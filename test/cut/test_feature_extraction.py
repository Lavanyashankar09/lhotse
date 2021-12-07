import multiprocessing
import sys
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import Mock

import pytest
import torch

from lhotse import (
    CutSet,
    Fbank,
    KaldiFbank,
    KaldiMfcc,
    KaldifeatFbank,
    KaldifeatMfcc,
    LibrosaFbank,
    LibrosaFbankConfig,
    LilcomHdf5Writer,
    Mfcc,
    MonoCut,
    Recording,
    validate,
)
from lhotse.audio import AudioSource
from lhotse.cut import MixedCut
from lhotse.features.io import LilcomFilesWriter
from lhotse.serialization import InvalidPathExtension
from lhotse.utils import is_module_available
from lhotse.utils import nullcontext as does_not_raise


@pytest.fixture
def recording():
    return Recording(
        id="rec",
        sources=[
            AudioSource(type="file", channels=[0, 1], source="test/fixtures/stereo.wav")
        ],
        sampling_rate=8000,
        num_samples=8000,
        duration=1.0,
    )


@pytest.fixture
def cut(recording):
    return MonoCut(id="cut", start=0, duration=1.0, channel=0, recording=recording)


def test_extract_features(cut):
    extractor = Fbank()
    arr = cut.compute_features(extractor=extractor)
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


def test_extract_and_store_features(cut):
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = cut.compute_and_store_features(
            extractor=extractor, storage=storage
        )
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 100
    assert arr.shape[1] == extractor.feature_dim(cut.sampling_rate)


@pytest.mark.parametrize("mix_eagerly", [False, True])
def test_extract_and_store_features_from_mixed_cut(cut, mix_eagerly):
    mixed_cut = cut.append(cut)
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir, LilcomFilesWriter(tmpdir) as storage:
        cut_with_feats = mixed_cut.compute_and_store_features(
            extractor=extractor, storage=storage, mix_eagerly=mix_eagerly
        )
        arr = cut_with_feats.load_features()
    assert arr.shape[0] == 200
    assert arr.shape[1] == extractor.feature_dim(mixed_cut.sampling_rate)


@pytest.fixture
def cut_set(cut):
    # The padding tests if feature extraction works correctly with a PaddingCut
    return CutSet.from_cuts([cut, cut.append(cut).pad(3.0)]).sort_by_duration()


# The lines below try to import Dask (a distributed computing library for Python)
# so that we can test that parallel feature extraction through the "executor"
# interface works correctly in this case. Dask in not a requirement or a dependency
# of Lhotse, so we make the tests with it optional as well.

try:
    import distributed
except:
    distributed = Mock()


def is_dask_availabe():
    try:
        import dask
        import distributed

        return True
    except:
        return False


@pytest.mark.parametrize("mix_eagerly", [False, True])
@pytest.mark.parametrize("storage_type", [LilcomFilesWriter, LilcomHdf5Writer])
@pytest.mark.parametrize(
    ["executor", "num_jobs"],
    [
        (None, 1),
        # For some reason in tests, we need to use the "spawn" context otherwise it hangs
        pytest.param(
            partial(
                ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")
            ),
            2,
            marks=pytest.mark.skipif(
                sys.version_info[0] == 3 and sys.version_info[1] < 7,
                reason="The mp_context argument is introduced in Python 3.7",
            ),
        ),
        pytest.param(
            distributed.Client,
            2,
            marks=pytest.mark.skipif(not is_dask_availabe(), reason="Requires Dask"),
        ),
    ],
)
def test_extract_and_store_features_from_cut_set(
    cut_set, executor, num_jobs, storage_type, mix_eagerly
):
    extractor = Fbank()
    with TemporaryDirectory() as tmpdir:
        cut_set_with_feats = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=tmpdir,
            num_jobs=num_jobs,
            mix_eagerly=mix_eagerly,
            executor=executor() if executor else None,
            storage_type=storage_type,
        ).sort_by_duration()  # sort by duration to ensure the same order of cuts

        # The same number of cuts
        assert len(cut_set_with_feats) == 2

        for orig_cut, feat_cut in zip(cut_set, cut_set_with_feats):
            # The ID is retained
            assert orig_cut.id == feat_cut.id
            # Features were attached
            assert feat_cut.has_features
            # Recording is retained unless mixing a MixedCut eagerly
            should_have_recording = not (mix_eagerly and isinstance(orig_cut, MixedCut))
            assert feat_cut.has_recording == should_have_recording

        cuts = list(cut_set_with_feats)

        arr = cuts[0].load_features()
        assert arr.shape[0] == 300
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)

        arr = cuts[1].load_features()
        assert arr.shape[0] == 100
        assert arr.shape[1] == extractor.feature_dim(cuts[0].sampling_rate)


@pytest.mark.parametrize(
    "extractor_type",
    [
        Fbank,
        Mfcc,
        KaldiFbank,
        KaldiMfcc,
        pytest.param(
            KaldifeatFbank,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            KaldifeatMfcc,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            lambda: LibrosaFbank(LibrosaFbankConfig(sampling_rate=16000)),
            marks=[
                pytest.mark.skipif(
                    not is_module_available("librosa"),
                    reason="Requires librosa to run.",
                ),
            ],
        ),
    ],
)
def test_cut_set_batch_feature_extraction(cut_set, extractor_type):
    extractor = extractor_type()
    cut_set = cut_set.resample(16000)
    with NamedTemporaryFile() as tmpf:
        cut_set_with_feats = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=tmpf.name,
            num_workers=0,
        )
        validate(cut_set_with_feats, read_data=True)


@pytest.mark.parametrize(
    ["suffix", "exception_expectation"],
    [
        (".jsonl", does_not_raise()),
        (".json", pytest.raises(InvalidPathExtension)),
    ],
)
def test_cut_set_batch_feature_extraction_manifest_path(
    cut_set, suffix, exception_expectation
):
    extractor = Fbank()
    cut_set = cut_set.resample(16000)
    with NamedTemporaryFile() as feat_f, NamedTemporaryFile(
        suffix=suffix
    ) as manifest_f:
        with exception_expectation:
            cut_set_with_feats = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=feat_f.name,
                manifest_path=manifest_f.name,
                num_workers=0,
            )
            validate(cut_set_with_feats, read_data=True)


@pytest.mark.parametrize(
    "extractor_type",
    [
        Fbank,
        Mfcc,
        KaldiFbank,
        KaldiMfcc,
        pytest.param(
            KaldifeatFbank,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            KaldifeatMfcc,
            marks=pytest.mark.skipif(
                not is_module_available("kaldifeat"),
                reason="Requires kaldifeat to run.",
            ),
        ),
        pytest.param(
            lambda: LibrosaFbank(LibrosaFbankConfig(sampling_rate=16000)),
            marks=[
                pytest.mark.skipif(
                    not is_module_available("librosa"),
                    reason="Requires librosa to run.",
                ),
            ],
        ),
    ],
)
def test_on_the_fly_batch_feature_extraction(cut_set, extractor_type):
    from lhotse.dataset import OnTheFlyFeatures

    extractor = OnTheFlyFeatures(extractor=extractor_type())
    cut_set = cut_set.resample(16000)
    feats, feat_lens = extractor(cut_set)  # does not crash
    assert isinstance(feats, torch.Tensor)
    assert isinstance(feat_lens, torch.Tensor)
