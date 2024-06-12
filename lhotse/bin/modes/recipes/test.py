import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.test import download_test, prepare_test
from lhotse.utils import Pathlike


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def test(target_dir: Pathlike):
    """yes_no dataset download."""
    download_test(target_dir)

@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def test(corpus_dir: Pathlike, output_dir: Pathlike):
    """yes_no data preparation."""
    prepare_test(corpus_dir, output_dir=output_dir)
