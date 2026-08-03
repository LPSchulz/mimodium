from importlib.metadata import version

import mimodium


def test_version_matches_distribution_metadata() -> None:
    assert mimodium.__version__ == version("mimodium")
