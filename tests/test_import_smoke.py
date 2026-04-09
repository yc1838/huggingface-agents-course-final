from pathlib import Path


def test_import_gaia_agent():
    import gaia_agent

    assert Path(gaia_agent.__file__).resolve() == Path("src/gaia_agent/__init__.py").resolve()
