from tidescout import paths


def test_repo_root_layout():
    assert (paths.REPO_ROOT / "backend" / "tidescout").is_dir()
    assert paths.FISHERIES_DIR == paths.REPO_ROOT / "fisheries"
    assert paths.DATA_DIR == paths.REPO_ROOT / "data"


def test_fishery_dirs_created(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    t = paths.tiles_dir("winyah-bay")
    assert d.is_dir() and d == tmp_path / "data" / "winyah-bay"
    assert t.is_dir() and t == d / "tiles"


def test_config_and_cache_still_resolve():
    from tidescout.config import FISHERIES_DIR as cfg_dir
    from tidescout.sources.cache import default_cache

    assert cfg_dir == paths.FISHERIES_DIR
    assert default_cache() is not None
