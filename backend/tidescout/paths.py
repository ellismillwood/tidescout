from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FISHERIES_DIR = REPO_ROOT / "fisheries"
DATA_DIR = REPO_ROOT / "data"


def fishery_data_dir(slug: str) -> Path:
    d = DATA_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def tiles_dir(slug: str) -> Path:
    t = fishery_data_dir(slug) / "tiles"
    t.mkdir(parents=True, exist_ok=True)
    return t
