from pathlib import Path

import yaml

from tidescout.models import Fishery

FISHERIES_DIR = Path(__file__).resolve().parents[2] / "fisheries"


def load_fishery(slug: str, root: Path | None = None) -> Fishery:
    path = (root or FISHERIES_DIR) / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No fishery config at {path}")
    raw = yaml.safe_load(path.read_text())
    return Fishery.model_validate(raw)
