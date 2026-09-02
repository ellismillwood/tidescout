from pathlib import Path

import yaml

from tidescout.models import Fishery, KnownSpot, SpeciesProfile
from tidescout.paths import FISHERIES_DIR


def load_fishery(slug: str, root: Path | None = None) -> Fishery:
    path = (root or FISHERIES_DIR) / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No fishery config at {path}")
    raw = yaml.safe_load(path.read_text())
    return Fishery.model_validate(raw)


def load_known_spots(slug: str) -> list[KnownSpot]:
    path = FISHERIES_DIR / f"{slug}.known-spots.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    return [KnownSpot.model_validate(s) for s in raw.get("spots") or []]


def load_species(path: Path | None = None) -> dict[str, SpeciesProfile]:
    p = path or (FISHERIES_DIR / "species_weights.yaml")
    if not p.exists():
        raise FileNotFoundError(f"No species weights at {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    return {k: SpeciesProfile.model_validate(v) for k, v in raw.items()}
