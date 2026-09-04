from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from tidescout.models import Fishery, KnownSpot, SpeciesProfile
from tidescout.paths import FISHERIES_DIR


def load_fishery(slug: str, root: Path | None = None) -> Fishery:
    path = (root or FISHERIES_DIR) / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No fishery config at {path}")
    raw = yaml.safe_load(path.read_text())
    return Fishery.model_validate(raw)


def fishery_now(slug: str) -> datetime:
    """Wall-clock time in the FISHERY's own zone -- not UTC, not system-local.

    Same reason `sources.weather._today` gives: "near the day boundary those
    can disagree with the fishery's own calendar day." Winyah Bay is
    `America/New_York`, so between 20:00 and 23:59 Eastern, UTC is already
    tomorrow. Deciding "today" in UTC there makes `tidescout warm --days 7`
    skip today entirely -- and that command's own docstring recommends running
    it overnight -- so the next morning the user hits a 202 and a 70-second
    wait for the one date the warming subsystem exists to have ready. The same
    slip shifts the API's forecast-horizon check and its staleness cutoff by a
    day.

    Returns an aware datetime rather than a `date` so callers that need an
    instant (`store.is_stale`) and callers that need a calendar day both get
    it from one place, in one zone.
    """
    return datetime.now(ZoneInfo(load_fishery(slug).timezone))


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
