"""Which fisheries exist, and which are processed enough to score.

ONE predicate, TWO callers (spec §2): the `ready` flag in `/api/fisheries` and
the `409` from the day endpoint. Written once so the list and the error can
never disagree about the same fishery.
"""

from dataclasses import dataclass

from tidescout.config import load_fishery
from tidescout.paths import DATA_DIR, FISHERIES_DIR

# `fisheries/` also holds per-fishery SIDECARS -- `<slug>.known-spots.yaml`,
# `<slug>.tiles.yaml` -- and one global `species_weights.yaml`. A plain
# `*.yaml` glob would offer all of them as pickable fisheries.
_SIDECAR_SUFFIXES = (".known-spots", ".tiles")
_GLOBAL_YAMLS = frozenset({"species_weights"})


def fishery_slugs() -> list[str]:
    slugs = []
    for path in sorted(FISHERIES_DIR.glob("*.yaml")):
        stem = path.stem
        if stem in _GLOBAL_YAMLS or stem.endswith(_SIDECAR_SUFFIXES):
            continue
        slugs.append(stem)
    return slugs


@dataclass(frozen=True)
class Readiness:
    ready: bool
    missing: tuple[str, ...]


def readiness(slug: str) -> Readiness:
    """What a fishery still needs before it can be scored.

    Deliberately does NOT call `paths.fishery_data_dir`, which mkdirs its
    argument: an unknown or traversing slug must not create anything. The
    membership check against `fishery_slugs()` comes first for that reason.
    """
    if slug not in fishery_slugs():
        return Readiness(False, ("unknown fishery",))

    data = DATA_DIR / slug
    missing = []
    if not (data / "flow").is_dir() or not any((data / "flow").iterdir()):
        missing.append("flow library")
    if not (data / "estuary_km.npy").exists():
        missing.append("along-estuary distance field")
    if not (data / "features.geojson").exists():
        missing.append("feature inventory")
    return Readiness(not missing, tuple(missing))


def fishery_summaries() -> list[dict]:
    rows = []
    for slug in fishery_slugs():
        f = load_fishery(slug)
        r = readiness(slug)
        row = {
            "slug": slug,
            "name": f.name,
            "center": list(f.center),
            "timezone": f.timezone,
            "ready": r.ready,
        }
        if r.missing:
            row["reason"] = ", ".join(r.missing)
        rows.append(row)
    return rows
