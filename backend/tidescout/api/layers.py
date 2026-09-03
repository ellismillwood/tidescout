"""The static-layer allowlist (spec §4).

`slug` and `name` are both user input that reaches the filesystem. Neither is
ever concatenated into a path: `slug` is checked for membership in
`fishery_slugs()` and `name` is used as a dict KEY, so an unlisted or
traversing value cannot name a file at all -- not even one that exists.
"""

from pathlib import Path

from tidescout.api.readiness import fishery_slugs
from tidescout.paths import DATA_DIR

LAYERS: dict[str, str] = {
    "features": "features.geojson",
    "contours": "contours.geojson",
    "oysters": "oyster_reefs.web.geojson",
    "hillshade": "hillshade.png",
    "hillshade-bounds": "hillshade.bounds.json",
    "depth-tint": "depth_tint.png",
}


def layer_path(slug: str, name: str) -> Path:
    if slug not in fishery_slugs():
        raise ValueError(f"unknown fishery: {slug!r}")
    filename = LAYERS[name]  # KeyError for anything unlisted
    return DATA_DIR / slug / filename
