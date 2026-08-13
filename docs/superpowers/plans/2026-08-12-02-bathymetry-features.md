# TideScout Plan 2: Bathymetry Pipeline & Static Ambush Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tidescout bathy build winyah-bay` + `tidescout features winyah-bay` produce, from real NOAA CUDEM bathymetry, a UTM analysis raster with terrain derivatives, human-checkable map artifacts (hillshade, depth quicklook, contour GeoJSON), and a typed ambush-feature inventory (`features.geojson`: drop-offs, walls, holes, flats, creek mouths, bars, jetties) — with detectors proven on synthetic DEMs before touching real data, and a known-spots validation aid for Ellis.

**Architecture:** Pure array math lives in `engine/` (`terrain.py`, `render.py`, `detect.py`, plus `tides.py` refactored in from Plan 1's sources); file/network I/O lives in `sources/cudem.py` (tile discovery) and `pipeline/` (download, mosaic, derivatives, artifacts, features). A committed tile manifest (`fisheries/winyah-bay.tiles.yaml`) makes CUDEM acquisition reproducible after one live discovery run. This is Phase 2 of the design spec (`docs/superpowers/specs/2026-08-11-tidescout-design.md` §6, §12). Read `docs/superpowers/plans/2026-08-12-plan1-carryover-notes.md` first — it binds this plan.

**Tech Stack:** Python 3.12; new deps rasterio, scipy, shapely, scikit-image, numpy (explicit); existing httpx/pydantic/PyYAML/typer/rich; pytest (+respx only for discovery HTTP fixtures); ruff gate `select = ["E","F","I","UP","B","DTZ"]`, 100 cols.

## Global Constraints

- Repo root `~/Documents/tidescout`; branch for this plan: `plan-02-bathymetry`; venv `~/.venvs/tidescout` only.
- Tests NEVER hit the network (synthetic rasters in tmp_path; respx/XML fixtures for discovery). Steps labeled **LIVE** are the only exception.
- `engine/` purity: numpy/scipy/shapely/scikit-image imports are fine, and `rasterio.features`/`rasterio.transform`/`rasterio.warp.transform`/`Affine` (pure compute helpers) are allowed — but no file opens, no httpx, no sqlite3, no `rasterio.open` in `engine/`.
- All raster artifacts land in `data/<slug>/` (gitignored, rebuildable). Downloaded tiles cache in `data/<slug>/tiles/` and are never re-downloaded if present with matching size.
- Analysis CRS EPSG:26917 (NAD83 / UTM 17N), cell 10 m, elevations meters NAVD88 (negative = below datum). Config-driven via the `bathymetry:` block (Task 3).
- Always-24 DST contract and Plan 1 interfaces are settled — do not touch `assemble_day`, fetchers, or the CLI `conditions`/`stations` commands except where a task says so.
- `make check` green before every commit.

---

### Task 1: Central paths module + geo dependencies

**Files:**
- Create: `backend/tidescout/paths.py`
- Modify: `backend/pyproject.toml` (deps)
- Modify: `backend/tidescout/config.py` (use paths)
- Modify: `backend/tidescout/sources/cache.py` (use paths)
- Test: `backend/tests/test_paths.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (every later task uses these): `tidescout.paths.REPO_ROOT: Path`, `FISHERIES_DIR: Path`, `DATA_DIR: Path`, `fishery_data_dir(slug: str) -> Path` (creates `data/<slug>/` on demand), `tiles_dir(slug: str) -> Path` (creates `data/<slug>/tiles/`).

- [ ] **Step 1: Write the failing test**

`backend/tests/test_paths.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_paths.py -v`
Expected: FAIL (ModuleNotFoundError: tidescout.paths).

- [ ] **Step 3: Implement**

`backend/tidescout/paths.py`:

```python
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
```

In `backend/tidescout/config.py`: replace the `FISHERIES_DIR = Path(__file__)...` line with `from tidescout.paths import FISHERIES_DIR` (keep the name exported so existing imports keep working).

In `backend/tidescout/sources/cache.py`: in `default_cache()`, replace the `Path(__file__).resolve().parents[3] / "data" / "cache.sqlite"` computation with:

```python
from tidescout.paths import DATA_DIR

db = DATA_DIR / "cache.sqlite"
```

(keep the `mkdir` behavior — `Cache.__init__` already creates the parent).

In `backend/pyproject.toml` `[project] dependencies`, add:

```toml
    "numpy>=1.26",
    "scipy>=1.11",
    "rasterio>=1.3.9",
    "shapely>=2.0",
    "scikit-image>=0.22",
```

- [ ] **Step 4: Install and verify green**

```bash
make install
cd backend && ~/.venvs/tidescout/bin/python -m pytest tests/test_paths.py -v && cd .. && make check
```

Expected: install succeeds (rasterio/scikit-image ship macOS arm64 wheels), 3 new tests pass, full suite green (51). If rasterio wheels fail, retry with `uv pip install -p ~/.venvs/tidescout/bin/python --only-binary :all: rasterio` and report the version installed.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/paths.py backend/tidescout/config.py backend/tidescout/sources/cache.py backend/pyproject.toml backend/tests/test_paths.py
git commit -m "feat: central paths module and geospatial dependencies"
```

---

### Task 2: Move pure tide math to engine/tides.py (carryover refactor)

**Files:**
- Create: `backend/tidescout/engine/tides.py`
- Create: `backend/tests/test_tides.py`
- Modify: `backend/tidescout/sources/noaa.py`
- Modify: `backend/tidescout/sources/dayloader.py`
- Modify: `backend/tidescout/engine/conditions.py`
- Modify: `backend/tests/test_noaa.py`

**Interfaces:**
- Produces: `tidescout.engine.tides` exporting the dataclasses `TideHour`, `TideEvent`, `TideStage`, `CurrentHour` and pure functions `stage_at`, `_cosine_height`, `interpolate_tide_hours`, `interpolate_current_hours` — with **identical signatures and behavior** as their Plan 1 versions in `sources/noaa.py`.
- After this task: `sources/noaa.py` imports the dataclasses FROM `engine.tides` (sources→engine direction is allowed; engine→sources runtime imports are now zero), and re-exports them (`from tidescout.engine.tides import TideHour, ...`) so any stale import keeps working.

- [ ] **Step 1: Create the new module by MOVING code (no rewrites)**

Cut these from `backend/tidescout/sources/noaa.py` and paste unchanged into new `backend/tidescout/engine/tides.py` (add the needed imports: `math`, `dataclass`, `date`, `datetime`, `timedelta`, `ZoneInfo`, `pairwise`): the four dataclasses `TideHour`, `TideEvent`, `TideStage`, `CurrentHour`; and the functions `stage_at`, `_cosine_height`, `interpolate_tide_hours`, `interpolate_current_hours`.

At the top of `sources/noaa.py` add:

```python
from tidescout.engine.tides import CurrentHour, TideEvent, TideHour, TideStage

__all__ = ["CurrentHour", "TideEvent", "TideHour", "TideStage"]
```

(noaa.py keeps constructing these from wire data; nothing else in it changes.)

- [ ] **Step 2: Update the three consumers**

- `sources/dayloader.py`: change interpolator calls from `noaa.interpolate_tide_hours` / `noaa.interpolate_current_hours` to `from tidescout.engine import tides` … `tides.interpolate_tide_hours(...)` / `tides.interpolate_current_hours(...)`.
- `engine/conditions.py`: replace the function-level `from tidescout.sources.noaa import stage_at` with a top-level `from tidescout.engine.tides import stage_at` (engine→engine; the sanctioned-exception comment can go). Adjust the `TYPE_CHECKING` imports of `CurrentHour, TideEvent, TideHour` to come from `tidescout.engine.tides`.

- [ ] **Step 3: Move the pure tests**

Create `backend/tests/test_tides.py` and MOVE (unchanged except the import line, which becomes `from tidescout.engine.tides import ...`) these tests out of `test_noaa.py`: `test_stage_at_interpolates` and every `interpolate_tide_hours` / `interpolate_current_hours` test added during Plan 1 fix rounds. Wire-format tests (respx ones) stay in `test_noaa.py`.

**CAUTION:** `test_dayloader.py` monkeypatches interpolators — check whether it patches `noaa.interpolate_tide_hours` or the dayloader's own reference; after the import change, patch targets must be `tidescout.engine.tides.interpolate_tide_hours` **as seen by dayloader** (i.e., patch `tidescout.sources.dayloader.tides.interpolate_tide_hours` if dayloader does `from tidescout.engine import tides`). Update the monkeypatch targets accordingly and say what you changed in your report.

- [ ] **Step 4: Verify green**

Run: `cd backend && ~/.venvs/tidescout/bin/python -m pytest -q`
Expected: same total (51) — nothing lost in the move, no duplicated tests. Then `make check`.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout backend/tests
git commit -m "refactor: move pure tide math to engine/tides (structural engine boundary)"
```

---

### Task 3: Fishery config — bathymetry, feature thresholds, jetty seeds

**Files:**
- Modify: `backend/tidescout/models.py`
- Modify: `fisheries/winyah-bay.yaml`
- Test: `backend/tests/test_config.py` (extend)

**Interfaces:**
- Produces (consumed by Tasks 4–10):
  - `BathymetryConfig`: `epsg: int = 26917`, `cell_m: float = 10.0`, `land_elev_m: float = 1.5`, `contour_depths_m: list[float] = [-2.0, -5.0, -10.0, -15.0]`.
  - `FeatureThresholds`: `dropoff_slope_deg: float = 8.0`, `wall_slope_deg: float = 20.0`, `hole_delta_m: float = 1.5`, `hole_min_area_m2: float = 2000.0`, `flat_max_slope_deg: float = 1.0`, `flat_band_m: tuple[float, float] = (-1.5, 0.5)`, `shallow_max_m: float = -0.3`, `deep_min_m: float = -3.0`, `bar_min_area_m2: float = 1500.0`, `mouth_search_radius_m: float = 60.0`.
  - `JettySeed`: `name: str`, `coords: list[tuple[float, float]]` (lon, lat vertices, ≥2).
  - `Fishery` gains: `bathymetry: BathymetryConfig = BathymetryConfig()`, `features: FeatureThresholds = FeatureThresholds()`, `jetties: list[JettySeed] = []`.

- [ ] **Step 1: Write the failing test (append to `backend/tests/test_config.py`)**

```python
def test_bathymetry_and_feature_config():
    f = load_fishery("winyah-bay")
    assert f.bathymetry.epsg == 26917
    assert f.bathymetry.cell_m == 10.0
    assert f.features.dropoff_slope_deg > 0
    assert f.features.wall_slope_deg > f.features.dropoff_slope_deg
    assert len(f.jetties) == 2
    for j in f.jetties:
        assert len(j.coords) >= 2
        for lon, lat in j.coords:
            assert f.bbox[0] <= lon <= f.bbox[2]
            assert f.bbox[1] <= lat <= f.bbox[3]
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_config.py -v` → FAIL (no attribute `bathymetry`).

- [ ] **Step 3: Implement**

Add the three models to `backend/tidescout/models.py` exactly as in Interfaces (pydantic BaseModel each; place above `Fishery`), and the three new `Fishery` fields with those defaults.

Append to `fisheries/winyah-bay.yaml`:

```yaml
bathymetry:
  epsg: 26917
  cell_m: 10.0
  land_elev_m: 1.5
  contour_depths_m: [-2.0, -5.0, -10.0, -15.0]
features: {} # all defaults; tune here during validation
jetties:
  # Approximate from memory of the chart — refine vertices against
  # imagery/ENC during the known-spots validation pass. Format: [lon, lat].
  - name: North Jetty
    coords: [[-79.1850, 33.2320], [-79.1630, 33.2160]]
  - name: South Jetty
    coords: [[-79.1910, 33.2220], [-79.1700, 33.2050]]
```

- [ ] **Step 4: Verify green** — `pytest tests/test_config.py -v` then `make check`.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/models.py fisheries/winyah-bay.yaml backend/tests/test_config.py
git commit -m "feat: bathymetry, feature-threshold, and jetty-seed config"
```

---

### Task 4: CUDEM tile discovery + committed manifest (LIVE)

**Files:**
- Create: `backend/tidescout/sources/cudem.py`
- Modify: `backend/tidescout/cli.py` (add `bathy` sub-app with `discover`)
- Create (LIVE output): `fisheries/winyah-bay.tiles.yaml`
- Test: `backend/tests/test_cudem.py`

**Interfaces:**
- Produces:
  - `TileRef` dataclass: `key: str`, `url: str`.
  - `list_s3_keys(bucket: str, prefix: str) -> list[str]` — public S3 ListObjectsV2 over HTTPS (`https://{bucket}.s3.amazonaws.com/?list-type=2&prefix=...`), XML-parsed, paginated via continuation tokens.
  - `parse_ninth_arc_name(key: str) -> tuple[float, float] | None` — extracts (north_edge_lat, west_edge_lon) from names like `ncei19_n33x50_w079x25_<anything>.tif` (n33x50 → 33.50, w079x25 → −79.25); returns None for non-matching keys.
  - `candidate_tiles(fishery: Fishery, keys: list[str], bucket: str) -> list[TileRef]` — keeps `.tif` keys whose 0.25°×0.25° tile (extending south+east from the parsed corner) intersects the fishery bbox.
  - `load_manifest(slug: str) -> list[dict]` / `write_manifest(slug: str, entries: list[dict]) -> Path` — YAML at `fisheries/<slug>.tiles.yaml`, entries `{key, url, bounds: [w, s, e, n], crs: str}`.
  - CLI: `tidescout bathy discover SLUG`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_cudem.py`:

```python
import respx
from httpx import Response

from tidescout.config import load_fishery
from tidescout.sources.cudem import candidate_tiles, list_s3_keys, parse_ninth_arc_name

S3_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>CUDEM_9as/ncei19_n33x50_w079x50_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/ncei19_n33x25_w079x25_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/ncei19_n41x00_w074x00_2018v1.tif</Key></Contents>
  <Contents><Key>CUDEM_9as/readme.txt</Key></Contents>
</ListBucketResult>"""


def test_parse_ninth_arc_name():
    assert parse_ninth_arc_name("x/ncei19_n33x50_w079x25_2018v1.tif") == (33.50, -79.25)
    assert parse_ninth_arc_name("x/ncei19_n41x00_w074x00_v3.tif") == (41.00, -74.00)
    assert parse_ninth_arc_name("x/readme.txt") is None


@respx.mock
def test_list_and_filter_candidates():
    respx.get(url__regex=r"https://testbucket\.s3\.amazonaws\.com/.*").mock(
        return_value=Response(200, text=S3_PAGE)
    )
    keys = list_s3_keys("testbucket", "CUDEM_9as/")
    assert len(keys) == 4
    f = load_fishery("winyah-bay")
    tiles = candidate_tiles(f, keys, "testbucket")
    got = {t.key.rsplit("/", 1)[-1] for t in tiles}
    # n33x50/w079x50 spans lat 33.25..33.50, lon -79.50..-79.25 -> intersects bbox
    # n33x25/w079x25 spans lat 33.00..33.25, lon -79.25..-79.00 -> intersects (south edge)
    # n41/w074 does not; readme.txt is not a tif
    assert got == {"ncei19_n33x50_w079x50_2018v1.tif", "ncei19_n33x25_w079x25_2018v1.tif"}
    assert all(t.url.startswith("https://testbucket.s3.amazonaws.com/") for t in tiles)
```

- [ ] **Step 2: Run to verify failure** — `pytest tests/test_cudem.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement**

`backend/tidescout/sources/cudem.py`:

```python
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from tidescout.models import Fishery
from tidescout.paths import FISHERIES_DIR

TILE_SPAN_DEG = 0.25
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
NAME_RE = re.compile(r"ncei19_n(\d+)x(\d+)_w(\d+)x(\d+).*\.tif$")


@dataclass
class TileRef:
    key: str
    url: str


def list_s3_keys(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        resp = httpx.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        keys += [el.text for el in root.iter(f"{S3_NS}Key") if el.text]
        truncated = (root.findtext(f"{S3_NS}IsTruncated") or "false") == "true"
        if not truncated:
            return keys
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not token:
            return keys


def parse_ninth_arc_name(key: str) -> tuple[float, float] | None:
    m = NAME_RE.search(key)
    if not m:
        return None
    lat = int(m.group(1)) + int(m.group(2)) / 100.0
    lon = -(int(m.group(3)) + int(m.group(4)) / 100.0)
    return lat, lon


def candidate_tiles(fishery: Fishery, keys: list[str], bucket: str) -> list[TileRef]:
    west, south, east, north = fishery.bbox
    out = []
    for key in keys:
        parsed = parse_ninth_arc_name(key)
        if parsed is None:
            continue
        tile_n, tile_w = parsed
        tile_s, tile_e = tile_n - TILE_SPAN_DEG, tile_w + TILE_SPAN_DEG
        if tile_w < east and tile_e > west and tile_s < north and tile_n > south:
            out.append(TileRef(key, f"https://{bucket}.s3.amazonaws.com/{key}"))
    return out


def manifest_path(slug: str) -> Path:
    return FISHERIES_DIR / f"{slug}.tiles.yaml"


def load_manifest(slug: str) -> list[dict]:
    p = manifest_path(slug)
    if not p.exists():
        return []
    return yaml.safe_load(p.read_text()) or []


def write_manifest(slug: str, entries: list[dict]) -> Path:
    p = manifest_path(slug)
    p.write_text(yaml.safe_dump(entries, sort_keys=False))
    return p
```

Add to `backend/tidescout/cli.py` a `bathy` sub-app (`bathy_app = typer.Typer(...)`, `app.add_typer(bathy_app, name="bathy")`) with:

```python
@bathy_app.command()
def discover(slug: str, bucket: str = typer.Option("noaa-nos-cudem-pds", "--bucket"),
             prefix: str = typer.Option("CUDEM_9as/", "--prefix")) -> None:
    """Find CUDEM tiles intersecting the fishery bbox; verify and record a manifest."""
    import rasterio

    from tidescout.config import load_fishery
    from tidescout.sources import cudem

    fishery = load_fishery(slug)
    keys = cudem.list_s3_keys(bucket, prefix)
    tiles = cudem.candidate_tiles(fishery, keys, bucket)
    console.print(f"{len(keys)} keys under prefix; {len(tiles)} intersect bbox")
    entries = []
    for t in tiles:
        with rasterio.open(t.url) as src:  # range-request header read only
            b = src.bounds
            entries.append({
                "key": t.key, "url": t.url,
                "bounds": [b.left, b.bottom, b.right, b.top], "crs": str(src.crs),
            })
        console.print(f"  ok {t.key} bounds={entries[-1]['bounds']}")
    path = cudem.write_manifest(slug, entries)
    console.print(f"manifest written: {path} ({len(entries)} tiles)")
```

- [ ] **Step 4: Tests green** — `pytest tests/test_cudem.py -v` then `make check`.

- [ ] **Step 5: LIVE VERIFICATION — resolve the real bucket and write the manifest**

The bucket/prefix defaults are best-guess. Resolution ladder — stop at the first that works, and record what you did:

1. `~/.venvs/tidescout/bin/tidescout bathy discover winyah-bay` (defaults).
2. If the S3 list 404s/errors: `curl -s https://registry.opendata.aws/index.yaml | grep -A6 -i cudem` — the AWS Open Data registry lists NOAA CUDEM's real bucket `Resources.ARN`; retry with `--bucket <name> --prefix <path>` accordingly. Also try `curl -s "https://<bucket>.s3.amazonaws.com/?list-type=2&max-keys=20&delimiter=/"` to see top-level prefixes and adjust `--prefix` (the ninth-arc folder may be named differently, e.g. contain `ninth`).
3. If no S3 route exists: NCEI THREDDS — fetch `https://www.ngdc.noaa.gov/thredds/catalog.html`, navigate/curl to a CUDEM ninth-arc catalog, and if found, construct direct HTTPS file URLs; adapt `discover` only if URL construction differs (report the change).
4. If all fail: report **BLOCKED** with every URL tried and the raw error bodies — the controller will research.

Success criteria: manifest at `fisheries/winyah-bay.tiles.yaml` with **2–8 tiles**, every entry's recorded `bounds` genuinely intersecting `[-79.45, 33.15, -79.05, 33.60]` and `crs` ≈ EPSG:4269 or 4326. If `parse_ninth_arc_name`'s corner convention disagrees with the real bounds rasterio reports (that's what the recorded bounds are for), FIX the parser to match reality, adjust the unit test's expectations with a comment citing the observed tile, and note it in your report.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/sources/cudem.py backend/tidescout/cli.py backend/tests/test_cudem.py fisheries/winyah-bay.tiles.yaml
git commit -m "feat: CUDEM tile discovery with live-verified manifest for winyah-bay"
```

---

### Task 5: Download, mosaic, clip, reproject → analysis raster (LIVE)

**Files:**
- Create: `backend/tidescout/pipeline/__init__.py` (empty)
- Create: `backend/tidescout/pipeline/bathy.py`
- Modify: `backend/tidescout/cli.py` (add `bathy build`)
- Test: `backend/tests/test_bathy_pipeline.py`

**Interfaces:**
- Produces:
  - `ensure_tiles(slug: str, entries: list[dict]) -> list[Path]` — streams each manifest URL to `tiles_dir(slug)/<basename>`, skipping files that already exist with size within 1% of the server's Content-Length; 300 s timeout per tile.
  - `build_bathy(fishery: Fishery, tile_paths: list[Path]) -> Path` — rasterio merge → clip to fishery bbox (in tile CRS) → `warp.reproject` to `EPSG:{fishery.bathymetry.epsg}` at `cell_m` resolution (bilinear) → write `data/<slug>/bathy_utm.tif` (float32, nodata −9999, LZW) and `data/<slug>/bathy_meta.json` `{crs, transform: [a,b,c,d,e,f], width, height, stats: {min, max, pct_nodata}}`. Returns the tif path.
  - `read_bathy(slug: str) -> tuple[numpy.ndarray, Affine, dict]` — loads array (nodata → numpy.nan), transform, meta. **Every later task reads the raster through this.**
  - CLI: `tidescout bathy build SLUG` (runs ensure_tiles + build_bathy, prints stats).

- [ ] **Step 1: Write the failing test (synthetic tiles — no network)**

`backend/tests/test_bathy_pipeline.py`:

```python
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from tidescout.config import load_fishery
from tidescout.pipeline.bathy import build_bathy, read_bathy


def _write_tile(path, west, north, value, size=60, res=0.005):
    transform = from_origin(west, north, res, res)
    data = np.full((size, size), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def test_build_bathy_mosaic_and_reproject(tmp_path, monkeypatch):
    from tidescout import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    f = load_fishery("winyah-bay")
    # two abutting tiles covering part of the bbox: values -5 (north) and -2 (south)
    t1 = tmp_path / "t1.tif"
    t2 = tmp_path / "t2.tif"
    _write_tile(t1, west=-79.40, north=33.50, value=-5.0)
    _write_tile(t2, west=-79.40, north=33.20, value=-2.0)
    out = build_bathy(f, [t1, t2])
    assert out.name == "bathy_utm.tif"
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 26917
        assert abs(src.transform.a - 10.0) < 1e-6  # 10 m cells
        data = src.read(1)
    vals = data[data > -9000]
    assert set(np.unique(np.round(vals))) <= {-5.0, -2.0}
    meta = json.loads((out.parent / "bathy_meta.json").read_text())
    assert meta["stats"]["min"] <= -5.0 <= meta["stats"]["max"] or meta["stats"]["min"] == -5.0

    arr, transform, meta2 = read_bathy("winyah-bay")
    assert np.isnan(arr[np.isnan(arr)]).all()  # nodata became nan
    assert arr.shape == (meta2["height"], meta2["width"])
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

`backend/tidescout/pipeline/bathy.py`:

```python
import json
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir, tiles_dir

NODATA = -9999.0


def ensure_tiles(slug: str, entries: list[dict]) -> list[Path]:
    out = []
    for e in entries:
        dest = tiles_dir(slug) / e["url"].rsplit("/", 1)[-1]
        if not dest.exists():
            with httpx.stream("GET", e["url"], timeout=300, follow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
        out.append(dest)
    return out


def build_bathy(fishery: Fishery, tile_paths: list[Path]) -> Path:
    west, south, east, north = fishery.bbox
    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, src_transform = merge(sources, bounds=(west, south, east, north), nodata=NODATA)
        src_crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    band = mosaic[0]
    dst_crs = f"EPSG:{fishery.bathymetry.epsg}"
    cell = fishery.bathymetry.cell_m
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, band.shape[1], band.shape[0],
        left=west, bottom=south, right=east, top=north, resolution=cell,
    )
    dst = np.full((height, width), NODATA, dtype="float32")
    reproject(
        band, dst, src_transform=src_transform, src_crs=src_crs,
        dst_transform=transform, dst_crs=dst_crs,
        src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.bilinear,
    )
    out_dir = fishery_data_dir(fishery.slug)
    out = out_dir / "bathy_utm.tif"
    with rasterio.open(
        out, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
        crs=dst_crs, transform=transform, nodata=NODATA, compress="lzw",
    ) as dst_file:
        dst_file.write(dst, 1)
    valid = dst[dst != NODATA]
    meta = {
        "crs": dst_crs,
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "width": width, "height": height,
        "stats": {
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "pct_nodata": float((dst == NODATA).mean() * 100.0),
        },
    }
    (out_dir / "bathy_meta.json").write_text(json.dumps(meta, indent=1))
    return out


def read_bathy(slug: str) -> tuple[np.ndarray, Affine, dict]:
    out_dir = fishery_data_dir(slug)
    meta = json.loads((out_dir / "bathy_meta.json").read_text())
    with rasterio.open(out_dir / "bathy_utm.tif") as src:
        arr = src.read(1).astype("float32")
    arr[arr == NODATA] = np.nan
    t = meta["transform"]
    return arr, Affine(t[0], t[1], t[2], t[3], t[4], t[5]), meta
```

CLI `bathy build`:

```python
@bathy_app.command()
def build(slug: str) -> None:
    """Download manifest tiles (cached) and build the UTM analysis raster."""
    from tidescout.config import load_fishery
    from tidescout.pipeline.bathy import build_bathy, ensure_tiles
    from tidescout.sources.cudem import load_manifest

    fishery = load_fishery(slug)
    entries = load_manifest(slug)
    if not entries:
        raise typer.BadParameter(f"no tile manifest — run `tidescout bathy discover {slug}` first")
    tile_paths = ensure_tiles(slug, entries)
    out = build_bathy(fishery, tile_paths)
    import json as _json

    meta = _json.loads((out.parent / "bathy_meta.json").read_text())
    console.print(f"built {out}: {meta['width']}x{meta['height']} @10m, stats={meta['stats']}")
```

- [ ] **Step 4: Tests green** — `pytest tests/test_bathy_pipeline.py -v` then `make check`.

- [ ] **Step 5: LIVE VERIFICATION — build the real raster**

```bash
~/.venvs/tidescout/bin/tidescout bathy build winyah-bay
```

Downloads are hundreds of MB per tile — run tile-by-tile patiently; re-runs skip cached tiles. Success criteria: stats show `min ≤ −10` (shipping channel), `max ≥ 1` (marsh/upland), `pct_nodata < 60`. If the merged bounds clip wrongly (all nodata), inspect one tile's real bounds vs the manifest and fix the clip logic; document what you found.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline backend/tidescout/cli.py backend/tests/test_bathy_pipeline.py
git commit -m "feat: bathymetry download, mosaic, and UTM analysis raster"
```

---

### Task 6: Terrain derivatives (pure engine math)

**Files:**
- Create: `backend/tidescout/engine/terrain.py`
- Create: `backend/tidescout/pipeline/derivatives.py`
- Modify: `backend/tidescout/cli.py` (extend `bathy build` to also write derivatives — one command, one pipeline)
- Test: `backend/tests/test_terrain.py`

**Interfaces:**
- Produces:
  - `engine.terrain.slope_deg(z: ndarray, cell_m: float) -> ndarray` — NaN-safe (NaN in → NaN out), via `np.gradient`.
  - `engine.terrain.curvature(z: ndarray, cell_m: float) -> ndarray` — Laplacian (∂²x+∂²y); positive = concave (pit-like).
  - `engine.terrain.zones(z: ndarray, land_elev_m: float, shallow_max_m: float, deep_min_m: float) -> ndarray(uint8)` — 0 nodata / 1 land (z ≥ land) / 2 intertidal-flat band (shallow_max ≤ z < land) / 3 shallow (deep_min ≤ z < shallow_max) / 4 deep (z < deep_min).
  - `pipeline.derivatives.build_derivatives(slug: str, fishery: Fishery) -> dict[str, Path]` — reads via `read_bathy`, writes `slope.tif`, `curv.tif`, `zones.tif` next to it (same grid/CRS), returns paths.

- [ ] **Step 1: Failing tests**

`backend/tests/test_terrain.py`:

```python
import numpy as np

from tidescout.engine.terrain import curvature, slope_deg, zones


def test_slope_on_inclined_plane():
    # z drops 1 m per cell along x; cell 10 m -> slope = atan(0.1) = 5.71 deg
    z = np.tile(np.arange(50, dtype="float32") * -1.0, (50, 1))
    s = slope_deg(z, cell_m=10.0)
    interior = s[5:-5, 5:-5]
    assert np.allclose(interior, np.degrees(np.arctan(0.1)), atol=0.01)


def test_slope_nan_safe():
    z = np.zeros((20, 20), dtype="float32")
    z[10, 10] = np.nan
    s = slope_deg(z, 10.0)
    assert np.isnan(s[10, 10])


def test_curvature_sign_in_pit():
    yy, xx = np.mgrid[0:41, 0:41]
    z = (((xx - 20) ** 2 + (yy - 20) ** 2) / 100.0).astype("float32")  # bowl, min at center
    c = curvature(z, 10.0)
    assert c[20, 20] > 0  # concave up at the pit bottom


def test_zones_bands():
    z = np.array([[2.0, 0.0, -1.0, -5.0, np.nan]], dtype="float32")
    out = zones(z, land_elev_m=1.5, shallow_max_m=-0.3, deep_min_m=-3.0)
    assert out.tolist() == [[1, 2, 3, 4, 0]]
```

- [ ] **Step 2: Verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

`backend/tidescout/engine/terrain.py`:

```python
import numpy as np


def slope_deg(z: np.ndarray, cell_m: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")


def curvature(z: np.ndarray, cell_m: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype("float64"), cell_m)
    gyy, _ = np.gradient(gy, cell_m)
    _, gxx = np.gradient(gx, cell_m)
    return (gxx + gyy).astype("float32")


def zones(
    z: np.ndarray, land_elev_m: float, shallow_max_m: float, deep_min_m: float
) -> np.ndarray:
    out = np.zeros(z.shape, dtype="uint8")
    valid = ~np.isnan(z)
    out[valid & (z >= land_elev_m)] = 1
    out[valid & (z >= shallow_max_m) & (z < land_elev_m)] = 2
    out[valid & (z >= deep_min_m) & (z < shallow_max_m)] = 3
    out[valid & (z < deep_min_m)] = 4
    return out
```

`backend/tidescout/pipeline/derivatives.py`:

```python
from pathlib import Path

import numpy as np
import rasterio

from tidescout.engine import terrain
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy


def _write(path: Path, arr: np.ndarray, template_meta: dict, dtype: str, nodata) -> None:
    t = template_meta["transform"]
    with rasterio.open(
        path, "w", driver="GTiff", height=template_meta["height"], width=template_meta["width"],
        count=1, dtype=dtype, crs=template_meta["crs"],
        transform=rasterio.transform.Affine(t[0], t[1], t[2], t[3], t[4], t[5]),
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def build_derivatives(slug: str, fishery: Fishery) -> dict[str, Path]:
    z, _, meta = read_bathy(slug)
    d = fishery_data_dir(slug)
    s = terrain.slope_deg(z, fishery.bathymetry.cell_m)
    c = terrain.curvature(z, fishery.bathymetry.cell_m)
    zn = terrain.zones(
        z, fishery.bathymetry.land_elev_m,
        fishery.features.shallow_max_m, fishery.features.deep_min_m,
    )
    s_out = np.where(np.isnan(s), -9999.0, s)
    c_out = np.where(np.isnan(c), -9999.0, c)
    paths = {
        "slope": d / "slope.tif", "curv": d / "curv.tif", "zones": d / "zones.tif",
    }
    _write(paths["slope"], s_out, meta, "float32", -9999.0)
    _write(paths["curv"], c_out, meta, "float32", -9999.0)
    _write(paths["zones"], zn, meta, "uint8", 0)
    return paths
```

Extend the CLI `bathy build` command: after printing raster stats, call `build_derivatives(slug, fishery)` and print the three paths.

- [ ] **Step 4: Tests green + LIVE** — `pytest tests/test_terrain.py -v`, `make check`, then re-run `tidescout bathy build winyah-bay` and confirm the three derivative files exist with nonzero size.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/engine/terrain.py backend/tidescout/pipeline/derivatives.py backend/tidescout/cli.py backend/tests/test_terrain.py
git commit -m "feat: terrain derivatives (slope, curvature, zones)"
```

---

### Task 7: Map artifacts — hillshade, quicklook, contours

**Files:**
- Create: `backend/tidescout/engine/render.py`
- Create: `backend/tidescout/pipeline/artifacts.py`
- Modify: `backend/tidescout/cli.py` (add `bathy artifacts`)
- Test: `backend/tests/test_render.py`

**Interfaces:**
- Produces:
  - `engine.render.hillshade(z, cell_m, azimuth_deg=315.0, altitude_deg=45.0) -> ndarray(uint8)` — standard Horn/ESRI formula, NaN → 0.
  - `engine.render.depth_rgba(z, deep_min_m, land_elev_m) -> ndarray(h, w, 4) uint8` — blue ramp for water (darker = deeper), tan for land, transparent for NaN.
  - `engine.render.contour_lines(z, transform, crs_epsg: int, depths_m: list[float]) -> list[dict]` — skimage `find_contours` per depth; pixel → CRS coords via the affine; each dict `{depth_m, coords: [(lon, lat), ...]}` reprojected to EPSG:4326 (use `rasterio.warp.transform` — allowed compute helper); drop rings shorter than 5 points.
  - `pipeline.artifacts.build_artifacts(slug, fishery) -> dict[str, Path]` — writes `hillshade.tif` (uint8, same grid), `quicklook.png` (hillshade multiplied into depth_rgba, PNG driver), `contours.geojson` (FeatureCollection of LineStrings with `depth_m` property, EPSG:4326).
  - CLI: `tidescout bathy artifacts SLUG`.

- [ ] **Step 1: Failing tests**

`backend/tests/test_render.py`:

```python
import numpy as np
from rasterio.transform import from_origin

from tidescout.engine.render import contour_lines, depth_rgba, hillshade


def _cone(size=80, depth=-10.0):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    z = (depth * (1 - r / (size / 2))).astype("float32")
    return np.minimum(z, 0.0)


def test_hillshade_range_and_nan():
    z = _cone()
    z[0, 0] = np.nan
    hs = hillshade(z, cell_m=10.0)
    assert hs.dtype == np.uint8
    assert hs[0, 0] == 0
    assert 0 < hs[40, 20] <= 255


def test_depth_rgba_shape_and_alpha():
    z = np.array([[np.nan, -5.0, 2.0]], dtype="float32")
    rgba = depth_rgba(z, deep_min_m=-3.0, land_elev_m=1.5)
    assert rgba.shape == (1, 3, 4)
    assert rgba[0, 0, 3] == 0      # nan transparent
    assert rgba[0, 1, 3] == 255    # water opaque
    assert rgba[0, 2, 3] == 255    # land opaque


def test_contours_on_cone():
    z = _cone()
    transform = from_origin(500000, 3690000, 10, 10)  # UTM-ish coords
    lines = contour_lines(z, transform, crs_epsg=26917, depths_m=[-5.0, -2.0])
    depths = {round(li["depth_m"], 1) for li in lines}
    assert depths == {-5.0, -2.0}
    for li in lines:
        assert len(li["coords"]) >= 5
        for lon, lat in li["coords"]:
            assert -180 <= lon <= 180 and -90 <= lat <= 90
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

`backend/tidescout/engine/render.py`:

```python
import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from skimage import measure


def hillshade(
    z: np.ndarray, cell_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0
) -> np.ndarray:
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    gy, gx = np.gradient(np.nan_to_num(z, nan=0.0).astype("float64"), cell_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(shaded * 255.0, 0, 255).astype("uint8")
    out[np.isnan(z)] = 0
    return out


def depth_rgba(z: np.ndarray, deep_min_m: float, land_elev_m: float) -> np.ndarray:
    h, w = z.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    valid = ~np.isnan(z)
    water = valid & (z < land_elev_m)
    land = valid & ~water
    # deeper -> darker blue: map z in [2*deep_min, 0] to shade 0..1
    frac = np.clip(z / (2.0 * deep_min_m), 0.0, 1.0)  # 0 at surface, 1 at 2x deep_min
    rgba[..., 0][water] = (30 + 40 * (1 - frac[water])).astype("uint8")
    rgba[..., 1][water] = (90 + 110 * (1 - frac[water])).astype("uint8")
    rgba[..., 2][water] = (120 + 135 * (1 - frac[water])).astype("uint8")
    rgba[..., 0][land] = 205
    rgba[..., 1][land] = 190
    rgba[..., 2][land] = 160
    rgba[..., 3][valid] = 255
    return rgba


def contour_lines(
    z: np.ndarray, transform: Affine, crs_epsg: int, depths_m: list[float]
) -> list[dict]:
    out = []
    filled = np.nan_to_num(z, nan=1000.0)
    for depth in depths_m:
        for ring in measure.find_contours(filled, level=depth):
            if len(ring) < 5:
                continue
            xs, ys = [], []
            for row, col in ring:
                x, y = transform * (col + 0.5, row + 0.5)
                xs.append(x)
                ys.append(y)
            lons, lats = warp_transform(f"EPSG:{crs_epsg}", "EPSG:4326", xs, ys)
            out.append({"depth_m": float(depth), "coords": list(zip(lons, lats, strict=True))})
    return out
```

`backend/tidescout/pipeline/artifacts.py`:

```python
import json
from pathlib import Path

import numpy as np
import rasterio

from tidescout.engine import render
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy
from tidescout.pipeline.derivatives import _write


def build_artifacts(slug: str, fishery: Fishery) -> dict[str, Path]:
    z, transform, meta = read_bathy(slug)
    d = fishery_data_dir(slug)
    hs = render.hillshade(z, fishery.bathymetry.cell_m)
    _write(d / "hillshade.tif", hs, meta, "uint8", 0)

    rgba = render.depth_rgba(z, fishery.features.deep_min_m, fishery.bathymetry.land_elev_m)
    shade = (hs.astype("float32") / 255.0) * 0.6 + 0.4
    for band in range(3):
        rgba[..., band] = (rgba[..., band] * shade).astype("uint8")
    ql = d / "quicklook.png"
    with rasterio.open(
        ql, "w", driver="PNG", height=rgba.shape[0], width=rgba.shape[1], count=4, dtype="uint8"
    ) as dst:
        for band in range(4):
            dst.write(rgba[..., band], band + 1)

    lines = render.contour_lines(
        z, transform, fishery.bathymetry.epsg, fishery.bathymetry.contour_depths_m
    )
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"depth_m": li["depth_m"]},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in li["coords"]],
                },
            }
            for li in lines
        ],
    }
    (d / "contours.geojson").write_text(json.dumps(fc))
    return {"hillshade": d / "hillshade.tif", "quicklook": ql, "contours": d / "contours.geojson"}
```

CLI:

```python
@bathy_app.command()
def artifacts(slug: str) -> None:
    """Render hillshade, quicklook PNG, and contour GeoJSON."""
    from tidescout.config import load_fishery
    from tidescout.pipeline.artifacts import build_artifacts

    fishery = load_fishery(slug)
    for name, path in build_artifacts(slug, fishery).items():
        console.print(f"{name}: {path} ({path.stat().st_size:,} bytes)")
```

- [ ] **Step 4: Tests green + LIVE** — `pytest tests/test_render.py -v`, `make check`; then `tidescout bathy artifacts winyah-bay` and **open `data/winyah-bay/quicklook.png`** (`open` command) — a human-recognizable Winyah Bay (channel dark, marsh tan, jetty area visible) is the acceptance bar; report what you see.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/engine/render.py backend/tidescout/pipeline/artifacts.py backend/tidescout/cli.py backend/tests/test_render.py
git commit -m "feat: hillshade, depth quicklook, and contour artifacts"
```

---

### Task 8: Feature detectors with synthetic-DEM gates (the spec's hard gate)

**Files:**
- Create: `backend/tidescout/engine/detect.py`
- Create: `backend/tests/synth.py` (synthetic DEM builders — shared fixture module)
- Test: `backend/tests/test_detect.py`

**Interfaces:**
- Produces (`engine/detect.py`; consumed by Task 9's pipeline):
  - `Feature` dataclass: `type: str` (`"dropoff" | "wall" | "hole" | "flat" | "creek_mouth" | "bar" | "jetty"`), `geometry` (shapely geometry, **in raster CRS coords**), `attrs: dict` (numeric summaries).
  - `detect_dropoffs(z, slope, thresholds, transform) -> list[Feature]` — cells with `dropoff_slope_deg ≤ slope` and z below `land_elev` grouped into polygons (rasterio.features.shapes on the mask); polygons with mean slope ≥ `wall_slope_deg` are typed `"wall"`, else `"dropoff"`; attrs: `area_m2`, `mean_slope_deg`, `min_z`, `max_z`, `orientation_deg`.
  - `detect_holes(z, thresholds, cell_m, transform) -> list[Feature]` — `scipy.ndimage.grey_closing(z, size=15)` minus z > `hole_delta_m` (pockets deeper than surroundings), water-only, labeled, area ≥ `hole_min_area_m2`; attrs: `area_m2`, `depth_below_rim_m`, `min_z`.
  - `detect_flats(z, slope, thresholds, transform) -> list[Feature]` — slope ≤ `flat_max_slope_deg` and z within `flat_band_m`; area ≥ 4×`hole_min_area_m2` (flats are big); attrs `area_m2`, `mean_z`.
  - `detect_creek_mouths(z, thresholds, cell_m, transform) -> list[Feature]` — wet mask (z < `shallow_max_m`… use z < `land_elev`? No: wet = z < 0.0 MLLW-ish → use z < `shallow_max_m` is too strict; use z < 0.0); open-water body = largest connected component of `binary_opening(wet, disk(round(60/cell_m)))` (kills creeks, keeps the bay); creeks = wet minus dilated open-water; skeletonize creeks; a **mouth** is a skeleton pixel adjacent (within `mouth_search_radius_m`) to the open-water body; cluster mouth pixels within 3×`mouth_search_radius_m` into single Point features; attrs: `creek_width_m` (2× distance-transform value at the mouth pixel).
  - `detect_bars(z, thresholds, cell_m, transform) -> list[Feature]` — shallow mask (`deep_min_m < z < shallow_max_m`) regions whose dilation (3 cells) touches deep (`z < deep_min_m`) on ≥ 25% of their boundary, area ≥ `bar_min_area_m2`; attrs `area_m2`, `pct_deep_boundary`, `orientation_deg`.
  - `seed_jetties(fishery, transform_4326_to_grid) -> list[Feature]` — config linestrings reprojected to raster CRS (caller passes a `pyproj`-free callable built with `rasterio.warp.transform`); attrs `{name}`.
  - `orientation_deg(geometry) -> float` — bearing (0–180) of the geometry's PCA major axis over its exterior/line coords.
  - All detectors are NaN-safe and pure (arrays in, features out).

- [ ] **Step 1: Write the synthetic builders**

`backend/tests/synth.py`:

```python
"""Idealized DEMs. Grid: 200x200 cells at 10 m; coords are raster CRS meters."""

import numpy as np
from rasterio.transform import from_origin

CELL = 10.0
TRANSFORM = from_origin(500000, 3700000, CELL, CELL)


def open_basin(depth=-5.0, size=200):
    return np.full((size, size), depth, dtype="float32")


def creek_mouth_dem(size=200):
    """Marsh plain (+1 m) north half; open water (-4 m) south half; a 30 m wide,
    -2 m creek carved north-south through the marsh, joining open water at row 100."""
    z = np.full((size, size), 1.0, dtype="float32")
    z[100:, :] = -4.0
    z[0:100, 98:101] = -2.0  # 3 cells = 30 m wide creek
    return z


def point_bar_dem(size=200):
    """-6 m basin with an elongated -1 m shoal ridge (140x20 cells) mid-grid."""
    z = np.full((size, size), -6.0, dtype="float32")
    z[90:110, 30:170] = -1.0
    return z


def dropoff_dem(size=200):
    """-1 m shelf west of column 100; sharp step to -8 m east of it."""
    z = np.full((size, size), -1.0, dtype="float32")
    z[:, 100:] = -8.0
    return z


def hole_dem(size=200):
    """-3 m flat with a -10 m pocket (radius 8 cells) at center."""
    z = np.full((size, size), -3.0, dtype="float32")
    yy, xx = np.mgrid[0:size, 0:size]
    z[np.hypot(xx - 100, yy - 100) < 8] = -10.0
    return z
```

- [ ] **Step 2: Write the failing gate tests**

`backend/tests/test_detect.py`:

```python
import numpy as np
from shapely.geometry import Point

from tidescout.config import load_fishery
from tidescout.engine import detect
from tidescout.engine.terrain import slope_deg

from . import synth


def _thresholds():
    return load_fishery("winyah-bay").features


def test_gate_creek_mouth_found():
    z = synth.creek_mouth_dem()
    feats = detect.detect_creek_mouths(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    mouths = [f for f in feats if f.type == "creek_mouth"]
    assert mouths, "spec gate: idealized creek mouth must be detected"
    true_mouth = Point(synth.TRANSFORM * (99.5, 100.5))
    assert min(m.geometry.distance(true_mouth) for m in mouths) < 100.0  # within 100 m


def test_gate_point_bar_found():
    z = synth.point_bar_dem()
    feats = detect.detect_bars(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats, "spec gate: idealized bar must be detected"
    ridge_center = Point(synth.TRANSFORM * (100.5, 100.5))
    assert any(f.geometry.buffer(0).contains(ridge_center) for f in feats)
    assert all(60 <= f.attrs["orientation_deg"] <= 120 for f in feats)  # E-W ridge


def test_gate_dropoff_found_and_typed():
    z = synth.dropoff_dem()
    s = slope_deg(z, synth.CELL)
    feats = detect.detect_dropoffs(z, s, _thresholds(), synth.TRANSFORM)
    assert feats
    assert {f.type for f in feats} <= {"dropoff", "wall"}
    assert any(f.type == "wall" for f in feats)  # 7 m over 10 m cell = 35 deg


def test_gate_hole_found():
    z = synth.hole_dem()
    feats = detect.detect_holes(z, _thresholds(), synth.CELL, synth.TRANSFORM)
    assert feats
    f = max(feats, key=lambda x: x.attrs["area_m2"])
    assert f.attrs["depth_below_rim_m"] >= 3.0
    assert f.attrs["min_z"] <= -9.0


def test_flat_detected_only_in_band():
    z = np.full((100, 100), -0.5, dtype="float32")  # in flat band
    s = slope_deg(z, synth.CELL)
    feats = detect.detect_flats(z, s, _thresholds(), synth.TRANSFORM)
    assert feats and all(f.type == "flat" for f in feats)
    z2 = np.full((100, 100), -6.0, dtype="float32")  # too deep
    assert detect.detect_flats(z2, slope_deg(z2, synth.CELL), _thresholds(), synth.TRANSFORM) == []


def test_no_features_on_empty_basin():
    z = synth.open_basin()
    s = slope_deg(z, synth.CELL)
    t = _thresholds()
    assert detect.detect_dropoffs(z, s, t, synth.TRANSFORM) == []
    assert detect.detect_holes(z, t, synth.CELL, synth.TRANSFORM) == []
    assert detect.detect_creek_mouths(z, t, synth.CELL, synth.TRANSFORM) == []
    assert detect.detect_bars(z, t, synth.CELL, synth.TRANSFORM) == []


def test_orientation_of_ew_line():
    from shapely.geometry import LineString

    assert abs(detect.orientation_deg(LineString([(0, 0), (100, 0)])) - 90.0) < 1.0
```

- [ ] **Step 3: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 4: Implement `backend/tidescout/engine/detect.py`**

```python
from dataclasses import dataclass, field

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from skimage.morphology import binary_opening, disk, skeletonize

from tidescout.models import Fishery, FeatureThresholds

WET_LEVEL_M = 0.0  # approximate mean-water wetness for static detection


@dataclass
class Feature:
    type: str
    geometry: object
    attrs: dict = field(default_factory=dict)


def orientation_deg(geometry) -> float:
    coords = np.asarray(
        geometry.exterior.coords if hasattr(geometry, "exterior") else geometry.coords
    )
    xy = coords - coords.mean(axis=0)
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, int(np.argmax(evals))]
    bearing = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 180.0
    return float(bearing)


def _mask_polygons(mask: np.ndarray, transform: Affine, min_area_m2: float, cell_m: float):
    polys = []
    cell_area = cell_m * cell_m
    for geom, val in rio_features.shapes(mask.astype("uint8"), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if g.area >= min_area_m2 and g.area >= cell_area:
            polys.append(g)
    return polys


def detect_dropoffs(
    z: np.ndarray, slope: np.ndarray, t: FeatureThresholds, transform: Affine
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < WET_LEVEL_M)
    mask = wet & (slope >= t.dropoff_slope_deg)
    cell = abs(transform.a)
    out = []
    for g in _mask_polygons(mask, transform, t.hole_min_area_m2 / 4.0, cell):
        sel = rio_features.geometry_mask([g], z.shape, transform, invert=True)
        mean_slope = float(np.nanmean(slope[sel])) if sel.any() else 0.0
        ftype = "wall" if mean_slope >= t.wall_slope_deg else "dropoff"
        out.append(
            Feature(
                ftype, g,
                {
                    "area_m2": float(g.area), "mean_slope_deg": mean_slope,
                    "min_z": float(np.nanmin(z[sel])), "max_z": float(np.nanmax(z[sel])),
                    "orientation_deg": orientation_deg(g),
                },
            )
        )
    return out


def detect_holes(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    filled = np.nan_to_num(z, nan=1000.0)
    closed = ndimage.grey_closing(filled, size=15)
    pocket = (closed - filled) > t.hole_delta_m
    pocket &= ~np.isnan(z) & (z < WET_LEVEL_M)
    out = []
    for g in _mask_polygons(pocket, transform, t.hole_min_area_m2, cell_m):
        sel = rio_features.geometry_mask([g], z.shape, transform, invert=True)
        rim = float(np.nanmax((closed - filled)[sel]))
        out.append(
            Feature(
                "hole", g,
                {"area_m2": float(g.area), "depth_below_rim_m": rim,
                 "min_z": float(np.nanmin(z[sel]))},
            )
        )
    return out


def detect_flats(
    z: np.ndarray, slope: np.ndarray, t: FeatureThresholds, transform: Affine
) -> list[Feature]:
    lo, hi = t.flat_band_m
    mask = ~np.isnan(z) & (z >= lo) & (z < hi) & (slope <= t.flat_max_slope_deg)
    cell = abs(transform.a)
    return [
        Feature("flat", g, {"area_m2": float(g.area)})
        for g in _mask_polygons(mask, transform, 4.0 * t.hole_min_area_m2, cell)
    ]


def detect_creek_mouths(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    wet = ~np.isnan(z) & (z < WET_LEVEL_M)
    if not wet.any():
        return []
    open_radius = max(1, round(60.0 / cell_m))
    opened = binary_opening(wet, disk(open_radius))
    labels, n = ndimage.label(opened)
    if n == 0:
        return []
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=range(1, n + 1))
    open_water = labels == (1 + int(np.argmax(sizes)))
    creeks = wet & ~ndimage.binary_dilation(open_water, iterations=2)
    if not creeks.any():
        return []
    skel = skeletonize(creeks)
    dist_to_open = ndimage.distance_transform_edt(~open_water) * cell_m
    width = ndimage.distance_transform_edt(creeks) * cell_m
    mouth_px = skel & (dist_to_open <= t.mouth_search_radius_m)
    rows, cols = np.nonzero(mouth_px)
    if rows.size == 0:
        return []
    pts = [Point(transform * (c + 0.5, r + 0.5)) for r, c in zip(rows, cols, strict=True)]
    widths = [2.0 * float(width[r, c]) for r, c in zip(rows, cols, strict=True)]
    # cluster nearby mouth pixels into one feature
    merged = unary_union([p.buffer(3.0 * t.mouth_search_radius_m) for p in pts])
    clusters = merged.geoms if hasattr(merged, "geoms") else [merged]
    out = []
    for cl in clusters:
        members = [(p, w) for p, w in zip(pts, widths, strict=True) if cl.contains(p)]
        if not members:
            continue
        cx = float(np.mean([p.x for p, _ in members]))
        cy = float(np.mean([p.y for p, _ in members]))
        out.append(
            Feature(
                "creek_mouth", Point(cx, cy),
                {"creek_width_m": float(np.median([w for _, w in members]))},
            )
        )
    return out


def detect_bars(
    z: np.ndarray, t: FeatureThresholds, cell_m: float, transform: Affine
) -> list[Feature]:
    shallow = ~np.isnan(z) & (z > t.deep_min_m) & (z < t.shallow_max_m)
    deep = ~np.isnan(z) & (z <= t.deep_min_m)
    if not shallow.any() or not deep.any():
        return []
    out = []
    labels, n = ndimage.label(shallow)
    deep_dilated = ndimage.binary_dilation(deep, iterations=3)
    for i in range(1, n + 1):
        region = labels == i
        area = float(region.sum()) * cell_m * cell_m
        if area < t.bar_min_area_m2:
            continue
        boundary = region & ~ndimage.binary_erosion(region)
        pct = float((boundary & deep_dilated).sum()) / max(1, int(boundary.sum()))
        if pct < 0.25:
            continue
        polys = _mask_polygons(region, transform, t.bar_min_area_m2, cell_m)
        for g in polys:
            out.append(
                Feature(
                    "bar", g,
                    {"area_m2": float(g.area), "pct_deep_boundary": pct,
                     "orientation_deg": orientation_deg(g)},
                )
            )
    return out


def seed_jetties(fishery: Fishery, lonlat_to_grid_xy) -> list[Feature]:
    from shapely.geometry import LineString

    out = []
    for j in fishery.jetties:
        lons = [c[0] for c in j.coords]
        lats = [c[1] for c in j.coords]
        xs, ys = lonlat_to_grid_xy(lons, lats)
        out.append(Feature("jetty", LineString(zip(xs, ys, strict=True)), {"name": j.name}))
    return out
```

- [ ] **Step 5: Run the gates** — `pytest tests/test_detect.py -v`.
These are the spec's hard gates. If a gate fails, fix the DETECTOR (or, if the synthetic scenario itself is geometrically wrong, fix it and explain the geometry in your report) — do NOT weaken an assertion to pass. Expected: 7 PASS. Then `make check`.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/engine/detect.py backend/tests/synth.py backend/tests/test_detect.py
git commit -m "feat: ambush feature detectors proven on synthetic DEMs"
```

---

### Task 9: Feature inventory on real Winyah Bay (LIVE) + features CLI

**Files:**
- Create: `backend/tidescout/pipeline/features.py`
- Modify: `backend/tidescout/cli.py` (add `features` command)
- Test: `backend/tests/test_features_pipeline.py`

**Interfaces:**
- Produces:
  - `pipeline.features.build_features(slug: str, fishery: Fishery) -> Path` — reads bathy + slope; runs all detectors; converts geometries to EPSG:4326 (`rasterio.warp.transform` on coordinate arrays; for polygons transform exterior rings, drop interior rings); writes `data/<slug>/features.geojson`: FeatureCollection where each feature has `id` (`"<type>-<n>"`), `properties: {type, **attrs}` (floats rounded 2), geometry Point/LineString/Polygon.
  - `load_features(slug: str) -> dict` — parsed GeoJSON.
  - CLI: `tidescout features SLUG [--rebuild]` — builds if missing or `--rebuild`; prints a Rich table of counts by type plus the 5 largest-by-area with centroid lat/lon.

- [ ] **Step 1: Failing test (synthetic end-to-end through the pipeline, tmp DATA_DIR)**

`backend/tests/test_features_pipeline.py`:

```python
import json

import numpy as np
import rasterio

from tidescout.config import load_fishery
from tidescout.pipeline.features import build_features, load_features

from . import synth


def _fake_bathy(tmp_path, monkeypatch, z):
    from tidescout import paths
    from tidescout.pipeline import bathy as bmod
    from tidescout.pipeline.derivatives import _write

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    t = synth.TRANSFORM
    meta = {
        "crs": "EPSG:26917",
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "width": z.shape[1], "height": z.shape[0],
        "stats": {"min": float(np.nanmin(z)), "max": float(np.nanmax(z)), "pct_nodata": 0.0},
    }
    (d / "bathy_meta.json").write_text(json.dumps(meta))
    _write(d / "bathy_utm.tif", np.nan_to_num(z, nan=-9999.0), meta, "float32", -9999.0)


def test_build_features_on_synthetic(tmp_path, monkeypatch):
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    out = build_features("winyah-bay", f)
    fc = load_features("winyah-bay")
    assert out.name == "features.geojson"
    types = {feat["properties"]["type"] for feat in fc["features"]}
    assert "creek_mouth" in types
    assert "jetty" in types  # seeds always present
    for feat in fc["features"]:
        gtype = feat["geometry"]["type"]
        coords = feat["geometry"]["coordinates"]
        flat = (
            [coords] if gtype == "Point"
            else coords if gtype == "LineString"
            else coords[0]
        )
        for lon, lat in flat:
            assert -180 <= lon <= 180 and -90 <= lat <= 90
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement `backend/tidescout/pipeline/features.py`**

```python
import json
from pathlib import Path

from rasterio.warp import transform as warp_transform
from shapely.geometry import LineString, Point, Polygon, mapping

from tidescout.engine import detect
from tidescout.engine.terrain import slope_deg
from tidescout.models import Fishery
from tidescout.paths import fishery_data_dir
from tidescout.pipeline.bathy import read_bathy


def _to4326(geom, epsg: int):
    src = f"EPSG:{epsg}"

    def tx(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        lons, lats = warp_transform(src, "EPSG:4326", xs, ys)
        return list(zip(lons, lats, strict=True))

    if isinstance(geom, Point):
        (lonlat,) = tx([(geom.x, geom.y)])
        return Point(lonlat)
    if isinstance(geom, LineString):
        return LineString(tx(list(geom.coords)))
    if isinstance(geom, Polygon):
        return Polygon(tx(list(geom.exterior.coords)))
    raise TypeError(f"unsupported geometry: {geom.geom_type}")


def build_features(slug: str, fishery: Fishery) -> Path:
    z, transform, meta = read_bathy(slug)
    cell = fishery.bathymetry.cell_m
    epsg = fishery.bathymetry.epsg
    t = fishery.features
    slope = slope_deg(z, cell)

    def lonlat_to_grid(lons, lats):
        return warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)

    feats = (
        detect.detect_dropoffs(z, slope, t, transform)
        + detect.detect_holes(z, t, cell, transform)
        + detect.detect_flats(z, slope, t, transform)
        + detect.detect_creek_mouths(z, t, cell, transform)
        + detect.detect_bars(z, t, cell, transform)
        + detect.seed_jetties(fishery, lonlat_to_grid)
    )
    counters: dict[str, int] = {}
    out = []
    for f in feats:
        counters[f.type] = counters.get(f.type, 0) + 1
        props = {"type": f.type}
        for k, v in f.attrs.items():
            props[k] = round(v, 2) if isinstance(v, float) else v
        out.append(
            {
                "type": "Feature",
                "id": f"{f.type}-{counters[f.type]}",
                "properties": props,
                "geometry": mapping(_to4326(f.geometry, epsg)),
            }
        )
    path = fishery_data_dir(slug) / "features.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": out}))
    return path


def load_features(slug: str) -> dict:
    return json.loads((fishery_data_dir(slug) / "features.geojson").read_text())
```

CLI:

```python
@app.command()
def features(slug: str, rebuild: bool = typer.Option(False, "--rebuild")) -> None:
    """Build (if needed) and summarize the ambush-feature inventory."""
    from tidescout.config import load_fishery
    from tidescout.paths import fishery_data_dir
    from tidescout.pipeline.features import build_features, load_features

    fishery = load_fishery(slug)
    path = fishery_data_dir(slug) / "features.geojson"
    if rebuild or not path.exists():
        build_features(slug, fishery)
    fc = load_features(slug)
    counts: dict[str, int] = {}
    for f in fc["features"]:
        counts[f["properties"]["type"]] = counts.get(f["properties"]["type"], 0) + 1
    table = Table(title=f"{fishery.name} — ambush features")
    table.add_column("type")
    table.add_column("count", justify="right")
    for k in sorted(counts):
        table.add_row(k, str(counts[k]))
    console.print(table)
    sized = [f for f in fc["features"] if "area_m2" in f["properties"]]
    for f in sorted(sized, key=lambda x: -x["properties"]["area_m2"])[:5]:
        c = f["geometry"]["coordinates"]
        pt = c if f["geometry"]["type"] == "Point" else c[0][0] if f["geometry"]["type"] == "Polygon" else c[0]
        console.print(
            f"  {f['id']}: {f['properties']['area_m2']:,.0f} m² near ({pt[1]:.4f}, {pt[0]:.4f})"
        )
```

- [ ] **Step 4: Tests green** — `pytest tests/test_features_pipeline.py -v` then `make check`.

- [ ] **Step 5: LIVE VERIFICATION — real inventory**

```bash
~/.venvs/tidescout/bin/tidescout features winyah-bay --rebuild
```

Acceptance gates (report the actual numbers):
- `jetty` count == 2 (the seeds).
- `dropoff` + `wall` combined ≥ 20 (the dredged shipping channel alone guarantees walls).
- `creek_mouth` count in [10, 500] — the marsh is creek-riddled; far outside that range means thresholds need tuning (tune in `fisheries/winyah-bay.yaml` `features:` block, re-run, document final values).
- `hole`, `flat`, `bar` each ≥ 1.
- Runtime: full detection under ~5 minutes on the 10 m grid. If it grossly exceeds that, note where time went (likely `grey_closing` or `shapes`) — do not optimize beyond getting it usable; ledger the rest.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/features.py backend/tidescout/cli.py backend/tests/test_features_pipeline.py fisheries/winyah-bay.yaml
git commit -m "feat: real-bathymetry ambush feature inventory for winyah-bay"
```

---

### Task 10: Known-spots file + nearest-feature validation aid

**Files:**
- Create: `fisheries/winyah-bay.known-spots.yaml`
- Modify: `backend/tidescout/models.py` (KnownSpot)
- Modify: `backend/tidescout/config.py` (loader)
- Modify: `backend/tidescout/cli.py` (add `spots`)
- Test: `backend/tests/test_spots.py`

**Interfaces:**
- Produces:
  - `KnownSpot` model: `name: str`, `lon: float`, `lat: float`, `kind_hint: str = ""`, `notes: str = ""`.
  - `load_known_spots(slug: str) -> list[KnownSpot]` (empty list if the file has no `spots:` entries).
  - CLI `tidescout spots SLUG`: for each spot, the nearest detected feature (type, id, distance in meters — computed in UTM via `rasterio.warp.transform`, shapely `distance`) — Ellis's static validation aid: his trusted spots should land near detected structure.

- [ ] **Step 1: Failing test**

`backend/tests/test_spots.py`:

```python
from tidescout.config import load_known_spots


def test_known_spots_template_loads():
    spots = load_known_spots("winyah-bay")
    assert isinstance(spots, list)  # template ships with zero uncommented spots


def test_known_spots_parse(tmp_path, monkeypatch):
    from tidescout import config, paths

    monkeypatch.setattr(paths, "FISHERIES_DIR", tmp_path)
    monkeypatch.setattr(config, "FISHERIES_DIR", tmp_path)
    (tmp_path / "x.known-spots.yaml").write_text(
        "spots:\n  - name: Jetty rip\n    lon: -79.17\n    lat: 33.21\n"
        "    kind_hint: eddy\n    notes: ebb only\n"
    )
    spots = load_known_spots("x")
    assert spots[0].name == "Jetty rip"
    assert spots[0].lat == 33.21
```

- [ ] **Step 2: Verify failure.**

- [ ] **Step 3: Implement**

`fisheries/winyah-bay.known-spots.yaml`:

```yaml
# Ellis's ground truth for Winyah Bay. Fill freely — this file is the
# validation input for feature detection now and the ANUGA flow gate in Plan 3.
# Each spot: name, lon, lat (decimal degrees), kind_hint (eddy | rip | slack |
# dropoff | hole | flat | creek_mouth | bar | other), notes (tide phase, season,
# anything). Example (leave commented):
#   - name: North Jetty rip
#     lon: -79.1680
#     lat: 33.2190
#     kind_hint: rip
#     notes: last of the ebb, trout stack on the seam
spots: []
```

`KnownSpot` model in models.py (fields per Interfaces). In `config.py`:

```python
def load_known_spots(slug: str) -> list[KnownSpot]:
    path = FISHERIES_DIR / f"{slug}.known-spots.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    return [KnownSpot.model_validate(s) for s in raw.get("spots") or []]
```

CLI:

```python
@app.command()
def spots(slug: str) -> None:
    """Show each known spot's nearest detected feature (static validation aid)."""
    from rasterio.warp import transform as warp_transform
    from shapely.geometry import Point, shape

    from tidescout.config import load_fishery, load_known_spots
    from tidescout.pipeline.features import load_features

    fishery = load_fishery(slug)
    known = load_known_spots(slug)
    if not known:
        console.print(f"no spots in fisheries/{slug}.known-spots.yaml — add some!")
        return
    fc = load_features(slug)
    epsg = fishery.bathymetry.epsg

    def to_utm(lons, lats):
        return warp_transform("EPSG:4326", f"EPSG:{epsg}", lons, lats)

    feats = []
    for f in fc["features"]:
        g = shape(f["geometry"])
        xs, ys = to_utm(*zip(*[(c[0], c[1]) for c in _all_coords(g)], strict=True))
        feats.append((f["id"], f["properties"]["type"], list(zip(xs, ys, strict=True))))
    table = Table(title=f"{fishery.name} — known spots vs detected features")
    for col in ("spot", "nearest feature", "type", "distance m"):
        table.add_column(col)
    for s in known:
        (sx,), (sy,) = to_utm([s.lon], [s.lat])
        p = Point(sx, sy)
        best = min(
            ((fid, ftype, min(p.distance(Point(x, y)) for x, y in coords))
             for fid, ftype, coords in feats),
            key=lambda r: r[2],
        )
        table.add_row(s.name, best[0], best[1], f"{best[2]:,.0f}")
    console.print(table)


def _all_coords(geom):
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    if geom.geom_type == "LineString":
        return list(geom.coords)
    if geom.geom_type == "Polygon":
        return list(geom.exterior.coords)
    return [(geom.centroid.x, geom.centroid.y)]
```

- [ ] **Step 4: Tests green** — `pytest tests/test_spots.py -v`, then `make check`; run `tidescout spots winyah-bay` (expect the "add some!" message).

- [ ] **Step 5: Commit**

```bash
git add fisheries/winyah-bay.known-spots.yaml backend/tidescout/models.py backend/tidescout/config.py backend/tidescout/cli.py backend/tests/test_spots.py
git commit -m "feat: known-spots ground-truth file and nearest-feature validation CLI"
```

---

### Task 11: Wrap-up — full gate, README, disk audit

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** `make check` — full suite green (expect ~75±5 tests).
- [ ] **Step 2:** Append to README's Running section:

```markdown
    # bathymetry pipeline (Plan 2): discover tiles once, then build + detect
    ~/.venvs/tidescout/bin/tidescout bathy discover winyah-bay
    ~/.venvs/tidescout/bin/tidescout bathy build winyah-bay      # ~1-2 GB of tiles cached in data/
    ~/.venvs/tidescout/bin/tidescout bathy artifacts winyah-bay  # open data/winyah-bay/quicklook.png
    ~/.venvs/tidescout/bin/tidescout features winyah-bay         # ambush-feature inventory
    ~/.venvs/tidescout/bin/tidescout spots winyah-bay            # your known spots vs detections
```

- [ ] **Step 3:** Verify each README command runs (all cached/local now — fast). Report `du -sh data/winyah-bay` and the feature counts.
- [ ] **Step 4:** Commit `docs: bathymetry pipeline run instructions`.

---

## Self-review checklist (applied by the plan author)

- Spec §6 coverage: drop-offs/walls ✓ holes ✓ creek mouths ✓ points/bars ✓ (as `bar`; land-promontory "points" fold into bars + jetty seeds for v1 — noted deviation, revisit with flow data in Plan 3) flats ✓ (static band; ANUGA wet/dry refines in Plan 3) dredged walls ✓ (slope-typed) jetties ✓ (config seeds) oyster ✗ (SCDNR layer explicitly deferred — spec allows).
- Placeholders: none — every step has runnable code; unknown-at-planning values (S3 bucket/prefix, tile-name corner convention) are resolved by Task 4's LIVE ladder with fix procedures and recorded into a committed manifest.
- Type consistency: `read_bathy` tuple shape used identically in Tasks 6/7/9; `FeatureThresholds` field names match every detector reference; `Feature.attrs` keys consumed by Task 9's rounding are produced by Task 8; `_write` helper reused (derivatives → artifacts) with the same signature.
- Carryover compliance: paths.py (Task 1), engine/tides.py (Task 2), explicit lint gate respected (code ≤100 cols, abc imports, tz-aware n/a here), always-24 contract untouched, no HTTP slug exposure.
