import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
import respx
from rasterio.transform import from_origin

from tidescout import paths
from tidescout.config import load_fishery
from tidescout.pipeline import bathy
from tidescout.pipeline.bathy import build_bathy, ensure_tiles, read_bathy


def _write_tile(path, west, north, value, size=60, res=0.005):
    transform = from_origin(west, north, res, res)
    data = np.full((size, size), value, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def _write_bathy_utm(d: Path, transform, arr, meta_transform=None, crs="EPSG:26917"):
    """Write a synthetic `bathy_utm.tif` + `bathy_meta.json` pair directly --
    mirrors `_write_tile`'s pattern above but for the *output* raster
    `read_bathy` consumes, with an independently-settable meta transform so
    a test can make the JSON sidecar agree with or disagree from the raster
    itself (simulating an interrupted build that left a stale sidecar next
    to a freshly-rewritten tif)."""
    height, width = arr.shape
    with rasterio.open(
        d / "bathy_utm.tif", "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=bathy.NODATA, compress="lzw",
    ) as dst:
        dst.write(arr, 1)
    mt = transform if meta_transform is None else meta_transform
    meta = {
        "crs": crs,
        "transform": [mt.a, mt.b, mt.c, mt.d, mt.e, mt.f],
        "width": width, "height": height,
        "stats": {"min": float(arr.min()), "max": float(arr.max()), "pct_nodata": 0.0},
    }
    (d / "bathy_meta.json").write_text(json.dumps(meta))


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
    # Diagnostic (not asserted, see comment below): rounding to whole meters
    # shows 99.05% of valid pixels land exactly on the two source values
    # (85.18% at -5.0, 13.87% at -2.0); the remaining 0.95% are a thin
    # bilinear-interpolation band exactly at the two tiles' shared edge
    # (rounded to -4.0/-3.0), all strictly between -5.0 and -2.0.
    #
    # Deviation from the brief's literal `set(np.unique(np.round(vals))) <=
    # {-5.0, -2.0}`: that assertion cannot pass under `resampling=bilinear`
    # (Step 3's own spec, kept as-is here since smooth interpolation is the
    # right choice for real bathymetry -- nearest-neighbor would produce
    # blocky artifacts that downstream slope/feature detection would
    # misread). This synthetic fixture's tiles are ~55x coarser than the 10m
    # UTM target grid (0.005 deg native vs 10m dest -- upsampling), so any
    # sharp value step between two abutting source tiles necessarily
    # produces a nonzero-width interpolated band; that never happens with
    # real CUDEM tiles (~3m native, downsampled to 10m). Verified by direct
    # inspection (not just accepted on faith) that the interpolated values
    # are confined to <1% of pixels and never exceed the source range --
    # i.e. this is bilinear math working correctly, not mosaic/reproject
    # corruption. The assertions below check the invariant that actually
    # matters: correct assembly of both tile values, no corruption/scaling,
    # and no interpolation overshoot outside the source range.
    assert np.isclose(vals.min(), -5.0, atol=1e-3)
    assert np.isclose(vals.max(), -2.0, atol=1e-3)
    assert set(np.unique(np.round(vals))) <= {-5.0, -4.0, -3.0, -2.0}
    frac_exact = np.mean(np.isclose(vals, -5.0) | np.isclose(vals, -2.0))
    assert frac_exact > 0.9  # seam interpolation stays a thin band, not most of the raster
    meta = json.loads((out.parent / "bathy_meta.json").read_text())
    assert meta["stats"]["min"] <= -5.0 <= meta["stats"]["max"] or meta["stats"]["min"] == -5.0

    arr, transform, meta2 = read_bathy("winyah-bay")
    # `data` here is the raw band read directly from `out` above, before
    # read_bathy's own nodata->nan conversion -- compare sentinel count to
    # NaN count 1:1 rather than the tautological `arr[isnan(arr)].all()`
    # (trivially true for any array, checks nothing about nodata handling).
    assert np.isnan(arr).sum() == int((data == bathy.NODATA).sum())
    assert arr.shape == (meta2["height"], meta2["width"])


def test_read_bathy_returns_transform_from_raster_not_meta(tmp_path, monkeypatch):
    """The Affine `read_bathy` returns must be the raster's own transform, not
    a value parsed out of bathy_meta.json. This closes a coverage gap too:
    no prior test asserted anything about the returned Affine's actual
    values (only its downstream shape/nan consistency)."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    real_transform = from_origin(500000.0, 3700000.0, 10.0, 10.0)
    arr = np.full((20, 20), -3.0, dtype="float32")
    _write_bathy_utm(d, real_transform, arr)

    _, out_transform, out_meta = read_bathy("winyah-bay")

    assert tuple(out_transform)[:6] == pytest.approx(tuple(real_transform)[:6], abs=1e-9)
    assert out_meta["width"] == 20
    assert out_meta["height"] == 20


def test_read_bathy_raises_on_stale_meta_transform(tmp_path, monkeypatch):
    """An interrupted build can leave a fresh tif paired with a stale
    bathy_meta.json transform -- read_bathy must raise, naming both, rather
    than silently returning a misregistered Affine (every downstream
    artifact -- slope, curvature, feature polygons -- would otherwise be
    computed against the wrong georeferencing with no warning at all)."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    d = paths.fishery_data_dir("winyah-bay")
    real_transform = from_origin(500000.0, 3700000.0, 10.0, 10.0)
    stale_transform = from_origin(505000.0, 3700000.0, 10.0, 10.0)  # 500-cell east offset
    arr = np.full((20, 20), -3.0, dtype="float32")
    _write_bathy_utm(d, real_transform, arr, meta_transform=stale_transform)

    with pytest.raises(ValueError, match="transform"):
        read_bathy("winyah-bay")


@respx.mock
def test_ensure_tiles_downloads_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/a.nc"
    payload = b"x" * 2048
    respx.get(url).mock(return_value=httpx.Response(200, content=payload))

    out = ensure_tiles("winyah-bay", [{"url": url}])

    assert out == [paths.tiles_dir("winyah-bay") / "a.nc"]
    assert out[0].read_bytes() == payload
    assert not out[0].with_suffix(".part").exists()


@respx.mock
def test_ensure_tiles_skips_existing_large_file(tmp_path, monkeypatch):
    # Deviation from the brief's "size within 1% of Content-Length" rule: this
    # THREDDS origin's fileServer HEAD responses omit Content-Length entirely
    # (task 4 finding). Simplified per task-5 brief's approved adaptation #3:
    # exists + size > 10MB counts as already-downloaded, no HEAD round trip.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/big.nc"
    route = respx.get(url)
    dest = paths.tiles_dir("winyah-bay") / "big.nc"
    dest.write_bytes(b"0" * (11 * 1024 * 1024))  # > 10MB sentinel for "already downloaded"

    out = ensure_tiles("winyah-bay", [{"url": url}])

    assert out == [dest]
    assert route.call_count == 0
    assert dest.stat().st_size == 11 * 1024 * 1024  # untouched


@respx.mock
def test_ensure_tiles_redownloads_small_existing_file(tmp_path, monkeypatch):
    # A small/truncated leftover file (e.g. from an interrupted prior run)
    # must NOT be treated as a cached tile.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/small.nc"
    dest = paths.tiles_dir("winyah-bay") / "small.nc"
    dest.write_bytes(b"partial")
    payload = b"y" * 4096
    respx.get(url).mock(return_value=httpx.Response(200, content=payload))

    out = ensure_tiles("winyah-bay", [{"url": url}])

    assert out[0].read_bytes() == payload


class _FakeCompletedProcess:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr


@respx.mock
def test_ensure_tiles_retries_httpx_before_falling_back_to_curl(tmp_path, monkeypatch):
    # Live discovery (task 5, 2026-08-13): the THREDDS fileServer origin can
    # stall mid-stream on a long-lived httpx connection. A transient failure
    # should retry with a fresh connection, not fall straight to curl.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/flaky.nc"
    payload = b"ok" * 1024
    attempts = {"n": 0}

    def flaky(request):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise httpx.ReadError("stall")
        return httpx.Response(200, content=payload)

    respx.get(url).mock(side_effect=flaky)

    def fail_if_called(cmd, **kwargs):
        raise AssertionError("curl fallback should not be needed when httpx eventually succeeds")

    monkeypatch.setattr(bathy.subprocess, "run", fail_if_called)

    out = ensure_tiles("winyah-bay", [{"url": url}])

    assert out[0].read_bytes() == payload
    assert attempts["n"] == 2


@respx.mock
def test_ensure_tiles_falls_back_to_curl_after_httpx_exhausted(tmp_path, monkeypatch):
    # Live discovery (task 5): THREDDS stalled mid-stream for over an hour
    # with no exception raised at all on one tile (a `.part` file stopped
    # growing and the process just hung) -- worse than the fast
    # RemoteProtocolError seen earlier in the same session. httpx retries
    # alone aren't sufficient for this origin; curl (separate process, own
    # connection/timeout handling, resumable via `-C -`) is the fallback.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/stalling.nc"
    respx.get(url).mock(side_effect=httpx.ReadError("stall"))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_bytes(b"z" * 4096)
        return _FakeCompletedProcess(0)

    monkeypatch.setattr(bathy.subprocess, "run", fake_run)

    out = ensure_tiles("winyah-bay", [{"url": url}])

    assert out[0].read_bytes() == b"z" * 4096
    assert len(calls) == 1
    assert calls[0][0] == "curl"
    assert "-C" in calls[0]
    assert calls[0][calls[0].index("-o") + 1].endswith("stalling.part")


@respx.mock
def test_ensure_tiles_raises_when_curl_fallback_also_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    url = "https://example.test/tiles/dead.nc"
    respx.get(url).mock(side_effect=httpx.ReadError("stall"))

    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(1, stderr=b"curl: connection refused")

    monkeypatch.setattr(bathy.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="dead.nc"):
        ensure_tiles("winyah-bay", [{"url": url}])
