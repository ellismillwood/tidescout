import json

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from tidescout.config import load_fishery
from tidescout.engine.terrain import zones as compute_zones
from tidescout.pipeline.artifacts import build_artifacts
from tidescout.pipeline.derivatives import build_derivatives

from . import synth
from .test_features_pipeline import _fake_bathy


def test_build_derivatives_writes_grid_aligned_rasters(tmp_path, monkeypatch, fishery):
    z = synth.dropoff_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    paths = build_derivatives("winyah-bay", fishery)
    assert set(paths) == {"slope", "curv", "zones"}
    for name, p in paths.items():
        assert p.exists(), name
        with rasterio.open(p) as src:
            assert src.width == z.shape[1]
            assert src.height == z.shape[0]
            assert src.transform == synth.TRANSFORM, f"{name} lost grid alignment"
            assert str(src.crs) == "EPSG:26917"
    # dtype/nodata for the float rasters -- zones' own dtype/nodata are covered
    # in more depth by test_zones_raster_is_categorical_and_nodata_zero below.
    with rasterio.open(paths["slope"]) as src:
        assert src.read(1).dtype == np.float32
        assert src.nodata == -9999.0
    with rasterio.open(paths["curv"]) as src:
        assert src.read(1).dtype == np.float32
        assert src.nodata == -9999.0


def test_zones_raster_is_categorical_and_nodata_zero(tmp_path, monkeypatch, fishery):
    # creek_mouth_dem (unlike dropoff_dem, whose two depths -1/-10 m fall into
    # only two zone bands -- 3 (mid-depth) and 4 (deep) -- under the fishery's
    # real thresholds) spans three separate zone bands, so the categorical
    # check below isn't trivially satisfied by an array of one repeated value.
    z = synth.creek_mouth_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    paths = build_derivatives("winyah-bay", fishery)
    with rasterio.open(paths["zones"]) as src:
        arr = src.read(1)
        assert src.nodata == 0
        assert arr.dtype == np.uint8
        # terrain.zones' real enum, confirmed by reading engine/terrain.py and
        # by the existing test_terrain.py::test_zones_bands, is 0=nodata,
        # 1=land (z >= land_elev_m), 2=shallow (shallow_max_m <= z < land_elev_m),
        # 3=mid-depth (deep_min_m <= z < shallow_max_m), 4=deep (z < deep_min_m)
        # -- five values, not the brief's guessed {0, 1, 2, 3}.
        assert set(np.unique(arr)) <= {0, 1, 2, 3, 4}, "zones must stay a small enum"
        assert set(np.unique(arr)) == {2, 3, 4}, "fixture should exercise multiple bands"
        # Cross-check the pipeline's write against a direct engine call, so this
        # regresses if build_derivatives ever diverges from what zones() itself
        # computes (wrong dtype cast, stale cache, transposed array, ...) rather
        # than only bounding the value range.
        expected = compute_zones(
            z,
            fishery.bathymetry.land_elev_m,
            fishery.bathymetry.zone_shallow_max_m,
            fishery.bathymetry.zone_deep_min_m,
        )
        np.testing.assert_array_equal(arr, expected)


def test_zone_thresholds_are_independent_of_bar_tuning(tmp_path, monkeypatch):
    """Retuning bar detection must not move the friction zones."""
    z = synth.point_bar_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    before = build_derivatives("winyah-bay", f)
    with rasterio.open(before["zones"]) as src:
        baseline = src.read(1).copy()

    f.features.shallow_max_m = -0.9      # aggressive bar retune
    f.features.deep_min_m = -6.0
    after = build_derivatives("winyah-bay", f)
    with rasterio.open(after["zones"]) as src:
        assert np.array_equal(src.read(1), baseline), "zones still alias bar thresholds"


def test_build_artifacts_produces_all_outputs(tmp_path, monkeypatch, fishery):
    z = synth.point_bar_dem()
    _fake_bathy(tmp_path, monkeypatch, z)
    build_derivatives("winyah-bay", fishery)
    # pipeline/artifacts.py writes quicklook.png via rasterio's PNG driver
    # without ever passing a crs/transform (unlike hillshade.tif, which is
    # written through the same _write() helper as the derivatives and is
    # fully georeferenced). That's a known, deliberately deferred Plan 2
    # limitation -- the quicklook is a display-only PNG, not a georeferenced
    # product -- not a bug for this test to fix. pytest.warns both documents
    # that expectation and scopes it tightly to the one write that causes
    # it, so a future genuine georeferencing regression in build_derivatives'
    # GeoTIFFs (checked below and in test_build_derivatives_writes_grid_
    # aligned_rasters) is not swallowed by this filter.
    with pytest.warns(NotGeoreferencedWarning, match="no geotransform"):
        out = build_artifacts("winyah-bay", fishery)
    # build_artifacts' real return keys (read from pipeline/artifacts.py) --
    # the brief's guessed "contours" key happened to be right, but the set
    # is asserted explicitly here so a dropped/renamed key fails loudly.
    assert set(out) == {"hillshade", "quicklook", "contours"}
    for name, p in out.items():
        assert p.exists() and p.stat().st_size > 0, name

    with rasterio.open(out["hillshade"]) as src:
        assert src.width == z.shape[1]
        assert src.height == z.shape[0]
        assert src.transform == synth.TRANSFORM, "hillshade lost grid alignment"
        assert str(src.crs) == "EPSG:26917"
        assert src.read(1).dtype == np.uint8

    # Reading the ungeoreferenced quicklook back also warns -- same known
    # limitation as above, scoped the same way.
    with pytest.warns(NotGeoreferencedWarning, match="no geotransform"):
        with rasterio.open(out["quicklook"]) as src:
            assert src.width == z.shape[1]
            assert src.height == z.shape[0]
            assert src.count == 4  # RGBA

    contours = json.loads(out["contours"].read_text())
    assert contours["type"] == "FeatureCollection"
    assert contours["features"], "point_bar_dem's ridge should cross a configured depth"
    feat = contours["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "LineString"
    assert isinstance(feat["properties"]["depth_m"], float)
