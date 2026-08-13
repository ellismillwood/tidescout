import pytest

from tidescout.config import load_fishery


def test_load_winyah_bay():
    f = load_fishery("winyah-bay")
    assert f.name == "Winyah Bay"
    assert f.timezone == "America/New_York"
    west, south, east, north = f.bbox
    assert west < east and south < north
    assert west < f.center[0] < east
    assert south < f.center[1] < north
    assert len(f.rivers) == 3
    assert f.discharge_buckets.low_below_cfs < f.discharge_buckets.high_above_cfs
    assert set(f.climatology.water_temp_f_by_month) == set(range(1, 13))
    assert set(f.climatology.salinity_ppt_by_month) == set(range(1, 13))


def test_unknown_fishery_raises():
    with pytest.raises(FileNotFoundError):
        load_fishery("lake-lanier")


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


def test_winyah_has_a_closed_model_domain():
    f = load_fishery("winyah-bay")
    assert f.model_domain is not None
    poly = f.model_domain.polygon_utm_km
    assert len(poly) >= 4
    assert poly[0] != poly[-1], "polygon is implicitly closed; do not repeat the first vertex"
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # sanity: inside the Winyah UTM 17N analysis grid (643.8-681.9 E, 3669.0-3719.5 N km)
    assert 643.0 < min(xs) and max(xs) < 682.0
    assert 3668.0 < min(ys) and max(ys) < 3720.0


def test_anuga_mass_tolerance_is_not_machine_precision():
    """A 1e-6 assert fails on a healthy wetting/drying run (measured 4.2e-4)."""
    f = load_fishery("winyah-bay")
    assert f.anuga.mass_tolerance >= 1e-4
