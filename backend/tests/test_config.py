import pytest
import yaml
from pydantic import ValidationError

from tidescout.config import load_fishery
from tidescout.models import Fishery
from tidescout.paths import FISHERIES_DIR


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


def test_mistyped_top_level_key_is_rejected():
    """A typo'd top-level key (`salinty:` for `salinity:`) must fail loudly,
    not silently fall back to that field's Python defaults. Before
    Fishery.model_config gained extra="forbid", this was undetectable by
    inspection once Task 5 writes fitted salinity numbers into the YAML: the
    block still parses, still looks complete, and the app runs on the
    unfitted theoretical constants while reporting nothing wrong."""
    raw = yaml.safe_load((FISHERIES_DIR / "winyah-bay.yaml").read_text())
    raw["salinty"] = raw.pop("salinity")
    with pytest.raises(ValidationError, match="salinty"):
        Fishery.model_validate(raw)


def test_real_fishery_document_has_no_unknown_top_level_keys():
    """Fishery.model_config's extra="forbid" is only safe repo-wide because
    the one real Fishery document declares every top-level key it uses --
    verified here so a future YAML addition that outruns the model fails
    fast in CI rather than as a surprise the next time someone edits it."""
    for path in FISHERIES_DIR.glob("*.yaml"):
        if path.name.endswith((".known-spots.yaml", ".tiles.yaml")):
            continue  # parsed by different models, never reach Fishery
        raw = yaml.safe_load(path.read_text())
        Fishery.model_validate(raw)  # raises ValidationError on any extra key
