import pytest
import yaml
from pydantic import ValidationError

from tidescout.config import load_fishery, load_species
from tidescout.models import Fishery
from tidescout.paths import FISHERIES_DIR

FACTORS = {
    "flow", "stage", "light", "solunar", "pressure", "wind", "water_temp",
    "salinity", "season",
}


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
        if path.name == "species_weights.yaml":
            continue  # per-species SpeciesProfile map, not a Fishery document
        raw = yaml.safe_load(path.read_text())
        Fishery.model_validate(raw)  # raises ValidationError on any extra key


# -- CDMO water-quality stations and the salt-intrusion axis ------------------


def test_cdmo_water_sensors_are_declarable():
    """`kind: cdmo` is a real sensor kind, not a typo. The CDMO stations are
    a manual historical backfill rather than a polled feed, but the fishery
    still has to declare WHICH stations belong to it -- the calibration path
    reads that list to decide what to look for in the store."""
    from tidescout.models import WaterSensor

    w = WaterSensor(kind="cdmo", station="NIWTAWQ", params=["SAL"], in_domain=True)
    assert w.kind == "cdmo"
    assert w.off_axis is False


def test_off_axis_marks_a_station_on_a_different_branch():
    from tidescout.models import WaterSensor

    w = WaterSensor(
        kind="cdmo", station="NIWCBWQ", params=["SAL"], off_axis=True, in_domain=True
    )
    assert w.off_axis is True


def test_in_domain_has_no_default_and_must_be_declared():
    """2026-08-26 re-review: `WaterSensor.in_domain` was made required, not
    defaulted, for the same reason `SalinityReading.fitted` (Minor 5, same
    review) was -- a `True` default on this field would silently reproduce
    Important 1's original bug for the next station a human declares.
    """
    from tidescout.models import WaterSensor

    with pytest.raises(ValidationError):
        WaterSensor(kind="usgs", station="00000000", params=["00480"])


def test_winyah_declares_all_six_nerrs_water_quality_stations():
    """All six are declared and all six are stored; what differs is whether
    the salt-intrusion fit may READ them."""
    fishery = load_fishery("winyah-bay")
    nerrs = {w.station: w for w in fishery.stations.water if w.kind in ("ndbc", "cdmo")}

    assert set(nerrs) == {
        "WYSS1", "NIWWBWQ", "NIWTAWQ", "NIWCBWQ", "NIWOLWQ", "NIWDCWQ",
    }


def test_north_inlet_stations_are_marked_off_axis_and_bay_stations_are_not():
    """Measured 2026-08-23 over the full 10.6-year record: the three North
    Inlet stations average 31.4-32.0 ppt while the three Winyah Bay stations
    average 6.0-9.6, at along-estuary distances that ORDER them the wrong
    way round once the field is seeded from the bay mouth (North Inlet
    12.88-14.18 km, the bay 16.68-19.03). Both branches respond to discharge
    (r(S, logQ) = -0.66 to -0.79 at all six), so they are not hydrologically
    isolated -- North Inlet takes the Winyah plume on the flood -- but a
    single distance axis cannot carry a 25 ppt baseline offset between two
    branches. They stay stored and are excluded from the fit.
    """
    fishery = load_fishery("winyah-bay")
    by_station = {w.station: w for w in fishery.stations.water}

    for station in ("NIWCBWQ", "NIWOLWQ", "NIWDCWQ"):
        assert by_station[station].off_axis is True, f"{station} is on North Inlet"
    for station in ("WYSS1", "NIWWBWQ", "NIWTAWQ"):
        assert by_station[station].off_axis is False, f"{station} is on the bay"


# -- Species profiles: weights and response curves ----------------------------


def test_all_three_species_load():
    species = load_species()
    assert set(species) == {"redfish", "speckled_trout", "southern_flounder"}


def test_every_species_covers_every_factor():
    """A factor with a weight but no curve silently scores NaN and gets
    excluded, which looks like a dark sensor rather than a config gap."""
    for name, profile in load_species().items():
        assert set(profile.weights) == FACTORS, f"{name} weights"
        assert FACTORS - {"season"} <= set(profile.curves) | {"salinity"}, f"{name} curves"


def test_a_negative_weight_is_rejected_rather_than_inverting_a_factor():
    """`combine` takes a weighted geometric MEAN, so a negative weight makes a
    factor push the score the wrong way and nothing raises -- the log-sum
    absorbs it and returns a plausible number. This file's header invites
    hand-editing, so the typo it invites must fail loudly.

    Zero stays legal: spec section 8 calls solunar "zeroable".
    """
    from tidescout.models import Curve, SpeciesProfile

    good = dict(
        curves={}, salinity=Curve(x=[0.0, 1.0], y=[0.0, 1.0]),
        months=dict.fromkeys(range(1, 13), 1.0), structure_weight=0.2,
        light_cloud_widen=0.35,
    )
    SpeciesProfile(weights={"flow": 0.0}, **good)          # zero is fine
    with pytest.raises(ValidationError, match="must be >= 0"):
        SpeciesProfile(weights={"flow": -0.5}, **good)


def test_a_negative_structure_weight_is_rejected_too():
    """`structure_weight` lives OUTSIDE `weights` (2026-08-26 review,
    Important 2: it cannot join `weights` without breaking `FACTORS`'s
    nine-factor pin), so it needs its own guard rather than inheriting the
    one above by accident. Without this, a validator that only ever looked
    at `self.weights` would silently let a negative `structure_weight`
    through -- the discriminating half of the check just above.
    """
    from tidescout.models import Curve, SpeciesProfile

    good = dict(
        weights={"flow": 0.0}, curves={}, salinity=Curve(x=[0.0, 1.0], y=[0.0, 1.0]),
        months=dict.fromkeys(range(1, 13), 1.0), light_cloud_widen=0.35,
    )
    SpeciesProfile(structure_weight=0.0, **good)          # zero is fine here too
    with pytest.raises(ValidationError, match="must be >= 0"):
        SpeciesProfile(structure_weight=-0.2, **good)


def test_light_cloud_widen_out_of_bounds_is_rejected():
    """2026-08-26 review, Important 2: `light_cloud_widen` moved out of
    Python and into this per-species field, so the same "this file's header
    invites hand-editing, so the typo it invites must fail loudly" argument
    the weight/structure_weight guards above make applies to it too -- a
    value outside [0, 1] can flip `effective`'s sign or make heavier cloud
    read as FARTHER from twilight (see `SpeciesProfile`'s own validator
    docstring).
    """
    from tidescout.models import Curve, SpeciesProfile

    good = dict(
        weights={"flow": 0.0}, curves={}, salinity=Curve(x=[0.0, 1.0], y=[0.0, 1.0]),
        months=dict.fromkeys(range(1, 13), 1.0), structure_weight=0.2,
    )
    SpeciesProfile(light_cloud_widen=0.0, **good)  # boundary values are fine
    SpeciesProfile(light_cloud_widen=1.0, **good)
    with pytest.raises(ValidationError, match="light_cloud_widen must be between 0 and 1"):
        SpeciesProfile(light_cloud_widen=-0.1, **good)
    with pytest.raises(ValidationError, match="light_cloud_widen must be between 0 and 1"):
        SpeciesProfile(light_cloud_widen=1.5, **good)


def test_every_month_has_a_season_modifier():
    for name, profile in load_species().items():
        assert sorted(profile.months) == list(range(1, 13)), name


def test_species_differ_from_one_another():
    """Three identical profiles would mean the species lens does nothing."""
    species = load_species()
    trout = species["speckled_trout"].salinity
    red = species["redfish"].salinity
    assert trout.y != red.y, "trout should be far less salinity-tolerant than redfish"


def test_trout_salinity_curve_penalises_near_fresh_water():
    """Spec section 7: trout ~10-30 ppt, avoid near-fresh."""
    from tidescout.engine.curves import evaluate

    trout = load_species()["speckled_trout"].salinity
    assert evaluate(trout, 2.0) < 0.3
    assert evaluate(trout, 20.0) > 0.85


def test_redfish_tolerate_the_whole_range():
    from tidescout.engine.curves import evaluate

    red = load_species()["redfish"].salinity
    assert all(evaluate(red, s) > 0.4 for s in (2.0, 12.0, 30.0))
