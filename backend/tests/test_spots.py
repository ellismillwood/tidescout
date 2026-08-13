import io
import json

import pytest
from rasterio.warp import transform as warp_transform
from rich.console import Console
from shapely.geometry import LineString, Point

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


def test_spots_cli_finds_exact_nearest_jetty_not_a_farther_decoy(tmp_path, monkeypatch):
    """Regression: a 2-vertex jetty LineString whose segment *interior* (not
    either endpoint) is the true closest point to a spot must still win over
    a decoy feature that only beats the jetty's endpoint-sampled distance.

    Fixture geometry is the real North Jetty / North Jetty rip coordinates
    from fisheries/winyah-bay.yaml and its known-spots.yaml example. True
    point-to-segment distance and the old endpoint-only-sampled distance were
    independently precomputed (see task-10-report.md fix addendum): true
    ~53.4 m, endpoint-sampled ~572.7 m. The decoy sits at exactly 300 m from
    the spot -- strictly between those two numbers -- so it wins under
    endpoint-only sampling (pre-fix) but loses to the jetty's true distance
    (post-fix).
    """
    from tidescout import cli, config, paths

    monkeypatch.setattr(paths, "FISHERIES_DIR", tmp_path)
    monkeypatch.setattr(config, "FISHERIES_DIR", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=120, no_color=True))

    (tmp_path / "jettytown.yaml").write_text(
        "slug: jettytown\nname: Jettytown\ntimezone: America/New_York\n"
        "bbox: [-79.3, 33.1, -79.1, 33.3]\ncenter: [-79.2, 33.2]\n"
        "orientation_deg: 135\nstations: {}\nrivers: []\n"
        "discharge_buckets: {low_below_cfs: 1, high_above_cfs: 2}\n"
        "climatology: {water_temp_f_by_month: {}, salinity_ppt_by_month: {}}\n"
    )
    jetty = [[-79.1850, 33.2320], [-79.1630, 33.2160]]
    decoy_lonlat = [-79.16478223063547, 33.21895256538265]  # exactly 300m from spot
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "jetty-1",
                "properties": {"type": "jetty"},
                "geometry": {"type": "LineString", "coordinates": jetty},
            },
            {
                "type": "Feature",
                "id": "bar-1",
                "properties": {"type": "bar"},
                "geometry": {"type": "Point", "coordinates": decoy_lonlat},
            },
        ],
    }
    data_dir = tmp_path / "data" / "jettytown"
    data_dir.mkdir(parents=True)
    (data_dir / "features.geojson").write_text(json.dumps(fc))
    (tmp_path / "jettytown.known-spots.yaml").write_text(
        "spots:\n  - name: North Jetty rip\n    lon: -79.1680\n    lat: 33.2190\n"
    )

    # Expected true distance, computed independently in the test via shapely
    # (not by importing/reusing the CLI's own distance logic).
    lons = [jetty[0][0], jetty[1][0], -79.1680]
    lats = [jetty[0][1], jetty[1][1], 33.2190]
    xs, ys = warp_transform("EPSG:4326", "EPSG:26917", lons, lats)
    line_utm = LineString([(xs[0], ys[0]), (xs[1], ys[1])])
    spot_utm = Point(xs[2], ys[2])
    expected_m = spot_utm.distance(line_utm)
    assert expected_m == pytest.approx(53.44, abs=0.1)  # pin the precomputed number

    cli.spots("jettytown")
    output = buf.getvalue()
    assert "jetty-1" in output
    assert "bar-1" not in output  # only the winning row is printed

    row = next(line for line in output.splitlines() if "jetty-1" in line)
    reported_m = float(row.split("│")[-2].strip().replace(",", ""))
    assert reported_m == pytest.approx(expected_m, abs=2)


def test_spots_cli_missing_features_geojson_prints_friendly_message(tmp_path, monkeypatch):
    from tidescout import cli, config, paths

    monkeypatch.setattr(paths, "FISHERIES_DIR", tmp_path)
    monkeypatch.setattr(config, "FISHERIES_DIR", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=120, no_color=True))

    (tmp_path / "jettytown.yaml").write_text(
        "slug: jettytown\nname: Jettytown\ntimezone: America/New_York\n"
        "bbox: [-79.3, 33.1, -79.1, 33.3]\ncenter: [-79.2, 33.2]\n"
        "orientation_deg: 135\nstations: {}\nrivers: []\n"
        "discharge_buckets: {low_below_cfs: 1, high_above_cfs: 2}\n"
        "climatology: {water_temp_f_by_month: {}, salinity_ppt_by_month: {}}\n"
    )
    (tmp_path / "jettytown.known-spots.yaml").write_text(
        "spots:\n  - name: North Jetty rip\n    lon: -79.1680\n    lat: 33.2190\n"
    )
    # No features.geojson written -- must not raise FileNotFoundError.
    cli.spots("jettytown")
    output = buf.getvalue()
    assert "tidescout features jettytown --rebuild" in output
