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
