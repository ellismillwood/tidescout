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
