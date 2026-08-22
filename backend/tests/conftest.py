import pytest

from tidescout.config import load_fishery
from tidescout.sources.cache import Cache


@pytest.fixture
def fishery():
    return load_fishery("winyah-bay")


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "c.sqlite")
