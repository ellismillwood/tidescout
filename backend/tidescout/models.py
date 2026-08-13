from typing import Literal

from pydantic import BaseModel


class RiverGauge(BaseModel):
    name: str
    usgs_site: str
    weight: float = 1.0


class WaterSensor(BaseModel):
    kind: Literal["usgs", "coops"]
    station: str
    params: list[str] = []


class Stations(BaseModel):
    tide: list[str] = []
    currents: list[str] = []
    water: list[WaterSensor] = []


class DischargeBuckets(BaseModel):
    low_below_cfs: float
    high_above_cfs: float


class Climatology(BaseModel):
    water_temp_f_by_month: dict[int, float]
    salinity_ppt_by_month: dict[int, float]


class Fishery(BaseModel):
    slug: str
    name: str
    timezone: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    center: tuple[float, float]  # lon, lat
    orientation_deg: float  # direction the bay mouth faces, degrees true
    stations: Stations
    rivers: list[RiverGauge]
    discharge_buckets: DischargeBuckets
    climatology: Climatology
