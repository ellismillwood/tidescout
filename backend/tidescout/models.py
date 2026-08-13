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


class BathymetryConfig(BaseModel):
    epsg: int = 26917
    cell_m: float = 10.0
    land_elev_m: float = 1.5
    contour_depths_m: list[float] = [-2.0, -5.0, -10.0, -15.0]


class FeatureThresholds(BaseModel):
    dropoff_slope_deg: float = 8.0
    wall_slope_deg: float = 20.0
    # Walls are typed on an upper percentile, not the mean: the polygon's own
    # boundary is cut at dropoff_slope_deg, so its mean slope is structurally
    # incapable of reaching wall_slope_deg. p90 is robust to the one-cell
    # artefacts that nanmax would latch onto.
    wall_slope_estimator: Literal["p90", "max", "mean"] = "p90"
    hole_delta_m: float = 1.5
    hole_min_area_m2: float = 2000.0
    flat_max_slope_deg: float = 1.0
    flat_band_m: tuple[float, float] = (-1.5, 0.5)
    shallow_max_m: float = -0.3
    deep_min_m: float = -3.0
    bar_min_area_m2: float = 1500.0
    mouth_search_radius_m: float = 60.0
    # Upper bounds. A feature larger than this is a basin, not an ambush point;
    # see the 47 km2 bar the real Winyah raster produced.
    bar_max_area_m2: float = 500_000.0     # 0.5 km2
    flat_max_area_m2: float = 2_000_000.0  # 2 km2 -- flats are legitimately broad
    hole_max_area_m2: float = 200_000.0    # 0.2 km2


class JettySeed(BaseModel):
    name: str
    coords: list[tuple[float, float]]  # lon, lat vertices, >=2


class KnownSpot(BaseModel):
    name: str
    lon: float
    lat: float
    kind_hint: str = ""
    notes: str = ""


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
    bathymetry: BathymetryConfig = BathymetryConfig()
    features: FeatureThresholds = FeatureThresholds()
    jetties: list[JettySeed] = []
