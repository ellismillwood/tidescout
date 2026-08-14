from typing import Literal

from pydantic import BaseModel


class RiverGauge(BaseModel):
    name: str
    usgs_site: str
    weight: float = 1.0
    # Where this river enters the model domain, lon/lat WGS84. Used to seed
    # anuga.Inlet_operator's injection region. None => inflow is not attached
    # for this river (see pipeline/regimes.py::_attach_river_inflows).
    inflow_lonlat: tuple[float, float] | None = None


class WaterSensor(BaseModel):
    kind: Literal["usgs", "coops"]
    station: str
    params: list[str] = []


class Stations(BaseModel):
    tide: list[str] = []
    currents: list[str] = []
    water: list[WaterSensor] = []
    # Added to CO-OPS predictions to convert MLLW -> NAVD88, the bathymetry
    # datum. Resolved from the station's own datums endpoint, not assumed.
    tide_datum_offset_m: float = 0.0


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
    static_wet_level_m: float = 0.0
    # Deliberately NOT FeatureThresholds.shallow_max_m/deep_min_m. Those two
    # drive bar detection; sharing them means retuning bars silently re-buckets
    # the Manning field and changes every simulation. Defaults match the
    # previous shared values so this split is a no-op at introduction.
    zone_shallow_max_m: float = -0.3
    zone_deep_min_m: float = -3.0


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


class ModelDomain(BaseModel):
    """Outer boundary of the hydrodynamic model, authored not inferred.

    Ocean and estuary are hydraulically connected through several inlets, so
    no automatic rule separates them -- see the Plan 3 spike findings. Vertices
    are (x_km, y_km) in the fishery's bathymetry EPSG, listed clockwise.
    """

    polygon_utm_km: list[tuple[float, float]]
    wet_level_m: float = 1.5  # cut the shoreline at highest simulated water
    simplify_m: float = 25.0  # shoreline generalisation before meshing
    clean_cells: int = 3  # morphological close/open radius, in cells


class AnugaConfig(BaseModel):
    base_edge_m: float = 60.0
    jetty_edge_m: float = 15.0
    jetty_radius_m: float = 300.0
    manning_channel: float = 0.022
    manning_flat: float = 0.030
    manning_marsh: float = 0.045
    spin_up_h: float = 6.0
    cycle_h: float = 12.42
    snapshot_minutes: float = 30.0
    mass_tolerance: float = 1e-3  # measured residual is ~4e-4; 1e-6 fails healthy runs
    max_workers: int = 6  # performance cores only -- see Task 11
    mean_range_m: float = 1.5  # amplitude base for range-regime boundary forcing


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
    model_domain: ModelDomain | None = None
    anuga: AnugaConfig = AnugaConfig()
