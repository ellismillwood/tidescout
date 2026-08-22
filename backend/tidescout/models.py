from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiverGauge(BaseModel):
    name: str
    usgs_site: str
    weight: float = 1.0
    # Where this river enters the model domain, lon/lat WGS84. Used to seed
    # anuga.Inlet_operator's injection region. None => inflow is not attached
    # for this river (see pipeline/regimes.py::_attach_river_inflows).
    inflow_lonlat: tuple[float, float] | None = None
    # Fraction of the composite discharge that enters the domain at THIS
    # river's inlet. Distinct from `weight`, which says how this gauge
    # contributes to the composite TOTAL (1.0 = include it in the sum). The
    # two were conflated until 2026-08-16, which split the composite into
    # equal thirds across three rivers whose measured split is 78/13/8.
    # None on every river reproduces the old equal-share behaviour, so a
    # fishery that has not authored shares still runs.
    inflow_share: float | None = None


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
    # CO-OPS physocean station supplying S_ocean. Springmaid Pier is the
    # closest one to Winyah Bay (mdapi probe, 2026-08-16, corrected 2026-08-22
    # -- it is not the only one within 100 km). Its salinity product currently
    # returns no data live-probed 2026-08-22 -- see sources/coops_water.py's
    # module docstring; the field is kept because it is the best documented
    # candidate and fetch_ocean_salinity degrades to None cleanly.
    ocean_salinity: str = ""


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


class StructureThresholds(BaseModel):
    """Derived-structure knobs. Tunable per fishery during validation."""

    # Radius a fish will move to intercept bait. Matches the radius the
    # known-spots validation gate already uses, so a spot that reads as an
    # ambush point in the gate reads as one here too.
    ambush_radius_m: float = 150.0
    # Dead band on |Okubo-Weiss| below which water is "quiet" rather than
    # seam or eddy. Not tuned to make anything pass -- it exists because
    # uniform flow sits at W = 0 and floating-point noise would otherwise
    # sign every cell of a featureless channel at random.
    quiet_w: float = 1e-5
    # Minimum convergence (negative divergence) counted as a bait-pinning
    # front, in s^-1. 1e-4 is ~0.002 m/s of closing speed across a 20 m cell.
    convergence_min: float = 1e-4


class JettySeed(BaseModel):
    name: str
    coords: list[tuple[float, float]]  # lon, lat vertices, >=2


class KnownSpot(BaseModel):
    name: str
    lon: float
    lat: float
    kind_hint: str = ""
    notes: str = ""
    # Machine-readable version of the tide phase the prose notes describe, so
    # the Task 13 validation gate can ASSERT rather than merely display. The
    # notes stay authoritative -- they carry detail this enum cannot, and the
    # gate quotes them when it reports.
    #
    # "" means unspecified, and the gate reports such a spot without passing or
    # failing it: silently treating an unfilled hint as "no expectation met" is
    # how a go/no-go gate turns into a rubber stamp.
    works_on: Literal["ebb", "flood", "slack", ""] = ""


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
    ocean_max_z_m: float = -2.0  # boundary segments below this bed level take the tide
    # Which boundary segments may carry the OCEAN tide. Authored, not inferred:
    # bed depth alone cannot tell a seaward opening from a deep river channel
    # 40 km inland, and inferring it drove the ocean tide into the Pee Dee head
    # and destroyed two library builds. Same lesson as polygon_utm_km itself.
    # (x_km, y_km) in the bathymetry EPSG, clockwise. Empty = no restriction.
    ocean_boundary_utm_km: list[tuple[float, float]] = []
    # islands at least this large become mesh holes (interior_holes) instead of
    # being meshed as land; smaller ones are filled as sub-mesh-scale noise.
    # Lowered 0.05 -> 0.002 2026-08-14: measured against Winyah's real 149
    # enclosed land islands (total area 0.75 km2), 0.05 kept only 6 of them as
    # holes (fills the other 143 -- meshed as water); 0.002 keeps 79. The
    # remaining fill is cheap (total island area is small) and this is a
    # fidelity INCREASE, not a mesh-cost tradeoff -- do not raise it back up.
    min_island_hole_km2: float = 0.002


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
    # Cell size of the stored flow-state library. Deliberately coarser than
    # bathymetry.cell_m (10 m): the naive full-grid float32 layout is ~52 GB
    # for nine regimes x 25 phases x 3 arrays, and 20 m masked to the domain
    # brings that to ~1.8 GB. Still inside the spec's "~10-20 m", and finer
    # than a 60 m base mesh can actually resolve, so nothing real is lost.
    # Matches anuga.jetty_edge_m = 20.0 by intent -- mesh structure finer than
    # the library grid cannot survive into the output.
    library_cell_m: float = 20.0
    # ANUGA writes a full .sww per regime (~170 MB each, 1.5 GB per build)
    # alongside our snap_*.npz. The pipeline consumes only the npz files, so
    # this is pure surplus -- but it is the only full-resolution record of the
    # run, and the frontend will likely want it for flow visualisation that a
    # 20 m masked grid cannot reconstruct. Kept ON, made switchable.
    store_sww: bool = True


class SalinityConfig(BaseModel):
    """Empirical salt-intrusion parameters. Fitted in Phase 2 Task 5.

    The defaults here are theoretical starting points, NOT calibrated values --
    k = 1/3 is the Savenije-family scaling exponent and l0_km is a rough guess
    at Winyah's intrusion length at median flow. Task 5 replaces them and
    records the fit residual alongside.

    S is a bounded logistic (sigmoid) function of the tidally-shifted
    distance, not a clipped exponential -- see `engine/salinity.py`'s module
    docstring for the real-data review that found the clipped form made
    47.40% of Winyah's domain read bit-identical salinity across a 19x
    discharge swing at high water, and why a single length scale could not
    be fixed without trading the mouth's salinity for the head's. Bounds
    below are load-bearing, not decorative: Task 5's fit is an unconstrained
    optimizer, and e.g. `l0_km=-18` produces salinity ABOVE ocean_ppt with no
    error anywhere unless these are enforced at construction.
    """

    model_config = ConfigDict(extra="forbid")

    ocean_ppt: float = Field(default=34.0, gt=0)
    # Along-estuary distance (net of tidal shift) at which salinity crosses
    # 50% of ocean_ppt -- the salt front's POSITION. Fitted.
    l0_km: float = Field(default=18.0, gt=0)
    q0_cfs: float = Field(default=4000.0, gt=0)
    # Power-law exponent: L ~ Q^-k. 1/3 is the theoretical value. Fitted.
    k: float = Field(default=0.33, ge=0)
    # Tidal excursion -- how far the salt field slides over a cycle.
    # u_tidal * T / pi with u ~ 0.5 m/s and T = 12.42 h gives ~7 km.
    excursion_km: float = Field(default=7.0, gt=0)
    # Half-width of the logistic transition, in km -- the salt front's
    # SHARPNESS, independent of l0_km's POSITION. Added because a single
    # length scale is over-constrained: forcing near-fresh (1 ppt) at the
    # real domain's 31.57 km head with a plain exponential forced l0_km down
    # to 8.95 km, which alone cost North Jetty (2.58 km) 8.5 of its 34 ppt.
    # Splitting position from sharpness fixes that -- under the same head
    # constraint here, North Jetty loses 0.01 ppt. 5.0 km is a starting
    # guess, sized so neither the mouth nor the head saturates to
    # bit-identical output across the full calibration range (verified
    # against the real 587,325-cell distance field; see Task 3's report).
    # Fitted in Task 5 alongside the rest.
    front_width_km: float = Field(default=5.0, gt=0)
    # Discharge span the fit was made over, (lo, hi) with lo < hi. Outside
    # it, results are flagged rather than silently trusted.
    calibration_range_cfs: tuple[float, float] = (1232.0, 22996.0)

    @field_validator("calibration_range_cfs")
    @classmethod
    def _calibration_range_is_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not lo < hi:
            raise ValueError(
                f"calibration_range_cfs must be (lo, hi) with lo < hi, got {v!r} -- "
                "a reversed or degenerate range makes every discharge read as "
                "extrapolated, which defeats the flag's whole purpose"
            )
        return v


class Fishery(BaseModel):
    # Verified 2026-08-22 against the real fishery document: winyah-bay.yaml
    # is the only Fishery YAML in the repo and all 16 of its top-level keys
    # are declared below (winyah-bay.known-spots.yaml and .tiles.yaml are
    # parsed by different models and never reach this class), so this is a
    # no-op today. It exists so a typo'd top-level key -- e.g. `salinty:`
    # for `salinity:` -- raises at load time instead of silently falling
    # back to that field's Python defaults. That failure mode is undetectable
    # by inspection once Task 5 has written fitted salinity numbers into the
    # YAML: the block still parses, still looks complete, and the app runs
    # on the unfitted theoretical constants while reporting nothing wrong.
    model_config = ConfigDict(extra="forbid")

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
    structure: StructureThresholds = StructureThresholds()
    jetties: list[JettySeed] = []
    model_domain: ModelDomain | None = None
    anuga: AnugaConfig = AnugaConfig()
    salinity: SalinityConfig = SalinityConfig()

    def branch_shares(self) -> list[float]:
        """Each river's fraction of total inflow, in `self.rivers` order.

        Shared by every caller that splits a composite discharge across
        branches (`pipeline.forcing.river_inflow_m3s`, the ANUGA boundary;
        `sources.usgs.branch_discharge_cfs`, the runtime salinity path) so the
        split logic and its guards exist in exactly one place. Three cases:

          - every river's `inflow_share` is None: fall back to equal shares.
            This is the state a fishery starts in before anyone measures its
            per-river split (see `RiverGauge.inflow_share`) -- not an error.
          - some but not all are None: raise, naming the missing rivers.
            Half a split half-guesses the rest; that must fail loudly.
          - shares are all present but do not sum to 1.0 (within 1e-6): raise.
            Renormalising silently would hide an authoring mistake.
        """
        shares = [r.inflow_share for r in self.rivers]
        if all(s is None for s in shares):
            n = len(self.rivers) or 1
            return [1.0 / n] * len(self.rivers)
        if any(s is None for s in shares):
            missing = [r.name for r in self.rivers if r.inflow_share is None]
            raise ValueError(
                f"inflow_share is set on some rivers but missing on {missing} -- "
                "author it on all of them or none, so the split is never half-guessed"
            )
        total_share = sum(shares)
        if abs(total_share - 1.0) > 1e-6:
            raise ValueError(
                f"inflow_share values sum to {total_share:.4f}, not 1.0 -- "
                "renormalising silently would hide an authoring mistake"
            )
        return shares
