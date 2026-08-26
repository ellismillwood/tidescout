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
    # "ndbc" added Phase 2 Task 8 for buoy water-quality stations (e.g. WYSS1)
    # -- see sources/ndbc.py. Its `params` are the station's own `.ocean`
    # column headers (DEPTH, OTMP, COND, SAL, ...), not USGS/CO-OPS codes;
    # nothing currently reads this list for "ndbc" sensors -- Task 8 stores
    # and exposes an accumulating history but wires it into no scoring path.
    # "cdmo" added Phase 2 Task 12 for the five NERRS water-quality stations
    # that have no NDBC mirror. Unlike the others this is not a polled feed:
    # CDMO history arrives via `tidescout salinity import-cdmo`, a manual
    # one-shot backfill. The entry still earns its place, because the
    # calibration path reads this list to decide which stations to look for
    # in the store -- a station nobody declares is a station nobody fits.
    kind: Literal["usgs", "coops", "ndbc", "cdmo"]
    station: str
    params: list[str] = []
    # True when a human has independently determined this station sits on a
    # branch the 1-D along-estuary coordinate cannot place: it is stored,
    # served and citable like any other, but the salt-intrusion fit must not
    # read it. Measured on Winyah 2026-08-23 -- North Inlet's three stations
    # average 31.4-32.0 ppt where the bay's own three average 6.0-9.6, at
    # distances that order them the wrong way round (North Inlet 12.88-14.18
    # km, the bay 16.68-19.03). Both branches respond to discharge, so this
    # is not "no signal"; it is a 25 ppt baseline offset that one distance
    # axis cannot carry.
    #
    # As of Task 5 (salinity anchoring), this flag is no longer what DECIDES
    # exclusion -- `pipeline.salinity_fit.is_off_axis` computes that from the
    # station's actual distance to the estuary's main stem, walked through
    # water (`pipeline.estuary.build_stem_distance_field`). This field is now
    # an OVERRIDE that can only ever EXCLUDE, never re-admit a station the
    # computed screen already excluded: `is_off_axis(stem_km, declared=True)`
    # is unconditionally `True` regardless of `stem_km`. A hand flag able to
    # force a station back IN would reintroduce exactly the hand-marking this
    # computed screen exists to remove; one that can only exclude is a safety
    # valve for geometry the criterion gets wrong (e.g. a station on a branch
    # too short or too oddly shaped for the stem walk to catch). Leave this
    # `False` unless you have independently confirmed the station is off the
    # main stem -- the computed screen will still catch it if the geometry
    # agrees, and will report it excluded either way.
    off_axis: bool = False


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
    # returns a CO-OPS {"error": ...} payload live-probed 2026-08-22 -- see
    # sources/coops_water.py's module docstring; the field is kept because it
    # is the best documented candidate. fetch_ocean_salinity raises
    # SourceUnavailable in that case (a real failure), not None -- None is
    # reserved for "the station responded with nothing usable."
    ocean_salinity: str = ""


class DischargeBuckets(BaseModel):
    low_below_cfs: float
    high_above_cfs: float
    # The observed maximum of the composite record -- the flow the `freshet`
    # regime is simulated at, NOT a bucket edge like the two above. Optional
    # because it is a per-fishery measurement and there is no defensible
    # default: guessing one would put a simulated regime at a discharge this
    # river system has never produced. A fishery that leaves it unset simply
    # has no freshet bucket, and `engine.flow.bucket_flows` omits it.
    #
    # Winyah's axis without it spans 2,774-6,292 cfs against an observed
    # 1,232-22,996, so the top of the simulated range sat at the p75 while
    # real freshets run 3.65x past it. That extrapolation is not small:
    # differencing a 22,996 cfs run against production `mean_high` moves the
    # velocity field by 17.20% (per-phase p99), 22x the 0.77% floor of a
    # change known to be negligible.
    freshet_cfs: float | None = None


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
    # Which opening the SALT FRONT advances from, for the along-estuary
    # distance field only. Empty = use ocean_boundary_utm_km, which is the
    # old behaviour and correct for any estuary with one seaward opening.
    #
    # These two fields answer different questions and conflating them is a
    # silent, measured defect. `ocean_boundary_utm_km` says which mesh
    # boundary segments take the TIDE, so every genuine opening belongs in
    # it -- and `mesh.classify_boundary` reads it, so narrowing it would
    # change the hydrodynamics and invalidate a built library. The distance
    # field asks something narrower: which mouth does the salt come in
    # through. Where a domain holds two openings, Dijkstra hands every cell
    # whichever is NEARER, with no error anywhere.
    #
    # Winyah, measured 2026-08-23: the authored ocean polygon spans one
    # contiguous 950-cell coastal strip covering BOTH the bay mouth and the
    # coast in front of North Inlet, so the mid/upper bay measured east
    # through Mud Bay and out North Inlet instead of down the bay past the
    # jetties. Re-seeding from the bay mouth alone: WYSS1 15.03 -> 19.03 km
    # (+27%), Thousand Acre 11.67 -> 16.68 (+43%), Mud Bay Cut 9.47 -> 13.05
    # (+38%); North Jetty (2.58) and Georgetown Lighthouse (5.52) unchanged,
    # since both already routed out the mouth. Stable across northern cuts
    # anywhere in y = 3677.2-3681.6 km, so it is the geometry talking, not
    # the threshold.
    salt_source_boundary_utm_km: list[tuple[float, float]] = []
    # islands at least this large become mesh holes (interior_holes) instead of
    # being meshed as land; smaller ones are filled as sub-mesh-scale noise.
    # Lowered 0.05 -> 0.002 2026-08-14: measured against Winyah's real 149
    # enclosed land islands (total area 0.75 km2), 0.05 kept only 6 of them as
    # holes (fills the other 143 -- meshed as water); 0.002 keeps 79. The
    # remaining fill is cheap (total island area is small) and this is a
    # fidelity INCREASE, not a mesh-cost tradeoff -- do not raise it back up.
    min_island_hole_km2: float = 0.002

    @property
    def salt_source_polygon_utm_km(self) -> list[tuple[float, float]]:
        """The polygon the along-estuary distance field seeds from.

        Falls back to `ocean_boundary_utm_km` so a fishery that has not
        authored a salt source keeps the old behaviour exactly, rather than
        losing its seed to an empty polygon.
        """
        return self.salt_source_boundary_utm_km or self.ocean_boundary_utm_km


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
    # SHARPNESS. Added because a single length scale is over-constrained:
    # forcing near-fresh (1 ppt) at the real domain's 31.57 km head
    # (36.19 km since the 2026-08-23 re-seeding) with a plain exponential
    # forced l0_km down to 8.95 km, which alone cost North Jetty (2.58 km)
    # 8.5 of its 34 ppt. Splitting position from sharpness fixes that --
    # under the same head constraint here, North Jetty loses 0.01 ppt.
    # 5.0 km is a starting guess, sized so neither the mouth nor the head
    # saturates to bit-identical output across the full calibration range
    # (verified against the real 587,325-cell distance field; see Task 3's
    # report). Fitted in Task 5 alongside the rest.
    #
    # NOT independent of l0_km's discharge response, though it IS a
    # separate knob from l0_km's POSITION (that decoupling is what fixes
    # the over-constrained problem above). Since 2026-08-25 this is the
    # front's width AT q0_cfs specifically -- away from that reference it
    # scales as (Q/q0)^-k, the SAME exponent l0_km's L(Q) uses (see
    # `engine.salinity._discharge_scale`), because a constant width could
    # not be sharp at high flow and broad at low flow at once.
    front_width_km: float = Field(default=5.0, gt=0)
    # Discharge span the fit was made over, (lo, hi) with lo < hi. Outside
    # it, results are flagged rather than silently trusted.
    calibration_range_cfs: tuple[float, float] = (1232.0, 22996.0)
    # True ONLY when a calibration run produced these values AND raised no
    # warning about its own data. False means every number above is
    # theoretical -- an authored starting point, not a measurement -- and any
    # salinity computed from them carries no observational signal whatsoever.
    #
    # FALSE IS THE SHIPPED STATE FOR WINYAH BAY, and not for want of trying.
    # Task 5 ran the calibration against every observation that exists. Both
    # of the bay's USGS 00480 sites lie OUTSIDE the model domain (1,362 m and
    # 9,498 m from the nearest in-domain cell) and both snap to the same
    # cell, the distance field's maximum at 31.57 km, so every observation
    # available carries ONE along-estuary distance at the extreme fresh end.
    # The 0-31.57 km reach the scoring layer actually reads has none. Fitted
    # anyway on 348 daily means, four parameter sets whose rmse differed by
    # 0.016 ppt -- 60x below the observations' own 1 ppt quantisation --
    # predicted anywhere from 19.8 to 34.0 ppt at North Jetty.
    #
    # This is deliberately SEPARATE from `salinity_field`'s `extrapolated`,
    # which asks a narrower question: was this DISCHARGE inside the span the
    # fit covered. That flag cannot express "no observation ever constrained
    # this cell's DISTANCE" -- true of every cell when this was written
    # (2026-08-23), so a caller checking only `extrapolated` saw green
    # everywhere. As of the salinity-anchoring branch (2026-08-24), WQP
    # anchors mean that is no longer true fishery-wide: `engine.salinity.
    # classify_coverage` reports 78.7% of Winyah's cells MEASURED or
    # INTERPOLATED, not EXTRAPOLATED. `extrapolated` still cannot express
    # per-cell distance coverage -- that is what `SalinityField.coverage`
    # (and its companion `nearest_observed_km`) exist to carry instead, and
    # a caller checking only `extrapolated` still cannot see it. `fitted`
    # remains a property of the CONFIG, identical at every cell -- it says
    # nothing about coverage even where coverage is now good; see
    # `engine.salinity.Coverage`'s docstring for the concrete case of
    # `coverage=MEASURED` coexisting with `fitted=False`.
    fitted: bool = False

    # Timescale, in days, over which the model integrates river discharge.
    # 0.0 means "read today's discharge only", which is what every version
    # before 2026-08-25 did and remains the default so existing configs are
    # unchanged.
    #
    # A salt front does not respond to a single day's flow. Measured on the
    # real record: the residual correlates with discharge averaged over PRIOR
    # days, strengthening with lag then weakening -- at 1/7/14/60 days,
    # -0.06/-0.13/-0.22/-0.15 at the surface sensor and -0.23/-0.39/-0.46/-0.37
    # at the bottom one. The bottom shows it about twice as strongly, which is
    # what a long-memory salt wedge should look like.
    #
    # NOTE the correlation peaks at 14 days but the rmse-minimizing candidate
    # on the tau scan is 7. Those are different questions, and neither
    # settles a value to adopt here: the gate's day-clustered bootstrap
    # (4,017 days, 2,000 reps) found tau=5 vs tau=7 differ by only +0.0115
    # ppt, 95% CI [-0.0010, +0.0237] -- spanning zero, UNRESOLVED. The honest
    # statement is a 5-7 day band, not tau=7 measured to the day (2026-08-25
    # gate report, section 6) -- do not treat 7 as more precise than that.
    # And nothing here is adopted: this field is 0.0, unchanged from every
    # version before 2026-08-25 (see above), and holds NO fitted answer at
    # all -- memory was MEASURED, never enabled.
    discharge_memory_days: float = Field(default=0.0, ge=0.0, le=365.0)

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
