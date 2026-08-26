# Plan 4 Phase 3 — Bite-Score Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn conditions, flow structure and salinity into two numbers a person can act on — an hourly 0–100 bite score per species, and a per-feature activation for the map — each carrying the reason it came out the way it did.

**Architecture:** One factor pipeline, two consumers. `engine/curves.py` evaluates piecewise-linear response curves authored in YAML; `engine/score.py` turns a `DayConditions` hour plus the spatial layers into nine sub-scores, each with a one-line reason, and combines them as a weighted geometric mean so a single dead factor tanks the hour instead of averaging away. The same sub-scores, evaluated against a feature's own `FeatureMetrics` and local salinity, become that feature's activation. Everything is pure; `pipeline/payload.py` is the only part that touches disk.

**Tech Stack:** Python 3.12, numpy 2.5.2, PyYAML, pydantic, typer, pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-tidescout-design.md` — §8 (bite-score engine) is what this plan implements; §7 (salinity effect on scoring), §9 (what the UI renders), §10 (resilience) and §11 (testing) constrain it.

**Depends on:** Phase 1 (all tasks — `FeatureMetrics`, `feature_key`, `CellSchedule`) and Phase 2 (Tasks 1–3, 6 — `SalinityField`, `blend_regimes`, `DischargeSummary.limb`). Phase 2 Task 7 is independent of this plan.

---

## Global Constraints

Identical to Phases 1 and 2 — Python ≥3.12, ruff `["E","F","I","UP","B","DTZ"]` at line-length 100, `make check` green before every commit, `engine/` pure, tests never hit live APIs, phase 0 is LOW water. Plus:

- **Scores are 0–100 integers at the top level, 0–1 floats internally.** Convert once, at the boundary.
- **No factor is ever silently defaulted.** A missing input excludes its factor and renormalises the remaining weights, and the exclusion appears in the output. Spec §8 is explicit about this.
- **Every sub-score carries a human-readable reason.** The UI renders factor bars with text; a sub-score with no explanation is an incomplete implementation, not a stylistic gap.
- **All curves and weights live in `fisheries/species_weights.yaml`.** No response shape is hard-coded in Python. Editing the YAML must change the answer with no code change.
- **A factor that is computed but UNCONSTRAINED must say so, in the reason a person reads.** Added 2026-08-26, after PRs #4-#9. Spec section 8 has only two states, present and missing-and-excluded; salinity needs a third. `SalinityConfig.fitted` is **False** for Winyah Bay: the residual is ~1,159x the observation resolution and, in `SalinityField`'s own words, a caller can get `extrapolated=False` on "a number that no observation anywhere ever constrained". Such a value is INCLUDED at full weight -- the owner's call, 2026-08-26 -- and carries `provisional=True` plus a reason that names it. A confident factor bar over an unconstrained number is the one outcome this project has spent five PRs learning to avoid.

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `fisheries/species_weights.yaml` | Every weight and response curve, per species |
| `backend/tidescout/models.py` (additions) | `SpeciesProfile`, `Curve` |
| `backend/tidescout/engine/curves.py` | Piecewise-linear curve evaluation |
| `backend/tidescout/engine/score.py` | Nine factors, geometric-mean combination, explanations |
| `backend/tidescout/engine/phase.py` | Wall-clock hour → library tide phase |
| `backend/tidescout/pipeline/payload.py` | Assemble the day payload (the only I/O here) |
| `backend/tests/test_curves.py` | Curve evaluation and validation |
| `backend/tests/test_score.py` | Factor behaviour, combination, property tests |
| `backend/tests/test_phase.py` | Phase mapping, including the low-water convention |
| `backend/tests/test_payload.py` | Payload shape and round-trip |

---

## Task 1: Response curves

Every factor maps a real-world quantity to 0–1 through a curve. Curves are **piecewise-linear breakpoints in YAML**, not named formula shapes: a fisherman tuning "what wind speed kills it" should be able to read and edit the numbers directly, and a breakpoint list is trivially testable in a way a parameterised bell is not.

**Files:**
- Create: `backend/tidescout/engine/curves.py`, `backend/tests/test_curves.py`
- Modify: `backend/tidescout/models.py`

**Interfaces:**
- Produces:
  - `Curve` pydantic model — `x: list[float]`, `y: list[float]`
  - `evaluate(curve: Curve, value: float) -> float`
  Every task below consumes `evaluate`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_curves.py
import pytest
from pydantic import ValidationError

from tidescout.engine.curves import evaluate
from tidescout.models import Curve


def test_evaluate_interpolates_between_breakpoints():
    c = Curve(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0])
    assert evaluate(c, 0.5) == pytest.approx(0.5)
    assert evaluate(c, 1.5) == pytest.approx(0.5)
    assert evaluate(c, 1.0) == pytest.approx(1.0)


def test_evaluate_clamps_outside_the_authored_range():
    """A curve authored to 40 knots must not go negative at 60. Extrapolating
    a hand-drawn response curve past its last breakpoint invents a shape
    nobody chose."""
    c = Curve(x=[0.0, 10.0, 40.0], y=[1.0, 0.8, 0.05])
    assert evaluate(c, -5.0) == pytest.approx(1.0)
    assert evaluate(c, 100.0) == pytest.approx(0.05)


def test_curve_rejects_unsorted_breakpoints():
    """np.interp returns silent nonsense for unsorted x rather than raising."""
    with pytest.raises(ValidationError, match="ascending"):
        Curve(x=[0.0, 2.0, 1.0], y=[0.0, 1.0, 0.5])


def test_curve_rejects_mismatched_lengths():
    with pytest.raises(ValidationError, match="same length"):
        Curve(x=[0.0, 1.0], y=[0.0])


def test_curve_rejects_outputs_outside_zero_to_one():
    """Sub-scores are 0-1 by contract; the geometric mean is undefined for
    negatives and a >1 factor would let one input inflate the whole score."""
    with pytest.raises(ValidationError, match="between 0 and 1"):
        Curve(x=[0.0, 1.0], y=[0.0, 1.4])


def test_curve_needs_at_least_two_points():
    with pytest.raises(ValidationError, match="at least two"):
        Curve(x=[1.0], y=[1.0])


def test_evaluate_returns_nan_for_a_missing_input():
    """None means "no data", which must reach the combiner as an exclusion --
    not as 0.0, which means "conditions are terrible"."""
    import math

    c = Curve(x=[0.0, 1.0], y=[0.0, 1.0])
    assert math.isnan(evaluate(c, None))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_curves.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.engine.curves`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/models.py
from pydantic import model_validator


class Curve(BaseModel):
    """A piecewise-linear response curve, authored as breakpoints.

    Deliberately not a parameterised shape ("bell(centre, width)"): the point of
    keeping these in YAML is that Ellis can read what the model believes about
    wind or water temperature and edit it directly, and a list of (x, y) pairs
    is legible in a way a formula is not. It is also exhaustively testable --
    every claim about a curve reduces to an assertion about a number.
    """

    x: list[float]
    y: list[float]

    @model_validator(mode="after")
    def _check(self):
        if len(self.x) != len(self.y):
            raise ValueError("curve x and y must be the same length")
        if len(self.x) < 2:
            raise ValueError("a curve needs at least two points to interpolate")
        if any(b <= a for a, b in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("curve x values must be in strictly ascending order")
        if any(v < 0.0 or v > 1.0 for v in self.y):
            raise ValueError("curve y values must be between 0 and 1")
        return self
```

```python
# backend/tidescout/engine/curves.py
"""Piecewise-linear response curves. Pure."""

import numpy as np

from tidescout.models import Curve


def evaluate(curve: Curve, value: float | None) -> float:
    """Curve value at `value`, clamped to the authored range.

    Clamping rather than extrapolating: a curve authored out to 40 knots has
    nothing to say about 60, and a linear extension would run it negative --
    inventing a response shape nobody chose.

    `None` returns NaN, not 0.0. Missing data and terrible conditions are
    different statements, and spec section 8 requires the first to exclude the
    factor and renormalise rather than score it zero.
    """
    if value is None:
        return float("nan")
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    return float(np.interp(v, curve.x, curve.y))
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_curves.py -v`

```bash
git add backend/tidescout/engine/curves.py backend/tidescout/models.py \
        backend/tests/test_curves.py
git commit -m "feat: piecewise-linear response curves"
```

---

## Task 2: Wall-clock hour → library tide phase

The scoring engine needs to know which flow state an hour corresponds to. The library indexes by phase where **phase 0 is low water**; the runtime knows tide events from CO-OPS. This maps between them.

**Files:**
- Create: `backend/tidescout/engine/phase.py`, `backend/tests/test_phase.py`

**Interfaces:**
- Consumes: `engine.tides.TideEvent`
- Produces: `library_phase(events: list[TideEvent], t: datetime) -> float | None`

**READ THIS BEFORE WRITING THE FUNCTION — a near-twin already exists.** `engine/tides.py:phase_at`
(added by PR #6 on 2026-08-24, EIGHT DAYS after this plan was authored) has the identical signature,
the identical return type, the identical "phase 0 is LOW water" convention and the identical
None-rather-than-guess policy. It is not the same function, and neither may be deleted — but do not
write this one as though the other did not exist.

They differ in what they interpolate between, and therefore in what they RETURN:

* `phase_at` interpolates between BRACKETING EVENTS, so high water is pinned at exactly 0.5.
  `salinity_at` needs that: it evaluates `cos(2*pi*phase)`, and a high water that is not exactly
  0.5 puts the tidal excursion's extreme somewhere other than high water.
* `library_phase` interpolates LOW TO LOW, so high water floats with the real diurnal inequality.
  The flow library needs THAT: its 26 snapshots are spaced uniformly in TIME through a simulated
  cycle, so choosing one is a question about elapsed time, not about tidal state.

Measured 2026-08-26 on a cycle with a 6.5 h flood and a 5.9 h ebb, they disagree by up to **0.024
of a cycle, about 18 minutes** — enough to select the wrong library snapshot, with nothing raising:

| hours after low | `phase_at` | `library_phase` |
|---|---|---|
| 0.00 | 0.0000 | 0.0000 |
| 3.25 | 0.2500 | 0.2621 |
| 6.50 | 0.5000 | 0.5242 |
| 9.45 | 0.7500 | 0.7621 |

Cross-reference `phase_at` in this module's docstring and say which consumer each serves. Two
functions returning "tidal phase in [0, 1)" that quietly disagree are the hazard `_discharge_scale`
was introduced to prevent on the salinity side. Here the answer is NOT one shared helper — the two
consumers genuinely need different numbers — so the divergence gets TESTED rather than merged away.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_phase.py
"""Phase mapping. Phase 0 is LOW water -- the convention every part of the
flow library depends on, and the one Plan 3 documented as the easiest to
invert without noticing."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tidescout.engine.phase import library_phase
from tidescout.engine.tides import TideEvent

TZ = ZoneInfo("America/New_York")
LOW = datetime(2026, 8, 16, 6, 0, tzinfo=TZ)
HIGH = LOW + timedelta(hours=6, minutes=13)     # half of 12.42 h
NEXT_LOW = LOW + timedelta(hours=12, minutes=25)


def _events():
    return [
        TideEvent(time=LOW, height_ft=0.2, kind="L"),
        TideEvent(time=HIGH, height_ft=4.8, kind="H"),
        TideEvent(time=NEXT_LOW, height_ft=0.3, kind="L"),
    ]


def test_phase_is_zero_at_low_water():
    assert library_phase(_events(), LOW) == pytest.approx(0.0, abs=0.01)


def test_phase_is_half_at_high_water():
    assert library_phase(_events(), HIGH) == pytest.approx(0.5, abs=0.02)


def test_phase_is_a_quarter_at_mid_flood():
    """Flood is the FIRST half of the cycle. If this returns 0.75 the ebb/flood
    convention has been inverted and every flow lookup is reading the wrong
    half of the tide."""
    mid_flood = LOW + timedelta(hours=3, minutes=6)
    assert library_phase(_events(), mid_flood) == pytest.approx(0.25, abs=0.02)


def test_phase_is_three_quarters_at_mid_ebb():
    mid_ebb = HIGH + timedelta(hours=3, minutes=6)
    assert library_phase(_events(), mid_ebb) == pytest.approx(0.75, abs=0.02)


def test_phase_wraps_into_the_next_cycle():
    assert library_phase(_events(), NEXT_LOW) == pytest.approx(0.0, abs=0.02)


def test_phase_is_none_before_the_first_low_water():
    """Rather than guessing backwards past the record and silently returning a
    plausible number for an hour we have no tide data for."""
    assert library_phase(_events(), LOW - timedelta(hours=3)) is None


def test_phase_is_none_when_there_are_no_lows():
    events = [TideEvent(time=HIGH, height_ft=4.8, kind="H")]
    assert library_phase(events, HIGH) is None


def test_library_phase_deliberately_disagrees_with_the_salinity_models_phase_at():
    """Both return "tidal phase in [0, 1)" with phase 0 at low water, and they
    are NOT interchangeable -- see this task's note above. `phase_at` brackets
    on EVENTS so high water is pinned at exactly 0.5; this one runs LOW TO LOW
    so high water floats with the real diurnal inequality.

    Pinned so anyone "unifying" the two has to come here and read why they
    differ. Measured 2026-08-26 on a 6.5 h flood / 5.9 h ebb -- deliberately
    NOT `_events()`, which is symmetric and would make the two agree, hiding
    the whole point.
    """
    from tidescout.engine.tides import phase_at

    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=TZ)
    events = [
        TideEvent(time=t0, height_ft=0.0, kind="L"),
        TideEvent(time=t0 + timedelta(hours=6.5), height_ft=5.0, kind="H"),
        TideEvent(time=t0 + timedelta(hours=12.4), height_ft=0.2, kind="L"),
    ]

    for hours, want_event, want_library in (
        (0.00, 0.0000, 0.0000),
        (3.25, 0.2500, 0.2621),
        (6.50, 0.5000, 0.5242),
        (9.45, 0.7500, 0.7621),
    ):
        t = t0 + timedelta(hours=hours)
        assert phase_at(events, t) == pytest.approx(want_event, abs=1e-4)
        assert library_phase(events, t) == pytest.approx(want_library, abs=1e-4)

    # The load-bearing assertion: they must actually DIFFER away from low
    # water. Without it this test would still pass if someone made
    # library_phase delegate straight to phase_at.
    mid = t0 + timedelta(hours=6.5)
    assert abs(library_phase(events, mid) - phase_at(events, mid)) > 0.02

```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_phase.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/phase.py
"""Map a wall-clock instant onto the flow library's tide phase.

The library's phase 0 is LOW water. That is not a convention anyone would guess
-- it falls out of spin_up_h / cycle_h = 6.0 / 12.42 = 0.4831 of a cycle -- and
assuming phase 0 is high water inverts flood and ebb everywhere. Measured
against the shipped config: phase 0.000 -> -0.547 m, 0.250 -> -0.058,
0.500 -> +0.547, 0.750 -> +0.058. Flood is the FIRST half.
"""

from datetime import datetime

from tidescout.engine.tides import TideEvent


def library_phase(events: list[TideEvent], t: datetime) -> float | None:
    """Fraction of the tidal cycle elapsed since the preceding low water.

    Interpolates between the bracketing low waters rather than assuming a fixed
    12.42 h period: real successive cycles differ by tens of minutes, and over a
    24-hour day a fixed period drifts far enough to select the wrong snapshot.

    Returns None -- never a guess -- when `t` falls outside the tide record.
    A plausible wrong phase would be read as fact by everything downstream.
    """
    lows = sorted(e.time for e in events if e.kind == "L")
    if len(lows) < 2:
        return None
    # Redundant with the loop's trailing `return None` TODAY -- `lows` is
    # freshly sorted above, so a `t` outside it can never satisfy `a <= t <= b`
    # for any pair and the loop always falls through. Proved on review over 16
    # boundary cases (duplicate lows, zero-length spans, unsorted input, naive
    # datetimes) and by that argument. Kept as an explicit statement of the
    # domain: it is what still refuses an out-of-record `t` if the loop below
    # is ever changed to wrap or extrapolate. No test can distinguish it.
    if t < lows[0] or t > lows[-1]:
        return None
    for a, b in zip(lows, lows[1:], strict=False):
        if a <= t <= b:
            span = (b - a).total_seconds()
            if span <= 0:
                return None
            return ((t - a).total_seconds() / span) % 1.0
    return None
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_phase.py -v`

```bash
git add backend/tidescout/engine/phase.py backend/tests/test_phase.py
git commit -m "feat: wall-clock hour to library tide phase"
```

---

## Task 3: The species profile and its YAML

Every weight and curve in one committed file, per spec §8's tunability requirement.

**Files:**
- Create: `fisheries/species_weights.yaml`
- Modify: `backend/tidescout/models.py`, `backend/tidescout/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `SpeciesProfile` (`weights: dict[str, float]`, `curves: dict[str, Curve]`, `months: dict[int, float]`, `salinity: Curve`), `load_species(path=None) -> dict[str, SpeciesProfile]`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_config.py (append)
import pytest

from tidescout.config import load_species

FACTORS = {
    "flow", "stage", "light", "solunar", "pressure", "wind", "water_temp",
    "salinity", "season",
}


def test_all_three_species_load():
    species = load_species()
    assert set(species) == {"redfish", "speckled_trout", "southern_flounder"}


def test_every_species_covers_every_factor():
    """A factor with a weight but no curve silently scores NaN and gets
    excluded, which looks like a dark sensor rather than a config gap."""
    for name, profile in load_species().items():
        assert set(profile.weights) == FACTORS, f"{name} weights"
        assert FACTORS - {"season"} <= set(profile.curves) | {"salinity"}, f"{name} curves"


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
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_config.py -k species -v`
Expected: FAIL — `ImportError: cannot import name 'load_species'`.

- [ ] **Step 3: Write the model and loader**

```python
# backend/tidescout/models.py
class SpeciesProfile(BaseModel):
    """One species' lens on the same factor pipeline.

    Weights are relative, not normalised -- the combiner renormalises whatever
    survives after missing factors are dropped, so authoring them as "flow
    matters about twice as much as wind" is the intended style.
    """

    weights: dict[str, float]
    curves: dict[str, Curve]
    salinity: Curve
    months: dict[int, float]
```

```python
# backend/tidescout/config.py (append)
from tidescout.models import SpeciesProfile

def load_species(path: Path | None = None) -> dict[str, SpeciesProfile]:
    p = path or (FISHERIES_DIR / "species_weights.yaml")
    if not p.exists():
        raise FileNotFoundError(f"No species weights at {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    return {k: SpeciesProfile.model_validate(v) for k, v in raw.items()}
```

- [ ] **Step 4: Author the YAML**

```yaml
# fisheries/species_weights.yaml
#
# EVERY NUMBER HERE IS A STARTING POINT, NOT A MEASUREMENT. These curves encode
# conventional SC inshore wisdom and are meant to be tuned against hindcasts of
# days Ellis remembers (Task 7). Edit freely -- nothing in Python hard-codes a
# response shape, so changing a number here changes the answer immediately.
#
# Curves are piecewise-linear breakpoints: `x` is the real-world quantity, `y`
# the 0-1 sub-score. Values outside the authored x range CLAMP to the end y.
#
# Units: flow m/s | stage tide_frac (0 = low water, 0.5 = high) | light hours
# from the nearest of sunrise/sunset | solunar minutes from the nearest
# major/minor | pressure mb change over 3 h | wind knots | water_temp degF |
# salinity ppt.
#
# Weights are RELATIVE, not normalised -- the combiner renormalises whatever
# survives after missing factors are dropped.

redfish:
  weights:
    flow: 1.0
    stage: 0.9
    light: 0.7
    solunar: 0.3        # spec section 8: "small default weight, zeroable"
    pressure: 0.6
    wind: 0.7
    water_temp: 0.8
    salinity: 0.5       # broadly euryhaline -- salinity moves them, rarely stops them
    season: 0.8
  curves:
    flow:
      x: [0.0, 0.10, 0.25, 0.50, 0.90, 1.50]
      y: [0.10, 0.45, 0.85, 1.00, 0.85, 0.45]
    stage:              # peaks late flood: bait pushed onto flooded flats
      x: [0.00, 0.20, 0.40, 0.55, 0.75, 1.00]
      y: [0.45, 0.80, 1.00, 0.90, 0.65, 0.45]
    light:
      x: [0.0, 1.0, 2.0, 4.0, 8.0]
      y: [1.00, 0.85, 0.65, 0.45, 0.35]
    solunar:
      x: [0.0, 30.0, 60.0, 120.0, 360.0]
      y: [1.00, 0.85, 0.72, 0.58, 0.50]
    pressure:
      x: [-4.0, -2.0, -0.5, 0.5, 2.0, 4.0]
      y: [0.90, 1.00, 0.82, 0.68, 0.42, 0.25]
    wind:
      x: [0.0, 5.0, 12.0, 18.0, 25.0, 35.0]
      y: [0.75, 1.00, 0.90, 0.60, 0.25, 0.05]
    water_temp:
      x: [45.0, 55.0, 65.0, 78.0, 88.0, 95.0]
      y: [0.30, 0.60, 0.95, 1.00, 0.70, 0.30]
  salinity:             # tolerant across the whole estuary
    x: [0.0, 2.0, 5.0, 15.0, 30.0, 36.0]
    y: [0.45, 0.60, 0.80, 1.00, 0.90, 0.70]
  months:               # fall bull-red run at the jetties
    {1: 0.55, 2: 0.55, 3: 0.70, 4: 0.85, 5: 0.95, 6: 0.90,
     7: 0.85, 8: 0.90, 9: 1.00, 10: 1.00, 11: 0.90, 12: 0.65}

speckled_trout:
  weights:
    flow: 1.0
    stage: 0.8
    light: 0.9          # markedly more dawn/dusk-driven than redfish
    solunar: 0.3
    pressure: 0.7
    wind: 0.7
    water_temp: 1.0     # cold snaps flip them to winter-hole behaviour
    salinity: 0.9       # spec section 7: ~10-30 ppt, avoid near-fresh
    season: 0.8
  curves:
    flow:
      x: [0.0, 0.10, 0.25, 0.50, 0.90, 1.50]
      y: [0.12, 0.50, 0.95, 1.00, 0.70, 0.30]
    stage:
      x: [0.00, 0.25, 0.50, 0.70, 1.00]
      y: [0.55, 0.95, 1.00, 0.85, 0.55]
    light:
      x: [0.0, 1.0, 2.0, 4.0, 8.0]
      y: [1.00, 0.80, 0.55, 0.35, 0.25]
    solunar:
      x: [0.0, 30.0, 60.0, 120.0, 360.0]
      y: [1.00, 0.85, 0.72, 0.58, 0.50]
    pressure:
      x: [-4.0, -2.0, -0.5, 0.5, 2.0, 4.0]
      y: [0.85, 1.00, 0.80, 0.62, 0.35, 0.20]
    wind:
      x: [0.0, 5.0, 12.0, 18.0, 25.0, 35.0]
      y: [0.80, 1.00, 0.85, 0.50, 0.20, 0.05]
    water_temp:
      x: [40.0, 50.0, 60.0, 75.0, 85.0, 92.0]
      y: [0.15, 0.50, 0.90, 1.00, 0.60, 0.20]
  salinity:             # the sharpest lens of the three -- near-fresh is close to a stop
    x: [0.0, 2.0, 6.0, 10.0, 20.0, 30.0, 36.0]
    y: [0.05, 0.10, 0.45, 0.85, 1.00, 0.90, 0.60]
  months:               # spring and fall aggregations
    {1: 0.50, 2: 0.55, 3: 0.75, 4: 0.95, 5: 1.00, 6: 0.85,
     7: 0.75, 8: 0.75, 9: 0.95, 10: 1.00, 11: 0.90, 12: 0.60}

southern_flounder:
  weights:
    flow: 0.9           # ambush predator: wants moving water, not fast water
    stage: 1.0          # the strongest stage bias of the three
    light: 0.5
    solunar: 0.3
    pressure: 0.5
    wind: 0.6
    water_temp: 1.0     # falling through the low 60s triggers the run
    salinity: 0.6
    season: 1.0
  curves:
    flow:
      x: [0.0, 0.08, 0.20, 0.45, 0.80, 1.50]
      y: [0.15, 0.60, 1.00, 0.80, 0.45, 0.15]
    stage:              # spec section 8: "flounder bias to ebb" -- peaks on the falling half
      x: [0.00, 0.30, 0.55, 0.75, 0.90, 1.00]
      y: [0.50, 0.60, 0.85, 1.00, 0.90, 0.60]
    light:
      x: [0.0, 1.0, 2.0, 4.0, 8.0]
      y: [1.00, 0.90, 0.75, 0.60, 0.50]
    solunar:
      x: [0.0, 30.0, 60.0, 120.0, 360.0]
      y: [1.00, 0.88, 0.78, 0.65, 0.58]
    pressure:
      x: [-4.0, -2.0, -0.5, 0.5, 2.0, 4.0]
      y: [0.85, 0.95, 0.85, 0.75, 0.55, 0.40]
    wind:
      x: [0.0, 5.0, 12.0, 18.0, 25.0, 35.0]
      y: [0.85, 1.00, 0.88, 0.55, 0.22, 0.05]
    water_temp:
      x: [50.0, 58.0, 65.0, 78.0, 86.0, 92.0]
      y: [0.20, 0.70, 1.00, 0.90, 0.50, 0.15]
  salinity:
    x: [0.0, 3.0, 8.0, 18.0, 32.0, 36.0]
    y: [0.20, 0.40, 0.80, 1.00, 0.90, 0.75]
  months:               # the fall run out of the estuary
    {1: 0.35, 2: 0.35, 3: 0.55, 4: 0.75, 5: 0.90, 6: 0.90,
     7: 0.85, 8: 0.90, 9: 1.00, 10: 1.00, 11: 0.70, 12: 0.40}
```

- [ ] **Step 5: Run the tests and commit**

Run: `pytest backend/tests/test_config.py -v && make check`

```bash
git add fisheries/species_weights.yaml backend/tidescout/models.py \
        backend/tidescout/config.py backend/tests/test_config.py
git commit -m "feat: species profiles with tunable weights and curves"
```

---

## Task 4: The nine factors

Each factor takes the hour's conditions and returns a sub-score plus the sentence explaining it.

**Files:**
- Create: `backend/tidescout/engine/score.py`, `backend/tests/test_score.py`

**Interfaces:**
- Consumes: `HourlyConditions`, `DayConditions`, `SpeciesProfile`, `SalinityField`, `curves.evaluate`
- Produces:
  - `SubScore` dataclass — `factor: str`, `value: float`, `weight: float`, `reason: str`, `missing: bool`
  - `score_factors(hour, day, profile, salinity=None, flow_speed=None) -> list[SubScore]`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_score.py
"""Factor behaviour. Each test isolates one factor and asserts the DIRECTION
of its response, not a calibrated value -- the curves are tuned by hindcasting
and every number in them will move."""

import math

from tidescout.config import load_species
from tidescout.engine.conditions import HourlyConditions
from tidescout.engine.score import FACTORS, score_factors


def _hour(**kw):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    base = {"time": datetime(2026, 10, 15, 15, 0, tzinfo=ZoneInfo("America/New_York"))}
    return HourlyConditions(**{**base, **kw})


def _by_factor(subs):
    return {s.factor: s for s in subs}


def _sal(ppt: float, *, fitted: bool = False, extrapolated: bool = False):
    """A salinity reading for tests.

    `fitted=False` is the DEFAULT because it is Winyah Bay's actual state --
    a helper defaulting to True would quietly exercise a configuration this
    project does not have.
    """
    from tidescout.engine.score import SalinityProvenance, SalinityReading

    return SalinityReading(ppt, SalinityProvenance.MODELLED,
                           fitted=fitted, extrapolated=extrapolated)


def test_the_factor_list_and_the_authored_yaml_cannot_drift_apart():
    """Task 3's test file declares its own FACTORS set and this module declares
    a FACTORS tuple. They agree today -- verified 2026-08-26 -- and nothing
    else makes them keep agreeing.

    Drift is silent and expensive in both directions: a factor named here but
    absent from the YAML raises KeyError deep inside `_scored` at scoring time,
    and a factor weighted in the YAML but missing here is simply never
    evaluated, so its weight quietly vanishes from the geometric mean and every
    score shifts with no error anywhere.

    This is the one place that can see both, because Task 3 ships before this
    module exists and its tests cannot import FACTORS from here.
    """
    for name, profile in load_species().items():
        assert set(FACTORS) == set(profile.weights), name


def test_slack_water_craters_the_flow_factor():
    p = load_species()["redfish"]
    slack = _by_factor(score_factors(_hour(), None, p, flow_speed=0.01))["flow"]
    running = _by_factor(score_factors(_hour(), None, p, flow_speed=0.5))["flow"]
    assert slack.value < 0.3
    assert running.value > slack.value


def test_every_sub_score_carries_a_reason():
    """The UI renders factor bars with text; a sub-score with no explanation
    is an incomplete implementation."""
    p = load_species()["redfish"]
    for s in score_factors(_hour(wind_speed_kn=8.0), None, p, flow_speed=0.4):
        assert s.reason, f"{s.factor} has no reason"
        assert not s.reason.endswith(".."), f"{s.factor} reason looks malformed"


def test_missing_input_marks_the_factor_missing_rather_than_scoring_zero():
    """A dark anemometer must not read as dead calm."""
    p = load_species()["redfish"]
    wind = _by_factor(score_factors(_hour(wind_speed_kn=None), None, p))["wind"]
    assert wind.missing is True
    assert "no data" in wind.reason.lower() or "missing" in wind.reason.lower()


def test_strong_wind_suppresses_the_wind_factor():
    p = load_species()["redfish"]
    calm = _by_factor(score_factors(_hour(wind_speed_kn=6.0), None, p))["wind"]
    gale = _by_factor(score_factors(_hour(wind_speed_kn=30.0), None, p))["wind"]
    assert gale.value < calm.value
    assert gale.value < 0.25


def test_falling_pressure_scores_above_sharply_rising_pressure():
    p = load_species()["redfish"]
    falling = _by_factor(score_factors(_hour(pressure_trend_mb_3h=-2.5), None, p))["pressure"]
    rising = _by_factor(score_factors(_hour(pressure_trend_mb_3h=+3.0), None, p))["pressure"]
    assert falling.value > rising.value


def test_near_fresh_water_penalises_trout_far_more_than_redfish():
    """Spec section 7: the same eddy scores near zero up-bay after a freshet."""
    trout = load_species()["speckled_trout"]
    red = load_species()["redfish"]
    t = _by_factor(score_factors(_hour(), None, trout, salinity=_sal(2.0)))["salinity"]
    r = _by_factor(score_factors(_hour(), None, red, salinity=_sal(2.0)))["salinity"]
    assert t.value < r.value
    assert t.value < 0.3


def test_an_uncalibrated_salinity_is_scored_but_marked_provisional():
    """The owner's 2026-08-26 call: include it, flag it. `fitted=False` is
    Winyah Bay's real state, so this is the path that actually runs -- the
    score must still be a number, and the caveat must be in the text a person
    reads, not only in a payload field the UI may not render."""
    red = load_species()["redfish"]
    sub = _by_factor(score_factors(_hour(), None, red, salinity=_sal(22.0)))["salinity"]
    assert sub.missing is False, "unconstrained is not the same as absent"
    assert math.isfinite(sub.value) and 0.0 <= sub.value <= 1.0
    assert sub.weight == red.weights["salinity"], "flagged, not discounted"
    assert sub.provisional is True
    assert "UNCALIBRATED" in sub.reason


def test_a_measured_salinity_carries_no_caveat():
    """The discriminating half. Without this, a bug marking EVERYTHING
    provisional would pass the test above."""
    from tidescout.engine.score import SalinityProvenance, SalinityReading

    red = load_species()["redfish"]
    reading = SalinityReading(22.0, SalinityProvenance.MEASURED)
    sub = _by_factor(score_factors(_hour(), None, red, salinity=reading))["salinity"]
    assert sub.provisional is False
    assert "UNCALIBRATED" not in sub.reason and "~" not in sub.reason


def test_extrapolation_is_disclosed_separately_from_calibration():
    """`fitted` and `extrapolated` answer different questions and a reading can
    fail either independently -- see `SalinityField`'s docstring."""
    red = load_species()["redfish"]
    sub = _by_factor(score_factors(
        _hour(), None, red, salinity=_sal(22.0, fitted=True, extrapolated=True),
    ))["salinity"]
    assert sub.provisional is True
    assert "outside the calibrated range" in sub.reason
    assert "UNCALIBRATED" not in sub.reason, "this one IS fitted"


def test_season_factor_uses_the_month_of_the_hour():
    p = load_species()["redfish"]
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    oct_ = _by_factor(score_factors(
        _hour(time=datetime(2026, 10, 15, 12, tzinfo=tz)), None, p))["season"]
    assert 0.0 <= oct_.value <= 1.0
    assert str(10) in oct_.reason or "octo" in oct_.reason.lower()


def test_all_nine_factors_are_always_present():
    """Even when every input is missing -- the UI shows nine bars, some greyed."""
    p = load_species()["redfish"]
    subs = score_factors(_hour(), None, p)
    assert len(subs) == 9
    assert len({s.factor for s in subs}) == 9
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.engine.score`.

- [ ] **Step 3: Implement `score_factors`**

```python
# backend/tidescout/engine/score.py
"""The factor pipeline. Pure -- conditions in, sub-scores and reasons out.

Two consumers share it: the fishery-wide hourly score and the per-feature
activation. They differ only in where `flow_speed` and `salinity` come
from -- the bay's representative values, or one feature's own.

Every factor obeys the same contract:
  - its weight comes from the species profile, never from code;
  - its response shape comes from a YAML curve, never from code;
  - a missing input yields missing=True and NaN, never 0.0, because "no data"
    and "conditions are dead" are different claims and spec section 8 requires
    the first to renormalise rather than score;
  - it always returns a reason, because the UI renders factor bars with text.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from tidescout.engine.curves import evaluate
from tidescout.models import SpeciesProfile

FACTORS = (
    "flow", "stage", "light", "solunar", "pressure", "wind",
    "water_temp", "salinity", "season",
)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)


class SalinityProvenance(StrEnum):
    """WHERE a salinity number came from, which is not the same question as
    whether it is in range."""

    MEASURED = "measured"   # a sensor read it -- `day.water.salinity_ppt`
    MODELLED = "modelled"   # `engine.salinity` computed it


@dataclass(frozen=True)
class SalinityReading:
    """A salinity value WITH the provenance a scorer needs to be honest.

    A bare float cannot distinguish a sensor reading from an uncalibrated
    model estimate, and this model is uncalibrated: `SalinityConfig.fitted`
    is False for Winyah Bay. Passing a float here would reproduce, at the
    factor level, exactly the confusion `SalinityField` was built to prevent.
    """

    ppt: float
    provenance: SalinityProvenance
    # Did a calibration EVER constrain these parameters. Config-level, so it
    # is identical at every cell and hour. False for Winyah Bay today.
    fitted: bool = True
    # Was THIS evaluation's discharge outside `calibration_range_cfs`.
    # Per-evaluation; changes with the river. Independent of `fitted`.
    extrapolated: bool = False

    @property
    def constrained(self) -> bool:
        """True only when the number is worth stating without a caveat."""
        return self.provenance is SalinityProvenance.MEASURED or (
            self.fitted and not self.extrapolated
        )


@dataclass
class SubScore:
    factor: str
    value: float
    weight: float
    reason: str
    missing: bool
    # Scored, and counted at full weight, but nothing observed constrains it.
    # DISTINCT from `missing`: missing means excluded and renormalised;
    # provisional means included and disclosed. A UI that renders these the
    # same way is not implementing spec section 10.
    provisional: bool = False


def _missing(factor: str, profile: SpeciesProfile, what: str) -> SubScore:
    return SubScore(factor, float("nan"), profile.weights[factor],
                    f"{factor}: no data ({what})", True)


def _scored(factor: str, profile: SpeciesProfile, x: float, reason: str) -> SubScore:
    return SubScore(factor, evaluate(profile.curves[factor], x),
                    profile.weights[factor], reason, False)


def _hours_from_twilight(t: datetime, sun) -> float | None:
    """Hours to the nearer of sunrise and sunset. Zero at the edge of light."""
    if sun is None or sun.sunrise is None or sun.sunset is None:
        return None
    return min(
        abs((t - sun.sunrise).total_seconds()), abs((t - sun.sunset).total_seconds())
    ) / 3600.0


def _minutes_from_solunar(t: datetime, periods) -> float | None:
    """Minutes to the nearest solunar major or minor."""
    if not periods:
        return None
    return min(abs((t - p.start).total_seconds()) for p in periods) / 60.0


def score_factors(
    hour,
    day,
    profile: SpeciesProfile,
    salinity: SalinityReading | None = None,
    flow_speed: float | None = None,
) -> list[SubScore]:
    """All nine sub-scores for one hour. Always nine, some possibly missing."""
    subs: list[SubScore] = []

    # 1. Tidal flow rate. Prefers the flow library's speed; falls back to the
    # CO-OPS current station, which is a single point and cannot describe the
    # bay, but beats nothing.
    speed = flow_speed
    if speed is None and hour.current_speed_kn is not None:
        speed = hour.current_speed_kn * 0.514444
    if speed is None:
        subs.append(_missing("flow", profile, "no flow state or current station"))
    else:
        kind = "slack" if speed < 0.1 else "moving" if speed < 0.8 else "ripping"
        subs.append(_scored("flow", profile, speed, f"flow {speed:.2f} m/s — {kind}"))

    # 2. Tide stage. tide_frac runs 0 at low water to 0.5 at high, matching the
    # flow library's phase convention -- NOT 0 at high water.
    if hour.tide_frac is None:
        subs.append(_missing("stage", profile, "no tide prediction"))
    else:
        half = "flooding" if hour.tide_frac < 0.5 else "ebbing"
        subs.append(_scored("stage", profile, hour.tide_frac,
                            f"tide {hour.tide_frac:.2f} of cycle — {half}"))

    # 3. Light. Cloud cover widens the low-light window, so heavy cloud is
    # credited as bringing the hour closer to twilight.
    #
    # CORRECTED 2026-08-26 (whole-branch review, Important 2): the sample
    # below originally hard-coded `0.35` as a Python literal, species-
    # independent, even though `light`'s WEIGHT and CURVE are both authored
    # per species in the same file. That is the same class of violation as
    # the `structure_ambush` ramp Task 6 had to move into YAML, and it
    # contradicts this plan's own Global Constraints ("All curves and
    # weights live in `fisheries/species_weights.yaml`. No response shape is
    # hard-coded in Python."). The shipped fix reads
    # `profile.light_cloud_widen` -- a new per-species field on
    # `SpeciesProfile`, sibling to `structure_weight`, authored in
    # `species_weights.yaml` at 0.35 for all three species (unchanged
    # default; only its home moved). Reproduced here as it actually ships,
    # not as originally drafted:
    hours_off = _hours_from_twilight(hour.time, getattr(day, "sun", None))
    if hours_off is None:
        subs.append(_missing("light", profile, "no sun times"))
    else:
        cloud = hour.cloud_cover_pct or 0.0
        effective = hours_off * (1.0 - profile.light_cloud_widen * cloud / 100.0)
        note = f", {cloud:.0f}% cloud widens it" if cloud > 50 else ""
        subs.append(_scored("light", profile, effective,
                            f"{hours_off:.1f} h from twilight{note}"))

    # 4. Solunar. Smallest default weight of the nine, per spec section 8.
    mins = _minutes_from_solunar(hour.time, getattr(day, "solunar", None))
    if mins is None:
        subs.append(_missing("solunar", profile, "no solunar periods"))
    else:
        subs.append(_scored("solunar", profile, mins,
                            f"{mins:.0f} min from a solunar period"))

    # 5. Pressure trend.
    if hour.pressure_trend_mb_3h is None:
        subs.append(_missing("pressure", profile, "no pressure trend"))
    else:
        p = hour.pressure_trend_mb_3h
        note = ("falling — pre-frontal feeding window" if p < -0.5
                else "rising sharply — post-frontal shutdown" if p > 2.0
                else "steady")
        subs.append(_scored("pressure", profile, p,
                            f"pressure {p:+.1f} mb/3h — {note}"))

    # 6. Wind.
    if hour.wind_speed_kn is None:
        subs.append(_missing("wind", profile, "no wind forecast"))
    else:
        w = hour.wind_speed_kn
        note = ("calm" if w < 5 else "light" if w < 12
                else "building" if w < 18 else "hard — fishability suffers")
        subs.append(_scored("wind", profile, w, f"wind {w:.0f} kn — {note}"))

    # 7. Water temperature.
    water = getattr(day, "water", None)
    temp = getattr(water, "temp_f", None) if water else None
    if temp is None:
        subs.append(_missing("water_temp", profile, "no water sensor or climatology"))
    else:
        trend = getattr(water, "temp_trend_f_3d", None)
        note = ""
        if trend is not None and abs(trend) >= 1.0:
            note = f", {'warming' if trend > 0 else 'cooling'} {abs(trend):.1f}F/3d"
        subs.append(_scored("water_temp", profile, temp, f"water {temp:.0f}F{note}"))

    # 8. Salinity. Spatial when scoring a feature, bay-representative otherwise.
    # The value is scored the same either way; only the REASON changes, and it
    # must change, because an uncalibrated model estimate and a sensor reading
    # are different claims about the world.
    if salinity is None or not math.isfinite(salinity.ppt):
        subs.append(_missing("salinity", profile, "no salinity estimate"))
    else:
        ppt = salinity.ppt
        note = "near-fresh" if ppt < 5 else "brackish" if ppt < 18 else "salty"
        if salinity.constrained:
            reason = f"salinity {ppt:.1f} ppt — {note}"
        else:
            caveats = []
            if not salinity.fitted:
                caveats.append("UNCALIBRATED model estimate, no observation constrains it")
            if salinity.extrapolated:
                caveats.append("discharge outside the calibrated range")
            # "~" on the number as well as the caveat: the tilde survives
            # truncation in a narrow UI column, the parenthetical may not.
            reason = f"salinity ~{ppt:.1f} ppt — {note} ({'; '.join(caveats)})"
        subs.append(SubScore("salinity", evaluate(profile.salinity, ppt),
                             profile.weights["salinity"], reason, False,
                             provisional=not salinity.constrained))

    # 9. Season. A table lookup, not a curve -- months are discrete.
    month = hour.time.month
    subs.append(SubScore("season", float(profile.months[month]),
                         profile.weights["season"],
                         f"{MONTH_NAMES[month - 1]} (month {month}) seasonal modifier",
                         False))

    return subs
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_score.py -v`

```bash
git add backend/tidescout/engine/score.py backend/tests/test_score.py
git commit -m "feat: nine scoring factors with explanations"
```

---

## Task 5: Combination — the weighted geometric mean

Spec §8: a near-zero critical factor must **tank** the hour, not average away. That is exactly what a geometric mean does and an arithmetic one does not.

**Files:**
- Modify: `backend/tidescout/engine/score.py`
- Test: `backend/tests/test_score.py`

**Interfaces:**
- Produces:
  - `HourScore` dataclass — `score: int` (0–100), `subs: list[SubScore]`, `excluded: list[str]`, `confidence: float`, `provisional: list[str]`, `constrained_share: float`
  - `combine(subs: list[SubScore]) -> HourScore`

**`confidence` is NOT redefined by the provenance work, and that is deliberate.** It stays
`live_weight / total_weight` -- the share of authored weight that survived exclusion -- so every
existing test of it keeps its meaning. But a provisional factor SURVIVES, so on Winyah Bay today an
hour can report `confidence == 1.0` while its heaviest factor (salinity, weight 0.9 for trout) is a
number no observation constrains. Reporting one figure that means both "how much data did we get"
and "how much of it is trustworthy" would collapse two different questions -- the same collapse
`SalinityField` documents at length for `extrapolated` vs `fitted`.

So `combine` reports a SECOND number beside it: `constrained_share`, the share of surviving weight
that is not provisional, and `provisional`, the factor names. Full data with an unconstrained
salinity reads `confidence=1.0, constrained_share=0.74` -- two facts, neither hidden. **Provisional
never alters a weight**; that was the owner's explicit call on 2026-08-26.

- [ ] **Step 1: Write the failing tests**

**This appends to the file Task 4 created.** It uses `pytest.approx`, which Task 4's version does
NOT import — Task 4 deliberately omits it, because an unused `import pytest` is ruff F401 and would
fail `make check` there. Add `import pytest` to the file's EXISTING top import block (its own group,
after `import math` and a blank line). Do not add an import mid-file: that is ruff E402.

```python
# backend/tests/test_score.py (append)
import math

from tidescout.engine.score import SubScore, combine


def _subs(**vals):
    return [SubScore(f, v, 1.0, f"{f} reason", False) for f, v in vals.items()]


def test_score_is_bounded_zero_to_one_hundred():
    for v in (0.0, 0.5, 1.0):
        s = combine(_subs(a=v, b=v, c=v))
        assert 0 <= s.score <= 100


def test_one_dead_factor_tanks_the_hour():
    """The whole reason for a geometric mean. Arithmetic would give 0.67 here."""
    tanked = combine(_subs(a=0.0, b=1.0, c=1.0))
    assert tanked.score < 20


def test_all_good_factors_score_near_one_hundred():
    assert combine(_subs(a=1.0, b=1.0, c=1.0)).score >= 99


def test_missing_factors_are_excluded_and_weights_renormalised():
    """A dark sensor must not drag the score down -- spec section 8 requires
    exclusion with renormalisation, never a silent default."""
    present = combine(_subs(a=0.8, b=0.8))
    with_missing = combine(
        _subs(a=0.8, b=0.8) + [SubScore("c", float("nan"), 1.0, "c: no data", True)]
    )
    assert with_missing.score == present.score
    assert with_missing.excluded == ["c"]


def test_confidence_falls_as_factors_go_missing():
    full = combine(_subs(a=0.8, b=0.8, c=0.8))
    partial = combine(
        _subs(a=0.8, b=0.8) + [SubScore("c", float("nan"), 1.0, "c: no data", True)]
    )
    assert partial.confidence < full.confidence
    assert full.confidence == pytest.approx(1.0)


def test_weights_actually_weight():
    heavy = [SubScore("a", 0.2, 9.0, "", False), SubScore("b", 1.0, 1.0, "", False)]
    light = [SubScore("a", 0.2, 1.0, "", False), SubScore("b", 1.0, 9.0, "", False)]
    assert combine(heavy).score < combine(light).score


def test_a_provisional_factor_keeps_confidence_but_lowers_constrained_share():
    """The two numbers answer different questions. If a regression made
    `constrained_share` an alias of `confidence`, this is what catches it."""
    full = combine([
        SubScore("flow", 0.8, 1.0, "", False),
        SubScore("salinity", 0.6, 1.0, "", False, provisional=True),
    ])
    assert full.confidence == pytest.approx(1.0), "nothing was excluded"
    assert full.constrained_share == pytest.approx(0.5)
    assert full.provisional == ["salinity"]


def test_constrained_share_is_one_when_nothing_is_provisional():
    """The discriminating half -- without it, hardcoding constrained_share to
    0.5 would pass the test above."""
    s = combine([
        SubScore("flow", 0.8, 1.0, "", False),
        SubScore("salinity", 0.6, 1.0, "", False),
    ])
    assert s.constrained_share == pytest.approx(1.0)
    assert s.provisional == []


def test_everything_missing_returns_zero_confidence_not_a_crash():
    s = combine([SubScore("a", float("nan"), 1.0, "a: no data", True)])
    assert s.confidence == 0.0
    assert s.score == 0


def test_score_is_monotone_in_a_single_factor():
    """Property test: improving one input can never lower the score."""
    previous = -1
    for v in [i / 20 for i in range(21)]:
        s = combine(_subs(a=v, b=0.7, c=0.7)).score
        assert s >= previous
        previous = s


def test_zero_is_floored_rather_than_producing_negative_infinity():
    """log(0) is -inf, which would propagate NaN through the whole payload.
    The floor must still tank the score -- it is a guard, not a rescue."""
    s = combine(_subs(a=0.0, b=1.0, c=1.0))
    assert math.isfinite(s.score)
    assert s.score < 20
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_score.py -k "combine or tank or missing or monotone" -v`
Expected: FAIL — `ImportError: cannot import name 'combine'`.

- [ ] **Step 3: Implement**

```python
# backend/tidescout/engine/score.py (append)
import math
from dataclasses import dataclass

# Floor applied to a sub-score before taking its log. log(0) is -inf, which
# would turn the whole day payload into NaN. 1e-3 still tanks the hour -- with
# nine equal weights it pulls a perfect score to about 46 -- so this is a guard
# against a numerical edge, not a rescue from a bad factor.
SCORE_FLOOR = 1e-3


@dataclass
class HourScore:
    score: int              # 0-100
    subs: list[SubScore]
    excluded: list[str]     # factors dropped for missing data
    confidence: float       # share of total authored weight that survived
    # Share of SURVIVING weight that is actually constrained by an observation.
    # Distinct from `confidence`: a provisional factor survives (so it does not
    # move `confidence`) while contributing nothing trustworthy (so it does
    # move this). On Winyah Bay today, a full-data hour reads confidence 1.0
    # and constrained_share well below it, which is the honest pair.
    constrained_share: float
    provisional: list[str]


def combine(subs: list[SubScore]) -> HourScore:
    """Weighted geometric mean of the present factors, as 0-100.

    Geometric, not arithmetic, because spec section 8 requires a near-zero
    critical factor to tank the hour rather than average away: dead slack water
    or a cold shock should not be rescued by a pleasant sky. Arithmetically,
    (0 + 1 + 1)/3 is a respectable 0.67; geometrically it is ~0.

    Missing factors are EXCLUDED and the remaining weights renormalised -- never
    defaulted to a middling value, which would invent data. `confidence` reports
    how much of the authored weight survived, so the UI can show that an hour
    scored 82 on six of nine factors.

    `constrained_share` answers the OTHER question: of the weight that did
    survive, how much rests on something observed. A provisional factor
    (scored, full weight, but unconstrained -- see `SalinityReading`) counts
    toward `confidence` and against this. On Winyah Bay today an all-factors
    hour reads confidence 1.0 with constrained_share well below it, and
    collapsing the two into one number would hide exactly that.
    """
    present = [s for s in subs if not s.missing and math.isfinite(s.value)]
    excluded = [s.factor for s in subs if s.missing or not math.isfinite(s.value)]
    total_weight = sum(s.weight for s in subs) or 1.0
    live_weight = sum(s.weight for s in present)

    if not present or live_weight <= 0:
        return HourScore(
            score=0, subs=subs, excluded=excluded, confidence=0.0,
            constrained_share=0.0, provisional=[],
        )

    log_sum = sum(s.weight * math.log(max(s.value, SCORE_FLOOR)) for s in present)
    value = math.exp(log_sum / live_weight)
    return HourScore(
        score=int(round(100 * min(max(value, 0.0), 1.0))),
        subs=subs,
        excluded=excluded,
        confidence=live_weight / total_weight,
        constrained_share=(
            sum(x.weight for x in present if not x.provisional) / live_weight
        ),
        provisional=[x.factor for x in present if x.provisional],
    )
```

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_score.py -v && make check`

```bash
git add backend/tidescout/engine/score.py backend/tests/test_score.py
git commit -m "feat: weighted geometric-mean combination with renormalisation"
```

---

## Task 6: Per-feature activation

The map half. The same factor pipeline, but a feature's flow comes from its own `FeatureMetrics` and its salinity from its own location — so the identical eddy scores differently up-bay and down-bay after a freshet, which is precisely spec §7's requirement.

**Files:**
- Modify: `backend/tidescout/engine/score.py`
- Test: `backend/tests/test_score.py`

**Interfaces:**
- Consumes: `FeatureMetrics` (Phase 1 Task 9), `SalinityField` (Phase 2 Task 3)
- Produces:
  - `FeatureActivation` dataclass — `key`, `type`, `activation: int`, `subs`, `reason: str`
  - `score_feature(metrics, hour, day, profile, salinity: SalinityReading | None) -> FeatureActivation`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_score.py (append)
from tidescout.engine.activation import FeatureMetrics
from tidescout.engine.score import score_feature


def _metrics(**kw):
    base = dict(
        key="dropoff-abc123", type="dropoff", speed=0.5, ambush=0.4, strain=2e-3,
        okubo_w=-1e-5, convergence=1e-4, wet_fraction=1.0, flood_phase=float("nan"),
        n_cells=42,
        # Required on FeatureMetrics and NOT optional -- omitting it is a
        # TypeError, not a default. Added to the dataclass after this plan was
        # written. 0.0 is the neutral value (no wet disc cell classifies as an
        # eddy) and nothing in this task's scoring reads it: the structure
        # sub-score is built from ambush, okubo_w and convergence.
        eddy_share=0.0,
    )
    return FeatureMetrics(**{**base, **kw})


def test_the_same_feature_scores_lower_in_fresh_water_for_trout():
    """Spec section 7: 'the same eddy scores near zero up-bay after a freshet
    and lights up 5 miles down-bay'."""
    trout = load_species()["speckled_trout"]
    salty = score_feature(_metrics(), _hour(), None, trout, salinity=_sal(22.0))
    fresh = score_feature(_metrics(), _hour(), None, trout, salinity=_sal(1.0))
    assert fresh.activation < salty.activation / 2


def test_a_strong_ambush_pocket_outscores_featureless_water():
    red = load_species()["redfish"]
    strong = score_feature(_metrics(ambush=0.9), _hour(), None, red, salinity=_sal(18.0))
    weak = score_feature(_metrics(ambush=0.0), _hour(), None, red, salinity=_sal(18.0))
    assert strong.activation > weak.activation


def test_a_feature_outside_the_domain_scores_zero_with_an_explanation():
    """n_cells == 0 means the feature has no library cells; it must not be
    silently scored on NaN metrics."""
    red = load_species()["redfish"]
    out = score_feature(_metrics(n_cells=0), _hour(), None, red, salinity=_sal(18.0))
    assert out.activation == 0
    assert "outside" in out.reason.lower() or "no cells" in out.reason.lower()


def test_activation_carries_the_feature_key_unchanged():
    """The frontend keys markers off this; Phase 1 Task 8 made it stable."""
    red = load_species()["redfish"]
    got = score_feature(_metrics(key="bar-9f2c1a7b4e05"), _hour(), None, red,
                        salinity=_sal(18.0))
    assert got.key == "bar-9f2c1a7b4e05"


def test_a_dry_flat_scores_zero_however_good_the_conditions():
    """You cannot fish a flat that has no water on it."""
    red = load_species()["redfish"]
    dry = score_feature(_metrics(type="flat", wet_fraction=0.0, speed=0.0),
                        _hour(), None, red, salinity=_sal(18.0))
    assert dry.activation < 10
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_score.py -k feature -v`
Expected: FAIL — `ImportError: cannot import name 'score_feature'`.

- [ ] **Step 3: Implement**

`score_feature` reuses `score_factors` with the feature's own `speed` as `flow_speed` and its own `salinity` reading, then adds a **structure** sub-score built from `ambush`, `okubo_w` and `convergence` — the Phase 1 fields — before calling `combine`. Return `activation=0` with an explanatory reason when `n_cells == 0`, and gate `flat`-type features on `wet_fraction`.

- [ ] **Step 4: Run the tests and commit**

Run: `pytest backend/tests/test_score.py -v && make check`

```bash
git add backend/tidescout/engine/score.py backend/tests/test_score.py
git commit -m "feat: per-feature activation with spatial salinity"
```

---

## Task 7: The day payload and the hindcast harness

One JSON per `(fishery, date, weather model)`, holding all 24 hours × 3 species pre-scored so the UI's scrubbing never refetches. Then the tuning loop: score a past day and compare against what actually happened.

**Files:**
- Create: `backend/tidescout/pipeline/payload.py`, `backend/tests/test_payload.py`
- Modify: `backend/tidescout/cli.py`

**Interfaces:**
- Produces: `build_payload(slug, day, model_label, cache) -> dict`, `tidescout score <slug> <date> [--species] [--model]`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_payload.py
"""Payload shape. The frontend contract lives here."""

import json


def test_payload_has_24_hours_for_every_species(synthetic_day):
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert set(p["species"]) == {"redfish", "speckled_trout", "southern_flounder"}
    for name, rows in p["species"].items():
        assert len(rows["hours"]) == 24, name


def test_every_hour_carries_its_sub_scores_and_reasons(synthetic_day):
    """Spec section 8: 'why is 3 PM an 82' always has a visible answer."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    hour = p["species"]["redfish"]["hours"][15]
    assert 0 <= hour["score"] <= 100
    assert len(hour["subs"]) == 9
    assert all(s["reason"] for s in hour["subs"])


def test_payload_is_json_serialisable(synthetic_day):
    """NaN is not valid JSON and numpy floats are not serialisable -- both are
    easy to leak from the scoring path."""
    from tidescout.pipeline.payload import build_payload

    text = json.dumps(build_payload(**synthetic_day), allow_nan=False)
    assert "NaN" not in text


def test_payload_records_missing_inputs_at_the_top_level(synthetic_day):
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    assert isinstance(p["missing"], list)
    assert "freshness" in p


def test_every_hour_carries_its_provenance_pair(synthetic_day):
    """The payload is the frontend contract, so the disclosure has to reach it.
    `confidence` and `constrained_share` answer different questions and BOTH
    must be present -- an hour on full data with an uncalibrated salinity reads
    1.0 and something lower, and a UI given only the first cannot tell."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    for name, rows in p["species"].items():
        for h in rows["hours"]:
            assert "confidence" in h and "constrained_share" in h, name
            assert isinstance(h["provisional"], list)
            assert 0.0 <= h["constrained_share"] <= 1.0


def test_an_uncalibrated_salinity_reaches_the_payload_as_provisional(synthetic_day):
    """Winyah Bay ships with `fitted: false`, so this is the live path, not an
    edge case. If the payload ever reports an empty `provisional` list for
    every hour while the fishery is unfitted, the disclosure has been lost
    somewhere between the factor and the JSON."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day)
    flagged = [h for rows in p["species"].values() for h in rows["hours"]
               if "salinity" in h["provisional"]]
    assert flagged, "an unfitted salinity model must surface on some hour"
    assert all(h["constrained_share"] < 1.0 for h in flagged)


def test_payload_flags_an_extrapolated_salinity(synthetic_day_freshet):
    """Spec section 10: degraded inputs surface, they do not hide."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_freshet)
    assert p["salinity"]["extrapolated"] is True


def test_payload_flags_a_clamped_discharge_blend(synthetic_day_freshet):
    """22,996 cfs is 3.7x the highest flow ever simulated. The payload must say
    the flow state was clamped rather than presenting it as a lookup."""
    from tidescout.pipeline.payload import build_payload

    p = build_payload(**synthetic_day_freshet)
    assert p["flow"]["clamped"] is True
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest backend/tests/test_payload.py -v`
Expected: FAIL — `ModuleNotFoundError: tidescout.pipeline.payload`.

- [ ] **Step 3: Implement**

`build_payload` calls **`dayloader.load_day`** — NOT `conditions.assemble_day` directly. `load_day`
is the only caller of `assemble_day` in the codebase: it wraps every source fetch in its own
`attempt()` helper that catches `SourceUnavailable`, records the source name in `missing`, and keeps
going. Calling `assemble_day` yourself means re-implementing that reject-and-report machinery, and
`test_payload_records_missing_inputs_at_the_top_level` is asserting on exactly the list it produces.

**Import the MODULE, not the name:** `from tidescout.sources import dayloader`, then call
`dayloader.load_day(...)`. The fixtures below monkeypatch `dayloader.load_day`, and a
`from ... import load_day` binds the function into `payload`'s namespace at import time, so the
patch would not take and the tests would silently hit the network.

Then it resolves the regime blend with `flow.blend_regimes`, maps each hour to a phase with `phase.library_phase`, loads and blends the flow states, computes `activation.structure_fields`, evaluates salinity per feature from the distance field, then scores all 24 hours × 3 species and all features. Convert NaN to `None` at the JSON boundary.

**The fixtures, verified against the real engine on 2026-08-26 before this task ran.** Add to
`backend/tests/conftest.py`. `build_payload` is called as `build_payload(**synthetic_day)`, so the
fixture yields its KWARGS and monkeypatches the day loader — the payload's own assembly is what is
under test, not the network.

```python
# backend/tests/conftest.py (append)
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

TZ = ZoneInfo("America/New_York")
_MID = datetime(2026, 8, 16, tzinfo=TZ)
_EVENTS = [(0, 0.2, "L"), (6.2, 4.8, "H"), (12.42, 0.3, "L"),
           (18.6, 4.9, "H"), (24.84, 0.2, "L")]


def _day_conditions(cfs: float, bucket: str):
    """A full 24-hour day with EVERY factor live.

    Verified 2026-08-26 against the real `score_factors`/`combine`: nothing is
    excluded on any hour, and the score varies 68..84 across the day. A fixture
    on which the score never moves would pass most payload assertions while
    testing nothing.
    """
    from tidescout.engine.conditions import DayConditions, HourlyConditions
    from tidescout.engine.tides import TideEvent, stage_at
    from tidescout.sources.astronomy import MoonInfo, SolunarPeriod, SunTimes
    from tidescout.sources.usgs import DischargeSummary, WaterSummary

    events = [TideEvent(time=_MID + timedelta(hours=o), height_ft=h, kind=k)
              for o, h, k in _EVENTS]
    hours = []
    for i in range(24):
        t = _MID + timedelta(hours=i)
        st = stage_at(events, t)
        hours.append(HourlyConditions(
            time=t, air_temp_f=82.0, wind_speed_kn=8.0, wind_dir_deg=180.0,
            pressure_mb=1014.0, pressure_trend_mb_3h=-0.8, cloud_cover_pct=30.0,
            tide_height_ft=2.5,
            tide_phase=(st.phase if st else None),
            tide_frac=(round(st.frac, 3) if st else None),
            current_speed_kn=1.2))
    return DayConditions(
        fishery_slug="winyah-bay", day=date(2026, 8, 16),
        model_label="gfs_seamless", hours=hours,
        sun=SunTimes(dawn=_MID + timedelta(hours=6), sunrise=_MID + timedelta(hours=6.5),
                     sunset=_MID + timedelta(hours=20), dusk=_MID + timedelta(hours=20.5)),
        moon=MoonInfo(phase_frac=0.5, rise=_MID + timedelta(hours=19),
                      set=_MID + timedelta(hours=7), transits=[_MID + timedelta(hours=13)]),
        solunar=[SolunarPeriod(kind="major", start=_MID + timedelta(hours=12.5),
                               end=_MID + timedelta(hours=14.5))],
        water=WaterSummary(temp_f=84.0, temp_trend_f_3d=0.4,
                           salinity_ppt=None, source="synthetic"),
        discharge=DischargeSummary(
            cfs_now=cfs, cfs_lagged=cfs * 0.95, bucket=bucket, sites=["02135200"],
            contributing=["02135200"], stale=[], trend=1.05, limb="steady"),
        missing=[])


def _payload_kwargs(monkeypatch, cfs: float, bucket: str):
    from tidescout.sources import dayloader

    monkeypatch.setattr(dayloader, "load_day", lambda *a, **k: _day_conditions(cfs, bucket))
    return dict(slug="winyah-bay", day=date(2026, 8, 16),
                model_label="gfs_seamless", cache=None)


@pytest.fixture
def synthetic_day(monkeypatch):
    """Median flow: 4,200 cfs, inside `calibration_range_cfs` (1232-22996)."""
    return _payload_kwargs(monkeypatch, 4_200.0, "med")


@pytest.fixture
def synthetic_day_freshet(monkeypatch):
    """22,996 cfs -- the TOP of the calibrated range and 3.7x the highest flow
    ever simulated, so this is the fixture that must surface both an
    extrapolated salinity and a clamped regime blend."""
    return _payload_kwargs(monkeypatch, 22_996.0, "freshet")
```

- [ ] **Step 4: Add the CLI and run a real day**

```bash
tidescout score winyah-bay 2026-08-16 --species redfish
```

Expected: 24 rows with scores, factor bars and reasons, plus a top-ranked feature list.

- [ ] **Step 5: Prepare the hindcast log — the OUTCOMES are Ellis's, not the implementer's**

**This step is split, because half of it cannot be done by whoever implements this task.** The
ground truth here is what actually happened on days Ellis fished. No agent has that, and inventing
plausible outcomes to fill the table would be the single most damaging thing that could happen to
this project — every curve would then be tuned against fiction, and the tuning would look rigorous.

The implementer does the half that is mechanical:

1. Pick three dates spanning the discharge range (one near median flow, one high, one low), run
   `tidescout score winyah-bay <date> --species <each>`, and record the PREDICTED scores.
2. Create `docs/superpowers/plans/2026-08-16-hindcast-log.md` with one row per date and species:
   date, discharge, predicted score, the two or three top factors driving it, `confidence`,
   `constrained_share` — and an **empty** `actual` column plus an empty `notes` column.
3. State in the log's header that no row is usable for tuning until Ellis fills the `actual` column,
   and that `constrained_share` below 1.0 means the salinity factor in that row rests on an
   uncalibrated model.

Then STOP. Do not tune any curve, and do not fill the `actual` column.

**Ellis's homework, recorded here so it is not lost:** pick days he remembers well — ideally one
excellent, one poor, one middling — and fill in what actually happened. Only then is tuning
meaningful, and even then: collect all three before adjusting anything. Tuning against a single day
fits noise, and these curves are the most over-fittable surface in the project.

- [ ] **Step 6: Commit**

```bash
git add backend/tidescout/pipeline/payload.py backend/tests/test_payload.py \
        backend/tests/conftest.py backend/tidescout/cli.py \
        docs/superpowers/plans/2026-08-16-hindcast-log.md
git commit -m "feat: day payload and hindcast harness"
```

---

## Phase 3 Completion Checklist

- [ ] `make check` green; test count ≥ 300
- [ ] All nine factors present in every hour, each with a reason, none silently defaulted
- [ ] A single dead factor tanks the hour; missing factors renormalise and lower confidence
- [ ] An UNCONSTRAINED factor is scored at full weight, marked `provisional`, names the reason in
      its own text, and lowers `constrained_share` WITHOUT lowering `confidence` — the two are
      separate questions and the payload carries both
- [ ] No reason string states an uncalibrated salinity as a bare number; `fitted=False` is Winyah
      Bay's live state, so this is the path that actually runs
- [ ] Score is monotone in each single factor (property test passing)
- [ ] The same feature scores differently for trout up-bay and down-bay
- [ ] Feature keys in the payload match `features.geojson` exactly
- [ ] Payload serialises with `allow_nan=False`
- [ ] Extrapolated salinity and clamped discharge both surface in the payload
- [ ] At least three hindcast days logged, **before** any curve tuning

**Then:** the spec's remaining phases are the API (§3) and the frontend (§9), which is where `frontend-design` and `dataviz` come in — both called for explicitly in the spec.
