# Salinity Model Form Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the salt front's width scale with discharge, and let the model read a discharge history rather than a single day, removing the two largest measured structures from the residual.

**Architecture:** Both changes are small edits to existing expressions. `intrusion_length_km` and a new `front_width_at` share ONE discharge-scaling helper so they cannot drift apart. Memory is a pure smoothing function over the `{date: cfs}` map the fit already builds, with its timescale stored in `SalinityConfig` so the prediction path smooths identically. `salinity_at`'s signature does not change — callers supply the smoothed discharge.

**Tech Stack:** Python 3.12, numpy, scipy, pydantic v2, typer, rich, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-25-salinity-model-form-design.md`

## Global Constraints

- Run from the repo root. `make check` (= ruff + pytest) green before every commit. Test count only ever goes UP (681 at the start of this plan).
- Python is `$(HOME)/.venvs/tidescout/bin/python`. Never `pip install`.
- Do NOT change `ocean_ppt`, `fitted`, or `calibration_range_cfs` in `fisheries/winyah-bay.yaml`. Task 4 reports; the owner decides.
- Do NOT touch `ocean_boundary_utm_km`, the ANUGA mesh, the flow library, or `ON_AXIS_MAX_KM`.
- Do NOT implement a second layer. Stratification is measured at +3.622 ppt and deliberately deferred — see the spec's §7.
- Reject-and-report: every rejection path carries a counter reaching the CLI.
- **`fitted` will remain False.** 3.42 ppt is still ~1,140x the observation resolution. No commit message, comment, or report may imply these changes calibrate the model.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/tidescout/engine/salinity.py` | **Modify.** Shared discharge scale; `front_width_at`; `salinity_at` uses it. |
| `backend/tidescout/models.py` | **Modify.** `SalinityConfig.discharge_memory_days`. |
| `backend/tidescout/pipeline/salinity_fit.py` | **Modify.** `smooth_discharge`; wire into `collect_observations`; profile τ. |
| `backend/tidescout/cli.py` | **Modify.** Report the memory window, exclusions, and the τ profile. |
| `fisheries/winyah-bay.yaml` | **Modify.** `front_width_km`'s comment only — its MEANING changes. No fitted values. |
| `backend/tests/test_salinity.py` | **Modify.** Shape, smoothing, parity and exclusion tests. |

---

## Task 1: Make the front width scale with discharge

**Files:**
- Modify: `backend/tidescout/engine/salinity.py`
- Modify: `fisheries/winyah-bay.yaml` (comment only)
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `_effective_cfs`, `SalinityConfig`.
- Produces:
  - `_discharge_scale(cfs: float, cfg: SalinityConfig) -> float` — the shared `(Q_eff/q0)^-k`
  - `front_width_at(cfs: float, cfg: SalinityConfig) -> float`
  - `intrusion_length_km` unchanged in behaviour, reimplemented on the shared helper

**Why.** `L(Q)` ranges 37.14 km → 1.13 km across the observed 257x discharge span while `front_width_km` is constant, so the front cannot be sharp at high flow and broad at low flow at once. Measured at fixed distance (removing any confounding with position), the mean residual trends monotonically with flow: −1.33 → −3.72 ppt at x=16.68 km, +1.33 → −2.03 at x=19.03. Making the width carry the same scaling as the length cuts the trend spread from 2.96 to 0.55 ppt.

**The load-bearing detail:** `intrusion_length_km` and `front_width_at` MUST read the same exponent from one place. If one is later changed to a different `k` or a different floor, the two silently diverge and the front's shape stops meaning anything. That is what `_discharge_scale` exists to prevent — do not inline the expression twice.

- [ ] **Step 1: Write the failing tests**

```python
def test_front_width_scales_down_as_discharge_rises():
    """A constant width cannot be sharp at high flow and broad at low flow.
    Measured consequence of the old form: the residual trended -1.33 -> -3.72
    ppt across flow quintiles at a FIXED distance."""
    from tidescout.engine.salinity import front_width_at

    low = front_width_at(1_000.0, CFG)
    high = front_width_at(100_000.0, CFG)
    assert high < low
    assert high < 0.25 * low, "a 100x discharge range must sharpen the front substantially"


def test_front_width_equals_the_authored_value_at_the_reference_discharge():
    """`front_width_km` now means 'the front's width at q0_cfs'. At exactly
    q0 the scaling is 1.0, so the authored number must come back unchanged --
    that is what keeps the config field readable."""
    from tidescout.engine.salinity import front_width_at

    assert front_width_at(CFG.q0_cfs, CFG) == pytest.approx(CFG.front_width_km)


def test_width_and_length_share_one_discharge_scaling():
    """If these ever read different exponents the front's shape stops meaning
    anything, and no output would look wrong. Pinned as a ratio so the test
    survives any future change to the scaling itself."""
    from tidescout.engine.salinity import front_width_at, intrusion_length_km

    for cfs in (500.0, 4_000.0, 30_000.0, 200_000.0):
        ratio = front_width_at(cfs, CFG) / intrusion_length_km(cfs, CFG)
        assert ratio == pytest.approx(CFG.front_width_km / CFG.l0_km)


def test_intrusion_length_is_unchanged_by_the_refactor():
    """`intrusion_length_km` is reimplemented on the shared helper. Its VALUES
    must not move -- l0_km's fitted meaning depends on them."""
    from tidescout.engine.salinity import intrusion_length_km

    for cfs, expected in ((1_000.0, CFG.l0_km * (1_000.0 / CFG.q0_cfs) ** -CFG.k),
                          (CFG.q0_cfs, CFG.l0_km),
                          (50_000.0, CFG.l0_km * (50_000.0 / CFG.q0_cfs) ** -CFG.k)):
        assert intrusion_length_km(cfs, CFG) == pytest.approx(expected)


def test_a_sharper_front_at_high_flow_drops_salinity_faster_with_distance():
    """The point of the change, stated as behaviour rather than parameters:
    at high discharge the profile must fall off over a shorter distance."""
    import numpy as np

    from tidescout.engine.salinity import salinity_at

    x = np.array([2.0, 6.0, 10.0, 14.0])
    lo = salinity_at(x, 2_000.0, 0.25, CFG)
    hi = salinity_at(x, 60_000.0, 0.25, CFG)
    lo_drop = float(lo[0] - lo[-1])
    hi_drop = float(hi[0] - hi[-1])
    assert hi[-1] < lo[-1], "high flow must be fresher far up the estuary"
    assert hi_drop > 0 and lo_drop > 0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k "front_width or discharge_scaling or intrusion_length_is_unchanged or sharper_front"`
Expected: `ImportError: cannot import name 'front_width_at'`

- [ ] **Step 3: Implement**

In `backend/tidescout/engine/salinity.py`:

```python
def _discharge_scale(cfs: float, cfg: SalinityConfig) -> float:
    """`(Q_eff / q0)^-k` -- the one place this scaling is computed.

    BOTH the intrusion length and the front width read it. They used to be
    independent: `L` scaled with discharge while `front_width_km` was a
    constant, and that mismatch was the model's largest systematic error.
    Measured 2026-08-25 at FIXED distance, so it cannot be confounded with
    position: the mean residual ran -1.33 -> -3.72 ppt across flow quintiles
    at x=16.68 km and +1.33 -> -2.03 at x=19.03, because `L` collapses
    37.14 -> 1.13 km across the observed 257x discharge span while the width
    did not move at all.

    Keeping it in one function is not tidiness. If a later change gave these
    two different exponents or different floors, the front's shape would stop
    meaning anything and NO output would look wrong -- the profile would still
    be smooth, monotonic and plausible.
    """
    return (_effective_cfs(cfs) / cfg.q0_cfs) ** (-cfg.k)


def front_width_at(cfs: float, cfg: SalinityConfig) -> float:
    """The salt front's width at this discharge.

    `cfg.front_width_km` is the width AT THE REFERENCE DISCHARGE `q0_cfs`, not
    everywhere -- at q0 the scaling is exactly 1.0 and this returns the
    authored value. A sharper front at high flow is the physical content: the
    same tidal excursion then sweeps a steeper gradient.
    """
    return cfg.front_width_km * _discharge_scale(cfs, cfg)
```

Rewrite `intrusion_length_km`'s body as `return cfg.l0_km * _discharge_scale(cfs, cfg)`, keeping its existing docstring (its explanation of the 1 cfs floor is still true and still valuable).

In `salinity_at`, replace `cfg.front_width_km` in the tanh denominator with `front_width_at(cfs, cfg)`.

- [ ] **Step 4: Re-examine EVERY existing shape test — do not just make them pass**

This changes the model's predictions everywhere. Some existing tests in `tests/test_salinity.py` will fail. For each one, decide whether it pinned something still true under the new form, or something that was only true of a constant width, and say which in your report. **A test adjusted until it goes green, without that judgement, is a test destroyed.**

Also expect `_swing`-related behaviour to shift: it evaluates `salinity_at` at phases 0.0 and 0.5, so a discharge-dependent width changes the modelled tidal swing, most at high flow. That is expected.

- [ ] **Step 5: Update the fishery YAML comment**

`front_width_km: 5.0`'s comment documents it as a constant width sized by a 3-8 km sweep. That meaning is now wrong. Rewrite it to say the value is the width AT `q0_cfs`, that the width scales as `(Q/q0)^-k` away from that reference, and why. Do NOT change the value — it is a theoretical starting point and `fitted` is False.

- [ ] **Step 6: Run the tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/engine/salinity.py backend/tests/test_salinity.py fisheries/winyah-bay.yaml
git commit -m "feat: scale the salt front's width with discharge"
```

---

## Task 2: A discharge memory window

**Files:**
- Modify: `backend/tidescout/models.py`
- Modify: `backend/tidescout/pipeline/salinity_fit.py`
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `composite_discharge_by_day`'s `dict[date, float]`.
- Produces:
  - `SalinityConfig.discharge_memory_days: float = 0.0`
  - `smooth_discharge(by_day: Mapping[date, float], tau_days: float) -> tuple[dict[date, float], int]` — returns the smoothed map and the count of days DROPPED for insufficient history

**Why.** The model reads same-day discharge, but a salt front integrates recent flow. Measured: the residual correlates with discharge averaged over prior days, strengthening with lag then weakening — WYSS1 −0.06/−0.13/−0.22/−0.15 and NIWWBWQ −0.23/−0.39/−0.46/−0.37 at 1/7/14/60 days. The bottom sensor shows it about twice as strongly, which is physically right: the bottom layer carries the long-memory salt wedge.

**A number that must be FITTED, not copied.** The raw correlation peaks at 14 days. The best-fit timescale is **7**. Those are different questions and only the second is being asked — Task 3 profiles it. Do not author 14.

**Insufficient history is a rejection, not a default.** A day whose preceding window is not fully covered by the discharge record must be DROPPED and counted, never smoothed over a short window and treated as equivalent — that would silently make early days' discharge mean something different from later days'.

- [ ] **Step 1: Write the failing tests**

```python
def test_zero_tau_returns_the_discharge_untouched():
    """The backward-compatibility guarantee: every existing caller and test
    must keep getting exactly today's behaviour."""
    from datetime import date

    from tidescout.pipeline.salinity_fit import smooth_discharge

    raw = {date(2026, 5, d): 1000.0 * d for d in range(1, 11)}
    out, dropped = smooth_discharge(raw, 0.0)
    assert out == raw
    assert dropped == 0


def test_a_constant_series_smooths_to_itself():
    """Any weighted mean of a constant is that constant. Catches a
    normalisation bug, which would otherwise show up only as a scale error in
    the fitted parameters."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61)}
    out, dropped = smooth_discharge(raw, 7.0)
    # Pin the POPULATION, not only the values: `all()` over an empty iterable
    # is True, so without these a regression that dropped the whole record
    # would pass this test in silence. tau=7 -> window 28 -> a day needs 29
    # days including itself, so days 29-60 survive.
    assert len(out) == 32
    assert dropped == 28
    assert all(v == pytest.approx(5000.0) for v in out.values())


def test_smoothing_lags_a_step_change():
    """The physical content: a discharge step must reach the model gradually.
    One day after a 10x step, a 7-day memory must have moved well short of
    the new value."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): (1000.0 if d <= 40 else 10000.0) for d in range(1, 61)}
    out, _ = smooth_discharge(raw, 7.0)
    assert out[day(41)] < 4000.0
    assert out[day(41)] > 1000.0
    assert out[day(60)] > out[day(41)]


def test_days_without_enough_history_are_dropped_and_counted():
    """A day smoothed over a SHORT window is not comparable with one smoothed
    over a full window -- its discharge would mean something different. Drop
    and count, never default."""
    from datetime import date

    from tidescout.pipeline.salinity_fit import smooth_discharge

    raw = {date(2026, 5, d): 5000.0 for d in range(1, 31)}
    out, dropped = smooth_discharge(raw, 7.0)
    assert dropped > 0
    assert min(out) > min(raw), "the earliest days cannot have a full window"
    assert dropped + len(out) == len(raw)


def test_a_gap_in_the_discharge_record_drops_the_days_it_covers():
    """A missing gauge day must not be interpolated across -- the composite
    already refuses to sum short when a gauge is dark, and this must not
    quietly undo that."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61) if d != 45}
    out, dropped = smooth_discharge(raw, 7.0)
    assert day(46) not in out
    # 59 days present; a day survives when n >= 29 AND its 28-day lookback
    # clears the hole, i.e. n <= 44. Pinned rather than `> 0` so that a
    # regression dropping most of the record cannot pass.
    assert len(out) == 16
    assert dropped == 43
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k smooth_discharge`
Expected: `ImportError: cannot import name 'smooth_discharge'`

- [ ] **Step 3: Implement**

Add to `backend/tidescout/models.py`'s `SalinityConfig`:

```python
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
    # NOTE the correlation peaks at 14 days but the best FIT is at 7. Those
    # are different questions; this field holds the fitted answer, not the
    # correlation peak.
    discharge_memory_days: float = Field(default=0.0, ge=0.0, le=365.0)
```

Add to `backend/tidescout/pipeline/salinity_fit.py`:

```python
# Windows shorter than this many multiples of tau lose meaningful weight; the
# exponential kernel is truncated here and days without that much history are
# dropped rather than smoothed over a stub.
_MEMORY_WINDOW_TAUS = 4.0


def smooth_discharge(
    by_day: Mapping[date, float], tau_days: float
) -> tuple[dict[date, float], int]:
    """Exponentially-weighted discharge over the preceding `tau_days`.

    Returns the smoothed map and the number of days DROPPED for insufficient
    history. A day whose preceding window is not fully covered is dropped, not
    smoothed over a shorter window: a short window is a different quantity,
    and mixing the two would make early days' discharge mean something other
    than later days'. `composite_discharge_by_day` already refuses to sum a
    day short when a gauge is dark, for the same reason -- this must not
    quietly undo that by averaging across the hole.

    `tau_days == 0.0` returns the input unchanged, which is every caller's
    behaviour before 2026-08-25.
    """
    if tau_days <= 0.0:
        return dict(by_day), 0
    window = int(round(_MEMORY_WINDOW_TAUS * tau_days))
    weights = np.exp(-np.arange(window + 1) / tau_days)
    weights /= weights.sum()
    out: dict[date, float] = {}
    dropped = 0
    for day in sorted(by_day):
        history = [by_day.get(day - timedelta(days=i)) for i in range(window + 1)]
        if any(h is None for h in history):
            dropped += 1
            continue
        out[day] = float(np.dot(weights, history))
    return out, dropped
```

- [ ] **Step 4: Run the tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/models.py backend/tidescout/pipeline/salinity_fit.py backend/tests/test_salinity.py
git commit -m "feat: an exponentially-weighted discharge memory window"
```

---

## Task 3: Wire memory into the fit and profile τ

**Files:**
- Modify: `backend/tidescout/pipeline/salinity_fit.py`
- Modify: `backend/tidescout/cli.py`
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `smooth_discharge` (Task 2), `front_width_at` (Task 1).
- Produces:
  - `CalibrationInput` gains `n_no_discharge_history: int` and `memory_days: float`
  - `profile_memory(...) -> list[tuple[float, float]]` — (tau, rmse) pairs
  - `profile_memory_row_counts(...) -> list[int]` — the row count each candidate tau was
    scored on; exists so a test can prove they are all equal
  - diagnostics gains `memory_days` and `memory_profile`

`smooth_discharge` needs `Mapping` from `collections.abc` and `timedelta` from `datetime`;
add whichever the module does not already import.

**τ is fitted by a PROFILED SCAN, not by adding it to the least-squares vector.** It changes the model's INPUT rather than its shape, so re-smoothing inside every residual evaluation is both slow and needlessly indirect. Fit the other parameters at each τ on a grid, take the best, and report the whole curve — which is exactly the evidence the spec requires, and which a single fitted number would not give.

Grid: at least `[0, 3, 5, 7, 10, 14, 21, 30]` days. The measured optimum is 7 with a sharp penalty either side; 30 was no better than no memory at all.

**Every τ must be scored on the SAME population.** Larger τ drops more early days for insufficient history, so a naive scan compares different row sets and would favour whichever τ discarded the hardest observations. Restrict every candidate to the rows the LARGEST τ retains, and say so in the report.

- [ ] **Step 1: Write the failing tests**

```python
def test_memory_scan_scores_every_tau_on_the_same_rows():
    """Larger tau drops more early days for insufficient history. Scoring each
    tau on whatever it happens to retain would let a tau win by discarding the
    hardest observations rather than by fitting better."""
    from tidescout.pipeline import salinity_fit

    counts = salinity_fit.profile_memory_row_counts(_synthetic_calibration_input(), [0, 7, 30])
    assert len(set(counts)) == 1, f"populations differ across tau: {counts}"


def test_a_longer_memory_drops_more_days_for_insufficient_history():
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    # Day index d maps to a sequential calendar day starting May 1, 2026 (not
    # day-of-month d, which overflows past May's 31 days for d > 31).
    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 5000.0 for d in range(1, 61)}
    # Both taus must leave SURVIVORS -- tau=21 gives a 84-day window against a
    # 60-day record, so it drops everything and the assertion would then hold
    # at saturation rather than because memory length drives exclusions.
    # tau=3 -> window 12 -> 12 dropped; tau=10 -> window 40 -> 40 dropped.
    _, few = smooth_discharge(raw, 3.0)
    _, many = smooth_discharge(raw, 10.0)
    assert many > few
    assert few == 12 and many == 40


def test_calibration_reports_the_memory_window_and_its_exclusions(monkeypatch):
    """A dropped observation must be visible, not inferred from a smaller n."""
    import re

    from typer.testing import CliRunner

    from tidescout.cli import app

    monkeypatch.setattr(
        salinity_fit, "collect_observations",
        lambda *a, **k: salinity_fit.CalibrationInput(
            [(19.03, 4000.0, 6.0), (16.68, 9000.0, 3.0), (19.03, 9000.0, 2.0)],
            [], [], 90, (date(2016, 1, 1), date(2026, 8, 22)), 90,
            n_no_discharge_history=17, memory_days=7.0,
        ),
    )
    result = CliRunner().invoke(app, ["salinity", "calibrate", "winyah-bay"])
    out = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "17" in out
    assert "history" in out.lower()
```

Write `_synthetic_calibration_input()` as a small helper building a `CalibrationInput` with a dated discharge span long enough for a 30-day window — the point is that the row-count check is real, not that the data is realistic.

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k "memory"`
Expected: `AttributeError: module 'tidescout.pipeline.salinity_fit' has no attribute 'profile_memory_row_counts'`

- [ ] **Step 3: Implement**

- `collect_observations` calls `smooth_discharge(by_day, fishery.salinity.discharge_memory_days)` and uses the smoothed map wherever it currently uses `by_day`. Count the dropped days into `n_no_discharge_history` and carry `memory_days` on `CalibrationInput`.
- `profile_memory` fits the other parameters at each candidate τ, on the common row set, and returns `(tau, rmse)` pairs.
- The CLI prints the memory window in use, the exclusion count, and the τ profile as a small table.

- [ ] **Step 4: Prove the fit path and the prediction path smooth identically**

The spec calls this "the failure mode that would silently invalidate every fitted parameter", and
nothing above tests it. If the calibration fits against smoothed discharge while a prediction
caller passes raw same-day discharge, every fitted parameter is applied to a different quantity
than it was fitted to — and nothing errors, because both are just a float.

There is no production caller of `salinity_field` yet, so this is a contract test for the callers
that will exist, and it must fail if that contract is broken:

```python
def test_the_fit_path_routes_its_discharge_through_smooth_discharge(monkeypatch):
    """If `collect_observations` inlines its own smoothing, or ignores
    `discharge_memory_days`, every fitted parameter silently describes a
    different quantity than a prediction caller supplies -- and nothing
    errors, because both are floats.

    This watches the fit path itself. Calling `smooth_discharge` twice and
    comparing the two results would NOT catch it: that only proves a pure
    function is deterministic, which is true of any implementation including
    an inlined duplicate.
    """
    from datetime import datetime, timedelta

    from tidescout.config import load_fishery

    fishery = load_fishery("winyah-bay")
    monkeypatch.setattr(fishery.salinity, "discharge_memory_days", 7.0)

    class _Store:
        def salinity_series(self, station):
            base = datetime(2026, 5, 1, 4, 0, tzinfo=UTC)
            return [(base + timedelta(minutes=15 * i), 10.0 + (i % 8)) for i in range(96)]

    monkeypatch.setattr(salinity_fit, "_open_store", lambda slug: _Store())
    monkeypatch.setattr(
        salinity_fit, "_store_distances",
        lambda slug, fishery, sites: {s: (19.03, 5.0) for s in sites},
    )
    # A discharge record long enough that a 7-day memory (28-day window) has
    # full history for the observation days above.
    by_day = {date(2026, 3, 1) + timedelta(days=i): 4000.0 + 10.0 * i for i in range(70)}
    monkeypatch.setattr(
        salinity_fit, "_usgs_inputs", lambda *a, **k: ({}, by_day, [], {}),
    )
    from tidescout.sources import noaa

    monkeypatch.setattr(noaa, "tide_events_range", lambda *a, **k: [])

    real = salinity_fit.smooth_discharge
    seen_taus: list[float] = []

    def spy(by_day, tau_days):
        seen_taus.append(tau_days)
        return real(by_day, tau_days)

    monkeypatch.setattr(salinity_fit, "smooth_discharge", spy)
    salinity_fit.collect_observations("winyah-bay", fishery, cache=None, days=90)

    assert seen_taus, "collect_observations never called smooth_discharge -- the fit path is reading raw discharge, or has inlined its own smoothing"
    assert seen_taus == [7.0], f"expected the CONFIGURED tau, saw {seen_taus}"


def test_smoothing_at_the_configured_tau_is_not_a_no_op():
    """Guards the test above: if tau were misread as 0 everywhere, the spy
    would still fire and still see the configured value, while the discharge
    reaching the fit was unchanged."""
    from datetime import date, timedelta

    from tidescout.pipeline.salinity_fit import smooth_discharge

    def day(n: int) -> date:
        return date(2026, 5, 1) + timedelta(days=n - 1)

    raw = {day(d): 1000.0 + 100.0 * d for d in range(1, 61)}
    out, _ = smooth_discharge(raw, 7.0)
    target = day(55)
    assert out[target] != pytest.approx(raw[target])
    assert out[target] < raw[target], "a backward-weighted mean of a rising series must lag it"
```

**Why this replaced a tautology.** The first draft of this step asserted
`smooth_discharge(raw, tau) == smooth_discharge(raw, tau)` and called that a parity check between
the fit and prediction paths. It is not: the same pure function called twice returns the same
answer no matter how many duplicate implementations exist elsewhere. It would have passed against
exactly the defect it was written to catch. The spy above fails if the fit path stops calling
`smooth_discharge` or passes a tau other than the configured one, which is the part that is
actually checkable today -- there is still no production caller of `salinity_field`, so the
prediction half of the contract rests on the module note in Step 4's last paragraph, not on a test
pretending to cover it.

Then add a module-level note in `engine/salinity.py` stating that `cfs` means the memory-smoothed
discharge whenever `cfg.discharge_memory_days > 0`, and that a caller passing raw same-day
discharge to a config fitted with memory is making a silent error.

- [ ] **Step 5: Run the tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/pipeline/salinity_fit.py backend/tidescout/engine/salinity.py backend/tidescout/cli.py backend/tests/test_salinity.py
git commit -m "feat: fit the discharge memory timescale by a profiled scan"
```

---

## Task 4: The gate — measure both changes against their pre-registered predictions

**Files:**
- Create: `docs/superpowers/plans/2026-08-25-salinity-model-form-gate-report.md`

**This task measures and reports. It changes no config and decides nothing.**

**The targets below are PRE-REGISTRATIONS, not goals.** They were written before the work, from a
diagnostic run on a different population. Missing one is a valid and useful result. Do not tune,
re-fit, re-bin, or re-scope anything to move a number toward its target — if you find yourself
adjusting a measurement until it agrees, stop and report the disagreement instead. A gate that
reports a miss honestly is doing its job; a gate that hits every target is evidence of nothing.

- [ ] **Step 1: Run the pipeline end to end**

```bash
cd /Users/ellismillwood/Documents/tidescout
$HOME/.venvs/tidescout/bin/tidescout salinity calibrate winyah-bay | tee /tmp/calibrate-form.txt
```

- [ ] **Step 2: Reconcile the populations BEFORE comparing any number**

The pre-registered figures came from a diagnostic on **11,688** observations that map to a
calendar day. The shipped scan reports **12,204** rows, and the headline fit a different count
again. These are not the same denominator, and comparing across them without saying so would make
the whole gate meaningless.

Open the report with a short table: each figure you are about to quote, the population it was
computed on, and whether it is comparable to its pre-registered counterpart. Where it is not,
say so and, where you can, recompute the OLD number on the NEW population rather than the reverse.

- [ ] **Step 3: Define the trend spread ONCE, then compute both sides with it**

**The spec never defines how "discharge-trend spread" is calculated, and it is the gate's primary
criterion.** Do not guess at the original recipe and do not quote `2.96` as though it were
reproducible. Instead:

1. Write the measurement as a small function, in the report or a scratch script — for each of the
   two fixed-distance NERRS stations (x = 16.68 km and x = 19.03 km), bin that station's rows into
   flow quintiles by the discharge the model actually reads, take the mean residual per quintile,
   and let that station's trend be `mean(highest quintile) - mean(lowest quintile)`. Report the
   spread as the mean of the two stations' absolute trends, and print the full 2x5 table of
   quintile means so a reader can recompute anything else they want.
2. Run that SAME function twice: once against the current shipped model, and once against the
   baseline form (constant width, same-day discharge — i.e. `discharge_memory_days = 0.0` and the
   width scaling disabled). **Both numbers must come out of the same code path on the same rows.**
3. Compare those two to each other. Quote the pre-registered `2.96 → ~0.27` alongside, clearly
   labelled as computed by a different method on a different population.

If your baseline recomputation lands far from 2.96, that is a finding about the pre-registration,
not an error to hide — report both numbers and say which one the conclusion rests on.

- [ ] **Step 4: Report against the predictions**

| prediction | target | actual | comparable? |
|---|---|---|---|
| rmse | 4.0875 → ~3.42 | ? | ? |
| **discharge-trend spread** | **2.96 → ~0.27 ppt** | ? | ? |
| fitted tau | ~7 days, worse at 3 and 14 | ? | ? |
| `fitted` | stays False | ? | — |

**The trend spread is the primary criterion, not rmse.** The trend is what these changes remove;
rmse is a side effect. If rmse falls but the trend does not flatten, the change did not do what it
claims and the report must say so in those words.

Report the tau profile as a curve. If tau lands on a grid bound, or the profile is flat within the
scatter of the rmse it is built from, memory is not identifiable on this data — say that plainly
rather than reporting the best grid point as though it were determined. State the margin between
the best and second-best tau and whether it is meaningful.

- [ ] **Step 5: Re-measure stratification, and say whether the spec's prediction held**

The spec predicted the surface/bottom signal would be **LARGER and cleaner** after these changes,
because removing two competing structures makes the depth split easier to see. WYSS1 and NIWWBWQ
share a piling at 19.03 km, so their residual difference is depth alone.

Method, stated so the before/after are comparable: pair the two stations on shared calendar days,
take `mean(bottom residual - surface residual)` over those paired days, with its sd and the share
of that station-pair's residual variance the split explains. Before this work: **+3.622 ppt, sd
2.085, 24.0% of variance**. Report the paired-day count alongside — if it moved, say so, because
the memory window drops early days and that changes which days pair.

If it shrank instead of growing, the ordering argument in the spec was wrong and the next plan
should know that.

- [ ] **Step 6: State what is binding now, and STOP**

Say whether `fitted` can become True (it cannot — the residual is still ~1,140x the observation
resolution) and what the largest remaining structure is. Do NOT edit the `salinity:` block, and do
NOT begin a two-layer model.

- [ ] **Step 7: Commit**

Write the report to `docs/superpowers/plans/2026-08-25-salinity-model-form-gate-report.md`,
matching the convention of `2026-08-24-salinity-anchoring-gate-report.md` and
`2026-08-24-tidal-phase-gate-report.md`. Note that `.superpowers/sdd/` is GITIGNORED — a report
written there is not committable and will be lost.

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add docs/superpowers/plans/2026-08-25-salinity-model-form-gate-report.md
# plus any file you had to touch to surface a number, named explicitly -- never `git add -A`
git commit -m "docs: gate report on the model-form changes"
```

## Completion Checklist

- [ ] `make check` green; test count only ever goes UP (681 at plan start)
- [ ] `front_width_at` and `intrusion_length_km` provably share one discharge scaling
- [ ] `front_width_at(q0_cfs, cfg)` returns the authored `front_width_km` exactly
- [ ] Every pre-existing shape test re-examined with a stated judgement, not adjusted until green
- [ ] `discharge_memory_days = 0.0` reproduces today's behaviour exactly
- [ ] Days without full history are dropped AND counted, never smoothed over a stub
- [ ] Every τ in the profile scored on the same row set
- [ ] The FIT path is proven to route through `smooth_discharge` at the configured tau by a spy
      that fails when the call is removed or the tau hardcoded. The PREDICTION half is NOT
      test-covered -- there is no production caller of `salinity_field` yet -- and rests on the
      module note in `engine/salinity.py`. The report must say so rather than claim parity.
- [ ] Gate report written against the pre-registered predictions; `fisheries/winyah-bay.yaml`'s values unchanged
