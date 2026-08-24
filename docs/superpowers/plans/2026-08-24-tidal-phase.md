# Tidal Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carry each observation's real tidal phase into the salinity fit, so instantaneous grab samples stop being scored as though they were tidal averages.

**Architecture:** `sources/noaa.py` already fetches CO-OPS hi/lo predictions with a permanent cache. A new pure function maps a timestamp to a model phase by interpolating between bracketing events. `fit_intrusion` gains a `phases` sequence parallel to `observations`, mirroring the `sources` argument it already has; `_levels` passes it straight through to `salinity_at`, which **already broadcasts over an array phase** — verified bit-identical to a per-row loop — so the model function itself needs no change beyond a type annotation.

**Tech Stack:** Python 3.12, httpx, numpy, scipy, pydantic v2, typer, rich, sqlite3, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-24-tidal-phase-and-ocean-endmember-design.md`

**SCOPE NOTE — this plan covers phase ONLY.** The spec's §4d (`ocean_ppt`) is deliberately excluded: which of its two routes to take is a decision the spec assigns to the owner at a gate, *after* this work lands and the refit is measured. Writing it here would mean a task with an unresolved fork in it. Task 4 is that gate; `ocean_ppt` gets its own short plan once the route is chosen.

## Global Constraints

- Run everything from the repo root. `make check` (= ruff + pytest) must be green before every commit. Test count only ever goes UP (655 at the start of this plan).
- Python is `$(HOME)/.venvs/tidescout/bin/python`. Never `pip install`.
- Do NOT change any value in the `salinity:` block of `fisheries/winyah-bay.yaml`. Task 4 reports; the owner decides.
- Do NOT touch `ocean_boundary_utm_km`, the ANUGA mesh, the flow library, or `ON_AXIS_MAX_KM`.
- Do NOT free `ocean_ppt` or change the model form. That is the NEXT plan, gated on this one's result.
- Reject-and-report, never silently drop: every rejection path carries a counter that reaches the CLI.
- Timestamps UTC and tz-aware internally. CO-OPS is requested with `time_zone=lst_ldt` and parsed into the fishery's zone — `sources/noaa.py:_parse_t` already does this correctly; do not re-derive it.
- **Phase convention: 0 = LOW water, 0.5 = high water.** `engine/salinity.py:salinity_at`'s docstring states it and the project's recorded conventions pin it. Reversing it inverts the tidal salinity swing across the entire bay.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/tidescout/engine/tides.py` | **Modify.** Add `phase_at(events, t)` — pure, no I/O. |
| `backend/tidescout/sources/noaa.py` | **Modify.** Add `tide_events_range(station, start, end, tz, cache)` for multi-year fetches. |
| `backend/tidescout/pipeline/salinity_fit.py` | **Modify.** `phases` through `fit_intrusion`/`_levels`; compute phases in `collect_observations`. |
| `backend/tidescout/engine/salinity.py` | **Modify.** Type annotation + docstring only — the maths already broadcasts. |
| `backend/tidescout/cli.py` | **Modify.** Report phase coverage and exclusions in `salinity calibrate`. |
| `backend/tests/test_tides.py` | **Modify/Create.** `phase_at` unit tests. |
| `backend/tests/test_noaa.py` | **Modify.** Range-fetch tests. |
| `backend/tests/test_salinity.py` | **Modify.** Fit-level and wiring tests. |

---

## Task 1: Derive a model phase from tide events

**Files:**
- Modify: `backend/tidescout/engine/tides.py`
- Modify: `backend/tests/test_tides.py` (create if absent)

**Interfaces:**
- Consumes: `TideEvent(time: datetime, kind: str, height_ft: float)` — already defined in this file; `kind` is `"H"` or `"L"`.
- Produces: `phase_at(events: Sequence[TideEvent], t: datetime) -> float | None`

**The convention, and why it is not negotiable.** `engine/salinity.py:salinity_at` documents: *"phase 0 is LOW water ... so cos(2*pi*phase) is +1 there, pushing x_eff UP and making a given cell fresher, and -1 at high water. Reversing that sign inverts the tidal salinity swing across the entire bay."* So:

| at | phase |
|---|---|
| low water | 0.0 |
| midway rising | 0.25 |
| high water | 0.5 |
| midway falling | 0.75 |

Interpolate **linearly in time** between the bracketing events: low→high maps 0.0→0.5, high→low maps 0.5→1.0. This is the same approximation `interpolate_tide_hours`/`_cosine_height` already rest on; do not introduce a second tidal model.

Return `None` when `t` is not bracketed by two events — before the first, after the last, or in a gap. **`None` means "cannot determine", and the caller excludes and counts it.** Never return a default.

- [ ] **Step 1: Write the failing tests**

```python
"""Mapping a timestamp to the salinity model's tidal phase.

The convention is load-bearing and stated in `engine/salinity.py`: phase 0
is LOW water, 0.5 is high water. Reversing it inverts the tidal salinity
swing across the entire bay, which no test of the fit itself would catch --
it would simply fit different parameters to compensate.
"""

from datetime import UTC, datetime, timedelta

import pytest

from tidescout.engine.tides import TideEvent, phase_at


def _events():
    """Low at 00:00, high at 06:00, low at 12:00, high at 18:00."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    return [
        TideEvent(base, "L", -0.5),
        TideEvent(base + timedelta(hours=6), "H", 4.0),
        TideEvent(base + timedelta(hours=12), "L", -0.4),
        TideEvent(base + timedelta(hours=18), "H", 4.2),
    ]


def test_low_water_is_phase_zero():
    assert phase_at(_events(), datetime(2026, 5, 1, 0, 0, tzinfo=UTC)) == pytest.approx(0.0)


def test_high_water_is_phase_one_half():
    assert phase_at(_events(), datetime(2026, 5, 1, 6, 0, tzinfo=UTC)) == pytest.approx(0.5)


def test_midway_rising_is_a_quarter():
    assert phase_at(_events(), datetime(2026, 5, 1, 3, 0, tzinfo=UTC)) == pytest.approx(0.25)


def test_midway_falling_is_three_quarters():
    """The falling limb runs 0.5 -> 1.0, so halfway down is 0.75. Getting
    this branch backwards would put ebb water where flood water belongs."""
    assert phase_at(_events(), datetime(2026, 5, 1, 9, 0, tzinfo=UTC)) == pytest.approx(0.75)


def test_the_next_low_wraps_to_zero_not_one():
    """Phase is in [0, 1). A low returning 1.0 would be a different number
    for the same physical state, and cos(2*pi*1.0) == cos(0) makes that
    invisible in the model but visible in any grouping or reporting."""
    p = phase_at(_events(), datetime(2026, 5, 1, 12, 0, tzinfo=UTC))
    assert p == pytest.approx(0.0)


def test_a_time_before_the_first_event_is_undeterminable():
    assert phase_at(_events(), datetime(2026, 4, 30, 23, 0, tzinfo=UTC)) is None


def test_a_time_after_the_last_event_is_undeterminable():
    assert phase_at(_events(), datetime(2026, 5, 1, 19, 0, tzinfo=UTC)) is None


def test_a_gap_in_the_events_is_undeterminable():
    """A missing prediction must not be interpolated across -- a 12-hour
    'interval' spanning a real gap would put phase 0.25 in the middle of
    what was actually a whole missing cycle."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    sparse = [TideEvent(base, "L", -0.5), TideEvent(base + timedelta(hours=30), "H", 4.0)]
    assert phase_at(sparse, base + timedelta(hours=15)) is None


def test_unsorted_events_are_handled():
    """CO-OPS returns sorted data, but a caller concatenating yearly
    fetches can produce unsorted input at the seams."""
    ev = list(reversed(_events()))
    assert phase_at(ev, datetime(2026, 5, 1, 6, 0, tzinfo=UTC)) == pytest.approx(0.5)


def test_two_consecutive_events_of_the_same_kind_are_undeterminable():
    """A missing intervening event: two lows in a row means the high
    between them was not predicted, and the interval is not half a cycle."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    bad = [TideEvent(base, "L", -0.5), TideEvent(base + timedelta(hours=12), "L", -0.4)]
    assert phase_at(bad, base + timedelta(hours=6)) is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_tides.py -q`
Expected: `ImportError: cannot import name 'phase_at'`

- [ ] **Step 3: Implement**

Add to `backend/tidescout/engine/tides.py`:

```python
# The longest interval between consecutive hi/lo predictions that can still
# be treated as half a tidal cycle. The M2 semidiurnal period is 12.42 h, so
# a real half-cycle is ~6.2 h; diurnal inequality stretches some intervals
# past 8 h. Anything beyond this is a GAP in the predictions, not a long
# half-cycle, and interpolating across it would place phase 0.25 in the
# middle of a cycle that was never predicted.
MAX_HALF_CYCLE_H = 9.0


def phase_at(events: Sequence[TideEvent], t: datetime) -> float | None:
    """The salinity model's tidal phase at `t`, or None if undeterminable.

    Phase 0 is LOW water and 0.5 is high water -- the convention
    `engine/salinity.py:salinity_at` documents and depends on. Reversing it
    inverts the tidal salinity swing across the whole bay, and no test of
    the FIT would catch that: least squares would simply choose different
    parameters to compensate.

    Interpolated LINEARLY IN TIME between the bracketing events, the same
    approximation `interpolate_tide_hours` already rests on. Returns a value
    in [0, 1): a low is always 0.0, never 1.0, so one physical state has one
    number.

    None -- never a default -- when `t` is outside the events, falls in a
    gap, or sits between two events of the same kind (which means the event
    between them was not predicted). A fabricated phase is exactly the error
    this codebase already refuses at parse time when it rejects rows with no
    usable timestamp.
    """
    if t.tzinfo is None:
        raise ValueError("phase_at needs a tz-aware timestamp; a naive one silently shifts phase")
    ordered = sorted(events, key=lambda e: e.time)
    for before, after in zip(ordered, ordered[1:], strict=False):
        if not (before.time <= t <= after.time):
            continue
        if before.kind == after.kind:
            return None  # the event between them was not predicted
        span = (after.time - before.time).total_seconds()
        if span <= 0 or span > MAX_HALF_CYCLE_H * 3600.0:
            return None
        frac = (t - before.time).total_seconds() / span
        # low -> high covers 0.0..0.5; high -> low covers 0.5..1.0
        phase = frac * 0.5 if before.kind == "L" else 0.5 + frac * 0.5
        return phase % 1.0
    return None
```

Add `from collections.abc import Sequence` to the imports if absent.

- [ ] **Step 4: Run the tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/engine/tides.py backend/tests/test_tides.py
git commit -m "feat: derive the salinity model's tidal phase from hi/lo predictions"
```

---

## Task 2: Fetch predictions over a multi-year range

**Files:**
- Modify: `backend/tidescout/sources/noaa.py`
- Modify: `backend/tests/test_noaa.py`

**Interfaces:**
- Consumes: `_get_json`, `_parse_t`, `PREDICTION_TTL`, `TideEvent` — all already in this module.
- Produces: `tide_events_range(station: str, start: date, end: date, tz: str, cache: Cache) -> list[TideEvent]`

**Why a range function rather than looping `tide_events`.** The existing `tide_events(station, day, ...)` fetches a 3-day window per call and caches on `hilo:{station}:{begin}:{end}`. The salinity fit needs phases for **1,260 unique dates spanning 1999-2026** — measured, not estimated. Looping the per-day function is 1,260 HTTP calls; fetching **yearly chunks is 28**. `PREDICTION_TTL` is already `None` because predictions are deterministic, so either way it is a one-time cost, but 28 is the right one.

Verified live before this plan was written: CO-OPS station `8662549` returns hi/lo predictions for 1999, 2008 and 2026 — all HTTP 200 with events.

- [ ] **Step 1: Write the failing tests**

```python
def test_tide_events_range_chunks_by_year(monkeypatch):
    """1,260 unique dates over 27 years is 28 yearly calls, not 1,260 daily
    ones. The cache makes both one-time, but not both cheap."""
    from datetime import date

    from tidescout.sources import noaa

    calls = []

    def fake_get_or_fetch(source, key, ttl, fetch):
        calls.append(key)
        return type("C", (), {"payload": {"predictions": []}})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    noaa.tide_events_range("8662549", date(1999, 1, 1), date(2001, 12, 31), "America/New_York", cache)

    assert len(calls) == 3, f"expected one call per year, got {calls}"


def test_tide_events_range_returns_events_in_time_order(monkeypatch):
    """Yearly chunks are concatenated; phase_at sorts defensively, but the
    seam is where unsorted input would first appear."""
    from datetime import date

    from tidescout.sources import noaa

    payloads = {
        "1999": {"predictions": [{"t": "1999-06-01 05:00", "type": "L", "v": "-0.5"}]},
        "2000": {"predictions": [{"t": "2000-06-01 05:00", "type": "H", "v": "4.0"}]},
    }

    def fake_get_or_fetch(source, key, ttl, fetch):
        year = key.split(":")[2][:4]
        return type("C", (), {"payload": payloads[year]})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    out = noaa.tide_events_range("8662549", date(1999, 1, 1), date(2000, 12, 31), "America/New_York", cache)

    assert [e.time for e in out] == sorted(e.time for e in out)
    assert len(out) == 2


def test_tide_events_range_uses_the_permanent_prediction_cache(monkeypatch):
    """Predictions are deterministic; re-fetching 28 years on every run
    would be pure waste."""
    from datetime import date

    from tidescout.sources import noaa

    seen_ttl = []

    def fake_get_or_fetch(source, key, ttl, fetch):
        seen_ttl.append(ttl)
        return type("C", (), {"payload": {"predictions": []}})()

    cache = type("Cache", (), {"get_or_fetch": staticmethod(fake_get_or_fetch)})()
    noaa.tide_events_range("8662549", date(2020, 1, 1), date(2020, 12, 31), "America/New_York", cache)

    assert seen_ttl == [noaa.PREDICTION_TTL]
    assert noaa.PREDICTION_TTL is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_noaa.py -q -k range`
Expected: `AttributeError: module 'tidescout.sources.noaa' has no attribute 'tide_events_range'`

- [ ] **Step 3: Implement**

```python
def tide_events_range(
    station: str, start: date, end: date, tz: str, cache: Cache
) -> list[TideEvent]:
    """Hi/lo predictions across a multi-year span, fetched a year at a time.

    The salinity calibration needs a tidal phase for every grab sample it
    holds -- measured 2026-08-24: 1,260 unique dates spanning 1999-2026.
    Looping `tide_events` would be 1,260 requests; yearly chunks are 28.
    Both are one-time (PREDICTION_TTL is None because predictions are
    deterministic), but only one is neighbourly to a federal service.

    Chunks are keyed per year, so extending the range later re-fetches only
    the years actually added.
    """
    zone = ZoneInfo(tz)
    out: list[TideEvent] = []
    for year in range(start.year, end.year + 1):
        begin = f"{year}0101"
        finish = f"{year}1231"
        params = {
            "product": "predictions",
            "application": "tidescout",
            "station": station,
            "begin_date": begin,
            "end_date": finish,
            "datum": "MLLW",
            "time_zone": "lst_ldt",
            "units": "english",
            "interval": "hilo",
            "format": "json",
        }
        cached = cache.get_or_fetch(
            "coops", f"hilo:{station}:{begin}:{finish}", PREDICTION_TTL, lambda p=params: _get_json(p)
        )
        out.extend(
            TideEvent(_parse_t(p["t"], zone), p["type"], float(p["v"]))
            for p in cached.payload.get("predictions", [])
        )
    out.sort(key=lambda e: e.time)
    return out
```

Note the `lambda p=params:` default-argument binding — a bare `lambda: _get_json(params)` inside a loop captures the *variable*, so every chunk would fetch the last year's parameters. This is a real closure bug, not a style preference.

- [ ] **Step 4: Add a real-CO-OPS-data test**

The spec requires a round trip against a real response, following this repo's standard that a
parser is not trusted until it has met one — the CDMO parser was built from documentation and four
of its inferences were wrong, one catastrophically. Capture a small real response as a fixture and
test `tide_events_range` against it:

```python
def test_tide_events_range_parses_a_real_coops_response():
    """A recorded real response, not a hand-built one. CO-OPS returns naive
    local-time strings with `time_zone=lst_ldt`; parsing them as UTC would
    shift every phase by 4-5 hours -- a third of a tidal cycle."""
    import json
    from datetime import date
    from pathlib import Path

    from tidescout.sources import noaa

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "coops_hilo_1999.json").read_text()
    )
    cache = type("Cache", (), {
        "get_or_fetch": staticmethod(lambda s, k, t, f: type("C", (), {"payload": payload})())
    })()

    out = noaa.tide_events_range("8662549", date(1999, 1, 1), date(1999, 12, 31),
                                 "America/New_York", cache)

    assert out, "the real fixture must yield events"
    assert all(e.time.tzinfo is not None for e in out)
    assert all(e.kind in ("H", "L") for e in out)
    assert out == sorted(out, key=lambda e: e.time)
```

Capture the fixture with:

```bash
curl -sS -G \
  "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter" \
  --data-urlencode "product=predictions" --data-urlencode "application=tidescout" \
  --data-urlencode "station=8662549" --data-urlencode "begin_date=19990101" \
  --data-urlencode "end_date=19990107" --data-urlencode "datum=MLLW" \
  --data-urlencode "time_zone=lst_ldt" --data-urlencode "units=english" \
  --data-urlencode "interval=hilo" --data-urlencode "format=json" \
  -o backend/tests/fixtures/coops_hilo_1999.json
```

- [ ] **Step 5: Run the tests, then fetch for real and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
cd backend && $HOME/.venvs/tidescout/bin/python -c "
from datetime import date
from tidescout.config import load_fishery
from tidescout.sources import noaa
from tidescout.sources.cache import default_cache
f = load_fishery('winyah-bay')
ev = noaa.tide_events_range(f.stations.tide[0], date(1999,1,1), date(2026,12,31), f.timezone, default_cache())
print(f'{len(ev):,} hi/lo events, {ev[0].time.date()} .. {ev[-1].time.date()}')
" 2>&1 | grep -v mpi4py
```

Expected: roughly 40,000 events (about 4 per day over 28 years), spanning 1999 to 2026.

```bash
git add backend/tidescout/sources/noaa.py backend/tests/test_noaa.py
git commit -m "feat: fetch tide predictions across a multi-year range"
```

---

## Task 3: Carry phase into the fit

**Files:**
- Modify: `backend/tidescout/pipeline/salinity_fit.py`
- Modify: `backend/tidescout/engine/salinity.py` (annotation + docstring ONLY)
- Modify: `backend/tests/test_salinity.py`

**Interfaces:**
- Consumes: `phase_at` (Task 1), `tide_events_range` (Task 2).
- Produces:
  - `fit_intrusion(observations, cfg, swings=(), sources=(), phases=())` — `phases` is `Sequence[float]`
  - `CalibrationInput` gains `observation_phases: list[float]` and `n_no_phase: int`
  - diagnostics gains `n_phase_resolved: int`

**`engine/salinity.py` needs NO maths change.** `salinity_at` computes `x + cfg.excursion_km * np.cos(2.0 * np.pi * phase)`; with `phase` an array of the same shape as `distance_km`, this broadcasts. Verified before this plan was written — array-phase output is **bit-identical** to a per-row loop, and scalar phase still works unchanged. Update the parameter's type hint to accept an array and say so in the docstring; change nothing else in that file.

**`phases` aligns with `observations`, NEVER with `swings`.** The two are already different lengths in practice (12,725 vs 10,865), so validating against the wrong one would either reject valid input or misalign every phase by an unknown offset. A swing observation has no single phase by construction — it is a difference *between* two phases. `_swing` stays exactly as it is.

- [ ] **Step 1: Write the failing tests**

```python
def test_phases_default_to_fit_phase_and_reproduce_todays_behaviour():
    """The backward-compatibility guarantee. Every existing caller passes no
    phases; all of them must keep getting exactly what they get today."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    a, da = salinity_fit.fit_intrusion(obs, cfg=CFG)
    b, db = salinity_fit.fit_intrusion(obs, cfg=CFG, phases=[salinity_fit.FIT_PHASE] * 3)
    assert da["rmse_ppt"] == pytest.approx(db["rmse_ppt"], rel=0, abs=0)
    assert a.l0_km == pytest.approx(b.l0_km, rel=0, abs=0)


def test_a_phase_actually_changes_the_residual():
    """If phase were ignored, these two fits would be identical -- which is
    exactly the bug this task exists to fix."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    _, flat = salinity_fit.fit_intrusion(obs, cfg=CFG, phases=[0.25, 0.25, 0.25])
    _, tidal = salinity_fit.fit_intrusion(obs, cfg=CFG, phases=[0.0, 0.5, 0.0])
    assert flat["rmse_ppt"] != pytest.approx(tidal["rmse_ppt"])


def test_phases_length_mismatch_raises():
    """Mirrors the contract `sources` already holds."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    with pytest.raises(ValueError, match="phases"):
        salinity_fit.fit_intrusion(obs, cfg=CFG, phases=[0.25, 0.5])


def test_phases_is_validated_against_observations_not_swings():
    """The two sequences are different lengths in the real fit (12,725 vs
    10,865). Validating against the wrong one would misalign every phase."""
    obs = [(5.0, 4000.0, 30.0), (12.0, 4000.0, 18.0), (20.0, 9000.0, 6.0)]
    swings = [(5.0, 4000.0, 8.0)]
    fitted, _ = salinity_fit.fit_intrusion(
        obs, cfg=CFG, swings=swings, phases=[0.1, 0.2, 0.3]
    )
    assert fitted is not None  # 3 phases for 3 observations, 1 swing -- valid
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && $HOME/.venvs/tidescout/bin/python -m pytest tests/test_salinity.py -q -k phase`
Expected: `TypeError: fit_intrusion() got an unexpected keyword argument 'phases'`

- [ ] **Step 3: Implement the fit change**

In `fit_intrusion`, add the parameter and validate it beside the existing `sources` check:

```python
    if phases and len(phases) != len(observations):
        raise ValueError(
            f"phases has {len(phases)} entries but observations has "
            f"{len(observations)} -- they must be the same length, in the same "
            "order. Note phases aligns with `observations`, never with `swings`: "
            "a swing is a DIFFERENCE between two phases and has no single one."
        )
```

`_finite_rows` drops non-finite observations, so the phase array must be filtered the same way to stay aligned. Build a per-row phase array over the SURVIVING rows and hand it to `_levels`:

```python
def _levels(groups, cfg: SalinityConfig, n: int, phases: np.ndarray) -> np.ndarray:
    """Modelled salinity per row, each at its OWN tidal phase.

    `salinity_at` broadcasts over an array phase -- `x + excursion *
    cos(2*pi*phase)` with both arrays the same shape -- so the existing
    group-by-discharge vectorisation is preserved exactly. Verified
    bit-identical to a per-row loop.
    """
    out = np.empty(n, dtype="float64")
    for q, idx, dist in groups:
        out[idx] = salinity_at(dist, q, phases[idx], cfg)
    return out
```

Default the array to `FIT_PHASE` when `phases` is empty, so behaviour is unchanged:

```python
    level_phases = (
        np.asarray(kept_phases, dtype="float64")
        if phases
        else np.full(len(obs), FIT_PHASE, dtype="float64")
    )
```

Report `n_phase_resolved` in the diagnostics — how many rows carry a real phase rather than the default.

- [ ] **Step 4: Wire it into `collect_observations`**

`_finite_rows` drops non-finite rows, so `kept_phases` must be filtered by the SAME predicate to
stay aligned. Build it exactly where `kept_sources` is already built (the code added in PR #5 for
`rmse_by_source_ppt` does this — follow it rather than inventing a second filter):

```python
    # Same predicate as `_finite_rows`, applied to the phase array so it stays
    # row-for-row aligned with `obs` after filtering. Duplicating the predicate
    # is deliberate and mirrors how `kept_sources` is built; the alternative is
    # `_finite_rows` returning indices, which changes a contract three callers
    # already depend on.
    kept_phases = [
        ph
        for (d, q, y_), ph in zip(observations, phases, strict=True)
        if np.isfinite(d) and np.isfinite(q) and np.isfinite(y_)
    ]
```

Then in `collect_observations`, fetch once and attach per row:

```python
    # ONE range fetch covering everything the observations span, rather than a
    # per-row lookup. Measured 2026-08-24: 1,260 unique dates over 1999-2026,
    # which is 28 yearly chunks against a permanently-cached, deterministic
    # product.
    from tidescout.engine.tides import phase_at
    from tidescout.sources import noaa

    events = []
    if by_day:
        events = noaa.tide_events_range(
            fishery.stations.tide[0], min(by_day), max(by_day), fishery.timezone, cache
        ) if fishery.stations.tide else []

    n_no_phase = 0
```

For the WQP grab loop, replace the bare append with a phase lookup that EXCLUDES on failure:

```python
        for ts, ppt in series:
            day = ts.astimezone(tz).date()
            if day not in by_day:
                n_wqp_no_discharge_day += 1
                continue
            ph = phase_at(events, ts) if events else None
            if ph is None:
                # A grab with no determinable phase is dropped, never scored at
                # FIT_PHASE. This module already refuses a fabricated timestamp
                # at parse time on the same reasoning -- a fabricated phase is
                # that error one layer down, and it is worth up to half the
                # local tidal swing (8.3-12.3 ppt where these samples sit).
                n_no_phase += 1
                continue
            observations.append((wqp_dist[site], by_day[day], ppt))
            sources.append("wqp")
            obs_phases.append(ph)
```

For the NERRS and USGS daily-mean appends, append `FIT_PHASE` alongside, with this comment at the
first such site:

```python
            # FIT_PHASE here is CORRECT, not a fallback: a daily mean IS a tidal
            # average, and 0.25 is exactly the phase at which the model's tidal
            # term vanishes. Only instantaneous samples need a real phase.
            obs_phases.append(FIT_PHASE)
```

Add `observation_phases: list[float]` and `n_no_phase: int` to `CalibrationInput`, populate both,
and print the resolved count and the exclusion count in `salinity calibrate`'s report beside the
existing `n_wqp_no_discharge_day` line.

- [ ] **Step 5: Run the tests and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tidescout/pipeline/salinity_fit.py backend/tidescout/engine/salinity.py backend/tests/test_salinity.py backend/tidescout/cli.py
git commit -m "feat: score each observation at its own tidal phase"
```

---

## Task 4: The gate — measure what phase changed

*(Task 5 below is a regression pin and may be done before or after this gate.)*

**Files:**
- Create: `.superpowers/sdd/2026-08-24-tidal-phase/gate-report.md`
- Modify: `backend/tidescout/cli.py` if needed to surface a number

**This task measures and reports. It does not decide, and it changes no config.**

- [ ] **Step 1: Run the pipeline end to end**

```bash
cd /Users/ellismillwood/Documents/tidescout
$HOME/.venvs/tidescout/bin/tidescout salinity calibrate winyah-bay | tee /tmp/calibrate-phase.txt
```

- [ ] **Step 2: Record the comparison, with real numbers**

| metric | before (PR #5) | after |
|---|---|---|
| observations | 12,725 | ? |
| grabs excluded, no phase | n/a | ? |
| rmse overall | 4.419 | ? |
| **rmse WQP** | **6.102** | **?** |
| rmse NERRS | 4.061 | ? |
| `l0_km` | 13.33 | ? |
| `front_width_km` | 14.68 | ? |
| condition number | 7.94 | ? |
| `fitted` | False | ? |

**The WQP figure is the one this work exists to move. The NERRS figure should NOT move** — those rows were already at the correct phase, so a change there means something went wrong.

- [ ] **Step 3: Re-run the score-spread probe**

Report whether the near-mouth trout sub-score spread (23–24 points at North Jetty in PR #5) narrowed. Say explicitly which question the probe answers — parameter uncertainty, or dependence on modelling choices — because conflating those is how a "resolved" claim gets overstated.

- [ ] **Step 4: State plainly what is now binding, and STOP**

Say whether `fitted` can become True and, if not, which single cause is binding now. Do NOT edit the `salinity:` block, do NOT free `ocean_ppt`, and do NOT choose an `ocean_ppt` route — that is the gate's output and the next plan's input.

- [ ] **Step 5: Commit**

```bash
git add backend/tidescout/cli.py
git commit -m "docs: gate report on the phase-corrected refit"
```

---

---

## Task 5: Pin the "one tide station suffices" ruling

**Files:**
- Modify: `backend/tests/test_salinity.py`

**Interfaces:** consumes `phase_at`, `tide_events_range`, and the NERRS store.

The spec's §2 rules that no per-location phase-lag model is needed, on the strength of a MEASURED
lag of <= 0.011 phase units at 16.68-19.03 km. That ruling is load-bearing — it is what keeps this
work small — and nothing currently protects it. A future change to the tide station, the phase
convention, or the interpolation would silently invalidate it.

- [ ] **Step 1: Write the regression**

```python
def test_up_estuary_tidal_lag_stays_negligible():
    """The spec rules that ONE tide station's phase serves the whole estuary,
    because the measured lag at 16.68-19.03 km is <= 0.011 phase units against
    the ~0.25 error the old code made. That ruling is what keeps this work
    small; this pins it.

    Measured 2026-08-24 over 2026-07-01..21 against CO-OPS station 8662549:
        NIWTAWQ  16.68 km   -2.0 min  (-0.003 phase)
        WYSS1    19.03 km   +4.0 min  (+0.005 phase)
        NIWWBWQ  19.03 km   +8.0 min  (+0.011 phase)

    Skips when the NERRS store is absent, like the other real-data tests here.
    """
    import numpy as np
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from tidescout.config import load_fishery
    from tidescout.sources import ndbc, noaa
    from tidescout.sources.cache import default_cache

    fishery = load_fishery("winyah-bay")
    store = ndbc.default_store("winyah-bay")
    if not store.stations():
        pytest.skip("NERRS store not present")

    zone = ZoneInfo(fishery.timezone)
    events = noaa.tide_events_range(
        fishery.stations.tide[0], date(2026, 7, 1), date(2026, 7, 21),
        fishery.timezone, default_cache(),
    )
    highs = sorted(e.time for e in events if e.kind == "H")

    for station in ("NIWTAWQ", "WYSS1", "NIWWBWQ"):
        rows = [
            (t, r.depth_m)
            for t, r in (
                (o.ts.astimezone(zone), o)
                for o in store.read(station,
                                    datetime(2026, 7, 1, tzinfo=zone),
                                    datetime(2026, 7, 21, tzinfo=zone))
            )
            if r.depth_m is not None
        ]
        if len(rows) < 500:
            pytest.skip(f"{station} has too few depth readings in the window")
        times = [t for t, _ in rows]
        depths = np.array([d for _, d in rows], dtype="float64")
        lags = []
        for h in highs:
            idx = [i for i, t in enumerate(times) if abs((t - h).total_seconds()) < 4 * 3600]
            if len(idx) < 20:
                continue
            j = idx[int(np.argmax(depths[idx]))]
            lags.append((times[j] - h).total_seconds() / 60.0)
        if not lags:
            continue
        phase_units = abs(float(np.median(lags))) / (12.42 * 60)
        assert phase_units <= 0.05, (
            f"{station}'s tidal lag is {phase_units:.3f} phase units -- the spec's "
            "'one station suffices' ruling assumed <= 0.011. If this is real, a "
            "per-location lag model is now needed and the spec must be revisited."
        )
```

The assertion threshold is 0.05 rather than the measured 0.011 deliberately: it is a guard against
the ruling becoming *wrong*, not a pin on the exact measurement, which will vary a little with the
window chosen.

- [ ] **Step 2: Run it and commit**

```bash
cd /Users/ellismillwood/Documents/tidescout && make check
git add backend/tests/test_salinity.py
git commit -m "test: pin the measured up-estuary tidal lag that keeps one station sufficient"
```

---

## Completion Checklist

- [ ] `make check` green; test count > 655
- [ ] `phase_at` returns 0.0 at low water and 0.5 at high water — the convention verified, not assumed
- [ ] An undeterminable phase excludes and counts; nothing defaults to `FIT_PHASE` silently
- [ ] Empty `phases` reproduces today's fit bit-identically
- [ ] `phases` validated against `observations`, never `swings`
- [ ] `engine/salinity.py` changed in annotation and docstring only — no maths
- [ ] Gate report written with real numbers; `fisheries/winyah-bay.yaml` unchanged
