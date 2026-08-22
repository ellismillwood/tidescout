
=== ROUND 1 RESULT 2026-08-15 14:53: ALL FOUR VARIANTS COMPLETED. BLOCKER RESOLVED. ===
FIRST TIME ANY REGIME HAS EVER FINISHED. Three builds and twelve-plus regimes had died before
this; all four round-1 variants ran the FULL 18.42 sim-h (t=66,312 s, 75 samples each).
  p1-mean-med         COMPLETED 3.30 h wall
  p1-neap-low         COMPLETED 3.21 h
  p1-ocean-dirichlet  COMPLETED 3.24 h
  p1-spring-high      COMPLETED 3.35 h
Health across the whole run, worst value of each over all 75 samples of all four runs:
  mass_resid   max 6.5e-13 .. 2.5e-11   (tolerance 1e-3; matches the "correct boundary closes at
                                         1e-13" expectation, so the identity holds end to end)
  dt_max       never below 0.195 s      (build #3 transmissive: 0.016-0.020 -- NO degradation
                                         anywhere, including the previously unreached final 30%)
  steps/yield  max 4,624 vs ~4,400 base (5% variation across the entire run; no cost blow-up)
  spd_max      0.880 / 1.129 / 1.108 / 1.439 m/s -- physical estuary and jetty speeds
  n>3 m/s      0 in every sample of every run
  hot20 on ocean/open/wall: max 0/1/1   -- the persistent fast cells are interior channel, never
                                          boundary-adjacent. Cause #1's signature is absent.
CONCLUSIONS
1. cebfb5a IS CONFIRMED. The `open` -> Reflective_boundary revert, committed but never run, is
   correct. Both the Pee Dee-head hotspot (cause #1) and the east-apron failure (cause #2) are
   gone.
2. CAUSE #2 NEVER EXISTED AS A SEPARATE PROBLEM. It was an artifact of Transmissive_boundary.
   The reviewer's warning -- that Transmissive_boundary appears in no ANUGA validation script and
   is weakly ill-posed for subcritical inflow -- was right on both counts, and process lesson 3
   ("weigh a reviewer's evidence above your own first-principles reasoning") is vindicated.
3. THE SEAWARD BC WAS NEVER IMPLICATED. p1-ocean-dirichlet swapped
   Transmissive_momentum_set_stage for Time_boundary([stage,0,0]) on the `ocean` tag and finished
   just as cleanly, at the same dt and residual. Production's existing seaward BC is fine and
   needs no change. This is exactly the value of running the contingency variant IN PARALLEL: it
   cost nothing and closed a question that would otherwise have needed its own 3.3 h round.
4. Damming the 3.6 km southern approach with Reflective did not destabilise anything. Still a
   physical liberty worth revisiting in Plan 4, but it is not a correctness blocker.

=== LIBRARY BUILD #4 LAUNCHED 2026-08-15 14:54:41 EDT ===
`tidescout flow run winyah-bay --workers 9`, pid 77042, log library-build4.log.
HEAD 26c2e39, 193 tests green, data/winyah-bay/flow empty, 746 GB free.
NINE WORKERS, deliberately oversubscribing 6 performance cores. Measured basis: 1 process runs
144 s per 900 s yieldstep, 4 processes 176 s each (only 1.22x slowdown, ~3.3x throughput), so the
machine is mildly bandwidth-contended rather than core-starved. 9 regimes on 6 workers is two
waves whose second is half-empty (~7 h); 9 at once has no tail (~5-5.5 h). This is the same gain
MPI would have bought for the ragged tail, at zero toolchain cost -- see the MPI analysis above.
Confirmed at launch: 9 worker processes each ~99-100% CPU, ~700 MB RSS (6.5 GB of 25.7 GB).
caffeinate re-armed for 10 h; the previous 8 h window would have expired mid-build.
Each regime writes snapshots only AFTER spin-up (6 sim-h, ~1 h wall), so 0 snapshots for the
first hour is expected, not a stall. 225 snapshots total expected (9 x 25).
