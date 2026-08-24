"""Does a config change move the water enough to justify rebuilding the library?

Every "should we rebuild?" question in Plans 3 and 4 has been decided by this
comparison, and the script that decided them was never committed. Carryover
lesson 5: the diagnostic harness has already been lost once and silently
drifted once. This is that harness, in the repo.

WHAT IT MEASURES. A variant is one regime re-run with exactly one thing
changed, written to `data/<slug>/flow-variant-<what>/<regime>/` instead of over
the shipped library. This differences it against the shipped regime of the same
name at MATCHED PHASE INDEX -- both runs share a tidal range bucket and
therefore the same boundary forcing, so snapshot i is the same point in the
cycle in both -- and reports two numbers:

  p99 delta          For each phase, p99(|dspeed|) as a percentage of that
                     phase's p99(speed) in the reference run; then the MEAN of
                     those 26 percentages. Per-phase-then-averaged, not pooled
                     into one percentile, so every point in the tidal cycle
                     counts equally -- a single pooled percentile is dominated
                     by the fast phases and reads ~25% low here (13.33% against
                     17.20% on the freshet probe). Both are printed; the
                     per-phase mean is the headline because it is the number
                     every recorded decision was made on.
  spot mean |dspeed| mean absolute speed change within `--radius` of each
                     known spot, in m/s. The domain figure can hide a large
                     local move; these are the places Ellis actually fishes.

WHY BOTH, AND WHY A NOISE FLOOR. Two runs of the same model are not bit
identical across a config change that should not matter, so a raw delta proves
nothing on its own -- it has to be read against a variant whose change is known
to be negligible. `flow-variant-southocean` (the corrected southern-approach
boundary vertex) is that reference: 0.77% pooled, 0.00018-0.00059 m/s at the
spots. Pass `--noise-floor 0.77` to have the verdict line compare against it.

Measured with this method (2026-08-16), against production `mean_high`:
  flow-variant-freshet        22,996 cfs, equal thirds   17.20%  22x the floor
  flow-variant-freshet-split  22,996 cfs, 78/13/8        17.62%  23x the floor
That is what opened the discharge axis to a fourth bucket.

All five figures above (both freshet variants, the floor, and both sample
counts) were reproduced to the digit by this file on 2026-08-23, which is the
only reason it can be trusted as THE harness rather than a re-implementation
that happens to agree in sign.

Usage:
    compare_variant.py <variant-dir> [--regime NAME] [--against DIR]
                       [--slug SLUG] [--radius M] [--noise-floor PCT]

    compare_variant.py data/winyah-bay/flow-variant-freshet --noise-floor 0.77
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rasterio.warp import transform as warp_transform

from tidescout.config import load_fishery, load_known_spots
from tidescout.paths import fishery_data_dir
from tidescout.pipeline import mesh

# A centroid is compared only where BOTH runs hold water. This is
# `engine.flow.wet_mask`'s tolerance, so the comparison sees exactly the cells
# the runtime will read -- and it is what the 2026-08-16 probes used (it
# reproduces their 5.48M / 5.47M sample counts). Raising it to 0.05 m, as
# discharge_axis_check.py does for a different question, drops 134k samples
# and moves the headline by 0.01 points; the metric is not sensitive to it.
WET_MIN_DEPTH_M = 0.01


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Difference a variant regime against the shipped library."
    )
    p.add_argument("variant_dir", type=Path, help="data/<slug>/flow-variant-<what>")
    p.add_argument("--regime", default=None, help="default: the variant's only subdirectory")
    p.add_argument("--against", type=Path, default=None, help="default: the shipped regime")
    p.add_argument("--slug", default="winyah-bay")
    p.add_argument("--radius", type=float, default=150.0, help="known-spot radius, m")
    p.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="pooled p99 %% of a variant known to be negligible, e.g. 0.77",
    )
    return p.parse_args(argv)


def _resolve_regime(variant_dir: Path, regime: str | None) -> str:
    """Which regime this variant re-ran.

    Inferred rather than required, because a variant directory holds exactly
    one regime by construction -- but an explicit --regime still wins, and an
    ambiguous directory is an error rather than an arbitrary pick.
    """
    if regime:
        return regime
    meta = variant_dir / "variant.json"
    if meta.exists():
        recorded = json.loads(meta.read_text()).get("regime")
        if recorded:
            return str(recorded)
    subdirs = sorted(d.name for d in variant_dir.iterdir() if (d / "regime.json").exists())
    if len(subdirs) != 1:
        raise SystemExit(
            f"cannot infer the regime: {variant_dir} holds {len(subdirs)} completed "
            f"regimes {subdirs} -- pass --regime"
        )
    return subdirs[0]


def _snapshot_count(regime_dir: Path) -> int:
    meta = json.loads((regime_dir / "regime.json").read_text())
    return len(meta["snapshots"])


def _load_speed(regime_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(regime_dir / f"snap_{index:03d}.npz")
    return np.hypot(z["u"], z["v"]), z["depth"]


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    variant_dir = args.variant_dir.resolve()
    regime = _resolve_regime(variant_dir, args.regime)
    var_reg = variant_dir / regime
    ref_reg = args.against or (fishery_data_dir(args.slug) / "flow" / regime)
    ref_reg = Path(ref_reg).resolve()

    for d in (var_reg, ref_reg):
        if not (d / "regime.json").exists():
            raise SystemExit(f"no completed regime at {d} (no regime.json)")

    n = min(_snapshot_count(var_reg), _snapshot_count(ref_reg))
    if n == 0:
        raise SystemExit("a run completed with zero snapshots -- nothing to compare")

    fishery = load_fishery(args.slug)
    print(f"variant   {var_reg}", flush=True)
    print(f"reference {ref_reg}")
    print(f"regime {regime!r}, {n} matched-phase snapshots\n")
    print("building mesh for centroid coordinates...", flush=True)
    domain = mesh.build_mesh(args.slug, fishery)
    cx, cy = domain.get_centroid_coordinates(absolute=True).T
    print(f"{len(cx):,} centroids", flush=True)

    # Known spots, projected once into the bathymetry CRS.
    spots = load_known_spots(args.slug)
    spot_masks: dict[str, np.ndarray] = {}
    if spots:
        xs, ys = warp_transform(
            "EPSG:4326",
            f"EPSG:{fishery.bathymetry.epsg}",
            [s.lon for s in spots],
            [s.lat for s in spots],
        )
        for spot, sx, sy in zip(spots, xs, ys, strict=True):
            spot_masks[spot.name] = np.hypot(cx - sx, cy - sy) <= args.radius

    deltas: list[np.ndarray] = []
    speeds: list[np.ndarray] = []
    per_phase_pct: list[float] = []
    spot_sums = {name: 0.0 for name in spot_masks}
    spot_counts = {name: 0 for name in spot_masks}

    for i in range(n):
        var_spd, var_depth = _load_speed(var_reg, i)
        ref_spd, ref_depth = _load_speed(ref_reg, i)
        if var_spd.shape != ref_spd.shape:
            raise SystemExit(
                f"phase {i}: variant has {var_spd.size:,} centroids and the reference "
                f"{ref_spd.size:,} -- these were run on different meshes and cannot be "
                "differenced cell by cell"
            )
        wet = (var_depth > WET_MIN_DEPTH_M) & (ref_depth > WET_MIN_DEPTH_M)
        if not wet.any():
            continue
        d = np.abs(var_spd[wet] - ref_spd[wet])
        ref_wet = ref_spd[wet]
        deltas.append(d.astype("float32"))
        speeds.append(ref_wet.astype("float32"))
        phase_p99_speed = float(np.percentile(ref_wet, 99))
        if phase_p99_speed > 0:
            per_phase_pct.append(100.0 * float(np.percentile(d, 99)) / phase_p99_speed)
        for name, sel in spot_masks.items():
            m = sel & wet
            if not m.any():
                continue
            spot_sums[name] += float(np.abs(var_spd[m] - ref_spd[m]).sum())
            spot_counts[name] += int(m.sum())

    if not deltas:
        raise SystemExit("no phase had a wet centroid in both runs -- nothing to compare")

    all_delta = np.concatenate(deltas)
    all_speed = np.concatenate(speeds)
    p99_delta = float(np.percentile(all_delta, 99))
    p99_speed = float(np.percentile(all_speed, 99))
    pooled_pct = 100.0 * p99_delta / p99_speed if p99_speed > 0 else float("nan")
    if not per_phase_pct:
        raise SystemExit("every phase had zero reference speed -- nothing to normalise by")
    # THE headline. Normalised by the REFERENCE run's own p99 speed so it reads
    # as "a percentage of how fast this bay actually moves", and averaged over
    # phases rather than pooled so slack water counts as much as peak ebb.
    pct = float(np.mean(per_phase_pct))

    print(f"\n{all_delta.size:,} wet-centroid samples "
          f"({n} phases, depth > {WET_MIN_DEPTH_M} m in both)")
    print(f"  P99 DELTA         {pct:.2f}%   (mean of {len(per_phase_pct)} per-phase "
          f"p99|dspeed| / p99 speed; range {min(per_phase_pct):.2f}-"
          f"{max(per_phase_pct):.2f}%)")
    print(f"  pooled equivalent {pooled_pct:.2f}%   (one percentile over all phases "
          "at once -- lower, the fast phases dominate it)")
    print(f"  p99 |dspeed|      {p99_delta:.5f} m/s")
    print(f"  p99 speed (ref)   {p99_speed:.5f} m/s")
    print(f"  mean |dspeed|     {float(all_delta.mean()):.5f} m/s")
    print(f"  max  |dspeed|     {float(all_delta.max()):.5f} m/s")

    if spot_masks:
        print(f"\nknown spots, mean |dspeed| within {args.radius:.0f} m:")
        for name in spot_masks:
            if spot_counts[name] == 0:
                print(f"  {name:24s} (no wet centroid within radius in any phase)")
                continue
            mean = spot_sums[name] / spot_counts[name]
            print(f"  {name:24s} {mean:.5f} m/s   ({spot_counts[name]:,} samples)")

    print("\nVERDICT")
    if args.noise_floor is None:
        print("  no --noise-floor given; a delta is only interpretable against a "
              "variant whose change is known to be negligible.")
        print("  Winyah's reference is flow-variant-southocean at 0.77%.")
    elif args.noise_floor <= 0:
        raise SystemExit("--noise-floor must be positive")
    else:
        ratio = pct / args.noise_floor
        print(f"  {pct:.2f}% is {ratio:.1f}x the {args.noise_floor:.2f}% noise floor.")
        if ratio >= 2.0:
            print("  MATERIAL: this change moves the water. A library built without "
                  "it is wrong by more than the model's own reproducibility.")
        else:
            print("  NEGLIGIBLE: indistinguishable from the noise floor. Correct to "
                  "inherit at the next rebuild, but not a reason to spend one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
