"""Static audit of the boundary classification, in ring order.

Cause #1 was a slot: four `ocean` segments embedded in an otherwise solid
`wall`, which imposed the coastal tide 40 km inland and acted as a jet
generator. That pathology is only visible in RING ORDER -- per-tag counts hide
it completely, and it took two five-hour builds to find.

Build #3 then failed on the east ocean apron at (672405,3690284) and
(672619,3679888). Both sit on the authored polygon's east cut line, which runs
through open water rather than along a shore. So the mirror-image pathology is
worth checking before spending another build on it: a `wall` run embedded in
`ocean` along that cut line is an invisible dam across open water, and would
generate exactly this kind of blow-up.

This needs no simulation -- it reads the same ring `build_mesh` classifies.
"""

import numpy as np
from shapely.geometry import Point, Polygon

from tidescout.config import load_fishery
from tidescout.pipeline import mesh
from tidescout.pipeline.bathy import read_bathy

SUSPECTS = [(672405, 3690284), (672619, 3679888), (662770, 3698800)]

f = load_fishery("winyah-bay")
md = f.model_domain
cfg = f.anuga

z, transform, meta = read_bathy("winyah-bay")
pixel_area_m2 = abs(transform.a) * abs(transform.e)
mask = mesh.clean_mask(
    mesh.domain_mask(z, transform, f),
    md.clean_cells,
    md.min_island_hole_km2,
    pixel_area_m2,
)
boundary = mesh.domain_polygon(mask, transform, md.simplify_m)
ring = [list(c) for c in boundary.exterior.coords[:-1]]

ocean_idx, wall_idx, open_idx = mesh.classify_boundary(
    ring, z, transform, md.ocean_max_z_m, md.ocean_boundary_utm_km
)
tag = {}
for i in ocean_idx:
    tag[i] = "ocean"
for i in wall_idx:
    tag[i] = "wall"
for i in open_idx:
    tag[i] = "open"

n = len(ring)
mid = np.zeros((n, 2))
zb = np.zeros(n)
seglen = np.zeros(n)
for i in range(n):
    ax, ay = ring[i]
    bx, by = ring[(i + 1) % n]
    mid[i] = (0.5 * (ax + bx), 0.5 * (ay + by))
    seglen[i] = float(np.hypot(bx - ax, by - ay))
    col, row = ~transform * (mid[i][0], mid[i][1])
    r = int(np.clip(row, 0, z.shape[0] - 1))
    c = int(np.clip(col, 0, z.shape[1] - 1))
    zb[i] = z[r, c]

print(f"ring segments: {n}  ocean={len(ocean_idx)} wall={len(wall_idx)} "
      f"open={len(open_idx)}")
print(f"total perimeter {seglen.sum()/1000:.1f} km")

# --- runs of consecutive same-tag segments, in ring order -------------------
runs = []
start = 0
for i in range(1, n + 1):
    if i == n or tag[i] != tag[start]:
        runs.append((tag[start], start, i - 1, i - start))
        start = i
# The ring is cyclic: merge the last run into the first if they share a tag.
if len(runs) > 1 and runs[0][0] == runs[-1][0]:
    t, s0, e0, c0 = runs[0]
    _, s1, e1, c1 = runs.pop()
    runs[0] = (t, s1, e0, c0 + c1)

print(f"\n{len(runs)} contiguous runs")
by_tag = {}
for t, _s, _e, c in runs:
    by_tag.setdefault(t, []).append(c)
for t, cs in sorted(by_tag.items()):
    print(f"  {t:6s}: {len(cs):3d} runs, lengths min={min(cs)} "
          f"median={int(np.median(cs))} max={max(cs)}")

# --- the pathology: a short run of one tag embedded in a different tag ------
print("\nEMBEDDED RUNS (a run whose two neighbours share a tag different from "
      "its own)\nThis is the shape that caused cause #1. Length in segments, "
      "then metres.")
for k, (t, s, e, c) in enumerate(runs):
    prev_t = runs[(k - 1) % len(runs)][0]
    next_t = runs[(k + 1) % len(runs)][0]
    if prev_t == next_t and prev_t != t:
        L = seglen[s:e + 1].sum()
        mx, my = mid[s:e + 1].mean(axis=0)
        print(f"  {t:6s} x{c:3d} ({L:7.1f} m) inside {prev_t:6s} "
              f"@ ({mx:.0f}, {my:.0f})  bed {zb[s:e+1].min():+.2f}.."
              f"{zb[s:e+1].max():+.2f} m")

# --- what is near each known failure location ------------------------------
ocean_poly = Polygon([(x * 1000.0, y * 1000.0) for x, y in md.ocean_boundary_utm_km])
for sx, sy in SUSPECTS:
    d = np.hypot(mid[:, 0] - sx, mid[:, 1] - sy)
    order = np.argsort(d)[:12]
    print(f"\nNEAREST BOUNDARY SEGMENTS TO ({sx}, {sy}) — 12 closest")
    for i in order:
        inside = ocean_poly.contains(Point(mid[i][0], mid[i][1]))
        print(f"  seg {i:5d} {tag[i]:6s} d={d[i]:8.1f} m  bed={zb[i]:+7.2f}  "
              f"len={seglen[i]:6.1f}  in_ocean_poly={inside}")

# --- bed depth distribution per tag ----------------------------------------
print("\nBED ELEVATION BY TAG (m, NAVD88)")
for t, idxs in (("ocean", ocean_idx), ("open", open_idx), ("wall", wall_idx)):
    if not idxs:
        continue
    v = zb[idxs]
    v = v[np.isfinite(v)]
    print(f"  {t:6s} n={len(idxs):5d}  min={v.min():+7.2f} p10={np.percentile(v,10):+7.2f} "
          f"median={np.median(v):+7.2f} p90={np.percentile(v,90):+7.2f} max={v.max():+7.2f}")

# --- the east cut line specifically ----------------------------------------
print("\nEAST CUT LINE (x > 670,500 m, 3,677,000 < y < 3,692,000) in ring order")
sel = np.where((mid[:, 0] > 670500) & (mid[:, 1] > 3677000) & (mid[:, 1] < 3692000))[0]
if len(sel) == 0:
    print("  none")
else:
    print(f"  {len(sel)} segments, {seglen[sel].sum()/1000:.2f} km")
    seq = "".join({"ocean": "O", "wall": "W", "open": "-"}[tag[i]] for i in sorted(sel))
    print(f"  ring-order tag sequence (O=ocean W=wall -=open):\n    {seq}")
    for i in sorted(sel):
        if tag[i] != "ocean":
            print(f"    NON-OCEAN seg {i:5d} {tag[i]:6s} @ ({mid[i][0]:.0f}, "
                  f"{mid[i][1]:.0f}) bed={zb[i]:+.2f} len={seglen[i]:.1f}")
