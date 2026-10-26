"""Plans an occluded pedestrian: occluder(s) + a walk that passes behind them.

Seen from the ego, the target pedestrian is first clearly visible, then walks
behind the car (hidden), then comes out and is clearly visible again:

    visible  ->  partly hidden  ->  most hidden  ->  partly hidden  ->  visible

1. Put the occluder(s) in place: 1-2 cars next to the ego, or a car parked at the curb.
2. Along the pedestrian's walk across the road, predict (camera model) how much of
   them is hidden at every 10 cm.
3. Find a stretch "visible -> hidden -> visible" whose most hidden point lies in the
   weather's occlusion range (min, max), and spread the frames over that stretch.
The real occlusion of every frame is measured later from segmentation.
"""
import advanced as C
from engine.camera import coverage, in_view, rects_overlap
from engine.people import crossing_line, make_ped, ped_box, pick_type
from engine.traffic import adjacent_occluders, boxes_of, parked_occluder


STEP = 0.1              # metres between predicted positions
EXTEND = 3.0            # the walk may go this far onto each sidewalk
CLEAR_VIEW = 0.8        # metres walked fully visible before / after the hidden stretch


def _occluder_categories(ctx, scene, cars, trucks):
    use_truck = trucks > 0 and (cars == 0 or ctx.rng.random() < C.OCCLUDER_TRUCK_CHANCE)
    if scene == "occluded_by_parked":
        return ["truck" if use_truck else "car"]
    first = "truck" if use_truck else "car"
    left = {"car": cars, "truck": trucks}
    left[first] -= 1
    cats = [first]
    if ctx.rng.random() < 0.5:          # one or two cars next to the ego
        if left["car"] > 0:
            cats.append("car")
        elif left["truck"] > 0:
            cats.append("truck")
    return cats


def _walk(ctx, direction, lateral):
    """Positions (s, x, y) every STEP m along a walk across the road, sidewalk to sidewalk."""
    (ox, oy), d = crossing_line(ctx.site, direction, lateral)
    out, s = [], -EXTEND
    while s <= ctx.site.length + EXTEND:
        out.append((s, ox + d[0] * s, oy + d[1] * s))
        s += STEP
    return out, d


def _arcs(curve, lo, hi):
    """Stretches "visible -> hidden -> visible" whose most hidden point is within [lo, hi].
    curve[i] = hidden share at step i, or -1 if out of the image.
    Returns (start, peak, end) indices; start / end are CLEAR_VIEW m into the visible parts."""
    def visible(i):
        return 0.0 <= curve[i] <= C.FULLY_VISIBLE

    margin = max(1, round(CLEAR_VIEW / STEP))
    out, i, n = [], 0, len(curve)
    while i < n:
        if curve[i] > C.FULLY_VISIBLE:
            a = i
            while i < n and curve[i] > C.FULLY_VISIBLE:
                i += 1
            b = i - 1                                   # hidden run a..b, all in view
            if a == 0 or b == n - 1 or not visible(a - 1) or not visible(b + 1):
                continue
            peak = max(range(a, b + 1), key=lambda k: curve[k])
            if not (lo - 0.01 <= curve[peak] <= hi + 0.01):
                continue
            start = a - 1
            while start > 0 and a - start < margin and visible(start - 1):
                start -= 1
            end = b + 1
            while end < n - 1 and end - b < margin and visible(end + 1):
                end += 1
            out.append((start, peak, end))
        else:
            i += 1
    return out


def _frame_indices(start, peak, end, n):
    """n frames spread evenly from start to end, one of them exactly at the peak."""
    if n == 1:
        return [peak]
    if n == 2:                                          # too short for the whole arc: visible, then hidden
        return [start, peak]
    idx = [round(start + (end - start) * k / (n - 1)) for k in range(n)]
    nearest = min(range(n), key=lambda k: abs(idx[k] - peak))
    idx[nearest] = peak
    return sorted(set(idx))


def plan_occluded(ctx, scene, occ_range, cars, trucks, n_frames):
    """Returns (target_ped, occluders, frames, rects, peak) for the best layout found, or None.
    occ_range = (min, max) hidden share of the target at its most hidden moment."""
    lo, hi = occ_range
    goal = ctx.rng.uniform(lo, hi)
    best = None
    for _ in range(40):
        ptype = pick_type(ctx)
        speed = ctx.rng.uniform(*C.SEQUENCE.get("walk_speed", (0.6, 1.0)))
        band = max(0.2, ctx.site.half_width - 0.5)
        lateral = ctx.rng.uniform(-band, band)
        # long position of the walking line in the road frame
        ped_long = ctx.frame.local(ctx.site.center[0] + ctx.site.along[0] * lateral,
                                   ctx.site.center[1] + ctx.site.along[1] * lateral)[0]

        rects = list(ctx.rects)
        cats = _occluder_categories(ctx, scene, cars, trucks)
        placed = (parked_occluder(ctx, cats[0], ped_long, rects) if scene == "occluded_by_parked"
                  else adjacent_occluders(ctx, cats, ped_long, rects))
        if placed is None:
            continue
        occluders, occ_rects = placed
        boxes = [b for v in occluders for b in boxes_of(ctx, v)]

        walk, d = _walk(ctx, ctx.rng.choice((1, -1)), lateral)
        probe = {"ground_z": ctx.site.center[2], "yaw": 0.0, "type": ptype}
        curve = []
        for _, x, y in walk:
            box = ped_box(ctx, probe, x, y)
            curve.append(coverage(box, boxes) if in_view(box) else -1.0)

        for start, peak, end in _arcs(curve, lo, hi):
            # the target must never walk into its own occluder
            if any(rects_overlap((*ctx.frame.local(walk[i][1], walk[i][2]), 0.5, 0.5), r, 0.0, 0.0)
                   for i in range(start, end + 1) for r in occ_rects):
                continue
            score = abs(curve[peak] - goal)
            if best is None or score < best[0]:
                best = (score, ptype, speed, d, walk, (start, peak, end), curve, occluders, rects + occ_rects)
        if best is not None and best[0] < 0.03:
            break

    if best is None:
        return None
    _, ptype, speed, d, walk, (start, peak, end), curve, occluders, rects = best
    idx = _frame_indices(start, peak, end, n_frames)

    # start walking a little before the first frame, so the pedestrian is mid-stride
    preroll = ctx.rng.uniform(*C.PED_PREROLL_SECONDS)
    s0 = walk[idx[0]][0]
    s_spawn = max(-EXTEND - 0.5, s0 - speed * preroll)
    sx, sy = walk[idx[0]][1] - d[0] * (s0 - s_spawn), walk[idx[0]][2] - d[1] * (s0 - s_spawn)

    frames, last_tick = [], -10 ** 9
    for i in idx:
        tick = round((walk[i][0] - s_spawn) / (speed * C.FIXED_DELTA_SECONDS))
        tick = max(tick, last_tick + C.CAPTURE_SETTLE_TICKS + 2, C.CAPTURE_SETTLE_TICKS + 2)
        frames.append({"tick": tick, "planned_occlusion": round(max(curve[i], 0.0), 3)})
        last_tick = tick
    target = make_ped(ctx, "crossing", ptype, speed, False, sx, sy, d, walk[idx[0]][1:])
    return target, occluders, frames, rects, round(curve[peak], 3)
