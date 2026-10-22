"""Plans an occluded pedestrian: occluder(s) + a slow walk from hidden to visible.

1. Put the occluder(s) in place: 1-2 cars next to the ego, or a car parked at the curb.
2. The target pedestrian walks across from the occluder's side of the road.
3. Along that walk, predict (camera model) how much of the pedestrian is hidden at
   every 10 cm, and pick the frames where the hidden share drops through the
   requested levels: e.g. 75 % -> 56 % -> 38 % -> 19 % -> 0 %.
The real occlusion of every frame is measured later from segmentation.
"""
import advanced as C
from engine.camera import coverage, in_view, rects_overlap
from engine.people import crossing_line, make_ped, ped_box, pick_type
from engine.traffic import adjacent_occluders, box_of, parked_occluder


STEP = 0.1          # metres between predicted positions


def frame_levels(top, n):
    """n occlusion levels for the frames of one walk, from `top` down to fully visible."""
    if n == 1:
        return [top]
    return [top * (n - 1 - i) / (n - 1) for i in range(n)]


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
    """Positions (s, x, y) along the walk, from 1.5 m back on the sidewalk to the far curb."""
    (ox, oy), d = crossing_line(ctx.site, direction, lateral)
    out, s = [], -1.5
    while s <= ctx.site.length:
        out.append((s, ox + d[0] * s, oy + d[1] * s))
        s += STEP
    return out, d


def _pick_frames(curve, levels):
    """Index into curve for each level: first point (after the peak) hidden <= level."""
    peak = max(range(len(curve)), key=lambda i: curve[i])
    if curve[peak] <= 0.0:
        return None             # never hidden at all
    top = min(levels[0], curve[peak])
    wanted = [top] + [lv for lv in levels[1:]]
    idx, start = [], peak
    for k, lv in enumerate(wanted):
        limit = C.FULLY_VISIBLE if k == len(wanted) - 1 and len(wanted) > 1 else lv + 1e-9
        hit = next((i for i in range(start, len(curve)) if 0.0 <= curve[i] <= limit), None)
        if hit is None:
            return None
        idx.append(hit)
        start = hit + 1
    return idx


def plan_occluded(ctx, scene, top, cars, trucks, n_frames):
    """Returns (target_ped, occluders, frames, rects) for the best layout found, or None."""
    levels = frame_levels(top, n_frames)
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
        boxes = [box_of(ctx, v) for v in occluders]

        # walk from the side of the road the occluders are on
        occ_lat = sum(r[1] for r in occ_rects) / len(occ_rects)
        start_lat = ctx.frame.local(*ctx.site.start[:2])[1]
        direction = 1 if (start_lat > 0) == (occ_lat > 0) else -1
        walk, d = _walk(ctx, direction, lateral)

        probe = {"ground_z": ctx.site.center[2], "yaw": 0.0, "type": ptype}
        curve = []
        for _, x, y in walk:
            box = ped_box(ctx, probe, x, y)
            curve.append(coverage(box, boxes) if in_view(box) else -1.0)
        idx = _pick_frames(curve, levels)
        if idx is None:
            continue
        # the target must not walk into its own occluder
        if any(rects_overlap((*ctx.frame.local(walk[i][1], walk[i][2]), 0.5, 0.5), r, 0.0, 0.0)
               for i in range(idx[0], idx[-1] + 1) for r in occ_rects):
            continue

        score = abs(curve[idx[0]] - levels[0])
        if best is None or score < best[0]:
            best = (score, ptype, speed, d, walk, idx, curve, occluders, occ_rects, rects + occ_rects)
        if score < 0.03:
            break

    if best is None:
        return None
    _, ptype, speed, d, walk, idx, curve, occluders, occ_rects, rects = best

    # start walking a little before the first frame, so the pedestrian is mid-stride
    preroll = ctx.rng.uniform(*C.PED_PREROLL_SECONDS)
    s0 = walk[idx[0]][0]
    s_spawn = max(-2.5, s0 - speed * preroll)
    sx, sy = walk[idx[0]][1] - d[0] * (s0 - s_spawn), walk[idx[0]][2] - d[1] * (s0 - s_spawn)

    frames, last_tick = [], -10 ** 9
    for i in idx:
        tick = round((walk[i][0] - s_spawn) / (speed * C.FIXED_DELTA_SECONDS))
        tick = max(tick, last_tick + C.CAPTURE_SETTLE_TICKS + 2, C.CAPTURE_SETTLE_TICKS + 2)
        frames.append({"tick": tick, "planned_occlusion": round(max(curve[i], 0.0), 3)})
        last_tick = tick
    target = make_ped(ctx, "crossing", ptype, speed, False, sx, sy, d, walk[idx[0]][1:])
    return target, occluders, frames, rects
