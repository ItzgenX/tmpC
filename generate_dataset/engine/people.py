"""Plans pedestrians (as data): who they are, where they start, how fast they walk.

Every walker moves in a straight line at constant speed from tick 0, so its
position at any frame is known in advance. A walk is accepted only if it never
comes closer than PED_MIN_GAP to another walker and never runs into a vehicle.
"""
import math

import advanced as C
from engine.roads import box_corners, forward_right, ground_z, rects_overlap, step_along


def weighted_choice(rng, weights):
    keys = [k for k, w in weights.items() if w > 0]
    total = sum(weights[k] for k in keys)
    x = rng.random() * total
    for k in keys:
        x -= weights[k]
        if x <= 0:
            return k
    return keys[-1]


def pick_type(ctx):
    mix = {t: w for t, w in C.PEDESTRIAN_MIX.items() if ctx.walker_pools.get(t)}
    if not mix:
        return None
    return weighted_choice(ctx.rng, mix)


def track(x, y, d, speed, seconds, step=0.25):
    """Positions every `step` seconds from t = 0 to `seconds`."""
    n = max(1, int(math.ceil(seconds / step)))
    return [(x + d[0] * speed * k * step, y + d[1] * speed * k * step) for k in range(n + 1)]


def walk_is_clear(ctx, path, tracks):
    for other in tracks:
        if any(math.hypot(a[0] - b[0], a[1] - b[1]) < C.PED_MIN_GAP for a, b in zip(path, other)):
            return False
    return not any(rects_overlap((*ctx.frame.local(*p), 0.6, 0.6), r, 0.2, 0.2)
                   for p in path for r in ctx.rects)


def ped_box(ctx, ped, x, y):
    return ctx.cam.box(box_corners(x, y, ped["ground_z"], ped["yaw"], 0.5, 0.5, C.PED_HEIGHT[ped["type"]]))


def position_at(ped, seconds):
    x, y = ped["spawn"][:2]
    d = ped["direction"]
    return x + d[0] * ped["speed"] * seconds, y + d[1] * ped["speed"] * seconds


def make_ped(ctx, role, ptype, speed, runner, x, y, d, capture_xy, **extra):
    return {
        "role": role, "type": ptype, "blueprint": ctx.rng.choice(ctx.walker_pools[ptype]),
        "speed": round(speed, 3), "runner": runner,
        "spawn": [x, y, ground_z(ctx.map, x, y, ctx.site.center[2]) + 1.0],
        "ground_z": ground_z(ctx.map, capture_xy[0], capture_xy[1], ctx.site.center[2]),
        "yaw": math.degrees(math.atan2(d[1], d[0])), "direction": [d[0], d[1]],
        "capture_xy": list(capture_xy), **extra,
    }


def crossing_line(site, direction, lateral):
    """Start point and unit direction of a walk across the site, shifted `lateral` m along the road."""
    origin, axis = (site.start, site.axis) if direction == 1 else (site.end, (-site.axis[0], -site.axis[1]))
    return (origin[0] + site.along[0] * lateral, origin[1] + site.along[1] * lateral), axis


def plan_crossing(ctx, n, first_frame_s, total_s, tracks):
    """n people crossing; each is at a random spot on the crossing at the first frame
    and keeps walking until the last frame."""
    L = ctx.site.length
    band = max(0.2, ctx.site.half_width - 0.5)
    peds = []
    for _ in range(n):
        for _try in range(40):
            ptype = pick_type(ctx)
            runner = ctx.rng.random() < C.PED_RUNNER_PROBABILITY
            speed = ctx.rng.uniform(*(C.PED_RUN_SPEED_RANGE if runner else C.PED_SPEED_RANGE[ptype]))
            direction = ctx.rng.choice((1, -1))
            p = ctx.rng.uniform(*C.PED_PROGRESS_RANGE)
            p0 = p - speed * first_frame_s / L
            if p0 < -1.5 / L:
                continue    # would start too far back on the sidewalk
            lat = ctx.rng.uniform(-band, band)
            (ox, oy), d = crossing_line(ctx.site, direction, lat)
            sx, sy = ox + d[0] * p0 * L, oy + d[1] * p0 * L
            path = track(sx, sy, d, speed, total_s)
            if not walk_is_clear(ctx, path, tracks):
                continue
            cx, cy = sx + d[0] * speed * first_frame_s, sy + d[1] * speed * first_frame_s
            peds.append(make_ped(ctx, "crossing", ptype, speed, runner, sx, sy, d, (cx, cy),
                                 capture_progress=round(p, 3), lateral_offset=round(lat, 3)))
            tracks.append(path)
            break
        else:
            return None
    return peds


def plan_sidewalk(ctx, n, first_frame_s, total_s, tracks):
    """People walking along / standing on the sidewalk, not crossing (negatives)."""
    walks = [w for w in (ctx.xs["same_sidewalk"], ctx.xs["opposite_sidewalk"]) if w is not None]
    peds = []
    ego_long = ctx.frame.local(ctx.ego["x"], ctx.ego["y"])[0]
    for _ in range(n if walks else 0):
        for _try in range(25):
            sw = ctx.rng.choice(walks)
            wp = step_along(sw, ctx.rng.uniform(*C.SIDEWALK_PED_DISTANCE), forward=ctx.rng.random() < 0.5)
            if wp is None:
                continue
            f, r = forward_right(wp.transform.rotation.yaw)
            j = ctx.rng.uniform(-0.5, 0.5)
            x, y = wp.transform.location.x + r[0] * j, wp.transform.location.y + r[1] * j
            lo, _ = ctx.frame.local(x, y)
            if abs(lo) < ctx.site.half_width + 2.0 or lo < ego_long + 3:
                continue    # not at the crossing mouth, not behind the camera
            ptype = pick_type(ctx)
            standing = ctx.rng.random() < C.SIDEWALK_PED_STANDING_PROBABILITY
            speed = 0.0 if standing else ctx.rng.uniform(*C.PED_SPEED_RANGE[ptype])
            sign = ctx.rng.choice((1, -1))
            d = (f[0] * sign, f[1] * sign)
            path = track(x, y, d, speed, total_s)
            if not walk_is_clear(ctx, path, tracks):
                continue
            cx, cy = x + d[0] * speed * first_frame_s, y + d[1] * speed * first_frame_s
            peds.append(make_ped(ctx, "sidewalk", ptype, speed, False, x, y, d, (cx, cy)))
            if standing:
                peds[-1]["yaw"] = ctx.rng.uniform(-180, 180)
            tracks.append(path)
            break
    return peds
