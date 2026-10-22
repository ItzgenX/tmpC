"""Places cars, trucks and cyclists for a scene (as plan data, nothing is spawned).

Normal traffic   : any lane, same way as the ego or oncoming, before or past the crossing.
Adjacent cars    : 1-2 cars in the lane next to the ego, between ego and crossing (occluders).
Parked car       : a car at the curb right before the pedestrian's path (occluder).
"""
import advanced as C
from engine.camera import box_corners, coverage, rects_overlap
from engine.roads import dot, forward_right, lane_key, step_along


SIZE = {"car": C.CAR_SIZE, "truck": C.TRUCK_SIZE, "cyclist": C.CYCLIST_SIZE}


def vehicle_at(ctx, lane, s, drift):
    """Pose `s` metres along the road frame (+ = past the crossing) on `lane`, shifted `drift` to the lane's right."""
    loc = lane.transform.location
    lane_f, _ = forward_right(lane.transform.rotation.yaw)
    delta = s - ctx.frame.local(loc.x, loc.y)[0]
    aligned = dot(lane_f, ctx.frame.f) > 0
    wp = step_along(lane, abs(delta), forward=(delta > 0) == aligned)
    if wp is None:
        return None
    t = wp.transform
    _, r = forward_right(t.rotation.yaw)
    return wp, t.location.x + r[0] * drift, t.location.y + r[1] * drift


def vehicle_dict(ctx, category, wp, x, y, direction, role):
    pool = ctx.blueprints[category]
    bp_id = ctx.rng.choice(sorted(pool))
    t = wp.transform
    lo, _ = ctx.frame.local(x, y)
    return {
        "blueprint": bp_id, "category": category, "role": role,
        "color": ctx.rng.choice(pool[bp_id]) if pool[bp_id] else None,
        "x": x, "y": y, "z": t.location.z, "yaw": t.rotation.yaw, "pitch": t.rotation.pitch,
        "direction": direction, "long": round(lo, 2), "lane": lane_key(wp),
    }


def box_of(ctx, v):
    length, width, height = SIZE[v["category"]]
    return ctx.cam.box(box_corners(v["x"], v["y"], v["z"], v["yaw"], length, width, height))


def rect_of(ctx, v):
    length, width, _ = SIZE[v["category"]]
    lo, la = ctx.frame.local(v["x"], v["y"])
    return (lo, la, length, width)


def max_drift(direction, lane, width):
    # same-direction traffic stays inside a lane treated as EGO_LANE_WIDTH wide
    if direction == "same":
        return max(0.0, C.EGO_LANE_WIDTH / 2 - width / 2)
    return min(C.ONCOMING_MAX_DRIFT, max(0.0, (lane.lane_width - width) / 2))


def _direction_of(ctx, group):
    return "same" if group == ctx.ego_group else "oncoming"


def _lanes_by_direction(ctx, direction):
    other = "opposite" if ctx.ego_group == "same" else "same"
    group = ctx.ego_group if direction == "same" else other
    if not ctx.xs[group]:
        group = other if group == ctx.ego_group else ctx.ego_group
    return group, ctx.xs[group]


# ---------- normal traffic ----------

def place_traffic(ctx, category, protect_boxes):
    """One car / truck / cyclist anywhere sensible that does not hide the target pedestrian."""
    length, width, _ = SIZE[category]
    corridor = ctx.site.half_width + C.CROSSING_CLEARANCE + length / 2
    near_lo = -(ctx.ego["distance"] - C.EGO_CLEARANCE - length / 2)

    for _ in range(60):
        want = "same" if ctx.rng.random() < C.SAME_DIRECTION_CHANCE else "oncoming"
        group, lanes = _lanes_by_direction(ctx, want)
        if not lanes:
            return None
        lane = ctx.rng.choice(lanes)
        direction = _direction_of(ctx, group)
        in_ego_lane = lane_key(lane) == ctx.ego["lane"]
        near_ok = (category == "cyclist" or not in_ego_lane) and near_lo < -corridor
        if near_ok and ctx.rng.random() < C.NEAR_SIDE_PROBABILITY:
            s = ctx.rng.uniform(near_lo, -corridor)
        elif corridor < C.TRAFFIC_FAR_MAX:
            s = ctx.rng.uniform(corridor, C.TRAFFIC_FAR_MAX)
        else:
            continue

        if category == "cyclist":       # ride near the lane's outer edge
            drift = max(0.0, lane.lane_width / 2 - width / 2 - 0.3) + ctx.rng.uniform(-0.2, 0.2)
        else:
            m = max_drift(direction, lane, width)
            drift = ctx.rng.uniform(-m, m)
        placed = vehicle_at(ctx, lane, s, drift)
        if placed is None:
            continue
        wp, x, y = placed
        v = vehicle_dict(ctx, category, wp, x, y, direction, "traffic")
        lo = v["long"]
        if abs(lo) < corridor or lo < near_lo or lo > C.TRAFFIC_FAR_MAX + 5:
            continue    # curvature moved it onto the crossing / next to the ego
        rect = rect_of(ctx, v)
        if any(rects_overlap(rect, r, C.VEHICLE_GAP, 0.3) for r in ctx.rects):
            continue
        box = box_of(ctx, v)
        if any(coverage(p, [box]) > C.PROTECT_TARGET_MAX_COVER for p in protect_boxes):
            continue
        ctx.rects.append(rect)
        return v
    return None


# ---------- occluders ----------

def adjacent_occluders(ctx, categories, ped_long, rects):
    """Cars in a lane right next to the ego, between ego and the pedestrian's path."""
    ego_lo, ego_la = ctx.frame.local(ctx.ego["x"], ctx.ego["y"])
    lanes = []
    for group in ("same", "opposite"):
        for lane in ctx.xs[group]:
            lo, la = ctx.frame.local(lane.transform.location.x, lane.transform.location.y)
            if 2.3 < abs(la - ego_la) < 4.8:
                lanes.append((lane, group))
    if not lanes:
        return None
    lane, group = ctx.rng.choice(lanes)
    direction = _direction_of(ctx, group)

    out, placed_rects = [], []
    s = ego_lo + ctx.rng.uniform(*C.ADJACENT_CARS_AHEAD)
    for category in categories:
        length, width, _ = SIZE[category]
        s += length / 2
        if s + length / 2 > ped_long - 0.8:
            return None             # no room before the pedestrian's path
        m = max_drift(direction, lane, width)
        placed = vehicle_at(ctx, lane, s, ctx.rng.uniform(-m, m))
        if placed is None:
            return None
        wp, x, y = placed
        v = vehicle_dict(ctx, category, wp, x, y, direction, "occluder")
        rect = rect_of(ctx, v)
        if rect[0] + length / 2 > ped_long - 0.8:
            return None
        if any(rects_overlap(rect, r, C.VEHICLE_GAP, 0.3) for r in rects + placed_rects):
            return None
        out.append(v)
        placed_rects.append(rect)
        s += length / 2 + ctx.rng.uniform(1.5, 4.0)
    return out, placed_rects


def parked_occluder(ctx, category, ped_long, rects):
    """A car at the curb whose front ends just before the pedestrian's path."""
    choices = []
    for group in ("same", "opposite"):
        if ctx.xs[group]:
            outer = ctx.xs[group][-1]
            if lane_key(outer) != ctx.ego["lane"]:     # never parked inside the ego's own lane
                choices.append((outer, group))
    if not choices:
        return None
    lane, group = ctx.rng.choice(choices)
    length, width, _ = SIZE[category]
    target_long = ped_long - ctx.rng.uniform(*C.PARKED_NOSE_GAP) - length / 2
    drift = max(0.0, lane.lane_width / 2 - width / 2 - 0.15)     # pulled over to the curb
    placed = vehicle_at(ctx, lane, target_long, drift)
    if placed is None:
        return None
    wp, x, y = placed
    v = vehicle_dict(ctx, category, wp, x, y, _direction_of(ctx, group), "occluder")
    v["parked"] = True
    rect = rect_of(ctx, v)
    if abs(rect[0] - target_long) > 1.0 or rect[0] + length / 2 > ped_long - 0.3:
        return None
    if any(rects_overlap(rect, r, 0.5, 0.2) for r in rects):
        return None
    return [v], [rect]
