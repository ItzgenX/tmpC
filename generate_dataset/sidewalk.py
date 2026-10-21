"""Road cross-section geometry: lanes across the road, sidewalks, stepping along a lane.

All of this runs client-side on the OpenDRIVE map, no server round trips.
"""
import math

import carla

MAX_LANE_STEPS = 12


def forward_right(yaw_deg):
    # CARLA is left-handed: right of heading (cos, sin) is (-sin, cos)
    yaw = math.radians(yaw_deg)
    fx, fy = math.cos(yaw), math.sin(yaw)
    return (fx, fy), (-fy, fx)


def heading_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def xy(loc):
    return (loc.x, loc.y)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def step_along(wp, dist, forward=True):
    """Waypoint `dist` metres ahead/behind on the same lane, straightest branch at junctions."""
    if dist < 0.05:
        return wp
    cands = wp.next(dist) if forward else wp.previous(dist)
    if not cands:
        return None
    yaw = wp.transform.rotation.yaw
    return min(cands, key=lambda c: heading_diff(c.transform.rotation.yaw, yaw))


def same_side(a, b):
    return (a.lane_id > 0) == (b.lane_id > 0)


def _walk_outward(start):
    # get_right_lane() always points away from the centre line on both sides
    lanes, sidewalk = [], None
    cur = start
    for _ in range(MAX_LANE_STEPS):
        if cur is None:
            break
        if cur.lane_type == carla.LaneType.Driving:
            lanes.append(cur)
        elif cur.lane_type == carla.LaneType.Sidewalk:
            sidewalk = cur
            break
        cur = cur.get_right_lane()
    return lanes, sidewalk


def road_cross_section(wp):
    """Driving lanes on both sides of the road at wp, plus the sidewalk on each edge.

    get_left_lane() flips meaning once it crosses the centre line (the left of an
    oncoming lane points back at you), so naive left-walking ping-pongs. Instead:
    walk inward to the innermost lane on wp's side, hop the centre once, then
    walk outward on each side with get_right_lane().
    """
    inner = wp
    for _ in range(MAX_LANE_STEPS):
        left = inner.get_left_lane()
        if left is None or not same_side(left, wp):
            break
        inner = left
    across = inner.get_left_lane()
    if across is not None and same_side(across, wp):
        across = None

    same_lanes, same_sidewalk = _walk_outward(inner)
    opp_lanes, opp_sidewalk = _walk_outward(across) if across is not None else ([], None)
    return {
        "same": same_lanes,             # travel in wp's direction
        "opposite": opp_lanes,          # travel against wp's direction
        "same_sidewalk": same_sidewalk,
        "opposite_sidewalk": opp_sidewalk,
    }


def lane_key(wp):
    return f"{wp.road_id}:{wp.section_id}:{wp.lane_id}"


def ground_z(carla_map, x, y, z_hint):
    """Surface height at (x, y): road, sidewalk, whatever lane is there."""
    wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z_hint), project_to_road=True,
                                lane_type=carla.LaneType.Any)
    return wp.transform.location.z if wp is not None else z_hint
