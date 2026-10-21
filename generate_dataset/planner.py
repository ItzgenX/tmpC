"""Plans a whole scene as plain data before anything is spawned.

Every position, blueprint, speed and direction is decided and checked here
(no overlaps, crossing kept clear, crossing in view, no repeated viewpoint,
target pedestrian occluded by the requested amount). Only a plan that passes
reaches the simulator, and the plan is saved next to the image.
"""
import math

import advanced as C
from sidewalk import (forward_right, road_cross_section, step_along, heading_diff,
                      lane_key, ground_z, dot, sub)


class PlanError(Exception):
    pass


def weighted_choice(rng, weights):
    keys = [k for k, w in weights.items() if w > 0]
    total = sum(weights[k] for k in keys)
    x = rng.random() * total
    for k in keys:
        x -= weights[k]
        if x <= 0:
            return k
    return keys[-1]


def sample_weather(rng, name):
    w = {k: round(rng.uniform(*r), 2) for k, r in C.WEATHER_LOOK[name].items()}
    w.update({
        "sun_altitude_angle": round(rng.uniform(*C.SUN_ALTITUDE_RANGE), 2),
        "sun_azimuth_angle": round(rng.uniform(*C.SUN_AZIMUTH_RANGE), 2),
        "precipitation": 0.0, "precipitation_deposits": 0.0, "wetness": 0.0,
    })
    return {"name": name, "params": w}


class Frame:
    """Ego-aligned 2D frame centred on the crossing: long = ahead of ego, lat = to its right."""

    def __init__(self, origin, yaw):
        self.o = origin
        self.f, self.r = forward_right(yaw)

    def local(self, x, y):
        d = sub((x, y), self.o)
        return dot(d, self.f), dot(d, self.r)


def rects_overlap(a, b, gap_long, gap_lat):
    # a, b = (long, lat, length, width), axis-aligned in the ego frame
    return (abs(a[0] - b[0]) < (a[2] + b[2]) / 2 + gap_long and
            abs(a[1] - b[1]) < (a[3] + b[3]) / 2 + gap_lat)


class PlanCamera:
    """Pinhole model of the dashcam at the planned ego pose (road roll ignored)."""

    def __init__(self, ego, z_offset):
        yaw = math.radians(ego["yaw"])
        pitch = math.radians(ego["pitch"] + C.CAMERA_PITCH)
        cx, cy, cz = C.CAMERA_LOCATION
        fx, fy = math.cos(yaw), math.sin(yaw)
        self.pos = (ego["x"] + fx * cx - fy * cy, ego["y"] + fy * cx + fx * cy, ego["z"] + z_offset + cz)
        cp, sp = math.cos(pitch), math.sin(pitch)
        self.F = (cp * fx, cp * fy, sp)
        self.R = (-fy, fx, 0.0)
        self.U = (-sp * fx, -sp * fy, cp)
        self.focal = C.IMAGE_W / (2.0 * math.tan(math.radians(C.CAMERA_FOV) / 2.0))

    def box(self, corners):
        """2D box [x0, y0, x1, y1] and mean depth of 3D corners, None if behind camera."""
        us, vs, ds = [], [], []
        for p in corners:
            d = (p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2])
            depth = sum(a * b for a, b in zip(d, self.F))
            if depth < 0.5:
                return None
            us.append(C.IMAGE_W / 2 + self.focal * sum(a * b for a, b in zip(d, self.R)) / depth)
            vs.append(C.IMAGE_H / 2 - self.focal * sum(a * b for a, b in zip(d, self.U)) / depth)
            ds.append(depth)
        return [min(us), min(vs), max(us), max(vs)], sum(ds) / len(ds)


def box_corners(x, y, z, yaw_deg, length, width, height):
    f, r = forward_right(yaw_deg)
    out = []
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            px = x + f[0] * a * length + r[0] * b * width
            py = y + f[1] * a * length + r[1] * b * width
            out += [(px, py, z), (px, py, z + height)]
    return out


def coverage(target, occluder):
    """Fraction of the target's on-screen box covered by a nearer occluder box."""
    if target is None or occluder is None or occluder[1] >= target[1]:
        return 0.0
    t, o = target[0], occluder[0]
    t = [max(t[0], 0), max(t[1], 0), min(t[2], C.IMAGE_W), min(t[3], C.IMAGE_H)]
    area = max(0.0, t[2] - t[0]) * max(0.0, t[3] - t[1])
    if area <= 0:
        return 0.0
    iw = max(0.0, min(t[2], o[2]) - max(t[0], o[0]))
    ih = max(0.0, min(t[3], o[3]) - max(t[1], o[1]))
    return iw * ih / area


class ScenePlanner:
    def __init__(self, carla_map, sites, env_vehicles, walker_pools, car_bps, truck_bps):
        self.map = carla_map
        self.sites = sites                    # list of CrossingSite
        self.env = env_vehicles               # list of (x, y, ext_x, ext_y, yaw)
        self.walker_pools = walker_pools      # {"child": [bp_id], ...}
        self.car_bps = car_bps                # {bp_id: [colors]}
        self.truck_bps = truck_bps
        self.ego_z_offset = 0.0               # set once the ego has settled on this map
        self.usage = {s.id: 0 for s in sites}
        self.used_signatures = set()
        self._xs_cache = {}
        self.S = C.settings_for(None)         # settings of the weather being planned

    def register(self, plan):
        """Count a saved scene so later plans spread across sites and never repeat a view."""
        sid = plan["site"]["id"]
        self.usage[sid] = self.usage.get(sid, 0) + 1
        self.used_signatures.add(plan["signature"])

    # ---------- site / ego ----------

    def _pick_site(self, rng):
        cw = [s for s in self.sites if s.kind == "crosswalk"]
        jw = [s for s in self.sites if s.kind == "jaywalk"]
        if cw and jw:
            pool = cw if rng.random() < self.S.crosswalk_chance else jw
        else:
            pool = cw or jw
        if not pool:
            raise PlanError("no crossing sites on this map")
        least = min(self.usage[s.id] for s in pool)
        return rng.choice([s for s in pool if self.usage[s.id] <= least + 1])

    def cross_section(self, site):
        if site.id not in self._xs_cache:
            self._xs_cache[site.id] = road_cross_section(site.road_wp)
        return self._xs_cache[site.id]

    def _env_rects(self, frame, max_long=60.0):
        rects = []
        heading = math.degrees(math.atan2(frame.f[1], frame.f[0]))
        for x, y, ex, ey, yaw in self.env:
            lo, la = frame.local(x, y)
            if abs(lo) > max_long or abs(la) > 30.0:
                continue
            d = heading_diff(yaw, heading) % 180.0
            if d < 30 or d > 150:
                rects.append((lo, la, 2 * ex, 2 * ey))
            elif 60 < d < 120:
                rects.append((lo, la, 2 * ey, 2 * ex))
            else:
                m = 2 * max(ex, ey)
                rects.append((lo, la, m, m))
        return rects

    def plan_ego(self, rng, site, distance_range=None):
        distance_range = distance_range or C.EGO_DISTANCE_RANGE
        xs = self.cross_section(site)
        groups = [g for g in ("same", "opposite") if xs[g]]
        if not groups:
            raise PlanError("no driving lanes at site")
        group = rng.choice(groups)
        lane = rng.choice(xs[group])
        d = rng.uniform(*distance_range)

        lane_loc = lane.transform.location
        lane_f, _ = forward_right(lane.transform.rotation.yaw)
        back = d - dot(sub(site.center[:2], (lane_loc.x, lane_loc.y)), lane_f)
        wp = step_along(lane, abs(back), forward=back < 0)
        if wp is None:
            raise PlanError("road ends before ego distance")

        t = wp.transform
        _, r = forward_right(t.rotation.yaw)
        lat_j = rng.uniform(-C.EGO_LATERAL_JITTER, C.EGO_LATERAL_JITTER)
        ex, ey = t.location.x + r[0] * lat_j, t.location.y + r[1] * lat_j
        yaw = t.rotation.yaw + rng.uniform(-C.EGO_YAW_JITTER_DEG, C.EGO_YAW_JITTER_DEG)

        # crossing must be ahead, roughly centred and at the planned distance
        to_c = sub(site.center[:2], (ex, ey))
        dist = math.hypot(*to_c)
        if heading_diff(math.degrees(math.atan2(to_c[1], to_c[0])), yaw) > C.MAX_VIEW_BEARING_DEG:
            raise PlanError("crossing not in view from ego")
        if not (distance_range[0] - 3 <= dist <= distance_range[1] + 5):
            raise PlanError("ego distance off after road curvature")

        ego = {
            "x": ex, "y": ey, "z": t.location.z, "yaw": yaw,
            "pitch": t.rotation.pitch, "roll": t.rotation.roll,
            "lane": lane_key(lane), "distance": round(dist, 2), "approach": group,
        }
        return ego, xs, group

    # ---------- pedestrians ----------

    def _plan_crossing_peds(self, rng, site, frame, blocked, preroll_s):
        types = {t: w for t, w in self.S.pedestrian_types.items() if self.walker_pools.get(t)}
        if not types:
            raise PlanError("no pedestrian blueprints")
        n = rng.randint(*self.S.crossing_pedestrians)
        L = site.length
        band = max(0.2, site.half_width - 0.5)
        peds, tracks = [], []

        for _ in range(n):
            for _try in range(30):
                ptype = weighted_choice(rng, types)
                runner = rng.random() < C.PED_RUNNER_PROBABILITY
                speed = rng.uniform(*(C.PED_RUN_SPEED_RANGE if runner else C.PED_SPEED_RANGE[ptype]))
                direction = rng.choice((1, -1))
                p = rng.uniform(*C.PED_PROGRESS_RANGE)
                p0 = p - speed * preroll_s / L
                if p0 < -1.5 / L:
                    continue    # would start too far back on the sidewalk
                lat = rng.uniform(-band, band)
                origin, axis = (site.start, site.axis) if direction == 1 else \
                               (site.end, (-site.axis[0], -site.axis[1]))
                sx = origin[0] + axis[0] * p0 * L + site.along[0] * lat
                sy = origin[1] + axis[1] * p0 * L + site.along[1] * lat
                walk = speed * preroll_s
                track = [(sx + axis[0] * walk * k / 5.0, sy + axis[1] * walk * k / 5.0) for k in range(6)]
                if any(math.hypot(a[0] - b[0], a[1] - b[1]) < C.PED_MIN_GAP
                       for other in tracks for a, b in zip(track, other)):
                    continue
                if any(rects_overlap((*frame.local(*pt), 0.6, 0.6), rect, 0.3, 0.3)
                       for pt in track for rect in blocked):
                    continue    # would walk into a parked car
                gz = ground_z(self.map, track[-1][0], track[-1][1], site.center[2])
                peds.append({
                    "role": "crossing", "blueprint": rng.choice(self.walker_pools[ptype]), "type": ptype,
                    "speed": round(speed, 3), "runner": runner,
                    "spawn": [sx, sy, ground_z(self.map, sx, sy, site.center[2]) + 1.0],
                    "capture_xy": list(track[-1]), "ground_z": gz,
                    "yaw": math.degrees(math.atan2(axis[1], axis[0])),
                    "direction": [axis[0], axis[1]], "capture_progress": round(p, 3),
                    "lateral_offset": round(lat, 3),
                })
                tracks.append(track)
                break
            else:
                raise PlanError("could not place pedestrian without conflicts")
        return peds, tracks

    def _plan_sidewalk_peds(self, rng, site, frame, ego, xs, blocked, tracks, preroll_s):
        """People walking along / standing on the sidewalk, not crossing (negatives)."""
        n = rng.randint(*self.S.sidewalk_pedestrians)
        walks = [w for w in (xs["same_sidewalk"], xs["opposite_sidewalk"]) if w is not None]
        types = {t: w for t, w in self.S.pedestrian_types.items() if self.walker_pools.get(t)}
        peds = []
        for _ in range(n if walks else 0):
            for _try in range(20):
                sw = rng.choice(walks)
                dist = rng.uniform(*C.SIDEWALK_PED_DISTANCE)
                wp = step_along(sw, dist, forward=rng.random() < 0.5)
                if wp is None:
                    continue
                f, r = forward_right(wp.transform.rotation.yaw)
                j = rng.uniform(-0.5, 0.5)
                x, y = wp.transform.location.x + r[0] * j, wp.transform.location.y + r[1] * j
                lo, _ = frame.local(x, y)
                if abs(lo) < site.half_width + 2.0 or lo < frame.local(ego["x"], ego["y"])[0] + 3:
                    continue    # not at the crossing mouth, not behind the camera
                ptype = weighted_choice(rng, types)
                standing = rng.random() < C.SIDEWALK_PED_STANDING_PROBABILITY
                speed = 0.0 if standing else rng.uniform(*C.PED_SPEED_RANGE[ptype])
                sign = rng.choice((1, -1))
                d = (f[0] * sign, f[1] * sign)
                walk = speed * preroll_s
                track = [(x + d[0] * walk * k / 5.0, y + d[1] * walk * k / 5.0) for k in range(6)]
                if any(math.hypot(a[0] - b[0], a[1] - b[1]) < C.PED_MIN_GAP
                       for other in tracks for a, b in zip(track, other)):
                    continue
                if any(rects_overlap((*frame.local(*pt), 0.6, 0.6), rect, 0.3, 0.3)
                       for pt in track for rect in blocked):
                    continue
                z = wp.transform.location.z
                peds.append({
                    "role": "sidewalk", "blueprint": rng.choice(self.walker_pools[ptype]), "type": ptype,
                    "speed": round(speed, 3), "runner": False,
                    "spawn": [x, y, z + 1.0], "capture_xy": list(track[-1]), "ground_z": z,
                    "yaw": math.degrees(math.atan2(d[1], d[0])) if not standing else rng.uniform(-180, 180),
                    "direction": [d[0], d[1]],
                })
                tracks.append(track)
                break
        return peds

    # ---------- traffic ----------

    def _vehicle_at(self, lane, s, frame, drift):
        """Pose `s` metres along the ego frame (+ = past the crossing) on `lane`."""
        loc = lane.transform.location
        lane_f, _ = forward_right(lane.transform.rotation.yaw)
        delta = s - frame.local(loc.x, loc.y)[0]
        aligned = dot(lane_f, frame.f) > 0
        wp = step_along(lane, abs(delta), forward=(delta > 0) == aligned)
        if wp is None:
            return None
        t = wp.transform
        _, r = forward_right(t.rotation.yaw)
        return wp, t.location.x + r[0] * drift, t.location.y + r[1] * drift

    def _vehicle_dict(self, rng, category, wp, vx, vy, direction, side, lo):
        pool = self.truck_bps if category == "truck" else self.car_bps
        bp_id = rng.choice(sorted(pool))
        t = wp.transform
        return {
            "blueprint": bp_id, "category": category,
            "color": rng.choice(pool[bp_id]) if pool[bp_id] else None,
            "x": vx, "y": vy, "z": t.location.z, "yaw": t.rotation.yaw, "pitch": t.rotation.pitch,
            "direction": direction, "side": side, "long": round(lo, 2), "lane": lane_key(wp),
        }

    @staticmethod
    def _max_drift(direction, lane, width):
        # same-direction traffic stays inside a lane treated as EGO_LANE_WIDTH wide
        if direction == "same":
            return max(0.0, C.EGO_LANE_WIDTH / 2 - width / 2)
        return min(C.ONCOMING_MAX_DRIFT, max(0.0, (lane.lane_width - width) / 2))

    def _vehicle_box(self, cam, category, vx, vy, z, yaw):
        l, w, h = C.TRUCK_SIZE if category == "truck" else C.CAR_SIZE
        return cam.box(box_corners(vx, vy, z, yaw, l, w, h))

    def _place_vehicle(self, rng, category, site, frame, ego, xs, ego_group, rects, cam, protect):
        """A normal traffic vehicle that must not cover the protected target pedestrian."""
        length, width, _ = C.TRUCK_SIZE if category == "truck" else C.CAR_SIZE
        other_group = "opposite" if ego_group == "same" else "same"
        corridor = site.half_width + C.CROSSING_CLEARANCE + length / 2
        near_lo = -(ego["distance"] - C.EGO_CLEARANCE - length / 2)

        for _ in range(40):
            direction = "same" if rng.random() < self.S.same_direction_chance else "oncoming"
            group = ego_group if direction == "same" else other_group
            if not xs[group]:
                direction, group = ("oncoming", other_group) if direction == "same" else ("same", ego_group)
            lane = rng.choice(xs[group])
            near_ok = lane_key(lane) != ego["lane"] and near_lo < -corridor
            if near_ok and rng.random() < C.NEAR_SIDE_PROBABILITY:
                s, side = rng.uniform(near_lo, -corridor), "near"
            elif corridor < C.TRAFFIC_FAR_MAX:
                s, side = rng.uniform(corridor, C.TRAFFIC_FAR_MAX), "far"
            else:
                continue

            max_drift = self._max_drift(direction, lane, width)
            placed = self._vehicle_at(lane, s, frame, rng.uniform(-max_drift, max_drift))
            if placed is None:
                continue
            wp, vx, vy = placed
            lo, la = frame.local(vx, vy)
            if abs(lo) < corridor or lo < near_lo or lo > C.TRAFFIC_FAR_MAX + 5:
                continue    # curvature moved it onto the crossing / next to the ego
            mine = (lo, la, length, width)
            if any(rects_overlap(mine, rect, C.VEHICLE_GAP, 0.3) for rect in rects):
                continue
            box = self._vehicle_box(cam, category, vx, vy, wp.transform.location.z, wp.transform.rotation.yaw)
            if protect is not None and coverage(protect, box) > C.PROTECT_TARGET_MAX_COVER:
                continue
            rects.append(mine)
            return self._vehicle_dict(rng, category, wp, vx, vy, direction, side, lo)
        return None

    def _place_occluder(self, rng, category, level, site, frame, ego, xs, ego_group, rects, cam, target_box):
        """Car/truck between camera and target, positioned so it covers the requested amount."""
        lo_t, hi_t = C.OCCLUSION_LEVELS[level]
        goal = (lo_t + hi_t) / 2
        length, width, _ = C.TRUCK_SIZE if category == "truck" else C.CAR_SIZE
        corridor = site.half_width + C.CROSSING_CLEARANCE + length / 2
        near_lo = -(ego["distance"] - C.OCCLUDER_MIN_AHEAD - length / 2)
        if near_lo >= -corridor:
            raise PlanError("no room for an occluder")
        lanes = [(ln, "same" if g == ego_group else "oncoming") for g in ("same", "opposite") for ln in xs[g]]

        best = None
        for _ in range(80):
            lane, direction = rng.choice(lanes)
            drift_max = self._max_drift(direction, lane, width)
            placed = self._vehicle_at(lane, rng.uniform(near_lo, -corridor), frame,
                                      rng.uniform(-drift_max, drift_max))
            if placed is None:
                continue
            wp, vx, vy = placed
            lo, la = frame.local(vx, vy)
            if abs(lo) < corridor or lo < near_lo:
                continue
            mine = (lo, la, length, width)
            if any(rects_overlap(mine, rect, C.VEHICLE_GAP, 0.3) for rect in rects):
                continue
            cov = coverage(target_box, self._vehicle_box(cam, category, vx, vy,
                                                         wp.transform.location.z, wp.transform.rotation.yaw))
            if lo_t <= cov <= hi_t and (best is None or abs(cov - goal) < best[0]):
                best = (abs(cov - goal), wp, vx, vy, direction, lo, mine, cov)
        if best is None:
            raise PlanError(f"no occluder position for level '{level}'")
        _, wp, vx, vy, direction, lo, mine, cov = best
        rects.append(mine)
        v = self._vehicle_dict(rng, category, wp, vx, vy, direction, "near", lo)
        v["occluder"] = True
        v["planned_cover"] = round(cov, 3)
        return v

    # ---------- whole scene ----------

    def plan(self, rng, weather_name, occlusion_level):
        self.S = S = C.settings_for(weather_name)
        n_cars = rng.randint(*S.cars)
        want_truck = rng.random() < S.truck_chance
        if n_cars and not self.car_bps:
            raise PlanError("no car blueprints")
        if want_truck and not self.truck_bps:
            raise PlanError("no truck blueprints")
        site = self._pick_site(rng)
        dist_range = C.EGO_DISTANCE_RANGE if occlusion_level == "none" else C.OCCLUDED_EGO_DISTANCE_RANGE
        ego, xs, ego_group = self.plan_ego(rng, site, dist_range)
        signature = (f"{site.id}|{ego['lane']}|{int(ego['distance'] // C.DISTANCE_BIN)}"
                     f"|{weather_name}|{occlusion_level}")
        if signature in self.used_signatures:
            raise PlanError("viewpoint already used")

        # measure along the road (not the jittered ego heading), pointing the way the ego drives
        road_yaw = math.degrees(math.atan2(site.along[1], site.along[0]))
        if heading_diff(road_yaw, ego["yaw"]) > 90:
            road_yaw += 180.0
        frame = Frame(site.center[:2], road_yaw)
        cam = PlanCamera(ego, self.ego_z_offset)
        env_rects = self._env_rects(frame)
        ego_lo, ego_la = frame.local(ego["x"], ego["y"])
        ego_rect = (ego_lo, ego_la, *C.EGO_SIZE)
        if any(rects_overlap(ego_rect, rect, 0.5, 0.2) for rect in env_rects):
            raise PlanError("ego spot overlaps a parked car")

        preroll_s = rng.uniform(*C.PED_PREROLL_SECONDS)
        crossing, tracks = self._plan_crossing_peds(rng, site, frame, env_rects, preroll_s)
        sidewalk = self._plan_sidewalk_peds(rng, site, frame, ego, xs, env_rects, tracks, preroll_s)

        # target = the crossing pedestrian whose occlusion level is controlled
        target = rng.randrange(len(crossing))
        t = crossing[target]
        target_box = cam.box(box_corners(t["capture_xy"][0], t["capture_xy"][1], t["ground_z"],
                                         t["yaw"], 0.5, 0.5, C.PED_HEIGHT[t["type"]]))
        if target_box is None or target_box[0][2] < 0 or target_box[0][0] > C.IMAGE_W:
            raise PlanError("target pedestrian out of view")

        # exact counts: the occluder is one of this image's cars, or its truck
        rects = env_rects + [ego_rect]
        vehicles = []
        if occlusion_level != "none":
            if n_cars == 0 and not want_truck:
                raise PlanError("occluded image rolled no car or truck to occlude with")
            use_truck = want_truck and (n_cars == 0 or rng.random() < C.OCCLUDER_TRUCK_PROBABILITY)
            vehicles.append(self._place_occluder(rng, "truck" if use_truck else "car", occlusion_level, site,
                                                 frame, ego, xs, ego_group, rects, cam, target_box))
            if use_truck:
                want_truck = False
            else:
                n_cars -= 1
        for _ in range(n_cars):
            v = self._place_vehicle(rng, "car", site, frame, ego, xs, ego_group, rects, cam, target_box)
            if v is None:
                raise PlanError("could not place car")
            vehicles.append(v)
        if want_truck:
            v = self._place_vehicle(rng, "truck", site, frame, ego, xs, ego_group, rects, cam, target_box)
            if v is None:
                raise PlanError("could not place truck")
            vehicles.append(v)

        return {
            "signature": signature,
            "weather": sample_weather(rng, weather_name),
            "occlusion_level": occlusion_level,
            "target_pedestrian": target,
            "site": site.to_json(),
            "ego": ego,
            "preroll_ticks": max(1, round(preroll_s / C.FIXED_DELTA_SECONDS)),
            "pedestrians": crossing + sidewalk,
            "vehicles": vehicles,
        }
