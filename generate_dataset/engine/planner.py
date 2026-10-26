"""Plans one job (a scene, 1 or more frames) as plain data before anything is spawned.

Order: crossing site -> ego pose -> (occluder + target pedestrian walk) ->
other pedestrians -> traffic -> sidewalk people. Each step checks the
earlier ones (no overlaps, crossing in view, target not hidden by normal
traffic). A plan that fails any check is thrown away and re-planned.
"""
import math

import advanced as C
from engine.camera import Frame, PlanCamera, in_view, rects_overlap
from engine.occlusion import plan_occluded
from engine.people import ped_box, plan_crossing, plan_sidewalk, position_at, track
from engine.roads import dot, forward_right, heading_diff, lane_key, road_cross_section, step_along, sub
from engine.traffic import place_traffic


CATEGORY_KEY = {"car": "vehicles", "truck": "trucks", "cyclist": "cyclists"}


class PlanError(Exception):
    pass


class SceneContext:
    """Everything the planning helpers need about the scene being built."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class ScenePlanner:
    def __init__(self, carla_map, sites, env_vehicles, walker_pools, blueprints):
        self.map = carla_map
        self.sites = sites                    # list of CrossingSite
        self.env = env_vehicles               # parked map cars: (x, y, ext_x, ext_y, yaw)
        self.walker_pools = walker_pools      # {"child": [bp_id], ...}
        self.blueprints = blueprints          # {"car": {bp_id: [colors]}, "truck": ..., "cyclist": ...}
        self.ego_z_offset = 0.0               # set once the ego has settled on this map
        self.usage = {}                       # (site id, weather) -> jobs made there
        self.views = {}                       # (site id, weather) -> [(lane, distance), ...]
        self._xs_cache = {}

    def register(self, plan):
        """Count a saved job: spreads jobs over all spots, and blocks near-identical views."""
        key = (plan["site"]["id"], plan["weather"])
        self.usage[key] = self.usage.get(key, 0) + 1
        self.views.setdefault(key, []).append((plan["ego"]["lane"], plan["ego"]["distance"]))

    def _view_is_new(self, site, weather, ego):
        """At the same spot (same weather), the same lane needs MIN_VIEW_DISTANCE_GAP more or
        less distance to count as a different picture. Another lane / direction always does."""
        return all(lane != ego["lane"] or abs(dist - ego["distance"]) >= C.MIN_VIEW_DISTANCE_GAP
                   for lane, dist in self.views.get((site.id, weather), []))

    # ---------- site / ego ----------

    def sites_of(self, kinds):
        return [s for s in self.sites if s.kind in kinds]

    def _pick_site(self, rng, kinds, weather):
        """A random spot among the least used ones (in this weather), never over MAX_USES_PER_SPOT."""
        pool = self.sites_of(kinds)
        if not pool:
            raise PlanError(f"no {'/'.join(kinds)} sites on this map")
        if C.MAX_USES_PER_SPOT:
            pool = [s for s in pool if self.usage.get((s.id, weather), 0) < C.MAX_USES_PER_SPOT]
            if not pool:
                raise PlanError(f"every {'/'.join(kinds)} spot already used {C.MAX_USES_PER_SPOT}x in {weather}")
        least = min(self.usage.get((s.id, weather), 0) for s in pool)
        return rng.choice([s for s in pool if self.usage.get((s.id, weather), 0) == least])

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
                rects.append((lo, la, 2 * max(ex, ey), 2 * max(ex, ey)))
        return rects

    def plan_ego(self, rng, site, distance_range=C.EGO_DISTANCE_RANGE):
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
        # camera height / tilt vary per image: the camera is fixed to the ego, so the
        # (physics-free) ego is lifted a little and tilted; the bonnet stays in the same place
        lift = rng.uniform(*C.CAMERA_LIFT_RANGE)
        tilt = rng.uniform(-C.CAMERA_PITCH_JITTER_DEG, C.CAMERA_PITCH_JITTER_DEG)

        # crossing must be ahead, roughly centred and at the planned distance
        to_c = sub(site.center[:2], (ex, ey))
        dist = math.hypot(*to_c)
        if heading_diff(math.degrees(math.atan2(to_c[1], to_c[0])), yaw) > C.MAX_VIEW_BEARING_DEG:
            raise PlanError("crossing not in view from ego")
        if not (distance_range[0] - 3 <= dist <= distance_range[1] + 5):
            raise PlanError("ego distance off after road curvature")

        ego = {"x": ex, "y": ey, "z": t.location.z + lift, "yaw": yaw,
               "pitch": t.rotation.pitch + tilt, "roll": t.rotation.roll, "camera_lift": round(lift, 3),
               "lane": lane_key(lane), "distance": round(dist, 2), "approach": group}
        return ego, xs, group

    # ---------- one job ----------

    def plan(self, rng, job, params):
        """job = {"name", "weather", "scene", "frames"}; params = the weather row to use."""
        scene = job["scene"]
        occluded = scene in C.OCCLUDED_SCENES
        counts = {k: C.roll(params.get(k), rng) for k in C.COUNT_KEYS}
        requested = dict(counts)
        occ_range = C.occlusion_range(params.get("occlusion")) if occluded else (0.0, 0.0)
        peak = 0.0
        for cat, key in CATEGORY_KEY.items():
            if counts[key] and not self.blueprints[cat]:
                raise PlanError(f"no {cat} blueprints on this server")
        if not any(self.walker_pools.values()):
            raise PlanError("no pedestrian blueprints")

        site = self._pick_site(rng, C.SCENE_SITE_KINDS[scene], job["weather"])
        ego, xs, ego_group = self.plan_ego(
            rng, site, C.OCCLUDED_EGO_DISTANCE_RANGE if occluded else C.EGO_DISTANCE_RANGE)
        if not self._view_is_new(site, job["weather"], ego):
            raise PlanError("too close to a view already used at this spot")
        signature = f"{site.id}|{ego['lane']}|{ego['distance']:.1f}|{job['weather']}|{scene}"

        # road-aligned frame (not the jittered ego heading), pointing the way the ego drives
        road_yaw = math.degrees(math.atan2(site.along[1], site.along[0]))
        if heading_diff(road_yaw, ego["yaw"]) > 90:
            road_yaw += 180.0
        frame = Frame(site.center[:2], road_yaw)
        env_rects = self._env_rects(frame)
        ego_rect = (*frame.local(ego["x"], ego["y"]), *C.EGO_SIZE)
        if any(rects_overlap(ego_rect, r, 0.5, 0.2) for r in env_rects):
            raise PlanError("ego spot overlaps a parked car")
        ctx = SceneContext(rng=rng, map=self.map, site=site, ego=ego, xs=xs, ego_group=ego_group,
                           frame=frame, cam=PlanCamera(ego, self.ego_z_offset),
                           rects=env_rects + [ego_rect], walker_pools=self.walker_pools,
                           blueprints=self.blueprints)

        # --- target pedestrian (+ occluders) and frame timing ---
        tracks = []
        vehicles = []
        if occluded:
            if counts["vehicles"] + counts["trucks"] == 0:
                raise PlanError("occluded scene needs a car or truck")
            result = plan_occluded(ctx, scene, occ_range, counts["vehicles"], counts["trucks"], job["frames"])
            if result is None:
                raise PlanError("no layout where the pedestrian walks behind the car (visible -> hidden "
                                "-> visible) within the occlusion range")
            target, vehicles, frames, ctx.rects, peak = result
            for v in vehicles:
                counts[CATEGORY_KEY[v["category"]]] -= 1
            total_s = frames[-1]["tick"] * C.FIXED_DELTA_SECONDS
            tracks.append(track(*target["spawn"][:2], target["direction"], target["speed"], total_s))
            peds = [target]
            n_others = counts["pedestrians"] - 1
        else:
            first = max(C.CAPTURE_SETTLE_TICKS + 2,
                        round(rng.uniform(*C.PED_PREROLL_SECONDS) / C.FIXED_DELTA_SECONDS))
            frames = [{"tick": first, "planned_occlusion": 0.0}]
            total_s = first * C.FIXED_DELTA_SECONDS
            peds = []
            n_others = counts["pedestrians"]

        first_s = frames[0]["tick"] * C.FIXED_DELTA_SECONDS
        others = plan_crossing(ctx, n_others, first_s, total_s, tracks)
        if others is None:
            raise PlanError("could not place pedestrians without conflicts")
        peds += others
        target_index = 0 if occluded else rng.randrange(len(peds))
        target = peds[target_index]

        protect = []
        for f in frames:
            box = ped_box(ctx, target, *position_at(target, f["tick"] * C.FIXED_DELTA_SECONDS))
            if not occluded and not in_view(box):
                raise PlanError("target pedestrian out of view")
            protect.append(box)

        # --- remaining traffic, none of it hiding the target ---
        for cat, key in (("truck", "trucks"), ("car", "vehicles"), ("cyclist", "cyclists")):
            for _ in range(counts[key]):
                v = place_traffic(ctx, cat, protect)
                if v is None:
                    raise PlanError(f"could not place {cat}")
                vehicles.append(v)

        peds += plan_sidewalk(ctx, C.roll(C.SIDEWALK_PEDESTRIANS, rng), first_s, total_s, tracks)

        return {
            "job": job["name"], "scene": scene, "weather": job["weather"], "signature": signature,
            "counts": requested, "occlusion_range": [round(v, 3) for v in occ_range], "occlusion_peak": peak,
            "site": site.to_json(), "ego": ego,
            "target_pedestrian": target_index, "frames": frames,
            "pedestrians": peds, "vehicles": vehicles,
        }
