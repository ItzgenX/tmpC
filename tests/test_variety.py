"""Variety: spots are used evenly (never more than MAX_USES_PER_SPOT), two views of the
same spot from the same lane are at least MIN_VIEW_DISTANCE_GAP apart, the camera height
varies, too few spots are reported up front, and maps are found by short name or path."""
import collections
import random
import unittest

import helpers as H
from helpers import C
from engine.planner import PlanError
from engine.run import build_jobs, resolve_maps, spot_shortages

USES = 3


class VarietyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H.configure(MAX_USES_PER_SPOT=USES)
        cls.planner = H.make_planner()
        cls.spots = cls.planner.sites_of(["road"])
        cls.made, cls.reasons = [], collections.Counter()
        row = dict(C.WEATHERS["clear"])
        for i in range(len(cls.spots) * USES + 5):          # ask for more than the spots allow
            job = {"name": f"clear_jaywalk_{i:05d}", "weather": "clear", "scene": "jaywalk_road", "frames": 1}
            for attempt in range(400):
                try:
                    plan = cls.planner.plan(random.Random(f"v:{i}:{attempt}"), job, row)
                except PlanError as exc:
                    cls.reasons[str(exc)] += 1
                    continue
                cls.planner.register(plan)
                cls.made.append(plan)
                break

    def test_each_spot_used_at_most_max_uses(self):
        use = collections.Counter(p["site"]["id"] for p in self.made)
        self.assertLessEqual(max(use.values()), USES)
        self.assertEqual(len(self.made), len(self.spots) * USES, "every spot used exactly MAX_USES_PER_SPOT times")
        self.assertTrue(any(f"already used {USES}x" in r for r in self.reasons))

    def test_spots_used_evenly(self):
        order = [p["site"]["id"] for p in self.made]
        for k in range(len(order)):
            seen = collections.Counter(order[:k + 1])
            self.assertLessEqual(max(seen.values()) - min(seen.get(s.id, 0) for s in self.spots), 1,
                                 "a spot was reused before all others were used as often")

    def test_views_of_one_spot_are_apart(self):
        by_spot_lane = collections.defaultdict(list)
        for p in self.made:
            by_spot_lane[(p["site"]["id"], p["ego"]["lane"])].append(p["ego"]["distance"])
        for key, ds in by_spot_lane.items():
            ds = sorted(ds)
            self.assertTrue(all(b - a >= C.MIN_VIEW_DISTANCE_GAP - 1e-6 for a, b in zip(ds, ds[1:])), (key, ds))

    def test_camera_height_varies(self):
        lifts = [p["ego"]["camera_lift"] for p in self.made]
        lo, hi = C.CAMERA_LIFT_RANGE
        self.assertTrue(lo <= min(lifts) and max(lifts) <= hi and max(lifts) - min(lifts) > (hi - lo) / 2)

    def test_too_few_spots_reported(self):
        H.configure(WEATHERS={"clear": dict(enabled=True, images=1000, pedestrians=1, vehicles=0, cyclists=0,
                                            trucks=0, occlusion=0)}, MAX_USES_PER_SPOT=USES)
        jobs = build_jobs("FakeCity", 0, 1, {"crosswalk", "road"})
        self.assertTrue(spot_shortages(self.planner, jobs))
        H.configure(MAX_USES_PER_SPOT=USES)

    def test_maps_by_short_name_or_path(self):
        avail = ["/Game/Carla/Maps/Town10HD_Opt", "/Game/Carla/Maps/Town03_Opt", "/Game/Custom/Maps/MyCity"]
        found, missing = resolve_maps(["town10hd_opt", "Town03_Opt", "/Game/Custom/Maps/MyCity", "MyCity", "Nope"],
                                      avail)
        self.assertEqual(found, [("Town10HD_Opt", avail[0]), ("Town03_Opt", avail[1]), ("MyCity", avail[2])])
        self.assertEqual(missing, ["Nope"])


if __name__ == "__main__":
    unittest.main()
