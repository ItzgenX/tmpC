"""Every occluder layout, checked on the plan with plain top-view geometry (not the
planner's own camera model):

  ego -> car -> pedestrian / ego -> car -> car -> pedestrian, in the ego's own lane,
  in the next lane, behind a parked car, and while turning left / right at a junction.

The car(s) must be where the layout says, in order between the ego and the pedestrian,
and the pedestrian's walk must go visible -> hidden -> visible, really passing behind a
car at the most hidden frame.
"""
import math
import unittest

import helpers as H
from helpers import C
from engine.people import position_at

SEEDS = (1, 2)


def ahead_axis(plan):
    """Unit vector along the road at the crossing, pointing the way the ego drives."""
    s = plan["site"]
    ax, ay = s["end"][0] - s["start"][0], s["end"][1] - s["start"][1]
    n = math.hypot(ax, ay)
    along = (-ay / n, ax / n)
    cx, cy = (s["start"][0] + s["end"][0]) / 2, (s["start"][1] + s["end"][1]) / 2
    if (cx - plan["ego"]["x"]) * along[0] + (cy - plan["ego"]["y"]) * along[1] < 0:
        along = (-along[0], -along[1])
    return along, (cx, cy), (ax / n, ay / n)


def ped_at(plan, k):
    target = plan["pedestrians"][plan["target_pedestrian"]]
    return position_at(target, plan["frames"][k]["tick"] * C.FIXED_DELTA_SECONDS)


def peak_index(plan):
    occ = [f["planned_occlusion"] for f in plan["frames"]]
    return occ.index(max(occ))


class LayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H.configure()
        cls.planner = H.make_planner()
        cls.plans = {name: [H.plan_layout(cls.planner, scene, layout, seed) for seed in SEEDS]
                     for name, scene, layout in H.LAYOUTS}

    def each(self):
        for name, scene, layout in H.LAYOUTS:
            for plan in self.plans[name]:
                if plan is not None:
                    yield name, scene, layout, plan

    def route_pos(self, plan, x, y):
        """How far along the ego's way a point is: along the road, or along the turn path."""
        if plan["scene"] != "occluded_turning":
            along, (cx, cy), _ = ahead_axis(plan)
            return (x - cx) * along[0] + (y - cy) * along[1]
        path = self.turn_path(plan)
        return min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2) * 0.5

    def turn_path(self, plan):
        site = next(s for s in self.planner.sites if s.id == plan["site"]["id"])
        e = plan["ego"]
        return min(self.planner.turn_paths(site),
                   key=lambda p: min((q[0] - e["x"]) ** 2 + (q[1] - e["y"]) ** 2 for q in p))

    def occluders(self, plan):
        return [v for v in plan["vehicles"] if v["role"] == "occluder"]

    # ------------------------------------------------------------------

    def test_every_layout_can_be_planned(self):
        for name, plans in self.plans.items():
            with self.subTest(name):
                self.assertTrue(all(plans), f"could not plan '{name}'")

    def test_layout_is_what_was_asked(self):
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                self.assertEqual(plan["layout"]["occluder_lane"], layout.get("lane", "parked"))
                self.assertEqual(plan["layout"]["occluders"], layout["cars"])
                self.assertEqual(plan["layout"]["turn"], layout.get("side"))
                self.assertEqual(len(self.occluders(plan)), layout["cars"])
                if layout.get("side"):
                    self.assertEqual(plan["ego"]["approach"], f"turn_{layout['side']}")

    def test_order_ego_then_cars_then_pedestrian(self):
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                ego = self.route_pos(plan, plan["ego"]["x"], plan["ego"]["y"])
                cars = [self.route_pos(plan, v["x"], v["y"]) for v in self.occluders(plan)]
                ped = self.route_pos(plan, *ped_at(plan, peak_index(plan)))
                order = [ego] + cars + [ped]
                self.assertEqual(order, sorted(order), f"ego, car(s), pedestrian along the way: {order}")

    def test_cars_are_in_the_right_lane(self):
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                along, _, _ = ahead_axis(plan)
                right = (-along[1], along[0])
                e = plan["ego"]
                for v in self.occluders(plan):
                    if scene == "occluded_turning":
                        path = self.turn_path(plan)
                        gap = min(math.hypot(q[0] - v["x"], q[1] - v["y"]) for q in path)
                        self.assertLess(gap, 0.6, "turning car is not on the ego's turn path")
                        continue
                    side = (v["x"] - e["x"]) * right[0] + (v["y"] - e["y"]) * right[1]
                    if layout.get("lane") == "ego":
                        self.assertLess(abs(side), 1.5, "car is not in the ego's lane")
                    elif layout.get("lane") == "next":
                        self.assertTrue(2.3 < abs(side) < 4.8, f"car is not in the next lane ({side:.2f} m)")
                    else:
                        self.assertTrue(v.get("parked") and abs(side) > 2.0, "parked car not at the curb")

    def test_visible_then_hidden_then_visible(self):
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                occ = [f["planned_occlusion"] for f in plan["frames"]]
                lo, hi = plan["occlusion_range"]
                self.assertLessEqual(occ[0], C.FULLY_VISIBLE, f"first frame not visible: {occ}")
                self.assertLessEqual(occ[-1], C.FULLY_VISIBLE, f"last frame not visible: {occ}")
                self.assertTrue(0 < peak_index(plan) < len(occ) - 1, f"most hidden frame at an end: {occ}")
                self.assertTrue(lo - 0.011 <= max(occ) <= hi + 0.011, f"peak {max(occ)} outside {lo}-{hi}")

    def test_pedestrian_really_walks_behind_a_car(self):
        """3D sight lines from the camera to the pedestrian's body: at the most hidden frame
        a good part of them runs through a car, at the first and last frames none does."""
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                cam = H.camera_xyz(plan["ego"], self.planner.ego_z_offset)
                solids = [b for v in self.occluders(plan) for b in H.vehicle_solids(v)]
                target = plan["pedestrians"][plan["target_pedestrian"]]
                peak = H.hidden_share(cam, target, *ped_at(plan, peak_index(plan)), solids)
                self.assertGreaterEqual(peak, 0.3, f"most hidden frame: only {peak:.0%} of sight lines blocked")
                for k in (0, len(plan["frames"]) - 1):
                    share = H.hidden_share(cam, target, *ped_at(plan, k), solids)
                    self.assertEqual(share, 0.0, f"frame {k} should be clearly visible, {share:.0%} blocked")

    def test_pedestrian_walks_on_the_crossing(self):
        for name, scene, layout, plan in self.each():
            with self.subTest(name):
                along, (cx, cy), axis = ahead_axis(plan)
                target = plan["pedestrians"][plan["target_pedestrian"]]
                d = target["direction"]
                self.assertAlmostEqual(abs(d[0] * axis[0] + d[1] * axis[1]), 1.0, places=6,
                                       msg="pedestrian does not walk straight across the road")
                for k in range(len(plan["frames"])):
                    px, py = ped_at(plan, k)
                    off = (px - cx) * along[0] + (py - cy) * along[1]
                    self.assertLessEqual(abs(off), plan["site"]["half_width"] + 0.01,
                                         f"frame {k}: pedestrian is off the crossing")


if __name__ == "__main__":
    unittest.main()
