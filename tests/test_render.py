"""Each occluder layout rendered in the fake simulator through the real code (MapSession,
render_job, label measurement):

  - the cars are spawned exactly where the plan put them
  - the pedestrian is exactly where the plan says at every captured frame (spawn + walk timing)
  - measured from the rendered segmentation, the pedestrian is visible in the first and last
    frame and mostly hidden in the middle: visible -> hidden -> visible behind the car
  - boxes get CARLA's classes (Pedestrian, Car)
"""
import math
import unittest

import helpers as H
from helpers import C
from engine.labels import target_label
from engine.people import position_at
from engine.render import render_job

TOLERANCE = 0.05            # metres; the fake simulator moves walkers exactly
VISIBLE_MAX = 0.15          # measured occlusion of a "visible" frame
HIDDEN_MIN = 0.4            # measured occlusion of the most hidden frame


class RenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H.configure()
        cls.session = H.open_session()
        cls.results = {}
        for name, scene, layout in H.LAYOUTS:
            plan = H.plan_layout(cls.session.planner, scene, layout, seed=1)
            frames = render_job(cls.session, plan) if plan else None
            cls.results[name] = (plan, frames)

    @classmethod
    def tearDownClass(cls):
        with H.quiet():
            cls.session.close()

    def each(self):
        for name, (plan, frames) in self.results.items():
            if plan is not None:
                yield name, plan, frames

    def test_every_layout_rendered(self):
        for name, (plan, frames) in self.results.items():
            with self.subTest(name):
                self.assertIsNotNone(plan, "could not plan it")
                self.assertEqual(len(frames), len(plan["frames"]))

    def test_cars_spawned_where_planned(self):
        for name, plan, frames in self.each():
            with self.subTest(name):
                for k, frame in enumerate(frames):
                    for p in frame["positions"]:
                        if p["kind"] == "pedestrian":
                            continue
                        v = plan["vehicles"][p["plan_index"]]
                        gap = math.hypot(p["center"][0] - v["x"], p["center"][1] - v["y"])
                        self.assertLess(gap, TOLERANCE, f"frame {k}: {v['role']} {gap:.2f} m off its planned spot")

    def test_pedestrian_where_planned_at_every_frame(self):
        for name, plan, frames in self.each():
            with self.subTest(name):
                i = plan["target_pedestrian"]
                target = plan["pedestrians"][i]
                for k, (planned, frame) in enumerate(zip(plan["frames"], frames)):
                    want = position_at(target, planned["tick"] * C.FIXED_DELTA_SECONDS)
                    got = next(p["center"] for p in frame["positions"]
                               if p["kind"] == "pedestrian" and p["plan_index"] == i)
                    gap = math.hypot(got[0] - want[0], got[1] - want[1])
                    self.assertLess(gap, TOLERANCE, f"frame {k}: pedestrian {gap:.2f} m off its planned spot")

    def test_rendered_pedestrian_goes_visible_hidden_visible(self):
        for name, plan, frames in self.each():
            with self.subTest(name):
                occ = []
                for frame in frames:
                    lb = target_label(frame["labels"])
                    self.assertIsNotNone(lb, "target pedestrian has no label")
                    occ.append(lb["occlusion"])
                self.assertLessEqual(occ[0], VISIBLE_MAX, f"first frame not visible: {occ}")
                self.assertLessEqual(occ[-1], VISIBLE_MAX, f"last frame not visible: {occ}")
                self.assertGreaterEqual(max(occ), HIDDEN_MIN, f"never really hidden: {occ}")
                self.assertTrue(0 < occ.index(max(occ)) < len(occ) - 1, f"most hidden frame at an end: {occ}")

    def test_boxes_get_carla_classes(self):
        for name, plan, frames in self.each():
            with self.subTest(name):
                for frame in frames:
                    for lb in frame["labels"]:
                        if lb["kind"] == "pedestrian":
                            self.assertEqual(lb["carla_class"], "Pedestrian")
                        else:
                            self.assertEqual((lb["kind"], lb["carla_class"]), ("vehicle", "Car"))
                    occluders = [lb for lb in frame["labels"] if lb["meta"].get("role") == "occluder"]
                    self.assertTrue(occluders, "the occluding car got no box")


if __name__ == "__main__":
    unittest.main()
