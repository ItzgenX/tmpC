"""A whole (small) run on the fake simulator, the way python main.py runs it: every image
has all its files under the same name in flat folders, every scene type is made, and a
second run finds everything done (resume)."""
import csv
import os
import shutil
import tempfile
import unittest

import helpers as H
from helpers import C
from engine.run import Generator

SMALL = {
    "clear": dict(enabled=True, images=4, pedestrians=1, vehicles=0, cyclists=0, trucks=0, occlusion=0),
    "rain": dict(enabled=True, images=20, pedestrians=1, vehicles=2, cyclists=0, trucks=0, occlusion=(50, 95)),
}


class RunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H.configure(WEATHERS=SMALL)
        cls.out = tempfile.mkdtemp(prefix="dataset_test_")
        with H.quiet():
            cls.code = Generator(cls.out).run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def files(self, folder):
        path = os.path.join(self.out, folder)
        return sorted(os.listdir(path)) if os.path.isdir(path) else []

    def test_run_finishes(self):
        self.assertEqual(self.code, C.EXIT_DONE)

    def test_every_image_has_all_its_files(self):
        images = [os.path.splitext(f)[0] for f in self.files("images")]
        self.assertEqual(len(images), 24)
        for folder, ext in (("depth", ".png"), ("seg_color", ".png"), ("labels", ".txt"), ("labels_3d", ".json"),
                            ("images_2d_bbox", ".jpg"), ("images_3d_bbox", ".jpg")):
            with self.subTest(folder):
                self.assertEqual([os.path.splitext(f)[0] for f in self.files(folder)], images)
                self.assertTrue(all(f.endswith(ext) for f in self.files(folder)))

    def test_folders_are_flat(self):
        for folder in ("images", "depth", "labels", "labels_3d"):
            path = os.path.join(self.out, folder)
            self.assertFalse(any(os.path.isdir(os.path.join(path, f)) for f in os.listdir(path)), folder)

    def test_every_scene_type_made(self):
        with open(os.path.join(self.out, "manifest.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 24)
        scenes = {r["scene"] for r in rows}
        # FakeCity's junction has 4 roads: no T-junction jaywalking there
        self.assertEqual(scenes, set(C.SCENE_SITE_KINDS) - {"jaywalk_junction"})
        self.assertIn("clear_crosswalk_00001.png", {r["image"] for r in rows})
        self.assertTrue(any(r["image"].startswith("rain_occluded_") for r in rows))

    def test_second_run_resumes_and_adds_nothing(self):
        before = self.files("images")
        with H.quiet():
            code = Generator(self.out).run()
        self.assertEqual(code, C.EXIT_DONE)
        self.assertEqual(self.files("images"), before)


class FlaggedTest(unittest.TestCase):
    """Images that fail a check: kept and listed (then --delete-flagged), or skipped."""
    TINY = {"clear": dict(enabled=True, images=3, pedestrians=1, vehicles=0, cyclists=0, trucks=0, occlusion=0)}

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="flagged_")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def run_failing(self, skip):
        # every target is "too small": every frame fails a check
        H.configure(WEATHERS=self.TINY, MIN_TARGET_PIXELS=10 ** 9, SKIP_FLAGGED_IMAGES=skip, MAX_RENDERS_PER_JOB=2)
        with H.quiet():
            return Generator(self.out).run()

    def test_flagged_images_kept_listed_and_deletable(self):
        from engine.tools import delete_flagged
        self.assertEqual(self.run_failing(skip=False), C.EXIT_DONE)
        with open(os.path.join(self.out, "flagged.csv"), encoding="utf-8") as f:
            flagged = list(csv.DictReader(f))
        self.assertEqual(len(flagged), 3)
        self.assertEqual(len(os.listdir(os.path.join(self.out, "images"))), 3)
        with H.quiet():
            delete_flagged(self.out, "", True)
        for folder in ("images", "depth", "labels", "labels_3d"):
            self.assertEqual(os.listdir(os.path.join(self.out, folder)), [], folder)

    def test_skip_flagged_images_saves_nothing(self):
        self.assertEqual(self.run_failing(skip=True), C.EXIT_DONE)
        self.assertEqual(os.listdir(os.path.join(self.out, "images")), [])
        with open(os.path.join(self.out, "skipped.csv"), encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)


if __name__ == "__main__":
    unittest.main()
