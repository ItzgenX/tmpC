"""python main.py --weather-preview, run against the fake simulator: one sheet with a row per
setting and a column per level, every picture saved, and the world left empty."""
import os
import shutil
import tempfile
import unittest

import cv2

import helpers as H
from helpers import C, carla
from engine import weather_preview as WP


class WeatherPreviewTest(unittest.TestCase):
    def setUp(self):
        H.configure()
        self.out = tempfile.mkdtemp(prefix="weather_")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_sheet_and_pictures(self):
        with H.quiet():
            code = WP.run_weather_preview(self.out)
        self.assertEqual(code, C.EXIT_DONE)
        folder = os.path.join(self.out, "weather_preview")
        rows = len(WP.ROWS) + 1                                     # + camera brightness
        sheet = cv2.imread(os.path.join(folder, "sheet.jpg"))
        self.assertEqual(sheet.shape[:2], (WP.HEADER_H + rows * WP.CELL_H, WP.LABEL_W + len(WP.LEVELS) * WP.CELL_W))
        pictures = [f for f in os.listdir(folder) if f != "sheet.jpg"]
        self.assertEqual(len(pictures), len(WP.ROWS) * len(WP.LEVELS) + len(WP.EXPOSURES))
        self.assertEqual(carla.Client().get_world().actors, {}, "actors left in the world")


if __name__ == "__main__":
    unittest.main()
