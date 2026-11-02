"""File names: flat folders, <weather>_<scene>_<number>, numbers counting up from 00001
per name with no gaps, the frames of one occluded walk as numbers in a row, and the map
code in front when several maps are used."""
import collections
import re
import unittest

import helpers as H
from helpers import C
from engine.run import build_jobs, map_codes

KINDS = {"crosswalk", "road", "junction", "turn"}
NAME = re.compile(r"^[a-z0-9-]+_[a-z0-9-]+_\d{5}$")


def all_images(jobs):
    return [name for job in jobs for name in job["images"]]


class NameTest(unittest.TestCase):
    def setUp(self):
        H.configure()
        self.jobs = build_jobs("FakeCity", 0, 1, KINDS)

    def test_names_are_flat_short_and_unique(self):
        names = all_images(self.jobs)
        self.assertEqual(len(names), len(set(names)), "two images would get the same name")
        for n in names:
            self.assertRegex(n, NAME)

    def test_numbers_count_up_from_1_without_gaps(self):
        numbers = collections.defaultdict(list)
        for n in all_images(self.jobs):
            base, num = n.rsplit("_", 1)
            numbers[base].append(int(num))
        for base, nums in numbers.items():
            self.assertEqual(sorted(nums), list(range(1, len(nums) + 1)), base)

    def test_all_occluded_scenes_share_one_name(self):
        rain = [j for j in self.jobs if j["weather"] == "rain" and j["scene"] in C.OCCLUDED_SCENES]
        self.assertEqual({j["scene"] for j in rain}, set(C.OCCLUDED_SCENES))
        self.assertTrue(all(n.startswith("rain_occluded_") for n in all_images(rain)))

    def test_a_walk_is_numbers_in_a_row(self):
        for job in self.jobs:
            nums = [int(n.rsplit("_", 1)[1]) for n in job["images"]]
            self.assertEqual(nums, list(range(nums[0], nums[0] + job["frames"])))
            self.assertEqual(job["name"], job["images"][0])

    def test_map_code_first_with_several_maps(self):
        codes = map_codes(["Town03_Opt", "Town10HD_Opt", "MyCity"])
        self.assertEqual(codes, {"Town03_Opt": "t03", "Town10HD_Opt": "t10hd", "MyCity": "mycity"})
        self.assertEqual(map_codes(["Town10HD", "Town10HD_Opt"]),
                         {"Town10HD": "town10hd", "Town10HD_Opt": "town10hd-opt"})
        jobs = build_jobs("Town03_Opt", 0, 2, KINDS, codes["Town03_Opt"])
        self.assertTrue(all(n.startswith("t03_") for n in all_images(jobs)))


if __name__ == "__main__":
    unittest.main()
