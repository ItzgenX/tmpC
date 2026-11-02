"""python main.py --self-test itself, run against the fake simulator: every layout passes
and its frames are saved for viewing."""
import os
import shutil
import tempfile
import unittest

import helpers as H
from helpers import C
from engine.selftest import LAYOUTS, run_self_test


class SelfTestCommand(unittest.TestCase):
    def setUp(self):
        H.configure()
        self.out = tempfile.mkdtemp(prefix="selftest_")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_every_layout_passes_and_is_saved(self):
        with H.quiet():
            code = run_self_test(self.out, 1)
        self.assertEqual(code, C.EXIT_DONE)
        saved = os.listdir(os.path.join(self.out, "self_test"))
        self.assertEqual(len(saved), len(LAYOUTS) * 5, "5 frames per layout")


if __name__ == "__main__":
    unittest.main()
