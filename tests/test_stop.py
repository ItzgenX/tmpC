"""Clean stop: Ctrl+C / Ctrl+Z / kill / closed terminal stop the run, and the CARLA world
is emptied and set back to normal (async) mode even if Ctrl+C is pressed again during
that cleanup."""
import os
import shutil
import signal
import tempfile
import unittest

import helpers as H
from helpers import carla
from engine.run import Generator
from engine.stop import EXTRA_STOP_SIGNALS, install_stop_handlers, protected


class StopTest(unittest.TestCase):
    def setUp(self):
        H.configure()
        self.old = {s: signal.getsignal(s) for s in [signal.SIGINT] + EXTRA_STOP_SIGNALS}

    def tearDown(self):
        for s, h in self.old.items():
            signal.signal(s, h)

    def test_stop_keys_become_a_clean_stop(self):
        install_stop_handlers()
        for sig in EXTRA_STOP_SIGNALS:
            with self.assertRaises(KeyboardInterrupt, msg=signal.Signals(sig).name):
                signal.raise_signal(sig)

    def test_cleanup_cannot_be_interrupted(self):
        with protected():
            signal.raise_signal(signal.SIGINT)          # ignored
        with self.assertRaises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)          # works again afterwards

    def test_ctrl_c_mid_scene_empties_the_world(self):
        out = tempfile.mkdtemp(prefix="stop_")
        orig, count = carla.World.tick, {"n": 0}

        def tick(world, seconds=10.0):
            count["n"] += 1
            if count["n"] == 400:                       # in the middle of the run: Ctrl+C
                raise KeyboardInterrupt
            if count["n"] > 400:                        # and again on every tick of the cleanup
                signal.raise_signal(signal.SIGINT)
            return orig(world, seconds)

        carla.World.tick = tick
        try:
            with self.assertRaises(KeyboardInterrupt), H.quiet():
                Generator(out).run()
        finally:
            carla.World.tick = orig
            shutil.rmtree(out, ignore_errors=True)
        world = carla.Client().get_world()
        self.assertEqual([a.type_id for a in world.actors.values()], [], "actors left in the world")
        self.assertFalse(world.settings.synchronous_mode, "world left in sync mode")


if __name__ == "__main__":
    unittest.main()
