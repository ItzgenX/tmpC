"""Auto-restart: keeps the generator running until the dataset is complete.

The generator runs in a child process. If it dies - including CARLA native
crashes / core dumps that no Python code can catch - it is started again and
continues from jobs_done.txt. If CARLA_EXECUTABLE is set, the CARLA server is
also started / restarted when it is down or hung. A stop key is passed on to
the generator, which empties the CARLA world itself; a final --cleanup pass
then makes sure nothing of ours is left.
"""
import os
import signal
import socket
import subprocess
import sys
import time

import advanced as C
from advanced import EXIT_DONE, EXIT_FATAL, EXIT_NO_SERVER, EXIT_RECYCLE, EXIT_STOPPED
from engine.stop import protected


# main.py, which the supervisor starts as the generator / cleanup process
MAIN_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


def server_reachable():
    try:
        with socket.create_connection((C.CARLA_HOST, C.CARLA_PORT), timeout=3):
            return True
    except OSError:
        return False


class ServerKeeper:
    def __init__(self):
        self.proc = None

    def restart(self):
        if not C.CARLA_EXECUTABLE:
            return False
        if self.proc is not None and self.proc.poll() is None:
            print("  stopping CARLA server")
            self.proc.kill()
            self.proc.wait()
        print(f"  starting CARLA: {C.CARLA_EXECUTABLE}")
        # own session: a Ctrl+C in this terminal must not kill the server mid-cleanup
        self.proc = subprocess.Popen([C.CARLA_EXECUTABLE] + list(C.CARLA_ARGS), start_new_session=True)
        time.sleep(C.CARLA_STARTUP_SECONDS)
        return True

    def wait(self):
        if server_reachable():
            return True
        if self.restart() and server_reachable():
            return True
        print(f"  CARLA not reachable at {C.CARLA_HOST}:{C.CARLA_PORT} - start the server, waiting...")
        end = time.time() + C.SERVER_WAIT_SECONDS
        while time.time() < end:
            time.sleep(10)
            if server_reachable():
                time.sleep(15)      # let it finish loading
                return True
        return False


def count_done(out):
    path = os.path.join(out, "jobs_done.txt")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_child(args):
    """Run main.py with `args`; on a stop key let it clean up instead of killing it.
    Returns its exit code, or EXIT_STOPPED after a stop key."""
    cmd = [sys.executable, MAIN_SCRIPT] + args
    # own session: the terminal's Ctrl+C / Ctrl+Z / hang-up reach only this supervisor,
    # which passes on exactly one SIGINT. a second Ctrl+C can then never land in the
    # moment before the generator's cleanup has started protecting itself.
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        while proc.poll() is None:
            time.sleep(0.2)
        return proc.returncode
    except KeyboardInterrupt:
        with protected():
            print(f"\nstopping: emptying the CARLA world (up to {C.STOP_WAIT_SECONDS} s)...")
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=C.STOP_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        return EXIT_STOPPED


def final_cleanup():
    """Fresh connection: remove anything left and switch the server to async mode."""
    if not server_reachable():
        return
    with protected():
        try:
            subprocess.run([sys.executable, MAIN_SCRIPT, "--cleanup"],
                           timeout=C.STOP_WAIT_SECONDS, start_new_session=True)
        except subprocess.TimeoutExpired:
            print("  cleanup timed out; run: python main.py --cleanup")


def supervise(out):
    try:
        return _supervise(out)
    except KeyboardInterrupt:       # stop key while no generator was running (e.g. waiting for the server)
        print("\nstopped")
        return EXIT_STOPPED


def _supervise(out):
    keeper = ServerKeeper()
    crashes = no_server = 0
    while True:
        if not keeper.wait():
            print("gave up waiting for the CARLA server")
            return 1
        before = count_done(out)
        code = run_child(["--worker", "--out", out])
        after = count_done(out)
        if code == EXIT_STOPPED:
            final_cleanup()
            print(f"stopped: {after} jobs done so far, CARLA world emptied. run python main.py to continue.")
            return EXIT_STOPPED
        if code == EXIT_DONE:
            print(f"\ndone: {after} jobs in {out}")
            return 0
        if code == EXIT_FATAL:
            print("stopped on a configuration error (see above)")
            return 2
        if code == EXIT_RECYCLE:
            crashes = no_server = 0
            print("\nworker recycled, continuing")
            continue
        if code == EXIT_NO_SERVER:
            no_server += 1
            print(f"\nserver connection lost ({no_server})")
            if no_server >= 2 and server_reachable():
                keeper.restart()    # port open but not answering: hung server
            continue
        crashes = 0 if after > before else crashes + 1
        print(f"\nworker died (exit code {code}), {after - before} jobs finished this run, "
              f"crash streak {crashes}/{C.MAX_CRASHES_WITHOUT_PROGRESS}")
        if crashes >= C.MAX_CRASHES_WITHOUT_PROGRESS:
            print("no progress across repeated crashes - check run_log.txt in the output folder")
            from engine.run import write_summary
            write_summary(out)
            return 1
        if not server_reachable():
            keeper.restart()
        time.sleep(5)
