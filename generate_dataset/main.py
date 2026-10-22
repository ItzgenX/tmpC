"""Pedestrian-crossing dataset generator. The only script you run.

    python main.py                          generate / resume the dataset (restarts itself after crashes)
    python main.py --plan-only 5            plan 5 jobs per weather + scene without rendering (sanity check)
    python main.py --check                  check the CARLA server against config.py
    python main.py --delete-flagged         list images that failed a check (deletes nothing)
    python main.py --delete-flagged --yes   delete them (add --reason <text> for one kind only)
    python main.py --summary                rebuild manifest.csv, summary.txt and split lists
    python main.py --cleanup                empty the CARLA world and switch it back to normal (async) mode

Stop any time with Ctrl+C, Ctrl+Z, kill <pid> or by closing the terminal: the
CARLA world is emptied and switched back to normal mode before the program
exits. Run python main.py again to continue.

Settings: config.py (everyday) and advanced.py (fine tuning). The code that
does the work is in engine/ - see CODE_GUIDE.md for what each file does.
"""
import argparse
import sys

import advanced as C
from engine.stop import install_stop_handlers


def run_worker(args):
    """The real generator (started by the supervisor, or directly for --plan-only)."""
    from engine.run import Generator, ConnectionLost, is_connection_error
    try:
        return Generator(args.out).run(plan_only=args.plan_only)
    except KeyboardInterrupt:
        print("\nstopped. the CARLA world has been emptied; run python main.py again to continue.")
        return C.EXIT_STOPPED
    except ConnectionLost as exc:
        print(f"  lost connection to CARLA: {exc}")
        return C.EXIT_NO_SERVER
    except RuntimeError as exc:
        if is_connection_error(exc):
            print(f"  lost connection to CARLA: {exc}")
            return C.EXIT_NO_SERVER
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=C.OUTPUT_DIR, help="dataset folder (resumes if it exists)")
    ap.add_argument("--plan-only", type=int, default=0, metavar="N", help="plan N jobs per weather/scene, no rendering")
    ap.add_argument("--check", action="store_true", help="check the CARLA server against config.py")
    ap.add_argument("--delete-flagged", action="store_true", help="list (or with --yes delete) flagged images")
    ap.add_argument("--reason", default="", help="with --delete-flagged: only reasons containing this text")
    ap.add_argument("--yes", action="store_true", help="with --delete-flagged: actually delete")
    ap.add_argument("--summary", action="store_true", help="only rebuild manifest / summary / splits")
    ap.add_argument("--cleanup", action="store_true", help="only empty the CARLA world and set async mode")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    install_stop_handlers()

    if args.check:
        from engine.tools import check_server
        return check_server()
    if args.delete_flagged:
        from engine.tools import delete_flagged
        return delete_flagged(args.out, args.reason, args.yes)
    if args.summary:
        from engine.run import write_summary
        write_summary(args.out)
        return 0
    if args.cleanup:
        from engine.tools import cleanup_only
        return cleanup_only()
    if args.worker or args.plan_only:
        return run_worker(args)
    from engine.supervisor import supervise
    return supervise(args.out)


if __name__ == "__main__":
    sys.exit(main())
