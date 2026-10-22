"""Clean stop: Ctrl+C / Ctrl+Z / kill empty the CARLA world before the program exits.

Ctrl+C (SIGINT), Ctrl+Z (SIGTSTP), kill (SIGTERM), closed terminal (SIGHUP) all
stop cleanly. Ctrl+Z would normally only *pause* the program and leave CARLA
frozen with our actors in it, so it is turned into a stop as well.
"""
import contextlib
import signal


EXTRA_STOP_SIGNALS = [signal.SIGTSTP, signal.SIGTERM, signal.SIGHUP]     # besides SIGINT


def _stop(signum, frame):
    raise KeyboardInterrupt


def install_stop_handlers():
    """Make Ctrl+Z / kill / a closed terminal behave like Ctrl+C."""
    for sig in EXTRA_STOP_SIGNALS:
        signal.signal(sig, _stop)


@contextlib.contextmanager
def protected():
    """Ignore every stop key until the block has finished, so cleanup always completes."""
    old = {}
    for sig in [signal.SIGINT] + EXTRA_STOP_SIGNALS:
        try:
            old[sig] = signal.signal(sig, signal.SIG_IGN)
        except ValueError:      # not the main thread: nothing to protect from
            pass
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def release_server(client):
    """Remove every vehicle, walker and sensor and switch the server back to async mode,
    so it does not sit frozen waiting for ticks that will never come."""
    from engine.carla_io import cleanup_world
    with protected():
        world = client.get_world()
        try:
            cleanup_world(client, world)
        finally:
            settings = world.get_settings()
            if settings.synchronous_mode:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
