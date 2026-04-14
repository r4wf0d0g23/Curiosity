#!/usr/bin/env python3
"""
Curiosity — Master process.
Spawns all 6 daemons, monitors health, restarts on failure.
Never stops. Never asks for directions.
"""
import subprocess
import time
import signal
import sys
import logging
import os
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = Path.home() / "curiosity" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "curiosity_master.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MASTER] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("curiosity.master")

# ── Daemon definitions ────────────────────────────────────────────────────────
DAEMONS = [
    ("assessor",   ["python3", "-m", "src.assessor.assessor"]),
    ("formulator", ["python3", "-m", "src.formulator.formulator"]),
    ("solver",     ["python3", "-m", "src.solver.solver"]),
    ("verifier",   ["python3", "-m", "src.verifier.verifier"]),
    ("memorizer",  ["python3", "-m", "src.memorizer.memorizer"]),
    ("trainer",    ["python3", "-m", "src.trainer.trainer"]),
]

MAX_BACKOFF = 60   # seconds
BASE_BACKOFF = 2   # seconds

# ── Global state ──────────────────────────────────────────────────────────────
running: dict[str, subprocess.Popen] = {}
backoffs: dict[str, float] = {name: BASE_BACKOFF for name, _ in DAEMONS}
shutting_down = False


def start_daemon(name: str, cmd: list[str]) -> subprocess.Popen | None:
    """Launch a daemon subprocess. Returns the Popen object or None on error."""
    try:
        log.info(f"Starting daemon: {name}  cmd={' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=Path.home() / "curiosity_code",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Daemon {name} started (pid={proc.pid})")
        return proc
    except Exception as exc:
        log.error(f"Failed to start daemon {name}: {exc}")
        return None


def shutdown_all(signum=None, frame=None):
    """Gracefully shut down all child daemons on SIGTERM / SIGINT."""
    global shutting_down
    shutting_down = True
    log.info("Shutdown signal received — stopping all daemons …")
    for name, proc in list(running.items()):
        try:
            log.info(f"Terminating {name} (pid={proc.pid})")
            proc.terminate()
        except Exception as exc:
            log.warning(f"Could not terminate {name}: {exc}")

    # Give processes a few seconds to exit cleanly
    deadline = time.time() + 10
    for name, proc in list(running.items()):
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
            log.info(f"Daemon {name} exited cleanly.")
        except subprocess.TimeoutExpired:
            log.warning(f"Daemon {name} did not exit in time — killing.")
            proc.kill()

    log.info("All daemons stopped. Curiosity master exiting.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, shutdown_all)
    signal.signal(signal.SIGINT, shutdown_all)

    log.info("═" * 60)
    log.info("Curiosity master starting …")
    log.info("═" * 60)

    # Initial launch of all daemons
    for name, cmd in DAEMONS:
        proc = start_daemon(name, cmd)
        if proc is not None:
            running[name] = proc
        else:
            log.error(f"Skipping daemon {name} due to launch failure.")

    # Monitor loop
    while not shutting_down:
        time.sleep(5)

        for name, cmd in DAEMONS:
            if shutting_down:
                break

            proc = running.get(name)
            if proc is None:
                # Never started (import error etc.) — try again after backoff
                backoff = backoffs[name]
                log.warning(f"Daemon {name} never started — retrying in {backoff:.0f}s …")
                time.sleep(backoff)
                new_proc = start_daemon(name, cmd)
                if new_proc is not None:
                    running[name] = new_proc
                    backoffs[name] = BASE_BACKOFF          # reset on success
                else:
                    backoffs[name] = min(backoff * 2, MAX_BACKOFF)
                continue

            ret = proc.poll()
            if ret is not None:
                # Process has exited
                backoff = backoffs[name]
                log.warning(
                    f"Daemon {name} (pid={proc.pid}) exited with code {ret}. "
                    f"Restarting in {backoff:.0f}s …"
                )
                time.sleep(backoff)

                new_proc = start_daemon(name, cmd)
                if new_proc is not None:
                    running[name] = new_proc
                    backoffs[name] = BASE_BACKOFF          # reset on success
                else:
                    del running[name]                      # will retry next cycle
                    backoffs[name] = min(backoff * 2, MAX_BACKOFF)


if __name__ == "__main__":
    main()
