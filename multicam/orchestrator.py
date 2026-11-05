"""
VIGIA 2.0 capture orchestrator.

Runs on uvcam (RPi5). The fast instruments run as their own services, each on
round UTC multiples of its own interval:

- ``capture-uv-monitor.service`` (uvcam): UV cameras + spectrometer
  (``[uv_sync] monitor_interval_s``)
- ``capture-ir-video.service`` (multicam): Optris IR
  (``[infrared] monitor_window_s``, one frame per window with
  ``monitor_keep = "baseline"``)

The orchestrator schedules the slower captures and watches the services:

- each task in ``[orchestrator] schedule`` (e.g. Canon every 30 s,
  environmental every 60 s) fires on round UTC multiples of its interval, in
  its own thread; a task still running at its next tick skips that tick
  (logged), it is never queued;
- a task in ``night_schedule`` (e.g. Canon every 120 s) uses that slower
  interval while the sun is below ``night_sun_elevation_deg`` at the site
  (``[uv_sync] latitude/longitude``); it must be a multiple of the day one;
- every command has a hard timeout; remote commands are also wrapped in
  ``timeout`` on multicam, so a hung capture there cannot outlive its slot;
- every ``supervise_interval_s`` it checks the services in ``supervise`` with
  ``systemctl is-active`` and, with ``supervise_restart``, starts any that are
  not running (``sudo -n``: needs passwordless sudo for systemctl);
- at the same time it warns about any instrument in ``freshness_max_age_min``
  with no new file for that long, whatever the cause: a service can be
  ``active`` and still be producing nothing. This only logs;
- after ``canon_usb_reset_after`` failed Canon shots in a row it resets the
  Canon's USB port on multicam, at most once per ``canon_usb_reset_every_min``.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import math
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path

from multicam.utilities import read_config
from multicam.utilities.sun import solar_elevation_deg

logger = logging.getLogger("orchestrator")

DEFAULT_SCHEDULE = {"dslr": 30, "environmental": 60}
DEFAULT_NIGHT_SUN_ELEVATION_DEG = 3.0
DEFAULT_SUPERVISE = ["capture-uv-monitor.service", "multicam:capture-ir-video.service"]
DEFAULT_FRESHNESS_MIN = {
    "ultraviolet": 2,
    "spectrometer": 2,
    "infrared": 2,
    "dslr": 5,
    "environmental": 5,
}
# These stop at night, when the UV monitor closes its devices.
DAYLIGHT_INSTRUMENTS = {"ultraviolet", "spectrometer"}
DEFAULT_MIN_SUN_ELEVATION_DEG = 5.0
# Margin above that elevation before daylight instruments are expected again.
DAYLIGHT_MARGIN_DEG = 1.0
FRESHNESS_GRACE_S = 300  # no warnings while everything is starting up
FRESHNESS_REMIND_S = 600
DEFAULT_CANON_USB_RESET_AFTER = 6  # failed shots in a row; 0 = never reset
DEFAULT_CANON_USB_RESET_EVERY_MIN = 30
DEFAULT_CANON_USB_PORT = "1-4"  # used when the Canon has dropped off the bus
CANON_USB_RESET_COMMAND = "/usr/local/sbin/vigia-canon-usb-reset"
CANON_USB_RESET_TIMEOUT_S = 30
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_path: str) -> None:
    """Log to a rotating file (10 MB × 5) and to stderr (journald)."""

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime  # log times in UTC, like the file names

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _find_multicamctl() -> str:
    """Return the path to the local multicamctl binary."""

    found = shutil.which("multicamctl")
    if found:
        return found
    # Fallback to the expected venv location on the RPi5
    return "/home/user/.venv/bin/multicamctl"


def _run_subprocess(
    cmd: list[str], label: str, timeout_s: float, dry_run: bool
) -> bool:
    """Run one command with a hard timeout; log the outcome; return success."""

    if dry_run:
        logger.info("[DRY RUN] %s: %s", label, " ".join(cmd))
        return True

    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        logger.error("%s timed out after %.0f s", label, timeout_s)
        return False
    except Exception as e:
        logger.error("%s error: %s", label, e)
        return False

    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()[-500:]
        logger.error(
            "%s failed (rc=%d, %.1f s): %s", label, proc.returncode, elapsed, tail
        )
        return False
    logger.info("%s ok (%.1f s)", label, elapsed)
    return True


class Remote:
    """Build commands for multicam over ssh."""

    def __init__(self, user: str, host: str) -> None:
        self.target = f"{user}@{host}"

    def command(
        self, remote_cmd: list[str], timeout_s: float | None = None
    ) -> list[str]:
        """``remote_cmd`` over ssh; with ``timeout_s``, killed on multicam too."""

        if timeout_s is not None:
            # SIGTERM first (lets the capture clean up), SIGKILL 5 s later.
            remote_cmd = ["timeout", "-k", "5", f"{timeout_s:.0f}"] + remote_cmd
        return ["ssh", *SSH_OPTIONS, self.target, *remote_cmd]


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def build_tasks(args, orch_config: dict) -> dict[str, tuple[float, float, list[str]]]:
    """
    Scheduled tasks as ``{name: (interval_s, timeout_s, command)}``.

    A task's timeout is just under its interval (at most 5 min), so a hung
    capture is stopped before it would have to skip a second tick.
    """

    multicamctl = _find_multicamctl()
    remote_bin = orch_config.get("remote_venv_bin", "/home/user/.venv/bin/multicamctl")
    remote = Remote(
        args.multicam_user or orch_config.get("multicam_user", "user"),
        args.multicam_host or orch_config.get("multicam_host", "192.168.30.150"),
    )

    schedule = dict(orch_config.get("schedule", DEFAULT_SCHEDULE))
    for name in args.skip or []:
        schedule.pop(name, None)

    tasks = {}
    for name, interval in schedule.items():
        interval = float(interval)
        timeout_s = min(300.0, max(10.0, interval - 2.0))
        if name == "dslr":
            # Remote timeout a little shorter than the local one, so multicam
            # stops the capture before ssh gives up on it.
            cmd = remote.command(
                # Closed loop: the previous Canon shot sets this one's exposure.
                [remote_bin, "capture", "dslr", "--meter-with", "previous"],
                timeout_s=timeout_s - 3,
            )
        elif name == "environmental":
            cmd = [multicamctl, "capture", "environmental"]
        else:
            raise ValueError(f"[orchestrator] schedule: unknown task {name!r}")
        tasks[name] = (interval, timeout_s, cmd)
    return tasks


def supervise_once(
    units: list[str], remote: Remote, restart: bool, dry_run: bool
) -> None:
    """Check each service; log problems; start it again if ``restart``."""

    for entry in units:
        where, _, unit = entry.rpartition(":")
        if where not in ("", "multicam"):
            logger.error("supervise: unknown host in %r (use 'multicam:<unit>')", entry)
            continue

        def wrap(cmd: list[str], where: str = where) -> list[str]:
            return remote.command(cmd) if where == "multicam" else cmd

        label = f"{where or 'uvcam'}:{unit}"
        if dry_run:
            logger.info("[DRY RUN] supervise %s", label)
            continue
        try:
            proc = subprocess.run(
                wrap(["systemctl", "is-active", unit]),
                capture_output=True,
                text=True,
                timeout=20,
            )
            state = (
                proc.stdout.strip() or proc.stderr.strip() or f"rc={proc.returncode}"
            )
        except Exception as e:
            state = f"check failed: {e}"

        if state == "active":
            continue
        # Only start a unit that is plainly down; "activating" is systemd's own
        # Restart= at work, and a failed check says nothing about the unit.
        if not restart or state not in ("inactive", "failed"):
            logger.warning("supervise: %s is %s", label, state)
            continue
        logger.warning("supervise: %s is %s — starting it", label, state)
        _run_subprocess(
            wrap(["sudo", "-n", "systemctl", "start", unit]),
            f"start {label}",
            timeout_s=60,
            dry_run=False,
        )


class CanonUsbReset:
    """Reset the Canon's USB port when its shots keep failing."""

    def __init__(self, command: list[str], after: int, every_s: float, dry_run: bool):
        self.command = command
        self.after = after
        self.every_s = every_s
        self.dry_run = dry_run
        self.failures = 0
        self.last_reset: float | None = None
        self.reset_since_good_shot = False

    def record(self, ok: bool) -> None:
        """Note one shot's outcome, and reset the port if it's time to."""

        if ok:
            if self.reset_since_good_shot:
                logger.info(
                    "dslr: back after USB reset (%d failed shots)", self.failures
                )
            self.failures = 0
            self.reset_since_good_shot = False
            return

        self.failures += 1
        if self.after <= 0 or self.failures < self.after:
            return
        now = time.monotonic()
        if self.last_reset is not None and now - self.last_reset < self.every_s:
            return
        self.last_reset = now
        self.reset_since_good_shot = True
        logger.warning(
            "dslr: %d failed shots in a row — USB reset of its port", self.failures
        )
        _run_subprocess(
            self.command, "dslr USB reset", CANON_USB_RESET_TIMEOUT_S, self.dry_run
        )


def _freshness_script(data_dir: str, max_age_min: dict[str, int]) -> str:
    """
    Shell script printing each instrument with a recent, non-empty file on this
    node. Files move between folders and nodes, so it looks in ``receive/`` and
    today's and yesterday's ``archive/`` folders (keeping the check cheap).
    """

    lines = [
        f"cd {shlex.quote(data_dir)} || exit 3",
        "d0=$(date -u +%j); d1=$(date -u -d yesterday +%j)",
    ]
    for name, minutes in max_age_min.items():
        i = shlex.quote(name)
        lines.append(
            f'[ -n "$(find {i}/receive {i}/archive/*/*/*/$d0 {i}/archive/*/*/*/$d1 '
            f"-maxdepth 1 -type f -mmin -{int(minutes)} ! -name '.*' -size +0c "
            f'-print -quit 2>/dev/null)" ] && echo {i}'
        )
    lines.append("exit 0")
    return "\n".join(lines)


def site_from_config(uv_sync: dict) -> tuple[float, float] | None:
    """Site (latitude, longitude) from ``[uv_sync]``, if it has one."""

    if "latitude" in uv_sync and "longitude" in uv_sync:
        return float(uv_sync["latitude"]), float(uv_sync["longitude"])
    return None


def daylight_expected(
    name: str,
    now: dt,
    site: tuple[float, float] | None,
    min_sun_elevation_deg: float,
) -> bool:
    """Should this instrument be producing files now?"""

    if name not in DAYLIGHT_INSTRUMENTS or site is None:
        return True
    elevation = solar_elevation_deg(now, *site)
    return elevation >= min_sun_elevation_deg + DAYLIGHT_MARGIN_DEG


class FreshnessCheck:
    """Warn about instruments that have stopped producing files."""

    def __init__(
        self,
        data_dir: str,
        max_age_min: dict[str, int],
        remote: Remote,
        site: tuple[float, float] | None,
        min_sun_elevation_deg: float,
    ) -> None:
        self.max_age_min = max_age_min
        self.script = _freshness_script(data_dir, max_age_min)
        self.remote = remote
        self.site = site
        self.min_sun_elevation_deg = min_sun_elevation_deg
        self.started = time.monotonic()
        self.warned: dict[str, float] = {}  # instrument: when last warned

    def _fresh_on(self, cmd: list[str], where: str) -> set[str] | None:
        """Instruments with recent files on one node; None if it can't be checked."""

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("freshness: could not check %s: %s", where, e)
            return None
        if proc.returncode != 0:
            logger.warning(
                "freshness: could not check %s (rc=%d)", where, proc.returncode
            )
            return None
        return set(proc.stdout.split())

    def check(self, dry_run: bool) -> None:
        """Log instruments that went stale, are still stale, or have recovered."""

        if dry_run:
            logger.info("[DRY RUN] freshness: %s", self.max_age_min)
            return
        now = time.monotonic()
        if now - self.started < FRESHNESS_GRACE_S:
            return

        local = self._fresh_on(["sh", "-c", self.script], "uvcam")
        remote = self._fresh_on(self.remote.command([self.script]), "multicam")
        if local is None or remote is None:
            return  # can't tell; a file may be on the node that didn't answer
        fresh = local | remote

        for name, minutes in self.max_age_min.items():
            expected = daylight_expected(
                name, dt.now(UTC), self.site, self.min_sun_elevation_deg
            )
            if name in fresh or not expected:
                if self.warned.pop(name, None) is not None:
                    logger.info("freshness: %s: back to normal", name)
            elif now - self.warned.get(name, -FRESHNESS_REMIND_S) >= FRESHNESS_REMIND_S:
                logger.warning(
                    "freshness: %s: no new file for over %d min", name, minutes
                )
                self.warned[name] = now


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _next_tick(interval_s: float, now: float) -> float:
    """Next round UTC multiple of ``interval_s`` strictly after ``now``."""

    return (math.floor(now / interval_s) + 1) * interval_s


def _night_schedule(
    orch_config: dict,
    intervals: dict[str, float],
    site: tuple[float, float] | None,
) -> dict[str, float]:
    """``{task: night interval}`` for scheduled tasks, validated; {} if unusable."""

    night = {
        name: float(iv)
        for name, iv in orch_config.get("night_schedule", {}).items()
        if name in intervals
    }
    if not night:
        return {}
    if site is None:
        logger.warning(
            "night_schedule ignored: no site coordinates ([uv_sync] latitude/longitude)"
        )
        return {}
    for name, iv in night.items():
        if iv <= 0 or round(iv) % round(intervals[name]) != 0:
            raise ValueError(
                f"[orchestrator] night_schedule: {name} = {iv:g} s must be a "
                f"multiple of its day interval ({intervals[name]:g} s)"
            )
    return night


def run_scheduler(
    args,
    orch_config: dict,
    site: tuple[float, float] | None = None,
    data_dir: str = "/home/user/data",
    min_sun_elevation_deg: float = DEFAULT_MIN_SUN_ELEVATION_DEG,
) -> None:
    """
    Fire each task on round UTC multiples of its interval until SIGTERM/SIGINT.

    A task that is still running at its next tick skips it. Supervision is one
    more task on the same clock. Tasks in ``night_schedule`` skip the ticks that
    are not on their night interval while the sun is below the threshold.
    """

    tasks = build_tasks(args, orch_config)
    remote = Remote(
        args.multicam_user or orch_config.get("multicam_user", "user"),
        args.multicam_host or orch_config.get("multicam_host", "192.168.30.150"),
    )
    supervise = (
        []
        if args.no_supervise
        else list(orch_config.get("supervise", DEFAULT_SUPERVISE))
    )
    supervise_restart = bool(orch_config.get("supervise_restart", True))
    supervise_s = float(orch_config.get("supervise_interval_s", 60))

    max_age_min = dict(orch_config.get("freshness_max_age_min", DEFAULT_FRESHNESS_MIN))
    for name in args.skip or []:
        max_age_min.pop(name, None)
    freshness = (
        FreshnessCheck(data_dir, max_age_min, remote, site, min_sun_elevation_deg)
        if supervise and max_age_min
        else None
    )

    usb_reset = CanonUsbReset(
        remote.command(
            [
                "sudo",
                "-n",
                CANON_USB_RESET_COMMAND,
                str(orch_config.get("canon_usb_port", DEFAULT_CANON_USB_PORT)),
            ]
        ),
        int(orch_config.get("canon_usb_reset_after", DEFAULT_CANON_USB_RESET_AFTER)),
        60.0
        * float(
            orch_config.get(
                "canon_usb_reset_every_min", DEFAULT_CANON_USB_RESET_EVERY_MIN
            )
        ),
        args.dry_run,
    )

    def run_task(name: str) -> None:
        if name == "supervise":
            supervise_once(supervise, remote, supervise_restart, args.dry_run)
            if freshness is not None:
                freshness.check(args.dry_run)
        else:
            _, timeout_s, cmd = tasks[name]
            ok = _run_subprocess(cmd, name, timeout_s, args.dry_run)
            if name == "dslr":
                usb_reset.record(ok)

    intervals = {name: iv for name, (iv, _, _) in tasks.items()}
    night = _night_schedule(orch_config, intervals, site)
    night_elevation = float(
        orch_config.get("night_sun_elevation_deg", DEFAULT_NIGHT_SUN_ELEVATION_DEG)
    )
    is_night: dict[str, bool] = {}
    if supervise:
        intervals["supervise"] = supervise_s

    logger.info(
        "Orchestrator starting — %s; supervising %s (restart=%s)",
        ", ".join(
            f"{n} every {iv:g} s" for n, iv in intervals.items() if n != "supervise"
        )
        or "no capture tasks",
        ", ".join(supervise) or "nothing",
        supervise_restart,
    )
    if night:
        logger.info(
            "Night (sun < %g°): %s",
            night_elevation,
            ", ".join(f"{n} every {iv:g} s" for n, iv in night.items()),
        )

    stop = threading.Event()

    def _handle_stop(signum, _frame) -> None:
        logger.info("Stop signal received — finishing running tasks")
        stop.set()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    now = time.time()
    next_tick = {name: _next_tick(iv, now) for name, iv in intervals.items()}
    running: dict[str, threading.Thread] = {}

    while not stop.is_set() and next_tick:
        name = min(next_tick, key=next_tick.get)
        if stop.wait(max(0.0, next_tick[name] - time.time())):
            break
        tick = next_tick[name]
        skip_for_night = False
        if name in night:
            elevation = solar_elevation_deg(dt.fromtimestamp(tick, UTC), *site)
            now_night = elevation < night_elevation
            if is_night.get(name) != now_night:
                logger.info(
                    "%s: sun at %.1f° — every %g s",
                    name,
                    elevation,
                    night[name] if now_night else intervals[name],
                )
                is_night[name] = now_night
            skip_for_night = now_night and round(tick) % round(night[name]) != 0
        thread = running.get(name)
        if skip_for_night:
            pass
        elif thread is not None and thread.is_alive():
            logger.warning(
                "%s still running at its %s tick — skipped",
                name,
                dt.fromtimestamp(tick, UTC).strftime("%H:%M:%S"),
            )
        else:
            running[name] = threading.Thread(
                target=run_task, args=(name,), name=name, daemon=True
            )
            running[name].start()
        next_tick[name] = _next_tick(intervals[name], max(time.time(), tick))

    for thread in running.values():
        thread.join(timeout=60)
    logger.info("Orchestrator stopped")


def run_once(args, orch_config: dict) -> bool:
    """Run every scheduled task once, in parallel, then supervise once."""

    tasks = build_tasks(args, orch_config)
    results: dict[str, bool] = {}
    threads = [
        threading.Thread(
            target=lambda n=name, t=timeout_s, c=cmd: results.__setitem__(
                n, _run_subprocess(c, n, t, args.dry_run)
            ),
            name=name,
        )
        for name, (_, timeout_s, cmd) in tasks.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not args.no_supervise:
        remote = Remote(
            args.multicam_user or orch_config.get("multicam_user", "user"),
            args.multicam_host or orch_config.get("multicam_host", "192.168.30.150"),
        )
        supervise_once(
            list(orch_config.get("supervise", DEFAULT_SUPERVISE)),
            remote,
            restart=False,
            dry_run=args.dry_run,
        )
    logger.info("Once: %d/%d tasks succeeded", sum(results.values()), len(results))
    return all(results.values())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the orchestrator."""

    parser = argparse.ArgumentParser(
        description="VIGIA 2.0 capture orchestrator — schedules the slow captures "
        "(Canon, environmental) on the UTC clock and supervises the UV and IR "
        "monitor services.",
    )
    parser.add_argument(
        "--multicam-host",
        type=str,
        default=None,
        help="IP of multicam SBC (default: from config or 192.168.30.150)",
    )
    parser.add_argument(
        "--multicam-user",
        type=str,
        default=None,
        help="SSH user on multicam (default: from config or user)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(Path.home() / ".config/multicam_config.toml"),
        help="Path to multicam_config.toml",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="TASK",
        help="Leave out a scheduled task (dslr, environmental); repeatable",
    )
    parser.add_argument(
        "--no-supervise", action="store_true", help="Don't check the services"
    )
    parser.add_argument(
        "--once", action="store_true", help="Run each task once, then exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Log commands without executing"
    )

    args = parser.parse_args()

    config = read_config(Path(args.config))
    orch_config = config.get("orchestrator", {})
    uv_sync = config.get("uv_sync", {})
    site = site_from_config(uv_sync)

    _setup_logging(orch_config.get("log_path", "/home/user/logs/orchestrator.log"))

    if args.once:
        sys.exit(0 if run_once(args, orch_config) else 1)
    run_scheduler(
        args,
        orch_config,
        site,
        config["metadata"]["data_archive"],
        float(uv_sync.get("min_sun_elevation_deg", DEFAULT_MIN_SUN_ELEVATION_DEG)),
    )


if __name__ == "__main__":
    main()
