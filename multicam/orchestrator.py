"""
VIGIA 2.0 capture orchestrator.

Runs a continuous acquisition loop on uvcam (RPi5), triggering one full
capture cycle every ``--interval`` seconds across all instruments:

- **Local (parallel):** UV-sync + environmental on uvcam
- **Remote (sequential):** IR → DSLR on multicam (SBC) via SSH

Local and remote run concurrently.  IR fires first on the remote so its
capture is closer in time to the UV cameras.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path

from multicam.utilities import read_config

logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_path: str) -> None:
    """Configure root logger to write to file and stderr."""

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
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
    default = "/home/user/.venv/bin/multicamctl"
    return default


def _run_subprocess(
    cmd: list[str],
    label: str,
    results: dict,
    dry_run: bool = False,
) -> None:
    """
    Run a subprocess command, capturing stdout/stderr.

    Results are stored in ``results[label]`` as a dict with keys
    ``success``, ``returncode``, ``stdout``, ``stderr``.
    """

    if dry_run:
        logger.info("[DRY RUN] %s: %s", label, " ".join(cmd))
        results[label] = {
            "success": True,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dry_run": True,
        }
        return

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute hard timeout per command
        )
        results[label] = {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            logger.error(
                "%s failed (rc=%d): %s", label, proc.returncode, proc.stderr[:500]
            )
        else:
            logger.info("%s completed successfully", label)
            if proc.stdout.strip():
                logger.debug("%s stdout: %s", label, proc.stdout[:500])
    except subprocess.TimeoutExpired:
        logger.error("%s timed out after 300 s", label)
        results[label] = {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }
    except Exception as e:
        logger.error("%s error: %s", label, e)
        results[label] = {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


# ---------------------------------------------------------------------------
# Cycle execution
# ---------------------------------------------------------------------------


def _run_group(tasks: list[tuple[list[str], str]], dry_run: bool) -> dict:
    """Run a list of (cmd, label) pairs in parallel and return results."""

    results: dict = {}
    threads = []
    for cmd, label in tasks:
        t = threading.Thread(
            target=_run_subprocess,
            args=(cmd, label, results, dry_run),
            name=label,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


def _run_remote_sequential(
    tasks: list[tuple[list[str], str]],
    dry_run: bool,
    results: dict,
) -> None:
    """Run remote tasks sequentially (IR then DSLR) to avoid /dev/video conflict."""

    for cmd, label in tasks:
        _run_subprocess(cmd, label, results, dry_run)


def run_cycle(args, orch_config: dict) -> dict:
    """
    Execute one full capture cycle.

    Local group (uv-sync + environmental, parallel internally) and remote
    group (IR → DSLR, sequential internally) run **concurrently** since they
    are on different hardware nodes.  IR runs first on the remote so its
    capture is closer in time to the UV cameras on uvcam.  IR and DSLR
    remain sequential to avoid the /dev/video2 conflict (Arducam vs Optris).
    """

    multicamctl = _find_multicamctl()
    host = args.multicam_host or orch_config.get("multicam_host", "192.168.30.150")
    user = args.multicam_user or orch_config.get("multicam_user", "user")
    remote_bin = orch_config.get("remote_venv_bin", "/home/user/.venv/bin/multicamctl")

    ssh_prefix = ["ssh", f"{user}@{host}"]

    # --- Build local task list (run in parallel internally) ---
    local_tasks: list[tuple[list[str], str]] = []

    if not args.no_uv:
        local_tasks.append(
            (
                [
                    multicamctl,
                    "capture",
                    "uv-sync",
                    "--check-saturation",
                    "--max-retries",
                    "3",
                ],
                "uv-sync",
            )
        )

    if not args.no_env:
        local_tasks.append(
            (
                [multicamctl, "capture", "environmental"],
                "environmental",
            )
        )

    # --- Build remote task list (sequential: IR first, then DSLR) ---
    # IR runs first so it captures closer in time to the UV cameras on uvcam.
    remote_tasks: list[tuple[list[str], str]] = []

    # The continuous capture-ir-video.service normally owns the Optris 24/7 (the
    # SDK is single-owner), so the per-cycle IR snapshot is disabled by default —
    # each monitor window's baseline frame is the IR snapshot. Enable
    # orchestrator.ir_snapshot only when running without the monitor service.
    ir_snapshot = orch_config.get("ir_snapshot", False)
    if ir_snapshot and not args.no_ir:
        remote_tasks.append(
            (
                ssh_prefix + [remote_bin, "capture", "infrared", "--check-saturation"],
                "infrared",
            )
        )

    if not args.no_dslr:
        remote_tasks.append(
            (
                ssh_prefix
                + [
                    remote_bin,
                    "capture",
                    "dslr",
                    "--meter-with",
                    "picam",
                ],
                "dslr",
            )
        )

    # --- Run local and remote concurrently ---
    local_results: dict = {}
    remote_results: dict = {}

    threads = []

    if local_tasks:
        logger.info("Local group: starting %d task(s)", len(local_tasks))
        local_thread = threading.Thread(
            target=lambda: local_results.update(_run_group(local_tasks, args.dry_run)),
            name="local-group",
        )
        threads.append(local_thread)

    if remote_tasks:
        logger.info("Remote group: starting %d task(s) (sequential)", len(remote_tasks))
        remote_thread = threading.Thread(
            target=_run_remote_sequential,
            args=(remote_tasks, args.dry_run, remote_results),
            name="remote-group",
        )
        threads.append(remote_thread)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {**local_results, **remote_results}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_loop(args, orch_config: dict) -> None:
    """Drift-corrected main loop."""

    interval = args.interval or int(orch_config.get("interval_s", 60))
    cycle = 0

    logger.info(
        "Orchestrator starting — interval=%d s, multicam=%s@%s",
        interval,
        args.multicam_user or orch_config.get("multicam_user", "user"),
        args.multicam_host or orch_config.get("multicam_host", "192.168.30.150"),
    )

    while True:
        t0 = time.monotonic()
        cycle += 1
        cycle_ts = dt.now(UTC)

        logger.info("=== Cycle %d at %s ===", cycle, cycle_ts.isoformat())

        try:
            results = run_cycle(args, orch_config)
        except Exception:
            logger.exception("Cycle %d failed with unhandled exception", cycle)
            results = {}

        elapsed = time.monotonic() - t0
        successes = sum(1 for r in results.values() if r.get("success"))
        total = len(results)

        logger.info(
            "Cycle %d complete: %d/%d succeeded in %.1f s",
            cycle,
            successes,
            total,
            elapsed,
        )

        sleep_s = max(0.0, interval - elapsed)
        if sleep_s == 0:
            logger.warning(
                "Cycle %d took %.1f s — exceeds interval (%d s), "
                "starting next immediately",
                cycle,
                elapsed,
                interval,
            )
        else:
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the orchestrator."""

    parser = argparse.ArgumentParser(
        description="VIGIA 2.0 capture orchestrator — "
        "coordinates multi-instrument capture cycles.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Capture interval in seconds (default: from config or 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory (default: from config)",
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
    parser.add_argument("--no-uv", action="store_true", help="Skip UV+spectrometer")
    parser.add_argument("--no-ir", action="store_true", help="Skip IR captures")
    parser.add_argument("--no-dslr", action="store_true", help="Skip DSLR captures")
    parser.add_argument("--no-env", action="store_true", help="Skip environmental")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )

    args = parser.parse_args()

    config = read_config(Path(args.config))
    orch_config = config.get("orchestrator", {})

    log_path = orch_config.get("log_path", "/home/user/logs/orchestrator.log")
    _setup_logging(log_path)

    try:
        run_loop(args, orch_config)
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user (KeyboardInterrupt)")
        sys.exit(0)


if __name__ == "__main__":
    main()
