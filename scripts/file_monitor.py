"""
Monitor for new instrument files and migrate them into the local archive.

Architecture
------------
Each instrument driver writes finished files into::

    {data_archive}/{instrument}/receive/

This script watches every ``receive/`` directory under ``data_archive``
using Linux ``inotify``.  On each ``IN_CLOSE_WRITE`` or ``IN_MOVED_TO``
event it:

1. Parses the filename to derive the archive sub-path.
2. Moves the file into the structured archive tree::

       {data_archive}/{instrument}/archive/{vnum}/{year}/{site}/{julday:03d}/

3. If ``[file_monitor] transmit = true``, hard-links it into ``transmit/``
   (for future telemetry integration; no extra disk space).

Files already in ``receive/`` at start-up (written while the monitor was
down) are archived as a backlog, and new ``receive/`` directories are
picked up by a rescan every 60 s.

If ``[file_monitor] forward_to`` is set (uvcam), a background thread moves
the local archive to that rsync destination (multicam) every
``forward_interval_s``.  Files leave this node only once transferred, so an
outage of the destination just builds a backlog here.

Run as a long-lived systemd service (see ``systemd-units/file-monitor.service``).

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import os
import pathlib
import shutil
import subprocess
import threading
import time

from multicam.utilities import read_config

RESCAN_INTERVAL_S = 60
# A backlog file modified more recently than this may still be open for
# writing; its IN_CLOSE_WRITE event (the watch is already in place) archives it.
BACKLOG_MIN_AGE_S = 10


def _parse_filename(name: str) -> tuple[str, str, str, int]:
    """
    Extract (vnum, site_code, year, julday) from a VIGIA filename.

    Expected stem format: ``{vnum}.{site_code}.{year}.{julday:03d}_…``
    """

    stem = name.split("_")[0]
    parts = stem.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {name!r}")

    vnum, site, year, julday = parts[0], parts[1], parts[2], int(parts[3])
    return vnum, site, year, julday


def _is_temp(name: str) -> bool:
    """Hidden/rsync temp files and the ``*.tmp`` files capture code renames."""

    return name.startswith(".") or name.endswith(".tmp")


def archive_file(source: pathlib.Path, transmit: bool = False) -> int:
    """
    Move a finished file from ``receive/`` into the structured archive tree
    and, optionally, hard-link it into ``transmit/`` for future telemetry.

    Returns
    -------
    return_code:
        0 on success (or if the file was already archived), 1 on failure.

    """

    try:
        vnum, site, year, julday = _parse_filename(source.name)
    except ValueError as exc:
        print(f"   ...skipping unrecognised filename: {exc}")
        return 1

    # Structure: {instrument}/archive/{vnum}/{year}/{site}/{julday:03d}/
    archive_dir = source.parents[1] / "archive" / vnum / year / site / f"{julday:03d}"
    target = archive_dir / source.name

    # receive/ and archive/ share a filesystem: a rename is atomic, so the
    # forwarder never sees a half-written file, and costs no copy.
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    except FileNotFoundError:
        # Already handled (backlog sweep and inotify both saw it).
        if not source.exists():
            return 0
        print(f"   ...archive failed for {source.name}: directory vanished")
        return 1
    except OSError as exc:
        print(f"   ...archive failed for {source.name}: {exc}")
        return 1

    if transmit:
        transmit_dir = source.parents[1] / "transmit"
        try:
            transmit_dir.mkdir(parents=True, exist_ok=True)
            os.link(target, transmit_dir / source.name)
        except FileExistsError:
            pass
        except OSError as exc:
            print(f"   ...transmit link failed for {source.name}: {exc}")
            return 1

    return 0


def _prune_empty_dirs(root: pathlib.Path) -> None:
    """Remove empty sub-directories of ``root`` (``root`` itself is kept)."""

    for dirpath, _, _ in os.walk(root, topdown=False):
        if pathlib.Path(dirpath) == root:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            pass  # not empty, or just re-created by archive_file


def forward_archive(data_dir: pathlib.Path, destination: str) -> int:
    """
    Move every ``{data_dir}/{instrument}/archive/`` tree to ``destination``
    (``user@host:/path`` or a local path) in one rsync call.

    Only files rsync has transferred are removed here, so a failed or
    interrupted run leaves the rest for the next one.

    Returns
    -------
    return_code:
        The rsync return code (0 = success).

    """

    command = [
        "rsync",
        "-a",
        "--remove-source-files",
        "--exclude=.*",
        "--exclude=*.tmp",
        "--include=/*/",
        "--include=/*/archive/***",
        "--exclude=*",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10 "
        "-o ServerAliveInterval=15 -o ServerAliveCountMax=4",
        f"{data_dir}/",
        f"{destination.rstrip('/')}/",
    ]
    ret = subprocess.run(command).returncode

    for archive_root in data_dir.glob("*/archive"):
        _prune_empty_dirs(archive_root)

    return ret


def _forward_loop(
    data_dir: pathlib.Path,
    destination: str,
    interval_s: float,
    min_free_gb: float,
) -> None:
    """Forward the archive every ``interval_s``, forever."""

    failing = False
    while True:
        start = time.monotonic()
        ret = forward_archive(data_dir, destination)
        if ret != 0:
            print(f"Forward to {destination} failed (rsync code {ret}); retrying.")
            failing = True
        elif failing:
            print(f"Forward to {destination} recovered.")
            failing = False

        free_gb = shutil.disk_usage(data_dir).free / 1e9
        if free_gb < min_free_gb:
            print(f"WARNING: only {free_gb:.1f} GB free under {data_dir}.")

        time.sleep(max(0.0, interval_s - (time.monotonic() - start)))


def _sweep(receive_dir: pathlib.Path, transmit: bool) -> None:
    """Archive files that were already in ``receive_dir``."""

    now = time.time()
    for f in sorted(receive_dir.iterdir()):
        if not f.is_file() or _is_temp(f.name):
            continue
        try:
            if now - f.stat().st_mtime < BACKLOG_MIN_AGE_S:
                continue
        except FileNotFoundError:
            continue
        print(f"Backlog file: {f.name}")
        archive_file(f, transmit)


def monitor_for_files(config_path: pathlib.Path | None = None) -> None:
    """
    Block indefinitely, watching all ``receive/`` directories under
    ``data_archive`` for new files.

    Parameters
    ----------
    config_path:
        Path to the TOML config file.  Defaults to
        ``~/.config/multicam_config.toml``.

    """

    import inotify.adapters

    config = read_config(config_path)
    data_dir = pathlib.Path(config["metadata"]["data_archive"])
    fm_config = config.get("file_monitor", {})
    transmit = fm_config.get("transmit", False)
    forward_to = fm_config.get("forward_to", "")

    if forward_to:
        interval_s = float(fm_config.get("forward_interval_s", 60))
        threading.Thread(
            target=_forward_loop,
            args=(
                data_dir,
                forward_to,
                interval_s,
                float(fm_config.get("min_free_gb", 5)),
            ),
            name="forward",
            daemon=True,
        ).start()
        print(
            f"Forwarding {data_dir}/*/archive/ to {forward_to} every {interval_s:g} s"
        )

    notifier = inotify.adapters.Inotify()
    watched: set[pathlib.Path] = set()

    def _rescan() -> None:
        for d in data_dir.glob("*/receive/"):
            if d not in watched:
                notifier.add_watch(str(d))
                watched.add(d)
                print(f"Watching: {d}")
                _sweep(d, transmit)

    _rescan()
    if not watched:
        print(
            f"No receive/ directories under {data_dir} yet; "
            f"rescanning every {RESCAN_INTERVAL_S} s."
        )
    print("File monitor running — waiting for new files...\n")

    last_rescan = time.monotonic()
    for event in notifier.event_gen(yield_nones=True, timeout_s=1):
        if time.monotonic() - last_rescan > RESCAN_INTERVAL_S:
            _rescan()
            last_rescan = time.monotonic()
        if event is None:
            continue

        _, type_names, path, filename = event

        if "IN_CLOSE_WRITE" not in type_names and "IN_MOVED_TO" not in type_names:
            continue

        # Ignore hidden/temporary files created by rsync pre-hash checks, and
        # the ``*.tmp`` files capture code writes before renaming into place
        # (the finished file arrives as IN_MOVED_TO).
        if _is_temp(filename):
            continue

        filepath = pathlib.Path(path) / filename
        print(f"New file: {filepath.name}")
        print("   ...archiving...")
        ret = archive_file(filepath, transmit)
        if ret == 0:
            print("   ...done.\n")
        else:
            print(f"   ...failed (code {ret}).\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VIGIA 2.0 file monitor")
    parser.add_argument(
        "-c",
        "--config",
        default=pathlib.Path.home() / ".config/multicam_config.toml",
        type=pathlib.Path,
        help="Path to multicam_config.toml",
    )
    args = parser.parse_args()
    monitor_for_files(args.config)
