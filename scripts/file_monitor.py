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
``forward_interval_s``.  Files leave this node only once transferred and flushed
to the destination's disk, so neither an outage nor a power cut there loses them.

Empty files are dropped with a warning, and temp files left behind by an
interrupted write or transfer are removed once they are old.

Run as a long-lived systemd service (see ``systemd-units/file-monitor.service``).

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import os
import pathlib
import shlex
import shutil
import subprocess
import threading
import time

from multicam.utilities import read_config

RESCAN_INTERVAL_S = 60
# A backlog file modified more recently than this may still be open for
# writing; its IN_CLOSE_WRITE event (the watch is already in place) archives it.
BACKLOG_MIN_AGE_S = 10
# Temp files older than this were left by an interrupted write or transfer.
STALE_TEMP_AGE_S = 3600
STALE_TEMP_CHECK_S = 3600
SSH_COMMAND = (
    "ssh -o BatchMode=yes -o ConnectTimeout=10 "
    "-o ServerAliveInterval=15 -o ServerAliveCountMax=4"
)
FLUSH_TIMEOUT_S = 120


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

    # Writers rename complete files into receive/, so this isn't one in progress.
    try:
        if source.stat().st_size == 0:
            print(f"WARNING: dropping empty file {source.name}")
            source.unlink(missing_ok=True)
            return 0
    except FileNotFoundError:
        return 0  # already handled (backlog sweep and inotify both saw it)

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


def _archived_files(data_dir: pathlib.Path) -> list[str]:
    """Finished files under ``{instrument}/archive/``, relative to ``data_dir``."""

    files = []
    for archive_root in data_dir.glob("*/archive"):
        for dirpath, _, names in os.walk(archive_root):
            files += [
                os.path.relpath(os.path.join(dirpath, name), data_dir)
                for name in names
                if not _is_temp(name)
            ]
    return sorted(files)


def _flush_destination(destination: str) -> int:
    """Force the destination's data to its disk (rsync alone doesn't); 0 = done."""

    host, separator, path = destination.partition(":")
    try:
        if not separator:
            os.sync()
            return 0
        command = [*shlex.split(SSH_COMMAND), host, f"sync -f {shlex.quote(path)}"]
        return subprocess.run(command, timeout=FLUSH_TIMEOUT_S).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Flushing {destination} failed: {exc}")
        return 1


def forward_archive(data_dir: pathlib.Path, destination: str) -> int:
    """
    Move every ``{data_dir}/{instrument}/archive/`` tree to ``destination``
    (``user@host:/path`` or a local path) in one rsync call.

    Files are removed here only after rsync has transferred all of them and the
    destination has flushed them to disk, so a failed run, or a power cut at the
    destination, leaves them for the next run (which skips those already there).

    Returns
    -------
    return_code:
        0 = success; otherwise the code from rsync or from the flush.

    """

    files = _archived_files(data_dir)
    if not files:
        return 0

    command = [
        "rsync",
        "-a",
        "--files-from=-",
        "-e",
        SSH_COMMAND,
        f"{data_dir}/",
        f"{destination.rstrip('/')}/",
    ]
    ret = subprocess.run(command, input="\n".join(files) + "\n", text=True).returncode
    if ret == 0:
        ret = _flush_destination(destination)
    if ret != 0:
        return ret

    for name in files:
        (data_dir / name).unlink(missing_ok=True)
    for archive_root in data_dir.glob("*/archive"):
        _prune_empty_dirs(archive_root)
    return 0


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
            print(f"Forward to {destination} failed (code {ret}); retrying.")
            failing = True
        elif failing:
            print(f"Forward to {destination} recovered.")
            failing = False

        free_gb = shutil.disk_usage(data_dir).free / 1e9
        if free_gb < min_free_gb:
            print(f"WARNING: only {free_gb:.1f} GB free under {data_dir}.")

        time.sleep(max(0.0, interval_s - (time.monotonic() - start)))


def remove_stale_temps(data_dir: pathlib.Path) -> None:
    """Remove old temp files from ``receive/`` and ``archive/`` trees."""

    oldest_allowed = time.time() - STALE_TEMP_AGE_S
    for root in [*data_dir.glob("*/receive"), *data_dir.glob("*/archive")]:
        for dirpath, _, names in os.walk(root):
            for name in names:
                path = pathlib.Path(dirpath) / name
                try:
                    if _is_temp(name) and path.stat().st_mtime < oldest_allowed:
                        path.unlink()
                        print(f"Removed stale temp file: {path}")
                except OSError:
                    pass  # gone already, or will be retried next time


def _stale_temp_loop(data_dir: pathlib.Path) -> None:
    """Clear out stale temp files now and every ``STALE_TEMP_CHECK_S``, forever."""

    while True:
        remove_stale_temps(data_dir)
        time.sleep(STALE_TEMP_CHECK_S)


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

    threading.Thread(
        target=_stale_temp_loop, args=(data_dir,), name="stale-temps", daemon=True
    ).start()

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
    # No ``timeout_s``: with it, event_gen *ends* after that many quiet seconds
    # and the daemon exits. Without it, a None arrives every ~1 s (epoll block)
    # and the loop runs forever.
    for event in notifier.event_gen(yield_nones=True):
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
