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
2. Copies the file into the structured archive tree::

       {data_archive}/{instrument}/archive/{vnum}/{year}/{site}/{julday:03d}/

3. Moves the file to ``transmit/`` (for future telemetry integration)
   and removes it from ``receive/``.

Run as a long-lived systemd service (see ``systemd-units/file-monitor.service``).

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
import sys

import inotify.adapters

from multicam.utilities import read_config, rsync


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


def archive_file(source: pathlib.Path) -> int:
    """
    Move a finished file from ``receive/`` into the structured archive tree
    and copy it to ``transmit/`` for future telemetry.

    Returns
    -------
    return_code:
        0 on success, non-zero on any rsync failure.

    """

    try:
        vnum, site, year, julday = _parse_filename(source.name)
    except ValueError as exc:
        print(f"   ...skipping unrecognised filename: {exc}")
        return 1

    # Structure: {instrument}/archive/{vnum}/{year}/{site}/{julday:03d}/
    archive_dir = (
        source.parents[1]
        / "archive"
        / vnum
        / year
        / site
        / f"{julday:03d}"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    ret = rsync(
        source=str(source),
        destination=str(archive_dir / source.name),
        mkpath=True,
    )
    if ret != 0:
        print(f"   ...archive rsync failed (code {ret}) for {source.name}")
        return ret

    # Copy to transmit/ and remove from receive/
    transmit_dir = source.parents[1] / "transmit"
    transmit_dir.mkdir(parents=True, exist_ok=True)

    ret = rsync(
        source=str(source),
        destination=str(transmit_dir / source.name),
        remove_source=True,
        mkpath=True,
    )
    if ret != 0:
        print(f"   ...transmit rsync failed (code {ret}) for {source.name}")

    return ret


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

    config = read_config(config_path)
    data_dir = pathlib.Path(config["metadata"]["data_archive"])

    receive_dirs = list(data_dir.glob("**/receive/"))
    if not receive_dirs:
        print(
            f"No receive/ directories found under {data_dir}. "
            "Run at least one capture first to create them.",
            file=sys.stderr,
        )
        sys.exit(1)

    notifier = inotify.adapters.Inotify()
    for d in receive_dirs:
        notifier.add_watch(str(d))
        print(f"Watching: {d}")

    print("File monitor running — waiting for new files...\n")

    for event in notifier.event_gen(yield_nones=False):
        _, type_names, path, filename = event

        if "IN_CLOSE_WRITE" not in type_names and "IN_MOVED_TO" not in type_names:
            continue

        # Ignore hidden/temporary files created by rsync pre-hash checks.
        if filename.startswith("."):
            continue

        filepath = pathlib.Path(path) / filename
        print(f"New file: {filepath.name}")
        print("   ...archiving...")
        ret = archive_file(filepath)
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
