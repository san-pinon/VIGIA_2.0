"""
Crash-safe file writing.

A plain write + rename survives a kill, but not a power cut: the filesystem can
commit the new name before the data, leaving a 0-byte file under its final name
(seen in the 2026-11-04 power-cut test). ``atomic_write`` forces the data to
disk before the rename, and the rename to disk after it.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable


def _fsync(path: pathlib.Path) -> None:
    """Force a file's or directory's contents to disk."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: pathlib.Path, write: Callable[[pathlib.Path], None]) -> None:
    """
    Write ``path`` via a hidden temp name, then rename it into place.

    The file monitor skips dot-files and ``*.tmp``, so it only ever sees complete
    files, and neither a kill nor a power cut leaves a partial or empty file under
    the final name. An interrupted write can leave the temp file behind; the file
    monitor removes old ones.

    Parameters
    ----------
    path:
        Final location of the file.
    write:
        Called with the temp path; must create the file there and close it.

    """

    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    _fsync(tmp)
    os.replace(tmp, path)
    _fsync(path.parent)


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    """:func:`atomic_write` for data already in memory."""

    atomic_write(path, lambda tmp: tmp.write_bytes(data))


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """:func:`atomic_write` for text already in memory."""

    atomic_write(path, lambda tmp: tmp.write_text(text))
