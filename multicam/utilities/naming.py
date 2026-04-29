"""
Centralized filename builder for the VIGIA naming convention.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from datetime import datetime


def build_filename(
    config_metadata: dict,
    timestamp: datetime,
    suffix: str = "",
    extension: str = "png",
    frame: int | None = None,
) -> str:
    """
    Build a filename following the VIGIA naming convention.

    Pattern::

        {vnum}.{site_code}.{year}.{julday:03d}_{HHMMSS}[-{frame:04d}][{suffix}].{ext}

    Parameters
    ----------
    config_metadata:
        The ``config["metadata"]`` dict containing ``vnum`` and ``site_code``.
    timestamp:
        UTC datetime used for the date/time portion of the filename.
    suffix:
        Optional suffix appended before the extension (e.g. ``"-310"``).
    extension:
        File extension without leading dot (e.g. ``"png"``, ``"npz"``).
    frame:
        Optional frame number for deduplication.

    Returns
    -------
    filename:
        The constructed filename string.

    """

    julday = timestamp.timetuple().tm_yday
    time_str = f"{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"

    parts = [
        f"{config_metadata['vnum']}.{config_metadata['site_code']}",
        f"{timestamp.year}.{julday:03d}_{time_str}",
    ]
    name = ".".join(parts)

    if frame is not None:
        name += f"-{frame:04d}"
    if suffix:
        name += f"-{suffix}"

    return f"{name}.{extension}"
