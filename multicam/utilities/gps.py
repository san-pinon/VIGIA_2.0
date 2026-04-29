"""
GPS timestamping utility via gpsd.

Connects to a local gpsd daemon to retrieve a GPS-derived UTC timestamp.
Falls back to system UTC if gpsd is unreachable or no fix is available.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def get_gps_timestamp(timeout_s: float = 5.0) -> tuple[datetime, bool]:
    """
    Attempt to read a GPS fix via gpsd and return its UTC timestamp.

    Parameters
    ----------
    timeout_s:
        Maximum seconds to wait for a valid GPS fix.

    Returns
    -------
    timestamp:
        UTC datetime from GPS, or system UTC as fallback.
    is_gps:
        ``True`` if the timestamp came from GPS, ``False`` if it is a
        system-clock fallback.

    """

    try:
        from gpsd import connect, get_current  # gpsd-py3
    except ImportError:
        logger.warning("gpsd-py3 not installed — using system UTC")
        return datetime.now(UTC), False

    try:
        connect()
    except Exception:
        logger.warning("Could not connect to gpsd — using system UTC")
        return datetime.now(UTC), False

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            packet = get_current()
            if packet.mode >= 2 and packet.time:
                gps_time = datetime.fromisoformat(packet.time.replace("Z", "+00:00"))
                return gps_time, True
        except Exception:
            pass
        time.sleep(0.25)

    logger.warning("No GPS fix within %.1f s — using system UTC", timeout_s)
    return datetime.now(UTC), False
