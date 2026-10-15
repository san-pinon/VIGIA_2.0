"""
Solar elevation from time and site coordinates (no external dependencies).

Low-precision solar position (Astronomical Almanac / NOAA style), good to
about 0.1° — plenty for a daylight gate.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import math
from datetime import datetime


def solar_elevation_deg(when: datetime, latitude: float, longitude: float) -> float:
    """
    Sun elevation above the horizon in degrees (negative below it).

    Parameters
    ----------
    when:
        Timezone-aware datetime (any zone; converted via its POSIX timestamp).
    latitude, longitude:
        Site coordinates in degrees (north and east positive).

    """

    # Days since J2000.0 (2000-01-01 12:00 TT ≈ UTC for this precision).
    n = when.timestamp() / 86400.0 + 2440587.5 - 2451545.0

    mean_longitude = (280.460 + 0.9856474 * n) % 360.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = math.radians(
        mean_longitude
        + 1.915 * math.sin(mean_anomaly)
        + 0.020 * math.sin(2 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)

    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_longitude), math.cos(ecliptic_longitude)
    )
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic_longitude))

    gmst_hours = (18.697374558 + 24.06570982441908 * n) % 24.0
    local_sidereal = math.radians((gmst_hours * 15.0 + longitude) % 360.0)
    hour_angle = local_sidereal - right_ascension

    lat = math.radians(latitude)
    elevation = math.asin(
        math.sin(lat) * math.sin(declination)
        + math.cos(lat) * math.cos(declination) * math.cos(hour_angle)
    )
    return math.degrees(elevation)
