"""
Histogram-based metering for Canon DSLR capture using the Arducam (V4L2).

The metering loop captures frames from the USB camera, computes the 95th percentile
brightness, and iteratively adjusts shutter speed and ISO to reach a target
exposure level before firing the Canon.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
from fractions import Fraction

import cv2
import numpy as np

from multicam.drivers.visible.v4l2cam import Camera as V4L2Camera
from multicam.drivers.visible.v4l2cam import capture as capture_v4l2

logger = logging.getLogger(__name__)

EOS_ISO_VALUES = [100, 200, 400, 800, 1600, 3200, 6400]
SHUTTER_MIN = Fraction(1, 4000)
SHUTTER_MAX = Fraction(30, 1)


def _parse_shutter(shutter_str: str) -> Fraction:
    """Convert a shutter speed string like ``'1/250'`` to a Fraction."""

    if "/" in shutter_str:
        parts = shutter_str.split("/")
        return Fraction(int(parts[0]), int(parts[1]))
    return Fraction(shutter_str)


def _shutter_to_str(shutter: Fraction) -> str:
    """Convert a Fraction shutter speed to the gphoto2 string format."""

    if shutter < 1:
        denom = round(1 / float(shutter))
        return f"1/{denom}"
    return str(int(shutter))


def _clamp_shutter(shutter: Fraction) -> Fraction:
    """Clamp a shutter speed to the valid EOS 4000D range."""

    if shutter < SHUTTER_MIN:
        return SHUTTER_MIN
    if shutter > SHUTTER_MAX:
        return SHUTTER_MAX
    return shutter


def _step_iso(current_iso: int, direction: int) -> int:
    """
    Step ISO up (direction=1) or down (direction=-1) in the EOS value table.

    Returns the current value if already at the limit.
    """

    try:
        idx = EOS_ISO_VALUES.index(current_iso)
    except ValueError:
        return current_iso
    new_idx = idx + direction
    new_idx = max(0, min(len(EOS_ISO_VALUES) - 1, new_idx))
    return EOS_ISO_VALUES[new_idx]


def meter_scene(
    picam: V4L2Camera,
    initial_iso: int = 800,
    initial_shutter: str = "1/250",
    target_p95: int = 204,
    max_iterations: int = 5,
    tolerance: float = 0.10,
) -> tuple[int, str, np.ndarray]:
    """
    Iterative histogram-based metering using a picam.

    Captures frames from the picam, computes the 95th percentile pixel
    brightness, and adjusts shutter speed (then ISO if needed) until
    the target exposure is reached.

    Parameters
    ----------
    picam:
        Initialised picam Camera object.
    initial_iso:
        Starting ISO value (must be a valid EOS 4000D value).
    initial_shutter:
        Starting shutter speed string (e.g. ``"1/250"``).
    target_p95:
        Target 95th percentile brightness (0–255).
    max_iterations:
        Maximum metering iterations.
    tolerance:
        Fractional tolerance around target_p95 (e.g. 0.10 = ±10%).

    Returns
    -------
    iso:
        Final metered ISO value.
    shutter_str:
        Final metered shutter speed string for gphoto2.
    last_frame:
        The last picam frame captured during metering.

    """

    iso = initial_iso
    shutter = _parse_shutter(initial_shutter)
    last_frame = None

    lo = target_p95 * (1 - tolerance)
    hi = target_p95 * (1 + tolerance)

    for iteration in range(max_iterations):
        result = capture_v4l2(picam)
        frame = result.artifacts["image"]
        last_frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        p95 = float(np.percentile(gray, 95))

        logger.info(
            "Meter iteration %d: ISO=%d shutter=%s p95=%.1f target=%d",
            iteration + 1,
            iso,
            _shutter_to_str(shutter),
            p95,
            target_p95,
        )

        if lo <= p95 <= hi:
            logger.info("Metering converged at iteration %d", iteration + 1)
            break

        if p95 < 1.0:
            ratio = 4.0
        else:
            ratio = target_p95 / p95

        new_shutter = _clamp_shutter(
            Fraction(float(shutter) * ratio).limit_denominator(10000)
        )

        if new_shutter == SHUTTER_MAX and p95 < lo:
            iso = _step_iso(iso, 1)
            shutter = new_shutter
        elif new_shutter == SHUTTER_MIN and p95 > hi:
            iso = _step_iso(iso, -1)
            shutter = new_shutter
        else:
            shutter = new_shutter

    return iso, _shutter_to_str(shutter), last_frame
