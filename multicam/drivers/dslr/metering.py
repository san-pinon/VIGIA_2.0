"""
Exposure control for the Canon DSLR.

Two ways to pick ISO + shutter:

- **Closed loop** (``--meter-with previous``, used by the orchestrator): the
  brightness of the previous Canon shot sets the next one (``next_stops``,
  ``settings_for_stops``), like the UV monitor. Works day and night.
- **Arducam EV metering** (``--meter-with picam``, ``meter_scene`` below):

  the Arducam is switched to manual exposure; a binary search finds the
  integration time that reaches a target brightness, converted to Canon
  ISO + shutter via a one-time calibration offset. Kept for manual use: the
  Arducam honours only a few ms of manual exposure, so it fails in dim light.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import math
import subprocess
import time
from fractions import Fraction

import cv2
import numpy as np

from multicam.drivers.visible.v4l2cam import Camera as V4L2Camera

logger = logging.getLogger(__name__)

EOS_ISO_VALUES = [100, 200, 400, 800, 1600, 3200, 6400]
SHUTTER_MIN = Fraction(1, 4000)
SHUTTER_MAX = Fraction(30, 1)

EOS_SHUTTER_STRINGS = [
    "1/4000", "1/3200", "1/2500", "1/2000", "1/1600", "1/1250", "1/1000",
    "1/800", "1/640", "1/500", "1/400", "1/320", "1/250", "1/200", "1/160",
    "1/125", "1/100", "1/80", "1/60", "1/50", "1/40", "1/30", "1/25", "1/20",
    "1/15", "1/13", "1/10", "1/8", "1/6", "1/5", "1/4", "0.3", "0.4", "0.5",
    "0.6", "0.8", "1", "1.3", "1.6", "2", "2.5", "3.2", "4", "5", "6", "8",
    "10", "13", "15", "20", "25", "30",
]
_EOS_SHUTTER_FLOATS = [float(Fraction(s)) for s in EOS_SHUTTER_STRINGS]

# V4L2 exposure_absolute units: 1 unit = 100 µs
_EXP_UNIT_S = 1e-4
_EXP_MIN = 1      # 0.1 ms
_EXP_MAX = 5000   # 500 ms


def _quantize_shutter(shutter: Fraction) -> str:
    """Nearest EOS shutter string, by ratio (so 1/3000 → 1/3200, not 1/4000)."""

    target = max(float(shutter), 1e-6)
    idx = min(
        range(len(_EOS_SHUTTER_FLOATS)),
        key=lambda i: abs(math.log(_EOS_SHUTTER_FLOATS[i] / target)),
    )
    return EOS_SHUTTER_STRINGS[idx]


def _parse_shutter(shutter_str: str) -> Fraction:
    if "/" in shutter_str:
        n, d = shutter_str.split("/")
        return Fraction(int(n), int(d))
    return Fraction(shutter_str)


def _clamp_shutter(shutter: Fraction) -> Fraction:
    return max(SHUTTER_MIN, min(SHUTTER_MAX, shutter))


def _step_iso(current_iso: int, direction: int) -> int:
    try:
        idx = EOS_ISO_VALUES.index(current_iso)
    except ValueError:
        return current_iso
    return EOS_ISO_VALUES[max(0, min(len(EOS_ISO_VALUES) - 1, idx + direction))]


def _v4l2_set(device_path: str, **controls) -> None:
    """Set one or more V4L2 controls atomically via v4l2-ctl."""
    ctrl_str = ",".join(f"{k}={v}" for k, v in controls.items())
    try:
        subprocess.run(
            ["v4l2-ctl", f"--device={device_path}", f"--set-ctrl={ctrl_str}"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("v4l2-ctl set failed: %s", exc)


def meter_scene(
    picam: V4L2Camera,
    initial_iso: int = 100,
    initial_shutter: str = "1/250",
    target_p95: int = 128,
    max_iterations: int = 8,
    calibration_stops: float = 0.0,
) -> tuple[int, str, np.ndarray]:
    """
    EV-based metering: fix the Arducam in manual mode, binary-search for
    the integration time that reaches *target_p95*, then derive Canon
    ISO + shutter from the measured scene EV.

    The Arducam is always restored to auto-exposure before returning.

    Parameters
    ----------
    picam:
        Initialised V4L2Camera (auto-exposure on, warmup done).
    initial_iso:
        Canon ISO to target.  Stepped up/down automatically when the
        computed shutter would exceed EOS limits.
    initial_shutter:
        Ignored — kept for API compatibility.
    target_p95:
        Target 95th-percentile brightness (0–255) for the metering frame.
    max_iterations:
        Binary-search iterations (each ~0.25 s).  8 covers ~8 stops of
        dynamic range from the starting exposure.
    calibration_stops:
        Additive EV offset applied to the Canon shutter after EV
        computation.  Tune once in the field: negative = Canon was too
        bright, positive = too dark.  Stored in ``[dslr]
        calibration_stops`` in the TOML config.

    Returns
    -------
    iso:
        Canon ISO to use.
    shutter_str:
        Canon shutter speed string for gphoto2.
    last_frame:
        The last Arducam frame captured during metering (BGR uint8).
    """
    device_path = getattr(picam, "_device_path", "/dev/video0")

    # Starting exposure: 50 ms (midrange, 5 stops from each limit)
    exp = 500
    last_frame = None
    p95_actual = float(target_p95)

    # Switch to manual exposure + minimum gain so pixel brightness
    # is proportional to scene luminance × integration time.
    _v4l2_set(device_path, exposure_auto=1, gain=0)
    time.sleep(0.2)

    try:
        for iteration in range(max_iterations):
            _v4l2_set(device_path, exposure_absolute=exp)
            time.sleep(0.15)  # let sensor apply new integration time

            picam._cap.read()  # flush stale frame
            ret, frame = picam._cap.read()
            if not ret or frame is None:
                logger.warning("EV meter iter %d: frame read failed", iteration + 1)
                continue

            last_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            p95_actual = float(np.percentile(gray, 95))

            print(
                f"      EV meter iter {iteration + 1}: "
                f"exp={exp} ({exp * 0.1:.1f} ms)  p95={p95_actual:.0f}  target={target_p95}"
            )

            if target_p95 * 0.9 <= p95_actual <= target_p95 * 1.1:
                break  # converged

            ratio = target_p95 / max(p95_actual, 1.0)
            new_exp = int(exp * ratio)
            new_exp = max(_EXP_MIN, min(_EXP_MAX, new_exp))
            if new_exp == exp:
                break  # at a sensor limit — stop iterating
            exp = new_exp

    finally:
        _v4l2_set(device_path, exposure_auto=3)  # always restore auto

    # --- EV → Canon shutter ---
    #
    # At manual exposure t_p (seconds), measured p95_actual:
    #   scene luminance L ∝ p95_actual / t_p
    #
    # Canon target brightness (same target_p95):
    #   target_p95 ∝ L × t_canon × (canon_iso / 100)
    #
    # Solving:
    #   t_canon = t_p × (target_p95 / p95_actual) × (100 / canon_iso) × 2^calibration_stops
    #
    t_p = exp * _EXP_UNIT_S
    t_canon_raw = (
        t_p
        * (target_p95 / max(p95_actual, 1.0))
        * (100.0 / initial_iso)
        * (2.0 ** calibration_stops)
    )

    iso = initial_iso

    # If shutter would exceed EOS limits, step ISO to compensate (one step).
    if t_canon_raw > float(SHUTTER_MAX) and iso < EOS_ISO_VALUES[-1]:
        iso = _step_iso(iso, 1)
        t_canon_raw /= 2.0
        logger.info("EV meter: shutter > 30 s — ISO stepped to %d", iso)
    elif t_canon_raw < float(SHUTTER_MIN) and iso > EOS_ISO_VALUES[0]:
        iso = _step_iso(iso, -1)
        t_canon_raw *= 2.0
        logger.info("EV meter: shutter < 1/4000 — ISO stepped to %d", iso)

    t_clamped = max(float(SHUTTER_MIN), min(float(SHUTTER_MAX), t_canon_raw))
    shutter_str = _quantize_shutter(Fraction(t_clamped).limit_denominator(10000))

    logger.info(
        "EV metering done: picam exp=%.1f ms  p95=%.0f → Canon ISO=%d shutter=%s",
        exp * 0.1, p95_actual, iso, shutter_str,
    )

    return iso, shutter_str, last_frame


# ---------------------------------------------------------------------------
# Closed loop: meter from the previous Canon shot
# ---------------------------------------------------------------------------

# p95 at or above this (8-bit) is treated as clipped: the true level is unknown.
CLIPPED_P95 = 250
# JPEG values are gamma-encoded; linear light ≈ (value / 255) ** JPEG_GAMMA.
JPEG_GAMMA = 2.2
# Changes smaller than this (stops) are ignored, so settings don't hunt.
DEADBAND_STOPS = 0.33
# Largest correction per shot (stops); clipped shots step down by CLIPPED_STEP.
MAX_STEP_UP = 4.0
MAX_STEP_DOWN = 3.0
CLIPPED_STEP = 3.0


def image_p95(image: np.ndarray) -> float:
    """95th-percentile grey level of an RGB (or grey) uint8 image, subsampled."""

    small = image[::4, ::4]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) if small.ndim == 3 else small
    return float(np.percentile(gray, 95))


def exposure_stops(iso: int, shutter_str: str) -> float:
    """Light gathered, in stops relative to 1 s at ISO 100 (aperture fixed)."""

    return math.log2(float(_parse_shutter(shutter_str)) * iso / 100.0)


def settings_for_stops(
    stops: float, blur_limit_s: float, max_iso: int, max_shutter_s: float
) -> tuple[int, str]:
    """
    ISO + shutter for an exposure of ``stops`` (see ``exposure_stops``).

    Policy: ISO 100 while the shutter stays ≤ ``blur_limit_s``; then the lowest
    ISO (up to ``max_iso``) that keeps the shutter there; then the shutter grows
    up to ``max_shutter_s``. Beyond that the shot is simply darker.
    """

    t100 = 2.0**stops  # shutter needed at ISO 100
    iso = EOS_ISO_VALUES[0]
    if t100 > blur_limit_s:
        allowed = [i for i in EOS_ISO_VALUES if i <= max_iso] or [EOS_ISO_VALUES[0]]
        iso = next(
            (i for i in allowed if t100 * 100.0 / i <= blur_limit_s), allowed[-1]
        )
    shutter = t100 * 100.0 / iso
    shutter = max(float(SHUTTER_MIN), min(max_shutter_s, shutter))
    return iso, _quantize_shutter(Fraction(shutter).limit_denominator(10000))


def next_stops(stops: float, p95: float, target_p95: float) -> float:
    """
    Exposure (stops) for the next shot, from this shot's exposure and p95.

    Unclipped: scale by the linear-light ratio (undoing the JPEG gamma),
    limited to MAX_STEP_UP / MAX_STEP_DOWN per shot, with a dead band.
    Clipped: step down CLIPPED_STEP stops, since the true level is unknown.
    """

    if p95 >= CLIPPED_P95:
        return stops - CLIPPED_STEP
    delta = JPEG_GAMMA * math.log2(target_p95 / max(p95, 1.0))
    if abs(delta) < DEADBAND_STOPS:
        return stops
    return stops + max(-MAX_STEP_DOWN, min(MAX_STEP_UP, delta))
