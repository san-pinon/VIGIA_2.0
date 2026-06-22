"""
Collection of drivers for IR camera systems.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import json
import pathlib
import sys
from datetime import UTC
from datetime import datetime as dt
from datetime import timedelta as td

import cv2
import numpy as np

from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from .optris import (
    Camera as OptrisCamera,
)
from .optris import (
    _convert_temp2image,
)
from .optris import (
    capture as capture_optris,
)


def _check_thermal_saturation(
    thermal_frame: np.ndarray,
    t_max: float,
    threshold: float = 0.05,
) -> bool:
    """
    Check if a thermal frame is saturated.

    Parameters
    ----------
    thermal_frame:
        Raw uint16 thermal data from the Optris camera.
    t_max:
        Maximum temperature (°C) of the sensor range.
    threshold:
        Fraction of pixels at or above T_max to trigger a warning.

    Returns
    -------
    saturated:
        ``True`` if more than ``threshold`` fraction of pixels are at T_max.

    """

    temp_frame = (thermal_frame.astype(np.float64) - 1000.0) / 10.0
    fraction_at_max = np.mean(temp_frame >= t_max)
    return bool(fraction_at_max > threshold)


def _thermal_stats(thermal_frame: np.ndarray) -> dict:
    """Compute min/max/mean temperature from raw Optris uint16 values."""

    temp_frame = (thermal_frame.astype(np.float64) - 1000.0) / 10.0
    return {
        "temperature_min_c": round(float(temp_frame.min()), 1),
        "temperature_max_c": round(float(temp_frame.max()), 1),
        "temperature_mean_c": round(float(temp_frame.mean()), 1),
    }


def _scene_background(temp_roi: np.ndarray, percentile: float) -> float:
    """
    Robust per-frame ambient estimate: a low percentile of the ROI temperatures.

    Using a percentile (not the min) shrugs off cold-pixel noise, and because it
    is recomputed every frame it tracks day/night ambient drift automatically.
    """

    return float(np.percentile(temp_roi, percentile))


def _monitor_frame_metrics(
    thermal_frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    bg_percentile: float,
    delta_c: float,
) -> dict:
    """
    Per-frame activity metrics for monitor-mode retention.

    Computes a robust scene background within the ROI and counts pixels that sit
    ``delta_c`` °C above it ("over-ambient"). Stateless and day/night-robust: the
    background is re-derived from the frame itself, so a uniform warm/cool scene
    yields ~0 hot pixels while a localised hotspot stands out.

    Parameters
    ----------
    thermal_frame:
        Raw uint16 thermal data from the Optris camera (``temp = (v-1000)/10``).
    roi:
        ``(x0, y0, x1, y1)`` sub-region to analyse, or ``None`` for the whole frame.
    bg_percentile:
        Percentile of the ROI used as the ambient/background estimate.
    delta_c:
        °C above background a pixel must exceed to count as "hot".

    Returns
    -------
    metrics:
        ``{ambient_c, hot_pixel_count, temperature_max_c}`` over the ROI.

    """

    temp_frame = (thermal_frame.astype(np.float64) - 1000.0) / 10.0
    if roi is not None:
        x0, y0, x1, y1 = roi
        temp_roi = temp_frame[y0:y1, x0:x1]
    else:
        temp_roi = temp_frame

    ambient = _scene_background(temp_roi, bg_percentile)
    hot_pixel_count = int(np.count_nonzero(temp_roi > ambient + delta_c))
    return {
        "ambient_c": ambient,
        "hot_pixel_count": hot_pixel_count,
        "temperature_max_c": float(temp_roi.max()),
    }


def _should_nuc(
    chip_temp: float,
    chip_temp_at_last_nuc: float,
    secs_since_nuc: float,
    drift_c: float,
    min_interval_s: float,
    max_interval_s: float,
) -> bool:
    """
    Decide whether to trigger a temperature-driven NUC.

    Minimises shutter actuations (the camera's main wear item) by NUC-ing on chip
    drift rather than a fixed cadence, while honouring a min-interval floor so a
    noisy/oscillating chip temperature cannot hammer the shutter, and an optional
    max-interval ceiling as a safety re-NUC.

    Parameters
    ----------
    chip_temp:
        Latest detector chip temperature (°C).
    chip_temp_at_last_nuc:
        Chip temperature recorded at the previous NUC (°C).
    secs_since_nuc:
        Seconds elapsed since the previous NUC.
    drift_c:
        Absolute chip-temp drift (°C) since the last NUC that triggers a new one.
    min_interval_s:
        Hard floor between NUCs, in seconds (protects the shutter).
    max_interval_s:
        Optional ceiling: force a NUC after this many seconds (0 = disabled).

    """

    if secs_since_nuc < min_interval_s:
        return False
    if max_interval_s > 0 and secs_since_nuc >= max_interval_s:
        return True
    return abs(chip_temp - chip_temp_at_last_nuc) >= drift_c


def capture_image(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handles queries to IR cameras attached to the multicam system.

    Parameters
    ----------
    config:
        Camera configuration information.
    extra_args:
        Additional CLI flags: ``--check-saturation``, ``--output-dir``,
        ``--max-retries``.

    """

    # --- Parse extra flags ---
    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument("--check-saturation", action="store_true", default=False)
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flag_parser.add_argument("--max-retries", type=int, default=3)
    flags = flag_parser.parse_args(extra_args or [])

    instrument_config = config["infrared"]

    print("Capturing images...")
    match instrument_config["model"]:
        case "optris":
            camera = OptrisCamera(instrument_config)
            capture_fn = capture_optris
        case _:
            raise ValueError("Invalid camera model.")

    time_between_frames = 1.0 / instrument_config["framerate"]

    if flags.output_dir:
        archive = pathlib.Path(flags.output_dir)
    else:
        archive = pathlib.Path(config["metadata"]["data_archive"]) / "infrared"
    (archive / "receive").mkdir(parents=True, exist_ok=True)

    metadata_cfg = config["metadata"]
    t_max = float(instrument_config.get("t_max", 900.0))
    sat_threshold = float(instrument_config.get("saturation_pixel_threshold", 0.05))

    frames, starttime = 0, dt.now(UTC)
    summaries: list[dict] = []

    try:
        print("   ...entering capture loop...")
        while frames < instrument_config["frame_count"]:
            now_check = dt.now(UTC)

            if now_check < starttime + td(seconds=time_between_frames) and frames != 0:
                continue

            try:
                capture_result = capture_fn(camera)
            except CaptureFailure:
                camera.close()
                raise

            utcnow = dt.now(UTC)
            raw_thermal = capture_result.artifacts["image"]
            stats = _thermal_stats(raw_thermal)

            saturation_warning = False
            if flags.check_saturation:
                saturation_warning = _check_thermal_saturation(
                    raw_thermal, t_max, sat_threshold
                )
                if saturation_warning:
                    print(
                        "      WARNING: THERMAL_SATURATION_WARNING — "
                        f">{sat_threshold * 100:.0f}% of pixels at T_max ({t_max}°C)"
                    )

            # Save 16-bit TIFF (raw thermal data)
            tiff_name = build_filename(
                metadata_cfg, utcnow, extension="tiff", frame=frames
            )
            tiff_path = archive / "receive" / tiff_name
            cv2.imwrite(str(tiff_path), raw_thermal)

            # Also save the false-colour image for quick inspection
            if instrument_config["model"] == "optris":
                colour_image = _convert_temp2image(
                    raw_thermal,
                    temp_min_c=float(instrument_config["colourmap_min_c"])
                    if "colourmap_min_c" in instrument_config
                    else None,
                    temp_max_c=float(instrument_config["colourmap_max_c"])
                    if "colourmap_max_c" in instrument_config
                    else None,
                )
                colour_name = build_filename(
                    metadata_cfg, utcnow, suffix="colour", extension="png", frame=frames
                )
                cv2.imwrite(str(archive / "receive" / colour_name), colour_image)

            # Write metadata JSON
            meta_name = build_filename(
                metadata_cfg, utcnow, suffix="metadata", extension="json", frame=frames
            )
            meta_payload = {
                "timestamp_utc": utcnow.isoformat(),
                **stats,
                "saturation_warning": saturation_warning,
                "files": {"thermal_tiff": tiff_name},
            }
            (archive / "receive" / meta_name).write_text(
                json.dumps(meta_payload, indent=2)
            )

            summaries.append(meta_payload)
            starttime += td(seconds=time_between_frames)
            frames += 1

        print("...capture sequence complete. Shutting down.")
    finally:
        camera.close()

    summary = {
        "instrument": "infrared",
        "frames_captured": frames,
        "captures": summaries,
    }
    print(json.dumps(summary))
    sys.exit(0)


# Imported at the bottom so ``video`` can reuse the helpers defined above
# (``_thermal_stats``, ``_check_thermal_saturation``) without a circular import.
from .video import capture_video as capture_video  # noqa: E402
