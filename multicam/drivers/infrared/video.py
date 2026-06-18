"""
Continuous (video) capture driver for Optris IR cameras.

Records a short radiometric "video": a fixed number of frames at a target rate,
kept as full 16-bit thermal data in a single compressed ``.npz`` cube plus a
summary JSON. See ``capture_video`` for details.

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
import time
from datetime import UTC
from datetime import datetime as dt

import numpy as np

from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from . import _check_thermal_saturation, _thermal_stats
from .optris import (
    Camera as OptrisCamera,
)
from .optris import (
    capture_stream_frame,
    trigger_nuc,
)


def capture_video(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Record a continuous radiometric IR "video" from an Optris camera.

    Captures ``frame_count`` frames at a target rate (throttled by wall clock,
    so it is robust to whatever hardware rate the imager XML delivers — the XML
    rate must be >= the target, since software can only throttle down) and
    persists them as one lossless ``.npz`` cube of uint16 thermal frames plus a
    summary JSON.

    Parameters
    ----------
    config:
        Camera configuration information.
    extra_args:
        Additional CLI flags: ``--frames``, ``--rate``, ``--nuc-interval``,
        ``--output-dir``, ``--check-saturation``.

    """

    instrument_config = config["infrared"]
    metadata_cfg = config["metadata"]

    # --- Parse extra flags; fall back to [infrared] config keys ---
    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument(
        "--frames",
        type=int,
        default=int(instrument_config.get("video_frame_count", 120)),
    )
    flag_parser.add_argument(
        "--rate",
        type=float,
        default=float(instrument_config.get("video_rate_hz", 2.0)),
    )
    flag_parser.add_argument(
        "--nuc-interval",
        type=float,
        default=float(instrument_config.get("nuc_interval_s", 0.0)),
    )
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flag_parser.add_argument("--check-saturation", action="store_true", default=False)
    flags = flag_parser.parse_args(extra_args or [])

    if flags.frames < 1:
        raise ValueError("--frames must be >= 1")
    if flags.rate <= 0:
        raise ValueError("--rate must be > 0")
    period_s = 1.0 / flags.rate

    t_max = float(instrument_config.get("t_max", 900.0))
    sat_threshold = float(instrument_config.get("saturation_pixel_threshold", 0.05))

    if flags.output_dir:
        archive = pathlib.Path(flags.output_dir)
    else:
        archive = pathlib.Path(metadata_cfg["data_archive"]) / "infrared"
    (archive / "receive").mkdir(parents=True, exist_ok=True)

    print("Capturing IR video...")
    if instrument_config["model"] != "optris":
        raise ValueError("Invalid camera model.")
    camera = OptrisCamera(instrument_config)

    start_utc = dt.now(UTC)
    nuc_events: list[dict] = []
    frame_stats: list[dict] = []
    frames: list[np.ndarray] = []
    timestamps: list[str] = []

    try:
        # Initial NUC so the first frames carry a fresh calibration.
        elapsed = trigger_nuc(camera)
        nuc_events.append(
            {"timestamp_utc": dt.now(UTC).isoformat(), "duration_s": round(elapsed, 2)}
        )
        print(f"   ...initial NUC complete ({elapsed:.1f}s)...")

        print(
            f"   ...entering capture loop ({flags.frames} frames @ {flags.rate} Hz)..."
        )
        last_accept = None
        last_nuc = time.monotonic()
        while len(frames) < flags.frames:
            # Periodic NUC, if requested. Freezes the scene briefly.
            if (
                flags.nuc_interval > 0
                and time.monotonic() - last_nuc >= flags.nuc_interval
            ):
                elapsed = trigger_nuc(camera)
                nuc_events.append(
                    {
                        "timestamp_utc": dt.now(UTC).isoformat(),
                        "duration_s": round(elapsed, 2),
                    }
                )
                last_nuc = time.monotonic()
                print(f"      ...periodic NUC complete ({elapsed:.1f}s)...")

            # Wall-clock throttle to the target rate.
            if last_accept is not None and time.monotonic() - last_accept < period_s:
                time.sleep(0.002)
                continue

            try:
                capture_result = capture_stream_frame(camera)
            except CaptureFailure:
                camera.close()
                raise

            last_accept = time.monotonic()
            utcnow = dt.now(UTC)
            raw_thermal = capture_result.artifacts["image"]
            idx = len(frames)

            stats = _thermal_stats(raw_thermal)
            temps = (raw_thermal.astype(np.float64) - 1000.0) / 10.0
            stats["temperature_median_c"] = round(float(np.median(temps)), 1)

            entry = {"frame": idx, "timestamp_utc": utcnow.isoformat(), **stats}
            if flags.check_saturation:
                entry["saturation_warning"] = _check_thermal_saturation(
                    raw_thermal, t_max, sat_threshold
                )

            frames.append(raw_thermal)
            timestamps.append(utcnow.isoformat())
            frame_stats.append(entry)

        print("...capture sequence complete. Shutting down.")
    finally:
        camera.close()

    # Stack to a single uint16 cube and persist losslessly.
    cube = np.stack(frames, axis=0)
    cube_name = build_filename(
        metadata_cfg, start_utc, suffix="thermal", extension="npz"
    )
    np.savez_compressed(
        archive / "receive" / cube_name,
        thermal=cube,
        timestamps=np.array(timestamps),
    )

    # Identify the hottest frames for quick triage.
    hottest_max = max(frame_stats, key=lambda e: e["temperature_max_c"])
    hottest_median = max(frame_stats, key=lambda e: e["temperature_median_c"])

    meta_name = build_filename(
        metadata_cfg, start_utc, suffix="metadata", extension="json"
    )
    meta_payload = {
        "instrument": "infrared-video",
        "start_utc": start_utc.isoformat(),
        "target_rate_hz": flags.rate,
        "frame_count": len(frames),
        "frame_shape": list(cube.shape[1:]),
        "nuc_events": nuc_events,
        "hottest_max_frame": {
            "frame": hottest_max["frame"],
            "temperature_max_c": hottest_max["temperature_max_c"],
        },
        "hottest_median_frame": {
            "frame": hottest_median["frame"],
            "temperature_median_c": hottest_median["temperature_median_c"],
        },
        "frames": frame_stats,
        "files": {"thermal_cube": cube_name},
    }
    (archive / "receive" / meta_name).write_text(json.dumps(meta_payload, indent=2))

    summary = {
        "instrument": "infrared-video",
        "frames_captured": len(frames),
        "files": {"thermal_cube": cube_name, "metadata": meta_name},
    }
    print(json.dumps(summary))
    sys.exit(0)
