"""
Continuous (video) capture driver for Optris IR cameras.

Two modes share one capture loop, throttle, and NUC machinery:

- **Burst (default):** records a fixed number of frames at a target rate and
  keeps every frame as a single compressed ``.npz`` cube plus a summary JSON.
  Useful for drift/NUC characterisation runs.
- **Monitor (``--monitor``):** runs continuously (24/7) at a target rate,
  emitting one file set per tumbling time window. Within each window it keeps the
  first valid frame (a baseline) plus only the frames that clear an activity
  threshold, so a quiet minute collapses to a single frame. Temperature-triggered
  NUC keeps shutter actuations to a minimum.

See ``capture_video`` for details.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import json
import os
import pathlib
import signal
import sys
import time
from datetime import UTC
from datetime import datetime as dt

import cv2
import numpy as np

from multicam.utilities.naming import build_filename

from . import (
    _check_thermal_saturation,
    _monitor_frame_metrics,
    _should_nuc,
    _thermal_stats,
)
from .optris import (
    Camera as OptrisCamera,
)
from .optris import (
    _convert_temp2image,
    capture_stream_frame,
    trigger_nuc,
)


def _parse_flags(
    instrument_config: dict, extra_args: list[str] | None
) -> argparse.Namespace:
    """Parse ``capture infrared-video`` flags, falling back to ``[infrared]`` config."""

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

    # --- Monitor-mode flags ---
    flag_parser.add_argument("--monitor", action="store_true", default=False)
    flag_parser.add_argument(
        "--window-s",
        type=float,
        default=float(instrument_config.get("monitor_window_s", 60.0)),
    )
    flag_parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,  # 0 = run until SIGTERM/SIGINT
    )
    flag_parser.add_argument(
        "--delta-c",
        type=float,
        default=float(instrument_config.get("monitor_delta_c", 8.0)),
    )
    flag_parser.add_argument(
        "--bg-percentile",
        type=float,
        default=float(instrument_config.get("monitor_bg_percentile", 20.0)),
    )
    flag_parser.add_argument(
        "--min-pixels",
        type=int,
        default=int(instrument_config.get("monitor_min_pixels", 25)),
    )
    flag_parser.add_argument(
        "--abs-max-c",
        type=float,
        default=float(instrument_config.get("monitor_abs_max_c", 0.0)),
    )
    flag_parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=instrument_config.get("monitor_roi"),
    )
    flag_parser.add_argument(
        "--nuc-drift-c",
        type=float,
        default=float(instrument_config.get("nuc_drift_c", 0.5)),
    )
    flag_parser.add_argument(
        "--nuc-min-interval-s",
        type=float,
        default=float(instrument_config.get("nuc_min_interval_s", 120.0)),
    )
    flag_parser.add_argument(
        "--nuc-max-interval-s",
        type=float,
        default=float(instrument_config.get("nuc_max_interval_s", 0.0)),
    )
    return flag_parser.parse_args(extra_args or [])


def capture_video(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Record a radiometric IR "video" from an Optris camera.

    Opens the camera, runs a single mandatory start-up NUC, then dispatches to
    either the fixed-frame burst recorder (default) or the continuous monitor
    (``--monitor``). The target rate is throttled by wall clock and so is robust
    to whatever hardware rate the imager XML delivers — the XML rate must be
    >= the target, since software can only throttle down.

    Parameters
    ----------
    config:
        Camera configuration information.
    extra_args:
        Additional CLI flags. Common: ``--rate``, ``--output-dir``,
        ``--check-saturation``. Burst: ``--frames``, ``--nuc-interval``.
        Monitor: ``--monitor``, ``--window-s``, ``--duration-s``, ``--delta-c``,
        ``--bg-percentile``, ``--min-pixels``, ``--abs-max-c``, ``--roi``,
        ``--nuc-drift-c``, ``--nuc-min-interval-s``, ``--nuc-max-interval-s``.

    """

    instrument_config = config["infrared"]
    metadata_cfg = config["metadata"]
    flags = _parse_flags(instrument_config, extra_args)

    if flags.rate <= 0:
        raise ValueError("--rate must be > 0")
    if not flags.monitor and flags.frames < 1:
        raise ValueError("--frames must be >= 1")

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
    try:
        # Initial NUC so the first frames carry a fresh calibration.
        elapsed = trigger_nuc(camera)
        initial_nuc = {
            "timestamp_utc": dt.now(UTC).isoformat(),
            "duration_s": round(elapsed, 2),
        }
        print(f"   ...initial NUC complete ({elapsed:.1f}s)...")

        if flags.monitor:
            _run_monitor(
                camera, flags, metadata_cfg, instrument_config, archive, initial_nuc
            )
        else:
            _run_burst(
                camera,
                flags,
                metadata_cfg,
                instrument_config,
                archive,
                start_utc,
                initial_nuc,
            )
    finally:
        camera.close()

    sys.exit(0)


def _run_burst(
    camera: OptrisCamera,
    flags: argparse.Namespace,
    metadata_cfg: dict,
    instrument_config: dict,
    archive: pathlib.Path,
    start_utc: dt,
    initial_nuc: dict,
) -> None:
    """Fixed-frame recorder: capture ``flags.frames`` frames into one ``.npz`` cube."""

    period_s = 1.0 / flags.rate
    t_max = float(instrument_config.get("t_max", 900.0))
    sat_threshold = float(instrument_config.get("saturation_pixel_threshold", 0.05))

    nuc_events: list[dict] = [initial_nuc]
    frame_stats: list[dict] = []
    frames: list[np.ndarray] = []
    timestamps: list[str] = []

    print(f"   ...entering capture loop ({flags.frames} frames @ {flags.rate} Hz)...")
    # Reference the throttle to when each capture *starts*, not when it returns.
    # capture_stream_frame() blocks until the SDK has a fresh frame (up to ~1
    # hardware period); anchoring to the post-capture time would stack that wait
    # on top of period_s and halve the effective rate. Anchoring to the start
    # overlaps the two so the achieved rate is min(target, hardware).
    last_capture_start = None
    last_nuc = time.monotonic()
    while len(frames) < flags.frames:
        # Periodic NUC, if requested. Freezes the scene briefly.
        if flags.nuc_interval > 0 and time.monotonic() - last_nuc >= flags.nuc_interval:
            elapsed = trigger_nuc(camera)
            nuc_events.append(
                {
                    "timestamp_utc": dt.now(UTC).isoformat(),
                    "duration_s": round(elapsed, 2),
                }
            )
            last_nuc = time.monotonic()
            # Resync the cadence so the long NUC stall doesn't fire an
            # immediate catch-up frame.
            last_capture_start = time.monotonic()
            print(f"      ...periodic NUC complete ({elapsed:.1f}s)...")

        # Wall-clock throttle to the target rate.
        if (
            last_capture_start is not None
            and time.monotonic() - last_capture_start < period_s
        ):
            time.sleep(0.002)
            continue

        last_capture_start = time.monotonic()
        capture_result = capture_stream_frame(camera)

        utcnow = dt.now(UTC)
        raw_thermal = capture_result.artifacts["image"]
        idx = len(frames)

        stats = _thermal_stats(raw_thermal)
        temps = (raw_thermal.astype(np.float64) - 1000.0) / 10.0
        stats["temperature_median_c"] = round(float(np.median(temps)), 1)

        entry = {
            "frame": idx,
            "timestamp_utc": utcnow.isoformat(),
            # Detector chip temperature — drives NUC scheduling. Watch its drift
            # across a run to tune nuc_interval_s (re-NUC ~every 0.5-1 C of chip
            # drift).
            "chip_temp_c": round(float(capture_result.metadata.tempChip), 2),
            **stats,
        }
        if flags.check_saturation:
            entry["saturation_warning"] = _check_thermal_saturation(
                raw_thermal, t_max, sat_threshold
            )

        frames.append(raw_thermal)
        timestamps.append(utcnow.isoformat())
        frame_stats.append(entry)

    print("...capture sequence complete. Shutting down.")

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
        "mode": "burst",
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


def _new_window(nuc_events: list[dict] | None = None) -> dict:
    """Fresh per-window accumulator for monitor mode."""

    return {
        "start_utc": dt.now(UTC),
        "start_mono": time.monotonic(),
        "frame_count": 0,
        "baseline_frame": None,
        "kept_frames": [],
        "kept_ts": [],
        "kept_indices": [],
        "kept_reasons": [],
        "all_stats": [],
        "nuc_events": list(nuc_events or []),
    }


def _run_monitor(
    camera: OptrisCamera,
    flags: argparse.Namespace,
    metadata_cfg: dict,
    instrument_config: dict,
    archive: pathlib.Path,
    initial_nuc: dict,
) -> None:
    """
    Continuous monitor: stream at the target rate, emit one file set per window.

    Keeps only the first valid frame of each window (baseline) plus frames that
    clear an activity threshold; compact stats for *every* frame are still
    recorded so thresholds can be retuned after the fact. NUC is temperature
    triggered to spare the shutter. Flushes the in-progress window on
    SIGTERM/SIGINT for a clean ``systemctl stop``.
    """

    period_s = 1.0 / flags.rate
    roi = tuple(flags.roi) if flags.roi else None

    stop = {"flag": False}

    def _handle_stop(signum, _frame) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    print(
        f"   ...entering monitor loop (@ {flags.rate} Hz, {flags.window_s:.0f}s "
        f"windows, ROI={roi or 'full'})..."
    )

    run_start = time.monotonic()
    last_capture_start = None
    last_nuc = time.monotonic()
    # Seeded from the mandatory start-up NUC the caller just ran.
    chip_temp_at_last_nuc = float(camera.metadata.tempChip)
    last_chip_temp = chip_temp_at_last_nuc

    win = _new_window([initial_nuc])

    while not stop["flag"]:
        if flags.duration_s > 0 and time.monotonic() - run_start >= flags.duration_s:
            break

        # Tumbling-window boundary: flush and start a fresh window.
        if time.monotonic() - win["start_mono"] >= flags.window_s:
            if win["kept_frames"]:
                _flush_window(win, flags, metadata_cfg, instrument_config, archive)
            win = _new_window()

        # Temperature-triggered NUC (replaces the burst's fixed interval).
        secs_since_nuc = time.monotonic() - last_nuc
        if _should_nuc(
            last_chip_temp,
            chip_temp_at_last_nuc,
            secs_since_nuc,
            flags.nuc_drift_c,
            flags.nuc_min_interval_s,
            flags.nuc_max_interval_s,
        ):
            elapsed = trigger_nuc(camera)
            win["nuc_events"].append(
                {
                    "timestamp_utc": dt.now(UTC).isoformat(),
                    "duration_s": round(elapsed, 2),
                }
            )
            last_nuc = time.monotonic()
            chip_temp_at_last_nuc = float(camera.metadata.tempChip)
            # Resync cadence so the NUC stall doesn't fire a catch-up frame.
            last_capture_start = time.monotonic()
            print(f"      ...temp-triggered NUC ({elapsed:.1f}s)...")
            continue

        # Wall-clock throttle to the target rate.
        if (
            last_capture_start is not None
            and time.monotonic() - last_capture_start < period_s
        ):
            time.sleep(0.002)
            continue

        last_capture_start = time.monotonic()
        capture_result = capture_stream_frame(camera)

        # Skip NUC/settling frames — they read as uniform and would skew metrics
        # and pollute the baseline. trigger_nuc already consumes shutter-closed
        # frames; this is the belt-and-suspenders guard.
        if capture_result.metadata.flagState != 0:
            continue

        utcnow = dt.now(UTC)
        raw_thermal = capture_result.artifacts["image"]
        last_chip_temp = float(capture_result.metadata.tempChip)

        metrics = _monitor_frame_metrics(
            raw_thermal, roi, flags.bg_percentile, flags.delta_c
        )
        idx = win["frame_count"]

        # Presence/activity trigger — no baseline comparison.
        triggered = metrics["hot_pixel_count"] >= flags.min_pixels or (
            flags.abs_max_c > 0 and metrics["temperature_max_c"] >= flags.abs_max_c
        )

        reason = None
        if win["baseline_frame"] is None:
            reason = "baseline"
            win["baseline_frame"] = raw_thermal
        elif triggered:
            reason = "anomaly"

        win["all_stats"].append(
            {
                "frame": idx,
                "timestamp_utc": utcnow.isoformat(),
                "chip_temp_c": round(last_chip_temp, 2),
                "ambient_c": round(metrics["ambient_c"], 1),
                "hot_pixel_count": metrics["hot_pixel_count"],
                "temperature_max_c": round(metrics["temperature_max_c"], 1),
                "flag_state": int(capture_result.metadata.flagState),
                "kept": reason is not None,
                "reason": reason,
            }
        )
        win["frame_count"] += 1

        if reason is not None:
            win["kept_frames"].append(raw_thermal)
            win["kept_ts"].append(utcnow.isoformat())
            win["kept_indices"].append(idx)
            win["kept_reasons"].append(reason)

    # Flush whatever is in flight (clean stop / duration reached).
    if win["kept_frames"]:
        _flush_window(win, flags, metadata_cfg, instrument_config, archive)
    print("...monitor stopped. Final window flushed.")


def _flush_window(
    win: dict,
    flags: argparse.Namespace,
    metadata_cfg: dict,
    instrument_config: dict,
    archive: pathlib.Path,
) -> None:
    """
    Write one window's outputs (``.npz`` cube + baseline ``.png`` + metadata JSON).

    Each file is written to a temp name and ``os.replace``-d into place so the
    ``file_monitor`` inotify watcher only ever sees complete files and a mid-write
    kill cannot leave a partial file in ``receive/``.
    """

    receive = archive / "receive"
    window_start = win["start_utc"]
    cube = np.stack(win["kept_frames"], axis=0)

    # --- Thermal cube (.npz) ---
    cube_name = build_filename(
        metadata_cfg, window_start, suffix="thermal", extension="npz"
    )
    cube_tmp = receive / (cube_name + ".tmp")
    with open(cube_tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            thermal=cube,
            timestamps=np.array(win["kept_ts"]),
            frame_indices=np.array(win["kept_indices"]),
            kept_reasons=np.array(win["kept_reasons"]),
        )
    os.replace(cube_tmp, receive / cube_name)

    # --- Baseline colour quick-look (.png) for the dashboard ---
    colour_name = build_filename(
        metadata_cfg, window_start, suffix="colour", extension="png"
    )
    colour = _convert_temp2image(
        win["baseline_frame"],
        temp_min_c=float(instrument_config["colourmap_min_c"])
        if "colourmap_min_c" in instrument_config
        else None,
        temp_max_c=float(instrument_config["colourmap_max_c"])
        if "colourmap_max_c" in instrument_config
        else None,
    )
    png_tmp = receive / (colour_name + ".tmp.png")
    cv2.imwrite(str(png_tmp), colour)
    os.replace(png_tmp, receive / colour_name)

    # --- Metadata JSON (stats for every frame, kept or not) ---
    meta_name = build_filename(
        metadata_cfg, window_start, suffix="metadata", extension="json"
    )
    meta_payload = {
        "instrument": "infrared-video",
        "mode": "monitor",
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": dt.now(UTC).isoformat(),
        "target_rate_hz": flags.rate,
        "window_s": flags.window_s,
        "frames_in_window": win["frame_count"],
        "frames_kept": len(win["kept_frames"]),
        "frame_shape": list(cube.shape[1:]),
        "retention": {
            "roi": list(flags.roi) if flags.roi else None,
            "bg_percentile": flags.bg_percentile,
            "delta_c": flags.delta_c,
            "min_pixels": flags.min_pixels,
            "abs_max_c": flags.abs_max_c,
        },
        "nuc": {
            "drift_c": flags.nuc_drift_c,
            "min_interval_s": flags.nuc_min_interval_s,
            "max_interval_s": flags.nuc_max_interval_s,
        },
        "nuc_events": win["nuc_events"],
        "frames": win["all_stats"],
        "files": {"thermal_window": cube_name, "colour": colour_name},
    }
    meta_tmp = receive / (meta_name + ".tmp")
    meta_tmp.write_text(json.dumps(meta_payload, indent=2))
    os.replace(meta_tmp, receive / meta_name)

    print(
        f"   ...window {window_start.isoformat()} flushed: "
        f"{len(win['kept_frames'])}/{win['frame_count']} frames kept "
        f"({cube_name})."
    )
