"""
DSLR capture with optional picam-based metering.

Uses the Arducam (picam) on the SBC to meter the scene and then fires
the Canon EOS 4000D with computed ISO and shutter settings.

:copyright:
    2026, Santiago Pinon.
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

import cv2
import numpy as np

from multicam.drivers.visible.canon import Camera as CanonCamera
from multicam.drivers.visible.canon import capture as capture_canon
from multicam.drivers.visible.v4l2cam import Camera as V4L2Camera
from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from .metering import (
    EOS_ISO_VALUES,
    _clamp_shutter,
    _parse_shutter,
    _quantize_shutter,
    meter_scene,
)


def _check_canon_saturation(image: np.ndarray, target: float = 0.8) -> bool:
    """Return True if the Canon image 95th percentile exceeds target fraction of 255."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    p95 = float(np.percentile(gray, 95))
    return p95 > (255 * target)


def capture_dslr(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle ``multicamctl capture dslr`` — meter with picam, fire Canon.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).
    extra_args:
        Additional CLI flags.

    """

    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument(
        "--meter-with", type=str, default=None, choices=["picam", "arducam"]
    )
    flag_parser.add_argument("--iso", type=int, default=None)
    flag_parser.add_argument("--shutter", type=str, default=None)
    flag_parser.add_argument("--check-saturation", action="store_true", default=False)
    flag_parser.add_argument("--max-retries", type=int, default=3)
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flags = flag_parser.parse_args(extra_args or [])

    dslr_config = config.get("dslr", {})
    vis_config = config.get("visible", {})
    metadata_cfg = config["metadata"]

    # Resolve output directories
    canon_archive = pathlib.Path(
        flags.output_dir
        or dslr_config.get("output_dir")
        or (config["metadata"]["data_archive"] + "/dslr")
    )
    picam_archive = pathlib.Path(
        dslr_config.get("picam_output_dir")
        or (config["metadata"]["data_archive"] + "/picam")
    )
    (canon_archive / "receive").mkdir(parents=True, exist_ok=True)
    (picam_archive / "receive").mkdir(parents=True, exist_ok=True)

    # Defaults from config
    canon_cfg = vis_config.get("canon", {})
    default_iso = flags.iso or canon_cfg.get("iso", 800)
    default_shutter = flags.shutter or canon_cfg.get("shutterspeed", "1/250")
    target_p95 = int(dslr_config.get("target_p95", 204))
    max_meter = int(dslr_config.get("max_meter_iterations", 5))
    tolerance = float(dslr_config.get("tolerance_pct", 10)) / 100.0

    iso = default_iso
    shutter_str = default_shutter
    last_picam_frame = None
    metering_iterations = 0

    # --- Metering with picam ---
    if flags.meter_with and not (flags.iso and flags.shutter):
        print("Metering scene with picam...")
        picam = V4L2Camera(vis_config)
        try:
            iso, shutter_str, last_picam_frame = meter_scene(
                picam,
                initial_iso=default_iso,
                initial_shutter=str(default_shutter),
                target_p95=target_p95,
                max_iterations=max_meter,
                tolerance=tolerance,
            )
            metering_iterations = max_meter  # actual count logged in meter_scene
            print(f"   Metered settings: ISO={iso} shutter={shutter_str}")
        finally:
            picam.close()
    else:
        if flags.iso:
            iso = flags.iso
        if flags.shutter:
            shutter_str = flags.shutter
        print(f"Manual settings: ISO={iso} shutter={shutter_str}")

    # --- Validate ISO ---
    if iso not in EOS_ISO_VALUES:
        closest = min(EOS_ISO_VALUES, key=lambda x: abs(x - iso))
        print(f"   WARNING: ISO {iso} not valid for EOS 4000D, using {closest}")
        iso = closest

    # --- Canon capture (with optional saturation retry) ---
    print("Firing Canon DSLR...")
    canon = CanonCamera(vis_config)
    canon_image = None
    retries = 0

    try:
        for attempt in range(flags.max_retries + 1):
            try:
                result = capture_canon(
                    canon,
                    settings={"iso": iso, "shutterspeed": shutter_str},
                )
            except CaptureFailure as e:
                print(f"   Canon capture failed: {e}")
                if attempt < flags.max_retries:
                    retries += 1
                    continue
                raise

            canon_image = result.artifacts["image"]
            timestamp = dt.now(UTC)

            if flags.check_saturation and _check_canon_saturation(canon_image):
                print(f"   Canon image saturated (attempt {attempt + 1})")
                if attempt < flags.max_retries:
                    retries += 1
                    current = _parse_shutter(shutter_str)
                    new_shutter = _clamp_shutter(current * 3 / 4)
                    shutter_str = _quantize_shutter(new_shutter)
                    continue
            break

    finally:
        canon.shutdown()

    # --- Save Canon image ---
    canon_fname = build_filename(
        metadata_cfg, timestamp, suffix="canon", extension="png"
    )
    cv2.imwrite(str(canon_archive / "receive" / canon_fname), canon_image)
    print(f"   Saved Canon image: {canon_fname}")

    # --- Save last picam frame ---
    picam_fname = None
    if last_picam_frame is not None:
        picam_fname = build_filename(
            metadata_cfg, timestamp, suffix="meter", extension="png"
        )
        cv2.imwrite(str(picam_archive / "receive" / picam_fname), last_picam_frame)
        print(f"   Saved picam frame: {picam_fname}")

    # --- Compute metering p95 for metadata ---
    picam_p95 = None
    if last_picam_frame is not None:
        gray = (
            cv2.cvtColor(last_picam_frame, cv2.COLOR_BGR2GRAY)
            if last_picam_frame.ndim == 3
            else last_picam_frame
        )
        picam_p95 = round(float(np.percentile(gray, 95)), 1)

    # --- Write metadata ---
    meta_fname = build_filename(
        metadata_cfg, timestamp, suffix="metadata", extension="json"
    )
    meta_payload = {
        "timestamp_utc": timestamp.isoformat(),
        "iso": iso,
        "shutterspeed": shutter_str,
        "metering_iterations": metering_iterations,
        "picam_p95": picam_p95,
        "saturation_retries": retries,
        "files": {
            "canon_image": canon_fname,
            "picam_image": picam_fname,
        },
    }
    (canon_archive / "receive" / meta_fname).write_text(
        json.dumps(meta_payload, indent=2)
    )

    summary = {
        "instrument": "dslr",
        "timestamp_utc": timestamp.isoformat(),
        "iso": iso,
        "shutterspeed": shutter_str,
        "saturation_retries": retries,
        "files": {
            "canon": canon_fname,
            "picam": picam_fname,
            "metadata": meta_fname,
        },
    }
    print(json.dumps(summary))
    sys.exit(0)
