"""
DSLR capture: fire the Canon EOS 4000D with metered or manual settings.

Metering (``--meter-with``):

- ``previous`` (orchestrator): closed loop — the previous Canon shot's
  brightness sets this shot's exposure; state in ``.canon_state.json``.
- ``picam`` / ``arducam``: meter with the Arducam first (unreliable in dim
  light, kept for manual use).
- none, or ``--iso`` + ``--shutter``: fixed settings.

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
import time
from datetime import UTC
from datetime import datetime as dt
from fractions import Fraction

import cv2
import numpy as np

from multicam.drivers.visible.canon import Camera as CanonCamera
from multicam.drivers.visible.canon import capture as capture_canon
from multicam.drivers.visible.v4l2cam import Camera as V4L2Camera
from multicam.errors import CaptureFailure
from multicam.utilities.files import atomic_write_bytes
from multicam.utilities.naming import build_filename

from .metering import (
    EOS_ISO_VALUES,
    SHUTTER_MIN,
    _clamp_shutter,
    _parse_shutter,
    _quantize_shutter,
    exposure_stops,
    image_p95,
    meter_scene,
    next_stops,
    settings_for_stops,
)

# A Canon image whose p95 is above this fraction of 255 counts as saturated.
SATURATION_FRACTION = 0.9


def _read_state(path: pathlib.Path, max_age_s: float) -> dict | None:
    """The closed-loop state, or None if missing, unreadable or too old."""

    try:
        state = json.loads(path.read_text())
        age_s = (dt.now(UTC) - dt.fromisoformat(state["updated_utc"])).total_seconds()
        float(state["next_stops"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return state if 0 <= age_s <= max_age_s else None


def _encode(
    image: np.ndarray, extension: str, params: list[int] | None = None
) -> bytes:
    """Encode an image in memory; raise CaptureFailure instead of saving nothing."""

    ok, buf = cv2.imencode(extension, image, params or [])
    if not ok:
        raise CaptureFailure(f"Encoding {extension} image failed")
    return buf.tobytes()


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
        "--meter-with", type=str, default=None, choices=["previous", "picam", "arducam"]
    )
    flag_parser.add_argument("--iso", type=int, default=None)
    flag_parser.add_argument("--shutter", type=str, default=None)
    flag_parser.add_argument("--check-saturation", action="store_true", default=True)
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
    target_p95 = int(dslr_config.get("target_p95", 180))
    max_meter = int(dslr_config.get("max_meter_iterations", 5))
    calibration_stops = float(dslr_config.get("calibration_stops", 0.0))
    # Closed-loop exposure policy (see metering.settings_for_stops).
    blur_limit_s = float(dslr_config.get("blur_limit_s", 1.0))
    max_iso = int(dslr_config.get("max_iso", 6400))
    max_shutter_s = float(dslr_config.get("max_shutter_s", 15.0))
    state_path = pathlib.Path(
        dslr_config.get("state_file") or (canon_archive / ".canon_state.json")
    )
    state_max_age_s = float(dslr_config.get("state_max_age_s", 1800))

    iso = default_iso
    shutter_str = default_shutter
    last_picam_frame = None
    metering_iterations = 0
    manual = bool(flags.iso and flags.shutter)
    closed_loop = flags.meter_with == "previous" and not manual
    metering = "previous" if closed_loop else (flags.meter_with or "fixed")
    if manual:
        metering = "manual"
    # Achievable range, so the state can't drift past what the camera can do
    # (e.g. all night at the longest exposure, then hours to come back).
    min_stops = exposure_stops(EOS_ISO_VALUES[0], str(SHUTTER_MIN))
    max_stops = exposure_stops(
        max_iso, _quantize_shutter(_clamp_shutter(Fraction(max_shutter_s)))
    )

    # --- Closed loop: settings from the previous Canon shot ---
    if closed_loop:
        state = _read_state(state_path, state_max_age_s)
        if state is not None:
            stops = float(state["next_stops"])
            source = "previous shot"
        else:
            stops = exposure_stops(default_iso, default_shutter)
            source = "config (no recent state)"
        stops = max(min_stops, min(max_stops, stops))
        iso, shutter_str = settings_for_stops(
            stops, blur_limit_s, max_iso, max_shutter_s
        )
        print(
            f"Closed-loop exposure from {source}: {stops:+.2f} stops → "
            f"ISO={iso} shutter={shutter_str}"
        )

    # --- Metering with picam ---
    elif flags.meter_with and not manual:
        print("Metering scene with picam...")
        picam = V4L2Camera(vis_config)
        try:
            iso, shutter_str, last_picam_frame = meter_scene(
                picam,
                initial_iso=default_iso,
                target_p95=target_p95,
                max_iterations=max_meter,
                calibration_stops=calibration_stops,
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
    canon = None
    canon_image = None
    retries = 0
    saturation_retries = 0
    # Closed loop: one re-shot at most; the next scheduled shot corrects the
    # rest. Each re-shot costs ~7 s and the orchestrator's timeout is ~25 s.
    max_saturation_retries = 1 if closed_loop else flags.max_retries

    try:
        for attempt in range(flags.max_retries + 1):
            try:
                if canon is None:
                    canon = CanonCamera(vis_config)
                result = capture_canon(
                    canon,
                    settings={"iso": iso, "shutterspeed": shutter_str},
                )
            except (CaptureFailure, RuntimeError) as e:
                # RuntimeError = CanonCamera() could not open the camera.
                print(f"   Canon capture failed: {e}")
                # A failed gphoto2 session usually stays broken: reconnect.
                if canon is not None:
                    canon.shutdown()
                    canon = None
                if attempt < flags.max_retries:
                    retries += 1
                    time.sleep(2.0)
                    continue
                raise

            canon_image = result.artifacts["image"]
            timestamp = dt.now(UTC)

            canon_p95 = image_p95(canon_image)
            saturated = canon_p95 > 255 * SATURATION_FRACTION
            if flags.check_saturation and saturated:
                print(
                    f"   Canon image saturated (p95={canon_p95:.0f}, "
                    f"attempt {attempt + 1})"
                )
                if (
                    attempt < flags.max_retries
                    and saturation_retries < max_saturation_retries
                ):
                    retries += 1
                    saturation_retries += 1
                    if closed_loop:
                        stops = max(
                            min_stops,
                            next_stops(
                                exposure_stops(iso, shutter_str), canon_p95, target_p95
                            ),
                        )
                        iso, shutter_str = settings_for_stops(
                            stops, blur_limit_s, max_iso, max_shutter_s
                        )
                    else:
                        current = _parse_shutter(shutter_str)
                        new_shutter = _clamp_shutter(current * 1 / 2)
                        shutter_str = _quantize_shutter(new_shutter)
                    continue
            break

    finally:
        if canon is not None:
            canon.shutdown()

    # --- Closed loop: exposure for the next shot ---
    shot_stops = exposure_stops(iso, shutter_str)
    following_stops = None
    if closed_loop:
        following_stops = max(
            min_stops, min(max_stops, next_stops(shot_stops, canon_p95, target_p95))
        )
        try:
            atomic_write_bytes(
                state_path,
                json.dumps(
                    {
                        "next_stops": round(following_stops, 3),
                        "last_stops": round(shot_stops, 3),
                        "last_p95": round(canon_p95, 1),
                        "last_iso": iso,
                        "last_shutterspeed": shutter_str,
                        "updated_utc": timestamp.isoformat(),
                    },
                    indent=2,
                ).encode(),
            )
        except OSError as e:
            print(f"   WARNING: could not write {state_path}: {e}")
        print(
            f"   Canon p95={canon_p95:.0f} (target {target_p95}) → next shot "
            f"{following_stops:+.2f} stops"
        )

    # --- Save Canon image ---
    canon_fname = build_filename(
        metadata_cfg, timestamp, suffix="canon", extension="jpg"
    )
    atomic_write_bytes(
        canon_archive / "receive" / canon_fname,
        _encode(
            cv2.cvtColor(canon_image, cv2.COLOR_RGB2BGR),
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ),
    )
    print(f"   Saved Canon image: {canon_fname}")

    # --- Save last picam frame ---
    picam_fname = None
    if last_picam_frame is not None:
        picam_fname = build_filename(
            metadata_cfg, timestamp, suffix="meter", extension="png"
        )
        atomic_write_bytes(
            picam_archive / "receive" / picam_fname, _encode(last_picam_frame, ".png")
        )
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
        "metering": metering,
        "metering_iterations": metering_iterations,
        "canon_p95": round(canon_p95, 1),
        "saturated": saturated,
        "exposure_stops": round(shot_stops, 3),
        "next_exposure_stops": (
            round(following_stops, 3) if following_stops is not None else None
        ),
        "picam_p95": picam_p95,
        "saturation_retries": retries,
        "files": {
            "canon_image": canon_fname,
            "picam_image": picam_fname,
        },
    }
    atomic_write_bytes(
        canon_archive / "receive" / meta_fname,
        json.dumps(meta_payload, indent=2).encode(),
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
