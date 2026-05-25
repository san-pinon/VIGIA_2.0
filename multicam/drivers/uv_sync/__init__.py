"""
UV-sync: barrier-synchronized capture of both UV cameras and spectrometer.

This subcommand triggers both UV cameras (310 nm and 330 nm filters) and the
spectrometer simultaneously using a threading barrier. Supports saturation
checking with automatic exposure adjustment and spectrum stacking.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import csv
import json
import logging
import pathlib
import sys
from datetime import UTC
from datetime import datetime as dt

import numpy as np

from multicam.drivers.spectrometer.oceaninsight import (
    Spectrometer,
    check_spectrum_saturation,
)
from multicam.drivers.ultraviolet.picam import (
    DualCamera,
    check_image_saturation,
)
from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from .sync import synchronized_capture

logger = logging.getLogger(__name__)

# Spectrometer integration time limits (µs) for OceanHR4
SPEC_INT_MIN = 3_800
SPEC_INT_MAX = 10_000_000


def _set_uv_exposure(cameras: DualCamera, exposure_us: int) -> None:
    """Apply the same ExposureTime to both UV cameras."""

    controls = {"ExposureTime": exposure_us}
    cameras.camera_1.set_controls(controls)
    # TODO: re-enable once replacement OV5647 is ready
    # cameras.camera_2.set_controls(controls)


def capture_uv_sync(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle ``multicamctl capture uv-sync``.

    Barrier-synchronized capture of both UV cameras and spectrometer
    with optional saturation-based exposure adjustment.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).
    extra_args:
        Additional CLI flags.

    """

    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument("--check-saturation", action="store_true", default=False)
    flag_parser.add_argument("--max-retries", type=int, default=3)
    flag_parser.add_argument("--stack", type=int, default=10)
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flags = flag_parser.parse_args(extra_args or [])

    uv_config = config["ultraviolet"]
    spec_config = config["spectrometer"]
    sync_config = config.get("uv_sync", {})
    metadata_cfg = config["metadata"]

    max_retries = flags.max_retries or int(sync_config.get("max_retries", 3))
    stack_count = flags.stack or int(sync_config.get("default_stack", 10))

    output_dir = pathlib.Path(
        flags.output_dir
        or sync_config.get("output_dir")
        or (config["metadata"]["data_archive"] + "/ultraviolet")
    )
    spec_output_dir = pathlib.Path(
        config["metadata"]["data_archive"] + "/spectrometer"
    )
    (output_dir / "receive").mkdir(parents=True, exist_ok=True)
    (spec_output_dir / "receive").mkdir(parents=True, exist_ok=True)

    # --- Initialise devices ---
    print("Initialising UV cameras and spectrometer...")
    cameras = DualCamera(uv_config)
    spectrometer = Spectrometer(spec_config)

    try:
        # Track current exposure settings
        current_uv_exposure = int(
            uv_config.get("controls", {}).get("ExposureTime", 50000)
        )
        current_spec_integration = spectrometer.integration_time_micros
        uv_bit_depth = 8  # RPi5 PiSP outputs 8-bit compressed raw

        uv1_result = None
        uv2_result = None
        spec_result = None
        saturation_retries = 0

        for attempt in range(max_retries + 1):
            print(
                f"   Capture attempt {attempt + 1} "
                f"(UV exp={current_uv_exposure} µs, "
                f"spec int={current_spec_integration} µs, "
                f"stack={stack_count})"
            )

            try:
                # TODO: unpack uv2_result once replacement OV5647 is ready
                uv1_result, spec_result = synchronized_capture(
                    cameras,
                    spectrometer,
                    stack_count=stack_count,
                    spec_config={
                        "integration_time_us": current_spec_integration,
                        "stacking": stack_count,
                    },
                )
                capture_timestamp = dt.now(UTC)
            except CaptureFailure as e:
                print(f"   Capture failed: {e}")
                if attempt < max_retries:
                    saturation_retries += 1
                    continue
                raise

            if not flags.check_saturation:
                break

            # --- UV saturation check ---
            img1 = uv1_result.artifacts["image"]
            # TODO: re-enable camera_2 saturation check once replacement OV5647 is ready
            # img2 = uv2_result.artifacts["image"]

            sat1 = check_image_saturation(
                img1, min_saturation=0.2, max_saturation=0.8, bit_depth=uv_bit_depth
            )
            # sat2 = check_image_saturation(
            #     img2, min_saturation=0.2, max_saturation=0.8, bit_depth=uv_bit_depth
            # )

            uv_needs_adjust = False
            if sat1 == -1:
                current_uv_exposure = max(1, int(current_uv_exposure * 0.75))
                uv_needs_adjust = True
                print(f"   UV over-exposed — reducing to {current_uv_exposure} µs")
            elif sat1 == 1:
                current_uv_exposure = int(current_uv_exposure * 1.25)
                uv_needs_adjust = True
                print(f"   UV under-exposed — increasing to {current_uv_exposure} µs")

            if uv_needs_adjust:
                _set_uv_exposure(cameras, current_uv_exposure)

            # --- Spectrum saturation check ---
            spectrum = spec_result.artifacts["spectrum"]
            spec_sat = check_spectrum_saturation(
                spectrum, spectrometer, min_saturation=0.2, max_saturation=0.8
            )

            spec_needs_adjust = False
            if spec_sat == -1:
                new_int = max(SPEC_INT_MIN, int(current_spec_integration * 0.75))
                if new_int == SPEC_INT_MIN and current_spec_integration == SPEC_INT_MIN:
                    print(
                        "   WARNING: Spectrum saturated at minimum integration — "
                        "saving anyway"
                    )
                else:
                    current_spec_integration = new_int
                    spectrometer.integration_time_micros = current_spec_integration
                    spec_needs_adjust = True
                    print(
                        f"   Spectrum over-exposed — reducing to "
                        f"{current_spec_integration} µs"
                    )
            elif spec_sat == 1:
                new_int = min(SPEC_INT_MAX, int(current_spec_integration * 1.25))
                current_spec_integration = new_int
                spectrometer.integration_time_micros = current_spec_integration
                spec_needs_adjust = True
                print(
                    f"   Spectrum under-exposed — increasing to "
                    f"{current_spec_integration} µs"
                )

            if not uv_needs_adjust and not spec_needs_adjust:
                print("   Exposure adequate")
                break

            if attempt < max_retries:
                saturation_retries += 1
                continue
            else:
                print("   Max retries reached — saving current captures")
                break

        # --- Save outputs ---
        timestamp = capture_timestamp
        ch1 = uv1_result.metadata.get("filter_nm", 310)

        # UV images as uncompressed npz
        uv1_fname = build_filename(
            metadata_cfg, timestamp, suffix=f"uv-{ch1}", extension="npz"
        )
        np.savez(
            output_dir / "receive" / uv1_fname,
            image=uv1_result.artifacts["image"],
        )
        print(f"   Saved UV image: {uv1_fname}")
        # TODO: re-enable once replacement OV5647 is ready
        # ch2 = uv2_result.metadata.get("filter_nm", 330)
        # uv2_fname = build_filename(
        #     metadata_cfg, timestamp, suffix=f"uv-{ch2}", extension="npz"
        # )
        # np.savez(
        #     output_dir / "receive" / uv2_fname,
        #     image=uv2_result.artifacts["image"],
        # )

        # Spectrum as CSV (wavelength, intensity)
        spec_fname = build_filename(
            metadata_cfg, timestamp, suffix="spectrum", extension="csv"
        )
        wavelengths = spec_result.metadata.get("wavelengths", [])
        intensities = spec_result.artifacts["spectrum"]
        csv_path = spec_output_dir / "receive" / spec_fname
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", "intensity"])
            for wl, inten in zip(wavelengths, intensities, strict=True):
                writer.writerow([round(wl, 4), round(float(inten), 4)])
        print(f"   Saved spectrum: {spec_fname}")

        # Metadata JSON
        meta_fname = build_filename(
            metadata_cfg, timestamp, suffix="metadata", extension="json"
        )
        meta_payload = {
            "timestamp_utc": timestamp.isoformat(),
            "uv_exposure_time_us": current_uv_exposure,
            "spectrometer_integration_time_us": current_spec_integration,
            "stack_count": stack_count,
            "saturation_retries": saturation_retries,
            "uv_camera_a_filter_nm": ch1,
            "spectrum_mean_intensity": round(float(np.mean(intensities)), 2),
            "spectrum_max_intensity": round(float(np.max(intensities)), 2),
            "files": {
                "uv_a": uv1_fname,
                "spectrum": spec_fname,
            },
        }
        (output_dir / "receive" / meta_fname).write_text(
            json.dumps(meta_payload, indent=2)
        )

        print("...uv-sync capture complete.")

        summary = {
            "instrument": "uv-sync",
            "timestamp_utc": timestamp.isoformat(),
            "saturation_retries": saturation_retries,
            "files": {
                "uv_a": uv1_fname,
                "spectrum": spec_fname,
                "metadata": meta_fname,
            },
        }
        print(json.dumps(summary))
        sys.exit(0)

    finally:
        spectrometer.shutdown()
        cameras.close()
