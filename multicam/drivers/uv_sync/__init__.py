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
    configured_integration_ms,
)
from multicam.drivers.ultraviolet.picam import (
    DualCamera,
    configured_exposure_ms,
    image_brightness,
)
from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from .sync import synchronized_capture

logger = logging.getLogger(__name__)

# Spectrometer integration time limits (ms) for OceanHR4
SPEC_INT_MIN_MS = 3.8
SPEC_INT_MAX_MS = 10_000.0


# Brightness band (fraction of full scale, brightest pixels) a UV image must
# fall in, and the level the auto-adjust aims for when it doesn't.
UV_MIN_BRIGHTNESS = 0.2
UV_MAX_BRIGHTNESS = 0.8
UV_TARGET_BRIGHTNESS = 0.6
UV_CLIPPED_BRIGHTNESS = 0.99


def _next_uv_exposure(exposure_ms: float, brightness: float) -> float | None:
    """Exposure (ms) that brings ``brightness`` into band, or None if in band."""

    if UV_MIN_BRIGHTNESS <= brightness <= UV_MAX_BRIGHTNESS:
        return None
    if brightness >= UV_CLIPPED_BRIGHTNESS:
        # Clipped: the true brightness is unknown, so halve instead of scaling.
        return exposure_ms * 0.5
    factor = UV_TARGET_BRIGHTNESS / max(brightness, 0.01)
    return exposure_ms * min(4.0, max(0.25, factor))


def _read_uv_state(state_path: pathlib.Path) -> dict:
    """Read the persisted exposure/integration state, or {} if unavailable."""

    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_uv_state(
    state_path: pathlib.Path,
    uv_exposure_ms: dict[int, float],
    config_exposure_ms: dict[int, float],
    spec_integration_ms: float,
    config_spec_integration_ms: float,
    timestamp_iso: str,
) -> None:
    """
    Persist the converged exposure settings for the next capture run.

    The config values are stored too: a saved exposure/integration time is
    only reused while the config value it started from is unchanged.
    """

    payload = {
        "uv1_exposure_ms": round(uv_exposure_ms[1], 3),
        "uv2_exposure_ms": round(uv_exposure_ms[2], 3),
        "uv1_config_exposure_ms": config_exposure_ms[1],
        "uv2_config_exposure_ms": config_exposure_ms[2],
        "spec_integration_ms": round(spec_integration_ms, 3),
        "spec_config_integration_ms": config_spec_integration_ms,
        "updated_utc": timestamp_iso,
    }
    try:
        state_path.write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.warning("Could not write UV state file %s: %s", state_path, e)


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
    flag_parser.add_argument("--check-saturation", action="store_true", default=True)
    flag_parser.add_argument("--max-retries", type=int, default=3)
    flag_parser.add_argument("--stack", type=int, default=10)
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flag_parser.add_argument("--uv1-exposure-ms", type=float, default=None)
    flag_parser.add_argument("--uv2-exposure-ms", type=float, default=None)
    flag_parser.add_argument("--spec-integration-ms", type=float, default=None)
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
    spec_output_dir = pathlib.Path(config["metadata"]["data_archive"] + "/spectrometer")
    (output_dir / "receive").mkdir(parents=True, exist_ok=True)
    (spec_output_dir / "receive").mkdir(parents=True, exist_ok=True)

    # Exposure state persisted between runs so each cycle starts from the last
    # converged value instead of the (often saturating) config default.
    state_path = pathlib.Path(
        sync_config.get("state_file") or (output_dir / ".uv_sync_state.json")
    )
    saved_state = _read_uv_state(state_path)

    # --- Initialise devices ---
    print("Initialising UV cameras and spectrometer...")
    cameras = DualCamera(uv_config)
    spectrometer = Spectrometer(spec_config)

    try:
        # Resolve each camera's starting exposure with precedence: explicit
        # CLI override > last converged value from a previous run (only while
        # the config value is unchanged) > config default. The two filters need
        # very different exposures, so each camera is resolved on its own.
        filters = {
            n: uv_config.get(f"camera_{n}_filter_nm", 310 if n == 1 else 330)
            for n in (1, 2)
        }
        config_exposure = {n: configured_exposure_ms(uv_config, n) for n in (1, 2)}
        for n in (1, 2):
            cli_ms = getattr(flags, f"uv{n}_exposure_ms")
            saved_ms = saved_state.get(f"uv{n}_exposure_ms")
            if cli_ms is not None:
                start_ms, source = cli_ms, "CLI"
            elif (
                saved_ms
                and saved_state.get(f"uv{n}_config_exposure_ms") == config_exposure[n]
            ):
                start_ms, source = saved_ms, "last run"
            else:
                start_ms, source = config_exposure[n], "config"
            if start_ms != cameras.exposure_ms[n]:
                cameras.set_exposure(n, start_ms)
            # Make sure manual exposure/gain are in effect before the first frame.
            cameras.settle_exposure(n)
            print(
                f"   UV camera_{n} ({filters[n]} nm): starting at "
                f"{cameras.exposure_ms[n]:.3f} ms ({source}), "
                f"gain {cameras.analogue_gain}"
            )

        # Same precedence for the spectrometer integration time.
        config_spec_ms = configured_integration_ms(spec_config)
        saved_spec_ms = saved_state.get("spec_integration_ms")
        if flags.spec_integration_ms is not None:
            current_spec_ms, source = flags.spec_integration_ms, "CLI"
        elif (
            saved_spec_ms
            and saved_state.get("spec_config_integration_ms") == config_spec_ms
        ):
            current_spec_ms, source = saved_spec_ms, "last run"
        else:
            current_spec_ms, source = config_spec_ms, "config"
        current_spec_ms = max(SPEC_INT_MIN_MS, min(SPEC_INT_MAX_MS, current_spec_ms))
        spectrometer.integration_time_ms = current_spec_ms
        print(f"   Spectrometer: starting at {current_spec_ms:.3f} ms ({source})")

        uv_bit_depth = 8  # RPi5 PiSP outputs 8-bit compressed raw

        uv1_result = None
        uv2_result = None
        spec_result = None
        saturation_retries = 0

        for attempt in range(max_retries + 1):
            print(
                f"   Capture attempt {attempt + 1} "
                f"(UV {filters[1]} nm={cameras.exposure_ms[1]:.3f} ms, "
                f"{filters[2]} nm={cameras.exposure_ms[2]:.3f} ms, "
                f"spec int={current_spec_ms:.3f} ms, "
                f"stack={stack_count})"
            )

            try:
                uv1_result, uv2_result, spec_result = synchronized_capture(
                    cameras,
                    spectrometer,
                    stack_count=stack_count,
                    spec_config={
                        "integration_time_ms": current_spec_ms,
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

            # --- UV saturation check (each camera adjusted independently) ---
            uv_changed = []
            for n, result in ((1, uv1_result), (2, uv2_result)):
                brightness = image_brightness(
                    result.artifacts["image"], bit_depth=uv_bit_depth
                )
                # Scale from the exposure this frame was actually taken at.
                frame_ms = (
                    result.metadata["exposure_ms_actual"]
                    or result.metadata["exposure_ms_requested"]
                )
                new_ms = _next_uv_exposure(frame_ms, brightness)
                if new_ms is None:
                    continue
                old_ms = cameras.exposure_ms[n]
                new_ms = cameras.set_exposure(n, new_ms)
                if new_ms == old_ms:
                    print(
                        f"   WARNING: UV camera_{n} ({filters[n]} nm) at exposure "
                        f"limit {old_ms:.3f} ms, brightness {brightness:.2f}"
                    )
                    continue
                uv_changed.append(n)
                print(
                    f"   UV camera_{n} ({filters[n]} nm) brightness "
                    f"{brightness:.2f} — exposure {old_ms:.3f} → {new_ms:.3f} ms"
                )
            for n in uv_changed:
                cameras.settle_exposure(n)
            uv_needs_adjust = bool(uv_changed)

            # --- Spectrum saturation check ---
            spectrum = spec_result.artifacts["spectrum"]
            spec_sat = check_spectrum_saturation(
                spectrum, spectrometer, min_saturation=0.2, max_saturation=0.8
            )

            spec_needs_adjust = False
            if spec_sat == -1:
                new_ms = max(SPEC_INT_MIN_MS, current_spec_ms * 0.75)
                if new_ms == current_spec_ms:
                    print(
                        "   WARNING: Spectrum saturated at minimum integration — "
                        "saving anyway"
                    )
                else:
                    current_spec_ms = new_ms
                    spectrometer.integration_time_ms = current_spec_ms
                    spec_needs_adjust = True
                    print(
                        f"   Spectrum over-exposed — reducing to "
                        f"{current_spec_ms:.3f} ms"
                    )
            elif spec_sat == 1:
                new_ms = min(SPEC_INT_MAX_MS, current_spec_ms * 1.25)
                if new_ms == current_spec_ms:
                    print(
                        "   WARNING: Spectrum under-exposed at maximum integration — "
                        "saving anyway"
                    )
                else:
                    current_spec_ms = new_ms
                    spectrometer.integration_time_ms = current_spec_ms
                    spec_needs_adjust = True
                    print(
                        f"   Spectrum under-exposed — increasing to "
                        f"{current_spec_ms:.3f} ms"
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
        ch2 = uv2_result.metadata.get("filter_nm", 330)

        # UV images as uncompressed npz
        uv1_fname = build_filename(
            metadata_cfg, timestamp, suffix=f"uv-{ch1}", extension="npz"
        )
        np.savez(
            output_dir / "receive" / uv1_fname,
            image=uv1_result.artifacts["image"],
        )
        print(f"   Saved UV image: {uv1_fname}")

        uv2_fname = build_filename(
            metadata_cfg, timestamp, suffix=f"uv-{ch2}", extension="npz"
        )
        np.savez(
            output_dir / "receive" / uv2_fname,
            image=uv2_result.artifacts["image"],
        )
        print(f"   Saved UV image: {uv2_fname}")

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
            "uv_a_exposure_ms": uv1_result.metadata["exposure_ms_requested"],
            "uv_b_exposure_ms": uv2_result.metadata["exposure_ms_requested"],
            "uv_analogue_gain": cameras.analogue_gain,
            "spectrometer_integration_time_ms": spec_result.metadata[
                "integration_time_ms"
            ],
            "stack_count": stack_count,
            "saturation_retries": saturation_retries,
            "uv_camera_a_filter_nm": ch1,
            "uv_camera_b_filter_nm": ch2,
            "uv_camera_a": uv1_result.metadata,
            "uv_camera_b": uv2_result.metadata,
            "spectrum_mean_intensity": round(float(np.mean(intensities)), 2),
            "spectrum_max_intensity": round(float(np.max(intensities)), 2),
            "files": {
                "uv_a": uv1_fname,
                "uv_b": uv2_fname,
                "spectrum": spec_fname,
            },
        }
        (output_dir / "receive" / meta_fname).write_text(
            json.dumps(meta_payload, indent=2)
        )

        # Persist the converged exposure so the next run starts from here.
        _write_uv_state(
            state_path,
            cameras.exposure_ms,
            config_exposure,
            current_spec_ms,
            config_spec_ms,
            timestamp.isoformat(),
        )

        print("...uv-sync capture complete.")

        summary = {
            "instrument": "uv-sync",
            "timestamp_utc": timestamp.isoformat(),
            "saturation_retries": saturation_retries,
            "files": {
                "uv_a": uv1_fname,
                "uv_b": uv2_fname,
                "spectrum": spec_fname,
                "metadata": meta_fname,
            },
        }
        print(json.dumps(summary))
        sys.exit(0)

    finally:
        spectrometer.shutdown()
        cameras.close()
