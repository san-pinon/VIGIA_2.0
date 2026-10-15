"""
Driver for the UV dual-camera system using two Raspberry Pi cameras via Picamera2.

Each camera sits behind a narrow-band UV filter (310 nm and 330 nm) for SO₂
column-amount retrieval using the dual-band ratio technique.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

try:
    from picamera2 import Picamera2 as PiCamera
except ModuleNotFoundError as e:
    PiCamera = None
    _PICAMERA2_IMPORT_ERROR = e
else:
    _PICAMERA2_IMPORT_ERROR = None


DEFAULT_EXPOSURE_MS = 50.0
MIN_EXPOSURE_MS = 0.1
DEFAULT_MAX_EXPOSURE_MS = 1000.0

# Controls DualCamera sets itself; ignored in [ultraviolet.controls].
_MANAGED_CONTROLS = {
    "ExposureTime",
    "AnalogueGain",
    "AeEnable",
    "ExposureTimeMode",
    "AnalogueGainMode",
}

# Width/height of the Y plane in the YUV420 still stream.
_IMAGE_SHAPE = (1944, 2592)


def _ms_to_us(exposure_ms: float) -> int:
    return max(1, int(round(exposure_ms * 1000)))


def configured_exposure_ms(config: Mapping[str, Any], n: int) -> float:
    """
    Starting exposure (ms) for UV camera ``n`` (1 or 2) from the config.

    Uses ``camera_<n>_exposure_ms``. Falls back to a legacy shared
    ``[ultraviolet.controls] ExposureTime`` (µs) so old configs still load.
    """

    key = f"camera_{n}_exposure_ms"
    if key in config:
        return float(config[key])
    legacy_us = config.get("controls", {}).get("ExposureTime")
    if legacy_us is not None:
        return float(legacy_us) / 1000.0
    return DEFAULT_EXPOSURE_MS


class DualCamera:
    """
    Long-lived controller for the UV dual-camera system.

    Each camera gets its own exposure time (the 310 nm and 330 nm filters
    pass very different amounts of light), while analogue gain is locked
    to a fixed value (default 1.0) with auto-exposure/auto-gain disabled,
    so pixel values scale only with exposure time.

    Expected config shape (``config["ultraviolet"]``):

    .. code-block:: toml

        [ultraviolet]
        model          = "picam"
        camera_1_port  = 0      # port 0 → 310 nm (field-confirmed)
        camera_2_port  = 1
        framerate      = 0.5
        frame_count    = 1
        camera_1_filter_nm = 310   # optional, default 310
        camera_2_filter_nm = 330   # optional, default 330
        camera_1_exposure_ms = 600.0
        camera_2_exposure_ms = 65.0
        max_exposure_ms      = 1000.0  # optional, auto-adjust ceiling
        analogue_gain        = 1.0     # optional, default 1.0
        start_delay_s  = 0.5       # optional warm-up delay after start()

        [ultraviolet.controls]
        # Extra libcamera control overrides applied to both cameras.
        # Exposure/gain controls here are ignored (set above instead).

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if PiCamera is None:
            raise RuntimeError(
                "picamera2 is not installed; cannot use UV camera driver."
            ) from _PICAMERA2_IMPORT_ERROR

        for key in ("camera_1_port", "camera_2_port"):
            if key not in config:
                raise ValueError(f"UV camera config missing required key: '{key}'")

        self._config = dict(config)
        self._closed = False
        self.camera_1: PiCamera | None = None
        self.camera_2: PiCamera | None = None

        self.analogue_gain = float(config.get("analogue_gain", 1.0))
        self.max_exposure_ms = float(
            config.get("max_exposure_ms", DEFAULT_MAX_EXPOSURE_MS)
        )
        self.exposure_ms = {
            n: self.clamp_exposure_ms(configured_exposure_ms(config, n)) for n in (1, 2)
        }

        try:
            self.camera_1 = PiCamera(config["camera_1_port"])
            self.camera_2 = PiCamera(config["camera_2_port"])

            still_cfg = {"format": "YUV420", "size": (2592, 1944)}
            self.camera_1.configure(
                self.camera_1.create_still_configuration(main=still_cfg)
            )
            self.camera_2.configure(
                self.camera_2.create_still_configuration(main=still_cfg)
            )

            # Exposure and gain are owned by this class; drop them from the
            # free-form overrides so they can't fight the manual settings.
            extra_controls = {
                k: v
                for k, v in config.get("controls", {}).items()
                if k not in _MANAGED_CONTROLS
            }
            for n in (1, 2):
                cam = self.camera(n)
                cam.set_controls(
                    {
                        **extra_controls,
                        **self._manual_controls(cam, self.exposure_ms[n]),
                    }
                )

            self.camera_1.start()
            self.camera_2.start()

            start_delay = float(config.get("start_delay_s", 0.0))
            if start_delay > 0:
                time.sleep(start_delay)
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialise UV camera pair: {e}") from e

    def camera(self, n: int) -> PiCamera:
        """Return the Picamera2 instance for camera ``n`` (1 or 2)."""

        return self.camera_1 if n == 1 else self.camera_2

    def clamp_exposure_ms(self, exposure_ms: float) -> float:
        return max(MIN_EXPOSURE_MS, min(self.max_exposure_ms, float(exposure_ms)))

    def _manual_controls(self, cam: PiCamera, exposure_ms: float) -> dict:
        """Controls for a fixed exposure and fixed analogue gain (AE/AGC off)."""

        controls = {
            "ExposureTime": _ms_to_us(exposure_ms),
            "AnalogueGain": self.analogue_gain,
        }
        available = cam.camera_controls
        # Older libcamera (e.g. v0.1 on the RPi5) uses AeEnable; newer releases
        # use per-control modes (1 = manual). Set whichever this build exposes.
        if "AeEnable" in available:
            controls["AeEnable"] = False
        if "ExposureTimeMode" in available:
            controls["ExposureTimeMode"] = 1
        if "AnalogueGainMode" in available:
            controls["AnalogueGainMode"] = 1
        return controls

    def set_exposure(self, n: int, exposure_ms: float) -> float:
        """Set camera ``n`` to a new exposure (ms); returns the clamped value."""

        exposure_ms = self.clamp_exposure_ms(exposure_ms)
        cam = self.camera(n)
        cam.set_controls(self._manual_controls(cam, exposure_ms))
        self.exposure_ms[n] = exposure_ms
        return exposure_ms

    def settle_exposure(
        self, n: int, max_frames: int = 6, tolerance: float = 0.05
    ) -> dict:
        """
        Discard frames from camera ``n`` until its reported exposure and
        analogue gain match the requested ones (new controls take a few frames
        to apply).

        Returns the metadata of the last frame read.
        """

        target_us = _ms_to_us(self.exposure_ms[n])
        metadata = {}
        for _ in range(max_frames):
            metadata = self.camera(n).capture_metadata()
            actual_us = metadata.get("ExposureTime", 0)
            actual_gain = metadata.get("AnalogueGain", 0.0)
            if (
                abs(actual_us - target_us) <= tolerance * target_us
                and abs(actual_gain - self.analogue_gain)
                <= tolerance * self.analogue_gain
            ):
                return metadata
        print(
            f"   WARNING: UV camera_{n} did not settle: requested "
            f"{target_us / 1000:.3f} ms / gain {self.analogue_gain}, sensor reports "
            f"{metadata.get('ExposureTime', 0) / 1000:.3f} ms / "
            f"gain {metadata.get('AnalogueGain', 0.0):.3f}"
        )
        return metadata

    def capture_one(self, n: int) -> CaptureResult:
        """Capture one frame from camera ``n`` with its per-frame sensor metadata."""

        try:
            request = self.camera(n).capture_request()
            try:
                yuv = request.make_array("main")
                frame_md = request.get_metadata()
            finally:
                request.release()
        except Exception as e:
            raise CaptureFailure(f"UV camera_{n} capture failed: {e}") from e

        ts = time.monotonic_ns()
        image = yuv[: _IMAGE_SHAPE[0], : _IMAGE_SHAPE[1]]  # Y plane (grayscale)
        return CaptureResult(
            metadata={
                "ts_monotonic_ns": ts,
                "port": self._config.get(f"camera_{n}_port"),
                "filter_nm": self._config.get(
                    f"camera_{n}_filter_nm", 310 if n == 1 else 330
                ),
                "stream": "main",
                "bit_depth": 8,
                "shape": tuple(image.shape),
                "dtype": str(image.dtype),
                "exposure_ms_requested": round(self.exposure_ms[n], 3),
                "exposure_ms_actual": round(frame_md.get("ExposureTime", 0) / 1000, 3),
                "analogue_gain_requested": self.analogue_gain,
                "analogue_gain_actual": round(frame_md.get("AnalogueGain", 0.0), 4),
                "digital_gain_actual": round(frame_md.get("DigitalGain", 0.0), 4),
                "sensor_timestamp_ns": frame_md.get("SensorTimestamp"),
            },
            artifacts={"image": image},
        )

    def close(self) -> None:
        """Close both cameras (idempotent)."""

        if self._closed:
            return
        self._closed = True
        for cam in (self.camera_1, self.camera_2):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass

    def __enter__(self) -> DualCamera:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __str__(self) -> str:
        return (
            f"DualCamera("
            f"port_1={self._config.get('camera_1_port')}, "
            f"port_2={self._config.get('camera_2_port')}, "
            f"exposure_ms={self.exposure_ms}, gain={self.analogue_gain})"
        )


def capture(
    cameras: DualCamera, settings: dict | None = None
) -> tuple[CaptureResult, CaptureResult]:
    """
    Capture one frame from each UV camera.

    The two captures are not hardware-synchronised; they happen as quickly
    as consecutive captures allow. Each result carries the sensor-reported
    exposure, gain and timestamp of its frame.

    Parameters
    ----------
    cameras:
        Initialised ``DualCamera`` object.
    settings:
        Unused; present for interface consistency.

    Returns
    -------
    result_1, result_2:
        ``CaptureResult`` for camera_1 and camera_2 respectively.

    Raises
    ------
    CaptureFailure:
        If either camera fails to produce a frame.

    """

    return cameras.capture_one(1), cameras.capture_one(2)


def image_brightness(
    image: np.ndarray, bit_depth: int = 8, pixel_count: int = 100
) -> float:
    """Mean of the ``pixel_count`` brightest pixels as a fraction of full scale."""

    flat = image.ravel()
    brightest = np.partition(flat, -pixel_count)[-pixel_count:]
    return float(np.mean(brightest)) / (2**bit_depth - 1)


def check_image_saturation(
    image: np.ndarray,
    min_saturation: float = 0.5,
    max_saturation: float = 0.8,
    bit_depth: int = 8,
    pixel_count: int = 100,
    rows: tuple[int, int] | None = None,
) -> int:
    """
    Check whether a UV image is over-/under-exposed.

    Parameters
    ----------
    image:
        The image to be analysed.
    min_saturation:
        Fraction below which the image is considered under-exposed.
    max_saturation:
        Fraction above which the image is considered over-exposed.
    bit_depth:
        ADC bit depth of the sensor (default 10 for RPi camera v2/HQ).
    pixel_count:
        Number of brightest pixels used to compute the average DN.
    rows:
        ``(start_row, end_row)`` slice to restrict the assessment to sky
        pixels only. If ``None``, the full image is used.

    Returns
    -------
    saturation_flag:
        ``1``  — under-exposed,
        ``0``  — adequately exposed,
        ``-1`` — over-exposed.

    """

    if rows is None:
        cropped_image = image.ravel()
    else:
        cropped_image = image[rows[0] : rows[1], :].ravel()

    average_DN = np.mean(cropped_image[cropped_image.argsort()[-pixel_count:]])

    max_DN = 2**bit_depth - 1
    saturation = average_DN / max_DN

    if saturation > max_saturation:
        return -1
    elif saturation < min_saturation:
        return 1
    else:
        return 0
