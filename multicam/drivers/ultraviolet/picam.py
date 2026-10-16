"""
Driver for the UV dual-camera system using two Raspberry Pi cameras via Picamera2.

Each camera sits behind a narrow-band UV filter (310 nm and 330 nm) for SO₂
column-amount retrieval using the dual-band ratio technique.

Images are taken from the sensor's **raw** stream (10-bit Bayer, no ISP
processing), so pixel values are linear in light — required for absorbance.
They can be binned on capture to cut noise and file size: each output pixel is
the **sum** of an N×N block of raw pixels, kept as uint16 (exact; divide by
N² for the per-pixel mean).

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

# OV5647 full-resolution sensor mode and its raw output.
SENSOR_SIZE = (2592, 1944)  # (width, height)
RAW_BIT_DEPTH = 10
RAW_MAX_DN = 2**RAW_BIT_DEPTH - 1
BAYER_ORDER = "GBRG"
DEFAULT_RAW_FORMAT = "SGBRG10"  # unpacked: one 16-bit container per pixel
DEFAULT_BLACK_LEVEL_DN = 16.0  # OV5647, 10-bit scale; used if metadata lacks it
# Raw pixels at or above this count as clipped.
SATURATED_DN = RAW_MAX_DN - 3
# The ISP-processed main stream is not used for data; keep it small.
_MAIN_STREAM = {"format": "YUV420", "size": (640, 480)}
_VALID_BINNING = (1, 2, 4, 8)


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


def unpack_raw(
    buffer: np.ndarray,
    fmt: str,
    width: int = SENSOR_SIZE[0],
    height: int = SENSOR_SIZE[1],
) -> np.ndarray:
    """
    Turn a raw-stream buffer into a ``(height, width)`` uint16 Bayer mosaic of
    10-bit values (0–1023).

    Handles MIPI CSI-2 packed 10-bit (``*_CSI2P``: 4 pixels in 5 bytes, the
    5th byte holding the 2 LSBs of each) and unpacked 16-bit containers in
    either alignment (LSB-aligned 0–1023, or MSB-aligned ×64). Compressed
    PiSP formats are rejected — they are not linear.
    """

    if "PISP_COMP" in fmt:
        raise CaptureFailure(
            f"Raw format {fmt} is compressed (not linear); set "
            f"[ultraviolet] raw_format = '{DEFAULT_RAW_FORMAT}'"
        )

    rows = np.ascontiguousarray(buffer).view(np.uint8).reshape(height, -1)

    if fmt.endswith("_CSI2P"):
        packed = rows[:, : width * 5 // 4].reshape(height, width // 4, 5)
        msb = packed[:, :, :4].astype(np.uint16) << 2
        lsb = packed[:, :, 4:5].astype(np.uint16)
        shifts = np.arange(4, dtype=np.uint16) * 2
        mosaic = msb | ((lsb >> shifts) & 0b11)
        return mosaic.reshape(height, width)

    mosaic = rows[:, : width * 2].copy().view("<u2").reshape(height, width)
    # LSB-aligned 10-bit data never exceeds 1023 (black level alone is ~16);
    # MSB-aligned data puts even black at ~1024, so this test is unambiguous.
    if mosaic.max() > RAW_MAX_DN:
        mosaic >>= 16 - RAW_BIT_DEPTH
    return mosaic


def black_level_dn(frame_metadata: Mapping[str, Any]) -> float:
    """Sensor black level on the 10-bit scale, from libcamera frame metadata."""

    levels = frame_metadata.get("SensorBlackLevels")
    if not levels:
        return DEFAULT_BLACK_LEVEL_DN
    # libcamera reports black levels on a 16-bit scale.
    return float(np.mean(levels)) / 2 ** (16 - RAW_BIT_DEPTH)


def bin_sum(mosaic: np.ndarray, factor: int) -> np.ndarray:
    """
    Sum ``factor``×``factor`` blocks of raw pixels into uint16 (exact: up to
    8×8 blocks of 10-bit values fit, 64 × 1023 = 65472). Even factors merge
    whole 2×2 Bayer cells. ``factor = 1`` returns the mosaic unchanged.
    """

    if factor == 1:
        return mosaic
    h = mosaic.shape[0] - mosaic.shape[0] % factor
    w = mosaic.shape[1] - mosaic.shape[1] % factor
    blocks = mosaic[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return blocks.sum(axis=(1, 3), dtype=np.uint32).astype(np.uint16)


def bin_mosaic(mosaic: np.ndarray, factor: int) -> np.ndarray:
    """
    Block-average ``factor``×``factor`` pixels (float32). Even factors merge
    whole 2×2 Bayer cells, so every output pixel mixes all four filter sites.
    ``factor = 1`` returns the uint16 mosaic unchanged.
    """

    if factor == 1:
        return mosaic
    h = mosaic.shape[0] - mosaic.shape[0] % factor
    w = mosaic.shape[1] - mosaic.shape[1] % factor
    blocks = mosaic[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


class DualCamera:
    """
    Long-lived controller for the UV dual-camera system.

    Each camera gets its own exposure time (the 310 nm and 330 nm filters
    pass very different amounts of light), while analogue gain is locked
    to a fixed value (default 1.0) with auto-exposure/auto-gain disabled,
    so pixel values scale only with exposure time. Frames come from the
    10-bit raw stream.

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
        camera_1_exposure_ms = 1400
        camera_2_exposure_ms = 100
        max_exposure_ms      = 2000    # optional, auto-adjust ceiling
        analogue_gain        = 1.0     # optional, default 1.0
        raw_format           = "SGBRG10"  # optional; or "SGBRG10_CSI2P"
        binning              = 4       # optional: 1, 2, 4 or 8
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
        self.binning = int(config.get("binning", 1))
        if self.binning not in _VALID_BINNING:
            raise ValueError(
                f"[ultraviolet] binning must be one of {_VALID_BINNING}, "
                f"got {self.binning}"
            )
        requested_raw = str(config.get("raw_format", DEFAULT_RAW_FORMAT))
        self.raw_format: dict[int, str] = {}

        try:
            self.camera_1 = PiCamera(config["camera_1_port"])
            self.camera_2 = PiCamera(config["camera_2_port"])

            for n in (1, 2):
                cam = self.camera(n)
                cam.configure(
                    cam.create_still_configuration(
                        main=_MAIN_STREAM,
                        raw={"format": requested_raw, "size": SENSOR_SIZE},
                        buffer_count=2,
                    )
                )
                # Record what libcamera actually granted (it may substitute).
                self.raw_format[n] = str(cam.camera_configuration()["raw"]["format"])

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
        # Older libcamera uses AeEnable; newer releases use per-control modes
        # (1 = manual). v0.5 on the RPi5 exposes both. Set whatever is present.
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
        """
        Capture one raw frame from camera ``n``.

        The artifact ``image`` is the raw mosaic as uint16; when binned, each
        pixel is the *sum* of a ``binning``×``binning`` block (divide by
        ``binning**2`` for the per-pixel mean). Black level is *not*
        subtracted; ``black_level_dn`` (per raw pixel) in the metadata gives it.
        ``saturated_fraction`` is measured on the unbinned raw pixels.
        """

        try:
            request = self.camera(n).capture_request()
            try:
                raw = request.make_array("raw")
                frame_md = request.get_metadata()
            finally:
                request.release()
            mosaic = unpack_raw(raw, self.raw_format[n])
        except CaptureFailure:
            raise
        except Exception as e:
            raise CaptureFailure(f"UV camera_{n} capture failed: {e}") from e

        ts = time.monotonic_ns()
        saturated_fraction = (
            float(np.count_nonzero(mosaic >= SATURATED_DN)) / mosaic.size
        )
        image = bin_sum(mosaic, self.binning)
        return CaptureResult(
            metadata={
                "ts_monotonic_ns": ts,
                "port": self._config.get(f"camera_{n}_port"),
                "filter_nm": self._config.get(
                    f"camera_{n}_filter_nm", 310 if n == 1 else 330
                ),
                "stream": "raw",
                "raw_format": self.raw_format[n],
                "bayer_order": BAYER_ORDER,
                "bit_depth": RAW_BIT_DEPTH,
                "binning": self.binning,
                "pixel_value": (
                    "raw DN" if self.binning == 1 else "sum of binning² raw DN"
                ),
                "black_level_dn": round(black_level_dn(frame_md), 3),
                "saturated_fraction": round(saturated_fraction, 6),
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
            f"exposure_ms={self.exposure_ms}, gain={self.analogue_gain}, "
            f"binning={self.binning})"
        )


def capture(
    cameras: DualCamera, settings: dict | None = None
) -> tuple[CaptureResult, CaptureResult]:
    """
    Capture one raw frame from each UV camera.

    The two captures are not hardware-synchronised; they happen as quickly
    as consecutive captures allow. Each result carries the sensor-reported
    exposure, gain, black level and timestamp of its frame.

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


def raw_brightness(result: CaptureResult, percentile: float = 99.9) -> float:
    """
    Exposure level of a raw capture: the ``percentile`` pixel value above the
    black level, as a fraction of the usable range (black level → 1023).

    A high percentile rather than the absolute maximum, so a few hot pixels
    (raw data has no defect correction) don't drive the exposure.
    """

    black = float(result.metadata["black_level_dn"])
    blocks = int(result.metadata.get("binning", 1)) ** 2
    value = float(np.percentile(result.artifacts["image"], percentile)) / blocks
    return max(0.0, (value - black) / (RAW_MAX_DN - black))


def quicklook(
    image: np.ndarray, black: float, binning: int = 1, max_width: int = 648
) -> np.ndarray:
    """
    8-bit preview of a raw UV image, for viewing only (not for analysis).

    Fixed scale (black level → full scale) with a display gamma, so frames
    stay comparable: a dim preview means a genuinely under-exposed image.
    Downsampled by block-averaging to at most ``max_width`` pixels wide.
    ``binning`` converts summed pixels back to per-raw-pixel values.
    """

    image = image.astype(np.float32) / binning**2

    factor = 1
    while image.shape[1] // factor > max_width:
        factor *= 2
    small = bin_mosaic(image, factor) if factor > 1 else image
    level = np.clip((small - black) / (RAW_MAX_DN - black), 0.0, 1.0)
    return np.round(255.0 * level ** (1 / 2.2)).astype(np.uint8)


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
