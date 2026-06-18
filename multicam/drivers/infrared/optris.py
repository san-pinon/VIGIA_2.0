"""
Driver for an Optris model IR camera using the 'irdirectsdk' framework.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import ctypes
import time
from collections.abc import Mapping
from ctypes.util import find_library
from typing import Any

import cv2 as cv
import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

libname = find_library("irdirectsdk")
if not libname:
    raise RuntimeError("Could not find irdirectsdk via find_library()")
try:
    LIBIR = ctypes.CDLL(libname)
except OSError as e:
    raise RuntimeError(f"Could not load irdirectsdk ({libname})") from e

# TSTAMP_FACTOR = 10_000_000  # Convert Optris software timestamp


class EvoIRFrameMetadata(ctypes.Structure):
    _fields_ = [
        ("counter", ctypes.c_uint),
        ("counterHW", ctypes.c_uint),
        ("timestamp", ctypes.c_longlong),  # / TSTAMP_FACTOR
        ("timestampMedia", ctypes.c_longlong),
        ("flagState", ctypes.c_int),
        ("tempChip", ctypes.c_float),
        ("tempFlag", ctypes.c_float),
        ("tempBox", ctypes.c_float),
    ]


# --- Declare ctypes prototypes so argument marshalling is correct on x86_64 ---
LIBIR.evo_irimager_usb_init.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
LIBIR.evo_irimager_usb_init.restype = ctypes.c_int

LIBIR.evo_irimager_terminate.argtypes = []
LIBIR.evo_irimager_terminate.restype = ctypes.c_int

LIBIR.evo_irimager_get_thermal_image_size.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
LIBIR.evo_irimager_get_thermal_image_size.restype = ctypes.c_int

LIBIR.evo_irimager_get_palette_image_size.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
LIBIR.evo_irimager_get_palette_image_size.restype = ctypes.c_int

LIBIR.evo_irimager_trigger_shutter_flag.argtypes = []
LIBIR.evo_irimager_trigger_shutter_flag.restype = ctypes.c_int

LIBIR.evo_irimager_get_thermal_palette_image_metadata.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ushort),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(EvoIRFrameMetadata),
]
LIBIR.evo_irimager_get_thermal_palette_image_metadata.restype = ctypes.c_int


class Camera:
    """
    Long-lived camera controller.

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if "xml_path" not in config:
            raise ValueError("PiCamera config missing required key: 'xml_path'")
        xml_path = ctypes.c_char_p(config["xml_path"].encode("UTF-8"))
        formats_def_path = ctypes.c_char_p()
        if "log_path" not in config:
            raise ValueError("PiCamera config missing required key: 'log_path'")
        log_path = ctypes.c_char_p(config["log_path"].encode("UTF-8"))

        self.metadata = EvoIRFrameMetadata()

        # Terminate any stale irdirectsdk session before initialising.
        # Use the C call directly — self.close() guards on _closed which
        # is not meaningful before init; terminate() on a clean process is
        # documented as a safe no-op but we guard anyway.
        self._closed = False
        try:
            LIBIR.evo_irimager_terminate()
        except Exception:
            pass

        ret = LIBIR.evo_irimager_usb_init(xml_path, formats_def_path, log_path)
        if ret != 0:
            raise RuntimeError(
                "Could not initialise USB connection to Optris camera, "
                f"return code {ret}. Exiting."
            )

        # Get thermal image dimensions and initialise data container
        self.thermal_width, self.thermal_height = ctypes.c_int(), ctypes.c_int()
        LIBIR.evo_irimager_get_thermal_image_size(
            ctypes.byref(self.thermal_width), ctypes.byref(self.thermal_height)
        )

        # Get image palette dimensions and initialise data container
        # Note: dimensions can differ from thermal image dimensions due to striding
        self.palette_width, self.palette_height = ctypes.c_int(), ctypes.c_int()
        LIBIR.evo_irimager_get_palette_image_size(
            ctypes.byref(self.palette_width), ctypes.byref(self.palette_height)
        )

        self._config = dict(config)

    def close(self) -> None:
        """Close camera resources (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            LIBIR.evo_irimager_terminate()
        except Exception:
            pass

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.close()

    def __str__(self) -> str:
        return (
            "Thermal image dimensions: "
            f"{self.thermal_width.value} x {self.thermal_height.value} px\n"
            "Image palette dimensions: "
            f"{self.palette_width.value} x {self.palette_height.value} px"
        )


def _convert_temp2image(
    frame: np.ndarray,
    temp_min_c: float | None = None,
    temp_max_c: float | None = None,
) -> np.ndarray:
    """
    Convert raw Optris uint16 thermal data to a false-colour RGB image.

    The Optris camera outputs values that can be related to temperature by:

        Temperature = (value - 1000) / 10

    Parameters
    ----------
    frame:
        The image as output by the Optris camera.
    temp_min_c:
        Lower bound of the colour scale in °C.  If ``None`` (default), the
        2nd percentile of the scene is used (auto-scale).
    temp_max_c:
        Upper bound of the colour scale in °C.  If ``None`` (default), the
        98th percentile of the scene is used (auto-scale).

    Returns
    -------
    image:
        The false-colour RGB image.

    """

    temps = (frame.astype(np.float64) - 1000.0) / 10.0

    lo = float(np.percentile(temps, 2)) if temp_min_c is None else temp_min_c
    hi = float(np.percentile(temps, 98)) if temp_max_c is None else temp_max_c

    if hi <= lo:
        hi = lo + 1.0  # avoid division by zero on a perfectly flat frame

    scaled = np.clip((temps - lo) / (hi - lo), 0.0, 1.0)
    gray8 = (scaled * 255).astype(np.uint8)

    image = cv.cvtColor(cv.applyColorMap(gray8, cv.COLORMAP_INFERNO), cv.COLOR_BGR2RGB)

    return image


def capture(camera: Camera) -> CaptureResult:
    """
    Capture an image from the Optris IR camera.

    Parameters
    ----------
    camera:
        A simple object that provides an instantiated camera object.

    Returns
    -------
    capture_result:
        Object containing metadata and artifacts.

    """

    thermal_data = np.zeros(
        [camera.thermal_width.value * camera.thermal_height.value], dtype=np.uint16
    )
    thermal_data_p = thermal_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))

    image_data = np.zeros(
        [camera.palette_width.value * camera.palette_height.value * 3], dtype=np.uint8
    )
    image_data_p = image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # Trigger a NUC (non-uniformity correction) flag cycle so the shutter
    # opens fresh and delivers real scene data at flagState==0.
    # Without this, the SDK stays at flagState==5 indefinitely.
    LIBIR.evo_irimager_trigger_shutter_flag()

    # Wait for flagState==0 (shutter open, scene data valid).
    # Flag cycle takes ~9 s on libirimager 8.9.0 arm64; allow up to 15 s.
    for _ in range(3000):
        ret = LIBIR.evo_irimager_get_thermal_palette_image_metadata(
            camera.thermal_width.value,
            camera.thermal_height.value,
            thermal_data_p,
            camera.palette_width.value,
            camera.palette_height.value,
            image_data_p,
            ctypes.byref(camera.metadata),
        )
        if ret == 0 and camera.metadata.flagState == 0:
            break
        time.sleep(0.005)
    else:
        raise CaptureFailure

    print(
        f"      ...image captured (flagState={camera.metadata.flagState}, chipT={camera.metadata.tempChip:.1f}C)..."
    )

    image = thermal_data.reshape(
        camera.thermal_height.value, camera.thermal_width.value
    )

    return CaptureResult(metadata=camera.metadata, artifacts={"image": image})


def trigger_nuc(camera: Camera, timeout_s: float = 15.0) -> float:
    """
    Trigger a NUC (non-uniformity correction) shutter flag cycle and wait for
    the shutter to reopen (``flagState==0``).

    Used by the video driver to (re)calibrate at the start of, or periodically
    during, a continuous recording. The scene is frozen while the shutter is
    closed, so the caller decides how often to pay this cost.

    Parameters
    ----------
    camera:
        An instantiated camera object.
    timeout_s:
        Maximum time to wait for the flag cycle to complete.

    Returns
    -------
    elapsed_s:
        How long the NUC cycle took, in seconds, so the caller can log/tune it.

    """

    thermal_data = np.zeros(
        [camera.thermal_width.value * camera.thermal_height.value], dtype=np.uint16
    )
    thermal_data_p = thermal_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    image_data = np.zeros(
        [camera.palette_width.value * camera.palette_height.value * 3], dtype=np.uint8
    )
    image_data_p = image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    start = time.monotonic()
    LIBIR.evo_irimager_trigger_shutter_flag()
    while time.monotonic() - start < timeout_s:
        ret = LIBIR.evo_irimager_get_thermal_palette_image_metadata(
            camera.thermal_width.value,
            camera.thermal_height.value,
            thermal_data_p,
            camera.palette_width.value,
            camera.palette_height.value,
            image_data_p,
            ctypes.byref(camera.metadata),
        )
        if ret == 0 and camera.metadata.flagState == 0:
            break
        time.sleep(0.005)
    else:
        raise CaptureFailure

    return time.monotonic() - start


def capture_stream_frame(camera: Camera, retries: int = 200) -> CaptureResult:
    """
    Grab the freshest thermal frame for continuous (video) capture.

    Unlike :func:`capture`, this does **not** trigger a shutter flag or wait for
    ``flagState==0`` — that would freeze the scene on every frame. It simply
    pulls the latest frame the SDK has, retrying only on a transient non-zero
    return code. NUC scheduling is left to the driver loop.

    Parameters
    ----------
    camera:
        An instantiated camera object.
    retries:
        Maximum number of pulls to attempt while the SDK returns a non-zero
        (frame-not-ready) code, at 5 ms spacing.

    Returns
    -------
    capture_result:
        Object containing metadata and the uint16 thermal frame.

    """

    thermal_data = np.zeros(
        [camera.thermal_width.value * camera.thermal_height.value], dtype=np.uint16
    )
    thermal_data_p = thermal_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort))
    image_data = np.zeros(
        [camera.palette_width.value * camera.palette_height.value * 3], dtype=np.uint8
    )
    image_data_p = image_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    for _ in range(retries):
        ret = LIBIR.evo_irimager_get_thermal_palette_image_metadata(
            camera.thermal_width.value,
            camera.thermal_height.value,
            thermal_data_p,
            camera.palette_width.value,
            camera.palette_height.value,
            image_data_p,
            ctypes.byref(camera.metadata),
        )
        if ret == 0:
            break
        time.sleep(0.005)
    else:
        raise CaptureFailure

    image = thermal_data.reshape(
        camera.thermal_height.value, camera.thermal_width.value
    )

    return CaptureResult(metadata=camera.metadata, artifacts={"image": image})
