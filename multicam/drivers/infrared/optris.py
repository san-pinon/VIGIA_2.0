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
    temp_min_c: float = -10.0,
    temp_max_c: float = 60.0,
) -> np.ndarray:
    """
    Convert raw Optris uint16 thermal data to a false-colour RGB image using a
    fixed temperature range.

    The Optris camera outputs values that can be related to temperature by:

        Temperature = (value - 1000) / 10

    Parameters
    ----------
    frame:
        The image as output by the Optris camera.
    temp_min_c:
        Lower bound of the colour scale in °C.
    temp_max_c:
        Upper bound of the colour scale in °C.

    Returns
    -------
    image:
        The false-colour RGB image.

    """

    raw_min = temp_min_c * 10 + 1000
    raw_max = temp_max_c * 10 + 1000

    clipped = np.clip(frame.astype(np.float64), raw_min, raw_max) - raw_min
    scaled = (255 * clipped / (raw_max - raw_min)).astype(np.uint8)

    image = cv.cvtColor(cv.applyColorMap(scaled, cv.COLORMAP_INFERNO), cv.COLOR_BGR2RGB)

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

    # Wait for a flag cycle then stable flagState==0 (normal operation).
    # Matches the working toughbook capture.py strategy:
    #   1. Poll until we see flagState != 0 (flag/NUC active)
    #   2. Then collect 10 consecutive flagState == 0 frames (stable post-NUC)
    # If the cycle never completes (e.g. flagState stuck at 5 on some firmware
    # versions), fall back to a direct grab of the first valid ret==0 frame.
    flag_seen = False
    stable = 0
    max_wait = 2000  # ~66 s at 30 fps before giving up on the flag cycle

    for _ in range(max_wait):
        ret = LIBIR.evo_irimager_get_thermal_palette_image_metadata(
            camera.thermal_width.value,
            camera.thermal_height.value,
            thermal_data_p,
            camera.palette_width.value,
            camera.palette_height.value,
            image_data_p,
            ctypes.byref(camera.metadata),
        )
        if ret != 0:
            time.sleep(0.002)
            continue
        if camera.metadata.flagState != 0:
            flag_seen = True
            stable = 0
        elif flag_seen:
            stable += 1
            if stable >= 10:
                break

    if stable < 10:
        # Flag cycle did not complete — firmware quirk (e.g. flagState stuck at
        # a non-zero value). Attempt a direct grab of the first valid frame.
        print(
            f"WARNING: flag cycle incomplete (last flagState={camera.metadata.flagState}); "
            "falling back to direct frame grab"
        )
        for _ in range(1000):
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
            time.sleep(0.002)
        else:
            raise CaptureFailure

    print("      ...image captured...")

    image = thermal_data.reshape(
        camera.thermal_height.value, camera.thermal_width.value
    )

    return CaptureResult(metadata=camera.metadata, artifacts={"image": image})
