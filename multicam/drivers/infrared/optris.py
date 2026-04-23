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
        self._closed = False
        self.close()
        self._closed = False

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


def _convert_temp2image(frame: np.ndarray) -> np.ndarray:
    """
    Utility function to convert from the outputs values from Optris camera to values
    in the range 0-255.

    The Optris camera outputs values that can be related to temperature by:

        Temperature = (value - 1000) / 10

    Parameters
    ----------
    frame:
        The image as output by the Optris camera.

    Returns
    -------
    image:
        The converted image.

    """

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(frame)
    min_val, max_val = [np.around((val - 1000) / 10, 1) for val in [min_val, max_val]]

    frame_mean, frame_std = frame.mean(), frame.std()
    image_min = int(max([frame.min(), frame_mean - 5 * frame_std]))
    image_max = int(min([frame.max(), frame_mean + 5 * frame_std]))

    clipped_image = np.clip(frame, image_min, image_max) - image_min

    image = (255 * (clipped_image / (image_max - image_min))).astype(np.uint8)

    image = cv.cvtColor(cv.applyColorMap(image, cv.COLORMAP_INFERNO), cv.COLOR_BGR2RGB)

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

    # Capture an image
    count = 0
    LIBIR.evo_irimager_trigger_shutter_flag()
    while count < 1000:
        ret = LIBIR.evo_irimager_get_thermal_palette_image_metadata(
            camera.thermal_width,
            camera.thermal_height,
            thermal_data_p,
            camera.palette_width,
            camera.palette_height,
            image_data_p,
            ctypes.byref(camera.metadata),
        )
        if ret != 0:
            print(f"error on evo_irimager_get_thermal_palette_image: {ret}")
            count += 1
            time.sleep(0.002)
            continue

        if camera.metadata.flagState == 3:
            print("      ...image captured...")
            break

        count += 1

    if count == 1000:
        raise CaptureFailure

    image = thermal_data.reshape(
        camera.thermal_height.value, camera.thermal_width.value
    )

    return CaptureResult(metadata=camera.metadata, artifacts={"image": image})
