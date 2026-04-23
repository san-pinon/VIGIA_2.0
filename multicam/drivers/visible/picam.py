"""
Driver for a Raspberry Pi Camera using Picamera2.

Design goals:
- Daemon-ready (stable lifecycle, predictable exceptions)
- Driver does device IO only.
- Consistent capture interface.

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


class Camera:
    """
    Long-lived camera controller.

    Expected config shape:

    {
        "camera_port": 0,
        "controls": {"ExposureTime": 10000, "AnalogueGain": 1.0, ...},
        "start_delay_s": 0.2,
    }

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if PiCamera is None:
            raise RuntimeError(
                "picamera2 is not installed; cannot use Raspberry Pi Camera driver."
            ) from _PICAMERA2_IMPORT_ERROR

        if "camera_port" not in config:
            raise ValueError("PiCamera config missing required key: 'camera_port'")

        self._config = dict(config)
        self._closed = False

        try:
            self.camera = PiCamera(config["camera_port"])
            controls = config.get("controls", {})
            if controls:
                self.camera.set_controls(dict(controls))
            self.camera.start()

            start_delay = float(self._config.get("start_delay_s", 0.0))
            if start_delay > 0:
                time.sleep(start_delay)
        except Exception as e:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(f"Failed to intialise PiCamera: {e}") from e

    def close(self) -> None:
        """Close camera resources (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self.camera.close()
        except Exception:
            pass

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.close()

    def __str__(self) -> str:
        props = getattr(self.camera, "camera_properties", None)
        if isinstance(props, dict):
            model = props.get("Model", "PiCamera")
            return f"{model} (port={self._config.get('camera_port')})"
        return f"PiCamera(port={self._config.get('camera_port')})"


def capture(camera: Camera, settings: dict | None = None) -> CaptureResult:
    """
    Capture a frame.

    Parameters
    ----------
    camera:
        Initialised camera object.
    settings:
        Local overrides for camera settings.

    Returns
    -------
    capture_result:
        Object containing metadata and artifacts.

    """

    try:
        image = camera.camera.capture_array()
    except Exception as e:
        raise CaptureFailure(f"PiCamera capture failed: {e}") from e

    meta = {
        "ts_monotonic_ns": time.monotonic_ns(),
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
    }

    return CaptureResult(metadata=meta, artifacts={"image": image})


def check_image_saturation(
    image: np.ndarray,
    min_saturation: float = 0.5,
    max_saturation: float = 0.8,
    bit_depth: int = 10,
    pixel_count: int = 100,
    rows: tuple[int, int] | None = None,
) -> int:
    """
    Check whether an image is over-/under-exposed.

    An average of the top N pixel values is used in order to avoid broken pixels
    causing erroneous estimates of the exposure. A specific set of rows may be chosen
    in order to ensure only sections of the image that represent the sky are used.

    Parameters
    ----------
    image:
        The image to be analysed.
    min_saturation:
        The threshold below which the image is considered under-exposed.
    max_saturation:
        The threshold above which the image is considered over-exposed.
    bit_depth:
        The number of bits used to digitise the values reported by the sensor.
    pixel_count:
        The number of pixels to use to compute the average pixel value.
    rows:
        The range of rows in the image used to compute the average pixel value.

    Returns
    -------
    saturation_flag:
        An integer flag indicated whether the image is under-exposed (1), adequately
        exposed (0), or over-exposed (-1).

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
