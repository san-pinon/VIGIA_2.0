"""
Driver for a USB camera via Video4Linux2 (OpenCV VideoCapture backend).

Suitable for any UVC-compatible USB camera (e.g. Arducam IMX477 in UVC mode).
Does not require picamera2 or libcamera.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure


class Camera:
    """
    Long-lived USB camera controller via V4L2 / OpenCV VideoCapture.

    Expected config shape (``config["visible"]``):

    .. code-block:: toml

        [visible]
        model        = "v4l2"
        camera_port  = 0        # /dev/videoN index
        frame_count  = 1
        framerate    = 1.0
        width        = 4056     # optional — omit to use camera default
        height       = 3040     # optional — omit to use camera default

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        device_index = int(config.get("camera_port", 0))

        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open /dev/video{device_index}. "
                "Check that the device exists and is not in use."
            )

        if "width" in config:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config["width"]))
        if "height" in config:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config["height"]))

        # Discard a few frames so the sensor exposure settles.
        warmup = int(config.get("warmup_frames", 5))
        for _ in range(warmup):
            self._cap.read()

        self._closed = False

    def close(self) -> None:
        """Release the VideoCapture (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.close()

    def __str__(self) -> str:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"V4L2Camera(port={int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))}, {w}x{h})"


def capture(camera: Camera, settings: dict | None = None) -> CaptureResult:
    """
    Capture a single frame from the USB camera.

    Parameters
    ----------
    camera:
        Initialised Camera object.
    settings:
        Unused; present for interface consistency.

    Returns
    -------
    capture_result:
        Metadata contains ``ts_monotonic_ns``, ``shape``, and ``dtype``.
        Artifacts contain ``image`` (H×W×3 uint8 BGR array).

    Raises
    ------
    CaptureFailure:
        If the frame read fails.

    """

    ret, frame = camera._cap.read()
    if not ret or frame is None:
        raise CaptureFailure("V4L2 frame read failed — camera may have disconnected.")

    meta = {
        "ts_monotonic_ns": time.monotonic_ns(),
        "shape": tuple(frame.shape),
        "dtype": str(frame.dtype),
    }

    return CaptureResult(metadata=meta, artifacts={"image": frame})
