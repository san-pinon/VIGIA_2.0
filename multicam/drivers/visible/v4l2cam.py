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

import subprocess
import time
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure


def _query_v4l2_card_name(device_index: int) -> str | None:
    """Return the V4L2 'Card type' string for /dev/videoN, or None on failure."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", f"--device=/dev/video{device_index}", "--info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in out.stdout.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _find_device_by_card_name(expected_name: str) -> int | None:
    """Scan /dev/video0..15 and return the first index whose card name matches."""
    for idx in range(16):
        card = _query_v4l2_card_name(idx)
        if card and expected_name.lower() in card.lower():
            return idx
    return None


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
        expected_name = config.get("expected_card_name")

        # When expected_card_name is set, auto-discover the correct device
        # index by scanning V4L2 nodes. This handles hot-plug scenarios where
        # devices shift (e.g. Optris disconnected/reconnected between cycles).
        if expected_name:
            discovered = _find_device_by_card_name(expected_name)
            if discovered is None:
                raise RuntimeError(
                    f"No V4L2 device matching '{expected_name}' found. "
                    f"Is the camera connected? "
                    f"Run 'v4l2-ctl --list-devices' to check."
                )
            if discovered != device_index:
                import logging
                logging.getLogger(__name__).warning(
                    "Config says camera_port=%d but '%s' is at /dev/video%d — using %d.",
                    device_index, expected_name, discovered, discovered,
                )
                device_index = discovered

        self._device_path = f"/dev/video{device_index}"
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open /dev/video{device_index}. "
                "Check that the device exists and is not in use."
            )

        # Request MJPG so the camera sends decoded frames.
        self._cap.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
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
