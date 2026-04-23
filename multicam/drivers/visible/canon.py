"""
Driver for a Canon DSLR camera using python-gphoto2.

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
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

try:
    import gphoto2 as gp
except ModuleNotFoundError as e:
    gp = None
    _GPHOTO2_IMPORT_ERROR = e
else:
    _GPHOTO2_IMPORT_ERROR = None



class Camera:
    """
    Long-lived Canon DSLR controller via python-gphoto2.

    Expected config shape (``config["visible"]``):

    .. code-block:: toml

        [visible]
        model = "canon"

        [visible.canon]
        iso          = 800
        shutterspeed = "1/250"
        aperture     = "5.6"

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if gp is None:
            raise RuntimeError(
                "python-gphoto2 is not installed; cannot use Canon DSLR driver."
            ) from _GPHOTO2_IMPORT_ERROR

        self._config = dict(config)
        self._closed = False

        try:
            self.camera = gp.Camera()
            self.camera.init()
        except gp.GPhoto2Error as e:
            raise RuntimeError(
                "Failed to initialise Canon camera "
                f"(is it connected and unlocked?): {e}"
            ) from e

    def shutdown(self) -> None:
        """Release camera resources (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self.camera.exit()
        except Exception:
            pass

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.shutdown()

    def __str__(self) -> str:
        try:
            abilities = self.camera.get_abilities()
            return f"{abilities.model} (Canon DSLR via gphoto2)"
        except Exception:
            return "Canon DSLR (via gphoto2)"


def _apply_config_values(camera: gp.Camera, settings: dict[str, Any]) -> None:
    """Apply multiple widget values in a single get_config/set_config round-trip."""

    cfg = camera.get_config()
    for name, value in settings.items():
        widget = cfg.get_child_by_name(name)
        widget.set_value(str(value))
    camera.set_config(cfg)


def capture(camera: Camera, settings: dict | None = None) -> CaptureResult:
    """
    Trigger the shutter and retrieve the captured image.

    The function applies any ``settings`` overrides (iso, shutterspeed,
    aperture) via the gphoto2 configuration interface, triggers the
    shutter, waits for the ``GP_EVENT_FILE_ADDED`` event, downloads the
    file into memory, and decodes it to a NumPy array using imageio.

    Parameters
    ----------
    camera:
        Initialised camera object.
    settings:
        Optional dict with keys ``iso``, ``shutterspeed``, ``aperture``
        that override the values stored in ``camera._config["canon"]``.

    Returns
    -------
    capture_result:
        Object containing metadata and image artifact.

    Raises
    ------
    CaptureFailure:
        If the camera does not signal a file-added event within the
        timeout period, or if image decoding fails.

    """

    if gp is None:
        raise RuntimeError(
            "python-gphoto2 is not installed."
        ) from _GPHOTO2_IMPORT_ERROR

    canon_cfg = dict(camera._config.get("canon", {}))
    if settings:
        canon_cfg.update(settings)

    # --- Apply camera settings (one round-trip for all widgets) ---
    widget_map = {"iso": "iso", "shutterspeed": "shutterspeed", "aperture": "aperture"}
    to_apply = {widget_map[k]: v for k, v in canon_cfg.items() if k in widget_map}
    if to_apply:
        try:
            _apply_config_values(camera.camera, to_apply)
        except gp.GPhoto2Error as e:
            raise CaptureFailure(f"Failed to apply Canon camera settings: {e}") from e

    # --- Trigger shutter ---
    try:
        camera.camera.trigger_capture()
    except gp.GPhoto2Error as e:
        raise CaptureFailure(f"Canon shutter trigger failed: {e}") from e

    ts_monotonic_ns = time.monotonic_ns()

    # --- Wait for file-added event (max 30 s) ---
    file_path = None
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        event_type, event_data = camera.camera.wait_for_event(1000)  # ms
        if event_type == gp.GP_EVENT_FILE_ADDED:
            file_path = event_data
            break
        if event_type == gp.GP_EVENT_CAPTURE_COMPLETE:
            continue  # keep waiting for the file

    if file_path is None:
        raise CaptureFailure("Timed out waiting for Canon file-added event.")

    # --- Download image into memory ---
    try:
        camera_file = camera.camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
        )
        raw_bytes = memoryview(camera_file.get_data_and_size())
        image: np.ndarray = iio.imread(BytesIO(bytes(raw_bytes)))
    except Exception as e:
        raise CaptureFailure(f"Failed to download/decode Canon image: {e}") from e

    meta = {
        "ts_monotonic_ns": ts_monotonic_ns,
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "filename": file_path.name,
    }

    return CaptureResult(metadata=meta, artifacts={"image": image})
