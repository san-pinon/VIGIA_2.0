"""
Collection of drivers for IR camera systems.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
from datetime import datetime as dt, timedelta as td, UTC

import cv2
import numpy as np

from multicam.errors import CaptureFailure
from .optris import (
    capture as capture_optris,
    Camera as OptrisCamera,
    _convert_temp2image,
)


def _write_image(
    timestamp: dt, image: np.ndarray, archive: pathlib.Path, config: dict
) -> None:
    """Utility function that writes an image to file."""

    metadata = config["metadata"]

    julday = timestamp.timetuple().tm_yday

    print("      ...writing image to file...")
    frame, frame_loop = 0, True
    while frame_loop:
        image_name = (
            archive
            / "receive"
            / (
                f"{metadata['vnum']}.{metadata['site_code']}.{timestamp.year}.{julday:03d}_"
                f"{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"
                f"-{frame:04d}.png"
            )
        )

        if image_name.is_file():
            frame += 1
            continue
        break

    cv2.imwrite(str(image_name), image)


def capture_image(config: dict) -> None:
    """
    Handles queries to IR cameras attached to the multicam system.

    Parameters
    ----------
    config:
        Camera configuration information.

    """

    instrument_config = config["infrared"]

    print("Capturing images...")
    match instrument_config["model"]:
        case "optris":
            camera = OptrisCamera(instrument_config)
            capture_fn = capture_optris
        case _:
            raise ValueError("Invalid camera model.")

    time_between_frames = 1.0 / instrument_config["framerate"]

    frames, starttime = 0, dt.now(UTC)
    try:
        print("   ...entering capture loop...")
        while frames < instrument_config["frame_count"]:
            utcnow = dt.now(UTC)

            # Capture first frame immediately, otherwise wait for frame interval to elapse
            if utcnow < starttime + td(seconds=time_between_frames) and frames != 0:
                continue

            try:
                capture_result = capture_fn(camera)
            except CaptureFailure:
                camera.close()
                raise

            image = capture_result.artifacts["image"]

            if instrument_config["model"] == "optris":
                image = _convert_temp2image(image)

            _write_image(
                utcnow,
                image,
                pathlib.Path(config["metadata"]["data_archive"]) / "infrared",
                config,
            )

            starttime += td(seconds=time_between_frames)
            frames += 1
        print("...capture sequence complete. Shutting down.")
    finally:
        camera.close()
