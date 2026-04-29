"""
Collection of drivers for the UV dual-camera system.

Both cameras connect to the Raspberry Pi 5 via dedicated MIPI-CSI ports.
Camera 1 (port ``camera_1_port``) sits behind the 310 nm filter;
Camera 2 (port ``camera_2_port``) sits behind the 330 nm filter.
Confirm the port → filter mapping against the physical wiring before
deploying (see TODO.md).

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
from datetime import UTC
from datetime import datetime as dt
from datetime import timedelta as td

import cv2

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

from .picam import DualCamera, capture


def _write_images(
    timestamp: dt,
    result_1: CaptureResult,
    result_2: CaptureResult,
    config: dict,
) -> None:
    """Write a pair of UV images to the receive directory."""

    instrument_config = config["ultraviolet"]
    metadata = config["metadata"]
    archive = pathlib.Path(metadata["data_archive"]) / "ultraviolet" / "receive"

    ch1 = result_1.metadata.get(
        "filter_nm", instrument_config.get("camera_1_filter_nm", 310)
    )
    ch2 = result_2.metadata.get(
        "filter_nm", instrument_config.get("camera_2_filter_nm", 330)
    )

    julday = timestamp.timetuple().tm_yday
    base = (
        f"{metadata['vnum']}.{metadata['site_code']}.{timestamp.year}.{julday:03d}_"
        f"{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"
    )

    print("      ...writing UV images to file...")
    frame = 0
    while True:
        name_1 = archive / f"{base}-{frame:04d}-{ch1}.tiff"
        if not name_1.is_file():
            break
        frame += 1

    name_2 = archive / f"{base}-{frame:04d}-{ch2}.tiff"

    cv2.imwrite(str(name_1), result_1.artifacts["image"])
    cv2.imwrite(str(name_2), result_2.artifacts["image"])


def capture_image(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle queries to the UV dual-camera system attached to the Raspberry Pi 5.

    Captures ``frame_count`` image pairs at ``framerate`` Hz, writing each
    pair to the data archive as it is captured.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).

    """

    instrument_config = config["ultraviolet"]
    time_between_frames = 1.0 / instrument_config["framerate"]

    print("Capturing UV images...")

    camera = DualCamera(instrument_config)
    try:
        archive = pathlib.Path(config["metadata"]["data_archive"]) / "ultraviolet"
        (archive / "receive").mkdir(parents=True, exist_ok=True)

        frames, starttime = 0, dt.now(UTC)
        print("   ...entering capture loop...")
        while frames < instrument_config["frame_count"]:
            utcnow = dt.now(UTC)

            # Capture first frame immediately; afterwards wait for the interval.
            if utcnow < starttime + td(seconds=time_between_frames) and frames != 0:
                continue

            try:
                result_1, result_2 = capture(camera)
            except CaptureFailure:
                raise

            _write_images(utcnow, result_1, result_2, config)

            starttime += td(seconds=time_between_frames)
            frames += 1

        print("...capture sequence complete. Shutting down.")
    finally:
        camera.close()
