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

import numpy as np

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename

from .picam import DualCamera, capture


def _write_images(
    timestamp: dt,
    results: tuple[CaptureResult, ...],
    output_dir: pathlib.Path,
    config: dict,
) -> None:
    """Save UV captures as NPZ files into output_dir."""

    metadata = config["metadata"]
    default_filters = [
        config["ultraviolet"].get("camera_1_filter_nm", 310),
        config["ultraviolet"].get("camera_2_filter_nm", 330),
    ]

    print("      ...writing UV images to file...")
    for i, result in enumerate(results):
        ch = result.metadata.get("filter_nm", default_filters[i])
        fname = build_filename(metadata, timestamp, suffix=f"uv-{ch}", extension="npz")
        np.savez(output_dir / fname, image=result.artifacts["image"])
        print(f"      Saved: {fname}")


def capture_image(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle queries to the UV dual-camera system attached to the Raspberry Pi 5.

    Captures ``frame_count`` image pairs at ``framerate`` Hz, writing each
    pair to the data archive as it is captured.

    Output goes to the same directory as ``uv-sync`` (``uv_sync.output_dir``
    or ``data_archive/uv``) so the dashboard picks up both capture paths.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).

    """

    instrument_config = config["ultraviolet"]
    sync_config = config.get("uv_sync", {})
    time_between_frames = 1.0 / instrument_config["framerate"]

    output_dir = pathlib.Path(
        sync_config.get("output_dir")
        or (config["metadata"]["data_archive"] + "/uv")
    ) / "receive"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Capturing UV images...")

    camera = DualCamera(instrument_config)
    try:
        frames, starttime = 0, dt.now(UTC)
        print("   ...entering capture loop...")
        while frames < instrument_config["frame_count"]:
            utcnow = dt.now(UTC)

            if utcnow < starttime + td(seconds=time_between_frames) and frames != 0:
                continue

            try:
                results = capture(camera)
            except CaptureFailure:
                raise

            _write_images(utcnow, results, output_dir, config)

            starttime += td(seconds=time_between_frames)
            frames += 1

        print("...capture sequence complete. Shutting down.")
    finally:
        camera.close()
