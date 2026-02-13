"""
Driver for a Canon DSLR camera using system calls to gphoto2.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import subprocess
import time
from typing import Any, Mapping

from multicam.drivers import CaptureResult


# Check availability of gphoto2


class Camera:
    def __init__(self):
        """
        Long-lived camera controller.

        """

        def __init__(self, config: Mapping[str, Any]) -> None:
            # Do something here to ensure gphoto2 is available.
            pass


def capture(camera: Camera, settings: dict, burst: bool = False):
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

    # if burst:
    #     shutter_sequence = ()
    # else:
    #     wait_event = ""

    # Build subprocess command
    cmd = [
        "gphoto2",
        f"--set-config iso={settings['iso']}",
        "--set-config capturetarget=1",
        f"--set-config shutterspeed={settings['shutterspeed']}",
        f"--set-config aperture={settings['aperture']}",
        '--set-config eosremoterelease="Immediate"',
        '--set-config eosremoterelease="Release Full"',
        "--wait-event-and-download=ObjectRemoved",
        "--force-overwrite",
    ]
    subprocess.call(cmd)

    meta = {
        "ts_monotonic_ns": time.monotonic_ns(),
        # "shape": tuple(image.shape),
        # "dtype": str(image.dtype),
    }

    return CaptureResult(metadata=meta, artifacts={"image": image})


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--iso",
#         help="Specify the ISO setting to be used.",
#         required=True,
#     )
#     parser.add_argument(
#         "--shutterspeed",
#         help="Specify the shutterspeed to be used.",
#         required=True,
#     )
#     parser.add_argument(
#         "--aperture",
#         help="Specify the aperture to be used.",
#         required=True,
#     )
#     settings = parser.parse_args()

#     camera = Camera()

#     metadata, image = capture(camera, settings)
