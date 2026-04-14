"""

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import pathlib
import sys

from multicam.drivers.environmental import capture_environmental
from multicam.drivers.infrared import capture_image as capture_ir_image
from multicam.drivers.spectrometer import capture_spectra
from multicam.drivers.ultraviolet import capture_image as capture_uv_images
from multicam.drivers.visible import capture_image as capture_vis_image
from multicam.utilities import read_config


FN_MAP = {
    "environmental": capture_environmental,
    "infrared": capture_ir_image,
    "spectrometer": capture_spectra,
    "ultraviolet": capture_uv_images,
    "visible": capture_vis_image,
}


def image_capture_handler(args=None):
    """
    A command-line entry point that handles parsing and dispatching of calls to query
    instruments attached to the node.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "instrument",
        help="Specify the type of instrument to be queried.",
        choices=FN_MAP.keys(),
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Specify the path to a configuration file to be used.",
        required=False,
        default=pathlib.Path.home() / ".config/multicam_config.toml",
    )
    args = parser.parse_args(sys.argv[2:])

    config = read_config(pathlib.Path(args.config))

    data_dir = pathlib.Path(config["metadata"]["data_archive"]) / args.instrument
    (data_dir / "receive").mkdir(exist_ok=True, parents=True)

    # --- Map arguments to appropriate instrument driver ---
    FN_MAP[args.instrument](config)
