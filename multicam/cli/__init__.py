"""
CLI for the multicam system.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import argparse
import sys

from .capture import image_capture_handler


FN_MAP = {
    "capture": image_capture_handler,
}


def entry_point(args=None):
    """Command-line interface for the VIGIA 2.0 system."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        help="top-level command used to select a sub-utility.",
        choices=FN_MAP.keys(),
    )
    args = parser.parse_args(sys.argv[1:2])

    FN_MAP[args.command]()
