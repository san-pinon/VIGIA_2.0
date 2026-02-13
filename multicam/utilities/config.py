"""
Utilities for reading in configuration files.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
import tomllib

from multicam.errors import InvalidConfigFile


def read_config(config_file: pathlib.Path | None) -> dict:
    """Utility function to read in configuration for the node/hub."""

    if config_file is None:
        config_file = pathlib.Path.home() / ".config/multicam_config.toml"

    if not config_file.is_file():
        raise InvalidConfigFile

    with config_file.open("rb") as f:
        config = tomllib.load(f)

    return config
