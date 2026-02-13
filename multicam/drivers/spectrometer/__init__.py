"""
Collection of drivers for spectrometer systems.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from datetime import datetime as dt, UTC

import numpy as np

from multicam.errors import CaptureFailure
from .oceaninsight import capture, Spectrometer_ as Spectrometer


def _write_spectrum(timestamp: dt, spectrum: np.ndarray, config: dict) -> None:
    """Utility function that writes an image to file."""

    pass


def capture_spectra(config: dict) -> None:
    """
    Handles queries to IR cameras attached to the multicam system.

    Parameters
    ----------
    config: System configuration file.

    """

    instrument_config = config["spectrum"]
    spectrometer = Spectrometer(config)
    print("Capturing spectrum...")

    frames, starttime = 0, dt.now(UTC)

    print("...capture sequence complete. Shutting down.")
