"""
Collection of drivers for spectrometer systems.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import pathlib
import time
from datetime import UTC
from datetime import datetime as dt

import numpy as np

from multicam.errors import CaptureFailure

from .oceaninsight import Spectrometer, capture


def _receive_dir(config: dict) -> pathlib.Path:
    """Return (and create) the spectrometer receive directory."""

    path = pathlib.Path(config["metadata"]["data_archive"]) / "spectrometer" / "receive"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_spectrum(timestamp: dt, spectrum: np.ndarray, config: dict) -> None:
    """Write a spectrum array to the receive directory as a NumPy .npy file."""

    meta = config["metadata"]
    year = timestamp.strftime("%Y")
    julday = int(timestamp.strftime("%j"))
    time_str = timestamp.strftime("%H%M%S")

    fname = f"{meta['vnum']}.{meta['site_code']}.{year}.{julday:03d}_{time_str}.npy"
    np.save(_receive_dir(config) / fname, spectrum)


def _write_wavelengths(timestamp: dt, wavelengths: np.ndarray, config: dict) -> None:
    """Write the wavelength axis once per session alongside the spectra."""

    meta = config["metadata"]
    year = timestamp.strftime("%Y")
    julday = int(timestamp.strftime("%j"))
    time_str = timestamp.strftime("%H%M%S")

    base = f"{meta['vnum']}.{meta['site_code']}.{year}.{julday:03d}"
    fname = f"{base}_{time_str}_wavelengths.npy"
    np.save(_receive_dir(config) / fname, wavelengths)


def capture_spectra(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle queries to a spectrometer attached to the multicam system.

    Reads ``frame_count`` spectra at the requested ``framerate``, writing
    each to the data archive as it is captured.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).

    """

    instrument_config = config["spectrometer"]
    frame_count = int(instrument_config.get("frame_count", 1))
    framerate = float(instrument_config["framerate"])

    print("Initialising spectrometer...")

    with Spectrometer(instrument_config) as spectrometer:
        print(f"Capturing {frame_count} spectrum/spectra at {framerate} Hz...")
        frames = 0
        start_time = dt.now(UTC)

        _write_wavelengths(start_time, spectrometer.wavelengths, config)

        while frames < frame_count:
            try:
                result = capture(spectrometer, instrument_config)
            except Exception as e:
                raise CaptureFailure(f"Spectrometer capture failed: {e}") from e

            timestamp = dt.now(UTC)
            _write_spectrum(timestamp, result.artifacts["spectrum"], config)
            frames += 1

            if frames < frame_count:
                elapsed = (dt.now(UTC) - start_time).total_seconds()
                sleep_s = max(0.0, frames / framerate - elapsed)
                time.sleep(sleep_s)

    print("...capture sequence complete. Shutting down.")
