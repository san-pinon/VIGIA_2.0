"""
Driver for an OceanInsight spectrometer using seabreeze.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import seabreeze

seabreeze.use("pyseabreeze")
from seabreeze.spectrometers import list_devices, Spectrometer as Spectrometer_

from multicam.drivers import CaptureResult


class Spectrometer:
    """
    Long-lived spectrometer controller.

    Attributes
    ----------
    spectrometer: The `seabreeze.Spectrometer` object.
    integration_time_micros: The integration time in microseconds.
    wavelengths: The wavelengths measured by the spectrometer in nanometers.

    """

    def __init__(self, config: Mapping[str, Any]):
        """Instantiate the Spectrometer object."""

        devices = list_devices()
        self.spectrometer = Spectrometer_(devices[0])
        self.spectrometer.trigger_mode(0)

        self._integration_time = 100_000
        self._closed = False

    def shutdown(self) -> None:
        """Release spectrometer resources (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self.spectrometer.close()
        except Exception:
            pass

    def __enter__(self) -> "Spectrometer":
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.shutdown()

    @property
    def integration_time_micros(self):
        """Return the currently set integration time."""

        return self._integration_time

    @integration_time_micros.setter
    def integration_time_micros(self, value):
        """Update the integration time for the Spectrometer."""

        lower_limit, upper_limit = self.spectrometer.integration_time_micros_limits

        if value < lower_limit or value > upper_limit:
            raise ValueError(
                f"Specified time ({value} microseconds) lies outside range of possible "
                f"integration times:\n\t{lower_limit} - {upper_limit} microseconds"
            )

        self._integration_time = value
        self.spectrometer.integration_time_micros(value)

    @property
    def wavelengths(self):
        """Return the set of wavelengths to which the spectrometer is sensitive."""

        return self.spectrometer.wavelengths()


def check_spectrum_saturation(
    spectrum: np.ndarray,
    spectrometer: Spectrometer,
    min_saturation: float = 0.5,
    max_saturation: float = 0.8,
    pixel_count: int = 100,
    wavelength_range: tuple[int, int] | None = None,
) -> int:
    """
    Check whether a spectrum is over-/under-exposed.

    An average of the top N pixel values is used in order to avoid broken pixels
    causing erroneous estimates of the exposure. A specific range of wavelengths may be
    chosen.

    Parameters
    ----------
    spectrum:
        The spectrum to be analysed.
    min_saturation:
        The threshold below which the spectrum is considered under-exposed.
    max_saturation:
        The threshold above which the spectrum is considered over-exposed.
    bit_depth:
        The number of bits used to digitise the values reported by the sensor.
    pixel_count:
        The number of pixels to use to compute the average pixel value.
    wavelength_raneg:
        The range of wavelengths in the spectrum used to compute the
        average pixel value.

    Returns
    -------
    saturation_flag:
        An integer flag indicated whether the spectrum is under-exposed (1), adequately
        exposed (0), or over-exposed (-1).

    """

    # Get spectrum sub-range
    subspectrum = spectrum

    average_DN = np.mean(subspectrum[subspectrum.argsort()[-pixel_count:]])

    max_DN = 2**spectrometer.bit_depth - 1
    saturation = average_DN / max_DN

    if saturation > max_saturation:
        return -1
    elif saturation < min_saturation:
        return 1
    else:
        return 0


def capture(spectrometer: Spectrometer, settings: dict | None = None) -> CaptureResult:
    """
    Capture spectra.

    Parameters
    ----------
    spectrometer:
        Initialised spectrometer object.
    settings:
        Local overrides for spectrometer settings.

    Returns
    -------
    capture_result:
        Object containing metadata and artifacts.

    """

    # devs = list_devices()
    # spec = Spectrometer(devs[0])

    import time as _time

    if settings:
        integration_time = settings.get("integration_time_us", spectrometer.integration_time_micros)
        n_stack = int(settings.get("stacking", 1))
    else:
        integration_time = spectrometer.integration_time_micros
        n_stack = 1

    spectrometer.integration_time_micros = integration_time

    # Capture and discard one spectrum as burn-in to flush the sensor.
    _ = spectrometer.spectrometer.intensities()

    spectra_stack = (
        np.add.reduce(
            [spectrometer.spectrometer.intensities() for _ in range(n_stack)]
        )
        / n_stack
    )

    meta = {
        "ts_monotonic_ns": _time.monotonic_ns(),
        "integration_time_us": integration_time,
        "stacking": n_stack,
        "wavelengths": spectrometer.wavelengths.tolist(),
    }

    return CaptureResult(metadata=meta, artifacts={"spectrum": spectra_stack})
