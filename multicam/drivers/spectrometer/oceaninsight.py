"""
Driver for an OceanInsight spectrometer using seabreeze.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import seabreeze

seabreeze.use("pyseabreeze")
from seabreeze.spectrometers import Spectrometer as Spectrometer_  # noqa: E402
from seabreeze.spectrometers import list_devices  # noqa: E402

from multicam.drivers import CaptureResult  # noqa: E402


class Spectrometer:
    """
    Long-lived spectrometer controller.

    Attributes
    ----------
    spectrometer:
        The underlying ``seabreeze.Spectrometer`` object.
    integration_time_micros:
        The currently applied integration time in microseconds.
    wavelengths:
        The wavelengths (nm) to which the spectrometer is sensitive.
    bit_depth:
        The ADC bit depth of the detector (read from the device).

    """

    def __init__(self, config: Mapping[str, Any]):
        """Instantiate the Spectrometer object."""

        devices = list_devices()
        self.spectrometer = Spectrometer_(devices[0])
        self.spectrometer.trigger_mode(0)

        self._integration_time = 100_000
        self._closed = False

        # Apply integration time from config immediately so the first capture
        # uses the correct value (avoids a silent mismatch if the default
        # 100 000 µs differs from the configured value).
        configured_time = int(config.get("integration_time_us", self._integration_time))
        self.integration_time_micros = configured_time

    def shutdown(self) -> None:
        """Release spectrometer resources (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self.spectrometer.close()
        except Exception:
            pass

    def __enter__(self) -> Spectrometer:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.shutdown()

    @property
    def integration_time_micros(self) -> int:
        """Return the currently set integration time in microseconds."""

        return self._integration_time

    @integration_time_micros.setter
    def integration_time_micros(self, value: int) -> None:
        """Update the integration time for the spectrometer."""

        lower_limit, upper_limit = self.spectrometer.integration_time_micros_limits

        if value < lower_limit or value > upper_limit:
            raise ValueError(
                f"Specified time ({value} µs) lies outside the device range: "
                f"{lower_limit}–{upper_limit} µs"
            )

        self._integration_time = value
        self.spectrometer.integration_time_micros(value)

    @property
    def bit_depth(self) -> int:
        """ADC bit depth of the detector, read from the seabreeze device features."""

        # seabreeze exposes max_intensity (= 2**bit_depth - 1).
        # Confirmed on OceanHR4: max_intensity = 65535.0  →  bit_depth = 16.
        max_intensity = self.spectrometer.max_intensity
        return math.floor(math.log2(max_intensity + 1))

    @property
    def wavelengths(self) -> np.ndarray:
        """Return the set of wavelengths (nm) to which the spectrometer is sensitive."""

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

    An average of the top N pixel values is used to avoid broken pixels
    causing erroneous exposure estimates. A specific wavelength sub-range
    may be selected (useful for restricting the assessment to the SO₂
    absorption band, ~310–330 nm).

    Parameters
    ----------
    spectrum:
        The spectrum to be analysed (intensity values).
    spectrometer:
        The ``Spectrometer`` object — used to derive the ADC bit depth.
    min_saturation:
        Fraction below which the spectrum is considered under-exposed.
    max_saturation:
        Fraction above which the spectrum is considered over-exposed.
    pixel_count:
        Number of brightest pixels used to compute the average DN.
    wavelength_range:
        ``(min_nm, max_nm)`` sub-range to restrict the assessment to.
        If ``None``, the full spectrum is used.

    Returns
    -------
    saturation_flag:
        ``1``  — under-exposed,
        ``0``  — adequately exposed,
        ``-1`` — over-exposed.

    """

    if wavelength_range is not None:
        wavelengths = spectrometer.wavelengths
        lo, hi = wavelength_range
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        subspectrum = spectrum[mask]
    else:
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
    Capture a (optionally stacked) spectrum.

    Parameters
    ----------
    spectrometer:
        Initialised ``Spectrometer`` object.
    settings:
        Per-call overrides: ``integration_time_us`` (int) and/or
        ``stacking`` (int, number of spectra to average).

    Returns
    -------
    capture_result:
        Metadata contains ``ts_monotonic_ns``, ``integration_time_us``,
        ``stacking``, and ``wavelengths``.
        Artifacts contain ``spectrum`` (1-D ``np.ndarray``).

    """

    if settings:
        integration_time = int(
            settings.get("integration_time_us", spectrometer.integration_time_micros)
        )
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
        "ts_monotonic_ns": time.monotonic_ns(),
        "integration_time_us": integration_time,
        "stacking": n_stack,
        "wavelengths": spectrometer.wavelengths.tolist(),
    }

    return CaptureResult(metadata=meta, artifacts={"spectrum": spectra_stack})
