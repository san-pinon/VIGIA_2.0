"""
Optris measuring range and raw-value conversion.

The imager XML selects one of the camera's calibrated ranges (``<temperature>``
``<min>``/``<max>``). The calibration file lists every range the unit has, with
the extended low limit (``TempLoExt``, used when
``enable_extended_temp_range`` is on) and the number of decimals of the raw
values (``Decimal``), which applies only when ``enable_high_precision`` is
on — otherwise every range is delivered with one decimal. Kept free of the
irdirectsdk import so it can be used and tested without the SDK.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_CALI_DIR = "/usr/share/libirimager/cali"


def _local(tag: str) -> str:
    """Tag name without an XML namespace."""

    return tag.rsplit("}", 1)[-1]


def _child_text(
    element: ET.Element, name: str, default: str | None = None
) -> str | None:
    """Text of the first direct child called ``name``, namespace-agnostic."""

    for child in element:
        if _local(child.tag) == name and child.text is not None:
            return child.text.strip()
    return default


def measuring_range(xml_path: str | pathlib.Path) -> dict:
    """
    Measuring range selected in the imager XML, completed from the calibration file.

    Returns
    -------
    range:
        ``min_c`` / ``max_c``: the selected range; ``extended``: whether
        ``enable_extended_temp_range`` is on; ``low_limit_c``: the lowest
        temperature the camera reports (``TempLoExt`` when extended, else
        ``min_c``); ``high_precision``: whether ``enable_high_precision`` is on;
        ``decimals``: raw-value decimals (``temp = (raw - 1000) / 10**decimals``),
        the range's ``Decimal`` with high precision on, else 1; ``source``:
        where the limits and decimals came from.

    """

    imager = ET.parse(xml_path).getroot()
    temperature = next(el for el in imager.iter() if _local(el.tag) == "temperature")
    t_min = float(_child_text(temperature, "min"))
    t_max = float(_child_text(temperature, "max"))
    extended = _child_text(imager, "enable_extended_temp_range", "0") == "1"
    high_precision = _child_text(imager, "enable_high_precision", "0") == "1"
    result = {
        "min_c": t_min,
        "max_c": t_max,
        "extended": extended,
        "high_precision": high_precision,
        "low_limit_c": t_min,
        "decimals": 1,
        "source": "imager XML only (calibration file not found or no matching range)",
    }

    serial = _child_text(imager, "serial", "")
    cali_dir = pathlib.Path(_child_text(imager, "calipath", DEFAULT_CALI_DIR))
    cali_path = cali_dir / f"Cali-{serial}.xml"
    try:
        cali = ET.parse(cali_path).getroot()
    except (OSError, ET.ParseError):
        return result

    for rng in cali.iter():
        if _local(rng.tag) != "TemperatureRange":
            continue
        lo, hi = _child_text(rng, "TempLo"), _child_text(rng, "TempHi")
        if lo is None or hi is None or float(lo) != t_min or float(hi) != t_max:
            continue
        if high_precision:
            result["decimals"] = int(_child_text(rng, "Decimal", "1"))
        if extended:
            result["low_limit_c"] = float(_child_text(rng, "TempLoExt", lo))
        result["source"] = str(cali_path)
        break
    return result


def raw_to_celsius(frame: np.ndarray, decimals: int = 1) -> np.ndarray:
    """Optris raw uint16 values to °C: ``(raw - 1000) / 10**decimals``."""

    return (frame.astype(np.float64) - 1000.0) / 10.0**decimals


def clip_fractions(temps: np.ndarray, irange: dict) -> dict:
    """Fractions of pixels at the top of the range and at its lowest reported value."""

    return {
        "fraction_at_max": round(float(np.mean(temps >= irange["max_c"])), 5),
        "fraction_at_low_limit": round(
            float(np.mean(temps <= irange["low_limit_c"])), 5
        ),
    }


def describe(irange: dict) -> str:
    """One-line summary for start-up logs, with a warning for unverified scaling."""

    text = (
        f"measuring range {irange['min_c']:g}…{irange['max_c']:g} °C"
        + (
            f" (extended, reads down to {irange['low_limit_c']:g} °C)"
            if irange["extended"]
            else ""
        )
        + f", {irange['decimals']} decimal(s)"
    )
    if irange["decimals"] != 1:
        text += (
            " — WARNING: high precision on, temperatures converted as "
            "(raw-1000)/100; unverified with this SDK, check against a known "
            "temperature"
        )
    return text
