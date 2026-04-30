"""
Driver for a Sensirion SHT30 temperature / humidity sensor via I2C.

Protocol follows the SHT3x-DIS datasheet (version 6, section 4).

Interface: Linux I2C via smbus2.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import time

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

try:
    from smbus2 import SMBus, i2c_msg
except ModuleNotFoundError as e:
    SMBus = None
    i2c_msg = None
    _SMBUS2_IMPORT_ERROR = e
else:
    _SMBUS2_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# SHT30 single-shot measurement commands (high repeatability, clock stretch
# disabled — datasheet Table 8).
# ---------------------------------------------------------------------------

_CMD_MEAS_HIGH = [0x24, 0x00]  # Single-shot, high repeatability

# Measurement duration for high repeatability: ≤15 ms (datasheet Table 4).
_MEAS_DELAY_S = 0.020

# CRC-8 polynomial and initial value (datasheet section 4.12).
_CRC_POLY = 0x31
_CRC_INIT = 0xFF


def _crc8(data: bytes) -> int:
    """Compute the CRC-8 checksum used by the SHT30 protocol."""

    crc = _CRC_INIT
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ _CRC_POLY
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


class SHT30:
    """
    Long-lived SHT30 sensor controller.

    Expected config shape (``config["environmental"]["sht30"]``):

    .. code-block:: toml

        [environmental.sht30]
        enabled = true
        i2c_bus = 1       # /dev/i2c-N
        address = 0x44    # 0x44 (ADDR low) or 0x45 (ADDR high)

    """

    def __init__(self, i2c_bus: int, address: int = 0x44) -> None:
        if SMBus is None:
            raise RuntimeError(
                "smbus2 is not installed; cannot use SHT30 driver."
            ) from _SMBUS2_IMPORT_ERROR

        self._bus = SMBus(i2c_bus)
        self._addr = address
        self._closed = False

        # Soft-reset to put sensor in a known state.
        self._soft_reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close the I2C bus (idempotent)."""

        if self._closed:
            return
        self._closed = True
        try:
            self._bus.close()
        except Exception:
            pass

    def __enter__(self) -> SHT30:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.shutdown()

    def __str__(self) -> str:
        return f"SHT30(bus={self._bus._fd}, addr=0x{self._addr:02X})"

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _soft_reset(self) -> None:
        """Issue a soft reset command (0x30, 0xA2) and wait for start-up."""

        write = i2c_msg.write(self._addr, [0x30, 0xA2])
        self._bus.i2c_rdwr(write)
        time.sleep(0.002)  # start-up time ≤1.5 ms

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def read_raw(self) -> tuple[int, int]:
        """
        Trigger a single-shot measurement and return raw (temp, hum) words.

        Returns
        -------
        temp_raw:
            Raw 16-bit temperature value.
        hum_raw:
            Raw 16-bit humidity value.

        Raises
        ------
        CaptureFailure:
            If CRC validation fails.

        """

        # Send measurement command.
        write = i2c_msg.write(self._addr, _CMD_MEAS_HIGH)
        self._bus.i2c_rdwr(write)

        # Wait for measurement to complete.
        time.sleep(_MEAS_DELAY_S)

        # Read 6 bytes: [T_MSB, T_LSB, T_CRC, H_MSB, H_LSB, H_CRC].
        read = i2c_msg.read(self._addr, 6)
        self._bus.i2c_rdwr(read)
        data = list(read)

        # Validate CRC for temperature bytes.
        if _crc8(bytes(data[0:2])) != data[2]:
            raise CaptureFailure("SHT30 CRC mismatch on temperature bytes.")

        # Validate CRC for humidity bytes.
        if _crc8(bytes(data[3:5])) != data[5]:
            raise CaptureFailure("SHT30 CRC mismatch on humidity bytes.")

        temp_raw = (data[0] << 8) | data[1]
        hum_raw = (data[3] << 8) | data[4]

        return temp_raw, hum_raw


# ---------------------------------------------------------------------------
# Public capture function
# ---------------------------------------------------------------------------


def capture(sensor: SHT30, settings: dict | None = None) -> CaptureResult:
    """
    Read temperature and relative humidity from the SHT30.

    Conversion formulas follow the SHT3x-DIS datasheet section 4.13:

    .. code-block::

        T [°C]  = -45 + 175 × S_T  / (2^16 - 1)
        RH [%]  = 100 × S_RH / (2^16 - 1)

    Parameters
    ----------
    sensor:
        Initialised SHT30 object.
    settings:
        Unused; present for interface consistency.

    Returns
    -------
    capture_result:
        Metadata contains the monotonic timestamp and I2C address.
        Artifacts contain ``temperature_c`` (°C) and ``humidity_pct`` (%RH).

    Raises
    ------
    CaptureFailure:
        If CRC validation fails or the I2C read errors.

    """

    try:
        temp_raw, hum_raw = sensor.read_raw()
    except CaptureFailure:
        raise
    except Exception as e:
        raise CaptureFailure(f"SHT30 read failed: {e}") from e

    temperature = -45.0 + 175.0 * temp_raw / 65535.0
    humidity = 100.0 * hum_raw / 65535.0
    humidity = max(0.0, min(100.0, humidity))

    meta = {
        "ts_monotonic_ns": time.monotonic_ns(),
        "i2c_address": f"0x{sensor._addr:02X}",
    }

    return CaptureResult(
        metadata=meta,
        artifacts={
            "temperature_c": round(temperature, 2),
            "humidity_pct": round(humidity, 2),
        },
    )
