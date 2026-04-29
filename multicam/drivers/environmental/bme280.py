"""
Driver for a Bosch BME280 temperature / pressure / humidity sensor via I2C.

Register map and compensation formulas follow the BME280 datasheet
(BST-BME280-DS002, rev. 1.9, section 4).

Interface: Linux I2C via smbus2.

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import struct
import time

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

try:
    from smbus2 import SMBus
except ModuleNotFoundError as e:
    SMBus = None
    _SMBUS2_IMPORT_ERROR = e
else:
    _SMBUS2_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Register addresses (BME280 datasheet Table 18)
# ---------------------------------------------------------------------------

_REG_ID = 0xD0  # Chip ID — should read 0x60
_REG_RESET = 0xE0  # Write 0xB6 for soft reset
_REG_CTRL_HUM = 0xF2  # Humidity oversampling
_REG_STATUS = 0xF3  # Measuring / updating flags
_REG_CTRL_MEAS = 0xF4  # Temperature + pressure oversampling, mode
_REG_CONFIG = 0xF5  # Standby time, filter, SPI 3-wire
_REG_DATA = 0xF7  # Start of 8-byte raw data burst
_REG_CALIB_00 = 0x88  # Start of trimming registers block 1 (24 bytes)
_REG_CALIB_26 = 0xE1  # Start of trimming registers block 2 (7 bytes)

# Forced mode: take one measurement, then return to sleep.
_MODE_FORCED = 0b01

# Oversampling ×1 for all channels.
_OSRS_1 = 0b001


class BME280:
    """
    Long-lived BME280 sensor controller.

    Expected config shape (``config["environmental"]["bme280"]``):

    .. code-block:: toml

        [environmental.bme280]
        enabled = true
        i2c_bus = 1       # /dev/i2c-N
        address = 0x76    # 0x76 (SDO low) or 0x77 (SDO high)

    """

    CHIP_ID = 0x60

    def __init__(self, i2c_bus: int, address: int = 0x76) -> None:
        if SMBus is None:
            raise RuntimeError(
                "smbus2 is not installed; cannot use BME280 driver."
            ) from _SMBUS2_IMPORT_ERROR

        self._bus_num = i2c_bus
        self._bus = SMBus(i2c_bus)
        self._addr = address
        self._closed = False

        chip_id = self._bus.read_byte_data(self._addr, _REG_ID)
        if chip_id != self.CHIP_ID:
            raise RuntimeError(
                f"BME280 not found at address 0x{address:02X} "
                f"(chip ID 0x{chip_id:02X}, expected 0x{self.CHIP_ID:02X})."
            )

        self._load_calibration()
        self._configure()

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

    def __enter__(self) -> BME280:
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        self.shutdown()

    def __str__(self) -> str:
        return f"BME280(bus={self._bus_num}, addr=0x{self._addr:02X})"

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_calibration(self) -> None:
        """Read and unpack the factory calibration trimming parameters."""

        # Block 1: 0x88..0x9F  (24 bytes → 12 unsigned/signed 16-bit words)
        b1 = self._bus.read_i2c_block_data(self._addr, _REG_CALIB_00, 24)
        (
            self._dig_T1,
            self._dig_T2,
            self._dig_T3,
            self._dig_P1,
            self._dig_P2,
            self._dig_P3,
            self._dig_P4,
            self._dig_P5,
            self._dig_P6,
            self._dig_P7,
            self._dig_P8,
            self._dig_P9,
        ) = struct.unpack_from("<HhhHhhhhhhhh", bytes(b1))

        # Block 1 byte 25 (0xA1) — H1
        self._dig_H1 = self._bus.read_byte_data(self._addr, 0xA1)

        # Block 2: 0xE1..0xE7 (7 bytes)
        b2 = self._bus.read_i2c_block_data(self._addr, _REG_CALIB_26, 7)
        self._dig_H2 = struct.unpack_from("<h", bytes(b2), 0)[0]
        self._dig_H3 = b2[2]
        self._dig_H4 = (b2[3] << 4) | (b2[4] & 0x0F)
        self._dig_H5 = (b2[5] << 4) | (b2[4] >> 4)
        self._dig_H6 = struct.unpack_from("<b", bytes(b2), 6)[0]

        # Convert H4/H5 to signed 12-bit
        if self._dig_H4 > 2047:
            self._dig_H4 -= 4096
        if self._dig_H5 > 2047:
            self._dig_H5 -= 4096

    def _configure(self) -> None:
        """Set oversampling and forced mode via ctrl_hum / ctrl_meas."""

        # Humidity oversampling ×1  (must be set BEFORE ctrl_meas)
        self._bus.write_byte_data(self._addr, _REG_CTRL_HUM, _OSRS_1)
        # Temperature ×1, pressure ×1, sleep mode (mode bits = 00)
        self._bus.write_byte_data(
            self._addr,
            _REG_CTRL_MEAS,
            (_OSRS_1 << 5) | (_OSRS_1 << 2) | 0b00,
        )

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _trigger_forced_measurement(self) -> None:
        """Kick off a one-shot forced-mode measurement."""

        ctrl = self._bus.read_byte_data(self._addr, _REG_CTRL_MEAS)
        ctrl = (ctrl & 0xFC) | _MODE_FORCED
        self._bus.write_byte_data(self._addr, _REG_CTRL_MEAS, ctrl)

    def _wait_for_measurement(self, timeout_s: float = 1.0) -> None:
        """Poll the status register until measurement is complete."""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._bus.read_byte_data(self._addr, _REG_STATUS)
            if not (status & 0x08):  # bit 3 = measuring
                return
            time.sleep(0.005)
        raise CaptureFailure("BME280 measurement timed out.")

    def read_raw(self) -> tuple[int, int, int]:
        """Trigger a forced measurement and return raw (press, temp, hum) ADC values."""

        self._trigger_forced_measurement()
        self._wait_for_measurement()

        data = self._bus.read_i2c_block_data(self._addr, _REG_DATA, 8)

        press_raw = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
        temp_raw = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
        hum_raw = (data[6] << 8) | data[7]

        return press_raw, temp_raw, hum_raw

    # ------------------------------------------------------------------
    # Compensation formulas (BME280 datasheet section 4.2.3)
    # ------------------------------------------------------------------

    def _compensate_temperature(self, adc_T: int) -> tuple[float, int]:
        """Return (temperature °C, t_fine) where t_fine is reused by P/H."""

        var1 = (adc_T / 16384.0 - self._dig_T1 / 1024.0) * self._dig_T2
        var2 = (adc_T / 131072.0 - self._dig_T1 / 8192.0) ** 2 * self._dig_T3
        t_fine = int(var1 + var2)
        temperature = (var1 + var2) / 5120.0
        return temperature, t_fine

    def _compensate_pressure(self, adc_P: int, t_fine: int) -> float:
        """Return pressure in hPa."""

        var1 = t_fine / 2.0 - 64000.0
        var2 = var1 * var1 * self._dig_P6 / 32768.0
        var2 += var1 * self._dig_P5 * 2.0
        var2 = var2 / 4.0 + self._dig_P4 * 65536.0
        var1 = (self._dig_P3 * var1 * var1 / 524288.0 + self._dig_P2 * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * self._dig_P1

        if var1 == 0.0:
            return 0.0  # avoid division by zero

        pressure = 1048576.0 - adc_P
        pressure = (pressure - var2 / 4096.0) * 6250.0 / var1
        var1 = self._dig_P9 * pressure * pressure / 2147483648.0
        var2 = pressure * self._dig_P8 / 32768.0
        pressure += (var1 + var2 + self._dig_P7) / 16.0

        return pressure / 100.0  # Pa → hPa

    def _compensate_humidity(self, adc_H: int, t_fine: int) -> float:
        """Return relative humidity in %RH."""

        x = t_fine - 76800.0
        if x == 0.0:
            return 0.0

        x = (adc_H - (self._dig_H4 * 64.0 + self._dig_H5 / 16384.0 * x)) * (
            self._dig_H2
            / 65536.0
            * (
                1.0
                + self._dig_H6 / 67108864.0 * x * (1.0 + self._dig_H3 / 67108864.0 * x)
            )
        )
        x *= 1.0 - self._dig_H1 * x / 524288.0
        return float(max(0.0, min(100.0, x)))


# ---------------------------------------------------------------------------
# Public capture function
# ---------------------------------------------------------------------------


def capture(sensor: BME280, settings: dict | None = None) -> CaptureResult:
    """
    Read temperature, pressure, and humidity from the BME280.

    Parameters
    ----------
    sensor:
        Initialised BME280 object.
    settings:
        Unused; present for interface consistency.

    Returns
    -------
    capture_result:
        Metadata contains the monotonic timestamp and I2C address.
        Artifacts contain ``temperature_c`` (°C), ``pressure_hpa`` (hPa),
        and ``humidity_pct`` (%RH).

    Raises
    ------
    CaptureFailure:
        If the measurement times out or the raw read fails.

    """

    try:
        press_raw, temp_raw, hum_raw = sensor.read_raw()
    except CaptureFailure:
        raise
    except Exception as e:
        raise CaptureFailure(f"BME280 read failed: {e}") from e

    temperature, t_fine = sensor._compensate_temperature(temp_raw)
    pressure = sensor._compensate_pressure(press_raw, t_fine)
    humidity = sensor._compensate_humidity(hum_raw, t_fine)

    meta = {
        "ts_monotonic_ns": time.monotonic_ns(),
        "i2c_address": f"0x{sensor._addr:02X}",
    }

    return CaptureResult(
        metadata=meta,
        artifacts={
            "temperature_c": round(temperature, 2),
            "pressure_hpa": round(pressure, 2),
            "humidity_pct": round(humidity, 2),
        },
    )
