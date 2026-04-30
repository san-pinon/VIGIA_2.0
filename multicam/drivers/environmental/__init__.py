"""
Collection of drivers for environmental sensors (temperature, pressure, humidity).

Supported hardware:
- Bosch BME280 — temperature, pressure, relative humidity (I2C)
- Sensirion SHT30 — temperature, relative humidity (I2C)

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import json
import pathlib
from datetime import UTC
from datetime import datetime as dt

from multicam.errors import CaptureFailure

from .bme280 import BME280
from .bme280 import capture as bme280_capture
from .sht30 import SHT30
from .sht30 import capture as sht30_capture


def _write_reading(timestamp: dt, readings: dict, config: dict) -> None:
    """Write an environmental reading to the receive directory as a JSON file."""

    meta = config["metadata"]
    year = timestamp.strftime("%Y")
    julday = int(timestamp.strftime("%j"))
    time_str = timestamp.strftime("%H%M%S")

    fname = f"{meta['vnum']}.{meta['site_code']}.{year}.{julday:03d}_{time_str}.json"
    dest = (
        pathlib.Path(config["metadata"]["data_archive"])
        / "environmental"
        / "receive"
        / fname
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_utc": timestamp.isoformat(),
        "readings": readings,
    }
    dest.write_text(json.dumps(payload, indent=2))


def capture_environmental(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle queries to environmental sensors attached to the multicam system.

    Reads from every enabled sensor in ``config["environmental"]`` and
    writes a single JSON file containing all readings to the data archive.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).

    """

    env_config = config["environmental"]
    bme_cfg = env_config.get("bme280", {})
    sht_cfg = env_config.get("sht30", {})

    readings: dict = {}

    # --- BME280 ---
    if bme_cfg.get("enabled", False):
        print("Reading BME280...")
        try:
            with BME280(bme_cfg["i2c_bus"], bme_cfg["address"]) as sensor:
                result = bme280_capture(sensor)
            readings["bme280"] = result.artifacts
        except CaptureFailure as e:
            print(f"  BME280 capture failed: {e}")
            readings["bme280"] = {"error": str(e)}
        except Exception as e:
            print(f"  BME280 error: {e}")
            readings["bme280"] = {"error": str(e)}

    # --- SHT30 ---
    if sht_cfg.get("enabled", False):
        print("Reading SHT30...")
        try:
            with SHT30(sht_cfg["i2c_bus"], sht_cfg["address"]) as sensor:
                result = sht30_capture(sensor)
            readings["sht30"] = result.artifacts
        except CaptureFailure as e:
            print(f"  SHT30 capture failed: {e}")
            readings["sht30"] = {"error": str(e)}
        except Exception as e:
            print(f"  SHT30 error: {e}")
            readings["sht30"] = {"error": str(e)}

    if not readings:
        print("No environmental sensors enabled in config — nothing to capture.")
        return

    timestamp = dt.now(UTC)
    _write_reading(timestamp, readings, config)
    print(f"...environmental reading written ({', '.join(readings.keys())}).")
