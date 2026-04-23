# TODO — VIGIA 2.0 Implementation

## Bugs — FIXED

### ~~1. CLI crashes at import — missing UV driver package~~ ✓
`multicam/drivers/ultraviolet/` created. CLI import works.

### ~~2. Optris IR driver — `AttributeError` on init~~ ✓
`multicam/drivers/infrared/optris.py` — `self._closed = False` now initialised before
the `self.close()` cleanup call.

### ~~3. Spectrometer — `AttributeError` in `check_spectrum_saturation`~~ ✓
`bit_depth` property added to `Spectrometer` (derived from `max_intensity`).
`import time` moved to top-level. Config `integration_time_us` applied in `__init__`.

### ~~4. Saturation check — rows branch result discarded~~ ✓
Fixed in `multicam/drivers/visible/picam.py` and the new UV driver.

### ~~5. Missing `receive/` directory creation~~ ✓
`spectrometer/__init__.py` and `environmental/__init__.py` did not create the output
directory before writing. Both now call `mkdir(parents=True, exist_ok=True)` before
the first write (UV driver already did this).

---

## UV camera driver — DONE

`multicam/drivers/ultraviolet/picam.py` and `multicam/drivers/ultraviolet/__init__.py`
created with all fixes applied:
- `DualCamera` implements `__enter__`/`__exit__`; `close()` releases both cameras.
- `capture()` returns `tuple[CaptureResult, CaptureResult]`.
- `capture_image()` uses `try/finally` to guarantee `camera.close()`.
- Filter channel (310 / 330 nm) stored in `CaptureResult.metadata` and written into
  the output filename as `-{filter_nm}` suffix.

**Pending — needs hardware confirmation:**
- [ ] Run `python3 -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"` on the RPi5 with both UV cameras connected to confirm which port index maps to the 310 nm filter and which to 330 nm.
- [ ] Set `camera_1_filter_nm` / `camera_2_filter_nm` in `multicam_config.toml` once confirmed.
- [ ] Tune `ExposureTime` in `[ultraviolet.controls]` — currently no config default.

---

## Canon DSLR driver — DONE (cleanup)

- Removed dead `_GP_EVENT_FILE_ADDED = 2` constant (was unused; `gp.GP_EVENT_FILE_ADDED` is used directly in comparisons).
- `_set_config_value` replaced with `_apply_config_values` which batches all widget updates in a single `get_config`/`set_config` round-trip.

**Pending:**
- [ ] End-to-end test with the physical camera.

---

## Spectrometer driver — mostly complete

- `bit_depth` property implemented. **Confirmed**: `max_intensity = 65535.0 → bit_depth = 16` on the OceanHR4. ✓
- `integration_time_us` from config applied on construction. Confirmed range: 3 800 – 10 000 000 µs. ✓
- `import math` moved to module level. ✓
- `check_spectrum_saturation` handles `wavelength_range` filtering. ✓

**Pending:**
- [x] Wavelength axis written as `{basename}_wavelengths.npy` once per session before the capture loop. ✓

---

## Environmental sensors — driver complete, BME280 unblocked

BME280 and SHT30 drivers fully implemented. BME280 confirmed on I2C bus 1 at address `0x77` (SDO pulled high). SHT30 still unconfirmed (not visible on any bus).

**Pending:**
- [x] Enable BME280 in `multicam_config.toml`: `enabled = true`, `i2c_bus = 1`, `address = 0x77`. ✓
- [x] Fix `BME280.__str__`: `self._bus_num` stored in `__init__`, replaces `self._bus._fd`. ✓
- [ ] Test BME280 end-to-end: `multicamctl capture environmental` on RPi5.
- [ ] Resolve SHT30 I2C issue (see **Environmental sensors — HARDWARE ISSUE** below).

---

## File monitor and archival — DONE

`scripts/file_monitor.py` ported from `.systems` with improvements:
- Cleaner `_parse_filename` that handles all file types (PNG, NPY, JSON).
- CLI `--config` flag.
- Prints structured output per archived file.

`systemd-units/file-monitor.service` created.

**Pending:**
- [ ] Confirm the path to `scripts/file_monitor.py` on each node and update `ExecStart` in `file-monitor.service` (currently set to `/home/user/multicam/scripts/file_monitor.py` — adjust to match the actual repo clone location).
- [ ] Test that `inotify` watches survive the `receive/` directory being created after the monitor starts (the current implementation globs at startup only — directories created later won't be watched). If this is a concern, re-check on first file arrival.

---

## Systemd units — DONE

All unit files created in `systemd-units/`:

| File | Node |
|---|---|
| `capture-ir.service` + `.timer` | SBC |
| `capture-vis.service` + `.timer` | SBC |
| `capture-uv.service` + `.timer` | RPi5 |
| `capture-spec.service` + `.timer` | RPi5 |
| `capture-env.service` + `.timer` | RPi5 |
| `file-monitor.service` | Both |

Default timer cadence is every minute for all instruments except the spectrometer (every 10 minutes). Adjust `OnCalendar=` per deployment needs.

**Pending:**
- [x] RPi5: `User=user` ✓, venv `/home/user/.venv/` ✓ — service files already correct.
- [x] SBC: `User=user` ✓, venv `/home/user/.venv/` ✓ — service files already correct.
- [ ] Install on each node: `sudo cp systemd-units/*.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now <unit>`.

---

## GNSS / time synchronisation

**Architecture** (confirmed): RPi5 controls the GNSS antenna and acts as NTP server. SBC syncs from RPi5.

No Python driver needed — this is a sysadmin task.

### RPi5 setup (GPS confirmed at `/dev/ttyUSB0`)
- [x] `sudo apt install gpsd gpsd-clients chrony`
- [x] Edit `/etc/default/gpsd`:
  ```
  DEVICES="/dev/ttyUSB0"
  GPSD_OPTIONS="-n"
  ```
- [x] Add refclock to `/etc/chrony/chrony.conf`:
  ```
  refclock SHM 0 refid GPS precision 1e-1 offset 0.9 delay 0.2
  ```
- [x] Add subnet allow to `/etc/chrony/chrony.conf`:
  ```
  allow 192.168.30.0/24
  ```
- [x] `sudo systemctl enable --now gpsd chrony`
- [x] Verify lock: `chronyc sources -v` → `#* GPS  0  4  77` ✓

### SBC setup (IP: 192.168.30.150)
- [x] `sudo apt install chrony`
- [x] Add to `/etc/chrony/chrony.conf`:
  ```
  server 192.168.30.101 iburst prefer   # RPi5 (GPS-disciplined NTP server)
  server 192.168.30.1 iburst            # router as fallback
  ```
- [x] `sudo systemctl enable --now chrony`
- [x] Verify sync: `chronyc sources -v` and `chronyc tracking`

### Python position utility (optional)
If GPS position is needed in capture metadata:
- [ ] Add `gpsd-py3` to `pyproject.toml` dependencies.
- [ ] Write `multicam/utilities/gnss.py` wrapping `gpsd.get_current()`.

---

## Environmental sensors — HARDWARE ISSUE (SHT30 only)

**BME280**: confirmed at address `0x77` on I2C bus 1 (SDO pulled high → 0x77, not default 0x76). ✓

**SHT30**: still not visible on bus 1 (only `0x08` Witty Pi and `0x77` BME280 appear). Buses 11/12 show a floating bus — not real devices.

**Remaining diagnosis steps for SHT30:**
- [ ] Check soldering on the SHT30 board (user noted this as likely cause).
- [ ] Once resoldered, rescan: `i2cdetect -y 1` should show `0x44`.
- [ ] Set `enabled = true` and `i2c_bus = 1` for SHT30 in `multicam_config.toml` once confirmed.

---

## RPi5 — confirmed hardware info

| Item | Value |
|---|---|
| OS | Debian GNU/Linux 12 (bookworm) |
| Kernel | 6.1.0-rpi7-rpi-2712 |
| Hostname | `uvcam` |
| Username | `user` |
| Python venv | `/home/user/.venv/` |
| `multicamctl` | `/home/user/.venv/bin/multicamctl` |
| libcamera | v0.1.0+118-563cd78e |
| UV cameras | 2× OV5647 at ports 0 and 1 (filter mapping not yet confirmed) |
| Spectrometer USB | `0999:1002` OceanInsight OceanHR4 |
| Spectrometer `max_intensity` | 65535.0 → bit_depth = 16 ✓ |
| Spectrometer integration limits | 3 800 – 10 000 000 µs ✓ |
| GPS port | `/dev/ttyUSB0` (USB-UART bridge; baud 4800; live fix confirmed ✓) |
| IP | 192.168.30.101 |
| I2C | bus 1: `0x08` (Witty Pi), `0x77` (BME280 ✓). Buses 11/12 floating. SHT30 (0x44) not yet visible. |

**Note on spectrometer model**: `lsusb` reports `OceanHR4` (USB device name), user confirms SR4 by serial number. These may be the same physical unit with a different commercial designation. seabreeze works with it; `bit_depth = 16` confirmed.

---

## OnLogic SBC — confirmed hardware info

| Item | Value |
|---|---|
| OS | Ubuntu 20.04.6 LTS (Focal Fossa) |
| Kernel | 5.4.0-186-generic |
| Hostname | `multicam` |
| Username | `user` |
| IP | 192.168.30.150 |
| Python | 3.11.9 at `/home/user/.venv/bin/python3` |
| venv | `/home/user/.venv/` |
| irdirectsdk | `/usr/lib/libirdirectsdk.so` (ctypes loads OK) |
| IR camera calibration XML | `/usr/share/libirimager/cali/Cali-24062074.xml` (SN 24062074) |
| Canon DSLR | EOS 4000D at `usb:001,007` ✓ |
| Log path | `/home/user/logs/thermal` ✓ |
| Visible camera | Arducam IMX477 HQ (`0c45:636d`) at `/dev/video0` ✓ — `camera_port = 0` confirmed correct |
| multicamctl | `/home/user/.venv/bin/multicamctl` ✓ |
| Optris IR camera | `0403:de37` FTDI "PI IMAGER" (Bus 001) ✓ — irdirectsdk communicates via FTDI directly, no `/dev/video` node used |
| Canon EOS 4000D | `04a9:32d9` (Bus 001) — creates `/dev/video2`+`3` in webcam mode but accessed via gphoto2 |

**Note on Python version**: SBC runs Python 3.11.9, RPi5 runs Python 3.11.2. Both covered by `>=3.11` in `pyproject.toml` ✓.

---

## Commands to run — RPi5 (remaining)

### 1 — UV camera port → filter mapping
No software command — physically inspect the two CSI cables and confirm which camera module carries the 310 nm filter vs 330 nm. Then update `multicam_config.toml`:
```toml
camera_1_filter_nm = 310   # whichever port has the 310 nm filter
camera_2_filter_nm = 330
```

### 2 — Repo location (needed for file-monitor.service)
```bash
find /home -name "file_monitor.py" 2>/dev/null
```
Update `ExecStart` in `systemd-units/file-monitor.service` with the actual path.

---

## Commands to run — OnLogic SBC (remaining)

### 1 — End-to-end driver tests (all cameras connected)

The config is at `~/.config/multicam_config.toml` (default path — no `-c` flag needed).
```bash
# IR camera
multicamctl capture infrared

# Visible — Arducam (model = "picam" in config)
multicamctl capture visible

# Visible — Canon (set model = "canon" in ~/.config/multicam_config.toml first)
multicamctl capture visible
```
