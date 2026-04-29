# multicamctl — CLI & Orchestrator Implementation Instructions

## Overview

Two computers connected via a local router (192.168.30.0/24):

| Host | Hostname | IP | Role |
|---|---|---|---|
| Raspberry Pi 5 | `uvcam` | 192.168.30.101 | UV cameras + spectrometer + env. sensors + GPS |
| OnLogic SBC | `multicam` | 192.168.30.150 | IR camera + Canon DSLR + picam |

Both machines run a CLI called `multicamctl` installed in `/home/user/.venv/bin/multicamctl`.  
The orchestrator script runs on `multicam` and coordinates captures across both machines via SSH.

### multicamctl subcommands to implement for `uvcam`

---

#### `multicamctl capture uv-sync`

Simultaneously captures one frame from both UV cameras (ports 0 and 1) and one spectrum from the spectrometer. All three must be triggered as close together as possible using a `threading.Barrier(3)` so all three threads are armed before any fires. Use `capture ultraviolet` and `capture spectrometer`

**Flags:**
- `--check-saturation` — after capture, check if either camera image or the spectrum is saturated. If saturated, adjust and retry.
- `--max-retries N` — maximum number of re-capture attempts when saturation is detected (default: 3)
- `--stack N` — number of spectra to co-add before returning the final spectrum (default: 10). Spectrum stacking must happen within the same trigger cycle: the spectrometer captures N integrations and averages them before saving.
- `--output-dir PATH` — directory to save outputs (default: `/home/user/data/uv/`)

**Saturation logic for UV cameras:**
- Compute the 95th percentile pixel value of each captured frame
- Target: 95th percentile should be ≤ 80% of max pixel value (i.e., ≤ 204 for 8-bit, scale accordingly)
- If saturated: reduce integration/exposure time by 25% and retry
- If underexposed (95th percentile < 20% of max): increase exposure by 25% and retry
- Track current exposure time in state between retries

**Saturation logic for spectrometer:**
- Check max intensity value against `max_intensity = 65535.0`
- Target: max intensity ≤ 80% of `max_intensity` (≤ 52428)
- If saturated: reduce integration time by 25% and retry
- Integration time must stay within `3800 – 10_000_000 µs`
- If at minimum integration time and still saturated: save frame anyway, log a warning

**Spectrum stacking:**
- Capture N spectra at the resolved integration time
- Average them element-wise into a single 1D array
- Save the stacked spectrum, not individual ones
- Log: number of stacks, integration time used, mean and max intensity

**GPS timestamping:**
- Read a GPS fix
- Attach GPS timestamp (UTC) to every saved file's metadata/filename
- If no GPS fix available within 5 seconds, fall back to system UTC time and log a warning

**Output files per capture cycle:**
- `uv_A_<timestamp>.tiff` — camera port 0 image
- `uv_B_<timestamp>.tiff` — camera port 1 image  
- `spectrum_<timestamp>.csv` — wavelength + intensity columns, stacked
- `metadata_<timestamp>.json` — GPS timestamp, integration times used, stack count, saturation retries, camera exposures


### multicamctl subcommands to implement for `multicam`

---

#### `multicamctl capture dslr --meter-with arducam`

Uses picam in SBC to meter the scene and then fires the Canon DSLR with computed settings.
You may merge what has already been developed in `capture visible` to add a flag `--camera CAM` where CAM=canon/picam and thus capture an image from the picam or canon depending on it. `capture dslr` may retrieve that code in the workflow to have less dead code. You should save the final images of the picam and the canon dsrl. Save last picam frame taken at `/home/user/data/picam/`

**Metering logic (histogram-based):**
1. Capture a frame from the picam at its current settings
2. Convert to grayscale (or use luminance channel)
3. Compute the 95th percentile pixel value of the frame
4. Target: 95th percentile = 80% of 255 (= ~204)
5. Compute `exposure_ratio = 204 / p95`
6. Apply ratio to current shutter speed first; clamp to EOS valid range
7. If shutter hits its limit (too dark), step up ISO to the next valid EOS ISO value
8. If shutter hits its minimum (too bright), step down ISO
9. Iterate metering loop up to `--max-meter-iterations N` (default: 5) until p95 is within ±10% of target
10. Once settings are stable, fire Canon with the computed ISO + shutter
11. Save the last picam image used for metering (discard the rest) and the canon image.

**Flags:**
- `--meter-with picam` — use picam for metering (required for auto mode)
- `--iso VALUE` — override ISO manually (skips metering)
- `--shutter VALUE` — override shutter manually (e.g. `1/60`)
- `--check-saturation` — verify Canon output is not blown; retry if so
- `--max-retries N` — max retries on saturation (default: 3)
- `--output-dir PATH` — directory to save canon (default: `/home/user/data/dslr/`)

**Valid EOS 4000D ISO values:** 100, 200, 400, 800, 1600, 3200, 6400  
**Valid EOS 4000D shutter range:** 1/4000 to 30 seconds

**Output:**
- Raw file in data archive
- `metadata_<timestamp>.json` — ISO used, shutter used, metering iterations, picam p95 value

---

#### `multicamctl capture infrared`

Modifications for already developed driver.

**Flags:**
- `--check-saturation` — check if the thermal image has pixels at or above the sensor's max temperature range; log a warning if so (IR cameras have a fixed integration, so no retry — just flag)
- `--output-dir PATH` — default: `/home/user/logs/thermal/`

**Saturation check for IR:**
- The Optris PI has a defined temperature range. If more than 5% of pixels are at or above `T_max` for the configured range, log a `THERMAL_SATURATION_WARNING` in metadata.
- Save the frame regardless.

**Output:**
- Thermal image as 16-bit TIFF (temperature-mapped)
- `metadata_<timestamp>.json` — min/max/mean temperature, saturation flag, timestamp

---

## Orchestrator Script

**File:** `orchestrator.py`  
**Runs on:** `uvcam`
**Language:** Python 3.11

### Behavior

Runs a continuous acquisition loop. Every **60 seconds**, trigger one full capture cycle across all instruments in this order:

1. **UV sync** — Run locally: `multicamctl capture uv-sync --check-saturation --max-retries 3 --stack 10`
2. **IR capture** — SSH into `multicam` and run: `multicamctl capture infrared --check-saturation --max-retries 3`
3. **DSLR capture** — SSH into `multicam` and run: `multicamctl capture dslr --meter-with arducam --check-saturation --max-retries 3`
4. **Environmental sensor** - Run locally: `multicamctl capture environmental`

Steps 2 and 3 can run in parallel (both local to `multicam`). Step 1 (UV sync on `uvcam`) and 4 can also run in parallel with steps 2 and 3 since it is on a separate machine. Use `threading` to fire all four concurrently and `join` before the next cycle starts.

### SSH to `multicam`

Use `paramiko` or `subprocess` with SSH keys (passwordless SSH must be configured between `multicam` and `uvcam`). The SSH call should:
- Connect to `192.168.30.150` as user `user`
- Run the `multicamctl` command in the remote venv: `/home/user/.venv/bin/multicamctl capture infrared ...`
- Capture stdout/stderr and log them locally

### Timing

- Use a **drift-corrected loop**: record the wall-clock time at the start of each cycle and sleep for `60 - elapsed` seconds, so cycles don't drift over time
- If a cycle takes longer than 60 seconds (e.g. max retries hit), log a warning and start the next cycle immediately
- For uv-sync, consider the timestamp only to capture images when they are in the interval of: 5:00am until 6:pm (local time. System will be installed in Mexico, so UTC-6.).

### Logging

- All orchestrator activity logged to `/home/user/logs/orchestrator.log`
- Each cycle logs: cycle number, UTC timestamp, duration, success/failure per instrument
- Instrument-level errors must not crash the orchestrator — catch exceptions per instrument, log them, and continue

### Flags / CLI interface for the orchestrator itself

```
python orchestrator.py [OPTIONS]

  --interval SECONDS     Capture interval in seconds (default: 60)
  --output-dir PATH      Root output directory (default: /home/user/data/)
  --uvcam-host IP        IP of uvcam (default: 192.168.30.101)
  --uvcam-user USER      SSH user on uvcam (default: user)
  --no-uv                Skip UV+spectrometer captures
  --no-ir                Skip IR captures
  --no-dslr              Skip DSLR captures
  --dry-run              Print commands without executing
```

---

## General Implementation Notes

- All `multicamctl` subcommands should return **exit code 0** on success and non-zero on failure
- All subcommands should print a **JSON summary to stdout** on completion (timestamp, files saved, settings used, any warnings) so the orchestrator can parse it
- Timestamps: always UTC, format `YYYYMMDD_HHMMSS` for filenames, ISO 8601 for metadata JSON
- All integration/exposure times and saturation thresholds should be configurable via a `multicam_config.toml`.
