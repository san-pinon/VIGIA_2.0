# VIGIA 2.0 — Field Commissioning Guide

Initial tests to verify each instrument's CLI works, focus cameras, and align the
spectrometer FOV within the UV camera FOV.

---

## Output and log locations

| Instrument | Output files | Logs |
|---|---|---|
| Ultraviolet | `/home/user/data/ultraviolet/receive/` | stdout |
| Spectrometer | `/home/user/data/spectrometer/receive/` | stdout |
| Environmental | `/home/user/data/environmental/receive/` | stdout |
| Infrared | `/home/user/data/infrared/receive/` | stdout (SDK logs: `/home/user/logs/thermal/`) |
| Visible (picam) | `/home/user/data/picam/receive/` | stdout |
| UV sync | `/home/user/data/uv/` | stdout |
| Orchestrator | — | `/home/user/logs/orchestrator.log` |

Copy files to the dev machine for inspection:

```bash
# From dev machine
scp -r user@192.168.30.101:/home/user/data/ultraviolet/receive/ /tmp/uv/
scp -r user@192.168.30.150:/home/user/data/infrared/receive/   /tmp/ir/
# etc.
```

---

## 1. Environmental sensors — uvcam

Requires no pointing or focusing. Run first as a sanity check.

```bash
# On uvcam
multicamctl capture environmental
```

Check output:
```bash
ls -lh /home/user/data/environmental/receive/
cat /home/user/data/environmental/receive/<latest>.json
```

Expected: JSON with temperature, pressure, humidity from BME280 and temperature +
humidity from SHT30. If SHT30 values are absent or zero, check I2C with
`i2cdetect -y 1` (expect 0x44).

---

## 2. Spectrometer — uvcam

```bash
# On uvcam
multicamctl capture spectrometer
```

Check output:
```bash
ls -lh /home/user/data/spectrometer/receive/
```

Expected: two `.npy` files — one named `*_wavelengths.npy` (wavelength axis) and
one with the intensity spectrum.

Inspect on-site:
```bash
cd /home/user/data/spectrometer/receive/
python3 - <<'EOF'
import numpy as np, glob
files = sorted(glob.glob("*.npy"))
wl_file = next(f for f in files if "wavelengths" in f)
sp_file = next(f for f in files if "wavelengths" not in f)
w = np.load(wl_file)
s = np.load(sp_file)
print(f"Wavelengths: {w[0]:.1f} – {w[-1]:.1f} nm")
print(f"Max intensity: {s.max():.0f} / 65535")
print(f"Peak at: {w[s.argmax()]:.1f} nm")
EOF
```

**What to look for:**
- Wavelengths should span roughly 290–460 nm for the OceanHR4.
- Intensity should not be at the 65535 ceiling (saturation). If it is, reduce
  `integration_time_us` in `~/.config/multicam_config.toml` and re-run.
- If intensity is very low (<10% of max), increase `integration_time_us`.
- Target: peak intensity at roughly 50–70% of 65535 (~33000–46000 counts).

---

## 3. UV cameras — uvcam

```bash
# On uvcam
multicamctl capture ultraviolet
```

Check output:
```bash
ls -lh /home/user/data/ultraviolet/receive/
```

Expected: two TIFF files per capture — one per camera, suffixed `-310.tiff`
(port 0 = 310 nm filter) and `-330.tiff` (port 1 = 330 nm filter).

Copy to dev machine and open to check focus:
```bash
scp user@192.168.30.101:/home/user/data/ultraviolet/receive/*.tiff /tmp/uv/
```

**Focusing:**
UV lenses are narrow-band so visual focus with the naked eye is not reliable.
1. Aim both cameras at a distant target with clear edges (rooftop, ridge line,
   ideally >500 m away to approximate infinity focus).
2. Capture a frame, inspect sharpness of edges in the TIFF.
3. Adjust each lens (port 0 and port 1 separately) until edges are sharp.
4. Tighten focus lock screws.

**Exposure check:**
- Image should not be fully white (saturated) or fully black.
- Adjust `ExposureTime` (µs) in `[ultraviolet.controls]` in the config if needed.
- A good starting point outdoors under sun: 50000–200000 µs.

---

## 4. Infrared camera — multicam

```bash
# On multicam
multicamctl capture infrared
```

Check output:
```bash
ls -lh /home/user/data/infrared/receive/
cat /home/user/data/infrared/receive/*-metadata.json
```

Expected: a 16-bit TIFF thermal image, a false-colour PNG, and a JSON metadata
file (suffixed `-metadata.json`) with min/max/mean temperature values and a
saturation flag. All three live in the same `receive/` directory.

Copy to dev machine:
```bash
scp user@192.168.30.150:/home/user/data/infrared/receive/*.tiff /tmp/ir/
```

**Focusing:**
The Optris PI 640 has a fixed factory focus. Confirm the image is sharp by
pointing at a heat source (warm cup, hand) at operational distance. If the
lens is adjustable on your unit, aim at a warm target at the expected
monitoring distance.

---

## 5. Visible camera — multicam

```bash
# On multicam — picam
multicamctl capture visible
```

Check output:
```bash
ls -lh /home/user/data/picam/receive/
```

Expected: PNG image files.

Copy and inspect:
```bash
scp user@192.168.30.150:/home/user/data/picam/receive/*.* /tmp/vis/
```

**Focusing:**
1. Aim at a target at the expected scene distance.
2. Capture, inspect sharpness.
3. Adjust the Arducam IMX477 lens ring until sharp; lock.

---

## 6. Spectrometer FOV alignment within the UV camera FOV

The spectrometer fiber has a narrow acceptance angle. You need to know which
region of the UV camera images corresponds to the spectrometer's line of sight
so that SO2 column amounts from the spectrometer can be co-registered with
the UV images.

### Equipment needed
- A UV LED flashlight or UV lamp (300–365 nm range)
- A second person, or a way to mount the UV source at a fixed location

### Procedure

1. **Point all instruments at the sky** (or at the scene you will monitor).
   Make sure UV cameras are focused (step 3 above done).

2. **Run a simultaneous UV + spectrometer capture loop** at short intervals:
   ```bash
   # On uvcam — repeat manually or wrap in a short loop
   for i in $(seq 1 20); do
       multicamctl capture ultraviolet
       multicamctl capture spectrometer
       sleep 2
   done
   ```

3. **While captures run**, have a helper slowly move the UV lamp across the
   scene (left–right, then up–down) at a distance of ~5–10 m, holding it
   stationary for ~4 s at each position.

4. **Copy all captures to the dev machine:**
   ```bash
   scp -r user@192.168.30.101:/home/user/data/ultraviolet/receive/ /tmp/uv_align/
   scp -r user@192.168.30.101:/home/user/data/spectrometer/receive/ /tmp/spec_align/
   ```

5. **Find the maximum-intensity spectrum.** The spectrum with the highest peak
   corresponds to the lamp being directly in the spectrometer's FOV.
   Note its timestamp.

6. **Open the UV image with the matching timestamp.** The bright spot in the
   TIFF is where the spectrometer fiber is pointed. Mark its pixel coordinates —
   this is the spectrometer FOV centre.

7. **Estimate the FOV boundary.** Repeat with the lamp at the edges of the
   spectrometer signal drop-off to estimate the angular extent. Draw a circle
   or rectangle on the UV image to annotate the spectrometer FOV.

8. **Record the pixel coordinates** (centre + radius/bbox) in the site config
   or a calibration JSON in `/home/user/data/` for use in SO2 retrieval.

### Quick check (no helper needed)
If a helper is not available, occlude the spectrometer fiber with your hand and
run a capture. The UV images will be unchanged but the spectrum will drop to
near-zero — confirming the fiber and camera system are both active.

---

## 7. UV sync test — uvcam

Once steps 3 and 2 are verified individually, test the synchronized capture:

```bash
# On uvcam
multicamctl capture uv-sync
```

Check output:
```bash
ls -lh /home/user/data/uv/
```

Expected: two UV TIFFs, one spectrum `.npy`, and one wavelengths `.npy` all
sharing the same timestamp, plus a metadata JSON.

---

## Notes

- All output filenames follow: `{vnum}.{site_code}.{year}.{julday}_{HHMMSS}[-{frame}].{ext}`
- The config at `~/.config/multicam_config.toml` on each node controls all
  integration times, frame counts, and output directories.
- To watch captures in real time: `tail -f /home/user/logs/orchestrator.log`
  (once the orchestrator is running).
