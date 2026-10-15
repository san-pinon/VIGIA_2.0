"""
UV-sync: barrier-synchronized capture of both UV cameras and spectrometer.

This subcommand triggers both UV cameras (310 nm and 330 nm filters) and the
spectrometer simultaneously using a threading barrier. UV images come from the
cameras' 10-bit raw stream (linear), optionally binned. Each camera's exposure
and the spectrometer's integration time are adjusted independently to stay out
of saturation, and the last good values are remembered between runs.

Two modes:

- **Single shot (default):** open devices, capture (with up to
  ``--max-retries`` exposure corrections), save one file set, exit. This is
  what the orchestrator runs each cycle.
- **Monitor (``--monitor``):** keep the devices open and capture on round
  UTC multiples of ``monitor_interval_s`` (default 5 s), correcting exposure
  from one capture to the next. Devices are closed while the sun is below
  ``min_sun_elevation_deg`` (when the site coordinates are configured). Stops
  cleanly on SIGTERM/SIGINT; run as ``capture-uv-monitor.service``.

``--kind dark|clear`` tags a capture set (filenames and metadata). Dark
captures keep the exposure fixed — cover the lenses/fibre first.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pathlib
import signal
import sys
import time
from collections.abc import Callable
from datetime import UTC
from datetime import datetime as dt

import cv2
import numpy as np

from multicam.drivers import CaptureResult
from multicam.drivers.spectrometer.oceaninsight import (
    Spectrometer,
    check_spectrum_saturation,
    configured_integration_ms,
)
from multicam.drivers.ultraviolet.picam import (
    DualCamera,
    configured_exposure_ms,
    quicklook,
    raw_brightness,
)
from multicam.errors import CaptureFailure
from multicam.utilities.naming import build_filename
from multicam.utilities.sun import solar_elevation_deg

from .sync import synchronized_capture

logger = logging.getLogger(__name__)

# Spectrometer integration time limits (ms) for OceanHR4
SPEC_INT_MIN_MS = 3.8
SPEC_INT_MAX_MS = 10_000.0

# Brightness band (fraction of the raw range above black level, 99.9th
# percentile) a UV image must fall in, and the level the auto-adjust aims for.
UV_MIN_BRIGHTNESS = 0.2
UV_MAX_BRIGHTNESS = 0.8
UV_TARGET_BRIGHTNESS = 0.6
# More than this fraction of raw pixels at full scale counts as clipped; above
# UV_HEAVILY_CLIPPED_FRACTION the exposure is quartered instead of halved.
UV_CLIPPED_FRACTION = 0.001
UV_HEAVILY_CLIPPED_FRACTION = 0.1

KINDS = ("plume", "dark", "clear")
DEFAULT_MONITOR_INTERVAL_S = 5.0
DEFAULT_MIN_SUN_ELEVATION_DEG = 5.0


def _next_uv_exposure(
    exposure_ms: float, brightness: float, clipped_fraction: float = 0.0
) -> float | None:
    """Exposure (ms) that brings ``brightness`` into band, or None if in band."""

    if clipped_fraction > UV_CLIPPED_FRACTION:
        # Clipped: the true brightness is unknown, so step down instead of scaling.
        return exposure_ms * (
            0.25 if clipped_fraction > UV_HEAVILY_CLIPPED_FRACTION else 0.5
        )
    if UV_MIN_BRIGHTNESS <= brightness <= UV_MAX_BRIGHTNESS:
        return None
    factor = UV_TARGET_BRIGHTNESS / max(brightness, 0.01)
    return exposure_ms * min(4.0, max(0.25, factor))


def _read_uv_state(state_path: pathlib.Path) -> dict:
    """Read the persisted exposure/integration state, or {} if unavailable."""

    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write(path: pathlib.Path, write: Callable[[pathlib.Path], None]) -> None:
    """
    Write via a temp name then ``os.replace`` into place, so the file monitor
    only ever sees complete files and a kill mid-write leaves no partial file.
    """

    # Hidden temp name: the file monitor skips dot-files and *.tmp.
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


class _UVSession:
    """Devices, exposure state and output writing shared by both modes."""

    def __init__(self, config: dict, flags: argparse.Namespace) -> None:
        self.uv_config = config["ultraviolet"]
        self.spec_config = config["spectrometer"]
        self.sync_config = config.get("uv_sync", {})
        self.metadata_cfg = config["metadata"]
        self.kind = flags.kind
        self.use_spec = not flags.no_spectrometer
        self.write_quicklook = bool(self.uv_config.get("quicklook_png", True))
        self.stack_count = flags.stack or int(self.sync_config.get("default_stack", 5))

        self.output_dir = pathlib.Path(
            flags.output_dir
            or self.sync_config.get("output_dir")
            or (self.metadata_cfg["data_archive"] + "/ultraviolet")
        )
        self.spec_output_dir = pathlib.Path(
            self.metadata_cfg["data_archive"] + "/spectrometer"
        )
        (self.output_dir / "receive").mkdir(parents=True, exist_ok=True)
        (self.spec_output_dir / "receive").mkdir(parents=True, exist_ok=True)

        self.filters = {
            n: self.uv_config.get(f"camera_{n}_filter_nm", 310 if n == 1 else 330)
            for n in (1, 2)
        }

        # Exposure state persisted between runs so each run starts from the
        # last good value instead of the config default. A saved value is only
        # reused while the config value it started from is unchanged.
        self.state_path = pathlib.Path(
            self.sync_config.get("state_file")
            or (self.output_dir / ".uv_sync_state.json")
        )
        saved = _read_uv_state(self.state_path)
        self.config_exposure = {
            n: configured_exposure_ms(self.uv_config, n) for n in (1, 2)
        }
        self.config_spec_ms = configured_integration_ms(self.spec_config)

        # Values worth remembering: start from what was saved (if still valid)
        # and replace only with exposures that actually landed in band.
        self.good_exposure: dict[int, float | None] = {}
        self.start_exposure: dict[int, tuple[float, str]] = {}
        for n in (1, 2):
            saved_ms = saved.get(f"uv{n}_exposure_ms")
            if (
                saved_ms
                and saved.get(f"uv{n}_config_exposure_ms") == self.config_exposure[n]
            ):
                self.good_exposure[n] = float(saved_ms)
            else:
                self.good_exposure[n] = None
            cli_ms = getattr(flags, f"uv{n}_exposure_ms")
            if cli_ms is not None:
                self.start_exposure[n] = (cli_ms, "CLI")
            elif self.good_exposure[n] is not None:
                self.start_exposure[n] = (self.good_exposure[n], "last run")
            else:
                self.start_exposure[n] = (self.config_exposure[n], "config")

        saved_spec = saved.get("spec_integration_ms")
        if (
            saved_spec
            and saved.get("spec_config_integration_ms") == self.config_spec_ms
        ):
            self.good_spec_ms: float | None = float(saved_spec)
        else:
            self.good_spec_ms = None
        if flags.spec_integration_ms is not None:
            spec_start, self.spec_source = flags.spec_integration_ms, "CLI"
        elif self.good_spec_ms is not None:
            spec_start, self.spec_source = self.good_spec_ms, "last run"
        else:
            spec_start, self.spec_source = self.config_spec_ms, "config"
        self.spec_ms = max(SPEC_INT_MIN_MS, min(SPEC_INT_MAX_MS, spec_start))

        self.cameras: DualCamera | None = None
        self.spectrometer: Spectrometer | None = None

    # --- devices -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self.cameras is not None

    def open(self) -> None:
        """Open cameras (+ spectrometer) and apply the starting settings."""

        if self.use_spec:
            print("Initialising UV cameras and spectrometer...")
        else:
            print("Initialising UV cameras (spectrometer disabled)...")
        self.cameras = DualCamera(self.uv_config)
        try:
            if self.use_spec:
                self.spectrometer = Spectrometer(self.spec_config)
            for n in (1, 2):
                start_ms, source = self.start_exposure[n]
                if start_ms != self.cameras.exposure_ms[n]:
                    self.cameras.set_exposure(n, start_ms)
                # Make sure manual exposure/gain are in effect before capturing.
                self.cameras.settle_exposure(n)
                print(
                    f"   UV camera_{n} ({self.filters[n]} nm): starting at "
                    f"{self.cameras.exposure_ms[n]:.3f} ms ({source}), "
                    f"gain {self.cameras.analogue_gain}, "
                    f"binning {self.cameras.binning}, "
                    f"raw {self.cameras.raw_format[n]}"
                )
            if self.spectrometer is not None:
                self.spectrometer.integration_time_ms = self.spec_ms
                print(
                    f"   Spectrometer: starting at {self.spec_ms:.3f} ms "
                    f"({self.spec_source})"
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.spectrometer is not None:
            self.spectrometer.shutdown()
            self.spectrometer = None
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None

    # --- capture + exposure control ---------------------------------------

    def capture(self) -> tuple[CaptureResult, CaptureResult, CaptureResult | None]:
        return synchronized_capture(
            self.cameras,
            self.spectrometer,
            stack_count=self.stack_count,
            spec_config={
                "integration_time_ms": self.spec_ms,
                "stacking": self.stack_count,
            },
        )

    def describe_settings(self) -> str:
        spec = (
            f"spec int={self.spec_ms:.3f} ms"
            if self.spectrometer is not None
            else "no spectrometer"
        )
        return (
            f"UV {self.filters[1]} nm={self.cameras.exposure_ms[1]:.3f} ms, "
            f"{self.filters[2]} nm={self.cameras.exposure_ms[2]:.3f} ms, "
            f"{spec}, stack={self.stack_count}"
        )

    def adjust(
        self,
        uv1: CaptureResult,
        uv2: CaptureResult,
        spec: CaptureResult | None,
        settle: bool,
        verbose: bool = True,
    ) -> bool:
        """
        Measure each capture and set new exposures for the next one.

        Annotates each UV result's metadata with ``brightness`` and
        ``exposure_status`` (ok / under / over / clipped / limit / fixed).
        Returns True if any exposure was changed (another try is useful).
        """

        changed = []
        for n, result in ((1, uv1), (2, uv2)):
            md = result.metadata
            brightness = raw_brightness(result)
            clipped = md["saturated_fraction"] > UV_CLIPPED_FRACTION
            md["brightness"] = round(brightness, 4)
            # Scale from the exposure this frame was actually taken at.
            frame_ms = md["exposure_ms_actual"] or md["exposure_ms_requested"]

            if self.kind == "dark":
                md["exposure_status"] = "fixed"
                continue
            new_ms = _next_uv_exposure(frame_ms, brightness, md["saturated_fraction"])
            if new_ms is None:
                md["exposure_status"] = "ok"
                self.good_exposure[n] = frame_ms
                continue
            md["exposure_status"] = (
                "clipped"
                if clipped
                else ("over" if brightness > UV_MAX_BRIGHTNESS else "under")
            )
            old_ms = self.cameras.exposure_ms[n]
            new_ms = self.cameras.set_exposure(n, new_ms)
            if new_ms == old_ms:
                md["exposure_status"] = "limit"
                if verbose:
                    print(
                        f"   WARNING: UV camera_{n} ({self.filters[n]} nm) at exposure "
                        f"limit {old_ms:.3f} ms, brightness {brightness:.2f}"
                    )
                continue
            changed.append(n)
            if verbose:
                print(
                    f"   UV camera_{n} ({self.filters[n]} nm) brightness "
                    f"{brightness:.2f}{' (clipped)' if clipped else ''} — exposure "
                    f"{old_ms:.3f} → {new_ms:.3f} ms"
                )
        if settle:
            for n in changed:
                self.cameras.settle_exposure(n)

        spec_changed = False
        if spec is not None and self.kind != "dark":
            spec_sat = check_spectrum_saturation(
                spec.artifacts["spectrum"],
                self.spectrometer,
                min_saturation=0.2,
                max_saturation=0.8,
            )
            if spec_sat == 0:
                self.good_spec_ms = self.spec_ms
            else:
                factor = 0.75 if spec_sat == -1 else 1.25
                new_ms = max(
                    SPEC_INT_MIN_MS, min(SPEC_INT_MAX_MS, self.spec_ms * factor)
                )
                if new_ms == self.spec_ms:
                    if verbose:
                        print(
                            "   WARNING: Spectrum "
                            f"{'saturated' if spec_sat == -1 else 'under-exposed'} "
                            "at integration limit"
                        )
                else:
                    if verbose:
                        print(
                            "   Spectrum "
                            f"{'over' if spec_sat == -1 else 'under'}-exposed — "
                            f"integration {self.spec_ms:.3f} → {new_ms:.3f} ms"
                        )
                    self.spec_ms = new_ms
                    self.spectrometer.integration_time_ms = new_ms
                    spec_changed = True

        return bool(changed) or spec_changed

    # --- output ------------------------------------------------------------

    def save(
        self,
        uv1: CaptureResult,
        uv2: CaptureResult,
        spec: CaptureResult | None,
        timestamp: dt,
        extra: dict,
    ) -> dict:
        """Write images, quick-looks, spectrum and metadata; return the file names."""

        kind_sfx = "" if self.kind == "plume" else f"-{self.kind}"
        receive = self.output_dir / "receive"
        files: dict[str, str | None] = {}

        for label, result in (("uv_a", uv1), ("uv_b", uv2)):
            md = result.metadata
            stem = f"uv-{md['filter_nm']}{kind_sfx}"
            image = result.artifacts["image"]

            npz_name = build_filename(
                self.metadata_cfg, timestamp, suffix=stem, extension="npz"
            )

            def _write_npz(tmp, image=image, md=md):
                with open(tmp, "wb") as fh:
                    np.savez(
                        fh,
                        image=image,
                        black_level_dn=md["black_level_dn"],
                        exposure_ms=md["exposure_ms_actual"],
                        binning=md["binning"],
                    )

            _atomic_write(receive / npz_name, _write_npz)
            files[label] = npz_name

            if self.write_quicklook:
                png_name = build_filename(
                    self.metadata_cfg, timestamp, suffix=stem, extension="png"
                )
                preview = quicklook(image, md["black_level_dn"])

                def _write_png(tmp, preview=preview):
                    ok, buf = cv2.imencode(".png", preview)
                    if not ok:
                        raise OSError("PNG encoding failed")
                    tmp.write_bytes(buf.tobytes())

                _atomic_write(receive / png_name, _write_png)
                files[f"{label}_quicklook"] = png_name

        files["spectrum"] = None
        if spec is not None:
            spec_name = build_filename(
                self.metadata_cfg,
                timestamp,
                suffix=f"spectrum{kind_sfx}",
                extension="csv",
            )
            wavelengths = spec.metadata.get("wavelengths", [])
            intensities = spec.artifacts["spectrum"]

            def _write_csv(tmp):
                with open(tmp, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["wavelength_nm", "intensity"])
                    for wl, inten in zip(wavelengths, intensities, strict=True):
                        writer.writerow([round(wl, 4), round(float(inten), 4)])

            _atomic_write(self.spec_output_dir / "receive" / spec_name, _write_csv)
            files["spectrum"] = spec_name

        meta_name = build_filename(
            self.metadata_cfg,
            timestamp,
            suffix=f"metadata{kind_sfx}",
            extension="json",
        )
        files["metadata"] = meta_name
        meta_payload = {
            "timestamp_utc": timestamp.isoformat(),
            "kind": self.kind,
            **extra,
            "uv_a_exposure_ms": uv1.metadata["exposure_ms_actual"],
            "uv_b_exposure_ms": uv2.metadata["exposure_ms_actual"],
            "uv_analogue_gain": self.cameras.analogue_gain,
            "spectrometer_integration_time_ms": (
                spec.metadata["integration_time_ms"] if spec else None
            ),
            "stack_count": self.stack_count if spec else None,
            "uv_camera_a_filter_nm": uv1.metadata["filter_nm"],
            "uv_camera_b_filter_nm": uv2.metadata["filter_nm"],
            "uv_camera_a": uv1.metadata,
            "uv_camera_b": uv2.metadata,
            "spectrum_mean_intensity": (
                round(float(np.mean(spec.artifacts["spectrum"])), 2) if spec else None
            ),
            "spectrum_max_intensity": (
                round(float(np.max(spec.artifacts["spectrum"])), 2) if spec else None
            ),
            "files": files,
        }
        _atomic_write(
            receive / meta_name,
            lambda tmp: tmp.write_text(json.dumps(meta_payload, indent=2)),
        )
        return files

    def persist(self, timestamp: dt) -> None:
        """Save the last *good* exposures so the next run starts from them."""

        payload: dict = {}
        for n in (1, 2):
            if self.good_exposure[n] is not None:
                payload[f"uv{n}_exposure_ms"] = round(self.good_exposure[n], 3)
                payload[f"uv{n}_config_exposure_ms"] = self.config_exposure[n]
        if self.good_spec_ms is not None:
            payload["spec_integration_ms"] = round(self.good_spec_ms, 3)
            payload["spec_config_integration_ms"] = self.config_spec_ms
        payload["updated_utc"] = timestamp.isoformat()
        try:
            _atomic_write(
                self.state_path,
                lambda tmp: tmp.write_text(json.dumps(payload, indent=2)),
            )
        except OSError as e:
            logger.warning("Could not write UV state file %s: %s", self.state_path, e)


def _parse_flags(extra_args: list[str] | None, sync_config: dict) -> argparse.Namespace:
    flag_parser = argparse.ArgumentParser(add_help=False)
    flag_parser.add_argument("--check-saturation", action="store_true", default=True)
    flag_parser.add_argument(
        "--max-retries", type=int, default=int(sync_config.get("max_retries", 3))
    )
    flag_parser.add_argument("--stack", type=int, default=None)
    flag_parser.add_argument("--output-dir", type=str, default=None)
    flag_parser.add_argument("--uv1-exposure-ms", type=float, default=None)
    flag_parser.add_argument("--uv2-exposure-ms", type=float, default=None)
    flag_parser.add_argument("--spec-integration-ms", type=float, default=None)
    flag_parser.add_argument("--no-spectrometer", action="store_true")
    flag_parser.add_argument("--kind", choices=KINDS, default="plume")
    flag_parser.add_argument("--monitor", action="store_true")
    flag_parser.add_argument(
        "--interval-s",
        type=float,
        default=float(
            sync_config.get("monitor_interval_s", DEFAULT_MONITOR_INTERVAL_S)
        ),
    )
    return flag_parser.parse_args(extra_args or [])


def capture_uv_sync(config: dict, extra_args: list[str] | None = None) -> None:
    """
    Handle ``multicamctl capture uv-sync``.

    Parameters
    ----------
    config:
        System configuration dictionary (parsed from TOML).
    extra_args:
        Additional CLI flags (see module docstring).

    """

    flags = _parse_flags(extra_args, config.get("uv_sync", {}))
    session = _UVSession(config, flags)
    if flags.monitor:
        _run_monitor(session, flags)
    else:
        _run_single(session, flags)


def _run_single(session: _UVSession, flags: argparse.Namespace) -> None:
    """One capture set, with up to ``--max-retries`` exposure corrections."""

    session.open()
    try:
        uv1 = uv2 = spec = None
        capture_timestamp = None
        retries = 0
        for attempt in range(flags.max_retries + 1):
            print(f"   Capture attempt {attempt + 1} ({session.describe_settings()})")
            try:
                uv1, uv2, spec = session.capture()
                capture_timestamp = dt.now(UTC)
            except CaptureFailure as e:
                print(f"   Capture failed: {e}")
                if attempt < flags.max_retries:
                    retries += 1
                    continue
                raise

            if not flags.check_saturation:
                break
            if not session.adjust(uv1, uv2, spec, settle=True):
                break
            if attempt < flags.max_retries:
                retries += 1
            else:
                print("   Max retries reached — saving current captures")

        statuses = {
            n: r.metadata.get("exposure_status", "unchecked")
            for n, r in ((1, uv1), (2, uv2))
        }
        if all(s in ("ok", "fixed", "unchecked") for s in statuses.values()):
            print(
                "   Exposure adequate" if session.kind != "dark" else "   Dark capture"
            )
        else:
            print(
                "   Saving with exposure not in range: "
                + ", ".join(
                    f"camera_{n} ({session.filters[n]} nm) {s}"
                    for n, s in statuses.items()
                    if s not in ("ok", "fixed")
                )
            )

        files = session.save(
            uv1,
            uv2,
            spec,
            capture_timestamp,
            {"mode": "single", "saturation_retries": retries},
        )
        for name in files.values():
            if name:
                print(f"   Saved: {name}")
        if session.kind != "dark":
            session.persist(capture_timestamp)

        print("...uv-sync capture complete.")
        print(
            json.dumps(
                {
                    "instrument": "uv-sync",
                    "timestamp_utc": capture_timestamp.isoformat(),
                    "kind": session.kind,
                    "saturation_retries": retries,
                    "files": files,
                }
            )
        )
        sys.exit(0)
    finally:
        session.close()


def _run_monitor(session: _UVSession, flags: argparse.Namespace) -> None:
    """
    Keep the devices open and capture on round UTC multiples of the interval.

    Exposure is corrected from one capture to the next (no retries within a
    tick). Ticks that are missed because a capture overran are skipped, not
    queued, so captures stay phase-aligned with the other instruments.
    """

    interval = float(flags.interval_s)
    if interval <= 0:
        raise ValueError("--interval-s must be positive")

    lat = session.sync_config.get("latitude")
    lon = session.sync_config.get("longitude")
    min_elev = float(
        session.sync_config.get("min_sun_elevation_deg", DEFAULT_MIN_SUN_ELEVATION_DEG)
    )
    gate = lat is not None and lon is not None
    if not gate:
        print(
            "   WARNING: [uv_sync] latitude/longitude not set — daylight gate "
            "disabled, capturing day and night"
        )

    stop = {"flag": False}

    def _handle_stop(signum, _frame) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    print(
        f"   ...entering UV monitor loop (every {interval:g} s on the UTC clock, "
        f"kind={session.kind}"
        + (f", daylight gate: sun ≥ {min_elev:g}°" if gate else "")
        + ")..."
    )

    failures = 0
    night = False
    try:
        while not stop["flag"]:
            tick_epoch = (math.floor(time.time() / interval) + 1) * interval
            while not stop["flag"]:
                remaining = tick_epoch - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.5))
            if stop["flag"]:
                break
            tick = dt.fromtimestamp(tick_epoch, UTC)

            if gate:
                elevation = solar_elevation_deg(tick, float(lat), float(lon))
                if elevation < min_elev:
                    if not night:
                        print(
                            f"[{tick:%Y-%m-%d %H:%M:%S}] sun at {elevation:.1f}° "
                            f"(< {min_elev:g}°) — closing devices until daylight"
                        )
                        night = True
                    session.close()
                    continue
                if night:
                    print(
                        f"[{tick:%Y-%m-%d %H:%M:%S}] sun at {elevation:.1f}° — resuming"
                    )
                    night = False

            if not session.is_open:
                try:
                    session.open()
                except Exception as e:
                    print(f"   ERROR opening devices: {e} — retrying in 30 s")
                    for _ in range(60):
                        if stop["flag"]:
                            break
                        time.sleep(0.5)
                    continue
                # The start-up settle took a while; restart on the next tick.
                continue

            try:
                uv1, uv2, spec = session.capture()
            except CaptureFailure as e:
                failures += 1
                print(f"[{tick:%H:%M:%S}] capture failed ({failures}): {e}")
                if failures >= 3:
                    print("   ...3 failures in a row — reopening devices")
                    session.close()
                    failures = 0
                continue
            failures = 0
            completed = dt.now(UTC)

            session.adjust(uv1, uv2, spec, settle=False, verbose=False)
            files = session.save(
                uv1,
                uv2,
                spec,
                tick,
                {
                    "mode": "monitor",
                    "interval_s": interval,
                    "capture_completed_utc": completed.isoformat(),
                },
            )
            session.persist(tick)

            def _cam(r: CaptureResult) -> str:
                m = r.metadata
                return (
                    f"{m['filter_nm']}: {m['exposure_ms_actual']:.1f} ms "
                    f"b={m['brightness']:.2f} {m['exposure_status']}"
                )

            elapsed = time.time() - tick_epoch
            print(
                f"[{tick:%H:%M:%S}] {_cam(uv1)} | {_cam(uv2)} | "
                f"{'spec ' + files['spectrum'] if files['spectrum'] else 'no spectrum'}"
                f" | {elapsed:.1f} s"
                + (
                    f"  WARNING: overran {interval:g} s interval"
                    if elapsed > interval
                    else ""
                )
            )
    finally:
        session.close()
        print("...UV monitor stopped.")
