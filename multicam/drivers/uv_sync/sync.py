"""
Barrier-synchronized capture of two UV cameras and a spectrometer.

Uses ``threading.Barrier(3)`` so all three devices trigger as close
together as possible.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import logging
import threading
from typing import Any

from multicam.drivers import CaptureResult
from multicam.errors import CaptureFailure

logger = logging.getLogger(__name__)


def synchronized_capture(
    cameras,
    spectrometer,
    stack_count: int = 10,
    spec_config: dict[str, Any] | None = None,
    spec_optional: bool = False,
) -> tuple[CaptureResult, CaptureResult, CaptureResult | None]:
    """
    Barrier-synchronized capture of both UV cameras and the spectrometer.

    All acquisitions wait on a ``threading.Barrier`` before triggering, so
    they fire as close together as possible. Pass ``spectrometer=None`` to
    capture the two UV cameras only.

    Parameters
    ----------
    cameras:
        Initialised ``DualCamera`` (from ultraviolet driver).
    spectrometer:
        Initialised ``Spectrometer`` (from spectrometer driver), or ``None``.
    stack_count:
        Number of spectra to co-add.
    spec_config:
        Spectrometer settings dict (passed to ``spectrometer.capture``).
    spec_optional:
        A failed spectrum doesn't fail the capture: the UV frames are still
        returned, with ``None`` for the spectrum.

    Returns
    -------
    uv1_result, uv2_result, spec_result:
        Capture results for camera 1, camera 2, and the spectrometer
        (``None`` when no spectrometer was given, or ``spec_optional`` and the
        spectrum failed).

    """

    from multicam.drivers.spectrometer.oceaninsight import (
        capture as capture_spectrum,
    )

    use_spec = spectrometer is not None
    barrier = threading.Barrier(3 if use_spec else 2, timeout=30)
    results: dict[str, Any] = {}
    errors: dict[str, Exception] = {}

    def _capture_uv(n: int):
        try:
            barrier.wait()
            results[f"uv{n}"] = cameras.capture_one(n)
        except Exception as e:
            errors[f"uv{n}"] = e

    def _capture_spec():
        try:
            barrier.wait()
            settings = dict(spec_config or {})
            settings["stacking"] = stack_count
            result = capture_spectrum(spectrometer, settings)
            results["spec"] = result
        except Exception as e:
            errors["spec"] = e

    threads = [
        threading.Thread(target=_capture_uv, args=(1,), name="uv1"),
        threading.Thread(target=_capture_uv, args=(2,), name="uv2"),
    ]
    if use_spec:
        # Daemon: a hung spectrometer must not keep the process alive.
        threads.append(threading.Thread(target=_capture_spec, name="spec", daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    if use_spec and spec_optional and ("spec" in errors or "spec" not in results):
        reason = errors.pop("spec", "no result in time")
        print(f"   WARNING: spectrum failed, keeping the UV frames: {reason}")
        use_spec = False
        results.pop("spec", None)

    if errors:
        msgs = [f"{k}: {v}" for k, v in errors.items()]
        raise CaptureFailure(f"Synchronized capture failed: {'; '.join(msgs)}")

    for key in ("uv1", "uv2", "spec") if use_spec else ("uv1", "uv2"):
        if key not in results:
            raise CaptureFailure(f"Synchronized capture: missing result for {key}")

    return results["uv1"], results["uv2"], results.get("spec")
