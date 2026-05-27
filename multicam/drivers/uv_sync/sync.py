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
) -> tuple[CaptureResult, CaptureResult]:
    """
    Barrier-synchronized capture of both UV cameras and the spectrometer.

    All three acquisitions wait on a ``threading.Barrier(3)`` before
    triggering, so they fire as close together as possible.

    Parameters
    ----------
    cameras:
        Initialised ``DualCamera`` (from ultraviolet driver).
    spectrometer:
        Initialised ``Spectrometer`` (from spectrometer driver).
    stack_count:
        Number of spectra to co-add.
    spec_config:
        Spectrometer settings dict (passed to ``spectrometer.capture``).

    Returns
    -------
    uv1_result, spec_result:
        Capture results for camera 1 and the spectrometer.

    """

    import time

    from multicam.drivers.spectrometer.oceaninsight import (
        capture as capture_spectrum,
    )

    # TODO: restore barrier to 3 and re-enable uv2 thread once replacement
    #       OV5647 is ready
    barrier = threading.Barrier(2, timeout=30)
    results: dict[str, Any] = {}
    errors: dict[str, Exception] = {}

    def _capture_uv1():
        try:
            barrier.wait()
            yuv = cameras.camera_1.capture_array("main")
            image = yuv[:1944, :2592]  # Y plane (grayscale luminance)
            ts = time.monotonic_ns()
            results["uv1"] = CaptureResult(
                metadata={
                    "ts_monotonic_ns": ts,
                    "port": cameras._config.get("camera_1_port"),
                    "filter_nm": cameras._config.get("camera_1_filter_nm", 310),
                    "stream": "main",
                    "bit_depth": 8,
                    "shape": tuple(image.shape),
                    "dtype": str(image.dtype),
                },
                artifacts={"image": image},
            )
        except Exception as e:
            errors["uv1"] = e

    # def _capture_uv2():
    #     try:
    #         barrier.wait()
    #         yuv = cameras.camera_2.capture_array("main")
    #         image = yuv[:1944, :2592]
    #         ts = time.monotonic_ns()
    #         results["uv2"] = CaptureResult(
    #             metadata={
    #                 "ts_monotonic_ns": ts,
    #                 "port": cameras._config.get("camera_2_port"),
    #                 "filter_nm": cameras._config.get("camera_2_filter_nm", 330),
    #                 "stream": "main",
    #                 "bit_depth": 8,
    #                 "shape": tuple(image.shape),
    #                 "dtype": str(image.dtype),
    #             },
    #             artifacts={"image": image},
    #         )
    #     except Exception as e:
    #         errors["uv2"] = e

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
        threading.Thread(target=_capture_uv1, name="uv1"),
        # threading.Thread(target=_capture_uv2, name="uv2"),
        threading.Thread(target=_capture_spec, name="spec"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    if errors:
        msgs = [f"{k}: {v}" for k, v in errors.items()]
        raise CaptureFailure(f"Synchronized capture failed: {'; '.join(msgs)}")

    for key in ("uv1", "spec"):
        if key not in results:
            raise CaptureFailure(f"Synchronized capture: missing result for {key}")

    return results["uv1"], results["spec"]
