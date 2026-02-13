"""

:copyright:
    2026, Conor A. Bacon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CaptureResult:
    """
    Standard capture return type for all instruments.

    metadata: JSON-serialisable information about the capture.
    artifacts: Captured data objects (images/arrays/paths/etc.)

    """

    metadata: Mapping[str, Any]
    artifacts: Mapping[str, Any]
