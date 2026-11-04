"""
Keep-alive messages for a systemd watchdog (``WatchdogSec=`` in the unit).

A capture loop calls :func:`ping` every pass. If it stops, for example because
a vendor SDK call never returns, systemd kills and restarts the service. That
has to come from outside the process: a thread blocked inside a C library can't
run a Python timeout or signal handler.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import os
import socket


def ping() -> None:
    """Tell systemd this process is alive; does nothing outside a service."""

    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    if address.startswith("@"):  # abstract socket
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"WATCHDOG=1", address)
    except OSError:
        pass  # a missed ping is harmless; repeated ones are what the watchdog is for
