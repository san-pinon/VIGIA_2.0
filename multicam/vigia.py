"""
VIGIA 2.0 master command: ``vigia start | stop | status``. Runs on uvcam.

- ``start``: file monitors on both nodes, then the orchestrator, which brings up
  the UV and IR services within a minute. All three are also enabled at boot, so
  the station comes back by itself after a power cut.
- ``stop``: the orchestrator first, or it would restart the services; then UV +
  file monitor on uvcam, IR + file monitor on multicam. The three are disabled at
  boot too, so a stopped station stays stopped through a reboot.
- ``status``: units on both nodes, disk, clock, Witty Pi default state, recent
  files per instrument and the orchestrator's problem lines. Read-only.

Exit code 0 if everything worked or looks right, 1 otherwise. An unreachable
multicam is reported, and never stops the work on uvcam.

Each ``sudo`` call names exactly one unit: the sudoers rules on multicam
(``vigia-capture``, ``vigia-file-monitor``) match the whole command line.
The UV and IR services are never enabled at boot: the orchestrator owns them.

:copyright:
    2026, Santiago Pinon.
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path

from multicam.orchestrator import (
    DEFAULT_FRESHNESS_MIN,
    DEFAULT_MIN_SUN_ELEVATION_DEG,
    Remote,
    daylight_expected,
    site_from_config,
)
from multicam.utilities import read_config

ORCHESTRATOR = "vigia-orchestrator.service"
FILE_MONITOR = "file-monitor.service"
UV_MONITOR = "capture-uv-monitor.service"
IR_MONITOR = "capture-ir-video.service"
UVCAM_UNITS = (ORCHESTRATOR, UV_MONITOR, FILE_MONITOR)
MULTICAM_UNITS = (IR_MONITOR, FILE_MONITOR)
BOOT_UNITS = (ORCHESTRATOR, FILE_MONITOR)  # enabled at boot while vigia is started

SSH_UNREACHABLE = 255  # ssh's own exit code: it never reached the command
START_TIMEOUT_S = 150  # the orchestrator waits for chrony for up to ~1 min
STOP_TIMEOUT_S = 120  # above the units' own stop timeouts (90 s)
BOOT_TIMEOUT_S = 30
STATUS_TIMEOUT_S = 60
RECENT_WINDOW_S = 180
MAX_CLOCK_DIFFERENCE_S = 2  # ssh takes well under this, drift doesn't
DEFAULT_MIN_FREE_GB = 5

WITTY_ADDRESS = "0x08"
WITTY_DEFAULT_STATE_REG = "17"
WITTY_DEFAULT_ON = "0x01"

PROBLEM_LINES = 8
PROBLEM_PATTERN = "supervise|failed|timed out|skipped|freshness|USB reset"


@dataclass
class Result:
    """Outcome of a command on a node."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def unreachable(self) -> bool:
        return self.returncode == SSH_UNREACHABLE

    @property
    def message(self) -> str:
        return (self.stderr or self.stdout).strip() or f"rc={self.returncode}"


class Node:
    """A node to run things on: this one, or another one over ssh."""

    def __init__(self, name: str, remote: Remote | None = None) -> None:
        self.name = name
        self.remote = remote

    def run(self, cmd: list[str] | str, timeout_s: float) -> Result:
        """Run an argument list, or a string as a shell script."""

        if self.remote is not None:
            argv = self.remote.command([cmd] if isinstance(cmd, str) else cmd)
        else:
            argv = ["sh", "-c", cmd] if isinstance(cmd, str) else cmd
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return Result(1, "", f"timed out after {timeout_s:.0f} s")
        except OSError as e:
            return Result(1, "", str(e))
        return Result(proc.returncode, proc.stdout, proc.stderr)


def _nodes(config: dict) -> tuple[Node, Node]:
    orch = config.get("orchestrator", {})
    remote = Remote(
        orch.get("multicam_user", "user"), orch.get("multicam_host", "192.168.30.150")
    )
    return Node("uvcam"), Node("multicam", remote)


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


def _act(node: Node, action: str, unit: str, timeout_s: float) -> Result:
    """Start or stop one unit and confirm its state; prints one line."""

    result = node.run(["sudo", "-n", "systemctl", action, unit], timeout_s)
    if result.ok:
        state = node.run(["systemctl", "is-active", unit], 20).stdout.strip()
        wanted = ("active",) if action == "start" else ("inactive", "failed")
        if state not in wanted:
            result = Result(1, "", f"is {state or 'unknown'} after {action}")
    label = f"{node.name} {unit}"
    if result.ok:
        print(f"  {label:<40} {action} ok", flush=True)
    elif result.unreachable:
        print(f"  {label:<40} UNREACHABLE", flush=True)
    else:
        print(f"  {label:<40} FAILED: {result.message}", flush=True)
    return result


def _set_boot(node: Node, unit: str, enabled: bool) -> Result:
    """Enable or disable one unit at boot; prints one line."""

    action = "enable" if enabled else "disable"
    result = node.run(["sudo", "-n", "systemctl", action, unit], BOOT_TIMEOUT_S)
    label = f"{node.name} {unit}"
    if result.ok:
        print(f"  {label:<40} {action} at boot ok", flush=True)
    else:
        print(f"  {label:<40} {action} at boot FAILED: {result.message}", flush=True)
    return result


def _start_at_boot(node: Node, unit: str, timeout_s: float) -> Result:
    """Enable, then start. A failed enable still starts: capturing comes first."""

    enabled = _set_boot(node, unit, True)
    if enabled.unreachable:
        return enabled
    started = _act(node, "start", unit, timeout_s)
    return started if not started.ok else enabled


def start(uvcam: Node, multicam: Node) -> int:
    """File monitors, then the orchestrator. Safe to run again."""

    failed = False
    result = _start_at_boot(multicam, FILE_MONITOR, START_TIMEOUT_S)
    failed = not result.ok
    if result.unreachable:
        print(
            "  Carrying on without multicam. Its file monitor isn't supervised: "
            "run `vigia start` again once it's back."
        )
    failed |= not _start_at_boot(uvcam, FILE_MONITOR, START_TIMEOUT_S).ok
    print(
        "  Starting the orchestrator (it can wait ~1 min for the clock)...", flush=True
    )
    failed |= not _start_at_boot(uvcam, ORCHESTRATOR, START_TIMEOUT_S).ok
    print("UV and IR come up within a minute. Check with `vigia status`.")
    return 1 if failed else 0


def stop(uvcam: Node, multicam: Node) -> int:
    """The orchestrator first, then everything else, on whichever node answers."""

    if not _act(uvcam, "stop", ORCHESTRATOR, STOP_TIMEOUT_S).ok:
        print(
            "The orchestrator is still up and would restart the rest: "
            "stopped nothing else."
        )
        return 1

    failed = not _set_boot(uvcam, ORCHESTRATOR, False).ok
    for unit in (UV_MONITOR, FILE_MONITOR):
        failed |= not _act(uvcam, "stop", unit, STOP_TIMEOUT_S).ok
    failed |= not _set_boot(uvcam, FILE_MONITOR, False).ok
    for unit in (IR_MONITOR, FILE_MONITOR):
        result = _act(multicam, "stop", unit, STOP_TIMEOUT_S)
        failed |= not result.ok
        if result.ok and unit in BOOT_UNITS:
            failed |= not _set_boot(multicam, unit, False).ok
        if result.unreachable:
            print(
                "  multicam may still be capturing: "
                "run `vigia stop` again once it's back."
            )
            break
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@dataclass
class NodeStatus:
    """What one node reports; see ``_status_script`` for the line format."""

    now: float | None = None
    uptime: str = "?"
    mem_free_mb: str = "?"
    disk_free_gb: int | None = None
    disk_used: str = "?"
    source: str = ""
    offset: str = "?"
    leap: str = "?"
    units: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    files: dict[str, tuple[int, int | None]] = field(default_factory=dict)
    data_dir_ok: bool = True


def _status_script(units: tuple[str, ...], data_dir: str) -> str:
    """
    Shell script printing one ``key value...`` line per fact. Files move between
    folders and nodes, so it looks in ``receive/`` and today's and yesterday's
    ``archive/`` folders, as the orchestrator's freshness check does.
    """

    unit_names = " ".join(shlex.quote(u) for u in units)
    return rf"""
now=$(date +%s)
echo "now $now"
echo "uptime $(uptime -p)"
echo "mem $(free -m | awk 'NR==2 {{print $7}}')"
for u in {unit_names}; do
  echo "unit $u $(systemctl is-active $u) \
$(systemctl show -p NRestarts --value $u) $(systemctl is-enabled $u 2>/dev/null)"
done
chronyc -n sources 2>/dev/null | awk '$1 ~ /^[#^]\*$/ {{print "source", $1, $2}}'
chronyc tracking 2>/dev/null | awk -F': ' '
  /^System time/ {{print "offset", $2}}
  /^Leap status/ {{print "leap", $2}}'
cd {shlex.quote(data_dir)} 2>/dev/null || {{ echo "nodata"; exit 0; }}
echo "disk $(df -BG --output=avail,pcent . | tail -1)"
d0=$(date -u +%j); d1=$(date -u -d yesterday +%j)
for d in */; do
  i=${{d%/}}
  [ "$i" = "lost+found" ] && continue
  find "$i/receive" "$i"/archive/*/*/*/"$d0" "$i"/archive/*/*/*/"$d1" \
      -maxdepth 1 -type f -size +0c ! -name '.*' ! -name '*.tmp' \
      -printf '%T@\n' 2>/dev/null |
    awk -v i="$i" -v now="$now" -v win={RECENT_WINDOW_S} '
      {{a = now - $1; if (a < win) n++; if (m == "" || a < m) m = a}}
      END {{printf "files %s %d %s\n", i, n, (m == "" ? "-" : int(m))}}'
done
"""


def parse_status(text: str) -> NodeStatus:
    """Read the lines printed by ``_status_script``."""

    status = NodeStatus()
    for line in text.splitlines():
        key, _, rest = line.partition(" ")
        parts = rest.split()
        if key == "now" and parts:
            status.now = float(parts[0])
        elif key == "uptime":
            status.uptime = rest.removeprefix("up ")
        elif key == "mem" and parts:
            status.mem_free_mb = parts[0]
        elif key == "unit" and parts:
            state = parts[1] if len(parts) > 1 else "?"
            restarts = parts[2] if len(parts) > 2 else "?"
            boot = parts[3] if len(parts) > 3 else "?"
            status.units[parts[0]] = (state, restarts, boot)
        elif key == "source":
            status.source = rest
        elif key == "offset":
            status.offset = rest
        elif key == "leap":
            status.leap = rest
        elif key == "nodata":
            status.data_dir_ok = False
        elif key == "disk" and len(parts) == 2:
            status.disk_free_gb = int(parts[0].rstrip("G"))
            status.disk_used = parts[1]
        elif key == "files" and len(parts) == 3:
            age = None if parts[2] == "-" else max(0, int(parts[2]))
            status.files[parts[0]] = (int(parts[1]), age)
    return status


def _wanted_running(statuses: dict[str, NodeStatus | None]) -> bool:
    """Was the station started? ``start`` enables the orchestrator, ``stop`` doesn't."""

    uvcam = statuses.get("uvcam")
    return uvcam is not None and uvcam.units.get(ORCHESTRATOR, ("", "", ""))[2] == (
        "enabled"
    )


def _overall_state(statuses: dict[str, NodeStatus | None]) -> str:
    """
    RUNNING or STOPPED only if every unit agrees with what was asked for: started
    (orchestrator enabled at boot) or stopped. Anything else is PARTIAL, including
    a node that didn't answer.
    """

    if None in statuses.values():
        return "PARTIAL"
    states = [unit[0] for status in statuses.values() for unit in status.units.values()]
    wanted = _wanted_running(statuses)
    if states and wanted and all(s == "active" for s in states):
        return "RUNNING"
    if states and not wanted and all(s == "inactive" for s in states):
        return "STOPPED"
    return "PARTIAL"


def _merge_files(
    statuses: dict[str, NodeStatus | None],
) -> dict[str, tuple[int, int | None]]:
    """Add the nodes together: a file can be on either, as uvcam forwards them."""

    merged: dict[str, tuple[int, int | None]] = {}
    for status in statuses.values():
        for name, (count, age) in (status.files if status else {}).items():
            old_count, old_age = merged.get(name, (0, None))
            ages = [a for a in (old_age, age) if a is not None]
            merged[name] = (old_count + count, min(ages) if ages else None)
    return merged


def find_problems(
    statuses: dict[str, NodeStatus | None],
    witty: str,
    config: dict,
    now: dt,
) -> list[str]:
    """Everything worth a warning. Empty means all looks right."""

    problems = []
    wanted = _wanted_running(statuses)
    capturing = wanted or any(
        unit[0] != "inactive"
        for status in statuses.values()
        if status is not None
        for unit in status.units.values()
    )
    min_free_gb = config.get("file_monitor", {}).get("min_free_gb", DEFAULT_MIN_FREE_GB)

    if wanted and statuses["uvcam"].units[ORCHESTRATOR][0] != "active":
        problems.append("the station was started, but the orchestrator is not running")
    elif capturing and not wanted:
        problems.append(
            "units are up, but not through `vigia start`: "
            "the station won't come back after a power cut"
        )

    for name, status in statuses.items():
        if status is None:
            problems.append(f"{name} unreachable")
            continue
        for unit, (state, _, boot) in status.units.items():
            if capturing and state != "active":
                problems.append(f"{name} {unit} is {state}")
            should_enable = wanted and unit in BOOT_UNITS
            if should_enable and boot != "enabled":
                problems.append(
                    f"{name} {unit} is not enabled at boot: "
                    "it won't come back after a power cut"
                )
            elif boot == "enabled" and not should_enable:
                problems.append(f"{name} {unit} is enabled at boot, but shouldn't be")
        if not status.data_dir_ok:
            problems.append(f"{name}: data folder missing")
        elif status.disk_free_gb is not None and status.disk_free_gb < min_free_gb:
            problems.append(f"{name}: only {status.disk_free_gb} GB free")

    uvcam, multicam = statuses.get("uvcam"), statuses.get("multicam")
    if uvcam is not None and uvcam.source.split()[1:] != ["GPS"]:
        problems.append(f"uvcam clock is not on GPS ({uvcam.source or 'no source'})")
    if multicam is not None and not multicam.source:
        problems.append("multicam clock has no time source")
    if uvcam is not None and multicam is not None and uvcam.now and multicam.now:
        difference = abs(uvcam.now - multicam.now)
        if difference > MAX_CLOCK_DIFFERENCE_S:
            problems.append(f"node clocks are {difference:.0f} s apart")

    if witty != WITTY_DEFAULT_ON:
        problems.append(
            f"Witty Pi default state (reg {WITTY_DEFAULT_STATE_REG}) is {witty}, "
            f"expected {WITTY_DEFAULT_ON}: uvcam won't start by itself"
        )

    # Files may be on the node that didn't answer, so only judge with both.
    if capturing and uvcam is not None and multicam is not None:
        files = _merge_files(statuses)
        for name, minutes in _freshness_limits(config).items():
            _, age = files.get(name, (0, None))
            stale = age is None or age > minutes * 60
            if stale and _expected(name, config, now):
                problems.append(f"{name}: no new file for over {minutes} min")
    return problems


def _freshness_limits(config: dict) -> dict[str, int]:
    return config.get("orchestrator", {}).get(
        "freshness_max_age_min", DEFAULT_FRESHNESS_MIN
    )


def _expected(name: str, config: dict, now: dt) -> bool:
    uv_sync = config.get("uv_sync", {})
    return daylight_expected(
        name,
        now,
        site_from_config(uv_sync),
        float(uv_sync.get("min_sun_elevation_deg", DEFAULT_MIN_SUN_ELEVATION_DEG)),
    )


def _print_status(
    statuses: dict[str, NodeStatus | None],
    witty: str,
    log_lines: list[str],
    problems: list[str],
    config: dict,
    now: dt,
) -> None:
    print(f"VIGIA {_overall_state(statuses)}   {now:%Y-%m-%d %H:%M:%S} UTC")

    print("\nUnits")
    for name, status in statuses.items():
        if status is None:
            print(f"  {name:<9} UNREACHABLE")
            continue
        for unit, (state, restarts, boot) in status.units.items():
            print(
                f"  {name:<9} {unit:<30} {state:<12} restarts={restarts:<4} boot={boot}"
            )

    print("\nNodes")
    for name, status in statuses.items():
        if status is None:
            continue
        print(
            f"  {name:<9} disk {status.disk_free_gb} GB free "
            f"({status.disk_used} used), "
            f"mem {status.mem_free_mb} MB free, up {status.uptime}"
        )
        print(
            f"  {'':<9} clock {status.source or 'no source'}, "
            f"{status.offset}, leap {status.leap}"
        )
    print(f"  Witty Pi  default state reg {WITTY_DEFAULT_STATE_REG} = {witty}")

    both = all(statuses.values())
    where = "both nodes" if both else "only the node that answered"
    print(f"\nFiles ({where})   last {RECENT_WINDOW_S // 60} min   newest")
    for name, (count, age) in sorted(_merge_files(statuses).items()):
        none = "none in 2 days" if both else "none on this node"
        newest = none if age is None else _format_age(age)
        note = "" if _expected(name, config, now) else "   (sun low: not expected)"
        print(f"  {name:<18} {count:>10}   {newest}{note}")

    print("\nOrchestrator log (latest problem lines)")
    for line in log_lines or ["(none)"]:
        print(f"  {line}")

    print("\nProblems" if problems else "\nNo problems")
    for problem in problems:
        print(f"  ! {problem}")


def _format_age(age_s: int) -> str:
    if age_s < 120:
        return f"{age_s} s ago"
    if age_s < 7200:
        return f"{age_s // 60} min ago"
    return f"{age_s // 3600} h ago"


def _read_witty(uvcam: Node) -> str:
    """The Witty Pi default state register; ``unreadable`` if that fails."""

    result = uvcam.run(
        ["sudo", "-n", "i2cget", "-y", "1", WITTY_ADDRESS, WITTY_DEFAULT_STATE_REG], 10
    )
    return (
        result.stdout.strip() if result.ok and result.stdout.strip() else "unreadable"
    )


def _read_log(uvcam: Node, log_path: str) -> list[str]:
    """Problem lines. ``-a`` and the ``tr``: a power cut can leave zero bytes in it."""

    script = (
        f"tr -d '\\000' < {shlex.quote(log_path)} 2>/dev/null | "
        f"grep -aE {shlex.quote(PROBLEM_PATTERN)} | tail -{PROBLEM_LINES} | cut -c1-150"
    )
    return uvcam.run(script, 20).stdout.splitlines()


def status(uvcam: Node, multicam: Node, config: dict) -> int:
    """Print the whole picture; exit code 1 if anything looks wrong."""

    data_dir = config["metadata"]["data_archive"]
    log_path = config.get("orchestrator", {}).get(
        "log_path", "/home/user/logs/orchestrator.log"
    )

    # Together, so the two node times can be compared.
    with ThreadPoolExecutor(max_workers=4) as pool:
        local = pool.submit(
            uvcam.run, _status_script(UVCAM_UNITS, data_dir), STATUS_TIMEOUT_S
        )
        remote = pool.submit(
            multicam.run, _status_script(MULTICAM_UNITS, data_dir), STATUS_TIMEOUT_S
        )
        witty = pool.submit(_read_witty, uvcam)
        log_lines = pool.submit(_read_log, uvcam, log_path)

    statuses: dict[str, NodeStatus | None] = {}
    for node, future in ((uvcam, local), (multicam, remote)):
        result = future.result()
        statuses[node.name] = parse_status(result.stdout) if result.ok else None

    now = dt.now(UTC)
    problems = find_problems(statuses, witty.result(), config, now)
    _print_status(statuses, witty.result(), log_lines.result(), problems, config, now)
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the master command."""

    parser = argparse.ArgumentParser(
        prog="vigia", description="Start, stop or check the whole VIGIA station."
    )
    parser.add_argument("command", choices=("start", "stop", "status"))
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(Path.home() / ".config/multicam_config.toml"),
        help="Path to multicam_config.toml",
    )
    args = parser.parse_args()

    if socket.gethostname() != "uvcam":
        sys.exit("vigia runs on uvcam. From elsewhere: ssh uvcam vigia " + args.command)

    config = read_config(Path(args.config))
    uvcam, multicam = _nodes(config)
    if args.command == "start":
        sys.exit(start(uvcam, multicam))
    if args.command == "stop":
        sys.exit(stop(uvcam, multicam))
    sys.exit(status(uvcam, multicam, config))


if __name__ == "__main__":
    main()
