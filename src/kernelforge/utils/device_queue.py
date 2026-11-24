# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One device, used by one command at a time, in the order the commands arrived.

Two callers share one queue. The loop imports this module and holds the device for its own measurements. A session
never imports it: the loop copies this file into the session's workspace beside a JSON configuration, and the session
runs every command that touches the device through the copy:

    python3 forge_device.py <command> [args...]

So the file uses the standard library only and runs under whatever interpreter the workspace has.

The queue is a directory of numbered tickets. A ticket is created with a number above every ticket that exists, and
its owner holds the device while it is the lowest one. A ticket whose owner has died is removed by whoever finds it.
Nothing here relies on ``flock``: on a network filesystem it is a per-process lock, and several lanes' stop checks
wait in the one loop process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNNER_NAME = "forge_device.py"
RUNNER_CONFIG_NAME = "forge_device.json"
# The variables that select the devices a process sees. A session's own environment carries them empty, so a process
# it starts outside the runner finds no device rather than timing on one another lane is measuring on.
DEVICE_VISIBILITY_VARIABLES = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
# Exit status of a command the runner stopped at the hold limit, as ``timeout(1)`` reports one.
HOLD_LIMIT_EXIT_STATUS = 124

_TICKET_DIGITS = 12
_POLL_SEC = 0.2
_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def hidden_device_environment() -> dict:
    """The overlay that hides every device from a process started with it."""
    return dict.fromkeys(DEVICE_VISIBILITY_VARIABLES, "")


def visible_device_environment(environ=None) -> dict:
    """What the visibility variables hold in ``environ``, ``None`` standing for one that is not set."""
    source = os.environ if environ is None else environ
    return {name: source.get(name) for name in DEVICE_VISIBILITY_VARIABLES}


class DeviceQueue:
    """The tickets of one device, under ``root``."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    def _owner_alive(self, ticket: Path) -> bool:
        """Whether the process that wrote ``ticket`` still exists; a ticket not yet filled in counts as alive."""
        try:
            pid = int(ticket.read_text().strip() or 0)
        except (OSError, ValueError):
            return True
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _tickets(self) -> list:
        """Every live ticket, lowest first."""
        live = []
        for ticket in sorted(self.root.glob("[0-9]" * _TICKET_DIGITS)):
            if self._owner_alive(ticket):
                live.append(ticket)
            else:
                with contextlib.suppress(OSError):
                    ticket.unlink()
        return live

    def _take_ticket(self) -> Path:
        """Create the ticket after every existing one. Exclusive creation makes two arrivals take different numbers."""
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            tickets = self._tickets()
            number = int(tickets[-1].name) + 1 if tickets else 1
            ticket = self.root / f"{number:0{_TICKET_DIGITS}d}"
            try:
                descriptor = os.open(ticket, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            with os.fdopen(descriptor, "w") as handle:
                handle.write(str(os.getpid()))
            return ticket

    def _is_first(self, ticket: Path) -> bool:
        return self._tickets()[:1] == [ticket]

    @contextlib.contextmanager
    def held(self):
        """Hold the device for the body; yields the seconds spent waiting for it."""
        ticket = self._take_ticket()
        arrived = time.monotonic()
        try:
            while not self._is_first(ticket):
                time.sleep(_POLL_SEC)
            yield time.monotonic() - arrived
        finally:
            with contextlib.suppress(OSError):
                ticket.unlink()

    @contextlib.asynccontextmanager
    async def held_async(self):
        """``held`` for a coroutine: the wait yields to the event loop, and a cancelled wait gives its place up."""
        ticket = self._take_ticket()
        arrived = time.monotonic()
        try:
            while not self._is_first(ticket):
                await asyncio.sleep(_POLL_SEC)
            yield time.monotonic() - arrived
        finally:
            with contextlib.suppress(OSError):
                ticket.unlink()


def append_usage(log_path, record: dict) -> None:
    """Add one line to the usage log. A log that cannot be written never fails the command it describes."""
    if not log_path:
        return
    with contextlib.suppress(OSError):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_runner_config(path, *, queue_root, usage_log, lane: str, hold_limit_sec: int, environment: dict) -> None:
    """Write the configuration a copied runner reads from beside itself."""
    config = {
        "queue_root": str(queue_root),
        "usage_log": str(usage_log),
        "lane": lane,
        "hold_limit_sec": int(hold_limit_sec),
        "environment": dict(environment),
    }
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_environment(visible: dict) -> dict:
    """This process's environment with the visibility variables as the campaign has them."""
    environment = dict(os.environ)
    for name, value in visible.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _stop_group(process) -> None:
    """End the command and everything it started."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _run_held(command: list, config: dict) -> tuple:
    """Run ``command`` with the device visible; its exit status, and whether the hold limit ended it."""
    # A group of its own, so the hold limit can end the command with everything it started. A termination signal
    # sent to this runner is passed on for the same reason: the group no longer receives it by itself.
    process = subprocess.Popen(command, env=_command_environment(config["environment"]), start_new_session=True)

    def forward(signum, _frame) -> None:
        _stop_group(process)
        sys.exit(128 + signum)

    previous = {signum: signal.signal(signum, forward) for signum in _TERMINATION_SIGNALS}
    try:
        return process.wait(timeout=config["hold_limit_sec"]), False
    except subprocess.TimeoutExpired:
        _stop_group(process)
        return HOLD_LIMIT_EXIT_STATUS, True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv: list) -> int:
    """Queue for the device, run the command, record the wait and the hold."""
    if not argv:
        sys.stderr.write(f"usage: python3 {RUNNER_NAME} <command> [args...]\n")
        return 2
    config = json.loads(Path(__file__).with_name(RUNNER_CONFIG_NAME).read_text(encoding="utf-8"))
    with DeviceQueue(config["queue_root"]).held() as waited:
        if waited >= 1.0:
            sys.stderr.write(f"[device] waited {waited:.0f}s for another lane's measurement\n")
            sys.stderr.flush()
        started = time.monotonic()
        status, limited = _run_held(argv, config)
        held = time.monotonic() - started
    if limited:
        sys.stderr.write(
            f"[device] stopped after {config['hold_limit_sec']}s: one command may hold the device no longer. "
            "Split the work into shorter commands.\n"
        )
    append_usage(
        config["usage_log"],
        {
            "time": time.time(),
            "lane": config["lane"],
            "wait_sec": round(waited, 3),
            "hold_sec": round(held, 3),
            "command": " ".join(argv)[:300],
            "status": status,
            "hold_limit_reached": limited,
        },
    )
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
