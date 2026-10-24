# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run Magpie with its leftover-server sweep confined to the caller's sandbox.

Magpie's ``BenchmarkMode._cleanup_server_processes`` SIGKILLs every process on the host whose command line names the
framework's server. When several local sandboxes measure on one host, one test's cleanup kills the others' servers
mid-run. Run as ``python magpie_confined.py <Magpie arguments>``: inside a sandbox (``E2B_SANDBOX_ID`` set) the sweep
signals only processes started in that sandbox, which inherit its id; outside one Magpie runs unchanged. Magpie's own
interpreter runs this file and need not have Hyperloom installed, so it imports the standard library only.

Remove once Magpie scopes its sweep, or once every test runs in a pod of its own.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

SANDBOX_TAG = "E2B_SANDBOX_ID"
SWEEP_CLASS = "BenchmarkMode"
SWEEP = "_cleanup_server_processes"


def process_environ(pid: int) -> list[bytes]:
    """The ``NAME=value`` entries ``pid`` started with; none when it is gone or unreadable."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            return handle.read().split(b"\0")
    except OSError:
        return []


class ConfinedOs:
    """``os`` whose ``kill`` reaches only processes whose environment carries ``tag``."""

    def __init__(self, tag: bytes, environ: Callable[[int], list[bytes]]) -> None:
        self._tag = tag
        self._environ = environ

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def kill(self, pid: int, sig: int) -> None:
        if self._tag in self._environ(pid):
            os.kill(pid, sig)


def confine_sweep(module: ModuleType, tag: bytes, *, environ: Callable[[int], list[bytes]] = process_environ) -> None:
    """Make ``module``'s sweep signal only processes tagged ``tag``; say so on stderr when it has no sweep."""
    owner = getattr(module, SWEEP_CLASS, None)
    sweep = getattr(owner, SWEEP, None)
    if sweep is None:
        print(f"magpie_confined: {SWEEP_CLASS}.{SWEEP} not found; Magpie's cleanup runs unconfined", file=sys.stderr)
        return
    confined = ConfinedOs(tag, environ)

    def confined_sweep(*args: Any, **kwargs: Any) -> Any:
        real = module.os
        module.os = confined
        try:
            return sweep(*args, **kwargs)
        finally:
            module.os = real

    setattr(owner, SWEEP, staticmethod(confined_sweep))


def main() -> None:
    sandbox = os.environ.get(SANDBOX_TAG, "")
    if sandbox:
        from Magpie.modes.benchmark import benchmarker

        confine_sweep(benchmarker, f"{SANDBOX_TAG}={sandbox}".encode())
    sys.argv[0] = "Magpie"
    runpy.run_module("Magpie", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
