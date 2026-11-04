# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run Magpie with its leftover-server sweep confined to the caller's sandbox.

Magpie's ``BenchmarkMode._cleanup_server_processes`` SIGKILLs every process on the host whose command line names the
framework's server. When several local sandboxes measure on one host, or a test on the host measures beside them, one
test's cleanup kills the others' servers mid-run. Run as ``python magpie_confined.py <Magpie arguments>``: the sweep
signals only processes of the caller's own sandbox, which inherit its ``E2B_SANDBOX_ID``; outside a sandbox, only
processes carrying no sandbox id. A process whose environment cannot be read is spared either way. Magpie's own
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


def process_environ(pid: int) -> list[bytes] | None:
    """The ``NAME=value`` entries ``pid`` started with; ``None`` when it is gone or unreadable."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            return handle.read().split(b"\0")
    except OSError:
        return None


def sandbox_of(environ: list[bytes]) -> bytes:
    """The sandbox id in ``environ``; empty for a process on the host."""
    prefix = f"{SANDBOX_TAG}=".encode()
    return next((entry[len(prefix) :] for entry in environ if entry.startswith(prefix)), b"")


class ConfinedOs:
    """``os`` whose ``kill`` reaches only processes of ``sandbox`` (empty: the host's own)."""

    def __init__(self, sandbox: bytes, environ: Callable[[int], list[bytes] | None]) -> None:
        self._sandbox = sandbox
        self._environ = environ

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def kill(self, pid: int, sig: int) -> None:
        environ = self._environ(pid)
        if environ is not None and sandbox_of(environ) == self._sandbox:
            os.kill(pid, sig)


def confine_sweep(
    module: ModuleType, sandbox: bytes, *, environ: Callable[[int], list[bytes] | None] | None = None
) -> None:
    """Make ``module``'s sweep signal only processes of ``sandbox``; say so on stderr when it has no sweep."""
    owner = getattr(module, SWEEP_CLASS, None)
    sweep = getattr(owner, SWEEP, None)
    if sweep is None:
        print(f"magpie_confined: {SWEEP_CLASS}.{SWEEP} not found; Magpie's cleanup runs unconfined", file=sys.stderr)
        return
    confined = ConfinedOs(sandbox, environ or process_environ)

    def confined_sweep(*args: Any, **kwargs: Any) -> Any:
        real = module.os
        module.os = confined
        try:
            return sweep(*args, **kwargs)
        finally:
            module.os = real

    setattr(owner, SWEEP, staticmethod(confined_sweep))


def main() -> None:
    from Magpie.modes.benchmark import benchmarker

    confine_sweep(benchmarker, os.environ.get(SANDBOX_TAG, "").encode())
    sys.argv[0] = "Magpie"
    runpy.run_module("Magpie", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
