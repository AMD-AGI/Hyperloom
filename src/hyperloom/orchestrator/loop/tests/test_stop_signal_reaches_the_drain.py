# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stop signal reaches the drain after native code replaced its handler behind the interpreter's back."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import importlib.util
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.signals import SignalDrain

_SIG_IGN = 1
_SYMBOL_MISSING_EXIT = 77

_RAY_DRIVER_SCENARIO = r"""
import asyncio
import ctypes
import os
import resource
import signal
import sys
import threading

import ray._raylet

from hyperloom.orchestrator.loop.signals import _RECLAIM_INTERVAL_SEC, SignalDrain

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
raylet = ctypes.CDLL(ray._raylet.__file__)
# What a driver's core worker installs when it connects: abseil's failure handler, chained to the previous one.
install = getattr(raylet, "_ZN3ray6RayLog27InstallFailureSignalHandlerEPKcb", None)
if install is None:
    sys.exit(int(sys.argv[1]))
install.argtypes = [ctypes.c_char_p, ctypes.c_bool]


async def main():
    stop = asyncio.Event()
    drain = SignalDrain(loop=asyncio.get_running_loop(), stop_event=stop)
    connect = threading.Thread(target=install, args=(None, True))
    connect.start()
    connect.join()
    await asyncio.sleep(2 * _RECLAIM_INTERVAL_SEC)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(stop.wait(), timeout=5.0)
    # Past the 3 s alarm abseil arms when it handles a SIGTERM.
    await asyncio.sleep(3.5)
    drain.close()
    print("teardown finished", flush=True)


asyncio.run(main())
"""


def _ignore_from_another_thread(sig: int) -> None:
    """Set ``sig`` to SIG_IGN through libc on a worker thread, out of the interpreter's sight."""
    libc = ctypes.CDLL(None)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal.restype = ctypes.c_void_p
    worker = threading.Thread(target=libc.signal, args=(sig, _SIG_IGN))
    worker.start()
    worker.join()


def test_a_sigterm_reaches_the_drain_after_native_code_replaced_its_handler():
    async def _exercise() -> tuple[bool, frozenset[int]]:
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        drain = SignalDrain(loop=loop, stop_event=stop, signals=(signal.SIGTERM,))
        try:
            assert drain.armed
            _ignore_from_another_thread(signal.SIGTERM)
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_IGN
            deadline = loop.time() + 5.0
            while not stop.is_set() and loop.time() < deadline:
                os.kill(os.getpid(), signal.SIGTERM)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=0.1)
            return stop.is_set(), drain.close()
        finally:
            drain.close()

    stopped, received = asyncio.run(_exercise())

    assert stopped is True
    assert received == {signal.SIGTERM}


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="Ray is not installed")
def test_a_ray_driver_outlives_a_sigterm_long_enough_to_finish_its_teardown():
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, "-c", _RAY_DRIVER_SCENARIO, str(_SYMBOL_MISSING_EXIT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode == _SYMBOL_MISSING_EXIT:
        pytest.skip("this Ray build does not export RayLog::InstallFailureSignalHandler")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "teardown finished" in result.stdout
    assert "SIGTERM received" not in result.stderr
