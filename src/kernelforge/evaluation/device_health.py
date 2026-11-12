# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the GPU this process measures on runs work: before a measurement, and again after the driver reset it.

The probe runs in a child process, never in the caller: a process that opens the GPU is ended by the next hang on
any GPU of the machine, and the caller is the campaign. It uses the interpreter and environment the measurement
driver uses, with the torch the container image supplies for every driver.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable

from kernelforge.evaluation.device_faults import DEVICE_UNRESPONSIVE, DeviceFault, classify
from kernelforge.utils.process_group import communicate_process_group

#: Allocates and reduces on the device, so it succeeds only when kernels launch and complete again.
_PROBE = (
    "import torch\n"
    "x = torch.ones(1 << 20, device='cuda')\n"
    "torch.cuda.synchronize()\n"
    "assert float(x.sum()) == float(1 << 20)\n"
)
PROBE_TIMEOUT_SEC = 120.0
#: Longer than a reset takes; a GPU still unusable after this needs an operator, not more waiting.
RECOVERY_TIMEOUT_SEC = 900.0
_BACKOFF_SEC = (5.0, 10.0, 20.0, 40.0, 60.0)


async def probe_device(timeout_sec: float = PROBE_TIMEOUT_SEC) -> DeviceFault | None:
    """``None`` when one small kernel runs to completion on the device within ``timeout_sec``, else why it did not.

    Why is a fault either stream names, or else the last line of stderr, where errors are written, and of stdout only
    when stderr is empty. The streams are read apart because two buffered streams merged into one arrive in the order
    their buffers were flushed, not the order they were written.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PROBE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await communicate_process_group(proc, timeout=timeout_sec)
    except asyncio.TimeoutError:
        return DeviceFault(DEVICE_UNRESPONSIVE, f"the probe kernel did not complete within {timeout_sec:.0f}s")
    if proc.returncode == 0:
        return None
    errors, output = stderr.decode(errors="replace").strip(), stdout.decode(errors="replace").strip()
    reported = (errors or output).splitlines()
    last_line = reported[-1] if reported else f"the probe exited with status {proc.returncode}"
    return classify(f"{errors}\n{output}") or DeviceFault(DEVICE_UNRESPONSIVE, last_line)


async def wait_for_device_recovery(
    *,
    timeout_sec: float = RECOVERY_TIMEOUT_SEC,
    probe: Callable[[], Awaitable[DeviceFault | None]] = probe_device,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Probe the device, backing off between tries, until it runs work again or ``timeout_sec`` has passed."""
    deadline = clock() + timeout_sec
    attempt = 0
    while True:
        if await probe() is None:
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        await sleep(min(_BACKOFF_SEC[min(attempt, len(_BACKOFF_SEC) - 1)], remaining))
        attempt += 1


__all__ = ["PROBE_TIMEOUT_SEC", "RECOVERY_TIMEOUT_SEC", "probe_device", "wait_for_device_recovery"]
