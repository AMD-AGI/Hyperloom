# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A server the scheduler watchdog ended while a rank was still in aiter's JIT build is ``jit_build_timeout``, an
infrastructure failure, not an unclear one with no known signature.

The log lines are those of the GLM-5.3 TP4 test whose tuned blockscale table made aiter rebuild its module at the first
prefill: the watchdog's line on every rank, then the stacks py-spy dumped.
"""

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.measure.failures import JIT_BUILD_TIMEOUT, NO_KNOWN_SIGNATURE_REASON, classify_test_failure

WATCHDOG = "[2026-10-27 06:58:27 TP0] Scheduler watchdog timeout (self.watchdog_timeout=300)"
PYSPY = """Thread 1234 (idle): "MainThread"
    mp_lock (aiter/jit/core.py:211)
    build_module (aiter/jit/core.py:497)
    gemm_a8w8_blockscale_bpreshuffle (aiter/ops/gemm_op_a8w8.py:330)
"""
TRACEBACK = """  File "/usr/local/lib/python3.12/dist-packages/aiter/jit/core.py", line 497, in build_module
    with mp_lock(lock_path):
"""


@pytest.mark.parametrize("stack", [PYSPY, TRACEBACK])
def test_the_watchdog_during_an_aiter_build_is_a_jit_build_timeout(stack: str) -> None:
    verdict = classify_test_failure(returncode=1, logs=[f"loading\n{WATCHDOG}\n{stack}"])

    assert verdict.failure_class is FailureClass.INFRA
    assert verdict.failure_reason == f"{JIT_BUILD_TIMEOUT}: {WATCHDOG}"


@pytest.mark.parametrize(
    "log",
    [
        f"loading\n{WATCHDOG}\n    event_loop_normal (sglang/srt/managers/scheduler.py:900)\n",
        f"[aiter] start build\n{PYSPY}",
    ],
    ids=["watchdog_elsewhere", "build_without_watchdog"],
)
def test_a_watchdog_or_a_build_alone_is_not_one(log: str) -> None:
    verdict = classify_test_failure(returncode=1, logs=[log])

    assert verdict.failure_class is FailureClass.UNCLEAR
    assert verdict.failure_reason == NO_KNOWN_SIGNATURE_REASON
