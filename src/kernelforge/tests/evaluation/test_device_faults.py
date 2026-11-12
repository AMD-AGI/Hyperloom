# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A GPU fault told apart from a program's own failure, and a hang from a memory access fault."""

from __future__ import annotations

import pytest

from kernelforge.evaluation.device_faults import DEVICE_HANG, MEMORY_FAULT, classify

HANG = "HW Exception by GPU node-2 (Agent handle: 0x5dac2f79a5b0) reason :GPU Hang"
MEMORY = "Memory access fault by GPU node-3 (Agent handle: 0x55a1) on address 0x7f0000. Reason: Page not present."


def test_a_hardware_exception_is_a_hang_on_the_gpu_it_names():
    fault = classify(f"allclose: False\n{HANG}\n")

    assert (fault.kind, fault.node, fault.line) == (DEVICE_HANG, 2, HANG)


def test_a_memory_access_fault_is_told_apart_from_a_hang():
    fault = classify(MEMORY)

    assert (fault.kind, fault.node) == (MEMORY_FAULT, 3)


def test_a_hang_outranks_a_memory_access_fault_reported_beside_it():
    assert classify(f"{MEMORY}\n{HANG}").kind == DEVICE_HANG


@pytest.mark.parametrize(
    "output",
    ["", "allclose: False\nRuntimeError: shape mismatch", "HIP error: invalid device function", "mean_ms: 1.0"],
)
def test_output_that_names_no_gpu_fault_is_not_one(output):
    assert classify(output) is None
