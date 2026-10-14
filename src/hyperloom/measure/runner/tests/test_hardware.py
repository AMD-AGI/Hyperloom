# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.measure.runner.hardware import GpuFacts, check_hardware, ecc_counts, read_gpu_facts


def _runner(answers):
    return lambda args: answers.get(" ".join(args))


STATIC = json.dumps(
    {
        "gpu_data": [
            {
                "gpu": i,
                "asic": {"market_name": "AMD Instinct MI355X"},
                "limit": {"ppt0": {"socket_power_limit": {"value": 1400, "unit": "W"}}},
            }
            for i in range(4)
        ]
    }
)
PARTITION = json.dumps(
    {
        "gpu_data": [
            {"gpu": i, "partition": {"accelerator_partition": "SPX", "memory_partition": "NPS1"}} for i in range(4)
        ]
    }
)
ECC = json.dumps(
    {"gpu_data": [{"gpu": i, "ecc": {"total_correctable_count": i, "total_uncorrectable_count": 0}} for i in range(4)]}
)


def test_facts_of_the_visible_gpus_only():
    run = _runner({"static --json": STATIC, "static --partition --json": PARTITION})

    facts = read_gpu_facts([1, 2], run)

    assert facts == [
        GpuFacts(1, "AMD Instinct MI355X", 1400.0, "SPX/NPS1"),
        GpuFacts(2, "AMD Instinct MI355X", 1400.0, "SPX/NPS1"),
    ]


def test_an_unreadable_tool_gives_none_and_not_a_guess():
    assert read_gpu_facts([0], _runner({})) is None
    assert read_gpu_facts([0], _runner({"static --json": "not json"})) is None
    assert ecc_counts([0], _runner({})) is None


def test_ecc_counts_need_every_visible_gpu():
    run = _runner({"metric --ecc --json": ECC})

    assert ecc_counts([1, 2], run) == {1: ecc_counts([1], run)[1], 2: ecc_counts([2], run)[2]}
    assert ecc_counts([1, 9], run) is None


def test_hardware_mismatches_are_named():
    facts = [GpuFacts(0, "AMD Instinct MI355X", 1400.0, "SPX/NPS1")]

    assert check_hardware(facts, gpu_type="mi355x", gpu_count=1, power_cap_w=1400, partition_mode="SPX/NPS1") == []
    problems = check_hardware(facts, gpu_type="mi300x", gpu_count=2, power_cap_w=750, partition_mode="CPX/NPS4")
    assert any("1 GPUs visible, 2 needed" in p for p in problems)
    assert any("not mi300x" in p for p in problems)
    assert any("power cap 1400.0 W, not 750 W" in p for p in problems)
    assert any("partition SPX/NPS1, not CPX/NPS4" in p for p in problems)


def test_a_value_the_machine_did_not_give_is_not_compared():
    facts = [GpuFacts(0, None, None, None)]

    assert check_hardware(facts, gpu_type="mi355x", gpu_count=1, power_cap_w=1400, partition_mode="SPX/NPS1") == []
