# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest
from hyperloom_common import amd_smi

from hyperloom.common.rocm_smi import GpuVram
from hyperloom.measure.runner.hardware import (
    EccCounts,
    GpuFacts,
    amd_smi_or_none,
    busy_gpus,
    check_hardware,
    ecc_counts,
    read_gpu_facts,
)
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD, MI355X_CARD

MI300X_CARD = dataclasses.replace(MI325X_CARD, name="AMD Instinct MI300X", device_id="74a1", memory_mib=196592)


def _runner(answers):
    return lambda args: answers.get(" ".join(args))


def _static_row(index: int, *, name: str = "AMD Instinct MI355X", **extra) -> dict:
    """An ``amd-smi static`` row of an MI355X as amd-smi 27 prints it, under ``name``."""
    return {
        "gpu": index,
        "asic": {
            "market_name": name,
            "device_id": "0x75A3",
            "num_compute_units": 256,
            "target_graphics_version": "gfx950",
        },
        "vram": {"size": {"value": 294912, "unit": "MB"}, "max_bandwidth": {"value": 8000, "unit": "GB/s"}},
        "clock": {
            "sys": {
                "frequency_levels": {
                    "Level 0": {"value": "400", "unit": "MHz"},
                    "Level 1": {"value": "2400", "unit": "MHz"},
                }
            }
        },
        **extra,
    }


STATIC = json.dumps(
    {
        "gpu_data": [
            _static_row(i, limit={"ppt0": {"socket_power_limit": {"value": 1400, "unit": "W"}}}) for i in range(4)
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

    assert facts == [GpuFacts(1, MI355X_CARD, 1400.0, "SPX/NPS1"), GpuFacts(2, MI355X_CARD, 1400.0, "SPX/NPS1")]


@pytest.mark.parametrize(
    "limit",
    [
        {"ppt": {"socket_power_limit": {"value": 750, "unit": "W"}}},
        {"socket_power_limit": {"value": 750, "unit": "W"}},
        {"power_cap": {"value": 750, "unit": "W"}},
    ],
)
def test_the_power_cap_is_read_in_every_layout_amd_smi_reports_it(limit):
    static = json.dumps({"gpu_data": [_static_row(0, limit=limit)]})

    (facts,) = read_gpu_facts([0], _runner({"static --json": static}))

    assert facts.power_cap_w == 750.0


def test_a_power_cap_within_the_rounding_of_amd_smi_matches_the_sheet():
    facts = [GpuFacts(0, MI355X_CARD, 749.8, "SPX/NPS1")]

    assert check_hardware(facts, card=MI355X_CARD, gpu_count=1, power_cap_w=750, partition_mode=None) == []
    assert check_hardware(facts, card=MI355X_CARD, gpu_count=1, power_cap_w=749, partition_mode=None) == [
        "GPU 0 power cap 749.8 W, not 749 W"
    ]


def test_an_amd_smi_that_fails_reads_as_none_instead_of_raising(monkeypatch):
    def run(cmd, **_kwargs):
        if cmd[1] == "missing":
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0 if cmd[1] == "static" else 2, "{}", "failed")

    monkeypatch.setattr(amd_smi.subprocess, "run", run)

    assert amd_smi_or_none(["static", "--json"]) == "{}"
    assert amd_smi_or_none(["metric", "--json"]) is None
    assert amd_smi_or_none(["missing"]) is None


def test_an_unreadable_tool_gives_none_and_not_a_guess():
    assert read_gpu_facts([0], _runner({})) is None
    assert read_gpu_facts([0], _runner({"static --json": "not json"})) is None
    assert ecc_counts([0], _runner({}), _runner({})) is None


def test_a_visible_card_amd_smi_cannot_describe_gives_none_and_one_that_is_not_visible_does_not_matter():
    undescribed = {"gpu": 1, "asic": {"market_name": "AMD Instinct MI355X"}}
    static = json.dumps({"gpu_data": [_static_row(0), undescribed]})

    assert read_gpu_facts([0, 1], _runner({"static --json": static})) is None
    assert read_gpu_facts([0], _runner({"static --json": static})) == [GpuFacts(0, MI355X_CARD, None, None)]


def test_ecc_counts_need_every_visible_gpu():
    run, no_ras = _runner({"metric --ecc --json": ECC}), _runner({})

    assert ecc_counts([1, 2], run, no_ras) == {1: EccCounts(1, 0), 2: EccCounts(2, 0)}
    assert ecc_counts([1, 9], run, no_ras) is None


def test_hardware_mismatches_are_named():
    facts = [GpuFacts(0, MI355X_CARD, 1400.0, "SPX/NPS1")]

    assert check_hardware(facts, card=MI355X_CARD, gpu_count=1, power_cap_w=1400, partition_mode="SPX/NPS1") == []
    problems = check_hardware(facts, card=MI300X_CARD, gpu_count=2, power_cap_w=750, partition_mode="CPX/NPS4")
    assert problems == [
        "1 GPUs visible, 2 needed",
        "GPU 0 is AMD Instinct MI355X (gfx950, 256 CUs, 294912 MiB, 2400 MHz), "
        "not AMD Instinct MI300X (gfx942, 304 CUs, 196592 MiB, 2100 MHz)",
        "GPU 0 power cap 1400.0 W, not 750 W",
        "GPU 0 partition SPX/NPS1, not CPX/NPS4",
    ]


def test_a_value_the_machine_did_not_give_is_not_compared():
    facts = [GpuFacts(0, MI355X_CARD, None, None)]

    assert check_hardware(facts, card=MI355X_CARD, gpu_count=1, power_cap_w=1400, partition_mode="SPX/NPS1") == []


def _kind_problems(card_on_the_machine, card_on_the_sheet):
    facts = [GpuFacts(0, card_on_the_machine, 1400.0, "SPX/NPS1")]
    return check_hardware(facts, card=card_on_the_sheet, gpu_count=1, power_cap_w=None, partition_mode=None)


def test_a_card_of_the_same_kind_passes_whatever_its_name_says():
    """A node whose VBIOS calls an MI355X 'AMD Instinct MI355 OAM' runs the sheet's MI355X tests."""
    oam = dataclasses.replace(MI355X_CARD, name="AMD Instinct MI355 OAM", device_id="")

    assert _kind_problems(oam, MI355X_CARD) == []


def test_another_kind_of_card_is_refused_whatever_its_name_says():
    assert _kind_problems(MI325X_CARD, MI355X_CARD) != []
    # the same arch and compute units with less memory is another card
    assert _kind_problems(MI300X_CARD, MI325X_CARD) != []
    # a name that matches the sheet's does not make a different card the same
    assert _kind_problems(dataclasses.replace(MI325X_CARD, name=MI355X_CARD.name), MI355X_CARD) != []
    # nor does a lower top engine clock pass for the sheet's
    assert _kind_problems(dataclasses.replace(MI355X_CARD, engine_clock_mhz=2100), MI355X_CARD) != []


def test_a_sheet_without_a_card_is_not_compared_with_the_machine_s_cards():
    assert _kind_problems(MI325X_CARD, None) == []


def _ras(*gpus: dict[str, tuple[int, int] | None]) -> str:
    """``rocm-smi --showrasinfo`` for each GPU's blocks: ``(correctable, uncorrectable)``, or ``None`` for a block that lists none."""
    lines = ["", "======================================== RAS Info ========================================", ""]
    for index, blocks in enumerate(gpus):
        lines += [f"GPU[{index}]: \tRAS INFO", "         Block       Status    Correctable Error  Uncorrectable Error"]
        for name, counts in blocks.items():
            cells = "" if counts is None else f"{counts[0]:>19}{counts[1]:>21}"
            lines.append(f"{name:>14}        ENABLED{cells}")
        lines += ["           SEM       DISABLED", "_" * 90, ""]
    return "\n".join(lines)


RAS = _ras(
    {"UMC": (3, 1), "SDMA": (0, 0), "GFX": (2, 0), "ATHUB": None},
    {"UMC": (0, 0), "SDMA": (0, 0), "ATHUB": None},
)


def test_ecc_counts_fall_back_to_the_ras_information_when_amd_smi_is_absent():
    counts = ecc_counts([0, 1], _runner({}), _runner({"--showrasinfo": RAS}))

    assert counts == {0: EccCounts(5, 1), 1: EccCounts(0, 0)}


def test_the_ras_counts_are_those_of_the_visible_gpus_only():
    assert ecc_counts([1], _runner({}), _runner({"--showrasinfo": RAS})) == {1: EccCounts(0, 0)}


def test_amd_smi_is_asked_first_and_the_ras_information_only_when_it_cannot_say():
    ras = _runner({"--showrasinfo": RAS})

    assert ecc_counts([0, 1], _runner({"metric --ecc --json": ECC}), ras) == {
        0: EccCounts(0, 0),
        1: EccCounts(1, 0),
    }
    assert ecc_counts([0, 1], _runner({"metric --ecc --json": "not json"}), ras) == {
        0: EccCounts(5, 1),
        1: EccCounts(0, 0),
    }


def test_ras_information_that_does_not_cover_every_visible_gpu_is_not_a_reading():
    assert ecc_counts([0, 2], _runner({}), _runner({"--showrasinfo": RAS})) is None
    assert ecc_counts([0], _runner({}), _runner({"--showrasinfo": "WARNING: no RAS"})) is None
    # a GPU whose blocks list no counts says nothing about its errors
    assert ecc_counts([0], _runner({}), _runner({"--showrasinfo": _ras({"ATHUB": None})})) is None


def test_busy_gpus_are_the_visible_ones_with_memory_resident():
    free, held = GpuVram(300.0, 300000.0), GpuVram(120000.0, 300000.0)

    assert busy_gpus([0, 1], lambda: [free, held]) == [1]
    assert busy_gpus([0], lambda: [free, held]) == [], "a card this sandbox cannot use is not its concern"
    assert busy_gpus([0, 1], lambda: [free, free]) == []


def test_memory_use_that_cannot_be_read_is_none_not_free():
    held = GpuVram(120000.0, 300000.0)

    assert busy_gpus([0], lambda: None) is None
    assert busy_gpus([0, 3], lambda: [held, held]) is None, "a visible card the tool did not list is unread"
