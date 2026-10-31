# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.common.rocm_smi import GpuVram
from hyperloom.measure.runner.hardware import (
    EccCounts,
    GpuFacts,
    busy_gpus,
    check_hardware,
    ecc_counts,
    read_gpu_facts,
)


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
    assert ecc_counts([0], _runner({}), _runner({})) is None


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


def _card(model, device_id, arch):
    return GpuFacts(0, model, 1400.0, "SPX/NPS1", device_id, arch)


def _wrong_card(facts, gpu_type):
    return check_hardware([facts], gpu_type=gpu_type, gpu_count=1, power_cap_w=None, partition_mode=None)


def test_the_device_id_and_arch_are_read_next_to_the_market_name():
    static = json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "asic": {
                        "market_name": "AMD Instinct MI355 OAM",
                        "device_id": "0x75A3",
                        "target_graphics_version": "gfx950",
                    },
                }
            ]
        }
    )

    facts = read_gpu_facts([0], _runner({"static --json": static}))

    assert facts == [GpuFacts(0, "AMD Instinct MI355 OAM", None, None, "75a3", "gfx950")]


def test_a_node_whose_market_name_is_mi355_oam_is_still_an_mi355x():
    """autopilot-crsuse2-m2m-188 reports 'AMD Instinct MI355 OAM' for device 0x75a3; the old name match refused every test."""
    oam = _card("AMD Instinct MI355 OAM", "75a3", "gfx950")

    for requested in ("mi355x", "MI355X", "MI355X "):
        assert _wrong_card(oam, requested) == []
    assert _wrong_card(_card("AMD Instinct MI355X", "75a3", "gfx950"), "MI355X") == []


def test_a_different_card_is_refused_whatever_its_name_says():
    mi300x = _card("AMD Instinct MI300X", "74a1", "gfx942")

    problems = _wrong_card(mi300x, "MI355X")

    assert problems == ["GPU 0 is AMD Instinct MI300X (device id 74a1, gfx942), not MI355X"]
    # the device id decides, so a name that happens to contain the requested type does not rescue a different board
    assert _wrong_card(_card("AMD Instinct MI355X", "74a1", "gfx942"), "MI355X")
    # and the reverse request is refused too
    assert _wrong_card(_card("AMD Instinct MI355 OAM", "75a3", "gfx950"), "MI300X")
    assert _wrong_card(_card("AMD Instinct MI325X", "74a5", "gfx942"), "MI300X")


def test_an_unlisted_device_id_is_judged_by_architecture_and_then_by_name():
    assert _wrong_card(_card("AMD Instinct MI355X", "ffff", "gfx942"), "MI355X")  # another architecture
    assert _wrong_card(_card("AMD Instinct MI355X", "ffff", "gfx950"), "MI355X") == []
    assert _wrong_card(
        _card("AMD Instinct MI300X", "ffff", "gfx950"), "MI355X"
    )  # same arch, a name that says otherwise
    # 0x74a2 is an MI308X and, at revision A1, an MI300X: it names no board, so the name decides
    assert _wrong_card(_card("AMD Instinct MI300X", "74a2", "gfx942"), "MI300X") == []
    assert _wrong_card(_card("AMD Instinct MI308X", "74a2", "gfx942"), "MI308X") == []
    assert _wrong_card(_card("AMD Instinct MI308X", "74a2", "gfx942"), "MI300X")


def test_a_type_without_a_device_id_table_is_compared_by_name_as_before():
    assert _wrong_card(_card("AMD Instinct MI400X", "75a3", "gfx950"), "mi400x") == []
    assert _wrong_card(_card("AMD Instinct MI355X", "75a3", "gfx950"), "mi400x")


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


def test_the_ras_fallback_tells_new_uncorrectable_errors_from_the_counts_before():
    before = ecc_counts([0], _runner({}), _runner({"--showrasinfo": _ras({"UMC": (3, 1)})}))
    after = ecc_counts([0], _runner({}), _runner({"--showrasinfo": _ras({"UMC": (3, 2)})}))

    assert before[0].uncorrectable < after[0].uncorrectable


def test_busy_gpus_are_the_visible_ones_with_memory_resident():
    free, held = GpuVram(284.0, 295000.0), GpuVram(120000.0, 295000.0)

    assert busy_gpus([0, 1], lambda: [free, held]) == [1]
    assert busy_gpus([0], lambda: [free, held]) == [], "a card this sandbox cannot use is not its concern"
    assert busy_gpus([0, 1], lambda: [free, free]) == []


def test_memory_use_that_cannot_be_read_is_none_not_free():
    held = GpuVram(120000.0, 295000.0)

    assert busy_gpus([0], lambda: None) is None
    assert busy_gpus([0, 3], lambda: [held, held]) is None, "a visible card the tool did not list is unread"
