# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_common import amd_smi, gpu_card
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom_common.gpu_card import (
    GpuCard,
    board_label,
    cards_of,
    one_kind,
    probed_cards,
    publish_session_card,
    read_cards,
    rocr_uuids,
    session_card,
    session_gfx,
)

_RECORDED_MI325X = Path(__file__).parent / "fixtures" / "amd_smi_static_mi325x.json"
_MI325X = GpuCard("AMD Instinct MI325X", "74a5", "gfx942", 304, 262128, 6144.0, 2100)
_MI355X = GpuCard("AMD Instinct MI355X", "75a3", "gfx950", 256, 294912, 8000.0, 2400)


def _row(index=0, *, asic=None, vram=None, clock=None):
    """One ``amd-smi static`` row of an MI325X; ``asic`` replaces single fields, ``vram`` and ``clock`` whole blocks."""
    return {
        "gpu": index,
        "asic": {
            "market_name": "AMD Instinct MI325X",
            "device_id": "0x74A5",
            "num_compute_units": 304,
            "target_graphics_version": "GFX942",
            **(asic or {}),
        },
        "vram": vram or {"size": {"value": 262128, "unit": "MB"}, "max_bandwidth": {"value": 6144, "unit": "GB/s"}},
        "clock": clock or {"sys": {"frequency_levels": {"Level 0": {"value": "112"}, "Level 1": {"value": "2100"}}}},
    }


def test_the_recorded_mi325x_host_reads_as_four_cards_of_one_kind():
    cards = cards_of(json.loads(_RECORDED_MI325X.read_text()))

    assert list(cards) == [0, 1, 2, 3]
    assert set(cards.values()) == {_MI325X}


def test_a_peak_is_compute_units_times_top_clock_times_the_arch_rate():
    assert [_MI325X.peak_tflops(p) for p in ("bf16", "fp16", "fp8", "fp32", "fp4")] == [
        1307.4,
        1307.4,
        2614.9,
        163.4,
        None,
    ]
    assert [_MI355X.peak_tflops(p) for p in ("bf16", "fp8", "fp4", "fp32")] == [2516.6, 5033.2, 10066.3, 157.3]


def test_a_card_without_a_clock_or_on_an_arch_without_rates_has_no_peak():
    assert dataclasses.replace(_MI325X, engine_clock_mhz=None).peak_tflops("bf16") is None
    assert dataclasses.replace(_MI325X, gfx="gfx1250").peak_tflops("bf16") is None


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("AMD Instinct MI300X", "mi300x"),
        ("AMD Instinct MI300X HF", "mi300x"),
        ("AMD Instinct MI300X VF", "mi300x"),
        ("AMD Instinct MI300X OAM", "mi300x"),
        ("AMD Instinct MI300XL", "mi300xl"),
        ("AMD Radeon AI PRO R9700", "r9700"),
        ("AMD Radeon AI PRO R9700S", "r9700s"),
        ("AMD Radeon PRO W7900 Dual Slot", "w7900"),
        ("AMD Radeon PRO W7900D", "w7900d"),
        ("Radeon 8060S Graphics", "radeon8060s"),
        ("AMD Radeon Graphics", ""),
        ("", ""),
    ],
)
def test_a_marketing_name_is_filed_under_its_model_designation(name, label):
    assert board_label(name) == label


@pytest.mark.parametrize("market_name", ["N/A", "0x1586"])
def test_a_card_the_driver_cannot_name_has_no_name(market_name):
    assert cards_of([_row(asic={"market_name": market_name})])[0].name == ""


def test_a_reading_the_card_does_not_report_is_none():
    card = cards_of([_row(vram={"size": {"value": 262128, "unit": "MB"}, "max_bandwidth": "N/A"}, clock="N/A")])[0]

    assert (card.memory_bandwidth_gbps, card.engine_clock_mhz) == (None, None)


@pytest.mark.parametrize(
    ("replaced", "missing"),
    [
        ({"asic": {"target_graphics_version": "N/A"}}, "no arch"),
        ({"asic": {"num_compute_units": "N/A"}}, "no compute-unit count"),
        ({"vram": {"size": "N/A"}}, "no memory size"),
    ],
)
def test_a_card_missing_a_required_fact_is_refused_by_index_and_fact(replaced, missing):
    with pytest.raises(AmdSmiError, match=rf"card 3 reports {missing}$"):
        cards_of({"gpu_data": [_row(3, **replaced)]})


def test_bare_numbers_and_value_unit_readings_parse_alike():
    bare = _row(
        vram={"size": 262128, "max_bandwidth": 6144},
        clock={"sys": {"frequency_levels": {"Level 0": 112, "Level 1": 2100}}},
    )

    assert cards_of([bare]) == cards_of([_row()]) == {0: _MI325X}


def test_cards_that_differ_in_name_bandwidth_or_clock_are_one_kind_named_by_one_that_has_a_label():
    unnamed = dataclasses.replace(
        _MI325X, name="AMD Radeon Graphics", memory_bandwidth_gbps=None, engine_clock_mhz=2102
    )

    assert one_kind([unnamed, _MI325X]) is _MI325X
    assert one_kind([unnamed]) is unnamed


def test_the_clock_is_the_top_level_not_a_live_reading_above_it():
    levels = {"Level 0": {"value": "500"}, "Level 1": {"value": "2402"}, "Level 2": {"value": "2400"}}

    assert cards_of([_row(clock={"sys": {"frequency_levels": levels}})])[0].engine_clock_mhz == 2400


@pytest.mark.parametrize("field", ["gfx", "compute_units", "memory_mib"])
def test_a_card_differing_in_a_computing_fact_is_another_kind(field):
    assert not _MI325X.same_kind(dataclasses.replace(_MI325X, **{field: "gfx950" if field == "gfx" else 1}))


def test_cards_of_more_than_one_kind_or_none_are_refused_listing_each_kind():
    with pytest.raises(ValueError) as mixed:
        one_kind([_MI325X, _MI355X, _MI325X])
    assert str(mixed.value) == (
        "the GPUs are not one kind of card: "
        "2x AMD Instinct MI325X (gfx942, 304 CUs, 262128 MiB, 2100 MHz); "
        "1x AMD Instinct MI355X (gfx950, 256 CUs, 294912 MiB, 2400 MHz)"
    )
    with pytest.raises(ValueError, match="there is none"):
        one_kind([])


def test_a_published_card_is_read_back_and_publishing_none_withdraws_it(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_GPU_CARD", "")
    assert (session_card(), session_gfx()) == (None, "")

    publish_session_card(_MI325X)
    assert (session_card(), session_gfx()) == (_MI325X, "gfx942")

    publish_session_card(None)
    assert session_card() is None


def _amd_smi_answers(monkeypatch, *, static, listed):
    answers = {("static",): static, ("list",): listed}
    monkeypatch.setattr(amd_smi, "read_amd_smi_json", lambda *args: answers[args])


def test_read_cards_numbers_the_cards_as_rocm_does(monkeypatch):
    static = [_row(0, asic={"num_compute_units": 10}), _row(1, asic={"num_compute_units": 11})]
    _amd_smi_answers(monkeypatch, static=static, listed=[{"gpu": 0, "node_id": 5}, {"gpu": 1, "node_id": 2}])

    assert [card.compute_units for card in read_cards()] == [11, 10]


_ROCM_SMI = {
    "card0": {
        "Card Series": "AMD Instinct MI325X",
        "Card Model": "0x74a5",
        "Node ID": "5",
        "GFX Version": "gfx942",
        "VRAM Total Memory (B)": str(262128 << 20),
    },
    "card1": {
        "Card Series": "AMD Instinct MI355X",
        "Card Model": "0x75a3",
        "Node ID": "2",
        "GFX Version": "gfx950",
        "VRAM Total Memory (B)": str(294912 << 20),
    },
}
_TORCH = {
    "torch": [{"name": "AMD Instinct MI325X", "gcnArchName": "gfx942:sramecc+:xnack-", "total_memory": 262128 << 20}]
}
_TORCH_MI325X = GpuCard("AMD Instinct MI325X", "", "gfx942", None, 262128, None, None)


def test_rocm_smi_describes_its_cards_in_rocm_order_without_compute_units_or_clock():
    cards = probed_cards(json.dumps(_ROCM_SMI))

    assert cards == [
        GpuCard("AMD Instinct MI355X", "75a3", "gfx950", None, 294912, None, None),
        GpuCard("AMD Instinct MI325X", "74a5", "gfx942", None, 262128, None, None),
    ]
    assert cards[0].peak_tflops("bf16") is None
    assert cards[0].describe() == "AMD Instinct MI355X (gfx950, 294912 MiB)"


def test_torch_describes_a_cards_name_arch_and_memory_among_other_lines():
    assert probed_cards(f"[INFO] started {{attempt: 1}}\n{json.dumps(_TORCH)}\n[INFO] done\n") == [_TORCH_MI325X]


def test_output_that_describes_no_card_or_one_without_its_arch_is_refused():
    with pytest.raises(AmdSmiError, match="no GPU is described in 'amd-smi: command not found'"):
        probed_cards("amd-smi: command not found\n")
    with pytest.raises(AmdSmiError, match="rocm-smi reports a card without its arch"):
        probed_cards(json.dumps({"card0": {"Card Series": "AMD Instinct MI300X", "VRAM Total Memory (B)": "1048576"}}))


def test_a_uuid_device_mask_names_a_card_by_its_serial_in_rocm_order(monkeypatch):
    static = [_row(0, asic={"asic_serial": "0x41311B60902C6BF9"}), _row(1, asic={"asic_serial": "0x93E19F9622C13CBB"})]
    _amd_smi_answers(monkeypatch, static=static, listed=[{"gpu": 0, "node_id": 5}, {"gpu": 1, "node_id": 2}])

    assert rocr_uuids() == ["gpu-93e19f9622c13cbb", "gpu-41311b60902c6bf9"]


def _probes_answer(monkeypatch, answers):
    """amd-smi cannot run; each other probe prints its entry of ``answers``, or fails when it has none."""

    def unreadable(*_args):
        raise AmdSmiError("amd-smi static could not run")

    def output(probe):
        tool = probe.split()[0]
        if tool not in answers:
            raise AmdSmiError(f"{tool} could not run")
        return json.dumps(answers[tool])

    monkeypatch.setattr(amd_smi, "read_amd_smi_json", unreadable)
    monkeypatch.setattr(gpu_card, "_probe_output", output)


def test_a_machine_amd_smi_cannot_describe_is_asked_through_rocm_smi_then_torch(monkeypatch):
    _probes_answer(monkeypatch, {"rocm-smi": _ROCM_SMI, "python3": _TORCH})
    assert [card.gfx for card in read_cards()] == ["gfx950", "gfx942"]

    _probes_answer(monkeypatch, {"python3": _TORCH})
    assert read_cards() == [_TORCH_MI325X]


def test_a_machine_no_probe_describes_cannot_be_read_and_gives_each_reason(monkeypatch):
    _probes_answer(monkeypatch, {})

    with pytest.raises(
        AmdSmiError, match="amd-smi static could not run; rocm-smi could not run; python3 could not run"
    ):
        read_cards()
