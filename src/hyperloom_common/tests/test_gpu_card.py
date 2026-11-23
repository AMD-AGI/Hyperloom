# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom_common.gpu_card import (
    GpuCard,
    board_label,
    cards_of,
    one_kind,
    publish_session_card,
    read_cards,
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


def test_cards_that_differ_only_in_name_are_one_kind():
    renamed = dataclasses.replace(_MI325X, name="AMD Instinct MI325X VF", device_id="74b9", memory_bandwidth_gbps=None)

    assert one_kind([_MI325X, renamed]) is _MI325X


@pytest.mark.parametrize("field", ["gfx", "compute_units", "memory_mib", "engine_clock_mhz"])
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


def test_a_machine_whose_amd_smi_lists_no_card_cannot_be_read(monkeypatch):
    _amd_smi_answers(monkeypatch, static={"gpu_data": []}, listed=[])

    with pytest.raises(AmdSmiError, match="reports no GPU"):
        read_cards()
