# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import dataclasses
import json
from pathlib import Path

import pytest

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom_common.gpu_card import (
    PEAK_PRECISIONS,
    SESSION_CARD_ENV,
    GpuCard,
    board_label,
    cards_of,
    one_kind,
    peak_rate,
    publish_session_card,
    read_cards,
    session_card,
)

_RECORDED_MI325X = Path(__file__).parent / "fixtures" / "amd_smi_static_mi325x.json"
_MI325X = GpuCard("AMD Instinct MI325X", "74a5", "gfx942", 304, 262128, 6144.0, 2100)
_MI355X = GpuCard("AMD Instinct MI355X", "75a3", "gfx950", 256, 294912, 8000.0, 2400)


def _row(index=0, *, asic=None, vram=None, clock=None):
    """One ``amd-smi static`` row of an MI325X, with whole blocks or single ``asic`` fields replaced."""
    return {
        "gpu": index,
        "asic": {
            "market_name": "AMD Instinct MI325X",
            "device_id": "0x74a5",
            "num_compute_units": 304,
            "target_graphics_version": "gfx942",
            **(asic or {}),
        },
        "vram": {"size": {"value": 262128, "unit": "MB"}, "max_bandwidth": {"value": 6144, "unit": "GB/s"}}
        if vram is None
        else vram,
        "clock": {"sys": {"frequency_levels": {"Level 0": {"value": "112"}, "Level 1": {"value": "2100"}}}}
        if clock is None
        else clock,
    }


def test_the_recorded_mi325x_host_reads_as_four_cards_of_one_kind():
    cards = cards_of(json.loads(_RECORDED_MI325X.read_text()))

    assert list(cards) == [0, 1, 2, 3]
    assert set(cards.values()) == {_MI325X}
    assert dataclasses.astuple(cards[0]) == ("AMD Instinct MI325X", "74a5", "gfx942", 304, 262128, 6144.0, 2100)


def test_a_card_peak_is_its_compute_units_times_its_clock_times_the_arch_rate():
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
    assert peak_rate("gfx1250", "bf16") is None


def test_only_gfx950_has_an_fp4_rate():
    arches = ("gfx942", "gfx950", "gfx1201", "gfx1100", "gfx1151")
    assert [arch for arch in arches if peak_rate(arch, "fp4")] == ["gfx950"]
    assert all(peak_rate("gfx950", precision) for precision in PEAK_PRECISIONS)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("AMD Instinct MI300X", "mi300x"),
        ("AMD Instinct MI300X HF", "mi300x"),
        ("AMD Instinct MI300X VF", "mi300x"),
        ("AMD Instinct MI300X OAM", "mi300x"),
        ("AMD Instinct MI308X", "mi308x"),
        ("AMD Instinct MI325X", "mi325x"),
        ("AMD Instinct MI350X", "mi350x"),
        ("AMD Instinct MI355X", "mi355x"),
        ("AMD Radeon AI PRO R9700", "r9700"),
        ("AMD Radeon AI Pro R9600D", "r9600d"),
        ("AMD Radeon PRO W7900", "w7900"),
        ("AMD Radeon PRO W7900 Dual Slot", "w7900"),
        ("AMD Radeon PRO W7900D", "w7900d"),
        ("Radeon 8060S Graphics", "radeon8060s"),
        ("AMD Radeon 8060S Graphics", "radeon8060s"),
        ("AMD Radeon8060S Graphics", "radeon8060s"),
        ("Radeon 8050S Graphics", "radeon8050s"),
    ],
)
def test_a_marketing_name_is_filed_under_its_model_designation(name, label):
    assert board_label(name) == label


def test_a_longer_designation_is_another_board():
    assert board_label("AMD Radeon PRO W7900D") != board_label("AMD Radeon PRO W7900")
    assert board_label("AMD Instinct MI300XL") == "mi300xl"
    assert board_label("AMD Radeon AI PRO R9700S") == "r9700s"
    assert board_label("AMD Radeon RX 9070 XT") != board_label("AMD Radeon RX 9070")


@pytest.mark.parametrize("name", ["AMD Radeon Graphics", ""])
def test_a_name_without_a_model_designation_has_no_label(name):
    assert board_label(name) == ""


@pytest.mark.parametrize("market_name", ["N/A", "0x1586", None])
def test_a_card_the_driver_cannot_name_has_no_name_and_so_no_label(market_name):
    name = cards_of([_row(asic={"market_name": market_name})])[0].name

    assert name == "" and board_label(name) == ""


def test_a_reading_the_card_does_not_report_is_none():
    card = cards_of([_row(vram={"size": {"value": 262128, "unit": "MB"}, "max_bandwidth": "N/A"}, clock="N/A")])[0]

    assert (card.memory_bandwidth_gbps, card.engine_clock_mhz) == (None, None)
    assert cards_of([_row(asic={"device_id": "N/A"})])[0].device_id == ""


@pytest.mark.parametrize(
    ("replaced", "missing"),
    [
        ({"asic": {"target_graphics_version": "N/A"}}, "no arch"),
        ({"asic": {"num_compute_units": 0}}, "no compute-unit count"),
        ({"vram": {"size": "N/A"}}, "no memory size"),
        ({"asic": {"target_graphics_version": None}, "vram": "N/A"}, "no arch, no memory size"),
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


@pytest.mark.parametrize(
    ("size", "mib"),
    [
        ({"value": 256, "unit": "GB"}, 262144),
        ({"value": 262128, "unit": "MiB"}, 262128),
        ({"value": 2**35, "unit": "B"}, 32768),
        ({"value": 262128}, 262128),
    ],
)
def test_memory_is_read_in_the_unit_it_is_reported_in(size, mib):
    assert cards_of([_row(vram={"size": size})])[0].memory_mib == mib


@pytest.mark.parametrize("key", ["num_compute_units", "compute_units", "num_cu", "cu_count"])
def test_the_compute_unit_count_is_read_under_each_spelling_amd_smi_has_used(key):
    asic = {"num_compute_units": "N/A", key: 228}
    assert cards_of([_row(asic=asic)])[0].compute_units == 228


def test_the_arch_and_device_id_are_lower_case():
    card = cards_of([_row(asic={"target_graphics_version": "GFX942", "device_id": "0x74A5"})])[0]
    assert (card.gfx, card.device_id) == ("gfx942", "74a5")


@pytest.mark.parametrize(
    ("reported", "device_id"),
    [(" 0x75A3 ", "75a3"), (0x75A3, "75a3"), ("75b0", "75b0"), (None, ""), ("0x", ""), ("not-an-id", ""), (True, "")],
)
def test_a_device_id_reads_as_bare_hex_and_one_that_is_not_an_id_as_empty(reported, device_id):
    assert cards_of([_row(asic={"device_id": reported})])[0].device_id == device_id


def test_an_arch_without_a_published_rate_at_a_precision_gives_no_peak_there():
    """AMD publishes only the FP16 rate of the Radeon 8060S's gfx1151; BF16 is read from it, FP8 and FP32 have none."""
    strix_halo = GpuCard("Radeon 8060S Graphics", "1586", "gfx1151", 40, 65536, None, 2900)

    assert strix_halo.peak_tflops("bf16") == strix_halo.peak_tflops("fp16") == 59.4
    assert [strix_halo.peak_tflops(p) for p in ("fp8", "fp32", "fp4")] == [None, None, None]


def test_cards_that_differ_only_in_name_are_one_kind():
    renamed = dataclasses.replace(_MI325X, name="AMD Instinct MI325X VF", device_id="74b9", memory_bandwidth_gbps=None)

    assert _MI325X.same_kind(renamed)
    assert one_kind([_MI325X, renamed]) is _MI325X


@pytest.mark.parametrize("field", ["gfx", "compute_units", "memory_mib", "engine_clock_mhz"])
def test_a_card_differing_in_a_computing_fact_is_another_kind(field):
    other = dataclasses.replace(_MI325X, **{field: "gfx950" if field == "gfx" else 1})
    assert not _MI325X.same_kind(other)


def test_mixed_cards_are_refused_with_each_kind_listed():
    with pytest.raises(ValueError) as refused:
        one_kind([_MI325X, _MI355X, _MI325X])

    assert str(refused.value) == (
        "the GPUs are not one kind of card: "
        "2x AMD Instinct MI325X (gfx942, 304 CUs, 262128 MiB, 2100 MHz); "
        "1x AMD Instinct MI355X (gfx950, 256 CUs, 294912 MiB, 2400 MHz)"
    )


def test_no_cards_are_not_a_kind():
    with pytest.raises(ValueError, match="no GPU"):
        one_kind([])


def test_a_published_card_is_read_back_and_publishing_none_withdraws_it(monkeypatch):
    monkeypatch.delenv(SESSION_CARD_ENV, raising=False)
    assert session_card() is None

    publish_session_card(_MI325X)
    assert session_card() == _MI325X
    assert session_card({SESSION_CARD_ENV: json.dumps(dataclasses.asdict(_MI355X))}) == _MI355X

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


def test_a_card_listed_but_not_described_cannot_be_read(monkeypatch):
    _amd_smi_answers(monkeypatch, static=[_row(0)], listed=[{"gpu": 0, "node_id": 1}, {"gpu": 1, "node_id": 2}])

    with pytest.raises(AmdSmiError, match="names card 1"):
        read_cards()
