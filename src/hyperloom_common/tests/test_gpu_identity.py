# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom_common.gpu_identity import AMD_GPU_BOARDS, gpu_type_for_device_id, normalise_device_id


def test_the_boards_known_by_their_name_alone_list_no_device_id():
    assert {board.product for board in AMD_GPU_BOARDS.values() if not board.device_ids} == {"r9700", "radeon8060s"}


def test_a_device_id_belongs_to_one_board():
    listed = [device_id for board in AMD_GPU_BOARDS.values() for device_id in board.device_ids]
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize(
    ("device_id", "board"),
    [
        ("0x75a3", "mi355x"),
        ("0x75A3", "mi355x"),
        (0x75A3, "mi355x"),
        ("0x75a0", "mi350x"),
        ("0x75b0", "mi350x"),
        ("0x74a1", "mi300x"),
        ("0x74a5", "mi325x"),
    ],
)
def test_the_boards_the_measured_nodes_report(device_id, board):
    assert gpu_type_for_device_id(device_id) == board


def test_the_mi350x_is_the_mi355x_die_at_the_lower_peak_clock():
    """AMD publishes the MI350X at a 2.2 GHz peak engine clock and the MI355X at 2.4 GHz, on the same gfx950 die."""
    mi350x, mi355x = AMD_GPU_BOARDS["mi350x"], AMD_GPU_BOARDS["mi355x"]
    assert (mi350x.gfx, mi350x.compute_units, mi350x.hbm_gb, mi350x.hbm_bw_gbps) == (
        mi355x.gfx,
        mi355x.compute_units,
        mi355x.hbm_gb,
        mi355x.hbm_bw_gbps,
    )
    assert mi350x.vendor_peak_tflops.keys() == mi355x.vendor_peak_tflops.keys()
    for precision, peak in mi355x.vendor_peak_tflops.items():
        assert mi350x.vendor_peak_tflops[precision] == pytest.approx(peak * 2.2 / 2.4, abs=0.1), precision


@pytest.mark.parametrize("device_id", [None, "", "0x", "not-an-id", "0x0000", True, "0x74a2", "0x74b6", "0x7551"])
def test_an_unknown_ambiguous_or_unreadable_id_names_no_board(device_id):
    assert gpu_type_for_device_id(device_id) is None


def test_normalise_device_id():
    assert normalise_device_id(" 0x75A3 ") == "75a3"
    assert normalise_device_id(None) is None
