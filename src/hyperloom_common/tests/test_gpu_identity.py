# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom_common.gpu_identity import (
    AMD_GPU_BOARDS,
    AMD_GPU_DEVICE_IDS,
    gpu_type_for_device_id,
    normalise_device_id,
)


def test_every_board_has_its_device_ids_listed_and_the_other_way_round():
    assert set(AMD_GPU_DEVICE_IDS) == set(AMD_GPU_BOARDS)
    assert {board for board, ids in AMD_GPU_DEVICE_IDS.items() if not ids} == {"r9700", "radeon8060s"}


def test_a_device_id_belongs_to_one_board():
    listed = [device_id for ids in AMD_GPU_DEVICE_IDS.values() for device_id in ids]
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize(
    ("device_id", "board"),
    [("0x75a3", "mi355x"), ("0x75A3", "mi355x"), (0x75A3, "mi355x"), ("0x74a1", "mi300x"), ("0x74a5", "mi325x")],
)
def test_the_boards_the_measured_nodes_report(device_id, board):
    assert gpu_type_for_device_id(device_id) == board


@pytest.mark.parametrize("device_id", [None, "", "0x", "not-an-id", "0x0000", True, "0x74a2", "0x74b6", "0x7551"])
def test_an_unknown_ambiguous_or_unreadable_id_names_no_board(device_id):
    assert gpu_type_for_device_id(device_id) is None


def test_normalise_device_id():
    assert normalise_device_id(" 0x75A3 ") == "75a3"
    assert normalise_device_id(None) is None
