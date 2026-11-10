# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor's map as the scheduler deals it: each direction read with the share it is in, and held by the branch that
share was dealt to."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import MapDirection
from hyperloom_multibranch.scheduling.agent_map import SharedMap

GEMM = {"id": "m-gemm", "kind": "increment", "title": "Dense   GEMMs: tuned kernels", "status": "open"}
FUSE = {"id": "m-fuse", "title": "Small kernels", "status": "open", "why": "rms_norm_kernel 4 %", "ops": ["rms_norm"]}
FP8 = {"id": "m-fp8", "title": "FP8 weights", "status": "killed", "why": "root is BF16"}


def _write(path: Path, *directions: dict[str, object]) -> str:
    path.write_text(json.dumps({"directions": list(directions)}))
    return str(path)


def _shared(tmp_path: Path) -> tuple[SharedMap, str, str]:
    one, two = _write(tmp_path / "share-1.json", GEMM), _write(tmp_path / "share-2.json", FUSE)
    return (
        SharedMap.read(Path(_write(tmp_path / "directions.json", GEMM, FUSE, FP8, {"title": "no id"})), (one, two)),
        one,
        two,
    )


def test_a_map_is_read_with_the_share_of_each_direction_and_no_holder_until_one_is_dealt(tmp_path: Path) -> None:
    shared, one, two = _shared(tmp_path)

    assert shared.directions == (
        MapDirection("m-gemm", "Dense GEMMs: tuned kernels", "open"),
        MapDirection("m-fuse", "Small kernels", "open", "rms_norm_kernel 4 %", ops=("rms_norm",)),
        MapDirection("m-fp8", "FP8 weights", "killed", "root is BF16"),
    )
    assert shared.shares == {"m-gemm": one, "m-fuse": two}


def test_a_share_dealt_to_a_branch_makes_it_the_holder_of_that_share_s_directions_only(tmp_path: Path) -> None:
    shared, one, _two = _shared(tmp_path)

    holders = {item.id: item.holder for item in shared.dealt(one, "b03").directions}

    assert holders == {"m-gemm": "b03", "m-fuse": "", "m-fp8": ""}


def test_a_map_that_cannot_be_read_is_empty(tmp_path: Path) -> None:
    assert SharedMap.read(tmp_path / "missing.json", ()) == SharedMap((), {})
