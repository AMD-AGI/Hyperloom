# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor's map as the scheduler deals it: each direction read with the share it is in, and held by the branch that
share was dealt to; and a worker's report and the next order Arbor's master deals folded back in."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import MapDirection
from hyperloom_multibranch.scheduling.agent_map import SharedMap
from hyperloom_multibranch.tests.agent_rig import worker_report

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


def test_a_report_closes_its_directions_and_a_dealt_order_moves_its_own_to_the_branch_it_opens_on(
    tmp_path: Path,
) -> None:
    shared, one, two = _shared(tmp_path)
    held = shared.dealt(one, "b01").dealt(two, "b02")
    closed = ({"id": "m-gemm", "status": "done", "why": "KEEP 2"}, {"id": "m-fp8", "status": "done", "why": "late"})
    report = worker_report(tmp_path / "worker_report.json", *closed, left_open=("m-fuse",))
    order = Path(_write(tmp_path / "order-1.json", FUSE))

    folded = held.folded(report, order)

    # m-fp8 was set aside in the map before the report closed it, and keeps the map's verdict.

    assert [(item.id, item.status, item.why, item.holder) for item in folded.directions] == [
        ("m-gemm", "done", "KEEP 2", "b01"),
        ("m-fuse", "open", "rms_norm_kernel 4 %", "b02"),
        ("m-fp8", "killed", "root is BF16", ""),
    ]
    assert folded.shares == {"m-gemm": one, "m-fuse": str(order)}
    holders = {item.id: item.holder for item in folded.dealt(str(order), "b04").directions}
    assert holders == {"m-gemm": "b01", "m-fuse": "b04", "m-fp8": ""}
    assert held.folded(report, None).shares == held.shares


def test_a_report_that_cannot_be_read_closes_nothing(tmp_path: Path) -> None:
    shared, _one, _two = _shared(tmp_path)

    assert shared.folded(tmp_path / "missing.json", None) == shared
