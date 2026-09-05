# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The trace nominator: trace-ranked, definition-resolved, share-weighted.

The three behaviours here are the ones the placeholder in ``nomination.stub``
explicitly does not have, so each is pinned against the stub's opposite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge import nomination as nom
from kernelforge.nomination import trace_nominator as tn


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request(tmp_path: Path, **overrides: object) -> nom.NominationRequest:
    payload: dict[str, object] = {
        "protocol_version": nom.PROTOCOL_VERSION,
        "lane": nom.LANE_REWRITE,
        "trace_path": "",
        "candidates_path": "/tmp/kernel_candidates.json",
        "lane_budget_sec": 6000,
        "max_kernels": 2,
        "trace_captured_after": "abc1234",
    }
    payload.update(overrides)
    return nom.read_request(_write(tmp_path / "req.json", payload))


def _candidates(tmp_path: Path, rows: list[dict[str, object]]) -> list[nom.Candidate]:
    payload = {"manifest_version": nom.MANIFEST_VERSION, "hot_kernels": rows}
    return nom.read_candidates(_write(tmp_path / "cand.json", payload))


def _row(name: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "kernel_name": name,
        "source_file": f"/repo/{name}.py",
        "gpu_pct": 1.0,
        "reason_class": "resolved",
        "attempts": 0,
        "rejected": False,
    }
    row.update(overrides)
    return row


def test_trace_shares_override_the_row(tmp_path: Path) -> None:
    """A row's own gpu_pct may be a whole-run average; the window wins."""
    trace = _write(
        tmp_path / "trace.json",
        {
            "manifest_version": 2,
            "hot_kernels": [
                {"kernel_name": "cold_in_row", "gpu_pct": 40.0},
                {"kernel_name": "hot_in_row", "gpu_pct": 2.0},
            ],
        },
    )
    request = _request(tmp_path, trace_path=str(trace))
    candidates = _candidates(
        tmp_path,
        [_row("hot_in_row", gpu_pct=90.0), _row("cold_in_row", gpu_pct=3.0)],
    )
    targets = tn.nominate_from_trace(request, candidates)
    # Ranked by the trace (40 vs 2), not by the rows (90 vs 3).
    assert [t.kernel_name for t in targets] == ["cold_in_row", "hot_in_row"]


def test_budget_is_split_by_share_not_evenly(tmp_path: Path) -> None:
    trace = _write(
        tmp_path / "trace.json",
        {
            "manifest_version": 2,
            "hot_kernels": [
                {"kernel_name": "big", "gpu_pct": 75.0},
                {"kernel_name": "small", "gpu_pct": 25.0},
            ],
        },
    )
    request = _request(tmp_path, trace_path=str(trace), lane_budget_sec=8000)
    targets = tn.nominate_from_trace(request, _candidates(tmp_path, [_row("big"), _row("small")]))
    budgets = {t.kernel_name: t.budget_sec for t in targets}
    assert budgets["big"] == 6000
    assert budgets["small"] == 2000


def test_budget_never_falls_below_the_floor(tmp_path: Path) -> None:
    """A session too short to finish a measure/keep cycle produces nothing."""
    trace = _write(
        tmp_path / "trace.json",
        {
            "manifest_version": 2,
            "hot_kernels": [
                {"kernel_name": "dominant", "gpu_pct": 99.0},
                {"kernel_name": "marginal", "gpu_pct": 1.5},
            ],
        },
    )
    request = _request(tmp_path, trace_path=str(trace), lane_budget_sec=1000)
    targets = tn.nominate_from_trace(request, _candidates(tmp_path, [_row("dominant"), _row("marginal")]))
    assert all(t.budget_sec >= tn.MIN_BUDGET_SEC for t in targets)


def test_rows_below_the_floor_are_dropped(tmp_path: Path) -> None:
    """Perfecting a 1%-of-GPU kernel cannot move an end-to-end target."""
    request = _request(tmp_path)
    targets = tn.nominate_from_trace(request, _candidates(tmp_path, [_row("tiny", gpu_pct=0.4)]))
    assert targets == []


def test_rejected_rows_stay_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    targets = tn.nominate_from_trace(request, _candidates(tmp_path, [_row("banned", gpu_pct=99.0, rejected=True)]))
    assert targets == []


def test_unresolved_row_is_rescued_from_a_definition_site(tmp_path: Path) -> None:
    """The rescue the stub cannot do: promote a row that has no source_file."""
    root = tmp_path / "aiter"
    (root / "csrc").mkdir(parents=True)
    (root / "csrc" / "fmoe.hip").write_text(
        "__global__ void fmoe_stage1_kernel(const float* x) { }\n", encoding="utf-8"
    )
    monkey = str(root)
    request = _request(tmp_path)
    candidates = _candidates(
        tmp_path, [_row("fmoe_stage1_kernel", source_file="", reason_class="source_not_resolved", gpu_pct=30.0)]
    )
    import os

    os.environ["KERNELFORGE_SOURCE_ROOTS"] = monkey
    try:
        targets = tn.nominate_from_trace(request, candidates)
    finally:
        del os.environ["KERNELFORGE_SOURCE_ROOTS"]
    assert len(targets) == 1
    assert targets[0].source_file.endswith("csrc/fmoe.hip")
    assert "rescued" in targets[0].reason


def test_a_mere_mention_does_not_resolve(tmp_path: Path) -> None:
    """The bug this guards: substring matching resolved 'blind' to any file."""
    root = tmp_path / "aiter"
    root.mkdir()
    (root / "notes.py").write_text("# we should test the blind case and blind_spot handling\n", encoding="utf-8")
    import os

    os.environ["KERNELFORGE_SOURCE_ROOTS"] = str(root)
    try:
        source_file, reason = tn.resolve_source("blind", (root,))
    finally:
        del os.environ["KERNELFORGE_SOURCE_ROOTS"]
    assert source_file == ""
    assert reason == "source_not_resolved"


def test_templated_mangled_symbol_reduces_to_its_identifier(tmp_path: Path) -> None:
    got = tn._identifier_candidates("void aiter::fmoe_kernel<128, true>(float const*, int)")
    assert got[0] == "fmoe_kernel"
    assert "fmoe" in got or len(got) == 1


def test_seam_defaults_to_stub_and_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping the default would silently change what an existing run does."""
    monkeypatch.delenv("KERNELFORGE_NOMINATOR", raising=False)
    assert nom._selected_nominator() == "stub"
    monkeypatch.setenv("KERNELFORGE_NOMINATOR", "trace")
    assert nom._selected_nominator() == "trace"
    monkeypatch.setenv("KERNELFORGE_NOMINATOR", "typo")
    assert nom._selected_nominator() == "stub"
