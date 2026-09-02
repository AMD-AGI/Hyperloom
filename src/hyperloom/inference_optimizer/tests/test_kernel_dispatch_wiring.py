# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pins how the KERNEL dispatch chain is wired, with every subprocess faked.

The chain from a candidate list to a KEEP/REVERT decision has no integration
test: the orchestrator tests each mock their own neighbour, so nothing asserts
that the pieces still fit together. Moving kernel selection into forge rewires
exactly this chain, so its current shape is pinned first.

Nothing here runs a GPU or a subprocess. What is pinned is the wiring: what the
selector hands the dispatcher, what the dispatcher hands the per-kernel tool, and
what reaches the pending-integration queue. The which-kernel fan-out is gone --
selection moved into forge nomination -- so the surviving path dispatches one
resolved candidate, not one per row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState


def _row(kernel_id: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kernel_id": kernel_id,
        "name": f"{kernel_id}_kernel",
        "source_file": f"/repo/{kernel_id}.py",
        "gpu_pct": 25.0,
        "reusable_native_kernel": True,
    }
    row.update(overrides)
    return row


def _candidates_file(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": rows}), encoding="utf-8")
    return str(path)


def _ok_result(kernel_id: str, *, decision: str = "KEEP", speedup: float = 1.5) -> dict[str, Any]:
    """The shape ``_run_optimization_single`` returns on a successful attempt.

    The decision and the measurement live in nested sections, not at the top
    level; the ledger reads them from there.
    """
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "source_file": f"/repo/{kernel_id}.py",
        "proposal": {"decision": decision, "reasons": []},
        "verification": {
            "micro_speedup": speedup,
            "best_artifact_path": f"/out/{kernel_id}.patch",
            "compile_passed": True,
            "correctness_passed": True,
        },
    }


@pytest.fixture
def one_candidate(tmp_path: Path) -> str:
    return _candidates_file(tmp_path, [_row("k001")])


@pytest.fixture
def three_candidates(tmp_path: Path) -> str:
    return _candidates_file(tmp_path, [_row("k001"), _row("k002"), _row("k003")])


@pytest.fixture
def forge_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-kernel dispatch only exists under this backend order; see below."""
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")


async def test_selector_output_feeds_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch, session_dir: Path, three_candidates: str, forge_backend: None
) -> None:
    """The selector resolves the candidate set and the dispatcher runs one kernel.

    The which-kernel fan-out is gone: selection now lives in forge nomination, so
    the surviving single-kernel path dispatches exactly one resolved candidate --
    not one per row -- and that one must be a real member of the selected set.
    """
    seen: list[str] = []

    async def _fake_single(payload: dict, **kwargs: Any) -> dict[str, Any]:
        kernel_id = str(payload.get("kernel_id") or "")
        seen.append(kernel_id)
        return _ok_result(kernel_id)

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    await krh.run_optimization_handler({"candidates_path": three_candidates}, session_dir=session_dir)
    assert len(seen) == 1
    assert seen[0] in {"k001", "k002", "k003"}


async def test_a_row_without_source_file_never_reaches_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch, session_dir: Path, tmp_path: Path, forge_backend: None
) -> None:
    """The silent drop happens before dispatch, not inside the per-kernel tool."""
    seen: list[str] = []

    async def _fake_single(payload: dict, **kwargs: Any) -> dict[str, Any]:
        kernel_id = str(payload.get("kernel_id") or "")
        seen.append(kernel_id)
        return _ok_result(kernel_id)

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    candidates = _candidates_file(tmp_path, [_row("k001", source_file=""), _row("k002")])
    await krh.run_optimization_handler({"candidates_path": candidates}, session_dir=session_dir)
    assert seen == ["k002"]


async def test_empty_selection_skips_cleanly(
    monkeypatch: pytest.MonkeyPatch, session_dir: Path, tmp_path: Path, forge_backend: None
) -> None:
    """Nothing eligible must not look like a dispatch failure."""

    async def _unreachable(payload: dict, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("dispatcher must not be called with an empty selection")

    monkeypatch.setattr(krh, "_run_optimization_single", _unreachable)
    candidates = _candidates_file(tmp_path, [_row("k001", source_file="")])
    result = await krh.run_optimization_handler({"candidates_path": candidates}, session_dir=session_dir)
    assert str(result.get("status") or "") in {"skipped", "ok"}


async def test_a_keep_reaches_the_pending_integration_queue(
    monkeypatch: pytest.MonkeyPatch, session_dir: Path, one_candidate: str, forge_backend: None
) -> None:
    """A KEEP must land in the queue integrate later drains, keyed per patch."""

    async def _fake_single(payload: dict, **kwargs: Any) -> dict[str, Any]:
        return _ok_result(str(payload.get("kernel_id") or ""))

    monkeypatch.setattr(krh, "_run_optimization_single", _fake_single)
    result = await krh.run_optimization_handler({"candidates_path": one_candidate}, session_dir=session_dir)

    state = SharedState.load_or_init(session_dir)
    state.record_kernel_opt(result)
    state.save(session_dir)

    queued = SharedState.load_or_init(session_dir).pending_kernel_integrations
    assert isinstance(queued, dict)
    assert any(str(entry.get("kernel_id") or "") == "k001" for entry in queued.values())
