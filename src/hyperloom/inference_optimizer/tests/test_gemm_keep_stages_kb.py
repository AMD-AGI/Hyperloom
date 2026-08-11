# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression: a GEMM-tuning KEEP must stage the kernel KB ``gemm`` column.

The per-round staging hook fires from ``record_gemm_tuning`` *before* the
promote/validate step appends the ``gemm_tuning`` row to
``optimization_stack``. Because ``build_gemm`` keys off that stack row, staging
at record time sees no row and writes nothing. ``_handle_gemm_tuning_result``
must therefore re-stage once more after promotion so the accepted GEMM lands in
the draft (and, in turn, is published to the KB at CLOSE).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


def _coord(tmp_path: Path, **state_kwargs) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(**state_kwargs)
    return coord


def _kernel_section(draft_dir: Path) -> dict:
    path = draft_dir / "sections" / "kernel.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_geak_gemm_keep_stages_gemm_column(tmp_path, monkeypatch):
    draft_dir = tmp_path / "kb_draft"
    draft_dir.mkdir()
    monkeypatch.setenv("KB_DRAFT_DIR", str(draft_dir))
    monkeypatch.delenv("KB_WARM_START_DIR", raising=False)

    tuned = tmp_path / "tuned_gemm.csv"
    tuned.write_text("M,N,K,kernelId\n16,512,7168,3\n", encoding="utf-8")

    coord = _coord(tmp_path, baseline_tput=100.0)

    await coord._handle_gemm_tuning_result(
        {
            "status": "ok",
            "decision": "KEEP",
            "best_speedup": 1.4,
            "backend": "geak",
            "tuned_file": str(tuned),
        }
    )

    # Promotion recorded the accepted GEMM on the stack ...
    assert len(coord.shared_state.optimization_stack) == 1
    assert coord.shared_state.optimization_stack[0]["action"] == "gemm_tuning"

    # ... and the kernel KB draft holds the matching gemm sub-column.
    section = _kernel_section(draft_dir)
    gemm = section.get("knowledge", section).get("gemm", {})
    assert gemm, "gemm column was not staged into the KB draft after a KEEP"
    opts = gemm.get("optimizations") or []
    assert len(opts) == 1
    # The tuned artifact was staged as a managed file ref, not a host path.
    assert opts[0].get("tuned_file", "").startswith("kernel/gemm/")
