# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every outcome an integration files under an upstream PR reaches the framework KB."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.agents.framework.kb import KB_ROOT_ENV, read_pr_ledger
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors._integrate_results import _PatchMeasurement
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.knowledge.kb_writeback import OUTCOME_KEPT_INERT
from hyperloom.orchestrator.state.shared_state import SharedState

PR_URL = "https://github.com/sgl-project/sglang/pull/1"


@pytest.mark.asyncio
async def test_a_dormant_keep_files_its_lesson(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(KB_ROOT_ENV, str(tmp_path / "kb"))
    attempt = IntegrateAttempt(
        task_id="t-int",
        specialist_task_id="t-spec",
        shared_state=SharedState(baseline_tput=1000.0),
        done_payload={"proposal_set": [{"provenance": "specialist:serving:framework", "fa_pr_url": PR_URL}]},
        output_root=tmp_path / "run",
        applied=[tmp_path / "001.patch"],
        switch_manifest=[{"switch": "SGLANG_FOO", "value": "1"}],
    )
    measured = _PatchMeasurement(
        bench_result={},
        new_tput=990.0,
        delta_pct=-1.0,
        base_tput=1000.0,
        keep_threshold_pct=1.0,
        accuracy_pass=True,
        acc_delta_pct=None,
        cfg_fingerprint="",
        measured_against={},
    )

    out = await IntegratePatchExecutor(session_dir=tmp_path / "session")._keep_inert_switches(
        attempt, measured, params={}, parity={"ran": True, "ok": True}
    )

    assert out["status"] == "kept_inert"
    assert [(row["pr_url"], row["outcome"], row["tps_delta_pct"]) for row in read_pr_ledger()] == [
        (PR_URL, OUTCOME_KEPT_INERT, -1.0)
    ]
