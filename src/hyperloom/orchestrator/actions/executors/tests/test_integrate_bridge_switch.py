# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A bridge patch reaches the benchmark only behind a default-off switch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._integrate_refusals import _unswitched_bridge_refusal
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.measurement.runtime_findings import RESTORED_PATH_RULES, restores_disabled_path
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, _section_output_protocol
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task

PATCH = [Path("001_gate.patch")]
SWITCH = [{"switch": "HL_GLM5_TP1_SHARED_FUSION", "value": "1"}]
GATE_FINDING = "feature_disabled:HIP shared experts fusion"


def _refusal(*, finding: str = "", gap: str = "", patches=PATCH, switches=()) -> str:
    done = {"resolves_finding": finding, "gap_canonical_id": gap}
    return _unswitched_bridge_refusal({}, done, patch_paths=list(patches), switch_manifest=list(switches))


def test_restored_path_rules_are_the_perf_path_fix_rules():
    assert RESTORED_PATH_RULES == {
        "feature_disabled",
        "capability_disabled",
        "comm.custom_ar_disabled",
        "comm.multimem_allgather_disabled",
    }
    assert restores_disabled_path(GATE_FINDING)
    assert not restores_disabled_path("vllm.unknown_env:VLLM_FOO")
    assert not restores_disabled_path("")


@pytest.mark.parametrize(
    ("kwargs", "refusal"),
    [
        pytest.param({"finding": GATE_FINDING}, f"it restores {GATE_FINDING}", id="declared-restore"),
        pytest.param(
            {"gap": "gap.static_recon.rocm.glm5_next.shared_expert_fusion_gate"},
            "it bridges gap.static_recon.rocm.glm5_next.shared_expert_fusion_gate",
            id="static-recon-gap",
        ),
        pytest.param({"finding": GATE_FINDING, "switches": SWITCH}, "", id="switched"),
        pytest.param({"finding": GATE_FINDING, "patches": ()}, "", id="config-only"),
        pytest.param({"finding": "vllm.unknown_env:VLLM_FOO"}, "", id="correctness-fix"),
        pytest.param({"gap": "gap.framework.local_explore.c1"}, "", id="other-gap"),
    ],
)
def test_only_an_unswitched_bridge_patch_is_refused(kwargs, refusal):
    assert _refusal(**kwargs) == refusal


def test_an_enablement_round_is_never_refused_as_a_bridge():
    refusal = _unswitched_bridge_refusal(
        {"enablement": True}, {"resolves_finding": GATE_FINDING}, patch_paths=list(PATCH), switch_manifest=[]
    )

    assert refusal == ""


def test_the_task_params_gap_marks_a_bridge_too():
    refusal = _unswitched_bridge_refusal(
        {"gap_canonical_id": "gap.static_recon.g1"}, {}, patch_paths=list(PATCH), switch_manifest=[]
    )

    assert refusal == "it bridges gap.static_recon.g1"


@pytest.mark.asyncio
async def test_executor_refuses_an_unswitched_bridge_before_benching(tmp_path, monkeypatch):
    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec-bridge"
    workspace.mkdir(parents=True)
    (workspace / "001_gate.patch").write_text("--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-a\n+b\n", encoding="utf-8")
    (workspace / "specialist_done.json").write_text(
        json.dumps({"patches_written": ["001_gate.patch"], "resolves_finding": GATE_FINDING}), encoding="utf-8"
    )

    async def _should_not_benchmark(*_args, **_kwargs):
        raise AssertionError("an un-switched bridge reached the benchmark")

    executor = IntegratePatchExecutor(session_dir=session_dir)
    monkeypatch.setattr(executor, "_bench_patch", _should_not_benchmark)
    state = SharedState(gpu_type="mi355x")
    state.record_specialist_patch_verdict("t-spec-bridge", "approve")
    task = Task(
        task_id="t-int-bridge",
        kind="integrate_patch",
        state="queued",
        params={"specialist_task_id": "t-spec-bridge"},
        idempotency_key="t-int-bridge",
        requires_lanes=tuple(),
    )
    result = await executor(RunnerContext(task=task, lease=None, extra={"shared_state": state}))

    assert (result["status"], result["error_class"]) == ("apply_failed", "bridge_not_switch_gated")
    assert result["error"] == f"bridge patch declares no framework_switches although it restores {GATE_FINDING}"
    assert "defaults OFF" in result["advisory"]
    assert result["prior_patches"] == [str(workspace / "001_gate.patch")]


GATE_FINDINGS = f"runtime findings for server.log [sglang]\n- detected [perf_path] {GATE_FINDING} x1: Disabling it"


@pytest.mark.parametrize(
    ("overrides", "told"),
    [
        pytest.param({"gap_canonical_id": "gap.static_recon.g1"}, True, id="static-recon-gap"),
        pytest.param({"runtime_findings": GATE_FINDINGS}, True, id="disabled-path-finding"),
        pytest.param({"runtime_findings": GATE_FINDINGS.replace(":", " ", 1)}, True, id="space-separated-id"),
        pytest.param(
            {"runtime_findings": "- detected [correctness] vllm.unknown_env VLLM_FOO x1: unknown"}, False, id="other"
        ),
        pytest.param({"gap_canonical_id": "gap.framework.c1"}, False, id="no-bridge"),
        pytest.param({"gap_canonical_id": "gap.static_recon.g1", "mode": "research"}, False, id="research"),
    ],
)
def test_patch_authors_are_told_to_switch_gate_a_bridge_only_where_one_is_likely(overrides, told):
    fields = {"domain": get_domain("serving_specialist"), "mode": "patch"} | overrides
    text = "\n".join(_section_output_protocol(SpecialistPromptInputs(**fields)))

    restored = "``" + "`` / ``".join(sorted(RESTORED_PATH_RULES)) + "`` ``resolves_finding``"
    assert ("MUST put the relaxed" in text) is told
    assert (restored in text) is told
