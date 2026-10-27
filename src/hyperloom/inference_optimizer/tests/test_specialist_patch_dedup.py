# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist patch that repeats one already queued or already in the tree is not measured again."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.orchestrator.framework import patch_overlap as po
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
from hyperloom.orchestrator.state.task_registry import Task

# Two wordings of one fix, as two freeform specialists wrote it for the GLM-5.3 NextN draft layer.
_FIX_A = """\
--- a/python/sglang/srt/models/glm4_moe.py
+++ b/python/sglang/srt/models/glm4_moe.py
@@ -1460,6 +1460,7 @@ class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
 class GlmMoeDsaForCausalLMNextN(GlmMoeDsaForCausalLM):
+    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"
     def __init__(self, config, quant_config=None, prefix=""):
"""
_FIX_B = """\
--- a/sglang/srt/models/glm4_moe.py
+++ b/sglang/srt/models/glm4_moe.py
@@ -1458,7 +1458,8 @@
 class GlmMoeDsaForCausalLMNextN(GlmMoeDsaForCausalLM):
+    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"
     # draft layer
"""
_OTHER = """\
--- a/python/sglang/srt/layers/attention/dsa/indexer.py
+++ b/python/sglang/srt/layers/attention/dsa/indexer.py
@@ -10,3 +10,4 @@
 def topk(rows):
+    rows = rows.contiguous()
"""
_FREEFORM = {"scope": "freeform", "mode": "patch", "task_description": "fuse the NextN shared expert"}


def _coordinator(session_dir: Path) -> Coordinator:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "observation", "body_md": "ok"}),
    )
    coord = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="orch"), "critic": MockCriticBackend()},
    )
    coord.shared_state.phase = "FRAMEWORK_AGENT"
    coord.shared_state.baseline_tput = 1500.0
    return coord


async def _specialist_wrote(coord: Coordinator, sid: str, diff: str, *, params: dict | None = None) -> None:
    """Complete a specialist whose deliverable names its patch relative to its worktree."""
    worktree = runs_dir(coord.session_dir, "specialist", sid) / "worktree"
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "fix.patch").write_text(diff, encoding="utf-8")
    await coord.phase_framework.maybe_autosubmit_specialist_patches(
        task=Task(
            task_id=sid, kind="specialist", state="running", params=dict(params or _FREEFORM), idempotency_key=sid
        ),
        done_payload={"patches_written": ["fix.patch"], "proposal_set": [{"name": f"name-{sid}"}]},
    )


async def _integrate_specialists(coord: Coordinator) -> list[str]:
    return [
        (t.params or {}).get("specialist_task_id") for t in await coord.tasks.queued() if t.kind == "integrate_patch"
    ]


@pytest.mark.asyncio
async def test_a_relative_patch_path_is_fingerprinted_on_the_integrate_task(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        await _specialist_wrote(coord, "s1", _FIX_A)
        await coord.reactor_pass("critic")

        (task,) = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
        identity = (task.params or {}).get("patch_identity") or {}
        assert identity.get("line_keys") == po.changed_line_keys([_FIX_A])
        assert identity.get("files") == ["python/sglang/srt/models/glm4_moe.py"]
    finally:
        await coord.stop()


@pytest.mark.parametrize(
    ("switch", "mode", "params", "screen"),
    [
        ("1", "agentx", _FREEFORM, "cycle"),
        ("", "agentx", _FREEFORM, None),
        ("1", "", _FREEFORM, None),
        ("1", "agentx", {**_FREEFORM, "enablement": True}, None),
    ],
)
@pytest.mark.asyncio
async def test_an_agentx_optimizing_patch_is_marked_for_the_cycle_screen_only_when_it_is_switched_on(
    session_dir: Path, monkeypatch, switch: str, mode: str, params: dict, screen: str | None
) -> None:
    """Enablement climbs to a booting model, so its patch is measured as before rather than admitted dormant."""
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", switch)
    coord = _coordinator(session_dir)
    coord.shared_state.benchmark_mode = mode
    try:
        await _specialist_wrote(coord, "s1", _FIX_A, params=params)
        await coord.reactor_pass("critic")

        (task,) = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
        assert (task.params or {}).get("screen") == screen
    finally:
        await coord.stop()


_TOKENIZER_FIX = """\
--- a/python/sglang/srt/managers/tokenizer_manager.py
+++ b/python/sglang/srt/managers/tokenizer_manager.py
@@ -400,3 +400,4 @@
 def _tokenize_one_request(self, obj):
+    cached = self._segment_cache.get(obj.text)
"""


@pytest.mark.parametrize(("diff", "screen"), [(_TOKENIZER_FIX, None), (_FIX_A, "cycle")])
@pytest.mark.asyncio
async def test_a_patch_the_decode_cycle_cannot_see_goes_to_its_own_replay_round(
    session_dir: Path, monkeypatch, diff: str, screen: str | None
) -> None:
    """Tokenizer, API and KV-cache paths stay out of steady decode, so the cycle screen would read them flat."""
    monkeypatch.setenv("HYPERLOOM_CYCLE_BENCH", "1")
    coord = _coordinator(session_dir)
    coord.shared_state.benchmark_mode = "agentx"
    try:
        await _specialist_wrote(coord, "s1", diff)
        await coord.reactor_pass("critic")

        (task,) = [t for t in await coord.tasks.queued() if t.kind == "integrate_patch"]
        assert (task.params or {}).get("screen") == screen
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_a_repeat_of_a_queued_patch_is_not_submitted(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        await _specialist_wrote(coord, "s1", _FIX_A)
        await coord.reactor_pass("critic")
        await _specialist_wrote(coord, "s2", _FIX_B)
        await _specialist_wrote(coord, "s3", _OTHER)
        await coord.reactor_pass("critic")

        assert await _integrate_specialists(coord) == ["s1", "s3"]
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_a_repeat_of_a_pending_review_is_not_submitted(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        await _specialist_wrote(coord, "s1", _FIX_A)
        await _specialist_wrote(coord, "s2", _FIX_B)
        await coord.reactor_pass("critic")

        assert await _integrate_specialists(coord) == ["s1"]
    finally:
        await coord.stop()


def _settled(params: dict, status: str) -> tuple[Task, SimpleNamespace]:
    task = Task(task_id="i1", kind="integrate_patch", state="done", params=params, idempotency_key="i1")
    return task, SimpleNamespace(result={"status": status})


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "landed"), [("kept_inert", True), ("kept", True), ("reverted", False)])
async def test_a_settled_round_records_its_patch_and_whether_it_stayed(
    session_dir: Path, status: str, landed: bool
) -> None:
    coord = _coordinator(session_dir)
    try:
        identity = {"files": ["f.py"], "line_keys": po.changed_line_keys([_FIX_A])}
        task, result = _settled({"patch_name": "nextn", "patch_identity": identity}, status)

        coord.phase_framework._record_measured_patch_set(task, result, Verdict.REVERTED)

        rows = [{k: v for k, v in row.items() if k != "ts"} for row in coord.shared_state.measured_patch_sets]
        assert rows == [
            {
                "task_id": "i1",
                "patch_name": "nextn",
                "files": ["f.py"],
                "line_keys": identity["line_keys"],
                "outcome": "reverted",
                "landed": landed,
            }
        ]
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_only_a_landed_repeat_is_refused_a_reverted_one_is_reviewed(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        keys = po.changed_line_keys([_FIX_A])
        coord.shared_state.record_measured_patch_set(
            task_id="old-landed",
            patch_name="a",
            identity={"files": ["f"], "line_keys": keys},
            outcome="reverted",
            landed=True,
        )
        await _specialist_wrote(coord, "s1", _FIX_B)
        coord.shared_state.measured_patch_sets.clear()
        coord.shared_state.record_measured_patch_set(
            task_id="old-reverted",
            patch_name="a",
            identity={"files": ["f"], "line_keys": keys},
            outcome="reverted",
            landed=False,
        )
        await _specialist_wrote(coord, "s2", _FIX_B)
        await coord.reactor_pass("critic")

        assert await _integrate_specialists(coord) == ["s2"]
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_an_authoring_round_repeat_still_reaches_review(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        await _specialist_wrote(coord, "s1", _FIX_A)
        await coord.reactor_pass("critic")
        authoring = {**_FREEFORM, "framework_agent_authoring": True}
        await _specialist_wrote(coord, "s2", _FIX_B, params=authoring)
        await coord.reactor_pass("critic")

        assert await _integrate_specialists(coord) == ["s1", "s2"]
    finally:
        await coord.stop()


@pytest.mark.asyncio
async def test_an_edit_of_a_landed_line_is_reviewed_not_refused(session_dir: Path) -> None:
    coord = _coordinator(session_dir)
    try:
        landed = po.changed_line_keys([_FIX_A])
        coord.shared_state.record_measured_patch_set(
            task_id="old-landed",
            patch_name="a",
            identity={"files": ["f"], "line_keys": landed},
            outcome="reverted",
            landed=True,
        )
        edit = _FIX_A.replace(
            '+    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"',
            '-    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"\n'
            '+    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN_v2"',
        )
        await _specialist_wrote(coord, "s1", edit)
        await coord.reactor_pass("critic")

        assert await _integrate_specialists(coord) == ["s1"]
    finally:
        await coord.stop()
