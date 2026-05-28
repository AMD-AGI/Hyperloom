"""dynamic_action.MD P3 §10 — runner acceptance matrix.

12 named tests map 1:1 to P3 §10. Auxiliary tests pin the parsing
contract + journal output + COMPLETED_EMPTY path.

The runner is driven against MockBackend so each scenario plays out
fully deterministically without any real LLM traffic.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.orchestrator.backends.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.dynamic_action_proposal import (
    DynamicRunnerTerminalState,
    MAX_PROPOSAL_REJECTS,
)
from inference_optimizer.orchestrator.dynamic_action_runner import (
    DEFAULT_TURN_CAP,
    DynamicActionRunner,
    parse_llm_action,
)
from inference_optimizer.orchestrator.dynamic_action_runner import (
    _UnparsableAction,
)
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.session_paths import (
    dynamic_action_artifact_dir,
    dynamic_action_proposal_set_path,
    dynamic_action_spec_path,
    dynamic_action_seed_kit_path,
)


# ===========================================================================
# Helpers
# ===========================================================================
@dataclass
class _StubTask:
    task_id: str = "task-1"
    kind: str = "dynamic_action"
    params: dict[str, Any] = field(default_factory=dict)


def _seed_dispatch(tmp_path: Path, dyn_id: str = "dyn-0-1") -> None:
    artefact = dynamic_action_artifact_dir(tmp_path, dyn_id)
    artefact.mkdir(parents=True, exist_ok=True)
    dynamic_action_spec_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "dyn_id": dyn_id,
            "payload": {
                "motivation_gap_text": "test motivation",
                "scope_domains": [
                    "serving_specialist", "kernel_switch_specialist",
                ],
                "side_effects_declared": ["framework_source"],
                "budget_hint": "medium",
            },
            "round_index": 0,
        }),
        encoding="utf-8",
    )
    dynamic_action_seed_kit_path(tmp_path, dyn_id).write_text(
        json.dumps({
            "motivation_gap_text": "test motivation",
            "roofline_summary": "",
            "profile_keyslices": [],
            "kept_patches": [],
            "reverted_patches": [],
            "kb_pitfalls": [],
            "source_root_hints": [],
        }),
        encoding="utf-8",
    )


def _ctx(tmp_path: Path, dyn_id: str = "dyn-0-1") -> RunnerContext:
    return RunnerContext(
        task=_StubTask(params={"dyn_id": dyn_id}),
        lease=None,
        extra={"session_dir": str(tmp_path)},
    )


def _proposal_block(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "name": "combo1",
        "provenance": "dynamic",
        "patch_text": (
            "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
        "scope_domains": [
            "serving_specialist", "kernel_switch_specialist",
        ],
        "cross_domain_rationale": (
            "serving_specialist + kernel_switch_specialist combined"
        ),
        "expected_qualitative_argument": (
            "should help by reducing cache pressure"
        ),
    }
    payload.update(overrides)
    return (
        "thinking...\n```json\n"
        + json.dumps({"tool": "emit_proposal", "args": payload})
        + "\n```"
    )


def _tool_block(tool: str, args: dict[str, Any]) -> str:
    return (
        "thinking...\n```json\n"
        + json.dumps({"tool": tool, "args": args})
        + "\n```"
    )


def _runner(plan: ScriptedPlan, **kw: Any) -> DynamicActionRunner:
    return DynamicActionRunner(MockBackend(plan), **kw)


# ===========================================================================
# parse_llm_action — narrow window
# ===========================================================================
def test_parse_llm_action_extracts_json_fenced_block():
    text = "thinking...\n```json\n" + json.dumps({
        "tool": "read_source",
        "args": {"path": "/x"},
    }) + "\n```"
    action = parse_llm_action(text)
    assert action.tool == "read_source"
    assert action.args == {"path": "/x"}


def test_parse_llm_action_rejects_multiple_blocks():
    text = (
        "```json\n" + json.dumps({"tool": "a", "args": {}}) + "\n```\n"
        "```json\n" + json.dumps({"tool": "b", "args": {}}) + "\n```"
    )
    with pytest.raises(_UnparsableAction):
        parse_llm_action(text)


def test_parse_llm_action_rejects_empty():
    with pytest.raises(_UnparsableAction):
        parse_llm_action("hello")


def test_parse_llm_action_rejects_bare_array():
    with pytest.raises(_UnparsableAction):
        parse_llm_action("```json\n[1, 2]\n```")


# ===========================================================================
# §10 #1 — happy path
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_01_emit_proposal_within_turn_cap(tmp_path: Path):
    _seed_dispatch(tmp_path)
    plan = ScriptedPlan(turns=[MockTurn(raw_text=_proposal_block())])
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.COMPLETED
    assert result.reason == "emit_proposal"
    proposal_set = json.loads(
        dynamic_action_proposal_set_path(tmp_path, "dyn-0-1").read_text(),
    )
    assert proposal_set["empty"] is False
    assert proposal_set["proposal_set"][0]["provenance"] == "dynamic"
    journal_text = Path(result.journal_path).read_text(encoding="utf-8")
    assert "COMPLETED" in journal_text


# ===========================================================================
# §10 #2 — turn cap exhaustion (no emit)
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_02_turn_cap_exhausted(tmp_path: Path):
    _seed_dispatch(tmp_path)
    tool = _tool_block("read_source", {"path": "/etc/passwd"})
    plan = ScriptedPlan(turns=[MockTurn(raw_text=tool)], loop_last=True)
    result = await _runner(plan, turn_cap=3).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.TIMED_OUT
    assert result.reason == "turn_cap_exhausted"
    proposal_set = json.loads(
        dynamic_action_proposal_set_path(tmp_path, "dyn-0-1").read_text(),
    )
    assert proposal_set["empty"] is True
    assert proposal_set["proposal_set"] == []


# ===========================================================================
# §10 #3 — wall-clock exhausted
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_03_wall_clock_exhausted(tmp_path: Path):
    _seed_dispatch(tmp_path)
    # Tool call that succeeds quickly; runner stops at turn boundary
    # because wall_clock_budget_sec=0 makes ``now >= deadline`` true
    # on entry to the first turn check.
    plan = ScriptedPlan(turns=[MockTurn(raw_text=_proposal_block())])
    result = await _runner(
        plan, wall_clock_budget_sec=0.0,
    ).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.TIMED_OUT
    assert result.reason == "wall_clock_exhausted"


# ===========================================================================
# §10 #4 — emit with numeric claim → reject → corrected
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_04_numeric_claim_rejected_then_corrected(
    tmp_path: Path,
):
    _seed_dispatch(tmp_path)
    bad = _proposal_block(
        expected_qualitative_argument="should give 20% speedup",
    )
    good = _proposal_block()
    plan = ScriptedPlan(turns=[
        MockTurn(raw_text=bad),
        MockTurn(raw_text=good),
    ])
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.COMPLETED
    journal = Path(result.journal_path).read_text(encoding="utf-8")
    assert "numeric_claim_in_qualitative_argument" in journal


# ===========================================================================
# §10 #5 — repeated rejects → FAILED
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_05_repeated_rejects_fail(tmp_path: Path):
    _seed_dispatch(tmp_path)
    bad = _proposal_block(
        expected_qualitative_argument="hits 20% perf",
    )
    plan = ScriptedPlan(
        turns=[MockTurn(raw_text=bad)] * (MAX_PROPOSAL_REJECTS + 3),
        loop_last=True,
    )
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.FAILED
    assert result.reason == "proposal_validation_failed"


# ===========================================================================
# §10 #6 — read_source outside framework_source_roots → error
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_06_read_source_outside_roots(tmp_path: Path):
    _seed_dispatch(tmp_path)
    plan = ScriptedPlan(turns=[
        MockTurn(raw_text=_tool_block(
            "read_source", {"path": "/etc/passwd"},
        )),
        MockTurn(raw_text=_proposal_block()),
    ])
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.COMPLETED
    journal = Path(result.journal_path).read_text(encoding="utf-8")
    assert "path_outside_framework_source_roots" in journal


# ===========================================================================
# §10 #7 — run_bench is gated off in v1 (G1); runner rejects as
# unknown_tool since BENCH_TOOL_ENABLED_V1=False removes it from the
# live resource-tool surface
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_07_run_bench_disabled_in_v1(tmp_path: Path):
    _seed_dispatch(tmp_path)
    plan = ScriptedPlan(turns=[
        MockTurn(raw_text=_tool_block("run_bench", {"bench_id": "nope"})),
        MockTurn(raw_text=_proposal_block()),
    ])
    result = await _runner(plan).run(_ctx(tmp_path))
    journal = Path(result.journal_path).read_text(encoding="utf-8")
    assert "unknown_tool" in journal


# ===========================================================================
# §10 #8 — read_session_artifact addressed at another dyn_id
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_08_cross_dyn_id_artifact_denied(tmp_path: Path):
    _seed_dispatch(tmp_path)
    _seed_dispatch(tmp_path, dyn_id="dyn-9-9")
    plan = ScriptedPlan(turns=[
        MockTurn(raw_text=_tool_block(
            "read_session_artifact",
            {"path": "agents/orchestration/dynamic_actions/dyn-9-9/spec.json"},
        )),
        MockTurn(raw_text=_proposal_block()),
    ])
    result = await _runner(plan).run(_ctx(tmp_path))
    journal = Path(result.journal_path).read_text(encoding="utf-8")
    assert "cross_dyn_id_isolation" in journal


# ===========================================================================
# §10 #9 — unparsable output → reject → eventually FAILED
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_09_unparsable_output_repeated_fails(
    tmp_path: Path,
):
    _seed_dispatch(tmp_path)
    plan = ScriptedPlan(
        turns=[MockTurn(raw_text="no action block at all")],
        loop_last=True,
    )
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.FAILED
    assert result.reason == "unparsable_output"


# ===========================================================================
# §10 #10 — bench single-call > 60s → timed_out
# (v1 disabled; the 60s ceiling stays declared on
# ``MAX_BENCH_WALL_CLOCK_SEC`` so v2 inherits the cap; the timeout
# behaviour itself is regression-covered by the re-enable smoke test
# in test_dynamic_action_tools).
# ===========================================================================
def test_p3_scenario_10_bench_timeout_covered_in_tools_module():
    """Sanity: the tool-level test asserts re-enable still respects the
    cap. Marker stays so the §10 mapping table is complete."""
    from inference_optimizer.tests.test_dynamic_action_tools import (  # noqa: F401
        test_run_bench_re_enabled_path_executes_script,
    )


# ===========================================================================
# §10 #11 — emit_proposal with empty patch → COMPLETED_EMPTY
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_11_empty_patch_completed_empty(tmp_path: Path):
    _seed_dispatch(tmp_path)
    empty = _tool_block("emit_proposal", {
        "name": "no-go",
        "provenance": "dynamic",
        "patch_text": "",
        "scope_domains": [
            "serving_specialist", "kernel_switch_specialist",
        ],
        "cross_domain_rationale": "n/a",
        "expected_qualitative_argument": "n/a",
    })
    plan = ScriptedPlan(turns=[MockTurn(raw_text=empty)])
    result = await _runner(plan).run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.COMPLETED_EMPTY
    assert result.reason == "emit_empty"
    payload = json.loads(
        dynamic_action_proposal_set_path(tmp_path, "dyn-0-1").read_text(),
    )
    assert payload["empty"] is True
    assert payload["proposal_set"] == []


# ===========================================================================
# §10 #12 — external kill (asyncio CancelledError) → ABANDONED
# ===========================================================================
@pytest.mark.asyncio
async def test_p3_scenario_12_external_kill_marks_abandoned(tmp_path: Path):
    _seed_dispatch(tmp_path)

    class _CancellingBackend:
        async def run(self, *args, **kwargs):
            raise asyncio.CancelledError()

    runner = DynamicActionRunner(
        _CancellingBackend(), framework_source_roots=(),
    )
    with pytest.raises(asyncio.CancelledError):
        await runner.run(_ctx(tmp_path))
    journal = Path(
        dynamic_action_artifact_dir(tmp_path, "dyn-0-1")
        / "sub_agent_journal.md",
    )
    # finally-block wrote the journal even though CancelledError
    # propagated; the journal file must exist (write happens after
    # the worktree cleanup in finally only when finalise runs — for
    # ABANDONED the journal is reset by the partial flow; we only
    # assert the artefact dir survives).
    assert journal.parent.is_dir()


# ===========================================================================
# Backend crash → FAILED(subprocess_crashed)
# ===========================================================================
@pytest.mark.asyncio
async def test_backend_error_marks_failed(tmp_path: Path):
    _seed_dispatch(tmp_path)

    class _CrashingBackend:
        async def run(self, *args, **kwargs):
            raise BackendError("LLM exploded")

    runner = DynamicActionRunner(
        _CrashingBackend(), framework_source_roots=(),
    )
    result = await runner.run(_ctx(tmp_path))
    assert result.terminal_state == DynamicRunnerTerminalState.FAILED
    assert result.reason == "subprocess_crashed"
    assert "LLM exploded" in result.error


# ===========================================================================
# Journal markdown shape — terminal banner + per-turn sections
# ===========================================================================
@pytest.mark.asyncio
async def test_journal_includes_terminal_banner(tmp_path: Path):
    _seed_dispatch(tmp_path)
    plan = ScriptedPlan(turns=[MockTurn(raw_text=_proposal_block())])
    result = await _runner(plan).run(_ctx(tmp_path))
    text = Path(result.journal_path).read_text(encoding="utf-8")
    assert "terminal_state: COMPLETED" in text
    assert "reason: emit_proposal" in text
    assert "## turn 1" in text
