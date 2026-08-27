# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Discovery must not turn its own failures into "no fusion opportunity".

Two of them: a response the gateway cut off mid-proposal, and a turn budget so
small that using the tools discovery was given always ends the session.
"""

from __future__ import annotations

import pytest
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentRunResult,
    AgentRuntimeConfig,
)

from kernelforge.fusion import discover as discover_module
from kernelforge.fusion.discover import _extract_json_array

TRUNCATED_MID_RATIONALE = """```json
[
  {
    "name": "rope_cache_write",
    "env_flag": "LLAMA_FUSED_ROPE_KVCACHE",
    "op_chain": "qkv.split -> rotary_emb(positions, q, k) -> reshape_and_cache",
    "rationale": "The trace shows rope followed by a separate cache write, which"""

TRUNCATED_IN_FIRST_FIELD = """```json
[
  {
    "name": "moe_prologue_addrms_rou"""

COMPLETE_THEN_TRUNCATED = """```json
[
  {"name": "first", "op_chain": "a -> b"},
  {
    "name": "second",
    "op_chain": "c -> d",
    "rationale": "cut here"""


def test_recovers_proposal_cut_mid_field() -> None:
    got = _extract_json_array(TRUNCATED_MID_RATIONALE)

    assert [item["name"] for item in got] == ["rope_cache_write"]
    assert got[0]["op_chain"].startswith("qkv.split")


def test_drops_proposal_cut_before_it_describes_anything() -> None:
    """A name on its own would send the author stage after nothing."""
    assert _extract_json_array(TRUNCATED_IN_FIRST_FIELD) == []


def test_keeps_complete_proposals_and_adds_the_repaired_one() -> None:
    got = _extract_json_array(COMPLETE_THEN_TRUNCATED)

    assert [item["name"] for item in got] == ["first", "second"]


def test_closed_array_is_returned_as_is() -> None:
    text = '```json\n[{"name": "a", "op_chain": "x -> y"}]\n```'

    assert _extract_json_array(text) == [{"name": "a", "op_chain": "x -> y"}]


def test_empty_text_yields_nothing() -> None:
    assert _extract_json_array("") == []


def _backend(captured: dict):
    class Backend:
        name = "codex"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(provider="codex", model="gpt-test", sandbox_mode="bypass")

        async def run(self, spec, usage=None):
            captured["spec"] = spec
            return AgentRunResult(text="[]")

    return Backend()


def test_turn_budget_leaves_room_for_the_tools_discovery_is_given(tmp_path) -> None:
    """One turn plus a read tool is a guaranteed turn_cap, not a budget."""
    captured: dict = {}
    fn = discover_module.registered_agent_llm_fn(
        _backend(captured), model="gpt-test", timeout_s=10, workdir=str(tmp_path)
    )

    fn("DISCOVERY PROMPT")

    policy = captured["spec"].tool_policy
    assert policy.read is True and policy.search is True
    assert policy.max_turns > 1


def test_turn_budget_is_configurable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FUSION_DISCOVERY_TURNS", "5")
    captured: dict = {}
    fn = discover_module.registered_agent_llm_fn(
        _backend(captured), model="gpt-test", timeout_s=10, workdir=str(tmp_path)
    )

    fn("DISCOVERY PROMPT")

    assert captured["spec"].tool_policy.max_turns == 5


def _backend_returning(results: list, calls: list):
    """A backend replaying ``results``, one per call."""

    class Backend:
        name = "claude"
        capabilities = AgentCapabilities(sandbox=True, requires_workspace_cwd=True)
        runtime = AgentRuntimeConfig(provider="claude", model="m", sandbox_mode="bypass")

        async def run(self, spec, usage=None):
            if spec.progress_log is not None:
                spec.progress_log.append("tool: Read qwen3.py")
            calls.append(spec)
            return results[min(len(calls) - 1, len(results) - 1)]

    return Backend()


PROPOSALS = '```json\n[{"name": "qk_norm_rope", "op_chain": "q_norm -> rope"}]\n```'


def test_proposals_survive_a_session_that_hit_the_turn_ceiling(tmp_path) -> None:
    """Discovery spends turns by design; brushing the ceiling is not a failure.

    Observed on Qwen3-14B-FP8: five attempts each ended ``turn_cap``, every
    answer was dropped, and the run published ``llm_unavailable`` having done
    the analysis five times.
    """
    calls: list = []
    fn = discover_module.registered_agent_llm_fn(
        _backend_returning([AgentRunResult(text=PROPOSALS, end_reason="turn_cap")], calls),
        model="m",
        timeout_s=10,
        workdir=str(tmp_path),
    )

    assert fn("DISCOVERY PROMPT") == PROPOSALS
    assert len(calls) == 1, "a parseable answer must not be retried"


def test_a_cut_short_session_with_no_proposals_still_fails(tmp_path) -> None:
    calls: list = []
    fn = discover_module.registered_agent_llm_fn(
        _backend_returning([AgentRunResult(text="I could not finish reading", end_reason="turn_cap")], calls),
        model="m",
        timeout_s=10,
        workdir=str(tmp_path),
        attempts=2,
        base_delay_sec=0,
        max_delay_sec=0,
    )

    with pytest.raises(discover_module.LlmUnavailableError):
        fn("DISCOVERY PROMPT")
    assert len(calls) == 2


def test_a_failed_discovery_leaves_a_transcript(tmp_path) -> None:
    """Without it the end reason is all there is, and it cannot be diagnosed."""
    log_path = tmp_path / "discovery_llm.txt"
    calls: list = []
    fn = discover_module.registered_agent_llm_fn(
        _backend_returning([AgentRunResult(text="", end_reason="turn_cap")], calls),
        model="m",
        timeout_s=10,
        workdir=str(tmp_path),
        log_path=str(log_path),
        attempts=2,
        base_delay_sec=0,
        max_delay_sec=0,
    )

    with pytest.raises(discover_module.LlmUnavailableError):
        fn("DISCOVERY PROMPT")

    assert log_path.is_file(), "no transcript written for a failed discovery"
    assert "tool: Read qwen3.py" in log_path.read_text(encoding="utf-8")
