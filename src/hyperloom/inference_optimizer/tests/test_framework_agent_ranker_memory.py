# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""FRAMEWORK ranker soft-guidance from working memory.

Covers the deterministic working-memory aggregation
(``_build_framework_working_memory``), its prompt rendering
(``_render_framework_memory_for_prompt``), and that the candidate ranker folds
the "already tried this session" negative-sample block into its prompt.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from hyperloom.orchestrator.loop.coordinator import Coordinator


class _StateStub:
    def __init__(self) -> None:
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.framework_agent_critic_decisions: list[dict[str, Any]] = []
        self.research_scout_seen_pr_ids: list[str] = []
        # Workload context read by the ranker prompt.
        self.model = "test-model"
        self.model_path = ""
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.precision = "fp8"
        self.tp = 4
        self.best_throughput = 0.0
        self.baseline_throughput = 0.0


class _MemCoord:
    _FRAMEWORK_KEEP_STATUSES = Coordinator._FRAMEWORK_KEEP_STATUSES
    _FRAMEWORK_TRIED_MEMORY_CAP = Coordinator._FRAMEWORK_TRIED_MEMORY_CAP
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _framework_known_candidate_ids = Coordinator._framework_known_candidate_ids
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _build_framework_working_memory = Coordinator._build_framework_working_memory
    _render_framework_memory_for_prompt = staticmethod(Coordinator._render_framework_memory_for_prompt)
    _match_framework_agent_candidate = staticmethod(Coordinator._match_framework_agent_candidate)
    _rank_framework_agent_candidates_llm = Coordinator._rank_framework_agent_candidates_llm

    def __init__(self) -> None:
        self.shared_state = _StateStub()
        self.framework_ranker_timeout_sec = 5.0


def test_build_working_memory_aggregates_tried_excluded_learnings():
    coord = _MemCoord()
    st = coord.shared_state
    st.framework_agent_phase_progress = [
        {"candidate_id": "PR:723", "status": "reverted", "gain_pct": 0.0, "rationale": "throughput == baseline"},
        {"candidate_id": "PR:1015", "status": "critic_denied", "rationale": "does not address mem-bw bottleneck"},
        {"candidate_id": "PR:900", "status": "kept", "gain_pct": 6.1},
    ]
    st.framework_agent_batches = [
        {
            "batch_id": "b1",
            "candidates": [
                {"candidate_id": "PR:723"},
                {"candidate_id": "PR:1015"},
                {"candidate_id": "PR:900"},
                {"candidate_id": "PR:2000"},
            ],
        },
    ]
    st.framework_agent_critic_decisions = [
        {"candidate_id": "PR:1015", "verdict": "reject", "rationale": "does not address mem-bw bottleneck"},
        {"candidate_id": "PR:5", "verdict": "approve", "rationale": "looks good"},
    ]

    mem = coord._build_framework_working_memory()

    refs = {t["ref"] for t in mem["tried_and_why"]}
    assert refs == {"PR:723", "PR:1015", "PR:900"}
    revert = next(t for t in mem["tried_and_why"] if t["ref"] == "PR:723")
    assert revert["status"] == "reverted"
    assert revert["gain_pct"] == 0.0
    assert "baseline" in revert["why"]
    # excluded_refs = known ids ∪ processed keys.
    assert {"PR:723", "PR:1015", "PR:900", "PR:2000"} <= set(mem["excluded_refs"])
    # learnings only from denied critic decisions.
    assert mem["learnings"] == ["does not address mem-bw bottleneck"]
    # pending = unprocessed candidate in the latest batch.
    assert mem["pending"] == ["PR:2000"]


def test_build_working_memory_empty_when_no_progress():
    coord = _MemCoord()
    mem = coord._build_framework_working_memory()
    assert mem["tried_and_why"] == []
    assert mem["learnings"] == []


def test_build_working_memory_caps_tried_rows():
    coord = _MemCoord()
    cap = coord._FRAMEWORK_TRIED_MEMORY_CAP
    coord.shared_state.framework_agent_phase_progress = [
        {"candidate_id": f"PR:{i}", "status": "reverted"} for i in range(cap + 5)
    ]
    mem = coord._build_framework_working_memory()
    assert len(mem["tried_and_why"]) == cap
    # Most-recent kept (last cap entries).
    assert mem["tried_and_why"][-1]["ref"] == f"PR:{cap + 4}"


def test_render_memory_empty_returns_blank():
    assert Coordinator._render_framework_memory_for_prompt(None) == ""
    assert Coordinator._render_framework_memory_for_prompt({"tried_and_why": [], "learnings": []}) == ""


def test_render_memory_lists_tried_and_learnings():
    mem = {
        "tried_and_why": [
            {"ref": "PR:723", "status": "reverted", "gain_pct": 0.0, "why": "no-op on SD3 path"},
            {"ref": "PR:1015", "status": "critic_denied", "gain_pct": None, "why": "wrong bottleneck"},
        ],
        "learnings": ["env-only fp8 levers are no-op"],
    }
    txt = Coordinator._render_framework_memory_for_prompt(mem)
    assert "Already tried THIS session" in txt
    assert "PR:723 [reverted] gain=+0.00% — no-op on SD3 path" in txt
    assert "PR:1015 [critic_denied] — wrong bottleneck" in txt
    assert "Learnings" in txt
    assert "env-only fp8 levers are no-op" in txt


class _FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def __aiter__(self):
        async def _gen():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=self._text))]
            )
        return _gen()


class _FakeCompletions:
    def __init__(self, captured: dict[str, Any], reply: str) -> None:
        self._captured = captured
        self._reply = reply

    async def create(self, *, model: str, messages: list[dict[str, Any]], **_kw: Any):
        self._captured["prompt"] = messages[0]["content"]
        return _FakeStream(self._reply)


class _FakeClient:
    def __init__(self, captured: dict[str, Any], reply: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(captured, reply))


def test_ranker_prompt_includes_tried_block_and_returns_match():
    coord = _MemCoord()
    coord.shared_state.framework_agent_phase_progress = [
        {"candidate_id": "PR:723", "status": "reverted", "gain_pct": 0.0, "rationale": "no-op on SD3 path"},
    ]
    captured: dict[str, Any] = {}
    coord._framework_agent_ranker_client = lambda: _FakeClient(captured, '{"candidate_id": "PR:2000", "reason": "attacks mem-bw"}')  # type: ignore[attr-defined]
    coord._framework_agent_ranker_model = lambda: "m"  # type: ignore[attr-defined]

    candidates = [
        {"candidate_id": "PR:2000", "title": "moe gemm fastpath", "repo": "sgl-project/sglang"},
        {"candidate_id": "PR:3000", "title": "kv cache dtype", "repo": "sgl-project/sglang"},
    ]
    chosen = asyncio.run(coord._rank_framework_agent_candidates_llm(candidates))

    # The already-tried negative-sample block is present in the ranker prompt.
    assert "Already tried THIS session" in captured["prompt"]
    assert "PR:723 [reverted]" in captured["prompt"]
    # And the ranker resolves the model's pick.
    assert chosen is not None
    assert chosen["candidate_id"] == "PR:2000"


def test_ranker_prompt_omits_tried_block_when_no_history():
    coord = _MemCoord()
    captured: dict[str, Any] = {}
    coord._framework_agent_ranker_client = lambda: _FakeClient(captured, '{"candidate_id": "PR:2000"}')  # type: ignore[attr-defined]
    coord._framework_agent_ranker_model = lambda: "m"  # type: ignore[attr-defined]

    candidates = [
        {"candidate_id": "PR:2000", "title": "x", "repo": "r"},
        {"candidate_id": "PR:3000", "title": "y", "repo": "r"},
    ]
    asyncio.run(coord._rank_framework_agent_candidates_llm(candidates))
    assert "Already tried THIS session" not in captured["prompt"]
