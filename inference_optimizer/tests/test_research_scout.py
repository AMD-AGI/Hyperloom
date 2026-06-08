# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Research scout: tag/domain wiring, hint artifacts, and state bookkeeping.

Covers the read-only PRELUDE research collector:

1. The ``research_scout`` knowledge-domain tag + ``research_scout_specialist``
   domain are registered (so PolicyGate accepts internal dispatch).
2. ``research_hints`` artifacts enforce the source-required contract,
   append-only merge, and the always-present PRELUDE skeleton.
3. ``competitor_target`` drops per-concurrency rows missing a source.
4. SharedState scout counters / seen-PR dedup round-trip through JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer import session_paths
from inference_optimizer.orchestrator import research_hints
from inference_optimizer.orchestrator import specialist_domains as sd
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.system_prompts import (
    specialist_prompt_builder as spb,
)


# ---------------------------------------------------------------------------
# 1. tag + domain wiring
# ---------------------------------------------------------------------------
def test_research_scout_tag_and_domain_registered():
    assert "research_scout" in sd.KNOWLEDGE_DOMAIN_TAGS
    assert "research_scout_specialist" in sd.SPECIALIST_DOMAIN_KEYS


def test_research_scout_has_focus_template():
    assert "research_scout_specialist" in spb._DOMAIN_FOCUS_TEMPLATES


# ---------------------------------------------------------------------------
# 2. research_hints artifacts
# ---------------------------------------------------------------------------
def test_skeleton_always_present(tmp_path: Path):
    research_hints.write_hints_skeleton(tmp_path)
    assert session_paths.research_hints_md(tmp_path).exists()
    assert research_hints.load_hints(tmp_path) == []


def test_append_drops_sourceless_and_dedups(tmp_path: Path):
    added, dropped = research_hints.append_hints(tmp_path, [
        {"what": "chunked prefill", "source": "ROCm/vllm#1",
         "domain_tags": ["serving"]},
        {"what": "no source"},
        {"what": "chunked prefill", "source": "ROCm/vllm#1"},
    ])
    assert (added, dropped) == (1, 1)
    hints = research_hints.load_hints(tmp_path)
    assert len(hints) == 1
    assert hints[0]["source"] == "ROCm/vllm#1"


def test_append_is_additive_across_runs(tmp_path: Path):
    research_hints.append_hints(tmp_path, [
        {"what": "a", "source": "s1"},
    ])
    research_hints.append_hints(tmp_path, [
        {"what": "b", "source": "s2"},
    ])
    whats = {h["what"] for h in research_hints.load_hints(tmp_path)}
    assert whats == {"a", "b"}


def test_skeleton_does_not_clobber_existing(tmp_path: Path):
    research_hints.append_hints(tmp_path, [{"what": "a", "source": "s1"}])
    research_hints.write_hints_skeleton(tmp_path)
    assert len(research_hints.load_hints(tmp_path)) == 1


# ---------------------------------------------------------------------------
# 3. competitor_target source contract
# ---------------------------------------------------------------------------
def test_competitor_target_requires_per_conc_source(tmp_path: Path):
    ok = research_hints.write_competitor_target(tmp_path, {
        "gpu": "MI300X",
        "per_conc": [
            {"conc": 8, "tput_per_gpu": 1000, "source": "mlperf"},
            {"conc": 16, "tput_per_gpu": 900},
        ],
    })
    assert ok is True
    data = json.loads(
        session_paths.competitor_target_json(tmp_path).read_text()
    )
    assert len(data["per_conc"]) == 1
    assert data["per_conc"][0]["source"] == "mlperf"


def test_competitor_target_all_sourceless_writes_nothing(tmp_path: Path):
    ok = research_hints.write_competitor_target(tmp_path, {
        "per_conc": [{"conc": 8, "tput_per_gpu": 1000}],
    })
    assert ok is False
    assert not session_paths.competitor_target_json(tmp_path).exists()


# ---------------------------------------------------------------------------
# 4. SharedState bookkeeping
# ---------------------------------------------------------------------------
def test_scout_counters_and_seen_pr_roundtrip():
    s = SharedState()
    assert s.research_scout_enabled is True
    assert s.research_scout_runs == 0
    assert s.bump_research_scout_runs() == 1
    assert s.register_seen_pr_ids(["a", "b", "a", ""]) == 2
    assert s.has_seen_pr_id("a") is True
    assert s.has_seen_pr_id("zzz") is False
    restored = SharedState.from_dict(s.to_dict())
    assert restored.research_scout_runs == 1
    assert restored.has_seen_pr_id("b") is True


@pytest.mark.asyncio
async def test_internal_research_scout_task_is_readonly(tmp_path: Path):
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="research-scout-readonly")
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "kernel": MockBackend(idle),
        "critic": MockBackend(idle),
        "robustness": MockBackend(idle),
    }
    coord = Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        cortex_kb=None,
        knowledge_plane=None,
    )

    task = await coord._enqueue_internal_research_scout_task(
        reason="test", round_id=0,
    )

    assert task is not None
    assert task.params["readonly"] is True
    assert task.allowed_tools == [
        "Read", "Grep", "Glob", "Write", "WebSearch", "WebFetch",
    ]
    assert "Bash" not in task.allowed_tools
    assert "Edit" not in task.allowed_tools
    assert "MultiEdit" not in task.allowed_tools
    assert task.side_effects == ["writes_results"]
