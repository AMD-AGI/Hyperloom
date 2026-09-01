# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Research scout: tag/domain wiring, hint artifacts, and state bookkeeping for the read-only PRELUDE collector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session import session_paths
from hyperloom.orchestrator.knowledge import research_hints
from hyperloom.orchestrator.specialists import domains as sd
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts import (
    specialist_prompt_builder as spb,
)


def test_research_scout_tag_and_domain_registered():
    assert "research_scout" in sd.KNOWLEDGE_DOMAIN_TAGS
    assert "research_scout_specialist" in sd.SPECIALIST_DOMAIN_KEYS


def test_research_scout_has_focus_template():
    assert "research_scout_specialist" in spb._DOMAIN_FOCUS_TEMPLATES


def test_skeleton_always_present(tmp_path: Path):
    research_hints.write_hints_skeleton(tmp_path)
    assert session_paths.research_hints_md(tmp_path).exists()
    assert research_hints.load_hints(tmp_path) == []


def test_append_drops_sourceless_and_dedups(tmp_path: Path):
    added, dropped = research_hints.append_hints(
        tmp_path,
        [
            {"what": "chunked prefill", "source": "ROCm/vllm#1", "domain_tags": ["serving"]},
            {"what": "no source"},
            {"what": "chunked prefill", "source": "ROCm/vllm#1"},
        ],
    )
    assert (added, dropped) == (1, 1)
    hints = research_hints.load_hints(tmp_path)
    assert len(hints) == 1
    assert hints[0]["source"] == "ROCm/vllm#1"


def test_append_is_additive_across_runs(tmp_path: Path):
    research_hints.append_hints(
        tmp_path,
        [
            {"what": "a", "source": "s1"},
        ],
    )
    research_hints.append_hints(
        tmp_path,
        [
            {"what": "b", "source": "s2"},
        ],
    )
    whats = {h["what"] for h in research_hints.load_hints(tmp_path)}
    assert whats == {"a", "b"}


def test_skeleton_does_not_clobber_existing(tmp_path: Path):
    research_hints.append_hints(tmp_path, [{"what": "a", "source": "s1"}])
    research_hints.write_hints_skeleton(tmp_path)
    assert len(research_hints.load_hints(tmp_path)) == 1


def test_competitor_target_requires_per_conc_source(tmp_path: Path):
    ok = research_hints.write_competitor_target(
        tmp_path,
        {
            "gpu": "MI300X",
            "per_conc": [
                {"conc": 8, "tput_per_gpu": 1000, "source": "mlperf"},
                {"conc": 16, "tput_per_gpu": 900},
            ],
        },
    )
    assert ok is True
    data = json.loads(session_paths.competitor_target_json(tmp_path).read_text())
    assert len(data["per_conc"]) == 1
    assert data["per_conc"][0]["source"] == "mlperf"


def test_competitor_target_all_sourceless_writes_nothing(tmp_path: Path):
    ok = research_hints.write_competitor_target(
        tmp_path,
        {
            "per_conc": [{"conc": 8, "tput_per_gpu": 1000}],
        },
    )
    assert ok is False
    assert not session_paths.competitor_target_json(tmp_path).exists()


def test_scout_counters_and_seen_pr_roundtrip():
    s = SharedState()
    assert s.research_scout_enabled is True
    assert s.research_scout_runs == 0
    assert s.bump_research_scout_runs() == 1
    assert s.register_seen_pr_ids(["a", "b", "a", ""]) == 2
    assert "a" in s.research_scout_seen_pr_ids
    assert "zzz" not in s.research_scout_seen_pr_ids
    restored = SharedState.from_dict(s.to_dict())
    assert restored.research_scout_runs == 1
    assert "b" in restored.research_scout_seen_pr_ids


@pytest.mark.asyncio
async def test_internal_research_scout_task_is_readonly(tmp_path: Path):
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import (
        MockBackend,
        MockTurn,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    state = SharedState(session_id="research-scout-readonly")
    state.register_seen_pr_ids(["https://pr/seen"])
    state.specialist_rounds = [
        {
            "domain": "research_scout_specialist",
            "residual_questions": ["Does this vLLM version support the backend?"],
        }
    ]
    state.explore_search = {
        "accepted": [
            {
                "name": "kept-variant",
                "source_evidence": ["https://pr/kept"],
            }
        ]
    }
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "critic": MockBackend(idle),
        "robustness": MockBackend(idle),
    }
    coord = Coordinator(
        session_dir=tmp_path,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )

    task = await coord._enqueue_internal_research_scout_task(
        reason="test",
        round_id=0,
    )

    assert task is not None
    assert task.params["mode"] == "research"
    assert task.params["source_phase"] == "PRELUDE"
    assert task.side_effects == ["writes_results"]
    assert task.params["seen_pr_ids"] == ["https://pr/seen"]
    assert "Does this vLLM version support the backend?" in task.params["notes"]
    assert {"name": "kept-variant", "source": "https://pr/kept"} in task.params["already_proven"]

    research_hints.append_hints(
        tmp_path,
        [{"what": "enable aiter", "source": "https://example.test/aiter"}],
    )
    coord._seed_gaps_from_research_hints()
    first_id = next(row["canonical_id"] for row in coord.shared_state.gaps if row.get("symptom") == "enable aiter")
    research_hints.append_hints(
        tmp_path,
        [{"what": "use hipblaslt", "source": "https://example.test/hipblaslt"}],
    )
    coord._seed_gaps_from_research_hints()
    assert any(row.get("canonical_id") == first_id for row in coord.shared_state.gaps)
