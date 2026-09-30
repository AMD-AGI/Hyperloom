# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ROCm/AITER build a KB config was tuned on is kept, compared with the pod's, and disclosed (#1507)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB, recipe_canonical_id
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import _find_config_donor, run_t0_anchor
from hyperloom.orchestrator.knowledge.stack_replay import (
    compare_stacks,
    config_stack_fingerprint,
    row_stack_fingerprint,
)

# --- the predicate -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "live"),
    [
        pytest.param("6.4.1", "6.4.1", id="same"),
        pytest.param("6.4.0", "6.4.3", id="older-patch-same-line"),
        pytest.param("6.4.1-120", "6.4.1", id="marker-build-suffix"),
    ],
)
def test_rocm_on_the_same_line_not_newer_matches(recorded, live):
    assert not compare_stacks({"rocm": recorded}, {"rocm": live}).conflicts


@pytest.mark.parametrize(
    ("recorded", "live"),
    [
        pytest.param("6.4.3", "6.4.1", id="recorded-newer"),
        pytest.param("6.3.4", "6.4.1", id="older-line"),
        pytest.param("7.0.0", "6.4.1", id="other-major"),
    ],
)
def test_rocm_off_the_line_or_newer_is_a_mismatch(recorded, live):
    comparison = compare_stacks({"rocm": recorded}, {"rocm": live})
    assert comparison.conflicts
    assert recorded in comparison.conflicts[0]


@pytest.mark.parametrize(
    ("recorded", "live"),
    [
        pytest.param("abc1234", "abc1234def5678", id="short-and-long-sha"),
        pytest.param("0.1.5", "0.1.5", id="same-version"),
        pytest.param("v0.1.5", "0.1.5", id="installer-tag-vs-dist"),
    ],
)
def test_aiter_matches(recorded, live):
    assert not compare_stacks({"aiter": recorded}, {"aiter": live}).conflicts


@pytest.mark.parametrize(
    ("recorded", "live"),
    [
        pytest.param("abc1234", "def5678", id="two-commits"),
        pytest.param("0.1.4", "0.1.5", id="two-versions"),
    ],
)
def test_aiter_known_disagreement_is_a_mismatch(recorded, live):
    assert compare_stacks({"aiter": recorded}, {"aiter": live}).conflicts


def test_an_aiter_commit_against_a_version_is_a_note_not_a_mismatch():
    """A SHA has no order and is not comparable to a version, so this is no claim either way."""
    comparison = compare_stacks({"aiter": "abc1234"}, {"aiter": "0.1.5"})
    assert not comparison.conflicts
    assert comparison.notes


@pytest.mark.parametrize("unknown", ["", "unknown", None])
def test_unknown_on_either_side_is_no_claim(unknown):
    for recorded, live in ((unknown, "6.4.1"), ("6.4.1", unknown)):
        comparison = compare_stacks({"rocm": recorded, "aiter": recorded}, {"rocm": live, "aiter": live})
        assert not comparison.conflicts
        assert not comparison.notes


def test_a_new_config_carries_this_sessions_stack_not_the_old_one():
    stored = {"stack_fingerprint": {"rocm_version": "6.3.0", "aiter_commit": "old1234", "vllm_version": "0.9"}}
    stamped = config_stack_fingerprint(stored, {"rocm": "6.4.1", "aiter": "unknown"})
    assert stamped == {"rocm_version": "6.4.1", "aiter_commit": "", "vllm_version": "0.9"}


# --- T0 end to end, through a real local KB ------------------------------------------------------------------------


@dataclass
class _State:
    recipe_kb_session_id: str = ""
    warm_start_ts: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    warm_start_context: dict[str, Any] = field(default_factory=dict)
    framework_name: str = "sglang"
    framework_version: str = "0.4.5"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    model_class: str = ""
    baseline_workload_extra: dict[str, Any] = field(default_factory=dict)
    compute_partition: dict[str, Any] = field(default_factory=dict)

    def save(self, _path: Path) -> None:
        """Persistence is not under test."""


def _seed_tuned_row(kb: RecipeKB, state: _State, *, rocm: str, aiter: str) -> str:
    cid = recipe_canonical_id(
        model="M",
        hardware="MI300X",
        framework_name="sglang",
        framework_version=state.framework_version,
        precision=state.precision,
    )
    kb.put_recipe(
        canonical_id=cid,
        model="M",
        hardware="MI300X",
        framework_name="sglang",
        framework_version=state.framework_version,
        precision=state.precision,
        best_config={"extra_server_args": "--attention-backend aiter", "extra_envs": {"SGLANG_USE_AITER": "1"}},
        best_throughput=1200.0,
        lessons=[{"statement": "aiter attention wins here", "measured_impact": "+12%"}],
        stack_fingerprint={"rocm_version": rocm, "aiter_commit": aiter},
        sessions=[{"session_id": "tuned", "gain_pct": 12.0}],
        provenance={"source": "seed", "generator": "ut"},
    )
    return cid


def _anchor(kb: RecipeKB, state: _State, tmp_path: Path, *, rocm: str, aiter: str) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    run_t0_anchor(
        kb,
        state,
        workload="M",
        hw="MI300X",
        stack_fingerprint={"rocm": rocm, "aiter": aiter},
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )


def test_a_config_tuned_on_another_aiter_still_replays_and_says_so(tmp_path):
    """Retrieval does not decide whether a config works on this stack; the warm replay measures it."""
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    state = _State()
    _seed_tuned_row(kb, state, rocm="6.4.1", aiter="abc1234")
    _anchor(kb, state, tmp_path, rocm="6.4.1", aiter="def5678")

    context = state.warm_start_context
    assert context["recommended_replay"]["extra_server_args"] == "--attention-backend aiter"
    assert context["stack_mismatch"]["conflicts"] == ["aiter commit abc1234 recorded, pod runs def5678"]
    assert [lesson["statement"] for lesson in context["lessons"]] == ["aiter attention wins here"]


def test_a_matching_stack_still_replays(tmp_path):
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    state = _State()
    _seed_tuned_row(kb, state, rocm="6.4.0", aiter="abc1234")
    _anchor(kb, state, tmp_path, rocm="6.4.1", aiter="abc1234def")

    replay = state.warm_start_context["recommended_replay"]
    assert replay["extra_server_args"] == "--attention-backend aiter"
    assert "stack_mismatch" not in state.warm_start_context


def test_the_anchor_does_not_overwrite_the_build_a_config_was_tuned_on(tmp_path):
    """T0 stamps its pod onto the row before the lookup; on a row with a config that would erase the evidence."""
    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    state = _State()
    cid = _seed_tuned_row(kb, state, rocm="6.4.1", aiter="abc1234")
    _anchor(kb, state, tmp_path, rocm="7.0.0", aiter="def5678")

    stored = row_stack_fingerprint(kb.get_recipe(canonical_id=cid))
    assert stored == {"rocm": "6.4.1", "aiter": "abc1234"}
    # And the next session on the same new pod still sees the real delta, not a restamped row that matches it.
    later = _State()
    _anchor(kb, later, tmp_path, rocm="7.0.0", aiter="def5678")
    assert later.warm_start_context["stack_mismatch"]["conflicts"]


# --- the borrowed-donor path ---------------------------------------------------------------------------------------


class _StubKB:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def search(self, *, label_match: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        return list(self._rows)


def _donor(cid: str, aiter: str) -> dict[str, Any]:
    return {
        "canonical_id": cid,
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hardware": "mi300x",
        "framework_version": "v1",
        "precision": "bf16",
        "validated_gain_pct": 10.0,
        "best_config": {"extra_server_args": f"--from-{cid}"},
        "stack_fingerprint": {"rocm_version": "6.4.1", "aiter_commit": aiter},
    }


def _find(kb):
    return _find_config_donor(
        kb,
        cid="self-cid",
        hardware="mi300x",
        framework="sglang",
        model_type="qwen2",
        arch_slug="qwen2forcausallm",
        framework_version="v1",
        precision="bf16",
        live_stack={"rocm": "6.4.1", "aiter": "def5678"},
    )


def test_among_equal_donors_the_one_on_the_pods_build_ranks_first():
    donor, _, _ = _find(_StubKB([_donor("mismatched", "abc1234"), _donor("compatible", "def5678")]))
    assert donor["canonical_id"] == "compatible"


def test_a_donor_on_another_build_is_still_offered_when_it_is_the_only_one():
    donor, _, _ = _find(_StubKB([_donor("mismatched", "abc1234")]))
    assert donor["canonical_id"] == "mismatched"


def test_a_better_gain_outranks_a_matching_build():
    """Stack agreement breaks ties; it does not override the measured gain."""
    better = {**_donor("better-gain-other-build", "abc1234"), "validated_gain_pct": 20.0}
    donor, _, _ = _find(_StubKB([_donor("compatible", "def5678"), better]))
    assert donor["canonical_id"] == "better-gain-other-build"


# --- the prompt ----------------------------------------------------------------------------------------------------


def test_the_prompt_discloses_the_stack_delta():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.warm_start_context = {
        "status": "hit",
        "match": {"tier": "exact", "confidence": 1.0},
        "stack_mismatch": {"conflicts": ["rocm 6.3.0 recorded, pod runs 6.4.1"], "notes": []},
    }
    summary = state.to_warm_start_summary()
    assert "tuned on a different stack (the replay measures it): rocm 6.3.0 recorded, pod runs 6.4.1" in summary
