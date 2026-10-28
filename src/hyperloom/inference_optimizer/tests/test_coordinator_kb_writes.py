# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Coordinator -> recipe-snapshot KB write chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles.mock_backend import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity
from hyperloom.orchestrator.knowledge.recipe_kb import (
    RecipeUpsert,
    LocalRecipeStore,
    LocalRecipeStoreError,
    RecipeKB,
    recipe_canonical_id,
)
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig
from hyperloom.orchestrator.knowledge.recipe_journal import (
    recipe_attrs_from_state,
)
from hyperloom.inference_optimizer.breakdown.recorder import close_out as _close_out


_MODEL = "qwen3-30b-a3b"
_HW = "mi300x"
_FW = "sglang"
_FWV = "0.4.5"
_PREC = "fp8"


def _make_coordinator(tmp_path: Path) -> Coordinator:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    backends = {
        "orchestration": MockBackend(idle),
        "critic": MockBackend(idle),
    }

    kb = RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"))
    coord = Coordinator(
        session_dir=session_dir,
        backends=backends,
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=KnowledgePlane(recipe_kb=kb)),
    )
    ss = coord.shared_state
    ss.model_name = _MODEL
    ss.gpu_type = _HW
    ss.framework = _FW
    ss.framework_version = _FWV
    ss.precision = _PREC
    return coord


def _expected_cid() -> str:
    return recipe_canonical_id(
        RecipeIdentity(
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
        )
    )


def test_workload_canonical_id_defined_and_consistent(tmp_path: Path) -> None:
    """``workload_canonical_id`` exists and agrees with ``recipe_canonical_id``."""
    coord = _make_coordinator(tmp_path)
    assert hasattr(coord.recipe_journal, "workload_canonical_id")
    assert coord.recipe_journal.workload_canonical_id() == _expected_cid()
    assert coord.recipe_journal.workload_canonical_id() == _expected_cid()


def test_kb_amend_recipe_persists_lesson(tmp_path: Path) -> None:
    """Appending a lesson lands in the local KB."""
    coord = _make_coordinator(tmp_path)
    coord.recipe_journal.kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "lesson write silently no-opped"
    statements = [l.get("statement") for l in (row.get("lessons") or [])]
    assert "raise tp to 8" in statements


def test_kb_amend_recipe_persists_pitfall(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    coord.recipe_journal.kb_amend_recipe(
        append_pitfall={"description": "ep=8 OOMs on 30B"},
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    descs = [p.get("description") for p in (row.get("pitfalls") or [])]
    assert "ep=8 OOMs on 30B" in descs


def test_kb_amend_recipe_is_noop_in_remote_mode(tmp_path: Path) -> None:
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)

    class _ForbiddenRecipeKB:
        def __getattr__(self, name: str):
            raise AssertionError(f"remote amend accessed RecipeKB: {name}")

    coord.knowledge_plane = KnowledgePlane(recipe_kb=_ForbiddenRecipeKB())
    coord.knowledge_plane = SimpleNamespace(
        config=KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_URL": "https://kb.test",
                "KB_STORE_TOKEN": "token",
            }
        )
    )
    coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "must not write", "measured_impact": ""})


def test_record_fact_per_variant_stamps_best_config_on_keep(tmp_path: Path) -> None:
    """KEEP with structured variant args must write best_config for warm-replay."""
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)
    task = SimpleNamespace(
        kind="explore",
        task_id="t-keep-bc",
        params={},
    )
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="sess-1",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "disable_radix",
            "variant": {"extra_server_args": "--disable-radix-cache"},
            "metrics": {"gain_pct": 0.66, "output_throughput": 6700.0},
        },
        adopted=True,
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    bc = row.get("best_config") or {}
    assert bc.get("extra_server_args") == "--disable-radix-cache"
    assert float(row.get("best_throughput") or 0.0) == 6700.0
    assert any("disable-radix-cache" in str(l.get("statement") or "") for l in (row.get("lessons") or []))


def test_record_fact_per_variant_writes_no_lesson_for_an_unadopted_keep(tmp_path: Path) -> None:
    """An executor KEEP whose lift did not land is not a lesson or a best_config."""
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)
    task = SimpleNamespace(kind="explore", task_id="t-keep-refused", params={})
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="sess-1",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "disable_radix",
            "variant": {"extra_server_args": "--disable-radix-cache"},
            "metrics": {"gain_pct": 0.66, "output_throughput": 6700.0},
        },
        adopted=False,
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid()) or {}
    assert not (row.get("lessons") or [])
    assert not (row.get("best_config") or {})


def test_record_fact_per_variant_does_not_clobber_better_best_config(
    tmp_path: Path,
) -> None:
    """A weaker KEEP must not overwrite an existing stronger best_config."""
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={"extra_server_args": "--page-size 32"},
            best_throughput=7000.0,
        )
    )
    task = SimpleNamespace(kind="explore", task_id="t-weaker", params={})
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="sess-1",
        variant_outcome={
            "outcome": "KEEP",
            "variant_name": "small_gain",
            "variant": {"extra_server_args": "--disable-radix-cache"},
            "metrics": {"gain_pct": 0.1, "output_throughput": 6600.0},
        },
        adopted=True,
    )
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    bc = row.get("best_config") or {}
    assert bc.get("extra_server_args") == "--page-size 32"
    assert float(row.get("best_throughput") or 0.0) == 7000.0


def test_kb_amend_recipe_stamps_architecture_tags(tmp_path: Path) -> None:
    """Amend stamps config.json architecture tags into the recipe."""
    coord = _make_coordinator(tmp_path)
    coord.shared_state.model_architectures = ["LlamaForCausalLM"]
    coord.shared_state.model_type = "llama"
    coord.recipe_journal.kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.recipe_kb.get_recipe(canonical_id=coord.recipe_journal.workload_canonical_id())
    assert row is not None
    assert row.get("architectures") == ["LlamaForCausalLM"]
    assert row.get("model_type") == "llama"


def test_kb_amend_recipe_skips_empty_architecture_tags(tmp_path: Path) -> None:
    """With no config.json tags the amend must NOT stamp empty ``architectures`` / ``model_type`` keys."""
    coord = _make_coordinator(tmp_path)
    coord.recipe_journal.kb_amend_recipe(
        append_lesson={"statement": "raise tp to 8", "measured_impact": "+12%"},
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    assert "architectures" not in row
    assert "model_type" not in row


def test_sdk_fallback_t0_anchors_into_self_recipe_kb(tmp_path: Path) -> None:
    """The SDK-fallback T0 anchor runs and writes into the SAME dispatcher the Coordinator holds."""
    coord = _make_coordinator(tmp_path)
    # Clear the markers and re-anchor the canonical Recipe identity.
    coord.shared_state.warm_start_ts = ""
    coord.shared_state.recipe_kb_session_id = ""
    coord.phase_machine.ensure_recipe_kb_t0_anchored()
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "SDK-fallback T0 did not anchor into self.recipe_kb"


# The on-disk row must preserve severity / dict measured_impact / session provenance, or warm-start + dedup lose data.
def _put(store: LocalRecipeStore, **kw) -> None:
    store.put_recipe(
        RecipeUpsert(
            canonical_id=_expected_cid(),
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            **kw,
        ),
    )


def test_local_store_preserves_pitfall_severity(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(store, pitfalls=[{"description": "ep=8 OOMs on 30B", "severity": "crash"}])
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    assert row["pitfalls"][0]["severity"] == "crash"


def test_local_store_preserves_lesson_dict_measured_impact(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(
        store,
        lessons=[
            {
                "statement": "raise tp to 8",
                "measured_impact": {"gain_pct": 12.0, "throughput_after": 1000.0},
            }
        ],
    )
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    mi = row["lessons"][0]["measured_impact"]
    assert isinstance(mi, dict), f"measured_impact got mangled to {type(mi)}"
    assert mi["gain_pct"] == 12.0


def test_local_store_preserves_session_provenance(tmp_path: Path) -> None:
    store = LocalRecipeStore(root=tmp_path / "kb")
    _put(store, sessions=[{"session_id": "sess-1", "gain_pct": 12.0, "stack_len": 3}])
    row = store.get_recipe(canonical_id=_expected_cid())
    assert row is not None
    s = row["sessions"][0]
    assert s["session_id"] == "sess-1"
    assert s["gain_pct"] == 12.0
    assert s["stack_len"] == 3


# kb_amend_recipe reads the LOCAL row and preserves T0-stamped extras + audit fields; appends accumulate instead of
# clobbering.
def test_amend_preserves_t0_extras_and_audit(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            authority="AUTHORITATIVE",
            confidence=0.99,
            extras={"model_class": "moe", "image_digest": "sha256:abc"},
        )
    )
    coord.recipe_journal.kb_amend_recipe(
        append_lesson={"statement": "x", "measured_impact": "+1%"},
    )
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row["model_class"] == "moe"
    assert row["image_digest"] == "sha256:abc"
    assert row["authority"] == "AUTHORITATIVE"
    assert row["confidence"] == 0.99


def test_amend_appends_lessons_cumulatively(tmp_path: Path) -> None:
    """Local read-modify-write accumulates, not overwrites."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "first", "measured_impact": "+1%"})
    coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "second", "measured_impact": "+2%"})
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert [l["statement"] for l in row["lessons"]] == ["first", "second"]


# CLOSE finalize must not clobber a better historical best_config with an empty/worse current result, and must merge
# the fingerprint.
def test_close_does_not_clobber_better_best_config(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={"name": "good", "tput": "1000"},
            best_throughput=1000.0,
            stack_fingerprint={"vllm_version": "0.6.0"},
        )
    )
    coord.shared_state.current_best = {}
    coord.recipe_journal.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 1000.0, "empty CLOSE clobbered a better config"
    assert row["best_config"].get("name") == "good"
    assert row["stack_fingerprint"].get("vllm_version") == "0.6.0"


def test_close_finalize_keeps_the_kernel_optimizations_an_earlier_session_recorded(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    prior = {"kernel_id": "k006", "micro_speedup": 1.3202, "decision": "KEEP", "e2e_gain_pct": -0.094}
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={"name": "good", "tput": "1000"},
            best_throughput=1000.0,
            extras={"kernel_optimizations": [prior]},
        )
    )

    assert coord.recipe_journal.finalize_recipe_and_journal()["result_type"] == "written"

    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row["version"] == 2
    assert [(k["kernel_id"], k["micro_speedup"]) for k in row["kernel_optimizations"]] == [("k006", 1.3202)]


# A bare-baseline CLOSE whose tput exceeds a historical best must NOT overwrite the validated best_config.
def test_close_does_not_clobber_with_bare_baseline_higher_tput(
    tmp_path: Path,
) -> None:
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={
                "name": "warm_replay",
                "extra_server_args": "--schedule-policy lpm --page-size 16",
                "tput": "2532",
            },
            best_throughput=2532.0,
        )
    )
    # Bare baseline: no validated stack/gain, but a higher tput the better-throughput guard alone would let through.
    ss = coord.shared_state
    ss.current_best = {"action": "baseline", "name": "baseline", "tput": 2813.5}
    ss.optimization_stack = []
    ss.cumulative_gain_validated = 0.0
    coord.recipe_journal.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 2532.0, "bare-baseline CLOSE clobbered a validated best_throughput"
    assert row["best_config"].get("extra_server_args") == ("--schedule-policy lpm --page-size 16"), (
        "warm_replay launch flags were dropped by a flagless baseline overwrite"
    )


def test_best_config_reads_stack_args_from_canonical_server_key(
    tmp_path: Path,
) -> None:
    """best_config reads the canonical ``extra_server_args`` stack key."""
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.current_best = {
        "name": "tuned",
        "extra_server_args": "--page-size 16 --page-size 16",
        "tput": 2200.0,
    }
    ss.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "page32",
            "candidate_extra_server_args": "--page-size 32 --schedule-policy lpm",
            "extra_server_args": "--page-size 32 --schedule-policy lpm",
        }
    ]
    ss.cumulative_gain_validated = 10.0
    attrs = recipe_attrs_from_state(coord.shared_state, coord.session_dir.name)
    assert attrs["best_config"]["extra_server_args"] == ("--page-size 32 --schedule-policy lpm"), (
        "stack-layer launch args must be read from the canonical "
        "extra_server_args key; got "
        f"{attrs['best_config'].get('extra_server_args')!r}"
    )


def test_close_overwrites_best_when_validated_win(tmp_path: Path) -> None:
    """Counterpart guard: a genuine validated win DOES update best_config/best_throughput."""
    coord = _make_coordinator(tmp_path)
    cid = _expected_cid()
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=cid,
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            best_config={"name": "old", "extra_server_args": "--page-size 16", "tput": "2000"},
            best_throughput=2000.0,
        )
    )
    ss = coord.shared_state
    ss.current_best = {
        "name": "tuned",
        "extra_server_args": "--page-size 32 --schedule-policy lpm",
        "tput": 2200.0,
    }
    ss.optimization_stack = [
        {
            "action": "explore",
            "variant_name": "page32",
            "extra_server_args": "--page-size 32 --schedule-policy lpm",
        }
    ]
    ss.cumulative_gain_validated = 10.0
    ss.cumulative_gain_validated_stack_len = 1
    coord.recipe_journal.finalize_recipe_and_journal()
    row = coord.recipe_kb.get_recipe(canonical_id=cid)
    assert row["best_throughput"] == 2200.0
    assert "--page-size 32" in row["best_config"].get("extra_server_args", "")


def _stack_one_keep(state) -> None:
    state.current_best = {"name": "page32", "extra_server_args": "--page-size 32", "tput": 2200.0}
    state.optimization_stack = [{"action": "explore", "variant_name": "page32", "extra_server_args": "--page-size 32"}]
    state.cumulative_gain_validated = 10.0


def test_close_skips_a_stack_that_grew_after_validation(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    _stack_one_keep(coord.shared_state)

    def _must_not_finalize_journal():
        raise AssertionError("an unvalidated working recipe reached journal finalization")

    coord.recipe_journal.ensure_journal = _must_not_finalize_journal

    outcome = coord.recipe_journal.finalize_recipe_and_journal()

    assert outcome == {
        "status": "skipped",
        "reason": "unvalidated_recipe_stack",
        "backend": "none",
        "result_type": "unvalidated_recipe",
    }
    assert coord.recipe_kb.get_recipe(canonical_id=_expected_cid()) is None


def test_close_skips_a_same_length_lift_the_watermark_cannot_see(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    state = coord.shared_state
    _stack_one_keep(state)
    state.cumulative_gain_validated_stack_len = 1
    state.working_recipe_generation = 2
    state.validated_recipe_generation = 1

    outcome = coord.recipe_journal.finalize_recipe_and_journal()

    assert outcome["reason"] == "unvalidated_recipe_stack"
    assert coord.recipe_kb.get_recipe(canonical_id=_expected_cid()) is None


def test_lift_then_validation_leaves_the_recipe_publishable(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    state = coord.shared_state
    state.baseline_tput = 1000.0
    state.current_best = {"action": "baseline", "tput": 1000.0, "extra_server_args": "", "extra_envs": {}}

    assert coord.writeback.lift_to_current_best(
        "explore",
        1100.0,
        {"name": "page16", "extra_server_args": "--page-size 16", "candidate_extra_server_args": "--page-size 16"},
    )
    assert (state.working_recipe_generation, state.validated_recipe_generation) == (1, 0)
    assert state.optimization_stack_has_unvalidated_keeps()

    assert coord.writeback.validate(1100.0, {"output_throughput": 1100.0})
    assert state.validated_recipe_generation == state.working_recipe_generation == 1

    outcome = coord.recipe_journal.finalize_recipe_and_journal()

    assert outcome["result_type"] == "written"
    assert coord.recipe_kb.get_recipe(canonical_id=_expected_cid())["best_throughput"] == 1100.0


def test_persisted_best_config_carries_the_full_stack_not_the_last_layer_delta(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    state = coord.shared_state
    state.baseline_tput = 1000.0
    state.current_best = {"action": "baseline", "tput": 1000.0, "extra_server_args": "", "extra_envs": {}}
    mtp = '--speculative-config {"method":"mtp","num_speculative_tokens":1} --max-model-len 13312'

    assert coord.writeback.lift_to_current_best(
        "explore", 1400.0, {"name": "mtp", "candidate_extra_server_args": mtp, "extra_server_args": mtp}
    )
    assert coord.writeback.lift_to_current_best(
        "explore",
        1480.0,
        {
            "name": "async-sched",
            "candidate_extra_server_args": "--async-scheduling",
            "extra_server_args": f"{mtp} --async-scheduling",
        },
    )
    assert coord.writeback.validate(1480.0, {"output_throughput": 1480.0})

    assert coord.recipe_journal.finalize_recipe_and_journal()["result_type"] == "written"
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    args = row["best_config"]["extra_server_args"]
    assert row["best_throughput"] == 1480.0
    assert '--speculative-config {"method":"mtp","num_speculative_tokens":1}' in args
    assert "--max-model-len 13312" in args
    assert "--async-scheduling" in args


# sessions[] entry must carry throughput_before/after, a date, and stack actions.
def test_session_entry_carries_throughput_date_and_actions(
    tmp_path: Path,
) -> None:
    coord = _make_coordinator(tmp_path)
    ss = coord.shared_state
    ss.baseline_tput = 2000.0
    ss.current_best = {
        "name": "tuned",
        "tput": 2150.0,
        "extra_server_args": "--page-size 32",
    }
    ss.optimization_stack = [
        {"action": "explore", "variant_name": "page32"},
        {"action": "explore", "variant_name": "stream_interval_4"},
    ]
    ss.cumulative_gain_validated = 7.5
    ss.cumulative_gain_validated_stack_len = 2
    attrs = recipe_attrs_from_state(coord.shared_state, coord.session_dir.name)
    sessions = attrs.get("sessions") or []
    assert sessions, "no session entry emitted"
    s = sessions[0]
    assert s["throughput_before"] == 2000.0
    assert s["throughput_after"] == 2150.0
    assert s["date"], "session date must not be empty"
    assert s["actions_taken"] == ["page32", "stream_interval_4"]


# A per-variant pitfall with an empty variant dict must still carry the variant NAME in its description, not collapse
# to the bare task kind.
def test_pitfall_description_uses_variant_name_not_bare_kind(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    coord = _make_coordinator(tmp_path)
    task = SimpleNamespace(kind="explore", task_id="t-1")
    coord.recipe_journal._record_fact_per_variant(
        task=task,
        source_session_id="sess-1",
        variant_outcome={
            "outcome": "REVERT",
            "variant_name": "page64_no_radix",
            "variant": {},
            "metrics": {"gain_pct": -10.0},
        },
        adopted=False,
    )
    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    descs = [p.get("description") for p in (row.get("pitfalls") or [])]
    assert any("page64_no_radix" in (d or "") for d in descs), descs
    assert not any((d or "") == f"[{_FW}] explore → regress on {_MODEL}/{_HW}" for d in descs), descs


# --- AgentX stays out of the cross-session KB ----------------------------------


def test_kb_amend_recipe_is_noop_under_agentx(native_benchmark, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)

    class _ForbiddenRecipeKB:
        def __getattr__(self, name: str):
            raise AssertionError(f"AgentX amend reached the recipe KB: {name}")

    coord.knowledge_plane = KnowledgePlane(recipe_kb=_ForbiddenRecipeKB())
    coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "must not reach the KB", "measured_impact": "+9%"})


def test_kb_amend_recipe_still_writes_without_agentx(tmp_path: Path, monkeypatch) -> None:
    """The gate must not cost the synthetic path its knowledge."""
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    coord = _make_coordinator(tmp_path)
    coord.recipe_journal.kb_amend_recipe(
        append_lesson={"statement": "synthetic still recorded", "measured_impact": "+1%"}
    )

    row = coord.recipe_kb.get_recipe(canonical_id=_expected_cid())
    assert row is not None, "the AgentX gate must not silence the synthetic path"
    statements = [x.get("statement") for x in (row.get("lessons") or [])]
    assert "synthetic still recorded" in statements


def test_finalize_recipe_is_skipped_under_agentx(native_benchmark, tmp_path, monkeypatch) -> None:
    """The sink the kb_amend_recipe gate cannot reach."""
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)

    def _must_not_run(*_a, **_k):
        raise AssertionError("AgentX finalize reached a Recipe KB sink")

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.remote_recipe.RemoteRecipeClient.from_env",
        _must_not_run,
    )
    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")
    assert out["status"] == "skipped"
    assert out["reason"] == "agentx_local_store_unsupported"


def test_finalize_recipe_reaches_agentx_remote_kb(native_benchmark, tmp_path, monkeypatch) -> None:
    """AgentX is isolated by its scheme and may now use the remote KB."""
    from types import SimpleNamespace

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)
    coord.knowledge_plane = SimpleNamespace(
        config=KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_URL": "https://kb.test",
                "KB_STORE_TOKEN": "token",
            }
        ),
        kb_disabled=False,
    )

    calls = []

    def _write(state, canonical_id, session_id, *, client):
        calls.append((canonical_id, state, session_id))
        return SimpleNamespace(
            status="written",
            reason="",
            canonical_id=canonical_id,
            session_id=session_id,
            primary_metric="interactivity_gain_pct",
            primary_value=20.0,
        )

    monkeypatch.setattr("hyperloom.orchestrator.knowledge.remote_recipe.RemoteRecipeClient.from_env", lambda: object())
    monkeypatch.setattr("hyperloom.orchestrator.knowledge.remote_recipe.write_final_remote_recipe", _write)
    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")
    assert out["status"] == "written"
    assert calls and calls[0][0].startswith("agentx:")
    from hyperloom.inference_optimizer.session.session_paths import (
        recipe_snapshot_audit_jsonl,
    )

    audit = json.loads(recipe_snapshot_audit_jsonl(coord.session_dir).read_text(encoding="utf-8"))
    assert audit["result"]["primary_metric"] == "interactivity_gain_pct"
    assert audit["result"]["primary_value"] == 20.0
    assert "best_throughput" not in audit["result"]


def test_agentx_remote_kb_never_receives_a_gain_measured_for_an_older_recipe(
    native_benchmark, tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)
    coord.knowledge_plane = SimpleNamespace(
        config=KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_URL": "https://kb.test",
                "KB_STORE_TOKEN": "token",
            }
        ),
        kb_disabled=False,
    )
    state = coord.shared_state
    state.current_best = {"name": "a+b", "extra_server_args": "--page-size 32", "tput": 120.0}
    state.optimization_stack = [
        {"action": "explore", "variant_name": "a"},
        {"action": "explore", "variant_name": "b"},
    ]
    state.cumulative_gain_validated = 20.0
    state.cumulative_gain_validated_stack_len = 1

    def _must_not_write():
        raise AssertionError("an unvalidated AgentX stack reached the remote KB")

    monkeypatch.setattr(
        "hyperloom.orchestrator.knowledge.remote_recipe.RemoteRecipeClient.from_env",
        _must_not_write,
    )

    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")

    assert out["reason"] == "unvalidated_recipe_stack"
    assert out["result_type"] == "unvalidated_recipe"


def test_agentx_remote_skip_audit_does_not_invent_throughput_metric(native_benchmark, tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)
    coord.knowledge_plane = SimpleNamespace(
        config=KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_URL": "https://kb.test",
                "KB_STORE_TOKEN": "token",
            }
        ),
        kb_disabled=False,
    )

    def _write(state, canonical_id, session_id, *, client):
        return SimpleNamespace(
            status="skipped",
            reason="no_new_keep_or_pure_warm_replay",
            canonical_id=canonical_id,
            session_id=session_id,
            primary_metric="",
            primary_value=0.0,
        )

    monkeypatch.setattr("hyperloom.orchestrator.knowledge.remote_recipe.RemoteRecipeClient.from_env", lambda: object())
    monkeypatch.setattr("hyperloom.orchestrator.knowledge.remote_recipe.write_final_remote_recipe", _write)
    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")
    assert out["status"] == "skipped"

    from hyperloom.inference_optimizer.session.session_paths import (
        recipe_snapshot_audit_jsonl,
    )

    audit = json.loads(recipe_snapshot_audit_jsonl(coord.session_dir).read_text(encoding="utf-8"))
    assert "primary_metric" not in audit["result"]
    assert "primary_value" not in audit["result"]
    assert "best_throughput" not in audit["result"]


def test_finalize_gate_honours_persisted_mode_without_the_env_var(tmp_path, monkeypatch) -> None:
    """benchmark_mode is stamped so the mode survives a restart; the gate should trust it rather than the shell that happens to be running."""
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    coord = _make_coordinator(tmp_path)
    coord.shared_state.benchmark_mode = "agentx"
    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")
    assert out["reason"] == "agentx_local_store_unsupported"


def test_finalize_recipe_still_runs_without_agentx(tmp_path, monkeypatch) -> None:
    """The gate must not cost the synthetic path its finalize."""
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    coord = _make_coordinator(tmp_path)
    out = coord.recipe_journal.finalize_recipe_and_journal(source="close")
    assert out.get("reason") != "agentx"


def test_t0_anchor_does_not_write_under_agentx(native_benchmark, tmp_path, monkeypatch) -> None:
    """The third Recipe sink, and the one the first two gates never saw."""
    from hyperloom.orchestrator.knowledge.recipe_kb_t0 import T0AnchorRequest, run_t0_anchor

    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(native_benchmark))
    coord = _make_coordinator(tmp_path)

    class _ForbiddenKB:
        mode = "local"

        def __getattr__(self, name: str):
            raise AssertionError(f"AgentX T0 anchor reached the recipe KB: {name}")

    run_t0_anchor(_ForbiddenKB(), coord.shared_state, T0AnchorRequest(workload=_MODEL, hw=_HW), session_dir=tmp_path)


def _seed_recipe_with_lesson(coord: Coordinator) -> dict:
    coord.recipe_kb.put_recipe(
        RecipeUpsert(
            canonical_id=_expected_cid(),
            model=_MODEL,
            hardware=_HW,
            framework_name=_FW,
            framework_version=_FWV,
            precision=_PREC,
            lessons=[{"statement": "kept", "measured_impact": "+1%"}],
        )
    )
    return coord.recipe_kb.get_recipe(canonical_id=_expected_cid())


def _fail_authoritative_read(coord: Coordinator) -> None:
    def _boom(**_kwargs):
        raise RuntimeError("authority read down")

    coord.recipe_kb.get_authoritative_recipe = _boom


def test_read_local_recipe_row_propagates_read_failure(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    _fail_authoritative_read(coord)
    with pytest.raises(RuntimeError, match="authority read down"):
        coord.recipe_journal.read_local_recipe_row()


def test_kb_amend_recipe_read_failure_leaves_recipe_untouched(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    before = _seed_recipe_with_lesson(coord)
    _fail_authoritative_read(coord)
    with pytest.raises(RuntimeError, match="authority read down"):
        coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "new", "measured_impact": "+2%"})
    assert coord.recipe_kb.get_recipe(canonical_id=_expected_cid()) == before


def test_kb_amend_recipe_propagates_put_failure(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)

    def _boom(_upsert):
        raise OSError("disk full")

    coord.recipe_kb.put_recipe = _boom
    with pytest.raises(OSError, match="disk full"):
        coord.recipe_journal.kb_amend_recipe(append_lesson={"statement": "new", "measured_impact": "+2%"})


def test_finalize_reports_error_when_authority_read_fails(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)
    before = _seed_recipe_with_lesson(coord)
    _fail_authoritative_read(coord)
    out = coord.recipe_journal.ensure_recipe_finalized(source="close")
    assert (out["status"], out["reason"], out["result_type"], out["error_class"]) == (
        "error",
        "RuntimeError",
        _close_out.RESULT_TRANSPORT_FAILED,
        "RuntimeError",
    )
    assert coord.recipe_kb.get_recipe(canonical_id=_expected_cid()) == before


def test_finalize_reports_error_when_put_recipe_fails(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)

    def _boom(_upsert):
        raise OSError("disk full")

    coord.recipe_kb.put_recipe = _boom
    out = coord.recipe_journal.ensure_recipe_finalized(source="close")
    assert (out["status"], out["reason"], out["result_type"], out["error_class"]) == (
        "error",
        "OSError",
        _close_out.RESULT_TRANSPORT_FAILED,
        "OSError",
    )


def _fact_write_for_three_kept_variants(coord) -> None:
    import asyncio
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.session.optimization_journal import Verdict

    task = SimpleNamespace(kind="explore", task_id="t-ledger", params={})
    variants = [
        {
            "variant_name": f"v{i}",
            "outcome": "KEEP",
            "fingerprint": f"fp{i}",
            "variant": {"extra_server_args": f"--flag-{i}"},
            "metrics": {"base_tput": 100.0, "tput": 110.0, "gain_pct": 10.0},
        }
        for i in range(3)
    ]
    asyncio.run(
        coord.recipe_journal.fact_write_hook(
            task=task,
            result={"round_id": "explore-001", "per_variant_outcomes": variants},
            verdict=Verdict.ADOPTED,
            adopted_variants={"fp0", "fp1", "fp2"},
        )
    )


@pytest.mark.parametrize(
    ("method", "failure"),
    [
        ("put_recipe", OSError("disk full")),
        ("get_authoritative_recipe", LocalRecipeStoreError("recipe row is corrupt")),
        ("get_authoritative_recipe", ValueError("not valid json")),
    ],
    ids=["write-os-error", "read-store-error", "read-value-error"],
)
def test_fact_write_hook_records_every_variant_when_the_recipe_store_fails(
    tmp_path: Path, caplog, method, failure
) -> None:
    """A cross-session KB failure must not cost the session its later variants' rows."""
    coord = _make_coordinator(tmp_path)

    def _boom(*_args, **_kwargs):
        raise failure

    setattr(coord.recipe_kb, method, _boom)

    _fact_write_for_three_kept_variants(coord)

    assert [row["variant_name"] for row in coord.shared_state.attempts] == ["v0", "v1", "v2"]
    assert [entry.variant_name for entry in coord.recipe_journal.ensure_journal().entries] == ["v0", "v1", "v2"]
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [
        f"recipe KB write for task t-ledger variant v{i} failed: {type(failure).__name__}: {failure}" for i in range(3)
    ]


def test_fact_write_hook_lets_an_unexpected_recipe_store_error_propagate(tmp_path: Path) -> None:
    coord = _make_coordinator(tmp_path)

    def _boom(_upsert):
        raise TypeError("put_recipe signature drifted")

    coord.recipe_kb.put_recipe = _boom

    with pytest.raises(TypeError, match="put_recipe signature drifted"):
        _fact_write_for_three_kept_variants(coord)
