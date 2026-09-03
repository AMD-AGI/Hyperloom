# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SharedState evolution and migration tests (Inv-10.1/10.2/10.3)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from hyperloom.orchestrator.state.shared_state import (
    LATEST_STATE_SCHEMA_VERSION,
    SharedState,
)


# 1. schema_version surface
def test_fresh_session_has_latest_schema_version():
    """Fresh SharedState carries the current schema version."""
    s = SharedState()
    assert s.schema_version == LATEST_STATE_SCHEMA_VERSION
    assert LATEST_STATE_SCHEMA_VERSION >= 2


def test_save_writes_schema_version_to_state_json(tmp_path):
    """Top-level ``schema_version`` visible in a fresh state.json."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState()
    s.session_id = "fresh-sid"
    s.baseline_tput = 250.0
    s.save(sd)
    raw = json.loads((sd / "state.json").read_text())
    assert raw.get("schema_version") == LATEST_STATE_SCHEMA_VERSION
    assert raw.get("baseline_tput") == 250.0


def test_v06_state_without_schema_version_is_migrated(tmp_path):
    """A legacy state.json with no ``schema_version`` is bumped to the current default."""
    sd = tmp_path / "session"
    sd.mkdir()
    legacy = {
        "session_id": "legacy-sid",
        "baseline_tput": 800.0,
        "current_best": {"variant_name": "warm-mla", "tput": 880.0},
        "cumulative_gain_validated": 10.0,
        "optimization_stack": [],
        "action_scores": {"backends": {"base_score": 5.0}},
        "cooldown_until_tick": {"backends": 12},
    }
    (sd / "state.json").write_text(json.dumps(legacy))
    loaded = SharedState.load_or_init(sd)
    assert loaded.schema_version == LATEST_STATE_SCHEMA_VERSION


# 2. Inv-10.1 — fact-layer survives migration unchanged
_FACT_LAYER_PAYLOAD: dict = {
    "session_id": "legacy",
    "baseline_tput": 1234.5,
    "baseline_accuracy": 0.81,
    "baseline_failure_streak": 0,
    "current_best": {
        "variant_name": "bs_a_b_c",
        "tput": 1450.0,
        "extra_server_args": "--mla",
        "extra_envs": {"FOO": "bar"},
    },
    "cumulative_gain_validated": 15.0,
    "cumulative_gain_validated_ts": "2025-01-01T00:00:00+00:00",
    "cumulative_gain_validated_stack_len": 2,
    "optimization_stack": [
        {"action": "params", "variant_name": "v1", "tput": 1300.0},
        {"action": "backends", "variant_name": "bs_a_b_c", "tput": 1450.0},
    ],
    "gain_per_stack_entry": [5.4, 11.5],
}


def test_fact_layer_fields_survive_v06_resume(tmp_path):
    """Fact-layer fields are bit-equal across the legacy-to-current migration."""
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))
    loaded = SharedState.load_or_init(sd)
    for key, expected in _FACT_LAYER_PAYLOAD.items():
        actual = getattr(loaded, key)
        assert actual == expected, (
            f"fact-layer field {key!r} drifted across migration (was {expected!r}, now {actual!r})"
        )


def test_fact_layer_md5_matches_post_save(tmp_path):
    """A migration + save round-trip keeps the fact-layer projection byte-identical."""
    import hashlib

    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))

    def _fact_md5(state: SharedState) -> str:
        projection = {k: getattr(state, k) for k in _FACT_LAYER_PAYLOAD}
        return hashlib.md5(json.dumps(projection, sort_keys=True).encode("utf-8")).hexdigest()

    loaded = SharedState.load_or_init(sd)
    md5_before = _fact_md5(loaded)
    loaded.save(sd)
    reloaded = SharedState.load_or_init(sd)
    md5_after = _fact_md5(reloaded)
    assert md5_before == md5_after, "fact-layer md5 changed across migration + save round-trip"


def test_legacy_dict_with_unknown_scoreboard_keys_loads_and_stamps():
    """A v1-like dict with unknown scoreboard keys loads, drops the unknowns,
    and is stamped to the latest schema version (no flags, no raise)."""
    payload = {
        "session_id": "legacy",
        "baseline_tput": 100.0,
        "action_scores": {"backends": {"base_score": 5.0}},
        "cooldown_until_tick": {"backends": 12},
        "score_violation": {"params": 3},
        "not_a_real_field": 123,
    }
    loaded = SharedState.from_dict(payload)
    assert loaded.session_id == "legacy"
    assert loaded.baseline_tput == 100.0
    for dropped in ("action_scores", "cooldown_until_tick", "score_violation", "not_a_real_field"):
        assert not hasattr(loaded, dropped)
    assert loaded.schema_version == LATEST_STATE_SCHEMA_VERSION
    assert isinstance(loaded.explore_search, dict)
    for key in ("tested", "accepted", "rejected", "winners_history", "synergy_attempted"):
        assert key in loaded.explore_search


# 3. Inv-10.3 — migration idempotence
def test_migration_is_idempotent(tmp_path):
    """Re-loading an already-migrated state.json produces the identical SharedState."""
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))
    first = SharedState.load_or_init(sd)
    first.save(sd)
    second = SharedState.load_or_init(sd)
    third = SharedState.load_or_init(sd)
    snap1 = {k: getattr(second, k) for k in _FACT_LAYER_PAYLOAD}
    snap2 = {k: getattr(third, k) for k in _FACT_LAYER_PAYLOAD}
    assert snap1 == snap2
    assert second.schema_version == third.schema_version == LATEST_STATE_SCHEMA_VERSION


def test_v2_kernel_keep_populates_stable_task_and_pending_patch():
    state = SharedState()
    state.kernel_opt_task_attempts["legacy-task"] = {
        "kernel_id": "k002",
        "current_kernel_id": "k002",
        "stable_task_key": "legacy-task",
        "task_group_key": "legacy-task",
        "last_decision": "KEEP",
        "last_source_file": "/repo/operator.py",
        "last_artifact_path": "/artifacts/operator.py",
        "last_micro_speedup": 1.2,
    }
    assert state.kernel_opt_task_attempts["legacy-task"]["current_kernel_id"] == "k002"
    pending = state.pending_kernel_integration_records()
    assert len(pending) == 1
    assert pending[0]["task_key"] == "legacy-task"
    assert pending[0]["artifact_path"] == "/artifacts/operator.py"


def test_v2_ungrouped_kernel_record_opt_accumulates():
    state = SharedState()
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k001",
            "source_file": "/repo/operator.py",
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 1.0},
            "attempts": [],
        }
    )
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k001",
            "source_file": "/repo/operator.py",
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 1.1},
            "attempts": [],
        }
    )

    assert len(state.kernel_opt_task_attempts) == 1


# 4. --reset-state behavior
def test_reset_state_backs_up_state_json(tmp_path):
    """``--reset-state`` renames state.json so the next load starts blank."""
    import hyperloom.inference_optimizer.cli as optimizer_cli

    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    (sd / "state.json").write_text(json.dumps(payload))
    optimizer_cli._reset_state_file(sd)
    assert not (sd / "state.json").exists()
    backups = [p for p in sd.iterdir() if p.name.startswith("state.json.preReset.")]
    assert len(backups) == 1, "exactly one pre-reset backup expected"
    loaded = SharedState.load_or_init(sd)
    assert loaded.baseline_tput == 0.0
    assert loaded.session_id == ""
    assert loaded.schema_version == LATEST_STATE_SCHEMA_VERSION


def test_reset_state_is_safe_when_no_state_file(tmp_path):
    import hyperloom.inference_optimizer.cli as optimizer_cli

    sd = tmp_path / "session"
    sd.mkdir()
    optimizer_cli._reset_state_file(sd)
    assert not (sd / "state.json").exists()


# 5. CLI flag wiring
def test_cli_exposes_reset_state_flag():
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--model",
            "/tmp/dummy",
            "--reset-state",
        ]
    )
    assert args.reset_state is True
    args2 = parser.parse_args(
        [
            "optimize",
            "--model",
            "/tmp/dummy",
        ]
    )
    assert args2.reset_state is False


# 6. Inv-10.2 — CORE_STATE_FIELDS blocks LLM update_state phase change
def test_core_state_fields_contains_v08_new_additions():
    """The new fields are locked in CORE_STATE_FIELDS."""
    from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS

    must_be_locked = {
        "phase",
        "phase_started_ts",
        "phase_history",
        "phase_budget_pct",
        "recipe_kb_session_id",
        "warm_start_recipe",
        "warm_start_pitfalls",
        "warm_start_lessons",
        "specialist_rounds",
        "research_lane_capacity",
        "stop_reason",
        "optimization_stack",
        "current_best",
    }
    missing = must_be_locked - CORE_STATE_FIELDS
    assert not missing, f"v0.8 §3.10 requires these to be CORE: {sorted(missing)}"


def test_policy_blocks_llm_phase_write():
    """LLM ``update_state`` setting ``phase=KERNEL`` is denied."""
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.inference_optimizer.protocol.intent import (
        Intent,
        IntentType,
    )
    from hyperloom.orchestrator.policy.gate import (
        PolicyDenied,
        PolicyGate,
    )

    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"phase": "KERNEL"}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


def test_policy_blocks_llm_schema_version_write():
    """An LLM cannot rewrite the ``schema_version`` migration breadcrumb."""
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.inference_optimizer.protocol.intent import (
        Intent,
        IntentType,
    )
    from hyperloom.orchestrator.policy.gate import (
        PolicyDenied,
        PolicyGate,
    )

    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"schema_version": 1}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


def test_policy_blocks_llm_optimization_stack_write():
    """An LLM update_state with ``optimization_stack`` is denied (Coordinator-only)."""
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.inference_optimizer.protocol.intent import (
        Intent,
        IntentType,
    )
    from hyperloom.orchestrator.policy.gate import (
        PolicyDenied,
        PolicyGate,
    )

    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"optimization_stack": []}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


# Search ledgers locked under CORE_STATE_FIELDS.
def test_search_ledgers_in_core_state_fields():
    """The ``explore_search`` ledger is locked as CORE."""
    from hyperloom.orchestrator.policy.gate import CORE_STATE_FIELDS

    assert "explore_search" in CORE_STATE_FIELDS, (
        "'explore_search' must be in CORE_STATE_FIELDS so LLM update_state cannot rewrite the search ledger"
    )


def test_enablement_accepted_config_path_roundtrips(tmp_path):
    """enablement_accepted_config_path is persisted and reloaded correctly."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState()
    s.enablement.accepted_config_path = "/runs/specialist/t-spec-1/integrate_patch.with_envs.yaml"
    s.enablement.active_runtime = {"bin_path": "/attempt/bin", "venv_root": "/attempt/venv"}
    s.save(sd)
    loaded = SharedState.load_or_init(sd)
    assert loaded.enablement.accepted_config_path == "/runs/specialist/t-spec-1/integrate_patch.with_envs.yaml"
    assert loaded.enablement.active_runtime == {"bin_path": "/attempt/bin", "venv_root": "/attempt/venv"}


def test_v5_migration_prefers_the_current_spelling(tmp_path):
    """A half-migrated state carrying both spellings keeps the current one."""
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "framework_pr_discover_failures": 9,
                "framework_agent_discover_failures": 1,
            }
        )
    )

    assert SharedState.load_or_init(sd).framework_agent_discover_failures == 1


def test_class_constants_are_not_persisted_fields():
    """A constant must be ``ClassVar`` (or module-level), never a bare annotation.

    A bare annotation makes it a dataclass field, which puts a value that was
    never session state into every ``state.json``, accepts it back on load, and
    exposes it to ``apply_changes`` -- ``CORE_STATE_FIELDS`` locks the fields
    someone thought to lock, and nobody locks a constant. Naming is the only
    signal available here, so this keys on the SCREAMING_CASE convention the
    file already follows.
    """
    leaked = sorted(f.name for f in dataclasses.fields(SharedState) if f.name.isupper())
    assert not leaked, (
        f"constants declared as dataclass fields (annotate them ClassVar[...] or move them to module level): {leaked}"
    )


def test_the_profile_identity_keys_stay_off_disk(tmp_path):
    """The projection keys decide trace staleness; a stored copy must not.

    An existing session's state.json still carries the key from when it was a
    field, and ``__init__`` no longer accepts it, so loading one has to ignore
    the key rather than raise.
    """
    sd = tmp_path / "session"
    sd.mkdir()
    SharedState(session_id="s").save(sd)
    raw = json.loads((sd / "state.json").read_text())
    assert "PROFILE_WORKLOAD_IDENTITY_KEYS" not in raw

    # An older state.json carrying the key cannot reintroduce it either.
    (sd / "state.json").write_text(json.dumps({"PROFILE_WORKLOAD_IDENTITY_KEYS": ["framework"]}))
    loaded = SharedState.load_or_init(sd)
    assert loaded.PROFILE_WORKLOAD_IDENTITY_KEYS == (SharedState.PROFILE_WORKLOAD_IDENTITY_KEYS)
    assert loaded.apply_changes({"PROFILE_WORKLOAD_IDENTITY_KEYS": ["framework"]}, allow_core=False) == {}


def test_v6_prefers_the_current_spelling_when_both_are_present(tmp_path):
    """A mid-migration state holding both keys resolves to the current one."""
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "continue_kernel_after_gemm": False,
                "auto_kernel_opt_enabled": True,
            }
        )
    )

    assert SharedState.load_or_init(sd).auto_kernel_opt_enabled is True


def test_v4_nested_enablement_roundtrips(tmp_path):
    """A v4 state.json with nested enablement dict survives save/load_or_init."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState()
    s.enablement.launch_log = "mla_gluon requires batch_size=1"
    s.enablement.attempts = 3
    s.enablement.kept_patches = ["/p/a.patch", "/p/b.patch"]
    s.save(sd)
    raw = json.loads((sd / "state.json").read_text())
    assert isinstance(raw.get("enablement"), dict), "enablement must be nested in state.json"
    assert raw["enablement"]["launch_log"] == "mla_gluon requires batch_size=1"
    assert "enablement_launch_log" not in raw, "flat keys must not appear in v4 output"
    loaded = SharedState.load_or_init(sd)
    assert loaded.enablement.launch_log == "mla_gluon requires batch_size=1"
    assert loaded.enablement.attempts == 3
    assert loaded.enablement.kept_patches == ["/p/a.patch", "/p/b.patch"]


def test_to_dict_emits_nested_enablement():
    """to_dict() produces enablement as a nested dict, not flat keys."""
    s = SharedState()
    s.enablement.launch_log = "test"
    d = s.to_dict()
    assert isinstance(d.get("enablement"), dict)
    assert d["enablement"]["launch_log"] == "test"
    assert "enablement_launch_log" not in d


@pytest.mark.parametrize("field_name", ["explore_search"])
def test_policy_blocks_llm_search_ledger_write(field_name):
    """LLM ``update_state`` of a search ledger surfaces a ``state_field`` denial."""
    from hyperloom.orchestrator.roles.agent_role import (
        default_role_registry,
    )
    from hyperloom.inference_optimizer.protocol.intent import (
        Intent,
        IntentType,
    )
    from hyperloom.orchestrator.policy.gate import (
        PolicyDenied,
        PolicyGate,
    )

    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {field_name: {"tested": {}}}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "state_field"


def _applyback_evidence():
    return {
        "artifact_kind": "framework_applyback",
        "artifact_schema_version": 2,
        "validation_scope": "reference",
        "reference_correctness_passed": True,
        "reference_snr_db": 48.5,
        "integration_validation_required": True,
        "integration_validation_status": "pending",
        "commit": "a" * 40,
        "commit_ref": "refs/hyperloom/applyback/attempt-1",
        "builder_symbol": "build_fused_gemm_module",
        "changed_files": ["flydsl_kernel.py", "kernel.py"],
    }


def _record_applyback_keep(state, **verification_overrides):
    verification = {
        "micro_speedup": 1.6,
        "compile_passed": True,
        "correctness_passed": True,
        "correctness_source": "forge_rewrite_reference",
        "integration_validation_status": "pending",
        "framework_applyback": _applyback_evidence(),
        "best_backend": "forge",
        "best_artifact_path": "/artifacts/flydsl_kernel.py",
        "deploy_patch_path": "/artifacts/forge.patch",
        "deploy_repo_root": "/repo",
        "deploy_snapshot_dir": "/artifacts/snapshot",
    }
    verification.update(verification_overrides)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k007",
            "source_file": "/repo/fused_gemm.py",
            "task_group_id": "tg001",
            "task_group_key": "tg-fused-gemm",
            "proposal": {
                "decision": "KEEP",
                "reasons": ["framework apply-back reference-verified; framework E2E/accuracy deferred to integrate"],
            },
            "verification": verification,
        }
    )


def test_reference_verified_applyback_queues_with_its_provenance():
    state = SharedState()

    _record_applyback_keep(state)

    attempt = state.kernel_opt_attempts["k007"]
    assert attempt["last_correctness_source"] == "forge_rewrite_reference"
    assert attempt["last_integration_validation_status"] == "pending"
    assert attempt["last_framework_applyback"] == _applyback_evidence()

    assert state.last_kernel_opt["correctness_source"] == "forge_rewrite_reference"
    assert state.last_kernel_opt["integration_validation_status"] == "pending"
    assert state.last_kernel_opt["framework_applyback"]["commit_ref"] == ("refs/hyperloom/applyback/attempt-1")

    pending = state.pending_kernel_integration_records()
    assert len(pending) == 1
    record = pending[0]
    assert record["status"] == "pending"
    assert record["artifact_kind"] == "framework_applyback"
    assert record["integration_validation_status"] == "pending"
    assert record["correctness_source"] == "forge_rewrite_reference"
    assert record["framework_applyback"]["changed_files"] == [
        "flydsl_kernel.py",
        "kernel.py",
    ]


def test_a_plain_keep_queues_without_applyback_provenance():
    state = SharedState()

    _record_applyback_keep(
        state,
        correctness_source="report_scan",
        integration_validation_status="",
        framework_applyback={},
    )

    record = state.pending_kernel_integration_records()[0]
    assert record["artifact_kind"] == ""
    assert record["integration_validation_status"] == ""
    assert record["framework_applyback"] == {}


def test_serving_accuracy_settles_the_pending_applyback_verdict():
    state = SharedState()
    _record_applyback_keep(state)
    pending = state.pending_kernel_integration_records()[0]

    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k007",
            "integration_id": pending["integration_id"],
            "task_group_key": "tg-fused-gemm",
            "patch_path": "/artifacts/forge.patch",
            "target_file": "/repo/fused_gemm.py",
            "gain_pct": 4.2,
            "accuracy_pass": True,
            "artifact_kind": "framework_applyback",
            "integration_validation_status": "passed",
            "validation_tier": "integrate_e2e_accuracy",
        }
    )

    record = state.pending_kernel_integrations[pending["integration_id"]]
    assert record["status"] == "integrated"
    assert record["integration_validation_status"] == "passed"
    assert record["validation_tier"] == "integrate_e2e_accuracy"

    attempt = state.kernel_opt_attempts["k007"]
    assert attempt["integration_status"] == "integrated"
    assert attempt["last_integration_validation_status"] == "passed"
    assert attempt["validation_tier"] == "integrate_e2e_accuracy"


def test_a_plain_integrated_keep_records_no_applyback_verdict():
    state = SharedState()
    _record_applyback_keep(
        state,
        correctness_source="report_scan",
        integration_validation_status="",
        framework_applyback={},
    )
    pending = state.pending_kernel_integration_records()[0]

    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "KEEP",
            "kernel_id": "k007",
            "integration_id": pending["integration_id"],
            "task_group_key": "tg-fused-gemm",
            "patch_path": "/artifacts/forge.patch",
            "target_file": "/repo/fused_gemm.py",
            "gain_pct": 4.2,
            "accuracy_pass": True,
        }
    )

    record = state.pending_kernel_integrations[pending["integration_id"]]
    assert record["status"] == "integrated"
    assert record["integration_validation_status"] == ""
    assert "validation_tier" not in record


def test_bare_kernel_id_integrate_resolves_the_pending_applyback(tmp_path):
    """Orchestration may send only a kernel_id; the queue supplies the rest."""
    from hyperloom.orchestrator.kernel.request_handlers import (
        _fill_integrate_defaults_from_state,
    )

    sd = tmp_path / "session"
    sd.mkdir()
    state = SharedState()
    state.baseline_tput = 100.0
    state.baseline_config_path = "/configs/baseline.yaml"
    _record_applyback_keep(state)
    state.save(sd)

    resolved = _fill_integrate_defaults_from_state({"kernel_id": "k007"}, session_dir=sd)

    assert resolved["kernel_id"] == "k007"
    assert resolved["task_group_key"] == "tg-fused-gemm"
    assert resolved["artifact_kind"] == "framework_applyback"
    assert resolved["integration_validation_status"] == "pending"
    assert resolved["integration_id"]
    assert resolved["config_path"] == "/configs/baseline.yaml"
