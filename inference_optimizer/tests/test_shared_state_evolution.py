# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 §3.10 — SharedState evolution / migration tests.

Covers KB_design/3.10_shared_state_evolution/README.md acceptance
criteria:

* Inv-10.1 — fact-layer fields survive v0.6 → v0.8 migration unchanged.
* Inv-10.2 — Coordinator is the sole writer for §4.1 new fields (the
  PolicyGate ``CORE_STATE_FIELDS`` denial check enforces this).
* Inv-10.3 — migration is idempotent: ``from_dict(from_dict(x))`` ≡
  ``from_dict(x)`` at the level of serialized JSON.
* §5.1 — top-level ``schema_version`` field; v0.6 absence → 1, v0.8
  default → 2.
* §5.3 — ``--migration-mode={strict,lenient}`` strictness; ``--reset-state``
  backs up the legacy file.
* §9 — fresh v0.8 session writes ``schema_version=2`` to state.json.
"""

from __future__ import annotations

import json
import logging

import pytest

from inference_optimizer.orchestrator.shared_state import (
    LATEST_STATE_SCHEMA_VERSION,
    SharedState,
)


# ===========================================================================
# 1. schema_version surface
# ===========================================================================
def test_fresh_session_has_latest_schema_version():
    """KB_design §3.10 §5.1 — fresh SharedState carries the current
    schema version so the next save imprints it on state.json."""
    s = SharedState()
    assert s.schema_version == LATEST_STATE_SCHEMA_VERSION
    assert LATEST_STATE_SCHEMA_VERSION >= 2


def test_save_writes_schema_version_to_state_json(tmp_path):
    """KB_design §3.10 §9 — top-level ``schema_version=2`` visible in
    a fresh state.json."""
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
    """KB_design §3.10 §5.1 — a v0.6 state.json has no
    ``schema_version`` field. Loading bumps it to the v0.8 default."""
    sd = tmp_path / "session"
    sd.mkdir()
    legacy = {
        "session_id": "legacy-sid",
        "baseline_tput": 800.0,
        "current_best": {"variant_name": "warm-mla", "tput": 880.0},
        "cumulative_gain": 10.0,
        "optimization_stack": [],
        # Various v0.6 cruft we expect to drop.
        "action_scores": {"backends": {"base_score": 5.0}},
        "cooldown_until_tick": {"backends": 12},
    }
    (sd / "state.json").write_text(json.dumps(legacy))
    loaded = SharedState.load_or_init(sd)
    assert loaded.schema_version == LATEST_STATE_SCHEMA_VERSION


# ===========================================================================
# 2. Inv-10.1 — fact-layer survives migration unchanged
# ===========================================================================
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
    "cumulative_gain": 17.5,
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
    """Inv-10.1 — baseline / current_best / cumulative_gain /
    optimization_stack / gain_per_stack_entry are bit-equal across
    the v0.6 → v0.8 migration."""
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))
    loaded = SharedState.load_or_init(sd)
    for key, expected in _FACT_LAYER_PAYLOAD.items():
        actual = getattr(loaded, key)
        assert actual == expected, (
            f"fact-layer field {key!r} drifted across migration "
            f"(was {expected!r}, now {actual!r})"
        )


def test_fact_layer_md5_matches_post_save(tmp_path):
    """Inv-10.1 stronger form — round-trip through migration +
    persistence keeps the fact-layer projection byte-identical (we
    compare a sorted-key JSON projection so ordering / whitespace
    don't trip the test)."""
    import hashlib
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))

    def _fact_md5(state: SharedState) -> str:
        projection = {k: getattr(state, k) for k in _FACT_LAYER_PAYLOAD}
        return hashlib.md5(
            json.dumps(projection, sort_keys=True).encode("utf-8")
        ).hexdigest()

    loaded = SharedState.load_or_init(sd)
    md5_before = _fact_md5(loaded)
    loaded.save(sd)
    reloaded = SharedState.load_or_init(sd)
    md5_after = _fact_md5(reloaded)
    assert md5_before == md5_after, (
        "fact-layer md5 changed across migration + save round-trip"
    )


# ===========================================================================
# 3. Inv-10.3 — migration idempotence
# ===========================================================================
def test_migration_is_idempotent(tmp_path):
    """Inv-10.3 — re-loading a state.json that has already been
    migrated produces the identical SharedState (modulo the same
    schema_version=2 tag)."""
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    payload["action_scores"] = {"backends": {"base_score": 5.0}}
    (sd / "state.json").write_text(json.dumps(payload))
    first = SharedState.load_or_init(sd)
    first.save(sd)
    second = SharedState.load_or_init(sd)
    third = SharedState.load_or_init(sd)
    # Three loads → identical core projection.
    snap1 = {k: getattr(second, k) for k in _FACT_LAYER_PAYLOAD}
    snap2 = {k: getattr(third, k) for k in _FACT_LAYER_PAYLOAD}
    assert snap1 == snap2
    assert second.schema_version == third.schema_version == LATEST_STATE_SCHEMA_VERSION


def test_v08_payload_short_circuits_migration(monkeypatch, caplog):
    """A v0.8 payload (schema_version == LATEST) skips the migration
    log line entirely."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MIGRATION_MODE", raising=False)
    payload = {
        "schema_version": LATEST_STATE_SCHEMA_VERSION,
        "session_id": "fresh-v08",
        "baseline_tput": 100.0,
    }
    with caplog.at_level(logging.INFO,
                          logger="inference_optimizer.orchestrator.shared_state"):
        SharedState.from_dict(payload)
    migrated = [
        r for r in caplog.records
        if "v0.8 §3.10: state.json migrated" in r.getMessage()
    ]
    assert migrated == [], "fresh v0.8 payload should not log a migration line"


# ===========================================================================
# 4. Migration log content
# ===========================================================================
def test_v06_migration_log_lists_scoreboard_drop(monkeypatch, caplog):
    """A v0.6 payload with action_scores produces a log line that
    names the §3.9 drop + the migrated schema_version."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MIGRATION_MODE", raising=False)
    payload = {
        "session_id": "legacy",
        "baseline_tput": 100.0,
        "action_scores": {"backends": {"base_score": 5.0}},
    }
    with caplog.at_level(logging.INFO,
                          logger="inference_optimizer.orchestrator.shared_state"):
        SharedState.from_dict(payload)
    migrated = [
        r for r in caplog.records
        if "v0.8 §3.10: state.json migrated" in r.getMessage()
    ]
    assert migrated, "v0.6 payload should log a migration line"
    msg = migrated[0].getMessage()
    assert "v1 → v2" in msg or "v1 \u2192 v2" in msg
    assert "§3.9 dropped scoreboard fields" in msg


# ===========================================================================
# 5. Strict / lenient migration mode
# ===========================================================================
def test_lenient_mode_allows_continue_on_fact_field_drop(monkeypatch, caplog):
    """KB_design §3.10 §5.3 — lenient mode downgrades a fact-layer
    discrepancy to WARNING and continues."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MIGRATION_MODE", "lenient")
    # Manually craft an inner discrepancy by patching the dataclass
    # field set the loader filters against. We monkey-patch
    # ``__dataclass_fields__`` to PRETEND ``baseline_tput`` is no longer
    # known — that's the only way to exercise the "raw has it,
    # filtered doesn't" branch without modifying the real schema.
    real_fields = SharedState.__dataclass_fields__
    fake_fields = {k: v for k, v in real_fields.items() if k != "baseline_tput"}
    monkeypatch.setattr(SharedState, "__dataclass_fields__", fake_fields)
    payload = {
        "session_id": "legacy",
        "baseline_tput": 100.0,  # field will be filtered out
    }
    with caplog.at_level(logging.WARNING,
                          logger="inference_optimizer.orchestrator.shared_state"):
        loaded = SharedState.from_dict(payload)
    assert loaded.session_id == "legacy"
    warned = [
        r for r in caplog.records
        if "Inv-10.1 violation" in r.getMessage()
    ]
    assert warned, "lenient mode should still log a WARNING about the drop"


def test_strict_mode_raises_on_fact_field_drop(monkeypatch):
    """KB_design §3.10 §5.3 — strict mode raises a ValueError when
    a fact-layer field would be silently lost."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MIGRATION_MODE", raising=False)
    real_fields = SharedState.__dataclass_fields__
    fake_fields = {k: v for k, v in real_fields.items() if k != "baseline_tput"}
    monkeypatch.setattr(SharedState, "__dataclass_fields__", fake_fields)
    payload = {
        "session_id": "legacy",
        "baseline_tput": 100.0,
    }
    with pytest.raises(ValueError, match="strict migration failed"):
        SharedState.from_dict(payload)


# ===========================================================================
# 6. --reset-state behavior
# ===========================================================================
def test_reset_state_backs_up_state_json(tmp_path):
    """KB_design §3.10 §5.3 bottom — ``--reset-state`` renames the
    existing state.json so the next ``load_or_init`` starts blank."""
    from inference_optimizer.cli import _reset_state_file
    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    (sd / "state.json").write_text(json.dumps(payload))
    _reset_state_file(sd)
    # state.json is gone.
    assert not (sd / "state.json").exists()
    # A pre-reset backup is present.
    backups = [p for p in sd.iterdir() if p.name.startswith("state.json.preReset.")]
    assert len(backups) == 1, "exactly one pre-reset backup expected"
    # Fresh load starts blank.
    loaded = SharedState.load_or_init(sd)
    assert loaded.baseline_tput == 0.0
    assert loaded.session_id == ""
    assert loaded.schema_version == LATEST_STATE_SCHEMA_VERSION


def test_reset_state_is_safe_when_no_state_file(tmp_path):
    from inference_optimizer.cli import _reset_state_file
    sd = tmp_path / "session"
    sd.mkdir()
    _reset_state_file(sd)
    assert not (sd / "state.json").exists()


# ===========================================================================
# 7. CLI flag wiring
# ===========================================================================
def test_cli_exposes_migration_mode_flag():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "optimize",
        "--model", "/tmp/dummy",
        "--migration-mode", "lenient",
    ])
    assert args.migration_mode == "lenient"
    args2 = parser.parse_args([
        "optimize", "--model", "/tmp/dummy",
    ])
    assert args2.migration_mode in ("strict", "lenient")


def test_cli_rejects_unknown_migration_mode():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize", "--model", "/tmp/dummy",
            "--migration-mode", "ultra",
        ])


def test_cli_exposes_reset_state_flag():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "optimize", "--model", "/tmp/dummy", "--reset-state",
    ])
    assert args.reset_state is True
    args2 = parser.parse_args([
        "optimize", "--model", "/tmp/dummy",
    ])
    assert args2.reset_state is False


# ===========================================================================
# 8. Inv-10.2 — CORE_STATE_FIELDS blocks LLM update_state phase change
# ===========================================================================
def test_core_state_fields_contains_v08_new_additions():
    """KB_design §3.10 §6.2 — verify the v0.8 §4.1 new fields are in
    the CORE_STATE_FIELDS lock so an LLM ``update_state`` cannot
    overwrite them."""
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    must_be_locked = {
        "phase",
        "phase_started_ts",
        "phase_history",
        "phase_budget_pct",
        "cortex_session_id",
        "cortex_session_summary",
        "warm_start_recipe",
        "warm_start_pitfalls",
        "warm_start_lessons",
        "specialist_rounds",
        "specialist_domain_empty_streak",
        "research_lane_capacity",
        "stop_reason",
        "optimization_stack",
        "current_best",
    }
    missing = must_be_locked - CORE_STATE_FIELDS
    assert not missing, (
        f"v0.8 §3.10 requires these to be CORE: {sorted(missing)}"
    )


def test_policy_blocks_llm_phase_write():
    """KB_design §3.10 §9 acceptance: LLM ``update_state`` that tries
    to set ``phase=KERNEL`` must be denied."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import (
        PolicyDenied, PolicyGate,
    )
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"phase": "KERNEL"}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


def test_policy_blocks_llm_schema_version_write():
    """KB_design §3.10 §5.1 — an LLM cannot rewrite the migration
    breadcrumb to roll the state.json back to a v0.6 reader."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import (
        PolicyDenied, PolicyGate,
    )
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"schema_version": 1}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


def test_policy_blocks_llm_optimization_stack_write():
    """KB_design §3.10 §6.2 — Coordinator is the sole writer for the
    KEEP ledger. An LLM update_state with ``optimization_stack`` in
    its changes set must be denied."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import (
        PolicyDenied, PolicyGate,
    )
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"optimization_stack": []}},
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)


# ===========================================================================
# 9. KB_gaps/Gap-14 — search ledgers locked under CORE_STATE_FIELDS
# ===========================================================================
def test_search_ledgers_in_core_state_fields():
    """KB_design §3.10 §6.2 — the unified ``explore_search`` ledger is a
    Coordinator-only fact-layer write. Locking it as CORE closes the
    Inv-10.2 defense surface KB_gaps/Gap-14 flagged."""
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "explore_search" in CORE_STATE_FIELDS, (
        "'explore_search' must be in CORE_STATE_FIELDS so LLM "
        "update_state cannot rewrite the search ledger"
    )


@pytest.mark.parametrize("field_name", ["explore_search"])
def test_policy_blocks_llm_search_ledger_write(field_name):
    """LLM ``update_state{changes: {<ledger>: ...}}`` must surface a
    ``state_field`` denial."""
    from inference_optimizer.orchestrator.agent_role import (
        default_role_registry,
    )
    from inference_optimizer.protocol.intent import (
        Intent, IntentType,
    )
    from inference_optimizer.orchestrator.policy import (
        PolicyDenied, PolicyGate,
    )
    gate = PolicyGate(role_registry=default_role_registry())
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {field_name: {"tested": {}}}},
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "state_field"
