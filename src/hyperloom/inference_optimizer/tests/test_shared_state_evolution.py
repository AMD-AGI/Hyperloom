# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SharedState save/load round trips, ``--reset-state``, and the kernel integration queue it persists."""

from __future__ import annotations

import dataclasses
import hashlib
import json

from hyperloom.orchestrator.kernel import kernel_decisions as kd
from hyperloom.orchestrator.state.shared_state import SharedState


_FACT_LAYER_PAYLOAD: dict = {
    "session_id": "seeded",
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


def test_fact_layer_md5_matches_post_save(tmp_path):
    """A load + save round trip keeps the fact-layer projection byte-identical."""
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "state.json").write_text(json.dumps(_FACT_LAYER_PAYLOAD))

    def _fact_md5(state: SharedState) -> str:
        projection = {k: getattr(state, k) for k in _FACT_LAYER_PAYLOAD}
        return hashlib.md5(json.dumps(projection, sort_keys=True).encode("utf-8")).hexdigest()

    loaded = SharedState.load_or_init(sd)
    assert {k: getattr(loaded, k) for k in _FACT_LAYER_PAYLOAD} == _FACT_LAYER_PAYLOAD
    md5_before = _fact_md5(loaded)
    loaded.save(sd)
    reloaded = SharedState.load_or_init(sd)
    md5_after = _fact_md5(reloaded)
    assert md5_before == md5_after, "fact-layer md5 changed across a load + save round trip"


# --reset-state behavior
def test_reset_state_backs_up_state_json(tmp_path):
    """``--reset-state`` backs up state.json so the next load starts blank, except for the session's framework."""
    import hyperloom.inference_optimizer.cli as optimizer_cli

    sd = tmp_path / "session"
    sd.mkdir()
    payload = dict(_FACT_LAYER_PAYLOAD)
    (sd / "state.json").write_text(json.dumps(payload))
    optimizer_cli._reset_state_file(sd, SharedState(framework="vllm"))
    backups = [p for p in sd.iterdir() if p.name.startswith("state.json.preReset.")]
    assert len(backups) == 1, "exactly one pre-reset backup expected"
    loaded = SharedState.load_or_init(sd)
    assert loaded.baseline_tput == 0.0
    assert loaded.session_id == ""
    assert loaded.framework == "vllm"


def test_reset_state_is_safe_when_no_state_file(tmp_path):
    import hyperloom.inference_optimizer.cli as optimizer_cli

    sd = tmp_path / "session"
    sd.mkdir()
    optimizer_cli._reset_state_file(sd, SharedState(framework="vllm"))
    assert not (sd / "state.json").exists()


# CLI flag wiring
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


def test_class_constants_are_not_persisted_fields():
    """A constant must be ``ClassVar`` (or module-level), never a bare annotation."""
    leaked = sorted(f.name for f in dataclasses.fields(SharedState) if f.name.isupper())
    assert not leaked, (
        f"constants declared as dataclass fields (annotate them ClassVar[...] or move them to module level): {leaked}"
    )


def test_the_profile_identity_keys_stay_off_disk(tmp_path):
    """The projection keys decide trace staleness; a stored copy must not."""
    sd = tmp_path / "session"
    sd.mkdir()
    SharedState(session_id="s").save(sd)
    raw = json.loads((sd / "state.json").read_text())
    assert "PROFILE_WORKLOAD_IDENTITY_KEYS" not in raw


def test_nested_enablement_roundtrips(tmp_path):
    """The nested enablement record survives save/load_or_init."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState()
    s.enablement.launch_log = "mla_gluon requires batch_size=1"
    s.enablement.kept_patches = ["/p/a.patch", "/p/b.patch"]
    s.save(sd)
    raw = json.loads((sd / "state.json").read_text())
    assert isinstance(raw.get("enablement"), dict), "enablement must be nested in state.json"
    assert raw["enablement"]["launch_log"] == "mla_gluon requires batch_size=1"
    assert "enablement_launch_log" not in raw, "flat keys must not appear in the saved state"
    loaded = SharedState.load_or_init(sd)
    assert loaded.enablement.launch_log == "mla_gluon requires batch_size=1"
    assert loaded.enablement.kept_patches == ["/p/a.patch", "/p/b.patch"]


def test_to_dict_emits_nested_enablement():
    """to_dict() produces enablement as a nested dict, not flat keys."""
    s = SharedState()
    s.enablement.launch_log = "test"
    d = s.to_dict()
    assert isinstance(d.get("enablement"), dict)
    assert d["enablement"]["launch_log"] == "test"
    assert "enablement_launch_log" not in d


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


def _record_applyback_keep(state, **overrides):
    """Put one apply-back KEEP on the integration queue."""
    record = {
        "integration_id": "kernel-integration:applyback-k007",
        "task_key": "tg-fused-gemm",
        "task_group_key": "tg-fused-gemm",
        "kernel_id": "k007",
        "source_file": "/repo/fused_gemm.py",
        "artifact_path": "/artifacts/flydsl_kernel.py",
        "artifact_bundle": {},
        "snapshot_dir": "/artifacts/snapshot",
        "deploy_patch_path": "/artifacts/forge.patch",
        "deploy_repo_root": "/repo",
        "micro_speedup": 1.6,
        "optimization_decision": "KEEP",
        "status": "pending",
        "correctness_source": "forge_rewrite_reference",
        "artifact_kind": "framework_applyback",
        "integration_validation_status": "pending",
        "framework_applyback": _applyback_evidence(),
        **overrides,
    }
    state.pending_kernel_integrations[record["integration_id"]] = record


def test_serving_accuracy_settles_the_pending_applyback_verdict():
    state = SharedState()
    _record_applyback_keep(state)
    pending = kd.pending_kernel_integration_records(state)[0]

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


def test_a_plain_integrated_keep_records_no_applyback_verdict():
    state = SharedState()
    _record_applyback_keep(
        state,
        correctness_source="report_scan",
        artifact_kind="",
        integration_validation_status="",
        framework_applyback={},
    )
    pending = kd.pending_kernel_integration_records(state)[0]

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
