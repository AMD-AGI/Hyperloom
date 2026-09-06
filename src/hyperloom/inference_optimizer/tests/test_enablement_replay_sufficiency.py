# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay sufficiency: the seven carried-over gaps and their fail-closed codes (R1b)."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.sessions import _build_attempt_summary
from hyperloom.orchestrator.enablement.recipe import (
    build_recipe_steps,
    classify_credential_class,
    detect_credential_channels,
    evaluate_replay_sufficiency,
    read_status,
)
from hyperloom.orchestrator.enablement.recipe.build_inputs import (
    ambient_closure,
    build_driver_for,
    build_input_record,
)
from hyperloom.orchestrator.enablement.recipe.projections import (
    project_accepted_config,
    project_launch_evidence,
    project_roots,
    project_runtime_provenance,
)
from hyperloom.orchestrator.enablement.recipe.setup_ledger import (
    build_execution_row,
    mark_round_disposition,
    setup_input_identity,
)

NO_FS = "/nonexistent-probe-root"


def _codes(decision):
    return [r["code"] for r in decision["reasons"]]


def _decide(enablement=None, section=None, delivered=None):
    enablement = dict(enablement or {})
    steps = build_recipe_steps(enablement, attempt_summary=_build_attempt_summary)
    return evaluate_replay_sufficiency(
        enablement,
        steps=steps,
        section=dict(section or {}),
        delivered_paths=delivered,
    )


def _row(cmd="pip install foo", *, seq=1, outcome="applied", task="r1", env=None, cwd="/tmp"):
    return build_execution_row(
        seq=seq,
        round_task_id=task,
        cmd_index=0,
        cmd=cmd,
        source="proposed",
        outcome=outcome,
        env=env,
        cwd=cwd,
        fs_root=NO_FS,
    )


def _accepted(rows, task="r1"):
    return mark_round_disposition(rows, round_task_id=task, disposition="kept", accepted=True)


# ---- 1. Occurrence identity survives the collapse (D4) ---------------------


def test_one_string_executed_twice_yields_two_steps_with_ordinals():
    cmd = "pip install foo"
    rows = _accepted([_row(cmd, seq=1, task="r1"), _row(cmd, seq=2, task="r2")], task="r2")
    state = {"setup_commands": [cmd], "setup_executions": rows}
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    assert [s["occurrence"] for s in steps] == [1, 2]
    assert state["setup_commands"] == [cmd]


def test_absent_ledger_falls_back_to_the_r1a_step_set():
    state = {"setup_commands": ["pip install a", "pip install b"]}
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    assert [s["occurrence"] for s in steps] == [None, None]
    assert "setup_occurrences_unknown" in _codes(_decide(state))


def test_empty_ledger_and_no_commands_raises_nothing_for_setup():
    assert "setup_occurrences_unknown" not in _codes(_decide({"setup_executions": [], "setup_commands": []}))


# ---- 11. Only applied executions become steps ------------------------------


def test_only_applied_executions_project_a_step():
    rows = [
        _row("pip install a", seq=1),
        _row("pip install b", seq=2, outcome="failed"),
        _row("pip install c", seq=3, outcome="skipped"),
    ]
    state = {"setup_commands": ["pip install a", "pip install b", "pip install c"], "setup_executions": rows}
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    assert [s["cmd"] for s in steps] == ["pip install a"]
    assert len(state["setup_executions"]) == 3


# ---- 2 / 12. Capture point vs validation point (D5) ------------------------


def test_command_reapplied_by_the_accepted_round_is_present_at_final_launch():
    cmd = "pip install foo"
    rows = _accepted([_row(cmd, seq=1, task="advanced"), _row(cmd, seq=2, task="kept")], task="kept")
    assert rows[1]["present_at_final_launch"] is True
    assert rows[0]["replayed_at_final_launch"] is True
    assert "setup_effect_outside_verified_launch" not in _codes(
        _decide({"setup_commands": [cmd], "setup_executions": rows})
    )


def test_command_from_a_discarded_round_raises_effect_outside_verified_launch():
    rows = mark_round_disposition(
        [_row("pip install stranded", seq=1, task="r1")],
        round_task_id="r1",
        disposition="apply_failed",
        accepted=False,
    )
    assert rows[0]["round_disposition"] == "apply_failed"
    assert rows[0]["present_at_final_launch"] is False
    assert "setup_effect_outside_verified_launch" in _codes(_decide({"setup_executions": rows}))


def test_ledger_survives_a_round_the_lane_never_reached():
    rows = [_row("pip install x", seq=1, task="never-reported")]
    assert rows[0]["round_disposition"] == "unreported"
    assert "setup_effect_outside_verified_launch" in _codes(_decide({"setup_executions": rows}))


def test_failed_occurrence_takes_no_succession_exemption():
    cmd = "pip install foo"
    rows = _accepted(
        [_row(cmd, seq=1, task="r1", outcome="failed"), _row(cmd, seq=2, task="kept")],
        task="kept",
    )
    assert rows[0]["present_at_final_launch"] is False
    assert rows[0]["replayed_at_final_launch"] is True
    assert "setup_effect_outside_verified_launch" in _codes(
        _decide({"setup_commands": [cmd], "setup_executions": rows})
    )


def test_durable_command_capped_out_of_the_accepted_round_is_reported():
    rows = _accepted([_row("pip install kept", seq=1, task="kept")], task="kept")
    state = {"setup_commands": ["pip install kept", "pip install capped"], "setup_executions": rows}
    assert "setup_ledger_truncated" in _codes(_decide(state))


def test_credentialed_command_is_sanitized_without_losing_its_digest():
    spellings = (
        "pip install --index-url https://user:token@host/simple pkg",
        "pip install --index-url 'https://user:token@host/simple' pkg",
        "pip install --index-url=https://user:token@host/simple pkg",
    )
    for cmd in spellings:
        rows = [_row(cmd, seq=i, outcome=o) for i, o in enumerate(("applied", "failed", "skipped"), start=1)]
        for row in rows:
            assert "token" not in row["cmd_sanitized"] and "host" not in row["cmd_sanitized"]
            assert row["credential_class"] == "index_url"
        assert len({row["cmd_digest"] for row in rows}) == 1


# ---- 3. Build join under concurrency (D6) ----------------------------------


def _attempt(task_id, **kw):
    row = {
        "ok": True,
        "attempt_root": f"/s/enablement/builds/{task_id}",
        "installed_versions": {"aiter_ref": "v1", "aiter_sha": "s" * 40, "arch": "gfx950"},
        "build_driver": "builtin_plan",
        "build_inputs": {
            "component": "aiter",
            "repo_url": "https://github.com/ROCm/aiter",
            "ref": "v1",
            "resolved_sha": "s" * 40,
            "gpu_arch": "gfx950",
            "max_jobs": 8,
            "torch_constraint_mode": "constraint_file",
            "env_digest": "sha256:e",
            "ambient_digest": "sha256:a",
            "build_command": None,
            "credential_class": None,
            "credential_channels": [],
        },
    }
    row.update(kw)
    return row


def _build_state(manifest):
    return {"build_manifest": manifest, "last_specialist_task_id": "probe"}


def test_build_binds_by_identity_not_by_position():
    state = _build_state([_attempt("bA"), _attempt("bB"), {"task_id": "bA", "probe_task_id": "probe"}])
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    assert steps[0]["build_task_id"] == "bA"
    assert "build_attempt_unjoined" not in _codes(_decide(state))


def test_unjoinable_build_step_raises_rather_than_binding_a_neighbour():
    state = _build_state([_attempt("bB"), {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_attempt_unjoined" in _codes(_decide(state))


# ---- 4. Incomplete build inputs (D6) ---------------------------------------


def test_complete_builtin_plan_raises_no_build_inputs_incomplete():
    state = _build_state([_attempt("bA"), {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_inputs_incomplete" not in _codes(_decide(state))


def test_each_missing_builtin_member_raises_build_inputs_incomplete():
    for member in ("component", "ref", "resolved_sha", "max_jobs", "env_digest", "ambient_digest"):
        row = _attempt("bA")
        row["build_inputs"][member] = "" if isinstance(row["build_inputs"][member], str) else 0
        state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
        assert "build_inputs_incomplete" in _codes(_decide(state)), member


def test_custom_command_build_is_incomplete_with_every_member_present():
    row = _attempt("bA", build_driver="custom_command")
    row["build_inputs"]["build_command"] = {"argv0": "bash", "digest": "sha256:d", "credential_class": None}
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_inputs_incomplete" in _codes(_decide(state))


def test_env_value_change_alone_changes_the_digest_and_emits_no_value():
    class _Action:
        component = "aiter"
        repo_url = ""
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 0
        torch_constraint_mode = "constraint_file"
        build_command = ()

        def __init__(self, envs):
            self.envs = envs

    one = build_input_record(_Action({"K": "1"}), installed_versions={}, ambient_env={}, fs_root=NO_FS)
    two = build_input_record(_Action({"K": "2"}), installed_versions={}, ambient_env={}, fs_root=NO_FS)
    assert one["env_digest"] != two["env_digest"]
    assert one["env_keys"] == two["env_keys"] == ["K"]
    # The driver's resolved value stands in where the action carried a blank.
    assert one["repo_url"] == "https://github.com/ROCm/aiter" and one["max_jobs"] == 8


def test_ambient_closure_tracks_build_effective_names_and_ignores_presentation():
    base = {"PATH": "/usr/bin", "ROCM_PATH": "/opt/rocm", "PIP_INDEX_URL": "https://a", "TERM": "xterm"}
    for changed in ("PATH", "ROCM_PATH", "PIP_INDEX_URL"):
        other = {**base, changed: "different"}
        assert ambient_closure(other, component="aiter") != ambient_closure(base, component="aiter")
    assert ambient_closure({**base, "TERM": "dumb"}, component="aiter") == ambient_closure(base, component="aiter")


def test_driver_overwritten_names_are_per_driver():
    env = {"PYTORCH_ROCM_ARCH": "gfx942", "HOME": "/root"}
    assert "PYTORCH_ROCM_ARCH" not in ambient_closure(env, component="aiter")
    assert "PYTORCH_ROCM_ARCH" in ambient_closure(env, component="sgl_kernel")
    assert "HOME" in ambient_closure(env, component="aiter")


def test_credentialed_repo_url_is_stripped_and_classified():
    class _Action:
        component = "aiter"
        repo_url = "https://user:token@github.com/org/repo"
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 8
        torch_constraint_mode = "constraint_file"
        build_command = ()
        envs: dict = {}

    record = build_input_record(_Action(), installed_versions={}, ambient_env={}, fs_root=NO_FS)
    assert record["repo_url"] == "https://github.com/org/repo"
    assert record["credential_class"] == "index_url"
    assert build_driver_for(_Action()) == "builtin_plan"


def test_build_command_travels_as_an_identity_never_as_text():
    class _Action:
        component = "aiter"
        repo_url = ""
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 8
        torch_constraint_mode = "constraint_file"
        build_command = ("bash", "-c", "pip install --index-url https://u:t@h/s pkg")
        envs: dict = {}

    record = build_input_record(_Action(), installed_versions={}, ambient_env={}, fs_root=NO_FS)
    identity = record["build_command"]
    assert identity["argv0"] == "bash" and identity["credential_class"] == "index_url"
    assert "u:t@h" not in str(identity)


# ---- 5. Launch-evidence sanitization (D2) ----------------------------------


def _evidence(**kw):
    evidence = {
        "schema_version": 1,
        "framework": "sglang",
        "recipe_digest": "sha256:cfg",
        "model_path": "/models/secret-model",
        "materialized_config_path": "/s/runs/materialized.yaml",
        "actual_server_log_path": "/s/runs/server.log",
        "requested_server_args": "--mem-fraction-static 0.9",
        "requested_server_env": {"HF_TOKEN": "shh", "SGLANG_X": "1"},
        "observed_server_launch_flags": "--mem-fraction-static 0.9",
        "observed_server_identity": {"model_path": "/models/secret-model", "tp_size": 8},
        "observed_model_binding": {"model_digest": "sha256:m", "tp": 8},
        "requested_model_digest": "sha256:m",
        "warm_reuse": {
            "reused_ready_server": False,
            "provenance": "fresh_or_unobserved",
            "source_server_log_path": "/s/x.log",
        },
    }
    evidence.update(kw)
    return evidence


def test_launch_evidence_projection_drops_paths_values_and_secret_env_names():
    projected, refused = project_launch_evidence(_evidence())
    assert refused is False
    flat = str(projected)
    assert "/models/secret-model" not in flat and "/s/runs" not in flat and "shh" not in flat
    assert projected["requested_server_env_keys"] == ["SGLANG_X"]
    assert projected["recipe_digest"] == "sha256:cfg"
    assert "model_path" not in projected["observed_server_identity"]
    assert projected["observed_server_identity"]["tp_size"] == 8


def test_evidence_with_no_observed_model_binding_is_activation_incomplete():
    projected, _ = project_launch_evidence(_evidence(observed_model_binding={}))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    assert "activation_incomplete" in _codes(_decide({}, section))


def test_requested_setting_no_observed_field_confirms_is_activation_incomplete():
    projected, _ = project_launch_evidence(_evidence(observed_server_launch_flags="", observed_server_identity={}))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    codes = _codes(_decide({}, section))
    assert "activation_incomplete" in codes


def test_observed_value_contradicting_the_requested_one_is_a_mismatch():
    projected, _ = project_launch_evidence(_evidence(observed_server_launch_flags="--mem-fraction-static 0.5"))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    assert "launch_evidence_mismatch" in _codes(_decide({}, section))


def test_model_digest_disagreement_is_a_mismatch():
    projected, _ = project_launch_evidence(_evidence(requested_model_digest="sha256:other"))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    assert "launch_evidence_mismatch" in _codes(_decide({}, section))


def test_untokenizable_argv_is_refused_rather_than_partially_represented():
    projected, refused = project_launch_evidence(_evidence(requested_server_args="--flag 'unterminated"))
    assert refused is True
    assert "requested_server_args" not in projected


# ---- 10. Absence is still not fabrication ----------------------------------


def test_empty_accepted_config_keys_are_not_emitted_as_defaults():
    assert project_accepted_config({"extra_envs": {}, "extra_server_args": "", "args_mode": ""}) == {}
    assert project_accepted_config(None) == {}


def test_five_accepted_config_keys_survive_the_projection():
    projected = project_accepted_config(
        {
            "extra_envs": {"A": "1"},
            "extra_server_args": "--x 1",
            "remove_args": ["--y"],
            "unset_envs": ["B"],
            "args_mode": "append",
        }
    )
    assert set(projected) == {"extra_envs", "extra_server_args", "remove_args", "unset_envs", "args_mode"}


def test_advanced_merge_source_is_activation_incomplete():
    section = {"accepted_config": {"config_path": "c.yaml"}, "accepted_config_source": "advanced_merge"}
    assert "activation_incomplete" in _codes(_decide({}, section))


# ---- 13. The runtime exports a rebuild path, not a location (D1) -----------


def _runtime_state(action):
    return {
        "active_runtime": {"python_path": "/attempt/venv/bin/python", "venv_root": "/attempt/venv"},
        "kept_stack_action": action,
    }


def test_runtime_provenance_carries_no_filesystem_path():
    provenance = project_runtime_provenance(
        _runtime_state({"acquisition_method": "editable_ref", "repo_url": "https://h/r", "ref": "main"})
    )
    assert "framework_python" in provenance["override_keys"]
    assert "/attempt" not in str(provenance)


def test_unpinned_acquisitions_require_a_runtime_rebuild():
    unpinned = (
        {"acquisition_method": "editable_ref", "repo_url": "https://h/r", "ref": "main"},
        {"acquisition_method": "wheel", "packages": ["sglang"]},
        {"acquisition_method": "wheel", "packages": ["sglang"], "resolved_packages": {"sglang": {"version": "1.0"}}},
    )
    for action in unpinned:
        state = _runtime_state(action)
        section = {"runtime_provenance": project_runtime_provenance(state)}
        assert "runtime_rebuild_required" in _codes(_decide(state, section)), action


def test_pinned_acquisition_needs_no_rebuild_note():
    for action in (
        {"acquisition_method": "editable_ref", "repo_url": "https://h/r", "ref": "main", "resolved_ref": "c" * 40},
        {
            "acquisition_method": "wheel",
            "packages": ["sglang"],
            "resolved_packages": {"sglang": {"version": "1.0", "artifact_digest": "sha256:w"}},
        },
    ):
        state = _runtime_state(action)
        section = {"runtime_provenance": project_runtime_provenance(state)}
        assert "runtime_rebuild_required" not in _codes(_decide(state, section)), action


def test_credentialed_acquisition_url_is_exported_stripped_with_its_class():
    provenance = project_runtime_provenance(
        _runtime_state({"acquisition_method": "editable_ref", "repo_url": "https://u:t@h/r", "ref": "main"})
    )
    assert provenance["acquisition"]["repo_url"] == "https://h/r"
    assert provenance["acquisition"]["credential_class"] == "index_url"


# ---- 6 / 14. Credential class and ambient channels (OD2) -------------------


def test_credential_classes_over_the_admitted_grammar():
    assert classify_credential_class("pip install --index-url https://user:token@host/simple foo") == "index_url"
    assert classify_credential_class("pip install git+https://user:token@host/repo@main") == "vcs_url"
    assert classify_credential_class("pip install --find-links https://u:t@h/links foo") == "find_links"
    assert classify_credential_class("conda install -c https://u:t@h/chan foo") == "channel"
    assert classify_credential_class("pip install foo") is None


def test_credential_class_records_no_userinfo_host_or_operand():
    found = classify_credential_class("pip install --index-url https://user:token@host/simple foo")
    assert "token" not in found and "host" not in found


def test_ambient_channels_are_names_only_and_block_replay():
    channels = detect_credential_channels({"PIP_INDEX_URL": "https://secret.host/simple"}, fs_root=NO_FS)
    assert channels == ["pip_index_env"]
    row = _row("pip install foo", env={"PIP_INDEX_URL": "https://secret.host/simple"})
    assert row["credential_class"] is None
    assert "credential_required" in _codes(_decide({"setup_executions": [row]}))
    assert "secret.host" not in str(row)


def test_no_channel_raises_nothing():
    assert detect_credential_channels({}, fs_root=NO_FS) == []
    row = _row("pip install foo", env={})
    assert "credential_required" not in _codes(_decide({"setup_executions": [row]}))


def test_clone_channels_are_classified_on_the_acquisition_path():
    assert detect_credential_channels({"SSH_AUTH_SOCK": "/tmp/sock"}, fs_root=NO_FS) == ["ssh_agent"]
    assert detect_credential_channels({"GIT_SSH_COMMAND": "ssh -i k"}, fs_root=NO_FS) == ["git_ssh_command"]


def test_a_git_config_helper_alone_classifies_nothing():
    """No config is read, so a helper named only in git config is invisible."""
    assert detect_credential_channels({"HOME": NO_FS}, fs_root=NO_FS) == []


# ---- 8. Root coverage and artifact payload (D3) ----------------------------


def _root(root_id="r1", anchor="framework_root", rel="", contributions=("patch_apply",), complete=True):
    return {
        "id": root_id,
        "kind": "framework_checkout",
        "contributions": list(contributions),
        "is_git": True,
        "base_sha": "a" * 40,
        "replay_target": {"anchor": anchor, "rel": rel},
    }


def _snapshot(root_id="r1", files=(("srt/a.py", "upsert"),), complete=True):
    return {
        "root_id": root_id,
        "schema_version": 2,
        "snapshot_ref": f"optimization_stack/enablement/{root_id}",
        "base_sha": "a" * 40,
        "provenance": "enablement_keep",
        "import_root": "python",
        "complete": complete,
        "files": [{"rel": rel, "op": op} for rel, op in files],
    }


def test_root_records_carry_no_absolute_path():
    projected = project_roots([{**_root(), "path": "/sgl-workspace/sglang"}])
    assert "path" not in projected[0] and "/sgl-workspace" not in str(projected)


def test_step_whose_root_is_unrecorded_raises_root_unidentified():
    state = {"kept_patches": ["/p/1.patch"], "framework_root": "/fr"}
    assert "root_unidentified" in _codes(_decide(state, {"roots": []}))


def test_contributing_root_bound_by_neither_resolver_raises_root_unidentified():
    state = {"kept_patches": ["/p/1.patch"], "framework_root": "/fr", "roots": [{**_root(), "path": "/fr"}]}
    section = {"roots": project_roots([{**_root(contributions=()), "path": "/fr"}])}
    assert "root_unidentified" in _codes(_decide(state, section))


def test_unmappable_anchor_and_colliding_anchors_raise_root_unmappable():
    unmappable = {"roots": project_roots([{**_root(anchor="unmappable"), "path": "/x"}])}
    assert "root_unmappable" in _codes(_decide({}, unmappable))
    collision = {
        "roots": project_roots(
            [
                {**_root("r1", anchor="site_packages", rel="pkg"), "path": "/a"},
                {**_root("r2", anchor="site_packages", rel="pkg"), "path": "/b"},
            ]
        )
    }
    assert "root_unmappable" in _codes(_decide({}, collision))


def test_listed_root_with_no_snapshot_entry_raises_source_snapshot_missing():
    section = {"roots": project_roots([{**_root(), "path": "/fr"}]), "source_snapshots": []}
    assert "source_snapshot_missing" in _codes(_decide({}, section))


def test_incomplete_snapshot_raises_source_snapshot_incomplete():
    section = {"roots": project_roots([{**_root(), "path": "/fr"}]), "source_snapshots": [_snapshot(complete=False)]}
    assert "source_snapshot_incomplete" in _codes(_decide({}, section))


def test_artifact_target_no_snapshot_captured_is_not_self_contained():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot()],
        "kept_artifacts": [{"target": "/fr/srt/b.py", "rel_target": "srt/b.py"}],
    }
    assert "artifact_not_self_contained" in _codes(_decide({}, section))


def test_declared_deletion_is_captured_as_a_deletion_and_raises_nothing():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot(files=(("srt/gone.py", "delete"),))],
        "accepted_stack_targets": {"r1": {"srt/gone.py": "delete"}},
    }
    assert "accepted_stack_not_launched" not in _codes(_decide({}, section))


def test_target_recorded_missing_raises_accepted_stack_not_launched():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot(files=(("srt/gone.py", "missing"),), complete=False)],
        "accepted_stack_targets": {"r1": {"srt/gone.py": "delete"}},
    }
    assert "accepted_stack_not_launched" in _codes(_decide({}, section))


def test_snapshot_missing_an_accepted_patch_target_raises_accepted_stack_not_launched():
    """A KEEP through a launch-only probe captures the base file, not the stack."""
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot(files=(("srt/other.py", "upsert"),))],
        "accepted_stack_targets": {"r1": {"srt/a.py": "upsert"}},
    }
    assert "accepted_stack_not_launched" in _codes(_decide({}, section))


# ---- 15. Environment closure at the KEEP (D6/D7) ---------------------------


def _sufficient_section():
    return {
        "accepted_config": {"extra_envs": {"A": "1"}, "config_path": "runs/materialized.yaml"},
        "accepted_config_source": "kept_bench",
        "launch_evidence": project_launch_evidence(_evidence())[0],
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot()],
        "accepted_stack_targets": {"r1": {"srt/a.py": "upsert"}},
        "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
        "installed_versions_at_keep": {"sglang": "0.4"},
    }


def _sufficient_state():
    cmd = "pip install foo"
    rows = _accepted([_row(cmd, seq=1, task="kept")], task="kept")
    return {
        "setup_commands": [cmd],
        "setup_executions": rows,
        "kept_patches": ["/p/1.patch"],
        "framework_root": "/fr",
        "roots": [{**_root(), "path": "/fr"}],
    }


def test_a_fully_recorded_enablement_is_sufficient():
    decision = _decide(_sufficient_state(), _sufficient_section())
    assert decision["status"] == "sufficient", decision["reasons"]
    assert decision["reasons"] == []


def test_absent_closure_and_assertions_fail_closed():
    section = {**_sufficient_section(), "environment_closure": None, "installed_versions_at_keep": {}}
    codes = _codes(_decide(_sufficient_state(), section))
    assert "environment_closure_absent" in codes and "assertions_not_at_keep" in codes


def test_non_python_installer_withholds_a_verified_closure():
    rows = _accepted([_row("apt-get install -y libfoo", seq=1, task="kept")], task="kept")
    state = {**_sufficient_state(), "setup_commands": ["apt-get install -y libfoo"], "setup_executions": rows}
    assert "closure_scope_incomplete" in _codes(_decide(state, _sufficient_section()))


# ---- 9. Fail-closed sufficiency (§3.1) -------------------------------------


def test_every_reason_carries_a_blocks_and_a_value_free_scope():
    decision = _decide({"kept_patches": ["/p/1.patch"], "framework_root": "/fr"}, {})
    assert decision["status"] == "insufficient"
    for reason in decision["reasons"]:
        assert reason["blocks"] in ("replay", "assertion_validation", "both")
        assert "/" not in reason["scope"] or reason["scope"].startswith("step[")


def test_absent_decision_reads_as_insufficient():
    assert read_status({})["status"] == "insufficient"
    assert read_status({})["reasons"][0]["code"] == "not_evaluated"


def test_unrecognized_code_is_itself_insufficient():
    forged = {"replay_sufficiency": {"status": "sufficient", "reasons": [{"code": "invented", "blocks": "replay"}]}}
    assert read_status(forged)["status"] == "insufficient"


# ---- B45: setup input identity for mutable and local inputs ----------------


def test_local_file_install_without_identity_blocks_replay(tmp_path):
    row = _row("pip install ./private.whl", cwd=tmp_path)
    assert row["unresolved_inputs"] == ["local_file"]
    assert "setup_inputs_incomplete" in _codes(_decide({"setup_executions": [row]}))


def test_local_file_install_with_the_payload_present_is_identified(tmp_path):
    (tmp_path / "private.whl").write_bytes(b"wheel-bytes")
    identities, unresolved = setup_input_identity("pip install ./private.whl", cwd=tmp_path)
    assert unresolved == []
    assert identities[0]["kind"] == "local_file" and len(identities[0]["sha256"]) == 64


def test_requirements_file_install_is_identified_or_blocks_replay(tmp_path):
    row = _row("pip install -r requirements.txt", cwd=tmp_path)
    assert row["unresolved_inputs"] == ["requirements_file"]
    assert "setup_inputs_incomplete" in _codes(_decide({"setup_executions": [row]}))
    (tmp_path / "requirements.txt").write_text("foo==1.0\n", encoding="utf-8")
    identities, unresolved = setup_input_identity("pip install -r requirements.txt", cwd=tmp_path)
    assert unresolved == [] and identities[0]["kind"] == "requirements_file"


def test_moving_vcs_ref_install_blocks_replay(tmp_path):
    row = _row("pip install git+https://host/repo@main", cwd=tmp_path)
    assert row["unresolved_inputs"] == ["vcs_ref"]
    assert "setup_inputs_incomplete" in _codes(_decide({"setup_executions": [row]}))


def test_commit_pinned_vcs_ref_is_identified(tmp_path):
    identities, unresolved = setup_input_identity(f"pip install git+https://host/repo@{'a' * 40}", cwd=tmp_path)
    assert unresolved == [] and identities[0]["resolved_ref"] == "a" * 40


def test_plain_package_spec_needs_no_input_identity(tmp_path):
    assert setup_input_identity("pip install foo==1.0", cwd=tmp_path) == ([], [])


# ---- B43: portable delivery contract ---------------------------------------


def _delivery_section():
    section = _sufficient_section()
    section["accepted_config"]["config_path"] = "runs/materialized.yaml"
    return section


def _delivered_everything():
    return ["runs/materialized.yaml", "optimization_stack/enablement/r1", "/p/1.patch"]


def test_fully_packaged_delivery_stays_sufficient():
    decision = _decide(_sufficient_state(), _delivery_section(), delivered=_delivered_everything())
    assert decision["status"] == "sufficient", decision["reasons"]


def test_unpackaged_snapshot_flips_the_delivery_to_insufficient():
    delivered = [p for p in _delivered_everything() if "optimization_stack" not in p]
    decision = _decide(_sufficient_state(), _delivery_section(), delivered=delivered)
    assert decision["status"] == "insufficient"
    assert "source_snapshot_missing" in _codes(decision)


def test_unpackaged_config_and_patch_are_not_self_contained():
    delivered = ["optimization_stack/enablement/r1"]
    codes = _codes(_decide(_sufficient_state(), _delivery_section(), delivered=delivered))
    assert codes.count("artifact_not_self_contained") == 2


def test_no_delivery_assembled_leaves_the_contract_unapplied():
    assert _decide(_sufficient_state(), _delivery_section())["status"] == "sufficient"


# ---- 16. Existing R1a assertions stay green --------------------------------


def test_new_keys_do_not_change_the_r1a_projections():
    from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement as collect

    out = collect(
        Path("/tmp/sess"),
        {
            "enablement_attempts": 1,
            "enablement_kept_patches": ["/tmp/sess/patches/001.patch"],
            "enablement_setup_commands": ["pip install -e ."],
            "enablement_framework_root": "/sgl-workspace/sglang",
        },
        [],
    )
    assert out["kept_patches"] == ["patches/001.patch"]
    assert out["setup_commands"] == ["pip install -e ."]
    assert out["framework_root"] == "/sgl-workspace/sglang"
    assert out["replay_sufficiency"]["status"] == "insufficient"
