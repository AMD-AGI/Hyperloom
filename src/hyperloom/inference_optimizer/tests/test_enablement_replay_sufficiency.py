# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replay sufficiency: the seven carried-over gaps and their fail-closed codes (R1b)."""

from __future__ import annotations

import json

import pytest
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.sessions import _build_attempt_summary
from hyperloom.orchestrator.enablement.recipe import (
    build_recipe_steps,
    classify_credential_class,
    classify_credential_value,
    detect_credential_channels,
    evaluate_replay_sufficiency,
    read_status,
    sanitize_command_text,
)
from hyperloom.orchestrator.enablement.recipe.build_inputs import (
    ambient_closure,
    build_driver_for,
    build_input_record,
)
from hyperloom.orchestrator.enablement.recipe.credentials import strip_url_userinfo, url_userinfo
from hyperloom.orchestrator.enablement.recipe.projections import (
    project_accepted_config,
    project_launch_evidence,
    project_roots,
    project_runtime_provenance,
)
from hyperloom.orchestrator.enablement.recipe.steps import command_digest
from hyperloom.orchestrator.enablement.recipe.sufficiency import REASON_BLOCKS, _BUILTIN_REQUIRED_INPUTS
from hyperloom.orchestrator.enablement.recipe.setup_ledger import (
    build_execution_row,
    mark_round_disposition,
)
from hyperloom.orchestrator.framework import targeted_build
from hyperloom.orchestrator.framework.build_actions import TargetedBuildAction

NO_FS = "/nonexistent-probe-root"


def _codes(decision):
    return [r["code"] for r in decision["reasons"]]


def _decide(enablement=None, section=None):
    enablement = dict(enablement or {})
    steps = build_recipe_steps(enablement, attempt_summary=_build_attempt_summary)
    return evaluate_replay_sufficiency(
        enablement,
        steps=steps,
        section=dict(section or {}),
    )


def _row(cmd="pip install foo", *, seq=1, outcome="applied", task="r1", env=None):
    return build_execution_row(
        seq=seq,
        round_task_id=task,
        cmd_index=0,
        cmd=cmd,
        source="proposed",
        outcome=outcome,
        env=env,
        fs_root=NO_FS,
    )


def _accepted(rows, task="r1"):
    return mark_round_disposition(rows, round_task_id=task, disposition="kept", accepted=True)


def test_an_outcome_outside_the_recorded_vocabulary_is_refused():
    """The ledger's rules read the outcome, so an unknown one is a silent gap."""
    with pytest.raises(ValueError):
        _row(outcome="partially_applied")


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


def test_command_reapplied_by_the_accepted_round_is_present_at_final_launch():
    cmd = "pip install foo"
    rows = _accepted([_row(cmd, seq=1, task="advanced"), _row(cmd, seq=2, task="kept")], task="kept")
    assert rows[1]["present_at_final_launch"] is True
    assert rows[0]["replayed_at_final_launch"] is True
    assert "setup_effect_outside_verified_launch" not in _codes(
        _decide({"setup_commands": [cmd], "setup_executions": rows})
    )


def test_a_later_keep_takes_presence_from_the_earlier_one():
    """Only one round terminates the lane, so only its rows may claim presence."""
    cmd = "pip install foo"
    first = _accepted([_row(cmd, seq=1, task="t1")], task="t1")
    assert first[0]["present_at_final_launch"] is True

    second = mark_round_disposition(
        [*first, _row("pip install bar", seq=2, task="t2")],
        round_task_id="t2",
        disposition="kept",
        accepted=True,
    )
    by_seq = {row["seq"]: row for row in second}
    assert by_seq[1]["present_at_final_launch"] is False
    assert by_seq[2]["present_at_final_launch"] is True


def test_a_command_only_the_earlier_keep_ran_is_capped_out_of_the_final_one():
    """The stale flag used to answer for it, so truncation could never fire."""
    first = _accepted([_row("pip install foo", seq=1, task="t1")], task="t1")
    rows = mark_round_disposition(
        [*first, _row("pip install bar", seq=2, task="t2")],
        round_task_id="t2",
        disposition="kept",
        accepted=True,
    )
    codes = _codes(_decide({"setup_commands": ["pip install foo", "pip install bar"], "setup_executions": rows}))
    assert "setup_ledger_truncated" in codes


@pytest.mark.parametrize("disposition", ["apply_failed", "no_patches", "reverted"])
def test_command_from_a_discarded_round_raises_effect_outside_verified_launch(disposition):
    rows = mark_round_disposition(
        [_row("pip install stranded", seq=1, task="r1")],
        round_task_id="r1",
        disposition=disposition,
        accepted=False,
    )
    assert rows[0]["round_disposition"] == disposition
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


def test_a_command_the_accepted_round_failed_was_not_capped_out_of_it():
    """A row of that round is a command the replay reached, whatever it returned."""
    cmd = "pip install foo"
    rows = _accepted(
        [_row(cmd, seq=1, task="kept"), _row("pip install bar", seq=2, outcome="failed", task="kept")],
        task="kept",
    )
    codes = _codes(_decide({"setup_commands": [cmd, "pip install bar"], "setup_executions": rows}))
    assert "setup_ledger_truncated" not in codes


def test_a_capped_command_is_reported_when_the_accepted_round_only_failed():
    """No row of that round is present, yet the round is still the accepted one."""
    rows = _accepted([_row("pip install foo", seq=1, outcome="failed", task="kept")], task="kept")
    assert all(row["present_at_final_launch"] is False for row in rows)
    state = {"setup_commands": ["pip install foo", "pip install capped"], "setup_executions": rows}
    assert "setup_ledger_truncated" in _codes(_decide(state))


def test_a_round_that_only_failed_still_reaches_its_own_commands():
    rows = _accepted([_row("pip install foo", seq=1, outcome="failed", task="kept")], task="kept")
    state = {"setup_commands": ["pip install foo"], "setup_executions": rows}
    assert "setup_ledger_truncated" not in _codes(_decide(state))


def test_a_durable_command_no_row_of_the_accepted_round_reached_is_truncated():
    cmd = "pip install foo"
    rows = _accepted([_row(cmd, seq=1, task="kept")], task="kept")
    codes = _codes(_decide({"setup_commands": [cmd, "pip install capped"], "setup_executions": rows}))
    assert "setup_ledger_truncated" in codes


def test_a_failed_non_python_installer_scopes_nothing_of_its_own():
    """The code names a completed mutation; a failed one is already refused as an
    effect outside the verified launch."""
    rows = _accepted([_row("apt-get install -y libfoo", seq=1, outcome="failed", task="kept")], task="kept")
    codes = _codes(_decide({"setup_executions": rows}))
    assert "closure_scope_incomplete" not in codes
    assert "setup_effect_outside_verified_launch" in codes


def test_a_skipped_non_python_installer_scopes_nothing():
    rows = _accepted([_row("apt-get install -y libfoo", seq=1, outcome="skipped", task="kept")], task="kept")
    assert "closure_scope_incomplete" not in _codes(_decide({"setup_executions": rows}))


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install --index-url https://user:token@host/simple pkg",
        "pip install --index-url=https://user:token@host/simple pkg",
        "pip install --index-url 'https://user:token@host/simple' pkg",
    ],
)
def test_credentialed_command_is_sanitized_without_losing_its_digest(cmd):
    """Sanitization is of the recorded text only: the digest that counts
    occurrences is over the verbatim string, so the three spellings the
    allowlist admits equally still collapse onto one command."""
    rows = [_row(cmd, seq=i, outcome=outcome) for i, outcome in enumerate(("applied", "failed", "skipped"), start=1)]
    for row in rows:
        assert "user:token" not in row["cmd_sanitized"] and "host" not in row["cmd_sanitized"]
        assert row["cmd_digest"] == command_digest(cmd)
    assert len({row["cmd_digest"] for row in rows}) == 1
    assert {row["credential_class"] for row in rows} == {"index_url"}


def _attempt(task_id, **kw):
    row = {
        "ok": True,
        "task_id": task_id,
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
            "ambient_keys": ["HOME", "PATH", "ROCM_PATH"],
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


def test_routing_merged_into_the_attempt_row_still_joins():
    """Production leaves one row: routing merges its fields into the attempt row.

    Recognizing a sentinel by the absence of an outcome skipped exactly that row,
    so every executed build projected as unjoined.
    """
    state = _build_state([_attempt("bA", probe_task_id="probe")])
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    assert steps[0]["build_task_id"] == "bA" and steps[0]["build_driver"] == "builtin_plan"
    assert "build_attempt_unjoined" not in _codes(_decide(state))


def test_unjoinable_build_step_raises_rather_than_binding_a_neighbour():
    state = _build_state([_attempt("bB"), {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_attempt_unjoined" in _codes(_decide(state))


def test_complete_builtin_plan_raises_no_build_inputs_incomplete():
    state = _build_state([_attempt("bA"), {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_inputs_incomplete" not in _codes(_decide(state))


@pytest.mark.parametrize("member", _BUILTIN_REQUIRED_INPUTS)
def test_each_missing_builtin_member_raises_build_inputs_incomplete(member):
    row = _attempt("bA")
    # Indexed, not fetched with a default: a required member the fixture never
    # carried would otherwise pass this test while proving nothing.
    value = row["build_inputs"][member]
    row["build_inputs"][member] = "" if isinstance(value, str) else 0
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_inputs_incomplete" in _codes(_decide(state))


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
    # The digest separates them while neither value travels.
    assert '"1"' not in json.dumps(one) and '"2"' not in json.dumps(two)
    # The driver's resolved value stands in where the action carried a blank.
    assert one["repo_url"] == "https://github.com/ROCm/aiter" and one["max_jobs"] == 8


@pytest.mark.parametrize(
    ("component", "default_prefix", "default_jobs"),
    [("aiter", "_AITER", 3), ("framework_ext", "_AITER", 3), ("sgl_kernel", "_SGLANG", 5), ("vllm_source", "_VLLM", 7)],
)
@pytest.mark.parametrize(("repo_url", "max_jobs"), [("", 0), ("   ", 0), ("https://example.com/override", 11)])
def test_build_inputs_follow_driver_defaults(monkeypatch, component, default_prefix, default_jobs, repo_url, max_jobs):
    default_repo = f"https://example.com/{component}"
    monkeypatch.setattr(targeted_build, f"{default_prefix}_DEFAULT_REPO", default_repo)
    monkeypatch.setattr(targeted_build, f"{default_prefix}_DEFAULT_MAX_JOBS", default_jobs)
    action = TargetedBuildAction(
        gap_id="gap",
        framework="vllm",
        component=component,
        capability="build",
        repo_url=repo_url,
        max_jobs=max_jobs,
    )

    record = build_input_record(action, installed_versions={}, ambient_env={}, fs_root=NO_FS)

    assert record["repo_url"] == (repo_url.strip() or default_repo)
    assert record["max_jobs"] == (max_jobs or default_jobs)


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


def test_a_build_spawned_with_a_credentialed_index_env_classifies_it():
    """The build inherits the whole environment, so the channel is its input too."""

    class _Action:
        component = "aiter"
        repo_url = "https://github.com/ROCm/aiter"
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 8
        torch_constraint_mode = "constraint_file"
        build_command = ()
        envs: dict = {}

    record = build_input_record(
        _Action(),
        installed_versions={"aiter_ref": "v1", "sha": "s"},
        ambient_env={"PIP_INDEX_URL": "https://user:token@h/simple", "PATH": "/b", "HOME": "/h"},
        fs_root=NO_FS,
    )
    assert record["credential_channels"] == ["pip_index_env"]
    assert "token" not in json.dumps(record)


@pytest.mark.parametrize("host", ["github.com", "host]", "[host"])
def test_credentialed_repo_url_is_stripped_and_classified(host):
    class _Action:
        component = "aiter"
        repo_url = f"https://user:token@{host}/org/repo"
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 8
        torch_constraint_mode = "constraint_file"
        build_command = ()
        envs: dict = {}

    record = build_input_record(_Action(), installed_versions={}, ambient_env={}, fs_root=NO_FS)
    assert record["repo_url"] == f"https://{host}/org/repo"
    # A repository URL is under no index option, so it takes the closed
    # vocabulary's catch-all rather than borrowing a flag's class.
    assert record["credential_class"] == "opaque_credential"
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


def test_an_ambient_closure_narrower_than_the_inherited_environment_is_incomplete():
    """A digest over a hand-picked subset is indistinguishable from a full one,
    so the key list beside it is what says which was computed."""
    for keys in ([], ["ROCM_PATH", "PIP_INDEX_URL"], ["PATH"]):
        row = _attempt("bA")
        row["build_inputs"] = {**row["build_inputs"], "ambient_keys": keys}
        state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
        assert "build_inputs_incomplete" in _codes(_decide(state)), keys


def test_a_row_carrying_no_input_record_at_all_is_incomplete():
    """State written before the input contract joins, and reproduces nothing."""
    row = _attempt("bA")
    row.pop("build_inputs")
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    codes = _codes(_decide(state))
    assert "build_inputs_incomplete" in codes and "build_attempt_unjoined" not in codes


def test_an_empty_installed_versions_map_is_incomplete():
    row = _attempt("bA", installed_versions={})
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    assert "build_inputs_incomplete" in _codes(_decide(state))


def test_a_credentialed_build_input_blocks_replay():
    for inputs in ({"credential_class": "opaque_credential"}, {"credential_channels": ["pip_index_env"]}):
        row = _attempt("bA")
        row["build_inputs"] = {**row["build_inputs"], **inputs}
        state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
        assert "credential_required" in _codes(_decide(state)), inputs


@pytest.mark.parametrize("host", ["github.com", "host]", "[host"])
def test_a_credentialed_repo_url_recorded_by_the_builder_blocks_replay(host):
    class _Action:
        component = "aiter"
        repo_url = f"https://user:token@{host}/org/repo"
        ref = "v1"
        gpu_arch = "gfx950"
        max_jobs = 8
        torch_constraint_mode = "constraint_file"
        build_command = ()
        envs: dict = {}

    row = _attempt("bA")
    row["build_inputs"] = {
        **row["build_inputs"],
        **build_input_record(_Action(), installed_versions={}, ambient_env={"PATH": "/b", "HOME": "/h"}, fs_root=NO_FS),
    }
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    codes = _codes(_decide(state))
    assert "credential_required" in codes
    assert "user:token" not in str(build_recipe_steps(state, attempt_summary=_build_attempt_summary))


def test_a_credentialed_build_command_blocks_replay():
    row = _attempt("bA", build_driver="custom_command")
    row["build_inputs"] = {
        **row["build_inputs"],
        "build_command": {"argv0": "bash", "digest": "sha256:d", "credential_class": "index_url"},
    }
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    assert "credential_required" in _codes(_decide(state))


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
        "observed_server_identity": {
            "model_path": "/models/secret-model",
            "tokenizer_path": "/models/secret-tokenizer",
            "served_model_name": "/models/secret-model",
            "tp_size": 8,
        },
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
    assert not {"model_path", "tokenizer_path", "served_model_name"}.intersection(projected["observed_server_identity"])
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


def _parallelism_section(**kw):
    projected, _ = project_launch_evidence(_evidence(**kw))
    return {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}


def test_a_width_the_binding_contradicts_is_a_mismatch():
    """The extractor strips parallelism from the launch line, so only the
    binding can confirm the width a recipe asked for."""
    section = _parallelism_section(
        requested_server_args="--tp-size 8 --mem-fraction-static 0.9",
        observed_model_binding={"model_digest": "sha256:m", "tp": 2},
    )
    codes = _codes(_decide({}, section))
    assert "launch_evidence_mismatch" in codes


def test_a_width_the_binding_agrees_with_raises_nothing():
    section = _parallelism_section(
        requested_server_args="--tensor-parallel-size 8 --mem-fraction-static 0.9",
        observed_model_binding={"model_digest": "sha256:m", "tp": 8},
    )
    codes = _codes(_decide({}, section))
    assert "launch_evidence_mismatch" not in codes and "activation_incomplete" not in codes


def test_a_width_no_binding_axis_reports_is_activation_incomplete():
    section = _parallelism_section(
        requested_server_args="--pp-size 4 --mem-fraction-static 0.9",
        observed_model_binding={"model_digest": "sha256:m", "tp": 8},
    )
    assert "activation_incomplete" in _codes(_decide({}, section))


def test_a_setting_only_the_observed_identity_confirms_is_confirmed():
    """On an SGLang log with no argv line the identity parse is the only
    observed evidence there is, so an argv-only test would refuse every one."""
    section = _parallelism_section(
        observed_server_launch_flags="", observed_server_identity={"mem_fraction_static": 0.9}
    )
    codes = _codes(_decide({}, section))
    assert "activation_incomplete" not in codes and "launch_evidence_mismatch" not in codes


def test_a_setting_the_observed_identity_contradicts_is_a_mismatch():
    section = _parallelism_section(
        observed_server_launch_flags="", observed_server_identity={"mem_fraction_static": 0.5}
    )
    assert "launch_evidence_mismatch" in _codes(_decide({}, section))


def test_observed_value_contradicting_the_requested_one_is_a_mismatch():
    projected, _ = project_launch_evidence(_evidence(observed_server_launch_flags="--mem-fraction-static 0.5"))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    assert "launch_evidence_mismatch" in _codes(_decide({}, section))


def test_the_attached_spelling_of_a_requested_flag_is_the_separated_one():
    """Both sides normalize before comparison, so a launcher's spelling choice
    is not a contradiction."""
    codes = _codes(_decide({}, _parallelism_section(requested_server_args="--mem-fraction-static=0.9")))
    assert "launch_evidence_mismatch" not in codes and "activation_incomplete" not in codes


def test_model_digest_disagreement_is_a_mismatch():
    projected, _ = project_launch_evidence(_evidence(requested_model_digest="sha256:other"))
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": projected}
    assert "launch_evidence_mismatch" in _codes(_decide({}, section))


def test_untokenizable_argv_is_refused_rather_than_partially_represented():
    projected, refused = project_launch_evidence(_evidence(requested_server_args="--flag 'unterminated"))
    assert refused is True
    assert "requested_server_args" not in projected


def test_an_accepted_config_with_no_path_names_the_missing_path():
    """The materialized config is an activation input, so its absence is named
    as itself rather than folded into the generic evidence reason."""
    decision = evaluate_replay_sufficiency({}, steps=[], section={"accepted_config": {"extra_envs": {"A": "1"}}})
    scoped = [r for r in decision["reasons"] if r["scope"] == "config_path"]
    assert scoped and scoped[0]["code"] == "activation_incomplete"


def test_an_accepted_config_carrying_its_path_names_nothing():
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": project_launch_evidence(_evidence())[0]}
    decision = evaluate_replay_sufficiency({}, steps=[], section=section)
    assert not [r for r in decision["reasons"] if r["scope"] == "config_path"]


def test_a_refused_argv_names_the_launch_line_it_could_not_represent():
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": project_launch_evidence(_evidence())[0]}
    decision = evaluate_replay_sufficiency({}, steps=[], section=section, launch_argv_refused=True)
    refusal = [r for r in decision["reasons"] if r["scope"] == "observed_server_launch_flags"]
    assert refusal and refusal[0]["code"] == "activation_incomplete"
    assert decision["status"] == "insufficient"


def test_a_represented_argv_names_no_refusal():
    section = {"accepted_config": {"config_path": "c.yaml"}, "launch_evidence": project_launch_evidence(_evidence())[0]}
    decision = evaluate_replay_sufficiency({}, steps=[], section=section)
    assert not [r for r in decision["reasons"] if r["scope"] == "observed_server_launch_flags"]


def test_empty_accepted_config_keys_are_not_emitted_as_defaults():
    assert project_accepted_config({"extra_envs": {}, "extra_server_args": "", "args_mode": ""}) == {}
    assert project_accepted_config(None) == {}


def test_an_empty_accepted_config_emits_a_null_source_and_null_evidence():
    """Neither key is guessed from the branch the round happened to take."""
    out = _collect({"kept_patches": ["/p/1.patch"], "framework_root": "/fr"})
    assert "accepted_config" not in out
    assert out["accepted_config_source"] is None and out["launch_evidence"] is None


def test_a_landed_patch_without_launch_evidence_is_activation_incomplete():
    codes = _codes(_decide({"kept_patches": ["/p/1.patch"], "framework_root": "/fr"}, {}))
    assert "activation_incomplete" in codes


def test_a_projected_configuration_no_launch_observed_is_activation_incomplete():
    section = {"accepted_config": {"config_path": "reports/enablement/spec-1/launch_config.yaml"}}
    assert "activation_incomplete" in _codes(_decide({"kept_patches": ["/p/1.patch"]}, section))


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


def test_a_runtime_env_naming_an_attempt_directory_does_not_travel():
    """The switches that decide what gets launched survive; the attempt
    directories beside them are re-derived by the rebuild."""
    state = _runtime_state({"acquisition_method": "editable_ref", "repo_url": "https://h/r", "ref": "main"})
    state["active_runtime"]["runtime_env"] = {
        "INFERENCE_OPTIMIZER_AITER_JIT_DIR": "/attempt/jit",
        "AITER_REBUILD": "1",
    }
    provenance = project_runtime_provenance(state)
    assert provenance["runtime_env"] == {"AITER_REBUILD": "1"}
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


def test_a_runtime_with_neither_rebuild_source_requires_a_rebuild():
    """No acquisition and no build is no path back to the graded venv."""
    section = {"runtime_provenance": project_runtime_provenance(_runtime_state(None))}
    assert "runtime_rebuild_required" in _codes(_decide({}, section))


def test_an_unjoined_build_does_not_certify_runtime_rebuildability():
    state = {
        **_runtime_state(None),
        **_build_state([_attempt("other"), {"task_id": "wanted", "probe_task_id": "probe"}]),
    }
    section = {"runtime_provenance": project_runtime_provenance(state)}
    codes = _codes(_decide(state, section))
    assert "build_attempt_unjoined" in codes
    assert "runtime_rebuild_required" in codes


def test_an_input_incomplete_build_does_not_certify_runtime_rebuildability():
    row = _attempt("bA")
    row["build_inputs"]["resolved_sha"] = ""
    state = {
        **_runtime_state(None),
        **_build_state([row, {"task_id": "bA", "probe_task_id": "probe"}]),
    }
    section = {"runtime_provenance": project_runtime_provenance(state)}
    codes = _codes(_decide(state, section))
    assert "build_inputs_incomplete" in codes
    assert "runtime_rebuild_required" in codes


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
    state = _runtime_state(
        {
            "acquisition_method": "editable_ref",
            "repo_url": "https://u:t@h/r",
            "ref": "main",
            "resolved_ref": "c" * 40,
        }
    )
    provenance = project_runtime_provenance(state)
    assert provenance["acquisition"]["repo_url"] == "https://h/r"
    assert provenance["acquisition"]["credential_class"] == "opaque_credential"
    assert "credential_required" in _codes(_decide(state, {"runtime_provenance": provenance}))


def test_credential_classes_over_the_admitted_grammar():
    assert classify_credential_class("pip install --index-url https://user:token@host/simple foo") == "index_url"
    assert classify_credential_class("pip install git+https://user:token@host/repo@main") == "vcs_url"
    assert classify_credential_class("pip install --find-links https://u:t@h/links foo") == "find_links"
    assert classify_credential_class("conda install -c https://u:t@h/chan foo") == "channel"
    assert classify_credential_class("npm install --registry https://u:t@h/ foo") == "registry"
    # An auth token is not a URL, so the option alone is what names the class.
    assert classify_credential_class("npm install --_authToken t0ken foo") == "registry"
    assert classify_credential_class("apt-get install -y https://u:t@h/foo.deb") == "apt_source"
    assert classify_credential_class("pip install foo") is None


@pytest.mark.parametrize(
    ("command", "credential_class"),
    [
        ("pip install -ihttps://user:token@host/simple foo", "index_url"),
        ("pip install -fhttps://user:token@host/links foo", "find_links"),
        ("conda install -chttps://user:token@host/channel foo", "channel"),
    ],
)
def test_compact_short_options_are_classified_and_sanitized(command, credential_class):
    row = _row(command)
    assert row["credential_class"] == credential_class
    assert "user:token" not in row["cmd_sanitized"]
    assert "host" not in row["cmd_sanitized"]


def test_an_inline_index_assignment_classifies_and_sanitizes_as_the_flag_does():
    """The allowlist strips a leading KEY=VALUE, so both spellings are admitted."""
    cmd = "PIP_INDEX_URL=https://user:token@host/simple pip install foo"
    assert classify_credential_class(cmd) == "index_url"
    row = _row(cmd)
    assert "user:token" not in row["cmd_sanitized"] and "host" not in row["cmd_sanitized"]
    assert row["credential_class"] == "index_url"
    assert row["cmd_digest"] == command_digest(cmd)
    state = {"setup_commands": [cmd], "setup_executions": _accepted([row])}
    assert "credential_required" in _codes(_decide(state))


def test_a_secret_shaped_assignment_keeps_its_own_class():
    assert classify_credential_class("HF_TOKEN=abc pip install foo") == "env_assignment"
    assert "abc" not in _row("HF_TOKEN=abc pip install foo")["cmd_sanitized"]


def test_an_unparseable_command_still_redacts_a_credentialed_url():
    sanitized = sanitize_command_text('pip install --index-url https://user:token@private.example/simple "unclosed')
    assert "user" not in sanitized
    assert "token" not in sanitized
    assert "private.example" not in sanitized
    assert "/simple" not in sanitized


@pytest.mark.parametrize(
    ("url", "userinfo", "stripped"),
    [
        ("https://user:token@host]/simple", "user:token", "https://host]/simple"),
        ("https://user:token@[host/simple", "user:token", "https://[host/simple"),
        ("https://user:token@[::1/simple", "user:token", "https://[::1/simple"),
        ("git+https://user:token@host]/repo@main", "user:token", "git+https://host]/repo@main"),
        ("https://user:tok@en@host]/simple?q=a@b#c@d", "user:tok@en", "https://host]/simple?q=a@b#c@d"),
    ],
)
def test_malformed_url_userinfo_is_detected_and_stripped(url, userinfo, stripped):
    assert url_userinfo(url) == userinfo
    assert strip_url_userinfo(url) == stripped
    assert classify_credential_value(url) is not None


@pytest.mark.parametrize("host", ["host]", "[host", "[::1"])
def test_malformed_credentialed_index_is_sanitized_and_blocks_replay(host):
    cmd = PINNED_INSTALL.replace("pip install", f"pip install --index-url https://user:token@{host}/simple", 1)
    row = _row(cmd)
    assert row["credential_class"] == "index_url"
    assert "user:token" not in row["cmd_sanitized"]
    assert f"{host}/simple" not in row["cmd_sanitized"]
    state = {**_sufficient_state(), "setup_commands": [cmd], "setup_executions": _accepted([row])}
    decision = _decide(state, _sufficient_section())
    assert _codes(decision) == ["credential_required"]
    assert decision["status"] == "insufficient"


def test_a_bare_credentialed_url_is_not_borrowed_from_the_index_flag():
    assert classify_credential_value("https://u:t@h/r") == "opaque_credential"
    assert classify_credential_value("git+https://u:t@h/r") == "vcs_url"
    assert classify_credential_value("https://u:t@h/simple", option="--index-url") == "index_url"
    assert classify_credential_value("https://h/r") is None


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


def test_a_git_root_naming_no_commit_is_an_unidentified_tree():
    """is_git standing over no base_sha names no tree a consumer can check out."""
    section = {"roots": project_roots([{**_root(), "path": "/fr", "base_sha": ""}])}
    assert "root_unidentified" in _codes(_decide({}, section))


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
        "kept_artifacts": [{"target": "/fr/srt/b.py", "rel_target": "srt/b.py", "root_id": "r1"}],
    }
    assert "artifact_not_self_contained" in _codes(_decide({}, section))


def test_an_artifact_bound_to_no_root_is_unidentified():
    """An artifact names a tree the same way a patch step does, or it names none."""
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot()],
        "kept_artifacts": [{"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "root_id": None}],
    }
    assert "root_unidentified" in _codes(_decide({}, section))


def test_an_artifact_bound_to_an_unrecorded_root_is_unidentified():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot()],
        "kept_artifacts": [{"target": "/pkg/srt/a.py", "rel_target": "srt/a.py", "root_id": "r2"}],
    }
    assert "root_unidentified" in _codes(_decide({}, section))


def test_another_roots_capture_does_not_contain_this_artifact():
    """The same framework-relative layout repeats across a checkout and its copy."""
    section = {
        "roots": project_roots(
            [
                {**_root(), "path": "/fr"},
                {**_root(root_id="r2", contributions=("artifact_install",), rel="sglang"), "path": "/pkg/sglang"},
            ]
        ),
        "source_snapshots": [_snapshot(), _snapshot(root_id="r2", files=())],
        "kept_artifacts": [{"target": "/pkg/sglang/srt/a.py", "rel_target": "srt/a.py", "root_id": "r2"}],
    }
    codes = _codes(_decide({}, section))
    assert "artifact_not_self_contained" in codes
    assert "root_unidentified" not in codes


def test_a_target_recorded_missing_is_not_a_captured_payload():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot(files=(("srt/a.py", "missing"),), complete=False)],
        "kept_artifacts": [{"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "root_id": "r1"}],
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


def test_a_stripped_round_names_no_target_and_raises_accepted_stack_not_launched():
    """A launch-only probe replays the stack while declaring none of its targets.

    Its own round contributes no applied patch, so the per-target comparison has
    nothing to walk and the previous KEEP's self-consistent records survive in
    durable state.
    """
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot()],
        "accepted_stack_targets": {},
    }
    decision = _decide({"kept_patches": ["/p/1.patch"], "framework_root": "/fr"}, section)
    reasons = [r for r in decision["reasons"] if r["code"] == "accepted_stack_not_launched"]
    assert reasons and reasons[0]["blocks"] == "both"
    assert decision["status"] == "insufficient"


def test_a_kept_artifact_alone_also_demands_a_named_target():
    section = {
        "roots": project_roots([{**_root(), "path": "/fr"}]),
        "source_snapshots": [_snapshot(files=(("srt/a.py", "upsert"),))],
        "kept_artifacts": [{"target": "/fr/srt/a.py", "rel_target": "srt/a.py", "root_id": "r1"}],
    }
    assert "accepted_stack_not_launched" in _codes(_decide({}, section))


def test_a_recipe_declaring_no_stack_is_not_faulted_for_naming_no_target():
    section = {"roots": project_roots([{**_root(), "path": "/fr"}]), "source_snapshots": [_snapshot()]}
    assert "accepted_stack_not_launched" not in _codes(_decide({}, section))


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


PINNED_INSTALL = f"pip install git+https://host/repo@{'a' * 40}"


def _sufficient_state():
    cmd = PINNED_INSTALL
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


def _perturbations():
    """One well-formed recipe, broken one way at a time."""
    snapshot_only_delete = _snapshot(files=(("srt/a.py", "delete"),))
    return {
        "activation_incomplete": ({}, {"accepted_config_source": "advanced_merge"}),
        "launch_evidence_mismatch": ({}, {"launch_evidence": _mismatched_evidence()}),
        "root_unidentified": ({}, {"roots": project_roots([{**_root(), "path": "/fr", "base_sha": ""}])}),
        "root_unmappable": ({}, {"roots": project_roots([{**_root(anchor="unmappable"), "path": "/fr"}])}),
        "source_snapshot_incomplete": ({}, {"source_snapshots": [_snapshot(complete=False)]}),
        "source_snapshot_missing": (
            {},
            # A second contributing root whose capture returned no manifest; the
            # first root's stack is captured exactly as it was declared.
            {"roots": project_roots([{**_root(), "path": "/fr"}, _unsnapshotted_root()])},
        ),
        "accepted_stack_not_launched": ({}, {"source_snapshots": [snapshot_only_delete]}),
        "environment_closure_absent": ({}, {"environment_closure": None}),
        "assertions_not_at_keep": ({}, {"installed_versions_at_keep": {}}),
        "setup_occurrences_unknown": ({"setup_executions": []}, {}),
        "credential_required": (
            {
                "setup_commands": [CREDENTIALED_INSTALL],
                "setup_executions": _accepted([_row(CREDENTIALED_INSTALL, task="kept")], task="kept"),
            },
            {},
        ),
    }


CREDENTIALED_INSTALL = f"pip install git+https://user:token@host/repo@{'a' * 40}"


def _unsnapshotted_root():
    return {
        "id": "r2",
        "path": "/pkg",
        "kind": "site_packages",
        "contributions": ["artifact_install"],
        "is_git": False,
        "base_sha": "",
        "replay_target": {"anchor": "site_packages", "rel": "pkg"},
    }


def _mismatched_evidence():
    evidence = project_launch_evidence(_evidence())[0]
    binding = dict(evidence["observed_model_binding"])
    binding["model_digest"] = "0" * 64
    return {**evidence, "observed_model_binding": binding}


@pytest.mark.parametrize("code", sorted(_perturbations()))
def test_each_fail_closed_case_carries_exactly_its_own_code(code):
    """A reason set wider than the defect makes the verdict unreadable."""
    state_delta, section_delta = _perturbations()[code]
    decision = _decide({**_sufficient_state(), **state_delta}, {**_sufficient_section(), **section_delta})
    assert _codes(decision) == [code], decision["reasons"]
    assert decision["status"] == "insufficient"


def test_absent_closure_and_assertions_fail_closed():
    section = {**_sufficient_section(), "environment_closure": None, "installed_versions_at_keep": {}}
    codes = _codes(_decide(_sufficient_state(), section))
    assert "environment_closure_absent" in codes and "assertions_not_at_keep" in codes


@pytest.mark.parametrize(
    "cmd",
    ["apt-get install -y libfoo", "npm install package", "conda install package"],
)
def test_non_python_installer_withholds_a_verified_closure(cmd):
    rows = _accepted([_row(cmd, seq=1, task="kept")], task="kept")
    state = {**_sufficient_state(), "setup_commands": [cmd], "setup_executions": rows}
    assert "closure_scope_incomplete" in _codes(_decide(state, _sufficient_section()))


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


def test_reason_code_vocabulary_matches_the_closed_contract():
    assert set(REASON_BLOCKS) == {
        "not_evaluated",
        "activation_incomplete",
        "launch_evidence_mismatch",
        "runtime_rebuild_required",
        "root_unidentified",
        "root_unmappable",
        "accepted_stack_not_launched",
        "source_snapshot_incomplete",
        "source_snapshot_missing",
        "artifact_not_self_contained",
        "setup_occurrences_unknown",
        "setup_effect_outside_verified_launch",
        "setup_ledger_truncated",
        "build_attempt_unjoined",
        "build_inputs_incomplete",
        "environment_closure_absent",
        "closure_scope_incomplete",
        "assertions_not_at_keep",
        "credential_required",
    }


def test_a_recorded_decision_reads_back_as_the_producer_wrote_it():
    """Only an absent key and an unrecognized code are overridden; a decision
    over the closed vocabulary is the one the consumer acts on."""
    recorded = _decide(_sufficient_state(), _sufficient_section())
    assert read_status({"replay_sufficiency": recorded}) == recorded


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


def _log(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_observed_model_binding_is_read_for_sglang_and_vllm(tmp_path):
    from hyperloom.common.launch_log_evidence import observed_model_binding_from_log, split_launch_flags

    sglang = _log(
        tmp_path,
        "sglang.log",
        "INFO python3 -m sglang.launch_server --model-path /models/a --tp-size 8 --mem-fraction-static 0.9\n",
    )
    vllm_serve = _log(tmp_path, "vllm_serve.log", "INFO vllm serve /models/b --tensor-parallel-size 4\n")
    vllm_module = _log(
        tmp_path,
        "vllm_module.log",
        "INFO python -m vllm.entrypoints.openai.api_server --model /models/c --tp 2\n",
    )
    for path, framework, tp in ((sglang, "sglang", "8"), (vllm_serve, "vllm", "4"), (vllm_module, "vllm", "2")):
        binding = observed_model_binding_from_log(path, framework)
        assert binding["model_digest"].startswith("sha256:")
        assert binding["tp"] == tp
        assert "/models/" not in str(binding)
    # The forwarded flags still carry no model or parallelism operand.
    forwarded = split_launch_flags("--model /models/b --tensor-parallel-size 4 --mem-fraction-static 0.9")
    assert forwarded == "--mem-fraction-static 0.9"


def test_framework_outside_the_marker_table_has_no_observable_binding(tmp_path):
    from hyperloom.common.launch_log_evidence import observed_model_binding_from_log

    atom = _log(tmp_path, "atom.log", "INFO atom_server --model-path /models/a\n")
    assert observed_model_binding_from_log(atom, "atom") == {}


def test_evidence_builder_carries_the_binding_and_the_requested_digest(tmp_path):
    from hyperloom.orchestrator.actions.executors._launch_evidence import build_launch_evidence

    config = tmp_path / "materialized.yaml"
    config.write_text("benchmark:\n  framework: vllm\n  model: /models/b\n", encoding="utf-8")
    log = _log(tmp_path, "server.log", "INFO vllm serve /models/b --tensor-parallel-size 4\n")
    evidence = build_launch_evidence(
        config_path=config,
        actual_server_log=log,
        framework="vllm",
        slot=tmp_path,
    )
    assert evidence["observed_model_binding"]["model_digest"] == evidence["requested_model_digest"]
    projected, _refused = project_launch_evidence(evidence)
    assert projected["observed_model_binding"]["tp"] == "4"


def test_provision_result_carries_the_resolved_identity_fields():
    from hyperloom.orchestrator.framework.stack_actions import ProvisionResult

    state = ProvisionResult(
        ok=True,
        resolved_ref="c" * 40,
        resolved_packages={"sglang": {"version": "0.4", "artifact_digest": "sha256:w"}},
    ).to_state()
    assert state["resolved_ref"] == "c" * 40
    assert state["resolved_packages"]["sglang"]["artifact_digest"] == "sha256:w"


def test_resolved_clone_ref_reads_the_commit_the_clone_landed_on():
    from hyperloom.orchestrator.framework.adapters import _resolved_clone_ref

    class _Completed:
        returncode = 0
        stdout = "d" * 40 + "\n"

    assert _resolved_clone_ref("/checkout", run=lambda *_a, **_k: _Completed()) == "d" * 40


def test_resolved_packages_reports_version_and_record_digest():
    from hyperloom.orchestrator.framework.adapters import _resolved_packages

    class _Completed:
        returncode = 0
        stdout = '{"sglang": {"version": "0.4", "artifact_digest": "sha256:w"}}'

    resolved = _resolved_packages("/py", ["sglang"], run=lambda *_a, **_k: _Completed())
    assert resolved == {"sglang": {"version": "0.4", "artifact_digest": "sha256:w"}}
    assert _resolved_packages("/py", [], run=lambda *_a, **_k: _Completed()) == {}


def _collect(state):
    from hyperloom.inference_optimizer.breakdown.collectors.sessions import collect_enablement

    return collect_enablement(Path("/tmp/sess"), {"enablement": {"attempts": 1, **state}}, [])


def test_closure_status_is_unverified_while_a_closure_reason_stands():
    out = _collect({"kept_patches": ["/p/1.patch"], "framework_root": "/fr"})
    assert out["dependency_closure_status"] == "unverified"
    assert out["replay_sufficiency"]["status"] == "insufficient"


def test_closure_status_is_verified_when_only_unrelated_reasons_stand():
    """A root a consumer cannot place does not make the dependency set unpinned."""
    out = _collect(
        {
            "kept_patches": ["/p/1.patch"],
            "framework_root": "/fr",
            "setup_commands": ["pip install foo"],
            "setup_executions": _accepted([_row("pip install foo", seq=1, task="kept")], task="kept"),
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "root_unidentified" in codes
    assert out["dependency_closure_status"] == "verified"


def test_a_session_with_no_ledger_certifies_no_closure():
    """The scope verdict is read off the ledger; without one nothing enumerated
    which installers ran."""
    out = _collect(
        {
            "setup_commands": ["apt-get install -y libfoo"],
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "setup_occurrences_unknown" in codes and "closure_scope_incomplete" not in codes
    assert out["dependency_closure_status"] == "unverified"


def test_an_absent_ledger_with_no_commands_still_certifies_no_closure():
    """The ledger is the only record of which installer families ran, so its
    absence leaves the scope unobserved rather than clean."""
    out = _collect(
        {
            "kept_patches": ["/p/1.patch"],
            "framework_root": "/fr",
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "setup_occurrences_unknown" not in codes
    assert out["dependency_closure_status"] == "unverified"


def test_a_capped_command_withholds_a_verified_closure():
    """A command the validated round never ran is an installer set nothing saw."""
    rows = _accepted([_row("pip install foo", seq=1, task="kept")], task="kept")
    out = _collect(
        {
            "setup_commands": ["pip install foo", "pip install capped"],
            "setup_executions": rows,
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "setup_ledger_truncated" in codes
    assert out["dependency_closure_status"] == "unverified"


@pytest.mark.parametrize(
    "cmd",
    ["apt-get install -y libfoo", "npm install package", "conda install package"],
)
def test_closure_status_is_unverified_for_a_non_python_installer(cmd):
    """Scope, not absence: the map is present and covers less than was installed."""
    rows = _accepted([_row(cmd, seq=1, task="kept")], task="kept")
    out = _collect(
        {
            "setup_commands": [cmd],
            "setup_executions": rows,
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "closure_scope_incomplete" in codes
    assert out["dependency_closure_status"] == "unverified"


def test_closure_status_is_verified_with_complete_build_inputs():
    row = _attempt("bA")
    out = _collect(
        {
            **_sufficient_state(),
            "build_manifest": [row, {"task_id": "bA", "probe_task_id": "probe"}],
            "last_specialist_task_id": "probe",
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    assert "build_inputs_incomplete" not in _codes(out["replay_sufficiency"])
    assert out["dependency_closure_status"] == "verified"


def test_closure_status_is_unverified_while_the_build_inputs_are_incomplete():
    row = _attempt("bA")
    row["build_inputs"] = {**row["build_inputs"], "resolved_sha": ""}
    out = _collect(
        {
            "build_manifest": [row, {"task_id": "bA", "probe_task_id": "probe"}],
            "last_specialist_task_id": "probe",
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    codes = [r["code"] for r in out["replay_sufficiency"]["reasons"]]
    assert "build_inputs_incomplete" in codes
    assert out["dependency_closure_status"] == "unverified"


def test_closure_status_is_unverified_while_the_build_attempt_is_unjoined():
    out = _collect(
        {
            "build_manifest": [_attempt("other"), {"task_id": "wanted", "probe_task_id": "probe"}],
            "last_specialist_task_id": "probe",
            "setup_commands": ["pip install foo"],
            "setup_executions": _accepted([_row("pip install foo", task="kept")], task="kept"),
            "environment_closure": {"interpreter_tag": "3.10.14", "distributions": {"sglang": "0.4"}},
            "installed_versions_at_keep": {"sglang": "0.4"},
        }
    )
    assert "build_attempt_unjoined" in _codes(out["replay_sufficiency"])
    assert out["dependency_closure_status"] == "unverified"


def test_build_inputs_reach_the_emitted_step_stripped_of_credential_material():
    row = _attempt("bA")
    row["build_inputs"]["repo_url"] = "https://user:token@github.com/org/repo"
    state = _build_state([row, {"task_id": "bA", "probe_task_id": "probe"}])
    steps = build_recipe_steps(state, attempt_summary=_build_attempt_summary)
    inputs = steps[0]["build_inputs"]
    assert inputs["repo_url"] == "https://github.com/org/repo"
    assert inputs["resolved_sha"] == "s" * 40


def test_a_round_with_no_task_id_claims_no_ledger_rows():
    """Rows a lane never identified stay unreported, never silently accepted."""
    rows = [_row("pip install x", seq=1, task="r1")]
    unchanged = mark_round_disposition(rows, round_task_id="", disposition="kept", accepted=True)
    assert unchanged[0]["round_disposition"] == "unreported"
    assert unchanged[0]["present_at_final_launch"] is False


def test_a_build_provisioned_runtime_names_that_build_as_its_rebuild_path():
    """A KEEP through a launch-only probe provisions nothing; the build is the path."""
    state = {
        **_build_state([_attempt("bA"), {"task_id": "bA", "probe_task_id": "probe"}]),
        "active_runtime": {"python_path": "/attempt/venv/bin/python"},
    }
    provenance = project_runtime_provenance(state)
    assert provenance["build_task_id"] == "bA"
    section = {"runtime_provenance": provenance}
    assert "runtime_rebuild_required" not in _codes(_decide(state, section))


def test_acquisition_channels_reach_runtime_provenance_and_block_replay():
    """A runtime acquired over an authenticated remote names the class it needed."""
    state = {
        "active_runtime": {"python_path": "/attempt/venv/bin/python"},
        "kept_stack_action": {
            "acquisition_method": "editable_ref",
            "repo_url": "https://h/r",
            "ref": "main",
            "resolved_ref": "c" * 40,
            "credential_channels": ["ssh_agent"],
        },
    }
    provenance = project_runtime_provenance(state)
    assert provenance["acquisition"]["credential_channels"] == ["ssh_agent"]
    assert "credential_required" in _codes(_decide(state, {"runtime_provenance": provenance}))
