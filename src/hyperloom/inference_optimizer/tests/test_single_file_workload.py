# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-file patch boundaries and portable export contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.deployment.export import export_custom_inference
from hyperloom.orchestrator.framework.optimization_scope import (
    aoti_config,
    check_patch_scope,
    optimization_file,
    scope_instructions,
)


def accept_source(session, hook, stack_entry_id="accepted0"):
    """Record ``hook``'s current bytes as an accepted stack snapshot.

    Mirrors what the integrate_patch executor writes on a KEEP. Export verifies the
    tree against these, so a workload with no snapshot has nothing proving what it
    benchmarked -- which is why the fixture writes one rather than relying on a git
    checkout to stand in for it.
    """
    snapshot = session / "optimization_stack" / "src" / stack_entry_id
    (snapshot / "files").mkdir(parents=True, exist_ok=True)
    (snapshot / "files" / hook.name).write_bytes(hook.read_bytes())
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "complete": True,
                "files": [{"rel": hook.name, "op": "upsert"}],
            }
        )
    )
    return {
        "stack_entry_id": stack_entry_id,
        "source_snapshot": str(snapshot),
        "source_snapshot_complete": True,
        "target_files": [hook.name],
    }


@pytest.fixture
def workload(tmp_path, monkeypatch):
    root, bench, session = [tmp_path / name for name in ("model", "bench", "session")]
    for directory in (root, bench, session):
        directory.mkdir()
    hook = root / "hyperloom_optimize.py"
    hook.write_text("def hyperloom_optimize(model):\n    return model\n")
    (bench / "optimization_scope.json").write_text(json.dumps({"schema_version": 1, "file": str(hook)}))
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(bench))
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(root))
    state = {
        "framework": "custom",
        "framework_repo_path": str(root),
        "bypass_scripts_dir": str(bench),
        "current_best": {"tput": 100},
        "optimization_stack": [accept_source(session, hook)],
    }
    return root, hook, session, state


def patch(name):
    return f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1 +1 @@\n-old\n+new\n"


def test_scope_allows_hook_only(workload, tmp_path):
    root, hook, _, _ = workload
    assert optimization_file() == hook
    check_patch_scope(root, [patch(hook.name)])
    for name in ("benchmark.py", "extra.py", "../transformers/model.py"):
        with pytest.raises(ValueError):
            check_patch_scope(root, [patch(name)])
    with pytest.raises(ValueError):
        check_patch_scope(tmp_path, [patch(hook.name)])


def declare(bench, hook, aoti=None, schema_version=2):
    """Rewrite the scope declaration, optionally carrying an aoti block."""
    scope = {"schema_version": schema_version, "file": str(hook)}
    if aoti is not None:
        scope["aoti"] = aoti
    (bench / "optimization_scope.json").write_text(json.dumps(scope))


def dim(**overrides):
    entry = {
        "input": "observation.language.tokens",
        "dim": 1,
        "min": 8,
        "max": 128,
        "derivation": "tokenizer output over the deployed instruction set; 4x margin",
    }
    entry.update(overrides)
    return entry


def test_schema_version_2_is_accepted_without_an_aoti_block(workload):
    root, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook)

    assert optimization_file() == hook
    assert aoti_config() is None
    assert "Ahead-of-time export" not in scope_instructions()


def test_unsupported_schema_version_is_refused(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=3)

    with pytest.raises(ValueError, match="Invalid single-file optimization scope"):
        optimization_file()


def test_aoti_declaration_is_parsed(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    config = aoti_config()

    assert config.enabled is True
    assert config.pin_all_others is True  # defaults on, so unlisted dims are static
    assert len(config.dynamic_dims) == 1
    assert config.dynamic_dims[0].input == "observation.language.tokens"
    assert (config.dynamic_dims[0].min, config.dynamic_dims[0].max) == (8, 128)


def test_declared_ranges_reach_the_agent_prompt(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    text = scope_instructions()

    assert "Ahead-of-time export is the objective" in text
    assert "observation.language.tokens axis 1 in [8, 128]" in text
    assert "must be pinned" in text
    # The bounds are baked into the artifact, so the agent must not re-derive them.
    assert "do not widen, narrow or invent one" in text


def test_disabled_aoti_leaves_the_prompt_untouched(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": False, "dynamic_dims": [dim()]})

    assert aoti_config().enabled is False
    assert "Ahead-of-time export" not in scope_instructions()


def test_export_without_dynamic_dims_says_fixed_shapes(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True})

    text = scope_instructions()

    assert "declares no dynamic dimensions" in text
    assert "export at fixed shapes" in text


@pytest.mark.parametrize(
    "entry,message",
    [
        (dim(derivation=""), "must say how the bound was measured"),
        (dim(min=128, max=128), "requires min < max"),
        (dim(min=200, max=128), "requires min < max"),
        (dim(min=0), "min must be at least 1"),
        (dim(dim=-1), "non-negative axis index"),
        (dim(input=""), "must be a non-empty string"),
        (dim(min="8"), "min must be an integer"),
    ],
)
def test_malformed_dynamic_dim_is_refused(workload, entry, message):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [entry]})

    with pytest.raises(ValueError, match=message):
        aoti_config()


def test_missing_dynamic_dim_field_is_refused(workload):
    _, hook, _, state = workload
    entry = dim()
    del entry["derivation"]
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [entry]})

    with pytest.raises(ValueError, match=r"missing \['derivation'\]"):
        aoti_config()


def test_duplicate_axis_declaration_is_refused(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim(), dim()]})

    with pytest.raises(ValueError, match="twice"):
        aoti_config()


def test_same_input_may_declare_two_axes(workload):
    _, hook, _, state = workload
    declare(
        Path(state["bypass_scripts_dir"]),
        hook,
        {"enabled": True, "dynamic_dims": [dim(), dim(dim=2)]},
    )

    assert len(aoti_config().dynamic_dims) == 2


def test_non_boolean_enabled_is_refused(workload):
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": "yes"})

    with pytest.raises(ValueError, match="aoti.enabled must be a boolean"):
        aoti_config()


def test_schema_1_ignores_an_aoti_block(workload):
    """Schema 1 predates the block, so a stray one must not silently activate export."""
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True}, schema_version=1)

    assert "Ahead-of-time export" not in scope_instructions()


def test_launch_scope_check_accepts_the_declared_hook(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors.bypass_scriptable import enforce_custom_scope

    root, hook, _, _ = workload
    assert optimization_file() == hook
    enforce_custom_scope("custom")
    enforce_custom_scope("sglang")


def test_launch_scope_check_refuses_an_out_of_boundary_root(workload, tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.bypass_scriptable import enforce_custom_scope

    _, hook, _, _ = workload
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="permits changes only"):
        enforce_custom_scope("custom")


def test_integration_cannot_use_recorded_foreign_root(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors.integrate_patch import _resolve_framework_root

    root, hook, _, _ = workload
    assert _resolve_framework_root(str(root), patch_texts=[patch(hook.name)], recorded_root=str(root)) == root
    assert _resolve_framework_root(str(root), patch_texts=[patch("extra.py")], recorded_root=str(root)) is None
    assert _resolve_framework_root(str(tmp_path), patch_texts=[patch(hook.name)], recorded_root=str(tmp_path)) is None


def test_kernel_direct_write_is_refused(workload, tmp_path):
    from hyperloom.orchestrator.kernel.apply_kernel_patch import apply_kernel_patch

    _, hook, _, _ = workload
    before = hook.read_bytes()
    result = apply_kernel_patch(patch_path=hook, target_file=hook, backup_root=tmp_path / "backup")
    assert result["error_class"] == "single_file_scope"
    assert hook.read_bytes() == before


def test_export_contains_only_module_and_metadata(workload):
    _, hook, session, state = workload
    result = export_custom_inference(session, state)
    assert result["status"] == "exported"
    assert result["reasons"] == []
    assert result["kind"] == "single_file"
    assert set(result["files"]) == {hook.name}
    assert {p.name for p in (session / "deployment").iterdir()} == {hook.name}
    assert json.loads((session / "reports/deployment.json").read_text())["kind"] == "single_file"
    assert (session / "deployment" / hook.name).read_bytes() == hook.read_bytes()


def test_dirty_source_does_not_export_as_accepted(workload):
    _, hook, session, state = workload
    hook.write_text("def hyperloom_optimize(model):\n    return None\n")
    result = export_custom_inference(session, state)
    assert result["status"] == "incomplete"
    assert not (session / "deployment" / hook.name).exists()


def test_export_bakes_accepted_switches_into_module(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1", "HL_HOIST": "1"}

    result = export_custom_inference(session, state)

    exported = session / "deployment" / hook.name
    assert result["status"] == "exported"
    assert "HL_FUSE_ATTN" in exported.read_text()
    assert "HL_HOIST" in exported.read_text()
    assert hook.read_bytes() == b"def hyperloom_optimize(model):\n    return model\n"
    assert result["files"][hook.name] == hashlib.sha256(exported.read_bytes()).hexdigest()
    assert result["settings"]["base_sha256"] == hashlib.sha256(hook.read_bytes()).hexdigest()


def test_export_reports_untranslatable_settings(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {
        "name": "PYTORCH_HIP_ALLOC_CONF",
        "value": "expandable_segments:True",
        "reason": "pre_context_init",
    } in result["settings"]["untranslated"]
    assert "PYTORCH_HIP_ALLOC_CONF" not in exported_text
    assert len(result["warnings"]) == 1
    assert "PYTORCH_HIP_ALLOC_CONF" in result["warnings"][0]


def test_export_excludes_benchmark_control_envs(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"RUN_EVAL": "false", "CONC": "1"}

    result = export_custom_inference(session, state)

    exported_text = (session / "deployment" / hook.name).read_text()
    assert result["status"] == "exported"
    assert {"name": "RUN_EVAL", "value": "false", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert {"name": "CONC", "value": "1", "reason": "benchmark_control"} in result["settings"]["excluded"]
    assert "RUN_EVAL" not in exported_text
    assert "CONC" not in exported_text


def test_export_is_idempotent_across_closes(workload):
    _, hook, session, state = workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1"}

    first_result = export_custom_inference(session, state)
    first_bytes = (session / "deployment" / hook.name).read_bytes()
    second_result = export_custom_inference(session, state)
    second_bytes = (session / "deployment" / hook.name).read_bytes()

    assert first_result["status"] == "exported"
    assert second_result["status"] == "exported"
    assert first_result["files"] == second_result["files"]
    assert first_bytes == second_bytes


AOTI_HOOK = """\
import torch


def hyperloom_optimize(model):
    ep = torch.export.export(model, (torch.zeros(1),))
    pkg = torch._inductor.aoti_compile_and_package(ep, package_path="/tmp/m.pt2")
    return torch._inductor.aoti_load_package(pkg)
"""


def reaccept_hook(session, hook, state, text):
    """Rewrite the hook and re-record it as accepted, as a fresh KEEP would.

    Re-snapshotting is what keeps the export verifiable: writing the file alone
    would leave it differing from the recorded stack, which export refuses.
    """
    hook.write_text(text)
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]


def test_export_flags_aoti_runtime_requirements(workload):
    _, hook, session, state = workload
    reaccept_hook(session, hook, state, AOTI_HOOK)

    result = export_custom_inference(session, state)

    assert result["status"] == "exported"
    assert result["runtime_requirements"]["compiles_on_first_call"] is True
    assert "TORCHINDUCTOR_CACHE_DIR" in result["runtime_requirements"]["writable_cache_dir"]
    assert any("aoti_first_call_compiles" in warning for warning in result["warnings"])
    # tput is steady state; the deployer must not read it as time-to-first-result.
    assert "compiles an AOTInductor package on its first call" in result["usage"]


def test_export_omits_aoti_metadata_for_an_eager_module(workload):
    _, _, session, state = workload

    result = export_custom_inference(session, state)

    assert "runtime_requirements" not in result
    assert result["warnings"] == []


def test_aoti_detection_survives_a_from_import(workload):
    _, hook, session, state = workload
    reaccept_hook(
        session,
        hook,
        state,
        "from torch._inductor import aoti_load_package\n\n\ndef hyperloom_optimize(model):\n"
        '    return aoti_load_package("/tmp/m.pt2")\n',
    )

    assert "runtime_requirements" in export_custom_inference(session, state)


def test_export_refuses_credential_shaped_setting(workload):
    _, _, session, state = workload
    state["current_best"]["extra_envs"] = {"HF_TOKEN": "x"}

    result = export_custom_inference(session, state)

    assert result["status"] == "incomplete"
    assert "HF_TOKEN" in result["reasons"][0]


def test_pycache_does_not_block_export(workload):
    root, _, session, state = workload
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "hyperloom_optimize.cpython-312.pyc").write_bytes(b"junk")

    result = export_custom_inference(session, state)

    assert result["status"] == "exported"


def test_scope_refuses_symlink(workload, tmp_path):
    _, hook, _, _ = workload
    other = tmp_path / "other.py"
    other.write_bytes(hook.read_bytes())
    hook.unlink()
    hook.symlink_to(other)
    with pytest.raises(ValueError, match="regular"):
        optimization_file()


def test_specialist_inherits_scope_and_writable_cli_config(workload, monkeypatch):
    from hyperloom.orchestrator.actions.executors._ray_runtime import safe_runtime_env
    from hyperloom.orchestrator.specialists.subprocess_ import _build_specialist_env

    _, _, _, state = workload
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/scratch/run/claude")
    env = _build_specialist_env()
    assert env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"
    worker_env = safe_runtime_env()["env_vars"]
    assert worker_env["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == state["bypass_scripts_dir"]
    assert worker_env["CLAUDE_CONFIG_DIR"] == "/scratch/run/claude"


def test_export_refuses_when_no_snapshot_covers_the_hook(workload):
    """A config-only KEEP records no snapshot; exporting then would verify nothing."""
    _, hook, session, state = workload
    state["optimization_stack"] = [{"stack_entry_id": "env0", "source_snapshot": ""}]
    hook.write_text("def hyperloom_optimize(model):\n    return None  # never benchmarked\n")

    result = export_custom_inference(session, state)

    assert result["status"] == "incomplete"
    assert "nothing proves what was benchmarked" in result["reasons"][0]
    assert not (session / "deployment" / hook.name).exists()


def test_export_refuses_a_hook_importing_a_sibling(workload):
    """Only the hook ships, so a sibling import would deploy broken."""
    root, hook, session, state = workload
    (root / "_kernels.py").write_text("X = 1\n")
    hook.write_text("import _kernels\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    result = export_custom_inference(session, state)

    assert result["status"] == "incomplete"
    assert "imports '_kernels' from beside itself" in result["reasons"][0]


def test_export_refuses_a_relative_import(workload):
    root, hook, session, state = workload
    hook.write_text("from . import helpers\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    assert export_custom_inference(session, state)["status"] == "incomplete"


def test_export_allows_stdlib_and_unrelated_scratch_files(workload):
    """Scratch files nothing imports are harmless; the old git check rejected them."""
    root, hook, session, state = workload
    (root / "scratch_probe.py").write_text("# left behind by a specialist\n")
    hook.write_text("import json\nimport torch\n\n\ndef hyperloom_optimize(model):\n    return model\n")
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]

    assert export_custom_inference(session, state)["status"] == "exported"


def test_module_uses_aoti_detects_an_exporting_hook(tmp_path):
    from hyperloom.orchestrator.deployment.single_file import module_uses_aoti

    exporting = tmp_path / "a.py"
    exporting.write_text(AOTI_HOOK)
    eager = tmp_path / "b.py"
    eager.write_text("def hyperloom_optimize(model):\n    return model\n")

    assert module_uses_aoti(exporting) is True
    assert module_uses_aoti(eager) is False


def test_module_uses_aoti_tolerates_unreadable_or_broken_modules(tmp_path):
    """The gate reads this during dispatch; it must never raise and wedge the loop."""
    from hyperloom.orchestrator.deployment.single_file import module_uses_aoti

    broken = tmp_path / "broken.py"
    broken.write_text("def hyperloom_optimize(  # unbalanced\n")

    assert module_uses_aoti(broken) is False
    assert module_uses_aoti(tmp_path / "does_not_exist.py") is False


def test_export_gate_instructions_ban_the_eager_fallback(workload):
    """A fallback would let a non-exporting module land a KEEP and open the gate."""
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    text = scope_instructions()

    assert "Do not fall back to eager on failure" in text
    assert "rule='export_gate'" in text


def test_export_gate_points_at_the_pattern_reference(workload):
    """The how-to lives in the catalog the specialist already receives, not in the prompt."""
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    text = scope_instructions()

    assert "framework_rewrite_patterns.md (h)" in text
    # The prompt is injected into the Coordinator's context too; keep it short.
    assert len(text) < 1600


class _ReportState:
    """Minimal stand-in for SharedState as report.py reads it."""

    def __init__(self, *, stop_reason="time_exhausted", export_landed=False, export_attempts=0):
        self.stop_reason = stop_reason
        self.export_landed = export_landed
        self.export_attempts = export_attempts


def test_unlanded_export_renames_any_terminal(workload):
    """time_exhausted reads as a normal closeout; an unlanded export must not hide behind it."""
    from hyperloom.orchestrator.actions.executors.report import _export_gate_stop_reason

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    for terminal in ("time_exhausted", "global_converged", "max_ticks", ""):
        assert _export_gate_stop_reason(_ReportState(stop_reason=terminal)) == "export_never_landed"


def test_landed_export_leaves_the_terminal_alone(workload):
    from hyperloom.orchestrator.actions.executors.report import _export_gate_stop_reason

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    assert _export_gate_stop_reason(_ReportState(export_landed=True)) == ""


def test_export_gate_is_inert_when_aoti_not_declared(workload):
    """Workloads that never asked for an export must be completely unaffected."""
    from hyperloom.orchestrator.actions.executors.report import _export_gate_stop_reason

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=1)

    assert _export_gate_stop_reason(_ReportState()) == ""


def test_export_never_landed_explanation_points_at_both_causes(workload):
    from hyperloom.orchestrator.actions.executors.report import _explain_stop_reason

    text = _explain_stop_reason("export_never_landed", None)

    assert "phase-budget-framework-pct" in text
    assert "tracing blocker" in text
    assert "ungated_close_reason" in text


def _budget_args(max_hours, extend_hours=0.0):
    import argparse

    return argparse.Namespace(max_hours=max_hours, extend_hours=extend_hours)


def test_export_gate_budget_warns_but_never_refuses(workload, capsys):
    """The default --max-hours affords none; refusing it would block the happy path."""
    from hyperloom.inference_optimizer.cli import _check_export_gate_budget
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _check_export_gate_budget(_budget_args(2.0), {PHASE_FRAMEWORK_AGENT: 0.38})

    assert "affords 0" in capsys.readouterr().err


def test_export_gate_budget_counts_extend_hours(workload, capsys):
    """--extend-hours lands in session state, not args.max_hours."""
    from hyperloom.inference_optimizer.cli import _check_export_gate_budget
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _check_export_gate_budget(_budget_args(1.0, extend_hours=5.0), {PHASE_FRAMEWORK_AGENT: 0.67})

    assert capsys.readouterr().err == ""


def test_export_gate_budget_warns_on_a_single_specialist(workload, capsys):
    from hyperloom.inference_optimizer.cli import _check_export_gate_budget
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    # The real default: 3h x 0.38 = 68 min, one 60-min patch specialist.
    _check_export_gate_budget(_budget_args(3.0), {PHASE_FRAMEWORK_AGENT: 0.38})

    err = capsys.readouterr().err
    assert "WARNING" in err and "affords 1" in err
    assert "--phase-budget-framework-pct 0.67" in err


def test_export_gate_budget_is_quiet_when_two_fit(workload, capsys):
    from hyperloom.inference_optimizer.cli import _check_export_gate_budget
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _check_export_gate_budget(_budget_args(3.0), {PHASE_FRAMEWORK_AGENT: 0.75})

    assert capsys.readouterr().err == ""


def test_export_gate_budget_ignores_workloads_without_aoti(workload, capsys):
    from hyperloom.inference_optimizer.cli import _check_export_gate_budget
    from hyperloom.orchestrator.phases.machine_state import PHASE_FRAMEWORK_AGENT

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=1)

    _check_export_gate_budget(_budget_args(0.5), {PHASE_FRAMEWORK_AGENT: 0.38})

    assert capsys.readouterr().err == ""


class _GateState:
    def __init__(self, export_landed=False):
        self.export_landed = export_landed


class _Graded:
    """Stand-in for resolve_graded_comparison's result."""

    def __init__(self, comparable=True, veto_reason=""):
        self.comparable = comparable
        self.veto_reason = veto_reason


def _exporting_workload(workload):
    """Declare aoti and put a genuinely exporting module at the hook path."""
    root, hook, session, state = workload
    hook.write_text(AOTI_HOOK)
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})
    return root, hook, session, state


def test_first_export_lands_despite_being_slower(workload):
    """Eager is not a deliverable here, so a correct export must not lose to it on latency."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    _exporting_workload(workload)

    assert _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=-15.4, accuracy_pass=True, acc_block=False
    )


def test_export_landing_requires_correctness(workload):
    """Only 'slow' is waived -- never a failed accuracy gate."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    _exporting_workload(workload)

    assert not _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=-15.4, accuracy_pass=False, acc_block=False
    )
    assert not _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=-15.4, accuracy_pass=True, acc_block=True
    )


def test_export_landing_requires_a_measurement(workload):
    """An unbenchmarked candidate is not evidence that the export works."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    _exporting_workload(workload)

    assert not _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=None, accuracy_pass=True, acc_block=False
    )


def test_export_landing_fires_only_once(workload):
    """After the gate opens, normal throughput scoring resumes against the export."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    _exporting_workload(workload)

    assert not _export_landing_keep(
        shared_state=_GateState(export_landed=True),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=True,
        acc_block=False,
    )


def test_export_landing_ignores_a_non_exporting_module(workload):
    """A slower eager rewrite must not ride the export carve-out into current_best."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    assert not _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=-15.4, accuracy_pass=True, acc_block=False
    )


def test_export_landing_inert_without_aoti_declared(workload):
    from hyperloom.orchestrator.actions.executors.integrate_patch import _export_landing_keep

    root, hook, session, state = workload
    hook.write_text(AOTI_HOOK)
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=1)

    assert not _export_landing_keep(
        shared_state=_GateState(), graded=_Graded(), delta_pct=-15.4, accuracy_pass=True, acc_block=False
    )


class _SummaryState:
    def __init__(self, export_landed=False, export_attempts=0, cumulative_gain_validated=0.0):
        self.export_landed = export_landed
        self.export_attempts = export_attempts
        self.cumulative_gain_validated = cumulative_gain_validated


def test_summary_explains_a_landed_but_slower_export(workload, capsys):
    """A negative headline gain is the expected shape of a first export, not a regression."""
    from hyperloom.inference_optimizer.cli.bootstrap import _print_export_gate_line

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _print_export_gate_line(_SummaryState(export_landed=True, cumulative_gain_validated=-14.9))

    out = capsys.readouterr().out
    assert "aoti_export" in out and "landed" in out
    assert "still behind the eager path" in out
    assert "which this workload does not ship" in out


def test_summary_does_not_excuse_an_export_that_won(workload, capsys):
    """Once the export overtakes eager, explaining away a negative gain reads as an excuse."""
    from hyperloom.inference_optimizer.cli.bootstrap import _print_export_gate_line

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _print_export_gate_line(_SummaryState(export_landed=True, cumulative_gain_validated=127.1))

    out = capsys.readouterr().out
    assert "faster than the eager path it replaces" in out
    assert "still behind" not in out


def test_summary_flags_an_export_that_never_landed(workload, capsys):
    from hyperloom.inference_optimizer.cli.bootstrap import _print_export_gate_line

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _print_export_gate_line(_SummaryState(export_attempts=4))

    out = capsys.readouterr().out
    assert "NEVER LANDED after 4 integrate round(s)" in out
    assert "no shippable artifact" in out


def test_summary_is_silent_without_aoti(workload, capsys):
    from hyperloom.inference_optimizer.cli.bootstrap import _print_export_gate_line

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=1)

    _print_export_gate_line(_SummaryState())

    assert capsys.readouterr().out == ""


def test_catalog_forbids_a_shared_package_path():
    """Keyed on anything coarser than the module, the next candidate measures the last one's artifact."""
    from pathlib import Path as _Path

    catalog = _Path(__file__).resolve().parents[1] / "references" / "framework_rewrite_patterns.md"
    text = catalog.read_text()

    assert "Write the package somewhere per-process" in text
    assert "HYPERLOOM_CACHE_DIR" in text
    assert "hand-bumped version constant" in text
