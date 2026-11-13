# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-file patch boundaries and portable export contracts."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from hyperloom.orchestrator.deployment.export import export_custom_inference
from hyperloom.orchestrator.framework.optimization_scope import (
    OptimizationScope,
    aoti_config,
    optimization_file,
    optimization_scope,
    scope_instructions,
)
from .conftest import _lift_coordinator, accept_source, declare


AOTI_HOOK = """\
import torch


def hyperloom_optimize(model):
    ep = torch.export.export(model, (torch.zeros(1),))
    pkg = torch._inductor.aoti_compile_and_package(ep, package_path="/tmp/m.pt2")
    return torch._inductor.aoti_load_package(pkg)
"""


EVIDENCE_SHAPES = [
    ("real export", {"compiled": 1, "loaded": 1, "invoked": 1}, True),
    ("dead code", {"compiled": 0, "loaded": 0, "invoked": 0}, False),
    ("caught fallback", {"compiled": 0, "loaded": 0, "invoked": 0}, False),
    ("compiled, not used", {"compiled": 1, "loaded": 1, "invoked": 0}, False),
]


class _GateState:
    def __init__(self, export_landed=False):
        self.export_landed = export_landed


class _Graded:
    """Stand-in for resolve_graded_comparison's result."""

    def __init__(self, comparable=True, veto_reason=""):
        self.comparable = comparable
        self.veto_reason = veto_reason


class _ReportState:
    """Minimal stand-in for SharedState as report.py reads it."""

    def __init__(self, *, stop_reason="time_exhausted", export_landed=False, export_attempts=0, exports=None):
        self.stop_reason = stop_reason
        self.export_landed = export_landed
        self.current_best_exports = export_landed if exports is None else exports
        self.export_attempts = export_attempts


class _SummaryState:
    def __init__(self, export_landed=False, export_attempts=0, cumulative_gain_validated=0.0, exports=None):
        self.export_landed = export_landed
        self.current_best_exports = export_landed if exports is None else exports
        self.export_attempts = export_attempts
        self.cumulative_gain_validated = cumulative_gain_validated


def _budget_args(max_hours, extend_hours=0.0):
    import argparse

    return argparse.Namespace(max_hours=max_hours, extend_hours=extend_hours)


def _exporting_workload(workload, *, invoked: int = 12):
    """Declare aoti, put an exporting module at the hook path, and leave run counters.

    The counters are what the waiver and the gate both read: a module whose source names
    the entry points proves nothing, since a caught fallback to eager parses identically.
    """
    import json as _json

    root, hook, session, state = workload
    hook.write_text(AOTI_HOOK)
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})
    counters = session / "aoti_evidence"
    counters.mkdir(parents=True, exist_ok=True)
    (counters / "1.json").write_text(
        _json.dumps({"instrumented": True, "compiled": 1, "loaded": 1, "invoked": invoked})
    )
    return root, hook, session, state


def _slower_winner(**over):
    return {"name": "aoti-export", "extra_server_args": "", "output_throughput": 850.0, **over}


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


def reaccept_hook(session, hook, state, text):
    """Rewrite the hook and re-record it as accepted, as a fresh KEEP would.

    Re-snapshotting is what keeps the export verifiable: writing the file alone
    would leave it differing from the recorded stack, which export refuses.
    """
    hook.write_text(text)
    state["optimization_stack"] = [accept_source(session, hook, "accepted1")]


def write_evidence(workspace, *, pid=1, instrumented=True, compiled=0, loaded=0, invoked=0, nested="bench"):
    """Write one process's export counters where the reader looks for them."""
    from hyperloom.orchestrator.measurement.aoti_evidence import EVIDENCE_DIR_NAME

    out = workspace / nested / EVIDENCE_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pid}.json").write_text(
        json.dumps(
            {
                "schema": "hyperloom.aoti_evidence/1",
                "pid": pid,
                "instrumented": instrumented,
                "compiled": compiled,
                "loaded": loaded,
                "invoked": invoked,
            }
        )
    )


def test_schema_version_2_is_accepted_without_an_aoti_block(workload):
    root, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook)

    assert optimization_file() == hook
    assert aoti_config() is None
    assert "Ahead-of-time export" not in scope_instructions()


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


def test_named_module_and_function_export(named_workload):
    _, hook, session, state = named_workload
    state["current_best"]["extra_envs"] = {"HL_FUSE_ATTN": "1"}
    assert optimization_scope() == OptimizationScope(file=hook, function="apply")
    assert "define apply(model)" in scope_instructions()

    result = export_custom_inference(session, state)

    assert result["status"] == "exported"
    assert result["entrypoint"] == "encoder_opt.apply"
    assert {p.name for p in (session / "deployment").iterdir()} == {"encoder_opt.py"}
    assert "HL_FUSE_ATTN" in (session / "deployment" / "encoder_opt.py").read_text()


def test_a_declaration_may_name_a_function_and_declare_an_export(named_workload):
    """Both schema-2 keys are independent: either, both or neither may appear."""
    _, hook, _, state = named_workload
    bench = Path(state["bypass_scripts_dir"])
    scope = {"schema_version": 2, "file": str(hook), "function": "apply", "aoti": {"enabled": True}}
    (bench / "optimization_scope.json").write_text(json.dumps(scope))

    assert optimization_scope() == OptimizationScope(file=hook, function="apply")
    assert aoti_config().enabled
    assert "define apply(model)" in scope_instructions()


def test_a_declaration_without_a_function_keeps_the_original_name(workload):
    """An aoti-only declaration predates the configurable entry function."""
    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True})

    assert optimization_scope() == OptimizationScope(file=hook, function="hyperloom_optimize")


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


def test_the_export_metadata_is_decided_by_source_but_the_gate_is_not(tmp_path):
    """``uses_aoti`` decides what the deployment metadata says, not whether the export
    landed. Only the run counters decide that -- the module below names the entry points
    and would still never export, which is why there is no source-reading path into the
    gate. Keep it that way: a helper taking a *path* here invites one."""
    import ast
    import inspect

    from hyperloom.orchestrator.deployment import single_file

    fallback = (
        "import torch\n\n\ndef hyperloom_optimize(model):\n"
        "    try:\n        return torch._inductor.aoti_load_package('m.pt2')\n"
        "    except Exception:\n        return model\n"
    )
    assert single_file.uses_aoti(ast.parse(fallback)) is True

    assert not hasattr(single_file, "module_uses_aoti")
    assert "read_bytes" not in inspect.getsource(single_file.uses_aoti)


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


def test_a_broken_harness_cannot_report_an_export_the_artifact_does_not_have(workload):
    """The integrate gate refuses a candidate that cannot be shown to export, so these two
    part only when the shim breaks. What ships is what the report must describe."""
    from hyperloom.orchestrator.actions.executors.report import _export_gate_stop_reason

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    assert _export_gate_stop_reason(_ReportState(export_landed=True, exports=False)) == "export_never_landed"


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


def test_first_export_lands_despite_being_slower(workload):
    """Eager is not a deliverable here, so a correct export must not lose to it on latency."""
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    _exporting_workload(workload)

    assert _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=True,
        acc_block=False,
        workspace=str(workload[2]),
    )


def test_export_landing_requires_correctness(workload):
    """Only 'slow' is waived -- never a failed accuracy gate."""
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    _exporting_workload(workload)

    assert not _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=False,
        acc_block=False,
        workspace=str(workload[2]),
    )
    assert not _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=True,
        acc_block=True,
        workspace=str(workload[2]),
    )


def test_export_landing_requires_a_measurement(workload):
    """An unbenchmarked candidate is not evidence that the export works."""
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    _exporting_workload(workload)

    assert not _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=None,
        accuracy_pass=True,
        acc_block=False,
        workspace=str(workload[2]),
    )


def test_export_landing_fires_only_once(workload):
    """After the gate opens, normal throughput scoring resumes against the export."""
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

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
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    assert not _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=True,
        acc_block=False,
        workspace=str(workload[2]),
    )


def test_export_landing_inert_without_aoti_declared(workload):
    from hyperloom.orchestrator.actions.executors._integrate_perf_gate import _export_landing_keep

    root, hook, session, state = workload
    hook.write_text(AOTI_HOOK)
    declare(Path(state["bypass_scripts_dir"]), hook, schema_version=1)

    assert not _export_landing_keep(
        shared_state=_GateState(),
        graded=_Graded(),
        delta_pct=-15.4,
        accuracy_pass=True,
        acc_block=False,
        workspace=str(workload[2]),
    )


def test_first_export_is_stacked_although_slower_than_the_baseline(tmp_path):
    """Without a stack entry nothing snapshots the hook, so the deployment export has no module to ship."""
    coord = _lift_coordinator(tmp_path)

    assert coord.writeback.lift_to_current_best("integrate_patch", 850.0, _slower_winner(keep_reason="export_landed"))
    assert coord.shared_state.current_best["tput"] == 850.0
    assert [entry["variant_name"] for entry in coord.shared_state.optimization_stack] == ["aoti-export"]


def test_a_slower_winner_without_the_export_reason_is_still_held(tmp_path):
    coord = _lift_coordinator(tmp_path)

    assert not coord.writeback.lift_to_current_best("integrate_patch", 850.0, _slower_winner())
    assert not coord.shared_state.optimization_stack


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


def test_the_summary_points_at_the_harness_when_the_counters_went_missing(workload, capsys):
    """Reachable only through a broken shim. Saying "landed" would describe the artifact as
    an export; saying "never landed" would be false. Name the harness instead."""
    from hyperloom.inference_optimizer.cli.bootstrap import _print_export_gate_line

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    _print_export_gate_line(_SummaryState(export_landed=True, exports=False, cumulative_gain_validated=31.0))

    out = capsys.readouterr().out
    assert "NOT IN THE ARTIFACT" in out and "broken harness" in out
    assert "NEVER LANDED" not in out

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


def test_launch_refuses_a_malformed_aoti_block(workload, tmp_path):
    """Nothing else reports it: every other aoti_config caller swallows the error.

    Left unchecked the run is silently useless -- scope_instructions omits the
    declaration it could not read, so the agent is never told to export, while the
    dispatcher holds the gate shut on the same error.
    """
    from hyperloom.orchestrator.actions.executors._suite.custom_workload import enforce_custom_scope

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim(min=128, max=8)]})

    with pytest.raises(ValueError, match="requires min < max"):
        enforce_custom_scope("custom")


def test_launch_accepts_a_well_formed_aoti_block(workload, tmp_path):
    from hyperloom.orchestrator.actions.executors._suite.custom_workload import enforce_custom_scope

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True, "dynamic_dims": [dim()]})

    enforce_custom_scope("custom")


def test_a_revert_is_not_an_export_attempt(workload):
    """export_attempts measures progress toward an export; a REVERT never reached the tree."""
    from hyperloom.orchestrator.loop.writeback import WritebackCollaborator

    _, hook, _, state = workload
    declare(Path(state["bypass_scripts_dir"]), hook, {"enabled": True})

    # Unbound: ``shared_state`` is a read-only property on the real collaborator, and
    # the method reads nothing else off ``self``.
    stub = SimpleNamespace(shared_state=SimpleNamespace(export_landed=False, export_attempts=0))

    assert WritebackCollaborator._record_export_gate(stub, kept_flag=False) is False
    assert stub.shared_state.export_attempts == 0


@pytest.mark.parametrize("name,counts,exported", EVIDENCE_SHAPES, ids=[s[0] for s in EVIDENCE_SHAPES])
def test_evidence_separates_a_real_export(tmp_path, name, counts, exported):
    from hyperloom.orchestrator.measurement.aoti_evidence import read_evidence

    write_evidence(tmp_path, **counts)

    evidence = read_evidence(tmp_path)

    assert evidence.observed is True
    assert evidence.exported is exported


def test_evidence_absent_is_not_the_same_as_did_not_export(tmp_path):
    """No counters means the shim never ran; the gate must not read that as a verdict."""
    from hyperloom.orchestrator.measurement.aoti_evidence import read_evidence

    evidence = read_evidence(tmp_path)

    assert evidence.observed is False
    assert evidence.exported is False


def test_evidence_ignores_a_process_that_never_instrumented(tmp_path):
    """Wrapped nothing, so its zeros are evidence of nothing."""
    from hyperloom.orchestrator.measurement.aoti_evidence import read_evidence

    write_evidence(tmp_path, instrumented=False, compiled=1, loaded=1, invoked=1)

    assert read_evidence(tmp_path).observed is False


def test_evidence_aggregates_across_processes(tmp_path):
    """A forked benchmark reports once per pid; the export may land in any of them."""
    from hyperloom.orchestrator.measurement.aoti_evidence import read_evidence

    write_evidence(tmp_path, pid=1, compiled=1, loaded=1, invoked=0)
    write_evidence(tmp_path, pid=2, compiled=0, loaded=0, invoked=1)

    evidence = read_evidence(tmp_path)

    assert evidence.processes == 2
    assert evidence.exported is True


def test_evidence_survives_an_unreadable_file(tmp_path):
    from hyperloom.orchestrator.measurement.aoti_evidence import EVIDENCE_DIR_NAME, read_evidence

    write_evidence(tmp_path, pid=1, compiled=1, loaded=1, invoked=1)
    (tmp_path / "bench" / EVIDENCE_DIR_NAME / "2.json").write_text("{ truncated")

    assert read_evidence(tmp_path).exported is True


def test_gate_stays_shut_when_the_shim_did_not_run(workload, tmp_path, monkeypatch):
    """Missing counters are a broken harness, not a verdict; there is no source fallback."""
    from hyperloom.orchestrator.loop.writeback import WritebackCollaborator

    _, hook, _, _ = workload
    # A module that would satisfy the old source check, with no counters beside it.
    hook.write_text(
        "import torch\n\n\ndef hyperloom_optimize(model):\n    return torch._inductor.aoti_load_package('m.pt2')\n"
    )
    empty = tmp_path / "ws-no-counters"
    empty.mkdir()

    assert WritebackCollaborator._keep_really_exported(object.__new__(WritebackCollaborator), str(empty)) is False
