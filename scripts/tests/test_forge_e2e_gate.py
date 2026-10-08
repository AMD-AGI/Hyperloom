# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for Hyperloom's path-gated KernelForge E2E smoke."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_GATE_SCRIPT = _ROOT / ".github" / "scripts" / "forge_e2e_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("forge_e2e_gate", _GATE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def test_vendored_forge_changes_trigger() -> None:
    for path in (
        "src/kernelforge/cli.py",
        "src/kernelforge/loop/runner.py",
        "src/kernelforge/data/examples/triton-softmax-forge-loop/driver.py",
        "pyproject.toml",
    ):
        assert gate.requires_forge_e2e([path]), path


def test_fusion_and_gemm_only_changes_do_not_trigger() -> None:
    assert not gate.requires_forge_e2e(
        [
            "src/kernelforge/fusion/loop.py",
            "src/kernelforge/tests/fusion/test_loop.py",
            "src/kernelforge/gemm_tune/router.py",
            "src/kernelforge/gemm_tune/tests/test_router.py",
        ]
    )


def test_shared_or_loop_change_still_triggers_alongside_excluded_changes() -> None:
    assert gate.requires_forge_e2e(
        [
            "src/kernelforge/gemm_tune/router.py",
            "src/kernelforge/agent_backends/codex.py",
        ]
    )


def test_forge_e2e_contract_changes_trigger() -> None:
    for path in (
        ".github/workflows/forge-e2e.yml",
        ".github/scripts/forge-ci-e2e-dispatch.sh",
        ".github/scripts/forge_e2e_gate.py",
        ".github/scripts/forge_e2e_report.py",
        "examples/triton-softmax-forge-loop/run_example.sh",
    ):
        assert gate.requires_forge_e2e([path]), path


def test_unrelated_hyperloom_changes_do_not_trigger() -> None:
    assert not gate.requires_forge_e2e(
        [
            "docs/user-guide/quickstart.md",
            "src/hyperloom/orchestrator/knowledge/config.py",
            "src/hyperloom/inference_optimizer/tests/test_local_recipe_store.py",
        ]
    )


def test_previous_filename_can_trigger_a_rename_out_of_forge() -> None:
    # The workflow feeds both filename and previous_filename from the Pull Files API, so deleting or renaming a Forge
    # file cannot evade the gate.
    assert gate.requires_forge_e2e(
        [
            "src/hyperloom/unrelated/new_home.py",
            "src/kernelforge/loop/old_home.py",
        ]
    )


def test_rename_confined_to_excluded_products_does_not_trigger() -> None:
    assert not gate.requires_forge_e2e(
        [
            "src/kernelforge/fusion/new_home.py",
            "src/kernelforge/gemm_tune/old_home.py",
        ]
    )


def test_workflow_dispatches_the_kernelforge_smoke_contract() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8")
    dispatcher = (_ROOT / ".github" / "scripts" / "forge-ci-e2e-dispatch.sh").read_text(encoding="utf-8")

    # Dispatron's CLI submits and polls. The PR commit reaches the run as its flags, and
    # Dispatron's platform row -- not a workspace named here -- decides where it lands.
    assert "dispatron-ci \\" in dispatcher
    for flag in ("--kind kernelforge", '--sha "$HEAD_SHA"', '--source-repo "$SRC_REPO"', '--pull-ref "$PULL_REF"'):
        assert flag in dispatcher, flag
    assert "STATUS_CONTEXT: ci-e2e/kernelforge" in workflow
    assert "FORGE_E2E_WORKSPACE" not in workflow
    assert "secrets.KERNEL_OPT_WORKSPACE" not in workflow
    assert "secrets.CI_E2E_API_BASE" not in workflow
    assert ".github/scripts/forge-ci-e2e-dispatch.sh" in workflow
    assert ".github/scripts/forge_e2e_report.py" in dispatcher


def test_workflow_path_gates_pr_runs_but_not_manual_runs() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8")

    assert 'if [ "$run" = true ] && [ "$EVENT" != "workflow_dispatch" ]; then' in workflow
    assert "(.previous_filename // empty)" in workflow
    assert "files?per_page=100&page=$page" in workflow
    assert "path_skipped=true" in workflow
    assert 'description="skipped: no Forge loop-related files changed"' in workflow


def test_a_pr_runs_only_once_the_run_label_is_on() -> None:
    """Adding the label starts a run; other PR events run only while it stays on. A
    comment or a label being removed is not a way in."""
    workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8"))
    triggers = workflow[True]  # YAML 1.1 reads the bare key `on` as the boolean True.
    resolve_condition = " ".join(workflow["jobs"]["resolve"]["if"].split())

    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert "unlabeled" not in triggers["pull_request"]["types"]
    assert "github.event.label.name == 'run-forge-e2e'" in resolve_condition
    assert "contains(github.event.pull_request.labels.*.name, 'run-forge-e2e')" in resolve_condition


def test_only_resolved_events_can_cancel_an_in_flight_forge_run() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8")
    workflow_header, _ = workflow.split("jobs:", 1)

    # Workflow-level concurrency is joined by every event on the PR before resolve.if, so adding an unrelated label
    # would cancel an expensive GPU run.
    assert "concurrency:" not in workflow_header

    group = "forge-e2e-${{ needs.resolve.outputs.pr_number || needs.resolve.outputs.head_sha || needs.resolve.outputs.head_ref || github.ref }}"
    assert workflow.count(group) == 2
    assert workflow.count("cancel-in-progress: true") == 2


def test_legacy_template_entry_point_runs_the_vendored_example() -> None:
    wrapper = (_ROOT / "examples" / "triton-softmax-forge-loop" / "run_example.sh").read_text(encoding="utf-8")

    assert '"${ROOT}[forge,forge-profiling]"' in wrapper
    assert "/src/kernelforge/data/examples/triton-softmax-forge-loop/run_example.sh" in wrapper
