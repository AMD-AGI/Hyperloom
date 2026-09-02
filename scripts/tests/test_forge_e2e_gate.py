# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contracts for Hyperloom's path-gated KernelForge E2E smoke."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
    # The workflow feeds both filename and previous_filename from the Pull Files
    # API, so deleting or renaming a Forge file cannot evade the gate.
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

    assert 'kind:"kernelforge"' in dispatcher
    assert "KERNELFORGE_SOURCE_REPO:$srcrepo" in dispatcher
    assert "KERNELFORGE_SOURCE_SHA:$sha" in dispatcher
    assert "KERNELFORGE_SOURCE_PULL_REF:$pullref" in dispatcher
    assert 'KF_USE_GIT:"1"' in dispatcher
    assert "STATUS_CONTEXT: ci-e2e/kernelforge" in workflow
    assert "CI_E2E_WORKSPACE: ${{ vars.FORGE_E2E_WORKSPACE || 'control-plan-hyperloom-ci' }}" in workflow
    assert "secrets.KERNEL_OPT_WORKSPACE" not in workflow
    assert ".github/scripts/forge-ci-e2e-dispatch.sh" in workflow
    assert ".github/scripts/forge_e2e_report.py" in dispatcher
    assert "__FORGE_RESULT__" in dispatcher


def test_workflow_gates_all_pr_backed_retries_but_not_manual_runs() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8")

    assert 'if [ "$run" = true ] && [ "$EVENT" != "workflow_dispatch" ]; then' in workflow
    assert "(.previous_filename // empty)" in workflow
    assert "files?per_page=100&page=$page" in workflow
    assert '[.labels[].name] | index("skip-e2e-test")' in workflow
    assert "path_skipped=true" in workflow
    assert 'description="skipped: no Forge loop-related files changed"' in workflow


def test_only_resolved_events_can_cancel_an_in_flight_forge_run() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "forge-e2e.yml").read_text(encoding="utf-8")
    workflow_header, jobs = workflow.split("jobs:", 1)

    # Workflow-level concurrency is evaluated before resolve.if and could let
    # an untrusted or malformed comment cancel an expensive GPU run.
    assert "concurrency:" not in workflow_header
    assert "github.event.comment.author_association == 'OWNER'" in jobs
    assert "github.event.comment.author_association == 'MEMBER'" in jobs
    assert "github.event.comment.author_association == 'COLLABORATOR'" in jobs

    group = "forge-e2e-${{ needs.resolve.outputs.pr_number || needs.resolve.outputs.head_ref || github.ref }}"
    assert workflow.count(group) == 2
    assert workflow.count("cancel-in-progress: true") == 2

    # Adding the opt-out label must reach skipped-status so it can enter the
    # same resolved concurrency group and cancel an already running workload.
    labeled_clause = workflow.split("(github.event.action != 'labeled'", 1)[1].split(") &&", 1)[0]
    assert "github.event.label.name == 'retest'" in labeled_clause
    assert "github.event.label.name == 'skip-e2e-test'" in labeled_clause


def test_legacy_template_entry_point_runs_the_vendored_example() -> None:
    wrapper = (_ROOT / "examples" / "triton-softmax-forge-loop" / "run_example.sh").read_text(encoding="utf-8")

    assert '"${ROOT}[forge,forge-profiling]"' in wrapper
    assert "/src/kernelforge/data/examples/triton-softmax-forge-loop/run_example.sh" in wrapper
