"""Regression tests for the long-horizon KernelForge CLI integration.

The forge-loop runs in a hard-killable subprocess, so a long-horizon campaign is
routinely terminated mid-iteration. Everything here pins the contract that makes
such a run salvageable rather than wasted:

  * the CLI invocation + isolated process group that the kill relies on,
  * the two recovery channels submit trusts, in order --
    ``<workspace>/forge_experiments/best_result.json`` (the published manifest)
    first, then ``<experiments_dir>/hyperloom.json`` (the caller-owned
    checkpoint) -- and what happens when they disagree,
  * the rule that a timed-out run with NO validated recovery discards its
    measurements and fails, while one WITH a validated recovery returns a
    salvaged, exportable best commit,
  * the precondition that makes any of that trustworthy -- a campaign starts
    fresh or not at all -- and the workspace hygiene around it: a stale campaign
    worktree is replaced rather than reused, and an in-place campaign (which
    runs *in* the developer's checkout) keeps every campaign-owned path under
    the output dir and hands the checkout back as it found it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import _flydsl_rewrite  # noqa: E402
import forge_submit  # noqa: E402


@pytest.fixture(autouse=True)
def _forget_rewrite_capabilities():
    """The capability answer is cached per process; never leak it across tests."""
    _flydsl_rewrite.reset_capability_cache()
    yield
    _flydsl_rewrite.reset_capability_cache()


def _git(repo: Path | str, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return proc.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.com")
    kernel = repo / "kernel.py"
    kernel.write_text("BASELINE\n")
    (repo / "forge_driver.py").write_text("TRACKED_DRIVER\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, kernel


def _published_manifest(commit_hash: str, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "commit_hash": commit_hash,
        "correctness_passed": True,
        "baseline_wall_ms": 3.0,
        "best_wall_ms": 2.0,
        "mean_case_speedup": 1.5,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "iteration": 2,
        "snr_db": 42.0,
    }
    payload.update(overrides)
    return payload


def _checkpoint(base_commit: str, best_commit: str, **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "experiment_id": "hyperloom",
        "state": "best_committed",
        "base_commit": base_commit,
        "best_commit": best_commit,
        "baseline_ms": 3.0,
        "best_ms": 1.5,
        "mean_case_speedup": 2.0,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "validation_passed": True,
        "case_coverage": [],
    }
    payload.update(overrides)
    return payload


def _stub_submit_environment(monkeypatch) -> None:
    """Neutralize everything submit does outside the loop/recovery contract."""
    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")


def test_observed_regression_score_is_preserved_for_diagnostics():
    observed = forge_submit._observed_mean_case_result_fields(
        {
            "mean_case_speedup": 0.95,
            "search_start_mean_case_speedup": 1.0,
        }
    )

    assert observed == (0.95, 1.0, False, False)


def test_regression_is_not_a_valid_recovery_best():
    fields = forge_submit._mean_case_result_fields(
        {
            "mean_case_speedup": 0.95,
            "search_start_mean_case_speedup": 1.0,
        }
    )

    assert fields is None


def test_all_kernel_sources_are_remapped_into_prepared_worktree(tmp_path):
    repo, kernel = _make_repo(tmp_path)
    sibling = repo / "kernels" / "device.py"
    sibling.parent.mkdir()
    sibling.write_text("DEVICE\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add device source")
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    prepared = forge_submit._prepare_worktree(
        str(kernel),
        str(repo),
        output_dir,
        "forge/test/source-remap",
    )
    assert prepared is not None
    workspace, worktree_kernel, _base = prepared

    sources = forge_submit._remap_implementation_sources(
        candidate={
            "kernel_sources": [
                str(sibling),
                str(kernel),
                str(sibling),
            ]
        },
        source_file=str(kernel),
        workspace=workspace,
        worktree_kernel=worktree_kernel,
        kernel_repo=str(repo),
    )

    assert sources == [
        str(Path(worktree_kernel).resolve()),
        str((Path(workspace) / "kernels" / "device.py").resolve()),
    ]
    assert all(Path(path).is_file() for path in sources)
    assert all(Path(path).is_relative_to(Path(workspace).resolve()) for path in sources)


def test_unmappable_declared_source_fails_remapping(tmp_path):
    repo, kernel = _make_repo(tmp_path)
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    prepared = forge_submit._prepare_worktree(
        str(kernel),
        str(repo),
        output_dir,
        "forge/test/unmappable-source",
    )
    assert prepared is not None
    workspace, worktree_kernel, _base = prepared

    with pytest.raises(
        forge_submit._WorktreePreparationError,
        match="declared implementation source could not be mapped",
    ):
        forge_submit._remap_implementation_sources(
            candidate={"kernel_sources": [str(tmp_path / "outside.py")]},
            source_file=str(kernel),
            workspace=workspace,
            worktree_kernel=worktree_kernel,
            kernel_repo=str(repo),
        )


def test_submit_fails_before_loop_when_declared_source_is_unmappable(
    tmp_path,
    monkeypatch,
):
    repo, kernel = _make_repo(tmp_path)
    outside = tmp_path / "external.py"
    outside.write_text("def external():\n    pass\n")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    _stub_submit_environment(monkeypatch)

    def must_not_run(**_kwargs):
        raise AssertionError("forge loop must not run with an incomplete source set")

    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", must_not_run)
    result = forge_submit.submit(
        source_file=str(kernel),
        prompt_file=prompt,
        output_dir=tmp_path / "attempt-submit",
        source_type="triton",
        candidate={
            "operation": "direct_kernel",
            "kernel_sources": [str(kernel), str(outside)],
            "platform": "mi355x",
        },
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 1
    assert "declared implementation source could not be mapped" in (
        result["stderr_tail"]
    )


def test_warm_start_best_is_exported_without_a_later_keep(tmp_path, monkeypatch):
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "warm-only"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("WARM_START_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "validated warm start")
        warm_commit = _git(workspace, "rev-parse", "HEAD")
        captured["warm_commit"] = warm_commit
        structured = {
            "baseline_ms": 1.2,
            "pristine_baseline_ms": 1.0,
            "search_start_ms": 1.2,
            "best_ms": 1.2,
            "mean_case_speedup": 1.25,
            "search_start_mean_case_speedup": 1.25,
            "improved": True,
            "total_improved": True,
            "incremental_improved": False,
            "improved_during_search": False,
            "best_commit": warm_commit,
            "kb_experience": {
                "read": {
                    "candidate": True,
                    "applied": True,
                    "validation_passed": True,
                    "pristine_ms": 1.0,
                    "keep_baseline_ms": 1.2,
                    "best_commit": warm_commit,
                }
            },
        }
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=1.2,
            best_ms=1.2,
            improved=True,
            output="warm start applied; no later KEEP",
            error=RuntimeError("forge-loop exited after warm-start validation"),
            timed_out=False,
            checkpoint=None,
            pristine_baseline_ms=1.0,
            search_start_ms=1.2,
            improved_during_search=False,
            structured_result=structured,
            mean_case_speedup=1.25,
            search_start_mean_case_speedup=1.25,
            total_improved=True,
            incremental_improved=False,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={
            "name": "direct_kernel",
            "operation": "direct_kernel",
            "device_kernel_names": ["direct_kernel"],
            "kernel_sources": [str(source)],
            "kernel_kind": "triton",
            "platform": "mi355x",
        },
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 0
    assert result["salvaged"] is True
    assert result["best_commit"] == captured["warm_commit"]
    assert result["kb_experience"]["read"]["applied"] is True
    assert result["pristine_baseline_ms"] == 1.0
    assert result["search_start_ms"] == 1.2
    assert result["best_ms"] == 1.2
    assert result["mean_case_speedup"] == 1.25
    assert result["total_improved"] is True
    assert result["incremental_improved"] is False
    assert result["improved"] is True
    assert result["improved_during_search"] is False
    assert "[micro_speedup] 1.2500x" in (
        output_dir / "optimization_report.md"
    ).read_text()
    assert (output_dir / "optimized_versions" / "v1_forge.py").read_text() == (
        "WARM_START_BEST\n"
    )
    report = (output_dir / "optimization_report.md").read_text()
    assert "[micro_speedup] 1.2500x" in report
    assert "improved_during_search=false" in report


def test_nonzero_exit_with_sidecar_timings_never_exports_dirty_worktree(
    tmp_path,
    monkeypatch,
):
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "failed-dirty"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")

    def fake_loop(**kwargs):
        Path(kwargs["worktree_kernel"]).write_text("UNVALIDATED_DIRTY_EDIT\n")
        structured = {
            "baseline_ms": 2.0,
            "pristine_baseline_ms": 2.0,
            "search_start_ms": 2.0,
            "best_ms": 1.0,
            "improved": True,
            "kb_experience": {"read": {"applied": False}},
        }
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=2.0,
            best_ms=1.0,
            improved=True,
            output="result sidecar was written before exit",
            error=RuntimeError("forge-loop exited rc=7"),
            timed_out=False,
            checkpoint=None,
            pristine_baseline_ms=2.0,
            search_start_ms=2.0,
            improved_during_search=True,
            structured_result=structured,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={
            "operation": "direct_kernel",
            "kernel_sources": [str(source)],
            "platform": "mi355x",
        },
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 1
    assert result["forge_result"]["best_ms"] == 1.0
    assert not (output_dir / "optimization_report.md").exists()
    assert not (output_dir / "optimized_versions").exists()


def test_generated_drivers_stage_in_the_workspace_without_clobbering(tmp_path):
    """Both driver generators write a hidden, unique driver into the workspace.

    ``campaign_config._relative_file`` rejects a ``--driver`` outside
    ``--workspace``, so a generated driver parked in the attempt output dir
    kills every forge-loop run with "driver must be inside workspace". Staging
    it in the workspace is therefore mandatory, and the ``.forge_driver_``
    mkstemp naming is what keeps that safe: a unique hidden name can never
    clobber a tracked file, ``git diff`` of tracked paths keeps it out of the
    keep/revert patch, and ``_finalize_forge_workspace`` deletes it by prefix.
    """
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    tracked_driver = workspace / "forge_driver.py"
    tracked_driver.write_text("TRACKED_DRIVER\n")
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()

    adapter = Path(
        forge_submit._build_driver_adapter(
            "python test.py",
            str(workspace),
        )
    )
    generated = Path(
        forge_submit._autogen_forge_driver(
            {"operation": "gemm"},
            str(workspace / "kernel.py"),
            workspace,
        )
    )

    assert {adapter.parent, generated.parent} == {workspace}
    assert adapter.name.startswith(".forge_driver_")
    assert generated.name.startswith(".forge_driver_")
    assert adapter.name != generated.name
    assert adapter.is_file() and generated.is_file()
    # The tracked file is untouched and the attempt dir stays clean.
    assert tracked_driver.read_text() == "TRACKED_DRIVER\n"
    assert list(output_dir.iterdir()) == []
    assert sorted(path.name for path in workspace.iterdir()) == sorted(
        ["forge_driver.py", adapter.name, generated.name]
    )


def test_finalize_removes_staged_drivers_from_the_live_repo(tmp_path):
    """In-place cleanup deletes every ``.forge_driver_*`` it staged."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    tracked = workspace / "kernel.py"
    tracked.write_text("TRACKED\n")
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    driver = Path(forge_submit._write_generated_driver(workspace, "print('drive')\n"))
    stray = Path(forge_submit._write_generated_driver(workspace, "print('stray')\n"))

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info=None,
        driver=str(driver),
        workspace=str(workspace),
        output_dir=output_dir,
        branch="forge/test",
        nogit_scratch=False,
    )

    assert not driver.exists()
    assert not stray.exists()
    assert tracked.read_text() == "TRACKED\n"


def test_finalize_keeps_both_campaigns_when_the_destination_is_populated(tmp_path):
    """A populated ``forge_experiments`` no longer aborts in-place cleanup.

    ``--experiments-dir`` points at ``output_dir/forge_experiments`` and mkdir's
    it, so the destination always exists; only real artifacts force a rename.
    """
    workspace = tmp_path / "repo"
    (workspace / "forge_experiments").mkdir(parents=True)
    (workspace / "forge_experiments" / "best_result.json").write_text("{}\n")
    output_dir = tmp_path / "attempt"
    (output_dir / "forge_experiments").mkdir(parents=True)
    (output_dir / "forge_experiments" / "campaign.json").write_text("{}\n")

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info=None,
        driver="",
        workspace=str(workspace),
        output_dir=output_dir,
        branch="forge/test",
        nogit_scratch=False,
    )

    assert (output_dir / "forge_experiments" / "campaign.json").is_file()
    assert (output_dir / "forge_experiments_workspace_1" / "best_result.json").is_file()
    assert not (workspace / "forge_experiments").exists()


def test_finalize_reuses_an_empty_destination_for_the_campaign(tmp_path):
    """The common case: the mkdir'd empty destination receives the campaign."""
    workspace = tmp_path / "repo"
    (workspace / "forge_experiments").mkdir(parents=True)
    (workspace / "forge_experiments" / "best_result.json").write_text("{}\n")
    output_dir = tmp_path / "attempt"
    (output_dir / "forge_experiments").mkdir(parents=True)

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info=None,
        driver="",
        workspace=str(workspace),
        output_dir=output_dir,
        branch="forge/test",
        nogit_scratch=False,
    )

    assert (output_dir / "forge_experiments" / "best_result.json").is_file()
    assert not (workspace / "forge_experiments").exists()


def test_inplace_restore_returns_the_original_working_tree_bytes(tmp_path):
    """In-place mode edits the live repo, so restore must be byte-exact.

    ``_prepare_inplace`` snapshots pre-existing dirty content into a baseline
    commit, so the index/working-tree split is folded into "unstaged" -- what is
    guaranteed is that the *content* on disk is identical afterwards, the files
    are still dirty, and the repo is back on its original branch with no forge
    temp branch left behind.
    """
    repo, source = _make_repo(tmp_path)
    binary = repo / "payload.bin"
    binary.write_bytes(b"\x00BASELINE\xff")
    _git(repo, "add", "payload.bin")
    _git(repo, "commit", "-m", "add binary fixture")

    source.write_text("STAGED\n")
    binary.write_bytes(b"\x00STAGED\xff")
    _git(repo, "add", "kernel.py", "payload.bin")
    source.write_text("STAGED\nUNSTAGED\n")
    binary.write_bytes(b"\x00STAGED\xffUNSTAGED")

    branch = "forge/test/inplace-restore"
    prepared = forge_submit._prepare_inplace(str(source), str(repo), branch)
    assert prepared is not None
    workspace, kernel, restore = prepared
    try:
        Path(kernel).write_text("FORGE_EDIT\n")
        binary.write_bytes(b"\x00FORGE_EDIT\xff")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "forge edit")
    finally:
        forge_submit._restore_inplace(restore)

    assert _git(repo, "branch", "--show-current") == "main"
    assert _git(repo, "branch", "--list", branch) == ""
    assert source.read_text() == "STAGED\nUNSTAGED\n"
    assert binary.read_bytes() == b"\x00STAGED\xffUNSTAGED"
    # Still dirty -- the developer's uncommitted work was not committed away.
    status = _git(repo, "status", "--short")
    assert "kernel.py" in status
    assert "payload.bin" in status
    # ... and the index is back at the original HEAD (the pre-forge staged /
    # unstaged split is deliberately collapsed into unstaged by the baseline
    # snapshot, so nothing is silently left staged).
    assert _git(repo, "diff", "--cached") == ""


def test_cli_invocation_pins_the_forge_loop_contract(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    program = tmp_path / "program.md"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    program.write_text("# Task\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    captured = {}

    class FakeProcess:
        pid = 43210
        returncode = 0

        def communicate(self, timeout=None):
            captured["communicate_timeout"] = timeout
            payload = {
                "baseline_ms": 2.0,
                "best_ms": 1.0,
                "mean_case_speedup": 2.0,
                "search_start_mean_case_speedup": 1.0,
                "total_improved": True,
                "incremental_improved": True,
            }
            return f"__FORGE_RESULT__{json.dumps(payload)}__FORGE_RESULT__", ""

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "/forge/src")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    deadline = time.time() + 120.0
    outcome = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        shapes={"primary": {"M": 128}},
        snr_threshold=30.0,
        max_iters=8,
        max_hours=1.0,
        branch="forge/session/kernel",
        gpu_target="gfx950",
        fellow="triton-fellow",
        program_md_file=str(program),
        invocation_spec_file="",
        experiments_dir=experiments,
        forge_log=tmp_path / "forge.log",
        timeout_s=120,
        deadline_unix=deadline,
        experience_id="attempt-1",
        operator_name="vllm::logical_op",
        source_files=[str(kernel)],
        target_functions=["kernel_impl", "device_kernel"],
    )

    # The loop result is a named outcome; unpacking it as a bare tuple is what
    # silently broke the recovery channels before.
    assert forge_submit.ForgeLoopOutcome._fields == (
        "baseline_ms",
        "best_ms",
        "improved",
        "output",
        "error",
        "timed_out",
        "checkpoint",
        "pristine_baseline_ms",
        "search_start_ms",
        "improved_during_search",
        "structured_result",
        "mean_case_speedup",
        "search_start_mean_case_speedup",
        "total_improved",
        "incremental_improved",
    )
    assert (outcome.baseline_ms, outcome.best_ms, outcome.improved) == (2.0, 1.0, True)
    assert outcome.error is None
    assert outcome.timed_out is False
    assert outcome.checkpoint is None
    assert outcome.mean_case_speedup == 2.0
    assert outcome.total_improved is True

    command = captured["command"]
    assert command[:5] == [
        sys.executable,
        "-m",
        "kernel_agents.cli",
        "forge-loop",
        "--kernel",
    ]
    expected_flags = {
        "--kernel": str(kernel),
        "--driver": str(driver),
        "--workspace": str(workspace),
        "--shapes-json": json.dumps({"primary": {"M": 128}}),
        "--snr-threshold": "30.0",
        "--max-iters": "8",
        "--max-hours": "1.0",
        "--git-branch": "forge/session/kernel",
        "--gpu-target": "gfx950",
        "--fellow": "triton-fellow",
        "--experiments-dir": str(experiments),
        "--experiment-id": "hyperloom",
        "--experience-id": "attempt-1",
        "--deadline-unix": str(deadline),
        "--result-json": str(experiments.parent / "forge_cli_result.json"),
        "--program-md-file": str(program),
        "--operator-name": "vllm::logical_op",
        "--source-files": str(kernel),
        "--target-functions": "kernel_impl,device_kernel",
    }
    for flag, value in expected_flags.items():
        assert flag in command, flag
        assert command[command.index(flag) + 1] == value, flag
    assert "--kernel-kind" not in command

    assert captured["env"]["GPU_TARGET"] == "gfx950"
    assert captured["env"]["PYTHONPATH"].startswith("/forge/src")
    # Isolated process group -- the timeout kill signals the group, not just pid.
    assert captured["popen_kwargs"]["start_new_session"] is True
    assert captured["popen_kwargs"]["stdout"] is subprocess.PIPE
    assert captured["popen_kwargs"]["stderr"] is subprocess.PIPE
    assert captured["popen_kwargs"]["cwd"] == str(workspace)
    # The subprocess wait is bounded by the absolute deadline, not by wall time
    # already spent before the loop started.
    assert 100.0 < captured["communicate_timeout"] <= 120.0


def test_generated_argv_matches_triton_wrapper_ck_and_flydsl_contracts(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    driver = workspace / "driver.py"
    driver.write_text("pass\n")
    direct = workspace / "direct.py"
    direct.write_text("@triton.jit\ndef direct_kernel(x):\n    return x\n")
    wrapper = workspace / "vllm" / "wrapper.py"
    wrapper.parent.mkdir()
    wrapper.write_text("def attention(x):\n    return x\n")
    aiter_impl = workspace / "aiter" / "attention.py"
    aiter_impl.parent.mkdir()
    aiter_impl.write_text(
        "@triton.jit\ndef attention_kernel(x):\n    return x\n"
    )
    ck_source = workspace / "aiter" / "gemm.cu"
    ck_source.write_text('__global__ void gemm_kernel() {}\n')
    fly_source = workspace / "flydsl" / "moe.py"
    fly_source.parent.mkdir()
    fly_source.write_text("@flydsl.jit\ndef moe_kernel(x):\n    return x\n")
    commands = []

    class FakeProcess:
        pid = 43210
        returncode = 0

        def communicate(self, timeout=None):
            payload = {"baseline_ms": 2.0, "best_ms": 2.0, "improved": False}
            return f"__FORGE_RESULT__{json.dumps(payload)}__FORGE_RESULT__", ""

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    cases = [
        {
            "candidate": {
                "operation": "custom::direct",
                "source_file": str(direct),
            },
            "source_type": "triton",
            "kernel": direct,
            "sources": [direct],
            "expected_framework": "",
            "expected_kind": "triton",
            "expected_fellow": "triton-fellow",
            "expected_symbols": ["direct_kernel"],
        },
        {
            "candidate": {
                "operation": "vllm::unified_attention",
                "source_file": str(wrapper),
                "kernel_sources": [str(aiter_impl)],
                "framework": "vllm",
            },
            "source_type": "python",
            "kernel": wrapper,
            "sources": [wrapper, aiter_impl],
            "expected_framework": "aiter",
            "expected_kind": "",
            "expected_fellow": "triton-fellow",
            "expected_symbols": ["attention_kernel"],
        },
        {
            "candidate": {
                "operation": "aiter::gemm",
                "source_file": str(ck_source),
                "kernel_kind": "aiter_ck",
            },
            "source_type": "hip_cpp",
            "kernel": ck_source,
            "sources": [ck_source],
            "expected_framework": "aiter",
            "expected_kind": "aiter_ck",
            "expected_fellow": "ck-fellow",
            "expected_symbols": ["gemm_kernel"],
        },
        {
            "candidate": {
                "operation": "pseudo_op::moe_flydsl_stage1",
                "source_file": str(fly_source),
            },
            "source_type": "flydsl",
            "kernel": fly_source,
            "sources": [fly_source],
            "expected_framework": "",
            "expected_kind": "flydsl",
            "expected_fellow": "flydsl-fellow",
            "expected_symbols": ["moe_kernel"],
        },
    ]

    for index, case in enumerate(cases):
        candidate = case["candidate"]
        kind = forge_submit._resolve_kernel_kind(
            case["source_type"],
            candidate.get("kernel_kind", ""),
        )
        source_values = [str(path.resolve()) for path in case["sources"]]
        symbols = forge_submit._stable_implementation_symbols(
            candidate,
            source_files=source_values,
        )
        framework = forge_submit._resolve_framework(
            candidate,
            str(case["kernel"]),
        )
        fellow = forge_submit._resolve_fellow(case["source_type"], kind)
        assert fellow is not None
        experiments = tmp_path / f"attempt-{index}" / "forge_experiments"
        experiments.mkdir(parents=True)
        forge_submit._run_loop_via_cli(
            worktree_kernel=str(case["kernel"]),
            driver=str(driver),
            workspace=str(workspace),
            shapes={},
            snr_threshold=30.0,
            max_iters=1,
            max_hours=1.0,
            branch=f"forge/test/{index}",
            gpu_target="gfx950",
            fellow=fellow,
            program_md_file="",
            invocation_spec_file="",
            experiments_dir=experiments,
            forge_log=tmp_path / f"forge-{index}.log",
            timeout_s=60,
            deadline_unix=time.time() + 60,
            operator_name=forge_submit._logical_operator(candidate),
            framework=framework,
            target_functions=symbols,
            source_files=source_values,
        )

        command = commands[-1]
        assert command[command.index("--operator-name") + 1] == (
            forge_submit._logical_operator(candidate)
        )
        assert command[command.index("--source-files") + 1] == ",".join(
            source_values
        )
        assert command[command.index("--fellow") + 1] == case["expected_fellow"]
        assert kind == case["expected_kind"]
        assert framework == case["expected_framework"]
        assert symbols == case["expected_symbols"]
        assert "--kernel-kind" not in command
        if framework:
            assert command[command.index("--framework") + 1] == framework
        else:
            assert "--framework" not in command
        if symbols:
            assert command[command.index("--target-functions") + 1] == ",".join(
                symbols
            )
        else:
            assert "--target-functions" not in command


def test_cli_timeout_recovers_only_this_run_s_checkpoint(tmp_path, monkeypatch):
    """A hard kill must yield THIS run's checkpoint, never a stale one.

    ``_run_loop_via_cli`` clears both recovery artifacts before launching, so a
    checkpoint returned after a kill can only have been written by the run that
    was killed.
    """
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    checkpoint_json = experiments / "hyperloom.json"
    result_json = experiments.parent / "forge_cli_result.json"
    # Artifacts left behind by a PREVIOUS campaign in the same output dir.
    checkpoint_json.write_text(
        json.dumps({"checkpoint": {"best_commit": "stale-commit"}})
    )
    result_json.write_text(json.dumps({"baseline_ms": 9.0, "best_ms": 9.0}))
    fresh = {"schema_version": 1, "state": "best_committed", "best_commit": "fresh"}

    class TimeoutPopen:
        pid = 43210
        returncode = None

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(["forge-loop"], timeout)

    def fake_terminate(_proc):
        # Mirrors the loop's KEEP callback landing before the SIGKILL.
        checkpoint_json.write_text(
            json.dumps({"experiment_id": "hyperloom", "checkpoint": fresh})
        )
        return "partial stdout", "partial stderr"

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", TimeoutPopen)
    monkeypatch.setattr(forge_submit, "_terminate_forge_process", fake_terminate)

    outcome = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        shapes={},
        snr_threshold=30.0,
        max_iters=8,
        max_hours=1.0,
        branch="forge/session/kernel",
        gpu_target="gfx950",
        fellow="triton-fellow",
        program_md_file="",
        invocation_spec_file="",
        experiments_dir=experiments,
        forge_log=tmp_path / "forge.log",
        timeout_s=10,
    )

    assert outcome.timed_out is True
    assert isinstance(outcome.error, RuntimeError)
    assert "10" in str(outcome.error)
    assert outcome.checkpoint == fresh
    assert "partial stdout" in outcome.output
    # The previous run's sidecar was cleared, so its numbers cannot leak in.
    assert not result_json.exists()
    assert (outcome.baseline_ms, outcome.best_ms, outcome.improved) == (None, None, False)


def test_forced_termination_escalates_to_sigkill_and_keeps_partial_output(monkeypatch):
    """SIGTERM, then SIGKILL to the whole group once the grace period expires."""
    signals = []
    descendants = [(9001, 4242)]
    killed = []

    class FakeProcess:
        pid = 43210
        returncode = None

        def __init__(self):
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(
                    ["forge-loop"],
                    timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            self.returncode = -signal.SIGKILL
            return "partial stdout\nfinal stdout", "partial stderr\nfinal stderr"

        def kill(self):
            raise AssertionError("killpg succeeded; direct fallback is invalid")

    process = FakeProcess()
    monkeypatch.setattr(
        forge_submit,
        "_descendant_processes",
        lambda _pid: list(descendants),
    )
    monkeypatch.setattr(
        forge_submit.os,
        "killpg",
        lambda process_group, sent_signal: signals.append(
            (process_group, sent_signal)
        ),
    )
    monkeypatch.setattr(
        forge_submit,
        "_signal_processes",
        lambda procs, sent_signal: killed.append((list(procs), sent_signal)),
    )

    stdout, stderr = forge_submit._terminate_forge_process(process, grace_sec=0.1)

    assert stdout == "partial stdout\nfinal stdout"
    assert stderr == "partial stderr\nfinal stderr"
    # SIGTERM, SIGKILL once the grace period expires, then a final sweep of the
    # group after the parent is reaped (a re-parented fellow child would
    # otherwise survive its parent).
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
        (process.pid, signal.SIGKILL),
    ]
    # The escalation also sweeps captured descendants, so a fellow's own
    # grandchildren cannot outlive the group.
    assert killed == [(descendants, signal.SIGKILL)]


def _grandchild_running(pid: int) -> bool:
    """True only while ``pid`` exists and is not a reaped zombie.

    Reads ``/proc/<pid>/stat`` defensively: the file can vanish between an
    existence check and the read once the kernel reaps the process, so a
    missing file (FileNotFoundError / ProcessLookupError) means "not running"
    -- the success condition here -- rather than a test error. Guarding the
    read this way removes a TOCTOU race that made the assertion flaky under
    load (it surfaced as ``FileNotFoundError: /proc/<pid>/stat`` on CI).
    """
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def test_forced_termination_leaves_no_running_grandchild(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
        "print('child-started', flush=True)\n"
        "time.sleep(60)\n"
    )
    child_pid = None

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(tmp_path),
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if child_pid_file.is_file() and child_pid_file.read_text().strip():
                break
            time.sleep(0.05)
        else:
            pytest.fail("spawned child never reported its pid")
        child_pid = int(child_pid_file.read_text())

        with pytest.raises(subprocess.TimeoutExpired):
            proc.communicate(timeout=1.0)
        stdout, _stderr = forge_submit._terminate_forge_process(proc, grace_sec=2)

        assert "child-started" in stdout
        # Generous: a loaded CI runner can take seconds to reap the group after
        # SIGKILL. The assertion is still "the grandchild must die" -- only the
        # patience is relaxed, so a real leak still fails here.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not _grandchild_running(child_pid):
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"child process {child_pid} survived process-group timeout")
    finally:
        if child_pid is not None and _grandchild_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except OSError:
                pass  # child already reaped between the check and the kill
        if proc.poll() is None:
            proc.kill()


def test_disagreeing_recovery_channels_keep_the_published_manifest(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Both channels validated but naming different commits is a forge bug.

    The published manifest is rewritten on every KEEP, the checkpoint only on
    the last KEEP callback, so the manifest wins -- loudly, never silently.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-disagree"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        base_commit = _git(workspace, "rev-parse", "HEAD")
        kernel.write_text("PUBLISHED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "published best")
        published_commit = _git(workspace, "rev-parse", "HEAD")
        kernel.write_text("CHECKPOINTED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "checkpointed best")
        checkpointed_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(published_commit))
        )
        kernel.write_text("UNVALIDATED_MID_ITERATION\n")
        captured.update(
            published_commit=published_commit,
            checkpointed_commit=checkpointed_commit,
        )
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=_checkpoint(base_commit, checkpointed_commit),
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    with caplog.at_level(logging.WARNING, logger=forge_submit.log.name):
        result = forge_submit.submit(
            source_file=str(source),
            prompt_file=prompt,
            output_dir=output_dir,
            test_command="python -c 'print(\"allclose: True\")'",
            source_type="triton",
            candidate={"platform": "mi355x"},
            timeout_s=10,
            kernel_repo=str(repo),
        )

    assert result["returncode"] == 0
    assert result["best_commit"] == captured["published_commit"]
    assert result["best_commit"] != captured["checkpointed_commit"]
    optimized = output_dir / "optimized_versions" / "v1_forge.py"
    assert optimized.read_text() == "PUBLISHED_BEST\n"
    # The warning only fires when BOTH channels validated, which is what makes
    # this a precedence assertion rather than a "checkpoint was ignored" one.
    assert "disagree" in caplog.text
    assert "keeping the published manifest" in caplog.text


def test_checkpoint_naming_an_unavailable_commit_is_rejected(tmp_path):
    """A checkpoint pointing at a commit that is not in the workspace is junk.

    Trusting it would export the current (unvalidated) worktree under a commit
    that never existed. Rejection must also leave the workspace untouched.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    branch = "forge/session/unavailable-commit"
    prepared = forge_submit._prepare_worktree(
        str(source),
        str(repo),
        output_dir,
        branch,
    )
    assert prepared is not None
    workspace, kernel, base_commit = prepared

    assert (
        forge_submit._validated_forge_checkpoint(
            _checkpoint(base_commit, "f" * 40),
            workspace=workspace,
            base_commit=base_commit,
            shapes={},
        )
        is None
    )
    # The same rejection holds for the published-manifest channel.
    assert (
        forge_submit._validated_forge_best_result(
            _published_manifest("f" * 40),
            workspace=workspace,
            base_commit=base_commit,
        )
        is None
    )

    assert _git(workspace, "rev-parse", "HEAD") == base_commit
    assert Path(kernel).read_text() == "BASELINE\n"


def test_submit_timeout_salvages_only_the_validated_best_commit(
    tmp_path,
    monkeypatch,
):
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-timeout"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(best_commit))
        )
        # The kill lands mid-iteration, so the working tree holds an unvalidated
        # candidate that must never reach the exported artifacts.
        kernel.write_text("UNVERIFIED_MID_ITERATION\n")
        _git(workspace, "add", "-u")
        captured.update(
            workspace=workspace,
            kernel=kernel,
            best_commit=best_commit,
            branch=_git(workspace, "branch", "--show-current"),
        )
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={"platform": "mi355x"},
        timeout_s=10,
        kernel_repo=str(repo),
    )

    optimized = output_dir / "optimized_versions" / "v1_forge.py"
    assert result["returncode"] == 0
    assert result["timed_out"] is True
    assert result["salvaged"] is True
    assert result["best_commit"] == captured["best_commit"]
    assert result["cli_workspace"] == str(output_dir)
    assert result["output_dir"] == str(output_dir)
    assert result["checkpoint_path"] == str(
        output_dir / "forge_experiments" / "hyperloom.json"
    )
    assert optimized.read_text() == "VERIFIED_BEST\n"

    patch = (output_dir / "optimized_versions" / "forge.patch").read_text()
    assert "VERIFIED_BEST" in patch
    assert "UNVERIFIED_MID_ITERATION" not in patch
    changed = (output_dir / "optimized_versions" / "changed_files.txt").read_text()
    assert changed.split() == ["kernel.py"]
    report = (output_dir / "optimization_report.md").read_text()
    assert "[micro_speedup] 1.5000x" in report
    assert "[correctness] pass" in report

    # Campaign state lives under the output dir, never inside the live repo, and
    # the isolated worktree + temp branch are retained afterwards for inspection.
    assert (output_dir / "forge_experiments").is_dir()
    assert not (repo / "forge_experiments").exists()
    assert captured["workspace"].exists()
    assert _git(repo, "branch", "--list", captured["branch"])
    assert source.read_text() == "BASELINE\n"


def test_submit_timeout_export_failure_writes_no_promotable_artifacts(
    tmp_path,
    monkeypatch,
):
    """A recovery that cannot be exported must not leave a promotable report."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-export-failure"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {"reports": 0}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(best_commit))
        )
        kernel.write_text("UNVERIFIED_MID_ITERATION\n")
        captured["kernel"] = kernel
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    def forbidden_report(*_args, **_kwargs):
        captured["reports"] += 1
        raise AssertionError("failed recovery must not write a promotable report")

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)
    monkeypatch.setattr(
        forge_submit,
        "_export_best_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("validated best commit has no exportable source diff")
        ),
    )
    monkeypatch.setattr(forge_submit, "_write_report", forbidden_report)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={"platform": "mi355x"},
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 1
    assert "no exportable source diff" in result["stderr_tail"]
    assert captured["reports"] == 0
    assert "best_commit" not in result
    assert not (output_dir / "optimization_report.md").exists()
    assert not (output_dir / "optimized_versions").exists()


def test_submit_timeout_without_validated_recovery_discards_measurements(
    tmp_path,
    monkeypatch,
):
    """No validated commit -> the sidecar's numbers are not evidence.

    After a forced termination only a validated commit may produce a passing
    report; the loop's self-reported baseline/best are dropped so nothing
    downstream can promote an unverified kernel.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-unrecoverable"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        captured.update(kwargs)
        Path(kwargs["worktree_kernel"]).write_text("UNVERIFIED_MID_ITERATION\n")
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={"platform": "mi355x"},
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 1
    assert result["timed_out"] is True
    assert result["salvaged"] is False
    assert "timed out without recoverable checkpoint" in result["stderr_tail"]
    assert "best_commit" not in result
    assert not (output_dir / "optimized_versions").exists()
    assert not (output_dir / "optimization_report.md").exists()

    # forge-loop rejects a soft budget below its own one-hour minimum, so submit
    # floors --max-hours there while still hard-killing at timeout_s.
    assert captured["max_hours"] >= forge_submit._FORGE_MIN_BUDGET_SEC / 3600.0
    assert captured["timeout_s"] == 10


def test_submit_non_timeout_error_fails_and_uses_unique_retained_branch(
    tmp_path,
    monkeypatch,
):
    repo, source = _make_repo(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    workspaces = []

    def fake_loop(**kwargs):
        workspaces.append(Path(kwargs["workspace"]))
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=None,
            best_ms=None,
            improved=False,
            output="failed",
            error=RuntimeError("loop failed"),
            timed_out=False,
            checkpoint=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    results = [
        forge_submit.submit(
            source_file=str(source),
            prompt_file=prompt,
            output_dir=tmp_path / "results" / f"attempt-{attempt}",
            test_command="python -c 'print(\"allclose: True\")'",
            source_type="triton",
            candidate={"platform": "mi355x"},
            timeout_s=10,
            kernel_repo=str(repo),
        )
        for attempt in (1, 2)
    ]

    branches = [_git(workspace, "branch", "--show-current") for workspace in workspaces]
    assert [result["returncode"] for result in results] == [1, 1]
    assert all("forge cli loop failed" in result["stderr_tail"] for result in results)
    # Each attempt retains its isolated worktree for inspection under its own
    # output dir, on a unique Forge branch, so a repeat run on the same repo is
    # never blocked by (or reuses) a prior attempt.
    assert len(workspaces) == 2
    assert all(workspace.is_dir() for workspace in workspaces)
    assert len(set(branches)) == 2
    assert all(branch.startswith("forge/") for branch in branches)
    assert source.read_text() == "BASELINE\n"


def test_finalization_failure_does_not_swallow_the_forge_result(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Workspace cleanup is best-effort; it must never eat a salvaged result."""
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-cleanup-failure"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(best_commit))
        )
        captured["best_commit"] = best_commit
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)
    monkeypatch.setattr(
        forge_submit,
        "_finalize_forge_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    with caplog.at_level(logging.ERROR, logger=forge_submit.log.name):
        result = forge_submit.submit(
            source_file=str(source),
            prompt_file=prompt,
            output_dir=output_dir,
            test_command="python -c 'print(\"allclose: True\")'",
            source_type="triton",
            candidate={"platform": "mi355x"},
            timeout_s=10,
            kernel_repo=str(repo),
        )

    assert result["returncode"] == 0
    assert result["salvaged"] is True
    assert result["best_commit"] == captured["best_commit"]
    assert (
        output_dir / "optimized_versions" / "v1_forge.py"
    ).read_text() == "VERIFIED_BEST\n"
    assert "forge workspace finalization failed" in caplog.text


def test_nogit_scratch_bootstraps_a_committable_scratch_repo(tmp_path):
    """Non-git sources get a scratch git repo so keep/revert works at all."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "kernel.py"
    source.write_text("pass\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    branch = "forge/session/kernel-attempt"

    prepared = forge_submit._prepare_worktree_nogit(
        str(source),
        str(source_root),
        output_dir,
        branch,
    )

    assert prepared is not None
    workspace, kernel, base_commit = prepared
    assert Path(workspace) == output_dir / "worktree"
    assert Path(kernel).read_text() == "pass\n"
    assert _git(workspace, "rev-parse", "HEAD") == base_commit
    assert _git(workspace, "log", "--oneline").count("\n") == 0  # single baseline
    assert _git(workspace, "config", "user.name") == "forge-bot"

    # The scratch repo must support the loop's commit/revert cycle.
    Path(kernel).write_text("OPTIMIZED\n")
    _git(workspace, "add", "-u")
    _git(workspace, "commit", "-m", "iter1")
    assert _git(workspace, "rev-parse", "HEAD") != base_commit
    _git(workspace, "reset", "--hard", base_commit)
    assert Path(kernel).read_text() == "pass\n"

    # The live (non-git) source tree is never converted into a repo.
    assert not (source_root / ".git").exists()
    assert source.read_text() == "pass\n"


def test_inplace_campaign_state_never_lands_in_the_live_repo(tmp_path, monkeypatch):
    """An in-place campaign must leave the developer's checkout as it found it.

    In-place mode hands forge the *live* repo as its workspace, so every
    campaign-owned path that can live outside it -- ``--experiments-dir``, the
    CLI sidecar -- is addressed at the output dir up front. The generated
    driver is the one exception: the long-horizon CLI resolves ``--driver``
    relative to ``--workspace`` and rejects anything outside it, so it is
    staged in the checkout under a hidden ``.forge_driver_`` name and removed
    during finalization. Pin both halves: what the loop is handed points at the
    output dir (driver aside), and after the run the repo's tracked state,
    branch and temp-branch set are exactly what they were before forge started.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-inplace"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(best_commit))
        )
        kernel.write_text("UNVERIFIED_MID_ITERATION\n")
        captured.update(
            workspace=workspace,
            kernel=kernel,
            driver=Path(kwargs["driver"]),
            experiments_dir=Path(kwargs["experiments_dir"]),
            branch=_git(workspace, "branch", "--show-current"),
            best_commit=best_commit,
        )
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_needs_inplace", lambda _repo: True)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python -c 'print(\"allclose: True\")'",
        source_type="triton",
        candidate={"platform": "mi355x"},
        timeout_s=10,
        kernel_repo=str(repo),
    )

    # The run really was in-place: forge edited the live checkout, not a copy.
    assert captured["workspace"] == repo
    assert captured["kernel"] == source
    assert captured["branch"].startswith("forge/")

    assert result["returncode"] == 0
    assert result["salvaged"] is True
    assert result["best_commit"] == captured["best_commit"]
    # Every campaign-owned path the loop was handed lives under the output dir.
    assert captured["experiments_dir"] == output_dir / "forge_experiments"
    # The driver is the CLI-mandated exception: inside the workspace, hidden.
    assert captured["driver"].parent == repo
    assert captured["driver"].name.startswith(".forge_driver_")
    assert result["checkpoint_path"] == str(
        output_dir / "forge_experiments" / "hyperloom.json"
    )
    assert (output_dir / "forge_experiments").is_dir()
    assert (
        output_dir / "optimized_versions" / "v1_forge.py"
    ).read_text() == "VERIFIED_BEST\n"

    # ... and the live checkout is handed back untouched: original branch, no
    # forge temp branch, pre-forge bytes, nothing tracked left dirty.
    assert _git(repo, "branch", "--show-current") == "main"
    assert _git(repo, "branch", "--list", captured["branch"]) == ""
    assert source.read_text() == "BASELINE\n"
    assert (repo / "forge_driver.py").read_text() == "TRACKED_DRIVER\n"
    assert _git(repo, "status", "--short", "--untracked-files=no") == ""
    # No generated driver or CLI sidecar was left behind in the checkout.
    assert not (repo / "forge_driver_adapter.py").exists()
    assert not (repo / "forge_task_driver.py").exists()
    assert not (repo / "forge_cli_result.json").exists()
    assert not captured["driver"].exists()
    assert list(repo.glob(".forge_driver_*.py")) == []


def test_inplace_restore_failure_is_surfaced_without_losing_the_result(
    tmp_path,
    monkeypatch,
    caplog,
):
    """In-place cleanup touches the live repo, so its failure must be loud.

    Restore is still attempted, the failure is reported rather than swallowed,
    and the salvaged forge result survives it.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-inplace-cleanup-failure"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}
    restore_attempts = []

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        (experiments / "best_result.json").write_text(
            json.dumps(_published_manifest(best_commit))
        )
        captured["best_commit"] = best_commit
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=None,
        )

    def failing_restore(restore_info):
        restore_attempts.append(restore_info)
        raise OSError("in-place restore failed")

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_needs_inplace", lambda _repo: True)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)
    monkeypatch.setattr(forge_submit, "_restore_inplace", failing_restore)

    with caplog.at_level(logging.ERROR, logger=forge_submit.log.name):
        result = forge_submit.submit(
            source_file=str(source),
            prompt_file=prompt,
            output_dir=output_dir,
            test_command="python -c 'print(\"allclose: True\")'",
            source_type="triton",
            candidate={"platform": "mi355x"},
            timeout_s=10,
            kernel_repo=str(repo),
        )

    # Restore was attempted with this run's snapshot (not skipped) ...
    assert len(restore_attempts) == 1
    assert restore_attempts[0]["repo"] == str(repo)
    assert restore_attempts[0]["orig_branch"] == "main"
    # ... its failure was surfaced ...
    assert "forge workspace finalization failed" in caplog.text
    # ... and it did not eat the salvaged result.
    assert result["returncode"] == 0
    assert result["salvaged"] is True
    assert result["best_commit"] == captured["best_commit"]
    assert (
        output_dir / "optimized_versions" / "v1_forge.py"
    ).read_text() == "VERIFIED_BEST\n"


def test_retained_worktree_collision_skips_without_delete_or_nogit_fallback(
    tmp_path,
    monkeypatch,
):
    """A retained worktree at the campaign path skips cleanly, never clobbering it.

    ``output_dir/worktree`` is the fixed campaign workspace path and prior
    attempts are retained for inspection. A collision must skip safely (rc 2)
    without deleting the retained attempt and without reinterpreting the path as
    a no-git scratch workspace.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "collision"
    retained = output_dir / "worktree"
    retained.mkdir(parents=True)
    marker = retained / "keep.txt"
    marker.write_text("retained\n")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    monkeypatch.setattr(
        forge_submit,
        "_prepare_worktree_nogit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not fall through to no-git scratch")
        ),
    )

    result = forge_submit.submit(
        source_file=str(source),
        prompt_file=prompt,
        output_dir=output_dir,
        test_command="python test.py",
        source_type="triton",
        candidate={"platform": "mi355x"},
        timeout_s=10,
        kernel_repo=str(repo),
    )

    assert result["returncode"] == 2
    assert result["skipped"] is True
    assert marker.read_text() == "retained\n"


def test_unclearable_stale_artifact_aborts_before_starting_a_campaign(
    tmp_path,
    monkeypatch,
):
    """Every campaign starts fresh, or it does not start at all.

    ``_run_loop_via_cli`` clears the two recovery artifacts up front so anything
    found afterwards provably belongs to this run. If one cannot be cleared the
    launch is refused -- silently proceeding would let a previous campaign's
    checkpoint be salvaged as if this run had produced it.
    """
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    result_json = experiments.parent / "forge_cli_result.json"
    result_json.write_text(json.dumps({"baseline_ms": 9.0, "best_ms": 9.0}))
    # Something at the checkpoint path that unlink() cannot remove.
    unclearable = experiments / "hyperloom.json"
    unclearable.mkdir()
    (unclearable / "blocker.txt").write_text("stale\n")

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("a campaign must not start on unclearable artifacts")

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", forbidden_popen)

    with pytest.raises(RuntimeError) as excinfo:
        forge_submit._run_loop_via_cli(
            worktree_kernel=str(kernel),
            driver=str(driver),
            workspace=str(workspace),
            shapes={},
            snr_threshold=30.0,
            max_iters=8,
            max_hours=1.0,
            branch="forge/session/kernel",
            gpu_target="gfx950",
            fellow="triton-fellow",
            program_md_file="",
            invocation_spec_file="",
            experiments_dir=experiments,
            forge_log=tmp_path / "forge.log",
            timeout_s=10,
        )

    assert "stale Forge recovery artifact" in str(excinfo.value)
    assert str(unclearable) in str(excinfo.value)
    # The refusal is an abort, not a partial start: nothing ran.
    assert not (tmp_path / "forge.log").exists()
    assert (unclearable / "blocker.txt").is_file()


def test_same_iteration_recovery_conflict_resolves_wholly_to_the_manifest(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Two validated bests claiming the same iteration must not be blended.

    Both channels can name a best for iteration N; when they name different
    commits one of them is stale. The published manifest wins *entirely* --
    commit AND measurements -- so the exported artifacts, the reported speedup
    and the returned commit all describe one coherent result rather than a mix.
    """
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-same-iteration"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        base_commit = _git(workspace, "rev-parse", "HEAD")
        kernel.write_text("PUBLISHED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "published best")
        published_commit = _git(workspace, "rev-parse", "HEAD")
        kernel.write_text("CHECKPOINTED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "checkpointed best")
        checkpointed_commit = _git(workspace, "rev-parse", "HEAD")
        experiments = workspace / "forge_experiments"
        experiments.mkdir(parents=True, exist_ok=True)
        # Same iteration, different commit, different timings.
        (experiments / "best_result.json").write_text(
            json.dumps(
                _published_manifest(
                    published_commit,
                    iteration=2,
                    baseline_wall_ms=3.0,
                    best_wall_ms=2.0,
                )
            )
        )
        kernel.write_text("UNVALIDATED_MID_ITERATION\n")
        captured.update(
            published_commit=published_commit,
            checkpointed_commit=checkpointed_commit,
        )
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=3.0,
            best_ms=2.0,
            improved=True,
            output="partial output",
            error=RuntimeError("forge-loop timed out after 10s"),
            timed_out=True,
            checkpoint=_checkpoint(
                base_commit,
                checkpointed_commit,
                iteration=2,
                baseline_ms=3.0,
                best_ms=1.5,
            ),
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    with caplog.at_level(logging.WARNING, logger=forge_submit.log.name):
        result = forge_submit.submit(
            source_file=str(source),
            prompt_file=prompt,
            output_dir=output_dir,
            test_command="python -c 'print(\"allclose: True\")'",
            source_type="triton",
            candidate={"platform": "mi355x"},
            timeout_s=10,
            kernel_repo=str(repo),
        )

    # The warning only fires when BOTH channels validated, so reaching it proves
    # the conflict was real and not a one-sided rejection.
    assert "disagree" in caplog.text
    assert captured["published_commit"][:12] in caplog.text
    assert captured["checkpointed_commit"][:12] in caplog.text

    assert result["returncode"] == 0
    assert result["best_commit"] == captured["published_commit"]
    assert result["best_commit"] != captured["checkpointed_commit"]
    assert (
        output_dir / "optimized_versions" / "v1_forge.py"
    ).read_text() == "PUBLISHED_BEST\n"

    # The manifest's own numbers are reported -- the checkpoint's faster
    # best_ms=1.5 (a 2.0x claim) is not merged in behind the manifest's commit.
    report = (output_dir / "optimization_report.md").read_text()
    assert "[micro_speedup] 1.5000x" in report
    assert "baseline_ms=3.0000 selected_ms=2.0000" in report
    assert "2.0000x" not in report
    assert "best_ms=1.5000" not in report


def test_trace_reads_llm_usage_from_the_cli_sidecar(tmp_path):
    """The loop runs out-of-process, so its cost is only recoverable on disk."""
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()

    assert forge_submit._forge_trace_from_sidecar(output_dir) == (None, None)

    (output_dir / "forge_cli_result.json").write_text(
        json.dumps(
            {
                "baseline_ms": 3.0,
                "llm_usage": {"input_tokens": 10, "output_tokens": 3},
                "steps": {"steps": [{"name": "baseline"}]},
            }
        )
    )

    usage, steps = forge_submit._forge_trace_from_sidecar(output_dir)

    assert usage == {"input_tokens": 10, "output_tokens": 3}
    assert steps == {"steps": [{"name": "baseline"}]}

    # A usage block with no canonical token counter is not a usage block.
    (output_dir / "forge_cli_result.json").write_text(
        json.dumps({"llm_usage": {"calls": 3}, "steps": {}})
    )
    assert forge_submit._forge_trace_from_sidecar(output_dir) == (None, None)


def test_non_inplace_finalization_retains_worktree(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    monkeypatch.setattr(
        forge_submit,
        "_remove_worktree",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must retain worktree")),
    )

    forge_submit._finalize_forge_workspace(
        inplace=False,
        restore_info=None,
        driver="",
        workspace=str(workspace),
        output_dir=tmp_path,
        branch="forge/test",
        nogit_scratch=False,
    )

    assert workspace.is_dir()


def test_inplace_finalization_moves_campaign_out_of_live_repo(tmp_path, monkeypatch):
    workspace = tmp_path / "live-repo"
    campaign = workspace / "forge_experiments"
    campaign.mkdir(parents=True)
    (campaign / "run_state.json").write_text("{}")
    driver = workspace / ".forge_driver_123.py"
    driver.write_text("pass\n")
    restored = []
    monkeypatch.setattr(
        forge_submit,
        "_restore_inplace",
        lambda restore: restored.append(restore),
    )

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info={"repo": str(workspace)},
        driver=str(driver),
        workspace=str(workspace),
        output_dir=tmp_path / "output",
        branch="forge/test",
        nogit_scratch=False,
    )

    assert not campaign.exists()
    assert not driver.exists()
    assert (tmp_path / "output" / "forge_experiments" / "run_state.json").is_file()
    assert restored == [{"repo": str(workspace)}]


def test_inplace_finalization_restores_then_raises_on_cleanup_failure(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "live-repo"
    campaign = workspace / "forge_experiments"
    campaign.mkdir(parents=True)
    driver = workspace / ".forge_driver_failure.py"
    driver.write_text("pass\n")
    restored = []
    monkeypatch.setattr(
        forge_submit.shutil,
        "move",
        lambda *_args: (_ for _ in ()).throw(OSError("move failed")),
    )
    monkeypatch.setattr(
        forge_submit,
        "_restore_inplace",
        lambda restore: restored.append(restore),
    )

    with pytest.raises(RuntimeError, match="in-place workspace cleanup failed"):
        forge_submit._finalize_forge_workspace(
            inplace=True,
            restore_info={"repo": str(workspace)},
            driver=str(driver),
            workspace=str(workspace),
            output_dir=tmp_path / "output",
            branch="forge/test",
            nogit_scratch=False,
        )

    assert restored == [{"repo": str(workspace)}]
    assert campaign.is_dir()
    assert not driver.exists()


def test_nogit_scratch_uses_supplied_non_main_branch(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "kernel.py"
    source.write_text("pass\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    branch = "forge/session/kernel-attempt"

    prepared = forge_submit._prepare_worktree_nogit(
        str(source),
        str(source_root),
        output_dir,
        branch,
    )

    assert prepared is not None
    workspace, _kernel, _base = prepared
    assert _git(workspace, "branch", "--show-current") == branch


def _capabilities_payload(**overrides) -> dict:
    payload = {
        "protocol_versions": [1],
        "artifact_schema_versions": [2],
        "driver_contract_versions": [1],
        "frameworks": ["aiter", "sglang", "vllm"],
        "result_sentinel": "__FORGE_RESULT__",
    }
    payload.update(overrides)
    return payload


def _stub_capability_process(monkeypatch, *, stdout: str, returncode: int = 0) -> dict:
    captured: dict = {"calls": 0}

    def fake_run(command, **kwargs):
        captured["calls"] += 1
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(_flydsl_rewrite.subprocess, "run", fake_run)
    return captured


class _RecordingProbe:
    """Stand-in for the producer probe that counts how often it is consulted."""

    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        return self.capabilities


_SUPPORTED_CAPABILITIES = _flydsl_rewrite.RewriteCapabilities(
    True,
    "capability_ok",
    "",
    ("aiter", "sglang", "vllm"),
    (1,),
)


def _rewrite_driver_contract(**overrides) -> dict:
    contract = {
        "contract_version": 1,
        "operator_family": "gemm",
        "logical_operator": "vllm::fused_gemm",
        "entry_symbols": ["matmul"],
        "case_selector_key": "CASE_ID",
        "cases": [
            {
                "case_id": "case_001",
                "selector": {"CASE_ID": "case_001"},
                "inputs": [
                    {"shape": [64, 32], "dtype": "fp16"},
                    {"shape": [32, 16], "dtype": "fp16"},
                ],
            }
        ],
    }
    contract.update(overrides)
    return contract


def _rewrite_route_kwargs(tmp_path, **overrides) -> dict:
    workspace = tmp_path / "worktree"
    kernel = workspace / "vllm" / "fused_gemm.py"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("TRITON\n")
    kwargs = {
        "candidate": {"name": "fused_gemm", "source_symbol": "matmul"},
        "source_type": "triton",
        "kernel_kind": "triton",
        "logical_operator": "vllm::fused_gemm",
        "source_kernel": str(kernel),
        "workspace": str(workspace),
        "implementation_sources": [str(kernel)],
        "implementation_symbols": ["matmul"],
        "framework": "vllm",
        "gpu_target": "gfx942",
        "driver_contract": _rewrite_driver_contract(),
        "shape_cases": [{"M": 8, "N": 16}],
        "shapes": {"M": 8},
        "branch": "forge/session/fused-gemm-abc123def456",
        "attempt_id": "attempt-1",
        "timeout_s": 7200,
    }
    kwargs.update(overrides)
    return kwargs


def test_capability_probe_reads_the_declared_rewrite_contract(monkeypatch):
    captured = _stub_capability_process(
        monkeypatch,
        stdout="loading kernel_agents...\n" + json.dumps(_capabilities_payload()) + "\ndone\n",
    )

    capabilities = _flydsl_rewrite.probe_capabilities(forge_root="/forge/src")

    assert capabilities.supported is True
    assert capabilities.reason == "capability_ok"
    assert capabilities.frameworks == ("aiter", "sglang", "vllm")
    # The flag is an eager short-circuit; nothing else may be guessed onto it.
    assert captured["command"] == [
        sys.executable,
        "-m",
        "kernel_agents.cli",
        "forge-rewrite-by-flydsl",
        "--capabilities-json",
    ]
    assert captured["env"]["PYTHONPATH"].startswith("/forge/src")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"protocol_versions": [2]}, "capability_protocol_unsupported"),
        ({"artifact_schema_versions": [1]}, "capability_artifact_schema_unsupported"),
        ({"driver_contract_versions": [2]}, "capability_driver_contract_unsupported"),
        ({"result_sentinel": "__FORGE_REWRITE_RESULT__"}, "capability_sentinel_mismatch"),
        ({"frameworks": []}, "capability_frameworks_missing"),
    ],
)
def test_capability_probe_rejects_an_incompatible_producer(monkeypatch, overrides, reason):
    _stub_capability_process(monkeypatch, stdout=json.dumps(_capabilities_payload(**overrides)))

    capabilities = _flydsl_rewrite.probe_capabilities()

    assert capabilities.supported is False
    assert capabilities.reason == reason


def test_capability_probe_reports_a_producer_that_rejects_the_flag(monkeypatch):
    _stub_capability_process(
        monkeypatch,
        stdout="Error: No such option '--capabilities-json'. Did you mean '--shapes-json'?",
        returncode=2,
    )

    capabilities = _flydsl_rewrite.probe_capabilities()

    assert capabilities.supported is False
    assert capabilities.reason == "capability_probe_failed"
    assert "rc=2" in capabilities.detail


def test_capability_probe_answers_once_per_process(monkeypatch):
    captured = _stub_capability_process(monkeypatch, stdout=json.dumps(_capabilities_payload()))

    first = _flydsl_rewrite.probe_capabilities()
    second = _flydsl_rewrite.probe_capabilities()

    assert first == second
    assert captured["calls"] == 1


def test_rewrite_route_needs_the_explicit_switch(tmp_path, monkeypatch):
    monkeypatch.delenv(_flydsl_rewrite.REWRITE_ENV, raising=False)
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path),
    )

    assert decision.eligible is False
    assert decision.reason == "route_disabled"
    assert probe.calls == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source_type": "hip_cpp", "kernel_kind": ""}, "source_type_unsupported"),
        ({"kernel_kind": "flydsl"}, "already_flydsl_source"),
        ({"kernel_kind": "aiter_asm"}, "prebuilt_binary_unsupported"),
        ({"logical_operator": "vllm::all_reduce"}, "collective_unsupported"),
        ({"framework": ""}, "framework_unsupported"),
        ({"framework": "torch"}, "framework_unsupported"),
        ({"timeout_s": 3600}, "budget_insufficient"),
        ({"implementation_symbols": []}, "target_functions_missing"),
        ({"driver_contract": {}}, "driver_unavailable"),
        ({"driver_contract": _rewrite_driver_contract(contract_version=2)}, "driver_contract_unsupported"),
    ],
)
def test_rewrite_route_rejects_candidates_before_probing_the_producer(
    tmp_path,
    monkeypatch,
    overrides,
    reason,
):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path, **overrides),
    )

    assert decision.eligible is False
    assert decision.reason == reason
    # An ineligible candidate must not spend a probe subprocess or any budget.
    assert probe.calls == 0


def test_rewrite_route_rejects_a_source_outside_the_prepared_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    stray = tmp_path / "elsewhere" / "helper.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("HELPER\n")
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path, implementation_sources=[str(stray)]),
    )

    assert decision.eligible is False
    assert decision.reason == "workspace_mapping_unresolved"
    assert str(stray) in decision.detail
    assert probe.calls == 0


def test_rewrite_route_rejects_multi_node_before_the_apply_stage(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path),
    )

    assert decision.eligible is False
    assert decision.reason == "multi_node_unsupported"
    assert probe.calls == 0


def test_rewrite_route_rejects_a_framework_the_producer_cannot_apply_back(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    probe = _RecordingProbe(
        _flydsl_rewrite.RewriteCapabilities(True, "capability_ok", "", ("aiter",), (1,))
    )

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path),
    )

    assert decision.eligible is False
    assert decision.reason == "capability_framework_unsupported"
    assert probe.calls == 1


def test_rewrite_route_forwards_the_incompatible_capability_reason(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    probe = _RecordingProbe(
        _flydsl_rewrite.RewriteCapabilities(False, "capability_probe_failed", "rc=2")
    )

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(tmp_path),
    )

    assert decision.eligible is False
    assert decision.reason == "capability_probe_failed"
    assert decision.capabilities is not None


def test_eligible_rewrite_route_carries_the_producer_candidate_fields(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    kwargs = _rewrite_route_kwargs(tmp_path)
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(capability_probe=probe, **kwargs)

    assert decision.eligible is True
    assert decision.reason == "eligible"
    assert decision.spec is not None
    assert decision.spec.as_dict() == {
        "logical_operator": "vllm::fused_gemm",
        "source_kernel": kwargs["source_kernel"],
        "implementation_symbols": ["matmul"],
        "source_entry": "matmul",
        "shape_cases": [{"M": 8, "N": 16}],
        "framework": "vllm",
        "gpu_target": "gfx942",
        "operator_family": "gemm",
        # The driver is generated only after the route is granted.
        "driver": "",
        "branch": "forge/session/fused-gemm-abc123def456",
        "attempt_id": "attempt-1",
    }
    assert decision.with_driver("/ws/.forge_driver_x.py").spec.driver == "/ws/.forge_driver_x.py"


def test_rewrite_spec_falls_back_to_the_single_shape_case(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    probe = _RecordingProbe(_SUPPORTED_CAPABILITIES)

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        capability_probe=probe,
        **_rewrite_route_kwargs(
            tmp_path,
            shape_cases=[],
            shapes={"M": 8, "K": 2048},
            candidate={"name": "fused_gemm", "source_entry": "fused_gemm_forward"},
        ),
    )

    assert decision.eligible is True
    assert decision.spec.shape_cases == ({"M": 8, "K": 2048},)
    assert decision.spec.source_entry == "fused_gemm_forward"


def _submit_with_rewrite_route(tmp_path, monkeypatch, captured=None, **submit_overrides) -> dict:
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "rewrite-route"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")

    def fake_loop(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return forge_submit.ForgeLoopOutcome(
            baseline_ms=2.0,
            best_ms=1.0,
            improved=True,
            output="forge-loop ran",
            error=None,
            timed_out=False,
            checkpoint=None,
            pristine_baseline_ms=2.0,
            search_start_ms=2.0,
            improved_during_search=True,
            structured_result=None,
        )

    _stub_submit_environment(monkeypatch)
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)

    submit_kwargs = {
        "source_file": str(source),
        "prompt_file": prompt,
        "output_dir": output_dir,
        "test_command": "python -c 'print(\"allclose: True\")'",
        "source_type": "triton",
        "candidate": {
            "name": "fused_gemm",
            "operation": "vllm::fused_gemm",
            "source_framework": "vllm",
            "source_symbol": "matmul",
            "kernel_sources": [str(source)],
            "kernel_kind": "triton",
            "platform": "mi355x",
            "input_shapes": [{"call_num": 4, "shape": "(64,32) fp16<br>(32,16) fp16"}],
            "input_dtypes": ["fp16", "fp16"],
        },
        "timeout_s": 7200,
        "kernel_repo": str(repo),
    }
    submit_kwargs.update(submit_overrides)
    return forge_submit.submit(**submit_kwargs)


def test_submit_omits_the_rewrite_verdict_when_the_route_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv(_flydsl_rewrite.REWRITE_ENV, raising=False)

    result = _submit_with_rewrite_route(tmp_path, monkeypatch)

    assert result["returncode"] == 0
    assert "flydsl_rewrite" not in result


_REWRITE_ARTIFACT_DIR = "forge_experiments/rewrite"
_REWRITE_PINNED_REF = "refs/hyperloom/applyback/attempt-1"


def _publish_applyback_in(workspace: str, base_commit: str, **outer_overrides) -> dict:
    """Commit an apply-back in the producer workspace and report its artifacts."""
    root = Path(workspace)
    (root / "kernel.py").write_text("def kernel(x):\n    return flydsl_kernel(x)\n")
    (root / "flydsl_kernel.py").write_text("def flydsl_kernel(x):\n    return x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "flydsl apply-back")
    best_commit = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", _REWRITE_PINNED_REF, best_commit)

    artifact_dir = root / _REWRITE_ARTIFACT_DIR
    (artifact_dir / "files").mkdir(parents=True, exist_ok=True)
    changed_files = ["flydsl_kernel.py", "kernel.py"]
    for relative in changed_files:
        (artifact_dir / "files" / relative).write_text((root / relative).read_text())
    (artifact_dir / "forge.patch").write_text(
        _git(root, "diff", f"{base_commit}..{best_commit}") + "\n"
    )
    scratch = artifact_dir / "scratch_port.py"
    scratch.write_text("SCRATCH\n")
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "validation_scope": "reference",
                "reference_correctness_passed": True,
                "reference_snr_db": 51.0,
                "integration_validation_required": True,
                "integration_validation_status": "pending",
                "base_commit": base_commit,
                "commit_hash": best_commit,
                "commit_ref": _REWRITE_PINNED_REF,
                "builder_symbol": "build_fused_gemm_module",
                "baseline_wall_ms": 4.0,
                "best_wall_ms": 2.0,
                "changed_files": changed_files,
            }
        ),
        encoding="utf-8",
    )
    outer = {
        "success": True,
        "applyback_required": True,
        "applyback_ok": True,
        "artifact_kind": "framework_applyback",
        "artifact_schema_version": 2,
        "best_commit": best_commit,
        "canonical_manifest": f"{_REWRITE_ARTIFACT_DIR}/manifest.json",
        "canonical_patch_path": f"{_REWRITE_ARTIFACT_DIR}/forge.patch",
        "canonical_files_root": f"{_REWRITE_ARTIFACT_DIR}/files",
        "temporary_paths": [f"{_REWRITE_ARTIFACT_DIR}/scratch_port.py"],
    }
    outer.update(outer_overrides)
    return outer


def _stub_rewrite_runner(monkeypatch, *, publish=True, error=None, timed_out=False):
    captured: dict = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        result = None
        if publish:
            base_commit = _git(Path(kwargs["workspace"]), "rev-parse", "HEAD")
            result = _publish_applyback_in(kwargs["workspace"], base_commit)
        return forge_submit.RewriteRunOutcome(
            result=result,
            output="forge rewrite ran",
            error=error,
            timed_out=timed_out,
        )

    monkeypatch.setattr(forge_submit, "_run_rewrite_via_cli", fake_runner)
    return captured


def test_submit_consumes_a_canonical_applyback_instead_of_the_forge_loop(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setattr(
        _flydsl_rewrite,
        "probe_capabilities",
        lambda **_kwargs: _SUPPORTED_CAPABILITIES,
    )
    rewrite_call = _stub_rewrite_runner(monkeypatch)

    loop_call: dict = {}
    result = _submit_with_rewrite_route(tmp_path, monkeypatch, captured=loop_call)

    assert result["returncode"] == 0
    # The generic loop is never reached once the rewrite route is granted.
    assert loop_call == {}
    verdict = result["flydsl_rewrite"]
    assert verdict["eligible"] is True
    assert verdict["spec"]["operator_family"] == "gemm"
    driver = Path(verdict["spec"]["driver"])
    assert driver.name.startswith(".forge_driver_")
    assert rewrite_call["driver"] == str(driver)
    assert rewrite_call["logical_op_name"] == "vllm::fused_gemm"

    applyback = result["flydsl_applyback"]
    assert applyback["artifact_kind"] == "framework_applyback"
    assert applyback["integration_validation_status"] == "pending"
    assert applyback["changed_files"] == ["flydsl_kernel.py", "kernel.py"]
    assert result["best_commit"] == applyback["best_commit"]
    assert result["best_ms"] == 2.0
    assert result["mean_case_speedup"] == 2.0
    assert result["salvaged"] is False
    output_dir = tmp_path / "results" / "rewrite-route"
    exported = output_dir / "optimized_versions"
    assert sorted(path.name for path in (exported / "files").iterdir()) == [
        "flydsl_kernel.py",
        "kernel.py",
    ]
    # The micro gate stays readable by the only report scanner in the repo,
    # while the integration verdict is stated separately.
    report = (output_dir / "optimization_report.md").read_text()
    assert "[correctness] pass" in report
    assert "[integration_validation] pending" in report


def test_reexported_patch_is_binary_safe(tmp_path, monkeypatch):
    repo, kernel = _make_repo(tmp_path)
    base_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "weights.bin").write_bytes(bytes(range(256)))
    kernel.write_text("OPTIMIZED\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "binary artifact")
    best_commit = _git(repo, "rev-parse", "HEAD")
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()

    _, changed = forge_submit._export_best_artifacts(
        str(repo),
        base_commit,
        str(kernel),
        str(kernel),
        output_dir,
        best_commit=best_commit,
    )

    assert sorted(changed) == ["kernel.py", "weights.bin"]
    patch = (output_dir / "optimized_versions" / "forge.patch").read_text()
    # Without --binary git emits a "Binary files differ" stub that cannot apply.
    assert "GIT binary patch" in patch
    assert "Binary files" not in patch


def test_submit_salvages_an_applyback_published_before_a_hard_kill(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setattr(
        _flydsl_rewrite,
        "probe_capabilities",
        lambda **_kwargs: _SUPPORTED_CAPABILITIES,
    )
    _stub_rewrite_runner(
        monkeypatch,
        error=RuntimeError("forge rewrite exceeded absolute deadline after 7200s"),
        timed_out=True,
    )

    result = _submit_with_rewrite_route(tmp_path, monkeypatch)

    assert result["returncode"] == 0
    assert result["timed_out"] is True
    assert result["salvaged"] is True
    assert result["flydsl_applyback"]["validation_scope"] == "reference"


@pytest.mark.parametrize(
    ("publish", "error", "timed_out"),
    [
        pytest.param(False, None, False, id="clean_exit_without_an_applyback"),
        pytest.param(False, RuntimeError("rc=1"), False, id="failed_without_an_applyback"),
        pytest.param(False, RuntimeError("deadline"), True, id="killed_before_publishing"),
    ],
)
def test_submit_produces_no_bundle_when_no_applyback_is_published(
    tmp_path,
    monkeypatch,
    publish,
    error,
    timed_out,
):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setattr(
        _flydsl_rewrite,
        "probe_capabilities",
        lambda **_kwargs: _SUPPORTED_CAPABILITIES,
    )
    _stub_rewrite_runner(monkeypatch, publish=publish, error=error, timed_out=timed_out)

    result = _submit_with_rewrite_route(tmp_path, monkeypatch)

    assert result["returncode"] == 1
    assert result["salvaged"] is False
    assert "best_commit" not in result
    output_dir = tmp_path / "results" / "rewrite-route"
    assert not (output_dir / "optimized_versions").exists()
    assert not (output_dir / "optimization_report.md").exists()


def test_submit_rejects_an_applyback_that_fails_its_own_contract(tmp_path, monkeypatch):
    """A published-but-invalid result is discarded, not degraded into a keep."""
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setattr(
        _flydsl_rewrite,
        "probe_capabilities",
        lambda **_kwargs: _SUPPORTED_CAPABILITIES,
    )

    def fake_runner(**kwargs):
        base_commit = _git(Path(kwargs["workspace"]), "rev-parse", "HEAD")
        outer = _publish_applyback_in(
            kwargs["workspace"], base_commit, temporary_paths=None
        )
        return forge_submit.RewriteRunOutcome(
            result=outer, output="", error=None, timed_out=False
        )

    monkeypatch.setattr(forge_submit, "_run_rewrite_via_cli", fake_runner)

    result = _submit_with_rewrite_route(tmp_path, monkeypatch)

    assert result["returncode"] == 1
    assert "flydsl_applyback" not in result
    assert not (tmp_path / "results" / "rewrite-route" / "optimization_report.md").exists()


def test_rewrite_cli_invocation_pins_the_producer_contract(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "fused_gemm.py"
    driver = workspace / ".forge_driver_abc.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    result_json = tmp_path / "attempt" / "forge_rewrite_result.json"
    result_json.write_text('{"stale": true}')
    captured = {}

    class FakeProcess:
        pid = 4242
        returncode = 0

        def communicate(self, timeout=None):
            captured["communicate_timeout"] = timeout
            payload = {"success": True, "applyback_ok": True}
            return f"__FORGE_RESULT__{json.dumps(payload)}", ""

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["popen_kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "/forge/src")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    deadline = time.time() + 7200.0
    outcome = forge_submit._run_rewrite_via_cli(
        source_kernel=str(kernel),
        driver=str(driver),
        logical_op_name="vllm::fused_gemm",
        source_entry="matmul",
        workspace=str(workspace),
        experiments_dir=experiments,
        result_json=result_json,
        target_functions=["matmul", "matmul_kernel"],
        shapes={"primary": {"CASE_ID": "case_001"}},
        snr_threshold=30.0,
        gpu_target="gfx950",
        max_iters=8,
        max_hours=2.0,
        branch="forge/session/fused-gemm",
        framework="vllm",
        forge_log=tmp_path / "forge.log",
        timeout_s=7200,
        deadline_unix=deadline,
    )

    command = captured["command"]
    assert command[:4] == [sys.executable, "-m", "kernel_agents.cli", "forge-rewrite-by-flydsl"]
    expected = {
        "--source-kernel": str(kernel),
        "--driver": str(driver),
        "--logical-op-name": "vllm::fused_gemm",
        "--source-entry": "matmul",
        "--workspace": str(workspace),
        "--experiments-dir": str(experiments),
        "--target-functions": "matmul,matmul_kernel",
        "--shapes-json": json.dumps({"primary": {"CASE_ID": "case_001"}}),
        "--snr-threshold": "30.0",
        "--gpu-target": "gfx950",
        "--max-iters": "8",
        "--max-hours": "2.0",
        "--deadline-unix": str(deadline),
        "--framework": "vllm",
        "--git-branch": "forge/session/fused-gemm",
        "--result-json": str(result_json),
    }
    for flag, value in expected.items():
        assert flag in command, flag
        assert command[command.index(flag) + 1] == value, flag
    # Options that only exist on the generic loop must never be smuggled across.
    for forbidden in (
        "--kernel",
        "--fellow",
        "--experiment-id",
        "--experience-id",
        "--e2e-pct",
        "--operator-name",
        "--source-files",
        "--program-md-file",
        "--invocation-spec-file",
        "--resume",
        "--task-type",
        "--workload-key",
        "--return-after-read-kb",
    ):
        assert forbidden not in command, forbidden

    assert captured["popen_kwargs"]["start_new_session"] is True
    assert captured["popen_kwargs"]["cwd"] == str(workspace)
    assert 7100.0 < captured["communicate_timeout"] <= 7200.0
    # A stale result from an earlier attempt is cleared before the child starts.
    assert outcome.result == {"success": True, "applyback_ok": True}
    assert outcome.error is None
    assert outcome.timed_out is False


def test_rewrite_cli_hard_kills_the_producer_at_the_deadline(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    result_json = tmp_path / "attempt" / "forge_rewrite_result.json"
    terminated = {}

    class HangingProcess:
        pid = 4321
        returncode = -9

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="forge-rewrite-by-flydsl", timeout=timeout)

    def fake_terminate(proc):
        terminated["pid"] = proc.pid
        return "partial stdout", "killed"

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", lambda command, **kwargs: HangingProcess())
    monkeypatch.setattr(forge_submit, "_terminate_forge_process", fake_terminate)

    outcome = forge_submit._run_rewrite_via_cli(
        source_kernel=str(workspace / "kernel.py"),
        driver=str(workspace / "driver.py"),
        logical_op_name="vllm::op",
        source_entry="",
        workspace=str(workspace),
        experiments_dir=experiments,
        result_json=result_json,
        target_functions=None,
        shapes={},
        snr_threshold=30.0,
        gpu_target="gfx942",
        max_iters=4,
        max_hours=1.0,
        branch="forge/session/op",
        framework="",
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
        deadline_unix=time.time() + 1.0,
    )

    # The whole process group is torn down, and a run that published nothing
    # yields no result for the validator to consider.
    assert terminated["pid"] == 4321
    assert outcome.timed_out is True
    assert outcome.result is None
    assert "deadline" in str(outcome.error)
    assert "killed" in outcome.output


def test_rewrite_cli_prefers_the_caller_named_result_file(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    result_json = tmp_path / "attempt" / "forge_rewrite_result.json"

    class FakeProcess:
        pid = 4243
        returncode = 0

        def communicate(self, timeout=None):
            result_json.write_text(json.dumps({"success": True, "from": "sidecar"}))
            return '__FORGE_RESULT__{"success": true, "from": "sentinel"}', ""

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", lambda command, **kwargs: FakeProcess())

    outcome = forge_submit._run_rewrite_via_cli(
        source_kernel=str(workspace / "kernel.py"),
        driver=str(workspace / "driver.py"),
        logical_op_name="vllm::op",
        source_entry="",
        workspace=str(workspace),
        experiments_dir=experiments,
        result_json=result_json,
        target_functions=None,
        shapes={},
        snr_threshold=30.0,
        gpu_target="gfx942",
        max_iters=4,
        max_hours=1.0,
        branch="forge/session/op",
        framework="",
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
    )

    assert outcome.result == {"success": True, "from": "sidecar"}


def test_finalizer_reclaims_only_declared_temporary_paths(tmp_path):
    repo, kernel = _make_repo(tmp_path)
    scratch_file = repo / "scratch_port.py"
    scratch_file.write_text("SCRATCH\n")
    scratch_dir = repo / "scratch_tree"
    scratch_dir.mkdir()
    (scratch_dir / "inner.py").write_text("INNER\n")
    untouched = repo / "untracked_note.txt"
    untouched.write_text("KEEP\n")

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info=None,
        driver="",
        workspace=str(repo),
        output_dir=tmp_path / "attempt",
        branch="forge/session/kernel",
        nogit_scratch=False,
        temporary_paths=[str(scratch_file), str(scratch_dir)],
    )

    assert not scratch_file.exists()
    assert not scratch_dir.exists()
    assert untouched.exists()
    assert kernel.exists()


def test_finalizer_refuses_temporary_paths_outside_the_workspace(tmp_path):
    repo, kernel = _make_repo(tmp_path)
    outsider = tmp_path / "outside.py"
    outsider.write_text("KEEP\n")

    with pytest.raises(RuntimeError, match="escapes the workspace"):
        forge_submit._finalize_forge_workspace(
            inplace=True,
            restore_info=None,
            driver="",
            workspace=str(repo),
            output_dir=tmp_path / "attempt",
            branch="forge/session/kernel",
            nogit_scratch=False,
            temporary_paths=[str(outsider), str(repo)],
        )

    assert outsider.exists()
    assert kernel.exists()


def test_retained_workspace_finalizer_ignores_temporary_paths(tmp_path):
    """Only the in-place route reclaims; a retained worktree is left intact."""
    repo, _kernel = _make_repo(tmp_path)
    scratch = repo / "scratch_port.py"
    scratch.write_text("SCRATCH\n")

    forge_submit._finalize_forge_workspace(
        inplace=False,
        restore_info=None,
        driver="",
        workspace=str(repo),
        output_dir=tmp_path / "attempt",
        branch="forge/session/kernel",
        nogit_scratch=False,
        temporary_paths=[str(scratch)],
    )

    assert scratch.exists()


def test_submit_falls_back_to_forge_loop_on_an_incompatible_producer(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    monkeypatch.setattr(
        _flydsl_rewrite,
        "probe_capabilities",
        lambda **_kwargs: _flydsl_rewrite.RewriteCapabilities(
            False,
            "capability_artifact_schema_unsupported",
            "producer artifact schemas [1] exclude 2",
        ),
    )

    result = _submit_with_rewrite_route(tmp_path, monkeypatch)

    assert result["returncode"] == 0
    assert result["skipped"] is False
    verdict = result["flydsl_rewrite"]
    assert verdict["eligible"] is False
    assert verdict["reason"] == "capability_artifact_schema_unsupported"
    assert result["best_ms"] == 1.0


_SOURCE_MODULE = """
import os


def matmul(a, b):
    with open(os.environ["REWRITE_CALL_LOG"], "a") as log:
        log.write("source\\n")
    return a @ b
"""

_CANDIDATE_MODULE = """
import os


def build_gemm_module():
    def launch(a, b):
        with open(os.environ["REWRITE_CALL_LOG"], "a") as log:
            log.write("candidate\\n")
        return (a.float() @ b.float()).to(a.dtype)

    return launch
"""


def _rewrite_driver_workspace(tmp_path, *, contract=None, candidate_module=_CANDIDATE_MODULE):
    pytest.importorskip("torch")
    workspace = tmp_path / "rewrite-ws"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "source.py").write_text(_SOURCE_MODULE)
    (workspace / "candidate.py").write_text(candidate_module)
    driver = _flydsl_rewrite.build_rewrite_driver(
        contract or _rewrite_driver_contract(),
        workspace=str(workspace),
        writer=forge_submit._write_generated_driver,
    )
    return workspace, driver


def _run_rewrite_driver(workspace, driver, *args, candidate=True, builder="build_gemm_module"):
    env = dict(os.environ)
    env["REWRITE_CALL_LOG"] = str(workspace / "calls.log")
    env[_flydsl_rewrite.ENV_SOURCE_KERNEL] = str(workspace / "source.py")
    env[_flydsl_rewrite.ENV_BUILDER_SYMBOL] = builder
    env[_flydsl_rewrite.ENV_LOGICAL_OP] = "vllm::fused_gemm"
    if candidate:
        env[_flydsl_rewrite.ENV_CANDIDATE_KERNEL] = str(workspace / "candidate.py")
    else:
        env.pop(_flydsl_rewrite.ENV_CANDIDATE_KERNEL, None)
    return subprocess.run(
        [sys.executable, driver, *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(workspace),
    )


def _called_implementations(workspace) -> list[str]:
    log = workspace / "calls.log"
    return sorted(set(log.read_text().split())) if log.exists() else []


def test_rewrite_driver_bench_modes_measure_different_implementations(tmp_path):
    workspace, driver = _rewrite_driver_workspace(tmp_path)

    reference = _run_rewrite_driver(workspace, driver, "--ref-bench-mode", "--warmup", "1", "--iters", "2")
    assert reference.returncode == 0, reference.stderr
    assert _called_implementations(workspace) == ["source"]

    (workspace / "calls.log").unlink()
    candidate = _run_rewrite_driver(workspace, driver, "--bench-mode", "--warmup", "1", "--iters", "2")
    assert candidate.returncode == 0, candidate.stderr
    assert _called_implementations(workspace) == ["candidate"]


def test_rewrite_driver_timing_is_readable_by_the_producer_parser(tmp_path):
    workspace, driver = _rewrite_driver_workspace(tmp_path)

    proc = _run_rewrite_driver(workspace, driver, "--ref-bench-mode", "--warmup", "1", "--iters", "3")

    assert proc.returncode == 0, proc.stderr
    # Same expression the generic harness adapter uses to recover a latency.
    match = re.search(r"(?:median_ms|wall_ms)\s*[:=]\s*([0-9.]+)", proc.stdout)
    assert match is not None, proc.stdout
    assert float(match.group(1)) > 0.0
    # Per-case detail must not collide with the aggregate the parser reads.
    assert match.group(1) == re.search(r"^median_ms:\s*([0-9.]+)$", proc.stdout, re.M).group(1)


def test_rewrite_driver_correctness_covers_every_grouped_case(tmp_path):
    contract = _rewrite_driver_contract(
        cases=[
            {
                "case_id": "case_001",
                "selector": {"CASE_ID": "case_001"},
                "inputs": [
                    {"shape": [64, 32], "dtype": "fp16"},
                    {"shape": [32, 16], "dtype": "fp16"},
                ],
            },
            {
                "case_id": "case_002",
                "selector": {"CASE_ID": "case_002"},
                "inputs": [
                    {"shape": [128, 64], "dtype": "fp32"},
                    {"shape": [64, 8], "dtype": "fp32"},
                ],
            },
        ]
    )
    workspace, driver = _rewrite_driver_workspace(tmp_path, contract=contract)

    proc = _run_rewrite_driver(workspace, driver)

    assert proc.returncode == 0, proc.stderr
    assert "[case] case_001 output=0 snr_db=" in proc.stdout
    assert "[case] case_002 output=0 snr_db=" in proc.stdout
    reported = float(re.search(r"^SNR:\s*([-0-9.]+) dB$", proc.stdout, re.M).group(1))
    per_case = [float(value) for value in re.findall(r"snr_db=([-0-9.]+)", proc.stdout)]
    assert len(per_case) == 2
    assert reported == pytest.approx(min(per_case), abs=0.01)
    assert _called_implementations(workspace) == ["candidate", "source"]


def test_rewrite_driver_selects_a_single_grouped_case(tmp_path):
    contract = _rewrite_driver_contract(
        cases=[
            {
                "case_id": "case_001",
                "selector": {"CASE_ID": "case_001"},
                "inputs": [{"shape": [64, 32], "dtype": "fp16"}, {"shape": [32, 16], "dtype": "fp16"}],
            },
            {
                "case_id": "case_002",
                "selector": {"CASE_ID": "case_002"},
                "inputs": [{"shape": [16, 8], "dtype": "fp16"}, {"shape": [8, 4], "dtype": "fp16"}],
            },
        ]
    )
    workspace, driver = _rewrite_driver_workspace(tmp_path, contract=contract)

    proc = _run_rewrite_driver(
        workspace, driver, "--ref-bench-mode", "--warmup", "1", "--iters", "2", "--shape", "CASE_ID=case_002"
    )

    assert proc.returncode == 0, proc.stderr
    assert "[case] case_002 ms=" in proc.stdout
    assert "case_001" not in proc.stdout


def test_rewrite_driver_reports_a_missing_candidate_without_a_metric(tmp_path):
    workspace, driver = _rewrite_driver_workspace(tmp_path)

    proc = _run_rewrite_driver(workspace, driver, candidate=False)

    assert proc.returncode == 1
    assert _flydsl_rewrite.ENV_CANDIDATE_KERNEL in proc.stderr
    # A harness failure is never reported as a correctness verdict.
    assert "SNR:" not in proc.stdout
    assert "allclose:" not in proc.stdout


def test_rewrite_driver_fails_closed_when_the_candidate_shape_diverges(tmp_path):
    diverging = """
import os


def build_gemm_module():
    def launch(a, b):
        with open(os.environ["REWRITE_CALL_LOG"], "a") as log:
            log.write("candidate\\n")
        return (a.float() @ b.float()).to(a.dtype)[:1]

    return launch
"""
    workspace, driver = _rewrite_driver_workspace(tmp_path, candidate_module=diverging)

    proc = _run_rewrite_driver(workspace, driver)

    assert proc.returncode == 1
    assert "allclose: False" in proc.stdout
    assert "SNR:" not in proc.stdout


def test_rewrite_driver_calls_a_parameterised_builder_symbol_directly(tmp_path):
    direct = """
import os


def flydsl_gemm(a, b):
    with open(os.environ["REWRITE_CALL_LOG"], "a") as log:
        log.write("candidate\\n")
    return (a.float() @ b.float()).to(a.dtype)
"""
    workspace, driver = _rewrite_driver_workspace(tmp_path, candidate_module=direct)

    proc = _run_rewrite_driver(workspace, driver, builder="flydsl_gemm")

    assert proc.returncode == 0, proc.stderr
    assert "SNR:" in proc.stdout
    assert _called_implementations(workspace) == ["candidate", "source"]


def test_rewrite_driver_is_staged_and_reclaimed_like_every_generated_driver(tmp_path):
    repo, kernel = _make_repo(tmp_path)
    driver = _flydsl_rewrite.build_rewrite_driver(
        _rewrite_driver_contract(),
        workspace=str(repo),
        writer=forge_submit._write_generated_driver,
    )
    staged = Path(driver)

    assert staged.parent == repo
    assert staged.name.startswith(".forge_driver_")
    # Excluded from git, so even a broad producer `git add` cannot carry the
    # driver into the framework patch.
    assert staged.name not in _git(repo, "status", "--porcelain")
    _git(repo, "add", "-A")
    assert staged.name not in _git(repo, "ls-files")

    forge_submit._finalize_forge_workspace(
        inplace=True,
        restore_info=None,
        driver=driver,
        workspace=str(repo),
        output_dir=tmp_path / "attempt",
        branch="forge/session/kernel",
        nogit_scratch=False,
    )

    assert not staged.exists()
    assert kernel.exists()
