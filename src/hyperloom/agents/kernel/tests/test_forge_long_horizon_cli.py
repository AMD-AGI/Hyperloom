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
    salvaged, exportable best commit.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


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
        "validation_passed": True,
        "case_coverage": [],
    }
    payload.update(overrides)
    return payload


def _stub_submit_environment(monkeypatch) -> None:
    """Neutralize everything submit does outside the loop/recovery contract."""
    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")


def test_generated_drivers_never_clobber_the_git_workspace(tmp_path):
    """Both driver generators write under the campaign output dir.

    The workspace is a git worktree of the user's repo, so writing a generated
    driver into it would either clobber a tracked file of the same name or show
    up as an agent "edit" in the keep/revert diff.
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
            output_dir,
        )
    )
    generated = Path(
        forge_submit._autogen_forge_driver(
            {"operation": "gemm"},
            str(workspace / "kernel.py"),
            output_dir,
        )
    )

    assert {adapter.parent, generated.parent} == {output_dir}
    assert adapter.name != generated.name
    assert adapter.is_file() and generated.is_file()
    # The workspace gained nothing and lost nothing.
    assert sorted(path.name for path in workspace.iterdir()) == ["forge_driver.py"]
    assert tracked_driver.read_text() == "TRACKED_DRIVER\n"


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
            payload = {"baseline_ms": 2.0, "best_ms": 1.0, "improved": True}
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
    )

    # The loop result is a 7-field outcome; unpacking it as a bare tuple is what
    # silently broke the recovery channels before.
    assert forge_submit.ForgeLoopOutcome._fields == (
        "baseline_ms",
        "best_ms",
        "improved",
        "output",
        "error",
        "timed_out",
        "checkpoint",
    )
    assert (outcome.baseline_ms, outcome.best_ms, outcome.improved) == (2.0, 1.0, True)
    assert outcome.error is None
    assert outcome.timed_out is False
    assert outcome.checkpoint is None

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
    }
    for flag, value in expected_flags.items():
        assert flag in command, flag
        assert command[command.index(flag) + 1] == value, flag

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
            proc.communicate(timeout=0.2)
        stdout, _stderr = forge_submit._terminate_forge_process(proc, grace_sec=2)

        assert "child-started" in stdout
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            stat_path = Path(f"/proc/{child_pid}/stat")
            if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"child process {child_pid} survived process-group timeout")
    finally:
        if child_pid is not None:
            try:
                stat_path = Path(f"/proc/{child_pid}/stat")
                if stat_path.exists() and stat_path.read_text().split()[2] != "Z":
                    os.kill(child_pid, signal.SIGKILL)
            except (OSError, IndexError):
                pass
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
    # the disposable worktree + temp branch are torn down afterwards.
    assert (output_dir / "forge_experiments").is_dir()
    assert not (repo / "forge_experiments").exists()
    assert not captured["workspace"].exists()
    assert _git(repo, "branch", "--list", captured["branch"]) == ""
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

    report = (output_dir / "optimization_report.md").read_text()
    assert "[micro_speedup]" not in report
    assert "[correctness] fail" in report
    # The discarded measurements must not survive even as an informational line.
    assert "observed timing" not in report
    assert "3.0000" not in report

    # forge-loop rejects a soft budget below its own one-hour minimum, so submit
    # floors --max-hours there while still hard-killing at timeout_s.
    assert captured["max_hours"] >= forge_submit._FORGE_MIN_BUDGET_SEC / 3600.0
    assert captured["timeout_s"] == 10


def test_submit_loop_error_without_measurement_fails_and_tears_down_worktree(
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

    assert [result["returncode"] for result in results] == [1, 1]
    assert all("forge cli loop failed" in result["stderr_tail"] for result in results)
    # Each attempt tears its disposable worktree + temp branch down, so a repeat
    # run on the same repo is not blocked by the previous one.
    assert len(workspaces) == 2
    assert not any(workspace.exists() for workspace in workspaces)
    assert _git(repo, "branch", "--list") == "* main"
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
        "_remove_worktree",
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
