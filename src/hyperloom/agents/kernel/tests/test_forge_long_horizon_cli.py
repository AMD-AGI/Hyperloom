"""Regression tests for the long-horizon KernelForge CLI integration."""

from __future__ import annotations

import json
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


def test_generated_driver_lives_inside_campaign_workspace(tmp_path):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    tracked_driver = workspace / "forge_driver.py"
    tracked_driver.write_text("TRACKED_DRIVER\n")

    first = Path(
        forge_submit._build_driver_adapter(
            "python test.py",
            str(workspace),
            tmp_path,
        )
    )
    second = Path(
        forge_submit._build_driver_adapter(
            "python test.py",
            str(workspace),
            tmp_path,
            inplace=True,
        )
    )
    generated = Path(
        forge_submit._autogen_forge_driver(
            {"operation": "gemm"},
            str(workspace / "kernel.py"),
            workspace,
        )
    )

    assert {first.parent, second.parent, generated.parent} == {workspace}
    assert len({first.name, second.name, generated.name}) == 3
    assert all(path.name.startswith(".forge_driver_") for path in (first, second, generated))
    assert tracked_driver.read_text() == "TRACKED_DRIVER\n"


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


def test_inplace_restore_preserves_original_staged_and_unstaged_diffs(tmp_path):
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
    staged_before = _git(repo, "diff", "--cached", "--binary", "--full-index")
    unstaged_before = _git(repo, "diff", "--binary", "--full-index")
    status_before = _git(repo, "status", "--short")

    branch = "forge/test/inplace-index-restore"
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
    assert _git(repo, "diff", "--cached", "--binary", "--full-index") == staged_before
    assert _git(repo, "diff", "--binary", "--full-index") == unstaged_before
    assert _git(repo, "status", "--short") == status_before
    assert source.read_text() == "STAGED\nUNSTAGED\n"
    assert binary.read_bytes() == b"\x00STAGED\xffUNSTAGED"


def test_cli_uses_simplified_fresh_campaign_contract(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    program = tmp_path / "program.md"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    program.write_text("# Task\n")
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

    result = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        shapes={"primary": {"M": 128}},
        max_hours=0.25,
        gpu_target="gfx950",
        fellow="triton-fellow",
        program_md_file=str(program),
        forge_log=tmp_path / "forge.log",
        timeout_s=30,
    )

    command = captured["command"]
    assert result[:3] == (2.0, 1.0, True)
    assert command[:5] == [
        sys.executable,
        "-m",
        "kernel_agents.cli",
        "forge-loop",
        "--kernel",
    ]
    assert command[command.index("--max-hours") + 1] == "1.0"
    assert "--program-md-file" in command
    for removed in (
        "--max-iters",
        "--experiments-dir",
        "--result-json",
        "--git-branch",
        "--gpu-target",
        "--fellow",
        "--snr-threshold",
    ):
        assert removed not in command
    assert captured["env"]["GPU_TARGET"] == "gfx950"
    assert captured["env"]["FORGE_FELLOW"] == "triton-fellow"
    assert captured["env"]["PYTHONPATH"].startswith("/forge/src")
    assert captured["popen_kwargs"]["start_new_session"] is True
    assert captured["communicate_timeout"] == 30


def test_cli_timeout_recovers_incremental_best_from_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "worktree"
    experiments = workspace / "forge_experiments"
    experiments.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    (experiments / "best_result.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "commit_hash": "old-commit",
                "baseline_wall_ms": 4.0,
                "best_wall_ms": 3.0,
                "correctness_passed": True,
            }
        )
    )
    (experiments / "run_state.json").write_text(
        json.dumps(
            {
                "iteration": 2,
                "baseline_wall_ms": 3.0,
                "best": {
                    "iteration": 2,
                    "commit_hash": "new-commit",
                    "wall_ms": 2.0,
                },
            }
        )
    )

    def timeout(command, **_kwargs):
        return (
            subprocess.CompletedProcess(
                command,
                -signal.SIGKILL,
                stdout="partial stdout",
                stderr="partial stderr",
            ),
            True,
        )

    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_apply_fellow_env", lambda _env: None)
    monkeypatch.setattr(forge_submit, "_run_isolated_process_group", timeout)

    baseline, best, improved, _output, error = forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        shapes={},
        max_hours=1.0,
        gpu_target="gfx950",
        fellow="triton-fellow",
        program_md_file="",
        forge_log=tmp_path / "forge.log",
        timeout_s=10,
    )

    assert (baseline, best, improved) == (3.0, 2.0, True)
    assert isinstance(error, forge_submit._ForgeLoopTimeout)
    selected = forge_submit._freshest_verified_best(workspace)
    assert selected["source"] == "run_state.json"
    assert selected["iteration"] == 2
    assert selected["commit_hash"] == "new-commit"


def test_isolated_process_timeout_terminates_and_kills_process_group(monkeypatch):
    captured = {}
    signals = []
    waits = []

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
            if self.communicate_calls == 2:
                raise subprocess.TimeoutExpired(["forge-loop"], timeout)
            self.returncode = -signal.SIGKILL
            return "partial stdout\nfinal stdout", "partial stderr\nfinal stderr"

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["process"] = FakeProcess()
        return captured["process"]

    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        forge_submit.os,
        "killpg",
        lambda process_group, sent_signal: signals.append(
            (process_group, sent_signal)
        ),
    )
    monkeypatch.setattr(
        forge_submit,
        "_wait_for_process_group_exit",
        lambda process_group, timeout_s: waits.append((process_group, timeout_s)),
    )

    completed, timed_out = forge_submit._run_isolated_process_group(
        ["forge-loop"],
        cwd="/workspace",
        env={"TEST": "1"},
        timeout_s=1,
        termination_grace_s=0.1,
    )

    assert timed_out is True
    assert completed.stdout == "partial stdout\nfinal stdout"
    assert completed.stderr == "partial stderr\nfinal stderr"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"] is subprocess.PIPE
    assert captured["kwargs"]["stderr"] is subprocess.PIPE
    assert signals == [
        (captured["process"].pid, signal.SIGTERM),
        (captured["process"].pid, signal.SIGKILL),
    ]
    assert waits == [(captured["process"].pid, 0.1)]


def test_isolated_process_timeout_leaves_no_running_child(tmp_path):
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

    try:
        completed, timed_out = forge_submit._run_isolated_process_group(
            [sys.executable, str(script)],
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout_s=0.5,
            termination_grace_s=0.2,
        )
        assert timed_out is True
        assert "child-started" in completed.stdout
        child_pid = int(child_pid_file.read_text())

        deadline = time.monotonic() + 2
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


def test_campaign_best_rejects_same_iteration_commit_mismatch(tmp_path):
    experiments = tmp_path / "forge_experiments"
    experiments.mkdir()
    (experiments / "run_state.json").write_text(
        json.dumps(
            {
                "best": {
                    "iteration": 2,
                    "commit_hash": "commit-a",
                    "wall_ms": 2.0,
                }
            }
        )
    )
    (experiments / "best_result.json").write_text(
        json.dumps(
            {
                "iteration": 2,
                "commit_hash": "commit-b",
                "baseline_wall_ms": 3.0,
                "best_wall_ms": 2.0,
                "correctness_passed": True,
            }
        )
    )

    with pytest.raises(ValueError, match="conflicting verified best commits"):
        forge_submit._freshest_verified_best(tmp_path)


def test_restore_verified_best_rejects_unavailable_commit(tmp_path):
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

    with pytest.raises(RuntimeError, match="verified best commit lookup failed"):
        forge_submit._restore_verified_best(
            workspace,
            branch,
            {
                "iteration": 1,
                "commit_hash": "f" * 40,
                "git_branch": branch,
            },
        )

    assert _git(workspace, "rev-parse", "HEAD") == base_commit
    assert Path(kernel).read_text() == "BASELINE\n"


def test_submit_timeout_exports_only_verified_commit_and_returns_124(
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
        branch = _git(workspace, "branch", "--show-current")
        experiments = workspace / "forge_experiments"
        experiments.mkdir()
        (experiments / "run_state.json").write_text(
            json.dumps(
                {
                    "iteration": 2,
                    "baseline_wall_ms": 3.0,
                    "git_branch": branch,
                    "best": {
                        "iteration": 2,
                        "commit_hash": best_commit,
                        "wall_ms": 2.0,
                    },
                }
            )
        )
        kernel.write_text("UNVERIFIED_MID_ITERATION\n")
        _git(workspace, "add", "-u")
        captured.update(
            workspace=workspace,
            kernel=kernel,
            best_commit=best_commit,
            branch=branch,
        )
        return (
            3.0,
            2.0,
            True,
            "partial output",
            forge_submit._ForgeLoopTimeout("timed out"),
        )

    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
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
    assert result["returncode"] == 124
    assert result["cli_workspace"] == str(output_dir)
    assert result["forge_workspace"] == str(captured["workspace"])
    assert result["output_dir"] == str(output_dir)
    assert result["optimized_artifact"] == str(optimized)
    assert optimized.read_text() == "VERIFIED_BEST\n"
    assert captured["kernel"].read_text() == "VERIFIED_BEST\n"
    assert _git(captured["workspace"], "rev-parse", "HEAD") == captured["best_commit"]
    assert _git(
        captured["workspace"],
        "status",
        "--porcelain",
        "--untracked-files=no",
    ) == ""
    assert captured["workspace"].is_dir()
    assert _git(repo, "branch", "--list", captured["branch"])


def test_submit_timeout_restore_failure_never_exports_current_workspace(
    tmp_path,
    monkeypatch,
):
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / "attempt-restore-failure"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")
    captured = {"exports": 0, "reports": 0}

    def fake_loop(**kwargs):
        workspace = Path(kwargs["workspace"])
        kernel = Path(kwargs["worktree_kernel"])
        kernel.write_text("VERIFIED_BEST\n")
        _git(workspace, "add", "-u")
        _git(workspace, "commit", "-m", "verified best")
        best_commit = _git(workspace, "rev-parse", "HEAD")
        branch = _git(workspace, "branch", "--show-current")
        experiments = workspace / "forge_experiments"
        experiments.mkdir()
        (experiments / "run_state.json").write_text(
            json.dumps(
                {
                    "iteration": 2,
                    "baseline_wall_ms": 3.0,
                    "git_branch": branch,
                    "best": {
                        "iteration": 2,
                        "commit_hash": best_commit,
                        "wall_ms": 2.0,
                    },
                }
            )
        )
        kernel.write_text("UNVERIFIED_MID_ITERATION\n")
        captured["workspace"] = workspace
        captured["kernel"] = kernel
        return (
            3.0,
            2.0,
            True,
            "partial output",
            forge_submit._ForgeLoopTimeout("timed out"),
        )

    def forbidden_export(*_args, **_kwargs):
        captured["exports"] += 1
        raise AssertionError("unverified workspace must not be exported")

    def forbidden_report(*_args, **_kwargs):
        captured["reports"] += 1
        raise AssertionError("failed recovery must not write a promotable report")

    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
    monkeypatch.setattr(forge_submit, "_run_loop_via_cli", fake_loop)
    monkeypatch.setattr(
        forge_submit,
        "_restore_verified_best",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )
    monkeypatch.setattr(forge_submit, "_export_best_artifacts", forbidden_export)
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

    assert result["returncode"] == 124
    assert "verified best recovery failed" in result["stderr_tail"]
    assert captured["exports"] == 0
    assert captured["reports"] == 0
    assert captured["kernel"].read_text() == "UNVERIFIED_MID_ITERATION\n"
    assert "optimized_artifact" not in result
    assert result["artifacts"] == []
    assert not (output_dir / "optimization_report.md").exists()
    assert not (output_dir / "optimized_versions").exists()


@pytest.mark.parametrize(
    ("loop_error", "expected_returncode"),
    [
        (forge_submit._ForgeLoopTimeout("timed out"), 124),
        (RuntimeError("loop failed"), 1),
    ],
    ids=["timeout", "error"],
)
def test_submit_failure_without_verified_best_writes_no_promotable_artifacts(
    tmp_path,
    monkeypatch,
    loop_error,
    expected_returncode,
):
    repo, source = _make_repo(tmp_path)
    output_dir = tmp_path / "results" / f"attempt-{expected_returncode}"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("# Optimize\n")

    def fake_loop(**kwargs):
        Path(kwargs["worktree_kernel"]).write_text("UNVERIFIED_MID_ITERATION\n")
        return 3.0, 2.0, True, "failed output", loop_error

    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
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

    assert result["returncode"] == expected_returncode
    assert result["artifacts"] == []
    assert result["changed_files"] == []
    assert "optimized_artifact" not in result
    assert not (output_dir / "optimization_report.md").exists()
    assert not (output_dir / "optimized_versions").exists()


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
        return 3.0, 2.0, True, "failed", RuntimeError("loop failed")

    monkeypatch.setenv("FORGE_BASELINE_GATE", "0")
    monkeypatch.setattr(forge_submit, "_ensure_forge_on_path", lambda: "")
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
    assert len(set(branches)) == 2
    assert all(branch.startswith("forge/") for branch in branches)
    assert all(workspace.is_dir() for workspace in workspaces)


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


def test_retained_worktree_collision_skips_without_delete_or_nogit_fallback(
    tmp_path,
    monkeypatch,
):
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


def test_trace_reads_latest_campaign_experiment(tmp_path):
    workspace = tmp_path / "worktree"
    experiments = workspace / "forge_experiments"
    experiments.mkdir(parents=True)
    (experiments / "run_state.json").write_text(
        json.dumps({"last_experiment_id": "segment-2"})
    )
    (experiments / "segment-2.json").write_text(
        json.dumps(
            {
                "experiment_id": "segment-2",
                "llm_usage": {"input_tokens": 10, "output_tokens": 3},
            }
        )
    )

    usage, steps = forge_submit._forge_trace_from_campaign(workspace)

    assert usage == {"input_tokens": 10, "output_tokens": 3}
    assert steps is None
