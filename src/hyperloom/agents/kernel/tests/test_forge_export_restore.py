#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for forge export/restore across all changed files."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


def _commit_all(repo: str, msg: str) -> None:
    """Mirror IterationLoop._git_commit (git add -u + commit)."""
    subprocess.run(["git", "-C", repo, "add", "-u"], capture_output=True)
    subprocess.run(["git", "-C", repo, "commit", "-m", msg], capture_output=True)


def _make_repo(tmp: Path) -> dict:
    repo = tmp / "repo"
    repo.mkdir()
    r = str(repo)
    _git(r, "init", "-b", "main")
    subprocess.run(["git", "-C", r, "config", "user.name", "t"], capture_output=True)
    subprocess.run(["git", "-C", r, "config", "user.email", "t@t"], capture_output=True)
    kernel = repo / "kernel.py"
    config = repo / "kernel_config.py"
    other = repo / "other.txt"
    kernel.write_text("KERNEL_ORIGINAL\n")
    config.write_text("BLOCK_SIZE = 64\n")
    other.write_text("committed_v1\n")
    subprocess.run(["git", "-C", r, "add", "."], capture_output=True)
    subprocess.run(["git", "-C", r, "commit", "-m", "initial"], capture_output=True)
    # Pre-existing dirty tracked file (must survive a forge run untouched).
    other.write_text("committed_v1\nPRE_EXISTING_DIRTY\n")
    # Pre-existing untracked file (must never be touched).
    (repo / "untracked.txt").write_text("untracked\n")
    return {"repo": r, "kernel_agent": kernel, "config": config, "other": other, "untracked": repo / "untracked.txt"}


def test_export_and_restore_cover_sibling_file():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel, config, other, untracked = (
            env["repo"],
            env["kernel_agent"],
            env["config"],
            env["other"],
            env["untracked"],
        )
        out = tmp / "out"
        out.mkdir()
        branch = "forge/test/kernel"

        orig_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        assert orig_branch == "main"

        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None, "prepare_inplace should succeed on a clean git repo"
        workspace, worktree_kernel, restore = prep
        base_commit = restore["base_commit"]
        # base_commit must have absorbed the pre-existing dirty file.
        assert base_commit, "base_commit must be set"

        # Winning edit lands in the SIBLING config file.
        # iter1 (kept): config 64 -> 128, kernel untouched.
        config.write_text("BLOCK_SIZE = 128\n")
        _commit_all(repo, "iter1: tune config")
        # iter2 (reverted): config 128 -> 256, then revert.
        config.write_text("BLOCK_SIZE = 256\n")
        _commit_all(repo, "iter2: worse")
        subprocess.run(["git", "-C", repo, "revert", "--no-edit", "HEAD"], capture_output=True)
        # best-kept state on disk now: config == 128.
        assert "128" in config.read_text()

        # --- export ---
        primary, changed = forge_submit._export_best_artifacts(workspace, base_commit, str(kernel), str(kernel), out)
        # The sibling config file must be in the changed set + exported.
        assert "kernel_config.py" in changed, f"changed should include config: {changed}"
        exported_cfg = out / "optimized_versions" / "files" / "kernel_config.py"
        assert exported_cfg.is_file(), "config file must be exported under files/"
        assert "128" in exported_cfg.read_text(), "exported config must carry the optimization"
        patch = (out / "optimized_versions" / "forge.patch").read_text()
        assert "BLOCK_SIZE" in patch and "128" in patch, "patch must contain the real change"

        # --- restore ---
        forge_submit._restore_inplace(restore)

        # 1. kernel + config reverted to pre-forge content.
        assert kernel.read_text() == "KERNEL_ORIGINAL\n"
        assert config.read_text() == "BLOCK_SIZE = 64\n", "config must be restored to pre-forge"
        # 2. pre-existing dirty file preserved (still dirty).
        assert other.read_text() == "committed_v1\nPRE_EXISTING_DIRTY\n"
        status = _git(repo, "status", "--porcelain")
        assert "other.txt" in status, f"pre-existing dirty must remain dirty: {status!r}"
        assert "kernel_config.py" not in status, f"config must be clean after restore: {status!r}"
        # 3. untracked file untouched.
        assert untracked.read_text() == "untracked\n"
        # 4. back on original branch, no forge temp branch.
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        branches = _git(repo, "branch", "--list", branch)
        assert branches.strip() == "", f"temp branch must be deleted: {branches!r}"
        print("PASS test_export_and_restore_cover_sibling_file")


def test_restore_from_detached_head_preserves_dirty():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel, config, other = (env["repo"], env["kernel_agent"], env["config"], env["other"])
        branch = "forge/test/kernel"
        # Detach HEAD at the current commit.
        head = _git(repo, "rev-parse", "HEAD")
        subprocess.run(["git", "-C", repo, "checkout", "--detach", head], capture_output=True)
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"

        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None
        _, _, restore = prep
        assert restore["orig_branch"] == "HEAD"

        config.write_text("BLOCK_SIZE = 128\n")
        _commit_all(repo, "iter1")

        forge_submit._restore_inplace(restore)

        # Restored content + detached at original commit + dirty preserved.
        assert config.read_text() == "BLOCK_SIZE = 64\n"
        assert _git(repo, "rev-parse", "HEAD") == head
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
        assert "PRE_EXISTING_DIRTY" in other.read_text()
        assert _git(repo, "branch", "--list", branch).strip() == ""
        print("PASS test_restore_from_detached_head_preserves_dirty")


def test_export_from_best_commit_ignores_unvalidated_worktree(tmp_path):
    env = _make_repo(tmp_path)
    repo = env["repo"]
    kernel = env["kernel_agent"]
    env["other"].write_text("committed_v1\n")
    base_commit = _git(repo, "rev-parse", "HEAD")

    kernel.write_text("KERNEL_VALIDATED_BEST\n")
    _commit_all(repo, "iter1: validated best")
    best_commit = _git(repo, "rev-parse", "HEAD")

    kernel.write_text("KERNEL_UNVALIDATED_TIMEOUT_CANDIDATE\n")
    out = tmp_path / "out"
    out.mkdir()

    forge_submit._export_best_artifacts(
        repo,
        base_commit,
        str(kernel),
        str(kernel),
        out,
        best_commit=best_commit,
    )

    exported = out / "optimized_versions" / "v1_forge.py"
    patch = (out / "optimized_versions" / "forge.patch").read_text()
    assert exported.read_text() == "KERNEL_VALIDATED_BEST\n"
    assert "KERNEL_VALIDATED_BEST" in patch
    assert "KERNEL_UNVALIDATED_TIMEOUT_CANDIDATE" not in patch


def test_canonical_forge_artifacts_resolve_from_campaign_root(tmp_path):
    workspace = tmp_path / "worktree"
    campaign = workspace / "forge_experiments"
    bundle = campaign / "best" / "iter_003"
    files = bundle / "files"
    files.mkdir(parents=True)
    (bundle / "forge.patch").write_text("diff --git a/a.py b/a.py\n")
    (files / "a.py").write_text("VALUE = 2\n")
    manifest = {
        "schema_version": 1,
        "artifact_dir": "best/iter_003",
        "patch_path": "best/iter_003/forge.patch",
        "changed_files": ["a.py"],
    }
    (campaign / "best" / "manifest.json").write_text(json.dumps(manifest))

    normalized = forge_submit._canonical_forge_artifacts(
        str(workspace),
        manifest,
    )

    assert normalized["best_manifest"] == str(campaign / "best" / "manifest.json")
    assert normalized["canonical_patch_path"] == str(bundle / "forge.patch")
    assert normalized["canonical_files_root"] == str(files)
    assert normalized["changed_files"] == ["a.py"]


def test_canonical_forge_artifacts_reject_path_escape(tmp_path, caplog):
    workspace = tmp_path / "worktree"
    campaign = workspace / "forge_experiments" / "best"
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text("{}")

    assert (
        forge_submit._canonical_forge_artifacts(
            str(workspace),
            {
                "artifact_dir": "../outside",
                "patch_path": "../outside.patch",
                "changed_files": ["../escape.py"],
            },
        )
        == {}
    )
    assert "changed file path escapes" in caplog.text


def test_canonical_forge_artifacts_logs_missing_files_root(tmp_path, caplog):
    workspace = tmp_path / "worktree"
    campaign = workspace / "forge_experiments"
    bundle = campaign / "best" / "iter_003"
    bundle.mkdir(parents=True)
    (bundle / "forge.patch").write_text(
        "diff --git a/a.py b/a.py\n",
        encoding="utf-8",
    )
    (campaign / "best" / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert (
        forge_submit._canonical_forge_artifacts(
            str(workspace),
            {
                "artifact_dir": "best/iter_003",
                "patch_path": "best/iter_003/forge.patch",
                "changed_files": ["a.py"],
            },
        )
        == {}
    )
    assert "expected files directory does not exist" in caplog.text
    assert str(bundle / "files") in caplog.text


def test_canonical_forge_artifacts_rejects_files_symlink_escape(
    tmp_path,
    caplog,
):
    workspace = tmp_path / "worktree"
    campaign = workspace / "forge_experiments"
    bundle = campaign / "best" / "iter_003"
    bundle.mkdir(parents=True)
    (bundle / "forge.patch").write_text(
        "diff --git a/a.py b/a.py\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside-files"
    outside.mkdir()
    (outside / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    (bundle / "files").symlink_to(outside, target_is_directory=True)
    (campaign / "best" / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    assert (
        forge_submit._canonical_forge_artifacts(
            str(workspace),
            {
                "artifact_dir": "best/iter_003",
                "patch_path": "best/iter_003/forge.patch",
                "changed_files": ["a.py"],
            },
        )
        == {}
    )
    assert "files directory resolves outside" in caplog.text
    assert str(outside) in caplog.text


def test_validated_checkpoint_requires_commit_metrics_and_coverage(tmp_path, caplog):
    env = _make_repo(tmp_path)
    repo = env["repo"]
    kernel = env["kernel_agent"]
    env["other"].write_text("committed_v1\n")
    base_commit = _git(repo, "rev-parse", "HEAD")
    kernel.write_text("KERNEL_VALIDATED_BEST\n")
    _commit_all(repo, "iter1: validated best")
    best_commit = _git(repo, "rev-parse", "HEAD")
    shapes = {
        "validation": [
            {"CASE_ID": "case_001"},
            {"CASE_ID": "case_002"},
        ]
    }
    checkpoint = {
        "schema_version": 1,
        "experiment_id": "hyperloom",
        "state": "best_committed",
        "base_commit": base_commit,
        "best_commit": best_commit,
        "baseline_ms": 1.0,
        "best_ms": 0.8,
        "mean_case_speedup": 1.25,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "validation_passed": True,
        "case_coverage": shapes["validation"],
    }

    recovered = forge_submit._validated_forge_checkpoint(
        checkpoint,
        workspace=repo,
        base_commit=base_commit,
        shapes=shapes,
    )

    assert recovered is not None
    assert recovered["best_commit"] == best_commit
    assert recovered["improved"] is True
    checkpoint["case_coverage"] = [{"CASE_ID": "case_001"}]
    with caplog.at_level(logging.WARNING):
        assert (
            forge_submit._validated_forge_checkpoint(
                checkpoint,
                workspace=repo,
                base_commit=base_commit,
                shapes=shapes,
            )
            is None
        )
    # Discarding a KEEP the producer already published is the expensive outcome
    # here, so it may not be inferred from a return value alone.
    assert "case coverage mismatch" in caplog.text


def test_a_checkpoint_that_reports_no_coverage_is_still_recoverable(tmp_path):
    """forge-loop stopped reporting case coverage when drivers took over the suite.

    Vetoing on the field's absence throws away every salvageable best from a
    timed-out campaign, silently: the run just looks like it recovered nothing.
    """
    env = _make_repo(tmp_path)
    repo = env["repo"]
    kernel = env["kernel_agent"]
    env["other"].write_text("committed_v1\n")
    base_commit = _git(repo, "rev-parse", "HEAD")
    kernel.write_text("KERNEL_VALIDATED_BEST\n")
    _commit_all(repo, "iter1: validated best")
    best_commit = _git(repo, "rev-parse", "HEAD")
    shapes = {"validation": [{"CASE_ID": "case_001"}, {"CASE_ID": "case_002"}]}
    checkpoint = {
        "schema_version": 1,
        "experiment_id": "hyperloom",
        "state": "best_committed",
        "base_commit": base_commit,
        "best_commit": best_commit,
        "baseline_ms": 1.0,
        "best_ms": 0.8,
        "mean_case_speedup": 1.25,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "validation_passed": True,
    }

    def recover(payload):
        return forge_submit._validated_forge_checkpoint(
            payload,
            workspace=repo,
            base_commit=base_commit,
            shapes=shapes,
        )

    # Absent: a current forge-loop reports no coverage key at all.
    assert recover(dict(checkpoint)) is not None
    # Empty: an older forge-loop reports the key with nothing in it.
    assert recover({**checkpoint, "case_coverage": []}) is not None
    # Present and disagreeing still vetoes: that is evidence, not silence.
    assert recover({**checkpoint, "case_coverage": [{"CASE_ID": "case_001"}]}) is None


def test_submit_salvages_validated_best_after_timeout(tmp_path, monkeypatch):
    env = _make_repo(tmp_path)
    repo = env["repo"]
    kernel = env["kernel_agent"]
    env["other"].write_text("committed_v1\n")
    base_commit = _git(repo, "rev-parse", "HEAD")
    kernel.write_text("KERNEL_VALIDATED_BEST\n")
    _commit_all(repo, "iter1: validated best")
    best_commit = _git(repo, "rev-parse", "HEAD")
    kernel.write_text("KERNEL_UNVALIDATED_TIMEOUT_CANDIDATE\n")

    prompt = tmp_path / "prompt.md"
    prompt.write_text("# optimize\n")
    output_dir = tmp_path / "forge-output"
    checkpoint = {
        "schema_version": 1,
        "experiment_id": "hyperloom",
        "state": "best_committed",
        "base_commit": base_commit,
        "best_commit": best_commit,
        "baseline_ms": 1.0,
        "best_ms": 0.8,
        "mean_case_speedup": 1.25,
        "search_start_mean_case_speedup": 1.0,
        "total_improved": True,
        "incremental_improved": True,
        "improved": True,
        "validation_passed": True,
        "case_coverage": [],
    }
    assert (
        forge_submit._validated_forge_checkpoint(
            checkpoint,
            workspace=repo,
            base_commit=base_commit,
            shapes={
                "primary": {},
                "minimal": {},
                "validation": [],
            },
        )
        is not None
    )

    monkeypatch.setattr(forge_submit, "_needs_inplace", lambda _repo: False)
    monkeypatch.setattr(
        forge_submit,
        "_prepare_worktree",
        lambda *_args, **_kwargs: (repo, str(kernel), base_commit),
    )
    monkeypatch.setattr(
        forge_submit,
        "_resolve_gpu_target",
        lambda _candidate: "gfx942",
    )
    monkeypatch.setattr(
        forge_submit,
        "_run_loop_via_cli",
        lambda **_kwargs: forge_submit.ForgeLoopOutcome(
            baseline_ms=1.0,
            best_ms=0.8,
            improved=True,
            output="timed out",
            error=RuntimeError("timeout"),
            timed_out=True,
            checkpoint=checkpoint,
        ),
    )
    monkeypatch.setattr(
        forge_submit,
        "_remove_worktree",
        lambda *_args, **_kwargs: None,
    )

    result = forge_submit.submit(
        source_file=str(kernel),
        prompt_file=prompt,
        output_dir=output_dir,
        source_type="triton",
        candidate={"operation": "unsupported_op"},
        timeout_s=60,
        kernel_repo=repo,
    )

    assert result["returncode"] == 0, result.get("stderr_tail")
    assert result["timed_out"] is True
    assert result["salvaged"] is True
    assert result["best_commit"] == best_commit
    exported = output_dir / "optimized_versions" / "v1_forge.py"
    assert exported.read_text() == "KERNEL_VALIDATED_BEST\n"
    assert "KERNEL_UNVALIDATED_TIMEOUT_CANDIDATE" not in (output_dir / "optimized_versions" / "forge.patch").read_text()


def test_default_branch_resolves_main():
    with tempfile.TemporaryDirectory() as td:
        env = _make_repo(Path(td))
        # No origin/HEAD configured -> falls back to a local 'main'/'master'.
        assert forge_submit._default_branch(env["repo"]) == "main"
        print("PASS test_default_branch_resolves_main")


def test_prepare_inplace_autorecovers_from_stale_forge_branch():
    """After a crash leaves the repo stranded on a stale forge temp branch,
    _prepare_inplace must auto-recover (force-checkout the default branch, drop
    the stale branch, snapshot a pristine baseline) instead of refusing."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        env = _make_repo(tmp)
        repo, kernel = env["repo"], env["kernel_agent"]

        # Strand HEAD on a forge/ temp branch carrying a leftover optimization.
        stale = "forge/20990101T000000Z/kernel"
        subprocess.run(["git", "-C", repo, "checkout", "-b", stale], capture_output=True)
        kernel.write_text("KERNEL_OPTIMIZED_LEFTOVER\n")
        _commit_all(repo, "forge: leftover optimization from crashed run")
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == stale

        branch = "forge/newrun/kernel"
        prep = forge_submit._prepare_inplace(str(kernel), repo, branch)
        assert prep is not None, "auto-recover must yield a usable prep, not None"
        _, _, restore = prep

        # Recovery force-checked-out main, so the leftover optimization is gone
        # and the baseline snapshot is pristine.
        assert kernel.read_text() == "KERNEL_ORIGINAL\n", "must recover to pristine kernel"
        # Stale forge branch deleted; restore targets the recovered default branch.
        assert _git(repo, "branch", "--list", stale).strip() == "", "stale forge branch must be deleted"
        assert restore["orig_branch"] == "main", f"orig_branch should be recovered main: {restore}"

        forge_submit._restore_inplace(restore)
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
        assert _git(repo, "branch", "--list", branch).strip() == "", "run temp branch must be deleted"
        print("PASS test_prepare_inplace_autorecovers_from_stale_forge_branch")


def test_terminate_forge_process_sweeps_group_after_parent_exit(monkeypatch):
    calls = []

    class Proc:
        pid = 123

        @staticmethod
        def communicate(timeout=None):
            return "stdout", "stderr"

    monkeypatch.setattr(forge_submit, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(
        forge_submit.os,
        "killpg",
        lambda pgid, sig: calls.append((pgid, sig)),
    )

    out, err = forge_submit._terminate_forge_process(Proc(), grace_sec=0)

    assert (out, err) == ("stdout", "stderr")
    assert calls == [
        (123, forge_submit.signal.SIGTERM),
        (123, forge_submit.signal.SIGKILL),
    ]


def test_terminate_forge_process_reports_unreaped_group(
    monkeypatch,
    caplog,
):
    group_signals = []
    process_signals = []

    class Proc:
        pid = 123

        @staticmethod
        def communicate(timeout=None):
            raise subprocess.TimeoutExpired("forge-loop", timeout)

        @staticmethod
        def kill():
            raise AssertionError("killpg succeeded; direct fallback is invalid")

    monkeypatch.setattr(forge_submit, "_descendant_processes", lambda _pid: [])
    monkeypatch.setattr(
        forge_submit,
        "_process_group_members",
        lambda _pgid: [(456, 99, "D")],
    )
    monkeypatch.setattr(
        forge_submit,
        "_proc_identity",
        lambda _pid: (1, 99),
    )
    monkeypatch.setattr(
        forge_submit.os,
        "killpg",
        lambda pgid, sig: group_signals.append((pgid, sig)),
    )
    monkeypatch.setattr(
        forge_submit.os,
        "kill",
        lambda pid, sig: process_signals.append((pid, sig)),
    )

    with caplog.at_level(logging.WARNING, logger=forge_submit.log.name):
        out, err = forge_submit._terminate_forge_process(
            Proc(),
            grace_sec=0,
        )

    assert (out, err) == ("", "")
    assert group_signals == [
        (123, forge_submit.signal.SIGTERM),
        (123, forge_submit.signal.SIGKILL),
        (123, forge_submit.signal.SIGKILL),
    ]
    assert process_signals == [(456, forge_submit.signal.SIGKILL)]
    assert "was not reaped after SIGKILL" in caplog.text


if __name__ == "__main__":
    test_export_and_restore_cover_sibling_file()
    test_restore_from_detached_head_preserves_dirty()
    test_default_branch_resolves_main()
    test_prepare_inplace_autorecovers_from_stale_forge_branch()
    print("ALL TESTS PASSED")
