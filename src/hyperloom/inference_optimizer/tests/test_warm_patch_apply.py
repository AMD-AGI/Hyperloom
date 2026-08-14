"""Tests for baseline executor warm-replay patch application (_apply_warm_patches)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
    _apply_warm_patches,
)


@pytest.fixture
def fake_repo(tmp_path):
    """Create a minimal git repo for testing git apply."""
    repo = tmp_path / "inferencex"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    # Create a file to patch
    (repo / "vllm").mkdir()
    (repo / "vllm" / "fp8.py").write_text("# fp8 module\noriginal = True\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    return repo


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


VALID_PATCH = """\
diff --git a/vllm/fp8.py b/vllm/fp8.py
index 0000000..1111111 100644
--- a/vllm/fp8.py
+++ b/vllm/fp8.py
@@ -1,2 +1,3 @@
 # fp8 module
 original = True
+patched = True
"""


def test_apply_single_patch(fake_repo, output_dir):
    """Successfully apply a single patch."""
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": VALID_PATCH,
                "patch_ref": "",
                "measured_gain_pct": 24.9,
                "repo": "ROCm/vllm",
            }
        ],
        "blocked_patches": [],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert len(result) == 1
    assert result[0]["patch_file"] == "vllm/fp8.py"
    # Verify file was patched
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" in content


def test_blocked_patch_skipped(fake_repo, output_dir):
    """Patches in blocked_patches should be skipped."""
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": VALID_PATCH,
                "patch_ref": "",
                "measured_gain_pct": 24.9,
                "repo": "ROCm/vllm",
            }
        ],
        "blocked_patches": [{"patch_file": "vllm/fp8.py"}],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" not in content


def test_no_patches_returns_empty(output_dir):
    """No patches -> empty result, no crash."""
    params = {"patches": [], "blocked_patches": []}
    result = _apply_warm_patches(params, "/some/path", output_dir)
    assert result == []


def test_empty_target_repo_returns_empty(output_dir):
    """Empty target_repo -> skip."""
    params = {
        "patches": [{"patch_file": "x.py", "patch_content": "diff...", "patch_ref": ""}],
    }
    result = _apply_warm_patches(params, "", output_dir)
    assert result == []


def test_invalid_patch_skipped(fake_repo, output_dir):
    """A malformed patch should be skipped, not crash."""
    params = {
        "patches": [
            {
                "patch_file": "bad.py",
                "patch_content": "this is not a valid diff",
                "patch_ref": "",
                "measured_gain_pct": 5.0,
                "repo": "x/y",
            }
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_patch_ref_fallback(fake_repo, output_dir, tmp_path):
    """When patch_content is empty, read from patch_ref file."""
    ref_file = tmp_path / "my.patch"
    ref_file.write_text(VALID_PATCH)
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": "",
                "patch_ref": str(ref_file),
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            }
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert len(result) == 1
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" in content


def test_patch_ref_missing_file_skipped(fake_repo, output_dir):
    """Non-existent patch_ref should be skipped."""
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": "",
                "patch_ref": "/nonexistent/path.patch",
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            }
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_no_content_no_ref_skipped(fake_repo, output_dir):
    """Entry with neither content nor ref should be skipped."""
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": "",
                "patch_ref": "",
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            }
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_git_apply_timeout_skipped(fake_repo, output_dir):
    """Timeout during git apply should be skipped gracefully."""
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": VALID_PATCH,
                "patch_ref": "",
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            }
        ],
    }
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30)):
        result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_patch_ref_read_oserror_skipped(fake_repo, output_dir, tmp_path):
    """OSError when reading patch_ref should be skipped."""
    ref_file = tmp_path / "unreadable.patch"
    ref_file.write_text("dummy")
    ref_file.chmod(0o000)
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": "",
                "patch_ref": str(ref_file),
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            }
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    # Skipped or read-succeeds depending on privileges; must not crash.
    assert isinstance(result, list)
    ref_file.chmod(0o644)  # cleanup


def test_multiple_patches_partial_success(fake_repo, output_dir):
    """When first patch fails and second succeeds, only second is in result."""
    bad_patch = "this is not a valid diff at all\n"
    params = {
        "patches": [
            {
                "patch_file": "bad.py",
                "patch_content": bad_patch,
                "patch_ref": "",
                "measured_gain_pct": 20.0,
                "repo": "ROCm/vllm",
            },
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": VALID_PATCH,
                "patch_ref": "",
                "measured_gain_pct": 10.0,
                "repo": "ROCm/vllm",
            },
        ],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert len(result) == 1
    assert result[0]["patch_file"] == "vllm/fp8.py"


def test_non_diff_patch_content_is_skipped(fake_repo, output_dir):
    # SWSPLAT-42326: KB-sourced patch_content that is not a unified diff must be
    # skipped before git apply (never applied), leaving the tree untouched.
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": "this is not a diff; just prose",
                "patch_ref": "",
                "repo": "ROCm/vllm",
            }
        ],
        "blocked_patches": [],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()


def test_tree_escaping_patch_content_is_skipped(fake_repo, output_dir):
    # SWSPLAT-42326: a patch whose header path escapes the tree (absolute path)
    # must be skipped, not git-applied.
    escaping = VALID_PATCH.replace("a/vllm/fp8.py", "/etc/evil").replace("b/vllm/fp8.py", "/etc/evil")
    params = {
        "patches": [
            {
                "patch_file": "vllm/fp8.py",
                "patch_content": escaping,
                "patch_ref": "",
                "repo": "ROCm/vllm",
            }
        ],
        "blocked_patches": [],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_required_patch_present_only_in_dirty_index_is_republished(
    fake_repo, output_dir
):
    params = {
        "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
        "required_patch_timeline": True,
    }
    first = _apply_warm_patches(params, str(fake_repo), output_dir)
    subprocess.run(
        ["git", "add", "vllm/fp8.py"],
        cwd=fake_repo,
        capture_output=True,
        check=True,
    )
    second = _apply_warm_patches(params, str(fake_repo), output_dir)

    assert first["status"] == "prepared"
    assert second["status"] == "prepared"
    assert (
        second["patches"][0]["status"]
        == "present_in_dirty_worktree"
    )


def test_required_patch_contained_in_committed_head_is_already_present(
    fake_repo, output_dir
):
    target = fake_repo / "vllm" / "fp8.py"
    target.write_text("# fp8 module\noriginal = True\npatched = True\n")
    subprocess.run(
        ["git", "add", "vllm/fp8.py"],
        cwd=fake_repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "patch in base"],
        cwd=fake_repo,
        capture_output=True,
        check=True,
    )

    result = _apply_warm_patches(
        {
            "patches": [
                {"patch_file": "p.patch", "patch_content": VALID_PATCH}
            ],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "prepared"
    assert result["patches"][0]["status"] == "already_present"


def test_required_patch_uses_three_way_after_checks_fail(
    fake_repo, output_dir, monkeypatch
):
    calls: list[list[str]] = []

    def _run(command, **_kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=b"0123456789abcdef\n",
                stderr=b"",
            )
        if "ls-files" in command or "diff" in command:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "--3way" in command:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if "--check" in command:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no")
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", _run)
    result = _apply_warm_patches(
        {
            "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "prepared"
    assert result["patches"][0]["status"] == "applied_3way"
    assert any("-R" in command for command in calls)
    assert any("--3way" in command for command in calls)


def test_required_patch_failure_rolls_back_and_stops(
    fake_repo, output_dir
):
    later = VALID_PATCH.replace("patched = True", "later = True")
    result = _apply_warm_patches(
        {
            "patches": [
                {"patch_file": "first.patch", "patch_content": VALID_PATCH},
                {"patch_file": "bad.patch", "patch_content": "not a diff"},
                {"patch_file": "later.patch", "patch_content": later},
            ],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["failed_ref"] == "bad.patch"
    assert len(result["patches"]) == 1
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" not in content
    assert "later = True" not in content


def test_pending_state_persist_failure_needs_no_file_rollback(
    fake_repo,
    output_dir,
) -> None:
    result = _apply_warm_patches(
        {
            "patches": [
                {"patch_file": "p.patch", "patch_content": VALID_PATCH}
            ],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
        before_mutation=lambda _manifest: False,
    )

    assert result["status"] == "failed"
    assert result["failure"] == "pending_state_persist_failed"
    assert result["applied"] == []
    assert result["rollback"] == {"ok": True, "errors": []}
    assert result["rolled_back"] is True
    assert "patched = True" not in (
        fake_repo / "vllm" / "fp8.py"
    ).read_text()


def test_required_patch_refuses_repo_without_head(tmp_path, output_dir):
    repo = tmp_path / "unborn"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    target = repo / "vllm" / "fp8.py"
    target.parent.mkdir()
    target.write_text("# fp8 module\noriginal = True\n")

    result = _apply_warm_patches(
        {
            "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
            "required_patch_timeline": True,
        },
        str(repo),
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["failure"] == "missing_git_head"
    assert "patched = True" not in target.read_text()


def test_legacy_patch_skips_when_rollback_snapshot_fails(
    fake_repo,
    output_dir,
    monkeypatch,
):
    def _fail_snapshot(*_args, **_kwargs):
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.baseline._create_patch_snapshot",
        _fail_snapshot,
    )

    result = _apply_warm_patches(
        {
            "patches": [
                {
                    "patch_file": "vllm/fp8.py",
                    "patch_content": VALID_PATCH,
                }
            ]
        },
        str(fake_repo),
        output_dir,
    )

    assert result == []
    assert "patched = True" not in (
        fake_repo / "vllm" / "fp8.py"
    ).read_text()


def test_three_way_residue_fails_and_rolls_back(
    fake_repo, output_dir, monkeypatch
):
    real_run = subprocess.run
    residue_checks = 0

    def _run(command, **kwargs):
        nonlocal residue_checks
        if command[:3] == ["git", "apply", "--check"]:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no")
        if command[:4] == ["git", "apply", "-R", "--check"]:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"no")
        if command[:3] == ["git", "apply", "--3way"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if command[:3] == ["git", "ls-files", "-u"]:
            residue_checks += 1
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    b""
                    if residue_checks == 1
                    else b"100644 deadbeef 1\tvllm/fp8.py\n"
                ),
                stderr=b"",
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)
    result = _apply_warm_patches(
        {
            "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["failed_ref"] == "p.patch"
    assert result["rolled_back"] is True


def test_required_rollback_preserves_unrelated_dirty_checkout(
    fake_repo, output_dir
):
    unrelated = fake_repo / "notes.txt"
    unrelated.write_text("committed\n")
    subprocess.run(
        ["git", "add", "notes.txt"],
        cwd=fake_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "notes"],
        cwd=fake_repo,
        check=True,
        capture_output=True,
    )
    unrelated.write_text("user dirty work\n")
    missing_target_patch = """\
diff --git a/missing.py b/missing.py
--- a/missing.py
+++ b/missing.py
@@ -1 +1 @@
-old
+new
"""
    result = _apply_warm_patches(
        {
            "patches": [
                {"patch_file": "first.patch", "patch_content": VALID_PATCH},
                {
                    "patch_file": "second.patch",
                    "patch_content": missing_target_patch,
                },
            ],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["rolled_back"] is True
    assert unrelated.read_text() == "user dirty work\n"
    assert "patched = True" not in (
        fake_repo / "vllm" / "fp8.py"
    ).read_text()


def test_rollback_does_not_erase_already_present_patch(
    fake_repo, output_dir
):
    target = fake_repo / "vllm" / "fp8.py"
    target.write_text("# fp8 module\noriginal = True\npatched = True\n")
    subprocess.run(
        ["git", "add", "vllm/fp8.py"],
        cwd=fake_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "patch landed upstream"],
        cwd=fake_repo,
        check=True,
        capture_output=True,
    )
    missing_target_patch = """\
diff --git a/missing.py b/missing.py
--- a/missing.py
+++ b/missing.py
@@ -1 +1 @@
-old
+new
"""
    result = _apply_warm_patches(
        {
            "patches": [
                {"patch_file": "upstream.patch", "patch_content": VALID_PATCH},
                {
                    "patch_file": "missing.patch",
                    "patch_content": missing_target_patch,
                },
            ],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "failed"
    assert target.read_text() == "# fp8 module\noriginal = True\npatched = True\n"


def test_real_git_three_way_merge_succeeds(tmp_path, output_dir):
    repo = tmp_path / "threeway"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    target = repo / "file.txt"
    base_lines = [f"line-{index}" for index in range(20)]
    target.write_text("\n".join(base_lines) + "\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    patched_lines = list(base_lines)
    patched_lines[10] = "PATCHED"
    target.write_text("\n".join(patched_lines) + "\n")
    patch_content = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout.decode()
    subprocess.run(
        ["git", "checkout", "--", "file.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    current_lines = list(base_lines)
    current_lines[7] = "CURRENT"
    target.write_text("\n".join(current_lines) + "\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "context drift"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    result = _apply_warm_patches(
        {
            "patches": [{"patch_file": "real.patch", "patch_content": patch_content}],
            "required_patch_timeline": True,
        },
        str(repo),
        output_dir,
    )

    assert result["status"] == "prepared"
    assert result["patches"][0]["status"] == "applied_3way"
    expected = list(current_lines)
    expected[10] = "PATCHED"
    assert target.read_text() == "\n".join(expected) + "\n"


@pytest.mark.asyncio
async def test_required_failure_does_not_run_config_only_fallback(monkeypatch):
    executor = object.__new__(BaselineExecutor)
    calls: list[dict] = []

    async def _run_once(ctx, **_kwargs):
        calls.append(dict(ctx.task.params))
        return {
            "status": "required_patch_failed",
            "failed_patch_ref": "bad.patch",
            "required_patch_failure": {"patches": [{"status": "failed"}]},
            "warm_replay_rollback": {"ok": True, "errors": []},
        }

    executor._run_once = _run_once  # type: ignore[method-assign]
    executor._maybe_stop_on_missing_baseline_accuracy = lambda *_a: None  # type: ignore[method-assign]
    executor._is_moe_runner_rooted_failure = lambda _r: False  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            kind="replay_warm_recipe",
            params={
                "patches": [{"patch_file": "bad.patch"}],
                "required_patch_timeline": True,
                "extra_server_args": "--recipe --kernel",
                "extra_envs": {"RECIPE": "1", "KERNEL": "1"},
                "recipe_extra_server_args": "--recipe",
                "recipe_extra_envs": {"RECIPE": "1"},
                "warm_kernel_apply_results": [{"status": "ok"}],
                "disable_run_eval": True,
            },
        )
    )

    result = await executor(ctx)

    # No Config/Env-only salvage: the failed timeline is returned as-is so
    # PRELUDE marks the warm replay failed and optimizes from the clean tree.
    assert len(calls) == 1
    assert result["status"] == "required_patch_failed"
    assert "warm_replay_partial" not in result
    assert "promoted_extra_server_args" not in result


@pytest.mark.asyncio
async def test_required_rollback_failure_is_returned_unchanged():
    executor = object.__new__(BaselineExecutor)
    calls = 0

    async def _run_once(_ctx, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "status": "required_patch_rollback_failed",
            "warm_replay_rollback": {
                "ok": False,
                "errors": ["kernel restore failed"],
            },
        }

    executor._run_once = _run_once  # type: ignore[method-assign]
    executor._maybe_stop_on_missing_baseline_accuracy = lambda *_a: None  # type: ignore[method-assign]
    executor._is_moe_runner_rooted_failure = lambda _r: False  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            kind="replay_warm_recipe",
            params={
                "recipe_extra_server_args": "--safe-only",
                "recipe_extra_envs": {"VLLM_SAFE": "1"},
                "disable_run_eval": True,
            },
        )
    )

    result = await executor(ctx)

    # The unverified rollback is surfaced verbatim; PRELUDE's combined rollback
    # guard is what stops the run on a dirty tree.
    assert calls == 1
    assert result["status"] == "required_patch_rollback_failed"
    assert result["warm_replay_rollback"]["ok"] is False
    assert "warm_replay_partial" not in result
