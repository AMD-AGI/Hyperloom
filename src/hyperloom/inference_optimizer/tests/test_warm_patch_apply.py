"""Tests for baseline executor warm-replay patch application (_apply_warm_patches)."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors.baseline import (
    _apply_warm_patches,
    _materialize_warm_env_path_refs,
    _prepare_warm_kernel_rebuild,
    _revert_applied_patch,
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


def test_reverse_patch_preserves_unrelated_dirty_and_untracked_files(
    fake_repo,
    output_dir,
):
    unrelated = fake_repo / "unrelated.txt"
    unrelated.write_text("committed\n")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=fake_repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add unrelated"],
        cwd=fake_repo,
        check=True,
        capture_output=True,
    )
    unrelated.write_text("operator change\n")
    untracked = fake_repo / "jit-cache.bin"
    untracked.write_text("cache")
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
    assert len(result) == 1
    _revert_applied_patch(
        result[0]["target_repo"],
        result[0]["patch_path"],
    )
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()
    assert unrelated.read_text() == "operator change\n"
    assert untracked.read_text() == "cache"


def test_bundle_patch_with_wrong_base_sha_is_skipped(fake_repo, output_dir):
    result = _apply_warm_patches(
        {
            "patches": [
                {
                    "patch_file": "vllm/fp8.py",
                    "patch_content": VALID_PATCH,
                    "target_repo": "inferencex",
                    "base_sha": "deadbeef",
                }
            ]
        },
        str(fake_repo),
        output_dir,
    )
    assert result == []
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()


def test_bundle_patch_resolves_aiter_meta_repo(
    tmp_path,
    output_dir,
    monkeypatch,
):
    repo = tmp_path / "aiter_meta"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    target = repo / "kernel.py"
    target.write_text("value = 1\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@test.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "base",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    patch_content = """\
diff --git a/kernel.py b/kernel.py
--- a/kernel.py
+++ b/kernel.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths.resolve_patch_target_roots",
        lambda: [f"{repo}/"],
    )
    result = _apply_warm_patches(
        {
            "patches": [
                {
                    "patch_file": "aiter-kernel.diff",
                    "patch_content": patch_content,
                    "target_repo": "aiter",
                    "base_sha": base_sha,
                }
            ]
        },
        "",
        output_dir,
    )
    assert len(result) == 1
    assert Path(result[0]["target_repo"]) == repo
    assert target.read_text() == "value = 2\n"


def test_aiter_cpp_patch_prepares_kernelforge_jit(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    package = types.ModuleType("kernel_agents")
    package.__path__ = []
    loop = types.ModuleType("kernel_agents.loop")
    loop.__path__ = []
    module = types.ModuleType("kernel_agents.loop.jit_rebuild")
    module.force_jit_rebuild = lambda paths: calls.append(list(paths))
    monkeypatch.setitem(sys.modules, "kernel_agents", package)
    monkeypatch.setitem(sys.modules, "kernel_agents.loop", loop)
    monkeypatch.setitem(sys.modules, "kernel_agents.loop.jit_rebuild", module)
    repo = tmp_path / "aiter_meta"
    ok, reason = _prepare_warm_kernel_rebuild(
        [
            {
                "target_repo": str(repo),
                "changed_files": ["csrc/kernels/tuned_kernel.hip"],
            }
        ]
    )
    assert ok is True
    assert reason == ""
    assert calls == [[str(repo / "csrc/kernels/tuned_kernel.hip")]]


def test_portable_env_path_materializes_from_applied_repo(tmp_path):
    repo = tmp_path / "aiter_meta"
    tuned = repo / "configs" / "tuned.csv"
    tuned.parent.mkdir(parents=True)
    tuned.write_text("M,N,K\n")
    params = {
        "extra_envs": {"STATIC": "1"},
        "env_path_refs": {
            "AITER_CONFIG": {
                "repo": "aiter",
                "path": "configs/tuned.csv",
            }
        },
    }
    ok, reason = _materialize_warm_env_path_refs(
        params,
        [{"logical_repo": "aiter", "target_repo": str(repo)}],
    )
    assert ok is True
    assert reason == ""
    assert params["extra_envs"] == {
        "STATIC": "1",
        "AITER_CONFIG": str(tuned),
    }


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
