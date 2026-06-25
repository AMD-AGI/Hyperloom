"""Tests for baseline executor warm-replay patch application (_apply_warm_patches)."""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.baseline import (
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
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo), capture_output=True, check=True,
    )
    # Create a file to patch
    (repo / "vllm").mkdir()
    (repo / "vllm" / "fp8.py").write_text("# fp8 module\noriginal = True\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(repo), capture_output=True, check=True,
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
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": VALID_PATCH,
            "patch_ref": "",
            "measured_gain_pct": 24.9,
            "repo": "ROCm/vllm",
        }],
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
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": VALID_PATCH,
            "patch_ref": "",
            "measured_gain_pct": 24.9,
            "repo": "ROCm/vllm",
        }],
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
        "patches": [{
            "patch_file": "bad.py",
            "patch_content": "this is not a valid diff",
            "patch_ref": "",
            "measured_gain_pct": 5.0,
            "repo": "x/y",
        }],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_patch_ref_fallback(fake_repo, output_dir, tmp_path):
    """When patch_content is empty, read from patch_ref file."""
    ref_file = tmp_path / "my.patch"
    ref_file.write_text(VALID_PATCH)
    params = {
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": "",
            "patch_ref": str(ref_file),
            "measured_gain_pct": 10.0,
            "repo": "ROCm/vllm",
        }],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert len(result) == 1
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" in content


def test_patch_ref_missing_file_skipped(fake_repo, output_dir):
    """Non-existent patch_ref should be skipped."""
    params = {
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": "",
            "patch_ref": "/nonexistent/path.patch",
            "measured_gain_pct": 10.0,
            "repo": "ROCm/vllm",
        }],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_no_content_no_ref_skipped(fake_repo, output_dir):
    """Entry with neither content nor ref should be skipped."""
    params = {
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": "",
            "patch_ref": "",
            "measured_gain_pct": 10.0,
            "repo": "ROCm/vllm",
        }],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_git_apply_timeout_skipped(fake_repo, output_dir):
    """Timeout during git apply should be skipped gracefully."""
    params = {
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": VALID_PATCH,
            "patch_ref": "",
            "measured_gain_pct": 10.0,
            "repo": "ROCm/vllm",
        }],
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
        "patches": [{
            "patch_file": "vllm/fp8.py",
            "patch_content": "",
            "patch_ref": str(ref_file),
            "measured_gain_pct": 10.0,
            "repo": "ROCm/vllm",
        }],
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    # Either skipped (permission denied) or read succeeds (root can read anything)
    # Either way, should not crash
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
