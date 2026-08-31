"""Tests for baseline executor warm-replay patch application (_apply_warm_patches)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
    _apply_warm_patches,
    _create_patch_snapshot,
    _restore_patch_snapshot,
    _revert_patches,
    _revert_warm_patch_state,
    _revert_warm_patch_trees,
)


@pytest.fixture
def fake_repo(tmp_path):
    """Create a minimal git repo for testing git apply."""
    repo = tmp_path / "inferencex"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
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


def _require_patch_cli() -> None:
    if not shutil.which("patch"):
        pytest.skip("patch CLI unavailable")


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
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert len(result) == 1
    assert result[0]["patch_file"] == "vllm/fp8.py"
    # Verify file was patched
    content = (fake_repo / "vllm" / "fp8.py").read_text()
    assert "patched = True" in content


def test_no_patches_returns_empty(output_dir):
    """No patches -> empty result, no crash."""
    params = {"patches": []}
    result = _apply_warm_patches(params, "/some/path", output_dir)
    assert result == []


def test_required_recipe_patch_fails_when_no_overlay_records_a_root(output_dir):
    """Nothing is probed for, so an overlay naming no checkout has nowhere to go."""
    params = {
        "required_patch_timeline": True,
        "patches": [{"patch_file": "framework/p.patch", "patch_content": VALID_PATCH}],
    }

    result = _apply_warm_patches(params, "", output_dir)

    assert result["status"] == "failed"
    assert result["failure"] == "missing_target_repo"


def _git_repo(root, rel_path, body):
    """A committed one-file git checkout, standing in for a recorded apply root."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(root), capture_output=True, check=True)
    for key, value in (("user.email", "t@t.com"), ("user.name", "T"), ("core.autocrlf", "false")):
        subprocess.run(["git", "config", key, value], cwd=str(root), capture_output=True, check=True)
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(root), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(root), capture_output=True, check=True)
    return root


def _one_line_diff(rel_path, before, after):
    return (
        f"diff --git a/{rel_path} b/{rel_path}\n--- a/{rel_path}\n+++ b/{rel_path}\n@@ -1 +1 @@\n-{before}\n+{after}\n"
    )


def test_overlays_from_two_trees_each_apply_into_their_own(tmp_path, output_dir):
    """A framework patch and a data-file overlay replay against their own checkouts.

    Resolving one root for the set would place one of them on a tree it was
    never measured against, so each entry carries the root it was recorded with.
    """
    sglang = _git_repo(tmp_path / "sglang", "python/sglang/layer.py", "original\n")
    tuning = _git_repo(tmp_path / "tuning", "shapes.csv", "1,2,3\n")
    params = {
        "required_patch_timeline": True,
        "patches": [
            {
                "patch_file": "patch/overlays/000001/00-sglang.patch",
                "patch_content": _one_line_diff("python/sglang/layer.py", "original", "patched"),
                "framework_root": str(sglang),
            },
            {
                "patch_file": "patch/overlays/000004/00-tuned-csv.patch",
                "patch_content": _one_line_diff("shapes.csv", "1,2,3", "4,5,6"),
                "framework_root": str(tuning),
            },
        ],
    }

    result = _apply_warm_patches(params, "", output_dir)

    assert result["status"] == "prepared"
    assert (sglang / "python" / "sglang" / "layer.py").read_text() == "patched\n"
    assert (tuning / "shapes.csv").read_text() == "4,5,6\n"
    # Each tree carries its own restore material, since one tree's snapshot
    # cannot restore another.
    assert [tree["root"] for tree in result["trees"]] == [str(sglang), str(tuning)]
    assert all(tree["snapshot_manifest"] for tree in result["trees"])


def test_one_overlay_failing_restores_every_tree_already_patched(tmp_path, output_dir):
    """The gain came from the whole set, so a partial application is not a smaller win."""
    sglang = _git_repo(tmp_path / "sglang", "python/sglang/layer.py", "original\n")
    tuning = _git_repo(tmp_path / "tuning", "shapes.csv", "1,2,3\n")
    params = {
        "required_patch_timeline": True,
        "patches": [
            {
                "patch_file": "patch/overlays/000001/00-sglang.patch",
                "patch_content": _one_line_diff("python/sglang/layer.py", "original", "patched"),
                "framework_root": str(sglang),
            },
            {
                "patch_file": "patch/overlays/000004/00-tuned-csv.patch",
                # Pre-image the second tree does not hold, so this cannot apply.
                "patch_content": _one_line_diff("shapes.csv", "9,9,9", "4,5,6"),
                "framework_root": str(tuning),
            },
        ],
    }

    result = _apply_warm_patches(params, "", output_dir)

    assert result["status"] == "failed"
    assert result["failure"] == "git_apply_failed"
    assert result["rolled_back"] is True
    assert (sglang / "python" / "sglang" / "layer.py").read_text() == "original\n"
    assert (tuning / "shapes.csv").read_text() == "1,2,3\n"


def test_two_roots_in_one_checkout_collapse_to_that_checkout(tmp_path, output_dir):
    """A recorded /sglang and /sglang/python/sglang are one tree, not two.

    A git diff names paths from the work-tree root and ``git apply`` resolves
    them there, silently ignoring any that fall outside the directory it runs
    in. Applying the narrower root on its own would therefore report success
    having written nothing.
    """
    outer = _git_repo(tmp_path / "sglang", "python/sglang/layer.py", "original\n")
    inner = outer / "python" / "sglang"
    (inner / "backend.py").write_text("backend\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(outer), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "backend"], cwd=str(outer), capture_output=True, check=True)
    params = {
        "required_patch_timeline": True,
        "patches": [
            {
                "patch_file": "patch/overlays/000001/00-outer.patch",
                "patch_content": _one_line_diff("python/sglang/layer.py", "original", "patched"),
                "framework_root": str(outer),
            },
            {
                "patch_file": "patch/overlays/000001/01-inner.patch",
                "patch_content": _one_line_diff("python/sglang/backend.py", "backend", "tuned"),
                "framework_root": str(inner),
            },
        ],
    }

    result = _apply_warm_patches(params, "", output_dir)

    assert result["status"] == "prepared"
    assert [tree["root"] for tree in result["trees"]] == [str(outer)]
    assert (inner / "layer.py").read_text() == "patched\n"
    assert (inner / "backend.py").read_text() == "tuned\n"

    restore = _revert_warm_patch_trees(result["trees"])

    assert restore["ok"] is True, restore["errors"]
    assert (inner / "layer.py").read_text() == "original\n"
    assert (inner / "backend.py").read_text() == "backend\n"


def test_an_absent_recorded_root_fails_the_whole_replay(tmp_path, output_dir):
    """Another host's layout is not this one's, and no other tree stands in for it."""
    sglang = _git_repo(tmp_path / "sglang", "python/sglang/layer.py", "original\n")
    params = {
        "required_patch_timeline": True,
        "patches": [
            {
                "patch_file": "patch/overlays/000001/00-sglang.patch",
                "patch_content": _one_line_diff("python/sglang/layer.py", "original", "patched"),
                "framework_root": str(sglang),
            },
            {
                "patch_file": "patch/overlays/000004/00-elsewhere.patch",
                "patch_content": _one_line_diff("shapes.csv", "1,2,3", "4,5,6"),
                "framework_root": str(tmp_path / "never-checked-out"),
            },
        ],
    }

    result = _apply_warm_patches(params, "", output_dir)

    assert result["status"] == "failed"
    assert result["failure"] == "apply_root_absent"
    # Refused before anything was written, so there is nothing to unwind.
    assert (sglang / "python" / "sglang" / "layer.py").read_text() == "original\n"


def test_no_root_is_ever_probed_for(tmp_path, output_dir, monkeypatch):
    """A tree found by probing is one the gain was never measured on.

    The allowlist search is gone, so an overlay that records no checkout fails
    rather than being placed on whatever tree its diff happens to fit.
    """

    def _fail(*_args, **_kwargs):
        raise AssertionError("warm replay must not probe for an apply root")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.integrate_patch._resolve_framework_root",
        _fail,
    )
    _git_repo(tmp_path / "sglang", "vllm/fp8.py", "# fp8 module\noriginal = True\n")

    result = _apply_warm_patches(
        {
            "required_patch_timeline": True,
            "patches": [{"patch_file": "patch/overlays/000000/00-a.patch", "patch_content": VALID_PATCH}],
        },
        "",
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["failure"] == "missing_target_repo"


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
    }
    result = _apply_warm_patches(params, str(fake_repo), output_dir)
    assert result == []


def test_required_patch_present_only_in_dirty_index_is_republished(fake_repo, output_dir):
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
    assert second["patches"][0]["status"] == "present_in_dirty_worktree"


def test_required_patch_contained_in_committed_head_is_already_present(fake_repo, output_dir):
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
            "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
            "required_patch_timeline": True,
        },
        str(fake_repo),
        output_dir,
    )

    assert result["status"] == "prepared"
    assert result["patches"][0]["status"] == "already_present"


def test_required_patch_uses_three_way_after_checks_fail(fake_repo, output_dir, monkeypatch):
    calls: list[list[str]] = []

    def _run(command, **_kwargs):
        calls.append(command)
        if "rev-parse" in command:
            if "--is-inside-work-tree" in command:
                return SimpleNamespace(
                    returncode=0,
                    stdout="true\n",
                    stderr="",
                )
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


def test_required_patch_failure_rolls_back_and_stops(fake_repo, output_dir):
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
            "patches": [{"patch_file": "p.patch", "patch_content": VALID_PATCH}],
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
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()


def test_snapshot_revert_validates_repo_and_head(
    fake_repo,
    output_dir,
) -> None:
    pre_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=fake_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = _create_patch_snapshot(
        str(fake_repo),
        [VALID_PATCH],
        output_dir,
    )
    target = fake_repo / "vllm/fp8.py"
    target.write_text("# fp8 module\npatched = True\n")

    result = _revert_patches(str(fake_repo), pre_sha, manifest)

    assert result == {"ok": True, "errors": []}
    assert "original = True" in target.read_text()


def test_snapshot_revert_rejects_repo_mismatch(
    fake_repo,
    output_dir,
    tmp_path,
) -> None:
    manifest = _create_patch_snapshot(
        str(fake_repo),
        [VALID_PATCH],
        output_dir,
    )
    other_repo = tmp_path / "other"
    other_repo.mkdir()

    result = _revert_patches(str(other_repo), "", manifest)

    assert result["ok"] is False
    assert result["errors"][0].startswith("repo_mismatch:")


def test_snapshot_revert_rejects_head_mismatch(
    fake_repo,
    output_dir,
) -> None:
    pre_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=fake_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = _create_patch_snapshot(
        str(fake_repo),
        [VALID_PATCH],
        output_dir,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "advance head"],
        cwd=fake_repo,
        capture_output=True,
        check=True,
    )

    result = _revert_patches(str(fake_repo), pre_sha, manifest)

    assert result["ok"] is False
    assert result["errors"][0].startswith("head_mismatch:")


def test_required_timeline_refuses_a_repo_with_no_head(tmp_path, output_dir):
    """prelude promotes this tree against a pre_sha it cannot get here.

    Applying via nogit made the run look prepared and then fail downstream with
    validated_recipe_checkout_incomplete, leaving a half-patched tree behind.
    Refusing up front is the outcome the caller can act on.
    """
    repo = tmp_path / "unborn"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    target = repo / "vllm" / "fp8.py"
    target.parent.mkdir(parents=True)
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
    assert "original = True" in target.read_text(), "must not leave a patched tree"


def test_required_timeline_refuses_a_non_git_install_tree(tmp_path, output_dir):
    """Same contract for an install tree that was never a repo."""
    install_root = tmp_path / "dist-packages"
    target = install_root / "vllm" / "fp8.py"
    target.parent.mkdir(parents=True)
    target.write_text("# fp8 module\noriginal = True\n")

    result = _apply_warm_patches(
        {
            "patches": [{"patch_file": "vllm/fp8.py", "patch_content": VALID_PATCH}],
            "required_patch_timeline": True,
        },
        str(install_root),
        output_dir,
    )

    assert result["status"] == "failed"
    assert result["failure"] == "missing_git_head"
    assert "original = True" in target.read_text()


def test_nogit_still_serves_the_legacy_list(tmp_path, output_dir):
    """Nothing downstream of a legacy patch needs a sha, so nogit stays."""
    _require_patch_cli()
    install_root = tmp_path / "dist-packages"
    target = install_root / "vllm" / "fp8.py"
    target.parent.mkdir(parents=True)
    target.write_text("# fp8 module\noriginal = True\n")

    applied = _apply_warm_patches(
        {"patches": [{"patch_file": "vllm/fp8.py", "patch_content": VALID_PATCH}]},
        str(install_root),
        output_dir,
    )

    assert [p["status"] for p in applied] == ["applied_nogit"]
    assert "patched = True" in target.read_text()
    assert (output_dir / "warm_patches" / "patch_backups").is_dir()


def test_nogit_apply_hands_teardown_the_backups_it_needs(tmp_path, output_dir):
    """A nogit apply has no sha, so its backups are the only way back."""
    _require_patch_cli()
    install_root = tmp_path / "dist-packages"
    target = install_root / "vllm" / "fp8.py"
    target.parent.mkdir(parents=True)
    target.write_text("# fp8 module\noriginal = True\n")
    params = {"patches": [{"patch_file": "vllm/fp8.py", "patch_content": VALID_PATCH}]}

    applied = _apply_warm_patches(params, str(install_root), output_dir)

    assert [p["status"] for p in applied] == ["applied_nogit"]
    assert not params.get("_warm_patch_snapshot_manifest"), "nogit has no git snapshot"
    assert params["_warm_patch_nogit_backups"], "teardown would have nothing to undo"


def test_teardown_undoes_a_nogit_apply(tmp_path):
    """Keying the revert on pre_sha alone leaked nogit patches into later tasks
    that reuse the same checkout."""
    target = tmp_path / "vllm" / "fp8.py"
    target.parent.mkdir(parents=True)
    target.write_text("# fp8 module\noriginal = True\n")
    backup = tmp_path / "backups" / "p__vllm__fp8.py__0000.bak"
    backup.parent.mkdir(parents=True)
    shutil.copy2(target, backup)
    target.write_text("# fp8 module\noriginal = True\npatched = True\n")

    result = _revert_warm_patch_state(
        str(tmp_path),
        pre_sha="",
        nogit_backups=[
            {
                "target": str(target),
                "existed": True,
                "backup_path": str(backup),
                "revert_action": "restore",
            }
        ],
    )

    assert result == {"ok": True, "errors": [], "channel": "nogit"}
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
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()


def test_three_way_residue_fails_and_rolls_back(fake_repo, output_dir, monkeypatch):
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
        if "ls-files" in command and command[command.index("ls-files") :][:2] == ["ls-files", "-u"]:
            residue_checks += 1
            return SimpleNamespace(
                returncode=0,
                stdout=(b"" if residue_checks == 1 else b"100644 deadbeef 1\tvllm/fp8.py\n"),
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


def test_required_rollback_preserves_unrelated_dirty_checkout(fake_repo, output_dir):
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
    assert "patched = True" not in (fake_repo / "vllm" / "fp8.py").read_text()


def test_rollback_does_not_erase_already_present_patch(fake_repo, output_dir):
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="git apply --3way merge baseline is validated on Linux CI/pod",
)
def test_real_git_three_way_merge_succeeds(tmp_path, output_dir):
    repo = tmp_path / "threeway"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
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


# ---------------------------------------------------------------------------
# Snapshot / restore must not depend on guessing the strip level
# ---------------------------------------------------------------------------


# Four components, so only -p2 strips down to the real ``pkg/mod.py``.
_DEEP_PREFIX_PATCH = """\
diff --git a/x/pkg/mod.py b/x/pkg/mod.py
--- a/x/pkg/mod.py
+++ b/x/pkg/mod.py
@@ -1 +1,2 @@
 original = True
+patched = True
"""


def _git_repo_with(tmp_path, rel, body):
    repo = tmp_path / "repo"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(body, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    for cfg in (["user.email", "t@t"], ["user.name", "t"]):
        subprocess.run(["git", "config", *cfg], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=str(repo), capture_output=True, check=True)
    return repo


def test_snapshot_covers_the_path_a_non_default_strip_level_touches(tmp_path, output_dir):
    """The header needs -p2 to reach the real file; the snapshot must still cover it.

    Assuming -p1 records ``x/pkg/mod.py``, which the apply never touches, so the
    restore is a no-op and the candidate's edit survives into the next bench.
    """
    repo = _git_repo_with(tmp_path, "pkg/mod.py", "original = True\n")
    before = (repo / "pkg/mod.py").read_text(encoding="utf-8")

    manifest = _create_patch_snapshot(str(repo), [_DEEP_PREFIX_PATCH], output_dir)
    assert "pkg/mod.py" in [row["path"] for row in manifest["paths"]]

    patch_file = tmp_path / "deep.patch"
    patch_file.write_text(_DEEP_PREFIX_PATCH, encoding="utf-8")
    applied = subprocess.run(
        ["git", "apply", "-p2", str(patch_file)],
        cwd=str(repo),
        capture_output=True,
    )
    assert applied.returncode == 0, applied.stderr.decode()
    assert (repo / "pkg/mod.py").read_text(encoding="utf-8") != before

    result = _restore_patch_snapshot(manifest)
    assert result["ok"], result["errors"]
    assert (repo / "pkg/mod.py").read_text(encoding="utf-8") == before


def test_snapshot_restores_a_file_the_patch_created(tmp_path, output_dir):
    """A created file must be removed on restore, not left behind."""
    repo = _git_repo_with(tmp_path, "pkg/keep.py", "keep = True\n")
    created = repo / "pkg/new.py"

    manifest = _create_patch_snapshot(
        str(repo),
        ["--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+added = True\n"],
        output_dir,
    )
    created.write_text("added = True\n", encoding="utf-8")

    result = _restore_patch_snapshot(manifest)
    assert result["ok"], result["errors"]
    assert not created.exists()
