# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``_patch_source_pr``: turning a PR candidate into a diff on disk."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hyperloom.orchestrator.actions.executors._patch_source_pr import (
    PrMaterialization,
    _fetch_diff_to_path,
    _materialize_pr_diff_from_head,
    materialize_candidate_patches,
)

_VALID_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "FRAMEWORK Test"
    env["GIT_AUTHOR_EMAIL"] = "fw-pr@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    return env


def _init_repo_with_pr_branch(path: Path, *, pr_ref: str = "pr-head") -> str:
    """Init ``main`` plus a divergent ``pr_ref`` branch; returns the PR head sha."""
    env = _git_env()
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True, env=env)
    subprocess.run(["git", "-C", str(path), "checkout", "-b", pr_ref], check=True, capture_output=True, env=env)
    (path / "src.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "commit", "-am", "pr head"], check=True, capture_output=True, env=env)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(path), "checkout", "main"], check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", str(path)],
        check=True,
        capture_output=True,
        env=env,
    )
    return head


def test_fetch_diff_to_path_rejects_file_url(tmp_path: Path):
    """A ``file://`` diff_url must not be fetched: the URL reaches us from a remote KB/API response, so honouring it would read the local filesystem."""
    src = tmp_path / "secret.patch"
    src.write_text(_VALID_PATCH, encoding="utf-8")
    dest = tmp_path / "out" / "got.patch"
    ok, err = _fetch_diff_to_path(f"file://{src}", dest, timeout_sec=5.0)
    assert not ok
    assert "unsupported URL scheme" in err
    assert not dest.exists()


def test_fetch_diff_to_path_fails_on_unreachable_url(tmp_path: Path):
    dest = tmp_path / "missing.patch"
    ok, err = _fetch_diff_to_path(
        "http://127.0.0.1:1/does-not-exist.patch",
        dest,
        timeout_sec=2.0,
    )
    assert not ok
    assert err
    assert not dest.exists()


def test_materialize_pr_diff_from_head_extracts_net_diff(tmp_path: Path):
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo, pr_ref="pr-head")
    dest = tmp_path / "out" / "pr.patch"
    cand = {"repo": "x/y", "pr_number": 7, "ref": "pr-head", "head_sha": head_sha}
    out = _materialize_pr_diff_from_head(repo, cand, dest, timeout_sec=60.0)
    assert out.failure is None, out.failure
    text = dest.read_text()
    assert "src.py" in text
    assert "return 2" in text
    assert not (dest.parent / "wt-x-y-pr-7").exists()
    assert (repo / "src.py").read_text().endswith("return 1\n")


def test_materialize_pr_diff_empty_when_no_ref(tmp_path: Path):
    repo = tmp_path / "framework"
    _init_repo_with_pr_branch(repo)
    dest = tmp_path / "pr.patch"
    out = _materialize_pr_diff_from_head(repo, {"repo": "x/y"}, dest, timeout_sec=30.0)
    assert out.failure is not None
    assert "cannot resolve PR head" in out.failure["error"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=_git_env())


def _merge_pr_head(repo: Path, *, revert: bool) -> None:
    """Merge ``pr-head`` into ``main`` with a merge commit, and optionally revert that merge."""
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "other.py")
    _git(repo, "commit", "-qm", "unrelated")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge pr", "pr-head")
    if revert:
        _git(repo, "revert", "--no-edit", "-m", "1", "HEAD")


def _materialize_head(repo: Path, tmp_path: Path, head_sha: str, **extra: object) -> PrMaterialization:
    return materialize_candidate_patches(
        candidate={"repo": "x/y", "ref": "pr-head", "head_sha": head_sha, "prefer_checkout": True, **extra},
        params={},
        framework_root=repo,
        output_root=tmp_path / "out",
        slug="s",
        diff_fetch_timeout_sec=30.0,
    )


def test_materialize_reports_a_merged_pr_head_as_already_present(tmp_path: Path):
    """The tree has the head's copy of every file the PR changes: nothing to apply, nothing failed."""
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo)
    _merge_pr_head(repo, revert=False)

    out = _materialize_head(repo, tmp_path, head_sha, changed_files=["src.py"])

    assert out.failure is None
    assert out.patches == []
    assert out.already_present == [{"diff_range": f"{head_sha}..{head_sha}", "check": "pr_head_files_in_tree"}]


def test_materialize_does_not_report_a_reverted_pr_head_as_already_present(tmp_path: Path):
    """A reverted merge leaves the head in the history but not its change in the tree."""
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo)
    _merge_pr_head(repo, revert=True)

    out = _materialize_head(repo, tmp_path, head_sha, changed_files=["src.py"])

    assert out.already_present == []
    assert out.failure is not None and out.failure["status"] == "fetch_failed"
    assert "empty diff" in out.failure["error"]


def test_materialize_needs_changed_files_that_exist_at_the_head(tmp_path: Path):
    """A path that names nothing at the head would compare equal in any tree, so it proves nothing."""
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo)
    _merge_pr_head(repo, revert=False)

    out = _materialize_head(repo, tmp_path, head_sha, changed_files=["python/src.py"])

    assert out.already_present == []
    assert out.failure is not None


def test_materialize_falls_back_to_the_diff_url_when_the_head_diff_is_empty(tmp_path: Path, monkeypatch):
    """With a diff_url the fetched diff goes to the apply, whose reverse check tells merged from reverted."""
    repo = tmp_path / "framework"
    head_sha = _init_repo_with_pr_branch(repo)
    _merge_pr_head(repo, revert=False)

    def fetch(_url, dest, *, timeout_sec):
        dest.write_text(_VALID_PATCH, encoding="utf-8")
        return True, ""

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._patch_source_pr._fetch_diff_to_path", fetch)

    out = _materialize_head(
        repo, tmp_path, head_sha, changed_files=["src.py"], diff_url="https://example.invalid/pr.diff"
    )

    assert out.already_present == []
    assert out.failure is None
    assert out.mode == "diff_url_fallback"
    assert out.patches == [(tmp_path / "out" / "s.patch").resolve()]


def test_materialize_explicit_patches_are_used_verbatim(tmp_path: Path):
    patch = tmp_path / "given.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    out = materialize_candidate_patches(
        candidate={},
        params={"patches": [str(patch)]},
        framework_root=tmp_path,
        output_root=tmp_path,
        slug="s",
        diff_fetch_timeout_sec=5.0,
    )
    assert out.failure is None
    assert out.mode == "explicit"
    assert [p.name for p in out.patches] == ["given.patch"]


def test_materialize_refuses_to_bench_when_every_explicit_patch_is_missing(tmp_path: Path):
    """An empty patch list would bench the unpatched tree and report the baseline as the candidate's verdict, so it must be a terminal result."""
    out = materialize_candidate_patches(
        candidate={},
        params={"patches": [str(tmp_path / "gone.patch")]},
        framework_root=tmp_path,
        output_root=tmp_path,
        slug="s",
        diff_fetch_timeout_sec=5.0,
    )
    assert out.patches == []
    assert out.failure is not None
    assert out.failure["status"] == "no_patch"
    assert out.failure["error_class"] == "explicit_patches_missing"


def test_fetch_diff_to_path_passes_max_filesize(tmp_path: Path, monkeypatch):
    dest = tmp_path / "out.patch"
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        dest.write_text(_VALID_PATCH, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._patch_source_pr.subprocess.run",
        fake_run,
    )
    from hyperloom.orchestrator.actions.executors._patch_source_pr import DEFAULT_DIFF_MAX_BYTES

    ok, err = _fetch_diff_to_path("https://example.test/pr.diff", dest, timeout_sec=5.0)
    assert ok, err
    assert "--max-filesize" in captured[0]
    assert str(DEFAULT_DIFF_MAX_BYTES) in captured[0]


def test_materialize_reports_no_patch_when_candidate_carries_no_source(tmp_path: Path):
    out = materialize_candidate_patches(
        candidate={},
        params={},
        framework_root=tmp_path,
        output_root=tmp_path,
        slug="s",
        diff_fetch_timeout_sec=5.0,
    )
    assert out.failure is not None
    assert out.failure["status"] == "no_patch"
