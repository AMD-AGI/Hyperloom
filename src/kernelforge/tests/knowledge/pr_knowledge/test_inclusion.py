# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the local-git probe that says whether an upstream PR is in a tree."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from kernelforge.knowledge.pr_knowledge import inclusion
from kernelforge.knowledge.pr_knowledge.inclusion import Inclusion, InclusionMethod, main, probe_inclusion
from kernelforge.llm.git import git
from kernelforge.knowledge.pr_knowledge import mcp_server as server

_MODULE = "kernelforge.knowledge.pr_knowledge.inclusion"


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "--quiet", cwd=root)
    for key, value in (("user.email", "t@t"), ("user.name", "t"), ("commit.gpgsign", "false")):
        git("config", key, value, cwd=root)
    _commit(root, "Initial import")
    return root


def _commit(root: Path, subject: str, body: str = "", name: str | None = None) -> str:
    """Commit ``subject`` as the content of file ``name`` (a new one by default) and return the sha."""
    (root / (name or f"{uuid.uuid4().hex}.txt")).write_text(subject)
    git("add", "-A", cwd=root)
    git("commit", "--quiet", "-m", subject, *(("-m", body) if body else ()), cwd=root)
    return _head(root)


def _head(root: Path) -> str:
    return git("rev-parse", "HEAD", cwd=root).stdout.strip()


def _fork_before(root: Path, upstream_subject: str) -> str:
    """Land ``upstream_subject`` upstream, then work on a fork branched just before it; return the upstream sha."""
    base = _head(root)
    upstream = _commit(root, upstream_subject)
    git("checkout", "--quiet", "-b", "fork", base, cwd=root)
    _commit(root, "Fork-only work")
    return upstream


def _fork_with_cherry_pick(root: Path, upstream_subject: str) -> str:
    """Land ``upstream_subject`` upstream, then cherry-pick it onto a fork branch; return the upstream sha."""
    upstream = _fork_before(root, upstream_subject)
    git("cherry-pick", upstream, cwd=root)
    return upstream


def _probe(root: Path, number: int, merge_commit_sha: str | None = None, commit: str = "HEAD"):
    return probe_inclusion(tree=root, commit=commit, number=number, merge_commit_sha=merge_commit_sha)


@pytest.mark.parametrize(
    ("subject", "body", "number", "expected"),
    [
        pytest.param("Tune the kernel (#12)", "", 12, Inclusion.INCLUDED, id="exact"),
        pytest.param("Tune the kernel (#7) (#12)", "", 12, Inclusion.INCLUDED, id="in-a-trailing-run"),
        pytest.param("Tune the kernel\u2028for small batches (#12)", "", 12, Inclusion.INCLUDED, id="line-separator"),
        pytest.param("Tune the kernel (#123)", "", 12, Inclusion.UNKNOWN, id="prefix-of-a-longer-number"),
        pytest.param("Tune the kernel (#112)", "", 12, Inclusion.UNKNOWN, id="suffix-of-a-longer-number"),
        pytest.param("Tune (#12) for small batches", "", 12, Inclusion.UNKNOWN, id="mid-subject"),
        pytest.param('Revert "Tune the kernel (#12)" (#200)', "", 12, Inclusion.UNKNOWN, id="revert"),
        pytest.param("Tune the kernel", "Backport of (#12)", 12, Inclusion.UNKNOWN, id="body-only"),
    ],
)
def test_only_a_subject_that_names_the_exact_pr_counts(tmp_path, subject, body, number, expected):
    root = _repo(tmp_path)
    sha = _commit(root, subject, body)

    verdict = _probe(root, number)

    assert verdict.inclusion is expected
    if expected is Inclusion.INCLUDED:
        assert verdict.method is InclusionMethod.SUBJECT
        assert sha in verdict.evidence
    else:
        assert verdict.method is InclusionMethod.NONE
        assert "no merge commit sha" in verdict.evidence


def test_a_subject_in_a_non_utf8_log_encoding_still_matches(tmp_path):
    root = _repo(tmp_path)
    git("config", "i18n.logOutputEncoding", "ISO-8859-1", cwd=root)
    _commit(root, "Tune the naïve kernel (#12)")

    assert _probe(root, 12).inclusion is Inclusion.INCLUDED


def test_a_merge_commit_in_the_history_is_included(tmp_path):
    root = _repo(tmp_path)
    merge = _commit(root, "Fuse the epilogue")
    _commit(root, "Later work")

    verdict = _probe(root, 42, merge)

    assert verdict.inclusion is Inclusion.INCLUDED
    assert verdict.method is InclusionMethod.MERGE_COMMIT
    assert merge in verdict.evidence


def test_a_cherry_pick_under_another_subject_is_included_by_patch_id(tmp_path):
    root = _repo(tmp_path)
    upstream = _fork_with_cherry_pick(root, "Fuse the epilogue (#42)")
    git("commit", "--quiet", "--amend", "-m", "[ROCm] Port the epilogue fusion", cwd=root)

    verdict = _probe(root, 42, upstream)

    assert verdict.inclusion is Inclusion.INCLUDED
    assert verdict.method is InclusionMethod.MERGE_COMMIT


@pytest.mark.parametrize(
    ("picked", "expected"),
    [
        pytest.param([0, 1], Inclusion.INCLUDED, id="every-commit"),
        pytest.param([1], Inclusion.UNKNOWN, id="one-commit"),
        pytest.param([], Inclusion.UNKNOWN, id="none"),
    ],
)
def test_every_commit_an_upstream_merge_brought_in_must_be_carried(tmp_path, picked, expected):
    root = _repo(tmp_path)
    base = _head(root)
    trunk = git("symbolic-ref", "--short", "HEAD", cwd=root).stdout.strip()
    git("checkout", "--quiet", "-b", "pr", cwd=root)
    pr_commits = [_commit(root, "Fuse the epilogue"), _commit(root, "Tune the epilogue")]
    git("checkout", "--quiet", trunk, cwd=root)
    _commit(root, "Other upstream work")
    git("merge", "--quiet", "--no-ff", "-m", "Land the epilogue work", "pr", cwd=root)
    merge = _head(root)
    git("checkout", "--quiet", "-b", "fork", base, cwd=root)
    _commit(root, "Fork-only work")
    for index in picked:
        git("cherry-pick", pr_commits[index], cwd=root)

    verdict = _probe(root, 42, merge)

    assert verdict.inclusion is expected
    if expected is Inclusion.UNKNOWN:
        assert f"{len(pr_commits) - len(picked)} non-merge commit(s)" in verdict.evidence


def test_a_merge_commit_git_cannot_find_is_unknown_with_gits_reason(tmp_path):
    """git exits 128 on the merge-commit call, after the subject search found nothing."""
    root = _repo(tmp_path)
    absent = "0123456789abcdef0123456789abcdef01234567"

    verdict = _probe(root, 42, absent)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert verdict.method is InclusionMethod.NONE
    assert "(128)" in verdict.evidence
    assert absent in verdict.evidence


def test_a_git_call_that_hangs_is_unknown(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _commit(root, "Tune the kernel (#12)")

    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired(["git", *args], kwargs["timeout"])

    monkeypatch.setattr(inclusion, "git_bytes", hangs)
    verdict = _probe(root, 12)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert "timed out" in verdict.evidence


@pytest.mark.parametrize("commit", ["no-such-branch", "--output={tmp}/written"], ids=["bad", "leading-dash"])
def test_a_revision_that_names_no_commit_is_unknown_and_never_reaches_git_as_an_option(tmp_path, commit):
    root = _repo(tmp_path)
    _commit(root, "Tune the kernel (#12)")
    revision = commit.format(tmp=tmp_path)

    verdict = _probe(root, 12, commit=revision)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert revision in verdict.evidence
    assert not list(tmp_path.glob("written*"))


def test_a_directory_outside_git_is_unknown(tmp_path):
    assert _probe(tmp_path, 42).inclusion is Inclusion.UNKNOWN


def _serving_any_sha(upstream: Path) -> None:
    """Serve any sha to a filtered clone, as a hosting service does, so a lazy fetch would succeed."""
    for key in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        git("config", key, "true", cwd=upstream)


def test_a_partial_clone_answers_from_the_objects_it_has_and_fetches_nothing(tmp_path):
    upstream = _repo(tmp_path)
    _serving_any_sha(upstream)
    trunk = git("symbolic-ref", "--short", "HEAD", cwd=upstream).stdout.strip()
    merge = _commit(upstream, "Fuse the epilogue")
    git("checkout", "--quiet", "-b", "side", cwd=upstream)
    unfetched = _commit(upstream, "Tune the kernel")
    clone = tmp_path / "clone"
    git("clone", "--quiet", "--filter=blob:none", "--single-branch", "--branch", trunk, upstream.as_uri(), str(clone))
    objects = clone / ".git" / "objects"
    stored = sorted(objects.rglob("*"))

    missing = _probe(clone, 42, unfetched)
    present = _probe(clone, 42, merge)

    assert missing.inclusion is Inclusion.UNKNOWN
    assert present.inclusion is Inclusion.INCLUDED
    assert sorted(objects.rglob("*")) == stored


def test_a_patch_id_that_needs_a_blob_a_partial_clone_lacks_is_unknown_and_fetches_nothing(tmp_path):
    """Both sides add the same path, so only the blobs can tell the two patches apart."""
    upstream = _repo(tmp_path)
    _serving_any_sha(upstream)
    base = _head(upstream)
    merge = _commit(upstream, "Fuse the epilogue", name="epilogue.py")
    git("checkout", "--quiet", "-b", "fork", base, cwd=upstream)
    _commit(upstream, "Fork's own fusion", name="epilogue.py")
    clone = tmp_path / "clone"
    filtered = ("--quiet", "--filter=blob:none", "--no-single-branch", "--branch", "fork")
    git("clone", *filtered, upstream.as_uri(), str(clone))
    objects = clone / ".git" / "objects"
    stored = sorted(objects.rglob("*"))

    verdict = _probe(clone, 42, merge)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert "git gave no answer" in verdict.evidence
    assert sorted(objects.rglob("*")) == stored


@pytest.fixture
def cherry_picked(tmp_path) -> tuple[Path, str]:
    root = _repo(tmp_path)
    return root, _fork_with_cherry_pick(root, "Fuse the epilogue (#42)")


@pytest.mark.parametrize(
    ("fork", "status", "expected"),
    [
        pytest.param(_fork_with_cherry_pick, 0, Inclusion.INCLUDED, id="included"),
        pytest.param(_fork_before, 3, Inclusion.UNKNOWN, id="unknown"),
    ],
)
def test_cli_prints_the_verdict_and_encodes_it_in_the_exit_status(tmp_path, capsys, fork, status, expected):
    root = _repo(tmp_path)
    upstream = fork(root, "Fuse the epilogue (#42)")
    git("commit", "--quiet", "--amend", "-m", "[ROCm] Port the epilogue fusion", cwd=root)

    assert main(["--tree", str(root), "--commit", "HEAD", "--pr", "42", "--merge-commit-sha", upstream]) == status
    record = json.loads(capsys.readouterr().out)
    assert set(record) == {"inclusion", "method", "evidence"}
    assert record["inclusion"] == expected.value


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--tree", "{tree}", "--commit", "HEAD"], id="no-pr"),
        pytest.param(["--tree", "{tree}/absent", "--commit", "HEAD", "--pr", "1"], id="tree-not-a-directory"),
    ],
)
def test_cli_usage_errors_exit_2(tmp_path, argv):
    with pytest.raises(SystemExit) as raised:
        main([part.format(tree=tmp_path) for part in argv])

    assert raised.value.code == 2


def test_cli_reports_a_tree_git_cannot_enter_as_unknown(tmp_path, capsys):
    root = _repo(tmp_path)
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        if os.access(locked, os.X_OK):
            pytest.skip("this user can enter a directory without permissions")
        status = main(["--tree", str(locked), "--commit", "HEAD", "--pr", "42"])
    finally:
        locked.chmod(0o755)

    assert status == 3
    assert json.loads(capsys.readouterr().out)["inclusion"] == Inclusion.UNKNOWN.value


def test_cli_runs_as_a_module(cherry_picked):
    root, _ = cherry_picked

    completed = subprocess.run(
        [sys.executable, "-m", _MODULE, "--tree", str(root), "--commit", "HEAD", "--pr", "42"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["method"] == InclusionMethod.SUBJECT.value


def _tool(arguments: dict) -> dict:
    reply = asyncio.run(server.handle_tool_call("pr_check_inclusion", arguments))
    return json.loads(reply["content"][0]["text"])


def test_mcp_tool_checks_the_tree_head_without_the_service(cherry_picked, monkeypatch):
    root, _ = cherry_picked
    monkeypatch.delenv("PR_KB_REPO", raising=False)
    monkeypatch.setattr(server, "_client", lambda: pytest.fail("must not reach the service"))

    result = _tool({"number": 42, "tree": str(root)})

    assert result["number"] == 42
    assert result["tree"] == str(root)
    assert result["commit"] == "HEAD"
    assert result["inclusion"] == Inclusion.INCLUDED.value
    assert result["method"] == InclusionMethod.SUBJECT.value
    assert _head(root) in result["evidence"]


def test_mcp_tool_uses_the_merge_sha_and_commit_it_is_given(cherry_picked):
    root, upstream = cherry_picked
    git("commit", "--quiet", "--amend", "-m", "[ROCm] Port the epilogue fusion", cwd=root)
    head = _head(root)

    result = _tool({"number": 42, "tree": str(root), "commit": head, "merge_commit_sha": upstream})

    assert result["commit"] == head
    assert result["inclusion"] == Inclusion.INCLUDED.value
    assert result["method"] == InclusionMethod.MERGE_COMMIT.value


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"tree": "{tree}"}, id="no-number"),
        pytest.param({"number": 42}, id="no-tree"),
        pytest.param({"number": 42, "tree": "{tree}/absent"}, id="tree-not-a-directory"),
    ],
)
def test_mcp_tool_rejects_invalid_arguments(tmp_path, arguments):
    filled = {key: value.format(tree=tmp_path) if isinstance(value, str) else value for key, value in arguments.items()}

    with pytest.raises(server.InvalidParamsError):
        _tool(filled)
