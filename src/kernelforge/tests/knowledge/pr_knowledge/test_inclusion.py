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
from kernelforge.llm.git import git, git_bytes
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


def _commit(root: Path, subject: str, body: str = "") -> str:
    """Commit one new file under ``subject`` and return the new sha."""
    (root / f"{uuid.uuid4().hex}.txt").write_text(subject)
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


def test_a_merge_commit_in_the_history_is_included(tmp_path):
    root = _repo(tmp_path)
    merge = _commit(root, "Fuse the epilogue")
    _commit(root, "Later work")

    verdict = _probe(root, 42, merge)

    assert verdict.inclusion is Inclusion.INCLUDED
    assert verdict.method is InclusionMethod.MERGE_ANCESTRY
    assert merge in verdict.evidence


def test_a_present_merge_commit_off_the_history_without_a_subject_is_not_included(tmp_path):
    root = _repo(tmp_path)
    merge = _fork_before(root, "Fuse the epilogue")

    verdict = _probe(root, 42, merge)

    assert verdict.inclusion is Inclusion.NOT_INCLUDED
    assert verdict.method is InclusionMethod.MERGE_ANCESTRY
    assert "is not an ancestor" in verdict.evidence


@pytest.mark.parametrize(
    "fork_subject",
    [
        pytest.param(None, id="cherry-pick"),
        pytest.param("Fuse the epilogue (#42) (#7)", id="merged-through-a-fork-pr"),
    ],
)
def test_a_cherry_pick_carrying_the_suffix_is_included(tmp_path, fork_subject):
    """The cherry-pick is a new commit, so ancestry says no and the subject decides."""
    root = _repo(tmp_path)
    upstream = _fork_with_cherry_pick(root, "Fuse the epilogue (#42)")
    if fork_subject:
        git("commit", "--quiet", "--amend", "-m", fork_subject, cwd=root)

    verdict = _probe(root, 42, upstream)

    assert verdict.inclusion is Inclusion.INCLUDED
    assert verdict.method is InclusionMethod.SUBJECT_SUFFIX
    assert _head(root) in verdict.evidence
    assert upstream != _head(root)


@pytest.mark.parametrize(
    ("subject", "body", "number", "expected"),
    [
        pytest.param("Tune the kernel (#12)", "", 12, Inclusion.INCLUDED, id="exact"),
        pytest.param("Tune the kernel (#7) (#12)", "", 12, Inclusion.INCLUDED, id="in-a-trailing-run"),
        pytest.param("Tune the kernel (#123)", "", 123, Inclusion.INCLUDED, id="longer-number-itself"),
        pytest.param("Tune the kernel\u2028for small batches (#12)", "", 12, Inclusion.INCLUDED, id="line-separator"),
        pytest.param("Tune the kernel\x0cfor small batches (#12)", "", 12, Inclusion.INCLUDED, id="form-feed"),
        pytest.param("Tune the kernel (#123)", "", 12, Inclusion.UNKNOWN, id="prefix-of-a-longer-number"),
        pytest.param("Tune the kernel (#112)", "", 12, Inclusion.UNKNOWN, id="suffix-of-a-longer-number"),
        pytest.param("Tune the kernel (#12)", "", 123, Inclusion.UNKNOWN, id="shorter-number"),
        pytest.param("Tune (#12) for small batches", "", 12, Inclusion.UNKNOWN, id="mid-subject"),
        pytest.param('Revert "Tune the kernel (#12)" (#200)', "", 12, Inclusion.UNKNOWN, id="revert"),
        pytest.param("Tune the kernel", "Backport of (#12)", 12, Inclusion.UNKNOWN, id="body-only"),
    ],
)
def test_only_a_trailing_suffix_with_the_exact_number_counts(tmp_path, subject, body, number, expected):
    root = _repo(tmp_path)
    _commit(root, subject, body)

    assert _probe(root, number).inclusion is expected


def test_a_subject_in_a_non_utf8_log_encoding_still_matches(tmp_path):
    root = _repo(tmp_path)
    git("config", "i18n.logOutputEncoding", "ISO-8859-1", cwd=root)
    _commit(root, "Tune the naïve kernel (#12)")

    assert _probe(root, 12).inclusion is Inclusion.INCLUDED


def test_a_merge_commit_missing_locally_is_unknown(tmp_path):
    root = _repo(tmp_path)

    verdict = _probe(root, 42, "0123456789abcdef0123456789abcdef01234567")

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert verdict.method is InclusionMethod.NONE
    assert "is not in the tree" in verdict.evidence


def test_without_a_merge_sha_a_subject_miss_is_unknown_never_not_included(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "Unrelated work (#41)")

    verdict = _probe(root, 42)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert "no merge commit sha" in verdict.evidence


def test_a_shallow_clone_cannot_prove_absence(tmp_path):
    """Ancestry stops at the shallow boundary, so a merged PR reads as not-ancestor there."""
    upstream = _repo(tmp_path)
    merge = _commit(upstream, "Fuse the epilogue")
    git("branch", "pr-head", merge, cwd=upstream)
    _commit(upstream, "Later work")
    clone = tmp_path / "clone"
    git("clone", "--quiet", "--depth", "1", "--no-single-branch", upstream.as_uri(), str(clone))

    verdict = _probe(clone, 42, merge)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert "shallow" in verdict.evidence


@pytest.mark.parametrize("commit", ["no-such-branch", "--all"])
def test_a_revision_that_names_no_commit_is_unknown(tmp_path, commit):
    root = _repo(tmp_path)

    verdict = _probe(root, 42, _head(root), commit=commit)

    assert verdict.inclusion is Inclusion.UNKNOWN
    assert verdict.method is InclusionMethod.NONE


def test_a_directory_outside_git_is_unknown(tmp_path):
    assert _probe(tmp_path, 42).inclusion is Inclusion.UNKNOWN


def test_a_partial_clone_answers_from_the_objects_it_has_and_fetches_nothing(tmp_path):
    upstream = _repo(tmp_path)
    # Serve any sha, as a hosting service does, so a lazy fetch of the side commit would succeed.
    for key in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        git("config", key, "true", cwd=upstream)
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
    assert "is not in the tree" in missing.evidence
    assert present.inclusion is Inclusion.INCLUDED
    assert present.method is InclusionMethod.MERGE_ANCESTRY
    assert sorted(objects.rglob("*")) == stored


def _hangs(*args, **kwargs):
    raise subprocess.TimeoutExpired(list(args), kwargs["timeout"])


def _fails(*args, **kwargs):
    return subprocess.CompletedProcess(list(args), 128, b"", b"fatal: simulated failure")


@pytest.mark.parametrize("failure", [_hangs, _fails], ids=["hangs", "fails"])
@pytest.mark.parametrize("command", ["merge-base", "--is-shallow-repository", "log"])
def test_a_git_call_that_fails_or_hangs_is_unknown_not_absence(tmp_path, monkeypatch, command, failure):
    """Git proves absence here while every call answers, so only the failed call can make the verdict unknown."""
    root = _repo(tmp_path)
    merge = _fork_before(root, "Fuse the epilogue")
    assert _probe(root, 42, merge).inclusion is Inclusion.NOT_INCLUDED

    monkeypatch.setattr(
        inclusion, "git_bytes", lambda *args, **kwargs: (failure if command in args else git_bytes)(*args, **kwargs)
    )

    assert _probe(root, 42, merge).inclusion is Inclusion.UNKNOWN


@pytest.fixture
def cherry_picked(tmp_path) -> tuple[Path, str]:
    root = _repo(tmp_path)
    return root, _fork_with_cherry_pick(root, "Fuse the epilogue (#42)")


@pytest.mark.parametrize(
    ("number", "with_merge_sha", "status", "expected"),
    [
        pytest.param(42, True, 0, Inclusion.INCLUDED, id="included"),
        pytest.param(41, True, 4, Inclusion.NOT_INCLUDED, id="not-included"),
        pytest.param(41, False, 3, Inclusion.UNKNOWN, id="unknown"),
    ],
)
def test_cli_prints_the_verdict_and_encodes_it_in_the_exit_status(
    cherry_picked, capsys, number, with_merge_sha, status, expected
):
    root, upstream = cherry_picked
    argv = ["--tree", str(root), "--commit", "HEAD", "--pr", str(number)]
    if with_merge_sha:
        argv += ["--merge-commit-sha", upstream]

    assert main(argv) == status
    record = json.loads(capsys.readouterr().out)
    assert set(record) == {"inclusion", "method", "evidence"}
    assert record["inclusion"] == expected.value


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--commit", "HEAD", "--pr", "1"], id="no-tree"),
        pytest.param(["--tree", "{tree}", "--commit", "HEAD"], id="no-pr"),
        pytest.param(["--tree", "{tree}", "--commit", "HEAD", "--pr", "twelve"], id="pr-not-a-number"),
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
    assert json.loads(completed.stdout)["method"] == InclusionMethod.SUBJECT_SUFFIX.value


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
    assert result["method"] == InclusionMethod.SUBJECT_SUFFIX.value
    assert _head(root) in result["evidence"]


def test_mcp_tool_uses_the_merge_sha_and_commit_it_is_given(cherry_picked):
    root, upstream = cherry_picked

    result = _tool({"number": 42, "tree": str(root), "commit": upstream, "merge_commit_sha": upstream})

    assert result["commit"] == upstream
    assert result["inclusion"] == Inclusion.INCLUDED.value
    assert result["method"] == InclusionMethod.MERGE_ANCESTRY.value


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
