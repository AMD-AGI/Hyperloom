# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Already-applied patches are a satisfied no-op, and a set of nothing else is ``already_present``, not benched."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hyperloom.agents.framework.kb import read_pr_ledger
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from hyperloom.orchestrator.actions.executors import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors import _patch_git as patch_git
from hyperloom.orchestrator.actions.executors import integrate_patch as ip
from hyperloom.orchestrator.actions.executors._apply_feedback import PatchApplyResult, PresenceCheck
from hyperloom.orchestrator.actions.executors._nogit_patch import (
    _apply_patch_no_git,
    _reverse_applies_cleanly,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.tests._helpers import git_commit_all


_SUBSET = """\
--- a/mod/alpha.py
+++ b/mod/alpha.py
@@ -1,3 +1,3 @@
 header
-old_alpha
+new_alpha
 footer
"""

# Superset: the subset's hunk plus a second file the subset never touches.
_SUPERSET = (
    _SUBSET
    + """\
--- a/mod/beta.py
+++ b/mod/beta.py
@@ -1,3 +1,3 @@
 header
-old_beta
+new_beta
 footer
"""
)

_UNRELATED = """\
--- a/mod/alpha.py
+++ b/mod/alpha.py
@@ -1,3 +1,3 @@
 header
-nothing_matches_this
+replacement
 footer
"""

# Deletes alpha.py; undoing it re-creates the file, which needs no matching content in the tree.
_DELETE_ALPHA = """\
diff --git a/mod/alpha.py b/mod/alpha.py
deleted file mode 100644
--- a/mod/alpha.py
+++ /dev/null
@@ -1,3 +0,0 @@
-header
-old_alpha
-footer
"""

_PR_URL = "https://github.com/example-org/example-framework/pull/1"


def _tree(root: Path) -> None:
    """Materialize the two-file source tree both patches target."""
    (root / "mod").mkdir(parents=True)
    (root / "mod" / "alpha.py").write_text("header\nold_alpha\nfooter\n", encoding="utf-8")
    (root / "mod" / "beta.py").write_text("header\nold_beta\nfooter\n", encoding="utf-8")


def _patch(root: Path, name: str, body: str) -> Path:
    """Write ``body`` as a patch file under ``root`` and return its path."""
    p = root / name
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def workspace(tmp_path):
    """A source tree, a patch dir and a backup dir, all under tmp_path."""
    src = tmp_path / "src"
    _tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    return src, patches, tmp_path / "backups"


def test_nogit_superset_then_subset_is_a_noop(workspace):
    """Subset after superset succeeds with no backups; both edits survive."""
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    first = _apply_patch_no_git(src, sup, backups, seq_offset=0)
    assert first.ok, first.error
    assert first.backups, "superset should have backed up the files it changed"
    assert first.present_by is None

    second = _apply_patch_no_git(src, sub, backups, seq_offset=len(first.backups))
    assert second.ok, second.error
    assert second.feedback is None
    assert second.present_by is PresenceCheck.PATCH_REVERSE_DRY_RUN
    # The no-op owns no backups — the patch that made the edits owns them, so a revert restores the tree exactly once.
    assert second.backups == []

    assert "new_alpha" in (src / "mod" / "alpha.py").read_text(encoding="utf-8")
    assert "new_beta" in (src / "mod" / "beta.py").read_text(encoding="utf-8")


def test_nogit_subset_then_superset_fails_closed(workspace):
    """The reverse order is only *partly* satisfied, so it must fail, not no-op."""
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    first = _apply_patch_no_git(src, sub, backups, seq_offset=0)
    assert first.ok
    second = _apply_patch_no_git(src, sup, backups, seq_offset=len(first.backups))
    assert not second.ok
    assert "failed at all strip levels" in second.error
    assert second.feedback is not None, "a real failure must carry reauthor feedback"
    assert second.present_by is None
    # Nothing was half-applied: beta is untouched.
    assert "old_beta" in (src / "mod" / "beta.py").read_text(encoding="utf-8")


def test_nogit_partial_overlap_still_fails(workspace):
    """A patch only *partly* present is a real failure, not a no-op."""
    src, patches, backups = workspace
    sup = _patch(patches, "superset.patch", _SUPERSET)
    assert _apply_patch_no_git(src, sup, backups, seq_offset=0).ok
    # Undo only beta, leaving alpha applied -> superset is half-present.
    (src / "mod" / "beta.py").write_text("header\nold_beta\nfooter\n", encoding="utf-8")

    assert not _reverse_applies_cleanly(src, sup)


def test_nogit_unrelated_patch_still_fails(workspace):
    """A patch that genuinely does not apply keeps failing with feedback."""
    src, patches, backups = workspace
    bad = _patch(patches, "bad.patch", _UNRELATED)

    result = _apply_patch_no_git(src, bad, backups, seq_offset=0)
    assert not result.ok
    assert "failed at all strip levels" in result.error
    assert result.feedback is not None
    assert result.backups == []


def _git_tree(root: Path) -> None:
    """Initialise ``root`` as a committed git work tree."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(root), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)


def _git(root: Path, *args: str) -> str:
    """Run git in ``root`` and return its stdout."""
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def test_git_channel_superset_then_subset_is_a_noop(tmp_path):
    """The git channel resolves the same ambiguity via ``git apply -R --check``."""
    src = tmp_path / "src"
    _tree(src)
    _git_tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    sup = _patch(patches, "superset.patch", _SUPERSET)
    sub = _patch(patches, "subset.patch", _SUBSET)

    first = patch_git._git_apply_collect_feedback(src, sup)
    assert first.ok, first.error
    assert first.present_by is None
    second = patch_git._git_apply_collect_feedback(src, sub)
    assert second.ok, second.error
    assert second.feedback is None
    assert second.present_by is PresenceCheck.GIT_REVERSE_APPLY
    assert "new_alpha" in (src / "mod" / "alpha.py").read_text(encoding="utf-8")


def test_git_channel_unrelated_patch_still_fails(tmp_path):
    """A non-applying patch still fails on the git channel, with feedback."""
    src = tmp_path / "src"
    _tree(src)
    _git_tree(src)
    patches = tmp_path / "patches"
    patches.mkdir()
    bad = _patch(patches, "bad.patch", _UNRELATED)

    result = patch_git._git_apply_collect_feedback(src, bad)
    assert not result.ok
    assert result.feedback is not None
    assert result.present_by is None


_CHANNEL_CHECKS = [(True, PresenceCheck.GIT_REVERSE_APPLY), (False, PresenceCheck.PATCH_REVERSE_DRY_RUN)]


def _apply_one(git_tree: bool, root: Path, patch: Path, backups: Path) -> PatchApplyResult:
    """Apply one patch through the channel ``git_tree`` selects."""
    if git_tree:
        return patch_git._git_apply_collect_feedback(root, patch)
    return _apply_patch_no_git(root, patch, backups)


def _channel_tree(tmp_path: Path, git_tree: bool) -> Path:
    """The two-file source tree, as a committed git checkout when ``git_tree``."""
    src = tmp_path / "src"
    _tree(src)
    if git_tree:
        _git_tree(src)
    return src


def _settle(root: Path, git_tree: bool, message: str) -> None:
    """Commit the tree's current state when it is a git checkout."""
    if git_tree:
        git_commit_all(root, message)


@pytest.mark.parametrize("git_tree", [True, False])
def test_deletion_of_a_drifted_file_is_a_failure_not_present(tmp_path, git_tree):
    """The file the patch deletes is still there with other content, so undoing the deletion cannot prove presence."""
    src = _channel_tree(tmp_path, git_tree)
    (src / "mod" / "alpha.py").write_text("header\ndrifted\nfooter\n", encoding="utf-8")
    _settle(src, git_tree, "drift")
    patch = _patch(tmp_path, "delete.patch", _DELETE_ALPHA)

    result = _apply_one(git_tree, src, patch, tmp_path / "bak")

    assert not result.ok
    assert result.present_by is None
    assert result.feedback is not None
    assert (src / "mod" / "alpha.py").read_text(encoding="utf-8") == "header\ndrifted\nfooter\n"


@pytest.mark.parametrize(("git_tree", "check"), _CHANNEL_CHECKS)
def test_deletion_already_carried_out_is_present(tmp_path, git_tree, check):
    """With the file gone there is nothing left to anchor on, and the deletion is a satisfied no-op."""
    src = _channel_tree(tmp_path, git_tree)
    (src / "mod" / "alpha.py").unlink()
    _settle(src, git_tree, "deleted upstream")
    patch = _patch(tmp_path, "delete.patch", _DELETE_ALPHA)

    result = _apply_one(git_tree, src, patch, tmp_path / "bak")

    assert result.ok, result.error
    assert result.present_by is check


# Shapes ``git apply`` and ``patch`` accept whose leading block names no file, so the manifest cannot be parsed.
_FORMAT_PATCH = (
    "From 0 Mon Sep 17 00:00:00 2001\nSubject: [PATCH] alpha\n\n---\ndiff --git a/mod/alpha.py b/mod/alpha.py\n"
)
_DIFF_RUN = "diff -ruN a/mod/alpha.py b/mod/alpha.py\n"


@pytest.mark.parametrize("preamble", [_FORMAT_PATCH, _DIFF_RUN], ids=["format_patch", "diff_ruN"])
@pytest.mark.parametrize(("git_tree", "check"), _CHANNEL_CHECKS)
def test_patch_with_unparseable_manifest_is_probed_at_every_level(tmp_path, git_tree, check, preamble):
    """A patch the manifest parser rejects is still reported present when the tree holds it."""
    src = _channel_tree(tmp_path, git_tree)
    (src / "mod" / "alpha.py").write_text("header\nnew_alpha\nfooter\n", encoding="utf-8")
    _settle(src, git_tree, "already merged")
    patch = _patch(tmp_path, "mail.patch", preamble + _SUBSET)

    result = _apply_one(git_tree, src, patch, tmp_path / "bak")

    assert result.ok, result.error
    assert result.present_by is check


def _make_ctx(task_id: str, params: dict, state: SharedState) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={"shared_state": state})


def _approved_state() -> SharedState:
    """Session state carrying the Critic's approval of the test PR."""
    state = SharedState()
    state.record_specialist_patch_verdict(_PR_URL, "approve")
    return state


@pytest.fixture
def executor_env(tmp_path, monkeypatch):
    """A session dir, a KB root, and an executor whose benchmark fails the test if it is reached."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    def no_bench(*_args, **_kwargs):
        pytest.fail("a candidate the tree already holds must not be benchmarked")

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    monkeypatch.setattr(ip.IntegratePatchExecutor, "_bench_patch", no_bench)
    monkeypatch.setattr(integrate_bench, "run_grid", no_bench)
    kb_root = tmp_path / "kb"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FA_KB_PATH", str(kb_root))
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    return ip.IntegratePatchExecutor(session_dir=session_dir), kb_root


def _repo_holding_the_superset(tmp_path: Path, git_tree: bool = True) -> Path:
    """A framework tree already carrying every hunk of ``_SUPERSET``: a git checkout at that HEAD when ``git_tree``."""
    repo = _channel_tree(tmp_path / "framework", git_tree)
    (repo / "mod" / "alpha.py").write_text("header\nnew_alpha\nfooter\n", encoding="utf-8")
    (repo / "mod" / "beta.py").write_text("header\nnew_beta\nfooter\n", encoding="utf-8")
    _settle(repo, git_tree, "upstream already merged it")
    return repo


def _specialist_round(session_dir: Path, repo: Path) -> tuple[dict, list[Path]]:
    """A specialist-authored round delivering the superset and its subset, with a PR identity."""
    sid = "spec-present"
    workspace = runs_dir(session_dir, "specialist", sid)
    patches = [
        _patch(_mkdir(workspace / "worktree"), "superset.patch", _SUPERSET),
        _patch(workspace / "worktree", "subset.patch", _SUBSET),
    ]
    done = {
        "patches_written": [p.name for p in patches],
        "proposal_set": [{"provenance": "specialist:serving:framework", "fa_pr_url": _PR_URL}],
    }
    (workspace / "specialist_done.json").write_text(json.dumps(done), encoding="utf-8")
    return {"specialist_task_id": sid, "framework_source_root": str(repo)}, patches


def _upstream_round(session_dir: Path, task_id: str, repo: Path) -> tuple[dict, list[Path]]:
    """An upstream-PR round landing the superset and its subset from the task's own scratch."""
    scratch = _mkdir(runs_dir(session_dir, "integrate_patch", task_id))
    patches = [_patch(scratch, "superset.patch", _SUPERSET), _patch(scratch, "subset.patch", _SUBSET)]
    params = {
        "patch_source": "upstream_pr",
        "candidate": {"pr_url": _PR_URL, "repo": "example-org/example-framework"},
        "framework_agent_candidate_id": _PR_URL,
        "framework_source_root": str(repo),
        "patches": [str(p) for p in patches],
    }
    return params, patches


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lane", "git_tree", "check"),
    [
        ("upstream_pr", True, PresenceCheck.GIT_REVERSE_APPLY),
        ("specialist_authored", True, PresenceCheck.GIT_REVERSE_APPLY),
        ("upstream_pr", False, PresenceCheck.PATCH_REVERSE_DRY_RUN),
    ],
)
async def test_executor_ends_an_all_noop_patch_set_as_already_present(tmp_path, executor_env, lane, git_tree, check):
    """Every patch already in the tree: no benchmark, an ``already_present`` result and one KB row saying so."""
    executor, kb_root = executor_env
    repo = _repo_holding_the_superset(tmp_path, git_tree)
    before = {f: (repo / "mod" / f).read_text(encoding="utf-8") for f in ("alpha.py", "beta.py")}
    head = _git(repo, "rev-parse", "HEAD") if git_tree else ""
    if lane == "upstream_pr":
        params, patches = _upstream_round(executor.session_dir, "t-present", repo)
    else:
        params, patches = _specialist_round(executor.session_dir, repo)
    state = SharedState()
    state.record_specialist_patch_verdict(_PR_URL if lane == "upstream_pr" else params["specialist_task_id"], "approve")

    result = await executor(_make_ctx("t-present", params, state))

    assert result["status"] == "already_present"
    assert result["already_present"] == [{"patch": str(p), "check": check.value} for p in patches]
    assert result["patches_applied"] == []
    assert result["patches_reverted"] == []
    rows = read_pr_ledger(kb_root=kb_root)
    assert [(row["outcome"], row["pr_url"]) for row in rows] == [("already_present", _PR_URL)]
    assert {f: (repo / "mod" / f).read_text(encoding="utf-8") for f in before} == before
    if git_tree:
        assert _git(repo, "rev-parse", "HEAD") == head
        assert _git(repo, "status", "--porcelain") == ""


def _record_bench(monkeypatch) -> list[str]:
    """Replace the benchmark with one that records who reached it and then fails, ending the round."""
    reached: list[str] = []

    def bench(_executor, bench_leg, **_kwargs):
        reached.append(bench_leg.specialist_task_id)
        raise OSError("benchmark stand-in")

    monkeypatch.setattr(ip.IntegratePatchExecutor, "_bench_patch", bench)
    return reached


def _repo_with_merged_pr_head(tmp_path: Path, *, revert: bool) -> tuple[Path, str]:
    """A checkout whose ``main`` merged ``pr-head`` (the superset) with a merge commit, optionally reverted since."""
    repo = tmp_path / "framework"
    _tree(repo)
    _git_tree(repo)
    _git(repo, "checkout", "-qb", "pr-head")
    (repo / "mod" / "alpha.py").write_text("header\nnew_alpha\nfooter\n", encoding="utf-8")
    (repo / "mod" / "beta.py").write_text("header\nnew_beta\nfooter\n", encoding="utf-8")
    git_commit_all(repo, "the pr")
    head = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "-q", "-")
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    git_commit_all(repo, "unrelated")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge the pr", "pr-head")
    if revert:
        _git(repo, "revert", "--no-edit", "-m", "1", "HEAD")
    _git(repo, "remote", "add", "origin", str(repo))
    return repo, head


def _pr_head_round(repo: Path, head: str, **candidate_extra: object) -> dict:
    """An upstream-PR round naming the PR by its head ref, as discovery does."""
    return {
        "patch_source": "upstream_pr",
        "candidate": {
            "pr_url": _PR_URL,
            "repo": "example-org/example-framework",
            "ref": "pr-head",
            "head_sha": head,
            **candidate_extra,
        },
        "framework_agent_candidate_id": _PR_URL,
        "framework_source_root": str(repo),
    }


def _serve_superset(monkeypatch) -> None:
    """Make every ``diff_url`` fetch return the PR's diff."""
    from hyperloom.orchestrator.actions.executors import _patch_source_pr

    def fetch(_url, dest, *, timeout_sec):
        dest.write_text(_SUPERSET, encoding="utf-8")
        return True, ""

    monkeypatch.setattr(_patch_source_pr, "_fetch_diff_to_path", fetch)


@pytest.mark.asyncio
async def test_executor_ends_a_merged_pr_fetched_by_diff_url_as_already_present(tmp_path, executor_env, monkeypatch):
    """An empty head diff defers to the diff_url, and the apply's reverse check finds the PR in the tree."""
    _serve_superset(monkeypatch)
    executor, kb_root = executor_env
    repo, head = _repo_with_merged_pr_head(tmp_path, revert=False)
    params = _pr_head_round(repo, head, diff_url="https://example.invalid/pr.diff", prefer_checkout=True)

    result = await executor(_make_ctx("t-merged-url", params, _approved_state()))

    assert result["status"] == "already_present"
    assert [row["check"] for row in result["already_present"]] == ["git_reverse_apply_check"]
    assert [row["outcome"] for row in read_pr_ledger(kb_root=kb_root)] == ["already_present"]


@pytest.mark.asyncio
async def test_executor_benchmarks_a_reverted_pr_head(tmp_path, executor_env, monkeypatch):
    """A merged-then-reverted PR is still in the history but not in the tree, so it is applied and measured."""
    _serve_superset(monkeypatch)
    reached = _record_bench(monkeypatch)
    executor, kb_root = executor_env
    repo, head = _repo_with_merged_pr_head(tmp_path, revert=True)
    params = _pr_head_round(repo, head, diff_url="https://example.invalid/pr.diff", prefer_checkout=True)

    result = await executor(_make_ctx("t-reverted", params, _approved_state()))

    assert reached == ["t-reverted"]
    assert result["status"] != "already_present"
    assert "already_present" not in [row["outcome"] for row in read_pr_ledger(kb_root=kb_root)]


@pytest.mark.asyncio
async def test_executor_benchmarks_an_all_noop_set_that_changes_server_args(tmp_path, executor_env, monkeypatch):
    """Patches already in the tree do not end a round that also changes the server args: those still need measuring."""
    reached = _record_bench(monkeypatch)
    executor, kb_root = executor_env
    repo = _repo_holding_the_superset(tmp_path)
    params, _patches = _upstream_round(executor.session_dir, "t-args", repo)
    params["extra_server_args"] = "--max-num-seqs 64"

    result = await executor(_make_ctx("t-args", params, _approved_state()))

    assert reached == ["t-args"]
    assert result["status"] != "already_present"
    assert "already_present" not in [row["outcome"] for row in read_pr_ledger(kb_root=kb_root)]


@pytest.mark.asyncio
async def test_executor_lands_a_set_whose_noop_rides_with_a_real_change(tmp_path, executor_env):
    """Only a set that changes nothing ends early; a no-op beside a real change lands with it."""
    executor, kb_root = executor_env
    repo = tmp_path / "framework"
    _tree(repo)
    _git_tree(repo)
    params, patches = _upstream_round(executor.session_dir, "t-mixed", repo)
    params["apply_only"] = True

    result = await executor(_make_ctx("t-mixed", params, _approved_state()))

    assert result["status"] == "applied_no_bench"
    assert result["patches_applied"] == [str(p) for p in patches]
    assert read_pr_ledger(kb_root=kb_root) == []
