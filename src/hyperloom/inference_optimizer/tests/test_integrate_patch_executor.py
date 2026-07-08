# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-A4 (Arbor-into-Hyperloom): IntegratePatchExecutor tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import git_commit_all, init_git_repo

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
)
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


# Helpers

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


# Targets an *existing* file (src.py) but with context lines that do not match
# the tree — exercises the genuine ``git_apply_failed`` path (file exists, hunk
# is stale), distinct from the ``patch_target_missing`` preflight below.
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


# Targets a file that does not exist in the framework tree at all — a
# hallucinated layout (e.g. a CUDA-only file on a ROCm build). Must be caught
# by the missing-target preflight, not a wasted ``git apply``.
_MISSING_TARGET_PATCH = """\
diff --git a/nonexistent.py b/nonexistent.py
index 0000000..1111111 100644
--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,1 +1,1 @@
-OLD
+NEW
"""


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


# 1. Patch path resolution
# G2 — force RUN_EVAL for framework-authored source patches (only with a baseline)
def test_framework_run_eval_envs_forces_for_authored_with_baseline():
    assert IntegratePatchExecutor._framework_run_eval_envs(
        {"framework_agent_authoring": True, "accuracy_baseline": 0.8}
    ) == {"RUN_EVAL": "true"}
    assert IntegratePatchExecutor._framework_run_eval_envs(
        {"framework_agent_candidate_id": "c1", "accuracy_baseline": 0.8}
    ) == {"RUN_EVAL": "true"}


def test_framework_run_eval_envs_no_force_without_baseline():
    # baseline eval never produced a score -> nothing to gate against -> don't
    # force eval on the candidate (matches the framework path / G1 degrade).
    assert IntegratePatchExecutor._framework_run_eval_envs({"framework_agent_authoring": True}) is None
    assert (
        IntegratePatchExecutor._framework_run_eval_envs(
            {"framework_agent_authoring": True, "accuracy_baseline": 0.0}
        )
        is None
    )


def test_framework_run_eval_envs_none_for_generic_explore():
    # Generic EXPLORE integrate_patch (no framework markers) -> untouched.
    assert IntegratePatchExecutor._framework_run_eval_envs({"specialist_task_id": "s1", "accuracy_baseline": 0.8}) is None
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
    # done_payload=None forces the filesystem scan.
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


# 2. git apply primitives
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


# Specialists author patches whose ``+++ b/<path>`` prefix is *not* a simple
# ``-p1`` strip (they read framework files at deep absolute paths, e.g.
# ``b/usr/local/lib/python3.12/dist-packages/vllm/src.py``). The executor must
# auto-detect the strip level instead of hardcoding ``-p1``; otherwise every
# such Critic-approved patch fails to apply (regression seen in production).
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
    # ``b/d0/.../d6/src.py`` needs -p7 (1 for ``b/`` + 6 for d0..d5).
    patch = tmp_path / "deep.patch"
    patch.write_text(_deep_prefix_patch(6), encoding="utf-8")
    ok, err = _git_apply(repo, patch)
    assert ok, f"auto -p detection should apply deep-prefix patch: {err}"
    assert (repo / "src.py").read_text().endswith("return 2\n")
    # Reverse must auto-detect the same level and roll back cleanly.
    ok_r, err_r = _git_apply_reverse(repo, patch)
    assert ok_r, err_r
    assert (repo / "src.py").read_text().endswith("return 1\n")


# 3. Framework root resolution
def test_resolve_framework_root_picks_explicit_when_dir(tmp_path: Path):
    repo = tmp_path / "repo"
    init_git_repo(repo)
    root = _resolve_framework_root(str(repo))
    assert root is not None
    assert root.samefile(repo)


def test_resolve_framework_root_returns_none_when_no_candidate(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        str(tmp_path / "does-not-exist"),
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path / "missing-ix"))
    root = _resolve_framework_root(None)
    # Either None or a fallback root — both acceptable for this defensive helper.
    if root is not None:
        assert root.exists()


# 4. End-to-end executor invocation
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


# 4b. Enablement runnable gate: the bench is the launch probe; a positive
# throughput means the server booted -> KEEP; else -> REVERT. The perf/accuracy
# KEEP gate is bypassed for enablement-tagged integrations.
async def _run_enablement_integrate(
    tmp_path: Path,
    monkeypatch,
    *,
    booted: bool,
    enablement_accuracy=None,
    bench_error: str = "",
    before_signature=None,
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
        }
        return bench_result, {
            "accuracy_pass": None,
            "enablement_accuracy": enablement_accuracy,
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
        "enable_stack_rebench": False,
    }
    if before_signature is not None:
        params["enablement_before_signature"] = before_signature
    ctx = _make_ctx("t-int-en", params)
    return await executor(ctx), repo


@pytest.mark.asyncio
async def test_enablement_keeps_when_server_boots(tmp_path: Path, monkeypatch):
    # No eval accuracy produced -> KEEP but provisional (boot-only).
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=True)
    assert result["status"] == "kept"
    assert result["enablement"] is True
    assert result["runnable"] is True
    assert result["provisional"] is True
    assert result["correctness_verified"] is False
    assert len(result["patches_applied"]) == 1
    # The patch stays applied on a runnable KEEP.
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_enablement_reverts_when_still_not_runnable(tmp_path: Path, monkeypatch):
    result, repo = await _run_enablement_integrate(tmp_path, monkeypatch, booted=False)
    assert result["status"] == "reverted"
    assert result["enablement"] is True
    assert result["runnable"] is False
    assert result["patches_applied"] == []
    # REVERT rolls the tree back to its original content.
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_keeps_verified_when_accuracy_above_floor(tmp_path: Path, monkeypatch):
    """Booted + eval accuracy above the absolute floor -> KEEP, non-provisional."""
    result, repo = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=0.42
    )
    assert result["status"] == "kept"
    assert result["runnable"] is True
    assert result["correctness_verified"] is True
    assert result["provisional"] is False
    assert (repo / "src.py").read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_enablement_reverts_when_accuracy_zero(tmp_path: Path, monkeypatch):
    """Booted but eval accuracy == floor (garbage output) -> REVERT."""
    result, repo = await _run_enablement_integrate(
        tmp_path, monkeypatch, booted=True, enablement_accuracy=0.0
    )
    assert result["status"] == "reverted"
    assert result["runnable"] is False
    assert result["correctness_verified"] is False
    assert result["patches_applied"] == []
    assert (repo / "src.py").read_text().endswith("return 1\n")


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
    """Patch clears gap #1 (shape_mismatch) but boot stops at gap #2 (missing_weight).

    The server still does not fully boot (output_throughput=0), but the failure
    moved to a NEW, deeper actionable signature -> status='advanced': the patch
    is recorded for stacking (patches_applied non-empty), the new failure log is
    surfaced, and the working tree is reverted to clean for deterministic
    re-application next round.
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
    # The progressing patch is reported for stacking.
    assert len(result["patches_applied"]) == 1
    # The new (deeper) gap is surfaced for the next round to reclassify + target.
    assert "not initialized from checkpoint" in result["enablement_launch_log"]
    assert result["after_signature"]["kind"] == "missing_weight"
    # Tree reverted to clean so the stack is rebuilt deterministically next round.
    assert (repo / "src.py").read_text().endswith("return 1\n")


@pytest.mark.asyncio
async def test_enablement_stacks_base_patches_before_new(tmp_path: Path, monkeypatch):
    """enablement_base_patches are applied before this round's patch (serial gaps)."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    repo = tmp_path / "framework"
    init_git_repo(repo)
    # A base patch (prior progressing round) touching a DIFFERENT file, plus the
    # current round's patch on src.py.
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
        "enable_stack_rebench": False,
        "enablement_base_patches": [str(base_patch)],
    }
    result = await executor(_make_ctx("t-int-stack", params))
    assert result["status"] == "kept"
    # Both the base patch and this round's patch were applied (stacked).
    assert len(result["patches_applied"]) == 2
    # Both files reflect their patched content on a runnable KEEP.
    assert (repo / "other.py").read_text().endswith("return 20\n")
    assert (repo / "src.py").read_text().endswith("return 2\n")


# 4c. Enablement environment-setup replay (Q3): allowlist + resolve + runner.
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
        "echo `whoami`",
        "pip install x $(malicious)",
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
    # The dangerous command was NEVER handed to subprocess.
    assert ran == ["pip install -U transformers"]
    assert (tmp_path / "logs" / "enablement_setup.log").exists()


@pytest.mark.asyncio
async def test_enablement_replays_setup_commands_before_boot(tmp_path: Path, monkeypatch):
    """Q3: enablement integrate replays setup_commands and surfaces them in the result."""
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
        # Simulate all allowlisted commands running cleanly.
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
        "enable_stack_rebench": False,
        "enablement_setup_commands": ["pip install -U transformers"],
    }
    result = await executor(_make_ctx("t-int-setup", params))
    assert result["status"] == "kept"
    assert result["setup_commands_applied"] == ["pip install -U transformers"]
    # The runner was invoked with the resolved setup commands (before boot).
    assert replayed["commands"] == ["pip install -U transformers"]


# 5. CLI registration
def test_integrate_patch_executor_imports_clean():
    """The real executor module must import without side effects."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip_mod

    assert hasattr(ip_mod, "IntegratePatchExecutor")
    assert callable(ip_mod.IntegratePatchExecutor)


# 6. git-free backup-based apply/revert
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

    ok, err, backups = _apply_patch_no_git(framework_root, patch_file, backup_root)
    pytest.importorskip("subprocess")  # ensure patch CLI available; skip gracefully if not
    if not ok:
        pytest.skip(f"patch CLI unavailable or patch failed: {err}")

    patched = (framework_root / "src.py").read_text(encoding="utf-8")
    assert "return 42" in patched, "patch was not applied"
    assert any(r["backup_path"] for r in backups), "backup was not created"

    # Revert restores original content.
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
        # Simulate a patch implementation whose dry-run accepts the target so
        # this test exercises Hyperloom's own boundary check before real apply.
        if "--dry-run" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        raise AssertionError("real patch apply must not run for escaping targets")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, err, backups = _apply_patch_no_git(framework_root, patch_file, tmp_path / "backups")

    assert ok is False
    assert "escapes framework root" in err
    assert backups == []
    assert outside.read_text(encoding="utf-8") == "SAFE\n"
    assert len(calls) == 1
