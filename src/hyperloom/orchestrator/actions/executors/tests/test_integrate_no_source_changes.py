# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``--no-source-changes``: integrate_patch keeps only server arguments and environment variables."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.integrate import _integrate_apply as integrate_apply
from hyperloom.orchestrator.actions.executors.integrate import _integrate_bench as integrate_bench
from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors import _setup_replay
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate._integrate_resolve import source_change_skip
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.enablement.runtime import adapters
from hyperloom.orchestrator.enablement.runtime.stack_actions import EnablementStackAction
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.orchestrator.tests._helpers import init_git_repo, patch_integrate_patch_roots

_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""

_SWITCH_GATED_PATCH = """\
diff --git a/src.py b/src.py
index 0000000..1111111 100644
--- a/src.py
+++ b/src.py
@@ -1,2 +1,5 @@
+import os
 def f():
-    return 1
+    if os.environ.get("HL_TEST_FUSE") == "1":
+        return 2
+    return 1
"""

_SPEC = "spec-1"


def _forbidden(*_args, **_kwargs):
    pytest.fail("a skipped round must not provision, apply, replay or measure anything")


@pytest.fixture(autouse=True)
def _nothing_runs(monkeypatch, tmp_path):
    patch_integrate_patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)
    monkeypatch.setattr(integrate_bench, "run_grid", _forbidden)
    monkeypatch.setattr(_setup_replay, "run_setup_commands", _forbidden)
    monkeypatch.setattr(
        adapters,
        "get_adapter",
        lambda _framework: SimpleNamespace(
            build_stack_action=lambda _request: EnablementStackAction(
                framework="vllm", gap_id="gap.test", capability="missing_model_arch"
            ),
            provision=_forbidden,
            probe=_forbidden,
            source_import_root=lambda root: root,
        ),
    )


@pytest.fixture
def framework_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "fw"
    init_git_repo(tree)
    return tree


def _state(*, no_source_changes: bool = True, **fields: Any) -> SharedState:
    state = SharedState(no_source_changes=no_source_changes, framework="vllm", **fields)
    state.record_specialist_patch_verdict(_SPEC, "approve")
    return state


def _specialist(session_dir: Path, *, patches: list[str] = (), **done: Any) -> Path:
    """A finished specialist's workspace whose done payload hands over ``patches`` and ``done``."""
    workspace = session_dir / "runs" / "specialist" / _SPEC
    (workspace / "worktree" / "patches").mkdir(parents=True, exist_ok=True)
    written = []
    for index, text in enumerate(patches, start=1):
        path = workspace / "worktree" / "patches" / f"{index:03d}.patch"
        path.write_text(text, encoding="utf-8")
        written.append(f"patches/{path.name}")
    payload = {"proposal_set": [], "patches_written": written, "summary": "test", **done}
    (workspace / "specialist_done.json").write_text(json.dumps(payload), encoding="utf-8")
    return workspace


async def _integrate(session_dir: Path, state: SharedState, params: dict[str, Any]) -> dict[str, Any]:
    task = Task(
        task_id="int-1",
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key="int-1",
        requires_lanes=tuple(),
    )
    executor = IntegratePatchExecutor(session_dir=session_dir)
    return await executor(RunnerContext(task=task, lease=None, extra={"shared_state": state}))


def _assert_skipped(result: dict[str, Any], state: SharedState, framework_tree: Path, carried: str) -> None:
    assert result["status"] == "skipped"
    assert result["skipped_reason"] == "no_source_changes"
    assert any(carried in item for item in result["source_changes"])
    assert result["patches_applied"] == []
    assert state.pending_integrate == {}
    assert (framework_tree / "src.py").read_text(encoding="utf-8").endswith("return 1\n")


async def test_a_patch_is_skipped_and_the_tree_is_untouched(tmp_path, framework_tree):
    session = tmp_path / "session"
    _specialist(session, patches=[_PATCH])
    state = _state()
    result = await _integrate(session, state, {"specialist_task_id": _SPEC})
    _assert_skipped(result, state, framework_tree, "patch")
    assert result["specialist_task_id"] == _SPEC


async def test_a_rewrite_switch_is_skipped_with_its_patch(tmp_path, framework_tree):
    session = tmp_path / "session"
    _specialist(
        session,
        patches=[_SWITCH_GATED_PATCH],
        framework_switches=[{"switch": "HL_TEST_FUSE", "category": "fuse_collectives", "target": "src.py:f"}],
    )
    state = _state()
    result = await _integrate(session, state, {"specialist_task_id": _SPEC})
    _assert_skipped(result, state, framework_tree, "rewrite switch")


async def test_a_tuned_artifact_without_a_diff_is_skipped(tmp_path, framework_tree):
    session = tmp_path / "session"
    workspace = _specialist(session)
    (workspace / "tuned.json").write_text("{}", encoding="utf-8")
    target = framework_tree / "configs" / "tuned.json"
    target.parent.mkdir()
    done = json.loads((workspace / "specialist_done.json").read_text(encoding="utf-8"))
    done["artifacts_written"] = [{"source": "tuned.json", "target": str(target), "kind": "config"}]
    (workspace / "specialist_done.json").write_text(json.dumps(done), encoding="utf-8")
    state = _state()
    result = await _integrate(session, state, {"specialist_task_id": _SPEC})
    _assert_skipped(result, state, framework_tree, "tuned artifact")
    assert not target.exists()


async def test_an_upstream_pr_port_is_skipped(tmp_path, framework_tree, monkeypatch):
    from hyperloom.orchestrator.actions.executors.integrate import _integrate_resolve

    session = tmp_path / "session"
    fetched = session / "runs" / "integrate_patch" / "int-1" / "pr.patch"

    def materialize(*, dest, **_kwargs):
        fetched.parent.mkdir(parents=True, exist_ok=True)
        fetched.write_text(_PATCH, encoding="utf-8")
        return SimpleNamespace(failure=None, patches=[fetched], mode="upstream_diff")

    monkeypatch.setattr(_integrate_resolve, "materialize_candidate_patches", materialize)
    state = _state()
    state.record_specialist_patch_verdict("https://github.com/vllm-project/vllm/pull/1", "approve")
    result = await _integrate(
        session,
        state,
        {
            "patch_source": "upstream_pr",
            "candidate": {"pr_url": "https://github.com/vllm-project/vllm/pull/1"},
            "framework_agent_candidate_id": "https://github.com/vllm-project/vllm/pull/1",
            "framework_source_root": str(framework_tree),
        },
    )
    _assert_skipped(result, state, framework_tree, "patch")


async def test_enablement_setup_commands_are_skipped(tmp_path, framework_tree):
    session = tmp_path / "session"
    _specialist(session, setup_commands=["pip install transformers==4.57.1"])
    state = _state(phase="ENABLEMENT")
    result = await _integrate(session, state, {"specialist_task_id": _SPEC, "enablement": True})
    _assert_skipped(result, state, framework_tree, "setup commands")
    assert result["enablement"] is True


async def test_an_enablement_runtime_request_is_skipped_before_it_is_provisioned(tmp_path, framework_tree):
    session = tmp_path / "session"
    _specialist(session, needs_runtime={"packages": ["transformers==4.57.1"]})
    state = _state(phase="ENABLEMENT")
    result = await _integrate(session, state, {"specialist_task_id": _SPEC, "enablement": True})
    _assert_skipped(result, state, framework_tree, "another runtime")


async def test_a_launch_only_bench_of_a_built_runtime_is_skipped(tmp_path, framework_tree):
    state = _state(phase="ENABLEMENT")
    result = await _integrate(
        tmp_path / "session",
        state,
        {"enablement": True, "enablement_launch_only": True, "runtime_override": {"venv_root": "/opt/build"}},
    )
    _assert_skipped(result, state, framework_tree, "pre-built runtime")


async def test_an_ordinary_round_carrying_a_runtime_override_is_skipped(tmp_path, framework_tree):
    """Server arguments alone would pass, but the bench would run them on another runtime."""
    session = tmp_path / "session"
    _specialist(session)
    state = _state()
    result = await _integrate(
        session,
        state,
        {
            "specialist_task_id": _SPEC,
            "extra_server_args": "--max-num-seqs 256",
            "runtime_override": {"venv_root": "/opt/build"},
        },
    )
    _assert_skipped(result, state, framework_tree, "pre-built runtime")


async def test_launch_configuration_passes_the_gate(tmp_path, framework_tree):
    session = tmp_path / "session"
    _specialist(session)
    result = await _integrate(
        session,
        _state(),
        {
            "specialist_task_id": _SPEC,
            "extra_server_args": "--max-num-seqs 256",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
            "apply_only": True,
        },
    )
    assert result["status"] == "applied_no_bench"
    assert result["extra_envs_applied"] == {"VLLM_ROCM_USE_AITER": "1"}


def test_an_enablement_round_with_only_flags_passes_the_gate(tmp_path):
    attempt = IntegrateAttempt(task_id="int-1")
    attempt.shared_state = _state(phase="ENABLEMENT")
    attempt.specialist_workspace = _specialist(tmp_path)
    attempt.done_payload = {"patches_written": []}
    params = {"enablement": True, "extra_server_args": "--enforce-eager", "extra_envs": {"VLLM_USE_V1": "1"}}
    assert source_change_skip(attempt, params) is None


async def test_without_the_flag_a_patch_is_applied(tmp_path, framework_tree, monkeypatch):
    monkeypatch.setattr(integrate_apply, "_candidate_mutation_roots", lambda **_kwargs: [])
    session = tmp_path / "session"
    _specialist(session, patches=[_PATCH])
    result = await _integrate(
        session,
        _state(no_source_changes=False),
        {"specialist_task_id": _SPEC, "framework_source_root": str(framework_tree), "apply_only": True},
    )
    assert result["status"] == "applied_no_bench"
    assert (framework_tree / "src.py").read_text(encoding="utf-8").endswith("return 2\n")
