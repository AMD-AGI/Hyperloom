"""P6 — flattened session_dir layout regression tests.

Locks the v0.6.1 contract:

* ``paths.session_dir()`` returns ``/workspace/hyperloom`` by default;
  ``$INFERENCE_OPTIMIZER_SESSION_DIR`` overrides for tests.
* ``paths.make_session_dir()`` is idempotent and creates the full
  skeleton (storage / agents / runs / logs / patches / ...).
* ``manifest.write_manifest()`` writes ``manifest.json`` atomically
  with the v1 schema.
* ``manifest.load_manifest()`` raises ``FileNotFoundError`` when the
  session has not been initialised.
* ``session_paths.runs_dir()`` rejects unknown action names so typos
  fail loudly.
* ``SubAgentRunner`` pre-creates ``runs/<action>/<task_id>/`` and
  injects it into ``RunnerContext.extra['workspace']``.
* ``PolicyGate`` with ``strict_paths=True`` rejects intents whose
  path-like fields escape ``session_dir`` and accepts ``source_file``
  values under the framework allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer import paths
from inference_optimizer.manifest import (
    SCHEMA_VERSION,
    build_session_id,
    load_manifest,
    write_manifest,
)
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import SubAgentRunner
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.session_paths import (
    agent_prompt_snapshot,
    kernel_workspace,
    manifest_path,
    patches_dir,
    runs_dir,
)
from inference_optimizer.storage.connection import SqliteConnection


# ---------------------------------------------------------------------------
# paths.session_dir + skeleton
# ---------------------------------------------------------------------------
def test_session_dir_default_is_workspace_hyperloom(monkeypatch):
    monkeypatch.delenv(paths.ENV_OVERRIDE_SESSION_DIR, raising=False)
    assert paths.session_dir() == Path("/workspace/hyperloom")


def test_session_dir_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE_SESSION_DIR, str(tmp_path / "alt"))
    assert paths.session_dir() == tmp_path / "alt"


def test_make_session_dir_creates_full_skeleton(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE_SESSION_DIR, str(tmp_path))
    sd = paths.make_session_dir()
    assert sd == tmp_path
    # Every entry in _SESSION_SKELETON must exist after the first call.
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir(), f"missing skeleton subdir: {sub}"
    # Re-running must be a no-op (idempotent).
    sd2 = paths.make_session_dir()
    assert sd2 == sd


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def test_write_manifest_writes_v1_schema(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE_SESSION_DIR, str(tmp_path))
    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="explicit-id-123")
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["session_id"] == "explicit-id-123"
    on_disk = json.loads(manifest_path(sd).read_text())
    assert on_disk == m


def test_build_session_id_includes_uuid_and_model(monkeypatch):
    sid = build_session_id("Qwen3-8B")
    assert sid.startswith("Qwen3-8B_")
    # Suffix is <UTC compact ts>_<8 hex chars>; sanity-check length.
    parts = sid.split("_")
    assert len(parts[-1]) == 8 and all(c in "0123456789abcdef" for c in parts[-1])


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "no-such-session")


# ---------------------------------------------------------------------------
# session_paths helpers
# ---------------------------------------------------------------------------
def test_runs_dir_layout(tmp_path):
    p = runs_dir(tmp_path, "baseline", "task-abcdef01")
    assert p == tmp_path / "runs" / "baseline" / "task-abcdef01"


def test_runs_dir_rejects_unknown_action(tmp_path):
    with pytest.raises(ValueError):
        runs_dir(tmp_path, "this_is_not_an_action", "x")


def test_kernel_workspace_and_patches(tmp_path):
    assert kernel_workspace(tmp_path, "k001") == tmp_path / "kernel-agent-workspace" / "k001"
    assert patches_dir(tmp_path, "k001") == tmp_path / "patches" / "k001"


def test_agent_prompt_snapshot_path(tmp_path):
    assert agent_prompt_snapshot(tmp_path, "orchestration") == (
        tmp_path / "agents" / "orchestration" / "system_prompt.snapshot.md"
    )


# ---------------------------------------------------------------------------
# SubAgentRunner pre-mkdir + workspace injection
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sub_agent_runner_premkdirs_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_OVERRIDE_SESSION_DIR, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    captured: dict = {}

    async def runner(ctx) -> dict:
        captured["workspace"] = ctx.extra.get("workspace")
        captured["session_dir"] = ctx.extra.get("session_dir")
        return {"status": "succeeded"}

    sub = SubAgentRunner(locks, tasks, session_dir=sd)
    sub.register_executor("baseline", runner)
    task = await tasks.create(
        kind="baseline", params={}, idempotency_key="ws-test-1",
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"
    assert captured["workspace"] == str(sd / "runs" / "baseline" / task.task_id)
    assert (sd / "runs" / "baseline" / task.task_id).is_dir()
    assert captured["session_dir"] == str(sd)


@pytest.mark.asyncio
async def test_sub_agent_runner_skips_unknown_action(tmp_path, monkeypatch):
    """`setup` is not in _RUNS_ACTIONS — runner shouldn't fabricate a path."""
    monkeypatch.setenv(paths.ENV_OVERRIDE_SESSION_DIR, str(tmp_path))
    sd = paths.make_session_dir()
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tasks = TaskRegistry(db)
    captured: dict = {}

    async def runner(ctx) -> dict:
        captured["workspace"] = ctx.extra.get("workspace")
        captured["session_dir"] = ctx.extra.get("session_dir")
        return {"status": "succeeded"}

    sub = SubAgentRunner(locks, tasks, session_dir=sd)
    sub.register_executor("setup", runner)
    task = await tasks.create(
        kind="setup", params={}, idempotency_key="setup-test-1",
    )
    await sub.run_task(task)
    db.close()
    # setup has no runs/ subtree — workspace stays unset, session_dir is plumbed.
    assert captured["workspace"] is None
    assert captured["session_dir"] == str(sd)


# ---------------------------------------------------------------------------
# PolicyGate strict path-containment
# ---------------------------------------------------------------------------
def _gate(tmp_path: Path, *, strict: bool = True) -> PolicyGate:
    return PolicyGate(
        role_registry=default_role_registry(),
        session_dir=tmp_path,
        strict_paths=strict,
    )


def test_policy_path_inside_session_dir_passes(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_input": str(tmp_path / "runs" / "profile" / "x.json.gz")},
        },
    )
    gate.validate_intent("orchestration", intent)


def test_policy_path_outside_session_dir_denied(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_input": "/tmp/some-trace.json.gz"},
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "path_outside_session_dir"


def test_policy_source_file_allowlist_passes(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.RESPONSE,
        payload={
            "in_reply_to": "m-1",
            "kind": "select_kernels_done",
            "result": {
                "hot_kernels": [
                    {"kernel_id": "k1",
                     "source_file": "/sgl-workspace/aiter/csrc/attn.cu"},
                ],
            },
        },
    )
    gate.validate_intent("kernel", intent)


def test_policy_source_file_outside_allowlist_denied(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.RESPONSE,
        payload={
            "in_reply_to": "m-1",
            "kind": "select_kernels_done",
            "result": {
                "hot_kernels": [
                    {"kernel_id": "k1",
                     "source_file": "/random/path/attn.cu"},
                ],
            },
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("kernel", intent)
    assert exc.value.rule == "source_file_not_allowlisted"


def test_policy_strict_off_skips_path_check(tmp_path):
    gate = _gate(tmp_path, strict=False)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_input": "/tmp/anywhere.json"},
        },
    )
    # Should not raise — strict_paths=False means the check is a no-op.
    gate.validate_intent("orchestration", intent)


def test_policy_env_var_enables_strict(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_PATHS", "1")
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=tmp_path,
    )
    assert gate.strict_paths is True
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_input": "/tmp/x.json"},
        },
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)
