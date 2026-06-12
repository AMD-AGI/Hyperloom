# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session_dir layout regression tests (N17 per-model/ts default)."""

from __future__ import annotations

import json
import logging
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
from inference_optimizer.protocol.intent import Intent, IntentType
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


# paths.session_dir + skeleton
def test_session_dir_default_is_workspace_hyperloom(monkeypatch):
    monkeypatch.delenv(paths.ENV_USER_DATA_PATH, raising=False)
    assert paths.session_dir() == Path("/workspace/hyperloom")


def test_session_dir_user_data_path_overrides_default(tmp_path, monkeypatch):
    """USER_DATA_PATH overrides the /workspace/hyperloom default."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "ud"))
    assert paths.session_dir() == tmp_path / "ud"


def test_make_session_dir_creates_full_skeleton(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    # No model_name -> flat layout, session_dir == workspace_root.
    assert sd == tmp_path
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir(), f"missing per-session skeleton subdir: {sub}"
    for sub in paths._WORKSPACE_SKELETON:
        assert (paths.workspace_root() / sub).is_dir(), (
            f"missing workspace skeleton subdir: {sub}"
        )
    # Re-running must be idempotent.
    sd2 = paths.make_session_dir()
    assert sd2 == sd


# N17 per-model/ts layout
def test_workspace_root_returns_user_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.workspace_root() == tmp_path


def test_workspace_root_independent_of_session_pin(tmp_path, monkeypatch):
    """workspace_root() never consults the session pin (runtime/ etc. are workspace-scoped)."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.setenv(paths.ENV_CURRENT_SESSION_DIR, str(tmp_path / "x/y/z"))
    assert paths.workspace_root() == tmp_path


def test_make_session_dir_per_model_ts_layout(tmp_path, monkeypatch):
    """N17 default: per-model/per-launch subdir + pin propagation."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(model_name="/wekafs/models/DeepSeek-R1-0528")
    # Layout: <ws>/DeepSeek-R1-0528/<UTC ts>/
    assert sd.parent.parent == tmp_path
    assert sd.parent.name == "DeepSeek-R1-0528"
    # Timestamp shape: YYYYMMDDTHHMMSSZ
    assert len(sd.name) == 16 and sd.name.endswith("Z") and "T" in sd.name
    import os as _os
    assert _os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(sd)
    assert paths.session_dir() == sd
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir()
    # Workspace skeleton landed under ws, not under sd.
    for sub in paths._WORKSPACE_SKELETON:
        assert (tmp_path / sub).is_dir()
        assert not (sd / sub).exists()


def test_make_session_dir_sanitises_model_basename(tmp_path, monkeypatch):
    """HF ids, absolute paths, and unsafe chars all reduce to a filename-safe basename."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(model_name="meta-llama/Llama-3.1-70B-Instruct")
    assert sd.parent.name == "Llama-3.1-70B-Instruct"


def test_make_session_dir_accepts_path_object(tmp_path, monkeypatch):
    """The helper must accept any os.PathLike (args.model is a Path in the CLI), not just str."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(
        model_name=Path("/wekafs/models/DeepSeek-R1-0528"),
    )
    assert sd.parent.name == "DeepSeek-R1-0528"


def test_make_session_dir_flat_layout_via_env(tmp_path, monkeypatch):
    """Env override forces legacy flat layout even when model_name is set."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.setenv(paths.ENV_SESSION_LAYOUT, "flat")
    sd = paths.make_session_dir(model_name="DeepSeek-R1-0528")
    assert sd == tmp_path


def test_make_session_dir_overwrites_stale_pin(tmp_path, monkeypatch):
    """A subsequent make_session_dir() overwrites the pin (no cross-test pollution)."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd1 = paths.make_session_dir(model_name="A")
    sd2 = paths.make_session_dir(model_name="B")
    assert sd1 != sd2
    import os as _os
    assert _os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(sd2)
    assert paths.session_dir() == sd2


def test_find_latest_per_session_dir_returns_none_on_empty(
    tmp_path, monkeypatch,
):
    """No per-session subdir under workspace_root -> None."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.find_latest_per_session_dir() is None
    assert paths.find_latest_per_session_dir(model_name="DSR1") is None


def test_find_latest_per_session_dir_picks_lex_latest_ts(
    tmp_path, monkeypatch,
):
    """Lex-sort on the YYYYMMDDTHHMMSSZ name picks the latest ts (robust to mtime touches)."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    (tmp_path / "MyModel").mkdir()
    (tmp_path / "MyModel" / "20260101T000000Z").mkdir()
    (tmp_path / "MyModel" / "20260520T120000Z").mkdir()
    (tmp_path / "MyModel" / "20260315T080000Z").mkdir()
    picked = paths.find_latest_per_session_dir(model_name="MyModel")
    assert picked is not None
    assert picked.name == "20260520T120000Z"


def test_find_latest_per_session_dir_no_model_scans_all(
    tmp_path, monkeypatch,
):
    """No model_name -> scan all model_basename subdirs for the latest ts across the workspace."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    (tmp_path / "Qwen-7B").mkdir()
    (tmp_path / "Qwen-7B" / "20260101T000000Z").mkdir()
    (tmp_path / "DSR1").mkdir()
    (tmp_path / "DSR1" / "20260520T120000Z").mkdir()
    picked = paths.find_latest_per_session_dir()
    assert picked is not None
    assert picked.name == "20260520T120000Z"
    assert picked.parent.name == "DSR1"


def test_find_latest_per_session_dir_skips_workspace_shared(
    tmp_path, monkeypatch,
):
    """workspace-shared subdirs (runtime/, logs/) must not be mistaken for model_basename subdirs."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "20260520T120000Z").mkdir()  # decoy
    (tmp_path / "logs").mkdir()
    (tmp_path / "MyModel").mkdir()
    (tmp_path / "MyModel" / "20260518T100000Z").mkdir()
    picked = paths.find_latest_per_session_dir()
    assert picked is not None
    assert picked.parent.name == "MyModel"
    assert "runtime" not in str(picked)


def test_find_latest_per_session_dir_ignores_non_ts_dirs(
    tmp_path, monkeypatch,
):
    """Only YYYYMMDDTHHMMSSZ-shaped dir names count as ts subdirs."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    (tmp_path / "MyModel").mkdir()
    (tmp_path / "MyModel" / "scratch").mkdir()  # ignored
    (tmp_path / "MyModel" / "backup-2026").mkdir()  # ignored
    (tmp_path / "MyModel" / "20260520T120000Z").mkdir()  # picked
    picked = paths.find_latest_per_session_dir(model_name="MyModel")
    assert picked is not None
    assert picked.name == "20260520T120000Z"


def test_runtime_dir_is_workspace_shared(tmp_path, monkeypatch):
    """N17: runtime/ lives under workspace_root, not the per-session subdir."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(model_name="DeepSeek-R1-0528")
    assert paths.runtime_dir(sd) == tmp_path / "runtime"
    # Also true when caller passes the historical no-arg form (back-compat)
    assert paths.runtime_dir() == tmp_path / "runtime"


def test_magpie_dir_is_pod_local_and_decoupled_from_user_data(tmp_path, monkeypatch):
    # Magpie resolves under the pod-local open-source root (mirrors install.sh),
    # NOT under $USER_DATA_PATH/runtime, so script + runtime agree on one checkout.
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "shared"))
    monkeypatch.delenv("HYPERLOOM_OPEN_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "podlocal"))
    expected = tmp_path / "podlocal" / "hyperloom" / "open-source-repos"
    assert paths.open_source_root() == expected
    assert paths.magpie_dir() == expected / "Magpie"
    assert str(tmp_path / "shared") not in str(paths.magpie_dir())


def test_open_source_root_honours_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_OPEN_SOURCE_ROOT", str(tmp_path / "custom"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "ignored"))
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    assert paths.open_source_root() == tmp_path / "custom"
    assert paths.magpie_dir() == tmp_path / "custom" / "Magpie"


def test_magpie_dir_honours_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGPIE_DIR", str(tmp_path / "operator-magpie"))
    monkeypatch.setenv("HYPERLOOM_OPEN_SOURCE_ROOT", str(tmp_path / "ignored"))
    assert paths.magpie_dir() == tmp_path / "operator-magpie"


# manifest
def test_write_manifest_writes_v1_schema(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="explicit-id-123")
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["session_id"] == "explicit-id-123"
    on_disk = json.loads(manifest_path(sd).read_text())
    assert on_disk == m


# manifest "dependencies" block — records each upstream's SHA/remote so we can
# answer "which upstream did this run hit?" (install.sh clones fresh per install).
def test_manifest_records_dependencies_block_empty_when_envs_unset(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.delenv("MAGPIE_DIR", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="empty-deps")
    deps = m["dependencies"]
    assert set(deps.keys()) == {"magpie", "inferencex"}
    for sub in deps.values():
        assert sub == {"path": "", "commit": "", "remote": ""}


def test_manifest_records_dependencies_block_picks_up_git_metadata(
    tmp_path, monkeypatch,
):
    """Plant two fake git checkouts and confirm we capture both SHA and origin URL."""
    import subprocess

    def _init_repo(path, remote_url, file_contents):
        path.mkdir(parents=True)
        (path / "stub.txt").write_text(file_contents, encoding="utf-8")
        for cmd in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "ci@hyperloom.test"],
            ["git", "config", "user.name", "ci"],
            ["git", "config", "commit.gpgsign", "false"],
            ["git", "add", "."],
            ["git", "commit", "-q", "-m", "init"],
            ["git", "remote", "add", "origin", remote_url],
        ):
            subprocess.run(cmd, cwd=path, check=True, capture_output=True)

    fake_magpie = tmp_path / "Magpie"
    fake_infx = tmp_path / "InferenceX"
    _init_repo(fake_magpie, "https://example.test/Magpie.git", "m")
    _init_repo(fake_infx, "https://example.test/InferenceX.git", "i")

    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "user_data"))
    monkeypatch.setenv("MAGPIE_DIR", str(fake_magpie))
    monkeypatch.setenv("INFERENCEX_PATH", str(fake_infx))

    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="full-deps")

    deps = m["dependencies"]
    assert deps["magpie"]["path"] == str(fake_magpie)
    assert deps["magpie"]["remote"] == "https://example.test/Magpie.git"
    assert deps["magpie"]["commit"], "expected non-empty magpie SHA"
    assert deps["inferencex"]["path"] == str(fake_infx)
    assert deps["inferencex"]["remote"] == "https://example.test/InferenceX.git"
    assert deps["inferencex"]["commit"], "expected non-empty inferencex SHA"


def test_manifest_dependencies_block_is_fail_soft_on_non_repo_paths(
    tmp_path, monkeypatch,
):
    """Path exists but isn't a git checkout -> path is recorded, sha/remote stay empty."""
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "user_data"))
    monkeypatch.setenv("MAGPIE_DIR", str(not_a_repo))
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="non-repo-deps")
    assert m["dependencies"]["magpie"] == {
        "path": str(not_a_repo), "commit": "", "remote": "",
    }
    assert m["dependencies"]["inferencex"] == {
        "path": "", "commit": "", "remote": "",
    }


def test_manifest_pod_local_dependency_warning_matches_default_policy(
    tmp_path, monkeypatch, caplog,
):
    from inference_optimizer import manifest

    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "user_data"))
    monkeypatch.setenv("MAGPIE_DIR", "/workspace/hyperloom_runtime_smoke/Magpie")

    with caplog.at_level(logging.WARNING, logger="inference_optimizer.manifest"):
        manifest._describe_dep("MAGPIE_DIR")

    messages = [r.message for r in caplog.records if "MAGPIE_DIR" in r.message]
    assert messages
    assert "defaults open-source dependencies to pod-local storage" in messages[0]
    assert "point MAGPIE_DIR back" not in messages[0]


def test_build_session_id_includes_uuid_and_model(monkeypatch):
    sid = build_session_id("Qwen3-8B")
    assert sid.startswith("Qwen3-8B_")
    # Suffix is <UTC compact ts>_<8 hex chars>.
    parts = sid.split("_")
    assert len(parts[-1]) == 8 and all(c in "0123456789abcdef" for c in parts[-1])


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "no-such-session")


# session_paths helpers
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


# SubAgentRunner pre-mkdir + workspace injection
@pytest.mark.asyncio
async def test_sub_agent_runner_premkdirs_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
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
    """`target_analysis` is not in _runs_actions() — runner shouldn't fabricate a path."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
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
    sub.register_executor("target_analysis", runner)
    task = await tasks.create(
        kind="target_analysis", params={}, idempotency_key="target-analysis-test-1",
    )
    await sub.run_task(task)
    db.close()
    # target_analysis has no runs/ subtree — workspace stays unset.
    assert captured["workspace"] is None
    assert captured["session_dir"] == str(sd)


# PolicyGate strict path-containment
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
            "kind": "trace_analyze",
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
            "kind": "trace_analyze",
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
            "kind": "trace_analyze_done",
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
            "kind": "trace_analyze_done",
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
            "kind": "trace_analyze",
            "params": {"trace_input": "/tmp/anywhere.json"},
        },
    )
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
            "kind": "trace_analyze",
            "params": {"trace_input": "/tmp/x.json"},
        },
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)
