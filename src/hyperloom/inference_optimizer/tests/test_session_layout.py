# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session_dir layout regression tests (per-model/ts default)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session import paths
from hyperloom.inference_optimizer.session.manifest import (
    SCHEMA_VERSION,
    build_session_id,
    load_manifest,
    write_manifest,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy import gate as policy_gate
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.inference_optimizer.session.session_paths import (
    agent_prompt_snapshot,
    manifest_path,
    patches_dir,
    runs_dir,
)
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection


def test_session_dir_default_follows_the_host(monkeypatch, tmp_path):
    """/workspace is a container convention, absent on bare metal off root."""
    import os as _os

    monkeypatch.delenv(paths.ENV_USER_DATA_PATH, raising=False)

    monkeypatch.setattr(_os, "access", lambda _p, _m: True)
    assert paths.session_dir() == Path("/workspace/hyperloom")

    monkeypatch.setattr(_os, "access", lambda _p, _m: False)
    monkeypatch.chdir(tmp_path)
    assert paths.session_dir() == tmp_path / "session"


def test_session_dir_user_data_path_overrides_default(tmp_path, monkeypatch):
    """USER_DATA_PATH overrides the /workspace/hyperloom default."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "ud"))
    assert paths.session_dir() == tmp_path / "ud"


def test_make_session_dir_creates_full_skeleton(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    # No model_name -> session_dir == workspace_root.
    assert sd == tmp_path
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir(), f"missing per-session skeleton subdir: {sub}"
    for sub in paths._WORKSPACE_SKELETON:
        assert (paths.workspace_root() / sub).is_dir(), f"missing workspace skeleton subdir: {sub}"
    # Re-running must be idempotent.
    sd2 = paths.make_session_dir()
    assert sd2 == sd


def test_workspace_root_returns_user_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.workspace_root() == tmp_path


def test_workspace_root_independent_of_session_pin(tmp_path, monkeypatch):
    """workspace_root() never consults the session pin (runtime/ etc. are workspace-scoped)."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.setenv(paths.ENV_CURRENT_SESSION_DIR, str(tmp_path / "x/y/z"))
    assert paths.workspace_root() == tmp_path


def test_relative_user_data_path_resolves_identically_from_every_cwd(tmp_path, monkeypatch):
    """A relative $USER_DATA_PATH must not follow each subprocess's cwd.

    Absolutising on read is not enough — every process would re-expand the
    relative value against its own cwd — so the CLI rewrites the env var itself.
    """
    from hyperloom.inference_optimizer import cli

    (tmp_path / "nested").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, "relws")
    with pytest.raises(SystemExit):
        cli.main([])  # no subcommand; the boundary normalises before parsing
    ws = paths.workspace_root()
    assert ws == Path.cwd() / "relws"
    sd = paths.make_session_dir(model_name="DeepSeek-R1-0528")

    monkeypatch.chdir(tmp_path / "nested")
    assert paths.workspace_root() == ws
    assert paths.session_dir() == sd


def test_make_session_dir_per_model_ts_layout(tmp_path, monkeypatch):
    """Default: per-model/per-launch subdir + pin propagation."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(model_name="/path/models/DeepSeek-R1-0528")
    # Layout: <ws>/DeepSeek-R1-0528/<UTC ts>-<rand8>/
    assert sd.parent.parent == tmp_path
    assert sd.parent.name == "DeepSeek-R1-0528"
    # Name shape: YYYYMMDDTHHMMSSZ-<8 hex>
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", sd.name), sd.name
    import os as _os

    assert _os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(sd)
    assert paths.session_dir() == sd
    for sub in paths._SESSION_SKELETON:
        assert (sd / sub).is_dir()
    # Workspace skeleton landed under ws, not under sd.
    for sub in paths._WORKSPACE_SKELETON:
        assert (tmp_path / sub).is_dir()
        assert not (sd / sub).exists()


def test_make_session_dir_same_second_launches_get_distinct_dirs(tmp_path, monkeypatch):
    """Two launches of one model inside the same UTC second must not share a dir.

    ``session_dir.name`` is also the de-facto session id (KB fact writes,
    per-session sinks), so a shared dir merges two runs' identity as well.
    """
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.setattr(paths, "utc_now_compact", lambda: "20260814T073026Z")
    first = paths.make_session_dir(model_name="DeepSeek-R1-0528")
    second = paths.make_session_dir(model_name="DeepSeek-R1-0528")
    assert first != second
    assert first.name[:16] == second.name[:16] == "20260814T073026Z"


def test_make_session_dir_sanitises_model_basename(tmp_path, monkeypatch):
    """HF ids, absolute paths, and unsafe chars all reduce to a filename-safe basename."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(model_name="meta-llama/Llama-3.1-70B-Instruct")
    assert sd.parent.name == "Llama-3.1-70B-Instruct"


def test_make_session_dir_accepts_path_object(tmp_path, monkeypatch):
    """The helper must accept any os.PathLike (args.model is a Path in the CLI), not just str."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir(
        model_name=Path("/path/models/DeepSeek-R1-0528"),
    )
    assert sd.parent.name == "DeepSeek-R1-0528"


def test_make_session_dir_overwrites_stale_pin(tmp_path, monkeypatch):
    """A subsequent make_session_dir() overwrites the pin (no cross-test pollution)."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd1 = paths.make_session_dir(model_name="A")
    sd2 = paths.make_session_dir(model_name="B")
    assert sd1 != sd2
    import os as _os

    assert _os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(sd2)
    assert paths.session_dir() == sd2


def test_runtime_dir_is_workspace_shared(tmp_path, monkeypatch):
    """runtime/ lives under workspace_root, not the per-session subdir."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    assert paths.runtime_dir() == tmp_path / "runtime"


def test_magpie_dir_is_cache_and_decoupled_from_user_data(tmp_path, monkeypatch):
    # Magpie resolves under the deps cache root (mirrors install.sh),
    # NOT under $USER_DATA_PATH/runtime, so script + runtime agree on one checkout.
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "shared"))
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    expected = tmp_path / "cache"
    assert paths.deps_cache_root() == expected
    assert paths.magpie_dir() == expected / "Magpie"
    assert str(tmp_path / "shared") not in str(paths.magpie_dir())


def test_deps_cache_root_defaults_to_repo_root_cache(tmp_path, monkeypatch):
    # Default is $REPO_ROOT/.cache: portable and repo-local.
    monkeypatch.delenv("HYPERLOOM_CACHE_DIR", raising=False)
    monkeypatch.setenv(paths.ENV_REPO_ROOT, str(tmp_path / "repo"))
    assert paths.deps_cache_root() == tmp_path / "repo" / ".cache"


def test_deps_cache_root_honours_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "custom"))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert paths.deps_cache_root() == tmp_path / "custom"
    assert paths.magpie_dir() == tmp_path / "custom" / "Magpie"


def test_magpie_dir_honours_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "operator-magpie"))
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "ignored"))
    assert paths.magpie_dir() == tmp_path / "operator-magpie"


def test_resolve_dep_dir_prefers_env_var(tmp_path, monkeypatch):
    # The installer-exported env var is the exact, preferred checkout.
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "operator-tl"))
    assert paths.resolve_dep_dir("TraceLens", "TRACELENS_ROOT") == tmp_path / "operator-tl"


def test_resolve_dep_dir_globs_pinned_checkout_when_env_unset(tmp_path, monkeypatch):
    # install.sh clones <name>@<sha>; a process that did NOT inherit the exported
    # env var must still resolve that checkout, not the bare path the installer
    # never created (the #3 regression).
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    pinned = tmp_path / "cache" / "TraceLens@deadbeef"
    pinned.mkdir(parents=True)
    assert paths.resolve_dep_dir("TraceLens", "TRACELENS_ROOT") == pinned
    assert paths.tracelens_root() == pinned  # rides the same resolver


def test_resolve_dep_dir_picks_newest_pinned_checkout(tmp_path, monkeypatch):
    import os as _os

    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    old = tmp_path / "cache" / "Magpie@1111111"
    new = tmp_path / "cache" / "Magpie@2222222"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    _os.utime(old, (1_000_000, 1_000_000))
    _os.utime(new, (2_000_000, 2_000_000))  # newest wins regardless of glob order
    assert paths.resolve_dep_dir("Magpie", "MAGPIE_PATH") == new


def test_resolve_dep_dir_falls_back_to_bare_when_no_pinned(tmp_path, monkeypatch):
    # No <name>@<sha> dir (pip-installed Magpie layout): bare default.
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    assert paths.resolve_dep_dir("Magpie", "MAGPIE_PATH") == tmp_path / "cache" / "Magpie"


# TraceLens root resolution: mirrors magpie_dir so trace analysis resolves the
# same checkout as install.sh even when TRACELENS_ROOT was not inherited.
def test_tracelens_root_derives_from_cache_root_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))
    expected = tmp_path / "cache" / "TraceLens"
    assert paths.tracelens_root() == expected


def test_tracelens_root_honours_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "operator-tracelens"))
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "ignored"))
    assert paths.tracelens_root() == tmp_path / "operator-tracelens"


def test_write_manifest_writes_v1_schema(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="explicit-id-123")
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["session_id"] == "explicit-id-123"
    on_disk = json.loads(manifest_path(sd).read_text())
    assert on_disk == m


# manifest "dependencies" block records each upstream's SHA/remote.
def test_manifest_records_dependencies_block_empty_when_envs_unset(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="empty-deps")
    deps = m["dependencies"]
    assert set(deps.keys()) == {"magpie", "inferencex"}
    for sub in deps.values():
        assert sub == {"path": "", "commit": "", "remote": ""}


def test_manifest_records_dependencies_block_picks_up_git_metadata(
    tmp_path,
    monkeypatch,
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
    monkeypatch.setenv("MAGPIE_PATH", str(fake_magpie))
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
    tmp_path,
    monkeypatch,
):
    """Path exists but isn't a git checkout -> path is recorded, sha/remote stay empty."""
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "user_data"))
    monkeypatch.setenv("MAGPIE_PATH", str(not_a_repo))
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)

    sd = paths.make_session_dir()
    m = write_manifest(sd, args=None, session_id="non-repo-deps")
    assert m["dependencies"]["magpie"] == {
        "path": str(not_a_repo),
        "commit": "",
        "remote": "",
    }
    assert m["dependencies"]["inferencex"] == {
        "path": "",
        "commit": "",
        "remote": "",
    }


def test_manifest_pod_local_dependency_warning_matches_default_policy(
    tmp_path,
    monkeypatch,
    caplog,
):
    from hyperloom.inference_optimizer.session import manifest

    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path / "user_data"))
    monkeypatch.setenv("MAGPIE_PATH", "/workspace/hyperloom_runtime_smoke/Magpie")

    with caplog.at_level(logging.WARNING, logger="hyperloom.inference_optimizer.session.manifest"):
        manifest._describe_dep("MAGPIE_PATH")

    messages = [r.message for r in caplog.records if "MAGPIE_PATH" in r.message]
    assert messages
    assert "defaults open-source dependencies to the repo-local cache" in messages[0]
    assert "point MAGPIE_PATH back" not in messages[0]


def test_build_session_id_includes_uuid_and_model(monkeypatch):
    sid = build_session_id("Qwen3-8B")
    assert sid.startswith("Qwen3-8B_")
    # Suffix is <UTC compact ts>_<8 hex chars>.
    parts = sid.split("_")
    assert len(parts[-1]) == 8 and all(c in "0123456789abcdef" for c in parts[-1])


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "no-such-session")


def test_runs_dir_layout(tmp_path):
    p = runs_dir(tmp_path, "baseline", "task-abcdef01")
    assert p == tmp_path / "runs" / "baseline" / "task-abcdef01"


def test_runs_dir_rejects_unknown_action(tmp_path):
    with pytest.raises(ValueError):
        runs_dir(tmp_path, "this_is_not_an_action", "x")


def test_patches_path(tmp_path):
    assert patches_dir(tmp_path, "k001") == tmp_path / "patches" / "k001"


def test_agent_prompt_snapshot_path(tmp_path):
    assert agent_prompt_snapshot(tmp_path, "orchestration") == (
        tmp_path / "agents" / "orchestration" / "system_prompt.snapshot.md"
    )


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
        kind="baseline",
        params={},
        idempotency_key="ws-test-1",
    )
    res = await sub.run_task(task)
    db.close()
    assert res.state == "succeeded"
    assert captured["workspace"] == str(sd / "runs" / "baseline" / task.task_id)
    assert (sd / "runs" / "baseline" / task.task_id).is_dir()
    assert captured["session_dir"] == str(sd)


@pytest.mark.asyncio
async def test_sub_agent_runner_skips_unknown_action(tmp_path, monkeypatch):
    """`target_analysis` is not in _RUNS_ACTIONS — runner shouldn't fabricate a path."""
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
        kind="target_analysis",
        params={},
        idempotency_key="target-analysis-test-1",
    )
    await sub.run_task(task)
    db.close()
    # target_analysis has no runs/ subtree — workspace stays unset.
    assert captured["workspace"] is None
    assert captured["session_dir"] == str(sd)


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
            "target_agent": "kernel_agent",
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
            "target_agent": "kernel_agent",
            "kind": "trace_analyze",
            "params": {"trace_input": "/tmp/some-trace.json.gz"},
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "path_outside_session_dir"


def test_policy_source_file_trusted_scope_passes(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "integrate",
            "params": {
                "kernel_id": "k1",
                "source_file": "/sgl-workspace/aiter/csrc/attn.cu",
            },
        },
    )
    gate.validate_intent("orchestration", intent)


def test_policy_source_file_any_installed_package_passes(tmp_path, monkeypatch):
    packages = tmp_path / "lib" / "python3.12" / "site-packages"
    source = packages / "unrelated_package" / "native" / "kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text("// source")
    monkeypatch.setattr(policy_gate, "resolve_source_file_allowlist", lambda: (f"{packages}/",))
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "integrate",
            "params": {"kernel_id": "k1", "source_file": str(source)},
        },
    )
    gate.validate_intent("orchestration", intent)


def test_policy_source_file_outside_trusted_scope_denied(tmp_path):
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "integrate",
            "params": {
                "kernel_id": "k1",
                "source_file": "/random/path/attn.cu",
            },
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "source_file_outside_trusted_scope"


def test_policy_framework_source_root_outside_trusted_scope_denied(tmp_path):
    # A framework_source_root override escaping trusted source scopes must be
    # rejected under strict_paths.
    gate = _gate(tmp_path)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "trace_analyze",
            "params": {
                "trace_input": str(tmp_path / "runs" / "x.json.gz"),
                "framework_source_root": "/root",
            },
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "source_file_outside_trusted_scope"


def test_policy_strict_off_skips_path_check(tmp_path):
    gate = _gate(tmp_path, strict=False)
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
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
            "target_agent": "kernel_agent",
            "kind": "trace_analyze",
            "params": {"trace_input": "/tmp/x.json"},
        },
    )
    with pytest.raises(PolicyDenied):
        gate.validate_intent("orchestration", intent)
