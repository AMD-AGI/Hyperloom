"""Tests for per-attempt AITER cache ownership and lock cleanup."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from kernelforge.loop import aiter_cache


@pytest.fixture(autouse=True)
def _isolate_aiter_env():
    """Keep the cache-isolation env vars from leaking across tests.

    ``configure_aiter_cache_isolation`` writes ``os.environ`` directly (its job
    is to steer aiter's build trees for child processes). We snapshot and restore
    those keys around each test so the temp paths it sets do not pollute later
    tests (e.g. resolve_aiter_root in kernelforge.gemm_tune). monkeypatch cannot cover
    this: it only rolls back keys it recorded, and delenv on an absent key
    records nothing.
    """
    keys = (
        "AITER_ROOT_DIR",
        "AITER_JIT_DIR",
        "AITER_REBUILD",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FORGE_AITER_CACHE_ROOT",
        "FORGE_AITER_CACHE_OWNER_PID",
    )
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_configure_isolates_every_aiter_build_tree(tmp_path):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)

    assert os.environ["AITER_ROOT_DIR"] == str(isolation.aiter_root_dir)
    assert os.environ["AITER_JIT_DIR"] == str(isolation.aiter_jit_dir)
    assert os.environ["FLYDSL_RUNTIME_CACHE_DIR"] == str(isolation.flydsl_cache_dir)
    assert isolation.aiter_root_dir.is_dir()
    assert isolation.aiter_jit_dir.is_dir()
    assert isolation.flydsl_cache_dir.is_dir()
    owner = json.loads(isolation.owner_file.read_text(encoding="utf-8"))
    assert owner["owner_pid"] == os.getpid()


def test_flydsl_cache_claim_survives_a_later_aiter_import(tmp_path):
    """The claim has to be made BEFORE aiter runs, and has to stick.

    ``aiter/__init__.py`` points FLYDSL_RUNTIME_CACHE_DIR at
    ``<aiter package>/jit/flydsl_cache`` whenever that directory exists and the
    variable is unset -- and in a run that package sits inside the workspace, so
    the cache lands in a git-visible directory the guard then fails the session
    over. aiter only claims the variable when it is absent, which is the whole
    reason setting it up front is sufficient. This reproduces aiter's rule
    rather than importing aiter, which is not a dependency of the test suite.
    """
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)

    in_workspace = tmp_path / "aiter" / "jit" / "flydsl_cache"
    in_workspace.mkdir(parents=True)
    # verbatim from aiter/__init__.py
    if in_workspace.is_dir() and "FLYDSL_RUNTIME_CACHE_DIR" not in os.environ:
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(in_workspace)

    assert os.environ["FLYDSL_RUNTIME_CACHE_DIR"] == str(isolation.flydsl_cache_dir)
    assert isolation.flydsl_cache_dir not in in_workspace.parents


def test_child_environment_carries_the_flydsl_cache_too(tmp_path):
    """A lane subprocess gets its own FlyDSL shard, not the workspace's.

    Lane sessions are where this bit hardest: each lane is a copy of the
    workspace with its own git index, so a FlyDSL entry written into the copy is
    an untracked file the lane's own guard rejects, losing the whole session.
    """
    env = aiter_cache.child_cache_environment(tmp_path / "shard")

    assert env["FLYDSL_RUNTIME_CACHE_DIR"] == str(tmp_path / "shard" / "flydsl_cache")
    assert (tmp_path / "shard" / "flydsl_cache").is_dir()


def test_source_hash_cache_reuses_and_rotates_on_edit(tmp_path):
    aiter_cache.configure_aiter_cache_isolation(tmp_path)
    source = tmp_path / "aiter" / "kernel.cu"
    source.parent.mkdir()
    source.write_text("version one", encoding="utf-8")

    first = aiter_cache.activate_aiter_cache_for_sources([str(source)])
    second = aiter_cache.activate_aiter_cache_for_sources([str(source)])
    source.write_text("version two", encoding="utf-8")
    third = aiter_cache.activate_aiter_cache_for_sources([str(source)])

    assert first is not None and second is not None and third is not None
    assert first.cache_root == second.cache_root
    assert third.cache_root != first.cache_root
    assert "AITER_REBUILD" not in os.environ


def test_source_hash_ignores_unrelated_tracked_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    source = repo / "aiter" / "kernel.cu"
    source.parent.mkdir()
    source.write_text("kernel", encoding="utf-8")
    unrelated = repo / "config.txt"
    unrelated.write_text("version one", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Forge Test",
            "-c",
            "user.email=forge-test@example.com",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=repo,
        check=True,
    )
    aiter_cache.configure_aiter_cache_isolation(tmp_path / "attempt")

    first = aiter_cache.activate_aiter_cache_for_sources([str(source)])
    unrelated.write_text("version two", encoding="utf-8")
    second = aiter_cache.activate_aiter_cache_for_sources([str(source)])

    assert first is not None and second is not None
    assert first.cache_root == second.cache_root


def test_source_hash_is_order_independent_for_multiple_inputs(tmp_path):
    aiter_cache.configure_aiter_cache_isolation(tmp_path / "attempt")
    first_source = tmp_path / "aiter" / "first.cu"
    second_source = tmp_path / "aiter" / "second.cuh"
    first_source.parent.mkdir()
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")

    first = aiter_cache.activate_aiter_cache_for_sources(
        [str(first_source), str(second_source)],
    )
    second = aiter_cache.activate_aiter_cache_for_sources(
        [str(second_source), str(first_source)],
    )
    second_source.write_text("changed", encoding="utf-8")
    third = aiter_cache.activate_aiter_cache_for_sources(
        [str(first_source), str(second_source)],
    )

    assert first is not None and second is not None and third is not None
    assert first.cache_root == second.cache_root
    assert third.cache_root != first.cache_root


def test_lru_prunes_old_inactive_shards_to_target(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(
        tmp_path,
        max_cache_bytes=1_000_000,
    )
    first_source = tmp_path / "aiter" / "first.cu"
    second_source = tmp_path / "aiter" / "second.cu"
    first_source.parent.mkdir()
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")

    first = aiter_cache.activate_aiter_cache_for_sources([str(first_source)])
    assert first is not None
    first_artifact = first.aiter_jit_dir / "build" / "artifact.so"
    first_artifact.parent.mkdir(parents=True)
    first_artifact.write_bytes(b"a" * 700_000)

    second = aiter_cache.activate_aiter_cache_for_sources([str(second_source)])
    assert second is not None
    second_artifact = second.aiter_jit_dir / "build" / "artifact.so"
    second_artifact.parent.mkdir(parents=True)
    second_artifact.write_bytes(b"b" * 700_000)
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [])

    stats = aiter_cache.prune_aiter_cache_shards(
        isolation.cache_root,
        protected_shard=second.cache_root,
    )

    assert not first.cache_root.exists()
    assert second.cache_root.exists()
    assert stats["deleted_shards"] == [str(first.cache_root)]
    assert stats["after_bytes"] <= stats["target_bytes"]


def test_lru_never_deletes_live_or_current_shards(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(
        tmp_path,
        max_cache_bytes=1_000_000,
    )
    first_source = tmp_path / "aiter" / "first.cu"
    second_source = tmp_path / "aiter" / "second.cu"
    first_source.parent.mkdir()
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    first = aiter_cache.activate_aiter_cache_for_sources([str(first_source)])
    second = aiter_cache.activate_aiter_cache_for_sources([str(second_source)])
    assert first is not None and second is not None
    for shard, value in ((first, b"a"), (second, b"b")):
        artifact = shard.aiter_jit_dir / "build" / "artifact.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(value * 700_000)
    monkeypatch.setattr(
        aiter_cache,
        "_live_cache_users",
        lambda candidate: [1234] if candidate.cache_root == first.cache_root else [],
    )

    stats = aiter_cache.prune_aiter_cache_shards(
        isolation.cache_root,
        protected_shard=second.cache_root,
    )

    assert first.cache_root.exists()
    assert second.cache_root.exists()
    assert stats["deleted_shards"] == []
    assert stats["skipped_live_shards"] == [str(first.cache_root)]


def test_finished_attempt_deletes_private_cache(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)
    source = tmp_path / "aiter" / "kernel.cu"
    source.parent.mkdir()
    source.write_text("kernel", encoding="utf-8")
    active = aiter_cache.activate_aiter_cache_for_sources([str(source)])
    assert active is not None
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [])

    stats = aiter_cache.cleanup_current_aiter_cache()

    assert stats is not None
    assert stats["deleted"] is True
    assert not isolation.cache_root.exists()
    assert "FORGE_AITER_CACHE_ROOT" not in os.environ
    assert "AITER_ROOT_DIR" not in os.environ
    assert "AITER_JIT_DIR" not in os.environ


def test_finished_attempt_preserves_cache_with_live_child(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)
    source = tmp_path / "aiter" / "kernel.cu"
    source.parent.mkdir()
    source.write_text("kernel", encoding="utf-8")
    assert aiter_cache.activate_aiter_cache_for_sources([str(source)]) is not None
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [4321])

    stats = aiter_cache.cleanup_current_aiter_cache()

    assert stats is not None
    assert stats["deleted"] is False
    assert stats["skipped_live_pids"] == [4321]
    assert isolation.cache_root.exists()


def test_cleanup_deletes_only_owned_lock_files(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)
    cpp_lock = isolation.aiter_root_dir / "build" / "pa_ragged" / "lock"
    jit_lock = isolation.aiter_jit_dir / "build" / "lock_module"
    artifact = isolation.aiter_root_dir / "build" / "pa_ragged" / "lib.so"
    cpp_lock.parent.mkdir(parents=True)
    jit_lock.parent.mkdir(parents=True)
    cpp_lock.write_text("", encoding="utf-8")
    jit_lock.write_text("", encoding="utf-8")
    artifact.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [])

    stats = aiter_cache.cleanup_owned_aiter_locks(isolation)

    assert stats["owner_verified"] is True
    assert stats["deleted"] == 2
    assert not cpp_lock.exists()
    assert not jit_lock.exists()
    assert artifact.exists()


def test_cleanup_refuses_when_another_cache_user_is_alive(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)
    lock = isolation.aiter_root_dir / "build" / "pa_ragged" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("", encoding="utf-8")
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [9876])

    stats = aiter_cache.cleanup_owned_aiter_locks(isolation)

    assert stats["deleted"] == 0
    assert stats["skipped_live_pids"] == [9876]
    assert lock.exists()


def test_cleanup_refuses_foreign_owner_marker(tmp_path, monkeypatch):
    isolation = aiter_cache.configure_aiter_cache_isolation(tmp_path)
    owner = json.loads(isolation.owner_file.read_text(encoding="utf-8"))
    owner["owner_pid"] = os.getpid() + 1
    isolation.owner_file.write_text(json.dumps(owner), encoding="utf-8")
    monkeypatch.setattr(aiter_cache, "_live_cache_users", lambda _isolation: [])

    stats = aiter_cache.cleanup_owned_aiter_locks(isolation)

    assert stats["owner_verified"] is False
    assert stats["deleted"] == 0


def test_profiler_droppings_do_not_fail_a_session(tmp_path):
    """rocprofv3 writes into the cwd it is handed; that is not the agent's doing.

    Observed in the archives: `.rocprofv3/<pid>-<pid>-counter_values.dat` and a
    `<pid>_results.db` beside it failed a session outright. The declaration is
    per-path on purpose -- an undeclared stray file is still a violation.
    """
    import subprocess

    from kernelforge.agent_backends.base import AgentRunSpec
    from kernelforge.agent_backends.workspace_guard import (
        WorkspaceGuard,
        WorkspaceSafetyError,
    )

    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "k.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(cmd, cwd=ws, check=True, capture_output=True)

    def run(globs):
        spec = AgentRunSpec(
            system_prompt="",
            user_prompt="",
            cwd=str(ws),
            target_files=[str(ws / "src" / "k.py")],
            ignored_untracked_globs=list(globs),
        )
        guard = WorkspaceGuard(spec, dirty_baseline_default=True)
        guard.prepare()
        (ws / ".rocprofv3").mkdir(exist_ok=True)
        (ws / ".rocprofv3" / "101-102-counter_values.dat").write_text("", encoding="utf-8")
        (ws / "101_results.db").write_text("", encoding="utf-8")
        return guard

    run([".rocprofv3/*", "*_results.db"]).verify()  # declared -> passes

    subprocess.run(["git", "clean", "-fdq"], cwd=ws, capture_output=True)
    with pytest.raises(WorkspaceSafetyError, match="new non-ignored files"):
        run([]).verify()  # undeclared -> still refused


def test_profiler_droppings_are_forgiven_below_the_git_toplevel(tmp_path):
    """The guard reports paths from the git toplevel; the profiler runs deeper.

    ``run_cwd`` is the kernel file's parent, not the workspace root (see
    ``orchestrator/agent.py``), and only a backend declaring
    ``requires_workspace_cwd`` moves it up. So the real observed droppings are
    nested -- ``aiter/ops/triton/.rocprofv3/...`` and ``<hash>/<pid>_results.db``
    -- and a pattern anchored at the root misses every one of them. ``fnmatch``
    crosses "/", so ``*_results.db`` reaches any depth on its own; ``.rocprofv3/``
    does not and needs the second spelling.
    """
    import subprocess

    from kernelforge.agent_backends.base import AgentRunSpec
    from kernelforge.agent_backends.workspace_guard import WorkspaceGuard

    ws = tmp_path / "nested"
    nested = ws / "aiter" / "ops" / "triton"
    nested.mkdir(parents=True)
    (nested / "k.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(cmd, cwd=ws, check=True, capture_output=True)

    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(nested),
        target_files=[str(nested / "k.py")],
        ignored_untracked_globs=[
            ".rocprofv3/*",
            "*/.rocprofv3/*",
            "*_results.db",
        ],
    )
    guard = WorkspaceGuard(spec, dirty_baseline_default=True)
    guard.prepare()
    (nested / ".rocprofv3").mkdir()
    (nested / ".rocprofv3" / "101-102-counter_values.dat").write_text("", encoding="utf-8")
    # The observed layout: a hash-named directory holding <pid>_results.db.
    hashed = nested / "4f1679f7dae9"
    hashed.mkdir()
    (hashed / "4135_results.db").write_text("", encoding="utf-8")

    assert guard.verify() == []


def test_an_undeclared_stray_file_is_still_refused(tmp_path):
    """The allowance is per-path, not a blanket one."""
    import subprocess

    from kernelforge.agent_backends.base import AgentRunSpec
    from kernelforge.agent_backends.workspace_guard import (
        WorkspaceGuard,
        WorkspaceSafetyError,
    )

    ws = tmp_path / "ws2"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "k.py").write_text("x = 1\n", encoding="utf-8")
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(cmd, cwd=ws, check=True, capture_output=True)

    spec = AgentRunSpec(
        system_prompt="",
        user_prompt="",
        cwd=str(ws),
        target_files=[str(ws / "src" / "k.py")],
        ignored_untracked_globs=[".rocprofv3/*"],
    )
    guard = WorkspaceGuard(spec, dirty_baseline_default=True)
    guard.prepare()
    (ws / "src" / "the_agent_left_this.py").write_text("y = 2\n", encoding="utf-8")
    with pytest.raises(WorkspaceSafetyError, match="the_agent_left_this"):
        guard.verify()
