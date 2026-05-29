"""Tests for the dynamic_action sub-agent tool whitelist."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.dynamic_action_tools import (
    ALL_DYNAMIC_TOOLS,
    BENCH_REGISTRY,
    MAX_BENCH_WALL_CLOCK_SEC,
    MAX_READ_SOURCE_CHARS,
    SESSION_ARTIFACT_ALLOWED_PREFIXES,
    SESSION_ARTIFACT_DENY_SEGMENTS,
    apply_patch_in_worktree,
    read_session_artifact,
    read_source,
    reset_worktree,
    run_bench,
)


# ===========================================================================
# Surface invariants
# ===========================================================================
def test_tool_set_is_exactly_four_in_v1():
    """The live surface is 3 resources + 1 terminal signal; ``run_bench``
    is gated off by ``BENCH_TOOL_ENABLED_V1=False``."""
    assert ALL_DYNAMIC_TOOLS == frozenset({
        "read_source",
        "read_session_artifact",
        "apply_patch_in_worktree",
        "emit_proposal",
    })


def test_bench_registry_caps_within_global_ceiling():
    """Every registered bench must respect ``MAX_BENCH_WALL_CLOCK_SEC``;
    vacuously true while the registry is empty."""
    for spec in BENCH_REGISTRY.values():
        assert spec.wall_clock_sec <= MAX_BENCH_WALL_CLOCK_SEC, (
            f"{spec.bench_id}: per-bench wall_clock_sec exceeds the "
            f"§4.1.c global ceiling"
        )


def test_bench_registry_disabled_in_v1():
    """``BENCH_TOOL_ENABLED_V1=False`` ⇒ registry empty and
    ``TOOL_RUN_BENCH`` absent from the live tool surface."""
    from inference_optimizer.orchestrator.dynamic_action_tools import (
        ALL_DYNAMIC_TOOLS,
        BENCH_TOOL_ENABLED_V1,
        TOOL_RUN_BENCH,
    )
    assert BENCH_TOOL_ENABLED_V1 is False
    assert BENCH_REGISTRY == {}
    assert TOOL_RUN_BENCH not in ALL_DYNAMIC_TOOLS


# ===========================================================================
# read_source — denials
# ===========================================================================
@pytest.mark.parametrize("path,expected", [
    ("",                            "path_required"),
    ("relative/path",               "path_must_be_absolute"),
    ("/etc/passwd",                 "path_outside_framework_source_roots"),
    ("/sgl-workspace/foo/*",        "globbing_not_allowed"),
])
def test_read_source_denials(path: str, expected: str):
    result = read_source(path)
    assert result["ok"] is False
    assert result["reason"] == expected


def test_read_source_truncates_large_files(tmp_path: Path, monkeypatch):
    big = tmp_path / "f.py"
    big.write_text("a" * (MAX_READ_SOURCE_CHARS + 1000), encoding="utf-8")
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(tmp_path),
    )
    res = read_source(str(big))
    assert res["ok"] is True
    assert res["truncated"] is True
    assert res["bytes_returned"] == MAX_READ_SOURCE_CHARS


def test_read_source_honours_max_bytes(tmp_path: Path, monkeypatch):
    f = tmp_path / "f.py"
    f.write_text("b" * 5000, encoding="utf-8")
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(tmp_path),
    )
    res = read_source(str(f), 1000)
    assert res["ok"] is True
    assert res["truncated"] is True
    assert res["bytes_returned"] == 1000


def test_read_source_max_bytes_clamped_to_hard_cap(tmp_path: Path, monkeypatch):
    f = tmp_path / "f.py"
    f.write_text("c" * (MAX_READ_SOURCE_CHARS + 500), encoding="utf-8")
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(tmp_path),
    )
    # A caller asking for more than the ceiling is clamped down.
    res = read_source(str(f), MAX_READ_SOURCE_CHARS * 10)
    assert res["bytes_returned"] == MAX_READ_SOURCE_CHARS


def test_read_source_max_bytes_none_uses_hard_cap(tmp_path: Path, monkeypatch):
    f = tmp_path / "f.py"
    f.write_text("d" * (MAX_READ_SOURCE_CHARS + 500), encoding="utf-8")
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", str(tmp_path),
    )
    res = read_source(str(f), None)
    assert res["bytes_returned"] == MAX_READ_SOURCE_CHARS


# ===========================================================================
# read_session_artifact — whitelist + cross-dyn_id isolation
# ===========================================================================
def test_read_session_artifact_rejects_absolute(tmp_path: Path):
    res = read_session_artifact(tmp_path, "/absolute", dyn_id="dyn-0-1")
    assert res["reason"] == "path_must_be_session_relative"


def test_read_session_artifact_rejects_unallowed_prefix(tmp_path: Path):
    res = read_session_artifact(
        tmp_path, "secrets/something", dyn_id="dyn-0-1",
    )
    assert res["reason"] == "path_not_in_allowed_prefixes"


def test_read_session_artifact_rejects_blacklisted_segments(tmp_path: Path):
    res = read_session_artifact(
        tmp_path, "runs/dynamic/inbox.jsonl", dyn_id="dyn-0-1",
    )
    assert res["reason"] == "path_in_deny_list"


def test_read_session_artifact_cross_dyn_id_isolation(tmp_path: Path):
    """A read addressed at another dyn_id's dispatch dir must fail
    even when the prefix matches."""
    other = (
        tmp_path / "agents/orchestration/dynamic_actions/dyn-9-9/spec.json"
    )
    other.parent.mkdir(parents=True)
    other.write_text("{}", encoding="utf-8")
    res = read_session_artifact(
        tmp_path,
        "agents/orchestration/dynamic_actions/dyn-9-9/spec.json",
        dyn_id="dyn-0-1",
    )
    assert res["reason"] == "cross_dyn_id_isolation"


def test_read_session_artifact_happy_path(tmp_path: Path):
    own = (
        tmp_path / "agents/orchestration/dynamic_actions/dyn-0-1/seed_kit.json"
    )
    own.parent.mkdir(parents=True)
    own.write_text('{"motivation_gap_text": "m"}', encoding="utf-8")
    res = read_session_artifact(
        tmp_path,
        "agents/orchestration/dynamic_actions/dyn-0-1/seed_kit.json",
        dyn_id="dyn-0-1",
    )
    assert res["ok"] is True
    assert res["truncated"] is False
    assert "motivation_gap_text" in res["content"]


def test_read_session_artifact_rejects_traversal(tmp_path: Path):
    res = read_session_artifact(
        tmp_path,
        "runs/dynamic/../secrets.txt",
        dyn_id="dyn-0-1",
    )
    assert res["reason"] == "path_traversal_denied"


# ===========================================================================
# run_bench — disabled by default
# ===========================================================================
@pytest.mark.asyncio
async def test_run_bench_disabled_in_v1(tmp_path: Path):
    """Every ``run_bench`` call returns ``bench_tool_disabled_v1``
    regardless of bench_id or worktree state."""
    res = await run_bench("anything", worktree=tmp_path, call_id="c1")
    assert res["ok"] is False
    assert res["reason"] == "bench_tool_disabled_v1"


@pytest.mark.asyncio
async def test_run_bench_disabled_with_unknown_id_too(tmp_path: Path):
    """The disabled gate fires before ``unknown_bench_id`` so the
    surface stays uniform — sub-agent never sees per-bench reasons."""
    res = await run_bench("not_a_bench", worktree=tmp_path, call_id="c1")
    assert res["reason"] == "bench_tool_disabled_v1"


@pytest.mark.asyncio
async def test_run_bench_re_enabled_path_executes_script(
    tmp_path: Path, monkeypatch,
):
    """When ``BENCH_TOOL_ENABLED_V1`` is flipped and a bench is
    registered, the subprocess path still works."""
    import inference_optimizer.orchestrator.dynamic_action_tools as tools_mod
    fake_dir = tmp_path / "benches"
    fake_dir.mkdir()
    script = fake_dir / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\necho probe ok\n", encoding="utf-8",
    )
    script.chmod(0o755)
    spec = tools_mod.BenchSpec(
        bench_id="probe",
        description="forward-guard probe",
        wall_clock_sec=5.0,
        script_path="probe.sh",
    )
    monkeypatch.setattr(tools_mod, "BENCH_TOOL_ENABLED_V1", True)
    monkeypatch.setitem(BENCH_REGISTRY, "probe", spec)
    res = await run_bench(
        "probe", worktree=tmp_path, call_id="c1", bench_dir_root=fake_dir,
    )
    assert res["ok"] is True
    assert res["exit_code"] == 0


# ===========================================================================
# apply_patch_in_worktree — git apply self-check
# ===========================================================================
def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "t"],
        check=True,
    )
    (tmp_path / "x.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "x.txt"], check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"],
        check=True,
    )
    return tmp_path


def test_apply_patch_rejects_out_of_tree(tmp_path: Path):
    repo = _init_repo(tmp_path)
    patch = (
        "--- a/../escape.txt\n"
        "+++ b/../escape.txt\n"
        "@@ -1 +1 @@\n"
        "-x\n+y\n"
    )
    res = apply_patch_in_worktree(repo, patch)
    assert res["ok"] is False
    assert res["reason"] == "patch_path_escapes_worktree"


def test_apply_patch_rejects_empty(tmp_path: Path):
    res = apply_patch_in_worktree(tmp_path, "")
    assert res["reason"] == "empty_patch"


def test_apply_patch_happy_path(tmp_path: Path):
    repo = _init_repo(tmp_path)
    patch = (
        "--- a/x.txt\n"
        "+++ b/x.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n+new\n"
    )
    res = apply_patch_in_worktree(repo, patch)
    assert res["ok"] is True
    assert (repo / "x.txt").read_text() == "new\n"
    reset_worktree(repo)
    assert (repo / "x.txt").read_text() == "old\n"
