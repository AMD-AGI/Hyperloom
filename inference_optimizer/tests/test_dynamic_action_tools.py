"""dynamic_action.MD P3 §4 — tool whitelist invariants."""

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
def test_tool_set_is_exactly_five():
    """P3 §4 — surface is 4 resources + 1 terminal signal. Any drift
    must be a design change (registry constants are the canonical
    source of truth)."""
    assert ALL_DYNAMIC_TOOLS == frozenset({
        "read_source",
        "read_session_artifact",
        "run_bench",
        "apply_patch_in_worktree",
        "emit_proposal",
    })


def test_bench_registry_caps_within_global_ceiling():
    for spec in BENCH_REGISTRY.values():
        assert spec.wall_clock_sec <= MAX_BENCH_WALL_CLOCK_SEC, (
            f"{spec.bench_id}: per-bench wall_clock_sec exceeds the "
            f"§4.1.c global ceiling"
        )


def test_bench_registry_has_required_starter_set():
    expected = {
        "kernel_attention_timing",
        "kernel_gemm_timing",
        "kernel_kvcache_layout",
        "inference_short_prompt",
    }
    assert set(BENCH_REGISTRY) == expected


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
# run_bench — unknown id + timeout
# ===========================================================================
@pytest.mark.asyncio
async def test_run_bench_unknown_id(tmp_path: Path):
    res = await run_bench("not_a_bench", worktree=tmp_path, call_id="c1")
    assert res["ok"] is False
    assert res["reason"] == "unknown_bench_id"
    assert "kernel_attention_timing" in res["allowed"]


@pytest.mark.asyncio
async def test_run_bench_executes_placeholder(tmp_path: Path):
    """The shipped placeholder bench writes a marker JSON; ``run_bench``
    surfaces exit_code 0 and the path to the scratch dir (which is
    NOT recovered)."""
    res = await run_bench(
        "kernel_attention_timing", worktree=tmp_path, call_id="c1",
    )
    assert res["ok"] is True, res
    assert res["exit_code"] == 0
    assert "kernel_attention_timing" in res["stdout_tail"]
    marker = Path(res["output_dir"]) / "result.json"
    assert marker.is_file()


@pytest.mark.asyncio
async def test_run_bench_times_out(tmp_path: Path, monkeypatch):
    """Force a bench that sleeps past its cap → ``timed_out`` reason."""
    fake_dir = tmp_path / "benches"
    fake_dir.mkdir()
    slow_script = fake_dir / "kernel_attention_timing.sh"
    slow_script.write_text(
        "#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8",
    )
    slow_script.chmod(0o755)
    monkeypatch.setitem(
        BENCH_REGISTRY,
        "kernel_attention_timing",
        type(BENCH_REGISTRY["kernel_attention_timing"])(
            bench_id="kernel_attention_timing",
            description="slow",
            wall_clock_sec=1.0,
            script_path="kernel_attention_timing.sh",
        ),
    )
    res = await run_bench(
        "kernel_attention_timing",
        worktree=tmp_path,
        call_id="c1",
        bench_dir_root=fake_dir,
    )
    assert res["ok"] is False
    assert res["reason"] == "timed_out"


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
