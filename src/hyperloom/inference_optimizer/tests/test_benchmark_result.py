# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``orchestrator.actions.executors.benchmark_result``."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import benchmark_result as br
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
    select_run_workspace,
    snapshot_workspaces,
)


# Unit helpers


# _candidate_raw_jsons ordering
class TestCandidateRawJsons:
    def test_orders_non_profile_first(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "inferencex_result.json").write_text("{}")
        (ws / "profile_result.json").write_text("{}")
        (ws / "benchmark_report.json").write_text("{}")
        out = br._candidate_raw_jsons(ws)
        names = [p.name for p in out]
        assert names[0] == "inferencex_result.json"
        assert names[1] == "profile_result.json"
        assert "benchmark_report.json" not in names

    def test_returns_empty_when_no_json_files(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert br._candidate_raw_jsons(ws) == []


# snapshot_workspaces / select_run_workspace
class TestWorkspaceSelection:
    def test_snapshot_returns_existing_benchmark_dirs(self, tmp_path):
        (tmp_path / "benchmark_vllm_20260101_000000").mkdir()
        (tmp_path / "benchmark_vllm_20260812_120000").mkdir()
        (tmp_path / "other_dir").mkdir()
        snap = snapshot_workspaces(tmp_path)
        names = {p.name for p in snap}
        assert "benchmark_vllm_20260101_000000" in names
        assert "benchmark_vllm_20260812_120000" in names
        assert "other_dir" not in names

    def test_snapshot_empty_on_missing_root(self, tmp_path):
        assert snapshot_workspaces(tmp_path / "nonexistent") == frozenset()

    def test_select_run_workspace_known_before_excludes_stale(self, tmp_path):
        stale = tmp_path / "benchmark_vllm_20260101_000000"
        stale.mkdir()
        known = snapshot_workspaces(tmp_path)
        fresh = tmp_path / "benchmark_vllm_20260812_120000"
        fresh.mkdir()
        result = select_run_workspace(tmp_path, known_before=known)
        assert result is not None
        assert result.name == "benchmark_vllm_20260812_120000"

    def test_select_run_workspace_all_known_returns_none(self, tmp_path):
        (tmp_path / "benchmark_vllm_20260101_000000").mkdir()
        known = snapshot_workspaces(tmp_path)
        result = select_run_workspace(tmp_path, known_before=known)
        assert result is None

    def test_select_run_workspace_no_dirs_returns_none(self, tmp_path):
        assert select_run_workspace(tmp_path, known_before=frozenset()) is None

    def test_select_run_workspace_picks_newest_of_several_fresh(self, tmp_path):
        known = snapshot_workspaces(tmp_path)
        (tmp_path / "benchmark_vllm_20260812_120000").mkdir()
        (tmp_path / "benchmark_vllm_20260812_130000").mkdir()
        result = select_run_workspace(tmp_path, known_before=known)
        assert result is not None
        assert result.name == "benchmark_vllm_20260812_130000"

    def test_select_run_workspace_stale_with_larger_name_not_selected(self, tmp_path):
        stale = tmp_path / "benchmark_vllm_29991231_235959"
        stale.mkdir()
        known = snapshot_workspaces(tmp_path)
        fresh = tmp_path / "benchmark_vllm_20260812_120000"
        fresh.mkdir()
        result = select_run_workspace(tmp_path, known_before=known)
        assert result is not None
        assert result.name == "benchmark_vllm_20260812_120000"


# _rescue_candidate_paths — env handling + workspace filter
class TestRescueCandidatePaths:
    def test_no_env_no_default_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
        ws = tmp_path / "ws"
        ws.mkdir()
        assert br._rescue_candidate_paths(ws) == []

    def test_explicit_file_included(self, tmp_path, monkeypatch):
        leak = tmp_path / "leak" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(ws)
        assert leak.resolve() in [p.resolve() for p in out]

    def test_directory_scanned_for_inferencex_pattern(self, tmp_path, monkeypatch):
        leak_dir = tmp_path / "leak"
        leak_dir.mkdir()
        a = leak_dir / "inferencex_result.json"
        a.write_text("{}")
        b = leak_dir / "inferencex_result_eval.json"
        b.write_text("{}")
        unrelated = leak_dir / "unrelated.json"
        unrelated.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(ws)
        names = {p.name for p in out}
        assert names == {"inferencex_result.json", "inferencex_result_eval.json"}

    def test_paths_inside_workspace_filtered_out(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        nested = ws / "inferencex_result.json"
        nested.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(nested))
        out = br._rescue_candidate_paths(ws)
        assert nested.resolve() not in [p.resolve() for p in out]

    def test_mtime_gate_drops_stale_leak(self, tmp_path, monkeypatch):
        leak = tmp_path / "old" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        old = leak.stat().st_mtime - 3600.0
        os.utime(leak, (old, old))
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(
            ws,
            subprocess_started_unix=leak.stat().st_mtime + 60.0,
        )
        assert out == []


# Rescue end-to-end: salvage adopts mtime-gated leaked results into the workspace.


def _write_inferencex(path: Path, tput: float = 1761.6, completed: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "output_throughput": tput,
                "request_throughput": tput / 10,
                "completed_requests": completed,
                "duration_seconds": 120.0,
            }
        ),
        encoding="utf-8",
    )


def test_rescue_from_env_path_adopted_after_subprocess_start(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_path = leak_dir / "inferencex_result.json"
    _write_inferencex(leak_path)
    cutoff = leak_path.stat().st_mtime - 5.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(1761.6)
    assert measurement["completed_requests"] == 640
    assert any(w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"])
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)
    assert json.loads(copied.read_text())["output_throughput"] == 1761.6
    assert leak_path.exists()
    assert any(w == f"rescued_from_leaked_path:{leak_path}" for w in measurement["nonfatal_warnings"])


def test_rescue_rejects_stale_leak_with_older_mtime(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    cutoff = leak_path.stat().st_mtime + 10.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is False
    assert measurement.get("output_throughput") is None
    assert not any(w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"])


def test_rescue_skips_when_in_workspace_salvage_succeeded(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    in_ws = workspace / "inferencex_result.json"
    _write_inferencex(in_ws, tput=2000.0)
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=9999.0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": True},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 5.0,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(2000.0)
    assert not any(w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"])


def test_rescue_no_candidates_keeps_invalid_measurement(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(tmp_path / "nope"))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=0.0,
    )
    assert measurement["valid_measurement"] is False
    assert not any(w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"])


def test_rescue_directory_scanned_for_inferencex_result_glob(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    a = leak_dir / "inferencex_result_eval.json"
    b = leak_dir / "inferencex_result.json"
    _write_inferencex(a, tput=100.0)
    _write_inferencex(b, tput=200.0)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=min(a.stat().st_mtime, b.stat().st_mtime) - 1.0,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] in (100.0, 200.0)
    assert any(w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"])


def test_subprocess_started_unix_none_disables_mtime_gate(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
    )
    assert measurement["valid_measurement"] is True
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)


def test_rescue_copies_leaked_file_into_workspace(tmp_path, monkeypatch):
    """Salvage must materialise the rescued file inside the workspace."""
    workspace = tmp_path / "session" / "runs" / "baseline" / "task-7" / "bench_run"
    workspace.mkdir(parents=True)
    leak_path = tmp_path / "workspace_leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=1234.5, completed=500)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 2.0,
    )
    copied = workspace / leak_path.name
    assert copied.exists(), (
        "salvage must physically copy the leaked file into the workspace "
        "so the NFS-clone of the task dir contains the InferenceX result"
    )
    body = json.loads(copied.read_text())
    assert body["output_throughput"] == 1234.5
    assert body["completed_requests"] == 500
    assert measurement["raw_result_path"] == str(copied)
    assert any(w == f"rescued_from_leaked_path:{leak_path}" for w in measurement["nonfatal_warnings"])
    assert leak_path.exists()


def test_rescue_copy_failure_falls_back_to_leak_path(
    tmp_path,
    monkeypatch,
):
    """If the copy step fails, salvage must still advertise the leaked measurement."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=999.9, completed=42)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    monkeypatch.setattr(
        br,
        "_materialize_rescue_into_workspace",
        lambda *_a, **_k: None,
    )

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=leak_path.stat().st_mtime - 1.0,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(999.9)
    assert measurement["raw_result_path"] == str(leak_path)
    warnings = measurement["nonfatal_warnings"]
    assert any(w.startswith("rescued_from_leaked_path:") for w in warnings)
    assert any(w.startswith("rescued_copy_into_workspace_failed:") for w in warnings)


# Harvest pass: copy diagnostics leaked under /workspace/ into the per-task workspace.


def _touch(path: Path, content: str = "x", *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_harvest_copies_every_default_glob(tmp_path):
    """Each leak glob in ``_DEFAULT_LEAK_ARTIFACT_GLOBS`` is harvested."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()

    _touch(leak_root / "server.log", "init OK\nstarted serving")
    _touch(leak_root / "gpu_metrics.csv", "timestamp,power\n0,300W")
    _touch(
        leak_root / "profile_run.trace.json.gz",
        '{"trace": "binary-ish"}',
    )
    _touch(
        leak_root / "inferencex_result.json",
        json.dumps({"output_throughput": 1761.6, "completed_requests": 640}),
    )
    _touch(
        leak_root / "results_2026.json",
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.42}}}),
    )

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )

    by_name = {src.name: dst for src, dst in harvested}
    assert {
        "server.log",
        "gpu_metrics.csv",
        "profile_run.trace.json.gz",
        "inferencex_result.json",
        "results_2026.json",
    } <= set(by_name)
    for dst in by_name.values():
        assert dst.parent == destination
        assert dst.exists()
    assert (destination / "server.log").read_text() == "init OK\nstarted serving"


def test_harvest_mtime_gating_skips_stale_leaks(tmp_path):
    """Files older than ``subprocess_started_unix`` are rejected."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()

    stale = _touch(
        leak_root / "server.log",
        "stale prior-run output",
        mtime=time.time() - 3600.0,
    )
    fresh = _touch(
        leak_root / "gpu_metrics.csv",
        "fresh post-launch csv",
    )

    # Cutoff rejects the ~1h-old stale leak while adopting the fresh file.
    cutoff = stale.stat().st_mtime + 60.0
    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=cutoff,
        leak_root=leak_root,
    )

    names = {src.name for src, _ in harvested}
    assert "gpu_metrics.csv" in names
    assert "server.log" not in names
    assert not (destination / "server.log").exists()
    assert (destination / "gpu_metrics.csv").read_text() == "fresh post-launch csv"
    assert fresh.exists()


def test_harvest_preserves_source_files(tmp_path):
    """``shutil.copy2`` semantics: source remains in place after harvest."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()
    source = _touch(leak_root / "server.log", "log body")

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )
    assert harvested
    assert source.exists()
    assert source.read_text() == "log body"


def test_harvest_returns_empty_when_leak_root_missing(tmp_path):
    """Missing leak root degrades to empty list without raising."""
    destination = tmp_path / "task"
    destination.mkdir()
    nonexistent = tmp_path / "does-not-exist"
    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=nonexistent,
    )
    assert harvested == []


def test_harvest_skips_files_already_in_destination(tmp_path):
    """If a leak path resolves inside the destination it's not re-copied."""
    leak_root = tmp_path / "workspace"
    destination = leak_root / "task"
    destination.mkdir(parents=True)

    in_ws = _touch(destination / "server.log", "already in place")
    leaked = _touch(leak_root / "gpu_metrics.csv", "real leak")

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )
    by_name = {src.name: src for src, _ in harvested}
    assert "gpu_metrics.csv" in by_name
    assert by_name["gpu_metrics.csv"] == leaked
    assert "server.log" not in by_name
    assert in_ws.read_text() == "already in place"


def test_harvest_extra_globs_extend_defaults(tmp_path):
    """Callers can register additional leak patterns without recompiling."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()
    _touch(leak_root / "server.log", "default")
    custom_leak = _touch(leak_root / "custom_diagnostic.log", "site-specific")

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
        extra_globs=("custom_diagnostic.log",),
    )
    names = {src.name for src, _ in harvested}
    assert "server.log" in names
    assert "custom_diagnostic.log" in names
    assert (destination / "custom_diagnostic.log").read_text() == "site-specific"
    assert custom_leak.exists()


def test_harvest_multiple_profile_traces_via_glob(tmp_path):
    """The ``profile_*.trace.json.gz`` glob catches per-variant trace names."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()
    _touch(leak_root / "profile_v1.trace.json.gz", "trace-1")
    _touch(leak_root / "profile_v2.trace.json.gz", "trace-2")

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )
    names = {src.name for src, _ in harvested}
    assert {"profile_v1.trace.json.gz", "profile_v2.trace.json.gz"} <= names
    assert (destination / "profile_v1.trace.json.gz").read_text() == "trace-1"
    assert (destination / "profile_v2.trace.json.gz").read_text() == "trace-2"


def test_harvest_reads_env_leak_roots(tmp_path, monkeypatch):
    """``INFERENCE_OPTIMIZER_LEAK_ROOTS`` selects the scan root(s)."""
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    destination = tmp_path / "task"
    destination.mkdir()
    _touch(root_a / "server.log", "from-a")
    _touch(root_b / "gpu_metrics.csv", "from-b")
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_LEAK_ROOTS",
        f"{root_a}:{root_b}",
    )

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
    )
    names = {src.name for src, _ in harvested}
    assert {"server.log", "gpu_metrics.csv"} <= names
    assert (destination / "server.log").read_text() == "from-a"
    assert (destination / "gpu_metrics.csv").read_text() == "from-b"


def test_harvest_explicit_leak_root_overrides_env(tmp_path, monkeypatch):
    """An explicit ``leak_root`` kwarg wins over the env var."""
    env_root = tmp_path / "env_root"
    explicit_root = tmp_path / "explicit_root"
    destination = tmp_path / "task"
    destination.mkdir()
    _touch(env_root / "server.log", "from-env")
    _touch(explicit_root / "server.log", "from-explicit")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(env_root))

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=explicit_root,
    )
    assert len(harvested) == 1
    assert (destination / "server.log").read_text() == "from-explicit"


def test_harvest_default_roots_include_inferencex_and_result_dir(tmp_path, monkeypatch):
    """Without an env override, the default scan roots also cover
    ``$INFERENCEX_PATH`` (eval ``mv ./`` lands in the checkout) and
    ``$RESULT_DIR``, not just ``/workspace``."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", raising=False)
    ix_root = tmp_path / "InferenceX@abc123"
    result_dir = tmp_path / "session" / "runs"
    destination = tmp_path / "task"
    destination.mkdir()
    _touch(ix_root / "inferencex_result.json", '{"output_throughput": 1.0}')
    _touch(result_dir / "gpu_metrics.csv", "from-result-dir")
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_root))
    monkeypatch.setenv("RESULT_DIR", str(result_dir))

    harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=None)
    names = {src.name for src, _ in harvested}
    assert {"inferencex_result.json", "gpu_metrics.csv"} <= names


def test_harvest_salvages_leaked_eval_results_json(tmp_path, monkeypatch):
    """#927: eval ``results*.json`` that leaked to ``$INFERENCEX_PATH`` (via
    ``append_lm_eval_summary``'s ``mv ./``) is harvested back into the session,
    so ``parse_eval_results`` (globs ``**/results*.json``) can still find it
    even when the patcher-based redirect missed."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", raising=False)
    monkeypatch.delenv("RESULT_DIR", raising=False)
    ix_root = tmp_path / "InferenceX@abc123"
    destination = tmp_path / "session" / "runs" / "baseline"
    destination.mkdir(parents=True)
    _touch(
        ix_root / "results_2026-07-17.json",
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.6694}}}),
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_root))

    harvested = harvest_leaked_artifacts(destination, subprocess_started_unix=None)

    by_name = {src.name: dst for src, dst in harvested}
    assert "results_2026-07-17.json" in by_name
    copied = by_name["results_2026-07-17.json"]
    assert copied.parent == destination and copied.exists()
    # The source is never moved (harvest copies).
    assert (ix_root / "results_2026-07-17.json").exists()


def test_rescue_candidate_paths_scan_inferencex_checkout(tmp_path, monkeypatch):
    """A leaked ``inferencex_result.json`` in the InferenceX checkout is a
    rescue candidate via the ``$INFERENCEX_PATH``-derived root."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
    ix_root = tmp_path / "InferenceX@abc123"
    ix_root.mkdir()
    leak = ix_root / "inferencex_result.json"
    leak.write_text("{}")
    monkeypatch.setenv("INFERENCEX_PATH", str(ix_root))
    ws = tmp_path / "ws"
    ws.mkdir()

    out = br._rescue_candidate_paths(ws)
    assert leak.resolve() in [p.resolve() for p in out]


def test_harvest_creates_destination_if_missing(tmp_path):
    """Destination doesn't have to exist before the call."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "newly_created_task" / "deep"
    _touch(leak_root / "server.log", "log body")

    assert not destination.exists()
    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )
    assert destination.is_dir()
    assert (destination / "server.log").read_text() == "log body"
    assert harvested


# Approximate throughput for killed-overtime variants (server.log parsing)
_SGLANG_LOG = """\
[2026-06-12 10:00:00] INFO: server started
Decode batch. #running-req: 64, #token: 1000, token usage: 0.10, gen throughput (token/s): 100.0, #queue-req: 0
Decode batch. #running-req: 64, #token: 2000, token usage: 0.20, gen throughput (token/s): 900.0, #queue-req: 0
Decode batch. #running-req: 64, #token: 3000, token usage: 0.30, gen throughput (token/s): 1000.0, #queue-req: 0
Decode batch. #running-req: 64, #token: 4000, token usage: 0.40, gen throughput (token/s): 1100.0, #queue-req: 0
Decode batch. #running-req: 64, #token: 5000, token usage: 0.50, gen throughput (token/s): 1200.0, #queue-req: 0
"""

_VLLM_LOG = """\
INFO: Started server process
Avg prompt throughput: 5000.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 64 reqs
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 800.0 tokens/s, Running: 64 reqs
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1000.0 tokens/s, Running: 64 reqs
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1200.0 tokens/s, Running: 64 reqs
"""


def test_estimate_from_sglang_server_log(tmp_path):
    """sglang ``gen throughput`` lines parse; warmup-trimmed steady mean returned."""
    log_path = tmp_path / "server.log"
    log_path.write_text(_SGLANG_LOG)
    est = br.estimate_output_throughput_from_server_log(log_path)
    assert est is not None
    # warmup_skip_frac 0.25 drops the 100.0 ramp; mean(900,1000,1100,1200) = 1050.0.
    assert est["output_throughput"] == pytest.approx(1050.0)
    assert est["num_samples"] == 5
    assert est["source_path"] == str(log_path)


def test_estimate_from_vllm_server_log_filters_zero(tmp_path):
    """vllm ``Avg generation throughput`` parses; prefill-only zeros dropped."""
    log_path = tmp_path / "server.log"
    log_path.write_text(_VLLM_LOG)
    est = br.estimate_output_throughput_from_server_log(log_path)
    assert est is not None
    # 3 positive samples (0.0 prefill filtered); mean(800,1000,1200)=1000.0.
    assert est["output_throughput"] == pytest.approx(1000.0)
    assert est["num_samples"] == 3


def test_estimate_returns_none_without_samples(tmp_path):
    """A log with no throughput markers yields no estimate."""
    log_path = tmp_path / "server.log"
    log_path.write_text("INFO: nothing useful here\nWARN: still nothing\n")
    assert br.estimate_output_throughput_from_server_log(log_path) is None


def test_estimate_returns_none_for_missing_file(tmp_path):
    """A non-existent log path is tolerated (returns None, never raises)."""
    assert br.estimate_output_throughput_from_server_log(tmp_path / "absent.log") is None


def test_estimate_killed_variant_prefers_richest_log(tmp_path):
    """``estimate_killed_variant_throughput`` scans the slot recursively and
    uses the largest server.log (the engine's full decode trace)."""
    slot = tmp_path / "variant_00_vA"
    (slot / "benchmark_sglang_x").mkdir(parents=True)
    # A tiny stub plus the full in-slot log; the larger one wins.
    (slot / "server.log").write_text(_SGLANG_LOG)
    (slot / "benchmark_sglang_x" / "server.log").write_text(
        "Decode batch. gen throughput (token/s): 5.0, #queue-req: 0\n"
    )
    est = br.estimate_killed_variant_throughput(slot)
    assert est is not None
    assert est["output_throughput"] == pytest.approx(1050.0)


def test_estimate_killed_variant_none_when_no_logs(tmp_path):
    """No server.log anywhere under the slot -> no estimate."""
    slot = tmp_path / "variant_00_vA"
    slot.mkdir()
    assert br.estimate_killed_variant_throughput(slot) is None


def test_explicitly_invalid_agentx_submission_is_always_rejected(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION", "1")
    # submission_valid=False means the scenario explicitly rejected it; the
    # escape hatch must not exempt this case.
    assert (
        br.is_valid_measurement(
            {
                "output_throughput": 170.0,
                "completed_requests": 10,
                "submission_valid": False,
            }
        )
        is False
    )


def test_unverified_submission_accepted_when_hatch_is_set(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION", "1")
    assert br.is_valid_measurement(
        {
            "output_throughput": 170.0,
            "completed_requests": 10,
            "submission_valid": None,
        }
    )


def test_unverified_submission_rejected_without_hatch(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.delenv("HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION", raising=False)
    assert (
        br.is_valid_measurement(
            {
                "output_throughput": 170.0,
                "completed_requests": 10,
                "submission_valid": None,
            }
        )
        is False
    )
