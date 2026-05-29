"""Tests for ``orchestrator.action_executors.benchmark_result``.

Combines three previously-separate suites:

* **Unit helpers** (``test_benchmark_result_units``) — ``_to_float`` /
  ``_to_int`` / ``_first_*`` / ``_load_json`` / ``_candidate_raw_jsons``
  / ``_rescue_candidate_paths`` env handling.
* **Rescue end-to-end** (``test_benchmark_result_rescue``) — the
  ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` second-chance salvage path
  surrounding :func:`extract_benchmark_measurement`, including the
  ``copy-into-workspace`` materialisation and fallback when the copy
  fails.
* **Harvest pass** (``test_harvest_leaked_artifacts``) — the broader
  ``harvest_leaked_artifacts`` pass that copies wrapper-side
  diagnostics (``server.log`` / ``gpu_metrics.csv`` / profile traces /
  rescue results) from ``$INFERENCE_OPTIMIZER_LEAK_ROOTS`` into the
  per-task workspace so the NFS clone is self-contained.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import benchmark_result as br
from inference_optimizer.orchestrator.action_executors.benchmark_result import (
    extract_benchmark_measurement,
    harvest_leaked_artifacts,
)


# ===========================================================================
# Unit helpers (formerly test_benchmark_result_units.py)
# ===========================================================================


# ---------------------------------------------------------------------------
# scalar coercion helpers
# ---------------------------------------------------------------------------

class TestScalarHelpers:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("3.14", 3.14),
            (42, 42.0),
            (None, None),
            (True, None),     # booleans are skipped intentionally
            ("nope", None),
            ([], None),
        ],
    )
    def test_to_float(self, value, expected):
        out = br._to_float(value)
        if expected is None:
            assert out is None
        else:
            assert out == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("7", 7),
            (3.9, 3),
            (None, None),
            (True, None),
            ("oops", None),
        ],
    )
    def test_to_int(self, value, expected):
        assert br._to_int(value) == expected

    def test_first_float_returns_first_valid(self):
        assert br._first_float(None, "x", "1.5", 9.0) == 1.5

    def test_first_int_returns_first_valid(self):
        assert br._first_int(None, "bad", "3", 5) == 3

    def test_first_helpers_return_none_when_all_invalid(self):
        assert br._first_float(None, "x") is None
        assert br._first_int(True, None) is None


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_returns_dict_on_valid_json(self, tmp_path):
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"a": 1}))
        assert br._load_json(path) == {"a": 1}

    def test_returns_none_on_missing_file(self, tmp_path):
        assert br._load_json(tmp_path / "ghost.json") is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{nope")
        assert br._load_json(path) is None

    def test_returns_none_when_not_dict(self, tmp_path):
        path = tmp_path / "lst.json"
        path.write_text(json.dumps([1, 2, 3]))
        assert br._load_json(path) is None


# ---------------------------------------------------------------------------
# _candidate_raw_jsons ordering
# ---------------------------------------------------------------------------

class TestCandidateRawJsons:
    def test_orders_non_profile_first(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "inferencex_result.json").write_text("{}")
        (ws / "profile_result.json").write_text("{}")
        (ws / "benchmark_report.json").write_text("{}")
        out = br._candidate_raw_jsons(ws)
        names = [p.name for p in out]
        # benchmark_report.json filtered out, non-profile sorted first.
        assert names[0] == "inferencex_result.json"
        assert names[1] == "profile_result.json"
        assert "benchmark_report.json" not in names

    def test_returns_empty_when_no_json_files(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        assert br._candidate_raw_jsons(ws) == []


# ---------------------------------------------------------------------------
# _rescue_candidate_paths — env handling + workspace filter
# ---------------------------------------------------------------------------

class TestRescueCandidatePaths:
    def test_no_env_no_default_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", raising=False)
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        assert br._rescue_candidate_paths(ws) == []

    def test_explicit_file_included(self, tmp_path, monkeypatch):
        leak = tmp_path / "leak" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
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
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
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
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        out = br._rescue_candidate_paths(ws)
        assert nested.resolve() not in [p.resolve() for p in out]

    def test_mtime_gate_drops_stale_leak(self, tmp_path, monkeypatch):
        leak = tmp_path / "old" / "inferencex_result.json"
        leak.parent.mkdir()
        leak.write_text("{}")
        # Force leak mtime way in the past.
        old = leak.stat().st_mtime - 3600.0
        os.utime(leak, (old, old))
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak))
        monkeypatch.setattr(br, "_DEFAULT_RESCUE_PATHS", ())
        ws = tmp_path / "ws"
        ws.mkdir()
        out = br._rescue_candidate_paths(
            ws, subprocess_started_unix=leak.stat().st_mtime + 60.0,
        )
        assert out == []


# ===========================================================================
# Rescue end-to-end (formerly test_benchmark_result_rescue.py)
#
# The Magpie ``dsr1_fp8_mi300x.sh`` script hardcodes
# ``--result-dir /workspace/`` so a benchmark that *numerically* succeeds
# can still leave the per-task workspace empty (no ``inferencex_result.json``).
# The optimizer's second-chance salvage:
#
# * honours ``$INFERENCE_OPTIMIZER_RESCUE_PATHS`` (files or dirs).
# * gates by ``subprocess_started_unix`` mtime so stale leaks from a
#   previous run cannot be misattributed to this run.
# * tags the adopted path in the ``nonfatal_warnings`` list as
#   ``rescued_from_leaked_path:<path>``.
# * COPIES the leaked file into the task workspace (best-effort) so the
#   canonical NFS-clone of ``<session>/runs/<action>/<task_id>/`` is
#   self-contained and ``raw_result_path`` points at the in-workspace
#   copy rather than the leak location.
# ===========================================================================


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
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_path = leak_dir / "inferencex_result.json"
    _write_inferencex(leak_path)
    # mtime cutoff strictly before the file's mtime
    cutoff = leak_path.stat().st_mtime - 5.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_dir))

    measurement = extract_benchmark_measurement(
        report={"success": False},  # wrapper reported failure
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(1761.6)
    assert measurement["completed_requests"] == 640
    assert any(
        w.startswith("rescued_from_leaked_path:") for w in measurement["nonfatal_warnings"]
    )
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)
    assert json.loads(copied.read_text())["output_throughput"] == 1761.6
    assert leak_path.exists()
    assert any(
        w == f"rescued_from_leaked_path:{leak_path}"
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_rejects_stale_leak_with_older_mtime(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    # subprocess "started" well *after* the leak file's mtime → stale
    cutoff = leak_path.stat().st_mtime + 10.0
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=cutoff,
    )
    assert measurement["valid_measurement"] is False
    assert measurement.get("output_throughput") is None
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


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
    # in-workspace value wins; rescue path should not have been used.
    assert measurement["valid_measurement"] is True
    assert measurement["output_throughput"] == pytest.approx(2000.0)
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_no_candidates_keeps_invalid_measurement(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # explicit empty env to avoid the default /workspace/ path from picking up
    # something under the test sandbox.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(tmp_path / "nope"))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        subprocess_started_unix=0.0,
    )
    assert measurement["valid_measurement"] is False
    assert not any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_rescue_directory_scanned_for_inferencex_result_glob(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    # Two leaked files; the helper sorts via Path.glob() then mtime-gates.
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
    assert any(
        w.startswith("rescued_from_leaked_path:")
        for w in measurement["nonfatal_warnings"]
    )


def test_subprocess_started_unix_none_disables_mtime_gate(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    measurement = extract_benchmark_measurement(
        report={"success": False},
        workspace=workspace,
        # No subprocess_started_unix → mtime gate disabled → adopt.
    )
    assert measurement["valid_measurement"] is True
    copied = workspace / leak_path.name
    assert copied.exists()
    assert measurement["raw_result_path"] == str(copied)


def test_rescue_copies_leaked_file_into_workspace(tmp_path, monkeypatch):
    """Salvage must materialise the rescued file inside the workspace so
    downstream NFS clones of ``<session>/runs/<action>/<task_id>/`` carry
    the canonical artifact instead of a path pointing at ``/workspace/``.
    """
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
    assert any(
        w == f"rescued_from_leaked_path:{leak_path}"
        for w in measurement["nonfatal_warnings"]
    )
    # The leak file itself is not touched (we copy, never move/delete).
    assert leak_path.exists()


def test_rescue_copy_failure_falls_back_to_leak_path(
    tmp_path, monkeypatch,
):
    """If the copy step fails, salvage must still advertise the leaked
    measurement so a usable baseline isn't discarded just because the
    artifact couldn't be materialised into the workspace.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    leak_path = tmp_path / "leak" / "inferencex_result.json"
    _write_inferencex(leak_path, tput=999.9, completed=42)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RESCUE_PATHS", str(leak_path))

    # Force the copy helper to fail so we exercise the fallback branch.
    monkeypatch.setattr(
        br, "_materialize_rescue_into_workspace", lambda *_a, **_k: None,
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
    assert any(
        w.startswith("rescued_copy_into_workspace_failed:")
        for w in warnings
    )


# ===========================================================================
# Harvest pass (formerly test_harvest_leaked_artifacts.py)
#
# Magpie's shell wrappers hardcode multiple output destinations under
# ``/workspace/`` (``server.log`` / ``gpu_metrics.csv`` /
# ``profile_*.trace.json.gz`` / ``inferencex_result*.json``). Without
# harvesting these into the per-task workspace the NFS clone of
# ``<session>/runs/<action>/<task_id>/`` is missing wrapper-side
# diagnostics even when the run succeeded numerically.
# ===========================================================================


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

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,  # gate disabled
        leak_root=leak_root,
    )

    by_name = {src.name: dst for src, dst in harvested}
    assert {"server.log", "gpu_metrics.csv", "profile_run.trace.json.gz",
            "inferencex_result.json"} <= set(by_name)
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

    cutoff = stale.stat().st_mtime + 1.0
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
        "INFERENCE_OPTIMIZER_LEAK_ROOTS", f"{root_a}:{root_b}",
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
