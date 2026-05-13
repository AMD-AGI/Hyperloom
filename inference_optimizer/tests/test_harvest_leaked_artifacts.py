"""Tests for the leak-artifact harvest pass in :mod:`benchmark_result`.

Magpie's shell wrappers hardcode multiple output destinations under
``/workspace/`` (``server.log`` / ``gpu_metrics.csv`` / ``profile_
*.trace.json.gz`` / ``inferencex_result*.json``). Without harvesting
these into the per-task workspace the NFS clone of
``<session>/runs/<action>/<task_id>/`` is missing wrapper-side
diagnostics even when the run succeeded numerically. This module
exercises:

* every default leak glob is picked up,
* mtime gating against ``subprocess_started_unix`` rejects stale
  leaks from previous runs,
* the source file is preserved (copy, never move),
* harvested files land in the destination with the original basename,
* the helper is robust to missing leak roots and read-only sources
  (degrades to an empty result without raising).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.benchmark_result import (
    harvest_leaked_artifacts,
)


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
    # Server log content is preserved verbatim.
    assert (destination / "server.log").read_text() == "init OK\nstarted serving"


def test_harvest_mtime_gating_skips_stale_leaks(tmp_path):
    """Files older than ``subprocess_started_unix`` are rejected."""
    leak_root = tmp_path / "workspace"
    destination = tmp_path / "task"
    destination.mkdir()

    # Two leak files: one stale (from a prior run), one fresh.
    stale = _touch(
        leak_root / "server.log",
        "stale prior-run output",
        mtime=time.time() - 3600.0,
    )
    fresh = _touch(
        leak_root / "gpu_metrics.csv",
        "fresh post-launch csv",
    )

    cutoff = stale.stat().st_mtime + 1.0  # strictly after stale, before fresh
    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=cutoff,
        leak_root=leak_root,
    )

    names = {src.name for src, _ in harvested}
    assert "gpu_metrics.csv" in names
    assert "server.log" not in names
    assert not (destination / "server.log").exists()
    # Fresh file is in the destination (and source is untouched).
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
    destination = leak_root / "task"  # inside leak_root
    destination.mkdir(parents=True)

    # File lives under the destination already; no harvest needed.
    in_ws = _touch(destination / "server.log", "already in place")
    # And one truly leaked file outside the destination.
    leaked = _touch(leak_root / "gpu_metrics.csv", "real leak")

    harvested = harvest_leaked_artifacts(
        destination,
        subprocess_started_unix=None,
        leak_root=leak_root,
    )
    by_name = {src.name: src for src, _ in harvested}
    assert "gpu_metrics.csv" in by_name
    assert by_name["gpu_metrics.csv"] == leaked
    # The in-workspace file is NOT in the harvested list — it was not
    # outside the destination to begin with.
    assert "server.log" not in by_name
    # And the in-workspace content was not clobbered.
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

    # No explicit leak_root → reads env, scans both.
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
