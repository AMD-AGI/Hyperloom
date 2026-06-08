# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for issue #432: GEAK Ray dispatch must auto-recover
from a *foreign* Ray cluster started under a different Python/Ray than the
submitter (the bug: cluster py3.10 vs submitter py3.12). Previously
``ray.init`` raised a "Version mismatch" RuntimeError in ~0.8s and the whole
GEAK attempt was recorded as a "compile failed" REVERT despite valid kernel
candidates.

``quiet_ray_init`` must now:
  1. detect the "Version mismatch" RuntimeError,
  2. tear the foreign cluster down + start a fresh LOCAL head under this
     interpreter (``force_restart_local_cluster`` -> ray stop/start), and
  3. retry ``ray.init`` exactly once.

A non-version-mismatch error must propagate unchanged (no restart, no retry).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

TOOLS_DIR = Path(__file__).resolve().parent
BACKENDS_DIR = TOOLS_DIR / "backends"
for d in (str(TOOLS_DIR), str(BACKENDS_DIR)):
    if d not in sys.path:
        sys.path.insert(0, d)

import ray_runtime  # noqa: E402


_VERSION_MISMATCH_MSG = (
    "Version mismatch: The cluster was started with:\n"
    "    Ray: 2.44.1\n"
    "    Python: 3.10.12\n"
    "This process on node 10.170.172.63 was started with:\n"
    "    Ray: 2.44.1\n"
    "    Python: 3.12.13\n"
)


def _make_fake_ray(init_side_effects):
    """Build a stand-in ``ray`` module whose ``init`` pops side effects in
    order: a ``BaseException`` instance is raised, ``None`` succeeds."""
    fake = types.ModuleType("ray")
    calls = {"init": 0, "shutdown": 0}
    effects = list(init_side_effects)

    def _init(*args, **kwargs):
        calls["init"] += 1
        eff = effects.pop(0)
        if isinstance(eff, BaseException):
            raise eff
        return None

    def _shutdown(*args, **kwargs):
        calls["shutdown"] += 1

    fake.init = _init
    fake.shutdown = _shutdown
    fake._calls = calls
    return fake


def test_is_version_mismatch_detects_banner():
    assert ray_runtime._is_ray_version_mismatch(_VERSION_MISMATCH_MSG)
    assert ray_runtime._is_ray_version_mismatch("ray Version Mismatch: foo")
    assert not ray_runtime._is_ray_version_mismatch("ConnectionError: GCS down")
    assert not ray_runtime._is_ray_version_mismatch("")
    assert not ray_runtime._is_ray_version_mismatch(None)  # type: ignore[arg-type]


def test_quiet_ray_init_recovers_from_version_mismatch(tmp_path, monkeypatch):
    """First init raises Version mismatch -> restart local cluster -> retry OK."""
    fake_ray = _make_fake_ray([RuntimeError(_VERSION_MISMATCH_MSG), None])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    restart_calls = []

    def _fake_restart(num_gpus=None, log_path=None):
        restart_calls.append({"num_gpus": num_gpus, "log_path": log_path})

    monkeypatch.setattr(ray_runtime, "force_restart_local_cluster", _fake_restart)

    log_path = tmp_path / "ray_lifecycle.log"
    runtime_env = ray_runtime.quiet_ray_init(num_gpus=2, log_path=log_path)

    assert fake_ray._calls["init"] == 2, "should retry init exactly once after restart"
    assert fake_ray._calls["shutdown"] == 1, "should shutdown stale driver before retry"
    assert len(restart_calls) == 1
    assert restart_calls[0]["num_gpus"] == 2
    assert restart_calls[0]["log_path"] == log_path
    assert "env_vars" in runtime_env


def test_quiet_ray_init_propagates_non_mismatch_error(monkeypatch):
    """A non-version-mismatch failure must NOT trigger a restart; it raises."""
    fake_ray = _make_fake_ray([ConnectionError("GCS handshake failed")])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    restart_calls = []
    monkeypatch.setattr(
        ray_runtime, "force_restart_local_cluster",
        lambda **kw: restart_calls.append(kw),
    )

    with pytest.raises(ConnectionError):
        ray_runtime.quiet_ray_init(num_gpus=1)

    assert fake_ray._calls["init"] == 1, "must not retry on non-mismatch errors"
    assert restart_calls == [], "must not restart cluster on non-mismatch errors"


def test_quiet_ray_init_no_mismatch_succeeds_first_try(monkeypatch):
    """Happy path: init succeeds immediately, no restart, no retry."""
    fake_ray = _make_fake_ray([None])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    restart_calls = []
    monkeypatch.setattr(
        ray_runtime, "force_restart_local_cluster",
        lambda **kw: restart_calls.append(kw),
    )
    ray_runtime.quiet_ray_init()
    assert fake_ray._calls["init"] == 1
    assert restart_calls == []


def test_force_restart_local_cluster_runs_stop_then_start(tmp_path):
    """``force_restart_local_cluster`` must ``ray stop --force`` then
    ``ray start --head`` with the requested num_gpus, logging to the audit
    file."""
    log_path = tmp_path / "ray_lifecycle.log"
    runs = []

    class _Proc:
        returncode = 0

    def _fake_run(cmd, **kwargs):
        runs.append(cmd)
        return _Proc()

    with mock.patch.object(ray_runtime.subprocess, "run", _fake_run):
        ray_runtime.force_restart_local_cluster(num_gpus=4, log_path=log_path)

    assert runs[0] == ["ray", "stop", "--force"]
    assert runs[1][:4] == ["ray", "start", "--head", "--port=6379"]
    assert "--num-gpus=4" in runs[1]
    assert log_path.exists()


def test_force_restart_raises_when_start_fails(tmp_path):
    """A non-zero ``ray start`` exit must raise so ``submit``'s except can
    record it as a backend-dispatch failure."""
    log_path = tmp_path / "ray_lifecycle.log"

    class _Proc:
        returncode = 1

    def _fake_run(cmd, **kwargs):
        return _Proc()

    with mock.patch.object(ray_runtime.subprocess, "run", _fake_run):
        with pytest.raises(RuntimeError, match="restart local Ray"):
            ray_runtime.force_restart_local_cluster(num_gpus=1, log_path=log_path)
