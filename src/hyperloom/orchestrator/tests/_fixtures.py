# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the orchestrator and inference_optimizer test packages.

Defined here rather than in a common ancestor ``conftest.py``, whose scope would
be every test package in the repo. Each consuming ``conftest.py`` imports them,
which is what registers them with pytest.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """Drop the session-dir pin and point MULTI_NODE_STATE_FILE at a missing sentinel so tests run single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


class NoLaunchBackendInstalled(BaseException):
    """A test launched a subprocess before installing a launch backend.

    A ``BaseException`` on purpose. Several production launch sites sit inside
    a bare ``except Exception`` -- the Magpie interpreter probe answers "this
    interpreter cannot import Magpie" that way -- so an ordinary exception here
    would be swallowed and the test would carry on against a wrong answer
    instead of stopping.
    """


@pytest.fixture
def launch_backend(monkeypatch):
    """Install a scripted stand-in for ``run_with_session_kill``.

    Call it with any object exposing that function's signature as ``run``. Both
    the definition and the names the eager importers bound are patched, so a
    launch made on a worker thread the test never sees is covered too.
    """
    from hyperloom.orchestrator.actions.executors import _grid_runner, _subprocess_kill, baseline

    installed: list = []

    def _run(cmd, **kwargs):
        if not installed:
            raise NoLaunchBackendInstalled(f"no launch backend is installed for this test; cmd={list(cmd)[:3]}")
        return installed[-1].run(cmd, **kwargs)

    for module in (_subprocess_kill, _grid_runner, baseline):
        monkeypatch.setattr(module, "run_with_session_kill", _run)

    def _install(backend):
        installed.append(backend)
        return backend

    return _install


@pytest.fixture
def virtual_clock():
    """A :class:`VirtualClock` for tests whose subject is a deadline.

    :class:`ProgressCadence` measures one path's reporting gaps and blocks for
    real (scaled) time so a heartbeat driver gets to run; that is the right
    instrument for cadence and the wrong one for a multi-tick round, where
    nothing may block at all and the readings production takes have to be the
    clock's. This one is that clock, and can be handed to ``ProgressCadence``
    so a test using both keeps one timeline.
    """
    from hyperloom.orchestrator.rehearsal import VirtualClock

    return VirtualClock()
