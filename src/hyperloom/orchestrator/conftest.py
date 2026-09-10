# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for all ``orchestrator`` test directories.

Scoped to ``src/hyperloom/orchestrator/**/tests/`` only — deliberately NOT
placed at ``src/conftest.py`` which would inject these into every test package
in the repo (17 directories).

Hoisted from ``inference_optimizer/tests/conftest.py`` so that test files
relocated out of that directory do not silently lose the session-layout
isolation and the subprocess launch backend.
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

    A ``BaseException`` on purpose: several production launch sites sit inside
    a bare ``except Exception`` so an ordinary exception here would be swallowed
    and the test would carry on against a wrong answer instead of stopping.
    """


@pytest.fixture
def launch_backend(monkeypatch):
    """Install a scripted stand-in for ``run_with_session_kill``.

    Call it with any object exposing that function's signature as ``run``.
    Both the definition and the names eager importers bound are patched, so a
    launch made on a worker thread the test never sees is covered too.
    """
    from hyperloom.orchestrator.actions.executors import _grid_runner, _subprocess_kill, baseline

    installed: list = []

    def _run(cmd, **kwargs):
        if not installed:
            raise NoLaunchBackendInstalled(f"no launch backend installed; cmd={list(cmd)[:3]}")
        return installed[-1].run(cmd, **kwargs)

    for module in (_subprocess_kill, _grid_runner, baseline):
        monkeypatch.setattr(module, "run_with_session_kill", _run)

    def _install(backend):
        installed.append(backend)
        return backend

    return _install


@pytest.fixture
def virtual_clock():
    """A :class:`VirtualClock` for tests whose subject is a deadline."""
    from hyperloom.orchestrator.rehearsal import VirtualClock

    return VirtualClock()
