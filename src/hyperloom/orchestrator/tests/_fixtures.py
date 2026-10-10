# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures shared by the orchestrator and inference_optimizer test packages.

Defined here rather than in a common ancestor ``conftest.py``, whose scope would
be every test package in the repo. Each consuming ``conftest.py`` imports them,
which is what registers them with pytest.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """Isolate a test from the host: no session-dir pin, single-node, no live GPU power sampling, and a sandbox aiter.

    Sampling follows ``amd-smi``, so on a GPU host every watchdog-driven test would otherwise query the real cards.
    Tests of the sampler turn it back on themselves.

    The serving-.so preflight and the kernel lane's cache invalidation move aiter's compiled modules aside, so inside
    an image that ships aiter they would strip the host install. Every discovery route (``find_spec``, the env
    overrides, the probe paths) is pointed at a sandbox; tests that model an install plant their own on top of it.
    """
    from hyperloom.orchestrator.actions.executors import _aiter_jit

    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "0")
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)

    aiter_sandbox = tmp_path_factory.mktemp("aiter_sandbox")
    (aiter_sandbox / "aiter" / "jit").mkdir(parents=True)
    (aiter_sandbox / "aiter" / "__init__.py").write_text(
        "raise ImportError('tests resolve aiter to a sandbox and must not import it')\n", encoding="utf-8"
    )
    monkeypatch.delitem(sys.modules, "aiter", raising=False)
    monkeypatch.syspath_prepend(str(aiter_sandbox))
    for name in ("AITER_JIT_DIR", "INFERENCE_OPTIMIZER_AITER_JIT_DIR", "VLLM_VENV_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_aiter_jit, "AITER_JIT_PROBE_PATHS", ())


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
