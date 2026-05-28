"""Pytest hooks and shared helpers for the inference_optimizer tests package.

This module hosts two responsibilities:

1. ``_bootstrap_kernel_agent_env`` (session-wide side effect on import).
   Points ``HYPERLOOM_KERNEL_AGENT_ROOT`` at the in-repo ``kernel-agent``
   checkout. ``kernel_request_handlers`` snapshots this path at import
   time from ``os.environ``; tests that need ``apply_kernel_patch.py`` /
   ``kernel_optimization.py`` must see a valid root before any handler
   module import, so we set the env here (conftest loads before test
   modules).

2. ``seed_target_analysis_marker`` (opt-in helper for individual tests).
   :class:`Coordinator` now hard-gates ``target_analysis`` as TODO 0
   *unconditionally*: when the marker JSON is missing it denies any
   sequence action other than ``target_analysis`` and the
   ``_required_next_step`` text demands it (independent of the
   ``--compare-against-gpu`` flag — when unset, the executor still runs
   and writes a ``reason='no_target_gpu_configured'`` marker).

   Most existing tests don't care about the prep gate; they construct a
   Coordinator and immediately exercise downstream behaviour (resume,
   proposal flow, e2e mock adapters, kernel handlers, ...). For those
   tests the cheapest fix is to write the marker JSON in the session
   fixture so the gate is satisfied from tick 0.

   Usage:

   .. code-block:: python

       from .conftest import seed_target_analysis_marker

       @pytest.fixture
       def session_dir(tmp_path, monkeypatch):
           monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
           sd = make_session_dir()
           seed_target_analysis_marker(sd)
           return sd

   Tests that *want* to exercise the gate itself (e.g.
   ``test_required_step_gates`` / the dedicated executor tests) should
   NOT call this helper; they construct their own session_dir fixture
   and write the marker JSON only inside the specific assertions where
   the gate-clearing transition is the subject under test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """N17 isolation: drop the in-process session-dir pin between tests.

    ``make_session_dir(model_name=...)`` writes
    ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` so every subprocess +
    every downstream ``paths.session_dir()`` call agree on the same
    location. In a long pytest run with hundreds of tests, that pin
    persists from test N into test N+1 if N+1 only does
    ``monkeypatch.setenv("USER_DATA_PATH", tmp_path)`` and never calls
    ``make_session_dir`` — and test N+1 then resolves session_dir to
    test N's tmp_path, which is already torn down. This autouse fixture
    deletes the pin (idempotent; ``raising=False``) at the start of
    every test so each test sees a fresh layout.

    Tests that *do* call ``make_session_dir`` immediately overwrite the
    pin with their own tmp_path, which is the production-equivalent
    behaviour.

    Also points ``MULTI_NODE_STATE_FILE`` at a per-test sentinel that
    does not exist by default. ``is_multi_node()`` and
    ``apply_kernel_patch._mn_state_path`` both honour this env var, so
    tests run as single-node unless they explicitly opt in (e.g. by
    writing their own state.json and re-setting the env). Without this,
    a stale ``/tmp/multi_node_state.json`` from a real run on the same
    host (or from an earlier test that didn't restore the env) would
    trip every baseline / profile / grid_runner test into the multi-
    node restart branch and surface as ``mn_server_restart_failed``.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SESSION_LAYOUT", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


@pytest.fixture(autouse=True)
def _clear_kernel_request_handler_caches():
    """Clear ``lru_cache`` state on env-bound helpers between tests.

    ``_default_geak_budget_minutes`` and ``_default_kernel_batch_parallel``
    cache their ``$GEAK_RUN_MODE`` / ``$KERNEL_AGENT_NUM_GPUS`` /
    ``torch.cuda.device_count()`` reads (these don't change in-session
    in production). Tests that monkeypatch any of those need a fresh
    cache so the helper actually re-reads the patched value.
    """
    from inference_optimizer.orchestrator import kernel_request_handlers as krh
    krh._default_geak_budget_minutes.cache_clear()
    krh._default_kernel_batch_parallel.cache_clear()


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout."""
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    # conftest.py -> tests -> inference_optimizer -> Hyperloom
    repo = Path(__file__).resolve().parents[2]
    kernel_agent = repo / "kernel-agent"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir.

    Returns the path of the written file. Idempotent: if the file already
    exists it is overwritten with the same contents (cheap; the JSON is
    a few hundred bytes).
    """
    from inference_optimizer.session_paths import target_baseline_json
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "skipped",
            "reason": "no_target_gpu_configured",
            "warning": "compare_against_gpu is empty",
        }),
        encoding="utf-8",
    )
    return path
