"""Cortex T0 anchor tests — post T2/T3 retirement.

The legacy ``session_begin`` step was removed alongside the
hypothesize/verify protocol; T0 is now a pure warm-start read plus
a best-effort recipe-anchor backfill (``update_recipe`` with
model-family / model-class / framework attrs).

Covers:

* ``--degraded-kb`` (client.enabled=False) short-circuits with a banner.
* Happy path: warm-start query fires, recipe backfill is attempted,
  ``cortex_session_id`` falls back to the session_dir name when no
  prior sid is on state.
* find_recipe failures are non-fatal.

HTTP calls are mocked via ``respx``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from inference_optimizer.cortex_kb_client import CortexKBClient
from inference_optimizer.orchestrator.cortex_t0 import (
    T0Result,
    run_t0_anchor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_pitfalls_json,
    cortex_warm_json,
)


KB_URL = "http://kb-test.local"


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("CORTEX_KB_URL", KB_URL)
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("CORTEX_KB_SMOKE", raising=False)
    return make_session_dir()


def _state_for_session(session_dir: Path) -> SharedState:
    s = SharedState()
    s.session_id = "sid-test"
    s.model_name = "Qwen-Qwen3-8B"
    s.gpu_type = "mi300x"
    s.framework = "sglang"
    s.save(session_dir)
    return SharedState.load_or_init(session_dir)


def _wire_t0_routes(router: respx.MockRouter) -> None:
    """Wire the surviving T0 HTTP calls with happy-path responses.

    Post T2/T3 retirement T0 only:
    1. POSTs ``/v1/points/propose`` (update_recipe backfill — best-effort)
    2. POSTs ``/v1/points/query`` (find_recipe_with_fallback — non-fatal)
    """
    router.post("/v1/points/propose").mock(
        return_value=httpx.Response(200, json={
            "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
        }),
    )
    router.post("/v1/points/query").mock(
        return_value=httpx.Response(200, json={"points": []}),
    )


# ===========================================================================
# 1. short-circuit branches
# ===========================================================================
def test_t0_disabled_client_emits_banner_and_skips_state_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    lines: list[str] = []
    result = run_t0_anchor(
        client, state,
        workload="Qwen-Qwen3-8B", hw="mi300x",
        on_status=lines.append,
        session_dir=session_dir,
    )
    assert isinstance(result, T0Result)
    assert result.status == "skipped_disabled"
    # No state writes happened.
    assert state.cortex_session_id == ""
    # Banner mentions DISABLED.
    assert any("DISABLED" in ln for ln in lines)


# ===========================================================================
# 2. happy path
# ===========================================================================
def test_t0_happy_path_writes_warm_and_pitfalls(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_routes(router)
        result = run_t0_anchor(
            client, state,
            workload="Qwen-Qwen3-8B", hw="mi300x",
            stack_fingerprint={"rocm": "7.2.0"},
            session_dir=session_dir,
        )

    assert result.status == "ok"
    # ``cortex_session_id`` is the hyperloom-local id (session_dir name
    # by default) — NOT a KB-side sid.
    assert state.cortex_session_id == session_dir.name
    # Warm-start file landed on disk even though the KB query returned
    # zero recipes (raw envelope is preserved).
    assert cortex_warm_json(session_dir).exists()
    assert cortex_pitfalls_json(session_dir).exists()


def test_t0_uses_existing_cortex_session_id_when_present(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    state.cortex_session_id = "carry-over-sid"
    state.save(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_routes(router)
        run_t0_anchor(
            client, state,
            workload="Qwen-Qwen3-8B", hw="mi300x",
            session_dir=session_dir,
        )
    assert state.cortex_session_id == "carry-over-sid"


# ===========================================================================
# 3. non-fatal degradation
# ===========================================================================
def test_t0_find_recipe_failure_is_non_fatal(session_dir):
    """find_recipe_with_fallback raising must NOT crash PRELUDE."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        # update_recipe backfill succeeds, query fails.
        router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
            }),
        )
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        result = run_t0_anchor(
            client, state,
            workload="Qwen-Qwen3-8B", hw="mi300x",
            session_dir=session_dir,
        )
    # T0 still returns ok; warm_present is False because the query failed.
    assert result.status == "ok"
    assert result.warm_present is False


def test_t0_recipe_backfill_failure_is_non_fatal(session_dir):
    """update_recipe backfill failure must NOT crash PRELUDE."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        # update_recipe propose fails; query returns empty.
        router.post("/v1/points/propose").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        result = run_t0_anchor(
            client, state,
            workload="Qwen-Qwen3-8B", hw="mi300x",
            session_dir=session_dir,
        )
    assert result.status == "ok"
