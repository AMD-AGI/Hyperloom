"""v0.8 KB_design §3.2 §5.1 + KB_gaps/Gap-12 — Cortex T0 anchor tests.

Covers the v0.8 contract for the T0 ritual after the cli /
Coordinator dual-entry refactor (KB_gaps/Gap-12):

* :func:`orchestrator.cortex_t0.run_t0_anchor` is the single source
  of truth for the four T0 steps (session_begin / propose_point
  recipe / find_recipe / traps) and the SharedState writes
  that go with them.
* cli is the **canonical** entry point — fail-fast on Cortex
  failure (``sys.exit(2)``), stdout banner the operator expects.
* :meth:`Coordinator._ensure_cortex_t0_anchored` is a **defensive
  fallback** for SDK / integration-test callers that construct a
  :class:`Coordinator` without going through the cli plumbing —
  fail-soft on Cortex failure, INFO-log banner instead of stdout.
* The fallback no-ops when ``cortex_kb`` is None, disabled, or a
  prior anchor already wrote ``shared_state.cortex_session_id``.

HTTP calls are mocked via ``respx``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from inference_optimizer.cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
)
from inference_optimizer.orchestrator.cortex_t0 import (
    T0Result,
    run_t0_anchor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_pitfalls_json,
    cortex_sid_file,
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


def _wire_t0_happy_path(router: respx.MockRouter, sid: int = 99) -> None:
    """Wire the four T0 HTTP calls with happy-path responses."""
    router.post("/v1/sessions/begin").mock(
        return_value=httpx.Response(200, json={
            "session_id": sid, "thinking_style": "recommendation", "lens_schedule": [],
        }),
    )
    router.post("/v1/points/propose").mock(
        return_value=httpx.Response(200, json={
            "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
        }),
    )
    router.post("/v1/points/query").mock(
        return_value=httpx.Response(200, json={"points": []}),
    )


# ===========================================================================
# 1. run_t0_anchor — short-circuit branches
# ===========================================================================
def test_t0_disabled_client_emits_banner_and_skips_state_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    lines: list[str] = []
    result = run_t0_anchor(
        client, state,
        workload="Qwen-Qwen3-8B", hw="mi300x",
        on_status=lines.append,
    )
    assert isinstance(result, T0Result)
    assert result.status == "skipped_disabled"
    assert any("DISABLED" in line for line in lines)
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}
    assert state.warm_start_pitfalls == []


def test_t0_already_anchored_session_id_does_not_re_begin(session_dir):
    """When ``cortex_session_id`` is already non-empty (e.g. cli already
    ran T0), the helper MUST NOT call ``session begin`` again. It still
    refreshes find_recipe / traps so a long-running session that
    survives Cortex outages picks up newer KB rows."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    state.cortex_session_id = "prior-sid"
    sentinel: list[str] = []

    def _boom(**kwargs):
        sentinel.append("session_begin")
        return "SHOULD-NOT-FIRE"

    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
            }),
        )
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        with patch.object(client, "session_begin", side_effect=_boom):
            result = run_t0_anchor(
                client, state,
                workload="w", hw="mi300x",
                on_status=lambda _l: None,
            )
    assert sentinel == []
    assert result.session_id == "prior-sid"
    assert state.cortex_session_id == "prior-sid"


def test_t0_fail_soft_returns_failed_session_begin_on_cortex_error(session_dir):
    """fail_fast=False (Coordinator fallback) absorbs the error and
    returns a failed result; warm_start fields stay empty."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("synthetic outage"),
    ):
        result = run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            fail_fast=False,
            on_status=lambda _l: None,
        )
    assert result.status == "failed_session_begin"
    assert "synthetic outage" in result.error
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_t0_fail_fast_propagates_cortex_error(session_dir):
    """fail_fast=True (cli path) re-raises so cli can sys.exit(2)."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("synthetic outage"),
    ):
        with pytest.raises(CortexKBError):
            run_t0_anchor(
                client, state,
                workload="w", hw="mi300x",
                fail_fast=True,
                on_status=lambda _l: None,
            )


# ===========================================================================
# 2. run_t0_anchor — happy path with HTTP mocks
# ===========================================================================
def test_t0_happy_path_writes_sid_warm_and_pitfalls(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    banner: list[str] = []
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_happy_path(router, sid=12345)
        result = run_t0_anchor(
            client, state,
            workload="Qwen-Qwen3-8B", hw="mi300x",
            image_digest="sha256:demo",
            stack_fingerprint={"rocm": "7.2.0"},
            extra_attrs={"framework": "sglang"},
            on_status=banner.append,
            session_dir=session_dir,
            save_state=True,
        )
    assert result.status == "ok"
    assert result.session_id == "12345"
    assert state.cortex_session_id == "12345"
    assert state.warm_start_recipe["workload"] == "Qwen-Qwen3-8B"
    assert state.warm_start_recipe["hw"] == "mi300x"
    assert cortex_sid_file(session_dir).read_text(encoding="utf-8").strip() == "12345"
    assert cortex_warm_json(session_dir).exists()
    assert cortex_pitfalls_json(session_dir).exists()
    assert any("session_id=12345" in line for line in banner)


def test_t0_skipped_already_when_sid_present_via_anchor(session_dir):
    """Two consecutive ``run_t0_anchor`` calls: the second sees a
    sid and reports ``skipped_already`` — no re-begin."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    state = _state_for_session(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_happy_path(router, sid=12345)
        run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            on_status=lambda _l: None,
        )
        sid_after_first = state.cortex_session_id
        assert sid_after_first == "12345"
        boom_calls: list[str] = []
        with patch.object(
            client, "session_begin",
            side_effect=lambda **kw: boom_calls.append("hit") or "WRONG",
        ):
            result = run_t0_anchor(
                client, state,
                workload="w", hw="mi300x",
                on_status=lambda _l: None,
            )
    assert result.status == "skipped_already"
    assert boom_calls == []
    assert state.cortex_session_id == sid_after_first


# ===========================================================================
# 3. Coordinator._ensure_cortex_t0_anchored — defensive SDK fallback
# ===========================================================================
def _bare_coord(session_dir: Path, *, client, state):
    from inference_optimizer.orchestrator.coordinator import Coordinator
    c = Coordinator.__new__(Coordinator)
    c.session_dir = session_dir
    c.shared_state = state
    c.cortex_kb = client
    return c


def test_coordinator_fallback_noop_when_cortex_disabled(session_dir):
    state = _state_for_session(session_dir)
    client = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    coord = _bare_coord(session_dir, client=client, state=state)
    coord._ensure_cortex_t0_anchored()
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_noop_when_client_is_none(session_dir):
    state = _state_for_session(session_dir)
    coord = _bare_coord(session_dir, client=None, state=state)
    coord._ensure_cortex_t0_anchored()
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_noop_when_session_id_already_set(session_dir):
    state = _state_for_session(session_dir)
    state.cortex_session_id = "from-cli"
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    coord = _bare_coord(session_dir, client=client, state=state)
    boom_calls: list[str] = []
    with patch.object(
        client, "session_begin",
        side_effect=lambda **kw: boom_calls.append("hit") or "WRONG",
    ):
        coord._ensure_cortex_t0_anchored()
    assert boom_calls == []
    assert state.cortex_session_id == "from-cli"


def test_coordinator_fallback_runs_t0_when_sid_missing(session_dir):
    """SDK / integration-test path: ``Coordinator(...)`` constructed
    without cli plumbing. cortex_session_id is empty → fallback
    fires and writes warm_start to SharedState."""
    state = _state_for_session(session_dir)
    assert state.cortex_session_id == ""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    coord = _bare_coord(session_dir, client=client, state=state)
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_happy_path(router, sid=98765)
        coord._ensure_cortex_t0_anchored()
    assert state.cortex_session_id == "98765"
    assert state.warm_start_recipe
    reloaded = SharedState.load_or_init(session_dir)
    assert reloaded.cortex_session_id == "98765"
    assert reloaded.warm_start_recipe


def test_coordinator_fallback_absorbs_cortex_error(session_dir):
    """Cortex outage during fallback → reactor boots cleanly with
    empty warm_start, no raise."""
    state = _state_for_session(session_dir)
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    coord = _bare_coord(session_dir, client=client, state=state)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("outage"),
    ):
        coord._ensure_cortex_t0_anchored()
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_uses_state_workload_hw(session_dir):
    """Workload / hw flow from SharedState, not from a manifest dict
    (SDK callers don't have a manifest)."""
    state = _state_for_session(session_dir)
    state.model_name = "Llama-3.1-70B"
    state.gpu_type = "mi325x"
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    coord = _bare_coord(session_dir, client=client, state=state)
    with respx.mock(base_url=KB_URL) as router:
        _wire_t0_happy_path(router, sid=11111)
        coord._ensure_cortex_t0_anchored()
    assert state.warm_start_recipe["workload"] == "Llama-3.1-70B"
    assert state.warm_start_recipe["hw"] == "mi325x"


# ===========================================================================
# 4. Re-export sanity — run_t0_anchor is importable from cli too
# ===========================================================================
def test_cli_imports_run_t0_anchor():
    """cli must import the helper so the refactored
    ``_bootstrap_cortex_kb`` keeps working without re-implementing
    the ritual locally."""
    from inference_optimizer import cli  # noqa: F401
    assert hasattr(cli, "run_t0_anchor")
    from inference_optimizer.orchestrator.cortex_t0 import (
        run_t0_anchor as canonical,
    )
    assert cli.run_t0_anchor is canonical
