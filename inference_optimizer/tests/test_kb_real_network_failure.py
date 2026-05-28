"""Real network-failure smoke tests for the soft-degrade contract.

The unit tests in ``test_kb_unavailable_resilience.py`` mock the KB
with respx and explicit ``CortexKBError``. This module goes one step
further and exercises the client against:

* a non-existent host (DNS failure / unroutable IP),
* a port that nothing is listening on (connection refused),
* a host that times out,

so we lock in that the **real** :class:`httpx.Client` failure modes
also degrade gracefully. These tests use real ``httpx`` retry +
backoff timing so they're slower (~10s each); guarded behind a
fixture that prunes the retry budget for test speed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from inference_optimizer.cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
)


# ===========================================================================
# Speed up the test by reducing retry attempts / backoff.
# ===========================================================================
@pytest.fixture
def fast_retry(monkeypatch):
    """Cut DEFAULT_RETRY_ATTEMPTS to 1 so connection refused / timeout
    failures don't waste seconds in backoff during tests."""
    import inference_optimizer.cortex_kb_constants as C
    monkeypatch.setattr(C, "DEFAULT_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(C, "DEFAULT_RETRY_BASE_MS", 0)
    monkeypatch.setattr(C, "DEFAULT_HTTP_TIMEOUT_SEC", 1.0)


# ===========================================================================
# Scenario A — port nothing listens on (connection refused)
# ===========================================================================
def test_connection_refused_propagates_as_cortex_kb_error(tmp_path, fast_retry):
    """When KB is unreachable on a port nobody listens on, the client
    raises ``CortexKBError`` (caught by caller / propose-write fallback
    to NDJSON). We assert the right error category so caller branches
    behave (transport → NDJSON enqueue retry path)."""
    # 127.0.0.1:1 is reserved; nothing listens on it.
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
    )
    with pytest.raises(CortexKBError) as exc_info:
        client.read_recipe_exact(model="M", hardware="H")
    assert exc_info.value.category == "transport"


def test_connection_refused_lessons_returns_empty(tmp_path, fast_retry):
    """T0 readers (lessons / pitfalls / find_recipe_with_fallback)
    handle CortexKBError internally and return empty — the warm-start
    just renders ``(none)`` in the prompt, no exception propagates."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
    )
    assert client.lessons(model="M", hardware="H") == []
    assert client.pitfalls(model="M", hardware="H") == []
    point, tier, conf = client.find_recipe_with_fallback(workload="M", hw="H")
    assert point == {}
    assert tier == "miss"


def test_connection_refused_propose_falls_back_to_ndjson(tmp_path, fast_retry):
    """When KB write hits connection refused, ``propose_lesson``
    catches the CortexKBError internally and writes to NDJSON. Caller
    sees ``status=queued``."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
    )
    out = client.propose_lesson(
        statement="X improves Y",
        measured_impact="gain_pct=10",
        applicable_models=["M"],
        applicable_hardware=["H"],
    )
    assert out["status"] == "queued"
    assert client.pending_path.read_text(encoding="utf-8").strip()


# ===========================================================================
# Scenario B — unresolvable host (DNS failure)
# ===========================================================================
def test_dns_failure_propagates_as_cortex_kb_error(tmp_path, fast_retry):
    """Unresolvable .invalid TLD — DNS fails, surfaces as transport
    error in CortexKBError. Same handling as connection refused."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://kb-test.invalid:443",
    )
    with pytest.raises(CortexKBError) as exc_info:
        client.read_recipe_exact(model="M", hardware="H")
    assert exc_info.value.category == "transport"


def test_dns_failure_warm_start_reads_return_empty(tmp_path, fast_retry):
    """Same internal-catch contract for warm-start readers on DNS
    failure."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://kb-test.invalid:443",
    )
    assert client.lessons(model="M", hardware="H") == []
    assert client.pitfalls(model="M", hardware="H") == []


# ===========================================================================
# Scenario C — drain_pending against a dead KB
# ===========================================================================
def test_drain_pending_with_unreachable_kb_returns_remaining(tmp_path, fast_retry):
    """When KB is dead, every queued NDJSON row goes ``transient`` and
    accumulates as ``remaining``. ``drain_pending`` returns gracefully
    (no exception); rows stay in the pending file for the next attempt."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
    )
    # Queue two rows by attempting writes; they auto-NDJSON-enqueue.
    client.propose_lesson(
        statement="X", measured_impact="y",
        applicable_models=["M"], applicable_hardware=["H"],
    )
    client.propose_pitfall(
        description="Z crashed", severity="crash",
        applicable_models=["M"], applicable_hardware=["H"],
    )
    # Drain attempts each row, hits transport error, leaves them in
    # the pending file.
    report = client.drain_pending(timeout_sec=5.0)
    assert isinstance(report, dict)
    assert report.get("remaining", 0) >= 1
    # Pending file still has rows.
    assert client.pending_path.exists()


# ===========================================================================
# Scenario D — speed test: a single failed write doesn't take forever
# ===========================================================================
def test_failed_write_returns_within_a_few_seconds(tmp_path, fast_retry):
    """Latency budget: even with retries, a single propose_* call
    against a dead KB must return within a small handful of seconds.
    Otherwise the main loop blocks for the full retry budget on every
    KEEP / REVERT, which violates the soft-degrade contract."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
    )
    start = time.monotonic()
    client.propose_lesson(
        statement="X", measured_impact="y",
        applicable_models=["M"], applicable_hardware=["H"],
    )
    elapsed = time.monotonic() - start
    # Generous bound: with the test fast_retry fixture (1 attempt,
    # 1s timeout) the call should return in < 5s. Without the fixture,
    # production retries (3 attempts × backoff) cap at ~10s — still
    # bounded.
    assert elapsed < 10.0, f"propose_lesson hung for {elapsed:.1f}s on dead KB"


# ===========================================================================
# Scenario E — disabled client never touches the network
# ===========================================================================
def test_disabled_client_no_network_call_even_with_bad_url(tmp_path):
    """``client.enabled=False`` must short-circuit BEFORE any HTTP
    attempt. Test by pointing at an unreachable URL and confirming the
    call returns instantly with the disabled-skip sentinel — no
    seconds-long timeout."""
    client = CortexKBClient(
        session_dir=tmp_path / "s",
        kb_url="http://127.0.0.1:1",
        enabled=False,
    )
    start = time.monotonic()
    out = client.propose_lesson(
        statement="X", measured_impact="y",
        applicable_models=["M"], applicable_hardware=["H"],
    )
    elapsed = time.monotonic() - start
    assert out["status"] == "skip_disabled"
    assert elapsed < 0.1, f"disabled client took {elapsed:.3f}s, must be ~instant"
