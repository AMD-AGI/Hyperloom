# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for SafeOptimizeClient submit-retry de-duplication.

The retry path must look a task up by modelId+workspace and reuse it instead
of creating a duplicate Claw session.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import optimize_submit  # noqa: E402


def _make_client() -> optimize_submit.SafeOptimizeClient:
    return optimize_submit.SafeOptimizeClient(
        base_url="https://fake.test",
        token="tok",
        register_workspace="ws-reg",
        submit_workspace="ws-sub",
        volume="/wekafs",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _install_fake_request(client, post_behaviors, list_items):
    """Wire a fake _request dispatching on method; track call counts."""
    state = {"post": 0, "get": 0}

    def fake(method, path, body=None, timeout=None):
        if method == "POST":
            i = state["post"]
            state["post"] += 1
            beh = post_behaviors[min(i, len(post_behaviors) - 1)]
            if isinstance(beh, Exception):
                raise beh
            return beh
        if method == "GET":
            state["get"] += 1
            return {"items": list_items, "total": len(list_items)}
        return {}

    client._request = fake  # type: ignore[assignment]
    return state


def _submit(client):
    return client.submit_task(
        model_id="m1",
        display_name="test-model",
        framework="sglang",
        precision="FP8",
        tp=8,
        concurrency=64,
        isl=1024,
        osl=1024,
        image="img:latest",
    )


def test_timeout_but_task_exists_reuses_without_resubmit(monkeypatch):
    """Transient POST failure + an already-created task -> reuse, no 2nd POST."""
    monkeypatch.setattr(optimize_submit.time, "sleep", lambda *_a, **_k: None)
    client = _make_client()
    existing = {"id": "existing-1", "modelId": "m1", "workspace": "ws-sub", "createdAt": _now_iso()}
    state = _install_fake_request(
        client,
        post_behaviors=[RuntimeError("POST api/v1/optimization/tasks -> read timed out")],
        list_items=[existing],
    )
    result = _submit(client)
    assert result["id"] == "existing-1"
    assert state["post"] == 1, "must NOT re-POST when the task already exists"
    assert state["get"] >= 1, "must look the task up before retrying"


def test_timeout_no_existing_task_retries_then_succeeds(monkeypatch):
    """Transient failure with no existing task -> retry POST and succeed."""
    monkeypatch.setattr(optimize_submit.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(optimize_submit.random, "uniform", lambda *_a, **_k: 0.0)
    client = _make_client()
    state = _install_fake_request(
        client,
        post_behaviors=[RuntimeError("HTTP 503: service unavailable"), {"id": "new-2"}],
        list_items=[],  # dedup finds nothing
    )
    result = _submit(client)
    assert result["id"] == "new-2"
    assert state["post"] == 2, "should retry the POST when no duplicate exists"


def test_non_transient_failure_raises_without_dedup(monkeypatch):
    """A non-transient error raises immediately; no dedup lookup, no retry."""
    monkeypatch.setattr(optimize_submit.time, "sleep", lambda *_a, **_k: None)
    client = _make_client()
    state = _install_fake_request(
        client,
        post_behaviors=[RuntimeError("HTTP 400: bad request")],
        list_items=[{"id": "x", "modelId": "m1", "createdAt": _now_iso()}],
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        _submit(client)
    assert state["post"] == 1
    assert state["get"] == 0, "non-transient must not trigger a dedup lookup"


def test_find_recent_submitted_task_picks_newest_matching(monkeypatch):
    """_find_recent_submitted_task: filter by modelId + recency, return newest."""
    client = _make_client()
    now = datetime.now(timezone.utc)
    items = [
        {"id": "old", "modelId": "m1", "createdAt": (now - timedelta(hours=3)).isoformat()},  # too old
        {"id": "other", "modelId": "m2", "createdAt": now.isoformat()},  # wrong model
        {"id": "recent-1", "modelId": "m1", "createdAt": (now - timedelta(seconds=10)).isoformat()},
        {"id": "recent-2", "modelId": "m1", "createdAt": now.isoformat()},  # newest matching
    ]
    client._request = lambda *a, **k: {"items": items}  # type: ignore[assignment]
    since = now.timestamp() - 60  # submit started ~60s ago
    got = client._find_recent_submitted_task("m1", "ws-sub", since)
    assert got is not None and got["id"] == "recent-2"


def test_find_recent_submitted_task_none_when_only_old(monkeypatch):
    """No match when the only same-model task predates the submit window."""
    client = _make_client()
    now = datetime.now(timezone.utc)
    items = [{"id": "ancient", "modelId": "m1", "createdAt": (now - timedelta(days=2)).isoformat()}]
    client._request = lambda *a, **k: {"items": items}  # type: ignore[assignment]
    since = now.timestamp()
    assert client._find_recent_submitted_task("m1", "ws-sub", since) is None
