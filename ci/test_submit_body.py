"""Unit tests for SafeOptimizeClient.submit_task body construction.

Verifies that the body always includes ``inferencexPath`` — even when
the caller passes an empty string — so the SaFE backend's Zod default
(``/hyperloom/InferenceX``) never kicks in and silently pins the sandbox
to the shared read-only mount.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from optimize_submit import SafeOptimizeClient  # noqa: E402


def _make_client() -> SafeOptimizeClient:
    c = SafeOptimizeClient(
        base_url="https://fake.test",
        token="tok",
        register_workspace="ws-reg",
        submit_workspace="ws-sub",
        volume="/wekafs",
    )
    return c


def _capture_body(client: SafeOptimizeClient) -> dict:
    """Call submit_task with minimal args and return the POST body."""
    captured: dict = {}

    def fake_request(_method: str, _path: str, body: dict | None = None) -> dict:
        captured.update(body or {})
        return {"id": "task-123"}

    client._request = fake_request  # type: ignore[assignment]
    client.submit_task(
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
    return captured


def test_default_inferencex_path_is_empty_string():
    """When no inferencex_path is given, the body key exists and is empty."""
    client = _make_client()
    body = _capture_body(client)
    assert "inferencexPath" in body, (
        "inferencexPath must always be present to prevent SaFE Zod default"
    )
    assert body["inferencexPath"] == ""


def test_explicit_inferencex_path_forwarded():
    """An explicit override is forwarded verbatim."""
    client = _make_client()
    captured: dict = {}

    def fake_request(_method, _path, body=None):
        captured.update(body or {})
        return {"id": "task-456"}

    client._request = fake_request  # type: ignore[assignment]
    client.submit_task(
        model_id="m2",
        display_name="test-model-2",
        framework="sglang",
        precision="FP8",
        tp=8,
        concurrency=64,
        isl=1024,
        osl=1024,
        image="img:latest",
        inferencex_path="/custom/InferenceX",
    )
    assert captured["inferencexPath"] == "/custom/InferenceX"


def test_empty_string_inferencex_path_stays_empty():
    """Explicitly passing '' still produces an empty-string body value."""
    client = _make_client()
    captured: dict = {}

    def fake_request(_method, _path, body=None):
        captured.update(body or {})
        return {"id": "task-789"}

    client._request = fake_request  # type: ignore[assignment]
    client.submit_task(
        model_id="m3",
        display_name="test-model-3",
        framework="sglang",
        precision="FP8",
        tp=8,
        concurrency=64,
        isl=1024,
        osl=1024,
        image="img:latest",
        inferencex_path="",
    )
    assert captured["inferencexPath"] == ""


def test_none_inferencex_path_becomes_empty():
    """None (the default) is normalised to ''."""
    client = _make_client()
    captured: dict = {}

    def fake_request(_method, _path, body=None):
        captured.update(body or {})
        return {"id": "task-000"}

    client._request = fake_request  # type: ignore[assignment]
    client.submit_task(
        model_id="m4",
        display_name="test-model-4",
        framework="sglang",
        precision="FP8",
        tp=8,
        concurrency=64,
        isl=1024,
        osl=1024,
        image="img:latest",
        inferencex_path=None,
    )
    assert "inferencexPath" in captured
    assert captured["inferencexPath"] == ""
