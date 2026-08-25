# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import urllib.error

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_mcp import (
    GbrainRemoteError,
    _GbrainMcp,
    _require_http_url,
    _select_mcp_response,
)


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "application/json",
        content_length: bool = True,
    ) -> None:
        self._body = body
        self._offset = 0
        self._lines = iter(body.splitlines(keepends=True))
        self.headers = {"Content-Type": content_type}
        if content_length:
            self.headers["Content-Length"] = str(len(body))

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def readline(self) -> bytes:
        return next(self._lines, b"")


def _rpc_body(result: object, *, request_id: str = "1") -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode()


def test_select_mcp_response_handles_json_sse_and_bare_results() -> None:
    response = _select_mcp_response(
        '{"jsonrpc":"2.0","id":"1","result":{"ok":true}}',
        "1",
    )
    assert response["result"] == {"ok": True}

    sse = 'event: message\ndata: {"jsonrpc":"2.0","id":"1","result":{"ok":"sse"}}\n\n'
    assert _select_mcp_response(sse, "1")["result"] == {"ok": "sse"}
    assert _select_mcp_response('{"nodes":["a"]}', allow_bare_result=True) == {"nodes": ["a"]}
    assert _select_mcp_response(
        '{"jsonrpc":"2.0","id":"other","result":{"fallback":true}}',
        "wanted",
    )["result"] == {"fallback": True}


def test_require_http_url_rejects_non_http_schemes() -> None:
    _require_http_url("https://gbrain.example/mcp")
    with pytest.raises(GbrainRemoteError, match="unsupported"):
        _require_http_url("file:///tmp/gbrain")


def test_call_decodes_json_rpc_text_content(monkeypatch) -> None:
    body = _rpc_body({"content": [{"type": "text", "text": '{"page":{"slug":"x"}}'}]})
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(body),
    )

    result = _GbrainMcp("https://gbrain.example", "token", 1.0).call(
        "get_page",
        {"slug": "x"},
    )

    assert result == {"page": {"slug": "x"}}


def test_call_reads_sse_and_native_bare_results(monkeypatch) -> None:
    sse = b'event: message\ndata: {"jsonrpc":"2.0","id":"1","result":{"content":[{"text":"{\\"ok\\":true}"}]}}\n\n'
    responses = iter(
        (
            _Response(sse, content_type="text/event-stream", content_length=False),
            _Response(b'{"nodes":["a"]}', content_length=False),
        )
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: next(responses),
    )
    client = _GbrainMcp("https://gbrain.example", "token", 1.0)

    assert client.call("search", {}) == {"ok": True}
    assert client.call("get_links", {}) == {"nodes": ["a"]}


def test_read_body_tolerates_header_and_read_signature_shims() -> None:
    body = _rpc_body({"content": [{"text": '{"ok":"shim"}'}]})

    class _HeaderlessResponse:
        def __init__(self) -> None:
            self.sent = False

        @property
        def headers(self):
            raise ValueError("headers unavailable")

        def read(self) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return body

    client = _GbrainMcp("https://gbrain.example", "token", 1.0)
    assert json.loads(client._read_body(_HeaderlessResponse()))["id"] == "1"


def test_call_returns_bare_non_dict_and_unparsed_text(monkeypatch) -> None:
    responses = iter(
        (
            _Response(b'["a","b"]', content_length=False),
            _Response(_rpc_body({"content": [{"type": "text", "text": "plain text"}]})),
        )
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: next(responses),
    )
    client = _GbrainMcp("https://gbrain.example", "token", 1.0)

    assert client.call("get_links", {}) == ["a", "b"]
    assert client.call("search", {}) == "plain text"


@pytest.mark.parametrize(
    "body,match",
    [
        (
            b'{"jsonrpc":"2.0","id":"1","error":{"code":-1}}',
            "JSON-RPC error",
        ),
        (
            _rpc_body({"isError": True, "content": [{"text": "failed"}]}),
            "tool error",
        ),
        (b"not-json", "bad envelope"),
    ],
)
def test_call_surfaces_remote_envelope_errors(
    monkeypatch,
    body: bytes,
    match: str,
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(body),
    )
    with pytest.raises(GbrainRemoteError, match=match):
        _GbrainMcp("https://gbrain.example", "token", 1.0).call("search", {})


def test_call_wraps_transport_errors(monkeypatch) -> None:
    def _fail(*_args, **_kwargs):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    with pytest.raises(GbrainRemoteError, match="transport error"):
        _GbrainMcp("https://gbrain.example", "token", 1.0).call("search", {})
