# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal GBrain MCP transport retained for KG integrations."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hyperloom.common.jsonio import iter_sse_objects


class GbrainRemoteError(RuntimeError):
    """Raised when a GBrain MCP interaction cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "transport",
        **_: Any,
    ) -> None:
        super().__init__(message)
        self.category = category


_BARE_RESULT_TOOLS = {
    "get_links",
    "get_backlinks",
    "traverse_graph",
}


def _select_mcp_response(
    raw: str,
    want_id: Any = None,
    *,
    allow_bare_result: bool = False,
) -> Any:
    """Select the matching JSON-RPC event from JSON or SSE response text."""

    fallback: Any = None
    bare: Any = None
    for obj in iter_sse_objects(raw):
        if isinstance(obj, dict):
            if want_id is not None and obj.get("id") == want_id:
                return obj
            if fallback is None and ("result" in obj or "error" in obj):
                fallback = obj
            elif allow_bare_result and bare is None:
                bare = obj
        elif allow_bare_result and bare is None:
            bare = obj
    return fallback if fallback is not None else bare


def _require_http_url(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        raise GbrainRemoteError(
            f"unsupported gbrain URL scheme: {scheme!r}",
            category="validation",
        )


class _GbrainMcp:
    """Small JSON-RPC-over-HTTP client used by :class:`KGClient`."""

    _RPC_ID = "1"

    def __init__(self, base_url: str, token: str, timeout_sec: float) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = max(0.5, float(timeout_sec))

    def _read_body(
        self,
        resp: Any,
        *,
        allow_bare_result: bool = False,
    ) -> str:
        """Read JSON or the first complete SSE response under a deadline."""

        deadline = time.monotonic() + self._timeout
        chunks: list[bytes] = []
        buffered = b""
        try:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            content_length = resp.headers.get("Content-Length") or ""
        except Exception:  # noqa: BLE001 - response shims may omit headers
            content_type = ""
            content_length = ""
        if "text/event-stream" not in content_type and content_length:
            return resp.read().decode()

        readline = getattr(resp, "readline", None)
        if "text/event-stream" in content_type and callable(readline):
            while time.monotonic() < deadline:
                try:
                    piece = readline()
                except (OSError, ValueError):
                    break
                if not piece:
                    break
                chunks.append(piece)
                buffered += piece
                if (
                    _select_mcp_response(
                        buffered.decode("utf-8", "replace"),
                        self._RPC_ID,
                        allow_bare_result=allow_bare_result,
                    )
                    is not None
                ):
                    break
            return b"".join(chunks).decode("utf-8", "replace")

        while time.monotonic() < deadline:
            try:
                piece = resp.read(4096)
            except TypeError:
                piece = resp.read()
            except (OSError, ValueError):
                break
            if not piece:
                break
            chunks.append(piece)
            buffered += piece
            if (
                _select_mcp_response(
                    buffered.decode("utf-8", "replace"),
                    self._RPC_ID,
                    allow_bare_result=allow_bare_result,
                )
                is not None
            ):
                break
        return b"".join(chunks).decode("utf-8", "replace")

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke one GBrain MCP tool and decode its result."""

        envelope = {
            "jsonrpc": "2.0",
            "id": self._RPC_ID,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        url = self._base + "/mcp"
        _require_http_url(url)
        request = urllib.request.Request(
            url,
            data=json.dumps(envelope).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # nosec B310 - scheme validated above
                request,
                timeout=self._timeout,
            ) as response:
                raw = self._read_body(
                    response,
                    allow_bare_result=tool in _BARE_RESULT_TOOLS,
                )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GbrainRemoteError(f"gbrain {tool} transport error: {exc!r}") from exc

        obj = _select_mcp_response(
            raw,
            self._RPC_ID,
            allow_bare_result=tool in _BARE_RESULT_TOOLS,
        )
        if obj is None:
            prefix = raw[:300].replace("\n", "\\n").replace("\r", "\\r")
            raise GbrainRemoteError(
                f"gbrain {tool} bad envelope: no parseable JSON-RPC response in body; body_prefix={prefix!r}"
            )
        if not isinstance(obj, dict):
            return obj
        if tool in _BARE_RESULT_TOOLS and "result" not in obj and "error" not in obj:
            return obj
        if obj.get("error") is not None:
            raise GbrainRemoteError(f"gbrain {tool} JSON-RPC error: {obj.get('error')!r}")
        result = obj.get("result") or {}
        if isinstance(result, dict) and result.get("isError"):
            raise GbrainRemoteError(f"gbrain {tool} tool error: {result.get('content')!r}")
        content = result.get("content") or []
        if content and isinstance(content[0], dict) and content[0].get("text"):
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return content[0]["text"]
        return result


__all__ = ["GbrainRemoteError", "_GbrainMcp"]
