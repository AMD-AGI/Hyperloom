# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal gbrain MCP page client for the standalone ``fa`` package.

Self-contained (no reverse import of ``inference_optimizer``) so the
framework-agent package stays independently installable. The MCP/SSE
contract mirrors ``inference_optimizer/recipe_kb/gbrain_remote_client.py``.

Exposes only the read surface PR KB consumption needs: ``get_page`` /
``query`` (MCP ``search`` tool) / ``list_pages``. All failures raise
:class:`GbrainPageError`; callers treat that as "source unavailable".
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any


class GbrainPageError(RuntimeError):
    """Raised on transport / envelope / tool-level gbrain failures."""


def _iter_sse_objects(raw: str):
    """Yield JSON objects from an MCP body (plain JSON or one/many SSE events)."""
    text = raw.lstrip()
    if text.startswith("{") or text.startswith("["):
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            # Malformed non-SSE payload: yield nothing.
            return
        return
    for block in re.split(r"\r?\n\r?\n", raw):
        parts: list[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                seg = line[5:]
                parts.append(seg[1:] if seg.startswith(" ") else seg)
        payload = "\n".join(parts).strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            # Skip a malformed SSE event block.
            continue


def _select_mcp_response(raw: str, want_id: Any = None) -> Any:
    """Pick the JSON-RPC response event matching ``want_id`` (else first result)."""
    fallback: Any = None
    for obj in _iter_sse_objects(raw):
        if isinstance(obj, dict):
            if want_id is not None and obj.get("id") == want_id:
                return obj
            if fallback is None and ("result" in obj or "error" in obj):
                fallback = obj
    return fallback


class GbrainPageClient:
    """Minimal JSON-RPC-over-HTTP MCP client (get_page / query / list_pages)."""

    _RPC_ID = "1"

    def __init__(self, base_url: str, token: str, *, timeout_sec: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = max(0.5, float(timeout_sec))

    def _read_body(self, resp: Any) -> str:
        """Read an MCP body under a wall-clock deadline (SSE-safe, no hang)."""
        deadline = time.monotonic() + self._timeout
        try:
            headers = resp.headers
            ctype = (headers.get("Content-Type") or "").lower()
            clen = headers.get("Content-Length")
        except Exception:  # noqa: BLE001
            ctype, clen = "", ""
        if "text/event-stream" not in ctype and clen:
            return resp.read().decode("utf-8", "replace")
        chunks: list[bytes] = []
        buf = b""
        readline = getattr(resp, "readline", None)
        reader = readline if ("text/event-stream" in ctype and callable(readline)) else None
        while True:
            if time.monotonic() >= deadline:
                break
            try:
                piece = reader() if reader is not None else resp.read(4096)
            except TypeError:
                piece = resp.read()
            except (OSError, ValueError):
                break
            if not piece:
                break
            chunks.append(piece)
            buf += piece
            if _select_mcp_response(buf.decode("utf-8", "replace"), self._RPC_ID) is not None:
                break
        return b"".join(chunks).decode("utf-8", "replace")

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke an MCP tool; return decoded result (parsed JSON when present)."""
        envelope = {
            "jsonrpc": "2.0",
            "id": self._RPC_ID,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        req = urllib.request.Request(
            self._base + "/mcp",
            data=json.dumps(envelope).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {self._token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = self._read_body(resp)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise GbrainPageError(f"gbrain {tool} transport error: {exc!r}") from exc
        obj = _select_mcp_response(raw, self._RPC_ID)
        if not isinstance(obj, dict):
            prefix = raw[:200].replace("\n", "\\n")
            raise GbrainPageError(f"gbrain {tool} bad envelope; body_prefix={prefix!r}")
        if obj.get("error") is not None:
            raise GbrainPageError(f"gbrain {tool} JSON-RPC error: {obj.get('error')!r}")
        result = obj.get("result") or {}
        if isinstance(result, dict) and result.get("isError"):
            raise GbrainPageError(f"gbrain {tool} tool error: {result.get('content')!r}")
        content = result.get("content") or []
        if content and isinstance(content[0], dict) and content[0].get("text"):
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return content[0]["text"]
        return result

    # -- read surface --------------------------------------------------------

    def get_page(self, slug: str) -> dict[str, Any] | None:
        """Return the page object for ``slug``, or ``None`` when absent."""
        res = self.call("get_page", {"slug": slug})
        return res if isinstance(res, dict) else None

    def query(self, text: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Semantic search via the gbrain MCP ``search`` tool (parity w/ recipe)."""
        res = self.call("search", {"query": text, "limit": int(limit)})
        return _as_hit_list(res)

    def list_pages(self, *, page_type: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """List pages, optionally filtered by ``type``."""
        args: dict[str, Any] = {"limit": int(limit)}
        if page_type:
            args["type"] = page_type
        res = self.call("list_pages", args)
        return _as_hit_list(res)

    def health(self) -> bool:
        """Lightweight liveness probe (single ``list_pages(limit=1)``)."""
        try:
            self.list_pages(limit=1)
            return True
        except GbrainPageError:
            return False


def _as_hit_list(res: Any) -> list[dict[str, Any]]:
    """Coerce a query/list_pages result into a list of dict hits."""
    if isinstance(res, list):
        return [h for h in res if isinstance(h, dict)]
    if isinstance(res, dict):
        for key in ("results", "hits", "data", "pages", "items"):
            val = res.get(key)
            if isinstance(val, list):
                return [h for h in val if isinstance(h, dict)]
    return []


def build_gbrain_page_client_from_env() -> GbrainPageClient | None:
    """Build a client from ``GBRAIN_BASE_URL`` / ``GBRAIN_TOKEN``; ``None`` if unset."""
    base_url = (os.environ.get("GBRAIN_BASE_URL", "") or "").strip()
    token = (os.environ.get("GBRAIN_TOKEN", "") or "").strip()
    if not base_url or not token:
        return None
    timeout_sec = 10.0
    raw = os.environ.get("GBRAIN_HTTP_TIMEOUT_SEC")
    if raw:
        try:
            timeout_sec = float(raw)
        except ValueError:
            timeout_sec = 10.0
    return GbrainPageClient(base_url, token, timeout_sec=timeout_sec)


__all__ = [
    "GbrainPageClient",
    "GbrainPageError",
    "build_gbrain_page_client_from_env",
]
