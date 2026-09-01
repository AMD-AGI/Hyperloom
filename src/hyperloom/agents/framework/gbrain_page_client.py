# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal gbrain MCP page client for the standalone ``fa`` package.

Deliberately avoids importing ``hyperloom.inference_optimizer`` /
``hyperloom.orchestrator`` so framework-agent has no reverse dependency on the
orchestrator layer (shared ``hyperloom.common`` helpers are fine). The MCP/SSE
contract mirrors the orchestrator's retained GBrain KG MCP transport.

Exposes only the read surface PR KB consumption needs: ``get_page`` /
``query`` (MCP ``search`` tool) / ``list_pages``. Failures raise
:class:`GbrainPageError`; callers treat that as "source unavailable". An absent
page is not a failure: ``get_page`` returns ``None`` for it, so a cold KB does
not read as an outage.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from hyperloom.common.jsonio import iter_sse_objects
from hyperloom.common.url_safety import require_http_url

log = logging.getLogger(__name__)


class GbrainPageError(RuntimeError):
    """Raised on transport / envelope / tool-level gbrain failures."""


def _is_page_missing(exc: Exception) -> bool:
    """Whether a gbrain tool error is an absent page rather than a real failure.

    gbrain reports an absent slug as an in-band ``isError``, which ``call``
    surfaces as an exception like any other tool error.
    """
    return "page_not_found" in str(exc)


def _select_mcp_response(raw: str, want_id: Any = None) -> Any:
    """Pick the JSON-RPC response event matching ``want_id`` (else first result)."""
    fallback: Any = None
    for obj in iter_sse_objects(raw):
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
        require_http_url(base_url, error=GbrainPageError, context="GBRAIN_BASE_URL")
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
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310 - URL scheme checked above.
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
        try:
            res = self.call("get_page", {"slug": slug})
        except GbrainPageError as exc:
            if not _is_page_missing(exc):
                raise
            return None
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
    """Build a client from ``GBRAIN_BASE_URL`` / ``GBRAIN_TOKEN``; ``None`` if unset.

    A configured URL whose scheme is not ``http``/``https`` is refused (logged
    and ``None``) so a bad ``GBRAIN_BASE_URL`` cannot reach ``urlopen``. One
    ``urllib.parse`` cannot parse at all (an unterminated IPv6 literal raises
    ``ValueError`` from under the scheme check) is refused the same way: every
    caller of this builder reads ``None`` as "source unconfigured", so a typo in
    the environment must not propagate as an exception.
    """
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
    try:
        return GbrainPageClient(base_url, token, timeout_sec=timeout_sec)
    except (GbrainPageError, ValueError) as exc:
        log.warning("gbrain: refusing client for GBRAIN_BASE_URL: %s", exc)
        return None


__all__ = [
    "GbrainPageClient",
    "GbrainPageError",
    "build_gbrain_page_client_from_env",
]
