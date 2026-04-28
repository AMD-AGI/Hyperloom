"""SageQueryService — DESIGN §5.1.2.

Lightweight always-on Codex-backed advisor. Other agents call ``recall``
synchronously; service hits KB and returns a markdown snippet. 30s
hard-timeout falls back to the empty string.

STATUS (v0.7):
    Pure-Python implementation. Caches results per
    ``(model_name, action_name)`` for 5 min so a marathon Sage run can
    pre-fetch in bulk without re-firing 200 backend calls / hour.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any


__all__ = [
    "SageQueryService",
    "SAGE_TIMEOUT_S",
    "SAGE_CACHE_TTL_S",
]


SAGE_TIMEOUT_S: float = 30.0
SAGE_CACHE_TTL_S: float = 300.0  # 5 min

log = logging.getLogger(__name__)


class SageQueryService:
    """Synchronous-feeling KB query channel (timeouts to empty string)."""

    def __init__(
        self,
        codex_backend: Any,
        kb: Any,
        *,
        timeout_s: float = SAGE_TIMEOUT_S,
        cache_ttl_s: float = SAGE_CACHE_TTL_S,
    ) -> None:
        self.backend = codex_backend
        self.kb = kb
        self.timeout_s = float(timeout_s)
        self.cache_ttl_s = float(cache_ttl_s)
        self._cache: dict[tuple[str, str], tuple[float, str]] = {}

    # ------------------------------------------------------------------
    def _cache_get(self, key: tuple[str, str]) -> str | None:
        hit = self._cache.get(key)
        if hit is None:
            return None
        ts, value = hit
        if time.time() - ts > self.cache_ttl_s:
            self._cache.pop(key, None)
            return None
        return value

    def _cache_put(self, key: tuple[str, str], value: str) -> None:
        self._cache[key] = (time.time(), value)

    # ------------------------------------------------------------------
    async def recall(
        self, model_name: str, action_name: str
    ) -> str:
        """Return ≤500-token markdown of relevant KB / persona slices.

        Behaviour summary:

            - cache hit (≤ ``SAGE_CACHE_TTL_S`` old) → return immediately
            - cold-start (no warm-start eligibility) → empty string
            - backend / KB failure → empty string (never raises)
            - timeout (> ``SAGE_TIMEOUT_S``) → empty string
        """
        key = (str(model_name), str(action_name))
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        try:
            value = await asyncio.wait_for(
                self._do_recall(model_name, action_name),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            log.info(
                "sage recall timed out: model=%s action=%s",
                model_name, action_name,
            )
            value = ""
        except Exception:  # noqa: BLE001 — never raise
            log.exception("sage recall failed: model=%s", model_name)
            value = ""
        self._cache_put(key, value)
        return value

    async def _do_recall(self, model_name: str, action_name: str) -> str:
        """Inner recall logic — KB lookup, optional backend annotation."""
        if self.kb is None:
            return ""
        snippet: str
        try:
            snippet = self.kb.recall_for_model(model_name, action_name) or ""
        except Exception:  # noqa: BLE001
            snippet = ""
        if not snippet:
            return ""
        if self.backend is None or not hasattr(self.backend, "run"):
            return snippet
        # Optional re-rank / annotate via Codex (no-tools); failure-tolerant.
        try:
            prompt = (
                f"# Sage hint\n"
                f"## model: {model_name}\n## action: {action_name}\n\n"
                f"## Raw KB extracts\n{snippet}\n\n"
                "## Task\n"
                "Compress these extracts to ≤500 tokens of bullets that "
                "directly help the agent decide. Return as markdown only."
            )
            intents = await self.backend.run(
                prompt,
                agent_name="sage",
                allowed_tools=(),
            )
        except Exception:  # noqa: BLE001
            return snippet
        # The Codex backend currently emits structured intents; if any of
        # them carry markdown body, prefer that — otherwise fall back to
        # the raw KB snippet.
        for intent in intents or []:
            payload = getattr(intent, "payload", {}) or {}
            md = str(payload.get("body_md") or payload.get("markdown") or "")
            if md.strip():
                return md
        return snippet

    async def prefetch(
        self, model_name: str, planned_actions: list[str]
    ) -> None:
        """Warm the cache for the next 5 min window — best-effort, never raises."""
        coros = [self.recall(model_name, a) for a in planned_actions]
        await asyncio.gather(*coros, return_exceptions=True)
