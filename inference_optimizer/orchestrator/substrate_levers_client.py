# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Optional client for the substrate KB ``/v2/reasoning/levers`` endpoint.

The forward (warm-start) counterpart to the Critic's ``/v2/reasoning/assess``
call: instead of judging a proposal after the fact, a specialist can be *steered*
before it spends a trial. The Coordinator queries this endpoint once per focus
(model / hardware / framework) and folds the substrate's known levers — each
labelled ``beneficial`` / ``neutral`` / ``harmful`` with provenance — into the
specialist prompt as an advisory block. It is never a gate.

Config mirrors :mod:`inference_optimizer.recipe_kb.remote_client` and the
Critic's :mod:`critic-agent.runtime.kb_assess_client`: ``CORTEX_KB_URL`` (URL,
no default — no URL means no network), ``CORTEX_KB_HTTP_TIMEOUT_SEC`` (timeout),
``KB_SERVICE_TOKEN`` (bearer).

Uses ``urllib`` (no ``httpx``) so it installs in the minimal container. All
failures are swallowed — the digest is advisory enrichment, never required.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

LEVERS_PATH = "/v2/reasoning/levers"
DEFAULT_TIMEOUT_SEC = 3.0


class SubstrateLeversClient:
    """Minimal sync POST client for ``/v2/reasoning/levers`` (best-effort)."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        """Configure the transport.

        Args:
            base_url (str): KB service base URL; trailing slash is stripped.
            token (str | None): Bearer token; falls back to ``KB_SERVICE_TOKEN``.
            timeout_sec (float): Per-request timeout in seconds.

        Raises:
            ValueError: If ``base_url`` is empty.
        """
        if not base_url:
            raise ValueError("SubstrateLeversClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_sec = float(timeout_sec)

    @classmethod
    def from_env(cls) -> "SubstrateLeversClient | None":
        """Build a client from env, or ``None`` when no cortex KB is configured.

        Reuses recipe-snapshot's config: ``CORTEX_KB_URL`` (no default by
        design) and ``CORTEX_KB_HTTP_TIMEOUT_SEC`` for the per-request timeout.

        Returns:
            SubstrateLeversClient | None: A client when ``CORTEX_KB_URL`` is
            set, else ``None``.
        """
        url = (os.environ.get("CORTEX_KB_URL") or "").strip()
        if not url:
            return None
        try:
            timeout = float(os.environ.get("CORTEX_KB_HTTP_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC
        return cls(url, timeout_sec=timeout)

    def recommend(self, *, focus: dict[str, Any]) -> dict[str, Any] | None:
        """POST one focus to ``/v2/reasoning/levers``.

        Args:
            focus (dict[str, Any]): Subject of the run; at least
                ``{"model": ...}``; optional ``hardware`` / ``framework`` /
                ``framework_version`` / ``precision``.

        Returns:
            dict[str, Any] | None: The decoded digest (``focus`` / ``seed`` /
            ``summary`` / ``levers``), or ``None`` on any error (best-effort;
            never raises).
        """
        if not isinstance(focus, dict) or not str(focus.get("model") or "").strip():
            return None
        body = {"focus": focus}
        url = f"{self.base_url}{LEVERS_PATH}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                payload = resp.read().decode("utf-8") or "{}"
                return json.loads(payload) if payload else {}
        except (urllib.error.URLError, ValueError, OSError) as exc:
            log.warning("specialist: KB levers call failed (%s): %r", url, exc)
            return None


__all__ = ["SubstrateLeversClient", "LEVERS_PATH", "DEFAULT_TIMEOUT_SEC"]
