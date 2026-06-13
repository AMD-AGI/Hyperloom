# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Optional client for the substrate KB ``/v2/reasoning/assess`` endpoint.

The Critic can *associate* each proposal's raw optimisation levers (server
args / env vars / params) to the knowledge substrate's calibrated mechanism
evidence to judge whether the proposal is reasonable. This is strictly
best-effort enrichment that reuses the **same** cortex KB endpoint the
recipe-snapshot integration already points at — there is no separate Critic
KB URL. It honours recipe-snapshot's "no URL = no network" contract:

* ``CORTEX_KB_URL`` set (or explicit ``base_url``) → POST per proposal to
  ``{CORTEX_KB_URL}/v2/reasoning/assess`` and fold the verdict into the bundle;
* unset → :func:`from_env` returns ``None`` and the Critic never calls out.

Config mirrors :mod:`inference_optimizer.recipe_kb.remote_client`:
``CORTEX_KB_URL`` (URL), ``CORTEX_KB_HTTP_TIMEOUT_SEC`` (timeout),
``KB_SERVICE_TOKEN`` (bearer).

Uses ``urllib`` (no ``httpx``) so it installs in the minimal Codex container,
consistent with :mod:`runtime.kb_client`. All failures are swallowed — the
verdict is advisory, never a gate on the review.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

ASSESS_PATH = "/v2/reasoning/assess"
DEFAULT_TIMEOUT_SEC = 3.0


class KBAssessClient:
    """Minimal sync POST client for ``/v2/reasoning/assess`` (best-effort)."""

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
            raise ValueError("KBAssessClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_sec = float(timeout_sec)

    @classmethod
    def from_env(cls) -> "KBAssessClient | None":
        """Build a client from env, or ``None`` when no cortex KB is configured.

        Reuses recipe-snapshot's config: ``CORTEX_KB_URL`` (no default by
        design — the Critic never silently connects to a remote KB) and
        ``CORTEX_KB_HTTP_TIMEOUT_SEC`` for the per-request timeout.

        Returns:
            KBAssessClient | None: A client when ``CORTEX_KB_URL`` is set,
            else ``None``.
        """
        url = (os.environ.get("CORTEX_KB_URL") or "").strip()
        if not url:
            return None
        try:
            timeout = float(
                os.environ.get("CORTEX_KB_HTTP_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC))
            )
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC
        return cls(url, timeout_sec=timeout)

    def assess(
        self,
        *,
        focus: dict[str, Any],
        params: dict[str, Any] | None = None,
        envs: dict[str, Any] | None = None,
        args: str = "",
    ) -> dict[str, Any] | None:
        """POST one proposal to ``/v2/reasoning/assess``.

        Args:
            focus (dict[str, Any]): Subject of the proposal; at least
                ``{"model": ...}``.
            params (dict[str, Any] | None): Numeric/string levers.
            envs (dict[str, Any] | None): Environment-variable levers.
            args (str): CLI arg string.

        Returns:
            dict[str, Any] | None: The decoded assessment, or ``None`` on any
            error (best-effort; never raises).
        """
        body: dict[str, Any] = {"focus": focus, "args": args or ""}
        if params:
            body["params"] = params
        if envs:
            body["envs"] = envs

        url = f"{self.base_url}{ASSESS_PATH}"
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
            log.warning("critic: KB assess call failed (%s): %r", url, exc)
            return None


__all__ = ["KBAssessClient", "ASSESS_PATH", "DEFAULT_TIMEOUT_SEC"]
