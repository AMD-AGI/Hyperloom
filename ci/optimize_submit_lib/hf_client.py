from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import config as _config

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})

# ── HuggingFace client ──────────────────────────────────────────────────────────


class HuggingFaceClient:
    """Minimal HF API client for model metadata + top-models discovery."""

    BASE = "https://huggingface.co"

    def __init__(self, token: str = "", timeout: int = 15, tokens: list[str] | None = None, seed: str = ""):
        """Initialise the HF client session.

        Supports a *pool* of HuggingFace tokens. Requests start on a token chosen
        by hashing ``seed`` (so parallel CI jobs spread their first hit across the
        pool) and, on HTTP 429 (Too Many Requests), transparently rotate to the
        next token with exponential backoff before retrying. This mitigates the
        bursty rate-limiting that hits the resolve/config.json endpoint when many
        optimize jobs fan out at once.

        Args:
            token (str): Primary HuggingFace token for gated-model access.
            timeout (int): Per-request timeout in seconds.
            tokens (list[str] | None): Additional tokens to alternate with.
            seed (str): Per-job string used to pick the starting token.
        """
        self.timeout = timeout
        pool: list[str] = []
        for t in [token, *(tokens or [])]:
            if t and t not in pool:
                pool.append(t)
        self._tokens = pool
        if pool:
            self._idx = (hash(seed) % len(pool)) if seed else 0
        else:
            self._idx = 0
        self._sess = requests.Session()
        self._sess.headers["User-Agent"] = "hyperloom-optimize-submit/1.0"
        self._apply_token()

    def _apply_token(self) -> None:
        """Set the Authorization header to the currently-selected token."""
        if self._tokens:
            tok = self._tokens[self._idx % len(self._tokens)]
            self._sess.headers["Authorization"] = f"Bearer {tok}"
        else:
            self._sess.headers.pop("Authorization", None)

    def _get(self, path: str) -> dict | list:
        """GET a HuggingFace API path and return the parsed JSON body.

        On HTTP 429 the client rotates to the next token in the pool and retries
        with exponential backoff (capped at 30s); other errors raise immediately.

        Args:
            path (str): Path appended to the HF base URL.

        Returns:
            dict | list: The decoded JSON response.

        Raises:
            requests.HTTPError: If the response status indicates an error.
        """
        attempts = max(4, len(self._tokens) * 2)
        last_exc: Exception | None = None
        for i in range(attempts):
            resp = self._sess.get(f"{self.BASE}{path}", timeout=self.timeout)
            if resp.status_code == 429:
                last_exc = requests.HTTPError(
                    f"429 Client Error: Too Many Requests for url: {self.BASE}{path}", response=resp
                )
                if len(self._tokens) > 1:
                    self._idx += 1
                    self._apply_token()
                time.sleep(min(2 ** i, 30))
                continue
            resp.raise_for_status()
            return resp.json()
        if last_exc is not None:
            raise last_exc
        raise requests.HTTPError(f"exhausted retries for url: {self.BASE}{path}")

    def model_info(self, repo_id: str) -> dict:
        """Fetch a repo's model metadata from the HF API.

        Args:
            repo_id (str): HuggingFace repo id (e.g. ``Qwen/Qwen3-8B``).

        Returns:
            dict: The model-info JSON (pipeline_tag, safetensors, tags, etc.).
        """
        return self._get(f"/api/models/{repo_id}")  # type: ignore[return-value]

    def model_config(self, repo_id: str) -> dict:
        """Fetch a repo's ``config.json`` from the HF resolve endpoint.

        Args:
            repo_id (str): HuggingFace repo id.

        Returns:
            dict: The parsed ``config.json`` contents.
        """
        return self._get(f"/{repo_id}/resolve/main/config.json")  # type: ignore[return-value]

    def top_models(self, limit: int, min_params_b: float = 0.0) -> list[str]:
        """Return top-N text-generation repos by downloads, optionally size-filtered.

        Pool-then-filter: the listing API matches on tags only, so re-validate
        per-repo on a generative architectures[0] suffix; failing that (or a
        gated 401) → skip. The pipeline_tag != text-generation gate has been
        removed so multimodal/other heads are allowed through.

        Args:
            limit (int): Maximum number of repos to return.
            min_params_b (float): Minimum parameter count in billions; ``0``
                disables the size filter.

        Returns:
            list[str]: Up to ``limit`` generative repo ids ordered by downloads.
        """
        pool_size = max(limit * 10, 100)
        listing = self._get(f"/api/models?sort=downloads&direction=-1&limit={pool_size}&filter=text-generation")
        repos: list[str] = []
        for m in listing:  # type: ignore[union-attr]
            if len(repos) >= limit:
                break
            repo = m.get("modelId") or m.get("id", "")
            if not repo or "/" not in repo:
                continue

            try:
                info = self.model_info(repo)
            except Exception:
                continue  # gated / network error

            if min_params_b > 0:
                total = (info.get("safetensors") or {}).get("total", 0)
                if (total / 1e9) < min_params_b:
                    continue

            # Final gate: config.json reachable AND architectures[0] generative.
            # Skip gated (401/403) or non-generative repos so the pool refills.
            try:
                cfg = self.model_config(repo)
            except Exception as e:
                log.info(
                    "skip %s: config.json unreachable (%s) — likely a gated repo your HF_TOKEN hasn't been granted",
                    repo,
                    e,
                )
                continue
            arch = (cfg.get("architectures") or [""])[0]
            if not is_generative_arch(arch):
                log.info("skip %s: arch=%s is non-generative", repo, arch)
                continue

            repos.append(repo)
        return repos
