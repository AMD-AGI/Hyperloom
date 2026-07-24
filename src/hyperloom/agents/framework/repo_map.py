# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical mapping from serving framework name to upstream git repo URL.

Lives in ``framework_agent`` so the standalone ``fa`` CLI need not
reverse-import ``inference_optimizer``. IO keeps an in-process copy in
``framework_agent_client.repo_url_for_framework`` that must not drift.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm": "https://github.com/ROCm/vllm.git",
    "atom": "https://github.com/ROCm/ATOM.git",
    "xdit": "https://github.com/xdit-project/xDiT.git",
}

VLLM_REPO_URL_ENV = "HYPERLOOM_VLLM_REPO_URL"
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


# Known framework names, derived from the URL dict.
KNOWN_FRAMEWORKS: frozenset[str] = frozenset(_FRAMEWORK_TO_REPO_URL.keys())


# Enablement bridging repos, keyed by ``bridge_layer``.
_BRIDGE_LAYER_TO_REPO_URLS: dict[str, tuple[str, ...]] = {
    "rocm_hip": (
        "https://github.com/ROCm/aiter.git",
        "https://github.com/ROCm/HIP.git",
        "https://github.com/ROCm/ROCm.git",
    ),
    "build": ("https://github.com/ROCm/aiter.git",),
}


def bridge_repo_urls(bridge_layer: str) -> tuple[str, ...]:
    """Return the bridging repo URLs to scout for a failure's ``bridge_layer``.

    The lookup is case-insensitive and whitespace-tolerant.

    Args:
        bridge_layer (str): The ``bridge_layer`` tag (e.g. ``"rocm_hip"``,
            ``"build"``). ``"framework"`` returns ``()``.

    Returns:
        tuple[str, ...]: Bridge repo URLs (empty for ``"framework"`` /
            unknown layers).
    """
    return _BRIDGE_LAYER_TO_REPO_URLS.get((bridge_layer or "").strip().lower(), ())


def canonical_github_repo_url(value: str) -> str:
    """Normalize a GitHub repository reference to one HTTPS clone URL."""

    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("git@github.com:"):
        repo = raw.split(":", 1)[1]
    elif "://" in raw:
        parsed = urlsplit(raw)
        if parsed.hostname is None or parsed.hostname.lower() != "github.com":
            raise ValueError(f"repository override must target github.com, got {value!r}")
        repo = parsed.path
    else:
        repo = raw
    repo = repo.strip("/")
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not _GITHUB_REPO_RE.fullmatch(repo):
        raise ValueError(f"repository override must be OWNER/REPO, got {value!r}")
    owner, name = repo.split("/", 1)
    return f"https://github.com/{owner}/{name}.git"


def github_repo_name(value: str) -> str:
    """Return ``OWNER/REPO`` for a valid GitHub repository reference."""

    canonical = canonical_github_repo_url(value)
    return canonical.split("github.com/", 1)[1][:-4] if canonical else ""


def default_repo_url_for_framework(framework: str) -> str:
    """Return the built-in GitHub repo URL for ``framework``."""

    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


def repo_url_for_framework(framework: str) -> str:
    """Return the effective GitHub repo URL for ``framework``.

    The lookup is case-insensitive and tolerant of surrounding whitespace.
    vLLM operators may override the AMD-fork default with
    ``HYPERLOOM_VLLM_REPO_URL``.

    Args:
        framework (str): Framework name (e.g. ``"sglang"``, ``"vllm"``,
            ``"atom"``). Compared case-insensitively after stripping.

    Returns:
        str: The canonical git repo URL, or an empty string for unknown
            frameworks; the caller is expected to bail out / log when this
            happens.
    """
    normalized = (framework or "").strip().lower()
    if normalized == "vllm":
        override = os.environ.get(VLLM_REPO_URL_ENV, "").strip()
        if override:
            return canonical_github_repo_url(override)
    return default_repo_url_for_framework(normalized)


__all__ = [
    "KNOWN_FRAMEWORKS",
    "VLLM_REPO_URL_ENV",
    "bridge_repo_urls",
    "canonical_github_repo_url",
    "default_repo_url_for_framework",
    "github_repo_name",
    "repo_url_for_framework",
]
