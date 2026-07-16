# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical mapping from serving framework name to upstream git repo URL.

Lives in ``framework_agent`` so the standalone ``fa`` CLI need not
reverse-import ``inference_optimizer``. IO keeps an in-process copy in
``framework_agent_client.repo_url_for_framework`` that must not drift.
"""

from __future__ import annotations

_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm": "https://github.com/ROCm/vllm.git",
    "atom": "https://github.com/ROCm/ATOM.git",
    "xdit": "https://github.com/xdit-project/xDiT.git",
}


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


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``.

    The lookup is case-insensitive and tolerant of surrounding whitespace.

    Args:
        framework (str): Framework name (e.g. ``"sglang"``, ``"vllm"``,
            ``"atom"``). Compared case-insensitively after stripping.

    Returns:
        str: The canonical git repo URL, or an empty string for unknown
            frameworks; the caller is expected to bail out / log when this
            happens.
    """
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


__all__ = ["KNOWN_FRAMEWORKS", "bridge_repo_urls", "repo_url_for_framework"]
