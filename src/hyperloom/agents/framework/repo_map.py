# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical mapping from serving framework name to upstream git repo URL."""

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
    """Return the bridging repo URLs to scout for a failure's ``bridge_layer``."""
    return _BRIDGE_LAYER_TO_REPO_URLS.get((bridge_layer or "").strip().lower(), ())


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``."""
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


__all__ = ["KNOWN_FRAMEWORKS", "bridge_repo_urls", "repo_url_for_framework"]
