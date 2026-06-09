# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Canonical mapping from serving framework name to upstream git repo URL.

Lives in ``framework_agent`` so the standalone ``fa`` CLI need not
reverse-import ``inference_optimizer`` (which breaks the standalone-package
invariant: IO subprocess-invokes ``fa``, never the reverse). IO keeps an
in-process copy in ``framework_agent_client.repo_url_for_framework``; the two
must not drift — ``tests/test_repo_map.py`` enforces byte-for-byte equality.
"""

from __future__ import annotations

_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
    "atom":   "https://github.com/ROCm/ATOM.git",
}


# Single source of truth for known framework names; derived from the URL
# dict so a new entry above automatically expands the set.
KNOWN_FRAMEWORKS: frozenset[str] = frozenset(_FRAMEWORK_TO_REPO_URL.keys())


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``.

    Returns an empty string for unknown frameworks; the caller is
    expected to bail out / log when this happens.
    """
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


__all__ = ["KNOWN_FRAMEWORKS", "repo_url_for_framework"]
