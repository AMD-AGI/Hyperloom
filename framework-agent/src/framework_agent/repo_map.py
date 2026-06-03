"""Canonical mapping from serving framework name to its upstream git repo URL.

Lives in the ``framework_agent`` package so the standalone ``fa`` CLI
does NOT need to reverse-import ``inference_optimizer`` to fetch the
default repo URL when a request omits ``repo_url`` — that import was
breaking the "framework-agent is a standalone package" invariant
called out in the project SKILL: inference_optimizer subprocess-invokes
``fa``, never the other way around.

The mapping covers the three supported frameworks (sglang, vllm, atom);
add new frameworks here as inference_optimizer grows its supported
backend list. IO has its own in-process copy in
``framework_agent_client.repo_url_for_framework`` which falls back to
importing this module when available; the two should never drift — a
sync test in ``framework-agent/tests/test_repo_map.py`` enforces
byte-for-byte equality between the canonical dict and the IO fallback.
"""

from __future__ import annotations

_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
    "atom":   "https://github.com/ROCm/ATOM.git",
}


# Single source of truth for "which framework names does framework-agent
# know about" — opt-in import from any module that previously hardcoded
# the ``{"sglang", "vllm"}`` literal. Derived from the URL dict so a
# new entry above automatically expands the set.
KNOWN_FRAMEWORKS: frozenset[str] = frozenset(_FRAMEWORK_TO_REPO_URL.keys())


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


__all__ = ["KNOWN_FRAMEWORKS", "repo_url_for_framework"]
