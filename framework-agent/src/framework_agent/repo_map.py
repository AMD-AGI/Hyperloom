"""Canonical mapping from serving framework name to its upstream git repo URL.

Lives in the ``framework_agent`` package so the standalone ``fa`` CLI
does NOT need to reverse-import ``inference_optimizer`` to fetch the
default repo URL when a request omits ``repo_url`` — that import was
breaking the "framework-agent is a standalone package" invariant
called out in the project SKILL: inference_optimizer subprocess-invokes
``fa``, never the other way around.

The mapping is intentionally tiny (sglang, vllm); add new frameworks
here as inference_optimizer grows its supported backend list. IO has
its own in-process copy in ``framework_agent_client.repo_url_for_framework``
which falls back to importing this module when available; the two
should never drift.
"""

from __future__ import annotations

_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
}


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``.

    Returns an empty string for unknown frameworks; the caller is
    expected to bail out / log when this happens.
    """
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


__all__ = ["repo_url_for_framework"]
