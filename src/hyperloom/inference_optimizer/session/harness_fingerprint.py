# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which harness a session ran: code revision, prompt-text digests, per-role model settings.

Stamped into ``manifest.json`` so a session's cost and score can be attributed to the exact
text / code / model set that produced them (the meta-RSI ledger reads it back). Only model
names, effort levels and endpoint hostnames are recorded; never keys or tokens.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_PKG_ROOT = Path(__file__).resolve().parents[2]

# Prompt text on disk: system prompts, action cards, review checklists, references.
_TEXT_GLOBS: tuple[str, ...] = (
    "orchestrator/prompts/*.md",
    "orchestrator/prompts/references/*.md",
    "agents/*/actions/*.md",
    "agents/*/references/*.md",
    "agents/*/SKILL.md",
    "inference_optimizer/actions/*.md",
    "inference_optimizer/references/*.md",
)

# Modules whose string literals are prompt text (the per-tick prompt is assembled here).
_PROMPT_CODE: tuple[str, ...] = (
    "orchestrator/prompts/prompt_builder.py",
    "orchestrator/prompts/specialist_prompt_builder.py",
    "orchestrator/loop/conversation.py",
)

_MODEL_ENV: tuple[str, ...] = (
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "GEAK_CLAUDE_MODEL",
    "FORGE_AGENT_MODEL",
    "FORGE_AGENT_BACKEND",
    "FORGE_AGENT_REASONING_EFFORT",
    "HYPERLOOM_REASONING_EFFORT",
    "OPENAI_REASONING_EFFORT",
    "INFERENCE_OPTIMIZER_CLAUDE_EFFORT",
    "INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT",
    "INFERENCE_OPTIMIZER_CLAUDE_KERNEL_EFFORT",
    "INFERENCE_OPTIMIZER_CLAUDE_THINKING",
    "HYPERLOOM_ROLE_MODELS",
)

_ENDPOINT_ENV: tuple[str, ...] = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", "-C", str(_PKG_ROOT), *args], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return ""
    return out.stdout.decode("utf-8", "replace") if out.returncode == 0 else ""


def code_fingerprint() -> dict[str, Any]:
    """Revision of the checkout, and a digest of any uncommitted change to tracked files."""
    revision = _git(["rev-parse", "--short", "HEAD"]).strip()
    diff = _git(["diff", "HEAD", "--", "."])
    return {
        "revision": revision or None,
        "dirty": bool(diff.strip()),
        "diff_sha": _sha(diff.encode()) if diff.strip() else None,
    }


def text_fingerprint(root: Path = _PKG_ROOT) -> dict[str, Any]:
    """Per-file digests of the prompt text, plus one combined digest to compare runs by."""
    files: dict[str, str] = {}
    for pattern in _TEXT_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                files[str(path.relative_to(root))] = _sha(path.read_bytes())
    for rel in _PROMPT_CODE:
        path = root / rel
        if path.is_file():
            files[rel] = _sha(path.read_bytes())
    combined = _sha("\n".join(f"{k}:{v}" for k, v in sorted(files.items())).encode())
    return {"digest": combined, "files": files}


def model_fingerprint(env: dict[str, str] | None = None, role_models: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model names and effort settings in force, and the hostnames of the LLM endpoints."""
    source = os.environ if env is None else env
    settings = {k: source[k] for k in _MODEL_ENV if source.get(k)}
    endpoints = {}
    for k in _ENDPOINT_ENV:
        raw = (source.get(k) or "").strip()
        if raw:
            endpoints[k] = urlparse(raw).hostname or raw.split("/")[0]
    return {"env": settings, "endpoints": endpoints, "role_models": role_models or {}}


def build_harness_fingerprint(role_models: dict[str, Any] | None = None) -> dict[str, Any]:
    """The three levers of the outer loop, as they stood when the session started."""
    return {
        "code": code_fingerprint(),
        "text": text_fingerprint(),
        "model": model_fingerprint(role_models=role_models),
    }


__all__ = [
    "build_harness_fingerprint",
    "code_fingerprint",
    "model_fingerprint",
    "text_fingerprint",
]
