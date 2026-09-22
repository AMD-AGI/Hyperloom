# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side server-flag denylist.

stdlib-only; shipped beside pod launcher scripts. Keep constants in sync
with multi_node/_internal/server_args_safety.py (the host-side authority).
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

_DENIED_SERVER_FLAGS: frozenset[str] = frozenset(
    {
        "--adapter-model-path",
        "--adapter-path",
        "--allowed-local-media-path",
        "--chat-template",
        "--code-revision",
        "--config",
        "--download-dir",
        "--hf-overrides",
        "--lora-dirs",
        "--lora-modules",
        "--lora-path",
        "--lora-paths",
        "--model",
        "--model-id",
        "--model-path",
        "--quantization-param-path",
        "--revision",
        "--tokenizer",
        "--tokenizer-path",
        "--tokenizer-revision",
    }
)

_DENIED_SERVER_FLAG_SUFFIXES: tuple[str, ...] = ("-dir", "-file", "-path")

_SUFFIX_EXEMPT_SERVER_FLAGS: frozenset[str] = frozenset({"--speculative-draft-model-path"})


def _is_denied_server_flag(flag: str) -> bool:
    """Return whether a single ``--flag`` token is denied at the pod boundary."""
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_SERVER_FLAGS:
        return True
    if name in _SUFFIX_EXEMPT_SERVER_FLAGS:
        return False
    return any(name.endswith(suffix) for suffix in _DENIED_SERVER_FLAG_SUFFIXES)


def _unsafe_path_value_reason(value: str | None) -> str:
    """Return why an exempt flag's path value is unsafe (empty string when acceptable)."""
    val = (value or "").strip()
    if not val:
        return "missing value"
    if not val.startswith("/"):
        return "must be an absolute path, not a repo id or URI"
    if ".." in PurePosixPath(val).parts:
        return "must not traverse with '..'"
    return ""


def _flag_value_pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    """Return ``(flag, value)`` pairs for both ``--flag=value`` and ``--flag value``."""
    pairs: list[tuple[str, str | None]] = []
    for idx, tok in enumerate(tokens):
        if not tok.startswith("--"):
            continue
        if "=" in tok:
            name, _, val = tok.partition("=")
            pairs.append((name, val))
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        pairs.append((tok, None if (nxt is None or nxt.startswith("--")) else nxt))
    return pairs


def _denied_extra_args(raw: str) -> list[str]:
    """Return rejected CLI flags in a pod-side extra-args string."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return ["<unparseable>"]
    out: list[str] = []
    for flag, value in _flag_value_pairs(tokens):
        if _is_denied_server_flag(flag):
            if flag not in out:
                out.append(flag)
            continue
        if flag not in _SUFFIX_EXEMPT_SERVER_FLAGS:
            continue
        reason = _unsafe_path_value_reason(value)
        entry = f"{flag}: {reason}"
        if reason and entry not in out:
            out.append(entry)
    return out
