# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical variant fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from typing import Any


__all__ = [
    "canonical_fingerprint",
    "workload_signature",
]


def _coerce_list(value: Any) -> list[str]:
    """Normalize optional list-like fingerprint inputs."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)] if str(value).strip() else []


def _is_flag(token: str) -> bool:
    """True when ``token`` looks like a CLI flag rather than a value."""
    if not token.startswith("-"):
        return False
    stripped = token.lstrip("-")
    if not stripped:
        return False
    return not stripped[0].isdigit()


def _args_pairs(args_text: str) -> list[list[str]]:
    """Return sorted last-wins ``[flag, value]`` / ``[flag]`` pairs for args."""
    try:
        tokens = shlex.split(args_text)
    except ValueError:
        # Shell-parse failure: fall back to whitespace split.
        tokens = args_text.split()
    last: dict[str, list[str]] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if not _is_flag(token):
            last[token] = [token]
        elif "=" in token:
            flag, _, value = token.partition("=")
            last[flag] = [flag, value]
        elif i < len(tokens) and not _is_flag(tokens[i]):
            last[token] = [token, tokens[i]]
            i += 1
        else:
            last[token] = [token]
    return sorted(last.values(), key=lambda pair: pair[0])


def _canonical_runtime_override(value: Any) -> Any:
    """Order-independent canonical form of a runtime_override payload."""
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, Any] = {}
    for k in sorted(value):
        v = value[k]
        if isinstance(v, (list, tuple)):
            out[str(k)] = sorted(str(x) for x in v)
        elif isinstance(v, dict):
            out[str(k)] = {str(ik): str(iv) for ik, iv in sorted(v.items())}
        else:
            out[str(k)] = str(v)
    return out


def canonical_fingerprint(
    extra_args: str | None,
    extra_envs: dict[str, Any] | None,
    *,
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None,
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None,
    args_mode: str = "append",
    runtime_override: dict[str, Any] | None = None,
) -> str:
    """Return the canonical 16-char fingerprint for a variant."""
    args_pairs = _args_pairs(str(extra_args or ""))
    env_pairs = sorted((str(k), str(v)) for k, v in (extra_envs or {}).items())
    mode = str(args_mode or "append").strip().lower()
    if mode not in {"append", "replace"}:
        mode = "append"
    remove_list = sorted(_coerce_list(remove_args))
    unset_list = sorted(_coerce_list(unset_envs))
    rt = _canonical_runtime_override(runtime_override)
    if not remove_list and not unset_list and mode == "append" and rt is None:
        payload_obj: Any = [args_pairs, [list(p) for p in env_pairs]]
    else:
        payload_obj = [
            args_pairs,
            [list(p) for p in env_pairs],
            remove_list,
            unset_list,
            mode,
        ]
        # Append the runtime override only when present so variants with removal/ replacement controls but no override
        # keep their historical fingerprint.
        if rt is not None:
            payload_obj.append(rt)
    payload = json.dumps(
        payload_obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def workload_signature(
    *,
    conc: int | str | None = None,
    isl: int | str | None = None,
    osl: int | str | None = None,
    precision: str | None = None,
    tp: int | str | None = None,
) -> str:
    """Return a stable 12-char digest of the workload contract."""
    fields = {
        "conc": str(conc if conc is not None else os.environ.get("CONC", "")).strip(),
        "isl": str(isl if isl is not None else os.environ.get("ISL", "")).strip(),
        "osl": str(osl if osl is not None else os.environ.get("OSL", "")).strip(),
        "precision": str(precision if precision is not None else os.environ.get("PRECISION", "")).strip(),
        "tp": str(tp if tp is not None else os.environ.get("TP", "")).strip(),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
