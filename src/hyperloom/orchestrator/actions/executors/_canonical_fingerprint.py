# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Canonical variant fingerprint.

A single content-addressed identity for any explore variant so the
``explore_search`` ledger has one canonical key per variant and dedup across
specialist / LLM / default_grid proposals collapses to the same row.

The on-disk fingerprint is content-only: sorted ``(flag, value)`` arg pairs +
sorted ``extra_envs`` pairs, plus the removal/replacement controls
(``remove_args``, ``unset_envs``, ``args_mode``) and a canonicalised
``runtime_override`` when any of those is non-default.  Discriminators such as
framework / tp / workload_signature are kept as side metadata rather than folded
into the hash.

Pairing semantics: repeated flags collapse to the last occurrence (last-wins).
Two arg strings that assign the same set of flag→value bindings, regardless of
order, produce the same fingerprint; strings that assign *different* values to
any flag produce different fingerprints.

**Ledger note**: changing the pairing algorithm invalidates all fingerprint keys
already persisted in ``explore_search.tested``.  Any session resumed after this
change will re-bench its full explored history.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ._grid_server_args import canonical_args_pairs


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


def _canonical_runtime_override(value: Any) -> Any:
    """Order-independent canonical form of a runtime_override payload.

    Sorts dict keys and list entries so semantically-equal overrides fingerprint
    identically; returns ``None`` for an empty/absent override so the hash for a
    plain variant is unchanged (back-compat).
    """
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
    """Return the canonical 16-char fingerprint for a variant.

    Single source of truth for the content hash; ``_grid_runner.variant_fingerprint``
    delegates here. Normalization: args parsed into sorted ``[flag, value]``
    pairs (last-wins for repeated flags); envs ``(str(k), str(v))`` sorted by
    key; removal / replacement controls sorted by value; a non-empty
    ``runtime_override`` folded as an order-independent canonical dict;
    16-char SHA-1.

    Args:
        extra_args: The variant's extra server args string, or ``None``.
        extra_envs: The variant's extra env mapping, or ``None``.
        remove_args: Base/server args to remove before appending
            ``extra_args``.
        unset_envs: Inherited env names to remove before applying
            ``extra_envs``.
        args_mode: ``"append"`` (default) or ``"replace"``.
        runtime_override: Attempt runtime override; folded into the hash only
            when non-empty so plain variants keep their historical fingerprint.

    Returns:
        The canonical 16-char SHA-1 content fingerprint for the variant.
    """
    args_pairs = canonical_args_pairs(str(extra_args or ""))
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
        # Append the runtime override only when present so variants with removal/
        # replacement controls but no override keep their historical fingerprint.
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
    """Return a stable 12-char digest of the workload contract.

    Stored as side metadata on each ``explore_search.tested`` entry so a
    cross-workload resume can warn when an old KEEP came from a different
    (CONC, ISL, OSL, precision, TP). Not part of the fingerprint hash today.
    Args default to the corresponding process env vars when omitted.

    Args:
        conc: Concurrency; defaults to ``$CONC`` when omitted.
        isl: Input sequence length; defaults to ``$ISL`` when omitted.
        osl: Output sequence length; defaults to ``$OSL`` when omitted.
        precision: Precision tag; defaults to ``$PRECISION`` when omitted.
        tp: Tensor-parallel size; defaults to ``$TP`` when omitted.

    Returns:
        A stable 12-char SHA-1 digest of the workload contract.
    """
    fields = {
        "conc": str(conc if conc is not None else os.environ.get("CONC", "")).strip(),
        "isl": str(isl if isl is not None else os.environ.get("ISL", "")).strip(),
        "osl": str(osl if osl is not None else os.environ.get("OSL", "")).strip(),
        "precision": str(precision if precision is not None else os.environ.get("PRECISION", "")).strip(),
        "tp": str(tp if tp is not None else os.environ.get("TP", "")).strip(),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
