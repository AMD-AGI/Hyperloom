# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Canonical variant fingerprint — v0.8 M3.

A single content-addressed identity for any explore variant so the
``explore_search`` ledger has one canonical key per variant and dedup across
specialist / LLM / default_grid proposals collapses to the same row.

KB_design §3.4 §4.2 (Inv-4.2): the canonical hash could include framework / tp
/ workload_signature, but for M3 the on-disk fingerprint is content-only
(``sorted(extra_args)`` + ``sorted(extra_envs)``); the discriminators are kept
as side metadata. Future milestones may fold workload_signature into the hash
(with a re-key migration).
"""

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


def canonical_fingerprint(
    extra_args: str | None,
    extra_envs: dict[str, Any] | None,
) -> str:
    """Return the canonical 16-char fingerprint for a variant.

    Produces the SAME hash as ``_grid_runner.variant_fingerprint`` for the same
    inputs (lossless legacy → ledger merge); kept separate so call-sites depend
    on the legacy canonical identity. Normalization: args ``shlex.split`` →
    sorted tokens; envs ``(str(k), str(v))`` sorted by key; 16-char SHA-1.
    """
    args_text = str(extra_args or "")
    try:
        args_tokens = sorted(shlex.split(args_text))
    except ValueError:
        # Shell-parse failure: fall back to whitespace split so we still
        # produce a fingerprint (identical bad strings still collide).
        args_tokens = sorted(args_text.split())
    env_pairs = sorted(
        (str(k), str(v)) for k, v in (extra_envs or {}).items()
    )
    payload = json.dumps(
        [args_tokens, [list(p) for p in env_pairs]],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


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
    """
    fields = {
        "conc": str(conc if conc is not None else os.environ.get("CONC", "")).strip(),
        "isl": str(isl if isl is not None else os.environ.get("ISL", "")).strip(),
        "osl": str(osl if osl is not None else os.environ.get("OSL", "")).strip(),
        "precision": str(
            precision if precision is not None
            else os.environ.get("PRECISION", "")
        ).strip(),
        "tp": str(tp if tp is not None else os.environ.get("TP", "")).strip(),
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
