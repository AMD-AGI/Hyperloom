"""Canonical variant fingerprint — v0.8 M3.

A single, content-addressed identity for any explore variant. Replaces
the legacy-era pair of fingerprint helpers (one inside ``params.py``, one
inside ``backends.py``) so that:

* the ``explore_search`` ledger has exactly one canonical key per variant,
* dedup across specialist / LLM / default_grid proposals collapses to
  the same row regardless of origin, and
* v0.6 ``backends_search`` / ``params_search`` ledgers migrate into
  ``explore_search`` losslessly (same hash for the same content).

Design rationale
----------------

KB_design §3.4 §4.2 (Inv-4.2) specifies:

    canonical_fingerprint = sha1(sorted(extra_args) + sorted(extra_envs)
                                 + framework + tp + workload_signature)

For M3, we intentionally keep the on-disk fingerprint **content-only**
(``sorted(extra_args)`` + ``sorted(extra_envs)``) to preserve hash
identity with v0.6 ``params_search`` / ``backends_search.tested`` keys
during resume migration. The optional framework / tp / workload
discriminators are stored as **side metadata** on each ledger entry
(``framework``, ``tp``, ``workload_signature``) so context is preserved
without re-hashing the existing universe.

Future milestones (M5/M6 with specialist provenance, M7 with workload
sweeps) MAY tighten this to include workload_signature in the hash; at
that point a second migration step will re-key old rows.
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

    Identical in shape to ``_grid_runner.variant_fingerprint`` — kept as
    a separate symbol so call-sites in ``explore.py`` / SharedState
    migration depend on the legacy canonical identity rather than the
    v0.6 helper. Both functions intentionally produce the SAME hash
    for the SAME inputs so the legacy → ledger merge is lossless.

    Normalization
    -------------
    * args: ``shlex.split`` → sorted token tuple.
    * envs: ``(str(k), str(v))`` pairs sorted by key.
    * 16-char SHA-1 prefix (collision-resistant for per-session ledger).
    """
    args_text = str(extra_args or "")
    try:
        args_tokens = sorted(shlex.split(args_text))
    except ValueError:
        # Unbalanced quotes / shell-parse failure: fall back to a stable
        # whitespace split so we still produce *some* fingerprint
        # instead of crashing a propose pre-flight. Two identically
        # malformed strings still collide; differently malformed
        # strings stay distinguishable.
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
    cross-workload resume can warn when an old KEEP came from a
    different (CONC, ISL, OSL, precision, TP). Not part of the
    fingerprint hash today (see module docstring).

    Args default to the corresponding process env vars (``CONC`` / ``ISL``
    / ``OSL`` / ``PRECISION`` / ``TP``) when omitted, matching the
    Magpie ``benchmark.envs`` materialization path.
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
