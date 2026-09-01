# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The record that decides whether a generated tuner is ever trusted.

A generated script starts as a *candidate*: sandboxed on every use, re-timed on
every use, and never registered anywhere. Promotion to *trusted* means it may be
driven like any other tuner -- so it is deliberately hard, and hard in ways that
match how these scripts fail.

The bar is three independent successes across at least two models, no recorded
regression, and a human sign-off. Each clause answers a specific failure:

* **Three successes** because one is a coincidence. Timing on a shared box moved
  2.5x between two readings of the same configuration.
* **Two models** because a script can encode one checkpoint's shapes and look
  perfect until it meets another.
* **No regression, ever** -- a single measured loss demotes it back. A tuner
  that is usually right is worse than none: it is trusted precisely when nobody
  is checking.
* **Human sign-off**, because everything above is a machine agreeing with a
  machine, and this is the point where the script stops being re-checked.

The ledger holds no code. It records what happened to a script identified by the
hash of its contents, so an edited script is a different script and starts over.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_SUCCESSES = 3
REQUIRED_MODELS = 2
# Promotion is an operator action, never a consequence of enough green runs.
TRUST_ENV = "FORGE_TIER3_TRUSTED"


@dataclass
class TunerRecord:
    """One generated script's history."""

    digest: str
    table: str
    successes: int = 0
    regressions: int = 0
    models: list[str] = field(default_factory=list)
    last_speedup: float | None = None
    first_seen: str = ""
    last_seen: str = ""

    @property
    def eligible_for_trust(self) -> bool:
        """Whether it has earned a *review*. Never whether it is trusted."""
        return (
            self.regressions == 0 and self.successes >= REQUIRED_SUCCESSES and len(set(self.models)) >= REQUIRED_MODELS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "table": self.table,
            "successes": self.successes,
            "regressions": self.regressions,
            "models": list(self.models),
            "last_speedup": self.last_speedup,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "eligible_for_trust": self.eligible_for_trust,
        }


def script_digest(script: Path | str) -> str:
    """Content hash. An edited script is a new script with no history."""
    try:
        return hashlib.sha256(Path(script).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _load(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record_outcome(
    ledger_path: Path,
    *,
    digest: str,
    table: str,
    model: str,
    improved: bool,
    speedup: float | None,
) -> TunerRecord:
    """Add one use to a script's history and return the updated record."""
    now = datetime.now(timezone.utc).isoformat()
    data = _load(ledger_path)
    raw = data.get(digest) or {}
    record = TunerRecord(
        digest=digest,
        table=str(raw.get("table") or table),
        successes=int(raw.get("successes") or 0),
        regressions=int(raw.get("regressions") or 0),
        models=list(raw.get("models") or []),
        first_seen=str(raw.get("first_seen") or now),
    )
    if improved:
        record.successes += 1
        if model and model not in record.models:
            record.models.append(model)
    else:
        # Not every non-improvement is a regression: finding nothing is a valid
        # outcome. Only a measured loss counts against the script.
        if speedup is not None and speedup < 1.0:
            record.regressions += 1
    record.last_speedup = speedup
    record.last_seen = now

    data[digest] = record.to_dict()
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log.warning("could not update the tier3 ledger at %s: %s", ledger_path, exc)

    if record.eligible_for_trust:
        log.warning(
            "tier3: generated tuner %s for %s has met the bar for review "
            "(%d successes across %d models, no regressions). It stays a "
            "candidate until an operator adds it to %s.",
            digest,
            record.table,
            record.successes,
            len(set(record.models)),
            TRUST_ENV,
        )
    return record


def is_trusted(digest: str) -> bool:
    """Whether an operator has signed this exact script off.

    Reads a list of digests. Eligibility never grants this: the ledger can say a
    script has earned a look, and only a person can say it has earned trust.
    """
    raw = os.environ.get(TRUST_ENV, "").strip()
    if not raw or not digest:
        return False
    return digest in {d.strip() for d in raw.split(",") if d.strip()}
