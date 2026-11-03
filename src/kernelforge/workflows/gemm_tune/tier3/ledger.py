# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The record that decides whether a generated tuner is ever trusted."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom_common.timeutil import now_iso

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


@dataclass(frozen=True)
class TunerUse:
    """One use of a generated script: which script, for which table and model, and how it fared."""

    digest: str
    table: str
    model: str
    improved: bool
    speedup: float | None


def record_outcome(ledger_path: Path, use: TunerUse) -> TunerRecord:
    """Add one use to a script's history and return the updated record."""
    now = now_iso("auto")
    data = _load(ledger_path)
    raw = data.get(use.digest) or {}
    record = TunerRecord(
        digest=use.digest,
        table=str(raw.get("table") or use.table),
        successes=int(raw.get("successes") or 0),
        regressions=int(raw.get("regressions") or 0),
        models=list(raw.get("models") or []),
        first_seen=str(raw.get("first_seen") or now),
    )
    if use.improved:
        record.successes += 1
        if use.model and use.model not in record.models:
            record.models.append(use.model)
    else:
        # Not every non-improvement is a regression: finding nothing is a valid outcome.
        if use.speedup is not None and use.speedup < 1.0:
            record.regressions += 1
    record.last_speedup = use.speedup
    record.last_seen = now

    data[use.digest] = record.to_dict()
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
            use.digest,
            record.table,
            record.successes,
            len(set(record.models)),
            TRUST_ENV,
        )
    return record


def is_trusted(digest: str) -> bool:
    """Whether an operator has signed this exact script off."""
    raw = os.environ.get(TRUST_ENV, "").strip()
    if not raw or not digest:
        return False
    return digest in {d.strip() for d in raw.split(",") if d.strip()}
