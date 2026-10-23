# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign's optimization directions, and how far each has been mined.

``forge_experiments/directions.json`` holds what cannot be derived: each direction's definition, the gain it was
expected to bring when listed, the ones retired with evidence, which Analysis commits have been surveyed, and the
audits run. How far a direction has been mined is folded from the ``iteration_result`` events that name
it, so a resume or an event replay can never leave the two disagreeing.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from kernelforge.orchestrator.direction_claims import DirectionClaim
from kernelforge.tools.durable_io import atomic_write_text

DIRECTIONS_FILE = "directions.json"

OPEN, ACTIVE, KEPT, EXHAUSTED, INFEASIBLE = "OPEN", "ACTIVE", "KEPT", "EXHAUSTED", "INFEASIBLE"
TERMINAL = frozenset({EXHAUSTED, INFEASIBLE})

# Decisions that tried a direction. Provider, orchestration and workspace failures say nothing about it.
ATTEMPT_DECISIONS = frozenset(
    {
        "KEEP",
        "REVERT_PERF",
        "NO_CHANGES",
        "REVERT_VALIDATION",
        "REVERT_VALIDATION_TIMEOUT",
        "REVERT_VALIDATION_ERROR",
        "BUILD_FAILED",
        "CRASH",
    }
)


def measured(event: dict) -> bool:
    """Whether an iteration's candidate was timed: a REVERT_PERF whose bench failed carries no score."""
    return event.get("decision") in {"KEEP", "REVERT_PERF"} and event.get("mean_case_speedup") is not None


def attempted(event: dict) -> bool:
    """Whether an iteration tested its direction. A bench that failed says nothing about it, like a provider error."""
    return event.get("decision") in ATTEMPT_DECISIONS and (event.get("decision") != "REVERT_PERF" or measured(event))


@dataclass(frozen=True)
class Direction:
    """One optimization direction: what it is, why it is believed, and who proposed it."""

    id: str
    title: str
    category: str = ""
    hypothesis: str = ""
    evidence: tuple[str, ...] = ()
    source: str = ""
    added_iteration: int = 0
    retired_reason: str = ""
    retired_iteration: int = 0
    # The fraction the agent that listed it expected it to raise the best score by, and why; None without one.
    priority: float | None = None
    priority_reason: str = ""


@dataclass(frozen=True)
class DirectionStanding:
    """How far one direction has been mined."""

    direction: Direction
    status: str
    attempts: int
    keeps: int
    attempts_since_keep: int
    gain: float

    def to_view(self) -> dict:
        return {
            "id": self.direction.id,
            "title": self.direction.title,
            "category": self.direction.category,
            "status": self.status,
            "attempts": self.attempts,
            "keeps": self.keeps,
            "attempts_since_keep": self.attempts_since_keep,
            "gain_pct": round(self.gain * 100, 2),
            "expected_gain_pct": round(self.direction.priority * 100, 2)
            if self.direction.priority is not None
            else None,
            "expected_gain_reason": self.direction.priority_reason,
        }

    @property
    def in_play(self) -> bool:
        return self.status not in TERMINAL


def by_priority(standings: tuple[DirectionStanding, ...]) -> tuple[DirectionStanding, ...]:
    """Directions still in play first, by expected gain with the unestimated after; mined out and ruled out last."""

    def key(standing: DirectionStanding) -> tuple:
        priority = standing.direction.priority
        return (not standing.in_play, priority is None, -(priority or 0.0))

    return tuple(sorted(standings, key=key))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "direction"


def _normalized(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def expected_fraction(entry: dict) -> float | None:
    """An agent's ``expected_gain_pct`` as a fraction, or None when it gave no non-negative number."""
    value = entry.get("expected_gain_pct")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value) / 100.0


class DirectionLedger:
    """The durable half of the ledger: definitions, retirements, surveys and audits."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.path = Path(campaign_root) / DIRECTIONS_FILE
        self._data = self._load()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("directions", [])
        data.setdefault("surveyed_commits", [])
        data.setdefault("audits", [])
        return data

    def _save(self) -> None:
        atomic_write_text(self.path, json.dumps(self._data, indent=2))

    def directions(self) -> tuple[Direction, ...]:
        fields = Direction.__dataclass_fields__
        return tuple(
            Direction(**{**{k: v for k, v in raw.items() if k in fields}, "evidence": tuple(raw.get("evidence", ()))})
            for raw in self._data["directions"]
        )

    def add(self, seeds: list[dict], *, source: str, iteration: int) -> list[str]:
        """Add the seeds whose title is new to the ledger; return the ids they were given."""
        known = {_normalized(d.title) for d in self.directions()}
        ids = {d.id for d in self.directions()}
        added = []
        for seed in seeds:
            title = str(seed.get("title") or "").strip()
            if not title or _normalized(title) in known:
                continue
            base = _slug(title)
            direction_id, suffix = base, 2
            while direction_id in ids:
                direction_id, suffix = f"{base}-{suffix}", suffix + 1
            direction = Direction(
                id=direction_id,
                title=title,
                category=str(seed.get("category") or "").strip(),
                hypothesis=str(seed.get("hypothesis") or "").strip(),
                evidence=tuple(str(item) for item in seed.get("evidence") or ()),
                source=source,
                added_iteration=iteration,
                priority=expected_fraction(seed),
                priority_reason=str(seed.get("reason") or "").strip(),
            )
            self._data["directions"].append(asdict(direction))
            known.add(_normalized(title))
            ids.add(direction_id)
            added.append(direction_id)
        if added:
            self._save()
        return added

    def resolve_claim(self, claim: DirectionClaim | None, *, iteration: int) -> str:
        """The ledger id a plan's claim refers to, adding a new direction it declares; "" when it names none."""
        if claim is None:
            return ""
        if claim.direction_id:
            return claim.direction_id if claim.direction_id in {d.id for d in self.directions()} else ""
        by_title = {_normalized(d.title): d.id for d in self.directions()}
        existing = by_title.get(_normalized(claim.new_title))
        if existing:
            return existing
        added = self.add(
            [{"title": claim.new_title, "category": claim.new_category}], source="planner", iteration=iteration
        )
        return added[0] if added else ""

    def retire(self, direction_id: str, *, reason: str, iteration: int) -> bool:
        for raw in self._data["directions"]:
            if raw.get("id") == direction_id and not raw.get("retired_reason"):
                raw["retired_reason"], raw["retired_iteration"] = reason, iteration
                self._save()
                return True
        return False

    def surveyed(self, commit: str) -> bool:
        return commit in self._data["surveyed_commits"]

    def mark_surveyed(self, commit: str) -> None:
        if commit and commit not in self._data["surveyed_commits"]:
            self._data["surveyed_commits"].append(commit)
            self._save()

    def audits(self) -> list[dict]:
        return list(self._data["audits"])

    def record_audit(self, *, iteration: int, ledger_size: int, added: list[str]) -> None:
        self._data["audits"].append({"iteration": iteration, "ledger_size": ledger_size, "added": added})
        self._save()


def keep_gains(results: list[dict], start_score: float | None) -> dict[int, float]:
    """Per KEEP iteration, the fraction it raised the best score by, from the scores the events carry."""
    gains: dict[int, float] = {}
    previous = start_score
    for event in results:
        best = event.get("best_after_mean_case_speedup")
        if event.get("decision") == "KEEP" and best and previous:
            gains[int(event.get("iter") or 0)] = float(best) / float(previous) - 1.0
        if best:
            previous = float(best)
    return gains


def direction_standings(
    directions: tuple[Direction, ...],
    results: list[dict],
    *,
    exhaust_after: int,
    gains: dict[int, float] | None = None,
) -> tuple[DirectionStanding, ...]:
    """Fold the attempts the events attribute to each direction into its standing."""
    gains = gains or {}
    tallies = {d.id: [0, 0, 0, 1.0] for d in directions}  # attempts, keeps, since keep, gain product
    for event in results:
        tally = tallies.get(str(event.get("direction_id") or ""))
        if tally is None or not attempted(event):
            continue
        tally[0] += 1
        if event.get("decision") == "KEEP":
            tally[1] += 1
            tally[2] = 0
            tally[3] *= 1.0 + gains.get(int(event.get("iter") or 0), 0.0)
        else:
            tally[2] += 1
    standings = []
    for direction in directions:
        attempts, keeps, since, gain = tallies[direction.id]
        if direction.retired_reason:
            status = INFEASIBLE
        elif attempts == 0:
            status = OPEN
        elif since >= exhaust_after:
            status = EXHAUSTED
        else:
            status = KEPT if keeps else ACTIVE
        standings.append(DirectionStanding(direction, status, attempts, keeps, since, gain - 1.0))
    return tuple(standings)
