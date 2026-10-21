# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The frontier scorers a session may choose from, and the pool that dates them.

* Built in: ``frontier-v1``, the default and the fallback of every other, with the
  :data:`~hyperloom_multibranch.notebook.rsi.calibration.VARIATIONS` of its parameters; and ``bandit-rv``, the island
  bandit's ``R / V`` (:mod:`rsi.bandit`).
* :func:`custom_scorers`: the place a scorer written for this repository is added; its docstring says how.
* Expressions the scheduler's LLM writes at CLOSE (:mod:`rsi.expression`, :mod:`rsi.authoring`), kept in the pool.

The pool, ``<archive>/scorers/<label>.json``, holds one :class:`PoolRecord` per label, written once: an expression
with its rationale and who wrote it, or the first sight of a scorer in code. ``registered_at`` is when it was written.
A session is evidence for a scorer only when it started after that, so no scorer wins on the sessions it could have
been fitted to. A session sets the :data:`MAX_EXPRESSIONS` most recent expressions beside the defaults, and one CLOSE
adds at most :data:`MAX_PER_SESSION`.

:func:`registered_scorers` is every scorer a session may choose besides ``frontier-v1``'s variations, each with when it
was registered, for :func:`~hyperloom_multibranch.notebook.rsi.calibration.frontier_scorer_for`; :func:`scorer_named`
finds the one a manifest recorded, or a person asked for, by its label.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.notebook.local.files import encode_json, publish_once
from hyperloom_multibranch.notebook.rsi.archive import TreeArchive
from hyperloom_multibranch.notebook.rsi.bandit import BanditScorer
from hyperloom_multibranch.notebook.rsi.calibration import VARIATIONS, Registered
from hyperloom_multibranch.notebook.rsi.expression import ExpressionError, ExpressionScorer
from hyperloom_multibranch.notebook.rsi.frontier import FRONTIER_V1, FRONTIER_V1_LABEL, SCORER_LABEL, FrontierScorer

log = logging.getLogger(__name__)

POOL_FORMAT = "multibranch.frontier-scorer.v1"
#: How many of the pool's expressions, the most recent, a session sets beside the defaults.
MAX_EXPRESSIONS = 24
#: How many expressions one CLOSE may add to the pool.
MAX_PER_SESSION = 3


def custom_scorers() -> tuple[FrontierScorer, ...]:
    """The scorers written for this repository besides the built-in ones; none yet. To add one, return it here.

    A scorer implements :class:`~hyperloom_multibranch.notebook.rsi.frontier.FrontierScorer`:

    * ``label`` matches :data:`~hyperloom_multibranch.notebook.rsi.frontier.SCORER_LABEL` and carries its version, such
      as ``acme-trend-v1``. A change to how it scores takes a new label, so the evidence gathered for one never counts
      for another.
    * ``estimate(branch, t_s, context, readings)`` returns an
      :class:`~hyperloom_multibranch.notebook.rsi.frontier.Estimate` for ``branch``, the
      :class:`~hyperloom_multibranch.notebook.rsi.archive.TreeBranch` as it stood ``t_s`` seconds into its session: its
      commits with the gain each reached, and when its tests were requested, measured, judged and ended. ``context``
      holds the session's task conditions, the archived sessions (``history``), the scoring policy, and ``tree``, the
      session's own search tree as it stood then; ``readings`` say whether a kernel-layer direction is under
      observation, the roofline headroom, the Amdahl ceiling of the head's largest op, and the gain of a candidate
      waiting for its confirm.
    * The estimate is deterministic and writes nothing. Its ``expected_gain_pct``, in points over the baseline, and its
      ``components`` are finite numbers, no component is one of
      :data:`~hyperloom_multibranch.notebook.rsi.frontier.RESERVED_COMPONENTS`, and its ``basis`` is ``custom``, or
      ``trend``, ``prior`` or ``observing`` when it is one of those.

    The envelope then applies, whatever the estimate says: the new-branch floor, the lift of a candidate waiting for
    its confirm, the roofline and Amdahl caps, never negative. The first session that sees a scorer registers it in the
    pool, and only the sessions that start after that are evidence for it: it replaces ``frontier-v1`` only once the
    calibration over those finds it ranks the branches better. A session in which it raises, or breaks the contract,
    scores that branch with ``frontier-v1``, and after
    :data:`~hyperloom_multibranch.notebook.rsi.frontier.FALLBACK_LIMIT` failures the rest of the session.
    ``python -m hyperloom_multibranch.notebook.rsi.studies frontier --archive DIR`` sets it beside ``frontier-v1`` and
    the bandit on an archive's sessions, for reference.
    """

    return ()


class ScorerSource(str, Enum):
    """Where a scorer comes from: code in this package, or an expression in the pool."""

    CODE = "code"
    EXPRESSION = "expression"


@dataclass(frozen=True)
class PoolRecord:
    """One scorer of the pool: its ``label``, where it came from, and when it was registered, in seconds since the
    epoch. An expression keeps its ``expression`` and its author's ``rationale``; ``provenance`` says who wrote it,
    such as the model and the session."""

    label: str
    source: ScorerSource
    registered_at: float
    expression: str = ""
    rationale: str = ""
    provenance: Mapping[str, str] = field(default_factory=dict)


class PoolConflict(FileExistsError):
    """The pool already holds another scorer under that label."""


def pool_document(record: PoolRecord) -> dict[str, object]:
    return {"format": POOL_FORMAT, "scorer": encode(record)}


def pool_record(document: Mapping[str, object]) -> PoolRecord:
    """The record :func:`pool_document` wrote; raises ``ValueError`` or ``TypeError`` on anything else."""

    if document.get("format") != POOL_FORMAT:
        raise ValueError(f"unsupported pool record format {document.get('format')!r}")
    return decode(PoolRecord, document.get("scorer"))


class ScorerPool:
    """The pool of registered scorers: ``<root>/scorers``, beside an archive's trees under ``<root>/trees``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "scorers"

    @classmethod
    def beside(cls, archive: TreeArchive) -> ScorerPool:
        return cls(archive.root.parent)

    def records(self) -> tuple[PoolRecord, ...]:
        """Every readable record, the earliest registered first; one this version cannot read is skipped with a
        warning."""

        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                found.append(pool_record(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                log.warning("skipping unreadable pool record %s: %s", path, exc)
        return tuple(sorted(found, key=lambda item: item.registered_at))

    def expressions(self) -> tuple[PoolRecord, ...]:
        """The :data:`MAX_EXPRESSIONS` most recent expressions, the earliest first."""

        return tuple(item for item in self.records() if item.source is ScorerSource.EXPRESSION)[-MAX_EXPRESSIONS:]

    def register(self, record: PoolRecord) -> PoolRecord:
        """``record`` written once under its label, or the record already there for the same scorer, which keeps its
        registration time. Raises :class:`PoolConflict` for another scorer under that label, and ``ValueError`` for a
        label :data:`~hyperloom_multibranch.notebook.rsi.frontier.SCORER_LABEL` refuses."""

        if not SCORER_LABEL.fullmatch(record.label):
            raise ValueError(f"{record.label!r} is no scorer label: {SCORER_LABEL.pattern}")
        path = self.root / f"{record.label}.json"
        if publish_once(path, encode_json(pool_document(record))):
            return record
        held = pool_record(json.loads(path.read_text(encoding="utf-8")))
        if (held.source, held.expression) != (record.source, record.expression):
            raise PoolConflict(f"{path} already holds another scorer labelled {record.label}")
        return held


def builtin_scorers() -> tuple[FrontierScorer, ...]:
    """The built-in scorers besides ``frontier-v1`` and its variations."""

    return (BanditScorer(),)


def _first_seen(pool: ScorerPool | None, record: PoolRecord) -> float:
    """When ``record``'s scorer was first registered in ``pool``; ``record``'s own time without a pool, or when the
    pool cannot take it, so that no archived session is evidence for it."""

    if pool is None:
        return record.registered_at
    try:
        return pool.register(record).registered_at
    except (OSError, ValueError) as exc:
        log.warning("could not register the frontier scorer %s in %s: %s", record.label, pool.root, exc)
        return record.registered_at


def registered_scorers(pool: ScorerPool | None, now: float) -> tuple[Registered, ...]:
    """Every scorer a session may choose besides ``frontier-v1`` and its variations, each with when it was registered:
    the scorers in code, registered in ``pool`` the first time a session sees them (``now``), and the pool's most recent
    expressions. A label already taken, by a variation or an earlier scorer, is left out."""

    taken = {FRONTIER_V1_LABEL, *(variation.label for variation in VARIATIONS)}
    found: list[Registered] = []
    for scorer in (*builtin_scorers(), *custom_scorers()):
        if scorer.label in taken:
            log.warning("left out the frontier scorer %s: its label is taken", scorer.label)
            continue
        taken.add(scorer.label)
        found.append(Registered(scorer, _first_seen(pool, PoolRecord(scorer.label, ScorerSource.CODE, now))))
    for record in pool.expressions() if pool is not None else ():
        if record.label in taken:
            continue
        try:
            found.append(Registered(ExpressionScorer.read(record.label, record.expression), record.registered_at))
        except ExpressionError as exc:
            log.warning("left out the pooled frontier scorer %s: %s", record.label, exc)
            continue
        taken.add(record.label)
    return tuple(found)


def scorer_named(label: str, pool: ScorerPool | None) -> FrontierScorer | None:
    """The scorer labelled ``label``: ``frontier-v1``, one in code, or an expression in ``pool``; ``None`` when none
    is."""

    if label == FRONTIER_V1_LABEL:
        return FRONTIER_V1
    coded = next((scorer for scorer in (*builtin_scorers(), *custom_scorers()) if scorer.label == label), None)
    if coded is not None:
        return coded
    records = pool.records() if pool is not None else ()
    record = next((item for item in records if item.label == label and item.source is ScorerSource.EXPRESSION), None)
    return None if record is None else ExpressionScorer.read(record.label, record.expression)


__all__ = [
    "MAX_EXPRESSIONS",
    "MAX_PER_SESSION",
    "POOL_FORMAT",
    "PoolConflict",
    "PoolRecord",
    "ScorerPool",
    "ScorerSource",
    "builtin_scorers",
    "custom_scorers",
    "pool_document",
    "pool_record",
    "registered_scorers",
    "scorer_named",
]
