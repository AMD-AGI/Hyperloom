# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling playbook: how the judge stops, sizes and opens branches, and the pool that improves it.

A :class:`Playbook` has two parts. Its numbers (:class:`PlaybookNumbers`) are bounded (:data:`BOUNDS`) and applied by
code: the stall bar, the probation and the new-branch grace the stall evidence reads (:meth:`Playbook.stop_rule`),
which the scheduler's grace reads too, how many branches run at once and how late a new one opens, which the
scheduler enforces on the judge's opens, and after how many failures an op leaves the open areas. Its rules, at most
:data:`MAX_RULES`, each say whether they govern stops, K or opening, and hold what numbers cannot, such as which of
two alike branches to stop.

:data:`H0` is the first champion: the evidence rules, no text rules, every number at its default. A playbook never
overrides the :data:`SAFETY_ENVELOPE`, which the judge reads beside it and the scheduler enforces: a branch with a
candidate waiting for its confirm or a near-bar or stranded lead is no stop candidate, a probation draw stands, the
numbers stay in their bounds, the audit stays on, and a playbook changes one thing at a time. A challenger session
whose judge ordered a stop the envelope refused plays the champion from then on.
:func:`problems` is the static check (R0) every new playbook passes before it is tried.

The pool beside the archive (:class:`PlaybookPool`, ``<root>/playbooks``) keeps every playbook written, the
champions crowned and the playbooks retired, each with its evidence. A session plays the champion, or, in a
challenger session, the oldest playbook still on trial (:func:`session_playbook`); a session the command line names
neither is a challenger session when its id draws it (:func:`drawn_challenger`), so the playbooks on trial are played
without anyone asking for them. :meth:`Playbook.render` is what
the judge reads of it every round.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from hyperloom_multibranch.blackboard.codec import decode, encode
from hyperloom_multibranch.blackboard.local.files import encode_json, publish_once
from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive, written_by
from hyperloom_multibranch.blackboard.rsi.frontier import PROBATION_SHARE, PROBATION_TESTS, SETTLED_TESTS, StopRule
from hyperloom_multibranch.blackboard.rsi.history import MAX_STALL_TESTS, MIN_STALL_TESTS

log = logging.getLogger(__name__)

PLAYBOOK_FORMAT = "multibranch.scheduling-playbook"
#: The key of the playbook a session played in its policy versions, as the scheduler records it.
PLAYBOOK_KEY = "playbook"
#: The key the scheduler records beside :data:`PLAYBOOK_KEY` when a challenger session broke the safety envelope and
#: played the champion for the rest of it, the breach as its value.
FALLBACK_KEY = "playbook_fallback"
#: The share of sessions their id draws as challenger sessions: an even split, so a challenger session finds a session
#: of the champion on its task to pair with soonest.
CHALLENGER_SHARE = 0.5
CROWNED_FORMAT = "multibranch.scheduling-playbook-crowned"
RETIRED_FORMAT = "multibranch.scheduling-playbook-retired"
#: A playbook's label: lower-case letters, digits and ``.``, ``_``, ``-``.
PLAYBOOK_LABEL = re.compile(r"[a-z][a-z0-9._-]{1,63}")
H0_LABEL = "h0"
MAX_RULES = 5
RULE_CHARS = 300
#: What a rule may not name, so a playbook holds for sessions it has not seen: a branch, a session or a model.
_PARTICULAR = re.compile(
    r"\bb\d{2,}\b|\b[0-9a-f]{8}\b|qwen|llama|deepseek|glm|mixtral|gpt-?oss|kimi|minimax", re.IGNORECASE
)
_JSON = "*.json"
_ENCODING = "utf-8"
_PLAYBOOKS = "playbooks"
_CROWNED = "crowned"
_RETIRED = "retired"


class Governs(str, Enum):
    """What a text rule decides."""

    STOP = "stop"
    K = "k"
    OPEN = "open"


@dataclass(frozen=True)
class PlaybookRule:
    governs: Governs
    text: str


@dataclass(frozen=True)
class PlaybookNumbers:
    """The playbook's numbers, each applied by code.

    ``stall_tests`` is the stop bar, ``None`` for the one the stall odds set each round; ``probation_share`` of the
    stop candidates run ``probation_tests`` more judged tests first; a new branch keeps its grace until
    ``grace_tests`` of its explore tests succeeded and were judged. ``max_open`` caps the branches running at once,
    ``None`` for the run's cap; no branch the judge refills opens in the last ``last_open_minutes`` of EXPLORE,
    ``None`` for the median time a new branch took to its first commit; an op that failed ``failed_op_limit`` times
    this session leaves the open areas.
    """

    stall_tests: int | None = None
    probation_share: float = PROBATION_SHARE
    probation_tests: int = PROBATION_TESTS
    grace_tests: int = SETTLED_TESTS
    max_open: int | None = None
    last_open_minutes: float | None = None
    failed_op_limit: int = 2


#: Each number's bounds, both included.
BOUNDS: Mapping[str, tuple[float, float]] = {
    "stall_tests": (MIN_STALL_TESTS, MAX_STALL_TESTS),
    "probation_share": (0.125, 0.5),
    "probation_tests": (2, 8),
    "grace_tests": (3, 8),
    "max_open": (1, 16),
    "last_open_minutes": (0, 240),
    "failed_op_limit": (2, 4),
}

#: What the judge reads beside every playbook and the scheduler enforces; no playbook changes it.
SAFETY_ENVELOPE = (
    "A branch with a candidate waiting for its confirm, a near-bar lead or a stranded lead is no stop candidate.",
    "A branch drawn for probation runs its probation tests: the draw stands, so stopping keeps being tested.",
    "The numbers stay within their bounds; the stop audit and the decision records stay on.",
    "A new playbook changes one number or one rule of its parent, so what follows can be laid to it.",
)


@dataclass(frozen=True)
class Playbook:
    """A playbook: its ``numbers`` and ``rules``; the ``parent`` it changed, why (``hypothesis``) and the decision
    measure it expects to improve (``measure``); when it was registered and who wrote it."""

    label: str
    numbers: PlaybookNumbers = field(default_factory=PlaybookNumbers)
    rules: tuple[PlaybookRule, ...] = ()
    parent: str = ""
    hypothesis: str = ""
    measure: str = ""
    registered_at: float = 0.0
    provenance: Mapping[str, str] = field(default_factory=dict)

    def stop_rule(self) -> StopRule:
        """The stop rule the stall evidence reads under this playbook."""

        numbers = self.numbers
        return StopRule(numbers.stall_tests, numbers.probation_share, numbers.probation_tests, numbers.grace_tests)

    def render(self, *, challenger: bool = False) -> str:
        """The playbook as the judge reads it every round: its numbers, its rules, then the safety envelope."""

        n = self.numbers
        bar = f"{n.stall_tests} judged tests" if n.stall_tests is not None else "what the stall odds set (stall_bar)"
        opened = f"{n.max_open} branches" if n.max_open is not None else "the run's cap"
        late = (
            f"{n.last_open_minutes:g} min"
            if n.last_open_minutes is not None
            else "the median time a new branch takes to its first commit"
        )
        lines = [
            f"Scheduling playbook {self.label}" + (", on trial in this session:" if challenger else ":"),
            f"- stop: a branch whose stop_candidate is 1 has stalled {bar} past its last commit that did not fail to "
            "merge, with nothing holding it; stop candidates, not the lowest expected gain.",
            f"- probation: {n.probation_share:.0%} of the stop candidates, drawn once a stall, run {n.probation_tests} "
            "more judged tests first (probation 1); leave them running.",
            f"- grace: a new branch is no stop candidate until {n.grace_tests} of its explore tests succeeded and "
            "were judged.",
            f"- open: at most {opened} at once; after the opening batch, none once less of EXPLORE is left than {late}. "
            "The scheduler cuts an open past either.",
            f"- an op that failed {n.failed_op_limit} times this session leaves the open areas.",
            *(f"- rule ({rule.governs.value}): {rule.text}" for rule in self.rules),
            "Safety envelope, whatever the playbook says:",
            *(f"- {line}" for line in SAFETY_ENVELOPE),
        ]
        return "\n".join(lines)


H0 = Playbook(
    H0_LABEL,
    hypothesis=(
        "the evidence rules: stop a branch on its stall, keep leads and probation, open where no branch covers the "
        "headroom while a new branch can still commit"
    ),
)


def changes(parent: Playbook, child: Playbook) -> tuple[str, ...]:
    """What ``child`` changes of ``parent``: each number by name, and ``rules`` when its rules differ."""

    changed = tuple(name for name in BOUNDS if getattr(parent.numbers, name) != getattr(child.numbers, name))
    return (*changed, "rules") if parent.rules != child.rules else changed


def problems(playbook: Playbook, parent: Playbook | None = None) -> tuple[str, ...]:
    """Why ``playbook`` may not be tried (R0): its label, a number out of its bounds, too many or empty rules, a rule
    that names a branch, a session or a model, or, against ``parent``, other than one change."""

    found = []
    if not PLAYBOOK_LABEL.fullmatch(playbook.label):
        found.append(f"{playbook.label!r} is no playbook label")
    for name, (low, high) in BOUNDS.items():
        value = getattr(playbook.numbers, name)
        if value is not None and not low <= value <= high:
            found.append(f"{name} {value:g} is outside {low:g} to {high:g}")
    if len(playbook.rules) > MAX_RULES:
        found.append(f"{len(playbook.rules)} rules, {MAX_RULES} at most")
    for index, rule in enumerate(playbook.rules, start=1):
        if not rule.text.strip() or len(rule.text) > RULE_CHARS:
            found.append(f"rule {index} is empty or longer than {RULE_CHARS} characters")
        elif _PARTICULAR.search(rule.text):
            found.append(f"rule {index} names a branch, a session or a model: it would not hold elsewhere")
    if parent is not None and len(changes(parent, playbook)) != 1:
        changed = ", ".join(changes(parent, playbook)) or "nothing"
        found.append(f"it changes {changed} of {parent.label}; a playbook changes one thing")
    return tuple(found)


@dataclass(frozen=True)
class Crowned:
    """``label`` became the champion at ``crowned_at``, on ``evidence``."""

    label: str
    crowned_at: float
    evidence: str


@dataclass(frozen=True)
class Retired:
    """``label`` left the trial at ``retired_at``, on ``evidence``."""

    label: str
    retired_at: float
    evidence: str


class PlaybookConflict(FileExistsError):
    """Another playbook is registered under the label."""


def _published(path: Path, document: Mapping[str, object]) -> bool:
    """``document`` written once at ``path``; ``False`` when ``path`` was written already."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_once(path, encode_json(document))


def _read(path: Path, kind: type, fmt: str) -> object:
    document = json.loads(path.read_text(encoding=_ENCODING))
    if document.get("format") != fmt:
        raise ValueError(f"unsupported format {document.get('format')!r}")
    return decode(kind, document.get("record"))


class PlaybookPool:
    """The playbooks written for sessions like the archive's: ``<root>/playbooks``, beside its trees under
    ``<root>/trees``; with ``as_of``, read as it stood then, the playbooks registered, crowned and retired by then."""

    def __init__(self, root: Path, as_of: float | None = None) -> None:
        self.root = Path(root) / _PLAYBOOKS
        self.as_of = as_of

    @classmethod
    def beside(cls, archive: TreeArchive) -> PlaybookPool:
        """The pool beside ``archive``, read as of the time it is."""

        return cls(archive.root.parent, archive.as_of)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PlaybookPool | None:
        """The pool a running session plays from, beside the archive the environment names; ``None`` when it names
        none or the session reads nothing from it (:meth:`TreeArchive.read_from_env`)."""

        archive = TreeArchive.read_from_env(env)
        return None if archive is None else cls.beside(archive)

    def _ledger(self, folder: str, kind: type, fmt: str) -> list:
        found = []
        for path in sorted((self.root / folder).glob(_JSON)):
            try:
                found.append(_read(path, kind, fmt))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                log.warning("skipping an unreadable playbook record %s: %s", path, exc)
        return found

    def playbooks(self) -> tuple[Playbook, ...]:
        """Every readable playbook, the earliest registered first."""

        found = [
            item
            for item in self._ledger(".", Playbook, PLAYBOOK_FORMAT)
            if isinstance(item, Playbook) and written_by(self.as_of, item.registered_at)
        ]
        return tuple(sorted(found, key=lambda item: item.registered_at))

    def register(self, playbook: Playbook) -> Playbook:
        """``playbook`` written once under its label, or the one already there with the same numbers and rules;
        raises :class:`PlaybookConflict` for another under that label, ``ValueError`` for a label the pool refuses."""

        if not PLAYBOOK_LABEL.fullmatch(playbook.label) or playbook.label == H0_LABEL:
            raise ValueError(f"{playbook.label!r} is no label a written playbook may take")
        path = self.root / f"{playbook.label}.json"
        if _published(path, {"format": PLAYBOOK_FORMAT, "record": encode(playbook)}):
            return playbook
        held = _read(path, Playbook, PLAYBOOK_FORMAT)
        if not isinstance(held, Playbook) or (held.numbers, held.rules) != (playbook.numbers, playbook.rules):
            raise PlaybookConflict(f"another playbook is registered as {playbook.label}")
        return held

    def named(self, label: str) -> Playbook | None:
        if label == H0_LABEL:
            return H0
        return next((item for item in self.playbooks() if item.label == label), None)

    def crowned(self) -> tuple[Crowned, ...]:
        found = [
            item
            for item in self._ledger(_CROWNED, Crowned, CROWNED_FORMAT)
            if isinstance(item, Crowned) and written_by(self.as_of, item.crowned_at)
        ]
        return tuple(sorted(found, key=lambda item: item.crowned_at))

    def retired(self) -> tuple[Retired, ...]:
        found = [
            item
            for item in self._ledger(_RETIRED, Retired, RETIRED_FORMAT)
            if isinstance(item, Retired) and written_by(self.as_of, item.retired_at)
        ]
        return tuple(sorted(found, key=lambda item: item.retired_at))

    def champion(self) -> Playbook:
        """The latest champion crowned whose playbook the pool still reads; :data:`H0` before any."""

        for record in reversed(self.crowned()):
            found = self.named(record.label)
            if found is not None:
                return found
        return H0

    def crown(self, record: Crowned) -> Crowned:
        path = self.root / _CROWNED / f"{record.crowned_at:.6f}-{record.label}.json"
        _published(path, {"format": CROWNED_FORMAT, "record": encode(record)})
        return record

    def retire(self, record: Retired) -> None:
        """Retire ``record``'s playbook, once: a second retirement of it adds nothing."""

        _published(self.root / _RETIRED / f"{record.label}.json", {"format": RETIRED_FORMAT, "record": encode(record)})

    def on_trial(self) -> tuple[Playbook, ...]:
        """The playbooks neither crowned nor retired that pass R0 against their parent, the earliest first."""

        done = {item.label for item in (*self.crowned(), *self.retired())}
        trial = []
        for item in self.playbooks():
            parent = self.named(item.parent) if item.parent else None
            if item.label not in done and not problems(item, parent):
                trial.append(item)
        return tuple(trial)


@dataclass(frozen=True)
class PlaybookChoice:
    """The playbook a session plays, the champion beside it, and whether it is on trial."""

    playbook: Playbook
    champion: Playbook
    challenger: bool = False

    def render(self) -> str:
        return self.playbook.render(challenger=self.challenger)


def session_playbook(pool: PlaybookPool | None, *, challenger: bool) -> PlaybookChoice:
    """What a session plays: the champion of ``pool``, :data:`H0` without one; in a ``challenger`` session the oldest
    playbook on trial, while there is one."""

    champion = pool.champion() if pool is not None else H0
    trial = pool.on_trial() if pool is not None and challenger else ()
    if trial:
        return PlaybookChoice(trial[0], champion, challenger=True)
    return PlaybookChoice(champion, champion)


def drawn_challenger(session_id: str) -> bool:
    """Whether the session's id draws it as one of the :data:`CHALLENGER_SHARE` of sessions that are challenger
    sessions: the same draw whenever it is asked, and apart from the menu's draw of its control branches."""

    digest = hashlib.sha256(f"{session_id}:playbook".encode()).hexdigest()
    return int(digest, 16) < CHALLENGER_SHARE * 16 ** len(digest)


@dataclass
class PlaybookPlay:
    """What a session plays as it goes: the ``choice`` it started with, then, once a challenger session broke the
    safety envelope, the champion for the rest of it, the first ``breach`` kept for :data:`FALLBACK_KEY`."""

    choice: PlaybookChoice
    breach: str = ""

    @property
    def playbook(self) -> Playbook:
        return self.choice.champion if self.breach else self.choice.playbook

    def stop_rule(self) -> StopRule:
        """The stop rule the stall evidence reads now."""

        return self.playbook.stop_rule()

    def render(self) -> str:
        """What the judge reads of the playbook played now."""

        return self.playbook.render(challenger=self.choice.challenger and not self.breach)

    def fall_back(self, breaches: Sequence[str]) -> bool:
        """Play the champion from now on, after the first of ``breaches`` in a challenger session still on trial;
        whether this call made it fall back."""

        if self.breach or not breaches or not self.choice.challenger:
            return False
        self.breach = breaches[0]
        return True

    def versions(self) -> dict[str, str]:
        """What the session's policy versions record of the play: the playbook it started with and, after a fallback,
        the breach under :data:`FALLBACK_KEY`."""

        found = {PLAYBOOK_KEY: self.choice.playbook.label}
        if self.breach:
            found[FALLBACK_KEY] = self.breach
        return found


def replaced(playbook: Playbook, **numbers: float | int | None) -> Playbook:
    """``playbook`` with some numbers changed; raises ``TypeError`` for a name :class:`PlaybookNumbers` lacks."""

    return dataclasses.replace(playbook, numbers=dataclasses.replace(playbook.numbers, **numbers))


def playbook_label(text: str) -> str:
    """``text`` as a label the pool takes (:data:`PLAYBOOK_LABEL`): in lower case, each run of characters other than
    letters, digits, ``.`` and ``_`` one ``-``, starting with a letter; ``""`` when none is left, or only
    :data:`H0_LABEL`."""

    slug = re.sub(r"[^a-z0-9._]+", "-", text.strip().lower()).strip("-._")
    if slug and not slug[0].isalpha():
        slug = f"p-{slug}"
    slug = slug[:64].rstrip("-._")
    return slug if PLAYBOOK_LABEL.fullmatch(slug) and slug != H0_LABEL else ""


__all__ = [
    "BOUNDS",
    "CHALLENGER_SHARE",
    "CROWNED_FORMAT",
    "FALLBACK_KEY",
    "H0",
    "H0_LABEL",
    "MAX_RULES",
    "PLAYBOOK_FORMAT",
    "PLAYBOOK_KEY",
    "PLAYBOOK_LABEL",
    "RETIRED_FORMAT",
    "RULE_CHARS",
    "SAFETY_ENVELOPE",
    "Crowned",
    "Governs",
    "Playbook",
    "PlaybookChoice",
    "PlaybookConflict",
    "PlaybookNumbers",
    "PlaybookPlay",
    "PlaybookPool",
    "PlaybookRule",
    "Retired",
    "changes",
    "drawn_challenger",
    "playbook_label",
    "problems",
    "replaced",
    "session_playbook",
]
