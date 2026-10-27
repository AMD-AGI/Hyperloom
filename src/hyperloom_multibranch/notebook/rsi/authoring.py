# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE: the scheduler's LLM improves the scheduling playbook, one change a session.

:func:`author_playbook` is the whole step: it shows the LLM how this session's decisions turned out (the stop audit
of :mod:`rsi.stops` and how each new branch opened), the playbook the session played, and the evidence ledger, every
playbook the pool holds with the change it made, its hypothesis and what came of it (:func:`playbook_prompt`). The
LLM answers with at most one change to the champion, a number or a rule, its hypothesis and the decision measure of
:data:`MEASURES` it should improve (:func:`read_change`). What passes the static check
(:func:`~hyperloom_multibranch.notebook.rsi.playbook.problems`) and repeats no retired playbook goes to the pool, on
trial in the challenger sessions after it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Awaitable, Callable, Mapping

from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import CommitOrigin, LineageKind
from hyperloom_multibranch.notebook.rsi.archive import NO_ARCHIVE, SKIPPED, SearchTree, TreeArchive
from hyperloom_multibranch.notebook.rsi.history import stall_odds
from hyperloom_multibranch.notebook.rsi.playbook import (
    BOUNDS,
    MAX_RULES,
    PLAYBOOK_KEY,
    Governs,
    Playbook,
    PlaybookNumbers,
    PlaybookPool,
    PlaybookRule,
    changes,
    problems,
)
from hyperloom_multibranch.notebook.rsi.replay import REPLAYED_NUMBERS
from hyperloom_multibranch.notebook.rsi.session import FAILED, NO_PROFILE, WRITTEN, CloseStep, archived_history
from hyperloom_multibranch.notebook.rsi.stops import StopAudit, stop_audit

log = logging.getLogger(__name__)

#: Asks the LLM with a system prompt and a prompt, and returns its answer.
Complete = Callable[[str, str], Awaitable[str]]
MAX_RATIONALE_CHARS = 600
NOT_ARCHIVED = "the session's search tree is not in the archive"
_MINUTE = 60.0

#: The decision measures a playbook change may aim at: the judge's early stops, the probations that committed, the
#: stops' worth by the stall odds, a new branch's minutes to its first commit, the share of new branches that
#: committed.
MEASURES = ("early_stops", "probation_commits", "stop_value", "first_commit_minutes", "opened_committed")
NO_CHANGE = "the LLM proposed no change"
#: The numbers that are whole though their default is ``None``.
_WHOLE = frozenset({"stall_tests", "max_open"})

PLAYBOOK_SYSTEM_PROMPT = f"""You improve the scheduling playbook of a multi-branch LLM inference optimization session.
Specialist branches run in parallel, each testing changes to a serving recipe. Every round a judge decides which
branches to stop, how many tests (K) each may run at once, and whether to open new ones. It follows the playbook:
numbers applied by code, within their bounds, and at most {MAX_RULES} text rules for what numbers cannot say.
You read how this session's decisions turned out, the playbook it played, and the evidence ledger: every change tried
before, its hypothesis and what came of it. Propose at most one change to the champion: one number, or one rule added
or removed. Prefer a number the replay of the archived sessions plays, {", ".join(sorted(REPLAYED_NUMBERS))}: a
change to one of them is evaluated before any session runs it, any other change only by sessions run under it. Do not
try again a change the ledger shows retired. A rule must hold for sessions you have not seen, so it
names no branch, session or model. The safety envelope stands whatever the playbook says.
Answer with one JSON object and nothing else, either
{{"label": "...", "change": {{"number": "<name>", "value": <number or null>}}, "hypothesis": "...", "measure": "..."}},
with "change" instead {{"add_rule": {{"governs": "stop" | "k" | "open", "text": "..."}}}} or {{"remove_rule": <its
number>}}, and "measure" one of {", ".join(MEASURES)}; or {{"none": "<why nothing is worth trying>"}}."""


def _number_line(name: str, champion: Playbook) -> str:
    low, high = BOUNDS[name]
    value = getattr(champion.numbers, name)
    return f"- {name}: {'default' if value is None else f'{value:g}'} (bounds {low:g} to {high:g})"


def _opening_lines(tree: SearchTree) -> list[str]:
    """Each steered branch opened after the root: when, how long to its first commit, its commits, how it ended."""

    lines = []
    for branch in tree.branches:
        if not branch.steered or branch.origin is LineageKind.ROOT:
            continue
        own = sorted(commit.t_s for commit in branch.commits if commit.origin is CommitOrigin.COMMIT)
        first = f"first commit after {(own[0] - branch.opened_s) / _MINUTE:.0f} min" if own else "no commit"
        ended = branch.stop_reason or "running at the end"
        lines.append(
            f"- {branch.name}: opened at {branch.opened_s / _MINUTE:.0f} min, {first}, {len(own)} commits; {ended}"
        )
    return lines


def _ledger_lines(pool: PlaybookPool) -> list[str]:
    """Each playbook of ``pool``: what it changed of its parent, why, and whether it was crowned, retired or is on
    trial."""

    crowned = {item.label: item for item in pool.crowned()}
    retired = {item.label: item for item in pool.retired()}
    lines = []
    for item in pool.playbooks():
        parent = pool.named(item.parent) if item.parent else None
        changed = ", ".join(changes(parent, item)) if parent is not None else "everything"
        if item.label in crowned:
            status = f"crowned: {crowned[item.label].evidence}"
        elif item.label in retired:
            status = f"retired: {retired[item.label].evidence}"
        else:
            status = "on trial"
        lines.append(
            f"- {item.label} (of {item.parent or 'none'}, changing {changed}): {item.hypothesis}; aims at "
            f"{item.measure or 'n/a'}; {status}"
        )
    return lines or ["- nothing tried yet: the champion is the first playbook"]


def playbook_prompt(tree: SearchTree, audit: StopAudit, champion: Playbook, played: str, pool: PlaybookPool) -> str:
    """What the playbook's author reads: the numbers and their bounds, the champion, the playbook this session
    played, its decision audit, how its new branches opened, and the evidence ledger."""

    lines = [
        "The numbers, the champion's value of each and its bounds:",
        *(_number_line(name, champion) for name in BOUNDS),
        "",
        "The champion:",
        champion.render(),
        "",
        f"This session, {tree.session_id}, played {played}.",
        "",
        audit.render(),
        "",
        "How its new branches opened:",
        *(_opening_lines(tree) or ["- none opened after the root"]),
        "",
        "The evidence ledger:",
        *_ledger_lines(pool),
    ]
    return "\n".join(lines)


def _changed_numbers(champion: Playbook, change: Mapping[str, object]) -> PlaybookNumbers:
    name, value = change.get("number"), change.get("value")
    if not isinstance(name, str) or name not in BOUNDS:
        raise ValueError(f"{name!r} is no number of the playbook")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{name} takes a number, not {value!r}")
    whole = isinstance(getattr(champion.numbers, name), int) or name in _WHOLE
    if whole and value is not None and value != int(value):
        raise ValueError(f"{name} takes a whole number, not {value!r}")
    return dataclasses.replace(champion.numbers, **{name: int(value) if whole and value is not None else value})


def _changed_rules(champion: Playbook, change: Mapping[str, object]) -> tuple[PlaybookRule, ...]:
    added, removed = change.get("add_rule"), change.get("remove_rule")
    if isinstance(added, Mapping):
        governs, text = added.get("governs"), added.get("text")
        if governs not in {item.value for item in Governs} or not isinstance(text, str):
            raise ValueError("a rule names what it governs, stop, k or open, and its text")
        return (*champion.rules, PlaybookRule(Governs(governs), text.strip()))
    if isinstance(removed, int) and not isinstance(removed, bool) and 1 <= removed <= len(champion.rules):
        return tuple(rule for index, rule in enumerate(champion.rules, start=1) if index != removed)
    raise ValueError("a change is a number, an added rule or the number of a rule to remove")


def read_change(
    answer: str, champion: Playbook, provenance: Mapping[str, str], now: float
) -> tuple[Playbook | None, str]:
    """The playbook ``answer`` proposes, one change to ``champion`` registered ``now`` with ``provenance``, and
    ``""``; or ``None`` and why there is none."""

    if not answer.strip():
        return None, "the answer is empty"
    start, end = answer.find("{"), answer.rfind("}")
    try:
        document = json.loads(answer[start : end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        document = None
    if not isinstance(document, dict):
        return None, "the answer holds no JSON object"
    if "none" in document:
        return None, f"{NO_CHANGE}: {document['none']}"
    change, measure = document.get("change"), document.get("measure")
    label, hypothesis = document.get("label"), document.get("hypothesis")
    if not isinstance(change, Mapping) or not isinstance(label, str) or not isinstance(hypothesis, str):
        return None, "a change names its label, the change and its hypothesis"
    if measure not in MEASURES:
        return None, f"{measure!r} is no decision measure; one of {', '.join(MEASURES)}"
    try:
        numbers = _changed_numbers(champion, change) if "number" in change else champion.numbers
        rules = champion.rules if "number" in change else _changed_rules(champion, change)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    proposed = Playbook(
        label.strip(),
        numbers,
        rules,
        parent=champion.label,
        hypothesis=hypothesis.strip()[:MAX_RATIONALE_CHARS],
        measure=str(measure),
        registered_at=now,
        provenance=dict(provenance),
    )
    found = problems(proposed, champion)
    return (None, "; ".join(found)) if found else (proposed, "")


def _repeats(pool: PlaybookPool, proposed: Playbook) -> str:
    """The label and evidence of a retired playbook with ``proposed``'s numbers and rules; ``""`` when none is."""

    retired = {item.label: item for item in pool.retired()}
    for item in pool.playbooks():
        if item.label in retired and (item.numbers, item.rules) == (proposed.numbers, proposed.rules):
            return f"it repeats {item.label}, retired: {retired[item.label].evidence}"
    return ""


async def author_playbook(
    snapshot: NotebookSnapshot,
    complete: Complete,
    *,
    model: str,
    archive: TreeArchive | None = None,
    pool: PlaybookPool | None = None,
) -> CloseStep:
    """CLOSE, after the search tree is archived: one change to the scheduling playbook. Ask the LLM, through
    ``complete``, with this session's decision audit, the playbook it played and the evidence ledger, and write the
    change to ``pool``, by default the one beside ``archive``, by default the environment's, when it passes the
    static check and repeats no retired playbook. ``model`` names the LLM in its provenance. A step that cannot be
    done says why and never stops CLOSE."""

    profile = snapshot.profile
    if profile is None:
        return CloseStep(SKIPPED, NO_PROFILE)
    target = archive if archive is not None else TreeArchive.from_env()
    if target is None:
        return CloseStep(SKIPPED, NO_ARCHIVE)
    history = archived_history(target)
    own = next((tree for tree in history if tree.session_id == profile.session_id), None)
    if own is None:
        return CloseStep(SKIPPED, NOT_ARCHIVED)
    writable = pool if pool is not None else PlaybookPool.beside(target)
    champion = writable.champion()
    versions = snapshot.fingerprints[-1].policy_versions if snapshot.fingerprints else {}
    audit = stop_audit(own, stall_odds(history, own.task, session=own))
    prompt = playbook_prompt(own, audit, champion, versions.get(PLAYBOOK_KEY, champion.label), writable)
    try:
        answer = await complete(PLAYBOOK_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        log.warning("CLOSE could not ask the LLM for a playbook change: %s", exc, exc_info=True)
        return CloseStep(FAILED, f"{type(exc).__name__}: {exc}")
    proposed, why = read_change(answer, champion, {"model": model, "session_id": profile.session_id}, snapshot.taken_at)
    why = why or (_repeats(writable, proposed) if proposed is not None else "")
    if proposed is None or why:
        return CloseStep(SKIPPED, why, {"written": 0, "refused": 0 if why.startswith(NO_CHANGE) else 1})
    try:
        written = writable.register(proposed)
    except (OSError, ValueError) as exc:
        return CloseStep(SKIPPED, f"{proposed.label}: {exc}", {"written": 0, "refused": 1})
    changed = ", ".join(changes(champion, written))
    return CloseStep(WRITTEN, f"{written.label}: {changed} of {champion.label}", {"written": 1, "refused": 0})


__all__ = [
    "MAX_RATIONALE_CHARS",
    "MEASURES",
    "NOT_ARCHIVED",
    "NO_ARCHIVE",
    "NO_CHANGE",
    "PLAYBOOK_SYSTEM_PROMPT",
    "Complete",
    "author_playbook",
    "playbook_prompt",
    "read_change",
]
