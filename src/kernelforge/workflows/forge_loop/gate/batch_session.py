# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A knowledge base batch session as the in-session gate holds it to its plan.

The session implements every part of the batch in order and ends with ``[playbook:...]``, ``[playbook-skip:...]`` and
``[playbook-untried:...]`` lines; until its wrap-up time the gate sends back a stop that leaves an item neither in the
kernel nor dropped with evidence. The loop credits the playbook's ledger from the same lines.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

# What a batch session keeps before its deadline for wrapping up: at least this, or this share of the session.
BATCH_RESERVE_MIN_SEC = 600.0
BATCH_RESERVE_SHARE = 1 / 6

# The Implementer's stopping rule in a batch session, in place of submitting the first improvement.
BATCH_STOP_RULE = """\
4. This session implements a knowledge base batch: every part of the plan, in
   order, on this one kernel. After each part, check correctness and time the
   cases; keep the part when the kernel is correct and not slower than before
   it, otherwise revert that part's edits, and go on to the next part either
   way. Do NOT submit after the first improvement: submit after the last part,
   or at the wrap-up time the session deadline gives. Leave a part out only
   when its precondition proves false or it measured worse, and record that
   with the evidence as [playbook-skip:...]; being hard or long is not a
   reason. A stop before then with parts left is sent back."""

BATCH_DEADLINE = (
    "## Session deadline\n"
    "This session ends at {ends} UTC, about {minutes} minutes from now; `date -u` shows the time. Keep working through "
    "the batch's parts until {wrap_up} UTC. From then on start no new part: finish or revert the part in hand, list "
    "every item you did not get to as [playbook-untried:...], and submit. A session cut off mid-turn hands off "
    "nothing.\n"
)


def _utc(epoch: float) -> str:
    return time.strftime("%H:%M", time.gmtime(epoch))


@dataclass(frozen=True)
class BatchSession:
    """What one Implementer session implements as a batch: each part's playbook group and open items, in order, and
    the session's clock once it starts. ``NO_BATCH`` for every other session."""

    parts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    ends_at: float = 0.0
    # From this wall-clock time on the session may submit with parts left.
    wrap_up_at: float = 0.0

    def timed(self, session_sec: float, *, now: float) -> BatchSession:
        reserve = max(BATCH_RESERVE_MIN_SEC, session_sec * BATCH_RESERVE_SHARE)
        return replace(self, ends_at=now + session_sec, wrap_up_at=now + session_sec - reserve)

    def stop_rule(self, default: str) -> str:
        return BATCH_STOP_RULE if self.parts else default

    def deadline_section(self, default: str, *, now: float) -> str:
        if not self.parts:
            return default
        minutes = max(1, round((self.ends_at - now) / 60))
        return BATCH_DEADLINE.format(ends=_utc(self.ends_at), minutes=minutes, wrap_up=_utc(self.wrap_up_at))


NO_BATCH = BatchSession()


_CITATION = re.compile(r"\[playbook:([^\]]*)\]", re.IGNORECASE)
_SKIP_CITATION = re.compile(r"\[playbook-skip:([^\]]*)\]", re.IGNORECASE)
_UNTRIED_CITATION = re.compile(r"\[playbook-untried:([^\]]*)\]", re.IGNORECASE)
_CITED_ID = re.compile(r"\bP\d{1,3}\b", re.IGNORECASE)
# Inside one citation, an entry is a run of item ids at the start of the block or after a `;`, `,` or line break,
# followed by its reason; an id that only appears inside another entry's reason is not cited. In a skip or untried
# citation, after a comma the run opens an entry only when its reason starts right after it ("P6 slower, P7:
# spilled"): in "slower than the split path, P21 would be needed first" P21 is part of P6's reason. A claim lists
# what is in the kernel, each id followed by what it is ("P18 bf16x3 tile, P19 per-shape dispatch"), so there any id
# after a comma opens an entry.
_ENTRY_START = re.compile(r"(?:^|[;,\n])\s*(?=P\d)", re.IGNORECASE)
_ID_RUN = re.compile(r"P\d{1,3}\b(?:\s*(?:,|/|&|\band\b)\s*P\d{1,3}\b)*", re.IGNORECASE)
_REASON_OPENER = re.compile(r"[ \t]*(?:$|[\n:;(=\-\u2013\u2014])")
# The line that closes each submission a session makes; the text after the one before the last is its latest report.
_SUBMIT_LINE = re.compile(r"^[ \t]*SUBMIT_CANDIDATE[ \t]*$", re.IGNORECASE | re.MULTILINE)
# A skip whose reason says the item was never tried or not finished leaves it open rather than settling it. "Not
# implemented" is not among them: a session also writes it of an item whose precondition proved false.
_UNTRIED_REASON = re.compile(
    r"not (?:attempted|tried|started|reached|finished)|untried|"
    r"(?:lack|out|short) of (?:session )?time|ran out of time|deadline|time budget|no time",
    re.IGNORECASE,
)
_NOTE_CHARS = 300

BATCH_UNFINISHED = (
    "The batch is not finished, and {minutes} minutes remain before the wrap-up time. The kernel as it stands is "
    "correct and faster, so keep it and continue from it with part {position} of {parts} (playbook group {group}): "
    "{items} is neither in the kernel nor dropped with evidence. Implement it, check and time it, keep or revert it, "
    "then go on with the parts after it. Leave an item out only as [playbook-skip:<id> <evidence>] when its "
    "precondition proves false or it measured worse; being hard or long is not a reason. End with the [playbook:...], "
    "[playbook-skip:...] and [playbook-untried:...] lines once every part is done or the wrap-up time is reached."
)


@dataclass(frozen=True)
class Citations:
    """What a plan or a session report cites: the items it claims, skips and did not try, and why for the last two."""

    claimed: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    untried: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def any(self) -> bool:
        return bool(self.claimed or self.skipped or self.untried)


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _entries(block: str, *, strict_commas: bool) -> list[tuple[list[str], str]]:
    """Each entry of one citation block: the item ids it opens with, and the reason that follows them."""
    opened: list[tuple[re.Match[str], re.Match[str]]] = []
    cursor = 0
    for start in _ENTRY_START.finditer(block):
        if start.end() < cursor:
            continue
        run = _ID_RUN.match(block, start.end())
        if run is None:
            continue
        if strict_commas and block[start.start()] == "," and not _REASON_OPENER.match(block, run.end()):
            continue
        opened.append((start, run))
        cursor = run.end()
    entries: list[tuple[list[str], str]] = []
    for index, (_, run) in enumerate(opened):
        following = opened[index + 1][0].start() if index + 1 < len(opened) else len(block)
        reason = _text(block[run.end() : following]).strip(" -\u2013\u2014:;,.")
        entries.append(([match.upper() for match in _CITED_ID.findall(run.group(0))], reason[:_NOTE_CHARS]))
    return entries


def latest_report(text: str) -> str:
    """The part of a session's text that holds its latest report, or ``""`` when it wrote none: the stop hook can send
    a session back after it submitted, and an earlier submission's lines describe a kernel that was not measured."""
    text = text or ""
    bounds = [0, *(match.end() for match in _SUBMIT_LINE.finditer(text))]
    segments = [text[start:end] for start, end in zip(bounds, [*bounds[1:], len(text)])]
    for segment in reversed(segments):
        if any(pattern.search(segment) for pattern in (_CITATION, _SKIP_CITATION, _UNTRIED_CITATION)):
            return segment
    return ""


def _cited(pattern: re.Pattern[str], text: str, known: set[str]) -> tuple[list[str], dict[str, str]]:
    found: list[str] = []
    reasons: dict[str, str] = {}
    cited = (
        (item_id, reason)
        for block in pattern.findall(text)
        for item_ids, reason in _entries(block, strict_commas=pattern is not _CITATION)
        for item_id in item_ids
    )
    for item_id, reason in cited:
        if item_id not in known or item_id in found:
            continue
        found.append(item_id)
        if reason:
            reasons[item_id] = reason
    return found, reasons


def read_citations(text: str, known: Iterable[str]) -> Citations:
    """The items ``text`` claims, skips and did not try, among ``known``; a claim wins over any other mention, since a
    report names an item it kept only in part both as kept and as dropped. ``text`` is one plan, or one session's
    latest report (``latest_report``)."""
    known_ids = set(known)
    text = text or ""
    claimed, _ = _cited(_CITATION, text, known_ids)
    skipped, skip_reasons = _cited(_SKIP_CITATION, text, known_ids)
    untried, untried_reasons = _cited(_UNTRIED_CITATION, text, known_ids)
    for item_id in list(skipped):
        if _UNTRIED_REASON.search(skip_reasons.get(item_id, "")):
            skipped.remove(item_id)
            if item_id not in untried:
                untried.append(item_id)
                untried_reasons[item_id] = skip_reasons[item_id]
    skipped = [item_id for item_id in skipped if item_id not in claimed]
    untried = [item_id for item_id in untried if item_id not in claimed and item_id not in skipped]
    reasons = {item_id: skip_reasons[item_id] for item_id in skipped if item_id in skip_reasons}
    reasons.update({item_id: untried_reasons[item_id] for item_id in untried if item_id in untried_reasons})
    return Citations(claimed=tuple(claimed), skipped=tuple(skipped), untried=tuple(untried), reasons=reasons)


def cited_items(plan_text: str, known: Iterable[str]) -> tuple[list[str], list[str]]:
    """The items a plan claims and the ones it declines (skipped or not tried), among ``known``."""
    cites = read_citations(plan_text, known)
    return list(cites.claimed), [*cites.skipped, *cites.untried]


def batch_unfinished(batch: BatchSession, message: str | None, *, now: float) -> str:
    """Why a batch session stopping with ``message`` as its last word must go on, or ``""`` when it may stop: past its
    wrap-up time, with every item in the kernel or dropped, or when its last word could not be read (``None``)."""
    if message is None or now >= batch.wrap_up_at:
        return ""
    cites = read_citations(latest_report(message), [item for _, items in batch.parts for item in items])
    settled = {*cites.claimed, *cites.skipped}
    for position, (group_id, items) in enumerate(batch.parts, start=1):
        left = [item for item in items if item not in settled]
        if left:
            return BATCH_UNFINISHED.format(
                minutes=max(1, round((batch.wrap_up_at - now) / 60)),
                position=position,
                parts=len(batch.parts),
                group=group_id,
                items=", ".join(left),
            )
    return ""
