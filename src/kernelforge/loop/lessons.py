# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-iteration factual records written by the resumed Implementer session.

Each iteration records actions that may exist only in the Implementer's
conversation, including attempts reverted before the final candidate:

  * The implementer session is RESUMED (so the model still has the whole
    conversation in context) under a READ-ONLY tool policy and with no hooks.
  * It is asked to record every direction it actually tried and the observed
    result of each, without deciding whether later iterations should continue or
    abandon a direction.
  * The returned text is written to ``forge_experiments/lessons/iter_NNN.md``
    by THIS module, not by the model (mirrors ``profile_analyst``), so the
    session needs no write access anywhere.
  * After the outer loop decides KEEP/REVERT, it appends one machine-written
    ``OUTCOME:`` line. The resumed session records attempted actions and observed
    results; the loop records what canonical validation and measurement decided.

The next iteration's prompt gets the last few documents verbatim plus the
absolute path of the directory, so the agent can inspect the full factual
history on demand rather than carrying it in context. The model's response is
stored as free-form text; apart from the ``HELD-FIXED:`` marker lines described
below, no output schema or headline contract is imposed.

Every document also carries the scope its observations were taken under: the
scored cases they were measured on, the constants that were pinned while
measuring, and whether the iteration measured a negative at all. A negative
result is evidence only inside that scope. Outside it — another scored case, or
a pinned value that has since moved in the declared source files — the record is
rendered as re-openable and the next iteration is told it needs a fresh
measurement rather than the note. A document that recorded no negative has
nothing to re-open, so an unrecorded premise does not re-open it.

A document may also close a direction by claiming it CANNOT be reached at
all. That claim is not a measurement and is not re-opened by the same things a
measurement is, so it carries its own obligation: the cheapest experiment that
would have falsified it, actually run. Until that experiment exists the claim
is rendered re-openable no matter how many numbers the document quotes around
it — a real measurement standing next to a false premise is exactly how the
premise survives review. Which sentences are such a claim is the summarizing
session's own answer, written on a marker line: nothing here reads the prose
for the word, so an untested premise stated without the marker is recorded as
unanswered rather than as an obligation, and what keeps it from closing an axis
is then the citation rule printed beside the document, not this check.

The same marker also carries the opposite outcome, because an experiment run
against a "cannot" can come out against it. A record reporting its own premise
FALSE is not an obligation discharged; it is the axis shown reachable, and it
is rendered as a direction the next iteration must re-enter rather than one it
may. One document carries one such verdict, and the strongest of its markers
wins: a record making three "cannot" claims while answering for one of them
certifies nothing about the other two, so every rendering that leaves a
document suppressing anything says which claim was answered and that the rest
were not.

Where a scope could not be checked — a source that could not be read, one that
could not be parsed, or a name the source mentions without binding it to
anything readable — the rendered note says it was not checked instead of
reporting the constant as gone. "Not checked" and "not assigned" are different
facts, and a wrong premise closes an axis that a missing one only re-opens.
Only a name absent from a source set that was checked in full is reported as
unassigned.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from kernelforge.durable_io import atomic_write_text

log = logging.getLogger(__name__)

# How many recent lesson documents are inlined into the implementer prompt. Older
# iterations stay on disk and are reachable through the directory pointer.
DEFAULT_RECENT_LESSONS = 5

# Hard ceiling on the inlined block. The per-document word budget below is a
# soft instruction the model can overshoot; this is the deterministic backstop.
# Oldest documents are dropped first (mirrors ``prompt_view``'s trimming), so a
# single verbose document degrades the window instead of blowing the prompt.
DEFAULT_MAX_PROMPT_CHARS = 10000

# Soft word budget stated to the summarizer. Deliberately not enforced by
# truncation: cutting a document mid-sentence would corrupt the record of an
# attempted direction.
SUMMARY_WORD_BUDGET = 250

# Below this many seconds left, skip the summarizer and record the outcome
# only. This is about whether there is time to PRODUCE the summary — it is
# deliberately NOT the loop's session-admission reserve, which is orders of
# magnitude larger. A campaign that stops for the day is resumed later, and
# that next session reads this very document, so the last iteration of a
# session is exactly the one whose record matters most.
SUMMARY_MIN_SECONDS = 120

# Session end reasons that mean the implementer was cut off rather than finishing.
# The summarizer is told to flag these, so a later iteration can tell an
# unfinished exploration apart from a settled negative result.
_CUTOFF_END_REASONS = frozenset({"turn_cap", "block_budget_exhausted"})

# Marker lines that carry a document's validity condition. ``SCOPE:`` is written
# by the loop from what it actually measured; ``HELD-FIXED:`` is asked of the
# summarizer, which is the only party that knows what a sweep pinned.
SCOPE_PREFIX = "SCOPE:"
HELD_FIXED_PREFIX = "HELD-FIXED:"

# The companion marker to ``HELD-FIXED:``. The loop can see its own verdict on
# the one candidate it measured; it cannot see the four directions the session
# tried and reverted before that one, and those are where most of a document's
# negatives live. So the summarizer -- the only party that can see them -- is
# asked to state on one line whether ANY direction measured worse.
NEGATIVES_PREFIX = "NEGATIVES:"

# The marker for the other kind of closure. ``NEGATIVES:`` answers "did
# anything measure worse"; this one answers "did anything here claim a
# direction cannot be reached, and what was run against that claim". The two
# are independent: the closures that suppressed winning routes in past
# campaigns quoted real measurements AND rested on an untested premise, so a
# document can need both lines.
DISPROOF_PREFIX = "DISPROOF:"

# What that line may say to mean "no direction measured worse". Anything else
# after the marker is read as naming at least one negative; an absent or empty
# marker is read as nothing recorded, which is not a "no".
_NO_NEGATIVES_WORDS = frozenset({"none", "no", "nothing", "n/a", "na"})

# What a ``DISPROOF:`` line may say to mean "this record claims no direction is
# unreachable", and the words that open one meaning "I ran the experiment".
# Everything else after the marker — including a named experiment nobody ran — is read as
# an outstanding obligation, because the direction that is safe to be wrong in
# is the one that re-opens an axis rather than the one that closes it.
_NO_CLAIM_WORDS = frozenset({"none", "no", "nothing", "n/a", "na"})
_DISPROOF_RUN_WORDS = ("tested", "ran")

# The words for the other outcome of that same experiment. They are deliberately
# NOT run words: "tested" says the falsifying experiment happened, these say it
# came out AGAINST the claim — the "cannot" is wrong and the axis it closed is
# reachable. Read as a run word, "DISPROOF: falsified — gfx950 accepts the
# instruction" scored an obligation as discharged and left the closure it had
# just destroyed still suppressing the route, which is the inversion this whole
# marker exists to prevent.
_DISPROVED_WORDS = ("disproved", "disproven", "falsified")

# What a scope field says when nothing was recorded for it. Spelled out rather
# than left blank so a reader cannot mistake an unrecorded scope for a universal
# one -- that mistake is what turns one measurement into a standing ban.
NOT_RECORDED = "(not recorded)"

_HELD_FIXED_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*([^,;]+)")

# What may sit inside a scored case id ("decode-t16", "gemm_4096.bf16"). Used as
# the boundary around an id so one id cannot match inside a longer one.
_ID_CHAR = r"[A-Za-z0-9_.\-]"

SUMMARIZER_ROLE = (
    "You are creating a factual record of an autonomous GPU-kernel optimization "
    "session. You are the same agent that ran the session and still have the "
    "whole conversation in context. Do not edit code in this turn. Report only "
    "actions actually attempted and results actually observed. Do not judge "
    "whether a direction is valuable, exhausted, or worth revisiting, and do not "
    "tell future iterations what they should or should not do."
)


def is_cutoff(end_reason: str) -> bool:
    """Whether a session ended by exhausting a budget rather than finishing."""
    return (end_reason or "").strip() in _CUTOFF_END_REASONS


@dataclass(frozen=True)
class LessonScope:
    """The conditions one iteration's observations were taken under.

    ``cases`` are the scored cases the iteration measured on — the whole suite,
    or the subset a restricted lane was assigned. ``held_fixed`` are the
    constants the session pinned while measuring, as ``(name, value)`` pairs.
    ``lane_restricted`` records that ``cases`` is narrower than the suite
    because the round said so, not because the measurement happened to skip
    the rest.

    ``carries_negative`` is whether any direction recorded in the document
    measured worse. It decides whether an unrecorded premise matters — a
    document with no negative in it has nothing to re-open.

    It has two sources, because neither sees the whole document. The loop sees
    only the one final candidate it measured: a revert, a crash, a build
    failure, an in-session rejection. It cannot see a direction the session
    tried and reverted before that candidate, and the record is explicitly
    asked to include those. So the summarizer's ``NEGATIVES:`` marker supplies
    the rest, and the loop's own verdict overrides it when the two disagree.

    ``None`` means neither could answer: a document written before this field
    existed, or one whose summarizer left the marker out. It is not a "no" —
    it is treated exactly as conservatively as a recorded negative.

    ``disproof`` answers the question a measurement cannot answer: when the
    document claims a direction CANNOT be reached, what was run against that
    claim. It exists because "does the closure carry a number" turned out not
    to discriminate. Three closures that each suppressed a winning route were
    reviewed: one carried no number, one quoted a real 0.206 → 0.237 ms
    regression, one quoted a real 153.9 us row. All three were wrong for the
    same reason — the feasibility premise beside the number ("this build
    cannot reach it", "this needs a data-dependent branch", "that is not one
    of the editable files") had never been tested, and two of them were false.
    A number is evidence about the variant that was run; it is not evidence
    about the route that was never attempted.

    Its five states are five different facts:

      * the experiment text — the cheapest falsifying experiment, named and
        actually run, with the claim surviving it. Only this keeps a "cannot"
        in scope and suppressing;
      * ``CLAIM_DISPROVED`` followed by what was run — the same experiment,
        come out the other way: the claim is FALSE and the direction it closed
        is reachable. This is the strongest answer the marker can carry, and
        the only one that is a fact about the route rather than about the
        variant that was run, so it does not merely re-open the direction, it
        tells the next iteration to re-enter it;
      * ``UNDISPROVEN_CLAIM`` — the document claims a direction cannot be
        reached and nothing was run against that claim. Re-openable, whatever
        else the document measured;
      * ``NO_FEASIBILITY_CLAIM`` — the document claims no such thing, so there
        is no obligation to discharge and its measured negatives are read on
        their own terms;
      * ``None`` — nobody answered: a document from before this field existed,
        or a summarizer that left the marker out. Not a "no claim": it does
        not certify that anything was tested, so a bare "cannot" sentence
        inside such a document closes nothing on its own. It is deliberately
        NOT rendered as an outstanding obligation either, because that would
        convict every document written before the field of a claim it may
        never have made, and a verdict every document receives stops
        discriminating between them.

    One value covers the whole document, and that is a known limit rather than
    a claim about every sentence in it. A record making three "cannot" claims
    and answering for one of them yields one verdict, so a discharged or
    disproved answer here says what happened to the claim someone answered
    for and nothing at all about the others; the silence about them is not
    visible in this field. It is bounded on the dangerous side only: an
    outstanding obligation beats a discharged one when both are recorded
    (``parse_disproof_marker``), and the renderings that leave a document
    suppressing anything say aloud that the other claims are uncertified.
    Making the verdict per-claim would have to put a list where this field
    holds one value, give the SCOPE line a repeatable field with its own
    separator, and keep a line written under the present format parseable by
    the reader that follows — a format change, and a wider one than the branch
    that made the obligation work at all.
    """

    cases: tuple[str, ...] = ()
    held_fixed: tuple[tuple[str, str], ...] = ()
    lane_restricted: bool = False
    carries_negative: bool | None = None
    disproof: str | None = None


# How the negative flag is spelled on the SCOPE line. All three states are
# written out, including the unknown one: claiming "no measured negative" over
# a document nobody checked is the false statement this whole line exists to
# prevent. A line carrying none of the three is one from before the flag
# existed, and that is the same third fact, not a "no".
CARRIES_NEGATIVE = "carries a measured negative"
NO_NEGATIVE = "no measured negative"
NEGATIVE_NOT_RECORDED = "whether anything measured worse was not recorded"

# How the disproof obligation is spelled on the SCOPE line. The two sentinel
# answers are their own rendering, so the round trip needs no second
# vocabulary; the two answers that carry evidence are that evidence written
# behind ``DISPROOF_RUN`` or ``CLAIM_DISPROVED``, which say which way the
# experiment came out. As with the negative flag, the unrecorded state is
# written out rather than left off the line: a reader who cannot see the
# difference between "no such claim" and "nobody asked" will collapse them
# into the first. No rendering here is a prefix of another, so the fields
# parse the same whatever order they are read in — and ``CLAIM_DISPROVED``
# and ``UNDISPROVEN_CLAIM`` are the pair that has to stay apart, since
# "disproved by X" and "not disproved" are opposite verdicts and a reader
# matching one inside the other would report an axis closed exactly where it
# was proved open.
NO_FEASIBILITY_CLAIM = "no feasibility claim"
UNDISPROVEN_CLAIM = "feasibility claim not disproved"
DISPROOF_RUN = "feasibility claim tested by "
CLAIM_DISPROVED = "feasibility claim disproved by "
DISPROOF_NOT_RECORDED = "whether a feasibility claim was disproved was not recorded"

# Longest named experiment kept on the SCOPE line. The line is inlined verbatim
# into the next iteration's prompt, and a summarizer that answers the "cheapest
# experiment" question with a paragraph must not push the fields after it out
# of a reader's sight. A cut rendering is marked, as elsewhere in this file.
_MAX_DISPROOF_CHARS = 120


def is_claim_disproved(disproof: str | None) -> bool:
    """Whether a disproof answer reports the document's own "cannot" as FALSE.

    The disproved answer carries evidence, so it cannot be one flat sentinel;
    it is that sentinel followed by what was run, and this is the one place
    that knows it. Callers ask here rather than comparing prefixes, so the
    verdict that re-opens an axis is never missed by a reader that only knew
    about the sentinels it could compare with ``==``.
    """
    return disproof is not None and disproof.startswith(CLAIM_DISPROVED)


def _disproved_evidence(text: str) -> str | None:
    """What stands behind a disproved answer, or ``None`` if it is not one.

    Matched against the prefix without its trailing space, so an answer that
    reports the claim false and carries nothing behind it is still recognised
    as that answer. Recognising it is what lets it be rendered as an open
    obligation; a reading that matched nothing would record it as a question
    nobody put, and the one thing the line certainly did was put it.
    """
    if text == CLAIM_DISPROVED.rstrip():
        return ""
    if text.startswith(CLAIM_DISPROVED):
        return text[len(CLAIM_DISPROVED) :].strip()
    return None


def _clipped(text: str) -> str:
    """One named experiment at the length the SCOPE line will carry."""
    if len(text) > _MAX_DISPROOF_CHARS:
        return text[:_MAX_DISPROOF_CHARS] + _TRUNCATION_MARK
    return text


def _disproof_field(disproof: str | None) -> str:
    """One scope's disproof answer as it appears on the SCOPE line.

    A named experiment is free text a model wrote, so it is folded onto one
    line and its pipes become slashes before it joins a pipe-separated line:
    an experiment name must not be able to forge a field. An answer that folds
    away to nothing is rendered as unrecorded rather than as an experiment,
    which is what an empty answer actually is.

    A disproved claim whose evidence folds away to nothing is rendered as an
    outstanding obligation instead. "The premise is false" with nothing behind
    it cannot be repeated by the iteration that reads it, exactly as an unnamed
    experiment cannot, and the answer that survives being wrong is the one that
    re-opens the axis without asserting anything about the route.
    """
    if disproof is None:
        return DISPROOF_NOT_RECORDED
    text = " ".join(disproof.split()).replace("|", "/")
    if not text:
        return DISPROOF_NOT_RECORDED
    if text in (NO_FEASIBILITY_CLAIM, UNDISPROVEN_CLAIM):
        return text
    evidence = _disproved_evidence(text)
    if evidence is not None:
        return CLAIM_DISPROVED + _clipped(evidence) if evidence else UNDISPROVEN_CLAIM
    return DISPROOF_RUN + _clipped(text)


def format_scope_line(scope: LessonScope) -> str:
    """One machine-written line stating what a document's results are valid for."""
    cases = ", ".join(scope.cases) if scope.cases else NOT_RECORDED
    held = ", ".join(f"{name}={value}" for name, value in scope.held_fixed) if scope.held_fixed else NOT_RECORDED
    parts = [f"{SCOPE_PREFIX} measured on {cases}", f"held fixed {held}"]
    if scope.lane_restricted:
        parts.append("lane restricted to the cases above")
    if scope.carries_negative is None:
        parts.append(NEGATIVE_NOT_RECORDED)
    else:
        parts.append(CARRIES_NEGATIVE if scope.carries_negative else NO_NEGATIVE)
    parts.append(_disproof_field(scope.disproof))
    return " | ".join(parts)


def parse_scope_line(text: str) -> LessonScope | None:
    """Recover a scope from a document, or None when it carries no scope line."""
    line = ""
    for candidate in (text or "").splitlines():
        if candidate.strip().startswith(SCOPE_PREFIX):
            line = candidate.strip()
    if not line:
        return None
    fields = [part.strip() for part in line[len(SCOPE_PREFIX) :].split("|")]
    cases: tuple[str, ...] = ()
    held: tuple[tuple[str, str], ...] = ()
    lane_restricted = False
    carries_negative: bool | None = None
    disproof: str | None = None
    for part in fields:
        if part.startswith("measured on "):
            listed = part[len("measured on ") :].strip()
            if listed != NOT_RECORDED:
                cases = tuple(item.strip() for item in listed.split(",") if item.strip())
        elif part.startswith("held fixed "):
            listed = part[len("held fixed ") :].strip()
            if listed != NOT_RECORDED:
                held = _parse_pairs(listed)
        elif part.startswith("lane restricted"):
            lane_restricted = True
        elif part.startswith(CARRIES_NEGATIVE):
            carries_negative = True
        elif part.startswith(NO_NEGATIVE):
            carries_negative = False
        elif part.startswith(NEGATIVE_NOT_RECORDED):
            carries_negative = None
        elif (evidence := _disproved_evidence(part)) is not None:
            disproof = CLAIM_DISPROVED + evidence if evidence else UNDISPROVEN_CLAIM
        elif part.startswith(DISPROOF_RUN):
            disproof = part[len(DISPROOF_RUN) :].strip() or UNDISPROVEN_CLAIM
        elif part.startswith(UNDISPROVEN_CLAIM):
            disproof = UNDISPROVEN_CLAIM
        elif part.startswith(NO_FEASIBILITY_CLAIM):
            disproof = NO_FEASIBILITY_CLAIM
        elif part.startswith(DISPROOF_NOT_RECORDED):
            disproof = None
    return LessonScope(
        cases=cases,
        held_fixed=held,
        lane_restricted=lane_restricted,
        carries_negative=carries_negative,
        disproof=disproof,
    )


def _parse_pairs(text: str) -> tuple[tuple[str, str], ...]:
    """``NAME=VALUE`` pairs from one comma-separated list. First value wins."""
    found: dict[str, str] = {}
    for name, value in _HELD_FIXED_PAIR.findall(text or ""):
        found.setdefault(name, value.strip())
    return tuple(found.items())


def parse_held_fixed(text: str) -> tuple[tuple[str, str], ...]:
    """The constants the summarizer recorded as pinned, across a document.

    Only ``HELD-FIXED:`` lines are read: a pair found anywhere in the prose is
    as likely to be a result as a premise, and a wrong premise is worse than a
    missing one — a missing one re-opens the axis, a wrong one closes it.
    """
    found: dict[str, str] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(HELD_FIXED_PREFIX):
            continue
        for name, value in _parse_pairs(stripped[len(HELD_FIXED_PREFIX) :]):
            found.setdefault(name, value)
    return tuple(found.items())


def parse_negatives_marker(text: str) -> bool | None:
    """Whether the document says any direction it records measured worse.

    Three outcomes, because they are three different facts:

      * ``True``  — a ``NEGATIVES:`` line names at least one direction that
        measured worse;
      * ``False`` — a ``NEGATIVES:`` line says none did;
      * ``None``  — there is no usable marker. An older document, or a reply
        that ignored the contract. That is not a "no": the question was never
        answered, and answering it "no" on the document's behalf would promote
        an unchecked negative into a standing ban.

    Any line naming something wins over a line saying none: the marker is a
    presence check, and a document that names one negative carries one.
    """
    verdict: bool | None = None
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("-*# ").strip()
        if not stripped.upper().startswith(NEGATIVES_PREFIX):
            continue
        payload = stripped[len(NEGATIVES_PREFIX) :].strip().strip("*.` ").strip()
        if not payload:
            continue
        if payload.lower() in _NO_NEGATIVES_WORDS:
            if verdict is None:
                verdict = False
            continue
        return True
    return verdict


def parse_disproof_marker(text: str) -> str | None:
    """What the document says it ran against its own "cannot" claims.

    Five outcomes, matching ``LessonScope.disproof``:

      * ``CLAIM_DISPROVED`` plus what was run — a line opens with a word from
        ``_DISPROVED_WORDS`` and names the evidence: the experiment happened
        and the claim lost;
      * the experiment text — a line opens with a run word and names the
        experiment: it happened and the claim survived;
      * ``UNDISPROVEN_CLAIM`` — a line says a direction cannot be reached but
        the experiment that would settle it was not run, or says it was run —
        or won — without naming what was run. An unnamed experiment is not a
        disproof either way: the whole point of the marker is that a later
        iteration can repeat it;
      * ``NO_FEASIBILITY_CLAIM`` — a line says this record claims no direction
        is unreachable;
      * ``None`` — no usable marker. An older document or a reply that ignored
        the contract; the question was never put, which is not an answer to it.

    A disproved claim wins over every other answer, an outstanding obligation
    wins over a discharged one, and all of them win over "no claim". The first
    two rankings point the same way: a disproved claim and an undisproven one
    both re-open a direction, and ranking the disproved one first only ever
    turns "you may re-enter this" into "this is reachable, re-enter it". A
    document that says "cannot" once and stays silent about it elsewhere
    carries the claim, exactly as one that names one negative carries a
    negative — and the direction to be wrong in is the one that re-opens an
    axis, never the one that closes it.

    ``DISPROOF: tested — <experiment>`` still means the experiment ran and the
    claim survived, which is the only answer that leaves a "cannot"
    suppressing. It is read that way and not conservatively because the
    summarizer is now taught three outcome words, not two: a session holding a
    falsifying result has ``disproved`` to write, so choosing ``tested`` is an
    answer about the outcome rather than silence about it. Reading ``tested``
    as ambiguous instead would leave no word in the contract that can ever
    discharge an obligation, which is not a stricter version of this mechanism
    but a different one — "no feasibility claim ever closes anything" — and
    that decision belongs to the loop's policy, not to the parser for one
    marker line. What is left of the risk is bounded: the named text is
    rendered verbatim beside the document, so a reader meets the evidence the
    word was attached to.
    """
    verdict: str | None = None
    disproved: str | None = None
    undisproven = False
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("-*# ").strip()
        if not stripped.upper().startswith(DISPROOF_PREFIX):
            continue
        payload = stripped[len(DISPROOF_PREFIX) :].strip().strip("*.` ").strip()
        if not payload:
            continue
        if payload.lower() in _NO_CLAIM_WORDS:
            if verdict is None:
                verdict = NO_FEASIBILITY_CLAIM
            continue
        head, _, rest = payload.partition(" ")
        named = rest.strip().lstrip("-—:,").strip()
        word = head.lower().strip(":,")
        if named and word in _DISPROVED_WORDS:
            if disproved is None:
                disproved = CLAIM_DISPROVED + named
            continue
        if named and word in _DISPROOF_RUN_WORDS:
            verdict = named
            continue
        undisproven = True
    if disproved is not None:
        return disproved
    if undisproven:
        return UNDISPROVEN_CLAIM
    return verdict


# Longest rendering of one assigned value kept for comparison and display. A
# constant pinned to a whole expression is rare; a wrapped one would only make
# the rendered note unreadable. A cut rendering is marked, so a truncated
# expression reaching a prompt cannot be read as a complete one.
_MAX_VALUE_CHARS = 60
_TRUNCATION_MARK = " ..."


def _assigned_names(target: ast.AST) -> list[str]:
    """The names one assignment target binds. Subscripts bind no name."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _assigned_names(element)]
    return []


def _value_text(node: ast.AST) -> str:
    """One assigned value as source text, for comparison against a pin."""
    rendered = " ".join(ast.unparse(node).split())
    if len(rendered) > _MAX_VALUE_CHARS:
        return rendered[:_MAX_VALUE_CHARS] + _TRUNCATION_MARK
    return rendered


def _is_constant_expr(node: ast.AST) -> bool:
    """Whether a node is a literal the source pins, not a name it forwards.

    ``num_warps=8`` pins 8; ``BLOCK_N=BLOCK_N`` forwards a caller's local and
    says nothing about the value. Only the first is a fact about the source.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_constant_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_constant_expr(node.left) and _is_constant_expr(node.right)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return all(_is_constant_expr(element) for element in node.elts)
    return False


def _unpacked_pairs(target: ast.AST, value: ast.AST) -> list[tuple[ast.AST, ast.AST]] | None:
    """``a, b = 1, 2`` element by element, or None when it cannot be paired.

    ``BLOCK_M, BLOCK_N = 64, 32`` pins each name to its own element. Recording
    the whole right-hand side against both would render "BLOCK_N is now
    (64, 32)" — a false statement about the source. A starred target, a length
    mismatch, or a right-hand side that is not a literal sequence cannot be
    paired at all, and the caller reports those as a value it did not read.
    """
    if not (
        isinstance(target, ast.Tuple | ast.List)
        and isinstance(value, ast.Tuple | ast.List)
        and len(target.elts) == len(value.elts)
    ):
        return None
    if any(isinstance(element, ast.Starred) for element in (*target.elts, *value.elts)):
        return None
    return list(zip(target.elts, value.elts, strict=True))


def _as_number(text: str) -> float | None:
    """One rendered value as a number, or None when it is not one.

    ``16`` and ``16.0`` are the same pin written two ways, and ``0x10`` is a
    third. Comparing the renderings as text reports the kernel as having moved
    when nothing moved, which re-opens a negative on a formatting difference.
    """
    try:
        value = ast.literal_eval((text or "").strip())
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _still_pinned(pinned: str, observed: Sequence[str]) -> bool:
    """Whether one recorded pin is among the values the source now assigns."""
    if pinned in observed:
        return True
    number = _as_number(pinned)
    if number is None:
        return False
    return any(_as_number(value) == number for value in observed)


def scan_constant_values(source: str, names: Iterable[str]) -> dict[str, tuple[str, ...]] | None:
    """What each named constant is bound to in ``source``.

    Four outcomes, because they are four different facts:

      * a name mapped to one or more values — it is bound, here is what to;
      * a name mapped to an EMPTY tuple — the name is in the source but nothing
        readable binds it: only a ``tl.constexpr`` parameter, only a keyword
        argument forwarding a caller's local, an unpairable tuple unpacking. Its
        current value was not checked, which is not the same as gone;
      * a name absent from the mapping — the source parsed and never mentions
        it at all, which is a change of premise: whatever was pinned is gone;
      * ``None`` — the source could not be parsed, so nothing is known about
        it. A caller must not report that as a name the source dropped.

    A binding is an ``ast.Assign`` target (paired element by element through a
    tuple unpacking), an ``ast.AnnAssign`` or ``ast.NamedExpr`` value (never the
    annotation), a string key of a dict literal, which is how a tuning table
    pins a constant, and a keyword argument whose value is a literal. That last
    one is where Triton tile sizes and warp counts actually live —
    ``num_warps=8``, ``BLOCK_N=128``, ``triton.Config({...}, num_warps=8)`` — so
    excluding it would report a pinned constant as gone. ``BLOCK_N=BLOCK_N``
    passes a name rather than a literal and is recorded as unread, not as a
    value.
    """
    wanted = {name for name in names if name}
    try:
        tree = ast.parse(source or "")
    except (SyntaxError, ValueError):
        return None
    if not wanted:
        return {}

    found: dict[str, list[str]] = {}

    def mention(name: str) -> None:
        """Mark a name as present in the source with no value read for it."""
        if name in wanted:
            found.setdefault(name, [])

    def record(name: str, node: ast.AST) -> None:
        if name not in wanted:
            return
        values = found.setdefault(name, [])
        value = _value_text(node)
        if value not in values:
            values.append(value)

    def bind(target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Tuple | ast.List):
            pairs = _unpacked_pairs(target, value)
            if pairs is None:
                # Bound, but to a share of the right-hand side this cannot
                # read. Recording the whole side would invent a value.
                for name in _assigned_names(target):
                    mention(name)
                return
            for element, element_value in pairs:
                bind(element, element_value)
            return
        for name in _assigned_names(target):
            record(name, value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target, node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    record(key.value, value)
        elif isinstance(node, ast.keyword) and node.arg:
            if _is_constant_expr(node.value):
                record(node.arg, node.value)
            else:
                mention(node.arg)
        elif isinstance(node, ast.arg):
            mention(node.arg)
        elif isinstance(node, ast.Name):
            mention(node.id)
        elif isinstance(node, ast.Attribute):
            mention(node.attr)
    return {name: tuple(values) for name, values in found.items()}


def scan_sources_with_coverage(
    sources: Sequence[str | None], names: Iterable[str]
) -> tuple[dict[str, tuple[str, ...]] | None, bool]:
    """``scan_constant_values`` across a source set, plus whether it was whole.

    A ``None`` entry is a declared file that could not be read; a file that
    could not be parsed is the same fact. Either one makes the coverage flag
    (the second element) ``False``, and a caller must then not report a name
    missing from the mapping as one the source set dropped — it may be sitting
    in the file that was never checked.

    A constant bound in any file that was checked is bound: tile, dispatch and
    JIT constants move between the anchor kernel and its siblings, and a
    constant that moved is not a constant that is gone. That union errs on the
    permissive side — a value matching the pin in dead code, or in an unrelated
    helper's local, reads as "unchanged" and keeps a negative in scope that a
    per-file check would re-open. It is the accepted cost of not reporting a
    moved constant as a deleted one.

    When nothing at all could be checked the mapping is ``None``.
    """
    wanted = list(names)
    combined: dict[str, list[str]] = {}
    checked_any = False
    complete = True
    for source in sources:
        found = None if source is None else scan_constant_values(source, wanted)
        if found is None:
            complete = False
            continue
        checked_any = True
        for name, values in found.items():
            merged = combined.setdefault(name, [])
            for value in values:
                if value not in merged:
                    merged.append(value)
    if not checked_any:
        return None, False
    return {name: tuple(values) for name, values in combined.items()}, complete


def scan_sources_for_constants(
    sources: Sequence[str | None], names: Iterable[str]
) -> dict[str, tuple[str, ...]] | None:
    """``scan_sources_with_coverage`` without the coverage flag.

    Only for a caller that does not distinguish a partly-checked source set
    from a whole one. A caller that renders a premise must use
    ``scan_sources_with_coverage``: without the flag, a name absent from the
    mapping cannot be told apart from a name in the one file that failed.
    """
    return scan_sources_with_coverage(sources, names)[0]


def _as_sources(
    kernel_source: str | Sequence[str | None] | None,
) -> list[str | None]:
    """One source text, several of them, or none, as one list.

    A ``None`` INSIDE the sequence is kept, not dropped: that is a declared
    file the caller could not read, and it has to reach the scan as a source
    that was not checked rather than vanish into a shorter list.
    """
    if kernel_source is None:
        return []
    if isinstance(kernel_source, str):
        return [kernel_source]
    return list(kernel_source)


def scope_conflicts(
    scope: LessonScope | None,
    *,
    current_cases: Sequence[str] = (),
    kernel_source: str | Sequence[str | None] | None = None,
) -> tuple[str, ...]:
    """Why a recorded negative cannot be cited as-is right now.

    Empty means the scope still holds and the negative stands. Every reason is
    phrased for the prompt, because the reader that has to act on it is the
    next Implementer session.

    ``kernel_source`` is the text of one source file, or of every declared one
    with ``None`` in place of any that could not be read, or ``None`` when none
    could be read at all. A source that could not be read or parsed, and a name
    the source mentions without binding it to a literal, both yield "was not
    checked". Only a name absent from a source set checked in full is reported
    as "is not assigned".

    A document whose scope records no measured negative is never re-opened over
    an unrecorded premise: there is no negative in it to re-open.

    A claim that a direction cannot be reached AT ALL is deliberately not
    answered here. It is not re-opened by a case it was not measured on or by a
    constant that has moved, but by never having been tested, so it is read off
    ``LessonScope.disproof`` in ``_validity_note`` instead. A caller asking
    whether a document still closes anything has to ask both.
    """
    if scope is None:
        return ()
    reasons: list[str] = []
    if scope.cases:
        outside = [case for case in current_cases if case not in scope.cases]
        if outside:
            reasons.append("not measured on " + ", ".join(outside))
    else:
        reasons.append("the cases it was measured on were not recorded")

    if not scope.held_fixed:
        # Only a document that actually carries a negative can be re-opened by
        # not knowing what was pinned. One that carries none — or was written
        # before the flag existed, which is not a "no" — is treated the same
        # way as a recorded negative.
        if scope.carries_negative is not False:
            reasons.append("the constants it was measured under were not recorded")
        return tuple(reasons)

    sources = _as_sources(kernel_source)
    observed, complete = scan_sources_with_coverage(sources, [name for name, _ in scope.held_fixed])
    if observed is None:
        if not sources:
            reasons.append("held-fixed values were not checked against the current kernel")
        elif any(source is None for source in sources):
            reasons.append("held-fixed values were not checked: the declared source could not be read or parsed")
        else:
            reasons.append("held-fixed values were not checked: the declared source could not be parsed")
        return tuple(reasons)

    for name, value in scope.held_fixed:
        values = observed.get(name)
        measured = f"(pinned at {value} when this was measured)"
        if values is None and complete:
            reasons.append(f"{name} is not assigned in the kernel source checked {measured}")
        elif values is None:
            reasons.append(
                f"{name} was not checked: part of the declared source could not be read or parsed {measured}"
            )
        elif not values:
            reasons.append(f"{name} was not checked: the source names it but binds no literal to it {measured}")
        elif not _still_pinned(value, values):
            # An observation, not an inference: these values were read out of
            # a source that was checked, whatever happened to the rest.
            reasons.append(f"{name} is now {'/'.join(values)} {measured}")
    return tuple(reasons)


def cases_named_in(text: str, case_ids: Iterable[str]) -> tuple[str, ...]:
    """The scored case ids ``text`` names as whole identifiers.

    How a restricted lane's scope is recovered: the assignment names the cases
    it is allowed to move, and that restriction is what must travel with the
    lane's negative results.

    A raw substring test would let one id swallow another — ``decode-t1`` reads
    as named by a plan that says ``decode-t16`` — and that is the dangerous
    direction: it widens a scope, making a negative look valid for a case it
    was never measured on. An id counts only where it is not part of a longer
    identifier.
    """
    body = text or ""
    return tuple(
        case_id
        for case_id in case_ids
        if case_id and re.search(rf"(?<!{_ID_CHAR}){re.escape(case_id)}(?!{_ID_CHAR})", body)
    )


# The four ways a "cannot" has been wrong before, named inline so that a record
# can be asked about them now rather than when the knowledge card documenting
# them with worked examples lands; that card's absolute path is appended to the
# end of this text.
#
# The question is deliberately "which of these did you consider and why does
# each not apply", never "did you try these four". The failure mode being
# corrected is enumerating a closed list of routes and calling it exhaustive,
# and a checklist read as a list of routes only makes the closed list longer.
REACH_CLASSES = """\
Wherever you write that something cannot be reached, also say which of the four
reach classes below you considered for it and why each one does not apply. They
are not routes to try and tick off. They are the four shapes an "unreachable"
claim has taken while being false, so read them as questions about your own
premise rather than as an inventory of what is left:

  (a) REBIND AN INSTALLED SYMBOL — replace a class, function, or attribute that
      an installed package binds, from inside a file you are permitted to edit.
  (b) INJECT DEVICE-SIDE SOURCE THROUGH THE FRAMEWORK'S OWN HOOK — import_source,
      pragma_import_c, an intrinsic, or inline asm, called through the
      framework's own extern-call path.
  (c) CHANGE A MODULE-LEVEL CONSTANT ANOTHER MODULE'S DISPATCH READS — including
      one whose default comes from os.environ, which does not put it outside the
      edit surface.
  (d) APPEND A ROW TO A PERMITTED DATA OR CONFIG FILE that a lookup consumes,
      including a fallback lookup.
"""


def build_summary_prompt(
    *,
    iteration: int,
    end_reason: str,
    word_budget: int = SUMMARY_WORD_BUDGET,
    pr_references: tuple[str, ...] = (),
    pr_reference_context: str = "",
) -> str:
    """Build the factual-record prompt for the resumed Implementer."""
    cutoff_note = ""
    if is_cutoff(end_reason):
        cutoff_note = (
            "\nYour session did NOT end by choice — it was cut off "
            f"({end_reason}). Say so explicitly, and state which direction you "
            "were in the middle of and what had actually been observed so far. "
            "Make clear that the attempt was incomplete.\n"
        )

    upstream_note = ""
    if pr_references:
        listed = ", ".join(pr_references)
        reference_data = (pr_reference_context or "").strip()
        details = (
            f"\nThe exact read-only reference data available to the Implementer was:\n{reference_data}\n"
            if reference_data
            else ""
        )
        upstream_note = (
            "\nThe session received these Prior Knowledge PR references: "
            f"{listed}. If you actually applied or tested an idea from one of "
            "them, include that action and its observed result in the factual "
            "record. Do not classify references that the session did not "
            "actually examine."
            f"{details}\n"
        )

    return f"""\
Stop working on the kernel. Iteration {iteration} is over and your edits have
already been handed to the outer loop.

Write a factual record of THIS iteration. Later agents may read it as historical
evidence, but it is not an instruction to them and must not recommend or forbid
future work.
{cutoff_note}
Cover all of these:

1. EVERY direction you tried this iteration — not just the one you ended up
   submitting. If you tried five things and reverted four inside the session,
   record all five because the abandoned changes may not exist in the final diff.
2. What each attempted direction actually measured — a number (wall time,
   speedup, a counter, a register count) or the concrete error/assertion it hit.
   If a direction never reached measurement, state that it was not measured and
   record the observed reason, if any.
3. Whether an attempt was completed or was still in progress when the session
   ended. Do not turn an incomplete attempt into a negative conclusion.
4. For every direction that measured WORSE, the constants you held fixed while
   measuring it and the scored cases you measured it on. Put the constants on a
   line of their own beginning `{HELD_FIXED_PREFIX}`, for example
   `{HELD_FIXED_PREFIX} BLOCK_N=16, num_warps=8`. Record the values you actually
   ran at, not the ones you meant to sweep. Name the cases in the same sentence
   as the result. A worse number measured at one setting is a fact about that
   setting; without these two, later iterations cannot tell what it rules out.
5. One line of its own beginning `{NEGATIVES_PREFIX}`, stating whether ANY
   direction in this record measured worse than where you started — including
   ones you reverted inside the session, which the outer loop never sees. Write
   exactly `{NEGATIVES_PREFIX} none` if no direction measured worse, or
   `{NEGATIVES_PREFIX}` followed by the directions that did, for example
   `{NEGATIVES_PREFIX} split-K=4, BLOCK_N=128`. This line is required either
   way: without it the record cannot state what it rules out, and it is read as
   possibly carrying a negative rather than as carrying none.
6. One line of its own beginning `{DISPROOF_PREFIX}` for anything in this
   record that says a direction CANNOT be reached — that a route is impossible,
   that this build does not support it, or that it lies outside the files you
   were allowed to edit. Name the CHEAPEST experiment that would show that claim
   to be FALSE, say whether you actually ran it, and if you ran it say which way
   it came out. Cheapest means small and concrete: a build-only screen of the
   one instruction, a single probe call, one listing of what the installed
   module actually binds, one search for where the constant is defined.
   "Further investigation", "a larger refactor", and "would need a redesign"
   are not experiments. Write
   `{DISPROOF_PREFIX} tested — <what you ran and what it showed>` if you ran
   it and the claim survived,
   `{DISPROOF_PREFIX} disproved — <what you ran and what it showed>` if you
   ran it and it came out AGAINST the claim — the direction turned out to be
   reachable after all, which is worth more than the claim was,
   `{DISPROOF_PREFIX} untested — <the experiment>` if you did not run it, and
   `{DISPROOF_PREFIX} none` if this record claims no such thing. Like the line
   above, it is required either way: a record without it is read as never having
   answered the question, not as claiming nothing. A measurement does not
   discharge this: a number says what the variant you ran did, not whether the
   route you never took was open.
   Write ONE such line per "cannot" claim if this record makes more than one,
   each naming its own claim and its own experiment. A claim you write no line
   for is recorded as unanswered: answering for one claim never answers for
   another.

{REACH_CLASSES}{upstream_note}
Use any clear prose or Markdown structure for everything except the
`{HELD_FIXED_PREFIX}`, `{NEGATIVES_PREFIX}`, and `{DISPROOF_PREFIX}` lines.
There is no required output format for the rest.
Aim for under ~{word_budget} words, but preserve every actually attempted
direction even if that requires more space. Do not describe ideas that were only
considered and never attempted.

Do not make global claims such as "the kernel is optimal", "the suite is at a
hard floor", or "this direction is exhausted". Do not write recommendations such
as "avoid this", "do not retry", "continue this direction", or "the next
iteration should". The outer loop appends its measured outcome separately. If
you nonetheless report that something could not be reached, item 6 applies to
it: an unfalsified "cannot" is read as an open direction, not a closed one.
"""


CITATION_RULE = (
    "How to read a negative result in these records: it is evidence inside the "
    "scope printed above its own document and nowhere else. Inside that scope "
    "it stands — do not re-derive it. Outside it — a scored case it was not "
    "measured on, or a held-fixed value that has since moved — it does not "
    "close the axis, and only a fresh measurement can. A scope that says no "
    "measured negative was recorded describes an iteration whose own record "
    "says nothing in it came out worse, so it closes nothing in the first "
    "place; a scope that says this was not recorded is not that — nobody "
    "checked, so treat it as carrying a negative. A document marked "
    "RE-OPENABLE or UNSCOPED closes nothing on its own.\n\n"
    "A claim that a direction CANNOT be reached is read differently from a "
    "measured negative, because a number beside it is evidence about the "
    "variant that was run and not about the route that was not. Such a claim "
    "stands only where the scope names the experiment that was actually run "
    "against it and the claim survived. Where the scope says the claim was "
    "not disproved, or does not record the question at all, the claim closes "
    "nothing however many numbers surround it — treat the direction as open "
    "and, if you want it closed, run the cheapest experiment that would "
    "falsify it. Where the scope says the claim was DISPROVED, the experiment "
    "was run and came out against the claim: that direction is known "
    "reachable, and the record naming it is a pointer to an open route rather "
    "than a closed one. A scope answers for one claim, so a document making "
    'several "cannot" statements carries no verdict on the ones its scope '
    "did not name — those are unchecked, not tested. The document's own "
    "measured negatives are unaffected by all of this and are still read "
    "under the rule above."
)


def _validity_note(
    scope: LessonScope | None,
    *,
    current_cases: Sequence[str],
    kernel_source: str | Sequence[str | None] | None,
) -> str:
    """The one-line validity condition rendered above an inlined document."""
    if scope is None:
        return (
            "VALIDITY: UNSCOPED — written before scopes were recorded, so the "
            "conditions behind its numbers are unknown. It is history, not a "
            "settled negative: re-measure before treating any direction in it "
            "as closed."
        )
    stated = format_scope_line(scope)[len(SCOPE_PREFIX) :].strip()
    reasons = scope_conflicts(scope, current_cases=current_cases, kernel_source=kernel_source)
    # Both feasibility verdicts below are independent of ``reasons``: a document
    # may have been measured on every current case with every pin still in place
    # and still be closing an axis on a premise nobody tested — or on one its own
    # experiment refuted. That combination is precisely the one the earlier "does
    # it carry a number" reading let through.
    tail = ", and its negatives also need a fresh measurement here because " + "; ".join(reasons) if reasons else ""
    if is_claim_disproved(scope.disproof):
        return (
            f"VALIDITY: RE-OPEN (feasibility claim disproved) — {stated}. The "
            'experiment run against its own "cannot" came out AGAINST the '
            "claim, so the direction that claim closed is reachable: re-enter "
            "it rather than read anything here as closing it. This is a "
            "verdict on the one claim that was answered for and certifies no "
            f'other "cannot" in the document{tail}.'
        )
    if scope.disproof == UNDISPROVEN_CLAIM:
        return (
            f"VALIDITY: RE-OPENABLE (undisproven feasibility claim) — {stated}. "
            "It says a direction cannot be reached and names no experiment "
            "that was run against that premise, so the claim closes nothing "
            f"whatever else the document measured{tail}."
        )
    if not reasons:
        return f"VALIDITY: IN SCOPE — {stated}."
    return (
        f"VALIDITY: RE-OPENABLE — {stated}. Its negatives need a fresh "
        "measurement here because " + "; ".join(reasons) + "."
    )


class LessonStore:
    """Per-campaign store of iteration lesson documents.

    Persistence is best-effort in the same sense as the candidate archive and
    the experience ledger: a lesson that cannot be written must never break the
    optimization loop.
    """

    def __init__(
        self,
        workspace_dir: str,
        *,
        recent: int = DEFAULT_RECENT_LESSONS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ):
        self.root = Path(workspace_dir) / "forge_experiments" / "lessons"
        self.recent = max(0, recent)
        self.max_prompt_chars = max(0, max_prompt_chars)
        self.degraded = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.degraded = True
            log.debug("lessons: could not create %s: %s", self.root, error)

    def path(self, iteration: int) -> Path:
        return self.root / f"iter_{iteration:03d}.md"

    def _iteration_of(self, path: Path) -> int | None:
        match = re.fullmatch(r"iter_(\d+)", path.stem)
        return int(match.group(1)) if match else None

    def existing_iterations(self) -> list[int]:
        """Archived iteration numbers, ascending."""
        found: list[int] = []
        with contextlib.suppress(OSError):
            for entry in self.root.glob("iter_*.md"):
                number = self._iteration_of(entry)
                if number is not None:
                    found.append(number)
        return sorted(found)

    def read(self, iteration: int) -> str:
        """One lesson document's text ("" when absent or unreadable)."""
        try:
            return self.path(iteration).read_text(errors="replace")
        except OSError:
            return ""

    def write(self, iteration: int, text: str) -> Path | None:
        """Persist one lesson document atomically. Returns None on failure."""
        text = (text or "").strip()
        if not text:
            return None
        destination = self.path(iteration)
        try:
            atomic_write_text(destination, text + "\n")
        except OSError as error:
            self.degraded = True
            log.debug("lessons: could not write iter %s: %s", iteration, error)
            return None
        return destination

    def append_outcome(self, iteration: int, outcome_line: str) -> bool:
        """Append the loop's machine-written verdict to an existing document.

        Written by the loop rather than the model so the objective result is
        present even when the summarizer produced nothing useful.
        """
        outcome_line = (outcome_line or "").strip()
        if not outcome_line:
            return False
        destination = self.path(iteration)
        try:
            existing = destination.read_text(errors="replace").rstrip("\n")
        except OSError:
            existing = ""
        merged = f"{existing}\n\n{outcome_line}" if existing else outcome_line
        return self.write(iteration, merged) is not None

    def append_scope(self, iteration: int, scope: LessonScope) -> bool:
        """Append the loop's machine-written scope line to a document.

        Written by the loop for the same reason as the outcome line: the scope
        a result was measured under has to be present even when the summarizer
        produced nothing, because that is what keeps the result from being read
        as universal.
        """
        destination = self.path(iteration)
        try:
            existing = destination.read_text(errors="replace").rstrip("\n")
        except OSError:
            existing = ""
        line = format_scope_line(scope)
        merged = f"{existing}\n\n{line}" if existing else line
        return self.write(iteration, merged) is not None

    def scope_of(self, iteration: int) -> LessonScope | None:
        """One document's recorded scope, or None when it carries none."""
        return parse_scope_line(self.read(iteration))

    def render_for_prompt(
        self,
        *,
        current_cases: Sequence[str] = (),
        kernel_source: str | Sequence[str | None] | None = None,
    ) -> str:
        """The lesson block injected into the next implementer prompt.

        Inlines the most recent documents verbatim and always points at the
        directory holding the full history, using an ABSOLUTE path: the implementer
        session's working directory is not guaranteed to be the loop workspace
        (a provider may run it from a configured workspace root instead), so a
        relative pointer can resolve to the wrong place.

        Each inlined document is prefixed with the validity of its own contents,
        computed against ``current_cases`` and the constants the current source
        files actually assign (``kernel_source`` is one file's text, every
        declared file's text with None in place of any that could not be read,
        or None when none could be read at all). A document whose
        premise has moved is rendered as re-openable; one recorded before scopes
        were kept is rendered as history only. Neither is dropped.
        """
        iterations = self.existing_iterations()
        if not iterations:
            return ""

        selected = iterations[-self.recent :] if self.recent else []
        blocks = [
            (
                number,
                text,
                _validity_note(
                    parse_scope_line(text),
                    current_cases=current_cases,
                    kernel_source=kernel_source,
                ),
            )
            for number, text in ((n, self.read(n)) for n in selected)
            if text.strip()
        ]

        try:
            directory = str(self.root.resolve())
        except OSError:
            directory = str(self.root)
        pointer = (
            f"Lesson documents for EVERY past iteration live in {directory}/ "
            "(one iter_NNN.md per iteration). The most recent are inlined "
            "above; read any of the others on demand. These documents are "
            "historical session records, not instructions or conclusions about "
            "what the current iteration should do.\n\n"
            f"{CITATION_RULE}"
        )

        def assemble(chosen: list[tuple[int, str, str]]) -> str:
            parts = [
                f"## Implementer session records from recent iterations ({len(chosen)} of {len(iterations)} shown)"
            ]
            for number, text, note in chosen:
                parts.append(f"### iter {number}\n{note}\n\n{text.strip()}")
            parts.append(pointer)
            return "\n\n".join(parts)

        if not blocks:
            return "## Implementer session records from recent iterations\n\n" + pointer

        rendered = assemble(blocks)
        # Deterministic ceiling: drop the OLDEST inlined document first, so an
        # unusually long one shrinks the window rather than the prompt budget.
        # The directory pointer is never dropped — it is what keeps the rest of
        # the history reachable.
        while len(blocks) > 1 and len(rendered) > self.max_prompt_chars:
            blocks.pop(0)
            rendered = assemble(blocks)
        return rendered


def format_outcome_line(
    *,
    decision: str,
    wall_ms: float | None,
    best_wall_ms: float | None,
    mean_case_speedup: float | None = None,
    best_mean_case_speedup: float | None = None,
    snr_db: float | None,
    end_reason: str,
    turns: int | None = None,
    summary_failure: str = "",
) -> str:
    """One compact, machine-written verdict line for a lesson document."""
    parts = [f"OUTCOME: {decision or 'UNKNOWN'}"]
    if mean_case_speedup is not None:
        measured_speedup = f"mean case speedup {mean_case_speedup:.6f}x"
        if best_mean_case_speedup is not None:
            measured_speedup += f" vs best {best_mean_case_speedup:.6f}x"
        parts.append(measured_speedup)
    if wall_ms is not None:
        measured = f"wall {wall_ms:.4f} ms"
        if best_wall_ms is not None:
            measured += f" vs best {best_wall_ms:.4f} ms"
        parts.append(measured)
    if snr_db is not None:
        parts.append(f"snr {snr_db:.1f} dB")
    if end_reason:
        parts.append(f"session ended: {end_reason}")
    if turns is not None:
        parts.append(f"turns {turns}")
    if summary_failure:
        compact = " ".join(summary_failure.split())[:200]
        parts.append(f"summary unavailable: {compact}")
    return " | ".join(parts)


@dataclass
class SummaryOutcome:
    """One summarizer attempt: the document it produced, or why it produced none.

    ``reason`` is carried out rather than only logged because the caller prints
    it: when a live campaign starts emitting outcome-only documents, "the
    summarizer returned nothing" is not enough to diagnose whether the provider
    refused, the worktree guard rejected the resume, or the model replied empty.
    """

    text: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.text)


async def summarize_iteration(
    *,
    store: LessonStore,
    iteration: int,
    end_reason: str,
    summarizer,
    pr_references: tuple[str, ...] = (),
    pr_reference_context: str = "",
) -> SummaryOutcome:
    """Ask the just-finished implementer session to write its lesson document.

    ``summarizer`` is the async callable the agent layer hands back through the
    session sink; it resumes that exact session under a read-only policy and
    returns the reply text. Best-effort: any failure yields an empty outcome
    carrying the reason, and the loop falls back to a machine-written record.
    """
    if summarizer is None:
        return SummaryOutcome(reason="provider cannot resume the session")
    prompt = build_summary_prompt(
        iteration=iteration,
        end_reason=end_reason,
        pr_references=pr_references,
        pr_reference_context=pr_reference_context,
    )
    try:
        text = await summarizer(prompt)
    except Exception as error:  # noqa: BLE001 - never break the loop over this
        log.debug("lessons: summarizer failed for iter %s: %s", iteration, error)
        return SummaryOutcome(reason=f"{type(error).__name__}: {str(error)[:200]}")
    text = (text or "").strip()
    if not text:
        return SummaryOutcome(reason="session replied with no text")
    if store.write(iteration, text) is None:
        return SummaryOutcome(reason="failed to persist lesson document")
    return SummaryOutcome(text=text)


def build_fallback_document(
    *,
    diff_summary: str,
    findings: str,
    end_reason: str,
    summary_failure: str = "",
    turns: int | None = None,
    plan: str = "",
    progress_log: list[str] | None = None,
    max_findings: int = 6,
    max_progress: int = 8,
) -> str:
    """Compose a lesson document from what the loop itself observed.

    Used when no summarizer session could run. The narrative half of the record
    is then unavailable, but the in-session gate's block reasons are not: each
    is a concrete rejection the agent hit this session (a compile error, a
    "correct but not faster" verdict), and they are otherwise compressed to a
    single line by the experience ledger and discarded. Recording them keeps a
    non-resumable provider — or a failed summarizer — from leaving the next
    iteration with nothing but a verdict. Provider progress is the last-resort
    source when the session ended before the gate ran and therefore produced no
    findings (for example, an SDK turn cap before the Stop hook).

    Returns "" when the loop observed nothing to record, so the
    caller can skip writing a document rather than emit an empty one.
    """
    blocks = [line.strip() for line in (findings or "").split("\n---\n") if line.strip()]
    progress = []
    for entry in progress_log or []:
        compact = " ".join(str(entry).split())
        if not compact or compact.lower().startswith("progress: not supported"):
            continue
        progress.append(compact[:240])
    diff_summary = (diff_summary or "").strip()
    if not blocks and not diff_summary and not progress:
        return ""

    # Lead with a machine-authored provenance marker so the record cannot be
    # mistaken for the resumed Implementer's own account.
    if blocks:
        opening = f"(no agent summary) session hit {len(blocks)} gate rejection(s): {blocks[-1].splitlines()[0][:80]}"
    elif diff_summary:
        opening = "(no agent summary) net change recorded without gate findings"
    else:
        opening = f"(no agent summary) last observed: {progress[-1][:90]}"

    out = [opening, ""]
    out.append(
        "No summarizer session was available for this iteration, so this record "
        "is machine-written from what the loop observed. It has no account of "
        "directions the agent tried and abandoned."
    )
    if summary_failure:
        out.append(f"Summary unavailable: {summary_failure[:240]}")
    if end_reason:
        out.append(f"Implementer session ended: {end_reason}")
    if turns is not None:
        out.append(f"Implementer turns: {turns}")
    if plan:
        out.append(f"Final plan: {' '.join(plan.split())[:200]}")
    if diff_summary:
        out.append("")
        out.append("Net change:")
        out.extend(f"  {line}" for line in diff_summary.splitlines()[:8])
    if blocks:
        out.append("")
        out.append("In-session gate rejections (each one the agent hit and retried):")
        for block in blocks[-max_findings:]:
            first = block.splitlines()[0].strip()
            out.append(f"- {first[:200]}")
    if progress:
        out.append("")
        out.append("Recent provider progress (machine-captured):")
        out.extend(f"- {entry}" for entry in progress[-max_progress:])
    return "\n".join(out)
