"""Read-only review of one synthesized forge-loop optimization plan."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kernelforge.agent_backends import (
    AgentBackend,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.orchestrator.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_text,
)
from kernelforge.orchestrator.contracts import (
    DispatchPlan,
    LaneDrop,
    OrchestrationContext,
    PlanCriticOutcome,
    SpecialistOutcome,
    SynthesizedPlan,
)
from kernelforge.orchestrator.structured_output import (
    build_repair_prompt,
    extract_json_object,
)


PLAN_CRITIC_MAX_TURNS = 100
PLAN_CRITIC_TIMEOUT_SEC = 600
# One repair pass for a width block the review did not deliver. Given no tools
# and two turns because nothing is being judged: the review already happened,
# and this call only restates its width decision in the shape the round reads.
WIDTH_REPAIR_MAX_TURNS = 2
WIDTH_REPAIR_TIMEOUT_SEC = 120
WIDTH_REPAIR_EFFORT = "low"
_ERROR_DETAIL_MAX_CHARS = 2000
_NARROWING_NOTE_MAX_CHARS = 240

log = logging.getLogger(__name__)

# The verdict stays a regex. It is one token from a closed three-word
# vocabulary, so the pattern is the whole grammar and there is nothing a
# structured block would add. The width ruling is a set of (lane, reason) pairs,
# which is why it is read as JSON below rather than out of the prose.
_VERDICT_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?"
    r"VERDICT(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?"
    r"(ACCEPT|REVISE|REPLACE)\b"
)

# The one key the review's trailing width block is required to carry, and the
# schema shown to the review and to the repair pass.
_WIDTH_BLOCK_KEY = "lane_narrowing"
_WIDTH_BLOCK_SCHEMA: dict[str, Any] = {
    _WIDTH_BLOCK_KEY: [
        {
            "lane_id": "The integer lane_id from draft_lane_plans",
            "reason": ("Why this lane is not worth an Implementer session of its own"),
        }
    ]
}
_WIDTH_BLOCK_LABEL = "plan critic width block"
_WIDTH_BLOCK_ABSENT = f"the review ended with no {_WIDTH_BLOCK_KEY} block"
# The two ways a round ends up not knowing what width the review wanted. Both
# survive the repair pass only when it, too, came back with nothing readable.
_UNREAD_WIDTH_STATUSES = frozenset({"absent", "malformed"})

# There is deliberately no prose fallback beside this parser. A `DROP LANE <n>`
# regex kept alongside a required block would give a review two ways to answer
# one question and the round no rule for ranking them when they disagree, and
# its own reach ended at whatever literal it was written around -- which is the
# defect this replaces. The failure mode that is therefore NOT protected
# against: a review that states a drop only in prose and whose one repair pass
# also fails to restate it as a block. That round runs every lane it planned,
# and says so under `lane_narrowing` with the note naming what was unread. It is
# never silent, and it is never narrowed on a reading nobody validated.


def _bounded_error_detail(error: Exception) -> str:
    """Return one bounded log-safe line for persisted critic failures."""
    message = " ".join(str(error).split())
    detail = f"{type(error).__name__}: {message}".rstrip()
    if len(detail) <= _ERROR_DETAIL_MAX_CHARS:
        return detail
    return detail[: _ERROR_DETAIL_MAX_CHARS - 3].rstrip() + "..."


def _bounded_note(note: str) -> str:
    """Return one bounded single line; every note here is persisted."""
    line = " ".join(str(note).split())
    if len(line) <= _NARROWING_NOTE_MAX_CHARS:
        return line
    return line[: _NARROWING_NOTE_MAX_CHARS - 3].rstrip() + "..."


def _entry_note(problem: str, entry: object) -> str:
    """Name one entry of a readable width block that could not be used.

    The note stops at what was found in the entry. What the round then does
    about it is ``status`` and the drops, and a note that also stated an
    outcome could contradict them: the repeated entry below names a lane the
    entry before it has already dropped.
    """
    try:
        rendered = json.dumps(entry, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - json.loads output only
        rendered = repr(entry)
    return _bounded_note(f"{problem}: {rendered}")


@dataclass(frozen=True)
class LaneNarrowingRuling:
    """What one review's trailing width block asked for, and how it was read.

    Four answers reach the round, and ``drops`` alone tells three of them apart
    from none. ``status`` is what separates them:

    - ``answered`` with no drops and no notes -- the block was read and it named
      no lane. This is the only one of the four that means "run every lane".
    - ``answered`` with notes -- the block was read and something it asked for
      was not a usable decision: a lane it did not number, a drop it gave no
      reason for, one lane named twice.
    - ``absent`` -- the review ended without a block at all.
    - ``malformed`` -- a block was there and could not be decoded, or its
      ``lane_narrowing`` was not a list, so what it wanted is unknown.

    The last two are what the review still owes an answer for, and the two a
    repair pass can settle. A block that was read and got a lane or a reason
    wrong is not repaired: correcting it would mean inventing the decision.

    ``notes`` says what was seen while reading the block and nothing more.
    ``unread`` is the separate question of whether the reading lost a width
    decision the review asked for, which is what a log line's severity has to
    follow: a note by itself is not a failure, and a lane named twice is a note
    with nothing lost, because the first entry dropped that lane.
    """

    drops: tuple[LaneDrop, ...] = ()
    notes: tuple[str, ...] = ()
    status: str = "absent"
    unread: bool = False

    @property
    def answered(self) -> bool:
        """Whether a block was read, whatever it went on to ask for."""
        return self.status == "answered"


_PLAN_CRITIC_SYSTEM_PROMPT = """\
You are the read-only critic for one GPU-kernel optimization plan. Review the
draft independently. Do not edit files, run shell commands, benchmark, profile,
or rewrite the plan yourself. You may read and search the supplied workspace
evidence paths when needed.

Use this checklist to guide judgment, but do not mechanically repeat every item:
- Question whether the current kernel, algorithm, programming model, and
  execution units should continue to exist.
- Verify that the bottleneck claim is supported by profiling and a plausible
  performance model.
- Compare the current route's performance ceiling with materially different
  alternatives.
- Search supplied source, dependency, and knowledge paths for existing GEMM,
  MFMA, fused-kernel, library, or alternate-backend implementations before
  recommending more work on the current implementation.
- Check for omitted structural options in fusion, algorithms, dataflow, layout,
  and hardware instructions.
- Require a clear causal link between every proposed change and the measured
  bottleneck.
- Detect whether repeated local gains have kept the search at one optimization
  level for too long.
- When useful, request one isolated challenger for a high-potential alternative.
  Exploration may regress temporarily, but the final candidate must still beat
  the canonical best under the unchanged correctness and KEEP gates.
- Check instruction, register, memory, occupancy, compiler, and implementation
  feasibility.
- Require explicit success, failure, stop, and route-switch conditions.
- Check correctness, boundary inputs, numerical accuracy, and representative
  workload coverage.
- Compare prior attempts and reject repetition without new evidence.
- Judge expected gain, implementation time, and opportunity cost for one
  iteration.

Cite concrete source, profiling, benchmark, candidate-history, or knowledge
paths for important claims. State uncertainty when evidence is missing. Focus
on issues that would change this iteration's plan rather than generic advice.

Include exactly one routing line somewhere in otherwise free-form Markdown:
VERDICT: ACCEPT
VERDICT: REVISE
VERDICT: REPLACE

ACCEPT means the draft is worth executing. REVISE means the same broad route
needs evidence, scope, sequencing, or risk corrections. REPLACE means the draft
continues a strategically dominated implementation route and should instead
validate a concrete alternative.
"""

_PLAN_CRITIC_ROUND_BLOCK = """\

This round was divided into several lanes, each planned on its own ground and
implemented concurrently by its own Implementer. Review the division as well as
the plans, and answer for the round as a whole:

- Is any lane's ground not worth an Implementer session of its own? A lane the
  round cannot use still costs a full session.
- Do two lanes amount to the same change described differently? Their scores
  would then be one answer bought twice, and neither could be stacked on the
  other.
- Would any lane have to edit code another lane owns to carry out its plan?
- Taken together, is the round still working at one optimization level that has
  stopped paying, when the evidence supports a materially different route?

One verdict covers the round and applies to every lane that runs. REVISE and
REPLACE are for what the round should do differently, not for a wording
preference in one plan.

How wide the round runs is a separate answer, given per lane, and it is the one
part of this review a machine reads rather than a person. End the review --
after all of your prose, as its last content -- with exactly one JSON object
naming every lane you judge not worth an Implementer session of its own: ground
the evidence does not support, or another lane's change in different words.

```json
{"lane_narrowing": [{"lane_id": 2, "reason": "the epilogue rewrite is lane 1's change in different words"}]}
```

A lane whose `joint` is true was given wider ground than a region on purpose:
it holds a change and the launch configuration that change invalidates, whose
parts cannot be measured apart, and its `fallback` is the smaller change inside
that same ground its Implementer lands if the joint step does not converge.
That width is already spent by the time you read this. Naming such a lane drops
it, exactly as for any other lane, and the round then measures nothing on the
ground it widened -- so weigh the `fallback` as what the lane still returns, and
name the lane only if even that is not worth its session.

`lane_narrowing` is a list. Each entry has `lane_id`, the integer lane_id from
draft_lane_plans, and `reason`, one non-empty sentence saying what you found.
Every lane you do not name is run, so a round you want whole still ends with the
block, empty:

```json
{"lane_narrowing": []}
```

Emit the block either way. An empty list and a missing block are different
answers, and only the first one means "run every lane". The reason is recorded
with the round, so a drop whose reason cannot be read keeps its lane.
At least one lane always runs, so drop only what the round is better off
without.
"""


_WIDTH_REPAIR_SYSTEM_PROMPT = """\
You are reformatting one machine-read block that a completed plan review was
required to end with and did not. Return exactly one JSON object and no other
text.

Carry over only the width decision the review already made in its own words: a
lane it said outright was not worth an Implementer session of its own. Use the
lane numbers and the reasons the review itself gave. If the review named no such
lane, return the empty list -- that is a complete answer, and it is the right
one whenever you would otherwise be guessing.

Do not read files, do not re-review the plans, and do not add a lane or a reason
the review did not state.
"""


def parse_plan_critic_verdict(text: str) -> str:
    """Parse the first explicit verdict; non-empty unmarked reviews revise."""
    review = str(text or "").strip()
    if not review:
        raise ValueError("plan critic returned no review")
    match = _VERDICT_PATTERN.search(review)
    return match.group(1).upper() if match else "REVISE"


def _lane_id_of(raw: object) -> int | None:
    """Return the positive lane a width-block entry names, or None.

    A quoted integer is read as the integer it spells. Nothing is invented by
    doing so -- "2" names lane 2 and no other -- and no schema shown to a model
    can stop one quoting its numbers.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        lane_id = raw
    elif isinstance(raw, str) and raw.strip().isdigit():
        lane_id = int(raw.strip())
    else:
        return None
    return lane_id if lane_id >= 1 else None


def parse_plan_critic_width_block(text: str) -> LaneNarrowingRuling:
    """Read the width ruling the review was required to end with.

    The review's product is prose -- a person reads it and the revision is fed
    it -- so only its width decision is structured, in one trailing JSON object.
    Whether the drops are obeyed is not decided here: the round's width belongs
    to whoever holds the lanes. What is decided here is that no answer leaves as
    nothing, because a round that quietly kept every lane would look exactly
    like a round the review wanted whole.

    The search runs from the end, anchored on the block's own key, because the
    block is asked for last and the prose before it is free to quote JSON --
    autotune configs and `structured_output.json` fragments are ordinary things
    for a kernel review to cite. Taking the first object in the response would
    hand the round a tuning dict and call the real ruling missing.
    """
    review = str(text or "")
    marker = review.rfind(f'"{_WIDTH_BLOCK_KEY}"')
    if marker < 0:
        return LaneNarrowingRuling(
            notes=(_bounded_note(_WIDTH_BLOCK_ABSENT),),
            status="absent",
            unread=True,
        )
    start = review.rfind("{", 0, marker)
    if start < 0:
        return LaneNarrowingRuling(
            notes=(_bounded_note(f"the review named {_WIDTH_BLOCK_KEY} outside any JSON object"),),
            status="malformed",
            unread=True,
        )
    try:
        payload = extract_json_object(review[start:], _WIDTH_BLOCK_LABEL)
    except ValueError as error:
        return LaneNarrowingRuling(
            notes=(_bounded_note(str(error)),),
            status="malformed",
            unread=True,
        )
    entries = payload.get(_WIDTH_BLOCK_KEY)
    if not isinstance(entries, list):
        return LaneNarrowingRuling(
            notes=(_bounded_note(f"{_WIDTH_BLOCK_KEY} was not a list"),),
            status="malformed",
            unread=True,
        )
    drops, notes, unread = _read_lane_drops(entries)
    return LaneNarrowingRuling(
        drops=drops,
        notes=notes,
        status="answered",
        unread=unread,
    )


def _read_lane_drops(
    entries: Sequence[object],
) -> tuple[tuple[LaneDrop, ...], tuple[str, ...], bool]:
    """Turn one readable width block's entries into drops, naming the rest.

    The third return value is whether any of those entries cost the round a
    decision. A lane named twice does not: the entry before it dropped that
    lane, so the second is worth recording and is nothing to raise.
    """
    drops: list[LaneDrop] = []
    notes: list[str] = []
    seen: set[int] = set()
    unread = False
    for entry in entries:
        if not isinstance(entry, dict):
            notes.append(_entry_note("unreadable lane drop", entry))
            unread = True
            continue
        lane_id = _lane_id_of(entry.get("lane_id"))
        reason = " ".join(str(entry.get("reason") or "").split())
        if lane_id is None:
            notes.append(_entry_note("lane drop names no lane", entry))
            unread = True
            continue
        if not reason:
            notes.append(_entry_note("lane drop states no reason", entry))
            unread = True
            continue
        if lane_id in seen:
            notes.append(_entry_note("lane drop repeats a lane", entry))
            continue
        seen.add(lane_id)
        drops.append(LaneDrop(lane_id=lane_id, reason=reason))
    return tuple(drops), tuple(notes), unread


def build_plan_critic_prompts(
    *,
    context: OrchestrationContext,
    drafts: Sequence[SynthesizedPlan],
    dispatch_plan: DispatchPlan,
    specialist_outcomes: Sequence[SpecialistOutcome],
    coverage: Mapping[str, object],
) -> tuple[str, str]:
    """Build one bounded critic request from persisted planning evidence.

    A round of several lanes is reviewed once, together: the lanes are one
    division of one round, so what is worth asking about them -- whether the
    division is right, whether two lanes are the same change twice, whether the
    round as a whole has stopped moving -- cannot be asked of any lane alone.
    One plan is reviewed exactly as it always was; there is no division to
    review and no sibling to compare against.
    """
    if not drafts:
        raise ValueError("plan critic needs at least one draft to review")
    payload = {
        "task": (
            "Audit the draft plan before implementation. Decide whether to "
            "accept it, revise it, or replace its implementation route."
        ),
        "context": context.to_prompt_dict(),
        "dispatch_plan": dispatch_plan.to_dict(),
        "specialist_coverage": dict(coverage),
        "specialist_outcomes": [outcome.to_dict() for outcome in specialist_outcomes],
    }
    if len(drafts) == 1:
        payload["draft_plan"] = drafts[0].text
        return (
            _PLAN_CRITIC_SYSTEM_PROMPT,
            json.dumps(payload, indent=2, sort_keys=True),
        )
    payload["task"] = (
        "Audit this round's lane plans and the division that produced them, "
        "before any of them is implemented. Decide whether to accept the "
        "round, revise it, or replace its implementation route."
    )
    payload["draft_lane_plans"] = [
        {
            "lane_id": index + 1,
            "ground": draft.ground,
            "joint": draft.joint,
            "fallback": draft.fallback,
            "draft_plan": draft.text,
        }
        for index, draft in enumerate(drafts)
    ]
    return (
        _PLAN_CRITIC_SYSTEM_PROMPT + _PLAN_CRITIC_ROUND_BLOCK,
        json.dumps(payload, indent=2, sort_keys=True),
    )


class PlanCriticAgent:
    """Run one fail-open, read-only plan review in an independent session."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        timeout_sec: int,
        max_turns: int = PLAN_CRITIC_MAX_TURNS,
        ceiling_sec: int | None = None,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.backend = backend
        # Budget for one plan. A round of several is several times the reading,
        # so the budget is spent per plan and capped by what the provider allows
        # a single call. Left equal to ``timeout_sec`` when no ceiling is given,
        # which keeps a one-plan review exactly as it was.
        self.timeout_sec = timeout_sec
        self.ceiling_sec = max(timeout_sec, int(ceiling_sec or timeout_sec))
        self.max_turns = max_turns

    def _budget_for(self, drafts: int) -> int:
        """The wall-clock a review of this many plans is allowed.

        Measured on one real two-lane round: the review took about eleven
        minutes against a ten-minute budget sized for one plan, so it failed
        open to ACCEPT and the round lost a verdict that had found a lane not
        worth its session.
        """
        return min(self.timeout_sec * max(1, drafts), self.ceiling_sec)

    async def review(
        self,
        *,
        context: OrchestrationContext,
        drafts: Sequence[SynthesizedPlan],
        dispatch_plan: DispatchPlan,
        specialist_outcomes: Sequence[SpecialistOutcome],
        coverage: Mapping[str, object],
        usage=None,
    ) -> PlanCriticOutcome:
        """Review one round; every backend/output failure accepts fail-open.

        One call whatever the round's width. Reviewing each lane on its own
        would multiply the cost by the width and still leave the one question a
        round raises -- whether the division is right -- asked of nobody.
        """
        system_prompt, user_prompt = build_plan_critic_prompts(
            context=context,
            drafts=drafts,
            dispatch_plan=dispatch_plan,
            specialist_outcomes=specialist_outcomes,
            coverage=coverage,
        )
        budget_sec = self._budget_for(len(drafts))
        started_at = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.backend.run(
                    AgentRunSpec(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        cwd=context.workspace,
                        writable=False,
                        timeout_sec=budget_sec,
                        reasoning_effort="max",
                        tool_policy=AgentToolPolicy(
                            read=True,
                            search=True,
                            write=False,
                            shell=False,
                            max_turns=self.max_turns,
                        ),
                        protected_globs=["*"],
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(budget_sec),
            )
        except Exception as error:  # noqa: BLE001 - provider boundary
            return self._fail_open(error, started_at=started_at)

        try:
            review = validated_agent_text(result, role="plan critic")
            verdict_explicit = _VERDICT_PATTERN.search(review) is not None
            verdict = parse_plan_critic_verdict(review)
        except (
            AgentResponseInfrastructureError,
            AgentResponseIncompleteError,
            ValueError,
        ) as error:
            return self._fail_open(error, started_at=started_at)

        ruling = await self._width_ruling(
            context=context,
            review=review,
            drafts=len(drafts),
            usage=usage,
        )
        duration_sec = time.monotonic() - started_at
        verdict_source = "explicit" if verdict_explicit else "inferred"
        if not verdict_explicit:
            log.warning("plan critic omitted an explicit VERDICT; inferring REVISE")
        self._log_width_ruling(ruling)
        log.info(
            "plan critic completed verdict=%s source=%s width=%s lane_drops=%d duration=%.3fs",
            verdict,
            verdict_source,
            ruling.status,
            len(ruling.drops),
            duration_sec,
        )
        return PlanCriticOutcome(
            verdict=verdict,
            review=review,
            duration_sec=duration_sec,
            verdict_source=verdict_source,
            lane_drops=ruling.drops,
            narrowing_notes=ruling.notes,
            narrowing_status=ruling.status,
        )

    async def _width_ruling(
        self,
        *,
        context: OrchestrationContext,
        review: str,
        drafts: int,
        usage=None,
    ) -> LaneNarrowingRuling:
        """Read the review's width block, repairing it once if it cannot be.

        A one-plan round is never asked for a block -- there is no division to
        rule on and no second lane to drop -- so it is not held to one, is not
        reported for omitting one, and never pays for a repair. A block it
        volunteers anyway is still read, because the floor that refuses it lives
        downstream in the round and is worth reaching rather than leaving as
        dead code.
        """
        if drafts <= 1:
            volunteered = parse_plan_critic_width_block(review)
            return volunteered if volunteered.answered else LaneNarrowingRuling(status="not_asked")
        ruling = parse_plan_critic_width_block(review)
        if ruling.answered:
            return ruling
        return await self._repaired_width_ruling(
            context=context,
            review=review,
            ruling=ruling,
            usage=usage,
        )

    async def _repaired_width_ruling(
        self,
        *,
        context: OrchestrationContext,
        review: str,
        ruling: LaneNarrowingRuling,
        usage=None,
    ) -> LaneNarrowingRuling:
        """Spend one call to recover a width ruling the review did not format.

        This is the round's only conditional call, and what it buys is an
        Implementer session: a drop the round cannot read is a lane it runs, and
        a lane costs a full session against a planning window already measured
        at 21.6 minutes a round. The repair is given no tools, two turns and two
        minutes, so at worst it costs a small fraction of the review that
        preceded it, and it is reached only when the block was absent or
        unreadable -- a review that answered and named a lane the round does not
        have has been read, and repairing a decision is how a parser starts
        inventing one.

        It cannot fail the round. A repair that errors, times out or comes back
        without a block leaves the original ruling standing, plus one note
        saying the pass was spent and what it did not recover.
        """
        detail = ruling.notes[0] if ruling.notes else _WIDTH_BLOCK_ABSENT
        try:
            result = await asyncio.wait_for(
                self.backend.run(
                    AgentRunSpec(
                        system_prompt=_WIDTH_REPAIR_SYSTEM_PROMPT,
                        user_prompt=build_repair_prompt(
                            label=_WIDTH_BLOCK_LABEL,
                            original_response=review,
                            validation_error=detail,
                            output_schema=_WIDTH_BLOCK_SCHEMA,
                        ),
                        cwd=context.workspace,
                        writable=False,
                        timeout_sec=self._repair_budget(),
                        reasoning_effort=WIDTH_REPAIR_EFFORT,
                        tool_policy=AgentToolPolicy(
                            read=False,
                            search=False,
                            write=False,
                            shell=False,
                            max_turns=WIDTH_REPAIR_MAX_TURNS,
                        ),
                        protected_globs=["*"],
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(self._repair_budget()),
            )
            repaired = validated_agent_text(
                result,
                role=_WIDTH_BLOCK_LABEL,
                allow_incomplete=True,
            )
        except Exception as error:  # noqa: BLE001 - provider boundary
            return self._unrepaired(
                ruling,
                f"one repair pass for the width block failed: {_bounded_error_detail(error)}",
            )
        recovered = parse_plan_critic_width_block(repaired)
        if not recovered.answered:
            return self._unrepaired(
                ruling,
                "one repair pass returned no readable width block either",
            )
        log.info(
            "plan critic width block was recovered by one repair pass (%s); it asks to drop %d lane(s)",
            ruling.status,
            len(recovered.drops),
        )
        return LaneNarrowingRuling(
            drops=recovered.drops,
            notes=(
                *ruling.notes,
                *recovered.notes,
                _bounded_note("the review did not end with a readable width block; one repair pass restated it"),
            ),
            status="repaired",
            # What the first reading could not find has been found. Only what
            # the restated block itself asked for and did not say is still lost.
            unread=recovered.unread,
        )

    def _repair_budget(self) -> int:
        """The wall-clock one repair pass is allowed, never above the review's."""
        return max(1, min(WIDTH_REPAIR_TIMEOUT_SEC, self.timeout_sec))

    @staticmethod
    def _unrepaired(
        ruling: LaneNarrowingRuling,
        note: str,
    ) -> LaneNarrowingRuling:
        """Keep the unreadable ruling, naming the repair pass that was spent.

        Nothing is logged here. The ruling is reported once, by the review that
        owns it, at the severity its outcome earns -- and this one earns the
        warning, because the width decision is now known to be unrecoverable.
        """
        return LaneNarrowingRuling(
            drops=ruling.drops,
            notes=(*ruling.notes, _bounded_note(note)),
            status=ruling.status,
            unread=True,
        )

    @staticmethod
    def _log_width_ruling(ruling: LaneNarrowingRuling) -> None:
        """Report how the width block read, at the severity that reading earns.

        A note says what was seen; ``status`` and the drops say what the ruling
        came to, and what the round then does with it is the round's own line.
        Logging every note as "narrowing was not applied" made the recovered
        path -- block absent, one repair pass restated it, a lane dropped --
        warn twice that nothing had been narrowed, immediately above the line
        saying the round had narrowed. An operator who sees a warning
        contradicted a few times stops reading it, which costs more than the
        wrong line does.
        """
        if not ruling.notes:
            return
        detail = "; ".join(ruling.notes)
        if ruling.status in _UNREAD_WIDTH_STATUSES:
            log.warning(
                "plan critic width ruling was never read (%s), so no lane can be dropped on it: %s",
                ruling.status,
                detail,
            )
        elif ruling.unread:
            log.warning(
                "plan critic width block was read and part of what it asked for was not: %s",
                detail,
            )
        else:
            log.info(
                "plan critic width block was read (%s): %s",
                ruling.status,
                detail,
            )

    @staticmethod
    def _fail_open(
        error: Exception,
        *,
        started_at: float,
    ) -> PlanCriticOutcome:
        detail = _bounded_error_detail(error)
        duration_sec = time.monotonic() - started_at
        log.warning(
            "plan critic failed open to the draft after %.3fs: %s",
            duration_sec,
            detail,
        )
        return PlanCriticOutcome(
            verdict="ACCEPT",
            error=detail,
            duration_sec=duration_sec,
            verdict_source="error",
        )
