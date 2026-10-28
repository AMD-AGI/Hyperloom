"""What the lane syntheses of one orchestration round are sent, and the order they start in.

Every lane of a round receives the same evidence and differs only in its own ``lane`` block. That block goes last,
so everything before it is one prefix the lanes share verbatim; the lanes after the first start once its response
begins, and with a backend that caches the prefix they read it instead of writing it again.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence

from kernelforge.orchestrator.contracts import LaneGround, OrchestrationContext, SpecialistRound

LANE_KEY = "lane"
# How long the siblings wait for the first lane to start answering before they start anyway and write the prefix
# themselves. With summarized thinking the first streamed event arrives within about 30 s; a lead still silent at
# this point is retrying or stuck, and holding the round for its whole timeout would serialize the lanes.
LEAD_RESPONSE_WAIT_SEC = 120.0

_LANE_TASK = (
    "Produce the optimization plan this lane's Implementer "
    "should execute on the ground this lane owns. Judge the "
    "analyses rather than copying them, and choose a clear "
    "implementation sequence."
)


def lane_prompts(
    context: OrchestrationContext,
    specialists: SpecialistRound,
    grounds: Sequence[LaneGround],
) -> list[tuple[str, str]]:
    """Each lane's prompt and the leading part of it that every lane of the round shares."""
    evidence = {
        "task": _LANE_TASK,
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_analyses": [
            {"assignment_id": outcome.assignment_id, "role_id": outcome.role_id, "analysis": outcome.content}
            for outcome in specialists.analyses
        ],
    }
    prompts = []
    for index, own in enumerate(grounds):
        lane = {
            "ground": own.ground,
            "joint": own.joint,
            "fallback": own.fallback,
            "ground_owned_by_other_lanes": [
                other.ground for position, other in enumerate(grounds) if position != index
            ],
        }
        prompts.append(lane_last_prompt({**evidence, LANE_KEY: lane}))
    return prompts


def lane_last_prompt(payload: Mapping[str, object]) -> tuple[str, str]:
    """``payload`` as JSON with its ``lane`` block after every other key, and the part before that block.

    Nested keys stay sorted, as a sorted dump would leave them; only the lane block moves.
    """
    canonical = json.loads(json.dumps(payload, sort_keys=True))
    ordered = {key: canonical[key] for key in sorted(canonical) if key != LANE_KEY}
    ordered[LANE_KEY] = canonical[LANE_KEY]
    text = json.dumps(ordered, indent=2)
    # A newline inside a JSON string is escaped, so this matches the top-level key and nothing nested.
    return text, text[: text.rindex(f'\n  "{LANE_KEY}": ')]


async def run_lanes_after_the_first_starts(
    run_lane: Callable[..., Awaitable[object]],
    width: int,
    *,
    shared_prompt_cache: bool,
) -> list[object]:
    """Every lane's answer or exception, in lane order, with the siblings started once the first lane is answering.

    Without ``shared_prompt_cache`` every lane starts at once. Otherwise the siblings start when the first lane begins
    its response, when it ends without one, or after :data:`LEAD_RESPONSE_WAIT_SEC`, whichever comes first.
    """
    if width <= 1 or not shared_prompt_cache:
        return await asyncio.gather(*(run_lane(index) for index in range(width)), return_exceptions=True)
    started = asyncio.Event()
    first = asyncio.ensure_future(run_lane(0, on_response_started=started.set))
    waiting = asyncio.ensure_future(started.wait())
    try:
        await asyncio.wait({first, waiting}, timeout=LEAD_RESPONSE_WAIT_SEC, return_when=asyncio.FIRST_COMPLETED)
        siblings = await asyncio.gather(*(run_lane(index) for index in range(1, width)), return_exceptions=True)
        (lead,) = await asyncio.gather(first, return_exceptions=True)
    finally:
        waiting.cancel()
        if not first.done():
            first.cancel()
    return [lead, *siblings]
