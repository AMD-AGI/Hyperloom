# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A read-only agent that lists and ranks a campaign's optimization directions.

It runs in three modes. ``survey`` once per Analysis commit, to turn the evidence into the directions the ledger
starts from. ``replan`` every few attempts, and whenever progress is flat, to rank what is left by the gain it is
now expected to bring, retiring duplicates and directions the measurements have ruled out. ``audit`` when progress
is flat and nothing promising is left, to look for the directions nobody listed before the campaign may stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kernelforge.agent_backends import AgentRunSpec, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.config import Config
from kernelforge.tools.durable_io import atomic_write_text

log = logging.getLogger(__name__)

SURVEY, REPLAN, AUDIT = "survey", "replan", "audit"
DIRECTION_AUDIT_MAX_TURNS = 40

_ROLE = (
    "You map the optimization directions of one GPU-kernel optimization campaign. You read evidence (the "
    "Analysis report and profiles, the campaign history, the knowledge maps and the optimization skill they "
    "point to) and list the distinct directions worth a measured attempt. You do not write code."
)

_SURVEY = (
    "Survey: list the directions this kernel should try, from the Analysis evidence and the skill's lever "
    "taxonomy. Cover the taxonomy broadly (grid and split-K, main loop, data path and LDS, scheduling and "
    "locality, epilogue, tuning, and whatever else the evidence names) rather than many variants of one idea. "
    "Do not repeat a direction the ledger already holds."
)

_REPLAN = (
    "Replan: rank every direction in the ledger whose status is OPEN, ACTIVE or KEPT by the gain you now expect "
    "from its next attempt, in percent of the best score. Weigh what the campaign has measured: each direction's "
    "tries, keeps and gain in the ledger, why its attempts failed (the lesson records, optimization_history.md), "
    "the Analysis evidence and the headroom left. A direction that failed for a reason its next attempt cannot "
    "avoid is worth little; one that kept and still has headroom is worth more. Retire a duplicate (reason "
    "`duplicate of <id>`) and a direction the measurements have ruled out (say which measurement). Add a "
    "direction only with evidence. Directions expected to gain {priority_gain} or more keep the campaign running, "
    "and once progress is flat every plan must pursue one of them."
)

_AUDIT = (
    "Audit: the best score has barely moved over the last hours of attempts, and no direction in the ledger is "
    "still expected to gain {priority_gain} or more. Before the campaign stops, find directions the ledger does "
    "not hold that the evidence still supports and that you expect to gain at least that much: a lever never "
    "tried, a bottleneck the profiles show that no direction addressed, or a direction retired on evidence that "
    "no longer holds. Return none if you find none; an empty list ends the campaign."
)

_NEW_DIRECTION = (
    '{"title": "...", "category": "...", "hypothesis": "...", "expected_gain_pct": 3.0, "reason": "why that much",\n'
    '   "evidence": ["<absolute path of a file that supports it>", "..."]}'
)

_OUTPUT = (
    """

Reply with your reasoning, then one fenced JSON block, exactly:
```json
{"new_directions": [%s]}
```
Each new direction needs at least one evidence path to an existing file; one without is dropped."""
    % _NEW_DIRECTION
)

_REPLAN_OUTPUT = (
    """

Reply with your reasoning, then one fenced JSON block, exactly:
```json
{"ranking": [{"id": "<ledger id>", "expected_gain_pct": 2.5, "reason": "..."}],
 "retire": [{"id": "<ledger id>", "reason": "..."}],
 "new_directions": [%s]}
```
Rank every direction still in play; give expected_gain_pct as a number. Each new direction needs at least one
evidence path to an existing file; one without is dropped."""
    % _NEW_DIRECTION
)

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class DirectionAuditFailure(RuntimeError):
    """The backend failed before producing an answer."""


@dataclass(frozen=True)
class DirectionReply:
    """What one pass proposed: directions to add, expected gains to set, and directions to retire."""

    new_directions: tuple[dict, ...] = ()
    ranking: tuple[dict, ...] = ()
    retire: tuple[dict, ...] = ()


def _task_prompt(mode: str, payload: dict, *, priority_gain: float) -> str:
    instruction = {SURVEY: _SURVEY, REPLAN: _REPLAN, AUDIT: _AUDIT}[mode].format(
        priority_gain=f"{priority_gain * 100:g}%"
    )
    output = _REPLAN_OUTPUT if mode == REPLAN else _OUTPUT
    return f"{instruction}\n\nCampaign state (JSON):\n{json.dumps(payload, indent=2, default=str)}{output}"


def _evidenced(entry: object, workspace: Path) -> bool:
    if not isinstance(entry, dict) or not str(entry.get("title") or "").strip():
        return False
    evidence = entry.get("evidence") or []
    if not isinstance(evidence, list):
        return False
    return any((workspace / str(item)).exists() for item in evidence if str(item).strip())


def _entries(payload: dict, key: str) -> list[dict]:
    entries = payload.get(key)
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def parse_reply(reply: str, *, workspace: Path) -> DirectionReply:
    """The reply's last JSON block: its evidenced new directions, its ranking, and its retirements that give a
    reason. Empty when it has none."""
    for block in reversed(_FENCED_JSON.findall(reply or "")):
        try:
            payload = json.loads(block)
        except ValueError:
            continue
        if not isinstance(payload, dict) or not {"new_directions", "ranking", "retire"} & payload.keys():
            continue
        return DirectionReply(
            new_directions=tuple(
                entry for entry in _entries(payload, "new_directions") if _evidenced(entry, workspace)
            ),
            ranking=tuple(_entries(payload, "ranking")),
            retire=tuple(entry for entry in _entries(payload, "retire") if str(entry.get("reason") or "").strip()),
        )
    return DirectionReply()


def make_direction_audit_fn(config: Config, *, priority_gain: float, usage=None) -> Callable[..., Awaitable]:
    """Build the read-only direction agent on the campaign's own provider."""
    from kernelforge.agent_backends import AgentToolPolicy
    from kernelforge.agent_backends.registry import create_registered_backend

    runtime = config.agent_runtime()
    timeout_sec = int(runtime.options.get("supervisor_timeout_sec") or runtime.timeout_sec)
    selected = None

    async def direction_audit_fn(mode: str, *, workspace: str, iteration: int, payload: dict) -> DirectionReply | None:
        """One bounded pass; None when it failed, so a failure never reads as \"found nothing\"."""
        nonlocal selected
        task = _task_prompt(mode, payload, priority_gain=priority_gain)
        reply, failed = "", False
        try:
            selected = selected or create_registered_backend(runtime)
            result = await asyncio.wait_for(
                selected.run(
                    AgentRunSpec(
                        system_prompt=_ROLE,
                        user_prompt=task,
                        role="direction_audit",
                        cwd=workspace,
                        writable=False,
                        timeout_sec=timeout_sec,
                        tool_policy=AgentToolPolicy(
                            read=True, search=True, write=False, shell=False, max_turns=DIRECTION_AUDIT_MAX_TURNS
                        ),
                        protected_globs=["*"],
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(timeout_sec),
            )
            if is_api_failure(result):
                raise DirectionAuditFailure(result.stderr_tail or result.end_reason or "no answer")
            reply = result.text or ""
        except (asyncio.TimeoutError, DirectionAuditFailure, OSError) as error:
            log.warning("direction %s failed: %s", mode, error)
            print(f"  [directions] {mode} failed ({error}) — retried next iteration", flush=True)
            failed = True
        record = Path(workspace) / "forge_experiments" / "directions" / f"{mode}_iter_{iteration:03d}.md"
        atomic_write_text(
            record, f"# Direction {mode}, iteration {iteration}\n\n## Task\n\n{task}\n\n## Reply\n\n{reply}\n"
        )
        return None if failed else parse_reply(reply, workspace=Path(workspace))

    return direction_audit_fn
