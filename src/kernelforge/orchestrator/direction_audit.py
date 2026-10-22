# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A read-only agent that lists a campaign's optimization directions.

It runs in two modes. ``survey`` once per Analysis commit, to turn the evidence into the directions the ledger
starts from. ``audit`` when every listed direction is mined out and recent rounds are flat, to look for the
directions nobody listed before the campaign may stop early.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from kernelforge.agent_backends import AgentRunSpec, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.config import Config
from kernelforge.tools.durable_io import atomic_write_text

log = logging.getLogger(__name__)

SURVEY, AUDIT = "survey", "audit"
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
    "Aim for at least {min_directions} directions in total, counting the ledger's. Do not repeat a direction the "
    "ledger already holds."
)

_AUDIT = (
    "Audit: every direction in the ledger is exhausted or ruled infeasible, and the last rounds gained little. "
    "Before the campaign stops, find directions the ledger does not hold that the evidence still supports: a "
    "lever never tried, a bottleneck the profiles show that no direction addressed, or a direction retired on "
    "evidence that no longer holds. Return none if you find none; an empty list ends the campaign."
)

_OUTPUT = """

Reply with your reasoning, then one fenced JSON block, exactly:
```json
{"new_directions": [{"title": "...", "category": "...", "hypothesis": "...", "expected_gain": "...",
  "evidence": ["<absolute path of a file that supports it>", "..."]}]}
```
Each direction needs at least one evidence path to an existing file; one without is dropped."""

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class DirectionAuditFailure(RuntimeError):
    """The backend failed before producing an answer."""


def _task_prompt(mode: str, payload: dict, *, min_directions: int) -> str:
    instruction = _SURVEY.format(min_directions=min_directions) if mode == SURVEY else _AUDIT
    return f"{instruction}\n\nCampaign state (JSON):\n{json.dumps(payload, indent=2, default=str)}{_OUTPUT}"


def _evidenced(entry: object, workspace: Path) -> bool:
    if not isinstance(entry, dict) or not str(entry.get("title") or "").strip():
        return False
    evidence = entry.get("evidence") or []
    if not isinstance(evidence, list):
        return False
    return any((workspace / str(item)).exists() for item in evidence if str(item).strip())


def parse_new_directions(reply: str, *, workspace: Path) -> list[dict]:
    """The evidenced directions in the reply's last JSON block; [] when it has none."""
    for block in reversed(_FENCED_JSON.findall(reply or "")):
        try:
            payload = json.loads(block)
        except ValueError:
            continue
        entries = payload.get("new_directions") if isinstance(payload, dict) else None
        if isinstance(entries, list):
            return [entry for entry in entries if _evidenced(entry, workspace)]
    return []


def make_direction_audit_fn(config: Config, *, usage=None, min_directions: int = 10) -> Callable[..., Awaitable]:
    """Build the read-only direction auditor on the campaign's own provider."""
    from kernelforge.agent_backends import AgentToolPolicy
    from kernelforge.agent_backends.registry import create_registered_backend

    runtime = config.agent_runtime()
    timeout_sec = int(runtime.options.get("supervisor_timeout_sec") or runtime.timeout_sec)
    selected = None

    async def direction_audit_fn(mode: str, *, workspace: str, iteration: int, payload: dict) -> list[dict] | None:
        """One bounded pass; None when it failed, so a failure never reads as \"found nothing\"."""
        nonlocal selected
        task = _task_prompt(mode, payload, min_directions=min_directions)
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
        return None if failed else parse_new_directions(reply, workspace=Path(workspace))

    return direction_audit_fn
