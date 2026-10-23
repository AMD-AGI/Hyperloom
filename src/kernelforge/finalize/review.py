# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The agent that reviews a finished campaign's kernel as a whole before its kept commits become one.

Each round of the campaign was judged alone, on speed; nobody read the sum. The review does: it tells the rounds that
carry the kernel from the passengers the ablation could not price above noise, removes what does not pay, and brings
the rest to the delivery rules. It edits only the kernel's sources; what it leaves is checked and timed afterwards.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kernelforge.agent_backends import AgentRunSpec, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.agent_backends.workspace_guard import WorkspaceSafetyError
from kernelforge.config import Config
from kernelforge.tools.durable_io import atomic_write_text
from kernelforge.workspace.edit_policy import TOOL_OWNED_UNTRACKED_GLOBS

log = logging.getLogger(__name__)

REVIEW_MAX_TURNS = 200
REPORT_HEADING = "## Review report"

_ROLE = (
    "You review the kernel a finished GPU-kernel optimization campaign delivers, as one change, before its kept "
    "commits are squashed into one. You remove what does not pay for itself and bring the rest to the delivery "
    "rules, without changing what the kernel computes. You edit only the files you are given."
)

DELIVERY_RULES = """Delivery rules (a reviewer rejects the change if any one is broken):
1. Concise. No dead code: no unused variable, function, import, branch, constant or knob; no disabled experiment
   arm, alternative path kept for later, commented-out code or probe. In traced kernel code (FlyDSL, Triton) a name
   only the traced body reads is used: search the file before deleting an "unused" name, and run the driver after.
2. Reuse. No private copy of a helper the kernel's library or repository already has.
3. Comments. English; why, not what; at most 3 lines per block, docstrings included; no measured numbers, dates,
   iteration numbers or change logs (they go in your report, which becomes the commit message); whole-line comments
   at most 4% of the lines the campaign added. A comment about code the campaign later changed is fixed or removed.
4. No debug traces: no print, debug or tmp names, TODO without an issue, breakpoint.
5. Names follow the file's existing style; no magic number where the value derives from the shape or a named
   constant.
6. The lint commands below pass on the files you edit."""


@dataclass(frozen=True)
class ReviewBrief:
    """What one review round is given."""

    base: str
    best: str
    kept_commits: int
    files: tuple[str, ...]
    driver: str
    pricing: str
    lint_commands: tuple[str, ...]
    open_findings: tuple[str, ...] = ()
    round_number: int = 1


def review_prompt(brief: ReviewBrief) -> str:
    files = " ".join(brief.files)
    lint = "\n".join(f"- `{command} {files}`" for command in brief.lint_commands) or "- none given"
    findings = (
        "\n\nThe previous round left these findings; fix them:\n"
        + "\n".join(f"- {line}" for line in brief.open_findings)
        if brief.open_findings
        else ""
    )
    return f"""Review round {brief.round_number}. The campaign started from {brief.base} and kept {brief.kept_commits} \
commit(s); HEAD ({brief.best}) is its best kernel. You may edit only: {files}. Never edit the driver \
({brief.driver}), program.md or any other file, and run no git command that changes the index or history.

1. Read the whole delivered change: `git diff {brief.base} HEAD -- {files}`. Where a part's purpose is unclear, \
`git log --reverse -p {brief.base}..HEAD -- {files}` shows the round that added it.
2. Tell the main changes from the passengers.
{brief.pricing}
   Code that a round within noise or hurting added, and that no paying round needs, is removed. A paying change \
stays whatever its size; one with many lines for its gain may be simplified only where its speed is kept.
3. Bring what remains to the delivery rules.
4. Keep what the kernel computes: run `python {brief.driver}` to check correctness after your edits. When you are \
done the kernel is checked again and timed beside HEAD; your edits are discarded if it is wrong or slower.

{DELIVERY_RULES}

Lint commands:
{lint}{findings}

End with a section headed `{REPORT_HEADING}`: one line per edit you made and why (for example "removed the \
disabled XCD-remap arm: round 17 within noise"), then one line naming the rules you checked and found nothing to \
fix."""


def report_of(reply: str) -> str:
    """The review's own account, or its whole reply when it wrote no report section."""
    _, heading, report = (reply or "").partition(REPORT_HEADING)
    return (report if heading else reply or "").strip()


class ReviewFailure(RuntimeError):
    """The backend failed before producing an answer."""


def make_review_fn(config: Config, *, usage=None) -> Callable[..., Awaitable[str | None]]:
    """Build the reviewer on the campaign's own provider."""
    from kernelforge.agent_backends import AgentToolPolicy
    from kernelforge.agent_backends.registry import create_registered_backend

    runtime = config.agent_runtime()
    selected = None

    async def review_fn(*, workspace: str, brief: ReviewBrief, record: Path) -> str | None:
        """One review session; its reply, or None when it failed, so a failure never reads as a clean review."""
        nonlocal selected
        prompt = review_prompt(brief)
        reply, failure = "", ""
        try:
            selected = selected or create_registered_backend(runtime)
            result = await asyncio.wait_for(
                selected.run(
                    AgentRunSpec(
                        system_prompt=_ROLE,
                        user_prompt=prompt,
                        role="delivery_review",
                        cwd=workspace,
                        writable=True,
                        timeout_sec=runtime.timeout_sec,
                        tool_policy=AgentToolPolicy(
                            read=True, search=True, write=True, shell=True, max_turns=REVIEW_MAX_TURNS
                        ),
                        target_files=list(brief.files),
                        driver_script=brief.driver,
                        allow_dirty_baseline=True,
                        # A probe the review leaves is deleted afterwards; refusing it here would roll back the review.
                        allow_untracked=True,
                        ignored_untracked_globs=list(TOOL_OWNED_UNTRACKED_GLOBS),
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(runtime.timeout_sec),
            )
            if is_api_failure(result):
                raise ReviewFailure(result.stderr_tail or result.end_reason or "no answer")
            reply = result.text or ""
        except (asyncio.TimeoutError, ReviewFailure, WorkspaceSafetyError, OSError) as error:
            log.warning("delivery review failed: %s", error)
            failure = f"{type(error).__name__}: {error}"
        outcome = f"## Failed\n\n{failure}\n" if failure else f"## Reply\n\n{reply}\n"
        atomic_write_text(record, f"# Delivery review, round {brief.round_number}\n\n## Task\n\n{prompt}\n\n{outcome}")
        return None if failure else reply

    return review_fn
