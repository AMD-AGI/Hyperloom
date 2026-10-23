# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn what a finished campaign learned into a proposed patch to the optimization skill its agents read.

An agent edits a copy of the skill, from the campaign's evidence: what each round gained beside the one before it,
each round's own account, the delivery review and the delivered change. Each candidate is audited rather than
copied, and the patch is written for a maintainer to review; nothing is applied to the skill itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kernelforge.agent_backends import AgentRunSpec, watchdog_timeout_sec
from kernelforge.agent_backends.session_resume import is_api_failure
from kernelforge.agent_backends.workspace_guard import WorkspaceSafetyError
from kernelforge.config import Config
from kernelforge.tools.durable_io import publish_report

log = logging.getLogger(__name__)

DISTILL_MAX_TURNS = 200
REPORT_HEADING = "## Distillation report"
PATCH_NAME, REPORT_NAME, RESULT_NAME = "knowledge_patch.diff", "distill.md", "distill.json"

_ROLE = (
    "You curate a shared GPU-kernel optimization skill. From the evidence one finished optimization campaign left, "
    "you propose the few edits that would change what the next engineer does on another kernel, each one checked. "
    "A maintainer reviews your patch and decides; you edit only the copy of the skill you are given."
)


class DistillError(RuntimeError):
    """The skill or the campaign cannot be distilled as given."""


@dataclass(frozen=True)
class DistillBrief:
    """Where the evidence and the skill copy are."""

    skill_copy: str
    workspace: str
    base: str
    delivered: str
    history: str
    references: tuple[str, ...] = ()


def distill_prompt(brief: DistillBrief) -> str:
    ws = brief.workspace
    references = "\n".join(f"- {path}" for path in brief.references) or "- the installed packages themselves"
    return f"""The skill to improve is the copy at {brief.skill_copy}; its entry is SKILL.md. Read SKILL.md and the \
cards it routes to before you write anything. Edit nothing outside that directory, and run no command that deletes or changes anything outside it: a card's own cleanup commands (a cache wipe, a glob) are text to check, not steps to run. Compile or probe in a new directory beside the copy.

The campaign's evidence, in {ws}:
- forge_experiments/ablation.md: what each kept round gained beside the version before it (pays, within noise, \
hurts) and what each change costs when reverted from the best.
- forge_experiments/lessons/iter_NNN.md: each round's own account of what it tried, measured and ruled out, kept or \
not.
- forge_experiments/finalize.md: the delivery review.
- program.md: the task. The delivered change: `git -C {ws} diff {brief.base} {brief.delivered}`; round by round: \
`git -C {ws} log -p {brief.base}..{brief.history}`.
Read-only references for checking claims:
{references}

What enters the skill (audit each candidate; do not copy):
1. General, not this kernel's tuned configuration: a lever, a mechanism, a pitfall, a dead end or a measurement fact \
that would change what someone does on another kernel. One shape's best tile or split count is not knowledge.
2. Not already there: search the skill first. Where it already says the same, add nothing; where the campaign's \
measurements contradict it, propose a correction.
3. Checked: an API, symbol or behaviour claim against the installed source; a hardware claim against the ISA \
documentation or the code the compiler emits. What you cannot check, leave out.
4. Numbers this campaign measured are marked "(field)" and given as an order of magnitude; a number you inferred \
rather than measured says so.
5. Cards carry no dates, campaign names, iteration numbers or source citations; the evidence goes in your report.
6. A dead end counts when the reason it failed is general: what the campaign tried that sat within noise or hurt.
7. Edit the card where the topic lives, in its style; create a card only when none fits, and route it from SKILL.md.

Keep the patch small enough for a maintainer to review. Each edit states one claim in one or two lines, as a row or bullet in a list or table the card already has wherever one fits; \
the mechanism, the numbers behind it and the caveats go in your report, not in the card. Where the skill says the \
same wrong thing in several places, correct each place with the smallest change and add no explanation. Few \
well-checked edits beat many.

End with a section headed `{REPORT_HEADING}`: one line per edit (addition or correction; the card; the claim in under \
twenty words; its evidence as artifact paths and numbers, or source file and line), then one line per candidate you \
rejected, with the reason."""


def report_of(reply: str) -> str:
    _, heading, report = (reply or "").partition(REPORT_HEADING)
    return (report if heading else reply or "").strip()


class DistillFailure(RuntimeError):
    """The backend failed before producing an answer."""


def make_distill_fn(config: Config, *, usage=None) -> Callable[..., Awaitable[str | None]]:
    """Build the curator on the campaign's own provider."""
    from kernelforge.agent_backends import AgentToolPolicy
    from kernelforge.agent_backends.registry import create_registered_backend

    runtime = config.agent_runtime()

    async def distill_fn(*, brief: DistillBrief) -> str | None:
        """One curation session; its reply, or None when it failed."""
        try:
            result = await asyncio.wait_for(
                create_registered_backend(runtime).run(
                    AgentRunSpec(
                        system_prompt=_ROLE,
                        user_prompt=distill_prompt(brief),
                        role="knowledge_distill",
                        cwd=brief.skill_copy,
                        writable=True,
                        timeout_sec=runtime.timeout_sec,
                        additional_directories=[brief.workspace, *brief.references],
                        tool_policy=AgentToolPolicy(
                            read=True, search=True, write=True, shell=True, max_turns=DISTILL_MAX_TURNS
                        ),
                        allow_dirty_baseline=True,
                        allow_untracked=True,
                    ),
                    usage=usage,
                ),
                timeout=watchdog_timeout_sec(runtime.timeout_sec),
            )
            if is_api_failure(result):
                raise DistillFailure(result.stderr_tail or result.end_reason or "no answer")
            return result.text or ""
        except (asyncio.TimeoutError, DistillFailure, WorkspaceSafetyError, OSError) as error:
            log.warning("knowledge distillation failed: %s", error)
            return None

    return distill_fn


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


def _skill_prefix(skill_dir: Path) -> str:
    """The skill's path inside its repository, so the patch applies from that repository's root."""
    root = _git(skill_dir, "rev-parse", "--show-toplevel")
    if root.returncode != 0:
        return ""
    return f"{skill_dir.resolve().relative_to(Path(root.stdout.strip()).resolve()).as_posix()}/"


def _campaign_refs(workspace: Path) -> tuple[str, str, str]:
    """The campaign's base, its delivered commit, and the last commit of its per-round history."""
    from kernelforge.finalize.delivery import FINALIZE_JSON, HISTORY_BRANCH
    from kernelforge.loop.campaign_config import CampaignConfigStore

    base = CampaignConfigStore(str(workspace)).load().base_commit
    head = _git(workspace, "rev-parse", "HEAD").stdout.strip()
    record = workspace / "forge_experiments" / FINALIZE_JSON
    delivered = json.loads(record.read_text()).get("final_commit", "") if record.is_file() else ""
    history = HISTORY_BRANCH if _git(workspace, "rev-parse", "--verify", HISTORY_BRANCH).returncode == 0 else head
    return base, delivered or head, history


def _fresh_copy(skill_dir: Path, copy: Path) -> None:
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(skill_dir, copy, ignore=shutil.ignore_patterns("__pycache__"))
    _git(copy, "init", "-q")
    _git(copy, "add", "-A")
    _git(copy, "-c", "user.name=distill", "-c", "user.email=distill@local", "commit", "-qm", "skill")


async def distill_campaign(
    workspace: str | Path,
    *,
    skill_dir: str | Path,
    out: str | Path,
    distill_fn,
    references: tuple[str, ...] = (),
) -> dict:
    """Propose a patch to ``skill_dir`` from the campaign in ``workspace``; write it, the report and the result
    under ``out``, and return the result."""
    workspace, skill_dir, out = Path(workspace).resolve(), Path(skill_dir).resolve(), Path(out).resolve()
    if not (skill_dir / "SKILL.md").is_file():
        raise DistillError(f"{skill_dir} holds no SKILL.md")
    base, delivered, history = _campaign_refs(workspace)
    copy = out / "skill"
    _fresh_copy(skill_dir, copy)
    brief = DistillBrief(
        str(copy),
        str(workspace),
        base,
        delivered,
        history,
        tuple(str(Path(r).resolve()) for r in references),
    )
    reply = await distill_fn(brief=brief)
    _git(copy, "add", "-A")
    prefix = _skill_prefix(skill_dir)
    patch = _git(copy, "diff", "--cached", f"--src-prefix=a/{prefix}", f"--dst-prefix=b/{prefix}").stdout
    files = [path for path in _git(copy, "diff", "--cached", "--name-only").stdout.splitlines() if path]
    report = report_of(reply) if reply is not None else ""
    publish_report(out / PATCH_NAME, patch)
    publish_report(out / REPORT_NAME, render_distill_md(skill_dir, report, patch))
    result = {
        "skill_dir": str(skill_dir),
        "patch": str(out / PATCH_NAME),
        "report": str(out / REPORT_NAME),
        "files": files,
        "failed": reply is None,
        "apply_from": str(Path(_git(skill_dir, "rev-parse", "--show-toplevel").stdout.strip() or skill_dir)),
    }
    publish_report(out / RESULT_NAME, json.dumps(result, indent=2))
    return result


def render_distill_md(skill_dir: Path, report: str, patch: str) -> str:
    """What a maintainer reviews in one place: each edit's evidence, then the patch itself."""
    diff = f"```diff\n{patch.rstrip()}\n```" if patch.strip() else "No edit proposed."
    return (
        f"# Proposed skill patch\n\nFor {skill_dir}; not applied.\n\n{report or '(no report)'}\n\n## Patch\n\n{diff}\n"
    )
