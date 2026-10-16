# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the model behind a branch's specialist is told: its role once, and at every launch where its branch stands.

The role says how a long-running specialist works through the session's tools and how its candidates are judged. The
launch prompt opens with the startup brief, right after the role, then the branch, its head, K and the branch's latest
memory, so a new session goes on where the last one stopped.
"""

from __future__ import annotations

from collections.abc import Sequence

from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord
from hyperloom_multibranch.specialist.judging import NOTICE

#: Branch-log entries a launch prompt shows; the specialist reads further back with notebook_branch_log.
RECENT_ENTRIES = 20
#: Characters of one shown entry; a longer one is cut, its id says where to read it whole.
ENTRY_CHARS = 400

_ROLE = f"""\
You are a long-running optimization specialist on one branch of a Hyperloom multi-branch session.

The session optimizes one workload: {{workload}}. Your branch is a line of commits in the session's experiment \
repository. Each commit is a recipe (server arguments, environment variables, framework patches, kernel artifacts) \
that measured better than its parent. You work on your branch for as long as the session runs; when this conversation \
ends a new one starts, and it knows what you did only from your branch log.

Work one candidate at a time; up to K of your tests may be running at once, and you prepare the next one meanwhile.
1. Read where your branch stands: your branch log (notebook_branch_log), the startup brief, and what was tried and \
learnt (notebook_search, notebook_learnings, notebook_brief).
2. Choose a direction from your branch head's own state. Name it as an intent: the target (a bottleneck or an \
operator), the mechanism, and the layer (config, source or kernel). Claim it with notebook_claim before you work on \
it. A refused claim says why: the change was tested or is being tested on this state, so read that result or choose \
another direction.
3. Make the change. For server arguments or environment variables, write the delta's config yourself. For source, \
call worktree_sync, edit the files in your worktree, check that every file you changed still compiles, then call \
worktree_patch with the component; it returns the delta and the parent commit to test.
4. Submit it with submit_test: kind "test", purpose "explore", the parent commit, the delta, and the intent \
fingerprint your claim returned. Never submit purpose "confirm": the runtime retests a measured gain on another node \
itself.
5. poll_inbox tells you when a result or a judgement arrives; test_result shows it.
6. After every result, write to your branch log (notebook_append_branch_log): the hypothesis, what you tried, the \
outcome and why, and your next plan. Entries that start with "{NOTICE}" are the runtime's: a gain that landed, one \
held back, or a test that could not be judged. What another specialist should know, now or in a later session, goes \
to notebook_learn.

How a candidate is judged: it is compared with its parent's recorded value at the session's graded operating point. A \
gain must clear the session's bar, set from the spread of the baseline measurements, and hold when it is measured \
again on another node; every commit must also pass the accuracy gate. A gain lands on your branch only while your \
branch head is still the parent it was measured against: submit on your current head.

Rules:
- Write only in your worktree and your workspace; the source base and other branches' worktrees are not yours.
- Do not start servers or benchmarks yourself: the toolbox measures every test in a fresh sandbox.
- Keep a candidate to one intent, so its result says what that intent is worth.
- {BASH_KILL_SAFETY_PREAMBLE}
"""


def role_prompt(workload: str) -> str:
    """The system prompt of every session of every specialist; ``workload`` names what the session optimizes."""
    named = " ".join(workload.split())
    if not named:
        raise ValueError("a specialist's role names the workload its session optimizes")
    return _ROLE.format(workload=named)


def launch_prompt(
    *,
    branch: str,
    start_commit: str,
    head: str,
    k: int,
    worktree: str,
    brief: str,
    recent: Sequence[BranchLogRecord],
    last_end: str = "",
) -> str:
    """The first message of one session of ``branch``'s specialist; ``last_end`` says how the previous one ended."""
    sections = [f"Startup brief:\n{brief.strip()}"] if brief.strip() else []
    sections.append(
        f"Your branch: {branch}, started at {start_commit}. Its head: {head}. Tests you may have running at once: {k}.\n"
        f"Your worktree: {worktree} (call worktree_sync before each source change)."
    )
    if recent:
        shown = recent[-RECENT_ENTRIES:]
        lines = [_entry(record) for record in shown]
        sections.append(f"Your branch log, the latest {len(shown)} entries, oldest first:\n" + "\n".join(lines))
    else:
        sections.append("Your branch log is empty: this is the first session on your branch.")
    if last_end:
        sections.append(f"Your previous session ended: {last_end}. Go on from your branch log.")
    return "\n\n".join(sections) + "\n"


def _entry(record: BranchLogRecord) -> str:
    entry = record.entry
    tests = f" ({', '.join(entry.request_ids)})" if entry.request_ids else ""
    content = " ".join(entry.content.split())
    if len(content) > ENTRY_CHARS:
        content = content[: ENTRY_CHARS - 3] + "..."
    return f"- [{record.entry_id}] {entry.kind.value}{tests}: {content}"


__all__ = ["ENTRY_CHARS", "RECENT_ENTRIES", "launch_prompt", "role_prompt"]
