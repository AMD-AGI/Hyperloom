# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prompts for the rsi agent steps. Each ends by asking for the JSON (or Markdown) the step parses."""

from __future__ import annotations

import json

SYSTEM = """You are one step of a Meta RSI round for the Hyperloom repository. A round studies \
Hyperloom's own session records to cut its LLM token spend without hurting the optimization \
result. Every number you state must come from a file you read in this session; cite the path. \
Never invent data. End your reply exactly as the task asks."""

FINDINGS = """Read the round's analysis in {analysis_dir} (the *.txt summaries and the JSON/JSONL \
tables) and the Hyperloom source in the current directory. Find where recent sessions spend tokens \
without benefit, and propose at most {max_levers} code changes ("levers") that cut that spend \
without changing what the optimizer decides.

For each lever give: the waste it removes and its evidence (numbers, each with the file it came \
from); an estimated saving and its basis (for example "% of orchestration weighted tokens"); the \
change and the files it touches; an environment variable that turns it off; the tests that pin it; \
the risk to the optimization result.

Also say whether a role could run on a cheaper model, citing a natural experiment in the data if \
there is one; leave the list empty otherwise.

End with:
```json
{{"levers": [{{"id": "kebab-case-id", "title": "...", "problem": "...", \
"evidence": [{{"metric": "...", "value": "...", "source": "path"}}], \
"estimated_saving": {{"basis": "...", "pct": 0.0}}, "change": "...", "files": ["src/..."], \
"off_switch": "ENV_VAR", "tests": ["..."], "risk": "..."}}], \
"model_choices": [{{"role": "...", "model": "...", "reason": "..."}}]}}
```"""

FINDINGS_RETRY = """Your previous reply could not be used: {problem}
Reply again with the complete JSON in the same shape."""

IMPLEMENT = """Implement this lever in the repository in the current directory, a git worktree on \
branch {branch}:

{lever}

Follow AGENTS.md and docs/contributing/style-guide.md. Keep the change to this one concern, behind \
the off switch {off_switch} (unset means on), with tests next to the code it changes that pin the \
behavior and its failure modes. Run the tests you add or touch and `{lint}` on the files you \
change, and fix what fails. Then commit with a message in the repository's style, \
"<type>(<area>): <what changed>" plus a short body with the evidence. Do not push.

End with:
```json
{{"commit": "<sha>", "tests": ["<pytest path or node id>"], "summary": "<one sentence>"}}
```"""

IMPLEMENT_RETRY = """

An earlier attempt left commits on this branch after {base}. The driver re-checked them and they \
did not pass:

{problem}

Fix this in the same worktree (amend or add commits) and end with the same JSON."""

DIAGNOSE = """Arms of one Meta RSI A/B ran the same scenario; the control is {control}. The \
comparisons are in {results}; each arm's session directory is under "session" there.

Explain the differences between the control and each other arm that matter for the result or the \
token spend: the exploration variants each arm tried and what they measured, the specialist \
findings each arm had or lacked (runs/specialist/*/specialist_done.json), specialist outcomes \
(timeouts, empty outputs), and where each arm's wall time went. Cite a file for every claim. Do not \
call an arm a failure: one run per arm cannot separate a code effect from the optimizer's \
run-to-run variance, so say what the evidence shows.

End with:
```json
{{"summary": "...", "differences": [{{"arm": "...", "finding": "...", \
"evidence": ["path: what it shows"], "affects": "result|tokens|both"}}]}}
```"""

REPORT = """Write the round report in Markdown from these files:
{inputs}

Sections: Summary; Data; Where the tokens go; Changes (one row per landed lever with its commits, \
evidence and off switch, then the dropped levers with the reason); Validation (the offline replays, \
the scenario, the arms, the comparison numbers, the diagnosis; an arm whose run record says \
"partial" was interrupted and is compared as it stands, so say so); Limitations. Give numbers \
exactly as the files state them and present the comparison rule's outcome as advisory. Reply with \
the Markdown only."""


def implement_prompt(lever: dict, branch: str, lint: str, base: str, problem: str = "") -> str:
    text = IMPLEMENT.format(
        branch=branch, lever=json.dumps(lever, indent=1), off_switch=lever.get("off_switch") or "an env var", lint=lint
    )
    return text + IMPLEMENT_RETRY.format(base=base, problem=problem) if problem else text
