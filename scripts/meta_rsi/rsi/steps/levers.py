# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent steps that decide and land the round's changes: findings, then one commit series per lever."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from meta_rsi.rsi import prompts
from meta_rsi.rsi.agent import READ_TOOLS, AgentResult, AgentSpec, extract_json
from meta_rsi.rsi.pipeline import RoundContext, StepFailed

LEVER_KEYS = (
    "id",
    "title",
    "problem",
    "evidence",
    "estimated_saving",
    "change",
    "files",
    "off_switch",
    "tests",
    "risk",
)
LEVER_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
IMPLEMENT_TOOLS = ("Bash", *READ_TOOLS, "Edit", "Write")
DEFAULT_BUDGET = {"findings": 20.0, "implement": 30.0, "diagnose": 10.0, "report": 10.0}
DEFAULT_TURNS = {"findings": 60, "implement": 150, "diagnose": 60, "report": 40}
OUTPUT_TAIL_LINES = 40


def ensure_worktrees(ctx: RoundContext) -> None:
    """The round's two checkouts: ``base`` detached at base_ref and ``candidate`` on the round branch."""
    t = ctx.config.target
    base, cand = ctx.worktree("base"), ctx.worktree("candidate")
    if not base.exists():
        ctx.git("worktree", "add", "--detach", str(base), t.base_ref, cwd=t.repo)
    if not cand.exists():
        ctx.git("worktree", "add", "--no-track", "-b", t.branch, str(cand), t.base_ref, cwd=t.repo)


def agent_spec(ctx: RoundContext, step: str, name: str, prompt: str, **kwargs) -> AgentSpec:
    a = ctx.config.agent
    return AgentSpec(
        name=name,
        prompt=prompt,
        system_prompt=prompts.SYSTEM,
        model=a.model,
        budget_usd=a.budget_usd.get(step, DEFAULT_BUDGET[step]),
        max_turns=a.max_turns.get(step, DEFAULT_TURNS[step]),
        env=ctx.agent_env(),
        transcript=ctx.path("agents", f"{name.replace(':', '_').replace('#', '_')}.jsonl"),
        **kwargs,
    )


def lever_problems(data: object, max_levers: int) -> list[str]:
    """What keeps a findings reply from being used; empty when it is usable."""
    if not isinstance(data, dict) or not isinstance(data.get("levers"), list):
        return ["the reply must be an object with a 'levers' list"]
    levers, problems, seen = data["levers"], [], set()
    if not levers:
        problems.append("no levers were proposed")
    if len(levers) > max_levers:
        problems.append(f"{len(levers)} levers proposed; the limit is {max_levers}")
    for i, lever in enumerate(levers):
        if not isinstance(lever, dict):
            problems.append(f"levers[{i}] is not an object")
            continue
        missing = [k for k in LEVER_KEYS if lever.get(k) in (None, "", [])]
        if missing:
            problems.append(f"levers[{i}] lacks {', '.join(missing)}")
        lever_id = str(lever.get("id") or "")
        if not LEVER_ID.match(lever_id) or lever_id in seen:
            problems.append(f"levers[{i}].id must be a unique kebab-case slug")
        seen.add(lever_id)
        if not isinstance((lever.get("estimated_saving") or {}).get("pct"), (int, float)):
            problems.append(f"levers[{i}].estimated_saving.pct must be a number")
    return problems


def render_findings(data: dict) -> str:
    lines = ["# Round findings", ""]
    for lever in data["levers"]:
        saving = lever["estimated_saving"]
        lines += [f"## {lever['title']} (`{lever['id']}`)", "", lever["problem"], ""]
        lines += [f"- {e.get('metric')}: {e.get('value')} ({e.get('source')})" for e in lever["evidence"]]
        lines += ["", f"Estimated saving: {saving['pct']}% of {saving.get('basis', '?')}.", ""]
        lines += [f"Change: {lever['change']}", "", f"Off switch: `{lever['off_switch']}`. Risk: {lever['risk']}", ""]
    if data.get("model_choices"):
        lines += ["## Model choices", ""]
        lines += [f"- {m.get('role')}: {m.get('model')} ({m.get('reason')})" for m in data["model_choices"]]
    return "\n".join(lines) + "\n"


def findings(ctx: RoundContext) -> dict:
    """A Claude Code agent reads the analysis and the source and proposes the round's levers."""
    ensure_worktrees(ctx)
    analysis = ctx.round_dir / "analysis"
    max_levers = ctx.config.agent.max_levers
    prompt = prompts.FINDINGS.format(analysis_dir=analysis, max_levers=max_levers)
    problems: list[str] = []
    for attempt in (1, 2):
        if problems:
            prompt = prompts.FINDINGS.format(analysis_dir=analysis, max_levers=max_levers)
            prompt += "\n\n" + prompts.FINDINGS_RETRY.format(problem="; ".join(problems))
        spec = agent_spec(
            ctx,
            "findings",
            f"findings#{attempt}",
            prompt,
            cwd=ctx.worktree("base"),
            tools=READ_TOOLS,
            add_dirs=(analysis,),
        )
        result = ctx.run_agent(spec)
        if result.is_error:
            raise StepFailed(f"the findings agent ended with an error: {result.error}")
        try:
            data = extract_json(result.text)
        except ValueError as exc:
            problems = [str(exc)]
            continue
        problems = lever_problems(data, max_levers)
        if not problems:
            ctx.path("levers.json").write_text(json.dumps(data, indent=1))
            ctx.path("findings.md").write_text(render_findings(data))
            return {"levers": [lever["id"] for lever in data["levers"]]}
    raise StepFailed("the findings reply was unusable: " + "; ".join(problems))


def _tail(text: str) -> str:
    return "\n".join(text.splitlines()[-OUTPUT_TAIL_LINES:])


def check_lever(ctx: RoundContext, cand: Path, base_sha: str, result: AgentResult) -> str:
    """Why the agent's commits cannot land (empty when they can): commit, clean tree, lint, tests."""
    if result.is_error:
        return f"the agent session ended with an error: {result.error}"
    try:
        reply = extract_json(result.text)
    except ValueError:
        return "the reply ended without the requested JSON"
    if ctx.git("rev-parse", "HEAD", cwd=cand) == base_sha:
        return "no commit was made"
    dirty = ctx.git("status", "--porcelain", cwd=cand)
    if dirty:
        return f"uncommitted changes remain:\n{dirty}"
    changed = [
        p
        for p in ctx.git("diff", "--name-only", "--diff-filter=AM", f"{base_sha}..HEAD", cwd=cand).split()
        if p.endswith(".py")
    ]
    checks = ctx.config.checks
    log = ctx.path("logs", "implement.log")
    if changed:
        lint = ctx.runner.run([*checks.lint, *changed], cwd=cand, log=log, check=False)
        if lint.returncode != 0:
            return f"`{' '.join(checks.lint)}` failed:\n{_tail(lint.stdout + lint.stderr)}"
    tests = [str(t) for t in reply.get("tests") or [] if str(t).strip()] if isinstance(reply, dict) else []
    if not tests:
        return "the reply named no tests"
    env = {**ctx.agent_env(), "PYTHONPATH": str(cand / "src")}
    try:
        run = ctx.runner.run(
            [str(checks.python), "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
            cwd=cand,
            env=env,
            log=log,
            check=False,
            timeout=60 * checks.timeout_min,
        )
    except StepFailed as exc:
        if not isinstance(exc.__cause__, subprocess.TimeoutExpired):
            raise
        return f"the named tests did not finish within checks.timeout_min ({checks.timeout_min:g} min)"
    if run.returncode != 0:
        return f"the named tests failed:\n{_tail(run.stdout + run.stderr)}"
    return ""


def implement_one(ctx: RoundContext, lever: dict, base_sha: str) -> dict:
    """Up to ``implement_attempts`` agent sessions for one lever; drops its commits if none passes."""
    cand, t = ctx.worktree("candidate"), ctx.config.target
    lint = " ".join(ctx.config.checks.lint)
    problem = ""
    for attempt in range(1, ctx.config.agent.implement_attempts + 1):
        prompt = prompts.implement_prompt(lever, t.branch, lint, base_sha, problem)
        spec = agent_spec(
            ctx,
            "implement",
            f"implement:{lever['id']}#{attempt}",
            prompt,
            cwd=cand,
            tools=IMPLEMENT_TOOLS,
            write_root=cand,
        )
        problem = check_lever(ctx, cand, base_sha, ctx.run_agent(spec))
        if not problem:
            commits = ctx.git("log", "--format=%h %s", f"{base_sha}..HEAD", cwd=cand).splitlines()
            return {"status": "landed", "commits": commits, "attempts": attempt}
        ctx.log(f"lever {lever['id']} attempt {attempt}: {problem.splitlines()[0]}")
    ctx.git("reset", "--hard", base_sha, cwd=cand)
    ctx.git("clean", "-fd", cwd=cand)
    return {"status": "dropped", "reason": problem, "attempts": ctx.config.agent.implement_attempts}


def implement(ctx: RoundContext) -> dict:
    """Land each lever as its own commits on the candidate branch, resuming per lever after a restart."""
    ensure_worktrees(ctx)
    cand = ctx.worktree("candidate")
    levers = json.loads((ctx.round_dir / "levers.json").read_text())["levers"][: ctx.config.agent.max_levers]
    done = ctx.state.data.setdefault("implement", {})
    for lever in levers:
        record = done.get(lever["id"]) or {}
        if record.get("status") in ("landed", "dropped"):
            continue
        if record.get("status") == "running":
            ctx.git("reset", "--hard", record["base"], cwd=cand)
        base_sha = ctx.git("rev-parse", "HEAD", cwd=cand)
        done[lever["id"]] = {"status": "running", "base": base_sha}
        ctx.state.save()
        done[lever["id"]] = implement_one(ctx, lever, base_sha)
        ctx.state.save()
    landed = [i for i, r in done.items() if r["status"] == "landed"]
    if not landed:
        raise StepFailed("no lever passed the driver's checks; see logs/implement.log and agents/")
    return {"landed": landed, "dropped": [i for i, r in done.items() if r["status"] == "dropped"]}
