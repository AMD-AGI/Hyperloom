# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Closing steps: an agent explains the arms' differences, another writes the report; then publish."""

from __future__ import annotations

import json

from meta_rsi.rsi import prompts
from meta_rsi.rsi.agent import READ_TOOLS, extract_json
from meta_rsi.rsi.pipeline import RoundContext, StepFailed
from meta_rsi.rsi.steps.ab import arm_dir
from meta_rsi.rsi.steps.levers import agent_spec

REPORT_INPUTS = (
    "findings.md",
    "levers.json",
    "suite/suite.json",
    "scenario.json",
    "diagnosis.md",
    "ab/environment.json",
)


def render_diagnosis(data: dict) -> str:
    lines = ["# A/B diagnosis", "", str(data.get("summary") or ""), ""]
    for d in data.get("differences") or []:
        lines += [f"## {d.get('arm')}: {d.get('finding')} ({d.get('affects')})", ""]
        lines += [f"- {e}" for e in d.get("evidence") or []]
        lines.append("")
    return "\n".join(lines)


def diagnose(ctx: RoundContext) -> dict:
    """A read-only agent compares the arms' logs and artifacts and cites what explains each difference."""
    results = sorted((ctx.round_dir / "results").glob("*.json"))
    if not results:
        raise StepFailed("no comparison results; run the compare step first")
    arms = ctx.config.ab.arms
    prompt = prompts.DIAGNOSE.format(control=arms[0].name, results=", ".join(str(p) for p in results))
    dirs = (ctx.round_dir, *(arm_dir(ctx, a.name) for a in arms))
    if ctx.config.router_log:
        dirs += (ctx.config.router_log.parent,)
    result = ctx.run_agent(
        agent_spec(ctx, "diagnose", "diagnose", prompt, cwd=ctx.round_dir, tools=READ_TOOLS, add_dirs=dirs)
    )
    if result.is_error:
        raise StepFailed(f"the diagnose agent ended with an error: {result.error}")
    try:
        data = extract_json(result.text)
    except ValueError as exc:
        raise StepFailed(f"the diagnosis reply was unusable: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("differences"), list):
        raise StepFailed("the diagnosis reply must be an object with a 'differences' list")
    ctx.path("diagnosis.json").write_text(json.dumps(data, indent=1))
    ctx.path("diagnosis.md").write_text(render_diagnosis(data))
    return {"differences": len(data["differences"])}


def report(ctx: RoundContext) -> dict:
    """A read-only agent writes report.md from the round's artifacts and the implement outcomes."""
    outcomes = ctx.path("implement.json")
    outcomes.write_text(json.dumps(ctx.state.data.get("implement", {}), indent=1))
    inputs = [ctx.round_dir / name for name in REPORT_INPUTS if (ctx.round_dir / name).exists()]
    inputs += [outcomes, *sorted((ctx.round_dir / "results").glob("*.json"))]
    prompt = prompts.REPORT.format(inputs="\n".join(f"- {p}" for p in inputs))
    result = ctx.run_agent(agent_spec(ctx, "report", "report", prompt, cwd=ctx.round_dir, tools=READ_TOOLS))
    if result.is_error or not result.text.strip():
        raise StepFailed(f"the report agent returned no report: {result.error or 'empty reply'}")
    ctx.path("report.md").write_text(result.text.strip() + "\n")
    return {"report": str(ctx.round_dir / "report.md")}


def publish(ctx: RoundContext) -> dict:
    """Push the candidate branch when publish.push is set; never opens a pull request."""
    if not ctx.config.publish_push:
        return {"skipped": True, "reason": "publish.push is off"}
    t = ctx.config.target
    ctx.git("push", ctx.config.publish_remote, t.branch, cwd=ctx.worktree("candidate"))
    return {"pushed": f"{ctx.config.publish_remote}/{t.branch}"}
