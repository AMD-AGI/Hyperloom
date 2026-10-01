# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic data steps: fetch Pulse sessions, run the analyses, pick the scenario, replay levers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from meta_rsi.rsi.pipeline import SCRIPTS_DIR, RoundContext, StepFailed

ANALYSES = ("an_ledger.py", "an_global.py", "an_era.py", "an_idle.py", "an_specialist.py")


def _script(ctx: RoundContext, name: str, *args: str, log: str, check: bool = True, env: dict | None = None) -> str:
    proc = ctx.runner.run(
        [sys.executable, str(SCRIPTS_DIR / name), *args],
        cwd=SCRIPTS_DIR,
        env=env or ctx.script_env(),
        log=ctx.path("logs", f"{log}.log"),
        check=check,
    )
    if not check and proc.returncode != 0:
        ctx.log(f"{name} exited {proc.returncode}; continuing (partial results are expected)")
    return proc.stdout


def _pulse_env(ctx: RoundContext) -> dict[str, str]:
    env = ctx.script_env()
    key_file = ctx.config.data.key_file
    if not env.get("PULSE_API_KEY") and key_file:
        env["PULSE_API_KEY"] = key_file.read_text().strip()
    if not env.get("PULSE_API_KEY"):
        raise StepFailed("set PULSE_API_KEY or data.key_file to read the Pulse API")
    return env


def fetch(ctx: RoundContext) -> dict:
    """Enumerate recent Pulse sessions, take a census of their archives and fetch the token files."""
    root, env = ctx.round_dir, _pulse_env(ctx)
    census = root / "01_census"
    census.mkdir(parents=True, exist_ok=True)
    days = str(ctx.config.data.last_days)
    _script(
        ctx,
        "pulse.py",
        "enum",
        "--last-days",
        days,
        "--layers",
        "global",
        "--out",
        str(root / "00_enum_global"),
        log="fetch",
        env=env,
    )
    _script(ctx, "targets.py", log="fetch", env=env)
    _script(
        ctx,
        "pulse.py",
        "census",
        "--targets",
        str(root / "targets_all.txt"),
        "--out",
        str(census / "ls.jsonl.gz"),
        log="fetch",
        check=False,
        env=env,
    )
    _script(
        ctx,
        "pulse.py",
        "census-retry",
        "--census",
        str(census / "ls.jsonl.gz"),
        "--index",
        str(root / "00_enum_global/index_rows.jsonl"),
        "--out",
        str(census / "retry.jsonl.gz"),
        log="fetch",
        check=False,
        env=env,
    )
    _script(ctx, "build_plan.py", log="fetch", env=env)
    ctx.runner.run(
        ["bash", str(SCRIPTS_DIR / "fetch_all.sh")], cwd=SCRIPTS_DIR, env=env, log=ctx.path("logs", "fetch.log")
    )
    recent = (root / "targets_recent.txt").read_text().split()
    return {"recent_sessions": len(recent)}


def analyze(ctx: RoundContext) -> dict:
    """Token accounting, era split, idle-tick replay and specialist turn patterns over the bundles."""
    for name in ANALYSES[:3]:
        _script(ctx, name, log="analyze")
    if ctx.config.data.local_sessions:
        _script(ctx, "an_local.py", log="analyze")
    for name in ANALYSES[3:]:
        _script(ctx, name, log="analyze")
    analysis = ctx.round_dir / "analysis"
    return {"analysis_dir": str(analysis), "files": sorted(p.name for p in analysis.iterdir())}


def scenario_args(manifest: dict) -> list[str]:
    """``optimize`` arguments that reproduce a session's model, framework and workload."""
    work = manifest.get("workload") or {}
    try:
        pairs = (
            ("--model", manifest["model_path"]),
            ("--framework", manifest["framework"]),
            ("--gpu-type", manifest["gpu_type"]),
            ("--tp", manifest["tp"]),
            ("--conc", work["conc"]),
            ("--isl", work["isl"]),
            ("--osl", work["osl"]),
        )
    except KeyError as exc:
        raise StepFailed(f"the example session's manifest has no {exc}") from exc
    args = [str(item) for pair in pairs for item in pair]
    if int(manifest.get("ep") or 1) > 1:
        args += ["--ep", str(manifest["ep"])]
    objective = manifest.get("objective") or {}
    if objective.get("kind") == "gain_pct" and objective.get("value") is not None:
        args += ["--target-gain", str(objective["value"])]
    return args


def scenario(ctx: RoundContext) -> dict:
    """The A/B scenario: the configured file, else the top-scored local scenario's example session."""
    dest = ctx.path("scenario.json")
    if ctx.config.ab.scenario:
        spec = json.loads(ctx.config.ab.scenario.read_text())
        if not isinstance(spec.get("args"), list):
            raise StepFailed(f"{ctx.config.ab.scenario} must hold an 'args' list")
    else:
        if not ctx.config.data.models_dir or not ctx.config.data.local_sessions:
            raise StepFailed("set ab.scenario, or data.models_dir and data.local_sessions to pick one")
        _script(ctx, "select_scenario.py", log="scenario")
        scores = json.loads((ctx.round_dir / "analysis/scenario_scores.json").read_text())
        if not scores:
            raise StepFailed("no local scenario passed select_scenario.py's filters; set ab.scenario")
        top = scores[0]
        manifest = json.loads((Path(top["examples"][0]) / "manifest.json").read_text())
        spec = {"args": scenario_args(manifest), "source": top["examples"][0], "score": top["score"]}
    dest.write_text(json.dumps(spec, indent=1))
    return {"scenario": str(dest), "args": spec["args"]}


def replay(ctx: RoundContext) -> dict:
    """Run the configured offline replays of the landed levers against the candidate's code."""
    commands = ctx.config.replay_commands
    if not commands:
        return {"skipped": True, "reason": "no replay commands configured"}
    env = ctx.script_env(PYTHONPATH=str(ctx.worktree("candidate") / "src"))
    outputs = []
    for i, cmd in enumerate(commands):
        proc = ctx.runner.run(list(cmd), cwd=SCRIPTS_DIR, env=env, log=ctx.path("logs", "replay.log"))
        out = ctx.path("replay", f"{i:02d}.txt")
        out.write_text(proc.stdout)
        outputs.append(str(out))
    return {"outputs": outputs}
