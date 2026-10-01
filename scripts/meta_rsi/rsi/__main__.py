# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Command line for the rsi driver.

    python -m meta_rsi.rsi init   --config round.yaml
    python -m meta_rsi.rsi run    --config round.yaml [--until STEP | --only STEP ... | --from STEP] [--rerun]
    python -m meta_rsi.rsi status --config round.yaml

``run`` is safe to repeat: finished steps are skipped, and an A/B arm that is still running is
watched rather than relaunched. A full round takes hours (each arm runs for ab.hours), so start
it detached, e.g. under ``setsid nohup``.
"""

from __future__ import annotations

import argparse
import functools
import shutil
import sys
from pathlib import Path

from meta_rsi.rsi.agent import run_agent
from meta_rsi.rsi.config import ConfigError, RoundConfig, load_config
from meta_rsi.rsi.pipeline import RoundContext, run_pipeline, select_steps
from meta_rsi.rsi.state import RoundState, round_lock
from meta_rsi.rsi.steps import STEPS


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m meta_rsi.rsi", description="Run a Meta RSI round.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("init", "run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path, help="round configuration (YAML)")
    run = sub.choices["run"]
    run.add_argument("--until", default="", help="stop after this step")
    run.add_argument("--from", dest="start", default="", help="start at this step")
    run.add_argument("--only", nargs="+", default=[], help="run just these steps")
    run.add_argument("--rerun", action="store_true", help="run selected steps even if they finished")
    return ap


def _context(config: RoundConfig) -> RoundContext:
    ledger = config.round_dir / "agent_calls.jsonl"
    return RoundContext(
        config=config, state=RoundState.load(config.round_dir), agent=functools.partial(run_agent, ledger=ledger)
    )


def _status(config: RoundConfig) -> None:
    state = RoundState.load(config.round_dir)
    for step in STEPS:
        rec = state.step(step.name)
        detail = rec.error.splitlines()[0] if rec.error else ""
        print(f"{step.name:10s} {step.kind:6s} {rec.status:8s} {rec.finished or rec.started:20s} {detail}")
    for name, arm in state.data.get("ab", {}).items():
        print(f"arm {name}: {arm.get('status')} (attempt {arm.get('attempt')}, pid {arm.get('pid')})")
    print(f"agent spend: ${state.agent_cost_usd:.2f} of ${config.agent.total_budget_usd:g}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    if args.command == "status":
        _status(config)
        return 0
    with round_lock(config.round_dir):
        if args.command == "init":
            shutil.copyfile(args.config, config.round_dir / "round.yaml")
            RoundState.load(config.round_dir).save()
            print(f"round initialized in {config.round_dir}")
            return 0
        steps = select_steps(STEPS, until=args.until, only=args.only, start=args.start)
        return 0 if run_pipeline(_context(config), steps, rerun=args.rerun) else 1


if __name__ == "__main__":
    sys.exit(main())
