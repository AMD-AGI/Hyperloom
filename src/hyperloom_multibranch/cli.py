# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed InferenceX workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and
the hard limits the judge works within. Exit codes: 0 when the session ran its course, 1 when it failed or was
stopped, 2 on a usage error. The peers a session needs are loaded before anything is created, so a missing one
fails the command without leaving a session directory behind.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS, MB_MIN_PRELUDE_HOURS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.exports import exports_from_env
from hyperloom_multibranch.scheduling.layout import create_session
from hyperloom_multibranch.scheduling.lifecycle import Session
from hyperloom_multibranch.scheduling.llm import anthropic_completion, default_model
from hyperloom_multibranch.scheduling.local_peers import GPUS_PER_MACHINE, load_peers
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload

PROG = "hyperloom-multibranch"
EXIT_FAILED = 1
EXIT_USAGE = 2
DEFAULT_MAX_BRANCHES = 4
DEFAULT_MAX_K_PER_BRANCH = 2
DEFAULT_JUDGE_INTERVAL_SECONDS = 600.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE.")
    run.add_argument("--model", required=True, help="Model path or id.")
    run.add_argument("--framework", required=True, help="Inference framework, e.g. vllm or sglang.")
    run.add_argument("--gpu-type", default="", help="GPU type; detected by the baseline prelude when left out.")
    for name in ("--tp", "--isl", "--osl", "--conc"):
        run.add_argument(name, type=int, required=True)
    run.add_argument("--precision", required=True)
    run.add_argument(
        "--max-hours", type=float, default=MB_DEFAULT_MAX_HOURS, help="Wall-clock budget, PRELUDE included; 0 = none."
    )
    run.add_argument(
        "--max-branches", type=int, default=DEFAULT_MAX_BRANCHES, help="Specialist branches not stopped, at most."
    )
    run.add_argument(
        "--max-parallel-tests", type=int, default=None, help=f"Tests running at once; default {GPUS_PER_MACHINE} // TP."
    )
    run.add_argument("--max-k-per-branch", type=int, default=DEFAULT_MAX_K_PER_BRANCH, help="Highest K of a branch.")
    run.add_argument(
        "--judge-interval-sec", type=float, default=DEFAULT_JUDGE_INTERVAL_SECONDS, help="Seconds between judge calls."
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Log at debug level.")
    return parser


def _positive(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    return [f"--{name.replace('_', '-')} must be positive" for name in names if getattr(args, name) <= 0]


def settings_from(args: argparse.Namespace) -> RunSettings:
    """The run's settings; raises ValueError naming the invalid options. :class:`RosterCaps` checks the caps."""

    errors = _positive(args, ("tp", "isl", "osl", "conc", "judge_interval_sec"))
    if args.max_hours < 0 or 0 < args.max_hours < MB_MIN_PRELUDE_HOURS:
        errors.append(f"--max-hours must be 0 (no limit) or at least {MB_MIN_PRELUDE_HOURS:.4f} (one minute)")
    if errors:
        raise ValueError("; ".join(errors))
    parallel = max(GPUS_PER_MACHINE // args.tp, 1) if args.max_parallel_tests is None else args.max_parallel_tests
    workload = Workload(
        args.model, args.framework, args.tp, args.isl, args.osl, args.conc, args.precision, args.gpu_type
    )
    caps = RosterCaps(args.max_branches, args.max_k_per_branch, parallel)
    return RunSettings(workload, args.max_hours, caps, args.judge_interval_sec)


def run_session(settings: RunSettings, peer_factory: PeerFactory) -> int:
    """Create the session directory, print it on one stdout line, run the session and return its exit code.

    The model and the Recipe KB client are resolved first, so an environment that cannot provide them fails before
    any session directory exists.
    """

    clock = Clock()
    model = default_model()
    judge = anthropic_completion(model, component=PROG, operation="judge")
    resolver = anthropic_completion(model, component=PROG, operation="patch_resolve")
    exports = exports_from_env()
    layout = create_session(workspace_root(), settings.workload.model_basename, clock)
    print(layout.root, flush=True)
    logging.getLogger().addHandler(_file_handler(layout.scheduler_log))
    session = Session(
        settings,
        layout,
        peer_factory,
        ProcessRegistry(layout.pids, clock),
        judge_completion=judge,
        resolver_completion=resolver,
        exports=exports,
        clock=clock,
    )
    return asyncio.run(session.run()).exit_code


def _file_handler(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from(args)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        peer_factory = load_peers()
    except PeersUnavailable as exc:
        print(f"{PROG}: cannot run a session, {exc}", file=sys.stderr)
        return EXIT_FAILED
    return run_session(settings, peer_factory)


__all__ = ["PROG", "build_parser", "main", "run_session", "settings_from"]
