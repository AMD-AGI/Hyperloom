# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and the kernel
optimizer its recipes are scoped to, and the hard limits the judge works within. ``--workload`` names the workload
kind: ``inferencex`` takes ``--isl`` and ``--osl``, and ``agentx`` refuses them, as the trace it replays fixes its
requests; an operator environment whose ``HYPERLOOM_AGENTX`` says otherwise than ``--workload`` is refused.
``--max-gpu-hours`` caps the GPU hours the session's tests use: EXPLORE ends with ``gpu_hours_exhausted`` once only
FINAL's share is left, and FINAL still runs. ``--max-llm-usd`` caps the dollars the specialists' LLM launches cost:
EXPLORE ends with ``llm_budget_exhausted`` once their recorded spend reaches it, and FINAL still runs. That spend is a
lower bound: a launch counts once it ends and its CLI reported its cost, so a launch cut short, lost or stopped, a CLI
that reports no cost and the scheduler's own judge and resolver are not in it, and the stop can come up to one launch
late. A session whose toolbox has fewer usable cards left than one test needs ends EXPLORE with ``no_measurement_cards``
and skips FINAL. Exit codes: 0 when the session ran its course, its wall-clock, GPU-hour or LLM budget spent included, 1
when it failed or was stopped, ``no_measurement_cards`` included, 2 on a usage error. The environment and the peers a
session needs are checked before anything is created, so a missing one fails the command without leaving a session
directory behind: the Recipe KB that PRELUDE reads warm-start leads from and CLOSE writes the final recipe back to is
configured by ``KB_STORE_URL`` and ``KB_STORE_TOKEN``, both or neither, and is not used without them; the RSI archive
defaults to :data:`RSI_ARCHIVE_DEFAULT`, set in this process's environment for every reader of it, and must be a
writable directory; the measurement pins must be readable; the framework's package must be installed in the image; and
the agent CLI the specialists run must be installed.
A session uses no Experience KB, whatever ``HYPERLOOM_KB_URL`` names (see
:data:`~hyperloom_multibranch.scheduling.mb_fakes.MB_FAKE_EXPERIENCE_KB`).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from hyperloom.common.env import EnvValueError
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient, RemoteRecipeConfigurationError
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.rulegate.critic import LlmCritic, anthropic_critic
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS, MB_MIN_PRELUDE_HOURS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.layout import create_session
from hyperloom_multibranch.scheduling.lifecycle import Session, SessionHost, SessionServices
from hyperloom_multibranch.scheduling.llm import anthropic_completion, default_model
from hyperloom_multibranch.scheduling.local_peers import AGENTX_ENV, load_peers, machine_parallel_tests
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, RunSettings, Workload

PROG = "hyperloom-multibranch"
EXIT_FAILED = 1
EXIT_USAGE = 2
DEFAULT_MAX_BRANCHES = 4
DEFAULT_MAX_K_PER_BRANCH = 2
DEFAULT_JUDGE_INTERVAL_SECONDS = 600.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
#: The shared folder of archived search trees, used when ``HYPERLOOM_RSI_ARCHIVE`` names none.
RSI_ARCHIVE_DEFAULT = "/shared_nfs/hyperloom/rsi-archive/"
#: The GPU types Magpie has a benchmark script of their own for. ``hyperloom optimize`` runs mi308x and mi325x on
#: mi300x's script through a mapping its package keeps private, so a session accepts only these until it is public.
MAGPIE_RUNNER_GPU_TYPES = ("mi300x", "mi355x")
#: The options an inferencex workload requires and an agentx one refuses.
SEQUENCE_LENGTHS = ("isl", "osl")


class EnvironmentIncomplete(RuntimeError):
    """The environment lacks what a session needs; the message names the variables to set."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE.")
    run.add_argument("--model", required=True, help="Model path or id.")
    run.add_argument("--framework", required=True, help="Inference framework, e.g. vllm or sglang.")
    run.add_argument("--gpu-type", required=True, choices=MAGPIE_RUNNER_GPU_TYPES, help="GPU type.")
    run.add_argument("--workload", required=True, choices=[kind.value for kind in WorkloadKind], help="Workload kind.")
    run.add_argument("--tp", type=int, required=True, help="Tensor-parallel size; must fit the machine's cards.")
    run.add_argument("--conc", type=int, required=True, help="Concurrency of the primary operating point.")
    for name in SEQUENCE_LENGTHS:
        run.add_argument(f"--{name}", type=int, default=None, help="Required for inferencex; refused for agentx.")
    run.add_argument("--precision", required=True, help="Model precision, as hyperloom optimize spells it.")
    run.add_argument(
        "--kernel-optimizer",
        choices=[optimizer.value for optimizer in KernelOptimizer],
        default=KernelOptimizer.FORGE.value,
        help="Kernel optimizer the session's recipes are scoped to.",
    )
    run.add_argument(
        "--max-hours", type=float, default=MB_DEFAULT_MAX_HOURS, help="Wall-clock budget, PRELUDE included; 0 = none."
    )
    run.add_argument(
        "--max-gpu-hours",
        type=float,
        default=0.0,
        help="GPU hours the session's tests may use; EXPLORE ends when FINAL's share is all that is left; 0 = none.",
    )
    run.add_argument(
        "--max-llm-usd",
        type=float,
        default=0.0,
        help="Dollars the specialists' recorded LLM launches may cost; EXPLORE ends when they reach it; 0 = none.",
    )
    run.add_argument(
        "--max-branches", type=int, default=DEFAULT_MAX_BRANCHES, help="Specialist branches not stopped, at most."
    )
    run.add_argument(
        "--max-parallel-tests",
        type=int,
        default=None,
        help="Tests running at once; default as many as the machine's cards run at TP.",
    )
    run.add_argument("--max-k-per-branch", type=int, default=DEFAULT_MAX_K_PER_BRANCH, help="Highest K of a branch.")
    run.add_argument(
        "--judge-interval-sec", type=float, default=DEFAULT_JUDGE_INTERVAL_SECONDS, help="Seconds between judge calls."
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Log at debug level.")
    return parser


def _positive(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    return [f"--{name.replace('_', '-')} must be positive" for name in names if getattr(args, name) <= 0]


def _shape_errors(args: argparse.Namespace) -> list[str]:
    given = [name for name in SEQUENCE_LENGTHS if getattr(args, name) is not None]
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return [f"--{name} is not taken by an agentx workload, whose trace fixes its requests" for name in given]
    missing = [f"--{name} is required for an inferencex workload" for name in SEQUENCE_LENGTHS if name not in given]
    return missing or _positive(args, SEQUENCE_LENGTHS)


def _workload_shape(args: argparse.Namespace) -> InferenceXShape | None:
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return None
    return InferenceXShape(args.isl, args.osl)


def settings_from(args: argparse.Namespace) -> RunSettings:
    """The run's settings; raises ValueError naming the invalid options. :class:`RosterCaps` checks the caps."""

    errors = _shape_errors(args) + _positive(args, ("tp", "conc", "judge_interval_sec"))
    if args.max_hours < 0 or 0 < args.max_hours < MB_MIN_PRELUDE_HOURS:
        errors.append(f"--max-hours must be 0 (no limit) or at least {MB_MIN_PRELUDE_HOURS:.4f} (one minute)")
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours < 0:
        errors.append("--max-gpu-hours must be 0 (no limit) or a positive number of hours")
    if not math.isfinite(args.max_llm_usd) or args.max_llm_usd < 0:
        errors.append("--max-llm-usd must be 0 (no limit) or a positive number of dollars")
    if errors:
        raise ValueError("; ".join(errors))
    try:
        fitting = machine_parallel_tests(args.tp)
    except ValueError as exc:
        raise ValueError(f"--tp does not fit the machine: {exc}") from exc
    parallel = fitting if args.max_parallel_tests is None else args.max_parallel_tests
    if parallel > fitting:
        raise ValueError(f"--max-parallel-tests must be at most {fitting}, the tests the machine's cards run at TP")
    workload = Workload(
        args.model,
        args.framework,
        args.tp,
        args.conc,
        args.precision,
        args.gpu_type,
        KernelOptimizer(args.kernel_optimizer),
        _workload_shape(args),
    )
    caps = RosterCaps(args.max_branches, args.max_k_per_branch, parallel)
    return RunSettings(workload, args.max_hours, args.max_gpu_hours, args.max_llm_usd, caps, args.judge_interval_sec)


def require_workload_switch(workload: Workload, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``env`` sets ``HYPERLOOM_AGENTX`` to anything but what ``workload``'s kind says; the
    session sets it for an AgentX workload itself, so an operator who set it means a workload ``--workload`` does not
    name."""

    if AGENTX_ENV not in env:
        return
    try:
        agentx = agentx_enabled(env)
    except EnvValueError as exc:
        raise ValueError(str(exc)) from exc
    if agentx != (workload.kind is WorkloadKind.AGENTX):
        raise ValueError(f"{AGENTX_ENV}={env[AGENTX_ENV]!r} contradicts --workload {workload.kind.value}")


def recipe_kb_from_env() -> RemoteRecipeClient | None:
    """The Recipe KB client ``KB_STORE_URL`` and ``KB_STORE_TOKEN`` configure, ``None`` when neither is set; raises
    :class:`EnvironmentIncomplete` when only one is."""

    try:
        return RemoteRecipeClient.from_env_optional()
    except RemoteRecipeConfigurationError as exc:
        raise EnvironmentIncomplete(f"the Recipe KB is half configured: {exc}") from exc


def require_rsi_archive(env: MutableMapping[str, str]) -> None:
    """Point ``HYPERLOOM_RSI_ARCHIVE`` at :data:`RSI_ARCHIVE_DEFAULT` when ``env`` names no archive; raise
    :class:`EnvironmentIncomplete` unless the archive is a writable directory."""

    if not env.get(ARCHIVE_ENV, "").strip():
        env[ARCHIVE_ENV] = RSI_ARCHIVE_DEFAULT
    archive = Path(env[ARCHIVE_ENV].strip())
    if not (archive.is_dir() and os.access(archive, os.W_OK)):
        raise EnvironmentIncomplete(
            f"the RSI archive {archive} is not a writable directory: mount it, or set {ARCHIVE_ENV} to one"
        )


def run_session(settings: RunSettings, peer_factory: PeerFactory, recipe_kb: RemoteRecipeClient | None) -> int:
    """Create the session directory, print it on one stdout line, run the session and return its exit code.

    The model is resolved first, so an environment that cannot provide it fails before any session directory exists.
    The session and its peers are built on the event loop they run on, which the specialist tools of the MCP service
    hand every toolbox call to.
    """

    return asyncio.run(_run_session(settings, peer_factory, recipe_kb))


async def _run_session(settings: RunSettings, peer_factory: PeerFactory, recipe_kb: RemoteRecipeClient | None) -> int:
    clock = Clock()
    model = default_model()
    judge = anthropic_completion(model, component=PROG, operation="judge")
    resolver = anthropic_completion(model, component=PROG, operation="patch_resolve")
    services = SessionServices(judge, resolver, recipe_kb, LlmCritic(anthropic_critic(model, component=PROG)))
    layout = create_session(workspace_root(), settings.workload.model_basename, clock)
    print(layout.root, flush=True)
    logging.getLogger().addHandler(_file_handler(layout.scheduler_log))
    host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
    session = Session(settings, peer_factory, host, services)
    return (await session.run()).exit_code


def _file_handler(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = settings_from(args)
        require_workload_switch(settings.workload, os.environ)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE) from exc
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        recipe_kb = recipe_kb_from_env()
        require_rsi_archive(os.environ)
        peer_factory = load_peers(settings.workload)
    except (EnvironmentIncomplete, PeersUnavailable) as exc:
        print(f"{PROG}: cannot run a session, {exc}", file=sys.stderr)
        return EXIT_FAILED
    return run_session(settings, peer_factory, recipe_kb)


__all__ = [
    "MAGPIE_RUNNER_GPU_TYPES",
    "PROG",
    "RSI_ARCHIVE_DEFAULT",
    "SEQUENCE_LENGTHS",
    "EnvironmentIncomplete",
    "build_parser",
    "main",
    "recipe_kb_from_env",
    "require_rsi_archive",
    "require_workload_switch",
    "run_session",
    "settings_from",
]
