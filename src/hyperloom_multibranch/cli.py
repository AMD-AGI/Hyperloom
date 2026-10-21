# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and the kernel
optimizer its recipes are scoped to, and the hard limits the judge works within. ``--workload`` names the workload
kind: ``inferencex`` takes ``--isl`` and ``--osl``, and ``agentx`` refuses them, as the trace it replays fixes its
requests; an operator environment whose ``HYPERLOOM_AGENTX`` says otherwise than ``--workload``, or that names a
benchmark YAML in ``HYPERLOOM_BENCHMARK_CONFIG``, is refused.
``--max-gpu-hours`` caps the GPU hours the session's tests use: EXPLORE ends with ``gpu_hours_exhausted`` once only
FINAL's share is left, and FINAL still runs. ``--max-llm-usd`` caps the dollars the session's LLM calls cost: EXPLORE
ends with ``llm_budget_exhausted`` once their recorded spend reaches it, and FINAL still runs. Each call is priced from
the price sheet, cached tokens at cache rates: a specialist launch's calls from its log when the launch ends, a launch
cut short, lost or stopped included, so the stop can come up to one launch late, and each call of the scheduler's own
judge, patch resolver and the rule gate's fix critic as it returns. A cap is a usage error when the price sheet has no
rate for the model the specialists run on or the one the judge, the patch resolver and the fix critic ask, whose calls
would record tokens only. ``--reuse-prelude DIR`` roots the session on the child session the PRELUDE of the
earlier session in ``DIR`` rooted on instead of starting a child, and still measures the baseline; it is a usage error
unless that session committed its root on the same workload and on the image ``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` names.
``--frontier-policy`` says which frontier policy the session scores its branches with once PRELUDE established the
baseline: ``auto`` (the default) takes the variation the calibration over similar archived sessions chooses, within
its guardrails, and ``default`` keeps the defaults; the manifest fingerprint and the session's manifest record
the label under ``frontier`` in ``policy_versions``. ``--frontier-scorer LABEL`` pins the scorer the session scores
with instead: ``frontier-v1``, a variation's label, a scorer in code or an expression in the archive's scorer pool; a
label none of these has, or one given beside ``--frontier-policy default``, is a usage error.
Each ``--specialist-prompt`` file holds a direction the user wrote for one specialist: its branch opens at PRELUDE's
root ahead of the opening batch, counted within ``--max-branches``, its specialist sees the text in every launch
prompt, and the judge neither stops it nor lowers its K for its first judged tests; an empty or unreadable file, or
more files than ``--max-branches``, is a usage error. A session whose toolbox has fewer
usable cards left than one test needs ends EXPLORE with ``no_measurement_cards`` and skips FINAL. Exit codes: 0 when
the session ran its course, its wall-clock, GPU-hour or LLM budget spent included, 1 when it failed or was stopped,
``no_measurement_cards`` included, 2 on a usage error. The environment and the peers a session needs are checked before
anything is created, so a missing one fails the command without leaving a session directory behind: the Recipe KB that
PRELUDE reads warm-start leads from and CLOSE writes the final recipe back to is configured by ``KB_STORE_URL`` and
``KB_STORE_TOKEN``, both or neither, and is not used without them; the RSI archive defaults to
:data:`RSI_ARCHIVE_DEFAULT`, set in this process's environment for every reader of it, and must be a writable
directory; the measurement pins must be readable, the image the session runs on named by both
``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` included; the framework's package must be
installed in the image; and the agent CLI the specialists run must be installed.
A session uses no Experience KB, whatever ``HYPERLOOM_KB_URL`` names (see
:data:`~hyperloom_multibranch.scheduling.mb_fakes.MB_FAKE_EXPERIENCE_KB`).
``final-only SESSION_DIR`` is the emergency recovery of a session that ended before FINAL ran, and takes nothing but
that directory: it runs FINAL and CLOSE over what the session persisted -- the settings and the frontier policy its
manifest holds, the branches its roster and notebook hold -- and rewrites its report, dashboard and breakdown. It
creates no session directory, so it cannot start one. It carries the stop reason that session ended with, which is
also what its exit code says: a session that did not run its course stays a failed session however FINAL goes. A
session that was killed with its machine wrote no report at all; it ends for ``killed``, and so fails too. The
search tree the session's own CLOSE archived is left as it was. It is refused with exit code 2, having written
nothing, when the directory is no session that ran, ran on another build than this one, committed no root, is still
running -- any process ``MB-pids.jsonl`` records, the scheduler included, still alive in this environment -- already
chose in FINAL, or has another ``final-only`` running over it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from functools import partial
from pathlib import Path

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path
from hyperloom.common.env import EnvValueError
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom.measure.pins import IMAGE_DIGEST_VAR
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient, RemoteRecipeConfigurationError
from hyperloom_multibranch.contracts.llm_prices import price_of
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.notebook.rsi.frontier import FrontierPolicy
from hyperloom_multibranch.rulegate.critic import LlmCritic, anthropic_critic
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS, MB_MIN_PRELUDE_HOURS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.layout import create_session
from hyperloom_multibranch.scheduling.lifecycle import (
    PersistedRun,
    Session,
    SessionHost,
    SessionServices,
    persisted_run,
    pinned_frontier_choice,
)
from hyperloom_multibranch.scheduling.llm import anthropic_completion, default_model
from hyperloom_multibranch.scheduling.local_peers import (
    AGENTX_ENV,
    load_peers,
    machine_parallel_tests,
    specialist_model,
)
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeersUnavailable
from hyperloom_multibranch.scheduling.prelude import reused_prelude
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, adopting_orphans
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    FrontierPolicyMode,
    InferenceXShape,
    KernelOptimizer,
    RunSettings,
    Workload,
)

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
    _add_run_options(commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE."))
    final_only = commands.add_parser(
        "final-only",
        help=(
            "Emergency recovery: run FINAL and CLOSE over a session directory that ended before FINAL, a session"
            " killed with its machine included. It rewrites that session's report, dashboard and breakdown, and"
            " exits as that session did, so a session that did not run its course still fails. Refused if the"
            " session is still running, already chose in FINAL, or ran on another build than this one."
        ),
    )
    final_only.add_argument("session_dir", type=Path, help="The session directory of the run to finish.")
    final_only.add_argument("-v", "--verbose", action="store_true", help="Log at debug level.")
    return parser


def _add_run_options(run: argparse.ArgumentParser) -> None:
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
        help=(
            "Dollars the session's recorded LLM calls may cost; EXPLORE ends there; 0 = none."
            " Refused when the price sheet lists no rate for the judge's or the specialists' model."
        ),
    )
    run.add_argument(
        "--max-branches", type=int, default=DEFAULT_MAX_BRANCHES, help="Specialist branches not stopped, at most."
    )
    run.add_argument(
        "--specialist-prompt",
        dest="specialist_prompts",
        action="append",
        type=Path,
        default=None,
        metavar="FILE",
        help="A direction for one specialist; repeatable, each opens one branch at the root within --max-branches.",
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
    run.add_argument(
        "--reuse-prelude",
        type=Path,
        default=None,
        metavar="DIR",
        help="An earlier session of this workload and image whose PRELUDE child to root on instead of starting one.",
    )
    run.add_argument(
        "--frontier-policy",
        choices=[mode.value for mode in FrontierPolicyMode],
        default=FrontierPolicyMode.AUTO.value,
        help=(
            "The frontier policy the session scores with after PRELUDE: auto takes the calibration's choice for similar"
            " archived sessions, default keeps the defaults."
        ),
    )
    run.add_argument(
        "--frontier-scorer",
        default="",
        metavar="LABEL",
        help="The frontier scorer the session scores with, instead of the one the calibration chooses.",
    )
    run.add_argument("-v", "--verbose", action="store_true", help="Log at debug level.")


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


def _user_directions(files: list[Path], max_branches: int) -> tuple[str, ...]:
    if len(files) > max_branches:
        raise ValueError(f"--specialist-prompt is given {len(files)} times, more than --max-branches {max_branches}")
    directions = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"--specialist-prompt {path} cannot be read: {exc}") from exc
        if not text:
            raise ValueError(f"--specialist-prompt {path} is empty")
        directions.append(text)
    return tuple(directions)


def settings_from(args: argparse.Namespace) -> RunSettings:
    """The run's settings; raises ValueError naming the invalid options. :class:`RosterCaps` checks the caps."""

    errors = _shape_errors(args) + _positive(args, ("tp", "conc", "judge_interval_sec"))
    if args.max_hours < 0 or 0 < args.max_hours < MB_MIN_PRELUDE_HOURS:
        errors.append(f"--max-hours must be 0 (no limit) or at least {MB_MIN_PRELUDE_HOURS:.4f} (one minute)")
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours < 0:
        errors.append("--max-gpu-hours must be 0 (no limit) or a positive number of hours")
    if not math.isfinite(args.max_llm_usd) or args.max_llm_usd < 0:
        errors.append("--max-llm-usd must be 0 (no limit) or a positive number of dollars")
    if args.frontier_scorer and FrontierPolicyMode(args.frontier_policy) is FrontierPolicyMode.DEFAULT:
        errors.append("--frontier-scorer pins a scorer, which --frontier-policy default rules out")
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
    return RunSettings(
        workload,
        args.max_hours,
        args.max_gpu_hours,
        args.max_llm_usd,
        caps,
        args.judge_interval_sec,
        user_directions=_user_directions(args.specialist_prompts or [], caps.max_branches),
        frontier_policy=FrontierPolicyMode(args.frontier_policy),
        frontier_scorer=args.frontier_scorer,
    )


def require_frontier_scorer(label: str, metric: str) -> None:
    """Raise ValueError unless ``label`` names a scorer a session of ``metric`` can score with; the scorer pool is
    read beside the archive ``HYPERLOOM_RSI_ARCHIVE`` names."""

    pinned_frontier_choice(label, FrontierPolicy(metric))


def require_workload_switch(workload: Workload, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``env`` sets ``HYPERLOOM_AGENTX`` to anything but what ``workload``'s kind says, or names
    a benchmark YAML in ``HYPERLOOM_BENCHMARK_CONFIG``; the session sets both for an AgentX workload itself, so an
    operator who set them means a workload ``--workload`` and its flags do not name."""

    if benchmark_config_path(env) is not None:
        raise ValueError(f"{BENCHMARK_CONFIG_ENV} names a workload YAML; --workload and its flags define the workload")
    if AGENTX_ENV not in env:
        return
    try:
        agentx = agentx_enabled(env)
    except EnvValueError as exc:
        raise ValueError(str(exc)) from exc
    if agentx != (workload.kind is WorkloadKind.AGENTX):
        raise ValueError(f"{AGENTX_ENV}={env[AGENTX_ENV]!r} contradicts --workload {workload.kind.value}")


def require_priced_models(max_llm_usd: float, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``max_llm_usd`` caps the session's LLM dollars and the price sheet lists no rate for the
    model the judge, the patch resolver and the fix critic ask or the one the specialists of ``env`` run on: their
    calls would record tokens only, which the cap cannot count."""

    if not max_llm_usd:
        return
    models = {"the judge's": default_model(), "the specialists'": specialist_model(env)}
    unpriced = [f"{whose} model {model}" for whose, model in models.items() if price_of(model) is None]
    if unpriced:
        raise ValueError(
            f"--max-llm-usd cannot count the dollars of {' or '.join(unpriced)}, which the price sheet does not list;"
            " run on a listed model or without a dollar cap"
        )


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


def session_services(recipe_kb: RemoteRecipeClient | None) -> SessionServices:
    """The judge, the patch resolver, the fix critic and CLOSE's scorer author on the default model, and
    ``recipe_kb``."""

    model = default_model()
    judge = anthropic_completion(model, component=PROG, operation="judge")
    resolver = anthropic_completion(model, component=PROG, operation="patch_resolve")
    critic = partial(LlmCritic, anthropic_critic(model, component=PROG))
    scorer = anthropic_completion(model, component=PROG, operation="author_scorers")
    return SessionServices(judge, resolver, recipe_kb, critic, scorer, model)


def run_session(settings: RunSettings, peer_factory: PeerFactory, services: SessionServices) -> int:
    """Create the session directory, print it on one stdout line, run the session and return its exit code.

    The session and its peers are built on the event loop they run on, which the specialist tools of the MCP service
    hand every toolbox call to. Whatever the session's processes orphan is reparented to this process, so the session
    stop kills it and reports it while it runs.
    """

    with adopting_orphans():
        return asyncio.run(_run_session(settings, peer_factory, services))


async def _run_session(settings: RunSettings, peer_factory: PeerFactory, services: SessionServices) -> int:
    clock = Clock()
    layout = create_session(workspace_root(), settings.workload.model_basename, clock)
    print(layout.root, flush=True)
    logging.getLogger().addHandler(_file_handler(layout.scheduler_log))
    host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
    session = Session(settings, peer_factory, host, services)
    return (await session.run()).exit_code


def final_only(persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices) -> int:
    """Run FINAL and CLOSE over the session directory ``persisted`` was read from, print it on one stdout line and
    return the session's exit code. No session directory is created, which is why this cannot start one.

    The run holds the session's lock file while it works; a second one over the same directory is refused rather
    than left to write over the first one's report.
    """

    lock = persisted.layout.final_only_lock
    try:
        holder = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"{PROG}: {lock} is held, by the pid it names; delete it if that run is gone", file=sys.stderr)
        return EXIT_USAGE
    os.write(holder, f"{os.getpid()}\n".encode())
    os.close(holder)
    try:
        with adopting_orphans():
            return asyncio.run(_final_only(persisted, peer_factory, services))
    finally:
        lock.unlink()


async def _final_only(persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices) -> int:
    clock = Clock()
    layout = persisted.layout
    print(layout.root, flush=True)
    logging.getLogger().addHandler(_file_handler(layout.scheduler_log))
    host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
    session = Session(persisted.settings, peer_factory, host, services)
    return (await session.finish(persisted.ended)).exit_code


def _file_handler(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _reused_image_digest(env: Mapping[str, str]) -> str:
    """The image digest an earlier PRELUDE must have run on; raises ValueError when the environment names none."""

    digest = env.get(IMAGE_DIGEST_VAR)
    if not digest:
        raise ValueError(
            f"--reuse-prelude compares the earlier session's image with {IMAGE_DIGEST_VAR}, which is not set"
        )
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    persisted = None
    try:
        if args.command == "final-only":
            persisted = persisted_run(args.session_dir)
            settings = persisted.settings
        else:
            settings = settings_from(args)
            if args.reuse_prelude is not None:
                reused = reused_prelude(args.reuse_prelude, settings.workload, _reused_image_digest(os.environ))
                settings = replace(settings, reuse_prelude=reused)
        require_workload_switch(settings.workload, os.environ)
        require_priced_models(settings.max_llm_usd, os.environ)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE) from exc
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        recipe_kb = recipe_kb_from_env()
        require_rsi_archive(os.environ)
        peer_factory = load_peers(settings.workload)
        services = session_services(recipe_kb)
    except (EnvironmentIncomplete, PeersUnavailable) as exc:
        print(f"{PROG}: cannot run a session, {exc}", file=sys.stderr)
        return EXIT_FAILED
    if settings.frontier_scorer:
        try:
            require_frontier_scorer(settings.frontier_scorer, settings.workload.metric)
        except ValueError as exc:
            parser.print_usage(sys.stderr)
            print(f"{parser.prog}: error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE) from exc
    if persisted is not None:
        return final_only(persisted, peer_factory, services)
    return run_session(settings, peer_factory, services)


__all__ = [
    "MAGPIE_RUNNER_GPU_TYPES",
    "PROG",
    "RSI_ARCHIVE_DEFAULT",
    "SEQUENCE_LENGTHS",
    "EnvironmentIncomplete",
    "build_parser",
    "final_only",
    "main",
    "recipe_kb_from_env",
    "require_frontier_scorer",
    "require_priced_models",
    "require_rsi_archive",
    "require_workload_switch",
    "run_session",
    "session_services",
    "settings_from",
]
