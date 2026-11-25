# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The argparse parser of ``hyperloom-multibranch`` and the run settings the options of ``run`` make."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections.abc import Callable
from pathlib import Path

from hyperloom.common.perf_metric import DATASET_PATH_ENV
from hyperloom.common.url_userinfo import CREDENTIALED_URL_RE
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom.inference_optimizer.framework_registry import native_component
from hyperloom_multibranch.blackboard.knowledge.reference_leads import INFERENCEX_REFERENCE
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.scheduling import review
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.local_peers import machine_parallel_tests
from hyperloom_multibranch.scheduling.reference import ReferenceSource, reference_source
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    AGENT_PORT,
    DEFAULT_AGENTX_OBJECTIVE,
    DOSSIER_MINUTES,
    MAX_BOOT_BUDGET_SEC,
    SHARE_MINUTES,
    InferenceXShape,
    KernelOptimizer,
    PlaybookMode,
    RunSettings,
    Workload,
    session_early_stop,
)
from hyperloom_multibranch.specialist.component_agent import AgentRefused, dossier_lacks, read_launch
from inference_suite import UserFileError, parse_inferencex_commit, parse_user_file
from inference_suite.contracts import BaseLaunchError

PROG = "hyperloom-multibranch"
MAX_PORT = 65535
DEFAULT_MAX_BRANCHES = 4
DEFAULT_MAX_K_PER_BRANCH = 2
DEFAULT_JUDGE_INTERVAL_SECONDS = 600.0
ENCODING = "utf-8"
FILE_METAVAR = "FILE"
#: The GPU types a session runs on; the toolbox's environment exports the board itself as ``GPU_TYPE``.
SESSION_GPU_TYPES = ("mi300x", "mi325x", "mi350x", "mi355x", "r9700", "w7900d")
#: Those of them a session runs on only when its framework builds its server from a checkout.
NATIVE_SERVER_GPU_TYPES = ("r9700", "w7900d")
#: The options ``run`` requires unless it is given ``--resume``, by their argparse names.
RUN_REQUIRED = ("model", "framework", "gpu_type", "workload", "tp", "conc", "precision")
#: The options an inferencex workload requires and an agentx one refuses.
SEQUENCE_LENGTHS = ("isl", "osl")
#: The argparse action of an option the user may give more than once, each giving one more value.
REPEATABLE = "append"
#: The argparse action of a switch, an option that takes no value and is on when given.
SWITCH = "store_true"
#: Where ``run`` measures: this machine's cards, or the sandboxes of a grant handed to it.
LOCAL_MEASUREMENT, HOSTED_MEASUREMENT = "local", "hosted"
#: The command that runs FINAL and CLOSE alone over a session directory.
FINAL_ONLY_COMMAND = "final-only"
#: The command that runs the concurrency sweep of a closed session.
CONC_SWEEP_COMMAND = "conc-sweep"
#: The command a host runs a session under, with the campaign spec it injects.
HOSTED_COMMAND = "hosted"


class _CommandParser(argparse.ArgumentParser):
    """A command's parser, which refuses a board of :data:`NATIVE_SERVER_GPU_TYPES` for a framework that builds no
    server, in the words argparse refuses a ``--gpu-type`` that is no choice."""

    def parse_known_args(self, args=None, namespace=None):  # type: ignore[no-untyped-def]
        parsed, rest = super().parse_known_args(args, namespace)
        board, framework = getattr(parsed, "gpu_type", None), getattr(parsed, "framework", None)
        if board in NATIVE_SERVER_GPU_TYPES and native_component(framework or "") is None:
            action = copy.copy(next(action for action in self._actions if action.dest == "gpu_type"))
            action.choices = [choice for choice in SESSION_GPU_TYPES if choice not in NATIVE_SERVER_GPU_TYPES]
            try:
                self._check_value(action, board)
            except argparse.ArgumentError as exc:
                self.error(str(exc))
        return parsed, rest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_CommandParser)
    _add_run_options(commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE."))
    review.add_parser(commands)
    final_only = commands.add_parser(
        FINAL_ONLY_COMMAND,
        help=(
            "Emergency recovery: run FINAL and CLOSE over a session directory that ended before FINAL, a session"
            " killed with its machine included. It rewrites that session's report, dashboard, breakdown and"
            " manifest, and exits as that session did, so a session that did not run its course still fails."
            " Refused if the session is still running, already chose in FINAL, or ran on another build than this one."
        ),
    )
    final_only.add_argument("session_dir", type=Path, help="The session directory of the run to finish.")
    _add_verbose_option(final_only)
    conc_sweep = commands.add_parser(
        CONC_SWEEP_COMMAND,
        help=(
            "Sweep a closed session's root and the commit FINAL chose over the concurrencies of the base's InferenceX"
            " recipe, measured on this machine; the points already measured are kept."
        ),
    )
    conc_sweep.add_argument("session_dir", type=Path, help="The session directory of a closed run.")
    _add_verbose_option(conc_sweep)
    hosted = commands.add_parser(HOSTED_COMMAND, help="Run under a host that injects the campaign spec and its inputs.")
    hosted_commands = hosted.add_subparsers(dest="hosted_command", required=True)
    hosted_commands.add_parser(
        "run",
        help="Run the injected spec's session, or resume or finish an earlier start's; exit.json says when it is over.",
    )
    return parser


def _add_run_options(run: argparse.ArgumentParser) -> None:
    resume = "Continue the session in SESSION_DIR, which ended before FINAL, on its own settings; no other option."
    run.add_argument("--resume", type=Path, metavar="SESSION_DIR", help=resume)
    _add_workload_options(run)
    _add_budget_options(run)
    _add_specialist_input_options(run)
    _add_concurrency_options(run)
    _add_playbook_option(run)
    _add_verbose_option(run)


def _add_verbose_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action=SWITCH, help="Log at debug level.")


def _add_agentx_options(run: argparse.ArgumentParser) -> None:
    """The options only an agentx workload takes."""

    run.add_argument(
        "--agentx-duration-sec",
        type=int,
        default=None,
        help="Trace window of an agentx workload's tests, written as the benchmark's DURATION; refused for inferencex.",
    )
    run.add_argument(
        "--agentx-objective",
        choices=[objective.value for objective in AgentXObjective],
        default=None,
        help=f"Axis an agentx workload is graded on, default {DEFAULT_AGENTX_OBJECTIVE.value}: a "
        f"{DEFAULT_AGENTX_OBJECTIVE.bar_pct:g}%% gain on it is "
        "kept while P90 interactivity and the other axis (total token throughput, or P50 interactivity) hold within "
        "the noise band. total_token_throughput suits a high --conc only: at a low one the trace's tool and think "
        "time paces the lanes, so a server gain barely moves it. Refused for inferencex.",
    )


def _add_workload_options(run: argparse.ArgumentParser) -> None:
    """The fixed workload, spelled as ``hyperloom optimize`` spells it, and the kernel optimizer its recipes are scoped
    to."""

    run.add_argument("--model", help="Model path, or an id found under HL_MODEL_BASE or in the Hugging Face cache.")
    run.add_argument("--framework", help="Inference framework, e.g. vllm or sglang.")
    run.add_argument("--gpu-type", type=str.lower, choices=SESSION_GPU_TYPES, help="GPU type, in any case.")
    run.add_argument("--workload", choices=[kind.value for kind in WorkloadKind], help="Workload kind.")
    run.add_argument("--tp", type=int, help="Tensor-parallel size; must fit the machine's cards.")
    run.add_argument(
        "--ep", type=int, default=None, help=f"Expert-parallel size the server runs at, default {DEFAULT_EP}."
    )
    run.add_argument("--conc", type=int, help="Concurrency of the primary operating point.")
    for name in SEQUENCE_LENGTHS:
        run.add_argument(f"--{name}", type=int, default=None, help="Required for inferencex; refused for agentx.")
    _add_agentx_options(run)
    run.add_argument("--precision", help="Model precision, as hyperloom optimize spells it.")
    run.add_argument(
        "--boot-budget-sec",
        type=int,
        default=None,
        help=(
            "How long, at least, the server of a test may take to be ready, in seconds, at most"
            f" {MAX_BOOT_BUDGET_SEC} (6 h); the round raises the server's own start timeout above it."
            " It only lengthens: a test runs for its trace window, if any, plus 1800 s plus"
            " this budget when that is longer than its default timeout, and a boot may take this budget when that is"
            " longer than half of the timeout. Not given, a boot may take half of a test's timeout, which is 3600 s"
            " for inferencex and the trace window plus 3600 s for agentx."
        ),
    )
    run.add_argument(
        "--kernel-optimizer",
        choices=[optimizer.value for optimizer in KernelOptimizer],
        default=KernelOptimizer.FORGE.value,
        help="Kernel optimizer the session's recipes are scoped to.",
    )
    run.add_argument(
        "--reference-script",
        dest="reference_script",
        type=Path,
        default=None,
        metavar=FILE_METAVAR,
        help=(
            "The session's base launch, as for hyperloom optimize: a shell launch script, or a base launch record"
            " (hyperloom-base-launch/1) frozen by another session, which is frozen without reading any recipe source"
            " (examples/multibranch-agent-launch). It wins over every recipe source; omitted, the base comes from"
            " InferenceX, then the framework's cookbook, then the default table. Every component agent is"
            " served the frozen base. Not the --reference directories the specialists read."
        ),
    )
    run.add_argument(
        "--inferencex-commit",
        dest="inferencex_commit",
        default="",
        metavar="SHA",
        help=(
            "Resolve the session's base launch from InferenceX at this commit (a full sha, or a prefix of 7 or more hex"
            " digits) instead of the head of its main, and record the full sha in the frozen base; the cookbooks are"
            " still read at main. A commit InferenceX does not have, or one that lacks code the resolver runs, stops"
            " the session before PRELUDE; a resumed session keeps its frozen base."
        ),
    )


def _add_budget_options(run: argparse.ArgumentParser) -> None:
    """What the session may spend: wall-clock time, GPU hours, LLM dollars and live branches."""

    run.add_argument(
        "--max-hours", type=float, default=MB_DEFAULT_MAX_HOURS, help="Wall-clock budget, PRELUDE included; 0 = none."
    )
    run.add_argument(
        "--max-gpu-hours",
        type=float,
        default=0.0,
        help=(
            "GPU hours the session may use, tests' GPU time plus card time any lease holds;"
            " EXPLORE ends when FINAL's share is all that is left; 0 = none."
        ),
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


def _add_concurrency_options(run: argparse.ArgumentParser) -> None:
    """How much runs at once and how often the judge looks: tests, a branch's K, dev sandboxes and judge calls."""

    run.add_argument(
        "--max-parallel-tests",
        type=int,
        default=None,
        help="Tests running at once; default as many as the machine's cards, or the grant, run at TP.",
    )
    run.add_argument("--max-k-per-branch", type=int, default=DEFAULT_MAX_K_PER_BRANCH, help="Highest specialist K.")
    run.add_argument(
        "--dev-leases",
        type=int,
        default=0,
        metavar="N",
        help="Dev sandboxes lent to the specialists at once, from the cards tests are measured on; 0 = none.",
    )
    run.add_argument(
        "--make-before-break",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hosted: keep a finished test's sandbox until the next one runs, where 3+ tests fit (one fewer runs).",
    )
    run.add_argument(
        "--enable-conc-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After CLOSE, sweep the root and the commit FINAL chose over the base's InferenceX recipe concurrencies.",
    )
    run.add_argument(
        "--conc-sweep-concs",
        type=_concurrencies,
        default=(),
        metavar="C1,C2,...",
        help="The conc sweep's concurrencies for a base no InferenceX recipe gave; ignored for one that did.",
    )
    run.add_argument(
        "--measurement",
        choices=(LOCAL_MEASUREMENT, HOSTED_MEASUREMENT),
        default=LOCAL_MEASUREMENT,
        help=(
            "Where tests run: this machine's cards, or the sandboxes of the grant the environment hands over"
            " (E2B_API_URL, E2B_SANDBOX_URL, HYPERLOOM_SANDBOX_KEY_MEASUREMENT, HYPERLOOM_SANDBOX_TEMPLATES,"
            " HYPERLOOM_GRANT_LIMITS, HYPERLOOM_RUN_CREATED_AT, HYPERLOOM_SOURCE_REPO, HYPERLOOM_SOURCE_SHA)."
        ),
    )
    run.add_argument(
        "--judge-interval-sec", type=float, default=DEFAULT_JUDGE_INTERVAL_SECONDS, help="Seconds between judge calls."
    )


def _add_playbook_option(run: argparse.ArgumentParser) -> None:
    """Which scheduling playbook the session plays after PRELUDE."""

    run.add_argument(
        "--playbook",
        choices=[mode.value for mode in PlaybookMode],
        default=PlaybookMode.AUTO.value,
        help=(
            "The scheduling playbook the session plays from the RSI archive: the champion, or, in a "
            "challenger session, the oldest playbook on trial; auto (the default) makes half the sessions, drawn by "
            "their id, challenger sessions."
        ),
    )


def _add_specialist_input_options(run: argparse.ArgumentParser) -> None:
    """What the user gives the specialists to work from: their own directions, the third-party agents some branches
    run, and the material they read."""

    run.add_argument(
        "--specialist-prompt",
        dest="specialist_prompts",
        action=REPEATABLE,
        type=Path,
        default=None,
        metavar=FILE_METAVAR,
        help="A direction for one specialist; repeatable, each opens one branch where EXPLORE starts, "
        "within --max-branches.",
    )
    _add_agent_options(run)
    run.add_argument(
        "--reference",
        dest="references",
        action=REPEATABLE,
        default=None,
        metavar="NAME=DIR",
        help="A directory of reference material the specialists read; repeatable. It is copied to MB-reference/NAME"
        " before PRELUDE, and the commit its checkout is on is recorded in the manifest. The specialists are briefed on"
        " leads from InferenceX's reference launches for the workload only when an InferenceX checkout is given as"
        f" --reference {INFERENCEX_REFERENCE}=DIR; under any other name it is plain reference.",
    )


def _add_agent_options(run: argparse.ArgumentParser) -> None:
    """The third-party agents some branches run, and what the component agents among them are given."""

    run.add_argument(
        "--autonomous-agent",
        dest="agents",
        action=REPEATABLE,
        default=None,
        metavar="SOURCE",
        help=(
            "A third-party agent to run end to end on one branch, a git URL (URL@ref for a ref) or a local path;"
            " repeatable, each opens one branch where EXPLORE starts, within --max-branches."
        ),
    )
    run.add_argument(
        "--agent-directions",
        dest="agent_directions",
        type=Path,
        default=None,
        metavar=FILE_METAVAR,
        help=(
            'A directions map, a JSON list of directions or {"directions": [...]} as a component agent\'s'
            " component.plan reads it; split into disjoint shares, one per component agent (an --autonomous-agent an"
            " installed adapter recognises), before EXPLORE opens their branches. Default: for two or more of them,"
            " the map of a dossier pass a mapping branch runs in EXPLORE (--agent-dossier-minutes), else one the"
            " adapter generates from the root's profile; each share then opens as a branch of its own."
        ),
    )
    _add_agent_minutes_options(run)
    _add_agent_start_options(run)
    _add_agent_review_options(run)


def _add_agent_minutes_options(run: argparse.ArgumentParser) -> None:
    """How long the dossier pass may take to make the agent map, and how long its shares must have left to open."""

    run.add_argument(
        "--agent-dossier-minutes",
        dest="agent_dossier_minutes",
        type=float,
        default=DOSSIER_MINUTES,
        metavar="MINUTES",
        help=(
            "Without --agent-directions, how long the dossier pass that maps two or more component agents' directions"
            " may run on its mapping branch before their share branches open, out of EXPLORE's time, while the other"
            " branches work; it ends once its map is set. A component agent starts no"
            f" iteration with under 63 minutes left, so less runs none; 0 skips it (default {DOSSIER_MINUTES:g})."
        ),
    )
    run.add_argument(
        "--agent-share-minutes",
        dest="agent_share_minutes",
        type=float,
        default=SHARE_MINUTES,
        metavar="MINUTES",
        help=(
            "The least time to their cutoff the share branches of a split map must still have, after the dossier pass"
            " and the review, to open: one iteration of a component agent. Below it none opens, and the session says"
            " why. It is also the least an agent branch whose lease finds no free cards must still have to run to"
            f" keep waiting for them; 0 always opens them and waits while any time is left (default {SHARE_MINUTES:g})."
        ),
    )


def _add_agent_start_options(run: argparse.ArgumentParser) -> None:
    """How a component agent is started: its pack, the launch its recipe adds to, and its servers' ports."""

    run.add_argument(
        "--agent-pack",
        dest="agent_pack",
        default=None,
        metavar="NAME",
        help="The pack a component agent runs, the name of one pack directory in its checkout; default: the adapter's.",
    )
    run.add_argument(
        "--agent-launch",
        dest="agent_launch",
        type=Path,
        default=None,
        metavar=FILE_METAVAR,
        help=(
            "For a component agent: the launch an AgentX recipe's adds to (server_flags, server_env,"
            " model_name; examples/multibranch-agent-launch), served instead of the session's frozen base,"
            " which the component agents are served without it. A base launch record is given as --reference-script."
        ),
    )
    run.add_argument(
        "--agent-port",
        dest="agent_port",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "The first port of the blocks the --autonomous-agent servers take, one block per agent and one for the"
            f" dossier pass; sessions that share a machine each need their own (default {AGENT_PORT})."
        ),
    )


def _add_agent_review_options(run: argparse.ArgumentParser) -> None:
    """The stages of a person's review of the agent map: stop once it is made, run on a reviewed one, or skip the
    review."""

    run.add_argument(
        "--stop-after-map",
        dest="stop_after_map",
        action=SWITCH,
        help=(
            "End the session before EXPLORE once the component agents' directions are mapped, those of a single one"
            " too: PRELUDE, the dossier pass (--agent-dossier-minutes; with 0, a map generated from the root's"
            " profile), then CLOSE. The map and its dossier are kept in the session's MB-agent-map/ for review."
        ),
    )
    run.add_argument(
        "--agent-dossier",
        dest="agent_dossier",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "The dossier the reviewed --agent-directions map was made with, a directory with the component agent's"
            " dossier index: every component agent's branch gets it in its checkout, as it would a dossier pass's"
            " (a --stop-after-map session keeps it under MB-agent-map/dossier/)."
        ),
    )
    run.add_argument(
        "--no-review-gate",
        dest="review_gate",
        action="store_false",
        help=(
            "Open the component agents' branches without a person's review: by default they wait once their map"
            " is made, holding no cards, until `hyperloom-multibranch review --session DIR approve` (or reject); the"
            " other branches go on, and a --agent-directions map is reviewed before EXPLORE. With this flag nothing"
            " waits, and the component agents are started with their own review gate off. For unattended runs, and for"
            " the second stage of a reviewed run."
        ),
    )


def _positive(args: argparse.Namespace, names: tuple[str, ...]) -> list[str]:
    return [f"--{name.replace('_', '-')} must be positive" for name in names if getattr(args, name) <= 0]


def _shape_errors(args: argparse.Namespace) -> list[str]:
    given = [name for name in SEQUENCE_LENGTHS if getattr(args, name) is not None]
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return [f"--{name} is not taken by an agentx workload, whose trace fixes its requests" for name in given]
    missing = [f"--{name} is required for an inferencex workload" for name in SEQUENCE_LENGTHS if name not in given]
    return missing or _positive(args, SEQUENCE_LENGTHS)


def _agentx_only_errors(args: argparse.Namespace) -> list[str]:
    given = [name for name in ("agentx_duration_sec", "agentx_objective") if getattr(args, name) is not None]
    if WorkloadKind(args.workload) is WorkloadKind.INFERENCEX:
        return [f"--{name.replace('_', '-')} is not taken by an inferencex workload" for name in given]
    return _positive(args, tuple(name for name in given if name != "agentx_objective"))


def _workload_shape(args: argparse.Namespace) -> InferenceXShape | None:
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return None
    return InferenceXShape(args.isl, args.osl, os.environ.get(DATASET_PATH_ENV, "").strip())


def _workload_model(args: argparse.Namespace) -> str:
    """``--model``, an agentx workload's model directory by its resolved path: the benchmark the session writes for it
    pins the directory so."""

    directory = Path(args.model).expanduser()
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX and directory.is_dir():
        return str(directory.resolve())
    return args.model


#: Schemes whose URL user is a login name rather than a credential; any other scheme's user, or a password, is one.
SSH_SCHEMES = frozenset({"ssh", "git+ssh", "svn+ssh"})


def _agents(sources: list[str], free: int) -> tuple[str, ...]:
    """The agent sources; refused when one carries credentials, which come from the sandbox's environment only, or
    when there are more than the ``free`` branches the user's directions leave."""

    if len(sources) > free:
        raise ValueError(f"--autonomous-agent is given {len(sources)} times, more than the {free} branches left")
    for source in sources:
        credentialed = CREDENTIALED_URL_RE.match(source)
        if credentialed is None:
            continue
        scheme = credentialed["scheme"].removesuffix("://")
        if ":" in credentialed["userinfo"] or scheme.lower() not in SSH_SCHEMES:
            raise ValueError(f"--autonomous-agent {scheme}://... carries credentials; give them in the environment")
    return tuple(sources)


def _agent_directions(path: Path | None, agents: list[str]) -> str:
    """The absolute path of the directions map the component agents split, ``""`` without one; refused with no agent
    to split it for, or when it holds no directions as a component agent's ``component.plan`` reads them."""

    if path is None:
        return ""
    if not agents:
        raise ValueError("--agent-directions needs an --autonomous-agent to split it for")
    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"--agent-directions {path} cannot be read: {exc}") from exc
    directions = document.get("directions") if isinstance(document, dict) else document
    if not isinstance(directions, list) or not directions:
        raise ValueError(f"--agent-directions {path} holds no directions")
    return str(path.absolute())


def _agent_pack(name: str | None, agents: list[str]) -> str:
    """The pack the component agents run, ``""`` for the adapter's default; refused with no agent to run it, or when
    it is not the name of one pack directory."""

    if name is None:
        return ""
    if not agents:
        raise ValueError("--agent-pack needs an --autonomous-agent to run it")
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"--agent-pack {name!r} is not the name of one pack directory")
    return name


def _agent_launch(path: Path | None, agents: list[str], reference_script: Path | None) -> str:
    """The absolute path of the launch file the component agents are served, ``""`` without one; refused with no agent
    to serve, beside a ``reference_script``, whose frozen base they are served, when it is a base launch record, which
    is given as ``reference_script``, or with every reason the adapter could not use it."""

    if path is None:
        return ""
    if not agents:
        raise ValueError("--agent-launch needs an --autonomous-agent to serve it")
    if reference_script is not None:
        raise ValueError("--agent-launch is not taken with --reference-script: the agents are served the frozen base")
    try:
        launch = read_launch(path)
    except AgentRefused as exc:
        raise ValueError(f"--agent-launch: {exc}") from exc
    if launch.record:
        raise ValueError(
            f"--agent-launch {path} is a base launch record; give it as --reference-script, which freezes it as the"
            " session's base and serves it to the agents"
        )
    return str(path.absolute())


def _reference_script(path: Path | None, framework: str) -> str:
    """The absolute path of the operator's launch file the session freezes as its base, ``""`` without one; refused
    when it cannot serve as a ``framework`` base: unreadable, no launch of ``framework``, a flag set from a shell
    variable, or a record that fails its schema or digest. Its topology is checked when the session freezes it."""

    if path is None:
        return ""
    try:
        parse_user_file(str(path), path.read_bytes(), framework=framework)
    except (OSError, UserFileError, BaseLaunchError) as exc:
        raise ValueError(f"--reference-script {path} cannot serve as the base launch: {exc}") from exc
    return str(path.absolute())


def _stop_after_map(stop: bool, agents: list[str], directions: Path | None) -> bool:
    """Whether the session stops after its map; refused with no agent to map for, or a map already given."""

    if stop and not agents:
        raise ValueError("--stop-after-map needs an --autonomous-agent whose directions it maps")
    if stop and directions is not None:
        raise ValueError("--stop-after-map maps directions, and --agent-directions already gives them")
    return stop


def _agent_dossier(path: Path | None, directions: Path | None) -> str:
    """The absolute path of the dossier every agent branch is given, ``""`` without one; refused without the reviewed
    map it was made with, or when it is no dossier: a directory with the component agent's dossier index."""

    if path is None:
        return ""
    if directions is None:
        raise ValueError("--agent-dossier is given with the --agent-directions map it was made with")
    if lacks := dossier_lacks(path):
        raise ValueError(f"--agent-dossier {path} is no dossier: it has no {lacks}")
    return str(path.absolute())


def _agent_port(port: int | None, agents: list[str]) -> int:
    """The first port of the agents' port blocks; refused with no agent to serve on it, or when it is no port."""

    if port is None:
        return AGENT_PORT
    if not agents:
        raise ValueError("--agent-port needs an --autonomous-agent to serve on it")
    if not 1 <= port <= MAX_PORT:
        raise ValueError(f"--agent-port {port} is not a port")
    return port


def _user_directions(files: list[Path], max_branches: int) -> tuple[str, ...]:
    if len(files) > max_branches:
        raise ValueError(f"--specialist-prompt is given {len(files)} times, more than --max-branches {max_branches}")
    directions = []
    for path in files:
        try:
            text = path.read_text(encoding=ENCODING).strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"--specialist-prompt {path} cannot be read: {exc}") from exc
        if not text:
            raise ValueError(f"--specialist-prompt {path} is empty")
        directions.append(text)
    return tuple(directions)


def _references(specs: list[str]) -> tuple[ReferenceSource, ...]:
    sources = tuple(reference_source(spec) for spec in specs)
    names = {source.name for source in sources}
    if len(names) != len(sources):
        raise ValueError("--reference is given one NAME twice; each names one directory under MB-reference")
    return sources


def settings_from(
    args: argparse.Namespace, tests_that_fit: Callable[[int], int] = machine_parallel_tests
) -> RunSettings:
    """The run's settings; raises ValueError naming the invalid options. :class:`RosterCaps` checks the caps.

    ``tests_that_fit`` says how many tests of a TP run at once where the session measures, raising ValueError when one
    does not fit: this machine's cards for ``run``, the grant for ``hosted run``."""

    missing = [f"--{name.replace('_', '-')}" for name in RUN_REQUIRED if getattr(args, name) is None]
    if missing:
        raise ValueError(f"the following arguments are required: {', '.join(missing)}")
    sizes = tuple(name for name in ("tp", "ep", "conc") if getattr(args, name) is not None)
    errors = _shape_errors(args) + _agentx_only_errors(args) + _positive(args, sizes)
    errors += _limit_errors(args)
    if errors:
        raise ValueError("; ".join(errors))
    parallel = _parallel_tests(args, tests_that_fit)
    workload = Workload(
        _workload_model(args),
        args.framework,
        args.tp,
        args.conc,
        args.precision,
        args.gpu_type,
        KernelOptimizer(args.kernel_optimizer),
        _workload_shape(args),
        DEFAULT_EP if args.ep is None else args.ep,
        args.agentx_duration_sec,
        AgentXObjective(args.agentx_objective or DEFAULT_AGENTX_OBJECTIVE),
        args.boot_budget_sec,
    )
    caps = RosterCaps(args.max_branches, args.max_k_per_branch, parallel)
    prompts, agents = args.specialist_prompts or [], args.agents or []
    return RunSettings(
        workload,
        args.max_hours,
        args.max_gpu_hours,
        args.max_llm_usd,
        caps,
        args.judge_interval_sec,
        user_directions=_user_directions(prompts, caps.max_branches),
        agents=_agents(agents, caps.max_branches - len(prompts)),
        references=_references(args.references or []),
        reference_script=_reference_script(args.reference_script, args.framework),
        dev_leases=args.dev_leases,
        playbook=PlaybookMode(args.playbook),
        make_before_break=args.make_before_break,
        agent_directions=_agent_directions(args.agent_directions, agents),
        agent_pack=_agent_pack(args.agent_pack, agents),
        agent_launch=_agent_launch(args.agent_launch, agents, args.reference_script),
        agent_dossier_minutes=args.agent_dossier_minutes,
        agent_share_minutes=args.agent_share_minutes,
        agent_port=_agent_port(args.agent_port, agents),
        stop_after_map=_stop_after_map(args.stop_after_map, agents, args.agent_directions),
        agent_dossier=_agent_dossier(args.agent_dossier, args.agent_directions),
        review_gate=args.review_gate,
        inferencex_commit=parse_inferencex_commit(args.inferencex_commit) if args.inferencex_commit else "",
        agentx_early_stop=session_early_stop(os.environ),
        conc_sweep=args.enable_conc_sweep,
        conc_sweep_concs=args.conc_sweep_concs,
    )


def _concurrencies(value: str) -> tuple[int, ...]:
    """``C1,C2,...`` as the distinct concurrencies it names, ascending; each must be positive."""
    concs = tuple(sorted({int(item) for item in value.split(",")}))
    if concs[0] < 1:
        raise argparse.ArgumentTypeError(f"concurrencies must be positive, got {value}")
    return concs


def _limit_errors(args: argparse.Namespace) -> list[str]:
    """What is wrong with the session's limits: its hours, GPU hours, LLM dollars, judge interval, agent minutes, dev
    sandboxes and boot budget."""

    hours, gpu_hours, usd = args.max_hours, args.max_gpu_hours, args.max_llm_usd
    judge, dossier, share = args.judge_interval_sec, args.agent_dossier_minutes, args.agent_share_minutes
    return [
        *_limit_error("--max-hours", hours, hours >= 0, "0 (no limit) or a positive number of hours"),
        *_limit_error("--max-gpu-hours", gpu_hours, gpu_hours >= 0, "0 (no limit) or a positive number of hours"),
        *_limit_error("--max-llm-usd", usd, usd >= 0, "0 (no limit) or a positive number of dollars"),
        *_limit_error("--judge-interval-sec", judge, judge > 0, "a positive number of seconds"),
        *_limit_error("--agent-dossier-minutes", dossier, dossier >= 0, "0 (runs no dossier pass) or positive"),
        *_limit_error("--agent-share-minutes", share, share >= 0, "0 (always opens the shares) or positive"),
        *_limit_error("--dev-leases", args.dev_leases, args.dev_leases >= 0, "0 (none) or a positive number"),
        *_boot_budget_errors(args.boot_budget_sec),
    ]


def _boot_budget_errors(budget: int | None) -> list[str]:
    if budget is None or 0 < budget <= MAX_BOOT_BUDGET_SEC:
        return []
    return [f"--boot-budget-sec must be a positive number of seconds, at most {MAX_BOOT_BUDGET_SEC} (6 h)"]


def _limit_error(option: str, value: float, in_range: bool, rule: str) -> list[str]:
    """``option`` must be ``rule`` unless ``value`` is finite and ``in_range``: a NaN is in no range, and an infinite
    limit is no limit, so neither may pass as one."""

    return [] if in_range and math.isfinite(value) else [f"{option} must be {rule}"]


def _parallel_tests(args: argparse.Namespace, tests_that_fit: Callable[[int], int]) -> int:
    """How many tests run at once: ``--max-parallel-tests``, at most the tests of the TP that fit where the session
    measures (``tests_that_fit``); raises ValueError when the TP does not fit, or more are asked for."""

    try:
        fitting = tests_that_fit(args.tp)
    except ValueError as exc:
        raise ValueError(f"--tp does not fit where the session measures: {exc}") from exc
    parallel = fitting if args.max_parallel_tests is None else args.max_parallel_tests
    if parallel > fitting:
        raise ValueError(f"--max-parallel-tests must be at most {fitting}, the tests that run at once at TP")
    return parallel


def refuse_options_beside_resume(args: argparse.Namespace) -> None:
    """Raise ValueError naming the options of ``run`` given beside ``--resume``, whose session's manifest sets them."""

    defaults = argparse.ArgumentParser()
    _add_run_options(defaults)
    unset = vars(defaults.parse_args([]))
    if given := [
        name for name, value in unset.items() if name not in ("resume", "verbose") and getattr(args, name) != value
    ]:
        raise ValueError(f"--resume takes the session's own settings and no other option; given: {', '.join(given)}")


__all__ = [
    "ENCODING",
    "FINAL_ONLY_COMMAND",
    "HOSTED_COMMAND",
    "HOSTED_MEASUREMENT",
    "PROG",
    "SEQUENCE_LENGTHS",
    "SESSION_GPU_TYPES",
    "build_parser",
    "refuse_options_beside_resume",
    "settings_from",
]
