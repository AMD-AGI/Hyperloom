# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and the kernel
optimizer its recipes are scoped to, and the hard limits the judge works within. ``--workload`` names the workload
kind: ``inferencex`` takes ``--isl`` and ``--osl``, and ``agentx`` refuses them, as the trace it replays fixes its
requests; ``agentx`` takes ``--ep``, 1 when not given, and ``inferencex`` refuses it. An agentx session serves its
model in the image ``HYPERLOOM_ACTIVE_IMAGE_REF`` names, and a ``--model`` that is a directory is recorded by its
resolved path, which its benchmark pins. An operator environment whose ``HYPERLOOM_AGENTX`` says otherwise than
``--workload``, or that names a benchmark YAML in ``HYPERLOOM_BENCHMARK_CONFIG``, is refused.
``--max-gpu-hours`` caps the GPU hours the session uses, tests' GPU time plus card time any lease holds: EXPLORE ends
with ``gpu_hours_exhausted`` once only FINAL's share is left, and FINAL still runs.
``--max-llm-usd`` caps the dollars the session's LLM calls cost: EXPLORE
ends with ``llm_budget_exhausted`` once their recorded spend reaches it, and FINAL still runs. Each call is priced from
the price sheet, cached tokens at cache rates: a specialist launch's calls from its log when the launch ends, a launch
cut short, lost or stopped included, so the stop can come up to one launch late, and each call of the scheduler's own
judge, patch resolver and the rule gate's fix critic as it returns. A cap is a usage error when the price sheet has no
rate for the model the specialists run on or the one the judge, the patch resolver and the fix critic ask, whose calls
would record tokens only. ``--reuse-prelude DIR`` roots the session on the child session the PRELUDE of the
earlier session in ``DIR`` rooted on instead of starting a child, and still measures the baseline; it is a usage error
unless that session committed its root on the same workload and on the image ``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` names.
``--playbook`` says which scheduling playbook the session plays once ``HYPERLOOM_RSI_ARCHIVE`` names an archive:
``champion`` (the default), or ``challenger``, which plays the oldest playbook on trial in the pool beside it and the
champion while none is; the manifest fingerprint and the session's manifest record the label under ``playbook`` in
``policy_versions``, and a challenger's fallback to the champion under ``playbook_fallback``. Without an archive, or
under ``HYPERLOOM_RSI_WRITE_ONLY=1``, the session plays none and is judged as before the RSI: no stall evidence, safety
envelope or probation.
Each ``--autonomous-agent SOURCE`` opens one branch where EXPLORE starts, after the user's directions and within
``--max-branches``, whose specialist runs that third-party agent end to end in a sandbox of its own; a source that
carries credentials in its URL is a usage error. ``--agent-directions FILE`` names a directions map that is split, after
PRELUDE, into disjoint shares, one per Arbor agent among them, each its ``component.plan``; a map with no directions, or
one given with no agent, is a usage error. Without one, two or more Arbors split the map of a dossier pass, the first
Arbor run once before their branches open for at most ``--agent-dossier-minutes`` (default 90), or, when it gives none
or the minutes are 0, a map Arbor generates from the root's profile (its op shares, else its kernel shares); a negative
value is a usage error. ``--agent-pack NAME`` names the pack those Arbors run instead of Arbor's
serving template; a name that is not one directory, or one given with no agent, is a usage error, and a pack the Arbor
checkout lacks is refused when the branch starts. ``--agent-launch FILE`` gives those Arbors the launch an AgentX
recipe's adds to, which Magpie resolves from InferenceX and MB does not carry yet; a file the adapter cannot use, or one
given with no agent, is a usage error. Each ``--specialist-prompt`` file holds a direction the user wrote
for one specialist: its branch opens at PRELUDE's
root ahead of the opening batch, counted within ``--max-branches``, its specialist sees the text in every launch
prompt, and the judge neither stops it nor lowers its K for its first judged tests; an empty or unreadable file, or
more files than ``--max-branches``, is a usage error.
Each ``--reference NAME=DIR`` names a directory the specialists read as reference, such as GEAK's tuning and kernel
knowledge: it is copied to ``MB-reference/NAME`` before PRELUDE and the commit its checkout is on is recorded in the
manifest, so every branch reads one fixed copy and the session says which. Only an InferenceX checkout named
:data:`~hyperloom_multibranch.blackboard.knowledge.reference_leads.INFERENCEX_REFERENCE` briefs the specialists on its
reference launches for the workload. A ``DIR`` that is no readable directory
in a git checkout, and a ``NAME`` given twice, are usage errors. A session never refreshes its copy; knowledge that
lands after it starts is what the next session takes. A session whose toolbox has fewer
usable cards left than one test needs ends EXPLORE with ``no_measurement_cards`` and skips FINAL. Exit codes: 0 when
the session ran its course, its wall-clock, GPU-hour or LLM budget spent included, 1 when it failed or was stopped,
``no_measurement_cards`` included, 2 on a usage error. The environment and the peers a session needs are checked before
anything is created, so a missing one fails the command without leaving a session directory behind: the Recipe KB that
PRELUDE reads warm-start leads from and CLOSE writes the final recipe back to is configured by ``KB_STORE_URL`` and
``KB_STORE_TOKEN``, both or neither, and is not used without them; the RSI archive ``HYPERLOOM_RSI_ARCHIVE``
names must be set and a writable directory (no path is assumed: each deployment names its own shared folder); a
session measured in hosted sandboxes also needs ``HYPERLOOM_RUNNER_CACHE``, this host's directory of the runner cache; the measurement pins must be readable, the image the session runs on named by both
``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST`` included; the framework's package must be
installed in the image; and the agent CLI the specialists run must be installed. A canonical measurement that cannot
be taken, one naming a launcher the InferenceX checkout lacks and Magpie cannot supply included, fails the command the
same way once the session directory exists. Before PRELUDE the session checks the rule pack's facts about outside
projects against this environment as ``HYPERLOOM_IMAGE_CHECK`` says: ``warn`` (the default) logs a failing fact,
``strict`` fails the command the same way before any phase runs, ``off`` checks nothing, and any other value is a usage
error. The manifest records the check's summary and CLOSE writes the whole report to ``MB-reports/image_check.json``.
A session uses no Experience KB, whatever ``HYPERLOOM_KB_URL`` names (see
:data:`~hyperloom_multibranch.scheduling.mb_fakes.MB_FAKE_EXPERIENCE_KB`).
``run --measurement hosted`` measures every test in the sandboxes of a grant handed to the session in the environment
(:data:`HANDED_GRANT`, read by :mod:`hyperloom.hosted.host_info`) instead of this machine's cards: as many tests run at
once as the grant has cards and sandboxes for, and the session stops before the grant ends as ``hosted run`` does. Its
run id is ``HYPERLOOM_RUN_ID`` when set, the session's id otherwise. A missing or malformed grant variable is a usage
error naming it. Its ``--dev-leases`` and ``--autonomous-agent`` run in ``dev`` sandboxes of the grant, so they also
need the ``dev`` key (``HYPERLOOM_SANDBOX_KEY_DEV``) and ``USER_DATA_PATH``, the storage the ``dev`` template mounts at
the same path; a usage error names what is missing.
``hosted run`` takes no arguments: it runs the session of the campaign spec a host injected, as
:mod:`~hyperloom_multibranch.scheduling.hosted_run` maps it onto the options of ``run``, measured in the runner
sandboxes of the host's sandbox service (:mod:`~hyperloom_multibranch.scheduling.hosted_peers`). Started again over a
run whose ``session.json`` names a session, it finishes that session as ``final-only`` does. It tells the host the run
is over -- ``final: true`` in ``exit.json`` -- once CLOSE ran, and when the spec, the environment, a missing package of
the ``hosted`` extra or the runner cache keeps a session from running, with exit code 2 for a refused spec or session
and 1 otherwise; a crash leaves no ``exit.json``, and the host starts the run again.
``final-only SESSION_DIR`` is the emergency recovery of a session that ended before FINAL ran, and takes nothing but
that directory: it runs FINAL and CLOSE over what the session persisted -- the settings and the frontier policy its
manifest holds, the branches its roster and blackboard hold -- and rewrites its report, dashboard and breakdown. It
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
import contextlib
import io
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path
from hyperloom.common.env import EnvValueError
from hyperloom.common.outside_facts.image_check import ImageCheckFailed, mode_from_environment
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom.hosted.host_info import HostGroup, HostInfo, take_host_info
from hyperloom.hosted.toolbox_inputs import ToolboxInputsError
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom.measure.pins import IMAGE_DIGEST_VAR
from hyperloom.measure.runner.cache_build import CacheBuildError
from hyperloom.measure.short_prompts.staging import CACHE_ENV as RUNNER_CACHE_ENV
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient, RemoteRecipeConfigurationError
from hyperloom_multibranch import points
from hyperloom_multibranch.contracts.llm_prices import price_of
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard.knowledge.reference_leads import INFERENCEX_REFERENCE
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.rulegate.critic import LlmCritic, anthropic_critic
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS, MB_MIN_PRELUDE_HOURS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.hosted_peers import PinnedToolboxInputs, load_hosted_peers
from hyperloom_multibranch.scheduling.hosted_run import (
    GrantDeadline,
    HostedExit,
    arm_grant_stop,
    campaign_spec,
    grant_deadline,
    record_session,
    recorded_session,
    refuse_unhosted_branches,
    reported_stop_reason,
    run_argv,
    toolbox_grant,
    write_final_exit,
)
from hyperloom_multibranch.scheduling import review
from hyperloom_multibranch.scheduling.layout import create_session
from hyperloom_multibranch.scheduling.lifecycle import (
    PersistedRun,
    Session,
    SessionHost,
    SessionServices,
    persisted_run,
)
from hyperloom_multibranch.scheduling.llm import (
    MB_RESOLVER_MAX_TOKENS,
    MB_RESOLVER_READ_TIMEOUT_SECONDS,
    anthropic_completion,
    default_model,
)
from hyperloom_multibranch.scheduling.local_peers import (
    AGENTX_ENV,
    load_peers,
    machine_parallel_tests,
    specialist_model,
)
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeersUnavailable
from hyperloom_multibranch.scheduling.prelude import reused_prelude
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, adopting_orphans
from hyperloom_multibranch.scheduling.reference import ReferenceSource, reference_source
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    AGENT_PORT,
    DOSSIER_MINUTES,
    InferenceXShape,
    KernelOptimizer,
    PlaybookMode,
    RunSettings,
    Workload,
)
from hyperloom_multibranch.specialist.arbor_adapter import DOSSIER_INDEX, ArborRefused, read_launch

PROG = "hyperloom-multibranch"
EXIT_FAILED = 1
EXIT_USAGE = 2
MAX_PORT = 65535
DEFAULT_MAX_BRANCHES = 4
DEFAULT_MAX_K_PER_BRANCH = 2
DEFAULT_JUDGE_INTERVAL_SECONDS = 600.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
ENCODING = "utf-8"
FILE_METAVAR = "FILE"
log = logging.getLogger(__name__)
#: The GPU types a session runs on: those Magpie has a benchmark script of their own for, and mi350x, which runs
#: mi355x's (the same gfx950 die) through ``gpu_runner_type``, as the toolbox's environment sets ``GPU_TYPE``.
#: ``hyperloom optimize`` also runs mi308x and mi325x on mi300x's script; a session does not accept them yet.
MAGPIE_RUNNER_GPU_TYPES = ("mi300x", "mi350x", "mi355x")
#: The options an inferencex workload requires and an agentx one refuses.
SEQUENCE_LENGTHS = ("isl", "osl")
#: The argparse action of an option the user may give more than once, each giving one more value.
REPEATABLE = "append"
#: Where ``run`` measures: this machine's cards, or the sandboxes of a grant handed to it.
LOCAL_MEASUREMENT, HOSTED_MEASUREMENT = "local", "hosted"
#: What a grant handed to ``run --measurement hosted`` must give: the sandbox service, the measurement key and
#: templates, the grant's limits and creation time, and the source the runner cache is built from.
HANDED_GRANT = (HostGroup.SANDBOX, HostGroup.MEASUREMENT_KEY, HostGroup.GRANT, HostGroup.SOURCE)


class EnvironmentIncomplete(RuntimeError):
    """The environment lacks what a session needs; the message names the variables to set."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run_options(commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE."))
    points.add_parser(commands)
    review.add_parser(commands)
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
    hosted = commands.add_parser("hosted", help="Run under a host that injects the campaign spec and its inputs.")
    hosted_commands = hosted.add_subparsers(dest="hosted_command", required=True)
    hosted_commands.add_parser(
        "run",
        help=(
            "Run the session of the injected campaign spec, or finish the one an earlier start of the run began, and"
            " tell the host in exit.json when the run is over."
        ),
    )
    return parser


def _add_run_options(run: argparse.ArgumentParser) -> None:
    _add_workload_options(run)
    _add_budget_options(run)
    _add_specialist_input_options(run)
    _add_concurrency_options(run)
    run.add_argument(
        "--reuse-prelude",
        type=Path,
        default=None,
        metavar="DIR",
        help="An earlier session of this workload and image whose PRELUDE child to root on instead of starting one.",
    )
    _add_playbook_option(run)
    run.add_argument("-v", "--verbose", action="store_true", help="Log at debug level.")


def _add_workload_options(run: argparse.ArgumentParser) -> None:
    """The fixed workload, spelled as ``hyperloom optimize`` spells it, and the kernel optimizer its recipes are scoped
    to."""

    run.add_argument("--model", required=True, help="Model path or id.")
    run.add_argument("--framework", required=True, help="Inference framework, e.g. vllm or sglang.")
    run.add_argument(
        "--gpu-type", required=True, type=str.lower, choices=MAGPIE_RUNNER_GPU_TYPES, help="GPU type, in any case."
    )
    run.add_argument("--workload", required=True, choices=[kind.value for kind in WorkloadKind], help="Workload kind.")
    run.add_argument("--tp", type=int, required=True, help="Tensor-parallel size; must fit the machine's cards.")
    run.add_argument("--conc", type=int, required=True, help="Concurrency of the primary operating point.")
    for name in SEQUENCE_LENGTHS:
        run.add_argument(f"--{name}", type=int, default=None, help="Required for inferencex; refused for agentx.")
    run.add_argument(
        "--ep",
        type=int,
        default=None,
        help=f"Expert-parallel size of an agentx workload, default {DEFAULT_EP}; refused for inferencex.",
    )
    run.add_argument("--precision", required=True, help="Model precision, as hyperloom optimize spells it.")
    run.add_argument(
        "--kernel-optimizer",
        choices=[optimizer.value for optimizer in KernelOptimizer],
        default=KernelOptimizer.FORGE.value,
        help="Kernel optimizer the session's recipes are scoped to.",
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
    run.add_argument("--max-k-per-branch", type=int, default=DEFAULT_MAX_K_PER_BRANCH, help="Highest K of a branch.")
    run.add_argument(
        "--dev-leases",
        type=int,
        default=0,
        metavar="N",
        help="Dev sandboxes lent to the specialists at once, from the cards tests are measured on; 0 = none.",
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
        default=PlaybookMode.CHAMPION.value,
        help=(
            "The scheduling playbook the session plays once the RSI archive is configured: the champion, or, in a "
            "challenger session, the oldest playbook on trial."
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
    """The third-party agents some branches run, and what the Arbors among them are given."""

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
            'A directions map, a JSON list of directions or {"directions": [...]} as Arbor\'s component.plan reads it;'
            " split into disjoint shares, one per Arbor --autonomous-agent, before EXPLORE opens their branches."
            " Default: for two or more Arbors, the map of a dossier pass (--agent-dossier-minutes), else one Arbor"
            " generates from the root's profile."
        ),
    )
    run.add_argument(
        "--agent-dossier-minutes",
        dest="agent_dossier_minutes",
        type=float,
        default=DOSSIER_MINUTES,
        metavar="MINUTES",
        help=(
            "Without --agent-directions, how long the dossier pass that maps two or more Arbor agents' directions may"
            " run before their branches open, out of EXPLORE's time; it ends once its map is set. Arbor starts no"
            f" iteration with under 63 minutes left, so less runs none; 0 skips it (default {DOSSIER_MINUTES:g})."
        ),
    )
    run.add_argument(
        "--agent-pack",
        dest="agent_pack",
        default=None,
        metavar="NAME",
        help="The pack an Arbor --autonomous-agent runs, a directory under the Arbor checkout's packs/; default serving.",
    )
    run.add_argument(
        "--agent-launch",
        dest="agent_launch",
        type=Path,
        default=None,
        metavar=FILE_METAVAR,
        help=(
            "The launch an AgentX recipe's adds to, as Magpie resolves it from InferenceX, for an Arbor"
            " --autonomous-agent: a JSON object of server_flags, server_env, model_name and client"
            " (examples/multibranch-arbor-launch)."
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
    run.add_argument(
        "--stop-after-map",
        dest="stop_after_map",
        action="store_true",
        help=(
            "End the session before EXPLORE once the Arbor --autonomous-agent directions are mapped, one Arbor's"
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
            "The dossier the reviewed --agent-directions map was made with, a directory with Arbor's dossier index:"
            " every Arbor --autonomous-agent branch gets it in its checkout, as it would a dossier pass's"
            " (a --stop-after-map session keeps it under MB-agent-map/dossier/)."
        ),
    )
    run.add_argument(
        "--no-review-gate",
        dest="review_gate",
        action="store_false",
        help=(
            "Open the Arbor --autonomous-agent branches without a person's review: by default the session pauses"
            " once their map is made, holding no cards, until `hyperloom-multibranch review --session DIR approve`"
            " (or reject). With this flag it does not pause, and its Arbors are started with their own review gate"
            " off. For unattended runs, and for the second stage of a reviewed run."
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


def _ep_errors(args: argparse.Namespace) -> list[str]:
    if args.ep is None:
        return []
    if WorkloadKind(args.workload) is WorkloadKind.INFERENCEX:
        return [f"--ep is not taken by an inferencex workload, which is served at EP {DEFAULT_EP}"]
    return _positive(args, ("ep",))


def _workload_shape(args: argparse.Namespace) -> InferenceXShape | None:
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return None
    return InferenceXShape(args.isl, args.osl)


def _workload_model(args: argparse.Namespace) -> str:
    """``--model``, an agentx workload's model directory by its resolved path: the benchmark the session writes for it
    pins the directory so, and its PRELUDE child's ``--model`` must name it as pinned."""

    directory = Path(args.model).expanduser()
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX and directory.is_dir():
        return str(directory.resolve())
    return args.model


#: Schemes whose URL user is a login name rather than a credential; any other scheme's user, or a password, is one.
SSH_SCHEMES = frozenset({"ssh", "git+ssh", "svn+ssh"})


def _agents(sources: list[str], free: int) -> tuple[str, ...]:
    """The agent sources, refused when one carries credentials, which come from the sandbox's environment only, or
    when there are more than the ``free`` branches the user's directions leave."""

    if len(sources) > free:
        raise ValueError(f"--autonomous-agent is given {len(sources)} times, more than the {free} branches left")
    for source in sources:
        scheme, found, rest = source.partition("://")
        user, at, _host = rest.split("/", 1)[0].rpartition("@")
        if found and at and (":" in user or scheme.lower() not in SSH_SCHEMES):
            raise ValueError(f"--autonomous-agent {scheme}://... carries credentials; give them in the environment")
    return tuple(sources)


def _agent_directions(path: Path | None, agents: list[str]) -> str:
    """The absolute path of the directions map the Arbor agents split, ``""`` without one; refused with no agent to
    split it for, or when it holds no directions as Arbor's ``component.plan`` reads them."""

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
    """The pack the Arbor agents run, ``""`` for the adapter's default; refused with no agent to run it, or when it
    is not one directory name under an Arbor checkout's ``packs/``."""

    if name is None:
        return ""
    if not agents:
        raise ValueError("--agent-pack needs an --autonomous-agent to run it")
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"--agent-pack {name!r} is not the name of one pack directory")
    return name


def _agent_launch(path: Path | None, agents: list[str]) -> str:
    """The absolute path of the launch file the Arbor agents are served, ``""`` without one; refused with no agent to
    serve, or with every reason the adapter could not use it."""

    if path is None:
        return ""
    if not agents:
        raise ValueError("--agent-launch needs an --autonomous-agent to serve it")
    try:
        read_launch(path)
    except ArborRefused as exc:
        raise ValueError(f"--agent-launch: {exc}") from exc
    return str(path.absolute())


def _stop_after_map(stop: bool, agents: list[str], directions: Path | None) -> bool:
    """Whether the session stops after its map; refused with no agent to map for, or a map already given."""

    if stop and not agents:
        raise ValueError("--stop-after-map needs an --autonomous-agent whose directions it maps")
    if stop and directions is not None:
        raise ValueError("--stop-after-map maps directions, and --agent-directions already gives them")
    return stop


def _agent_dossier(path: Path | None, directions: Path | None) -> str:
    """The absolute path of the dossier every Arbor branch is given, ``""`` without one; refused without the reviewed
    map it was made with, or when it is no dossier: a directory with Arbor's dossier index."""

    if path is None:
        return ""
    if directions is None:
        raise ValueError("--agent-dossier is given with the --agent-directions map it was made with")
    if not (path / DOSSIER_INDEX).is_file():
        raise ValueError(f"--agent-dossier {path} is no dossier: it has no {DOSSIER_INDEX}")
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
    does not fit: this machine's cards for ``run``, the grant for ``hosted run``.
    """

    errors = _shape_errors(args) + _ep_errors(args) + _positive(args, ("tp", "conc", "judge_interval_sec"))
    if args.max_hours < 0 or 0 < args.max_hours < MB_MIN_PRELUDE_HOURS:
        errors.append(f"--max-hours must be 0 (no limit) or at least {MB_MIN_PRELUDE_HOURS:.4f} (one minute)")
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours < 0:
        errors.append("--max-gpu-hours must be 0 (no limit) or a positive number of hours")
    if not math.isfinite(args.max_llm_usd) or args.max_llm_usd < 0:
        errors.append("--max-llm-usd must be 0 (no limit) or a positive number of dollars")
    if args.dev_leases < 0:
        errors.append("--dev-leases must be 0 (none) or a positive number of sandboxes")
    if errors:
        raise ValueError("; ".join(errors))
    try:
        fitting = tests_that_fit(args.tp)
    except ValueError as exc:
        raise ValueError(f"--tp does not fit where the session measures: {exc}") from exc
    parallel = fitting if args.max_parallel_tests is None else args.max_parallel_tests
    if parallel > fitting:
        raise ValueError(f"--max-parallel-tests must be at most {fitting}, the tests that run at once at TP")
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
        agents=_agents(args.agents or [], caps.max_branches - len(args.specialist_prompts or [])),
        references=_references(args.references or []),
        dev_leases=args.dev_leases,
        playbook=PlaybookMode(args.playbook),
        agent_directions=_agent_directions(args.agent_directions, args.agents or []),
        agent_pack=_agent_pack(args.agent_pack, args.agents or []),
        agent_launch=_agent_launch(args.agent_launch, args.agents or []),
        agent_dossier_minutes=_dossier_minutes(args.agent_dossier_minutes),
        agent_port=_agent_port(args.agent_port, args.agents or []),
        stop_after_map=_stop_after_map(args.stop_after_map, args.agents or [], args.agent_directions),
        agent_dossier=_agent_dossier(args.agent_dossier, args.agent_directions),
        review_gate=args.review_gate,
    )


def _dossier_minutes(minutes: float) -> float:
    if minutes < 0:
        raise ValueError(f"--agent-dossier-minutes {minutes:g} is negative; 0 runs no dossier pass")
    return minutes


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

    if not any(os.environ.get(name, "").strip() for name in ("KB_STORE_URL", "KB_STORE_TOKEN")):
        return None
    try:
        return RemoteRecipeClient.from_env()
    except RemoteRecipeConfigurationError as exc:
        raise EnvironmentIncomplete(f"the Recipe KB is half configured: {exc}") from exc


def require_rsi_archive(env: Mapping[str, str]) -> None:
    """Raise :class:`EnvironmentIncomplete` naming ``HYPERLOOM_RSI_ARCHIVE`` when ``env`` does not set it, and unless
    the archive it names is a writable directory. There is no default: the archive is the deployment's shared folder."""

    if not env.get(ARCHIVE_ENV, "").strip():
        raise EnvironmentIncomplete(
            f"{ARCHIVE_ENV} is not set: name the writable shared directory the session archives its search tree in"
        )
    archive = Path(env[ARCHIVE_ENV].strip())
    if not (archive.is_dir() and os.access(archive, os.W_OK)):
        raise EnvironmentIncomplete(
            f"the RSI archive {archive} is not a writable directory: mount it, or set {ARCHIVE_ENV} to one"
        )


def session_services(recipe_kb: RemoteRecipeClient | None) -> SessionServices:
    """The judge, the patch resolver, the fix critic and CLOSE's playbook author and curator on the default model,
    and ``recipe_kb``."""

    model = default_model()
    judge = anthropic_completion(model, component=PROG, operation="judge")
    resolver = anthropic_completion(
        model,
        component=PROG,
        operation="patch_resolve",
        read_seconds=MB_RESOLVER_READ_TIMEOUT_SECONDS,
        max_tokens=MB_RESOLVER_MAX_TOKENS,
    )
    critic = partial(LlmCritic, anthropic_critic(model, component=PROG))
    author = anthropic_completion(model, component=PROG, operation="rsi_authoring")
    return SessionServices(judge, resolver, recipe_kb, critic, author, model, model)


def _print_root(root: Path) -> None:
    print(root, flush=True)


def run_session(
    settings: RunSettings,
    peer_factory: PeerFactory,
    services: SessionServices,
    report_root: Callable[[Path], None] = _print_root,
) -> int:
    """Create the session directory, hand it to ``report_root`` -- by default printed on one stdout line -- before
    anything else is built in it, run the session and return its exit code.

    The session and its peers are built on the event loop they run on, which the specialist tools of the MCP service
    hand every toolbox call to. Whatever the session's processes orphan is reparented to this process, so the session
    stop kills it and reports it while it runs.
    """

    with adopting_orphans():
        return asyncio.run(_run_session(settings, peer_factory, services, report_root))


async def _run_session(
    settings: RunSettings, peer_factory: PeerFactory, services: SessionServices, report_root: Callable[[Path], None]
) -> int:
    clock = Clock()
    layout = create_session(workspace_root(), settings.workload.model_basename, clock)
    report_root(layout.root)
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
    handler = logging.FileHandler(path, encoding=ENCODING)
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


def require_session_environment(settings: RunSettings, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``env`` contradicts the workload of ``settings``, cannot price its dollar cap or sets an
    image check mode there is none of."""

    require_workload_switch(settings.workload, env)
    require_priced_models(settings.max_llm_usd, env)
    mode_from_environment(env)


@dataclass(frozen=True)
class ReadySession:
    """The peers a session is built on and the services it calls out to."""

    peer_factory: PeerFactory
    services: SessionServices


def hosted_ready_session(host: HostInfo, workload: Workload) -> ReadySession:
    """:func:`ready_session` measured in the sandboxes of ``host``'s grant: the one builder of the peers of
    ``hosted run`` and ``run --measurement hosted``."""

    inputs = PinnedToolboxInputs(host, hosted_runner_cache(os.environ))
    return ready_session(workload, partial(load_hosted_peers, inputs=inputs))


def hosted_runner_cache(env: Mapping[str, str]) -> Path:
    """This host's runner cache directory, which ``HYPERLOOM_RUNNER_CACHE`` must name as an absolute path; raises
    :class:`EnvironmentIncomplete` naming the variable otherwise. Hosted sessions take no default: a cache built where
    the measurement sandboxes do not mount it leaves them with no runner, and a sealed sandbox cannot fetch one."""

    value = env.get(RUNNER_CACHE_ENV, "").strip()
    if not value or not Path(value).is_absolute():
        raise EnvironmentIncomplete(
            f"{RUNNER_CACHE_ENV} is not set to an absolute path: name the directory of the runner cache on the shared"
            " storage the measurement sandboxes mount"
        )
    return Path(value)


def grant_tests_that_fit(host: HostInfo) -> Callable[[int], int]:
    """How many tests of a TP the grant ``host`` holds runs at once; raises ValueError when one does not fit, and
    ImportError when the ``hosted`` extra is not installed."""

    # hosted_toolbox loads the sandbox SDK, which only the hosted extra installs.
    from hyperloom_multibranch.toolbox.hosted_toolbox import parallel_tests_for

    return partial(parallel_tests_for, toolbox_grant(host.grant_limits))


def ready_session(workload: Workload, load: Callable[[Workload], PeerFactory]) -> ReadySession:
    """The peers ``load`` has for ``workload`` and the services, the Recipe KB and the RSI archive checked first;
    raises :class:`EnvironmentIncomplete` or :class:`PeersUnavailable` naming what is missing. Creates nothing."""

    recipe_kb = recipe_kb_from_env()
    require_rsi_archive(os.environ)
    peer_factory = load(workload)
    return ReadySession(peer_factory, session_services(recipe_kb))


@dataclass(frozen=True)
class HostedStart:
    """What ``hosted run`` runs, and the seconds after which the grant's end stops it, ``None`` when it has none."""

    run: Callable[[], int]
    stop_after_sec: float | None


def hosted_run() -> int:
    """``hosted run``: the session of the campaign spec the host injected, or the finish of the one an earlier start
    of the run began; writes ``exit.json`` with ``final: true`` once CLOSE ran or when the run cannot be served."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    host = take_host_info(os.environ, require=())
    try:
        start = _hosted_start(host)
    except ValueError as exc:
        return _not_run(host.state_dir, EXIT_USAGE, exc)
    except (EnvironmentIncomplete, PeersUnavailable, ImageCheckFailed) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    except ImportError as exc:
        log.exception("hosted run: a module the session needs is not installed")
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    if start.stop_after_sec is not None:
        arm_grant_stop(max(0.0, start.stop_after_sec))
    try:
        returncode = start.run()
    except (CacheBuildError, ToolboxInputsError) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    root = recorded_session(host.state_dir)
    stop_reason = None if root is None else reported_stop_reason(root)
    write_final_exit(host.state_dir, HostedExit(returncode, stop_reason=stop_reason))
    return returncode


def _hosted_start(host: HostInfo) -> HostedStart:
    """A new session of the spec, or ``final-only`` over the session ``session.json`` names; raises ValueError when
    the spec or that session is refused, ImportError when a package of the ``hosted`` extra is missing, and what
    :func:`ready_session` raises."""

    deadline = grant_deadline(host, datetime.now(timezone.utc))
    previous = recorded_session(host.state_dir)
    persisted = None if previous is None else persisted_run(previous)
    settings = _hosted_settings(host, deadline.session_hours) if persisted is None else persisted.settings
    require_session_environment(settings, os.environ)
    ready = hosted_ready_session(host, settings.workload)
    if persisted is not None:
        run = partial(final_only, persisted, ready.peer_factory, ready.services)
    else:
        run = partial(run_session, settings, ready.peer_factory, ready.services, partial(_record_root, host.state_dir))
    return HostedStart(run, deadline.stop_after_sec)


def _hosted_settings(host: HostInfo, session_hours: Callable[[float], float]) -> RunSettings:
    spec = campaign_spec(host)
    tests_that_fit = grant_tests_that_fit(host)
    argv = run_argv(spec, session_hours(spec.max_hours), tests_that_fit(spec.tp))
    settings = settings_from(_parsed_run(argv), tests_that_fit)
    refuse_unhosted_branches(settings, host, os.environ)
    return settings


def _parsed_run(argv: list[str]) -> argparse.Namespace:
    """``argv`` parsed as the command line it is; raises ValueError with argparse's message where it would exit."""

    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            return build_parser().parse_args(argv)
    except SystemExit as exc:
        message = errors.getvalue().strip().splitlines()[-1]
        raise ValueError(f"the spec does not map onto the options of run: {message}") from exc


def _record_root(state_dir: Path, root: Path) -> None:
    _print_root(root)
    record_session(state_dir, root)


def _not_run(state_dir: Path, returncode: int, exc: Exception) -> int:
    """End a run no session ran in for good: a start of it again would meet what kept this one from running."""

    print(f"{PROG}: hosted run: {exc}", file=sys.stderr)
    write_final_exit(state_dir, HostedExit(returncode, error=str(exc)))
    return returncode


@dataclass(frozen=True)
class HandedGrant:
    """The grant ``run --measurement hosted`` measures in, and how much of it is left."""

    host: HostInfo
    deadline: GrantDeadline


def _handed_grant(args: argparse.Namespace) -> HandedGrant:
    """The grant handed to ``run --measurement hosted``, taken out of the environment; ``args.max_hours`` is cut to
    what it leaves. Raises ValueError naming what is missing or refused, never a value."""

    host = take_host_info(os.environ, require=HANDED_GRANT, require_run=False)
    deadline = grant_deadline(host, datetime.now(timezone.utc))
    args.max_hours = deadline.session_hours(args.max_hours)
    return HandedGrant(host, deadline)


def _run_settings(args: argparse.Namespace, grant: HandedGrant | None) -> RunSettings:
    settings = settings_from(args) if grant is None else settings_from(args, grant_tests_that_fit(grant.host))
    if grant is not None:
        refuse_unhosted_branches(settings, grant.host, os.environ)
    if args.reuse_prelude is not None:
        reused = reused_prelude(args.reuse_prelude, settings.workload, _reused_image_digest(os.environ))
        settings = replace(settings, reuse_prelude=reused)
    return settings


def _start(settings: RunSettings, grant: HandedGrant | None) -> int:
    if grant is None:
        ready = ready_session(settings.workload, load_peers)
    else:
        ready = hosted_ready_session(grant.host, settings.workload)
        if grant.deadline.stop_after_sec is not None:
            arm_grant_stop(max(0.0, grant.deadline.stop_after_sec))
    return run_session(settings, ready.peer_factory, ready.services)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "hosted":
        return hosted_run()
    if args.command == "points":
        try:
            return points.main(args)
        except ValueError as exc:
            print(f"{parser.prog} points: error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    if args.command == "review":
        return review.main(args)
    persisted = grant = None
    try:
        if args.command == "final-only":
            persisted = persisted_run(args.session_dir)
            settings = persisted.settings
        else:
            grant = _handed_grant(args) if args.measurement == HOSTED_MEASUREMENT else None
            settings = _run_settings(args, grant)
        require_session_environment(settings, os.environ)
    except ValueError as exc:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE) from exc
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        if persisted is not None:
            ready = ready_session(settings.workload, load_peers)
            return final_only(persisted, ready.peer_factory, ready.services)
        return _start(settings, grant)
    except (EnvironmentIncomplete, PeersUnavailable, ImageCheckFailed, CacheBuildError, ToolboxInputsError) as exc:
        print(f"{PROG}: cannot run a session, {exc}", file=sys.stderr)
        return EXIT_FAILED


__all__ = [
    "MAGPIE_RUNNER_GPU_TYPES",
    "PROG",
    "SEQUENCE_LENGTHS",
    "EnvironmentIncomplete",
    "HANDED_GRANT",
    "HandedGrant",
    "HostedStart",
    "ReadySession",
    "build_parser",
    "final_only",
    "grant_tests_that_fit",
    "hosted_ready_session",
    "hosted_runner_cache",
    "hosted_run",
    "main",
    "recipe_kb_from_env",
    "ready_session",
    "require_priced_models",
    "require_rsi_archive",
    "require_session_environment",
    "require_workload_switch",
    "run_session",
    "session_services",
    "settings_from",
]
