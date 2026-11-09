# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and the kernel
optimizer its recipes are scoped to, and the hard limits the judge works within. ``--workload`` names the workload
kind: ``inferencex`` takes ``--isl`` and ``--osl``, and ``agentx`` refuses them, as the trace it replays fixes its
requests; ``agentx`` takes ``--ep``, 1 when not given, and ``--agentx-duration-sec``, the trace window its tests
replay (the measuring harness's default when not given), and ``--agentx-objective``, the axis it is graded on
(``p50_interactivity`` when not given, or ``total_token_throughput``), and ``inferencex`` refuses all three. An agentx session serves its
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
would record tokens only.
``--playbook`` says which scheduling playbook the session plays from the RSI archive ``HYPERLOOM_RSI_ARCHIVE`` names:
``champion``, ``challenger``, which plays the oldest playbook on trial in the pool beside it and the champion while
none is, or ``auto`` (the default), a challenger session when the session's id draws it, half of them, so the
playbooks on trial are played without anyone asking; the manifest fingerprint and the session's manifest record the
label under ``playbook`` in
``policy_versions``, and a challenger's fallback to the champion under ``playbook_fallback``. Under
``HYPERLOOM_RSI_WRITE_ONLY=1`` the session plays none and is judged as before the RSI: no stall evidence, safety
envelope or probation.
Each ``--autonomous-agent SOURCE`` opens one branch where EXPLORE starts, after the user's directions and within
``--max-branches``, whose specialist runs that third-party agent end to end in a sandbox of its own; a source that
carries credentials in its URL is a usage error. ``--agent-directions FILE`` names a directions map that is split, after
PRELUDE, into disjoint shares, one per Arbor agent among them, each its ``component.plan``; a map with no directions, or
one given with no agent, is a usage error. Without one, two or more Arbors split the map of a dossier pass, the first
Arbor run once on a mapping branch EXPLORE opens beside the others, for at most ``--agent-dossier-minutes`` (default
90), or, when it gives none or the minutes are 0, a map Arbor generates from the root's profile (its op shares, else its
kernel shares); each share then opens as a branch of its own, when it still has ``--agent-share-minutes`` (default 63,
0 for always) to its cutoff, the least in which Arbor runs an iteration. A negative or non-finite value of either,
as of any other limit, is a usage error.
``--agent-pack NAME`` names the pack those Arbors run instead of Arbor's serving template; a name that is not one
directory, or one given with no agent, is a usage error, and a pack the Arbor checkout lacks is refused when the branch
starts. ``--agent-launch FILE`` gives those Arbors the launch an AgentX recipe's adds to, as resolved from InferenceX;
a file the adapter cannot use, an unfilled template among them, a base launch record, or one given with no agent or
with ``--reference-script``, is a usage error. ``--reference-script FILE`` is the session's base launch, as for
``hyperloom optimize``: a shell launch script, or a base launch record (``hyperloom-base-launch/1``) frozen elsewhere,
which the session freezes without reading any recipe source; every Arbor is then served the frozen base and checks its
launch against it. A file that cannot serve as a base of the session's framework is a usage error, and one on another
topology than the workload's is refused when the session freezes it. ``--inferencex-commit SHA`` resolves a fresh
session's base launch from InferenceX at that commit instead of the head of its ``main``; one that is not 7 to 40 hex
digits is a usage error. Each ``--specialist-prompt`` file holds a direction the user
wrote for one specialist: its branch opens at PRELUDE's root ahead of the opening batch, counted within
``--max-branches``, its specialist sees the text in every launch prompt, and the judge neither stops it nor lowers its K
for its first judged tests; an empty or unreadable file, or more files than ``--max-branches``, is a usage error.
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
session's runners read the inference_suite staging from ``HYPERLOOM_RUNNER_CACHE``, this host's directory of the runner
cache, which a session measured in hosted sandboxes must set and one on this machine's cards takes as ``/runner-cache``
when unset; the image the session runs on must be named by both
``HYPERLOOM_ACTIVE_IMAGE_REF`` and ``HYPERLOOM_ACTIVE_IMAGE_DIGEST``; the application's inference_suite tool cache
must hold a staging of the InferenceX client that verifies; the framework's package must be installed in the image;
and the agent CLI the specialists run must be installed. A canonical measurement that cannot be taken fails the command
the same way once the session directory exists. Before PRELUDE the session checks the rule pack's facts about outside
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
sandboxes of the host's sandbox service (:mod:`~hyperloom_multibranch.scheduling.hosted_peers`) or on this container's
cards, as the spec says. Started again over a run whose ``session.json`` names a session, it finishes that session as
``final-only`` does. It tells the host the run is over -- ``final: true`` in ``exit.json`` -- once CLOSE ran, and when
the spec, the environment, agents it cannot isolate, a missing package of the ``hosted`` extra or the runner cache keeps
a session from running, with exit code 2 for a refused spec or session and 1 otherwise; a crash leaves no ``exit.json``,
and the host starts the run again. Each command says at start how agents are isolated; manifest and report record it.
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

from hyperloom_common.shared_storage import open_shared_append
from hyperloom.common.agentx_early_stop import EarlyStopMode, early_stop_mode
from hyperloom.common.outside_facts.image_check import ImageCheckFailed
from hyperloom.common.url_userinfo import CREDENTIALED_URL_RE
from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom.hosted.campaign import MEASUREMENT_LOCAL, CampaignSpec
from hyperloom.hosted.host_info import HostGroup, HostInfo, take_host_info
from hyperloom.hosted.toolbox_inputs import ToolboxInputsError
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom.measure.runner.cache_build import CacheBuildError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard.knowledge.reference_leads import INFERENCEX_REFERENCE
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.rulegate.critic import LlmCritic, anthropic_critic
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.hosted_peers import PinnedToolboxInputs, load_hosted_peers
from hyperloom_multibranch.scheduling.hosted_run import (
    GrantDeadline,
    HostedExit,
    arm_grant_stop,
    campaign_spec,
    earlier_session,
    grant_deadline,
    grant_tests_that_fit,
    hosted_storage,
    record_session,
    recorded_session,
    refuse_unhosted_branches,
    reported_stop_reason,
    require_hosted_isolation,
    require_local_image,
    run_argv,
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
    load_peers,
    machine_parallel_tests,
)
from hyperloom_multibranch.scheduling.peers import PeerFactory, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry, adopting_orphans
from hyperloom_multibranch.scheduling.reference import ReferenceSource, reference_source
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.session_environment import (
    EnvironmentIncomplete,
    hosted_runner_cache,
    recipe_kb_from_env,
    require_session_environment,
    require_writable_storage,
    rsi_archive_dir,
)
from hyperloom_multibranch.scheduling.settings import (
    AGENT_PORT,
    DEFAULT_AGENTX_OBJECTIVE,
    DOSSIER_MINUTES,
    SHARE_MINUTES,
    InferenceXShape,
    KernelOptimizer,
    PlaybookMode,
    RunSettings,
    Workload,
)
from hyperloom_multibranch.specialist.arbor_adapter import DOSSIER_INDEX, ArborRefused, read_launch
from inference_suite import UserFileError, parse_inferencex_commit, parse_user_file
from inference_suite.contracts import BaseLaunchError

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
#: The GPU types a session runs on; the toolbox's environment exports the board itself as ``GPU_TYPE``.
MAGPIE_RUNNER_GPU_TYPES = ("mi300x", "mi350x", "mi355x")
#: The options an inferencex workload requires and an agentx one refuses.
SEQUENCE_LENGTHS = ("isl", "osl")
#: The argparse action of an option the user may give more than once, each giving one more value.
REPEATABLE = "append"
#: The argparse action of a switch, an option that takes no value and is on when given.
SWITCH = "store_true"
#: Where ``run`` measures: this machine's cards, or the sandboxes of a grant handed to it.
LOCAL_MEASUREMENT, HOSTED_MEASUREMENT = "local", "hosted"
#: The command a host runs a session under, with the campaign spec it injects.
HOSTED_COMMAND = "hosted"
#: What a grant handed to ``run --measurement hosted`` must give: the sandbox service, the measurement key and
#: templates, the grant's limits and creation time, and the source the runner cache is built from.
HANDED_GRANT = (HostGroup.SANDBOX, HostGroup.MEASUREMENT_KEY, HostGroup.GRANT, HostGroup.SOURCE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Run a multi-branch optimization session.")
    commands = parser.add_subparsers(dest="command", required=True)
    _add_run_options(commands.add_parser("run", help="Run one session: PRELUDE, EXPLORE, FINAL and CLOSE."))
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
    final_only.add_argument("-v", "--verbose", action=SWITCH, help="Log at debug level.")
    hosted = commands.add_parser(HOSTED_COMMAND, help="Run under a host that injects the campaign spec and its inputs.")
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
    _add_playbook_option(run)
    run.add_argument("-v", "--verbose", action=SWITCH, help="Log at debug level.")


def _add_agentx_options(run: argparse.ArgumentParser) -> None:
    """The options only an agentx workload takes."""

    run.add_argument(
        "--ep",
        type=int,
        default=None,
        help=f"Expert-parallel size of an agentx workload, default {DEFAULT_EP}; refused for inferencex.",
    )
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
    _add_agentx_options(run)
    run.add_argument("--precision", required=True, help="Model precision, as hyperloom optimize spells it.")
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
            " (examples/multibranch-arbor-launch). It wins over every recipe source; omitted, the base comes from"
            " InferenceX, then the framework's cookbook, then the default table. Every Arbor --autonomous-agent is"
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
    """The third-party agents some branches run, and what the Arbors among them are given."""

    run.add_argument(
        "--autonomous-agent",
        dest="agents",
        action=REPEATABLE,
        default=None,
        metavar="SOURCE",
        help=(
            "A third-party agent to run end to end on one branch, a git URL (URL@ref for a ref), a local path, or"
            f" {ARBOR_KEYWORD} for the Arbor vendored in this source checkout (src/hyperloom_arbor); repeatable, each"
            " opens one branch where EXPLORE starts, within --max-branches."
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
            " Default: for two or more Arbors, the map of a dossier pass a mapping branch runs in EXPLORE"
            " (--agent-dossier-minutes), else one Arbor generates from the root's profile; each share then opens as a"
            " branch of its own."
        ),
    )
    _add_agent_minutes_options(run)
    _add_arbor_start_options(run)
    _add_agent_review_options(run)


def _add_agent_minutes_options(run: argparse.ArgumentParser) -> None:
    """How long the dossier pass may map the Arbors' directions, and how long their shares must have left to open."""

    run.add_argument(
        "--agent-dossier-minutes",
        dest="agent_dossier_minutes",
        type=float,
        default=DOSSIER_MINUTES,
        metavar="MINUTES",
        help=(
            "Without --agent-directions, how long the dossier pass that maps two or more Arbor agents' directions may"
            " run on its mapping branch before their share branches open, out of EXPLORE's time, while the other"
            " branches work; it ends once its map is set. Arbor starts no"
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
            " and the review, to open: Arbor starts no iteration with less. Below it none opens, and the session says"
            " why. It is also the least an agent branch whose lease finds no free cards must still have to run to"
            f" keep waiting for them; 0 always opens them and waits while any time is left (default {SHARE_MINUTES:g})."
        ),
    )


def _add_arbor_start_options(run: argparse.ArgumentParser) -> None:
    """How an Arbor --autonomous-agent is started: its pack, the launch its recipe adds to, and its servers' ports."""

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
            "For an Arbor --autonomous-agent: the launch an AgentX recipe's adds to, as resolved from InferenceX"
            " (server_flags, server_env, model_name, client; examples/multibranch-arbor-launch). A base launch record"
            " is given as --reference-script, whose frozen base the Arbors are served instead."
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
    """The stages of a person's review of the Arbor agents' map: stop once it is made, run on a reviewed one, or skip
    the review."""

    run.add_argument(
        "--stop-after-map",
        dest="stop_after_map",
        action=SWITCH,
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
            "Open the Arbor --autonomous-agent branches without a person's review: by default they wait once their map"
            " is made, holding no cards, until `hyperloom-multibranch review --session DIR approve` (or reject); the"
            " other branches go on, and a --agent-directions map is reviewed before EXPLORE. With this flag nothing"
            " waits, and its Arbors are started with their own review gate off. For unattended runs, and for the"
            " second stage of a reviewed run."
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
    given = [name for name in ("ep", "agentx_duration_sec", "agentx_objective") if getattr(args, name) is not None]
    if WorkloadKind(args.workload) is WorkloadKind.INFERENCEX:
        return [f"--{name.replace('_', '-')} is not taken by an inferencex workload" for name in given]
    return _positive(args, tuple(name for name in given if name != "agentx_objective"))


def _workload_shape(args: argparse.Namespace) -> InferenceXShape | None:
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX:
        return None
    return InferenceXShape(args.isl, args.osl)


def _workload_model(args: argparse.Namespace) -> str:
    """``--model``, an agentx workload's model directory by its resolved path: the benchmark the session writes for it
    pins the directory so."""

    directory = Path(args.model).expanduser()
    if WorkloadKind(args.workload) is WorkloadKind.AGENTX and directory.is_dir():
        return str(directory.resolve())
    return args.model


#: Schemes whose URL user is a login name rather than a credential; any other scheme's user, or a password, is one.
SSH_SCHEMES = frozenset({"ssh", "git+ssh", "svn+ssh"})
#: The --autonomous-agent source that names the Arbor vendored in this checkout, which a wheel does not ship.
ARBOR_KEYWORD = "arbor"
VENDORED_ARBOR = Path(__file__).resolve().parent.parent / "hyperloom_arbor"


def _agents(sources: list[str], free: int) -> tuple[str, ...]:
    """The agent sources, :data:`ARBOR_KEYWORD` as :data:`VENDORED_ARBOR`'s absolute path; refused when one carries
    credentials, which come from the sandbox's environment only, when there are more than the ``free`` branches the
    user's directions leave, or when :data:`ARBOR_KEYWORD` is given and this install has no vendored Arbor."""

    if len(sources) > free:
        raise ValueError(f"--autonomous-agent is given {len(sources)} times, more than the {free} branches left")
    for source in sources:
        credentialed = CREDENTIALED_URL_RE.match(source)
        if credentialed is None:
            continue
        scheme = credentialed["scheme"].removesuffix("://")
        if ":" in credentialed["userinfo"] or scheme.lower() not in SSH_SCHEMES:
            raise ValueError(f"--autonomous-agent {scheme}://... carries credentials; give them in the environment")
    if ARBOR_KEYWORD in sources and not VENDORED_ARBOR.is_dir():
        raise ValueError(
            f"--autonomous-agent {ARBOR_KEYWORD} names the Arbor vendored at {VENDORED_ARBOR}, which this install does"
            " not have: it is in a source checkout only; give an Arbor checkout's path or URL instead"
        )
    return tuple(str(VENDORED_ARBOR) if source == ARBOR_KEYWORD else source for source in sources)


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


def _agent_launch(path: Path | None, agents: list[str], reference_script: Path | None) -> str:
    """The absolute path of the launch file the Arbor agents are served, ``""`` without one; refused with no agent to
    serve, beside a ``reference_script``, whose frozen base they are served, when it is a base launch record, which
    is given as ``reference_script``, or with every reason the adapter could not use it."""

    if path is None:
        return ""
    if not agents:
        raise ValueError("--agent-launch needs an --autonomous-agent to serve it")
    if reference_script is not None:
        raise ValueError("--agent-launch is not taken with --reference-script: the Arbors are served the frozen base")
    try:
        launch = read_launch(path)
    except ArborRefused as exc:
        raise ValueError(f"--agent-launch: {exc}") from exc
    if launch.record:
        raise ValueError(
            f"--agent-launch {path} is a base launch record; give it as --reference-script, which freezes it as the"
            " session's base and serves it to the Arbors"
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

    errors = _shape_errors(args) + _agentx_only_errors(args) + _positive(args, ("tp", "conc"))
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
        agentx_early_stop=early_stop_mode(os.environ, EarlyStopMode.SHADOW),
    )


def _limit_errors(args: argparse.Namespace) -> list[str]:
    """What is wrong with the session's limits: its hours, GPU hours, LLM dollars, judge interval, agent minutes and
    dev sandboxes."""

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
    ]


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
    handler = _file_handler(layout.scheduler_log)
    logging.getLogger().addHandler(handler)
    try:  # a hosted run that closes this session goes on to start another, which logs to its own directory
        host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
        session = Session(persisted.settings, peer_factory, host, services)
        return (await session.finish(persisted.ended)).exit_code
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _file_handler(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding=ENCODING, delay=True)  # writes to the one stream set below
    handler.stream = os.fdopen(open_shared_append(path), "a", encoding=ENCODING)  # a later leg appends; no link
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


@dataclass(frozen=True)
class ReadySession:
    """The peers a session is built on and the services it calls out to."""

    peer_factory: PeerFactory
    services: SessionServices


def hosted_ready_session(host: HostInfo, workload: Workload, *, local: bool = False) -> ReadySession:
    """:func:`ready_session` measured in the sandboxes of ``host``'s grant, or this container's cards with ``local``:
    the one builder of the peers of ``hosted run`` and ``run --measurement hosted``."""

    cache = hosted_runner_cache(os.environ)
    load = load_peers if local else partial(load_hosted_peers, inputs=PinnedToolboxInputs(host, cache))
    return ready_session(workload, load, hosted_storage(host, os.environ, cache))


def ready_session(
    workload: Workload, load: Callable[[Workload], PeerFactory], storage: Mapping[str, Path] | None = None
) -> ReadySession:
    """The peers ``load`` has for ``workload`` and the services, the Recipe KB checked first, then that this account
    can write the RSI archive and each of ``storage`` (:func:`require_writable_storage`); raises
    :class:`EnvironmentIncomplete` or :class:`PeersUnavailable` naming what is missing. Creates nothing."""

    recipe_kb = recipe_kb_from_env()
    require_writable_storage({**(storage or {}), **rsi_archive_dir(os.environ)})
    peer_factory = load(workload)
    return ReadySession(peer_factory, session_services(recipe_kb))


@dataclass(frozen=True)
class HostedStart:
    """What ``hosted run`` runs, and the deadline of the grant it runs in."""

    run: Callable[[], int]
    deadline: GrantDeadline


def hosted_run() -> int:
    """``hosted run``: the session of the campaign spec the host injected, or the finish of the one an earlier start
    of the run began; writes ``exit.json`` with ``final: true`` once CLOSE ran or when the run cannot be served."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    host = take_host_info(os.environ, require=())
    try:
        start = _hosted_start(host)
    except ValueError as exc:
        return _not_run(host.state_dir, EXIT_USAGE, exc)
    except (EnvironmentIncomplete, PeersUnavailable) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    except ImportError as exc:
        log.exception("hosted run: a module the session needs is not installed")
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    _stop_before_grant_ends(start.deadline)
    try:
        returncode = start.run()
    except (PeersUnavailable, ImageCheckFailed, CacheBuildError, ToolboxInputsError, PermissionError) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    root = recorded_session(host.state_dir)
    stop_reason = None if root is None else reported_stop_reason(root)
    write_final_exit(host.state_dir, HostedExit(returncode, stop_reason=stop_reason))
    return returncode


def _hosted_start(host: HostInfo) -> HostedStart:
    """A new session of the spec, or the close or finish of the one ``session.json`` names (:func:`earlier_session`);
    raises ValueError when the spec or that session is refused, ImportError when a ``hosted`` package is missing,
    and what :func:`require_hosted_isolation` and :func:`ready_session` raise."""

    deadline = grant_deadline(host, datetime.now(timezone.utc))
    spec = campaign_spec(host)
    require_hosted_isolation(spec, os.environ)
    require_local_image(spec, os.environ)
    earlier = earlier_session(host.state_dir)
    if earlier.closed is not None:  # it closed with no baseline, which ended the run, and was killed before exit.json
        return HostedStart(lambda: EXIT_FAILED, deadline)
    persisted = earlier.finish
    settings = _hosted_settings(host, spec, deadline.session_hours) if persisted is None else persisted.settings
    require_session_environment(settings, os.environ)
    ready = hosted_ready_session(host, settings.workload, local=spec.measurement == MEASUREMENT_LOCAL)
    if persisted is not None:
        run = partial(final_only, persisted, ready.peer_factory, ready.services)
    elif earlier.left is not None:
        run = partial(_start_again, earlier.left, settings, ready, host.state_dir)
    else:
        run = partial(run_session, settings, ready.peer_factory, ready.services, partial(_record_root, host.state_dir))
    return HostedStart(run, deadline)


def _start_again(left: PersistedRun, settings: RunSettings, ready: ReadySession, state_dir: Path) -> int:
    """Close ``left``, which ended before its baseline was recorded, with :func:`final_only`, which ends the sandboxes
    of the tests it left, then start the spec's session on ``settings`` unless a stop during the close ended the run."""

    log.warning("%s ended before PRELUDE recorded its baseline: closing it, then starting again", left.layout.root)
    closed = final_only(left, ready.peer_factory, ready.services)
    if reported_stop_reason(left.layout.root) != left.ended.value:
        return closed
    return run_session(settings, ready.peer_factory, ready.services, partial(_record_root, state_dir))


def _stop_before_grant_ends(deadline: GrantDeadline) -> None:
    """Arm this process's stop one CLOSE grace before ``deadline``'s grant ends (at once if passed; never untimed)."""

    if deadline.stop_after_sec is not None:
        arm_grant_stop(max(0.0, deadline.stop_after_sec))


def _hosted_settings(host: HostInfo, spec: CampaignSpec, session_hours: Callable[[float], float]) -> RunSettings:
    local = spec.measurement == MEASUREMENT_LOCAL
    tests_that_fit = machine_parallel_tests if local else grant_tests_that_fit(host)
    argv = run_argv(spec, session_hours(spec.max_hours), tests_that_fit(spec.tp))
    settings = settings_from(_parsed_run(argv), tests_that_fit)
    if not local:
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
    """End a run that cannot be served for good: a start of it again would meet what kept this one from running."""

    print(f"{PROG}: hosted run: {exc}", file=sys.stderr)
    write_final_exit(state_dir, HostedExit(returncode, error=str(exc)))
    return returncode


@dataclass(frozen=True)
class HandedGrant:
    """The grant ``run --measurement hosted`` measures in, how much of it is left, and the session's ``--max-hours``
    cut to what it leaves."""

    host: HostInfo
    deadline: GrantDeadline
    max_hours: float


def _handed_grant(args: argparse.Namespace) -> HandedGrant:
    """The grant handed to ``run --measurement hosted``, taken out of the environment. Raises ValueError naming what
    is missing or refused, never a value."""

    host = take_host_info(os.environ, require=HANDED_GRANT, require_run=False)
    deadline = grant_deadline(host, datetime.now(timezone.utc))
    return HandedGrant(host, deadline, deadline.session_hours(args.max_hours))


def _run_settings(args: argparse.Namespace, grant: HandedGrant | None) -> RunSettings:
    if grant is None:
        return settings_from(args)
    settings = replace(settings_from(args, grant_tests_that_fit(grant.host)), max_hours=grant.max_hours)
    refuse_unhosted_branches(settings, grant.host, os.environ)
    return settings


def _start(settings: RunSettings, grant: HandedGrant | None) -> int:
    if grant is None:
        ready = ready_session(settings.workload, load_peers)
    else:
        ready = hosted_ready_session(grant.host, settings.workload)
        _stop_before_grant_ends(grant.deadline)
    return run_session(settings, ready.peer_factory, ready.services)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == HOSTED_COMMAND:
        return hosted_run()
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
    "HANDED_GRANT",
    "HandedGrant",
    "HostedStart",
    "ReadySession",
    "build_parser",
    "final_only",
    "grant_tests_that_fit",
    "hosted_ready_session",
    "hosted_run",
    "main",
    "ready_session",
    "run_session",
    "session_services",
    "settings_from",
]
