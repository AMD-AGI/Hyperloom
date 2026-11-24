# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-multibranch``: command-line entry for a multi-branch session.

``run`` takes the fixed workload, spelled as ``hyperloom optimize`` spells it, the wall-clock budget and the kernel
optimizer its recipes are scoped to, and the hard limits the judge works within. ``--workload`` names the workload
kind: ``inferencex`` takes ``--isl`` and ``--osl``, and ``agentx`` refuses them, as the trace it replays fixes its
requests; either takes ``--ep``, the expert-parallel size its server runs at, 1 when not given; ``agentx`` takes
``--agentx-duration-sec``, the trace window its tests replay (the measuring harness's default when not given), and
``--agentx-objective``, the axis it is graded on (``p50_interactivity`` when not given, or
``total_token_throughput``), and ``inferencex`` refuses both. An agentx session serves its
model in the image ``HYPERLOOM_ACTIVE_IMAGE_REF`` names, and a ``--model`` that is a directory is recorded by its
resolved path, which its benchmark pins. An operator environment whose ``HYPERLOOM_AGENTX`` says otherwise than
``--workload``, or that names a benchmark YAML in ``HYPERLOOM_BENCHMARK_CONFIG``, is refused.
``--boot-budget-sec`` (at most 6 h; the round raises the server's own start timeout above it) is how long, at least,
a test's server may take to be ready: it only lengthens, so a test runs for its trace window, if any, half an hour more
and that budget when that is longer than its default timeout, and a boot may take the budget when that is longer than
half of the timeout.
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
PRELUDE, into disjoint shares, one per component agent (an agent an installed adapter recognises), each its
``component.plan``; a map with no directions, or one given with no agent, is a usage error. Without one, two or more
component agents split the map of a dossier pass, one run of the first on a mapping branch EXPLORE opens beside the
others, for at most ``--agent-dossier-minutes`` (default 90), or, when it gives none or the minutes are 0, a map the
adapter generates from the root's profile (its op shares, else its kernel shares); each share then opens as a branch of
its own, when it still has ``--agent-share-minutes`` (default 63, 0 for always) to its cutoff, the least in which a
component agent runs one iteration. A negative or non-finite value of either, as of any other limit, is a usage error.
``--agent-pack NAME`` names the pack those component agents run instead of the adapter's default; a name that is not
one directory, or one given with no agent, is a usage error, and a pack the agent's checkout lacks is refused when the
branch starts. The component agents are served the session's frozen base launch, resolved from the recipe sources or
``--reference-script``, and check their launch against it. ``--agent-launch FILE`` serves them that launch file
instead; a file the adapter cannot use, an unfilled template among them, a base launch record, or one given with no
agent or with ``--reference-script``, is a usage error. ``--reference-script FILE`` is the session's base launch, as for
``hyperloom optimize``: a shell launch script, or a base launch record (``hyperloom-base-launch/1``) frozen elsewhere,
which the session freezes without reading any recipe source. A file that cannot serve as a base of the session's
framework is a usage error, and one on another topology than the workload's is refused when the session freezes it. ``--inferencex-commit SHA`` resolves a fresh
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
when unset and publishes the staging into once the session directory exists; the image the session runs on must be named by both
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
need the ``dev`` key (``HYPERLOOM_SANDBOX_KEY_DEV``), ``USER_DATA_PATH``, the storage the ``dev`` template mounts at
the same path, a ``dev`` template of the session's TP in ``HYPERLOOM_SANDBOX_TEMPLATES``, and a grant that fits one
lease beside a test (at least twice the TP in GPUs and two sandboxes); a usage error names what is missing or short.
``hosted run`` takes no arguments: it runs the session of the campaign spec a host injected, as
:mod:`~hyperloom_multibranch.scheduling.hosted_run` maps it onto the options of ``run``, measured in the runner
sandboxes of the host's sandbox service (:mod:`~hyperloom_multibranch.scheduling.hosted_peers`) or on this container's
cards, as the spec says. Started again over a run whose ``session.json`` names a session, it takes it over
(``hosted_restart_leg``): resumed to its first deadline while its baseline holds, else finished as ``final-only`` does
(a FINAL that chooses ends it ``finished_after_restart``, exit 0), or, before its baseline, closed and started again.
It tells the host the run is over -- ``final: true`` in ``exit.json`` -- once CLOSE ran, or when the spec, the
environment, agents it cannot isolate, a missing ``hosted`` package, the runner cache or an unreadable file keeps a
session from running (exit 2 for a refused spec or session, else 1); a crash leaves no ``exit.json`` and is restarted.
``final-only SESSION_DIR`` is the emergency recovery of a session that ended before FINAL ran, and takes nothing but
that directory: it runs FINAL and CLOSE over what the session persisted -- the settings and the frontier policy its
manifest holds, the branches its roster and blackboard hold -- and rewrites its report, dashboard, breakdown and
manifest. It
creates no session directory, so it cannot start one. It carries the stop reason that session ended with, which is
also what its exit code says: a session that did not run its course stays a failed session however FINAL goes. A
session that was killed with its machine wrote no report at all; it ends for ``killed``, and so fails too. The
search tree the session's own CLOSE archived is left as it was. It is refused with exit code 2, having written
nothing, when the directory is no session that ran, ran on another build than this one, committed no root, is still
running -- any process ``MB-pids.jsonl`` records, the scheduler included, still alive in this environment -- already
chose in FINAL, or has another ``final-only`` running over it.
``run --resume SESSION_DIR`` continues such a session on its manifest's settings and no option but ``-v``: no PRELUDE,
new branches on the integration head, the deadline moved by the time it was down. It is refused, holds the lock and
archives as ``final-only`` does, and is refused too as :func:`resumable_run` says.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from hyperloom_common.shared_storage import open_shared_append
from hyperloom.common.outside_facts.image_check import ImageCheckFailed
from hyperloom.hosted.campaign import MEASUREMENT_LOCAL, CampaignSpec
from hyperloom.hosted.host_info import HostGroup, HostInfo, take_host_info
from hyperloom.hosted.toolbox_inputs import ToolboxInputsError
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom.measure.runner.cache_build import CacheBuildError
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient
from hyperloom_multibranch.cli_options import (
    ENCODING,
    FINAL_ONLY_COMMAND,
    HOSTED_COMMAND,
    HOSTED_MEASUREMENT,
    PROG,
    build_parser,
    refuse_options_beside_resume,
    settings_from,
)
from hyperloom_multibranch.rulegate.critic import LlmCritic, anthropic_critic
from hyperloom_multibranch.scheduling.budget import ExploreCap
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.hosted_peers import PinnedToolboxInputs, load_hosted_peers, local_grant_peers
from hyperloom_multibranch.scheduling.hosted_run import (
    GrantDeadline,
    HostedExit,
    campaign_spec,
    close_then_start,
    grant_deadline,
    grant_explore_cap,
    grant_tests_that_fit,
    hosted_restart_leg,
    hosted_storage,
    recorded_session,
    refuse_unhosted_branches,
    reported_stop_reason,
    require_hosted_isolation,
    require_local_image,
    run_argv,
    share_with_the_group,
    spec_make_before_break,
    started_session,
    stop_before_grant_ends,
    take_over_earlier_session,
    write_final_exit,
)
from hyperloom_multibranch.scheduling import review
from hyperloom_multibranch.scheduling.layout import create_session
from hyperloom_multibranch.scheduling.lifecycle import Session
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
from hyperloom_multibranch.scheduling.resumption import (
    PersistedRun,
    deadline_after_downtime,
    persisted_run,
    resumable_run,
)
from hyperloom_multibranch.scheduling.session_environment import (
    EnvironmentIncomplete,
    hosted_runner_cache,
    recipe_kb_from_env,
    require_session_environment,
    require_writable_storage,
)
from hyperloom_multibranch.scheduling.session_types import SessionHost, SessionOutcome, SessionServices
from hyperloom_multibranch.scheduling.specialist_reach import require_specialist_reach
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.scheduling.settings import (
    RunSettings,
    Workload,
)

EXIT_FAILED = 1
EXIT_USAGE = 2
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
log = logging.getLogger(__name__)
#: What a grant handed to ``run --measurement hosted`` must give: the sandbox service, the measurement key and
#: templates, the grant's limits and creation time, and the source the runner cache is built from.
HANDED_GRANT = (HostGroup.SANDBOX, HostGroup.MEASUREMENT_KEY, HostGroup.GRANT, HostGroup.SOURCE)


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
    explore_cap: ExploreCap | None = None,
) -> int:
    """Create the session directory, hand it to ``report_root`` -- by default printed on one stdout line -- before
    anything else is built in it, run the session (:meth:`Session.run`, given ``explore_cap``) and return its exit code.
    A session whose specialists run as an account that cannot reach its directory is refused before it runs
    (:func:`~hyperloom_multibranch.scheduling.specialist_reach.require_specialist_reach`, raising
    :class:`EnvironmentIncomplete`), unless it stops once its map is made: that session starts no specialist.

    The session and its peers are built on the event loop they run on, which the specialist tools of the MCP service
    hand every toolbox call to. Whatever the session's processes orphan is reparented to this process, so the session
    stop kills it and reports it while it runs."""

    with adopting_orphans():
        return asyncio.run(_run_session(settings, peer_factory, services, report_root, explore_cap))


async def _run_session(
    settings: RunSettings,
    peer_factory: PeerFactory,
    services: SessionServices,
    report_root: Callable[[Path], None],
    explore_cap: ExploreCap | None = None,
) -> int:
    clock = Clock()
    layout = create_session(workspace_root(), settings.workload.model_basename, clock)
    report_root(layout.root)
    logging.getLogger().addHandler(_file_handler(layout.scheduler_log))
    if not settings.stop_after_map:
        require_specialist_reach(layout.branches)
    host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
    session = Session(settings, peer_factory, host, services)
    return (await session.run(explore_cap)).exit_code


def final_only(
    persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices, *, restart: bool = False
) -> int:
    """Run FINAL and CLOSE over the session directory ``persisted`` was read from, print it on one stdout line and
    return the session's exit code. No session directory is created, which is why this cannot start one. For
    ``restart``, a hosted run's, see :meth:`Session.finish`."""

    return _over_session(persisted, peer_factory, services, partial(Session.finish, restart=restart))


def resume(
    persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices, until: float, cap: ExploreCap | None
) -> int:
    """:func:`final_only` running :meth:`Session.resume` up to the deadline ``until`` with EXPLORE capped by ``cap``."""

    return _over_session(persisted, peer_factory, services, lambda session, ended: session.resume(ended, until, cap))


#: What a run over a session directory runs on the session it builds there, given how that session ended.
SessionLeg = Callable[[Session, SessionStopReason], Awaitable[SessionOutcome]]


def _over_session(
    persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices, leg: SessionLeg
) -> int:
    """Run ``leg`` over the session directory ``persisted`` was read from, holding the session's lock file while it
    works; a second run over the same directory is refused rather than left to write over the first one's report."""

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
            return asyncio.run(_run_leg(persisted, peer_factory, services, leg))
    finally:
        lock.unlink()


async def _run_leg(
    persisted: PersistedRun, peer_factory: PeerFactory, services: SessionServices, leg: SessionLeg
) -> int:
    clock = Clock()
    layout = persisted.layout
    print(layout.root, flush=True)
    handler = _file_handler(layout.scheduler_log)
    logging.getLogger().addHandler(handler)
    try:  # a hosted run that closes this session goes on to start another, which logs to its own directory
        host = SessionHost(layout, ProcessRegistry(layout.pids, clock), clock)
        session = Session(persisted.settings, peer_factory, host, services)
        return (await leg(session, persisted.ended)).exit_code
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _file_handler(path: Path) -> logging.Handler:
    os.close(open_shared_append(path))  # 664: final-only of a later leg, maybe another account, appends to it
    handler = logging.FileHandler(path, encoding=ENCODING)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


@dataclass(frozen=True)
class ReadySession:
    """The peers a session is built on and the services it calls out to."""

    peer_factory: PeerFactory
    services: SessionServices


def hosted_ready_session(host: HostInfo, workload: Workload, *, local: bool = False) -> ReadySession:
    """:func:`ready_session` measured in the sandboxes of ``host``'s grant, or this container's cards with ``local``:
    the one builder of the peers of ``hosted run`` and ``run --measurement hosted``. Either way the tests end by the
    stop of the grant (:func:`local_grant_peers`)."""

    cache = hosted_runner_cache(os.environ)
    load = local_grant_peers(host) if local else partial(load_hosted_peers, inputs=PinnedToolboxInputs(host, cache))
    return ready_session(workload, load, **hosted_storage(cache))


def ready_session(workload: Workload, load: Callable[[Workload], PeerFactory], **storage: Path) -> ReadySession:
    """The peers ``load`` has for ``workload`` and the services, the Recipe KB, the RSI archive and the directories of
    ``storage`` checked first; raises :class:`EnvironmentIncomplete` or :class:`PeersUnavailable` naming the problem."""

    recipe_kb = recipe_kb_from_env()
    require_writable_storage(os.environ, storage)
    return ReadySession(load(workload), session_services(recipe_kb))


@dataclass(frozen=True)
class HostedStart:
    """What ``hosted run`` runs."""

    run: Callable[[], int]


def hosted_run() -> int:
    """``hosted run``: the session of the injected campaign spec, or the resume or finish of the one an earlier start
    of the run began; writes ``exit.json`` with ``final: true`` once CLOSE ran or when the run cannot be served."""

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    host = take_host_info(os.environ, require=())
    try:
        start = _hosted_start(host)
    except ValueError as exc:
        return _not_run(host.state_dir, EXIT_USAGE, exc)
    except (EnvironmentIncomplete, PeersUnavailable, PermissionError) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    except ImportError as exc:
        log.exception("hosted run: a module the session needs is not installed")
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    _stop_before_grant_ends(host)
    try:
        returncode = start.run()
    except (
        EnvironmentIncomplete,
        PeersUnavailable,
        ImageCheckFailed,
        CacheBuildError,
        ToolboxInputsError,
        PermissionError,
    ) as exc:
        return _not_run(host.state_dir, EXIT_FAILED, exc)
    root = recorded_session(host.state_dir)
    stop_reason = None if root is None else reported_stop_reason(root)
    write_final_exit(host.state_dir, HostedExit(returncode, stop_reason=stop_reason))
    return returncode


def _hosted_start(host: HostInfo) -> HostedStart:
    """A new session of the spec, or the close, resume or finish of the one ``session.json`` names, which the restart
    takes over (:func:`take_over_earlier_session`), measured where the spec says; raises ValueError on a refused spec or
    session, ImportError when a ``hosted`` package is missing, and what :func:`require_hosted_isolation` and
    :func:`ready_session` raise."""

    deadline = grant_deadline(host, datetime.now(timezone.utc))
    spec = campaign_spec(host)
    require_hosted_isolation(spec, os.environ)
    require_local_image(spec, os.environ)
    earlier = take_over_earlier_session(host.state_dir)
    if earlier.closed is not None:  # it closed with no baseline, which ended the run, and was killed before exit.json
        return HostedStart(lambda: EXIT_FAILED)
    persisted = earlier.finish
    if (taken := persisted or earlier.left) is not None:
        share_with_the_group(taken.layout.root)
    settings = _hosted_settings(host, spec, deadline.session_hours) if persisted is None else persisted.settings
    require_session_environment(settings, os.environ)
    ready = hosted_ready_session(host, settings.workload, local=spec.measurement == MEASUREMENT_LOCAL)
    if persisted is not None:
        leg = hosted_restart_leg(persisted, partial(final_only, restart=True), resume, grant_explore_cap(host))
        return HostedStart(partial(leg, persisted, ready.peer_factory, ready.services))
    close = partial(final_only, peer_factory=ready.peer_factory, services=ready.services)
    record = partial(started_session, host.state_dir)
    start = partial(run_session, settings, ready.peer_factory, ready.services, record, grant_explore_cap(host))
    return HostedStart(partial(close_then_start, earlier.left, close, start))


def _stop_before_grant_ends(host: HostInfo) -> None:
    """Arm this process's stop one CLOSE grace before ``host``'s grant ends (at once if passed; never untimed). What is
    left of the grant is read now, as the timer starts, not when the start was prepared: that took time of its own."""

    stop_before_grant_ends(grant_deadline(host, datetime.now(timezone.utc)))


def _hosted_settings(host: HostInfo, spec: CampaignSpec, session_hours: Callable[[float], float]) -> RunSettings:
    local = spec.measurement == MEASUREMENT_LOCAL
    tests_that_fit = machine_parallel_tests if local else grant_tests_that_fit(host, spec_make_before_break(spec))
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


def _not_run(state_dir: Path, returncode: int, exc: Exception) -> int:
    """End a run no session ran in for good: a start of it again would meet what kept this one from running."""

    log.error("hosted run: not served", exc_info=exc)
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
    fits = grant_tests_that_fit(grant.host, args.make_before_break)
    settings = replace(settings_from(args, fits), max_hours=grant.max_hours)
    refuse_unhosted_branches(settings, grant.host, os.environ)
    return settings


def _start(settings: RunSettings, grant: HandedGrant | None) -> int:
    if grant is None:
        ready = ready_session(settings.workload, load_peers)
    else:
        ready = hosted_ready_session(grant.host, settings.workload)
        _stop_before_grant_ends(grant.host)
    cap = None if grant is None else grant_explore_cap(grant.host)
    return run_session(settings, ready.peer_factory, ready.services, explore_cap=cap)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == HOSTED_COMMAND:
        return hosted_run()
    if args.command == "review":
        return review.main(args)
    persisted = grant = None
    try:
        if args.command == FINAL_ONLY_COMMAND:
            persisted = persisted_run(args.session_dir)
            settings = persisted.settings
        elif args.resume is not None:
            refuse_options_beside_resume(args)
            persisted = resumable_run(args.resume)
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
            if args.command == FINAL_ONLY_COMMAND:
                return final_only(persisted, ready.peer_factory, ready.services)
            deadline = deadline_after_downtime(persisted.layout, Clock().now())
            require_specialist_reach(persisted.layout.branches)
            return resume(persisted, ready.peer_factory, ready.services, deadline, None)
        return _start(settings, grant)
    except (EnvironmentIncomplete, PeersUnavailable, ImageCheckFailed, CacheBuildError, ToolboxInputsError) as exc:
        print(f"{PROG}: cannot run a session, {exc}", file=sys.stderr)
        return EXIT_FAILED


__all__ = [
    "HANDED_GRANT",
    "HandedGrant",
    "HostedStart",
    "ReadySession",
    "final_only",
    "grant_tests_that_fit",
    "hosted_ready_session",
    "hosted_run",
    "main",
    "ready_session",
    "resume",
    "run_session",
    "session_services",
]
