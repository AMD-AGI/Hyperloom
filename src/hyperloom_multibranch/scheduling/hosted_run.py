# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What ``hyperloom-multibranch hosted run`` reads from the host and writes back to it.

The campaign spec maps onto the options of ``run`` (:func:`run_argv`), so a hosted session is built by the one builder
of :class:`~hyperloom_multibranch.scheduling.settings.RunSettings` a command-line session is. Besides the common part
the legacy entry validated, a multi-branch session needs ``multibranch.workload`` (``--workload``),
``workload.conc`` and ``workload.precision``; ``multibranch.inferencex_commit`` is ``--inferencex-commit``,
``multibranch.max_branches`` ``--max-branches``,
``multibranch.dev_leases`` ``--dev-leases``, each source of ``multibranch.autonomous_agents`` one
``--autonomous-agent``, ``multibranch.make_before_break`` (``true`` by default) ``--make-before-break`` or
``--no-make-before-break``, and ``placement.max_parallel_tests`` is ``--max-parallel-tests``, at most what the grant
runs at once. The fields ``run`` has no option for -- ``objective``, ``options``, ``placement.profile`` and
``placement.specialist_capacity`` -- are not read. A spec that sets ``launch.server_args`` or ``launch.extra_env`` is
refused rather than run without them: a multi-branch session resolves its base launch from the recipe sources, as
``run`` does, and reads neither (:func:`run_argv`). Dev sandboxes and agents run in ``dev`` sandboxes of the grant, so
they need its ``dev`` key and the session directory on the storage ``USER_DATA_PATH`` names, which the ``dev``
template mounts at the same path (:func:`refuse_unhosted_branches`).
Every test, PRELUDE's included, runs in a measurement sandbox, so the application's container needs no card:
``hosted-cpu`` serves as ``hosted-gpu`` does.

The grant bounds the session in time (:func:`grant_deadline`): its budget ends one CLOSE grace and a margin before the
grant does, and the process is sent SIGTERM one CLOSE grace before the grant ends, which ends FINAL and leaves CLOSE
that grace. The grant's runtime counts from the run's creation (``HYPERLOOM_RUN_CREATED_AT``), which the platform
stamps when it makes the run, before this process starts, so a session's own deadline, counted from its start, can
leave FINAL less of the grant than it needs. The grant's end (:func:`grant_end`) therefore also caps EXPLORE: it ends a
FINAL reserve (:func:`final_reserve_seconds`) and a margin before that SIGTERM, whichever comes first of that and the
end the session's budget gives it, and the tests' infra retries never wait past the time a test can still end by it
(:attr:`GrantEnd.tests_end_by`). The reserve is the validation FINAL would run as the session stands, which grows as
candidates and regression cases come, so EXPLORE's end is asked again while it runs.

A hosted run's agents must be isolated: they run with the application's credentials in reach otherwise. Its start
marks the process so (:func:`~hyperloom.common.agent_isolation.require_agent_isolation`), with the hosting layer's
bubblewrap, ``bin/bwrap`` in the directory ``HYPERLOOM_HOST_LAYER`` names
(:func:`~hyperloom.common.agent_isolation.layer_bwrap`), and no other, and refuses a run that would launch anything
unconfined on this host, as the path that launches it would (:func:`require_hosted_isolation`): its specialists run
here, its agent and dev sandboxes and its tests in the grant's sandboxes.

The host decides from ``$HYPERLOOM_STATE_DIR/exit.json`` whether to start the run again: never once it holds
``final: true`` (:func:`write_final_exit`), and again when it is missing. ``session.json`` there names the session
directory as soon as it exists (:func:`record_session`), so a start that finds it finishes that session instead of
starting another, once that session got through PRELUDE; one that ended before its baseline was recorded is closed,
which ends the measurement sandboxes it left, and the spec's session starts again. The two file names and their fields
are the legacy entry's, ``hyperloom.hosted.entry``, which this package does not import: importing it loads the legacy
flow's state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.shared_storage import child_umask, ownership
from hyperloom.common.agent_isolation import IsolationRefused, layer_bwrap, require_agent_isolation
from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import GrantLimits as HostGrantLimits
from hyperloom.hosted.host_info import HostInfo, Var
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.scheduling.budget import (
    MB_CLOSE_GRACE_CAP_SECONDS,
    MB_SECONDS_PER_HOUR,
    ExploreCap,
    FinalWork,
)
from hyperloom_multibranch.scheduling.launch_isolation import launch_isolation
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.lifecycle import PersistedRun, persisted_run, persisted_session, recorded_baseline
from hyperloom_multibranch.scheduling.manifest import read_manifest
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.resumption import ResumeRefused, require_resumable
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.toolbox.hosted_leases import GrantLimits
from hyperloom_multibranch.toolbox.resume_guard import BaselineNotReusable, check_hosted_baseline_reusable

# hyperloom/hosted/entry.py EXIT_FILE and SESSION_FILE
EXIT_FILE = "exit.json"
SESSION_FILE = "session.json"
#: Seconds between the session's hard stop and the grant's end, for the process to write ``exit.json`` and exit.
GRANT_MARGIN_SECONDS = 60.0
#: The attempts of a validation test FINAL's reserve covers, once per FINAL however many tests it runs: two that meet
#: an infrastructure failure a retry waits out (cards a process outside the test still holds, a platform with no sandbox
#: to give), and the one that measures.
FINAL_ATTEMPTS = 3
#: A fresh sandbox's create and its runner's start, once, which a test measured on a warm node may not have included.
FINAL_COLD_START_SECONDS = 300.0
#: From EXPLORE's end to FINAL's first test: the branches stopped, the candidates read and the last merges settled. A
#: live run took over five minutes for it.
FINAL_WIND_DOWN_SECONDS = 600.0
#: FINAL's reserve at the least, and all of it while the session has measured no test.
FINAL_RESERVE_FLOOR_SECONDS = 1800.0
#: The blackboard writer name a restart reads the baseline as.
RESTART_READER = "restart"
#: ``exit.json`` and ``session.json`` reach the disk before their names, and are ``664``: the run's next leg, maybe
#: another account of the group, reads ``session.json`` to finish the session it names.
_DURABLE = WriteOptions(fsync=True, shared=True)
_ENCODING = "utf-8"

log = logging.getLogger(__name__)


class HostedRefused(ValueError):
    """The run cannot be served as the host gave it, and a restart would not change that; names the field."""


def campaign_spec(host: HostInfo) -> CampaignSpec:
    """The campaign spec ``host`` injected; raises :class:`~hyperloom.hosted.campaign.CampaignError` naming every
    problem with it."""

    return parse_campaign(json.loads(host.input_file.read_text(encoding=_ENCODING)))


def run_argv(spec: CampaignSpec, max_hours: float, parallel_tests: int) -> list[str]:
    """The ``run`` command line of ``spec``, its session lasting ``max_hours`` and running at most ``parallel_tests``
    tests at once; every value is written ``--option=value`` so that one starting with ``-`` is no option.

    Raises:
        HostedRefused: When the spec lacks what a multi-branch session needs, naming each missing field, or sets a
            launch field the session would not read.
    """

    section = _multibranch_section(spec.multibranch)
    _refuse_launch_fields(spec)
    values: dict[str, Any] = {
        "model": spec.model_path,
        "framework": spec.framework_name,
        "gpu-type": spec.gpu_type,
        "workload": section.get("workload"),
        "tp": spec.tp,
        "conc": spec.conc,
        "precision": spec.precision,
    }
    required = {"workload": "multibranch.workload", "conc": "workload.conc", "precision": "workload.precision"}
    missing = [field for option, field in required.items() if values[option] is None]
    if missing:
        raise HostedRefused(f"a multibranch run needs {', '.join(missing)}")
    values |= {
        "isl": spec.isl,
        "osl": spec.osl,
        "ep": spec.ep,
        "inferencex-commit": _inferencex_commit(section.get("inferencex_commit")),
        "max-hours": max_hours,
        "max-gpu-hours": spec.max_gpu_hours,
        "max-llm-usd": spec.max_llm_usd,
        "max-branches": section.get("max_branches"),
        "dev-leases": section.get("dev_leases"),
        "max-parallel-tests": min(spec.max_parallel_tests, parallel_tests),
    }
    agents = [f"--autonomous-agent={source}" for source in _agent_sources(section.get("autonomous_agents"))]
    switches = _switches(section)
    return [
        "run",
        *(f"--{option}={value}" for option, value in values.items() if value is not None),
        *agents,
        *switches,
    ]


def _switches(section: Mapping[str, Any]) -> list[str]:
    """The on/off options of ``section``: ``multibranch.make_before_break`` as ``--[no-]make-before-break``, nothing
    when the spec leaves it to ``run``'s default (on)."""

    keep = _make_before_break(section)
    return [] if keep is None else ["--make-before-break" if keep else "--no-make-before-break"]


def spec_make_before_break(spec: CampaignSpec) -> bool:
    """The operator's ``multibranch.make_before_break`` of ``spec``, on when it is not given; raises
    :class:`HostedRefused` when it is neither true nor false."""

    keep = _make_before_break(_multibranch_section(spec.multibranch))
    return True if keep is None else keep


def _make_before_break(section: Mapping[str, Any]) -> bool | None:
    keep = section.get("make_before_break")
    if keep is not None and not isinstance(keep, bool):
        raise HostedRefused("multibranch.make_before_break must be true or false")
    return keep


def _inferencex_commit(commit: object) -> str | None:
    if commit is not None and not isinstance(commit, str):
        raise HostedRefused("multibranch.inferencex_commit must be a string")
    return commit


def _refuse_launch_fields(spec: CampaignSpec) -> None:
    given = [
        field
        for field, value in (("launch.server_args", spec.server_args), ("launch.extra_env", spec.extra_env))
        if value is not None
    ]
    if given:
        raise HostedRefused(
            f"a multibranch run does not read {' or '.join(given)}: its base launch is resolved from the recipe sources"
            " (InferenceX, the framework's cookbook, the default table), as hyperloom-multibranch run resolves it"
        )


def _agent_sources(sources: object) -> list[str]:
    if sources is None:
        return []
    if not isinstance(sources, list) or not all(isinstance(source, str) and source.strip() for source in sources):
        raise HostedRefused("multibranch.autonomous_agents must be a list of agent sources")
    return sources


def refuse_unhosted_branches(settings: RunSettings, host: HostInfo, env: Mapping[str, str]) -> None:
    """Refuse a session whose dev sandboxes or autonomous agents could not run in ``dev`` sandboxes of ``host``'s
    grant: one the grant gives no ``dev`` key or no limits for, or whose session directory is not where ``env``'s
    ``USER_DATA_PATH`` names, the storage the ``dev`` template mounts at the same path.

    Raises:
        HostedRefused: Naming every variable that is missing.
    """

    if not settings.dev_leases and not settings.agents:
        return
    given = {
        Var.SANDBOX_KEY_DEV: host.sandbox_keys.dev,
        Var.GRANT_LIMITS: host.grant_limits,
        ENV_USER_DATA_PATH: env.get(ENV_USER_DATA_PATH, "").strip(),
    }
    missing = [str(name) for name, value in given.items() if not value]
    if missing:
        raise HostedRefused(
            f"dev sandboxes and autonomous agents run in the grant's dev sandboxes and need {', '.join(missing)}"
        )


def require_hosted_isolation(env: Mapping[str, str]) -> None:
    """Mark this process as a hosted run's, whose agents must be isolated, with the hosting layer's bubblewrap
    (:func:`~hyperloom.common.agent_isolation.layer_bwrap`) and no other, and refuse now a run that would launch one
    unconfined on this host, as the path that launches it would
    (:func:`~hyperloom_multibranch.scheduling.launch_isolation.launch_isolation`): its specialists; its agent and dev
    sandboxes and its tests run in the grant's sandboxes.

    Raises:
        PeersUnavailable: A launch on this host would run unconfined or be refused, naming which and why; the run ends
            for good.
        TimeoutError: Bubblewrap gave no answer in time; a later start of the run may find it.
    """

    try:
        require_agent_isolation(bwrap=layer_bwrap(env))
    except IsolationRefused as refused:
        if refused.transient:
            raise TimeoutError(str(refused)) from refused
        raise PeersUnavailable(str(refused)) from refused
    problems = launch_isolation(local=False).problems()
    if problems:
        raise PeersUnavailable(f"a hosted run launches nothing unconfined beside its credentials, and here {problems}")


def _multibranch_section(section: object) -> dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise HostedRefused("multibranch must be an object")
    return section


def grant_tests_that_fit(host: HostInfo, make_before_break: bool = True) -> Callable[[int], int]:
    """How many tests of a TP the grant ``host`` holds runs at once, as the hosted toolbox runs them (``handover_for``
    with ``make_before_break``); raises ValueError when one does not fit, ImportError without the ``hosted`` extra."""

    # hosted_toolbox loads the sandbox SDK, which only the hosted extra installs.
    from hyperloom_multibranch.toolbox.hosted_toolbox import handover_for, parallel_tests_for

    grant = toolbox_grant(host.grant_limits)

    def tests_that_fit(tp: int) -> int:
        return parallel_tests_for(grant, tp, handover_for(grant, tp) if make_before_break else None)

    return tests_that_fit


def toolbox_grant(limits: HostGrantLimits | None) -> GrantLimits | None:
    """The grant ``limits`` as the hosted toolbox reads it, ``None`` when the host gave none.

    Raises:
        HostedRefused: When the grant states only some of its totals: the toolbox needs all of them or none.
        ValueError: When a total is below what one test needs.
    """

    if limits is None:
        return None
    totals = {
        "max_sandboxes": limits.max_sandboxes,
        "max_gpus": limits.max_gpus,
        "gpu_hours": limits.gpu_hours,
        "max_runtime_sec": limits.max_runtime_sec,
    }
    unstated = [name for name, total in totals.items() if total is None]
    if unstated:
        raise HostedRefused(f"{Var.GRANT_LIMITS} states no {', '.join(unstated)}; a hosted session needs all or none")
    return GrantLimits(**totals)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GrantDeadline:
    """The seconds left in the grant, ``None`` when it states no runtime."""

    left_sec: float | None

    @property
    def stop_after_sec(self) -> float | None:
        """When the process is sent SIGTERM, in seconds from now: one CLOSE grace before the grant ends."""

        return None if self.left_sec is None else self.left_sec - MB_CLOSE_GRACE_CAP_SECONDS

    def session_hours(self, spec_hours: float) -> float:
        """``spec_hours``, cut so that the session's hard stop -- its deadline and one CLOSE grace -- comes a margin
        before the grant ends; ``0``, no limit of its own, is all the grant leaves.

        Raises:
            HostedRefused: When the grant has no more time left than a CLOSE grace and the margin.
        """

        if self.left_sec is None:
            return spec_hours
        usable = self.left_sec - MB_CLOSE_GRACE_CAP_SECONDS - GRANT_MARGIN_SECONDS
        if usable <= 0:
            raise HostedRefused(
                f"the grant has {max(self.left_sec, 0.0):.0f} s left, no more than a CLOSE grace of"
                f" {MB_CLOSE_GRACE_CAP_SECONDS:.0f} s and a margin of {GRANT_MARGIN_SECONDS:.0f} s"
            )
        usable_hours = usable / MB_SECONDS_PER_HOUR
        return min(spec_hours, usable_hours) if spec_hours else usable_hours


def grant_deadline(host: HostInfo, now: datetime) -> GrantDeadline:
    """How much of the grant ``host`` states is left at ``now``: its runtime from when the run was created."""

    limits = host.grant_limits
    if limits is None or limits.max_runtime_sec is None:
        return GrantDeadline(None)
    return GrantDeadline(limits.max_runtime_sec - (now - host.run_created_at).total_seconds())


def final_reserve_seconds(work: FinalWork) -> float:
    """The seconds of the grant FINAL is kept for the validation ``work`` it would run: its wind-down from EXPLORE
    (:data:`FINAL_WIND_DOWN_SECONDS`), one cold start (:data:`FINAL_COLD_START_SECONDS`), its tests queued on the slots
    FINAL has (:attr:`~hyperloom_multibranch.scheduling.budget.FinalWork.run_seconds`), and once per FINAL the retries
    of :data:`FINAL_ATTEMPTS`, each a test at the primary conc and the infra backoff before it
    (:data:`~hyperloom_multibranch.toolbox.attempts.INFRA_BACKOFF_SECONDS`, as
    :class:`~hyperloom_multibranch.toolbox.attempts.InfraBackoff` waits it); no less than
    :data:`FINAL_RESERVE_FLOOR_SECONDS`."""

    # attempts loads the measurement's failure names, which load the main flow's runtime: the command imports none.
    from hyperloom_multibranch.toolbox.attempts import INFRA_BACKOFF_SECONDS

    waits = INFRA_BACKOFF_SECONDS
    backoff = sum(waits[min(retry, len(waits)) - 1] for retry in range(1, FINAL_ATTEMPTS)) if waits else 0.0
    retries = (FINAL_ATTEMPTS - 1) * max(work.test_seconds, 0.0) + backoff
    needed = FINAL_WIND_DOWN_SECONDS + FINAL_COLD_START_SECONDS + work.run_seconds + retries
    return max(FINAL_RESERVE_FLOOR_SECONDS, needed)


@dataclass(frozen=True)
class GrantEnd:
    """When the grant ends, Unix seconds: its runtime from the run's creation."""

    at: float

    @property
    def tests_end_by(self) -> float:
        """When the session's tests must have ended: when the process is sent SIGTERM, one CLOSE grace before the
        grant ends (:attr:`GrantDeadline.stop_after_sec`), which ends FINAL."""

        return self.at - MB_CLOSE_GRACE_CAP_SECONDS

    def explore_cap(self, work: FinalWork) -> float:
        """The latest EXPLORE may end at for FINAL to end a margin before :attr:`tests_end_by`, given the validation
        ``work`` it would run: an :data:`~hyperloom_multibranch.scheduling.budget.ExploreCap`."""

        return self.tests_end_by - GRANT_MARGIN_SECONDS - final_reserve_seconds(work)


def grant_end(host: HostInfo) -> GrantEnd | None:
    """When the grant ``host`` states ends, ``None`` when it states no runtime. Every start of the run reads the same
    creation time, so a restart's grant ends when its first start's did."""

    limits = host.grant_limits
    if limits is None or limits.max_runtime_sec is None:
        return None
    return GrantEnd(host.run_created_at.timestamp() + limits.max_runtime_sec)


def grant_tests_end_by(host: HostInfo) -> float | None:
    """:attr:`GrantEnd.tests_end_by` of ``host``'s grant, ``None`` without a grant runtime."""

    end = grant_end(host)
    return None if end is None else end.tests_end_by


def grant_explore_cap(host: HostInfo) -> ExploreCap | None:
    """:meth:`GrantEnd.explore_cap` of ``host``'s grant, ``None`` without a grant runtime."""

    end = grant_end(host)
    return None if end is None else end.explore_cap


def arm_grant_stop(seconds: float) -> threading.Timer:
    """Send this process SIGTERM in ``seconds``, the stop signal a session ends on; returns the started timer."""

    timer = threading.Timer(seconds, os.kill, (os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    return timer


def stop_before_grant_ends(deadline: GrantDeadline) -> None:
    """Arm this process's stop one CLOSE grace before ``deadline``'s grant ends (at once if passed; never untimed)."""

    if deadline.stop_after_sec is not None:
        arm_grant_stop(max(0.0, deadline.stop_after_sec))


@dataclass(frozen=True)
class HostedExit:
    """How the run ended for good: its return code, and the stop reason its session reported or why it was refused."""

    returncode: int
    stop_reason: str | None = None
    error: str | None = None


def write_final_exit(state_dir: Path, ending: HostedExit) -> None:
    """Tell the host the run is over and must not be started again."""

    detail = {"error": ending.error} if ending.error is not None else {"stop_reason": ending.stop_reason}
    document = {"final": True, "returncode": ending.returncode, **detail}
    atomic_write_json(state_dir / EXIT_FILE, document, options=_DURABLE)


def record_session(state_dir: Path, root: Path) -> None:
    """Name the run's session directory in ``session.json``, for the host and for a restart."""

    atomic_write_json(state_dir / SESSION_FILE, {"session_dir": str(root)}, options=_DURABLE)


def share_with_the_group(root: Path) -> None:
    """Below shared storage, run the rest of this process under the umask of a child that writes below the session
    directory ``root`` (:func:`~hyperloom_common.shared_storage.child_umask`), so every process it starts here, the
    commands of its local agent sandboxes among them, inherits it: what Arbor, an agent or a tool writes there with the
    default modes stays writable by the group, whose other accounts may run the run's next leg. A remote sandbox's
    commands run in another container and take the umask from its launcher instead. Elsewhere the umask is left as
    it is."""

    umask = child_umask(root)
    if umask != -1:
        os.umask(umask)


def started_session(state_dir: Path, root: Path) -> None:
    """Take the session just created in ``root`` as the run's, before anything is built in it: shared with the group
    (:func:`share_with_the_group`), printed on one stdout line and named in ``session.json`` (:func:`record_session`)."""

    share_with_the_group(root)
    print(root, flush=True)
    record_session(state_dir, root)


def recorded_session(state_dir: Path) -> Path | None:
    """The session directory an earlier start of the run named, ``None`` when none did.

    Raises:
        HostedRefused: When this account cannot read ``session.json``, naming it with its owner and mode: an earlier
            leg of another account wrote it owner-only, and a restart would meet the same file.
    """

    path = state_dir / SESSION_FILE
    try:
        text = path.read_text(encoding=_ENCODING)
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise HostedRefused(f"this account cannot read {path}{ownership(path)}: {exc.strerror}") from None
    return Path(json.loads(text)["session_dir"])


@dataclass(frozen=True)
class EarlierSession:
    """The session an earlier start of the run began: ``finish``, which a restart finishes, when it got through
    PRELUDE; ``left``, which a restart closes before it starts the spec's session again, when it committed its root
    and ended before its baseline was recorded without closing, or was closed by such a restart; ``closed``, the
    directory of one that closed on its own before its baseline was recorded, which ended the run; none of them when
    there was none or it committed no root, and so started no test."""

    finish: PersistedRun | None = None
    left: PersistedRun | None = None
    closed: Path | None = None


def take_over_earlier_session(state_dir: Path) -> EarlierSession:
    """The session ``session.json`` names, read back and taken over from the earlier start; raises what :func:`recorded_session` and
    :func:`~hyperloom_multibranch.scheduling.lifecycle.persisted_run` raise. Blocks.

    The host starts a run again only once its earlier start has ended, so a ``final-only`` lock that start left in the
    session is that start's, whose pid names a process of a machine that is gone: it is removed, or the finish would be
    refused for good. A session that is closed and started again holds no measurement this build has to read back, so
    it may have run on another build: a restart onto a new build is how a fix reaches a run that is failing."""

    previous = recorded_session(state_dir)
    if previous is None or not SessionLayout(previous).root_recipe.is_file():
        return EarlierSession()
    lock = SessionLayout(previous).final_only_lock
    if lock.is_file():
        log.warning("removing %s, which the earlier start of this run left", lock)
        lock.unlink(missing_ok=True)
    if prelude_landed(previous):
        return EarlierSession(finish=persisted_run(previous))
    reported = reported_stop_reason(previous)
    if reported not in (None, SessionStopReason.KILLED.value):
        return EarlierSession(closed=previous)
    return EarlierSession(left=persisted_session(previous))


def close_then_start(left: PersistedRun | None, close: Callable[[PersistedRun], int], start: Callable[[], int]) -> int:
    """Close ``left``, a session that ended before its baseline was recorded, when there is one, with ``close``
    (``final-only``, which ends the sandboxes of the tests it left; :func:`close_before_starting_again`), then ``start``
    the spec's session, unless a stop during the close ended the run: then the close's exit code."""

    if left is not None and (ended := close_before_starting_again(partial(close, left), left)) is not None:
        return ended
    return start()


def hosted_resume_deadline(persisted: PersistedRun, now: float) -> float:
    """The deadline a restart of the run resumes ``persisted`` up to, the ``finish`` of :func:`take_over_earlier_session`, with
    ``cli.resume``: the one the run's first start set, so the session's budget counts from that start for
    ``--max-hours`` as it does for ``--max-gpu-hours``, whose spend the session reads from its blackboard, and the time
    it was down is not given back. That the earlier start is gone is the host's to know -- it starts a run again only
    once its earlier start ended -- not the pids that start recorded in another container. Blocks.

    Raises:
        ResumeRefused: When :func:`~hyperloom_multibranch.scheduling.resumption.require_resumable` does, or ``now`` is
            past that deadline; the restart finishes the session instead.
    """

    require_resumable(persisted)
    deadline = read_manifest(persisted.layout.manifest).deadline
    if now >= deadline:
        raise ResumeRefused(
            f"the session in {persisted.layout.root} used its budget, which ended {now - deadline:.0f} s ago"
        )
    return deadline


def hosted_restart_leg(
    persisted: PersistedRun, finish: Callable[..., int], resume: Callable[..., int]
) -> Callable[..., int]:
    """What a restart of the run runs over ``persisted``, the ``finish`` of :func:`take_over_earlier_session`: ``resume``
    (``cli.resume``) with ``deadline`` bound to :func:`hosted_resume_deadline`'s, so its budget counts from the run's
    first start, when the session was measured in the grant's sandboxes and its baseline still measures what a test
    measured now would (:func:`~hyperloom_multibranch.toolbox.resume_guard.check_hosted_baseline_reusable`: the same
    build, ``MEASUREMENT_EPOCH``, image, base launch and runner template cards; not the node or the account, which a
    reschedule changes by design); else ``finish`` (``cli.final_only``), and one warning names why. Writes nothing.
    Blocks."""

    root = persisted.layout.root
    if not persisted.layout.runner_source.exists():  # what a session measured in the grant's sandboxes cloned
        log.warning("hosted run: finishing %s, whose tests did not run in the grant's sandboxes", root)
        return finish
    try:
        check_hosted_baseline_reusable(persisted.layout.toolbox, persisted.settings.workload.tp)
        deadline = hosted_resume_deadline(persisted, time.time())
    except (BaselineNotReusable, ResumeRefused) as exc:
        log.warning("hosted run: finishing %s instead of resuming it: %s", root, exc)
        return finish
    log.info("hosted run: resuming %s up to the deadline its first start set", root)
    return partial(resume, deadline=deadline)


#: What reading back an earlier session fails with when this account cannot read what it left: a file an earlier leg of
#: another account of the group made owner-only, or one that is not the text it wrote.
CANNOT_READ_SESSION = (PermissionError, UnicodeDecodeError, json.JSONDecodeError)


def close_before_starting_again(close: Callable[[], int], left: PersistedRun) -> int | None:
    """Close ``left``, which ended before PRELUDE recorded its baseline, with ``close`` (which ends the sandboxes of the
    tests it left) before the run starts the spec's session again: the exit code the run ends with when a stop during
    the close ended it, else ``None``. A close that cannot read ``left`` (:data:`CANNOT_READ_SESSION`, and for a file
    the OS refuses, one inside ``left``'s directory) leaves it as it is, one warning names the file and why, and
    ``None`` says the run goes on with a new session in a directory of its own, as that session measured nothing a
    restart carries on from. A file outside that directory the OS refuses is not the earlier session's to skip: it
    raises, and ends the run as before."""

    root = left.layout.root
    log.warning("%s ended before PRELUDE recorded its baseline: closing it, then starting again", root)
    try:
        closed = close()
    except CANNOT_READ_SESSION as exc:
        if not _names_a_file_of(root, exc):
            raise
        log.warning("cannot read %s to close it, so it is left as it is and a new session starts: %s", root, _why(exc))
        return None
    return None if reported_stop_reason(root) == left.ended.value else closed


def _names_a_file_of(root: Path, exc: BaseException) -> bool:
    """Whether ``exc`` is about what the session in ``root`` left: a refused file inside ``root``, or text that is not
    what was written (which names no file)."""

    if not isinstance(exc, OSError):
        return True
    return exc.filename is not None and Path(os.path.abspath(os.fsdecode(exc.filename))).is_relative_to(
        os.path.abspath(root)
    )


def _why(exc: BaseException) -> str:
    """The file an error names, with its owner and mode, and the reason; else the error."""

    if isinstance(exc, OSError) and exc.filename:
        return f"{exc.filename}{ownership(Path(exc.filename))}: {exc.strerror}"
    return f"{type(exc).__name__}: {exc}"


def prelude_landed(root: Path) -> bool:
    """Whether the session in ``root`` got through PRELUDE far enough for FINAL to validate it: it committed its root
    and the rule gate wrote its baseline. A session that did not measured nothing a later start could carry on from.
    Creates nothing. Blocks, and runs an event loop of its own."""

    layout = SessionLayout(root)
    if not layout.root_recipe.is_file():
        return False
    reader = LocalBlackboard(layout.blackboard, Writer(Role.OBSERVER, RESTART_READER))
    return asyncio.run(recorded_baseline(reader)) is not None


def reported_stop_reason(root: Path) -> str | None:
    """The stop reason the CLOSE of the session in ``root`` reported, ``None`` when it wrote no report."""

    try:
        text = SessionLayout(root).report.read_text(encoding=_ENCODING)
    except FileNotFoundError:
        return None
    return json.loads(text)["stop_reason"]


__all__ = [
    "CANNOT_READ_SESSION",
    "EXIT_FILE",
    "FINAL_ATTEMPTS",
    "FINAL_COLD_START_SECONDS",
    "FINAL_RESERVE_FLOOR_SECONDS",
    "FINAL_WIND_DOWN_SECONDS",
    "GRANT_MARGIN_SECONDS",
    "SESSION_FILE",
    "EarlierSession",
    "GrantDeadline",
    "GrantEnd",
    "HostedExit",
    "HostedRefused",
    "arm_grant_stop",
    "campaign_spec",
    "close_before_starting_again",
    "close_then_start",
    "final_reserve_seconds",
    "grant_deadline",
    "grant_end",
    "grant_explore_cap",
    "grant_tests_end_by",
    "grant_tests_that_fit",
    "hosted_restart_leg",
    "hosted_resume_deadline",
    "prelude_landed",
    "record_session",
    "recorded_session",
    "refuse_unhosted_branches",
    "reported_stop_reason",
    "require_hosted_isolation",
    "run_argv",
    "share_with_the_group",
    "spec_make_before_break",
    "started_session",
    "stop_before_grant_ends",
    "take_over_earlier_session",
    "toolbox_grant",
    "write_final_exit",
]
