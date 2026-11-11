# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What ``hyperloom-multibranch hosted run`` reads from the host and writes back to it.

The campaign spec maps onto the options of ``run`` (:func:`run_argv`), so a hosted session is built by the one builder
of :class:`~hyperloom_multibranch.scheduling.settings.RunSettings` a command-line session is. Besides the common part
the legacy entry validated, a multi-branch session needs ``multibranch.workload`` (``--workload``),
``workload.conc`` and ``workload.precision``; ``multibranch.max_branches`` is ``--max-branches``,
``multibranch.dev_leases`` ``--dev-leases``, each source of ``multibranch.autonomous_agents`` one
``--autonomous-agent``, ``multibranch.make_before_break`` (``true`` by default) ``--make-before-break`` or
``--no-make-before-break``, and ``placement.max_parallel_tests`` is ``--max-parallel-tests``, at most what the grant
runs at once. The fields ``run`` has no option for -- ``objective``, ``launch``, ``options``, ``placement.profile`` and
``placement.specialist_capacity`` -- are not read. Dev sandboxes and agents run in ``dev`` sandboxes of the grant, so
they need its ``dev`` key and the session directory on the storage ``USER_DATA_PATH`` names, which the ``dev``
template mounts at the same path (:func:`refuse_unhosted_branches`).
Every test, PRELUDE's included, runs in a measurement sandbox, so the application's container needs no card:
``hosted-cpu`` serves as ``hosted-gpu`` does.

The grant bounds the session in time (:func:`grant_deadline`): its budget ends one CLOSE grace and a margin before the
grant does, and the process is sent SIGTERM one CLOSE grace before the grant ends, which ends FINAL and leaves CLOSE
that grace.

The host decides from ``$HYPERLOOM_STATE_DIR/exit.json`` whether to start the run again: never once it holds
``final: true`` (:func:`write_final_exit`), and again when it is missing. ``session.json`` there names the session
directory as soon as it exists (:func:`record_session`), so a start that finds it finishes that session instead of
starting another, once that session got through PRELUDE; one that ended before its baseline was recorded is closed,
which ends the measurement sandboxes it left, and the spec's session starts again. The two file names and their fields
are the legacy entry's, ``hyperloom.hosted.entry``, which this package does not import: importing it loads the legacy
flow's state.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.shared_storage import child_umask, ownership
from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import GrantLimits as HostGrantLimits
from hyperloom.hosted.host_info import HostInfo, Var
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS, MB_SECONDS_PER_HOUR
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.lifecycle import PersistedRun, persisted_run, persisted_session, prelude_landed
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.scheduling.stops import SessionStopReason
from hyperloom_multibranch.toolbox.hosted_leases import GrantLimits

# hyperloom/hosted/entry.py EXIT_FILE and SESSION_FILE
EXIT_FILE = "exit.json"
SESSION_FILE = "session.json"
#: Seconds between the session's hard stop and the grant's end, for the process to write ``exit.json`` and exit.
GRANT_MARGIN_SECONDS = 60.0
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
        HostedRefused: When the spec lacks what a multi-branch session needs, naming each missing field.
    """

    section = _multibranch_section(spec.multibranch)
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


def arm_grant_stop(seconds: float) -> threading.Timer:
    """Send this process SIGTERM in ``seconds``, the stop signal a session ends on; returns the started timer."""

    timer = threading.Timer(seconds, os.kill, (os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    return timer


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
    directory ``root`` (:func:`~hyperloom_common.shared_storage.child_umask`), so every process it starts, and the
    commands of the agent sandboxes it lends, inherit it: what Arbor, an agent or a tool writes there with the default
    modes stays writable by the group, whose other accounts may run the run's next leg. Elsewhere the umask is left
    as it is."""

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
    (``final-only``, which ends the sandboxes of the tests it left), then ``start`` the spec's session, unless a stop
    during the close ended the run: then the close's exit code."""

    if left is not None:
        log.warning("%s ended before PRELUDE recorded its baseline: closing it, then starting again", left.layout.root)
        closed = close(left)
        if reported_stop_reason(left.layout.root) != left.ended.value:
            return closed
    return start()


def reported_stop_reason(root: Path) -> str | None:
    """The stop reason the CLOSE of the session in ``root`` reported, ``None`` when it wrote no report."""

    try:
        text = SessionLayout(root).report.read_text(encoding=_ENCODING)
    except FileNotFoundError:
        return None
    return json.loads(text)["stop_reason"]


__all__ = [
    "EXIT_FILE",
    "GRANT_MARGIN_SECONDS",
    "SESSION_FILE",
    "EarlierSession",
    "GrantDeadline",
    "HostedExit",
    "HostedRefused",
    "arm_grant_stop",
    "campaign_spec",
    "close_then_start",
    "grant_deadline",
    "grant_tests_that_fit",
    "record_session",
    "recorded_session",
    "refuse_unhosted_branches",
    "reported_stop_reason",
    "run_argv",
    "share_with_the_group",
    "spec_make_before_break",
    "started_session",
    "take_over_earlier_session",
    "toolbox_grant",
    "write_final_exit",
]
