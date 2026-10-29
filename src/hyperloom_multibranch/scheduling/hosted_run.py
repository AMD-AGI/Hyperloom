# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What ``hyperloom-multibranch hosted run`` reads from the host and writes back to it.

The campaign spec maps onto the options of ``run`` (:func:`run_argv`), so a hosted session is built by the one builder
of :class:`~hyperloom_multibranch.scheduling.settings.RunSettings` a command-line session is. Besides the common part
the legacy entry validated, a multi-branch session needs ``multibranch.workload`` (``--workload``),
``workload.conc`` and ``workload.precision``; ``multibranch.max_branches`` is ``--max-branches``,
``multibranch.dev_leases`` ``--dev-leases``, each source of ``multibranch.autonomous_agents`` one
``--autonomous-agent``, and ``placement.max_parallel_tests`` is ``--max-parallel-tests``, at most what the grant runs at
once. The fields ``run`` has no option for -- ``objective``, ``launch``, ``options``, ``placement.profile`` and
``placement.specialist_capacity`` -- are not read. Dev sandboxes and agents run in ``dev`` sandboxes of the grant, so
they need its ``dev`` key and the session directory on the storage ``USER_DATA_PATH`` names, which the ``dev``
template mounts at the same path (:func:`refuse_unhosted_branches`).
Every test, PRELUDE's included, runs in a measurement sandbox, as PRELUDE starts no child on a hosted session, so the
application's container needs no card: ``hosted-cpu`` serves as ``hosted-gpu`` does.

The grant bounds the session in time (:func:`grant_deadline`): its budget ends one CLOSE grace and a margin before the
grant does, and the process is sent SIGTERM one CLOSE grace before the grant ends, which ends FINAL and leaves CLOSE
that grace.

The host decides from ``$HYPERLOOM_STATE_DIR/exit.json`` whether to start the run again: never once it holds
``final: true`` (:func:`write_final_exit`), and again when it is missing. ``session.json`` there names the session
directory as soon as it exists (:func:`record_session`), so a start that finds it finishes that session instead of
starting another. The two file names and their fields are the legacy entry's, ``hyperloom.hosted.entry``, which this
package does not import: importing it loads the legacy flow's state.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.io import WriteOptions, atomic_write_json
from hyperloom.hosted.campaign import CampaignSpec, parse_campaign
from hyperloom.hosted.host_info import GrantLimits as HostGrantLimits
from hyperloom.hosted.host_info import HostInfo, Var
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS, MB_SECONDS_PER_HOUR
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.settings import RunSettings
from hyperloom_multibranch.toolbox.hosted_leases import GrantLimits

# hyperloom/hosted/entry.py EXIT_FILE and SESSION_FILE
EXIT_FILE = "exit.json"
SESSION_FILE = "session.json"
#: Seconds between the session's hard stop and the grant's end, for the process to write ``exit.json`` and exit.
GRANT_MARGIN_SECONDS = 60.0
_DURABLE = WriteOptions(fsync=True)
_ENCODING = "utf-8"


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
    return ["run", *(f"--{option}={value}" for option, value in values.items() if value is not None), *agents]


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


def recorded_session(state_dir: Path) -> Path | None:
    """The session directory an earlier start of the run named, ``None`` when none did."""

    try:
        text = (state_dir / SESSION_FILE).read_text(encoding=_ENCODING)
    except FileNotFoundError:
        return None
    return Path(json.loads(text)["session_dir"])


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
    "GrantDeadline",
    "HostedExit",
    "HostedRefused",
    "arm_grant_stop",
    "campaign_spec",
    "grant_deadline",
    "record_session",
    "recorded_session",
    "refuse_unhosted_branches",
    "reported_stop_reason",
    "run_argv",
    "toolbox_grant",
    "write_final_exit",
]
