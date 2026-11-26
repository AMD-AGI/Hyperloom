# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The boundary of a hosted ``conc_sweep`` run: which closed session it sweeps, the run it takes the session over from,
and how it ended.

The spec names the session by its path; the run sweeps it only where the run's own user data lives, as the session the
spec describes, and records it in ``session.json`` as any run records its session, so a restart of the same run knows
its earlier start and takes over the lock that start left. ``exit.json`` carries the sweep's status as its stop
reason, so the host settles the run without opening the report.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hyperloom.hosted.campaign import CampaignSpec
from hyperloom.inference_optimizer.session.paths import ENV_USER_DATA_PATH
from hyperloom.orchestrator.conc_sweep.report import read_summary
from hyperloom_multibranch.scheduling.hosted_run import (
    HostedRefused,
    record_session,
    recorded_session,
    release_earlier_lock,
    share_with_the_group,
)
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.settings import Workload

#: The prefix of the stop reason a hosted sweep's ``exit.json`` carries, before the sweep's status.
SWEEP_STOP_PREFIX = "conc_sweep_"


def swept_session(spec: CampaignSpec, env: Mapping[str, str]) -> SessionLayout:
    """The closed multi-branch session ``conc_sweep.session_dir`` names.

    Raises:
        HostedRefused: When the path is not the directory it resolves to (a link or a ``..`` on the way), lies outside
            the run's ``USER_DATA_PATH``, or holds no closed multi-branch session.
    """

    named = Path(str(spec.sweep_session_dir))
    resolved = named.resolve()
    if resolved != named:
        raise HostedRefused(f"conc_sweep.session_dir {named} resolves to {resolved}: name the directory itself")
    user_data = Path(env.get(ENV_USER_DATA_PATH, "")).resolve()
    if not resolved.is_relative_to(user_data) or resolved == user_data:
        raise HostedRefused(f"conc_sweep.session_dir {named} is not below the run's {ENV_USER_DATA_PATH} {user_data}")
    layout = SessionLayout(resolved)
    if not (layout.manifest.is_file() and layout.report.is_file()):
        raise HostedRefused(f"conc_sweep.session_dir {named} holds no closed multi-branch session")
    return layout


def require_spec_matches(spec: CampaignSpec, workload: Workload) -> None:
    """Refuse a spec that describes another workload than the swept session's: its TP, GPU type and framework.

    Raises:
        HostedRefused: Naming the first field that differs.
    """

    for field, asked, swept in (
        ("hardware.tp", spec.tp, workload.tp),
        ("hardware.gpu_type", spec.gpu_type.lower(), workload.gpu_type.lower()),
        ("framework.name", spec.framework_name.lower(), workload.framework.lower()),
    ):
        if asked != swept:
            raise HostedRefused(f"{field} {asked} is not the swept session's {swept}")


def take_over_swept_session(state_dir: Path, layout: SessionLayout) -> None:
    """Take ``layout`` as the run's session: the lock an earlier start of this run left in it is released, the rest of
    the process writes under the group's umask, and ``session.json`` names it. A lock of another run stays."""

    if recorded_session(state_dir) == layout.root:
        release_earlier_lock(layout)
    share_with_the_group(layout.root)
    record_session(state_dir, layout.root)


def sweep_stop_reason(layout: SessionLayout) -> str | None:
    """``conc_sweep_<status>``, with ``:<skip_reason>`` for a skipped sweep, as the report in ``layout`` says; ``None``
    when the sweep wrote none."""

    summary = read_summary(layout.reports)
    if summary is None:
        return None
    reason = f"{SWEEP_STOP_PREFIX}{summary['status']}"
    return f"{reason}:{summary['skip_reason']}" if summary.get("skip_reason") else reason


__all__ = [
    "SWEEP_STOP_PREFIX",
    "require_spec_matches",
    "swept_session",
    "sweep_stop_reason",
    "take_over_swept_session",
]
