# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The plain values the multibranch hosted toolbox needs, taken from :class:`HostInfo`.

The toolbox package takes plain values only and does not import this package; the multibranch entry builds its input
as ``HostedToolboxInputs(**hosted_toolbox_inputs(host, session_dir=..., tp=...))``.  Nothing here imports the
multibranch package, so the main package does not depend on it.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.hosted.host_info import GrantLimits, HostInfo, Var


class ToolboxInputsError(ValueError):
    """A value the hosted toolbox needs is missing from the host inputs.  Names variables, never values."""


def grant_limits_totals(limits: GrantLimits | None) -> dict[str, float] | None:
    """The four grant totals as a dict for the toolbox, ``None`` when the host gave no limits.

    Raises :class:`ToolboxInputsError` naming each total the grant leaves unstated: the toolbox needs all or none.
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
        raise ToolboxInputsError(
            f"{Var.GRANT_LIMITS} must state all of max_sandboxes, max_gpus, gpu_hours, max_runtime_sec "
            f"(or be unset); missing {', '.join(unstated)}"
        )
    return totals  # type: ignore[return-value]


def hosted_toolbox_inputs(host: HostInfo, *, session_dir: Path, tp: int) -> dict[str, object]:
    """The keyword arguments of the toolbox's input, from *host*.

    SECRET: the result carries the revealed measurement key (and the dev key when set).  Pass it straight to the
    toolbox input, which holds them with ``repr=False``; never log, print or persist the dict.

    Raises :class:`ToolboxInputsError` naming every required variable that is not set.
    """
    required = {
        Var.E2B_API_URL: host.e2b_api_url,
        Var.E2B_SANDBOX_URL: host.e2b_sandbox_url,
        Var.SANDBOX_KEY_MEASUREMENT: host.sandbox_keys.measurement,
        Var.SANDBOX_TEMPLATES: host.sandbox_templates,
        Var.SOURCE_REPO: host.source_repo,
        Var.SOURCE_SHA: host.source_sha,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ToolboxInputsError("hosted toolbox needs " + ", ".join(missing))
    dev_key = host.sandbox_keys.dev
    return {
        "run_id": host.run_id,
        "session_dir": session_dir,
        "api_url": host.e2b_api_url,
        "sandbox_url": host.e2b_sandbox_url,
        "measurement_key": host.sandbox_keys.measurement.reveal(),  # type: ignore[union-attr]
        "sandbox_templates": host.sandbox_templates,
        "source_repo": host.source_repo,
        "source_sha": host.source_sha,
        "tp": tp,
        "dev_key": None if dev_key is None else dev_key.reveal(),
        "grant_limits": grant_limits_totals(host.grant_limits),
    }
