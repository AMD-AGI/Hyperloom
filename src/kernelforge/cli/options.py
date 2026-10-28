# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Click options and validators shared by the kernelforge commands."""

from __future__ import annotations

import json

import click

from kernelforge.cli.option_groups import option_group
from kernelforge.config import CEILING_OFF, CEILING_ON, AgentRuntimeOptions, CeilingSwitch, GpuOptions


MIN_MAX_HOURS = 1.0  # a run shorter than this can't complete a productive campaign


def _validate_max_hours(ctx, param, value):
    """Reject a runtime budget below the minimum a real campaign needs."""
    if value is not None and value < MIN_MAX_HOURS:
        raise click.BadParameter(
            f"must be >= {MIN_MAX_HOURS} (a forge run needs at least {MIN_MAX_HOURS:g} hour to be productive)"
        )
    return value


def _normalize_gpu_type(_ctx, _param, value):
    """Canonicalize the hardware SKU used in KB identities."""
    if value is None:
        return "mi355x"
    return str(value).strip().lower()


def _gpu_options(gpu_type_help: str):
    """The ``--gpu-target`` and ``--gpu-type`` options, with the command's own wording for the SKU."""
    return option_group(
        "gpu",
        GpuOptions,
        click.option(
            "--gpu-target",
            default=None,
            help="ROCm compilation architecture, e.g. gfx950 (also exported to env)",
        ),
        click.option(
            "--gpu-type",
            default=None,
            callback=_normalize_gpu_type,
            help=gpu_type_help,
        ),
    )


def _roofline_ceiling_option(help_text: str):
    """The ``--roofline-ceiling`` switch, spelled once for every command that offers it.

    Off by default everywhere: an estimate costs a profiler pass and an analyst
    session, so nobody pays for one without asking.
    """
    return option_group(
        "roofline_ceiling",
        CeilingSwitch,
        click.option(
            "--roofline-ceiling",
            "roofline_ceiling",
            type=click.Choice([CEILING_ON, CEILING_OFF], case_sensitive=False),
            default=CEILING_OFF,
            help=help_text,
        ),
    )


def _validate_agent_provider(ctx, param, value):
    """Validate a dynamic built-in or entry-point provider name."""
    if value is None:
        return None
    from kernelforge.agent_backends import get_agent_provider

    try:
        return get_agent_provider(value).name
    except ValueError as exc:
        raise click.BadParameter(str(exc), ctx=ctx, param=param) from exc


def _validate_fallback_provider(ctx, param, value):
    """Validate a fallback provider while allowing explicit disablement."""
    if value is not None and value.strip().lower() in {"", "none", "off"}:
        return ""
    return _validate_agent_provider(ctx, param, value)


def _validate_optional_agent_provider(ctx, param, value):
    """Validate an optional provider override, empty meaning "inherit"."""
    if value is None or not value.strip():
        return ""
    return _validate_agent_provider(ctx, param, value)


def _parse_agent_options(ctx, param, value):
    """Read ``--agent-options-json`` as the JSON object of provider extension options it must be."""
    if value is None:
        return None
    try:
        provider_options = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"invalid JSON: {exc}", ctx=ctx, param_hint="--agent-options-json") from exc
    if not isinstance(provider_options, dict):
        raise click.BadParameter("must be a JSON object", ctx=ctx, param_hint="--agent-options-json")
    return provider_options


# Provider-neutral Agent runtime options, one ``AgentRuntimeOptions`` per command.
_agent_runtime_options = option_group(
    "agent",
    AgentRuntimeOptions,
    click.option("--model", default=None, help="Selected provider model"),
    click.option(
        "--agent-backend",
        default=None,
        callback=_validate_agent_provider,
        help="Registered local Agent provider",
    ),
    click.option(
        "--agent-cli",
        default=None,
        help="Provider executable path or command",
    ),
    click.option(
        "--agent-timeout-sec",
        type=click.IntRange(min=1),
        default=None,
        help="Timeout for one Agent session",
    ),
    click.option(
        "--agent-reasoning-effort",
        default=None,
        help="Provider reasoning effort value",
    ),
    click.option(
        "--agent-sandbox-mode",
        default=None,
        help="Provider sandbox mode",
    ),
    click.option(
        "--agent-fallback-provider",
        default=None,
        callback=_validate_fallback_provider,
        help="Fallback provider name, or 'none'. Under model relay it is the member preferred when the primary "
        "cannot run, and 'none' refuses to start a session that has no profile to resume on any other member",
    ),
    click.option(
        "--agent-precheck/--no-agent-precheck",
        default=None,
        help="Enable provider preflight and capability probe",
    ),
    click.option(
        "--agent-options-json",
        "agent_options",
        default=None,
        callback=_parse_agent_options,
        help="Provider extension options as a JSON object",
    ),
)
