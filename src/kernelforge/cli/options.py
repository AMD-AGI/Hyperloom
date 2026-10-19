# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Click options and validators shared by the kernelforge commands."""

from __future__ import annotations

import json

import click


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


CEILING_ON = "on"
CEILING_OFF = "off"


def _roofline_ceiling_option(help_text: str):
    """The ``--roofline-ceiling`` switch, spelled once for every command that offers it.

    Off by default everywhere: an estimate costs a profiler pass and an analyst
    session, so nobody pays for one without asking.
    """
    return click.option(
        "--roofline-ceiling",
        "roofline_ceiling",
        type=click.Choice([CEILING_ON, CEILING_OFF], case_sensitive=False),
        default=CEILING_OFF,
        help=help_text,
    )


def _validate_roofline_target(target: float, ceiling_enabled: bool) -> float:
    """Check the attainment target is a fraction, and that a ceiling can reach it.

    A target above one is unreachable by construction -- attainment is bounded
    by the ceiling it divides by -- and a target without ``--roofline-ceiling
    on`` is a stop condition that can never fire. Both are refused rather than
    run, because either one leaves a campaign quietly ignoring the limit its
    operator thought they had set.
    """
    value = float(target or 0.0)
    if value <= 0:
        return 0.0
    if value > 1.0:
        raise click.BadParameter(
            f"--roofline-target {value} is a fraction of the ceiling, so it cannot exceed 1.0",
            param_hint="--roofline-target",
        )
    if not ceiling_enabled:
        raise click.BadParameter(
            "--roofline-target needs a ceiling to measure against; pass --roofline-ceiling on",
            param_hint="--roofline-target",
        )
    return value


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


def _agent_runtime_options(func):
    """Attach provider-neutral Agent runtime options to one command."""
    decorators = (
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
            help="Fallback provider name, or 'none'",
        ),
        click.option(
            "--agent-precheck/--no-agent-precheck",
            default=None,
            help="Enable provider preflight and capability probe",
        ),
        click.option(
            "--agent-options-json",
            default=None,
            help="Provider extension options as a JSON object",
        ),
    )
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _agent_runtime_overrides(
    *,
    model: str | None,
    agent_backend: str | None,
    agent_cli: str | None,
    agent_timeout_sec: int | None,
    agent_reasoning_effort: str | None,
    agent_sandbox_mode: str | None,
    agent_fallback_provider: str | None,
    agent_precheck: bool | None,
    agent_options_json: str | None,
) -> dict:
    """Convert provider-neutral CLI values into Config overrides."""
    values = {
        "agent_model": model,
        "agent_backend": agent_backend,
        "agent_cli": agent_cli,
        "agent_timeout_sec": agent_timeout_sec,
        "agent_reasoning_effort": agent_reasoning_effort,
        "agent_sandbox_mode": agent_sandbox_mode,
        "agent_fallback_provider": agent_fallback_provider,
        "agent_precheck": agent_precheck,
    }
    overrides = {key: value for key, value in values.items() if value is not None}
    if agent_options_json is not None:
        try:
            options = json.loads(agent_options_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"invalid JSON: {exc}",
                param_hint="--agent-options-json",
            ) from exc
        if not isinstance(options, dict):
            raise click.BadParameter(
                "must be a JSON object",
                param_hint="--agent-options-json",
            )
        overrides["agent_options"] = options
    return overrides
