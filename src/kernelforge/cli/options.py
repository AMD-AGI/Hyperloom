# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Click options and validators shared by the kernelforge commands."""

from __future__ import annotations

import json
from dataclasses import dataclass

import click

from kernelforge.option_groups import option_group


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


@dataclass(frozen=True)
class CeilingSwitch:
    """The ``--roofline-ceiling`` choice."""

    roofline_ceiling: str

    @property
    def enabled(self) -> bool:
        return self.roofline_ceiling == CEILING_ON


@dataclass(frozen=True)
class GpuOptions:
    """The ROCm compilation architecture and the hardware SKU used in KB identities."""

    gpu_target: str | None
    gpu_type: str


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


def _validate_speedup_target(target: float) -> float:
    """Check the speedup target is one the baseline has not already met.

    The baseline's mean case speedup is one by definition, so a target at or
    below one would end every campaign before its first round. It is refused
    rather than run.
    """
    value = float(target or 0.0)
    if value <= 0:
        return 0.0
    if value <= 1.0:
        raise click.BadParameter(
            f"--target-mean-case-speedup {value} is already met by the baseline; give a speedup above 1.0",
            param_hint="--target-mean-case-speedup",
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


@dataclass(frozen=True)
class AgentRuntimeOptions:
    """Provider-neutral Agent runtime options; ``None`` leaves a setting to ``Config.from_env``."""

    model: str | None
    agent_backend: str | None
    agent_cli: str | None
    agent_timeout_sec: int | None
    agent_reasoning_effort: str | None
    agent_sandbox_mode: str | None
    agent_fallback_provider: str | None
    agent_precheck: bool | None
    agent_options_json: str | None


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
        default=None,
        help="Provider extension options as a JSON object",
    ),
)


def _agent_runtime_overrides(options: AgentRuntimeOptions) -> dict:
    """Convert provider-neutral CLI values into Config overrides."""
    values = {
        "agent_model": options.model,
        "agent_backend": options.agent_backend,
        "agent_cli": options.agent_cli,
        "agent_timeout_sec": options.agent_timeout_sec,
        "agent_reasoning_effort": options.agent_reasoning_effort,
        "agent_sandbox_mode": options.agent_sandbox_mode,
        "agent_fallback_provider": options.agent_fallback_provider,
        "agent_precheck": options.agent_precheck,
    }
    overrides = {key: value for key, value in values.items() if value is not None}
    if options.agent_options_json is not None:
        try:
            provider_options = json.loads(options.agent_options_json)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(
                f"invalid JSON: {exc}",
                param_hint="--agent-options-json",
            ) from exc
        if not isinstance(provider_options, dict):
            raise click.BadParameter(
                "must be a JSON object",
                param_hint="--agent-options-json",
            )
        overrides["agent_options"] = provider_options
    return overrides
