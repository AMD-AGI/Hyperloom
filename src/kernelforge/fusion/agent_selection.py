# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Choose and build the agent backend that forge-fuse's own LLM calls run on."""

from __future__ import annotations

import logging
import os
from typing import Optional

from kernelforge.config import resolve_agent_model, resolve_agent_reasoning_effort
from kernelforge.fusion.errors import FuseUsageError
from kernelforge.agent_backends.base import AgentProviderUnavailableError, AgentRuntimeSettings
from kernelforge.agent_backends.registry import (
    create_registered_backend,
    get_agent_provider,
    resolve_agent_runtime,
    select_default_agent_provider,
)

log = logging.getLogger("forge_fusion")


_AGENT_SANDBOX_MODES = frozenset({"workspace-write", "read-only", "bypass"})


def _resolve_agent_choice(
    agent_backend: str,
    llm_model: Optional[str],
) -> tuple[str, str]:
    """Resolve provider from the registry's auto rule, then model from provider precedence."""
    requested = (agent_backend or "auto").strip().lower()
    if requested == "auto":
        # No credential test of its own: credentials-then-SDK with Claude ahead
        # of Codex is the registry's single rule, and a second copy here is how
        # forge-fusion and forge-loop came to disagree about the same box. No
        # model is passed, so dual-configured selection never guesses from a
        # model prefix.
        provider = select_default_agent_provider().name
    else:
        provider = get_agent_provider(requested).name

    registration = get_agent_provider(provider)
    # The same ladder forge-loop reads. Resolving it here rather than reading
    # one variable directly is what keeps forge-fusion and forge-loop agreeing
    # about which model a box is configured for.
    model = str(llm_model or "").strip() or resolve_agent_model(provider) or registration.default_model
    return provider, model


def _campaign_agent_provider(agent_backend: str, llm_model: Optional[str]) -> str:
    """The provider forge-loop is handed, or the requested spelling when no provider is installed.

    forge-loop rejects the literal ``auto``, this command's own spelling, so the choice is
    resolved here rather than forwarded. An unresolvable choice is not fatal at this point:
    a run that only replays a compile pass authors nothing, and a run that does author still
    fails on its own when it asks the registry for the backend it actually needs.
    """
    try:
        provider, _model = _resolve_agent_choice(agent_backend, llm_model)
    except AgentProviderUnavailableError as exc:
        log.debug("no agent provider resolved for the campaign (%s); forwarding %r", exc, agent_backend)
        return agent_backend
    return provider


def _resolve_agent_sandbox_mode(explicit: Optional[str]) -> str:
    """Resolve and validate the global Agent sandbox policy for this run."""
    mode = (
        str(explicit or "").strip().lower()
        or os.environ.get("FORGE_AGENT_SANDBOX_MODE", "").strip().lower()
        or "workspace-write"
    )
    if mode not in _AGENT_SANDBOX_MODES:
        choices = ", ".join(sorted(_AGENT_SANDBOX_MODES))
        raise FuseUsageError(f"unsupported agent sandbox mode {mode!r}; choose one of: {choices}")
    return mode


# Per-attempt agent wall clock.
_AGENT_TIMEOUT_DEFAULT_SEC = 7200


def _agent_timeout_sec() -> int:
    """Resolve the per-attempt agent wall clock, defaulting to two hours."""
    raw = os.environ.get("FORGE_FUSION_AGENT_TIMEOUT_SEC", "").strip()
    if not raw:
        return _AGENT_TIMEOUT_DEFAULT_SEC
    try:
        value = int(raw)
    except ValueError as exc:
        raise FuseUsageError(
            f"FORGE_FUSION_AGENT_TIMEOUT_SEC must be an integer number of seconds, got {raw!r}"
        ) from exc
    if value <= 0:
        raise FuseUsageError("FORGE_FUSION_AGENT_TIMEOUT_SEC must be greater than zero")
    return value


def _create_agent_backend(
    agent_backend: str,
    llm_model: Optional[str],
    agent_sandbox_mode: Optional[str] = None,
):
    """Create one no-cross-provider-fallback backend for the complete run."""
    sandbox_mode = _resolve_agent_sandbox_mode(agent_sandbox_mode)
    provider, model = _resolve_agent_choice(agent_backend, llm_model)
    runtime = resolve_agent_runtime(
        provider,
        model=model,
        settings=AgentRuntimeSettings(
            timeout_sec=_agent_timeout_sec(),
            # Same switches forge-loop reads. Omitting this pinned every fusion
            # session to the default depth no matter what the box asked for.
            reasoning_effort=resolve_agent_reasoning_effort(),
            sandbox_mode=sandbox_mode,
            fallback_provider="",
        ),
    )
    return create_registered_backend(runtime)
