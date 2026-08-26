# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Operator-configured attribution header for every Hyperloom LLM call.

Hyperloom knows which component and phase a call belongs to; the gateway that
meters the call knows only that some token spent money. This module carries that
context out on one HTTP header whose *name* and *field selection* come entirely
from the operator via :data:`HEADER_SPEC_ENV`, so no gateway-specific vocabulary
lives in this repo and a deployment that leaves the variable unset behaves
exactly as it did before.

The spec is ``<header-name>:<field>,<field>,...`` and renders to
``field=value,field=value``. Selected fields with no value are dropped, and when
nothing survives the header is not emitted at all::

    HYPERLOOM_LLM_HEADER_COMBINED='x-tags:session,component,phase'
    # -> x-tags: session=<CLAW_SESSION_ID>,component=geak,phase=KERNEL_AGENT

Two delivery shapes exist because Hyperloom reaches providers two ways.
:func:`call_headers` returns the header for in-process clients that build their
own request. :func:`inject_env` merges it into the ``*_CUSTOM_HEADERS`` variables
that agent SDKs and CLI children read, which is the only channel available once
the transport belongs to a child process.
"""

from __future__ import annotations

import json
import os
import re
from typing import Mapping, MutableMapping

#: Operator setting that names the header and picks the fields it carries.
HEADER_SPEC_ENV = "HYPERLOOM_LLM_HEADER_COMBINED"
#: PrimusClaw session id, already exported by the session bootstrap.
CLAW_SESSION_ID_ENV = "CLAW_SESSION_ID"

ANTHROPIC_CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"
OPENAI_CUSTOM_HEADERS_ENV = "OPENAI_CUSTOM_HEADERS"

# Codex maps every gateway header onto a TOML bare key, so a name it would
# reject must never reach ``resolve_codex_provider_config`` (which raises).
_VALID_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_NEWLINE_RE = re.compile(r"[\r\n]+")

__all__ = [
    "CLAW_SESSION_ID_ENV",
    "HEADER_SPEC_ENV",
    "attribution_context",
    "call_headers",
    "current_phase",
    "inject_env",
    "sdk_env_overlay",
    "set_current_phase",
]

# Publishing the phase here rather than threading it through every signature is
# deliberate: the sites that spawn an LLM child are spread across orchestration,
# specialists and kernel tools, and most have no route to ``SharedState``. A
# Hyperloom process drives exactly one session, so a process-wide value is the
# accurate scope rather than a shortcut.
_current_phase = ""


def set_current_phase(phase: str) -> None:
    """Publish the phase the orchestrator just entered.

    Args:
        phase: The phase being entered.
    """
    global _current_phase
    _current_phase = _sanitize(phase)


def current_phase() -> str:
    """Return the phase last published by :func:`set_current_phase`.

    Returns:
        The current phase, or ``""`` before the first transition.
    """
    return _current_phase


def _sanitize(value: object) -> str:
    """Strip anything that would corrupt the ``Name: value`` header encoding.

    Newlines end a header record, and ``$`` would be re-read as a ``${VAR}``
    reference by ``llm_config.parse_custom_headers`` and expanded against the
    environment.

    Args:
        value: Raw field value from a call site.

    Returns:
        The value reduced to characters that survive both encodings.
    """
    return _NEWLINE_RE.sub(" ", str(value or "")).replace("$", "").strip()


def _parse_spec(env: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    """Split :data:`HEADER_SPEC_ENV` into a header name and its field order.

    Args:
        env: Environment mapping holding the operator setting.

    Returns:
        ``(header_name, fields)``; ``("", ())`` when unset, malformed, or naming
        a header Codex could not represent.
    """
    name, separator, field_list = (env.get(HEADER_SPEC_ENV) or "").strip().partition(":")
    name = name.strip()
    if not separator or not _VALID_HEADER_NAME_RE.match(name):
        return "", ()
    fields = tuple(field.strip() for field in field_list.split(",") if field.strip())
    return (name, fields) if fields else ("", ())


def attribution_context(
    *,
    component: str,
    phase: str | None = None,
    env: Mapping[str, str] | None = None,
    **extra: str,
) -> dict[str, str]:
    """Collect the attribution fields known at one call site.

    ``session`` is read from the environment and ``phase`` defaults to the
    published one, so a call site only has to name what it alone knows. ``extra``
    keeps the vocabulary open: a site with more context (``kernel_id``,
    ``task_id``, ``attempt``) can add it without changing this signature, and the
    operator decides whether to select it.

    Args:
        component: Producer label for the call, e.g. ``geak`` or ``specialist``.
        phase: Orchestrator phase the call belongs to. ``None`` takes the phase
            published by :func:`set_current_phase`; pass ``""`` to force none.
        env: Environment mapping to read the session id from (defaults to
            :data:`os.environ`).
        **extra: Additional attribution fields.

    Returns:
        Field name to sanitized value, with empty fields dropped.
    """
    source = env if env is not None else os.environ
    fields: dict[str, str] = {
        "session": source.get(CLAW_SESSION_ID_ENV, ""),
        "component": component,
        "phase": current_phase() if phase is None else phase,
        **extra,
    }
    return {key: text for key, value in fields.items() if (text := _sanitize(value))}


def call_headers(
    *,
    component: str,
    phase: str | None = None,
    env: Mapping[str, str] | None = None,
    **extra: str,
) -> dict[str, str]:
    """Render the attribution header for an in-process request.

    Args:
        component: Producer label for the call.
        phase: Orchestrator phase the call belongs to; ``None`` uses the
            published phase.
        env: Environment mapping supplying the spec and session id.
        **extra: Additional attribution fields.

    Returns:
        A single-entry header mapping, or ``{}`` when the operator configured no
        header or none of the selected fields has a value.
    """
    source = env if env is not None else os.environ
    name, fields = _parse_spec(source)
    if not name:
        return {}
    context = attribution_context(component=component, phase=phase, env=source, **extra)
    rendered = ",".join(f"{field}={context[field]}" for field in fields if context.get(field))
    return {name: rendered} if rendered else {}


def _merge_raw(raw: str | None, headers: Mapping[str, str]) -> str:
    """Add ``headers`` to a raw ``*_CUSTOM_HEADERS`` setting, preserving its text.

    The existing setting is never parsed-and-re-serialized. ``codex_session``
    inspects the *unexpanded* form to recognize a value that is exactly
    ``${VAR}`` and forward the variable name instead of its value; expanding it
    here would materialize the operator's gateway secret into a new variable.

    Re-injecting the same header replaces the previous copy rather than stacking
    another one, because the env-facing hooks run once per turn.

    Args:
        raw: Current value of the setting, possibly unset.
        headers: Header name to value pairs to add.

    Returns:
        The merged setting, in whichever encoding the original used.
    """
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed.update(headers)
            return json.dumps(parsed)

    replaced = {name.lower() for name in headers}
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and line.partition(":")[0].strip().lower() not in replaced
    ]
    lines.extend(f"{name}: {value}" for name, value in headers.items())
    return "\n".join(lines)


def _merge_targets(env: Mapping[str, str]) -> tuple[str, ...]:
    """Pick the ``*_CUSTOM_HEADERS`` variables safe to carry attribution.

    ``llm_config.resolve_openai_client_config`` reads the Anthropic headers only
    while ``OPENAI_CUSTOM_HEADERS`` parses empty, which is how an Anthropic-only
    deployment authenticates its OpenAI-protocol calls. Creating that variable
    where it did not exist would end the fallback and drop the gateway auth
    header, so it is only written when it already carries something or when no
    fallback could have applied.

    Args:
        env: Environment mapping to inspect.

    Returns:
        The variable names to merge into.
    """
    targets = [ANTHROPIC_CUSTOM_HEADERS_ENV]
    fallback_available = bool((env.get(ANTHROPIC_CUSTOM_HEADERS_ENV) or "").strip())
    if (env.get(OPENAI_CUSTOM_HEADERS_ENV) or "").strip() or not fallback_available:
        targets.append(OPENAI_CUSTOM_HEADERS_ENV)
    return tuple(targets)


def inject_env(
    env: MutableMapping[str, str],
    *,
    component: str,
    phase: str | None = None,
    source: Mapping[str, str] | None = None,
    **extra: str,
) -> None:
    """Merge the attribution header into a child environment, in place.

    The spec and session id are read from ``source`` (this process) rather than
    from ``env``, because a child environment is often an allowlisted subset that
    deliberately carries neither. Which variables get written is still decided
    from ``env``, since that is what the child will resolve its gateway
    credentials from.

    No-op when the operator configured no header, so an unconfigured deployment
    spawns children with a byte-identical environment.

    Args:
        env: Child environment to enrich (mutated in place).
        component: Producer label for the calls this child will make.
        phase: Orchestrator phase the child belongs to; ``None`` uses the
            published phase.
        source: Environment to read configuration from (defaults to
            :data:`os.environ`).
        **extra: Additional attribution fields.
    """
    headers = call_headers(
        component=component,
        phase=phase,
        env=source if source is not None else os.environ,
        **extra,
    )
    if not headers:
        return
    for variable in _merge_targets(env):
        env[variable] = _merge_raw(env.get(variable), headers)


def sdk_env_overlay(
    *,
    component: str,
    phase: str | None = None,
    **extra: str,
) -> dict[str, str]:
    """Return the header variables an agent-SDK child needs overlaid on its env.

    ``claude_agent_sdk`` merges ``options.env`` over the inherited environment,
    so an overlay carrying a bare tag would *replace* the operator's gateway
    header rather than join it. Merging against this process's environment first
    keeps both.

    Args:
        component: Producer label for the child's calls.
        phase: Orchestrator phase; ``None`` uses the published phase.
        **extra: Additional attribution fields.

    Returns:
        Only the variables whose value the overlay actually changes; ``{}`` when
        the operator configured no header.
    """
    merged = dict(os.environ)
    inject_env(merged, component=component, phase=phase, **extra)
    return {
        variable: merged[variable]
        for variable in (ANTHROPIC_CUSTOM_HEADERS_ENV, OPENAI_CUSTOM_HEADERS_ENV)
        if merged.get(variable) != os.environ.get(variable)
    }
