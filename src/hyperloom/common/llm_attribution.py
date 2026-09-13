# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gateway attribution headers for every Hyperloom LLM call."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, MutableMapping, Sequence

#: Selects the gateway whose headers to emit; unset emits nothing.
ATTRIBUTION_ENV = "HYPERLOOM_LLM_ATTRIBUTION"
#: PrimusClaw session id, already exported by the session bootstrap.
CLAW_SESSION_ID_ENV = "CLAW_SESSION_ID"
#: Product label on a shared gateway; every Hyperloom call carries this.
DEFAULT_APPLICATION = "hyperloom"

ANTHROPIC_CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"
OPENAI_CUSTOM_HEADERS_ENV = "OPENAI_CUSTOM_HEADERS"

# Codex maps every gateway header onto a TOML bare key, so a name it would reject must never reach
# ``resolve_codex_provider_config`` (which raises).
_VALID_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_NEWLINE_RE = re.compile(r"[\r\n]+")
_SEPARATOR_RE = re.compile(r"[,=]")

__all__ = [
    "ATTRIBUTION_ENV",
    "AttributionHeader",
    "CLAW_SESSION_ID_ENV",
    "DEFAULT_APPLICATION",
    "PRESETS",
    "attribution_context",
    "call_headers",
    "current_action",
    "current_action_scope",
    "gateway_selected",
    "current_phase",
    "inject_env",
    "sdk_env_overlay",
    "set_current_phase",
]

# Publishing the phase here rather than threading it through every signature is deliberate: the sites that spawn an
# LLM child are spread across orchestration, specialists and kernel tools, and most have no route to ``SharedState``.
_current_phase = ""


def set_current_phase(phase: str) -> None:
    """Publish the phase the orchestrator just entered."""
    global _current_phase
    _current_phase = _sanitize(phase)


def current_phase() -> str:
    """Return the phase last published by :func:`set_current_phase`."""
    return _current_phase


# The action is *not* process-wide the way the phase is: the dispatcher runs several actions at once, so a module
# global would label every call with whichever action started last.
_current_action: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hyperloom_llm_attribution_action",
    default="",
)


@contextlib.contextmanager
def current_action_scope(action: str) -> Iterator[None]:
    """Label every LLM call made while one action runs."""
    token = _current_action.set(_sanitize(action))
    try:
        yield
    finally:
        _current_action.reset(token)


def current_action() -> str:
    """Return the action whose scope this code is running inside."""
    return _current_action.get()


def _sanitize(value: object) -> str:
    """Strip anything that would corrupt an encoding the value passes through."""
    text = _NEWLINE_RE.sub(" ", str(value or "")).replace("$", "")
    return _SEPARATOR_RE.sub("_", text).strip()


def attribution_context(
    *,
    component: str,
    operation: str = "",
    phase: str | None = None,
    env: Mapping[str, str] | None = None,
    **extra: str,
) -> dict[str, str]:
    """Collect the attribution fields known at one call site."""
    source = env if env is not None else os.environ
    fields: dict[str, str] = {
        "application": DEFAULT_APPLICATION,
        "session": source.get(CLAW_SESSION_ID_ENV, ""),
        "component": component,
        "phase": current_phase() if phase is None else phase,
        "type": current_action(),
        "operation": operation,
        **extra,
    }
    return {key: text for key, value in fields.items() if (text := _sanitize(value))}


def _render_combined(fields: Sequence[str], context: Mapping[str, str]) -> str:
    """Join the selected fields as ``field=value`` pairs."""
    return ",".join(f"{field}={context[field]}" for field in fields if context.get(field))


def _render_raw(fields: Sequence[str], context: Mapping[str, str]) -> str:
    """Emit the first selected field that has a value, with no ``field=`` prefix."""
    return next((context[field] for field in fields if context.get(field)), "")


def _render_json(fields: Sequence[str], context: Mapping[str, str]) -> str:
    """Emit the selected fields as a compact JSON object."""
    selected = {field: context[field] for field in fields if context.get(field)}
    return json.dumps(selected, separators=(",", ":")) if selected else ""


#: Value shape by name. This is the whole of the gateway-specific knowledge;
#: everything else a preset states is which header carries which fields.
_RENDERERS: dict[str, Callable[[Sequence[str], Mapping[str, str]], str]] = {
    "combined": _render_combined,
    "raw": _render_raw,
    "json": _render_json,
}


def _parse_combined(fields: Sequence[str], value: str) -> dict[str, str]:
    """Recover ``field=value`` pairs, keeping only the fields declared."""
    recovered: dict[str, str] = {}
    for chunk in value.split(","):
        name, separator, text = chunk.partition("=")
        if separator and name.strip() in fields and text.strip():
            recovered[name.strip()] = text.strip()
    return recovered


def _parse_raw(fields: Sequence[str], value: str) -> dict[str, str]:
    """Recover the single field a prefix-less value carries."""
    return {fields[0]: value.strip()} if len(fields) == 1 and value.strip() else {}


def _parse_json(fields: Sequence[str], value: str) -> dict[str, str]:
    """Recover the declared fields from a JSON object value."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: str(item).strip() for key, item in parsed.items() if key in fields and str(item).strip()}


#: Reverse of :data:`_RENDERERS`, for reading a tag a parent already wrote.
_PARSERS: dict[str, Callable[[Sequence[str], str], dict[str, str]]] = {
    "combined": _parse_combined,
    "raw": _parse_raw,
    "json": _parse_json,
}

#: Fields a child may take from the tag its parent wrote. They describe *where*
#: a call happens rather than what makes it: ``session`` identifies the run, and
#: ``phase`` and ``type`` are ambient state that lives in one process only --
#: :data:`_current_phase` is a module global and :data:`_current_action` a
#: context variable, so a spawned child starts with both empty and could not
#: restate them if it wanted to. ``application`` is absent because this module
#: always supplies it, so an inherited copy could never be reached. ``component``
#: and ``operation`` are absent by intent: a call site that names itself is
#: declaring a new producer, and inheriting the parent's purpose would label its
#: calls with work they are not doing.
_INHERITED_FIELDS = ("session", "phase", "type")

#: The inherited fields describing the *running process* rather than the run's
#: identity. Only a genuinely different process may take these; see
#: :func:`inject_env` for why re-reading them into their own writer is unsound.
_AMBIENT_FIELDS = ("phase", "type")


@dataclass(frozen=True)
class AttributionHeader:
    """One header a gateway preset emits."""

    name: str
    shape: str
    fields: tuple[str, ...]


PRESETS: dict[str, tuple[AttributionHeader, ...]] = {
    "litellm": (
        # Comma-separated tags land in the LiteLLM_SpendLogs request_tags column, which is what gives a per-component
        # spend rollup.
        AttributionHeader(
            "x-litellm-tags",
            "combined",
            ("application", "session", "component", "phase", "type", "operation"),
        ),
        # Sets the spend log's session_id column and propagates to nested MCP and A2A calls, so it is the column a
        # per-session reconciliation joins on.
        AttributionHeader("x-litellm-trace-id", "raw", ("session",)),
    ),
}


def _validate_presets(presets: Mapping[str, Sequence[AttributionHeader]]) -> None:
    """Reject a preset that some path downstream could not carry."""
    for gateway, headers in presets.items():
        for header in headers:
            if not _VALID_HEADER_NAME_RE.match(header.name):
                raise ValueError(f"{gateway} preset: header name {header.name!r} is not a TOML bare key")
            if header.shape not in _RENDERERS:
                raise ValueError(f"{gateway} preset: header {header.name!r} has unknown shape {header.shape!r}")
            if not header.fields:
                raise ValueError(f"{gateway} preset: header {header.name!r} selects no fields")


_validate_presets(PRESETS)


def _configured_headers(env: Mapping[str, str]) -> tuple[AttributionHeader, ...]:
    """Return the headers the selected gateway preset emits."""
    return PRESETS.get((env.get(ATTRIBUTION_ENV) or "").strip().lower(), ())


def gateway_selected(env: Mapping[str, str] | None = None) -> bool:
    """Whether a known gateway preset is selected, so attribution is emitted."""
    return bool(_configured_headers(env if env is not None else os.environ))


def call_headers(
    *,
    component: str,
    operation: str = "",
    phase: str | None = None,
    env: Mapping[str, str] | None = None,
    base: Mapping[str, str] | None = None,
    **extra: str,
) -> dict[str, str]:
    """Render the selected gateway's attribution headers for a request."""
    source = env if env is not None else os.environ
    headers = _configured_headers(source)
    if not headers:
        return {}
    context = attribution_context(
        component=component,
        operation=operation,
        phase=phase,
        env=source,
        **extra,
    )
    if base:
        inherited = dict(base)
        if phase is not None and not _sanitize(phase):
            inherited.pop("phase", None)
        # An explicitly empty field suppresses an inherited one the way an empty ``phase`` does.
        for name, value in extra.items():
            if not _sanitize(value):
                inherited.pop(name, None)
        context = {**inherited, **context}
    rendered = {header.name: _RENDERERS[header.shape](header.fields, context) for header in headers}
    return {name: value for name, value in rendered.items() if value}


def _json_object(text: str) -> dict[str, object] | None:
    """Decode a ``*_CUSTOM_HEADERS`` setting written as a JSON object."""
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _raw_headers(raw: str | None) -> dict[str, str]:
    """Read a raw ``*_CUSTOM_HEADERS`` setting back into header name and value."""
    text = (raw or "").strip()
    if not text:
        return {}
    if (parsed := _json_object(text)) is not None:
        return {str(key).strip().lower(): str(value).strip() for key, value in parsed.items()}
    found: dict[str, str] = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip():
            found[name.strip().lower()] = value.strip()
    return found


def _merge_raw(raw: str | None, headers: Mapping[str, str]) -> str:
    """Add ``headers`` to a raw ``*_CUSTOM_HEADERS`` setting, preserving its text."""
    text = (raw or "").strip()
    if (parsed := _json_object(text)) is not None:
        parsed.update(headers)
        return json.dumps(parsed)

    replaced = {name.lower() for name in headers}
    lines = [
        line for line in text.splitlines() if line.strip() and line.partition(":")[0].strip().lower() not in replaced
    ]
    lines.extend(f"{name}: {value}" for name, value in headers.items())
    return "\n".join(lines)


def _merge_targets(env: Mapping[str, str]) -> tuple[str, ...]:
    """Pick the ``*_CUSTOM_HEADERS`` variables safe to carry attribution."""
    targets = [ANTHROPIC_CUSTOM_HEADERS_ENV]
    fallback_available = bool((env.get(ANTHROPIC_CUSTOM_HEADERS_ENV) or "").strip())
    if (env.get(OPENAI_CUSTOM_HEADERS_ENV) or "").strip() or not fallback_available:
        targets.append(OPENAI_CUSTOM_HEADERS_ENV)
    return tuple(targets)


def _inherited_context(
    env: Mapping[str, str],
    headers: Sequence[AttributionHeader],
) -> dict[str, str]:
    """Recover the ambient attribution fields already present in ``env``."""
    recovered: dict[str, str] = {}
    for variable in _merge_targets(env):
        present = _raw_headers(env.get(variable))
        if not present:
            continue
        for header in headers:
            if header.shape == "raw":
                continue
            value = present.get(header.name.lower())
            if not value:
                continue
            for field, text in _PARSERS[header.shape](header.fields, value).items():
                if field in _INHERITED_FIELDS and (clean := _sanitize(text)):
                    recovered.setdefault(field, clean)
    return recovered


def inject_env(
    env: MutableMapping[str, str],
    *,
    component: str,
    operation: str = "",
    phase: str | None = None,
    source: Mapping[str, str] | None = None,
    **extra: str,
) -> None:
    """Merge the attribution headers into a child environment, in place."""
    configuration = source if source is not None else os.environ
    configured = _configured_headers(configuration)
    if not configured:
        return
    inherited = _inherited_context(env, configured)
    if env is configuration:
        # Injecting into this process's own environment -- as forge_fusion does, so the CLI it spawns later inherits
        # the tag -- leaves that tag in place for the life of the process.
        inherited = {name: value for name, value in inherited.items() if name not in _AMBIENT_FIELDS}
    headers = call_headers(
        component=component,
        operation=operation,
        phase=phase,
        env=configuration,
        base=inherited,
        **extra,
    )
    if not headers:
        return
    for variable in _merge_targets(env):
        env[variable] = _merge_raw(env.get(variable), headers)


def sdk_env_overlay(
    *,
    component: str,
    operation: str = "",
    phase: str | None = None,
    **extra: str,
) -> dict[str, str]:
    """Return the header variables an agent-SDK child needs overlaid on its env."""
    merged = dict(os.environ)
    inject_env(merged, component=component, operation=operation, phase=phase, **extra)
    return {
        variable: merged[variable]
        for variable in (ANTHROPIC_CUSTOM_HEADERS_ENV, OPENAI_CUSTOM_HEADERS_ENV)
        if merged.get(variable) != os.environ.get(variable)
    }
