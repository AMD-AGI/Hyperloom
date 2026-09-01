# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Gateway attribution headers for every Hyperloom LLM call.

Hyperloom knows where a call came from and what it was for; the gateway that
meters the call knows only that some token spent money. This module carries that
context out on the headers a named gateway understands, selected by a single
setting::

    HYPERLOOM_LLM_ATTRIBUTION=litellm
    # x-litellm-tags: application=hyperloom,session=<id>,component=geak,
    #                 phase=KERNEL_AGENT,type=kernel_opt,operation=generate_candidate
    # x-litellm-trace-id: <id>

The fields narrow from the product down to the individual call: ``application``
names Hyperloom on a shared gateway, ``session`` is the run, ``phase`` the stage
it reached, ``type`` the action executing inside that stage, ``component`` the
code that made the call, and ``operation`` what that one call was for.

Leaving it unset is the default and emits nothing, so an unconfigured deployment
behaves exactly as it did before.

Gateways disagree about the *shape* of a header value, not about its content, so
shape is the only axis a preset chooses between: ``combined`` renders
``field=value`` pairs, ``raw`` renders one field's bare value for slots that
expect an identifier, and ``json`` renders an object. Supporting a new gateway is
adding a :data:`PRESETS` entry naming its headers and their shapes; the tests
check every entry, which a free-form environment string could not.

Two delivery paths exist because Hyperloom reaches providers two ways.
:func:`call_headers` returns headers for in-process clients that build their own
request. :func:`inject_env` merges them into the ``*_CUSTOM_HEADERS`` variables
that agent SDKs and CLI children read, which is the only channel available once
the transport belongs to a child process.
"""

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

# Codex maps every gateway header onto a TOML bare key, so a name it would
# reject must never reach ``resolve_codex_provider_config`` (which raises).
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


# The action is *not* process-wide the way the phase is: the dispatcher runs
# several actions at once, so a module global would label every call with
# whichever action started last. A context variable is copied into each
# ``asyncio.Task``, which is exactly the scope an action occupies -- the calls it
# awaits see its value, and its siblings never do.
_current_action: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hyperloom_llm_attribution_action",
    default="",
)


@contextlib.contextmanager
def current_action_scope(action: str) -> Iterator[None]:
    """Label every LLM call made while one action runs.

    Resetting on exit matters because not every caller wraps this in its own
    task: the ones that ``await`` the action directly would otherwise keep its
    label after it returned.

    Args:
        action: The action's task kind, e.g. ``baseline`` or ``kernel_opt``.

    Yields:
        ``None``, with the action published for the duration of the block.
    """
    token = _current_action.set(_sanitize(action))
    try:
        yield
    finally:
        _current_action.reset(token)


def current_action() -> str:
    """Return the action whose scope this code is running inside.

    Returns:
        The action's task kind, or ``""`` outside any action -- which is the
        honest answer for the orchestration call that is *choosing* the action.
    """
    return _current_action.get()


def _sanitize(value: object) -> str:
    """Strip anything that would corrupt an encoding the value passes through.

    Three of them: a newline ends a header record; ``$`` would be re-read as a
    ``${VAR}`` reference by ``llm_config.parse_custom_headers`` and expanded
    against the environment; and ``,`` and ``=`` are the ``combined`` shape's own
    delimiters, so a value carrying either would arrive at the gateway split into
    tags nobody wrote. The separators are replaced rather than dropped, so two
    values differing only there stay two values.

    Args:
        value: Raw field value from a call site.

    Returns:
        The value reduced to characters that survive every encoding.
    """
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
    """Collect the attribution fields known at one call site.

    Five of the six fields answer *where* the call came from and are filled in
    without the call site's help: ``application`` is always Hyperloom,
    ``session`` from the environment, ``phase`` from the phase machine, and
    ``type`` from the action currently running. ``operation`` is the exception,
    because what a call is *for* is knowable only where it is written.

    ``extra`` keeps the vocabulary open: a site with more context (``kernel_id``,
    ``task_id``, ``attempt``) can add it without changing this signature, and a
    preset decides whether to select it. No preset does today -- a per-call
    identifier belongs in a header of its own, since spreading unbounded
    cardinality across the ``combined`` tag is what stops it rolling up.

    Args:
        component: Producer label for the call, e.g. ``geak`` or ``specialist``.
        operation: What this particular call does, e.g. ``review_commit``.
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
    """Recover ``field=value`` pairs, keeping only the fields declared.

    Unambiguous because :func:`_sanitize` replaces both separators in values, so
    no value written by this module can contain a ``,`` or an ``=``.
    """
    recovered: dict[str, str] = {}
    for chunk in value.split(","):
        name, separator, text = chunk.partition("=")
        if separator and name.strip() in fields and text.strip():
            recovered[name.strip()] = text.strip()
    return recovered


def _parse_raw(fields: Sequence[str], value: str) -> dict[str, str]:
    """Recover the single field a prefix-less value carries.

    ``raw`` emits the first field that *has* a value, so a header selecting
    several of them cannot be reversed: the value alone does not say which one
    won. Only a single-field header is recovered; the rest yield nothing.
    """
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
    """One header a gateway preset emits.

    Attributes:
        name: The HTTP header name. Must be a TOML bare key so the Codex config
            path can represent it.
        shape: A key of :data:`_RENDERERS`, choosing how the value is rendered.
        fields: Attribution fields to carry, in order. ``raw`` uses the first
            one that has a value; the others use all of them.
    """

    name: str
    shape: str
    fields: tuple[str, ...]


PRESETS: dict[str, tuple[AttributionHeader, ...]] = {
    "litellm": (
        # Comma-separated tags land in the LiteLLM_SpendLogs request_tags column,
        # which is what gives a per-component spend rollup.
        AttributionHeader(
            "x-litellm-tags",
            "combined",
            ("application", "session", "component", "phase", "type", "operation"),
        ),
        # Sets the spend log's session_id column and propagates to nested MCP and
        # A2A calls, so it is the column a per-session reconciliation joins on.
        AttributionHeader("x-litellm-trace-id", "raw", ("session",)),
    ),
}


def _validate_presets(presets: Mapping[str, Sequence[AttributionHeader]]) -> None:
    """Reject a preset that some path downstream could not carry.

    Checked at import so a bad entry fails the process that added it rather than
    the first run that happens to spawn the wrong child: ``codex_session`` maps
    each gateway header onto a TOML bare key and raises on a name it cannot
    write, and an unknown shape would surface much later as a ``KeyError`` from
    inside rendering. Adding a gateway is the only way to reach any of these.

    Args:
        presets: The preset table to check.

    Raises:
        ValueError: If a header names something unwritable, renders through an
            unknown shape, or selects no fields at all.
    """
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
    """Return the headers the selected gateway preset emits.

    Args:
        env: Environment mapping holding :data:`ATTRIBUTION_ENV`.

    Returns:
        The preset's headers, empty when unset or naming an unknown gateway.
    """
    return PRESETS.get((env.get(ATTRIBUTION_ENV) or "").strip().lower(), ())


def gateway_selected(env: Mapping[str, str] | None = None) -> bool:
    """Whether a known gateway preset is selected, so attribution is emitted.

    Distinct from :data:`ATTRIBUTION_ENV` being set: a value naming no preset
    selects nothing, and a caller that treats the two as the same reports a
    deployment as instrumented when it emits nothing at all.

    Args:
        env: Environment mapping to inspect; defaults to :data:`os.environ`.

    Returns:
        True when the selection names a preset in :data:`PRESETS`.
    """
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
    """Render the selected gateway's attribution headers for a request.

    Args:
        component: Producer label for the call.
        operation: What this particular call does.
        phase: Orchestrator phase the call belongs to; ``None`` uses the
            published phase.
        env: Environment mapping supplying the gateway selection and session id.
        base: Fields to fall back on where this call site knows none, normally
            those :func:`_inherited_context` recovered from a parent's tag. A
            field this call site fills always wins, and passing one as empty
            suppresses the inherited copy -- for ``phase`` the way it suppresses
            the published one, and for anything in ``extra`` the same way.
        **extra: Additional attribution fields.

    Returns:
        Header name to value, empty when no gateway is selected or none of the
        selected fields has a value.
    """
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
        # An explicitly empty field suppresses an inherited one the way an empty
        # ``phase`` does. Without it a call site could override an inherited
        # value but never state that it has none, which is the difference
        # between correcting a stale label and removing it.
        for name, value in extra.items():
            if not _sanitize(value):
                inherited.pop(name, None)
        context = {**inherited, **context}
    rendered = {header.name: _RENDERERS[header.shape](header.fields, context) for header in headers}
    return {name: value for name, value in rendered.items() if value}


def _json_object(text: str) -> dict[str, object] | None:
    """Decode a ``*_CUSTOM_HEADERS`` setting written as a JSON object.

    Sole arbiter of whether a setting is in the JSON encoding, so that reading
    one and writing it back cannot disagree: a malformed object that
    :func:`_merge_raw` appends to as lines must not read back as nothing, which
    would drop the parent's tag exactly where the setting is already suspect.

    Args:
        text: The stripped setting value.

    Returns:
        The decoded object, or ``None`` when the setting is not one -- including
        when it opens like one but does not parse, which is handled as lines.
    """
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _raw_headers(raw: str | None) -> dict[str, str]:
    """Read a raw ``*_CUSTOM_HEADERS`` setting back into header name and value.

    Deliberately does not expand ``${VAR}``: this reads the same text
    :func:`_merge_raw` preserves, and materializing a reference here would put
    the operator's gateway secret somewhere they did not put it. Attribution
    values never contain one, because :func:`_sanitize` strips ``$``.

    Args:
        raw: Current value of the setting, possibly unset.

    Returns:
        Lowercased header name to value; empty when the setting is unset or
        carries no header.
    """
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
    """Add ``headers`` to a raw ``*_CUSTOM_HEADERS`` setting, preserving its text.

    The existing setting is never parsed-and-re-serialized. ``codex_session``
    inspects the *unexpanded* form to recognize a value that is exactly
    ``${VAR}`` and forward the variable name instead of its value; expanding it
    here would materialize the operator's gateway secret into a new variable.

    Re-injecting the same header replaces the previous copy rather than stacking
    another one, because the env-facing hooks run once per turn. Replacement is
    by name and does not ask who wrote the previous copy, so a preset naming a
    header the operator already sets for their own purposes takes it over
    wholesale. A preset should name headers of its own.

    Args:
        raw: Current value of the setting, possibly unset.
        headers: Header name to value pairs to add.

    Returns:
        The merged setting, in whichever encoding the original used.
    """
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


def _inherited_context(
    env: Mapping[str, str],
    headers: Sequence[AttributionHeader],
) -> dict[str, str]:
    """Recover the ambient attribution fields already present in ``env``.

    A process spawned with a tag carries its parent's ``phase`` and ``type``
    in that tag and nowhere else, so re-injecting from inside the child would
    otherwise drop them: both come from in-process state a new interpreter
    starts empty. Reading them back makes a nested injection refine the tag
    rather than replace it -- the child names itself and keeps the rest.

    Recovery is per field across every variable in :func:`_merge_targets` order,
    not a choice of one variable to read. The two are written together, so they
    disagree only when one was partly set by hand -- and taking the first that
    parses at all would then drop exactly the ``phase``/``type`` this exists to
    carry, because the variable that answered first happened to hold less.

    Only self-describing headers are read. A ``raw`` header carries a bare
    value, so reading it back would be a guess that this module wrote it: an
    operator setting ``x-litellm-trace-id`` for their own tracing is
    indistinguishable from our copy of it, and taking that would make their
    trace id the run's ``session`` and misjoin every reconciliation. Nothing is
    lost by declining, because a tag this module wrote always includes the
    ``combined`` header -- ``application`` is never empty.

    Values are sanitized on the way back in. They are re-rendered into a tag
    this process sends, so a separator surviving here would forge fields nobody
    wrote, and a surviving ``${VAR}`` would be expanded downstream into a header
    the gateway logs -- turning an inherited tag into a way to read this
    process's secrets.

    Args:
        env: Environment mapping the tag would be merged into.
        headers: Headers the selected preset emits.

    Returns:
        Field name to value, restricted to :data:`_INHERITED_FIELDS`; empty when
        nothing recoverable is present.
    """
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
    """Merge the attribution headers into a child environment, in place.

    The gateway selection and session id are read from ``source`` (this process)
    rather than from ``env``, because a child environment is often an allowlisted
    subset that deliberately carries neither. Which variables get written is
    still decided from ``env``, since that is what the child will resolve its
    gateway credentials from.

    A tag already in ``env`` is refined rather than overwritten: this call site
    names itself, and the ambient fields it cannot know are carried over from
    whoever wrote that tag. See :func:`_inherited_context` for which those are.

    No-op when no gateway is selected, so an unconfigured deployment hands its
    children the same header variables it always did.

    Args:
        env: Child environment to enrich (mutated in place).
        component: Producer label for the calls this child will make.
        operation: What the child is being spawned to do.
        phase: Orchestrator phase the child belongs to; ``None`` uses the
            published phase.
        source: Environment to read configuration from (defaults to
            :data:`os.environ`).
        **extra: Additional attribution fields.
    """
    configuration = source if source is not None else os.environ
    configured = _configured_headers(configuration)
    if not configured:
        return
    inherited = _inherited_context(env, configured)
    if env is configuration:
        # Injecting into this process's own environment -- as forge_fusion does,
        # so the CLI it spawns later inherits the tag -- leaves that tag in place
        # for the life of the process. Reading our own ``phase``/``type`` back
        # out of it would pin the first action this process ran onto every call
        # that follows, long after its scope exited, which is precisely the
        # mislabelling the tag exists to prevent. The live values are
        # authoritative here because this process is the one publishing them.
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
    """Return the header variables an agent-SDK child needs overlaid on its env.

    ``claude_agent_sdk`` merges ``options.env`` over the inherited environment,
    so an overlay carrying a bare tag would *replace* the operator's gateway
    header rather than join it. Merging against this process's environment first
    keeps both.

    Args:
        component: Producer label for the child's calls.
        operation: What the child is being spawned to do.
        phase: Orchestrator phase; ``None`` uses the published phase.
        **extra: Additional attribution fields.

    Returns:
        Only the variables whose value the overlay actually changes; ``{}`` when
        no gateway is selected.
    """
    merged = dict(os.environ)
    inject_env(merged, component=component, operation=operation, phase=phase, **extra)
    return {
        variable: merged[variable]
        for variable in (ANTHROPIC_CUSTOM_HEADERS_ENV, OPENAI_CUSTOM_HEADERS_ENV)
        if merged.get(variable) != os.environ.get(variable)
    }
