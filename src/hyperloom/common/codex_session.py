# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One-shot Codex Agent SDK turns for Hyperloom's OpenAI-side runners.

Hyperloom never issues bare LLM API calls: every interaction runs inside an
agent runtime. This module is the Codex half of that contract. It wraps
``openai_codex`` so callers inherit the SDK's shell/file tools, sandbox, turn
management and usage accounting instead of hand-rolling a tool-calling loop.

The SDK plumbing follows ``kernel_agents.agent_backends.codex.CodexBackend``,
but that class cannot be reused: its workspace guard requires the session cwd
to be a git worktree and enforces KernelForge's benchmark-file protection.
Hyperloom's Codex sessions run against plain output directories, so only the
patterns are shared.

Sandboxing follows that backend too. Codex builds its ``read-only`` and
``workspace-write`` presets on bubblewrap, which Hyperloom's runtime container
does not ship: under either preset every shell command the agent issues dies
with ``bwrap: Failed to make / slave: Permission denied`` before its body runs,
so the agent runtime is unusable. ``HYPERLOOM_CODEX_SANDBOX_MODE``
(:data:`CODEX_SANDBOX_MODE_ENV`) therefore selects which preset family a
session may use, and defaults to ``bypass``: a writing session gets
``Sandbox.full_access`` and containment rests on the container Hyperloom
already runs inside. Deployments whose host provides ``bwrap`` can set
``workspace-write`` or ``read-only`` to hand containment back to Codex.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hyperloom.common.llm_config import LLMConfigError, resolve_openai_client_config

# Name Codex records the gateway under in its own TOML config. Only stability
# matters: the thread's ``model_provider`` refers back to this key.
CODEX_PROVIDER_NAME = "hyperloom"

_CLIENT_NAME = "hyperloom"
_CLIENT_TITLE = "Hyperloom"

# Env vars ``resolve_openai_client_config`` accepts for the API key, in its own
# precedence order. Codex is handed the winning variable's NAME, so the secret
# never reaches the app-server's argv.
_API_KEY_ENV_FALLBACKS: tuple[str, ...] = ("OPENAI_API_KEY", "LLM_GATEWAY_KEY")

# TOML bare-key charset. Header names outside it would need a quoted key, which
# the Codex ``-c key=value`` override parser does not accept.
_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Sandbox preset selector, mirroring KernelForge's ``agent_sandbox_mode``
# (including its ``bypass`` default, which is what makes Codex usable in a
# container without bubblewrap -- see the module docstring).
CODEX_SANDBOX_MODE_ENV = "HYPERLOOM_CODEX_SANDBOX_MODE"
DEFAULT_CODEX_SANDBOX_MODE = "bypass"
# Ordered so the error raised for an unknown mode lists them predictably.
CODEX_SANDBOX_MODES: tuple[str, ...] = ("bypass", "workspace-write", "read-only")

# Grace period for tearing a timed-out turn down before giving up on it.
_INTERRUPT_TIMEOUT_SEC = 5.0


class CodexSessionError(RuntimeError):
    """Raised when a Codex SDK turn cannot start or does not complete."""


class CodexSessionUnavailableError(CodexSessionError):
    """Raised when the Codex SDK is missing or its configuration is unusable."""


class CodexSessionTimeoutError(CodexSessionError):
    """Raised when a Codex turn outlived its timeout and was interrupted."""


@dataclass(frozen=True)
class CodexSessionResult:
    """Normalized outcome of one Codex SDK turn.

    Attributes:
        text: The agent's final response.
        items: One JSON-ready mapping per typed SDK thread item, in order.
        usage: Token accounting for the turn (empty when the SDK reported none).
        thread_id: The SDK thread handle.
        error: The in-band SDK error message, or ``""`` when the turn was clean.
            A provider-side failure can complete the turn without an answer, so
            callers must treat a non-empty value as a failed run.
    """

    text: str = ""
    items: tuple[dict[str, Any], ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    thread_id: str = ""
    error: str = ""


def load_codex_sdk() -> Any:
    """Import ``openai_codex`` lazily and return the module.

    Returns:
        Any: The ``openai_codex`` module.

    Raises:
        CodexSessionUnavailableError: If the Codex SDK is not installed.
    """
    try:
        import openai_codex  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CodexSessionUnavailableError(
            "openai_codex is not installed; install the Codex SDK "
            "(pip install 'hyperloom-inference_optimizer[llm]') before running a Codex session"
        ) from exc
    return openai_codex


def _toml_string(value: str) -> str:
    """Encode a TOML basic string for one Codex ``-c key=value`` override."""
    return json.dumps(value)


def api_key_env_name(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    env: dict[str, str] | None = None,
) -> str:
    """Return the NAME of the env var holding the OpenAI-side API key.

    Mirrors the precedence in
    :func:`hyperloom.common.llm_config.resolve_openai_client_config` so Codex
    reads the same credential as Hyperloom's other OpenAI-side callers, while
    only the variable name crosses into Codex's configuration.

    Args:
        api_key_env (str): Preferred env var name to check first.
        env (dict[str, str] | None): Environment to read; ``os.environ`` when
            ``None``.

    Returns:
        str: The name of the first env var in precedence order that is set.

    Raises:
        CodexSessionUnavailableError: If none of the candidates is set.
    """
    source = env if env is not None else os.environ
    candidates = list(dict.fromkeys([api_key_env, *_API_KEY_ENV_FALLBACKS]))
    for name in candidates:
        if (source.get(name) or "").strip():
            return name
    raise CodexSessionUnavailableError(f"none of {' / '.join(candidates)} is set in env; Codex cannot authenticate")


def codex_provider_overrides(
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    env: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Build the ``model_providers`` overrides that point Codex at the gateway.

    The API key is passed by env-var NAME (``env_key``), never by value, so the
    secret stays out of the app-server's argv. ``wire_api="responses"`` selects
    the OpenAI Responses protocol that Codex speaks.

    Args:
        api_key_env (str): Preferred API-key env var name.
        base_url_env (str): Preferred base-URL env var name.
        env (dict[str, str] | None): Environment to read; ``os.environ`` when
            ``None``.

    Returns:
        tuple[str, ...]: Codex ``key=value`` config overrides.

    Raises:
        CodexSessionUnavailableError: If the credential or the base URL is
            missing, or a gateway header name cannot be expressed as a Codex
            config key.
    """
    source = dict(env) if env is not None else dict(os.environ)
    try:
        config = resolve_openai_client_config(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            env=source,
        )
    except LLMConfigError as exc:
        raise CodexSessionUnavailableError(f"Codex gateway credential is missing: {exc}") from exc
    if not config.base_url:
        raise CodexSessionUnavailableError(
            f"{base_url_env} is not set; Codex needs an explicit OpenAI-compatible gateway base URL"
        )

    key_env = api_key_env_name(api_key_env=api_key_env, env=source)
    provider = CODEX_PROVIDER_NAME
    overrides = [
        f"model_provider={_toml_string(provider)}",
        f"model_providers.{provider}.name={_toml_string(provider)}",
        f"model_providers.{provider}.base_url={_toml_string(config.base_url)}",
        f"model_providers.{provider}.wire_api={_toml_string('responses')}",
        f"model_providers.{provider}.env_key={_toml_string(key_env)}",
    ]
    # Gateway headers are operator-supplied routing/identity values, not
    # credentials (see llm_config.resolve_openai_client_config).
    for header, value in config.default_headers.items():
        if not _TOML_BARE_KEY_RE.match(header):
            raise CodexSessionUnavailableError(f"gateway header name {header!r} is not a valid Codex config key")
        overrides.append(f"model_providers.{provider}.http_headers.{header}={_toml_string(value)}")
    return tuple(overrides)


def _writable_root_overrides(writable_roots: Sequence[Path]) -> tuple[str, ...]:
    """Widen the ``workspace_write`` sandbox to the given roots.

    The session cwd is already writable under that preset, so only the extra
    roots are declared. Codex reads the table only when that preset is active,
    so the override is emitted for every sandbox mode and simply goes unread
    under the others.
    """
    if not writable_roots:
        return ()
    roots = [str(Path(root).resolve()) for root in writable_roots]
    return (f"sandbox_workspace_write.writable_roots={json.dumps(roots)}",)


def _validated_sandbox_mode(mode: str) -> str:
    """Return ``mode`` when it names a known preset family, else fail loudly."""
    if mode not in CODEX_SANDBOX_MODES:
        raise CodexSessionUnavailableError(
            f"unknown Codex sandbox mode {mode!r}; set {CODEX_SANDBOX_MODE_ENV} to one of "
            f"{' / '.join(CODEX_SANDBOX_MODES)}"
        )
    return mode


def resolve_codex_sandbox_mode(*, sandbox_mode: str = "", env: dict[str, str] | None = None) -> str:
    """Resolve which Codex sandbox preset family a session may use.

    Args:
        sandbox_mode (str): Mode stated by the caller; outranks the
            environment. Blank defers to :data:`CODEX_SANDBOX_MODE_ENV`.
        env (dict[str, str] | None): Environment to read; ``os.environ`` when
            ``None``.

    Returns:
        str: One of :data:`CODEX_SANDBOX_MODES`, defaulting to
            :data:`DEFAULT_CODEX_SANDBOX_MODE`.

    Raises:
        CodexSessionUnavailableError: If the resolved value names no known mode.
    """
    stated = sandbox_mode.strip().lower()
    if stated:
        return _validated_sandbox_mode(stated)
    source = env if env is not None else os.environ
    configured = (source.get(CODEX_SANDBOX_MODE_ENV) or "").strip().lower()
    return _validated_sandbox_mode(configured or DEFAULT_CODEX_SANDBOX_MODE)


def codex_sandbox(sdk: Any, *, writable_roots: Sequence[Path], sandbox_mode: str) -> Any:
    """Map a sandbox mode and the requested write scope onto a Codex preset.

    The mode only decides how a *writing* session is contained: a session that
    declares no writable root stays read-only under every mode. ``bypass``
    gives a writing session ``Sandbox.full_access`` because Codex's other two
    presets need a ``bwrap`` this deployment may not have; ``read-only`` is a
    ceiling that a declared write scope cannot raise.

    Args:
        sdk (Any): The loaded ``openai_codex`` module.
        writable_roots (Sequence[Path]): Extra roots the session may write.
        sandbox_mode (str): A mode already resolved by
            :func:`resolve_codex_sandbox_mode`.

    Returns:
        Any: The ``Sandbox`` preset to run the session under.

    Raises:
        CodexSessionUnavailableError: If ``sandbox_mode`` names no known mode.
    """
    mode = _validated_sandbox_mode(sandbox_mode)
    if not writable_roots or mode == "read-only":
        return sdk.Sandbox.read_only
    return sdk.Sandbox.full_access if mode == "bypass" else sdk.Sandbox.workspace_write


def _item_dict(item: Any) -> dict[str, Any]:
    """Convert one typed SDK thread item into a JSON-ready mapping."""
    root = getattr(item, "root", item)
    if hasattr(root, "model_dump"):
        dumped = root.model_dump(by_alias=True, mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return dict(root) if isinstance(root, dict) else {}


def normalize_codex_items(result: Any) -> tuple[dict[str, Any], ...]:
    """Normalize the typed thread items of one turn into JSON-ready mappings.

    Args:
        result (Any): A ``TurnResult`` (or a stand-in exposing ``items``).

    Returns:
        tuple[dict[str, Any], ...]: One mapping per item, in stream order.
    """
    return tuple(_item_dict(item) for item in (getattr(result, "items", None) or ()))


def codex_item_type(item: dict[str, Any]) -> str:
    """Return an item's SDK type folded to a separator-insensitive key.

    The SDK emits camelCase (``commandExecution``); folding to ``commandexecution``
    keeps callers independent of that spelling.
    """
    return str(item.get("type") or "").replace("_", "").lower()


def describe_codex_item(item: dict[str, Any]) -> str:
    """Summarize one normalized thread item as a single log line.

    Args:
        item (dict[str, Any]): A mapping from :func:`normalize_codex_items`.

    Returns:
        str: A one-line summary, or ``""`` for items with nothing to report.
    """
    kind = codex_item_type(item)
    if kind == "agentmessage":
        return str(item.get("text") or "").strip()
    if kind == "commandexecution":
        command = str(item.get("command") or "").strip()
        if not command:
            return ""
        exit_code = item.get("exitCode")
        suffix = "" if exit_code is None else f" (exit {exit_code})"
        return f"$ {command}{suffix}"
    if kind == "filechange":
        paths = codex_file_changes(item)
        return f"wrote {', '.join(paths)}" if paths else ""
    return ""


def codex_file_changes(item: dict[str, Any]) -> tuple[str, ...]:
    """Return the paths a ``fileChange`` item reports, in order."""
    changes = item.get("changes")
    if not isinstance(changes, (list, tuple)):
        return ()
    return tuple(
        change["path"] for change in changes if isinstance(change, dict) and isinstance(change.get("path"), str)
    )


def _usage_int(payload: dict[str, Any], key: str) -> int:
    """Read one non-negative token count, treating unusable values as 0.

    Token accounting is diagnostic, so a malformed count must not abort a run
    that otherwise produced its artifacts.
    """
    value = payload.get(key)
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def normalize_codex_usage(usage: Any) -> dict[str, int]:
    """Normalize ``ThreadTokenUsage`` into Hyperloom's transcript usage fields.

    Only the ``last`` breakdown is read: it covers the turn that just ran,
    whereas ``total`` accumulates across the whole thread. Reasoning tokens are
    reported separately because on a reasoning model they dominate the output
    budget and are invisible in the response text.

    Args:
        usage (Any): The SDK usage object, or ``None``.

    Returns:
        dict[str, int]: Token counts, or ``{}`` when the SDK reported none.
    """
    if usage is None:
        return {}
    breakdown = usage.get("last", usage) if isinstance(usage, dict) else getattr(usage, "last", usage)
    if hasattr(breakdown, "model_dump"):
        breakdown = breakdown.model_dump()
    if not isinstance(breakdown, dict):
        return {}
    return {
        "input_tokens": _usage_int(breakdown, "input_tokens"),
        "output_tokens": _usage_int(breakdown, "output_tokens"),
        "cache_read_input_tokens": _usage_int(breakdown, "cached_input_tokens"),
        "reasoning_output_tokens": _usage_int(breakdown, "reasoning_output_tokens"),
    }


def _turn_error_message(result: Any) -> str:
    """Extract the in-band SDK error message from a completed turn."""
    error = getattr(result, "error", None)
    if error is None:
        return ""
    return str(getattr(error, "message", None) or error)


def normalize_codex_result(result: Any, thread_id: str) -> CodexSessionResult:
    """Normalize one completed SDK turn into a :class:`CodexSessionResult`.

    Args:
        result (Any): The SDK ``TurnResult``.
        thread_id (str): The thread handle the turn ran on.

    Returns:
        CodexSessionResult: The normalized turn outcome.
    """
    return CodexSessionResult(
        text=str(getattr(result, "final_response", "") or "").strip(),
        items=normalize_codex_items(result),
        usage=normalize_codex_usage(getattr(result, "usage", None)),
        thread_id=thread_id,
        error=_turn_error_message(result),
    )


async def run_codex_turn(
    *,
    prompt: str,
    developer_instructions: str,
    cwd: Path,
    model: str,
    timeout_sec: float,
    writable_roots: Sequence[Path] = (),
    sandbox_mode: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    base_url_env: str = "OPENAI_BASE_URL",
    codex_bin: str = "",
    env: dict[str, str] | None = None,
) -> CodexSessionResult:
    """Run one non-interactive Codex turn and normalize its typed result.

    The turn runs with ``ApprovalMode.deny_all`` (no approval prompt can block
    an unattended run). Thread and turn share one sandbox preset, chosen from
    the resolved sandbox mode and ``writable_roots``. ``CODEX_HOME`` is
    redirected to a per-run temp dir so concurrent sessions and the operator's
    own Codex state stay independent.

    Args:
        prompt (str): The user prompt for the turn.
        developer_instructions (str): System-level instructions for the thread.
        cwd (Path): Working directory for the thread; writable under the
            ``workspace_write`` and ``full_access`` sandboxes.
        model (str): The Codex model id.
        timeout_sec (float): Wall-clock budget for the turn. On expiry the turn
            is interrupted and :class:`CodexSessionTimeoutError` is raised.
        writable_roots (Sequence[Path]): Extra roots the session may write.
            Empty keeps the session read-only under every sandbox mode.
        sandbox_mode (str): One of :data:`CODEX_SANDBOX_MODES`. Blank defers to
            :data:`CODEX_SANDBOX_MODE_ENV`, then to
            :data:`DEFAULT_CODEX_SANDBOX_MODE`.
        api_key_env (str): Preferred API-key env var name.
        base_url_env (str): Preferred base-URL env var name.
        codex_bin (str): Optional Codex runtime path; the SDK resolves its own
            when empty.
        env (dict[str, str] | None): Base environment for the app-server;
            ``os.environ`` when ``None``.

    Returns:
        CodexSessionResult: The normalized turn outcome.

    Raises:
        CodexSessionUnavailableError: If the SDK is missing, or its gateway
            config or sandbox mode is unusable.
        CodexSessionTimeoutError: If the turn outlived ``timeout_sec``.
        CodexSessionError: If the SDK failed for any other reason.
    """
    sdk = load_codex_sdk()
    sandbox = codex_sandbox(
        sdk,
        writable_roots=writable_roots,
        sandbox_mode=resolve_codex_sandbox_mode(sandbox_mode=sandbox_mode, env=env),
    )
    config_overrides = (
        "features.memories=false",
        *codex_provider_overrides(api_key_env=api_key_env, base_url_env=base_url_env, env=env),
        *_writable_root_overrides(writable_roots),
    )
    thread_id = ""
    turn_task: asyncio.Task[Any] | None = None

    with tempfile.TemporaryDirectory(prefix="hyperloom-codex-home-") as codex_home:
        child_env = dict(env) if env is not None else os.environ.copy()
        child_env["CODEX_HOME"] = codex_home
        config = sdk.CodexConfig(
            codex_bin=codex_bin or None,
            config_overrides=config_overrides,
            cwd=str(cwd),
            env=child_env,
            client_name=_CLIENT_NAME,
            client_title=_CLIENT_TITLE,
        )
        try:
            async with sdk.AsyncCodex(config) as client:
                thread = await client.thread_start(
                    approval_mode=sdk.ApprovalMode.deny_all,
                    cwd=str(cwd),
                    developer_instructions=developer_instructions,
                    model=model,
                    model_provider=CODEX_PROVIDER_NAME,
                    sandbox=sandbox,
                )
                thread_id = str(getattr(thread, "id", "") or "")
                turn_handle = await thread.turn(
                    prompt,
                    approval_mode=sdk.ApprovalMode.deny_all,
                    cwd=str(cwd),
                    model=model,
                    sandbox=sandbox,
                )
                turn_task = asyncio.create_task(turn_handle.run())
                completed, _pending = await asyncio.wait({turn_task}, timeout=timeout_sec)
                if not completed:
                    # Teardown of an already-failed turn: the timeout below is
                    # the reported failure, so interrupt errors add no signal.
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(turn_handle.interrupt(), timeout=_INTERRUPT_TIMEOUT_SEC)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(asyncio.shield(turn_task), timeout=_INTERRUPT_TIMEOUT_SEC)
                    raise CodexSessionTimeoutError(f"Codex turn timed out after {timeout_sec:g}s")
                sdk_result = turn_task.result()
        except CodexSessionError:
            raise
        except Exception as exc:
            raise CodexSessionError(f"Codex SDK turn failed: {exc}") from exc
        finally:
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                # Let the cancellation settle. The turn's own outcome is already
                # decided, so the task's result is discarded rather than raised;
                # a cancellation of *this* task still propagates.
                await asyncio.gather(turn_task, return_exceptions=True)

    return normalize_codex_result(sdk_result, thread_id)


__all__ = [
    "CODEX_PROVIDER_NAME",
    "CODEX_SANDBOX_MODES",
    "CODEX_SANDBOX_MODE_ENV",
    "CodexSessionError",
    "CodexSessionResult",
    "CodexSessionTimeoutError",
    "CodexSessionUnavailableError",
    "DEFAULT_CODEX_SANDBOX_MODE",
    "api_key_env_name",
    "codex_file_changes",
    "codex_item_type",
    "codex_provider_overrides",
    "codex_sandbox",
    "describe_codex_item",
    "load_codex_sdk",
    "normalize_codex_items",
    "normalize_codex_result",
    "normalize_codex_usage",
    "resolve_codex_sandbox_mode",
    "run_codex_turn",
]
