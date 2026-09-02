# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Subprocess-based specialist dispatcher.

Per-task git worktree under ``runs/specialist/<task_id>/worktree/``, an agent
CLI subprocess scoped via ``--add-dir``, and a ``specialist_done.json``
(+ ``worktree/patches/``) exit signal harvested into the final
:class:`SpecialistRunResult`. The explicit in-process dispatch mode is wired
separately by the CLI to the matching provider's Agent SDK backend.

Two agent CLIs can drive that contract, and the deployment's credential shape
picks one (:func:`resolve_specialist_agent_backend`): ``claude --print
--output-format stream-json`` authenticates against the Anthropic side, and
``codex exec --json`` against the OpenAI side. An OpenAI-only deployment has no
Anthropic credential at all, so spawning the Claude CLI there produced a
``Not logged in`` exit on every specialist task and silently cost the session
those domains. Everything around the spawn — worktree setup, the reap loop,
heartbeat/staleness, patch discovery, done-file harvesting and the Ray
GPU-specialist actor path — is backend-agnostic and shared; only the argv and
the ``process.log`` parsers differ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from hyperloom.common.codex_session import (
    CodexSessionError,
    probe_codex_sandbox_capability,
    resolve_codex_provider_config,
    resolve_codex_sandbox_mode,
)
from hyperloom.common.env import is_truthy
from hyperloom.common.hermes_runtime import running_in_container
from hyperloom.common.llm_attribution import inject_env as inject_attribution_env
from hyperloom.common.env_safety import (
    BLOCKED_CHILD_ENV_NAMES,
    scrub_child_process_env,
    valid_env_key,
)

from ..trace.parse_usage import (
    parse_claude_stream_json_response,
    parse_claude_stream_json_tool_calls,
    parse_claude_stream_json_turn_usages,
    parse_claude_stream_json_usage,
    parse_codex_jsonl_response,
    parse_codex_jsonl_error,
    parse_codex_jsonl_tool_calls,
    parse_codex_jsonl_turn_usages,
    parse_codex_jsonl_usage,
)


log = logging.getLogger(__name__)


class SpecialistAgentUnavailableError(RuntimeError):
    """Raised when the agent CLI the deployment needs cannot be assembled.

    A missing runtime or an unconfigurable gateway means this deployment cannot
    run specialists at all. It surfaces as the task's failure rather than being
    absorbed into a fallback CLI that would fail to authenticate.
    """


# The two agent CLIs that can drive the specialist contract (module docstring).
AGENT_BACKEND_CLAUDE = "claude"
AGENT_BACKEND_CODEX = "codex"
AGENT_BACKEND_HERMES = "hermes"


def resolve_specialist_agent_backend(env: Mapping[str, str] | None = None) -> str:
    """Return the agent CLI the deployment's credentials can actually drive.

    An OpenAI-only deployment holds no Anthropic credential, so the Claude CLI
    starts and immediately fails with ``Not logged in``; the Codex CLI is the
    only runtime that can authenticate there. Every other shape — Anthropic-only,
    both configured, or nothing configured (a CLI logged in by other means, or
    Bedrock) — keeps the Claude CLI, so this only ever redirects the shape that
    could not work at all.

    The shape test itself belongs to :mod:`hyperloom.common.llm_config`, so this
    cannot disagree with backend selection, the TraceLens runner or the forge
    kernel_backend.

    Args:
        env: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        :data:`AGENT_BACKEND_CODEX` for an OpenAI-only deployment, else
        :data:`AGENT_BACKEND_CLAUDE`.
    """
    source = os.environ if env is None else env
    explicit = str(source.get("HYPERLOOM_AGENT_BACKEND") or "").strip().lower()
    if explicit in {AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX, AGENT_BACKEND_HERMES}:
        return explicit
    from hyperloom.common import llm_config  # local import: keep module import-light

    return AGENT_BACKEND_CODEX if llm_config.is_openai_only(env) else AGENT_BACKEND_CLAUDE


def resolve_codex_executable(explicit: str = "") -> str:
    """Resolve the Codex CLI a specialist subprocess should spawn.

    Order: an explicit path, then ``codex`` on ``$PATH`` (what the runtime
    container installs), then the version-pinned runtime shipped with the Codex
    SDK. The SDK runtime is last so an operator's own installation still wins,
    but present at all so a pod that never ran the npm install can still start a
    specialist.

    Args:
        explicit: Operator-configured path; returned as-is when non-empty.

    Returns:
        The resolved executable path, or ``""`` when no Codex runtime exists —
        the caller reports that instead of spawning a name that cannot run.
    """
    pinned = (explicit or "").strip()
    if pinned:
        return pinned
    on_path = shutil.which(AGENT_BACKEND_CODEX)
    if on_path:
        return on_path
    try:
        # Installed as the ``openai-codex`` SDK's own runtime dependency.
        from codex_cli_bin import bundled_codex_path  # type: ignore[import-not-found]

        bundled = Path(bundled_codex_path())
    except ImportError:
        return ""
    return str(bundled) if bundled.exists() else ""


_SPECIALIST_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        # Session identity, so a child that reports its own spend files it under
        # the same session the parent does. Not a credential.
        "CLAW_SESSION_ID",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "OPENAI_BASE_URL",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
        "USERNAME",
    }
)
_SPECIALIST_SECRET_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "LLM_GATEWAY_KEY",
        "OPENAI_API_KEY",
        "OPENAI_CUSTOM_HEADERS",
    }
)

_CODEX_MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CODEX_MCP_ENV_PREFIX = "HYPERLOOM_CODEX_MCP_ENV_"
_CODEX_PROVIDER_ENV_PREFIX = "HYPERLOOM_CODEX_HTTP_HEADER_"
_CODEX_MCP_RESERVED_ENV_NAMES: frozenset[str] = frozenset(
    {
        *_SPECIALIST_ENV_ALLOWLIST,
        *_SPECIALIST_SECRET_ENV_ALLOWLIST,
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CODEX_HOME",
        "CUDA_VISIBLE_DEVICES",
        "DISABLE_AUTOUPDATER",
        "HIP_VISIBLE_DEVICES",
        "INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS",
        "IS_SANDBOX",
        "ROCR_VISIBLE_DEVICES",
    }
)


def _toml_string(value: str) -> str:
    """Encode a string for the JSON-compatible subset of TOML."""
    return json.dumps(str(value), ensure_ascii=False)


def _resolve_env_references(value: str, *, source: Mapping[str, str], context: str) -> str:
    """Expand ``${NAME}`` references without reading them into generated config."""

    def _replacement(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in source:
            raise SpecialistAgentUnavailableError(f"{context} references unset environment variable {name!r}")
        return source[name]

    return _ENV_REFERENCE_RE.sub(_replacement, value)


def _private_mcp_env_name(
    index: int,
    *,
    source: Mapping[str, str],
    additions: Mapping[str, str],
    protected_env_names: frozenset[str],
) -> str:
    """Return a collision-free child-only variable name for one MCP header."""
    base = f"{_CODEX_MCP_ENV_PREFIX}{index}"
    candidate = base
    suffix = 1
    while candidate in source or candidate in additions or candidate in protected_env_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _codex_provider_resolver_overlay(base_env: Mapping[str, str]) -> dict[str, str]:
    """Mask every omitted specialist secret before resolver environment overlay.

    The canonical provider resolver intentionally overlays its input on
    ``os.environ``. Empty sentinels preserve that contract while preventing an
    omitted child secret from falling back to the parent process.
    """
    overlay = dict(base_env)
    for name in _SPECIALIST_SECRET_ENV_ALLOWLIST:
        overlay.setdefault(name, "")
    return overlay


def _codex_mcp_config(
    config_path: str | None,
    *,
    source: Mapping[str, str],
    child_env: Mapping[str, str],
    protected_env_names: frozenset[str],
) -> tuple[list[str], dict[str, str]]:
    """Translate Claude-style MCP JSON into secret-free Codex TOML lines.

    Stdio environment values and HTTP header values are copied into the child
    environment. The generated TOML contains only their variable names through
    Codex's verified ``env_vars`` and ``env_http_headers`` fields.
    """
    if not config_path:
        return [], {}
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecialistAgentUnavailableError(f"cannot read specialist MCP config {path}: {exc}") from exc
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(servers, dict):
        raise SpecialistAgentUnavailableError(f"specialist MCP config {path} must contain an object at 'mcpServers'")

    lines: list[str] = []
    env_additions: dict[str, str] = {}
    header_index = 0
    for raw_name, raw_server in servers.items():
        name = str(raw_name or "")
        if not _CODEX_MCP_SERVER_NAME_RE.fullmatch(name):
            raise SpecialistAgentUnavailableError(
                f"Codex MCP server name {name!r} is not translatable; "
                "use a letter followed by letters, digits, '_' or '-'"
            )
        if not isinstance(raw_server, dict):
            raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} must be an object")
        server = dict(raw_server)
        declared_type = str(server.get("type") or "").strip().lower()
        if not declared_type:
            declared_type = "http" if server.get("url") else "stdio" if server.get("command") else ""
        if declared_type not in {"http", "stdio"}:
            raise SpecialistAgentUnavailableError(
                f"Codex MCP server {name!r} uses unsupported transport "
                f"{declared_type or '<missing>'!r}; expected 'http' or 'stdio'"
            )

        lines.extend([f"[mcp_servers.{name}]"])
        if declared_type == "http":
            unsupported = set(server) - {"type", "url", "headers"}
            if unsupported:
                raise SpecialistAgentUnavailableError(
                    f"Codex MCP server {name!r} has unsupported HTTP fields: {sorted(unsupported)!r}"
                )
            url = server.get("url")
            if not isinstance(url, str) or not url.strip():
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} requires a non-empty URL")
            parsed_url = urlsplit(url.strip())
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} has invalid HTTP URL")
            if parsed_url.username is not None or parsed_url.password is not None:
                raise SpecialistAgentUnavailableError(
                    f"Codex MCP server {name!r} URL embeds credentials; use environment-backed headers instead"
                )
            lines.append(f"url = {_toml_string(url.strip())}")
            headers = server.get("headers") or {}
            if not isinstance(headers, dict):
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} headers must be an object")
            header_rows: list[tuple[str, str]] = []
            for raw_header, raw_value in headers.items():
                header = str(raw_header or "").strip()
                if not header or not isinstance(raw_value, str):
                    raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} has an invalid HTTP header")
                value = _resolve_env_references(
                    raw_value,
                    source=source,
                    context=f"Codex MCP server {name!r} header {header!r}",
                )
                env_name = _private_mcp_env_name(
                    header_index,
                    source=source,
                    additions=env_additions,
                    protected_env_names=protected_env_names,
                )
                header_index += 1
                env_additions[env_name] = value
                header_rows.append((header, env_name))
            if header_rows:
                lines.extend(["", f"[mcp_servers.{name}.env_http_headers]"])
                lines.extend(f"{_toml_string(header)} = {_toml_string(env_name)}" for header, env_name in header_rows)
        else:
            unsupported = set(server) - {"type", "command", "args", "env"}
            if unsupported:
                raise SpecialistAgentUnavailableError(
                    f"Codex MCP server {name!r} has unsupported stdio fields: {sorted(unsupported)!r}"
                )
            command = server.get("command")
            if not isinstance(command, str) or not command.strip():
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} requires a non-empty command")
            args = server.get("args") or []
            if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} args must be a list of strings")
            raw_env = server.get("env") or {}
            if not isinstance(raw_env, dict):
                raise SpecialistAgentUnavailableError(f"Codex MCP server {name!r} env must be an object")
            env_names: list[str] = []
            for raw_key, raw_value in raw_env.items():
                key = str(raw_key or "").strip()
                if not valid_env_key(key) or key in BLOCKED_CHILD_ENV_NAMES or not isinstance(raw_value, str):
                    raise SpecialistAgentUnavailableError(
                        f"Codex MCP server {name!r} has an unsafe or invalid env entry {key!r}"
                    )
                if (
                    key in _CODEX_MCP_RESERVED_ENV_NAMES
                    or key in protected_env_names
                    or key.startswith(_CODEX_PROVIDER_ENV_PREFIX)
                    or key.startswith(_CODEX_MCP_ENV_PREFIX)
                ):
                    raise SpecialistAgentUnavailableError(
                        f"Codex MCP server {name!r} env key {key!r} is reserved "
                        "for specialist control or provider configuration"
                    )
                resolved_value = _resolve_env_references(
                    raw_value,
                    source=source,
                    context=f"Codex MCP server {name!r} env {key!r}",
                )
                existing_value = env_additions.get(key, child_env.get(key))
                if existing_value is not None and existing_value != resolved_value:
                    raise SpecialistAgentUnavailableError(
                        f"Codex MCP server {name!r} env key {key!r} collides with an existing child environment value"
                    )
                if existing_value is None:
                    env_additions[key] = resolved_value
                env_names.append(key)
            lines.append(f"command = {_toml_string(command.strip())}")
            lines.append(f"args = {json.dumps(args, ensure_ascii=False)}")
            if env_names:
                lines.append(f"env_vars = {json.dumps(env_names)}")
        lines.append("")
    return lines, env_additions


def _write_private_codex_config(
    *,
    codex_home: Path,
    developer_instructions: str,
    mcp_lines: list[str],
) -> Path:
    """Write task-local Codex configuration with private filesystem modes."""
    codex_home.mkdir(parents=True, mode=0o700, exist_ok=True)
    codex_home.chmod(0o700)
    config_path = codex_home / "config.toml"
    lines: list[str] = []
    if developer_instructions:
        lines.extend(
            [
                f"developer_instructions = {_toml_string(developer_instructions)}",
                "",
            ]
        )
    lines.extend(mcp_lines)
    temporary = config_path.with_suffix(".toml.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            # fdopen owns and may already have closed the descriptor.
            pass
        raise
    os.replace(temporary, config_path)
    config_path.chmod(0o600)
    return config_path


def _build_specialist_env() -> dict[str, str]:
    """Build a minimal env for Bash-enabled specialist subprocesses."""
    inherit_setting = os.environ.get("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV")
    inherit_secrets = True if inherit_setting is None else is_truthy(inherit_setting)
    allowed = set(_SPECIALIST_ENV_ALLOWLIST)
    if inherit_secrets:
        allowed.update(_SPECIALIST_SECRET_ENV_ALLOWLIST)
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env = scrub_child_process_env(env)
    # claude's bypassPermissions/--dangerously-skip-permissions refuses to start
    # under root unless IS_SANDBOX=1 (SWSPLAT-42390). Mirror the kernel-agent
    # forge tools (forge_fusion / forge_submit) so specialist authoring
    # subprocesses run on bare-root pods (non-Claw hosts) instead of crashing
    # immediately. setdefault only under root keeps the guard intact elsewhere.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("IS_SANDBOX", "1")
    return env


# Live wall-budget extensions granted by ``extend_lease`` while a specialist is
# already spawned. The reap loop re-reads this every poll, so an extension moves
# the hard kill deadline of a run that is in flight — without it, extend_lease
# would push the task / lane / GPU leases out while the subprocess still died at
# its original ``wall_budget_sec``. Keyed by task_id; the dispatcher clears the
# entry when the run finishes.
_WALL_BUDGET_EXTENSIONS: dict[str, float] = {}


def grant_wall_budget_extension(task_id: str, extra_sec: float) -> float:
    """Add ``extra_sec`` to a live specialist's hard wall-clock deadline.

    Safe to call for a task that is not running a subprocess (the entry is
    simply never read and is cleared on the next dispatch of that task_id).

    Args:
        task_id: The specialist task whose deadline should move.
        extra_sec: Seconds to add; non-positive values are a no-op.

    Returns:
        The task's cumulative granted extension in seconds.
    """
    key = str(task_id or "").strip()
    if not key or extra_sec <= 0:
        return _WALL_BUDGET_EXTENSIONS.get(key, 0.0)
    total = _WALL_BUDGET_EXTENSIONS.get(key, 0.0) + float(extra_sec)
    _WALL_BUDGET_EXTENSIONS[key] = total
    return total


def wall_budget_extension(task_id: str) -> float:
    """Return the cumulative live extension granted to ``task_id`` (0 if none).

    Args:
        task_id: The specialist task to look up.

    Returns:
        Seconds of extension granted so far.
    """
    return _WALL_BUDGET_EXTENSIONS.get(str(task_id or "").strip(), 0.0)


def clear_wall_budget_extension(task_id: str) -> None:
    """Drop any recorded extension for ``task_id``.

    Args:
        task_id: The specialist task whose entry should be removed.
    """
    _WALL_BUDGET_EXTENSIONS.pop(str(task_id or "").strip(), None)


# Configuration
@dataclass(frozen=True)
class SpecialistSubprocessConfig:
    """Static config for spawning agent-CLI subprocesses per specialist.

    Captured once at CLI boot and reused for every dispatch; per-task state is
    passed at run time via :meth:`SpecialistSubprocessDispatcher.run`.
    """

    agent_backend: str = ""
    """Which agent CLI to spawn: ``"claude"``, ``"codex"``, ``"hermes"``, or ``""``.

    Empty resolves the deployment's credential shape per dispatch via
    :func:`resolve_specialist_agent_backend`. The CLI pins it explicitly at boot
    so the backend cannot disagree with the executable and model chosen next to
    it; leaving it empty is for callers that construct a config directly.
    """

    claude_executable: str = "claude"
    """Path / name of the claude CLI binary. Default looks it up on $PATH."""

    codex_executable: str = ""
    """Path / name of the codex CLI binary. Empty resolves it per
    :func:`resolve_codex_executable`."""

    hermes_executable: str = "hermes"
    """Path / name of the Hermes Agent CLI."""

    hermes_profile: str = "hyperloomfaithful"
    """Hermes profile carrying terminal/file tools and no fallback chain."""

    hermes_provider: str = "openai-codex"
    """Inference provider selected inside Hermes."""

    hermes_external_sandbox: bool = False
    """Whether a verifiable outer container confines Hermes terminal/file tools."""

    model: str = ""
    """Model id for the selected agent CLI. Empty = that CLI's own default."""

    permission_mode: str = "bypassPermissions"
    """claude-cli ``--permission-mode``.

    A Claude runtime policy only. Worktrees and the tool denylist scope task
    behavior but are not filesystem containment.
    """

    framework_source_roots: tuple[str, ...] = ()
    """Roots used to seed ``git worktree add`` and as ``--add-dir`` parents.

    The first existing root becomes the worktree base; the rest are exposed
    to the CLI as additional ``--add-dir`` entries.
    """

    mcp_config_path: str | None = None
    """Optional path to a JSON file holding ``{"mcpServers": {...}}``."""

    output_format: str = "stream-json"
    """claude-cli ``--output-format`` flag; ``stream-json`` matches Arbor.

    Codex has no equivalent knob — ``codex exec --json`` is its only streaming
    event format — so this applies to the claude backend only."""

    extra_claude_args: tuple[str, ...] = ()
    """Operator escape hatch — appended verbatim to the claude command."""

    extra_codex_args: tuple[str, ...] = ()
    """Operator escape hatch — appended verbatim to the codex command."""

    leaf_agents_json: str | None = None
    """``--agents`` JSON declaring leaf sub-agent types. None = built-in leaf."""

    per_turn_max_seconds: float = 600.0
    """Per-turn wall-clock fallback.

    Only callers that omit ``wall_budget_sec`` fall back to
    ``max_turns * per_turn_max_seconds`` as a per-task hard timeout.
    """

    poll_interval_seconds: float = 5.0
    """How often the reaper polls done.json / process exit / heartbeat."""

    heartbeat_stale_seconds: float = 300.0
    """Liveness window for heartbeat.json or process.log activity.

    A process that never creates either file is reaped after this window.
    After either file has appeared, its mtime can refresh the liveness clock
    for one window before the subsequent stale window expires, yielding about
    two windows of silence before reap.
    """


# Result
@dataclass
class SpecialistSubprocessResult:
    """Outcome of one specialist subprocess invocation.

    The SpecialistRunner translates this into its own
    :class:`SpecialistRunResult`.
    """

    done_payload: dict[str, Any] | None = None
    """Parsed ``specialist_done.json`` content, or None when the file never
    appeared (the runner then falls back to ``build_empty_specialist_done``)."""

    exit_code: int | None = None
    """Subprocess exit code (None when killed before exit)."""

    elapsed_seconds: float = 0.0

    timed_out: bool = False
    """True when the dispatcher killed the subprocess past the wall-clock cap.

    The cap is normally ``wall_budget_sec``, falling back to
    ``max_turns * per_turn_max_seconds`` when no budget is supplied.
    """

    stale_heartbeat: bool = False
    """True when the heartbeat went stale and the dispatcher killed
    the subprocess."""

    process_log_path: str = ""
    patches: list[str] = field(default_factory=list)
    """Full filesystem paths of patch files discovered under
    ``<worktree>/patches/`` and ``<workspace>/patches/`` (absolute, since both
    roots are session-absolute). Consumers read and sandbox-check these
    directly — do not join them onto a base."""

    patch_roots: dict[str, str] = field(default_factory=dict)
    """Apply root per patch path, for patches harvested from a worktree.

    Empty for patches merely found on disk, whose target tree is not knowable at
    collection time and must therefore still be resolved from their contents."""

    usage: dict[str, Any] | None = None
    """Token usage recovered from the agent CLI's ``process.log``. Carries the
    four canonical counters (``input_tokens`` / ``output_tokens`` /
    ``cache_creation_input_tokens`` / ``cache_read_input_tokens``); the two
    ``cache_*`` may be ``None``, and the codex backend adds
    ``reasoning_output_tokens``. ``None`` when no row carried a ``usage`` block.
    Re-enters the unified ledger the production specialist's token spend."""

    response: str | None = None
    """Assistant reply text recovered from the same log. The prompt is held by
    the parent; pairing it with this response lands the production specialist
    turn in ``conversations.jsonl``. ``None`` when no response text could be
    recovered."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Intel/tool calls (``{"tool", "query"}``) recovered from the same log
    (WebSearch / WebFetch / pr_monitor / Bash / ...). Empty when none were made
    or the log was missing/truncated."""

    turn_usages: list[dict[str, int | None]] = field(default_factory=list)
    """Per-turn token usage recovered from the same log so the parent can trace
    the multi-turn subprocess as one ledger row per model turn. Empty when no
    per-turn usage was present (parent falls back to ``usage``)."""

    error: str = ""


# Worktree management
def _pick_worktree_base(
    roots: tuple[str, ...],
    *,
    preferred: str = "",
) -> Path | None:
    """Return the checkout to branch the specialist's worktree off.

    ``preferred`` wins whenever it is a checkout. It names the framework the
    session is actually optimising, which ``roots`` cannot express: that is the
    source-file *allowlist*, a permission set whose order says nothing about the
    session. Selecting by position worked only while exactly one trusted root
    happened to be a git checkout. When a pod started shipping aiter as one it
    sorted first, so WorldPlay specialists were handed an aiter worktree; the
    patches they wrote against ``hyvideo/`` paths could not be grounded against
    it and patch-safety dropped every one as ``missing_target``, leaving the
    session to bench switches with no code behind them.

    Falls back to None when nothing qualifies — the runner then runs the
    specialist without an isolated worktree.

    Args:
        roots: Candidate root paths to probe for a ``.git`` marker.
        preferred: Checkout of the framework under optimisation, if any. Skipped
            when it is absent or not a checkout, so a pip-installed framework
            costs the specialist nothing.

    Returns:
        The chosen checkout root, or ``None`` when none qualify.
    """

    def _is_checkout(path: str) -> Path | None:
        p = Path(path)
        # ``.git`` may be a file (worktree) or a dir (repo).
        return p if p.is_dir() and (p / ".git").exists() else None

    if preferred:
        chosen = _is_checkout(preferred)
        if chosen is not None:
            return chosen
    for r in roots:
        chosen = _is_checkout(r)
        if chosen is not None:
            return chosen
    return None


def _setup_worktree(
    base: Path,
    worktree_path: Path,
    branch: str,
) -> tuple[Path | None, str]:
    """Create a fresh git worktree at ``worktree_path`` branched off
    ``base``'s HEAD.

    Best-effort: on git error returns ``(None, err)`` so the caller can
    proceed without isolation or hard-fail.

    Args:
        base: Git checkout the worktree is branched off of.
        worktree_path: Destination path for the new worktree.
        branch: Branch name to create for the worktree.

    Returns:
        A ``(worktree_path, "")`` tuple on success, or ``(None, error)`` on
        git failure.
    """
    if worktree_path.exists():
        # Resume / retry: reuse an existing worktree.
        log.warning(
            "specialist worktree already exists at %s; reusing",
            worktree_path,
        )
        return worktree_path, ""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "-C",
        str(base),
        "worktree",
        "add",
        "-b",
        branch,
        str(worktree_path),
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git worktree add failed to spawn: {exc!r}"
    if cp.returncode != 0:
        return None, (f"git worktree add rc={cp.returncode}: stderr={cp.stderr.strip()[:400]!r}")
    return worktree_path, ""


# §3.3: how often to poll a PENDING GPU-specialist Ray actor for its pid.
_RAY_PENDING_POLL_INTERVAL_SEC: float = 1.0


def _ray_specialist_pending_deadline_sec() -> float:
    """Max seconds to wait for a GPU-specialist actor to schedule before failing.

    Reads ``INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC``. A pending
    request that exceeds this becomes a structured task failure rather than an
    unbounded stall (§3.3 / invariant §6.4: pending time is bounded and tracked
    separately from the running wall budget).
    """
    try:
        return float(os.environ.get("INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC", "300"))
    except (TypeError, ValueError):
        return 300.0


class _RayLeaseProcess:
    """``Popen``-like adapter for a specialist that runs inside a Ray actor."""

    def __init__(self, lease: Any, pid: int) -> None:
        self._lease = lease
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """Return ``None`` while running; exit code once done.

        When the actor is unreachable (dead) and ``exit_code()`` is ``None``,
        latches :data:`_RAY_ACTOR_DIED_RC` so the reap loop treats it as a
        real failure immediately rather than looping until the wall-clock cap.
        """
        from hyperloom.orchestrator.actions.executors._ray_serving import (  # noqa: PLC0415
            _RAY_ACTOR_DIED_RC,
        )

        if self.returncode is not None:
            return self.returncode
        if self._lease.is_alive():
            return None
        rc = self._lease.exit_code()
        if rc is None:
            rc = _RAY_ACTOR_DIED_RC
        self.returncode = rc
        return self.returncode

    def reap(self) -> None:
        """Reap the subprocess tree via the actor (lease released separately)."""
        self._lease.stop()


# Dispatcher
class SpecialistSubprocessDispatcher:
    """Spawn + reap one agent-CLI subprocess for a specialist task.

    Reusable across many specialist tasks; owns no per-task state.
    """

    def __init__(self, config: SpecialistSubprocessConfig):
        """Store the static spawn config for reuse across dispatches.

        Args:
            config (SpecialistSubprocessConfig): Session-wide config
                captured at CLI boot; reused for every :meth:`run` call.
        """
        self.config = config

    # Public entry point
    async def run(
        self,
        *,
        task_id: str,
        workspace: Path,
        worktree: Path | None,
        worktree_base: Path | None,
        system_prompt: str,
        user_prompt: str,
        disallowed_tools: frozenset[str] | set[str] | tuple[str, ...] = frozenset(),
        max_turns: int,
        gpu_ids: tuple[int, ...] = (),
        wall_budget_sec: float | None = None,
        gpu_lease: Any = None,
        progress_cb: Any = None,
    ) -> SpecialistSubprocessResult:
        """Spawn an agent-CLI subprocess, reap it, return the parsed result.

        The CLI is the one the deployment's credentials can drive (module
        docstring); the spawn, reap and harvest contract is identical either way.

        Args:
            task_id (str): Task identifier used for logging / workspace
                layout.
            workspace (Path): ``runs/specialist/<task_id>/`` — where
                prompt.md, process.log, heartbeat.json, and
                specialist_done.json live.
            worktree (Path | None): Per-task git worktree (None when
                worktree setup failed; the dispatcher still spawns the agent
                but it has no write-isolated tree, only
                ``--add-dir <workspace>``).
            worktree_base (Path | None): Base checkout the worktree was
                branched off. Unused here; the runner uses it as the clean
                base for patch git-grounding.
            system_prompt (str): System prompt assembled by
                :func:`specialist_prompt_builder.build_specialist_prompts`.
            user_prompt (str): User prompt from the same builder.
            disallowed_tools: Tool names to deny in the agent CLI subprocess.
                Passed as ``--disallowedTools`` to the Claude CLI; has no
                Codex equivalent (Codex containment uses ``--sandbox``).
            max_turns (int): Turn budget. This dispatcher never enforces it
                mechanically — neither agent CLI is passed a turn-cap flag (the
                cap reaches the specialist only as advisory prompt text baked
                into ``system_prompt``). Its sole effect here is the
                ``max_turns × per_turn_max_seconds`` fallback wall-clock
                ceiling, used when ``wall_budget_sec`` is not supplied.
            gpu_ids (tuple[int, ...]): GPU ids to expose to the subprocess.
            wall_budget_sec (float | None): WS1 explicit wall-clock budget
                (seconds). When provided it overrides the
                ``max_turns × per_turn_max_seconds`` ceiling as the reaper's
                hard kill deadline — turns are no longer the stop signal.
            gpu_lease (Any): When set (Ray-managed GPU execution, §12 T4), a
                started-on-demand ``GpuSpecialistLease``; the whole subprocess
                runs inside its actor holding ``num_gpus`` (Ray sets the visible
                devices, so any GPU command the specialist issues stays within
                its lease). ``None`` keeps the local ``Popen`` path, with
                ``gpu_ids`` pinned into ``*_VISIBLE_DEVICES`` as before.
            progress_cb (Any): Optional async callback invoked with each new
                partial checkpoint the specialist writes while it is still
                alive. Exceptions from it never affect the run.

        Returns:
            SpecialistSubprocessResult: Parsed outcome — done payload (if
                any), exit code, timing, timeout / stale-heartbeat flags,
                process log path, and discovered patches.
        """
        # Drop any extension left over from a prior run of this task id.
        clear_wall_budget_extension(task_id)
        workspace.mkdir(parents=True, exist_ok=True)
        prompt_file = workspace / "prompt.md"
        process_log = workspace / "process.log"
        # Poll worktree first (prompt-advertised path), then workspace as fallback.
        done_candidates: list[Path] = []
        if worktree is not None:
            done_candidates.append(worktree / "specialist_done.json")
        done_candidates.append(workspace / "specialist_done.json")
        # Incremental checkpoint recovered as best-so-far on a budget kill; does
        # NOT trigger reap. Same worktree-first / workspace-fallback order.
        partial_candidates: list[Path] = []
        if worktree is not None:
            partial_candidates.append(worktree / "specialist_done.partial.json")
        partial_candidates.append(workspace / "specialist_done.partial.json")
        heartbeat_file = workspace / "heartbeat.json"

        # Compose a minimal env. Provider credentials are inherited by default
        # for compatibility with deployments that authenticate the agent CLI
        # via env; set HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=0 to disable.
        env = _build_specialist_env()
        # Bound the spawned CLI's request transport so a stalled gateway stream
        # raises client-side instead of hanging forever.
        from ..roles._llm_stability_env import apply_llm_stability_env

        apply_llm_stability_env(env)
        # The child spends against the gateway, so tag it or its spend lands
        # under no component at all. The task is offered but no preset selects
        # it: one tag per task would give the spend rollup as many buckets as
        # there are tasks, which is the opposite of what it is read for. Reading
        # spend per task needs a header of its own, not a value in this one.
        inject_attribution_env(env, component="specialist", operation="run_agent", task_id=task_id)

        backend = ""
        try:
            backend = self._agent_backend()
            if backend == AGENT_BACKEND_CODEX:
                # Codex reads the user turn from stdin. Its higher-priority
                # developer instructions live in private task-local config.
                prompt_file.write_text(user_prompt, encoding="utf-8")
                prompt_file.chmod(0o600)
                cmd, launch_env_additions = self._build_codex_launch(
                    prompt_file=prompt_file,
                    workspace=workspace,
                    worktree=worktree,
                    system_prompt=system_prompt,
                    base_env=env,
                    probe_sandbox=True,
                )
                env.update(launch_env_additions)
            elif backend == AGENT_BACKEND_HERMES:
                prompt_file.write_text(user_prompt, encoding="utf-8")
                prompt_file.chmod(0o600)
                cmd = self._build_hermes_cmd(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            else:
                prompt_file.write_text(user_prompt, encoding="utf-8")
                prompt_file.chmod(0o600)
                cmd = self._build_claude_cmd(
                    system_prompt_file=prompt_file.parent / "system_prompt.md",
                    system_prompt=system_prompt,
                    user_prompt_file=prompt_file,
                    workspace=workspace,
                    worktree=worktree,
                    disallowed_tools=frozenset(disallowed_tools),
                )
        except SpecialistAgentUnavailableError as exc:
            # No runtime for this deployment's credential shape. Report it as the
            # task's failure rather than degrading to a CLI that cannot auth.
            return SpecialistSubprocessResult(
                done_payload=None,
                exit_code=None,
                elapsed_seconds=0.0,
                process_log_path=str(process_log),
                error=f"specialist agent runtime unavailable (backend={backend or 'unresolved'}): {exc}",
            )

        if backend == AGENT_BACKEND_CODEX:
            # Per-task CODEX_HOME so concurrent specialists and the operator's
            # own Codex state stay independent. It lives in the task workspace
            # rather than a temp dir because Codex refuses to create its helper
            # binaries under /tmp and would run without its PATH aliases.
            codex_home = workspace / ".codex"
            env["CODEX_HOME"] = str(codex_home)
        if gpu_lease is not None:
            # Ray-managed GPU execution (§12 T4): Ray sets *_VISIBLE_DEVICES in
            # the actor's worker; never let the caller env pin them (that would
            # override Ray's card assignment). ``gpu_ids`` here is the logical
            # 0..N-1 view the specialist sees under Ray's mask — kept only as the
            # informational count env for specialist tooling.
            for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                env.pop(var, None)
            if gpu_ids:
                env["INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"] = ",".join(str(g) for g in gpu_ids)
        elif gpu_ids:
            visible = ",".join(str(g) for g in gpu_ids)
            env["HIP_VISIBLE_DEVICES"] = visible
            env["CUDA_VISIBLE_DEVICES"] = visible
            env["ROCR_VISIBLE_DEVICES"] = visible
            env["INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS"] = visible
        else:
            # CPU specialists must not inherit serving GPU visibility.
            for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                env.pop(var, None)

        log_fh: Any = None
        proc_started: float
        if gpu_lease is not None:
            # §3.3 non-blocking start: submit the actor launch, then poll for
            # the pid with ``asyncio.sleep`` between polls. This keeps the
            # Coordinator event loop responsive while Ray schedules the actor
            # (no blocking ``ray.get``), and — combined with the timing split
            # below — excludes Ray *pending* time from the specialist's running
            # wall budget. A bounded pending deadline turns a permanently
            # unschedulable request into a structured task failure instead of an
            # unbounded stall. The actor opens process.log inside its worker
            # (same host on single-node), so the reaper below reads it directly.
            try:
                gpu_lease.start_async(
                    cmd,
                    env=env,
                    cwd=str(worktree or workspace),
                    log_path=str(process_log),
                    env_mode="replace",
                    stdin_path=str(prompt_file),
                )
            except Exception as exc:  # noqa: BLE001 — surface a submit failure as a result
                return SpecialistSubprocessResult(
                    done_payload=None,
                    exit_code=None,
                    elapsed_seconds=0.0,
                    process_log_path=str(process_log),
                    error=f"failed to submit specialist GPU actor: {exc!r}",
                )
            pending_deadline_sec = _ray_specialist_pending_deadline_sec()
            pending_start = time.monotonic()
            pid: int | None = None
            while True:
                try:
                    pid = gpu_lease.poll_started()
                except Exception as exc:  # noqa: BLE001 — dead actor / ray error mid-schedule
                    gpu_lease.close()
                    return SpecialistSubprocessResult(
                        done_payload=None,
                        exit_code=None,
                        elapsed_seconds=0.0,
                        process_log_path=str(process_log),
                        error=f"specialist GPU actor start failed: {exc!r}",
                    )
                if pid is not None:
                    break
                pending_elapsed = time.monotonic() - pending_start
                if pending_elapsed >= pending_deadline_sec:
                    gpu_lease.close()
                    return SpecialistSubprocessResult(
                        done_payload=None,
                        exit_code=None,
                        elapsed_seconds=0.0,
                        process_log_path=str(process_log),
                        error=(
                            f"specialist GPU actor did not schedule within "
                            f"{pending_deadline_sec:.0f}s (Ray pending deadline); "
                            "cluster fully occupied"
                        ),
                    )
                await asyncio.sleep(_RAY_PENDING_POLL_INTERVAL_SEC)
            proc: Any = _RayLeaseProcess(gpu_lease, pid)
            # §3.3 timing split (invariant §6.4): the wall-budget clock starts
            # only now that a real pid exists — the Ray pending time above is
            # excluded so a slow-to-schedule actor is never mis-reaped.
            proc_started = time.monotonic()
        else:
            proc_started = time.monotonic()
            log_fh = process_log.open("w", encoding="utf-8")
            stdin_fh: Any = None
            try:
                stdin_fh = prompt_file.open("rb")
                proc = subprocess.Popen(
                    cmd,
                    stdin=stdin_fh,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(worktree or workspace),
                    start_new_session=True,
                )
            except (FileNotFoundError, OSError) as exc:
                log_fh.close()
                return SpecialistSubprocessResult(
                    done_payload=None,
                    exit_code=None,
                    elapsed_seconds=0.0,
                    process_log_path=str(process_log),
                    error=f"failed to spawn {backend} subprocess: {exc!r}",
                )
            finally:
                # Popen duplicates the descriptor before returning. Close the
                # parent's copy on success and every spawn failure.
                if stdin_fh is not None:
                    try:
                        stdin_fh.close()
                    except OSError:
                        log.warning("failed to close specialist prompt stdin", exc_info=True)

        # Reap loop — poll done-file / exit / heartbeat staleness / timeout.
        # Prefer the explicit wall budget; fall back to ``max_turns × per_turn``.
        if wall_budget_sec and wall_budget_sec > 0:
            max_seconds = float(wall_budget_sec)
        else:
            max_seconds = float(max_turns) * float(self.config.per_turn_max_seconds)
        try:
            outcome = await self._reap_loop(
                proc=proc,
                workspace=workspace,
                done_files=tuple(done_candidates),
                partial_files=tuple(partial_candidates),
                heartbeat_file=heartbeat_file,
                max_seconds=max_seconds,
                started=proc_started,
                progress_cb=progress_cb,
                task_id=task_id,
            )
        finally:
            if log_fh is not None:
                log_fh.close()
            clear_wall_budget_extension(task_id)

        # Patches: harvest from the worktree via git diff first; fall back to disk scan.
        patches, collected_patch_roots = self._collect_patches(worktree, workspace, worktree_base)

        # Parse done.json (best-effort) — first existing candidate.
        done_payload = None
        for cand in done_candidates:
            if cand.exists():
                done_payload = self._read_done(cand)
                if done_payload is not None:
                    break

        # No final done.json — fall back to the most recent incremental partial
        # so a killed-but-productive specialist still surfaces its findings.
        if done_payload is None:
            for cand in partial_candidates:
                if cand.exists():
                    partial = self._read_done(cand)
                    if partial is not None:
                        partial["_recovered_from_partial"] = True
                        done_payload = partial
                        if not outcome.get("error"):
                            outcome["error"] = "recovered_from_partial"
                        break

        # Harvest the trace from process.log with the parsers for the CLI that
        # wrote it: cumulative session token usage, the reply text that lands in
        # conversations.jsonl, the intel/tool calls, and per-turn usage for
        # fine-grained tracing (which falls back to ``usage`` when absent).
        if backend == AGENT_BACKEND_CODEX:
            usage: dict[str, Any] | None = parse_codex_jsonl_usage(process_log)
            response = parse_codex_jsonl_response(process_log)
            tool_calls = parse_codex_jsonl_tool_calls(process_log)
            turn_usages = parse_codex_jsonl_turn_usages(process_log)
            structured_error = parse_codex_jsonl_error(process_log)
            if structured_error and (outcome["exit_code"] not in (None, 0) or done_payload is None):
                prior_error = str(outcome.get("error") or "").strip()
                outcome["error"] = (
                    f"{prior_error}; codex: {structured_error}" if prior_error else f"codex: {structured_error}"
                )
        elif backend == AGENT_BACKEND_HERMES:
            usage = None
            response = process_log.read_text(encoding="utf-8", errors="replace") if process_log.is_file() else None
            tool_calls = []
            turn_usages = []
        else:
            usage = parse_claude_stream_json_usage(process_log)
            response = parse_claude_stream_json_response(process_log)
            tool_calls = parse_claude_stream_json_tool_calls(process_log)
            turn_usages = parse_claude_stream_json_turn_usages(process_log)

        return SpecialistSubprocessResult(
            done_payload=done_payload,
            exit_code=outcome["exit_code"],
            elapsed_seconds=outcome["elapsed"],
            timed_out=outcome["timed_out"],
            stale_heartbeat=outcome["stale_heartbeat"],
            process_log_path=str(process_log),
            patches=patches,
            patch_roots=collected_patch_roots,
            usage=usage,
            response=response,
            tool_calls=tool_calls,
            turn_usages=turn_usages,
            error=outcome["error"],
        )

    # Internals
    def _agent_backend(self) -> str:
        """Return the agent CLI this dispatch should spawn.

        An explicitly configured backend wins; otherwise the deployment's
        credential shape decides (:func:`resolve_specialist_agent_backend`).

        Returns:
            str: :data:`AGENT_BACKEND_CLAUDE` or :data:`AGENT_BACKEND_CODEX`.

        Raises:
            SpecialistAgentUnavailableError: If the configured backend is not
                one this dispatcher can spawn.
        """
        pinned = (self.config.agent_backend or "").strip().lower()
        if not pinned:
            return resolve_specialist_agent_backend()
        if pinned not in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX, AGENT_BACKEND_HERMES):
            raise SpecialistAgentUnavailableError(
                f"agent_backend={self.config.agent_backend!r} is not one of "
                f"{AGENT_BACKEND_CLAUDE!r} / {AGENT_BACKEND_CODEX!r} / {AGENT_BACKEND_HERMES!r}"
            )
        return pinned

    def _writable_dirs(self, workspace: Path, worktree: Path | None) -> list[str]:
        """Return the dirs an agent CLI may write, in precedence order.

        Worktree first (where patches are authored), then the workspace (where
        ``specialist_done.json`` lands), then each distinct framework source root.

        Args:
            workspace (Path): Task workspace.
            worktree (Path | None): Per-task worktree, when present.

        Returns:
            list[str]: Directory paths, de-duplicated, in order.
        """
        dirs: list[str] = []
        if worktree is not None:
            dirs.append(str(worktree))
        dirs.append(str(workspace))
        for root in self.config.framework_source_roots:
            if root and Path(root).is_dir() and root not in dirs:
                dirs.append(root)
        return dirs

    def _build_codex_cmd(
        self,
        *,
        prompt_file: Path,
        workspace: Path,
        worktree: Path | None,
        system_prompt: str = "",
    ) -> list[str]:
        """Assemble a test/introspection Codex argv without running the probe."""
        cmd, _env_additions = self._build_codex_launch(
            prompt_file=prompt_file,
            workspace=workspace,
            worktree=worktree,
            system_prompt=system_prompt,
            base_env=_build_specialist_env(),
            probe_sandbox=False,
        )
        return cmd

    def _build_codex_launch(
        self,
        *,
        prompt_file: Path,
        workspace: Path,
        worktree: Path | None,
        system_prompt: str,
        base_env: Mapping[str, str],
        probe_sandbox: bool,
    ) -> tuple[list[str], dict[str, str]]:
        """Build secure Codex argv, private config, and child-only env values.

        Codex 0.144.4 accepts ``developer_instructions`` in ``config.toml`` as a
        real developer-role message. The same schema declares MCP servers under
        ``mcp_servers``. The user prompt is represented only by the positional
        ``-`` marker and is supplied through file-backed stdin by both local and
        Ray launchers.
        """
        cfg = self.config
        executable = resolve_codex_executable(cfg.codex_executable)
        if not executable:
            raise SpecialistAgentUnavailableError(
                "no codex CLI found: set SpecialistSubprocessConfig.codex_executable, "
                "install `codex` on $PATH, or install the Codex SDK runtime "
                "(pip install 'hyperloom-inference_optimizer[llm]')"
            )
        resolver_env = _codex_provider_resolver_overlay(base_env)
        try:
            provider_config = resolve_codex_provider_config(env=resolver_env)
        except CodexSessionError as exc:
            raise SpecialistAgentUnavailableError(f"codex gateway is not configured: {exc}") from exc
        try:
            sandbox_mode = resolve_codex_sandbox_mode(env=dict(base_env))
        except CodexSessionError as exc:
            raise SpecialistAgentUnavailableError(f"codex sandbox policy is not usable: {exc}") from exc
        if sandbox_mode == "read-only":
            raise SpecialistAgentUnavailableError(
                "Codex sandbox mode 'read-only' cannot run a specialist: "
                "the specialist must write its result and may need to author patches"
            )
        if probe_sandbox and sandbox_mode != "bypass" and not probe_codex_sandbox_capability(env=dict(base_env)):
            raise SpecialistAgentUnavailableError(
                f"Codex sandbox mode {sandbox_mode!r} requires a working bubblewrap "
                "sandbox, but the capability probe failed; refusing to fall back to bypass"
            )

        provider_env_additions = dict(provider_config.env_additions)
        child_env_before_mcp = dict(base_env)
        child_env_before_mcp.update(provider_env_additions)
        effective_source = os.environ.copy()
        effective_source.update(resolver_env)
        effective_source.update(provider_env_additions)
        mcp_lines, mcp_env_additions = _codex_mcp_config(
            cfg.mcp_config_path,
            source=effective_source,
            child_env=child_env_before_mcp,
            protected_env_names=frozenset(provider_env_additions),
        )
        env_additions = dict(provider_env_additions)
        env_additions.update(mcp_env_additions)
        codex_home = workspace / ".codex"
        try:
            _write_private_codex_config(
                codex_home=codex_home,
                developer_instructions=system_prompt,
                mcp_lines=mcp_lines,
            )
        except OSError as exc:
            raise SpecialistAgentUnavailableError(
                f"cannot prepare private Codex config under {codex_home}: {exc}"
            ) from exc

        writable_dirs = self._writable_dirs(workspace, worktree)
        cmd: list[str] = [
            executable,
            "exec",
            "--json",
            "--strict-config",
            "--skip-git-repo-check",
        ]
        if sandbox_mode == "bypass":
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", sandbox_mode])
        # Primary working root; the reap loop spawns with the same cwd.
        cmd.extend(["-C", str(worktree or workspace)])
        for extra_dir in writable_dirs[1:]:
            cmd.extend(["--add-dir", extra_dir])
        # ``features.memories=false`` matches the SDK session: a specialist must
        # not carry state between tasks.
        for override in ("features.memories=false", *provider_config.overrides):
            cmd.extend(["-c", override])
        if cfg.model:
            cmd.extend(["-m", cfg.model])
        if cfg.extra_codex_args:
            cmd.extend(list(cfg.extra_codex_args))
        cmd.append("-")
        return cmd, env_additions

    def _build_hermes_cmd(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> list[str]:
        """Assemble a Hermes argv for the unchanged specialist contract."""

        cfg = self.config
        if not cfg.hermes_external_sandbox or not running_in_container():
            raise SpecialistAgentUnavailableError(
                "Hermes specialists require a verifiable outer container; set "
                "HYPERLOOM_HERMES_EXTERNAL_SANDBOX=1 only inside that boundary"
            )
        executable = shutil.which(cfg.hermes_executable) or (
            cfg.hermes_executable if Path(cfg.hermes_executable).is_file() else ""
        )
        if not executable:
            raise SpecialistAgentUnavailableError(
                "no Hermes CLI found: set SpecialistSubprocessConfig.hermes_executable or install hermes"
            )
        prompt = "\n\n".join(part for part in (system_prompt.strip(), user_prompt) if part)
        cmd = [
            executable,
            "--profile",
            cfg.hermes_profile,
            "--provider",
            cfg.hermes_provider,
        ]
        if cfg.model:
            cmd.extend(["--model", cfg.model])
        cmd.extend(["--safe-mode", "--toolsets", "terminal,file", "-z", prompt])
        return cmd

    def _build_claude_cmd(
        self,
        *,
        system_prompt_file: Path,
        system_prompt: str,
        user_prompt_file: Path,
        workspace: Path,
        worktree: Path | None,
        disallowed_tools: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Assemble the ``claude`` CLI argv for a specialist subprocess.

        System and user prompts travel through separate channels: system via
        ``--system-prompt-file``, user via stdin.

        Args:
            system_prompt_file (Path): Destination for the written system prompt;
                passed to ``--system-prompt-file``.
            system_prompt (str): The system prompt text to write.
            user_prompt_file (Path): Pre-written user prompt file fed to stdin.
            workspace (Path): Task workspace surfaced as an ``--add-dir``.
            worktree (Path | None): Write-isolated worktree surfaced as the
                first ``--add-dir`` when present.
            disallowed_tools: Tool names to remove from the available set.

        Returns:
            list[str]: The full command argv to spawn.
        """
        fd = os.open(system_prompt_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(system_prompt)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        system_prompt_file.chmod(0o600)

        cfg = self.config
        cmd: list[str] = [
            cfg.claude_executable,
            "--print",
            "--output-format",
            cfg.output_format,
            "--verbose",
            "--permission-mode",
            cfg.permission_mode,
            "--system-prompt-file",
            str(system_prompt_file),
        ]
        if cfg.model:
            cmd.extend(["--model", cfg.model])
        if disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(sorted(disallowed_tools))])
        from .leaf import build_leaf_agents_json

        cmd.extend(["--agents", cfg.leaf_agents_json or build_leaf_agents_json()])
        if cfg.mcp_config_path:
            cmd.extend(["--mcp-config", cfg.mcp_config_path])
        for d in self._writable_dirs(workspace, worktree):
            cmd.extend(["--add-dir", d])
        if cfg.extra_claude_args:
            cmd.extend(list(cfg.extra_claude_args))
        return cmd

    async def _publish_partial_progress(
        self,
        *,
        partial_files: tuple[Path, ...],
        since_mtime: float,
        elapsed: float,
        progress_cb: Any,
    ) -> float:
        """Forward a freshly-rewritten partial checkpoint to ``progress_cb``.

        Publishes at most one file per call: worktree first, then workspace.

        Args:
            partial_files: Candidate checkpoint paths, worktree first.
            since_mtime: Newest mtime already published.
            elapsed: Seconds since spawn, passed to the callback.
            progress_cb: Async callback receiving ``(payload, elapsed)``.

        Returns:
            The newest mtime seen, so the caller can skip unchanged files.
        """
        newest = since_mtime
        for cand in partial_files:
            try:
                mtime = cand.stat().st_mtime
            except OSError:
                continue
            if mtime <= since_mtime:
                continue
            payload = self._read_done(cand)
            if payload is None:
                continue
            newest = max(newest, mtime)
            try:
                await progress_cb(payload, elapsed)
            except Exception:  # noqa: BLE001 — never let telemetry kill a run
                log.exception("specialist progress callback raised")
            break
        return newest

    async def _reap_loop(
        self,
        *,
        proc: Any,
        workspace: Path,
        done_files: tuple[Path, ...],
        heartbeat_file: Path,
        max_seconds: float,
        started: float,
        partial_files: tuple[Path, ...] = (),
        progress_cb: Any = None,
        task_id: str = "",
    ) -> dict[str, Any]:
        """Poll the subprocess until it finishes, stalls, or times out.

        Each tick checks (in order): a done-file at any candidate path
        (graceful exit with a short grace window), natural process exit,
        activity staleness (heartbeat.json OR process.log), and the hard
        wall-clock cap. Stale / timed-out
        runs are killed via :meth:`_kill`. Partial checkpoints written along
        the way are forwarded to ``progress_cb`` as they change.

        Args:
            proc (Any): The running agent subprocess — a ``subprocess.Popen``
                (local) or a :class:`_RayLeaseProcess` (Ray GPU-specialist
                actor). Only ``poll`` / ``returncode`` / ``pid`` are used.
            workspace (Path): Task workspace; supplies the ``process.log``
                whose mtime is the second liveness signal.
            done_files (tuple[Path, ...]): Candidate done-file paths to poll.
            heartbeat_file (Path): Heartbeat file whose mtime is one of the two
                activity signals ORed together for the liveness check.
            max_seconds (float): Hard wall-clock ceiling for the run.
            started (float): ``time.monotonic()`` value at spawn time.
            partial_files (tuple[Path, ...]): Candidate partial-checkpoint
                paths polled for live progress; never an exit signal.
            progress_cb (Any): Optional async callback for each new checkpoint.
            task_id (str): Task identifier used to pick up live
                ``extend_lease`` wall-budget extensions each poll.

        Returns:
            dict[str, Any]: Outcome with ``exit_code``, ``elapsed``,
                ``timed_out``, ``stale_heartbeat``, and ``error`` keys.
        """
        cfg = self.config
        outcome: dict[str, Any] = {
            "exit_code": None,
            "elapsed": 0.0,
            "timed_out": False,
            "stale_heartbeat": False,
            "error": "",
        }
        last_heartbeat_seen: float = started
        # process.log mtime is a reliable "still working" signal even when the
        # agent never self-writes heartbeat.json.
        process_log = workspace / "process.log"
        last_partial_mtime: float = 0.0

        while True:
            await asyncio.sleep(cfg.poll_interval_seconds)
            now = time.monotonic()
            elapsed = now - started
            outcome["elapsed"] = elapsed

            # done.json appeared — graceful exit with up to 30s grace.
            if any(p.exists() for p in done_files):
                grace_until = now + 30.0
                while time.monotonic() < grace_until and proc.poll() is None:
                    await asyncio.sleep(2.0)
                # Still alive after grace — reap it so no orphaned subprocess leaks.
                if proc.poll() is None:
                    self._kill(proc)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

            # Process exited on its own.
            if proc.poll() is not None:
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = elapsed
                break

            # Still running: republish any checkpoint written since the last tick.
            if progress_cb is not None:
                last_partial_mtime = await self._publish_partial_progress(
                    partial_files=partial_files,
                    since_mtime=last_partial_mtime,
                    elapsed=elapsed,
                    progress_cb=progress_cb,
                )

            # Liveness check: alive if EITHER heartbeat.json was refreshed OR
            # process.log is still growing. The hard wall-clock cap below still
            # bounds genuinely hung subprocesses.
            for activity_file in (heartbeat_file, process_log):
                try:
                    if not activity_file.exists():
                        continue
                    a_mtime = activity_file.stat().st_mtime
                except OSError:
                    continue
                if max(0.0, time.time() - a_mtime) <= cfg.heartbeat_stale_seconds:
                    last_heartbeat_seen = now
                    break

            if (now - last_heartbeat_seen) > cfg.heartbeat_stale_seconds:
                outcome["stale_heartbeat"] = True
                outcome["error"] = (
                    f"heartbeat stale for {now - last_heartbeat_seen:.0f}s "
                    f"(> {cfg.heartbeat_stale_seconds:.0f}s threshold)"
                )
                self._kill(proc)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

            # Hard wall-clock cap — re-read each poll so an ``extend_lease``
            # granted mid-run actually moves this deadline.
            deadline = max_seconds + wall_budget_extension(task_id)
            if elapsed > deadline:
                outcome["timed_out"] = True
                outcome["error"] = f"specialist subprocess exceeded {deadline:.0f}s wall-clock cap"
                self._kill(proc)
                outcome["exit_code"] = proc.poll()
                outcome["elapsed"] = time.monotonic() - started
                break

        return outcome

    @staticmethod
    def _kill(proc: Any) -> None:
        """Tear down an agent subprocess.

        Kills the whole process group (SIGTERM, then SIGKILL after a 5s
        grace) so child SDK / curl invocations die with it. No-op if the
        process already exited. For a :class:`_RayLeaseProcess` (Ray
        GPU-specialist actor) the reap is delegated to the actor, which reaps
        the whole tree inside its worker (the lease is released separately).

        Args:
            proc (Any): The subprocess to terminate — ``subprocess.Popen`` or
                :class:`_RayLeaseProcess`.
        """
        if isinstance(proc, _RayLeaseProcess):
            proc.reap()
            return
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        # Give SIGTERM 5s before SIGKILL.
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _harvest_worktree_diff(worktree: Path) -> str:
        """Return the worktree's edits as one ``-p1`` diff, or ``""``.

        A new file is untracked, and ``git diff`` does not see untracked paths.
        Intent-to-add stages their existence so they render as creations without
        committing anything, which is what keeps "edited a file and added one"
        from harvesting a patch that silently omits the addition. ``patches/`` is
        excluded because the harvest itself lands there.

        Args:
            worktree: Per-task worktree holding a ``.git`` marker.

        Returns:
            str: The diff text, or ``""`` when there is nothing to harvest or
                git could not be run.
        """

        def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
            try:
                return subprocess.run(
                    ["git", "-C", str(worktree), *args],
                    capture_output=True,
                    text=True,
                    timeout=120.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("specialist: git %s in %s failed: %r", args[0], worktree, exc)
                return None

        _git("add", "-A", "-N", "--", ".", ":(exclude)patches")
        diff = _git("diff", "HEAD", "--", ".", ":(exclude)patches")
        if diff is None or diff.returncode != 0:
            return ""
        return diff.stdout if diff.stdout.strip() else ""

    @staticmethod
    def _collect_patches(
        worktree: Path | None,
        workspace: Path,
        worktree_base: Path | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """Harvest the specialist's edits, else collect the patch files it wrote.

        Editing the worktree is the primary contract: ``git diff HEAD`` renders
        those edits as one canonical ``-p1`` patch and names its apply root, so
        nothing downstream has to infer which tree it belongs to. Scanning
        ``patches/`` remains for dispatches that never got a worktree, and for
        any git failure here -- a harvest that raises would otherwise discard
        the whole specialist result, done-file included.

        Args:
            worktree: Per-task worktree, or None.
            workspace: Task workspace.
            worktree_base: Checkout the worktree was branched off, which is the
                apply root of anything harvested from it.

        Returns:
            ``(patch_paths, patch_roots)``; the latter is empty for scanned files.
        """
        if worktree is not None and (worktree / ".git").exists():
            harvested_diff = SpecialistSubprocessDispatcher._harvest_worktree_diff(worktree)
            if harvested_diff:
                harvested = worktree / "patches" / "_worktree_diff.patch"
                harvested.parent.mkdir(exist_ok=True)
                harvested.write_text(harvested_diff, encoding="utf-8")
                return [str(harvested)], {str(harvested): str(worktree_base or worktree)}

        out: list[str] = []
        for base in (worktree, workspace):
            if base is None:
                continue
            patches_dir = base / "patches"
            if not patches_dir.is_dir():
                continue
            for ext in ("*.patch", "*.diff"):
                for p in sorted(patches_dir.glob(ext)):
                    out.append(str(p))
        return out, {}

    @staticmethod
    def _read_done(done_file: Path) -> dict[str, Any] | None:
        """Parse a ``specialist_done.json`` file, unwrapping intent envelopes.

        Tolerates missing files and parse errors (logged, returns None).
        When the file holds a ``specialist_done`` intent envelope, the
        inner ``payload`` is merged with the outer keys so callers always
        see a flat dict.

        Args:
            done_file (Path): Path to the candidate done-file.

        Returns:
            dict[str, Any] | None: The parsed (and possibly unwrapped)
                payload, or None when missing / unparseable / not a dict.
        """
        if not done_file.exists():
            return None
        try:
            data = json.loads(done_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "specialist_done.json parse failed at %s: %r",
                done_file,
                exc,
            )
            return None
        if not isinstance(data, dict):
            log.warning(
                "specialist_done.json at %s is not a dict (%r); ignoring",
                done_file,
                type(data).__name__,
            )
            return None
        if str(data.get("intent_type") or "") == "specialist_done" and isinstance(data.get("payload"), dict):
            inner = data["payload"]
            merged: dict[str, Any] = {}
            for k, v in data.items():
                if k in ("intent_type", "payload"):
                    continue
                merged[k] = v
            for k, v in inner.items():
                merged[k] = v
            log.info(
                "_read_done: unwrapped specialist_done intent envelope at %s (proposal_set_len=%d, empty=%s)",
                done_file,
                len(inner.get("proposal_set") or []) if isinstance(inner.get("proposal_set"), list) else 0,
                inner.get("empty"),
            )
            return merged
        return data


__all__ = [
    "AGENT_BACKEND_CLAUDE",
    "AGENT_BACKEND_CODEX",
    "SpecialistAgentUnavailableError",
    "SpecialistSubprocessConfig",
    "SpecialistSubprocessDispatcher",
    "SpecialistSubprocessResult",
    "_pick_worktree_base",
    "_setup_worktree",
    "resolve_codex_executable",
    "resolve_specialist_agent_backend",
]
