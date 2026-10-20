# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch a specialist on the DeepSeek Harness ``dsh`` CLI and read back what it did.

``dsh --profile headless`` answers one task and exits. Each task gets a
private ``DSH_HOME`` holding the system prompt as ``AGENTS.md`` (sent as the
run's baseline instructions) and a patch overlay for the shipped profile;
the run's record is the session log the harness writes under that home.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hyperloom.common.env_safety import BLOCKED_CHILD_ENV_NAMES, valid_env_key
from hyperloom.common.llm_config import DEFAULT_DEEPSEEK_HARNESS_MODEL, LLMConfigError, resolve_openai_client_config
from hyperloom.inference_optimizer.trace.agent_trace import AgentTrace
from hyperloom.inference_optimizer.trace.parse_dsh_session import parse_dsh_session_trace

from .agent_cli_config import (
    MCP_HTTP,
    MCP_STDIO,
    SpecialistAgentUnavailableError,
    _codex_provider_resolver_overlay,
    _private_mcp_env_name,
    _resolve_env_references,
)
from .subprocess_ import SpecialistSpawn, SpecialistSubprocessConfig

# ``dsh --profile headless`` reads its task from argv alone, and Linux caps one argument at MAX_ARG_STRLEN (32 pages).
_TASK_ARG_LIMIT_BYTES = 32 * 4096
# dsh-mcp-client's ``serverName`` rule.
_MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_MCP_ENV_PREFIX = "HYPERLOOM_DSH_MCP_ENV_"
# The shipped headless profile's default instruction budget; raised when the system prompt needs more.
_DEFAULT_INSTRUCTION_BYTES = 65536
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_ENCODING = "utf-8"


@dataclass(frozen=True)
class DshLaunch:
    """The argv of one headless ``dsh`` run and the env values only its child sees."""

    cmd: list[str]
    env_additions: dict[str, str]


def dsh_home(workspace: Path) -> Path:
    """The private ``DSH_HOME`` of the task whose workspace is ``workspace``."""
    return workspace / ".dsh"


def resolve_dsh_executable(explicit: str = "") -> str:
    """Resolve the DeepSeek Harness CLI a specialist subprocess should spawn.

    Order: an explicit path, then ``dsh`` on ``$PATH``, then the runtime the
    harness SDK ships, so an operator's own installation still wins.

    Args:
        explicit: Operator-configured path; returned as-is when non-empty.

    Returns:
        The resolved executable path, or ``""`` when no harness runtime exists.
    """
    pinned = (explicit or "").strip()
    if pinned:
        return pinned
    on_path = shutil.which("dsh")
    if on_path:
        return on_path
    try:
        # Installed as the ``deepseek-harness-sdk`` package's own runtime dependency.
        from deepseek_harness_runtime import resolve_bundled_launch_args  # type: ignore[import-not-found]

        launch = tuple(resolve_bundled_launch_args())
    except (ImportError, OSError):
        return ""
    return launch[0] if len(launch) == 1 and Path(launch[0]).exists() else ""


def _yaml_string(value: str) -> str:
    """Encode a string as a YAML double-quoted scalar (JSON strings are a subset)."""
    return json.dumps(str(value), ensure_ascii=False)


class _McpRows:
    """Translate Claude-style MCP servers into ``dsh-mcp-client`` patch rows.

    Header and env values are resolved into child-only variables that the rows
    reference, so the patch file carries no secret.
    """

    def __init__(self, source: Mapping[str, str], protected_env_names: frozenset[str]) -> None:
        self.source = source
        self.protected_env_names = protected_env_names
        self.rows: list[str] = []
        self.env_additions: dict[str, str] = {}

    def add(self, name: str, server: Any) -> None:
        if not _MCP_SERVER_NAME_RE.fullmatch(name):
            raise SpecialistAgentUnavailableError(
                f"dsh MCP server name {name!r} is not translatable; use 1-32 letters, digits, '_' or '-'"
            )
        if not isinstance(server, dict):
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} must be an object")
        transport = str(server.get("type") or "").strip().lower()
        if not transport:
            transport = MCP_HTTP if server.get("url") else MCP_STDIO if server.get("command") else ""
        if transport not in (MCP_HTTP, MCP_STDIO):
            raise SpecialistAgentUnavailableError(
                f"dsh MCP server {name!r} uses unsupported transport {transport or '<missing>'!r}; "
                "expected 'http' or 'stdio'"
            )
        unsupported = set(server) - {
            "type",
            *(("url", "headers") if transport == MCP_HTTP else ("command", "args", "env")),
        }
        if unsupported:
            raise SpecialistAgentUnavailableError(
                f"dsh MCP server {name!r} has unsupported {transport} fields: {sorted(unsupported)!r}"
            )
        self.rows.extend(
            [
                f"    - id: {_yaml_string('mcp-' + name)}",
                "      name: '@deepseek-ai/dsh-mcp-client'",
                "      config:",
                f"        serverName: {_yaml_string(name)}",
            ]
        )
        if transport == MCP_HTTP:
            self._http(name, server)
        else:
            self._stdio(name, server)

    def _private_reference(self, raw_value: str, *, context: str) -> str:
        """Hold one value in a child-only variable; return the patch expression that reads it."""
        env_name = _private_mcp_env_name(
            len(self.env_additions),
            source=self.source,
            additions=self.env_additions,
            protected_env_names=self.protected_env_names,
            prefix=_MCP_ENV_PREFIX,
        )
        self.env_additions[env_name] = _resolve_env_references(raw_value, source=self.source, context=context)
        return f"!!js process.env.{env_name}"

    def _http(self, name: str, server: dict[str, Any]) -> None:
        url = server.get("url")
        parsed = urlsplit(url.strip()) if isinstance(url, str) else None
        if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} has invalid HTTP URL")
        if parsed.username is not None or parsed.password is not None:
            raise SpecialistAgentUnavailableError(
                f"dsh MCP server {name!r} URL embeds credentials; use environment-backed headers instead"
            )
        headers = server.get("headers") or {}
        if not isinstance(headers, dict):
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} headers must be an object")
        self.rows.extend(["        transport: streamable-http", f"        url: {_yaml_string(url.strip())}"])
        if headers:
            self.rows.append("        headers:")
        for raw_header, raw_value in headers.items():
            header = str(raw_header or "").strip()
            if not header or not isinstance(raw_value, str):
                raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} has an invalid HTTP header")
            reference = self._private_reference(raw_value, context=f"dsh MCP server {name!r} header {header!r}")
            self.rows.append(f"          {_yaml_string(header)}: {reference}")

    def _stdio(self, name: str, server: dict[str, Any]) -> None:
        command = server.get("command")
        if not isinstance(command, str) or not command.strip():
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} requires a non-empty command")
        args = server.get("args") or []
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} args must be a list of strings")
        raw_env = server.get("env") or {}
        if not isinstance(raw_env, dict):
            raise SpecialistAgentUnavailableError(f"dsh MCP server {name!r} env must be an object")
        self.rows.extend(
            [
                "        transport: stdio",
                f"        command: {_yaml_string(command.strip())}",
                f"        args: {json.dumps(args, ensure_ascii=False)}",
            ]
        )
        if raw_env:
            self.rows.append("        env:")
        for raw_key, raw_value in raw_env.items():
            key = str(raw_key or "").strip()
            if not valid_env_key(key) or key in BLOCKED_CHILD_ENV_NAMES or not isinstance(raw_value, str):
                raise SpecialistAgentUnavailableError(
                    f"dsh MCP server {name!r} has an unsafe or invalid env entry {key!r}"
                )
            reference = self._private_reference(raw_value, context=f"dsh MCP server {name!r} env {key!r}")
            self.rows.append(f"          {_yaml_string(key)}: {reference}")


def _mcp_rows(config_path: str | None, *, source: Mapping[str, str], protected_env_names: frozenset[str]) -> _McpRows:
    """The ``dsh-mcp-client`` rows for the servers of a Claude-style MCP config file."""
    rows = _McpRows(source, protected_env_names)
    if not config_path:
        return rows
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding=_ENCODING))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecialistAgentUnavailableError(f"cannot read specialist MCP config {path}: {exc}") from exc
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(servers, dict):
        raise SpecialistAgentUnavailableError(f"specialist MCP config {path} must contain an object at 'mcpServers'")
    for name, server in servers.items():
        rows.add(str(name or ""), server)
    return rows


def _render_patch(*, model: str, instruction_bytes: int, mcp_rows: list[str]) -> str:
    """The patch overlay a specialist's headless ``dsh`` run boots with."""
    lines = [
        # The shipped headless profile lists a session-title plugin whose
        # package the runtime does not ship, and boot fails while it is enabled.
        "- id: session-title-llm",
        "  disabled: true",
        "- id: agent-default-model",
        "  config:",
        "    provider: deepseek-official",
        f"    model: {_yaml_string(model)}",
        # ``$DSH_HOME/AGENTS.md`` carries the system prompt; never clip it.
        "- id: agent-instructions",
        "  config:",
        f"    maxBytes: {instruction_bytes + _DEFAULT_INSTRUCTION_BYTES}",
    ]
    if mcp_rows:
        lines.append("- insert:")
        lines.extend(mcp_rows)
    return "\n".join(lines) + "\n"


def _write_private_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable by the owner only."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w", encoding=_ENCODING) as handle:
        handle.write(text)
    path.chmod(_PRIVATE_FILE_MODE)


def _gateway_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """The harness's ``DEEPSEEK_*`` gateway names, set from the OpenAI-compatible line it speaks."""
    try:
        gateway = resolve_openai_client_config(env=_codex_provider_resolver_overlay(base_env))
    except LLMConfigError as exc:
        raise SpecialistAgentUnavailableError(
            f"the DeepSeek Harness needs OPENAI_BASE_URL and OPENAI_API_KEY: {exc}"
        ) from exc
    if not gateway.base_url or not gateway.api_key:
        raise SpecialistAgentUnavailableError("the DeepSeek Harness needs both OPENAI_BASE_URL and OPENAI_API_KEY")
    return {"DEEPSEEK_BASE_URL": gateway.base_url, "DEEPSEEK_API_KEY": gateway.api_key}


def _prepare_home(home: Path, *, instructions: str, patch: str) -> Path:
    """Write the task's private ``DSH_HOME``; return its patch file."""
    patch_path = home / "specialist_patch.yml"
    try:
        home.mkdir(parents=True, mode=_PRIVATE_DIR_MODE, exist_ok=True)
        home.chmod(_PRIVATE_DIR_MODE)
        # The harvest reads every session under DSH_HOME; a rerun of this task id must not count the last one.
        shutil.rmtree(home / "sessions", ignore_errors=True)
        _write_private_text(home / "AGENTS.md", instructions)
        _write_private_text(patch_path, patch)
    except OSError as exc:
        raise SpecialistAgentUnavailableError(f"cannot prepare private dsh home under {home}: {exc}") from exc
    return patch_path


def build_dsh_launch(
    config: SpecialistSubprocessConfig,
    request: SpecialistSpawn,
    base_env: Mapping[str, str],
) -> DshLaunch:
    """Build the ``dsh --profile headless`` argv of one specialist task and its child-only env values.

    Raises:
        SpecialistAgentUnavailableError: When no ``dsh`` exists, the gateway is
            unset, the user prompt cannot travel as one argument, or the MCP
            config or the private home cannot be prepared.
    """
    executable = resolve_dsh_executable(config.dsh_executable)
    if not executable:
        raise SpecialistAgentUnavailableError(
            "no dsh CLI found: set SpecialistSubprocessConfig.dsh_executable, install `dsh` on $PATH, "
            "or install the DeepSeek Harness SDK (pip install 'hyperloom-inference_optimizer[deepseek]')"
        )
    task_bytes = len(request.user_prompt.encode(_ENCODING))
    if task_bytes >= _TASK_ARG_LIMIT_BYTES:
        raise SpecialistAgentUnavailableError(
            f"the specialist user prompt is {task_bytes} bytes; dsh headless takes its "
            f"task as one argument, which the kernel caps below {_TASK_ARG_LIMIT_BYTES} bytes"
        )
    env_additions = {
        **_gateway_env(base_env),
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DSH_TELEMETRY_MODE": "DISABLED",
    }
    source = {**os.environ, **_codex_provider_resolver_overlay(base_env)}
    mcp = _mcp_rows(config.mcp_config_path, source=source, protected_env_names=frozenset(env_additions))
    env_additions.update(mcp.env_additions)
    instructions = request.system_prompt or ""
    home = dsh_home(request.workspace)
    patch_path = _prepare_home(
        home,
        instructions=instructions,
        patch=_render_patch(
            model=config.model or DEFAULT_DEEPSEEK_HARNESS_MODEL,
            instruction_bytes=len(instructions.encode(_ENCODING)),
            mcp_rows=mcp.rows,
        ),
    )
    env_additions["DSH_HOME"] = str(home)
    # The launcher consumes the first ``--`` and the headless app the second,
    # so a task that starts with ``-`` is not parsed as an option.
    cmd = [executable, "--profile", "headless", "--patch", str(patch_path), "--", "--", request.user_prompt]
    return DshLaunch(cmd=cmd, env_additions=env_additions)


def harvest_dsh_trace(workspace: Path) -> AgentTrace:
    """What a finished ``dsh`` run left under its ``DSH_HOME``; its profile tree is removed after.

    dsh prints only its final message and reasoning, so the session logs are the record.
    """
    home = dsh_home(workspace)
    trace = parse_dsh_session_trace(home)
    # dsh materializes megabytes of profile modules per DSH_HOME; a finished task never boots it again.
    shutil.rmtree(home / "profiles", ignore_errors=True)
    return trace
