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

from hyperloom.common.deepseek_session import DEEPSEEK_PROVIDER_ROUTE, DeepSeekSessionUnavailableError, gateway_env
from hyperloom.common.fs_utils import write_text_nofollow
from hyperloom.inference_optimizer.trace.agent_trace import AgentTrace
from hyperloom.inference_optimizer.trace.parse_dsh_session import parse_dsh_session_trace

from .agent_cli_config import (
    McpServer,
    SpecialistAgentUnavailableError,
    _codex_provider_resolver_overlay,
    _private_mcp_env_name,
    _resolve_env_references,
    load_mcp_servers,
)

# ``dsh --profile headless`` reads its task from argv alone, and Linux caps one argument at MAX_ARG_STRLEN (32 pages).
_TASK_ARG_LIMIT_BYTES = 32 * 4096
# dsh-mcp-client's ``serverName`` rule.
_PATCH_CONFIG = "  config:"
_MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_MCP_ENV_PREFIX = "HYPERLOOM_DSH_MCP_ENV_"
# The shipped headless profile's default instruction budget; raised when the system prompt needs more.
_DEFAULT_INSTRUCTION_BYTES = 65536
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_ENCODING = "utf-8"


@dataclass(frozen=True)
class DshLaunchRequest:
    """What one specialist's headless ``dsh`` run is launched with.

    Attributes:
        executable: The resolved ``dsh`` CLI, ``""`` when none exists.
        model: Harness model id.
        mcp_config_path: Claude-style MCP config to translate, if any.
        workspace: The task workspace; its ``.dsh`` is the run's ``DSH_HOME``.
        system_prompt: Sent as the run's baseline instructions.
        user_prompt: The headless task.
    """

    executable: str
    model: str
    mcp_config_path: str | None
    workspace: Path
    system_prompt: str
    user_prompt: str


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
    """Render validated MCP servers as ``dsh-mcp-client`` patch rows.

    Header and env values are resolved into child-only variables that the rows
    reference, so the patch file carries no secret.
    """

    def __init__(self, source: Mapping[str, str], protected_env_names: frozenset[str]) -> None:
        self.source = source
        self.protected_env_names = protected_env_names
        self.rows: list[str] = []
        self.env_additions: dict[str, str] = {}

    def add(self, server: McpServer) -> None:
        name = server.name
        if not _MCP_SERVER_NAME_RE.fullmatch(name):
            raise SpecialistAgentUnavailableError(
                f"dsh MCP server name {name!r} is not translatable; use 1-32 letters, digits, '_' or '-'"
            )
        self.rows.extend(
            [
                f"    - id: {_yaml_string('mcp-' + name)}",
                "      name: '@deepseek-ai/dsh-mcp-client'",
                "      config:",
                f"        serverName: {_yaml_string(name)}",
            ]
        )
        if server.is_http:
            self._http(server)
        else:
            self._stdio(server)

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

    def _http(self, server: McpServer) -> None:
        self.rows.extend(["        transport: streamable-http", f"        url: {_yaml_string(server.url)}"])
        if server.headers:
            self.rows.append("        headers:")
        for header, raw_value in server.headers:
            reference = self._private_reference(raw_value, context=f"dsh MCP server {server.name!r} header {header!r}")
            self.rows.append(f"          {_yaml_string(header)}: {reference}")

    def _stdio(self, server: McpServer) -> None:
        self.rows.extend(
            [
                "        transport: stdio",
                f"        command: {_yaml_string(server.command)}",
                f"        args: {json.dumps(list(server.args), ensure_ascii=False)}",
            ]
        )
        if server.env:
            self.rows.append("        env:")
        for key, raw_value in server.env:
            reference = self._private_reference(raw_value, context=f"dsh MCP server {server.name!r} env {key!r}")
            self.rows.append(f"          {_yaml_string(key)}: {reference}")


def _mcp_rows(config_path: str | None, *, source: Mapping[str, str], protected_env_names: frozenset[str]) -> _McpRows:
    """The ``dsh-mcp-client`` rows for the servers of a Claude-style MCP config file."""
    rows = _McpRows(source, protected_env_names)
    for server in load_mcp_servers(config_path):
        rows.add(server)
    return rows


def _render_patch(*, model: str, instruction_bytes: int, mcp_rows: list[str]) -> str:
    """The patch overlay a specialist's headless ``dsh`` run boots with."""
    lines = [
        # The shipped headless profile enables a session-title plugin whose
        # implementation package (@deepseek-ai/dsh-session-title-llm) the
        # runtime does not ship, and boot fails while it is enabled.
        "- id: session-title-llm",
        "  disabled: true",
        # The harness compresses its session log by default, a format its own
        # documentation says is to be read through the backend; the harvest
        # reads the plain JSONL instead.
        "- id: session-persistence-jsonl",
        _PATCH_CONFIG,
        "    root: !!js dshHomePath('sessions')",
        "    compression: none",
        "- id: agent-default-model",
        _PATCH_CONFIG,
        f"    provider: {DEEPSEEK_PROVIDER_ROUTE}",
        f"    model: {_yaml_string(model)}",
        # ``$DSH_HOME/AGENTS.md`` carries the system prompt; never clip it.
        "- id: agent-instructions",
        _PATCH_CONFIG,
        f"    maxBytes: {instruction_bytes + _DEFAULT_INSTRUCTION_BYTES}",
    ]
    if mcp_rows:
        lines.append("- insert:")
        lines.extend(mcp_rows)
    return "\n".join(lines) + "\n"


def _write_private_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable by the owner only."""
    write_text_nofollow(path.parent, (path.name,), text, mode=_PRIVATE_FILE_MODE)


def _prepare_home(home: Path, *, instructions: str, patch: str) -> Path:
    """Write the task's private ``DSH_HOME``; return its patch file."""
    patch_path = home / "specialist_patch.yml"
    try:
        home.mkdir(parents=True, mode=_PRIVATE_DIR_MODE, exist_ok=True)
        # A previous run's account could have replaced the directory with a link.
        dir_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fchmod(dir_fd, _PRIVATE_DIR_MODE)
        finally:
            os.close(dir_fd)
        # The harvest reads every session under DSH_HOME; a rerun of this task id must not count the last one.
        shutil.rmtree(home / "sessions", ignore_errors=True)
        _write_private_text(home / "AGENTS.md", instructions)
        _write_private_text(patch_path, patch)
    except OSError as exc:
        raise SpecialistAgentUnavailableError(f"cannot prepare private dsh home under {home}: {exc}") from exc
    return patch_path


def build_dsh_launch(request: DshLaunchRequest, base_env: Mapping[str, str]) -> DshLaunch:
    """Build the ``dsh --profile headless`` argv of one specialist task and its child-only env values.

    Raises:
        SpecialistAgentUnavailableError: When no ``dsh`` exists, the gateway is
            unset, the user prompt cannot travel as one argument, or the MCP
            config or the private home cannot be prepared.
    """
    executable = request.executable
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
    try:
        gateway = gateway_env(_codex_provider_resolver_overlay(base_env))
    except DeepSeekSessionUnavailableError as exc:
        raise SpecialistAgentUnavailableError(str(exc)) from exc
    env_additions = {
        **gateway,
        "DSH_PERMISSION_MODE": "danger-full-access",
        "DSH_TELEMETRY_MODE": "DISABLED",
    }
    source = {**os.environ, **_codex_provider_resolver_overlay(base_env)}
    mcp = _mcp_rows(request.mcp_config_path, source=source, protected_env_names=frozenset(env_additions))
    env_additions.update(mcp.env_additions)
    instructions = request.system_prompt or ""
    home = dsh_home(request.workspace)
    patch_path = _prepare_home(
        home,
        instructions=instructions,
        patch=_render_patch(
            model=request.model,
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
    """What a finished ``dsh`` run left under its ``DSH_HOME``.

    dsh prints only its final message and reasoning, so the session logs are the record.
    """
    return parse_dsh_session_trace(dsh_home(workspace))


def remove_dsh_profiles(workspace: Path) -> None:
    """Remove the megabytes of profile modules dsh materializes per ``DSH_HOME``; a finished task never boots it again."""
    home = dsh_home(workspace)
    if not home.is_symlink():  # a link left by the task's account must not send the removal elsewhere
        shutil.rmtree(home / "profiles", ignore_errors=True)
