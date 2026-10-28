# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What every specialist agent CLI is launched with, whichever CLI it is.

The minimal child environment and the allowlists behind it, the validated
Claude-style MCP server list that each CLI renders in its own format, the
errors a launch that cannot be assembled raises, and Codex's private
``config.toml`` (developer instructions and MCP servers, with secret values
kept in child-only environment variables).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from hyperloom.common.env import env_bool
from hyperloom.common.fs_utils import write_text_nofollow
from hyperloom.common.env_safety import BLOCKED_CHILD_ENV_NAMES, scrub_child_process_env, valid_env_key
from hyperloom.common.visible_devices import GPU_MASK_ENV_NAMES


class SpecialistAgentUnavailableError(RuntimeError):
    """Raised when the agent CLI the deployment needs cannot be assembled.

    A missing runtime or an unconfigurable gateway means this deployment cannot
    run specialists at all. It surfaces as the task's failure rather than being
    absorbed into a fallback CLI that would fail to authenticate.
    """


_SPECIALIST_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        # Session identity, so a child that reports its own spend files it under
        # the same session the parent does. Not a credential.
        "CLAW_SESSION_ID",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        # Hugging Face Hub cache, offline switch and mirror. HF_TOKEN stays out; a login under HF_HOME is still read.
        "HF_ENDPOINT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "INFERENCEX_PATH",
        "LANG",
        "LC_ALL",
        # The agent CLIs are Node processes, which read neither SSL_CERT_FILE nor REQUESTS_CA_BUNDLE; behind a
        # TLS-intercepting gateway these are the only trust knobs that reach them.
        "NODE_EXTRA_CA_CERTS",
        "NODE_TLS_REJECT_UNAUTHORIZED",
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
        "OPENAI_API_KEY",
        "OPENAI_CUSTOM_HEADERS",
    }
)


def _build_specialist_env() -> dict[str, str]:
    """Build a minimal env for Bash-enabled specialist subprocesses."""
    inherit_secrets = env_bool("HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV", True)
    allowed = set(_SPECIALIST_ENV_ALLOWLIST)
    if inherit_secrets:
        allowed.update(_SPECIALIST_SECRET_ENV_ALLOWLIST)
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env = scrub_child_process_env(env)
    # claude's bypassPermissions/--dangerously-skip-permissions refuses to start
    # under root unless IS_SANDBOX=1 (SWSPLAT-42390). Mirror the kernel-agent
    # forge tools (forge_fusion) so specialist authoring
    # subprocesses run on bare-root pods (non-Claw hosts) instead of crashing
    # immediately. setdefault only under root keeps the guard intact elsewhere.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("IS_SANDBOX", "1")
    return env


# The two MCP transports a Claude-style ``mcpServers`` entry may declare.
_MCP_HTTP = "http"
_MCP_STDIO = "stdio"
_CODEX_MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CODEX_MCP_ENV_PREFIX = "HYPERLOOM_CODEX_MCP_ENV_"
_CODEX_PROVIDER_ENV_PREFIX = "HYPERLOOM_CODEX_HTTP_HEADER_"
_CODEX_MCP_RESERVED_ENV_NAMES: frozenset[str] = frozenset(
    {
        *_SPECIALIST_ENV_ALLOWLIST,
        *_SPECIALIST_SECRET_ENV_ALLOWLIST,
        # Every mask spelling, not the three canonical ones: the reason a mask
        # is reserved is that setting it re-pins the specialist's cards, and a
        # guard that names only the modern spellings is bypassed by the legacy
        # ones it honours just as well.
        *GPU_MASK_ENV_NAMES,
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
        "CODEX_HOME",
        "DISABLE_AUTOUPDATER",
        "INFERENCE_OPTIMIZER_SPECIALIST_GPU_IDS",
        "IS_SANDBOX",
    }
)


@dataclass(frozen=True)
class McpServer:
    """One Claude-style MCP server, validated independently of the CLI that will run it.

    ``headers`` and ``env`` keep their values as written, so any ``${NAME}``
    reference is still in them; resolving it is the renderer's job.
    """

    name: str
    transport: str
    url: str = ""
    command: str = ""
    args: tuple[str, ...] = ()
    headers: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    @property
    def is_http(self) -> bool:
        """Whether the server is reached over HTTP rather than spawned over stdio."""
        return self.transport == _MCP_HTTP


def _http_server(name: str, server: dict) -> McpServer:
    """Validate the HTTP fields of one ``mcpServers`` entry."""
    unsupported = set(server) - {"type", "url", "headers"}
    if unsupported:
        raise SpecialistAgentUnavailableError(
            f"MCP server {name!r} has unsupported HTTP fields: {sorted(unsupported)!r}"
        )
    url = server.get("url")
    if not isinstance(url, str) or not url.strip():
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} requires a non-empty URL")
    parsed_url = urlsplit(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} has invalid HTTP URL")
    if parsed_url.username is not None or parsed_url.password is not None:
        raise SpecialistAgentUnavailableError(
            f"MCP server {name!r} URL embeds credentials; use environment-backed headers instead"
        )
    headers = server.get("headers") or {}
    if not isinstance(headers, dict):
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} headers must be an object")
    rows: list[tuple[str, str]] = []
    for raw_header, raw_value in headers.items():
        header = str(raw_header or "").strip()
        if not header or not isinstance(raw_value, str):
            raise SpecialistAgentUnavailableError(f"MCP server {name!r} has an invalid HTTP header")
        rows.append((header, raw_value))
    return McpServer(name=name, transport=_MCP_HTTP, url=url.strip(), headers=tuple(rows))


def _stdio_server(name: str, server: dict) -> McpServer:
    """Validate the stdio fields of one ``mcpServers`` entry."""
    unsupported = set(server) - {"type", "command", "args", "env"}
    if unsupported:
        raise SpecialistAgentUnavailableError(
            f"MCP server {name!r} has unsupported stdio fields: {sorted(unsupported)!r}"
        )
    command = server.get("command")
    if not isinstance(command, str) or not command.strip():
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} requires a non-empty command")
    args = server.get("args") or []
    if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} args must be a list of strings")
    raw_env = server.get("env") or {}
    if not isinstance(raw_env, dict):
        raise SpecialistAgentUnavailableError(f"MCP server {name!r} env must be an object")
    rows: list[tuple[str, str]] = []
    for raw_key, raw_value in raw_env.items():
        key = str(raw_key or "").strip()
        if not valid_env_key(key) or key in BLOCKED_CHILD_ENV_NAMES or not isinstance(raw_value, str):
            raise SpecialistAgentUnavailableError(f"MCP server {name!r} has an unsafe or invalid env entry {key!r}")
        rows.append((key, raw_value))
    return McpServer(name=name, transport=_MCP_STDIO, command=command.strip(), args=tuple(args), env=tuple(rows))


def load_mcp_servers(config_path: str | None) -> list[McpServer]:
    """Read and validate the servers of a Claude-style MCP config file.

    Raises:
        SpecialistAgentUnavailableError: When the file cannot be read, or a
            server declares an unsupported transport or field, an unsafe URL,
            or a malformed header, argument or environment entry.
    """
    if not config_path:
        return []
    path = Path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecialistAgentUnavailableError(f"cannot read specialist MCP config {path}: {exc}") from exc
    servers = payload.get("mcpServers") if isinstance(payload, dict) else None
    if not isinstance(servers, dict):
        raise SpecialistAgentUnavailableError(f"specialist MCP config {path} must contain an object at 'mcpServers'")

    loaded: list[McpServer] = []
    for raw_name, raw_server in servers.items():
        name = str(raw_name or "")
        if not isinstance(raw_server, dict):
            raise SpecialistAgentUnavailableError(f"MCP server {name!r} must be an object")
        declared_type = str(raw_server.get("type") or "").strip().lower()
        if not declared_type:
            declared_type = _MCP_HTTP if raw_server.get("url") else _MCP_STDIO if raw_server.get("command") else ""
        if declared_type == _MCP_HTTP:
            loaded.append(_http_server(name, raw_server))
        elif declared_type == _MCP_STDIO:
            loaded.append(_stdio_server(name, raw_server))
        else:
            raise SpecialistAgentUnavailableError(
                f"MCP server {name!r} uses unsupported transport {declared_type or '<missing>'!r}; "
                "expected 'http' or 'stdio'"
            )
    return loaded


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
    prefix: str = _CODEX_MCP_ENV_PREFIX,
) -> str:
    """Return a collision-free child-only variable name for one MCP header."""
    base = f"{prefix}{index}"
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
    """Render the MCP servers of a Claude-style config as secret-free Codex TOML lines.

    Stdio environment values and HTTP header values are copied into the child
    environment. The generated TOML contains only their variable names through
    Codex's verified ``env_vars`` and ``env_http_headers`` fields.
    """
    lines: list[str] = []
    env_additions: dict[str, str] = {}
    header_index = 0
    for server in load_mcp_servers(config_path):
        name = server.name
        if not _CODEX_MCP_SERVER_NAME_RE.fullmatch(name):
            raise SpecialistAgentUnavailableError(
                f"Codex MCP server name {name!r} is not translatable; "
                "use a letter followed by letters, digits, '_' or '-'"
            )
        lines.append(f"[mcp_servers.{name}]")
        if server.is_http:
            lines.append(f"url = {_toml_string(server.url)}")
            header_rows: list[tuple[str, str]] = []
            for header, raw_value in server.headers:
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
            env_names: list[str] = []
            for key, raw_value in server.env:
                if (
                    key in _CODEX_MCP_RESERVED_ENV_NAMES
                    or key in protected_env_names
                    or key.startswith((_CODEX_PROVIDER_ENV_PREFIX, _CODEX_MCP_ENV_PREFIX))
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
            lines.append(f"command = {_toml_string(server.command)}")
            lines.append(f"args = {json.dumps(list(server.args), ensure_ascii=False)}")
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
    # A previous run's account could own, or have replaced with a link, what is here.
    dir_fd = os.open(codex_home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(dir_fd, 0o700)
    finally:
        os.close(dir_fd)
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
    temporary = write_text_nofollow(
        codex_home, (config_path.with_suffix(".toml.tmp").name,), "\n".join(lines).rstrip() + "\n", mode=0o600
    )
    os.replace(temporary, config_path)
    config_path.chmod(0o600)
    return config_path
