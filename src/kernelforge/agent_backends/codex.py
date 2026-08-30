# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codex Python SDK backend for Forge implementer sessions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderError,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.llm import LlmGateway, resolve_openai_gateway
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard

log = logging.getLogger(__name__)

_TOML_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")

DEFAULT_CODEX_MODEL = "gpt-5.6"
FALLBACK_CODEX_MODEL = "gpt-5.5"


class CodexBackendError(AgentProviderError):
    """Base error for Codex backend failures."""


class CodexUnavailableError(
    CodexBackendError,
    AgentProviderUnavailableError,
):
    """Report a missing CLI or incomplete gateway configuration."""


class CodexExecutionError(CodexBackendError):
    """Report a failed or timed-out Codex SDK session.

    ``session_id`` carries the thread handle when the failure happened AFTER the
    thread existed. By then the session already holds every turn it spent reading,
    building and benchmarking, so the caller has to continue that thread rather
    than open a new one — which is what ``session_resume`` reads it for.
    """

    def __init__(self, *args: Any, session_id: str = "") -> None:
        super().__init__(*args)
        self.session_id = str(session_id or "")


def _write_git_guard(directory: Path) -> dict[str, str]:
    """Create a PATH wrapper that rejects git repository mutations."""
    real_git = shutil.which("git")
    if not real_git:
        raise CodexUnavailableError("git is required for Codex workspace guards")

    wrapper = directory / "git"
    allowed = (
        '""|blame|cat-file|describe|diff|for-each-ref|grep|help|log|ls-files|'
        "ls-tree|merge-base|name-rev|rev-parse|shortlog|show|status|version"
    )
    wrapper.write_text(
        "#!/bin/sh\n"
        "cmd=''\n"
        "expect_value=0\n"
        'for arg in "$@"; do\n'
        '  if [ "$expect_value" -eq 1 ]; then expect_value=0; continue; fi\n'
        '  case "$arg" in\n'
        "    -C|-c|--git-dir|--work-tree) expect_value=1 ;;\n"
        "    --git-dir=*|--work-tree=*|-*) ;;\n"
        '    *) cmd="$arg"; break ;;\n'
        "  esac\n"
        "done\n"
        f'case "$cmd" in {allowed}) ;;\n'
        '  *) echo "forge: non-read-only git command denied: $cmd" >&2\n'
        "     exit 126 ;;\n"
        "esac\n"
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{directory}{os.pathsep}{env.get('PATH', '')}"
    return env


def resolve_codex_cli(explicit: str = "") -> str:
    """Resolve an optional SDK runtime override from generic configuration."""
    selected = explicit.strip() or os.environ.get("FORGE_AGENT_CLI", "").strip()
    if selected:
        path = Path(selected).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return ""
    return ""


def resolve_codex_model(explicit: str = "") -> str:
    """Resolve a Codex model from generic runtime input or built-in default."""
    model = explicit.strip() or DEFAULT_CODEX_MODEL
    if re.search(r"(^|[/.:_-])claude(?:[/.:_-]|$)", model, re.IGNORECASE):
        raise CodexExecutionError(
            f"model {model!r} is a Claude model and cannot be used by the "
            "Codex provider; select a Codex-compatible model"
        )
    return model


def resolve_codex_reasoning_effort(explicit: str = "") -> str:
    """Map the generic maximum effort onto Codex's highest supported level."""
    effort = (explicit or "high").strip().lower()
    if effort == "max":
        return "xhigh"
    if effort not in {"none", "low", "medium", "high", "xhigh"}:
        raise CodexExecutionError(f"unsupported Codex reasoning effort: {explicit!r}")
    return effort


def resolve_codex_gateway() -> LlmGateway:
    """Derive the Codex OpenAI-compatible gateway from process environment."""
    return resolve_openai_gateway()


def _resolve_gateway() -> LlmGateway:
    """Resolve the built-in Codex gateway configuration."""
    return resolve_codex_gateway()


def _toml_string(value: str) -> str:
    """Encode a safe TOML basic string for Codex SDK config overrides."""
    return json.dumps(value)


def _toml_key(name: str) -> str:
    """Encode one segment of a TOML dotted key, quoting it only when required."""
    return name if _TOML_BARE_KEY_RE.fullmatch(name) else _toml_string(name)


def _provider_overrides(gateway: LlmGateway) -> list[str]:
    """Build SDK config overrides without copying API secrets."""
    if not gateway.is_complete():
        raise CodexUnavailableError(
            "Codex gateway is not configured; it speaks the OpenAI-compatible "
            "protocol and needs both OPENAI_BASE_URL and OPENAI_API_KEY. The "
            "ANTHROPIC_* line belongs to Claude and is not a substitute."
        )

    provider = "forge"
    overrides = [
        f"model_provider={_toml_string(provider)}",
        f"model_providers.{provider}.name={_toml_string(provider)}",
        f"model_providers.{provider}.base_url={_toml_string(gateway.base_url)}",
        f"model_providers.{provider}.wire_api={_toml_string('responses')}",
        f"model_providers.{provider}.env_key={_toml_string(gateway.key_env)}",
    ]
    # Every header the operator configured for THIS provider, not just the
    # gateway's mandatory ``user``: an APIM subscription key is equally required.
    overrides.extend(
        f"model_providers.{provider}.http_headers.{_toml_key(name)}={_toml_string(value)}"
        for name, value in sorted(gateway.headers.items())
    )
    return overrides


def _codex_instructions(spec: AgentRunSpec) -> str:
    """Adapt shared system instructions to Codex SDK capabilities."""
    write_guidance = (
        "Use your native patch/edit capability for the files the current request explicitly allows you to change."
        if spec.writable
        else ("This is a read-only session. Do not edit, create, delete, or rename any file.")
    )
    runtime = f"""\
## Codex runtime mapping
Use shell commands to inspect and search files. References below to Read, Edit,
Write, Grep, Glob, or Bash describe equivalent capabilities, not literal tool
names. {write_guidance}

Do not run git commands that change repository state. Do not commit, reset,
checkout, stash, clean, or alter branches. The Forge loop owns all git state.
Follow the task-specific write scope and output format below exactly. Stop when
that task is complete; do not continue open-ended exploration.
"""
    return f"{runtime}\n## System instructions\n{spec.system_prompt}"


def _usage_value(payload: dict[str, Any], *keys: str) -> int:
    """Read the first valid non-negative integer from usage aliases."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return 0


def _normalize_sdk_usage(usage: Any) -> dict[str, Any]:
    """Normalize the latest SDK turn usage into Forge accounting fields."""
    if usage is None:
        return {}
    breakdown = getattr(usage, "last", usage)
    if hasattr(breakdown, "model_dump"):
        payload = breakdown.model_dump()
    elif isinstance(breakdown, dict):
        payload = breakdown
    else:
        return {}
    return {
        "input_tokens": _usage_value(payload, "input_tokens"),
        "output_tokens": _usage_value(payload, "output_tokens"),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": _usage_value(
            payload,
            "cached_input_tokens",
        ),
    }


def _sdk_item_dict(item: Any) -> dict[str, Any]:
    """Convert one typed SDK thread item into a provider-neutral mapping."""
    root = getattr(item, "root", item)
    if hasattr(root, "model_dump"):
        try:
            dumped = root.model_dump(by_alias=True, mode="json")
        except TypeError:
            dumped = root.model_dump(by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    return dict(root) if isinstance(root, dict) else {}


def _normalize_sdk_result(result: Any, session_id: str) -> AgentRunResult:
    """Normalize one completed SDK turn into the shared backend result."""
    text = str(getattr(result, "final_response", "") or "").strip()
    file_changes: list[str] = []
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    findings: list[str] = []
    edit_count = 0
    for raw_item in getattr(result, "items", ()) or ():
        item = _sdk_item_dict(raw_item)
        item_type = str(item.get("type") or "")
        normalized_type = item_type.replace("_", "").lower()
        if normalized_type == "agentmessage" and not text:
            message = item.get("text")
            if isinstance(message, str):
                text = message.strip()
        elif normalized_type == "commandexecution":
            command = item.get("command") or item_type
            metadata = {key: item[key] for key in ("exitCode", "status") if key in item}
            tool_calls.append((str(command), metadata))
        elif normalized_type == "mcptoolcall":
            server = str(item.get("server") or "mcp")
            tool = str(item.get("tool") or "tool")
            metadata = {key: item[key] for key in ("status", "error") if key in item}
            tool_calls.append((f"{server}.{tool}", metadata))
        elif normalized_type in {"collabagenttoolcall", "collabtoolcall"}:
            command = item.get("tool") or "collab_agent"
            metadata = {
                key: item[key]
                for key in (
                    "status",
                    "senderThreadId",
                    "receiverThreadIds",
                )
                if key in item
            }
            tool_calls.append((str(command), metadata))
        elif normalized_type == "subagentactivity":
            command = item.get("kind") or "subagent"
            tool_calls.append(
                (
                    str(command),
                    {
                        "agentThreadId": item.get("agentThreadId", ""),
                    },
                )
            )
        elif normalized_type == "filechange":
            item_paths = [
                change["path"]
                for change in item.get("changes", ())
                if isinstance(change, dict) and isinstance(change.get("path"), str)
            ]
            file_changes.extend(item_paths)
            if item_paths:
                edit_count += 1

    # The SDK reports a provider-side failure in-band: the turn "completes" while
    # the model never answered. Reporting that as a successful agent_stopped made
    # a rate limit indistinguishable from a deliberate no-op -- resume never
    # fired, and the empty diff was recorded as NO_CHANGES, i.e. an optimization
    # verdict about a kernel nobody looked at. Label it the way ClaudeBackend
    # does so `session_resume.is_api_failure` sees it.
    error = getattr(result, "error", None)
    subtype = "success"
    end_reason = "agent_stopped"
    stderr_tail = ""
    if error is not None:
        message = str(getattr(error, "message", None) or error)
        findings.append(message)
        stderr_tail = message[:2000]
        lowered = message.lower()
        if "maximum number of turns" in lowered or "max_turns" in lowered:
            # A turn ceiling is a limit the caller chose, so it is an answer.
            subtype = "error_max_turns"
            end_reason = "turn_cap"
        else:
            subtype = "error"
            end_reason = "sdk_error"
        text = text or f"[session ended with SDK error: {message}]"

    unique_changes = list(dict.fromkeys(file_changes))
    return AgentRunResult(
        text=text,
        subtype=subtype,
        num_turns=1,
        end_reason=end_reason,
        session_id=session_id,
        tool_calls=tool_calls,
        file_changes=unique_changes,
        usage=_normalize_sdk_usage(getattr(result, "usage", None)),
        findings=findings,
        edit_count=edit_count,
        stderr_tail=stderr_tail,
    )


def _load_codex_sdk() -> Any:
    """Load the optional Codex SDK only when this provider is selected."""
    try:
        import openai_codex
    except ImportError as exc:
        raise CodexUnavailableError("Codex Python SDK is not installed; install kernelforge[codex]") from exc
    return openai_codex


class CodexBackend:
    """Execute Forge sessions through the Codex Python SDK."""

    name = "codex"
    capabilities = AgentCapabilities(
        writable=True,
        resumable=True,
        native_subagents=True,
        mcp=True,
        sandbox=True,
        probe=True,
        requires_workspace_cwd=True,
        session_env=True,
        workspace_guard=True,
    )

    def __init__(
        self,
        codex_bin: str = "",
        gateway: LlmGateway | Mapping[str, object] | None = None,
        bypass_sandbox: bool | None = None,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        """Capture transport overrides while deferring checks until execution."""
        self.runtime = runtime or AgentRuntimeConfig(
            provider=self.name,
            model=DEFAULT_CODEX_MODEL,
            fallback_model=FALLBACK_CODEX_MODEL,
            executable=codex_bin,
            sandbox_mode=("bypass" if bypass_sandbox is not False else "workspace-write"),
        )
        configured_gateway = self.runtime.options.get("gateway")
        if gateway is None and isinstance(configured_gateway, Mapping):
            gateway = configured_gateway
        # An empty override is "no override": converting it would yield a
        # gateway object that reads as configured and shadow the environment.
        if isinstance(gateway, Mapping):
            gateway = LlmGateway.from_mapping(gateway) if gateway else None
        self._configured_codex_bin = (
            codex_bin or self.runtime.executable or os.environ.get("FORGE_AGENT_CLI", "").strip()
        )
        self.codex_bin = str(Path(self._configured_codex_bin).expanduser()) if self._configured_codex_bin else ""
        self.gateway = gateway
        self.bypass_sandbox = self.runtime.sandbox_mode == "bypass" if bypass_sandbox is None else bypass_sandbox
        self._preflight_done = False
        self._codex_home_owner: Any = None
        self._codex_home = ""

    def _child_environment(
        self,
        base: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build an isolated Codex environment outside the system temp tree."""
        env = dict(base) if base is not None else os.environ.copy()
        configured = str(self.runtime.options.get("home", "")).strip()
        if configured:
            codex_home = Path(configured).expanduser().resolve()
        else:
            if not self._codex_home:
                root = Path.home() / ".cache" / "kernelforge" / "codex_home"
                root.mkdir(parents=True, exist_ok=True)
                self._codex_home_owner = tempfile.TemporaryDirectory(
                    prefix="run-",
                    dir=root,
                )
                self._codex_home = self._codex_home_owner.name
            codex_home = Path(self._codex_home)
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        # These outrank -C and cwd, so a caller that set them to reach one
        # repository would answer every git command the session runs anywhere.
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        return env

    def _effective_gateway(self) -> LlmGateway:
        """Return the explicit override, else what the environment configures.

        Spelled out rather than ``self.gateway or ...`` because ``LlmGateway``
        has no truthiness: an override that resolved to nothing would otherwise
        shadow the environment instead of deferring to it.
        """
        if self.gateway is not None and self.gateway.is_complete():
            return self.gateway
        return _resolve_gateway()

    def preflight(self) -> None:
        """Validate SDK runtime and gateway prerequisites without spending tokens."""
        if self._preflight_done:
            return
        sdk = _load_codex_sdk()
        if self.codex_bin and (not Path(self.codex_bin).is_file() or not os.access(self.codex_bin, os.X_OK)):
            raise CodexUnavailableError(f"Codex SDK runtime is not executable: {self.codex_bin}")
        _provider_overrides(self._effective_gateway())
        if self.codex_bin:
            try:
                version = subprocess.run(
                    [self.codex_bin, "--version"],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CodexUnavailableError(f"Codex SDK runtime version check failed: {exc}") from exc
            version_text = b"\n".join([version.stdout, version.stderr]).decode(errors="replace").strip()
            if version.returncode != 0 or not re.search(
                r"\bcodex(?:-cli)?\b",
                version_text,
                re.IGNORECASE,
            ):
                raise CodexUnavailableError(
                    f"configured SDK runtime does not appear to be Codex: "
                    f"{self.codex_bin}; --version returned {version_text!r}"
                )
        client = None
        try:
            client = sdk.Codex(
                self._sdk_config(
                    sdk=sdk,
                    spec=None,
                    child_env=self._child_environment(),
                )
            )
        except Exception as exc:
            raise CodexUnavailableError(f"Codex SDK app-server initialization failed: {exc}") from exc
        finally:
            if client is not None:
                client.close()
        self._preflight_done = True

    def _agent_role_overrides(self, spec: AgentRunSpec) -> list[str]:
        """Materialize Codex custom-agent roles as SDK config overrides."""
        raw_roles = spec.subagents
        if not raw_roles:
            return []
        codex_home = Path(self._child_environment()["CODEX_HOME"])
        roles_dir = codex_home / "forge-agent-roles"
        roles_dir.mkdir(parents=True, exist_ok=True)
        overrides = ["features.multi_agent=true"]
        for role_name, role in raw_roles.items():
            if not isinstance(role_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", role_name):
                raise CodexExecutionError(f"invalid Codex agent role name: {role_name!r}")
            description = role.description or f"Forge {role_name} specialist"
            instructions = role.instructions
            role_model = resolve_codex_model(role.model or spec.model)
            effort = role.reasoning_effort or spec.reasoning_effort
            effort = resolve_codex_reasoning_effort(effort)
            # config.toml spells full access with the warning in the name.
            # Derived from the role alone, never from the parent's ``bypass``.
            # Bypass withdraws OS confinement because the operator placed the
            # process in an external sandbox, but a role's read-only-ness is
            # carried by nothing else here -- the role config has no tool
            # allowlist -- so widening it would hand a reviewer full write access
            # with only its prompt to stop it. A host with no bubblewrap
            # therefore cannot run native roles at all, which is a limit on the
            # paths that use them, not a reason to remove the only enforcement
            # they have.
            sandbox_mode = "workspace-write" if role.writable else "read-only"
            role_path = (roles_dir / f"{role_name}.toml").resolve()
            role_path.write_text(
                "\n".join(
                    [
                        f"name = {_toml_string(role_name)}",
                        f"description = {_toml_string(description)}",
                        f"model = {_toml_string(role_model)}",
                        f"model_reasoning_effort = {_toml_string(effort)}",
                        f"sandbox_mode = {_toml_string(sandbox_mode)}",
                        f"developer_instructions = {_toml_string(instructions)}",
                        "",
                    ]
                )
            )
            overrides.extend(
                [
                    f"agents.{role_name}.description={_toml_string(description)}",
                    f"agents.{role_name}.config_file={_toml_string(str(role_path))}",
                ]
            )
        return overrides

    def _mcp_server_overrides(self, spec: AgentRunSpec) -> list[str]:
        """Convert backend-neutral stdio MCP definitions into Codex overrides."""
        raw_servers = spec.mcp_servers
        if not raw_servers:
            return []
        overrides: list[str] = []
        for server_name, server in raw_servers.items():
            if not isinstance(server_name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", server_name):
                raise CodexExecutionError(f"invalid Codex MCP server name: {server_name!r}")
            command = server.command.strip()
            command_args = list(server.args)
            if not command:
                raise CodexExecutionError(f"Codex MCP server {server_name!r} requires a command")
            if not isinstance(command_args, list) or not all(isinstance(value, str) for value in command_args):
                raise CodexExecutionError(f"Codex MCP server {server_name!r} args must be strings")

            prefix = f"mcp_servers.{server_name}"
            overrides.extend(
                [
                    f"{prefix}.command={_toml_string(command)}",
                    f"{prefix}.args={json.dumps(command_args)}",
                    f"{prefix}.enabled=true",
                ]
            )
            if server.env:
                if not all(isinstance(key, str) and isinstance(value, str) for key, value in server.env.items()):
                    raise CodexExecutionError(f"Codex MCP server {server_name!r} env must contain strings")
                encoded_env = ",".join(
                    f"{_toml_key(key)}={_toml_string(value)}" for key, value in sorted(server.env.items())
                )
                overrides.append(f"{prefix}.env={{{encoded_env}}}")
            startup_timeout = server.startup_timeout_sec
            if startup_timeout is not None:
                overrides.append(
                    f"{prefix}.startup_timeout_sec={int(startup_timeout)}",
                )
            tool_timeout = server.tool_timeout_sec
            if tool_timeout is not None:
                overrides.append(
                    f"{prefix}.tool_timeout_sec={int(tool_timeout)}",
                )
        return overrides

    def _sdk_sandbox(self, sdk: Any, spec: AgentRunSpec) -> Any:
        """Map the generic runtime sandbox into a Codex SDK preset."""
        if self.runtime.sandbox_mode == "bypass" or self.bypass_sandbox:
            return sdk.Sandbox.full_access
        elif spec.writable and self.runtime.sandbox_mode != "read-only":
            return sdk.Sandbox.workspace_write
        return sdk.Sandbox.read_only

    def _config_overrides(self, spec: AgentRunSpec | None) -> tuple[str, ...]:
        """Collect process-wide SDK app-server configuration overrides."""
        overrides = [
            "features.memories=false",
            *_provider_overrides(self._effective_gateway()),
        ]
        if spec is not None:
            overrides.extend(self._agent_role_overrides(spec))
            overrides.extend(self._mcp_server_overrides(spec))
        return tuple(overrides)

    def _sdk_config(
        self,
        *,
        sdk: Any,
        spec: AgentRunSpec | None,
        child_env: dict[str, str],
    ) -> Any:
        """Build one isolated Codex SDK app-server configuration.

        The app server is the parent of every command the session runs, so the
        spec's environment is applied over ``child_env`` here: that is the one
        place this session's shell commands can be given values that differ from
        the Forge process's own.
        """
        return sdk.CodexConfig(
            codex_bin=self.codex_bin or None,
            config_overrides=self._config_overrides(spec),
            cwd=spec.cwd if spec is not None else None,
            env={**child_env, **(spec.env if spec is not None else {})},
            client_name="kernel_forge",
            client_title="KernelForge",
        )

    def _thread_start_options(
        self,
        sdk: Any,
        spec: AgentRunSpec,
    ) -> dict[str, Any]:
        """Build SDK options for one new thread."""
        return {
            "approval_mode": sdk.ApprovalMode.deny_all,
            "cwd": spec.cwd,
            "developer_instructions": _codex_instructions(spec),
            "model": resolve_codex_model(spec.model),
            "model_provider": "forge",
            "sandbox": self._sdk_sandbox(sdk, spec),
        }

    def _turn_options(
        self,
        sdk: Any,
        spec: AgentRunSpec,
    ) -> dict[str, Any]:
        """Build SDK options for one non-interactive Codex turn."""
        return {
            "approval_mode": sdk.ApprovalMode.deny_all,
            "cwd": spec.cwd,
            "effort": resolve_codex_reasoning_effort(spec.reasoning_effort),
            "model": resolve_codex_model(spec.model),
            "sandbox": self._sdk_sandbox(sdk, spec),
        }

    def probe(
        self,
        *,
        cwd: str,
        model: str = "",
        reasoning_effort: str = "",
        timeout_sec: int | None = None,
        usage: Any = None,
    ) -> AgentRunResult:
        """Probe gateway, model, and SDK compatibility before a long run."""
        self.preflight()
        sdk = _load_codex_sdk()
        model = model.strip() or self.runtime.model
        reasoning_effort = reasoning_effort.strip() or self.runtime.reasoning_effort
        timeout_sec = timeout_sec or min(60, self.runtime.timeout_sec)
        spec = AgentRunSpec(
            system_prompt="",
            user_prompt="",
            cwd=cwd,
            model=model,
            writable=False,
            timeout_sec=timeout_sec,
            reasoning_effort=reasoning_effort,
        )
        client = None
        turn = None
        outcome: dict[str, Any] = {}
        completed = threading.Event()

        def run_turn() -> None:
            """Run the blocking SDK turn in a bounded daemon thread."""
            try:
                outcome["result"] = turn.run()
            except Exception as exc:
                outcome["error"] = exc
            finally:
                completed.set()

        try:
            with tempfile.TemporaryDirectory(prefix="forge-codex-git-") as tmpdir:
                child_env = self._child_environment(_write_git_guard(Path(tmpdir)))
                client = sdk.Codex(
                    self._sdk_config(
                        sdk=sdk,
                        spec=spec,
                        child_env=child_env,
                    )
                )
                thread_options = self._thread_start_options(sdk, spec)
                turn_options = self._turn_options(sdk, spec)
                thread = client.thread_start(**thread_options)
                turn = thread.turn(
                    "Reply with exactly OK. Do not inspect files or run commands.",
                    **turn_options,
                )
                worker = threading.Thread(
                    target=run_turn,
                    name="forge-codex-probe",
                    daemon=True,
                )
                worker.start()
                if not completed.wait(timeout_sec):
                    with contextlib.suppress(Exception):
                        turn.interrupt()
                    raise CodexUnavailableError(f"Codex gateway precheck timed out after {timeout_sec}s")
                if "error" in outcome:
                    raise CodexUnavailableError(f"Codex gateway precheck failed: {outcome['error']}") from outcome[
                        "error"
                    ]
                result = _normalize_sdk_result(
                    outcome["result"],
                    thread.id,
                )
        except CodexUnavailableError:
            raise
        except Exception as exc:
            raise CodexUnavailableError(f"Codex gateway precheck failed: {exc}") from exc
        finally:
            if client is not None:
                client.close()

        if result.usage and usage is not None:
            usage.add_usage(
                result.usage,
                total_cost_usd=result.usage.get("total_cost_usd"),
            )
        if not result.text:
            raise CodexUnavailableError("Codex gateway precheck returned an empty SDK response")
        return result

    async def _execute(
        self,
        *,
        spec: AgentRunSpec,
        prompt: str,
        session_id: str = "",
        usage: Any = None,
    ) -> AgentRunResult:
        """Execute one guarded SDK turn and normalize its typed result."""
        self.preflight()
        sdk = _load_codex_sdk()
        guard = WorkspaceGuard(spec)
        guard.prepare()
        turn_handle = None
        turn_task: asyncio.Task[Any] | None = None
        # Set as soon as the thread exists, so a failure past that point can hand
        # the handle to the caller instead of stranding the turns it already paid
        # for. Falls back to the id we were asked to resume.
        thread_id = session_id
        try:
            with tempfile.TemporaryDirectory(prefix="forge-codex-git-") as tmpdir:
                child_env = self._child_environment(_write_git_guard(Path(tmpdir)))
                config = self._sdk_config(
                    sdk=sdk,
                    spec=spec,
                    child_env=child_env,
                )
                async with sdk.AsyncCodex(config) as client:
                    if session_id:
                        thread = await client.thread_resume(session_id)
                    else:
                        thread = await client.thread_start(**self._thread_start_options(sdk, spec))
                    thread_id = thread.id
                    turn_handle = await thread.turn(
                        prompt,
                        **self._turn_options(sdk, spec),
                    )
                    turn_task = asyncio.create_task(turn_handle.run())
                    try:
                        completed, _ = await asyncio.wait(
                            {turn_task},
                            timeout=spec.timeout_sec,
                        )
                    except asyncio.CancelledError:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(
                                turn_handle.interrupt(),
                                timeout=5,
                            )
                        raise
                    if not completed:
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(
                                turn_handle.interrupt(),
                                timeout=5,
                            )
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(
                                asyncio.shield(turn_task),
                                timeout=5,
                            )
                        raise CodexExecutionError(
                            f"Codex timed out after {spec.timeout_sec}s",
                            session_id=thread_id,
                        )
                    sdk_result = turn_task.result()
        except asyncio.CancelledError:
            guard.rollback()
            raise
        except CodexExecutionError as exc:
            guard.rollback()
            if not exc.session_id and thread_id:
                exc.session_id = thread_id
            raise
        except Exception as exc:
            guard.rollback()
            raise CodexExecutionError(
                f"Codex SDK execution failed: {exc}",
                session_id=thread_id,
            ) from exc
        finally:
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    _ = await turn_task

        result = _normalize_sdk_result(sdk_result, thread_id)

        if usage is not None:
            usage.add_usage(
                result.usage,
                total_cost_usd=result.usage.get("total_cost_usd"),
            )
        try:
            actual_changes = guard.verify()
        except Exception:
            # verify() restores the baseline itself before raising a rejection, so
            # this second call only covers the paths that fail before it gets
            # there. Its own failure must not replace the exception it was
            # recovering from: under allow_dirty_baseline a restore that cannot
            # finish raises, and the caller would then be told about a restore
            # instead of about the paths the session was rejected for.
            with contextlib.suppress(Exception):
                guard.rollback()
            raise
        result.file_changes = actual_changes
        result.target_edit_count = guard.count_target_edits()
        result.edit_count = max(result.edit_count, result.target_edit_count)
        return result

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        """Run one new Codex SDK thread."""
        spec = spec.resolved(self.runtime)
        if spec.progress_log is not None and not spec.progress_log:
            spec.progress_log.append("progress: not supported by codex backend")
        return await self._execute(
            spec=spec,
            prompt=spec.user_prompt,
            usage=usage,
        )

    async def resume(
        self,
        spec: AgentRunSpec,
        session_id: str,
        feedback: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Resume an exact Codex session with canonical Forge gate feedback."""
        if not session_id.strip():
            raise CodexExecutionError("Codex resume requires a session ID")
        spec = spec.resolved(self.runtime)
        result = await self._execute(
            spec=spec,
            prompt=feedback,
            session_id=session_id,
            usage=usage,
        )
        if not result.session_id:
            result.session_id = session_id
        return result


__all__ = [
    "CodexBackend",
    "CodexBackendError",
    "CodexExecutionError",
    "CodexUnavailableError",
    "DEFAULT_CODEX_MODEL",
    "FALLBACK_CODEX_MODEL",
    "resolve_codex_cli",
    "resolve_codex_gateway",
    "resolve_codex_model",
    "resolve_codex_reasoning_effort",
]
