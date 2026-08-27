"""Tests for the Codex implementer backend transport and workspace guards."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from kernelforge.agent_backends import create_registered_backend
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderUnavailableError,
    AgentRole,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
    AgentToolPolicy,
    StdioMcpServer,
)
from kernelforge.agent_backends.registry import resolve_agent_runtime
from kernelforge.agent_backends.session_resume import (
    is_api_failure,
    resumable_session_id,
)
from kernelforge.agent_backends.workspace_guard import (
    WorkspaceGuard,
    WorkspaceSafetyError,
)
from kernelforge.agent_backends.codex import (
    CodexBackend,
    CodexExecutionError,
    CodexUnavailableError,
    _normalize_sdk_result,
    resolve_codex_model,
    resolve_codex_reasoning_effort,
)
from kernelforge.config import Config
from kernelforge.cli import _agent_runtime_overrides
from kernelforge.orchestrator import agent as agent_module
from kernelforge.tracker.usage import UsageAccumulator


def _git(cwd: Path, *args: str) -> str:
    """Run one git command for a temporary test repository."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a clean repository with one target and one ignored driver."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("forge_driver.py\n")
    kernel = repo / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    driver = repo / "forge_driver.py"
    driver.write_text("DRIVER = 'original'\n")
    _git(repo, "config", "user.name", "KernelForge Test")
    _git(repo, "config", "user.email", "kernelforge-test@example.invalid")
    _git(repo, "add", ".gitignore", "kernel.py")
    _git(repo, "commit", "-q", "-m", "test baseline")
    return repo, kernel, driver


def _write_fake_codex(path: Path, body: str) -> Path:
    """Write an executable fake Codex CLI with a custom exec body."""
    script = path / "fake-codex"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "if '--version' in sys.argv:\n"
        "    print('codex-cli 0.100.0')\n"
        "    raise SystemExit(0)\n"
        "workspace = Path(os.environ['FAKE_CODEX_WORKSPACE'])\n"
        "prompt = sys.stdin.read()\n"
        "if prompt.startswith('Reply with exactly OK'):\n"
        "    print(json.dumps({\n"
        "        'type': 'item.completed',\n"
        "        'item': {'type': 'agent_message', 'text': 'OK'},\n"
        "    }))\n"
        "    print(json.dumps({\n"
        "        'type': 'turn.completed',\n"
        "        'usage': {'input_tokens': 1, 'output_tokens': 1},\n"
        "    }))\n"
        "    raise SystemExit(0)\n"
        "is_resume = 'resume' in sys.argv\n"
        "if not is_resume and ('System instructions' not in prompt or 'Current request' not in prompt):\n"
        "    print('prompt was not supplied on stdin', file=sys.stderr)\n"
        "    raise SystemExit(9)\n" + textwrap.dedent(body)
    )
    script.chmod(0o755)
    return script


class _FakeCodexConfig:
    """Capture the public CodexConfig values used by the backend."""

    def __init__(self, **kwargs: object) -> None:
        """Store arbitrary SDK configuration fields for the transport fake."""
        self.__dict__.update(kwargs)


class _FakeApprovalMode:
    """Expose the approval mode consumed by the backend."""

    deny_all = "deny_all"


class _FakeSandbox:
    """Expose the sandbox presets consumed by the backend."""

    full_access = "full_access"
    read_only = "read_only"
    workspace_write = "workspace_write"


def _fake_thread_id(config: _FakeCodexConfig) -> str:
    """Read a deterministic thread ID embedded in one fake runtime."""
    codex_bin = getattr(config, "codex_bin", None)
    if not codex_bin:
        return "thread-fake"
    text = Path(codex_bin).read_text()
    match = re.search(
        r"""["']thread_id["']\s*:\s*["']([^"']+)""",
        text,
    )
    return match.group(1) if match else "thread-fake"


def _fake_prompt(prompt: str, options: dict[str, object]) -> str:
    """Combine SDK input and developer instructions for the CLI test fixture."""
    instructions = str(options.get("developer_instructions") or "")
    return f"{prompt}\n{instructions}\n## Current request\n{prompt}\n"


def _fake_turn_result(stdout: str) -> SimpleNamespace:
    """Convert fake runtime JSONL into a typed-SDK-shaped turn result."""
    final_response = ""
    items: list[dict[str, object]] = []
    usage: dict[str, object] = {}
    completed = False
    for raw in stdout.splitlines():
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict):
                items.append(item)
                if item.get("type") == "agent_message":
                    final_response = str(item.get("text") or "")
        elif event_type == "turn.completed":
            completed = True
            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                usage = event_usage
        elif event_type in {"turn.failed", "error"}:
            raise RuntimeError(str(event.get("message") or event_type))
    if not completed:
        raise RuntimeError("fake SDK turn did not complete")
    breakdown = {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0)),
    }
    return SimpleNamespace(
        final_response=final_response,
        items=items,
        usage=SimpleNamespace(last=breakdown),
        error=None,
    )


class _FakeSyncTurn:
    """Run one fake SDK turn synchronously for gateway probes."""

    def __init__(
        self,
        config: _FakeCodexConfig,
        prompt: str,
        options: dict[str, object],
        resumed: bool,
    ) -> None:
        """Capture runtime state used by one synchronous turn."""
        self.config = config
        self.prompt = prompt
        self.options = options
        self.resumed = resumed
        self.process: subprocess.Popen[str] | None = None

    def run(self) -> SimpleNamespace:
        """Execute the fake runtime and return an SDK-shaped result."""
        command = [self.config.codex_bin, "resume" if self.resumed else "exec"]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.config.cwd,
            env=self.config.env,
        )
        stdout, stderr = self.process.communicate(_fake_prompt(self.prompt, self.options))
        if self.process.returncode != 0:
            raise RuntimeError(stderr.strip() or f"exit {self.process.returncode}")
        return _fake_turn_result(stdout)

    def interrupt(self) -> SimpleNamespace:
        """Terminate an active synchronous fake turn."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        return SimpleNamespace()


class _FakeSyncThread:
    """Represent one synchronous Codex SDK thread."""

    def __init__(
        self,
        config: _FakeCodexConfig,
        thread_id: str,
        options: dict[str, object],
        resumed: bool,
    ) -> None:
        """Capture thread options for subsequent turns."""
        self.config = config
        self.id = thread_id
        self.options = options
        self.resumed = resumed

    def turn(self, prompt: str, **options: object) -> _FakeSyncTurn:
        """Create one synchronous fake turn."""
        merged = {**self.options, **options}
        return _FakeSyncTurn(self.config, prompt, merged, self.resumed)


class _FakeCodex:
    """Provide the synchronous public SDK surface."""

    def __init__(self, config: _FakeCodexConfig) -> None:
        """Capture one fake app-server configuration."""
        self.config = config

    def close(self) -> None:
        """Close the no-op fake app-server."""

    def thread_start(self, **options: object) -> _FakeSyncThread:
        """Start one synchronous fake SDK thread."""
        return _FakeSyncThread(
            self.config,
            _fake_thread_id(self.config),
            options,
            False,
        )

    def thread_resume(
        self,
        thread_id: str,
        **options: object,
    ) -> _FakeSyncThread:
        """Resume one synchronous fake SDK thread."""
        return _FakeSyncThread(self.config, thread_id, options, True)


class _FakeAsyncTurn:
    """Run one fake SDK turn without blocking the event loop."""

    def __init__(
        self,
        config: _FakeCodexConfig,
        prompt: str,
        options: dict[str, object],
        resumed: bool,
    ) -> None:
        """Capture runtime state used by one asynchronous turn."""
        self.config = config
        self.prompt = prompt
        self.options = options
        self.resumed = resumed
        self.process: asyncio.subprocess.Process | None = None

    async def run(self) -> SimpleNamespace:
        """Execute the fake runtime and return an SDK-shaped result."""
        command = [self.config.codex_bin, "resume" if self.resumed else "exec"]
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.cwd,
            env=self.config.env,
        )
        stdout, stderr = await self.process.communicate(_fake_prompt(self.prompt, self.options).encode())
        if self.process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(detail or f"exit {self.process.returncode}")
        return _fake_turn_result(stdout.decode(errors="replace"))

    async def interrupt(self) -> SimpleNamespace:
        """Terminate an active asynchronous fake turn."""
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            await self.process.wait()
        return SimpleNamespace()


class _FakeAsyncThread:
    """Represent one asynchronous Codex SDK thread."""

    def __init__(
        self,
        config: _FakeCodexConfig,
        thread_id: str,
        options: dict[str, object],
        resumed: bool,
    ) -> None:
        """Capture thread options for subsequent turns."""
        self.config = config
        self.id = thread_id
        self.options = options
        self.resumed = resumed

    async def turn(self, prompt: str, **options: object) -> _FakeAsyncTurn:
        """Create one asynchronous fake turn."""
        merged = {**self.options, **options}
        return _FakeAsyncTurn(self.config, prompt, merged, self.resumed)


class _FakeAsyncCodex:
    """Provide the asynchronous public SDK surface."""

    def __init__(self, config: _FakeCodexConfig) -> None:
        """Capture one fake app-server configuration."""
        self.config = config

    async def __aenter__(self) -> _FakeAsyncCodex:
        """Enter the no-op asynchronous app-server context."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Exit the no-op asynchronous app-server context."""

    async def thread_start(self, **options: object) -> _FakeAsyncThread:
        """Start one asynchronous fake SDK thread."""
        return _FakeAsyncThread(
            self.config,
            _fake_thread_id(self.config),
            options,
            False,
        )

    async def thread_resume(
        self,
        thread_id: str,
        **options: object,
    ) -> _FakeAsyncThread:
        """Resume one asynchronous fake SDK thread."""
        if options:
            raise AssertionError("SDK thread resume must preserve stored options")
        return _FakeAsyncThread(self.config, thread_id, options, True)


class _FakeCodexSdk:
    """Collect the fake classes exposed by the public Python SDK."""

    ApprovalMode = _FakeApprovalMode
    AsyncCodex = _FakeAsyncCodex
    Codex = _FakeCodex
    CodexConfig = _FakeCodexConfig
    Sandbox = _FakeSandbox


@pytest.fixture(autouse=True)
def _install_fake_codex_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route backend tests through the deterministic SDK transport fake."""

    def load_sdk() -> type[_FakeCodexSdk]:
        """Return the test SDK facade."""
        return _FakeCodexSdk

    monkeypatch.setattr(
        "kernelforge.agent_backends.codex._load_codex_sdk",
        load_sdk,
    )


def _spec(repo: Path, kernel: Path, driver: Path, timeout: int = 2) -> AgentRunSpec:
    """Build one writable Codex run specification for tests."""
    return AgentRunSpec(
        system_prompt="Optimize the kernel.",
        user_prompt="Make the change now.",
        cwd=str(repo),
        model="gpt-5.3-codex",
        timeout_sec=timeout,
        target_files=[str(kernel)],
        driver_script=str(driver),
    )


def _backend(fake: Path) -> CodexBackend:
    """Build a Codex backend with a deterministic fake gateway."""
    return CodexBackend(
        codex_bin=str(fake),
        gateway={
            "base_url": "https://gateway.example.invalid/v1",
            "key_env": "FAKE_CODEX_API_KEY",
            "headers": {"user": "test-user"},
        },
        bypass_sandbox=True,
    )


@pytest.mark.parametrize(
    ("sandbox_mode", "writable", "expected"),
    [
        ("bypass", False, _FakeSandbox.full_access),
        ("bypass", True, _FakeSandbox.full_access),
        ("workspace-write", False, _FakeSandbox.read_only),
        ("workspace-write", True, _FakeSandbox.workspace_write),
        ("read-only", False, _FakeSandbox.read_only),
        ("read-only", True, _FakeSandbox.read_only),
    ],
)
def test_codex_sdk_sandbox_keeps_runtime_and_write_policy_separate(
    sandbox_mode: str,
    writable: bool,
    expected: str,
) -> None:
    """Let explicit bypass own OS isolation without granting logical writes."""
    backend = CodexBackend(
        runtime=AgentRuntimeConfig(
            provider="codex",
            model="gpt-test",
            sandbox_mode=sandbox_mode,
        )
    )
    spec = AgentRunSpec(
        system_prompt="Inspect only.",
        user_prompt="Return findings.",
        cwd=".",
        writable=writable,
    )

    sandbox = backend._sdk_sandbox(_FakeCodexSdk, spec)

    assert sandbox == expected


def test_normalize_codex_sdk_result() -> None:
    """Normalize SDK items, session identity, and canonical token usage."""
    sdk_result = SimpleNamespace(
        final_response="PLAN: vectorize loads",
        items=[
            {
                "type": "fileChange",
                "changes": [{"path": "kernel.py", "kind": "update"}],
            },
            {
                "type": "collabAgentToolCall",
                "tool": "spawn_agent",
                "status": "completed",
                "senderThreadId": "thread-1",
                "receiverThreadIds": ["thread-child"],
            },
        ],
        usage=SimpleNamespace(
            last={
                "input_tokens": 120,
                "cached_input_tokens": 20,
                "output_tokens": 30,
            }
        ),
        error=None,
    )

    result = _normalize_sdk_result(sdk_result, "thread-1")

    assert result.session_id == "thread-1"
    assert result.text == "PLAN: vectorize loads"
    assert result.file_changes == ["kernel.py"]
    assert result.usage["input_tokens"] == 120
    assert result.usage["output_tokens"] == 30
    assert result.usage["cache_read_input_tokens"] == 20
    assert result.end_reason == "agent_stopped"
    assert result.tool_calls == [
        (
            "spawn_agent",
            {
                "status": "completed",
                "senderThreadId": "thread-1",
                "receiverThreadIds": ["thread-child"],
            },
        )
    ]


def test_codex_backend_materializes_custom_agent_roles(
    tmp_path: Path,
) -> None:
    """Inject stable multi-agent roles through absolute TOML config paths."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(tmp_path, "raise SystemExit(0)\n")
    backend = _backend(fake)
    spec = replace(
        _spec(repo, kernel, driver),
        subagents={
            "forge_reviewer": AgentRole(
                description="Forge correctness reviewer",
                instructions="Review only. Never edit files.",
                reasoning_effort="medium",
            ),
        },
        mcp_servers={
            "gpu": StdioMcpServer(
                command=sys.executable,
                args=(
                    "-m",
                    "kernelforge.mcp_server.pr_stdio_server",
                ),
                env={"PR_KB_REPO": "ROCm/aiter"},
                startup_timeout_sec=15,
            ),
        },
    )

    overrides = backend._config_overrides(spec)

    assert "features.multi_agent=true" in overrides
    config_value = next(value for value in overrides if value.startswith("agents.forge_reviewer.config_file="))
    role_path = Path(json.loads(config_value.split("=", 1)[1]))
    assert role_path.is_absolute()
    role_text = role_path.read_text()
    assert 'name = "forge_reviewer"' in role_text
    assert 'model = "gpt-5.3-codex"' in role_text
    assert 'sandbox_mode = "read-only"' in role_text
    assert "Review only. Never edit files." in role_text
    assert f'mcp_servers.gpu.command="{sys.executable}"' in overrides
    assert ('mcp_servers.gpu.args=["-m", "kernelforge.mcp_server.pr_stdio_server"]') in overrides
    assert 'mcp_servers.gpu.env={PR_KB_REPO="ROCm/aiter"}' in overrides
    assert "mcp_servers.gpu.startup_timeout_sec=15" in overrides


@pytest.mark.parametrize(
    ("sandbox_mode", "writable", "expected"),
    [
        ("bypass", False, "read-only"),
        ("bypass", True, "workspace-write"),
        ("workspace-write", False, "read-only"),
        ("workspace-write", True, "workspace-write"),
        ("read-only", False, "read-only"),
        # The role is not clamped to the parent in either direction. No caller
        # declares a writable role under a read-only parent today; the trio and
        # the orchestrator both give a writable parent read-only roles.
        ("read-only", True, "workspace-write"),
    ],
)
def test_codex_role_sandbox_comes_from_the_role_not_the_parent(
    tmp_path: Path,
    sandbox_mode: str,
    writable: bool,
    expected: str,
) -> None:
    """Confine a subagent by what the role may do, whatever the parent resolved.

    The sandbox is the only enforcement a native role has: the role config takes
    a description, a config file and nickname candidates, and the config file it
    points at carries no tool allowlist. Widening a read-only reviewer to match a
    parent running under ``bypass`` would leave its prompt as the only thing
    standing between it and the worktree. A host with no bubblewrap therefore
    cannot run native roles, which limits the paths that use them rather than
    what those roles are allowed to do.
    """
    backend = CodexBackend(
        runtime=AgentRuntimeConfig(
            provider="codex",
            model="gpt-5.3-codex",
            sandbox_mode=sandbox_mode,
            options={"home": str(tmp_path / "codex-home")},
        ),
    )
    spec = AgentRunSpec(
        system_prompt="Lead the work.",
        user_prompt="Delegate it.",
        cwd=str(tmp_path),
        model="gpt-5.3-codex",
        subagents={
            "forge_reviewer": AgentRole(
                description="Forge correctness reviewer",
                instructions="Review the change.",
                writable=writable,
            ),
        },
    )

    overrides = backend._agent_role_overrides(spec)

    config_value = next(value for value in overrides if value.startswith("agents.forge_reviewer.config_file="))
    role_text = Path(json.loads(config_value.split("=", 1)[1])).read_text()
    assert f'sandbox_mode = "{expected}"' in role_text


def test_normalize_codex_sdk_usage_uses_last_turn() -> None:
    """Count the resumed turn without recounting thread-total usage."""
    sdk_result = SimpleNamespace(
        final_response="PLAN: resumed response",
        items=[],
        usage=SimpleNamespace(
            last={
                "input_tokens": 8,
                "output_tokens": 3,
                "cached_input_tokens": 2,
            },
            total={
                "input_tokens": 80,
                "output_tokens": 30,
                "cached_input_tokens": 20,
            },
        ),
        error=None,
    )

    result = _normalize_sdk_result(sdk_result, "resumed-session")

    assert result.session_id == "resumed-session"
    assert result.text == "PLAN: resumed response"
    assert result.usage["input_tokens"] == 8
    assert result.usage["output_tokens"] == 3
    assert result.usage["cache_read_input_tokens"] == 2


def test_normalize_codex_sdk_result_reports_an_sdk_error_as_an_api_failure() -> None:
    """An in-band SDK error is not a finished agent.

    The SDK reports a provider-side failure on an otherwise "completed" turn, so
    labelling it ``agent_stopped`` made a rate limit indistinguishable from a
    deliberate no-op: resume never fired, and the empty diff was recorded as
    NO_CHANGES -- an optimization verdict about a kernel nobody looked at.
    """
    result = _normalize_sdk_result(
        SimpleNamespace(
            final_response=None,
            items=[],
            usage=None,
            error=SimpleNamespace(message="rate limited"),
        ),
        "thread-textless",
    )

    assert result.end_reason == "sdk_error"
    assert result.subtype == "error"
    assert is_api_failure(result) is True
    assert result.findings == ["rate limited"]
    assert result.stderr_tail == "rate limited"
    assert "rate limited" in result.text


def test_normalize_codex_sdk_result_keeps_a_turn_cap_terminal() -> None:
    """A turn ceiling is a limit the caller chose, so it is an answer, not weather."""
    result = _normalize_sdk_result(
        SimpleNamespace(
            final_response=None,
            items=[],
            usage=None,
            error=SimpleNamespace(message="reached the maximum number of turns"),
        ),
        "thread-capped",
    )

    assert result.end_reason == "turn_cap"
    assert is_api_failure(result) is False


def test_codex_execution_error_carries_the_thread_it_established() -> None:
    """A transport failure after ``thread_start`` must not strand the session.

    By then the thread holds every turn spent reading, building and benchmarking,
    so ``session_resume`` continues it instead of opening a new one.
    """
    exc = CodexExecutionError("Codex SDK execution failed: connection reset", session_id="thread-7")

    assert exc.session_id == "thread-7"
    assert resumable_session_id(exc) == "thread-7"
    # Unset by default, so a pre-thread failure still reads as "nothing to resume".
    assert CodexExecutionError("no thread yet").session_id == ""


def test_config_loads_generic_provider_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load provider-neutral model and sandbox settings from environment."""
    monkeypatch.setenv("FORGE_AGENT_BACKEND", "codex")
    monkeypatch.setenv("FORGE_AGENT_MODEL", "gpt-test-codex")
    monkeypatch.setenv("FORGE_AGENT_SANDBOX_MODE", "workspace-write")

    config = Config.from_env()
    runtime = config.agent_runtime()

    assert config.agent_backend == "codex"
    assert runtime.model == "gpt-test-codex"
    assert runtime.sandbox_mode == "workspace-write"


def test_shared_model_option_is_provider_neutral() -> None:
    """Map --model directly to the selected provider runtime."""
    overrides = _agent_runtime_overrides(
        model="provider-model",
        agent_backend="codex",
        agent_cli=None,
        agent_timeout_sec=None,
        agent_reasoning_effort=None,
        agent_sandbox_mode=None,
        agent_fallback_provider=None,
        agent_precheck=None,
        agent_options_json=None,
    )

    assert overrides == {
        "agent_model": "provider-model",
        "agent_backend": "codex",
    }
    assert resolve_codex_model("provider-model") == "provider-model"
    assert resolve_codex_model("") == "gpt-5.6"
    assert resolve_codex_reasoning_effort("") == "high"
    assert resolve_codex_reasoning_effort("max") == "xhigh"
    assert resolve_codex_reasoning_effort("xhigh") == "xhigh"
    with pytest.raises(CodexExecutionError, match="reasoning effort"):
        resolve_codex_reasoning_effort("ultra")


def test_backend_factory_falls_back_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downgrade missing Codex preflight to Claude only when configured."""

    def missing_codex_sdk() -> object:
        """Simulate a host without the optional Codex Python SDK."""
        raise CodexUnavailableError("Codex Python SDK is not installed")

    def fake_claude_sdk() -> tuple[object, object]:
        """Avoid importing a real Claude transport in the factory test."""
        return object(), object()

    monkeypatch.setattr(
        "kernelforge.agent_backends.codex._load_codex_sdk",
        missing_codex_sdk,
    )
    monkeypatch.setattr(
        "kernelforge.agent_backends.claude._load_claude_sdk",
        fake_claude_sdk,
    )

    with pytest.raises(CodexUnavailableError):
        create_registered_backend(
            resolve_agent_runtime(
                "codex",
                fallback_provider="",
            )
        )

    backend = create_registered_backend(
        resolve_agent_runtime(
            "codex",
            fallback_provider="claude",
        )
    )

    assert backend.name == "claude"
    assert "Codex Python SDK is not installed" in backend.fallback_reason


def test_make_agent_fn_dispatches_codex_without_claude_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch the implementer through Codex with its provider-specific model."""
    captured: dict[str, AgentRunSpec] = {}

    class FakeCodexBackend:
        """Capture one integration-level backend call."""

        name = "codex"
        capabilities = AgentCapabilities()

        async def run(
            self,
            spec: AgentRunSpec,
            usage: object = None,
        ) -> AgentRunResult:
            """Record the spec and return a deterministic agent answer."""
            captured["spec"] = spec.resolved(self.runtime)
            return AgentRunResult(text="PLAN: vectorize loads\nLESSON: aligned loads are faster")

    def fake_factory(runtime, **kwargs: object) -> FakeCodexBackend:
        """Return an integration fake for one registered runtime."""
        backend = FakeCodexBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(
        agent_module,
        "create_registered_backend",
        fake_factory,
    )
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    driver = tmp_path / "forge_driver.py"
    driver.write_text("print('allclose: True')\n")
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
    )
    session: dict[str, object] = {}

    agent_fn = agent_module.make_agent_fn(
        config=config,
        program_md="Optimize VALUE.",
        agent_backend="codex",
        insession_gate=True,
        driver_script=str(driver),
    )
    rationale = asyncio.run(agent_fn(str(kernel), "", session_sink=session))

    assert captured["spec"].model == "gpt-codex-test"
    assert captured["spec"].provider_options == {}
    assert captured["spec"].reasoning_effort == "max"
    assert config.max_turns == 500
    assert captured["spec"].tool_policy.max_turns == config.max_turns
    assert "ONE self-correcting session" in captured["spec"].system_prompt
    assert "Do NOT create or leave new non-ignored files" in captured["spec"].system_prompt
    assert getattr(agent_fn, "backend_name") == "codex"
    assert getattr(agent_fn, "backend_model") == "gpt-codex-test"
    assert session["session_started"] is True
    assert session["progress_log"] == []
    assert captured["spec"].progress_log is session["progress_log"]
    assert session["plan"] == "vectorize loads"
    # The implementer no longer authors its own takeaway: a stray LESSON: line in
    # its output is ignored, and the record is written afterwards by a dedicated
    # summarizer session. This fake provider cannot resume, so there is none.
    assert "lesson" not in session
    assert session["summarize"] is None
    assert session["end_reason"] == "resume_unavailable"
    assert rationale.startswith("[gate edits=0 pass=False end=resume_unavailable")


@pytest.mark.parametrize(
    ("text", "provider_end_reason", "expected_end_reason"),
    [
        ("PLAN: stop at cap", "turn_cap", "turn_cap"),
        (
            "PLAN: submit unsafe candidate\nSUBMIT_CANDIDATE",
            "agent_stopped",
            "candidate_submitted",
        ),
    ],
)
def test_final_integrity_verdict_survives_session_end_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    provider_end_reason: str,
    expected_end_reason: str,
) -> None:
    """Scan after turn caps and keep integrity independent from SUBMIT text."""

    captured: dict[str, AgentRunSpec] = {}
    driver = tmp_path / "forge_driver.py"
    driver.write_text("DRIVER = 'original'\n")
    source_oracle = tmp_path / "source_oracle.py"
    source_oracle.write_text("ORACLE = 'original'\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")

    class FakeBackend:
        name = "codex"
        capabilities = AgentCapabilities(stop_hooks=True)

        async def run(
            self,
            spec: AgentRunSpec,
            usage: object = None,
        ) -> AgentRunResult:
            captured["spec"] = spec
            source_oracle.write_text("ORACLE = 'gamed'\n")
            return AgentRunResult(
                text=text,
                end_reason=provider_end_reason,
            )

    def fake_factory(runtime, **_kwargs: object) -> FakeBackend:
        backend = FakeBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(agent_module, "create_registered_backend", fake_factory)
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
    )
    session: dict[str, object] = {}
    agent_fn = agent_module.make_agent_fn(
        config=config,
        program_md="Optimize VALUE.",
        agent_backend="codex",
        insession_gate=True,
        driver_script=str(driver),
        extra_protected_paths=[str(source_oracle)],
        extra_protected_globs=["golden*.json"],
    )

    asyncio.run(agent_fn(str(kernel), "", session_sink=session))

    assert captured["spec"].protected_paths == [str(source_oracle)]
    assert "golden*.json" in captured["spec"].protected_globs
    assert session["end_reason"] == expected_end_reason
    assert session["integrity_verdict"] == "violation"
    assert session["integrity_violation"] is True
    assert "source_oracle.py" in str(session["integrity_reason"])
    session["integrity_restore"]()
    assert source_oracle.read_text() == "ORACLE = 'original'\n"


def test_final_integrity_scan_runs_when_backend_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = tmp_path / "forge_driver.py"
    driver.write_text("DRIVER = 'original'\n")
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")

    class FailingBackend:
        name = "codex"
        capabilities = AgentCapabilities(stop_hooks=True)

        async def run(
            self,
            spec: AgentRunSpec,
            usage: object = None,
        ) -> AgentRunResult:
            driver.write_text("DRIVER = 'gamed'\n")
            raise RuntimeError("SDK stream failed")

    def fake_factory(runtime, **_kwargs: object) -> FailingBackend:
        backend = FailingBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(agent_module, "create_registered_backend", fake_factory)
    session: dict[str, object] = {}
    agent_fn = agent_module.make_agent_fn(
        config=Config(
            workspace=str(tmp_path),
            agent_backend="codex",
            agent_model="gpt-codex-test",
            agent_precheck=False,
        ),
        program_md="Optimize VALUE.",
        agent_backend="codex",
        insession_gate=True,
        driver_script=str(driver),
    )

    with pytest.raises(RuntimeError, match="SDK stream failed"):
        asyncio.run(agent_fn(str(kernel), "", session_sink=session))

    assert session["integrity_verdict"] == "violation"
    assert session["integrity_violation"] is True
    session["integrity_restore"]()
    assert driver.read_text() == "DRIVER = 'original'\n"


def test_implementer_turn_inherits_the_worktree_the_loop_dirtied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge an implementer turn against what it inherited, not against HEAD.

    forge-loop writes its own ledger -- campaign_config.json, events.jsonl,
    lessons, supervisor notes -- into the very workspace it then hands the
    implementer, and the kernel's runtime leaves a JIT cache there too. A turn
    judged against HEAD is refused for that inherited state before the agent is
    asked anything, so every iteration is skipped and the whole kernel budget
    goes to refusals. What the turn itself did is still judged, by comparing
    against the snapshot taken when it started.
    """
    captured: dict[str, AgentRunSpec] = {}

    class FakeCodexBackend:
        """Capture one integration-level backend call."""

        name = "codex"
        capabilities = AgentCapabilities()

        async def run(self, spec: AgentRunSpec, usage: object = None) -> AgentRunResult:
            """Record the spec and return a deterministic agent answer."""
            captured["spec"] = spec.resolved(self.runtime)
            return AgentRunResult(text="PLAN: vectorize loads")

    def fake_factory(runtime, **kwargs: object) -> FakeCodexBackend:
        """Return an integration fake for one registered runtime."""
        backend = FakeCodexBackend()
        backend.runtime = runtime
        return backend

    monkeypatch.setattr(agent_module, "create_registered_backend", fake_factory)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    driver = tmp_path / "forge_driver.py"
    driver.write_text("print('allclose: True')\n")
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
    )

    agent_fn = agent_module.make_agent_fn(
        config=config,
        program_md="Optimize VALUE.",
        agent_backend="codex",
        driver_script=str(driver),
    )
    asyncio.run(agent_fn(str(kernel), ""))

    assert captured["spec"].allow_dirty_baseline is True


def test_outer_gate_counts_only_incremental_resume_target_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use per-turn target counts instead of cumulative dirty file lists."""
    resume_calls = 0
    stop_calls = 0

    class FakeCodexBackend:
        """Return one edit followed by an unchanged resumed candidate."""

        name = "codex"
        capabilities = AgentCapabilities(resumable=True)

        async def run(
            self,
            spec: AgentRunSpec,
            usage: object = None,
        ) -> AgentRunResult:
            """Report the initial target edit."""
            return AgentRunResult(
                text="PLAN: initial candidate",
                session_id="thread-incremental-edits",
                file_changes=["kernel.py"],
                target_edit_count=1,
            )

        async def resume(
            self,
            spec: AgentRunSpec,
            session_id: str,
            feedback: str,
            usage: object = None,
        ) -> AgentRunResult:
            """Return the same cumulative diff without a new target edit."""
            nonlocal resume_calls
            resume_calls += 1
            return AgentRunResult(
                text="PLAN: unchanged candidate",
                session_id=session_id,
                file_changes=["kernel.py"],
                target_edit_count=0,
            )

    def fake_factory(runtime, **kwargs: object) -> FakeCodexBackend:
        """Return the deterministic resumable backend."""
        backend = FakeCodexBackend()
        backend.runtime = runtime
        return backend

    async def fake_on_stop(self, *_args, **_kwargs):
        """Block once, then allow the unchanged resumed candidate."""
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            return {"decision": "block", "reason": "recheck candidate"}
        self.end_reason = "converged"
        return {}

    monkeypatch.setattr(agent_module, "create_registered_backend", fake_factory)
    monkeypatch.setattr(
        "kernelforge.loop.insession_gate.InSessionGate._on_stop",
        fake_on_stop,
    )
    kernel = tmp_path / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    driver = tmp_path / "forge_driver.py"
    driver.write_text("print('allclose: True')\n")
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_model="gpt-codex-test",
        agent_precheck=False,
    )
    session: dict[str, object] = {}
    agent_fn = agent_module.make_agent_fn(
        config=config,
        program_md="Optimize VALUE.",
        agent_backend="codex",
        insession_gate=True,
        driver_script=str(driver),
    )

    asyncio.run(agent_fn(str(kernel), "", session_sink=session))

    assert resume_calls == 1
    assert session["edit_count"] == 1


def test_codex_gateway_probe_validates_sdk_and_counts_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe the configured model through the Python SDK."""
    fake = _write_fake_codex(tmp_path, "raise SystemExit(7)\n")
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    usage = UsageAccumulator()

    result = _backend(fake).probe(
        cwd=str(tmp_path),
        model="gpt-5.3-codex",
        reasoning_effort="high",
        usage=usage,
    )

    assert result.text == "OK"
    assert usage.totals()["calls"] == 1
    assert usage.totals()["input_tokens"] == 1


def test_make_agent_fn_falls_back_after_gateway_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the whole implementer run to Claude after a failed Codex gateway probe."""

    class FakeClaudeBackend:
        """Represent the fallback backend without importing the real SDK."""

        name = "claude"
        fallback_reason = ""
        capabilities = AgentCapabilities(stop_hooks=True)

        async def run(
            self,
            spec: AgentRunSpec,
            usage: object = None,
        ) -> AgentRunResult:
            """Return a placeholder result if the callback is invoked."""
            return AgentRunResult(text="PLAN: fallback")

    def fake_factory(runtime, **kwargs: object) -> object:
        """Return the fallback selected by the registered backend factory."""
        backend = FakeClaudeBackend()
        backend.runtime = resolve_agent_runtime(
            "claude",
            model="claude-fallback-model",
            fallback_provider="",
        )
        backend.fallback_reason = "gateway returned 401"
        return backend

    monkeypatch.setattr(
        agent_module,
        "create_registered_backend",
        fake_factory,
    )
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_precheck=True,
        agent_fallback_provider="claude",
    )

    agent_fn = agent_module.make_agent_fn(
        config=config,
        program_md="Optimize the kernel.",
        agent_backend="codex",
    )

    assert getattr(agent_fn, "backend_name") == "claude"
    assert getattr(agent_fn, "backend_model") == "claude-fallback-model"


def test_make_agent_fn_does_not_system_exit_when_claude_fallback_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface a Codex error instead of terminating a pure-Codex process."""

    def fake_factory(runtime, **kwargs: object) -> object:
        """Raise one provider-level error for unavailable generic fallback."""
        raise AgentProviderUnavailableError("codex unavailable; fallback claude unavailable")

    monkeypatch.setattr(
        agent_module,
        "create_registered_backend",
        fake_factory,
    )
    config = Config(
        workspace=str(tmp_path),
        agent_backend="codex",
        agent_precheck=True,
        agent_fallback_provider="claude",
    )

    with pytest.raises(
        AgentProviderUnavailableError,
        match="fallback claude unavailable",
    ):
        agent_module.make_agent_fn(
            config=config,
            program_md="Optimize the kernel.",
            agent_backend="codex",
        )


def test_codex_backend_keeps_allowed_edit_and_counts_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a target edit while accumulating one Codex usage record."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 2\\n")
        print(json.dumps({"type": "thread.started", "thread_id": "thread-ok"}))
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: raise value\\nLESSON: ok"},
        }))
        print(json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 4, "cached_input_tokens": 3},
        }))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    usage = UsageAccumulator()

    result = asyncio.run(_backend(fake).run(_spec(repo, kernel, driver), usage))

    assert kernel.read_text() == "VALUE = 2\n"
    assert driver.read_text() == "DRIVER = 'original'\n"
    assert result.file_changes == ["kernel.py"]
    assert result.target_edit_count == 1
    assert result.edit_count == 1
    assert usage.totals()["input_tokens"] == 10
    assert usage.totals()["output_tokens"] == 4
    assert usage.totals()["cache_read_input_tokens"] == 3
    assert usage.totals()["calls"] == 1


def test_codex_backend_counts_repeated_edits_to_same_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count completed file-change events instead of unique changed paths."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 2\\n")
        print(json.dumps({"type": "thread.started", "thread_id": "thread-edits"}))
        for _ in range(2):
            print(json.dumps({
                "type": "item.completed",
                "item": {
                    "type": "file_change",
                    "changes": [{"path": "kernel.py", "kind": "update"}],
                },
            }))
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: edit twice"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    result = asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert result.file_changes == ["kernel.py"]
    assert result.edit_count == 2


def test_codex_backend_resumes_with_dirty_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume one exact session while preserving its unstaged target candidate."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        if is_resume:
            assert (workspace / "kernel.py").read_text() == "VALUE = 2\\n"
            (workspace / "kernel.py").write_text("VALUE = 3\\n")
            message = "PLAN: resume candidate\\nLESSON: gate feedback helped"
        else:
            (workspace / "kernel.py").write_text("VALUE = 2\\n")
            message = "PLAN: initial candidate\\nLESSON: first attempt"
        print(json.dumps({"type": "thread.started", "thread_id": "thread-resume"}))
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        }))
        print(json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    backend = _backend(fake)
    spec = _spec(repo, kernel, driver)

    initial = asyncio.run(backend.run(spec))
    resumed = asyncio.run(
        backend.resume(
            replace(spec, allow_dirty_targets=True),
            initial.session_id,
            "Canonical gate rejected the first candidate.",
        )
    )

    assert initial.session_id == "thread-resume"
    assert resumed.session_id == "thread-resume"
    assert resumed.text.startswith("PLAN: resume candidate")
    assert resumed.file_changes == ["kernel.py"]
    assert kernel.read_text() == "VALUE = 3\n"


def test_codex_read_only_resume_preserves_arbitrary_dirty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarize over staged/non-target/untracked state without changing it."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper = repo / "helper.py"
    helper.write_text("HELPER = 1\n")
    _git(repo, "add", "helper.py")
    _git(repo, "commit", "-q", "-m", "add helper")
    fake = _write_fake_codex(
        tmp_path,
        """
        message = "lesson summary" if is_resume else "PLAN: initial session"
        print(json.dumps({"type": "thread.started", "thread_id": "thread-summary"}))
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    backend = _backend(fake)
    spec = _spec(repo, kernel, driver)
    initial = asyncio.run(backend.run(spec))

    kernel.write_text("VALUE = 'staged candidate'\n")
    _git(repo, "add", "kernel.py")
    helper.write_text("HELPER = 'non-target change'\n")
    note = repo / "session-note.txt"
    note.write_text("untracked context\n")
    note_link = repo / "session-note-link"
    note_link.symlink_to(note.name)
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--binary")
    unstaged_before = _git(repo, "diff", "--binary")

    summary_spec = replace(
        spec,
        writable=False,
        allow_dirty_targets=True,
        allow_untracked=True,
        read_only_resume=True,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=4,
        ),
    )
    resumed = asyncio.run(
        backend.resume(
            summary_spec,
            initial.session_id,
            "Record the iteration lesson.",
        )
    )

    assert resumed.text == "lesson summary"
    assert resumed.file_changes == []
    assert resumed.target_edit_count == 0
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--binary") == staged_before
    assert _git(repo, "diff", "--binary") == unstaged_before
    assert note.read_text() == "untracked context\n"
    assert note_link.is_symlink()
    assert note_link.readlink() == Path(note.name)


def test_codex_read_only_discovery_accepts_unchanged_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a new read-only discovery turn over staged, unstaged, and untracked state."""
    repo, kernel, driver = _make_repo(tmp_path)
    kernel.write_text("VALUE = 'staged candidate'\n")
    _git(repo, "add", "kernel.py")
    kernel.write_text("VALUE = 'runtime patch'\n")
    note = repo / "runtime-note.txt"
    note.write_text("untracked runtime state\n")
    fake = _write_fake_codex(
        tmp_path,
        """
        assert (workspace / "kernel.py").read_text() == "VALUE = 'runtime patch'\\n"
        assert (workspace / "runtime-note.txt").read_text() == "untracked runtime state\\n"
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "[]"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--binary")
    unstaged_before = _git(repo, "diff", "--binary")
    spec = replace(
        _spec(repo, kernel, driver),
        writable=False,
        read_only_resume=True,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=1,
        ),
    )

    result = asyncio.run(_backend(fake).run(spec))

    assert result.text == "[]"
    assert result.file_changes == []
    assert result.target_edit_count == 0
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--binary") == staged_before
    assert _git(repo, "diff", "--binary") == unstaged_before
    assert note.read_text() == "untracked runtime state\n"


def test_codex_read_only_discovery_rejects_and_restores_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore an arbitrary dirty baseline before reporting a read-only violation."""
    repo, kernel, driver = _make_repo(tmp_path)
    kernel.write_text("VALUE = 'staged candidate'\n")
    _git(repo, "add", "kernel.py")
    kernel.write_text("VALUE = 'runtime patch'\n")
    note = repo / "runtime-note.txt"
    note.write_text("untracked runtime state\n")
    real_git = shutil.which("git")
    assert real_git is not None
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 'agent mutation'\\n")
        subprocess.run(
            [os.environ["REAL_GIT"], "add", "kernel.py"],
            cwd=workspace,
            check=True,
        )
        (workspace / "runtime-note.txt").write_text("mutated note\\n")
        (workspace / "new-source.py").write_text("MUTATED = True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "[]"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    monkeypatch.setenv("REAL_GIT", real_git)
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--binary")
    unstaged_before = _git(repo, "diff", "--binary")
    spec = replace(
        _spec(repo, kernel, driver),
        writable=False,
        read_only_resume=True,
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=1,
        ),
    )

    with pytest.raises(
        WorkspaceSafetyError,
        match="read-only.*changed the workspace.*restored",
    ):
        asyncio.run(_backend(fake).run(spec))

    assert kernel.read_text() == "VALUE = 'runtime patch'\n"
    assert note.read_text() == "untracked runtime state\n"
    assert not (repo / "new-source.py").exists()
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--binary") == staged_before
    assert _git(repo, "diff", "--binary") == unstaged_before


def test_codex_backend_reports_no_target_edit_for_unchanged_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not recount a dirty target when a resumed turn leaves it unchanged."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        if not is_resume:
            (workspace / "kernel.py").write_text("VALUE = 2\\n")
        message = "PLAN: inspect candidate\\nLESSON: no extra edit needed"
        print(json.dumps({"type": "thread.started", "thread_id": "thread-no-edit"}))
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": message},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    backend = _backend(fake)
    spec = _spec(repo, kernel, driver)

    initial = asyncio.run(backend.run(spec))
    resumed = asyncio.run(
        backend.resume(
            replace(spec, allow_dirty_targets=True),
            initial.session_id,
            "Recheck the unchanged candidate.",
        )
    )

    assert initial.target_edit_count == 1
    assert resumed.file_changes == ["kernel.py"]
    assert resumed.target_edit_count == 0


def test_codex_backend_restores_driver_and_target_on_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject metric-surface edits and roll back every tracked candidate edit."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 99\\n")
        (workspace / "forge_driver.py").write_text("DRIVER = 'gamed'\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: game metric"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="protected ignored files changed"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert kernel.read_text() == "VALUE = 1\n"
    assert driver.read_text() == "DRIVER = 'original'\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_codex_guard_honors_additional_exact_protected_paths(
    tmp_path: Path,
) -> None:
    repo, kernel, driver = _make_repo(tmp_path)
    oracle = repo / "source_oracle.py"
    oracle.write_text("ORACLE = 'original'\n")
    _git(repo, "add", "source_oracle.py")
    _git(repo, "commit", "-q", "-m", "add source oracle")
    guard = WorkspaceGuard(
        replace(
            _spec(repo, kernel, driver),
            protected_paths=[str(oracle)],
        )
    )
    guard.prepare()

    oracle.write_text("ORACLE = 'gamed'\n")

    with pytest.raises(
        WorkspaceSafetyError,
        match="protected tracked files changed",
    ):
        guard.verify()
    assert oracle.read_text() == "ORACLE = 'original'\n"


def _read_only_spec(cwd: Path) -> AgentRunSpec:
    """A session with every route to the filesystem closed."""
    return AgentRunSpec(
        system_prompt="Analyze.",
        user_prompt="Report.",
        cwd=str(cwd),
        model="gpt-5.3-codex",
        timeout_sec=2,
        writable=False,
        tool_policy=AgentToolPolicy(read=False, search=False, write=False, shell=False, max_turns=1),
        protected_globs=["*"],
    )


def test_workspace_guard_skips_a_read_only_session_outside_git(tmp_path: Path) -> None:
    """A session that cannot write has no rollback to protect.

    Demanding a git worktree of it refuses to run for a caller who simply has
    none -- discovery analyzing an installed framework, say.
    """
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()
    guard = WorkspaceGuard(_read_only_spec(plain_dir))

    guard.prepare()

    assert guard.skipped is True
    assert guard.verify() == []
    guard.rollback()  # must not reach for git state it never recorded


def test_workspace_guard_skips_a_read_only_session_in_a_dirty_worktree(
    tmp_path: Path,
) -> None:
    """The clean-worktree rule would otherwise block any caller mid-loop."""
    repo, kernel, _driver = _make_repo(tmp_path)
    kernel.write_text("VALUE = 'uncommitted work'\n")
    guard = WorkspaceGuard(_read_only_spec(repo))

    guard.prepare()

    assert guard.skipped is True
    assert kernel.read_text() == "VALUE = 'uncommitted work'\n"


def test_workspace_guard_still_runs_for_a_read_only_resume(tmp_path: Path) -> None:
    """Its whole point is the verify() check that dirty state came back intact.

    Skipping on read-only alone would drop that silently, and the only thing
    keeping it alive today is that this path happens to declare target files.
    """
    repo, _kernel, _driver = _make_repo(tmp_path)
    spec = replace(_read_only_spec(repo), read_only_resume=True)

    assert WorkspaceGuard.is_read_only_session(spec) is False


def test_codex_backend_preserves_preexisting_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed without discarding a user-owned tracked modification."""
    repo, kernel, driver = _make_repo(tmp_path)
    kernel.write_text("VALUE = 'user change'\n")
    fake = _write_fake_codex(tmp_path, "raise SystemExit(8)\n")
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="requires a clean"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert kernel.read_text() == "VALUE = 'user change'\n"
    assert _git(repo, "status", "--porcelain") == "M kernel.py"


def test_codex_backend_removes_new_untracked_source_on_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject and delete an untracked source file that the loop cannot commit."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 5\\n")
        (workspace / "helper.py").write_text("HELPER = True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: add helper"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="new non-ignored files"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert kernel.read_text() == "VALUE = 1\n"
    assert not (repo / "helper.py").exists()
    assert _git(repo, "status", "--porcelain") == ""


def test_codex_backend_allows_orchestrator_source_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep new source files only when the orchestrator explicitly allows them."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper = repo / "helper.py"
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "helper.py").write_text("HELPER = True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: add source helper"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    spec = replace(
        _spec(repo, kernel, driver),
        target_files=[str(kernel), str(helper)],
        allow_untracked=True,
    )

    result = asyncio.run(_backend(fake).run(spec))

    assert helper.read_text() == "HELPER = True\n"
    assert result.file_changes == ["helper.py"]


def _dirty_author_baseline(repo: Path) -> tuple[Path, Path, Path]:
    """Leave the unrelated tracked/staged/untracked state a long run accumulates."""
    helper = repo / "helper.py"
    helper.write_text("HELPER = 1\n")
    _git(repo, "add", "helper.py")
    _git(repo, "commit", "-q", "-m", "add helper")
    helper.write_text("HELPER = 'operator edit'\n")
    staged = repo / "server_args.py"
    staged.write_text("ARGS = 'operator staged'\n")
    _git(repo, "add", "server_args.py")
    note = repo / "runtime-note.txt"
    note.write_text("untracked runtime state\n")
    return helper, staged, note


def _author_spec(repo: Path, kernel: Path, driver: Path) -> AgentRunSpec:
    """Build the writable author spec the fusion author phase submits."""
    return replace(
        _spec(repo, kernel, driver),
        writable=True,
        allow_dirty_targets=True,
        allow_untracked=True,
        allow_dirty_baseline=True,
    )


def test_codex_dirty_baseline_rejects_an_undone_inherited_stage(
    tmp_path: Path,
) -> None:
    """Judge an index change on its own, not by where the path ends up.

    A turn that unstages a file the caller had staged leaves it untracked on
    disk. The deviation is detected -- the index record for that path no longer
    matches the baseline -- but reporting it in whichever bucket the path now
    occupies hands it to the rule for that bucket, and ``allow_untracked``
    forgives untracked paths. The caller's staged work is undone and the turn is
    accepted. An index that no longer matches the one the turn inherited is a
    violation wherever the file itself went.
    """
    from kernelforge.agent_backends import codex as codex_module

    repo, kernel, driver = _make_repo(tmp_path)
    inherited = repo / "caller_staged.py"
    inherited.write_text("CALLER = True\n")
    _git(repo, "add", "caller_staged.py")
    guard = codex_module.WorkspaceGuard(_author_spec(repo, kernel, driver))
    guard.prepare()

    _git(repo, "reset", "--quiet", "--", "caller_staged.py")

    with pytest.raises(WorkspaceSafetyError, match="index"):
        guard.verify()


def test_codex_writable_author_restore_failure_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """A recovery that could not run must not report a clean rollback.

    ``rollback()`` turns a failed ``allow_dirty_baseline`` recovery into a raised
    rejection, so the restore it calls cannot suppress the error. A Git-ignored
    target is recorded nowhere but its own snapshot, and a suppressed write would
    leave the rejected turn's edit on disk.
    """
    from kernelforge.agent_backends import codex as codex_module

    repo, kernel, driver = _make_repo(tmp_path)
    guard = codex_module.WorkspaceGuard(_author_spec(repo, kernel, driver))
    guard.prepare()

    # Fail only the target-snapshot step. Patching every write would raise from an
    # earlier recovery step, which never suppressed anything, and the test would
    # pass whether or not this step propagates.
    targets = set(guard.target_snapshots)
    assert targets, "the author spec must allowlist at least one target"
    real_write_bytes = Path.write_bytes

    def selective_write(self: Path, data: bytes, *args, **kwargs):
        if self in targets:
            raise OSError("read-only filesystem")
        return real_write_bytes(self, data, *args, **kwargs)

    kernel.write_text("VALUE = 'authored'\n")
    with mock.patch.object(Path, "write_bytes", selective_write):
        with pytest.raises(WorkspaceSafetyError, match="could not be restored") as raised:
            guard.rollback()
    # A restore that could not run says nothing about what the session did, and the
    # author classifies it by this marker: marked, it abandoned the recipe -- and
    # rollback also runs while unwinding a plain timeout.
    assert raised.value.agent_safety_rejection is False


def test_codex_safety_error_marks_a_verdict_and_not_a_failed_query(
    tmp_path: Path,
) -> None:
    """The two things this one class carries must be distinguishable.

    The fusion author refuses to retry a workspace-safety VERDICT, and used to
    recognise one by class name -- so a ``git`` call that timed out on NFS
    abandoned the recipe exactly like a session that edited a protected file.
    """
    from kernelforge.agent_backends import workspace_guard as guard_module

    repo, _kernel, _driver = _make_repo(tmp_path)

    assert WorkspaceSafetyError("the session changed HEAD").agent_safety_rejection is True
    with pytest.raises(WorkspaceSafetyError) as raised:
        guard_module._git_output(repo, "rev-parse", "--verify", "refs/heads/absent")
    assert raised.value.agent_safety_rejection is False


def test_codex_verify_rejection_survives_a_failing_second_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The author needs the violating paths, not a complaint about a restore.

    ``verify()`` already restored the baseline before raising, so the caller's
    rollback is a second one; under ``allow_dirty_baseline`` its own failure raises
    and replaced the violation list the author logs and hands to the next attempt.
    """
    from kernelforge.agent_backends import codex as codex_module

    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "forge_driver.py").write_text("DRIVER = 'agent rewrote it'\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: retune the driver"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    real_rollback = codex_module.WorkspaceGuard.rollback
    calls: list[int] = []

    def failing_second_rollback(self):
        # verify() rolls back itself before raising, so the caller's is the second.
        calls.append(1)
        if len(calls) == 1:
            return real_rollback(self)
        raise WorkspaceSafetyError(
            "Codex run ended and the inherited workspace state could not be restored: [Errno 30] Read-only file system",
            rejection=False,
        )

    monkeypatch.setattr(
        codex_module.WorkspaceGuard,
        "rollback",
        failing_second_rollback,
    )

    with pytest.raises(WorkspaceSafetyError, match="forge_driver.py"):
        asyncio.run(_backend(fake).run(_author_spec(repo, kernel, driver)))


def test_codex_writable_author_accepts_inherited_dirty_non_target_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Author into a worktree the caller already left dirty outside the targets."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper, staged, note = _dirty_author_baseline(repo)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 'authored'\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: fuse the decode chain"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    result = asyncio.run(_backend(fake).run(_author_spec(repo, kernel, driver)))

    assert kernel.read_text() == "VALUE = 'authored'\n"
    # Only the paths this turn actually changed are reported; the inherited dirty
    # files are not the author's edits and must not be attributed to it.
    assert result.file_changes == ["kernel.py"]
    assert result.target_edit_count == 1
    assert helper.read_text() == "HELPER = 'operator edit'\n"
    assert staged.read_text() == "ARGS = 'operator staged'\n"
    assert note.read_text() == "untracked runtime state\n"


def test_codex_writable_author_rejection_restores_inherited_dirty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undo the turn's own changes on rejection without discarding the baseline."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper, staged, note = _dirty_author_baseline(repo)
    status_before = _git(repo, "status", "--porcelain")
    staged_before = _git(repo, "diff", "--cached", "--binary")
    unstaged_before = _git(repo, "diff", "--binary")
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 'authored'\\n")
        oracle = workspace / "tests" / "test_fake.py"
        oracle.parent.mkdir()
        oracle.write_text("assert True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: fake the oracle"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="protected files created"):
        asyncio.run(_backend(fake).run(_author_spec(repo, kernel, driver)))

    assert kernel.read_text() == "VALUE = 1\n"
    assert not (repo / "tests").exists()
    assert helper.read_text() == "HELPER = 'operator edit'\n"
    assert staged.read_text() == "ARGS = 'operator staged'\n"
    assert note.read_text() == "untracked runtime state\n"
    assert driver.read_text() == "DRIVER = 'original'\n"
    assert _git(repo, "status", "--porcelain") == status_before
    assert _git(repo, "diff", "--cached", "--binary") == staged_before
    assert _git(repo, "diff", "--binary") == unstaged_before


def test_codex_writable_author_rejects_reverting_an_inherited_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notice a turn that quietly restores an inherited edit to its committed form."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper, _staged, _note = _dirty_author_baseline(repo)
    protected = repo / "scripts" / "cal_kernel_perf.py"
    protected.parent.mkdir()
    protected.write_text("MEASURE = 'operator edit'\n")
    _git(repo, "add", "scripts/cal_kernel_perf.py")
    _git(repo, "commit", "-q", "-m", "add measurement script")
    protected.write_text("MEASURE = 'operator patch'\n")
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "scripts" / "cal_kernel_perf.py").write_text(
            "MEASURE = 'operator edit'\\n"
        )
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: clean the measurement"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="protected tracked files changed"):
        asyncio.run(_backend(fake).run(_author_spec(repo, kernel, driver)))

    assert protected.read_text() == "MEASURE = 'operator patch'\n"
    assert helper.read_text() == "HELPER = 'operator edit'\n"


def test_codex_dirty_target_resume_still_rejects_inherited_non_target_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the strict resume contract for callers that did not opt into a baseline."""
    repo, kernel, driver = _make_repo(tmp_path)
    helper, _staged, _note = _dirty_author_baseline(repo)
    fake = _write_fake_codex(tmp_path, "raise SystemExit(8)\n")
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    spec = replace(
        _spec(repo, kernel, driver),
        allow_dirty_targets=True,
        allow_untracked=True,
    )

    with pytest.raises(WorkspaceSafetyError, match="only unstaged target changes"):
        asyncio.run(_backend(fake).run(spec))

    assert helper.read_text() == "HELPER = 'operator edit'\n"


def test_codex_backend_allows_preexisting_untracked_preparation_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept explicit untracked scaffolding before a writable prepare turn."""
    repo, kernel, driver = _make_repo(tmp_path)
    prep_driver = repo / "driver.py"
    prep_driver.write_text("BROKEN = True\n")
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "driver.py").write_text("READY = True\\n")
        (workspace / "helper.py").write_text("HELPER = True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: prepare driver"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    spec = replace(
        _spec(repo, kernel, driver),
        target_files=[str(prep_driver)],
        allow_dirty_targets=True,
        allow_untracked=True,
    )

    result = asyncio.run(_backend(fake).run(spec))

    assert prep_driver.read_text() == "READY = True\n"
    assert (repo / "helper.py").read_text() == "HELPER = True\n"
    assert result.file_changes == ["driver.py", "helper.py"]
    assert result.target_edit_count == 1


def test_codex_backend_accepts_untracked_state_the_caller_declared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start an implementer turn beside untracked state the orchestrator allowed.

    forge-loop writes its own experiment ledger into the workspace it hands the
    implementer, so every iteration starts beside untracked files nobody asked the
    agent about. The orchestrator says that is expected with ``allow_untracked``,
    which the resume branch honours -- an implementer branch that ignores it rejects
    the loop's own bookkeeping and skips every candidate without spending a turn.
    """
    repo, kernel, driver = _make_repo(tmp_path)
    ledger = repo / "forge_experiments"
    ledger.mkdir()
    (ledger / "events.jsonl").write_text('{"iteration": 1}\n')
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 2\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: raise the value"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    spec = replace(_spec(repo, kernel, driver), allow_untracked=True)

    result = asyncio.run(_backend(fake).run(spec))

    assert kernel.read_text() == "VALUE = 2\n"
    assert (ledger / "events.jsonl").exists()
    assert result.target_edit_count == 1


def test_codex_backend_names_the_state_that_blocked_the_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report which paths made the worktree dirty, not merely that it was.

    The bare refusal costs an operator a manual worktree inspection to learn
    what to clean, which is the whole content of the answer.
    """
    repo, kernel, driver = _make_repo(tmp_path)
    ledger = repo / "forge_experiments"
    ledger.mkdir()
    (ledger / "events.jsonl").write_text('{"iteration": 1}\n')
    fake = _write_fake_codex(tmp_path, "raise SystemExit(8)\n")
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError, match="forge_experiments"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))


def test_codex_backend_summarizes_a_long_list_of_blocking_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the refusal readable when a workspace inherits hundreds of files."""
    repo, kernel, driver = _make_repo(tmp_path)
    ledger = repo / "forge_experiments" / "lessons"
    ledger.mkdir(parents=True)
    for index in range(40):
        (ledger / f"iter_{index:03d}.md").write_text("note\n")
    fake = _write_fake_codex(tmp_path, "raise SystemExit(8)\n")
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(WorkspaceSafetyError) as failure:
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    message = str(failure.value)
    assert "and 30 more" in message
    assert message.count("untracked: ") == 10


def test_codex_backend_rejects_new_protected_file_when_creation_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject new measurement files even when source creation is enabled."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        target = workspace / "tests" / "test_fake.py"
        target.parent.mkdir()
        target.write_text("assert True\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: fake validation"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    spec = replace(
        _spec(repo, kernel, driver),
        allow_untracked=True,
    )

    with pytest.raises(WorkspaceSafetyError, match="protected files created"):
        asyncio.run(_backend(fake).run(spec))

    assert not (repo / "tests" / "test_fake.py").exists()
    assert _git(repo, "status", "--porcelain") == ""


def test_codex_child_cannot_mutate_git_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deny mutating git commands while preserving a valid source edit."""
    repo, kernel, driver = _make_repo(tmp_path)
    original_head = _git(repo, "rev-parse", "HEAD")
    fake = _write_fake_codex(
        tmp_path,
        """
        attempt = subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "forbidden"],
            cwd=workspace,
            capture_output=True,
        )
        if attempt.returncode == 0:
            raise SystemExit(8)
        (workspace / "kernel.py").write_text("VALUE = 3\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: safe edit"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    result = asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert result.file_changes == ["kernel.py"]
    assert _git(repo, "rev-parse", "HEAD") == original_head
    assert kernel.read_text() == "VALUE = 3\n"


def test_codex_backend_recovers_absolute_git_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject and recover a commit that bypasses the PATH git wrapper."""
    repo, kernel, driver = _make_repo(tmp_path)
    original_head = _git(repo, "rev-parse", "HEAD")
    real_git = shutil.which("git")
    assert real_git is not None
    fake = _write_fake_codex(
        tmp_path,
        """
        subprocess.run(
            [os.environ["REAL_GIT"], "commit", "--allow-empty", "-m", "forbidden"],
            cwd=workspace,
            check=True,
        )
        (workspace / "kernel.py").write_text("VALUE = 44\\n")
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "PLAN: unsafe commit"},
        }))
        print(json.dumps({"type": "turn.completed", "usage": {}}))
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")
    monkeypatch.setenv("REAL_GIT", real_git)

    with pytest.raises(WorkspaceSafetyError, match="changed HEAD"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver)))

    assert _git(repo, "rev-parse", "HEAD") == original_head
    assert kernel.read_text() == "VALUE = 1\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_codex_backend_rolls_back_partial_edit_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kill a timed-out process group and restore its partial tracked edit."""
    repo, kernel, driver = _make_repo(tmp_path)
    fake = _write_fake_codex(
        tmp_path,
        """
        (workspace / "kernel.py").write_text("VALUE = 77\\n")
        time.sleep(5)
        """,
    )
    monkeypatch.setenv("FAKE_CODEX_WORKSPACE", str(repo))
    monkeypatch.setenv("FAKE_CODEX_API_KEY", "test-secret")

    with pytest.raises(CodexExecutionError, match="timed out"):
        asyncio.run(_backend(fake).run(_spec(repo, kernel, driver, timeout=1)))

    assert kernel.read_text() == "VALUE = 1\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_the_guard_does_not_hold_the_repositorys_own_bookkeeping(tmp_path: Path) -> None:
    """`.git` moves on its own; snapshotting it makes git housekeeping a rejection."""
    repo, _kernel, _driver = _make_repo(tmp_path)
    guard = WorkspaceGuard(replace(_read_only_spec(repo), read_only_resume=True))

    guard.prepare()

    assert guard.snapshots
    assert not [path for path in guard.snapshots if ".git" in Path(path).parts]


def test_git_housekeeping_during_a_session_is_not_a_violation(tmp_path: Path) -> None:
    """git rewrites its own bookkeeping unprompted -- refreshing a stale stat
    cache rewrites the index, and a build touching files is enough to cause it.
    Held as bytes, that housekeeping read as the session tampering."""
    repo, _kernel, _driver = _make_repo(tmp_path)
    guard = WorkspaceGuard(replace(_read_only_spec(repo), read_only_resume=True))
    guard.prepare()

    (repo / ".git" / "COMMIT_EDITMSG").write_text("rewritten by git\n")

    assert guard.verify() == []


def test_a_protected_file_is_still_caught_once_git_is_excluded(tmp_path: Path) -> None:
    """The exclusion must not cost the check the guard exists for."""
    repo, _kernel, driver = _make_repo(tmp_path)
    guard = WorkspaceGuard(replace(_read_only_spec(repo), read_only_resume=True))
    guard.prepare()

    driver.write_text("DRIVER = 'tampered'\n")

    with pytest.raises(WorkspaceSafetyError, match="read-only session changed the workspace"):
        guard.verify()
    assert driver.read_text() == "DRIVER = 'original'\n"
