# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the shared Codex Agent SDK session helper.

Hyperloom routes every OpenAI-side LLM interaction through the Codex Agent SDK
rather than bare API calls, so this module owns the gateway mapping, the sandbox
and approval policy, the turn timeout and the usage normalization for all of
them. The SDK is mocked throughout: no test may reach the network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import codex_session as cs

_GATEWAY_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "LLM_GATEWAY_KEY",
    "CODEX_HOME",
)

_SECRET = "sk-do-not-leak-this-value"


@pytest.fixture(autouse=True)
def _clear_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from a known-empty gateway environment."""
    for name in _GATEWAY_ENV:
        monkeypatch.delenv(name, raising=False)


def _gateway_env(**extra: str) -> dict[str, str]:
    """Build a minimal OpenAI-compatible gateway environment."""
    env = {
        "OPENAI_API_KEY": _SECRET,
        "OPENAI_BASE_URL": "https://gateway.example/api/v1/llm-proxy/Unified/v1",
    }
    env.update(extra)
    return env


def _override_value(overrides: tuple[str, ...], key: str) -> str:
    """Return the value of one ``key=value`` Codex override."""
    for override in overrides:
        name, sep, value = override.partition("=")
        if sep and name == key:
            return value
    raise AssertionError(f"override {key!r} not found in {overrides!r}")


# --------------------------------------------------------------------------- #
# Fake SDK
# --------------------------------------------------------------------------- #


class _FakeSandbox:
    """Sentinels standing in for ``openai_codex.Sandbox``."""

    read_only = "read-only"
    workspace_write = "workspace-write"
    full_access = "full-access"


class _FakeApprovalMode:
    """Sentinels standing in for ``openai_codex.ApprovalMode``."""

    deny_all = "deny_all"
    auto_review = "auto_review"


class _FakeSDKModule:
    """Fake exposing only the sandbox presets, for the selection helper."""

    Sandbox = _FakeSandbox


class _FakeItem:
    """Typed thread item stand-in exposing a pydantic-style dump."""

    def __init__(self, payload: dict[str, Any], *, wrap_root: bool = False) -> None:
        self._payload = payload
        self.root = self if not wrap_root else _FakeItem(payload)

    def model_dump(self, by_alias: bool = False, mode: str = "python") -> dict[str, Any]:
        return dict(self._payload)


class _FakeUsageBreakdown:
    """``TokenUsageBreakdown`` stand-in."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)


class _FakeTurnResult:
    """``TurnResult`` stand-in."""

    def __init__(
        self,
        *,
        final_response: str | None = "done",
        items: tuple[Any, ...] = (),
        usage: Any = None,
        error: Any = None,
    ) -> None:
        self.final_response = final_response
        self.items = list(items)
        self.usage = usage
        self.error = error


class _FakeTurnHandle:
    """``AsyncTurnHandle`` stand-in that can hang until interrupted."""

    def __init__(self, record: dict[str, Any], *, result: Any, hang: bool, honor_interrupt: bool) -> None:
        self._record = record
        self._result = result
        self._hang = hang
        self._honor_interrupt = honor_interrupt
        self._interrupted = asyncio.Event()
        record["interrupt_calls"] = 0

    async def run(self) -> Any:
        if self._hang:
            await self._interrupted.wait()
            raise RuntimeError("turn interrupted")
        return self._result

    async def interrupt(self) -> None:
        self._record["interrupt_calls"] += 1
        if self._honor_interrupt:
            self._interrupted.set()


class _FakeThread:
    """``AsyncThread`` stand-in."""

    id = "thread-fake"

    def __init__(self, record: dict[str, Any], *, result: Any, hang: bool, honor_interrupt: bool) -> None:
        self._record = record
        self._result = result
        self._hang = hang
        self._honor_interrupt = honor_interrupt

    async def turn(self, prompt: str, **options: Any) -> _FakeTurnHandle:
        self._record["prompt"] = prompt
        self._record["turn_options"] = options
        return _FakeTurnHandle(
            self._record,
            result=self._result,
            hang=self._hang,
            honor_interrupt=self._honor_interrupt,
        )


class _FakeAsyncCodex:
    """``AsyncCodex`` stand-in usable as an async context manager."""

    def __init__(
        self,
        config: Any,
        record: dict[str, Any],
        *,
        result: Any,
        hang: bool,
        honor_interrupt: bool,
        thread_start_error: Exception | None,
    ) -> None:
        record["config"] = config
        self._record = record
        self._result = result
        self._hang = hang
        self._honor_interrupt = honor_interrupt
        self._thread_start_error = thread_start_error

    async def __aenter__(self) -> "_FakeAsyncCodex":
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        self._record["closed"] = True
        return False

    async def thread_start(self, **options: Any) -> _FakeThread:
        self._record["thread_options"] = options
        if self._thread_start_error is not None:
            raise self._thread_start_error
        return _FakeThread(
            self._record,
            result=self._result,
            hang=self._hang,
            honor_interrupt=self._honor_interrupt,
        )


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any = None,
    hang: bool = False,
    honor_interrupt: bool = True,
    thread_start_error: Exception | None = None,
) -> dict[str, Any]:
    """Point :func:`load_codex_sdk` at a fake SDK and return its record dict."""
    record: dict[str, Any] = {}
    turn_result = result if result is not None else _FakeTurnResult()

    class _Config:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def _async_codex(config: Any) -> _FakeAsyncCodex:
        return _FakeAsyncCodex(
            config,
            record,
            result=turn_result,
            hang=hang,
            honor_interrupt=honor_interrupt,
            thread_start_error=thread_start_error,
        )

    fake_sdk = type(
        "_FakeCodexSDK",
        (),
        {
            "CodexConfig": _Config,
            "AsyncCodex": staticmethod(_async_codex),
            "ApprovalMode": _FakeApprovalMode,
            "Sandbox": _FakeSandbox,
        },
    )
    monkeypatch.setattr(cs, "load_codex_sdk", lambda: fake_sdk)
    return record


# --------------------------------------------------------------------------- #
# Gateway override construction
# --------------------------------------------------------------------------- #


def test_provider_overrides_pass_the_key_env_name_not_its_value():
    """The API key must reach Codex by variable NAME only.

    Config overrides become app-server launch arguments, so a copied secret
    would be visible to anything that can read the process table.
    """
    overrides = cs.codex_provider_overrides(env=_gateway_env())

    assert _override_value(overrides, "model_providers.hyperloom.env_key") == '"OPENAI_API_KEY"'
    assert not any(_SECRET in override for override in overrides)


def test_provider_overrides_select_the_responses_wire_api():
    """Codex talks to the gateway over the OpenAI Responses protocol."""
    overrides = cs.codex_provider_overrides(env=_gateway_env())

    assert _override_value(overrides, "model_provider") == '"hyperloom"'
    assert _override_value(overrides, "model_providers.hyperloom.wire_api") == '"responses"'
    assert (
        _override_value(overrides, "model_providers.hyperloom.base_url")
        == '"https://gateway.example/api/v1/llm-proxy/Unified/v1"'
    )
    assert _override_value(overrides, "model_providers.hyperloom.name") == '"hyperloom"'


def test_provider_overrides_name_the_gateway_key_variable_that_is_set():
    """``LLM_GATEWAY_KEY`` deployments must be pointed at their own variable."""
    env = {
        "LLM_GATEWAY_KEY": _SECRET,
        "OPENAI_BASE_URL": "https://gateway.example/Unified/v1",
    }

    overrides = cs.codex_provider_overrides(env=env)

    assert _override_value(overrides, "model_providers.hyperloom.env_key") == '"LLM_GATEWAY_KEY"'
    assert not any(_SECRET in override for override in overrides)


def test_provider_overrides_carry_operator_gateway_headers():
    """Gateway routing headers are operator-supplied and must be forwarded."""
    overrides = cs.codex_provider_overrides(env=_gateway_env(OPENAI_CUSTOM_HEADERS="user: ntid42"))

    assert _override_value(overrides, "model_providers.hyperloom.http_headers.user") == '"ntid42"'


def test_provider_overrides_reject_a_header_codex_cannot_express():
    """A header name outside the TOML bare-key charset must fail loudly.

    Codex's ``-c key=value`` parser reads a dotted bare-key path, so silently
    emitting a dotted header name would corrupt the whole provider table.
    """
    with pytest.raises(cs.CodexSessionUnavailableError, match="not a valid Codex config key"):
        cs.codex_provider_overrides(env=_gateway_env(OPENAI_CUSTOM_HEADERS="x.api.key: value"))


def test_provider_overrides_require_an_explicit_base_url():
    """Codex needs the gateway endpoint; guessing one would misroute traffic."""
    with pytest.raises(cs.CodexSessionUnavailableError, match="OPENAI_BASE_URL"):
        cs.codex_provider_overrides(env={"OPENAI_API_KEY": _SECRET})


def test_provider_overrides_require_a_credential():
    """A missing credential fails before the SDK spends a turn."""
    with pytest.raises(cs.CodexSessionUnavailableError, match="credential is missing"):
        cs.codex_provider_overrides(env={"OPENAI_BASE_URL": "https://gateway.example/Unified/v1"})


def test_api_key_env_name_follows_llm_config_precedence():
    """The preferred variable wins, then OPENAI_API_KEY, then LLM_GATEWAY_KEY."""
    both = {"OPENAI_API_KEY": "a", "LLM_GATEWAY_KEY": "b"}

    assert cs.api_key_env_name(env=both) == "OPENAI_API_KEY"
    assert cs.api_key_env_name(env={"LLM_GATEWAY_KEY": "b"}) == "LLM_GATEWAY_KEY"
    assert cs.api_key_env_name(api_key_env="SAFE_API_KEY", env={"SAFE_API_KEY": "c", **both}) == "SAFE_API_KEY"


def test_api_key_env_name_lists_every_candidate_when_none_is_set():
    """The error must tell the operator which variables were checked."""
    with pytest.raises(cs.CodexSessionUnavailableError) as excinfo:
        cs.api_key_env_name(env={})

    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "LLM_GATEWAY_KEY" in message


# --------------------------------------------------------------------------- #
# Sandbox and approval selection
# --------------------------------------------------------------------------- #


def test_codex_sandbox_is_read_only_without_writable_roots():
    """A session that declares no write scope must not get one."""
    assert cs.codex_sandbox(_FakeSDKModule, writable_roots=()) == _FakeSandbox.read_only


def test_codex_sandbox_is_workspace_write_for_a_declared_scope():
    """Declaring a writable root selects the scoped write preset, never full access."""
    sandbox = cs.codex_sandbox(_FakeSDKModule, writable_roots=(Path("/tmp/out"),))

    assert sandbox == _FakeSandbox.workspace_write
    assert sandbox != _FakeSandbox.full_access


def test_run_codex_turn_denies_approvals_and_scopes_writes(tmp_path, monkeypatch):
    """The unattended turn must deny approvals and write only where told."""
    workspace = tmp_path / "workspace"
    output = tmp_path / "out"
    workspace.mkdir()
    output.mkdir()
    record = _install_fake_sdk(monkeypatch)

    asyncio.run(
        cs.run_codex_turn(
            prompt="analyze",
            developer_instructions="instructions",
            cwd=workspace,
            model="gpt-5.6-sol",
            timeout_sec=30.0,
            writable_roots=(output,),
            env=_gateway_env(),
        )
    )

    thread_options = record["thread_options"]
    assert thread_options["approval_mode"] == _FakeApprovalMode.deny_all
    assert thread_options["sandbox"] == _FakeSandbox.workspace_write
    assert thread_options["cwd"] == str(workspace)
    assert thread_options["model"] == "gpt-5.6-sol"
    assert thread_options["model_provider"] == cs.CODEX_PROVIDER_NAME
    assert thread_options["developer_instructions"] == "instructions"

    # Turn-level options gate the tools for the turn that actually runs, so they
    # are set explicitly rather than inherited from the thread.
    turn_options = record["turn_options"]
    assert turn_options["approval_mode"] == _FakeApprovalMode.deny_all
    assert turn_options["sandbox"] == _FakeSandbox.workspace_write
    assert record["prompt"] == "analyze"

    overrides = record["config"].kwargs["config_overrides"]
    assert "features.memories=false" in overrides
    assert _override_value(overrides, "sandbox_workspace_write.writable_roots") == f'["{output.resolve()}"]'


def test_run_codex_turn_uses_a_read_only_sandbox_with_no_write_scope(tmp_path, monkeypatch):
    """No writable roots means no writable-roots override and a read-only sandbox."""
    record = _install_fake_sdk(monkeypatch)

    asyncio.run(
        cs.run_codex_turn(
            prompt="inspect",
            developer_instructions="instructions",
            cwd=tmp_path,
            model="gpt-5.6-sol",
            timeout_sec=30.0,
            env=_gateway_env(),
        )
    )

    assert record["thread_options"]["sandbox"] == _FakeSandbox.read_only
    overrides = record["config"].kwargs["config_overrides"]
    assert not any(override.startswith("sandbox_workspace_write.writable_roots") for override in overrides)


def test_run_codex_turn_isolates_codex_home_per_run(tmp_path, monkeypatch):
    """Each run gets a private CODEX_HOME that does not survive it."""
    homes: list[str] = []

    def _capture(record: dict[str, Any]) -> None:
        homes.append(record["config"].kwargs["env"]["CODEX_HOME"])

    for _ in range(2):
        record = _install_fake_sdk(monkeypatch)
        asyncio.run(
            cs.run_codex_turn(
                prompt="p",
                developer_instructions="i",
                cwd=tmp_path,
                model="m",
                timeout_sec=5.0,
                env=_gateway_env(),
            )
        )
        _capture(record)

    assert homes[0] != homes[1]
    assert not Path(homes[0]).exists()
    assert not Path(homes[1]).exists()


def test_run_codex_turn_identifies_hyperloom_to_the_app_server(tmp_path, monkeypatch):
    """The SDK client identity must name Hyperloom, not the SDK default."""
    record = _install_fake_sdk(monkeypatch)

    asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="i",
            cwd=tmp_path,
            model="m",
            timeout_sec=5.0,
            env=_gateway_env(),
        )
    )

    kwargs = record["config"].kwargs
    assert kwargs["client_name"] == "hyperloom"
    assert kwargs["client_title"] == "Hyperloom"
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["codex_bin"] is None


def test_run_codex_turn_honors_an_explicit_codex_binary(tmp_path, monkeypatch):
    """An operator-pinned runtime path is forwarded verbatim."""
    record = _install_fake_sdk(monkeypatch)

    asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="i",
            cwd=tmp_path,
            model="m",
            timeout_sec=5.0,
            codex_bin="/opt/bin/codex",
            env=_gateway_env(),
        )
    )

    assert record["config"].kwargs["codex_bin"] == "/opt/bin/codex"


# --------------------------------------------------------------------------- #
# Timeout and failure handling
# --------------------------------------------------------------------------- #


def test_run_codex_turn_interrupts_a_turn_that_outlives_its_timeout(tmp_path, monkeypatch):
    """An overrunning turn is interrupted and reported, never left running."""
    record = _install_fake_sdk(monkeypatch, hang=True)

    with pytest.raises(cs.CodexSessionTimeoutError, match=r"timed out after 0\.05s"):
        asyncio.run(
            cs.run_codex_turn(
                prompt="p",
                developer_instructions="i",
                cwd=tmp_path,
                model="m",
                timeout_sec=0.05,
                env=_gateway_env(),
            )
        )

    assert record["interrupt_calls"] == 1
    assert record["closed"] is True


def test_run_codex_turn_cancels_a_turn_that_ignores_the_interrupt(tmp_path, monkeypatch):
    """A turn that will not stop must still be cancelled, not leaked.

    Leaving the task pending would keep the SDK transport alive past the run and
    surface later as an unretrieved-exception warning from an unrelated phase.
    """
    monkeypatch.setattr(cs, "_INTERRUPT_TIMEOUT_SEC", 0.05)
    record = _install_fake_sdk(monkeypatch, hang=True, honor_interrupt=False)

    with pytest.raises(cs.CodexSessionTimeoutError):
        asyncio.run(
            cs.run_codex_turn(
                prompt="p",
                developer_instructions="i",
                cwd=tmp_path,
                model="m",
                timeout_sec=0.05,
                env=_gateway_env(),
            )
        )

    assert record["interrupt_calls"] == 1


def test_run_codex_turn_wraps_sdk_failures_with_context(tmp_path, monkeypatch):
    """An SDK failure surfaces as a Codex session error naming the cause."""
    _install_fake_sdk(monkeypatch, thread_start_error=RuntimeError("app-server refused the handshake"))

    with pytest.raises(cs.CodexSessionError, match="app-server refused the handshake"):
        asyncio.run(
            cs.run_codex_turn(
                prompt="p",
                developer_instructions="i",
                cwd=tmp_path,
                model="m",
                timeout_sec=5.0,
                env=_gateway_env(),
            )
        )


def test_run_codex_turn_fails_before_starting_when_the_gateway_is_unconfigured(tmp_path, monkeypatch):
    """A misconfigured gateway must fail without opening a thread."""
    record = _install_fake_sdk(monkeypatch)

    with pytest.raises(cs.CodexSessionUnavailableError):
        asyncio.run(
            cs.run_codex_turn(
                prompt="p",
                developer_instructions="i",
                cwd=tmp_path,
                model="m",
                timeout_sec=5.0,
                env={},
            )
        )

    assert "config" not in record


def test_load_codex_sdk_reports_a_missing_install(monkeypatch):
    """A missing SDK must name the package and how to install it."""
    monkeypatch.setitem(sys.modules, "openai_codex", None)

    with pytest.raises(cs.CodexSessionUnavailableError, match="openai_codex is not installed"):
        cs.load_codex_sdk()


def test_load_codex_sdk_returns_the_installed_module():
    """The happy path hands back the real SDK module."""
    sdk = pytest.importorskip("openai_codex")

    assert cs.load_codex_sdk() is sdk


# --------------------------------------------------------------------------- #
# Result and usage normalization
# --------------------------------------------------------------------------- #


def test_run_codex_turn_normalizes_typed_items_and_usage(tmp_path, monkeypatch):
    """Typed SDK items and the turn's token usage become plain data."""
    result = _FakeTurnResult(
        final_response="  wrote the report  ",
        items=(
            _FakeItem({"type": "commandExecution", "command": "ls", "exitCode": 0}),
            _FakeItem({"type": "agentMessage", "text": "hello"}, wrap_root=True),
        ),
        usage=type(
            "_Usage",
            (),
            {
                "last": _FakeUsageBreakdown(
                    {
                        "input_tokens": 120,
                        "output_tokens": 34,
                        "cached_input_tokens": 12,
                        "reasoning_output_tokens": 4096,
                    }
                )
            },
        ),
    )
    _install_fake_sdk(monkeypatch, result=result)

    session = asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="i",
            cwd=tmp_path,
            model="m",
            timeout_sec=5.0,
            env=_gateway_env(),
        )
    )

    assert session.text == "wrote the report"
    assert session.thread_id == "thread-fake"
    assert session.error == ""
    assert [item["type"] for item in session.items] == ["commandExecution", "agentMessage"]
    assert session.usage == {
        "input_tokens": 120,
        "output_tokens": 34,
        "cache_read_input_tokens": 12,
        "reasoning_output_tokens": 4096,
    }


def test_run_codex_turn_surfaces_an_in_band_turn_error(tmp_path, monkeypatch):
    """A turn that completes carrying an error is not a success.

    A provider-side failure can end the turn with no answer, so the message has
    to reach the caller instead of looking like a deliberate no-op.
    """
    error = type("_Error", (), {"message": "rate limit exceeded"})()
    _install_fake_sdk(monkeypatch, result=_FakeTurnResult(final_response=None, error=error))

    session = asyncio.run(
        cs.run_codex_turn(
            prompt="p",
            developer_instructions="i",
            cwd=tmp_path,
            model="m",
            timeout_sec=5.0,
            env=_gateway_env(),
        )
    )

    assert session.error == "rate limit exceeded"
    assert session.text == ""


def test_normalize_codex_usage_reads_only_the_last_turn():
    """``total`` accumulates across the thread; only ``last`` describes this turn."""
    usage = type(
        "_Usage",
        (),
        {
            "last": _FakeUsageBreakdown({"input_tokens": 7, "output_tokens": 9}),
            "total": _FakeUsageBreakdown({"input_tokens": 700, "output_tokens": 900}),
        },
    )

    assert cs.normalize_codex_usage(usage)["input_tokens"] == 7


def test_normalize_codex_usage_accepts_a_plain_mapping():
    """Mapping-shaped usage is normalized the same way as the typed model."""
    assert cs.normalize_codex_usage({"last": {"input_tokens": 3, "output_tokens": 4}}) == {
        "input_tokens": 3,
        "output_tokens": 4,
        "cache_read_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def test_normalize_codex_usage_tolerates_missing_and_malformed_counts():
    """Token accounting is diagnostic; a bad count must not sink a good run."""
    assert cs.normalize_codex_usage(None) == {}
    assert cs.normalize_codex_usage(object()) == {}
    assert cs.normalize_codex_usage({"last": {"input_tokens": "abc", "output_tokens": True}}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def test_normalize_codex_items_tolerates_mapping_items():
    """Plain mappings pass through so callers can build fixtures without pydantic."""
    result = _FakeTurnResult(items=({"type": "agentMessage", "text": "hi"},))

    assert cs.normalize_codex_items(result) == ({"type": "agentMessage", "text": "hi"},)


def test_codex_item_type_folds_the_sdk_spelling():
    """Callers must not depend on the SDK's camelCase item type spelling."""
    assert cs.codex_item_type({"type": "commandExecution"}) == "commandexecution"
    assert cs.codex_item_type({"type": "command_execution"}) == "commandexecution"
    assert cs.codex_item_type({}) == ""


def test_describe_codex_item_summarizes_the_three_reported_kinds():
    """Command, file-change and message items each get a readable log line."""
    assert cs.describe_codex_item({"type": "agentMessage", "text": " done "}) == "done"
    assert cs.describe_codex_item({"type": "commandExecution", "command": "ls", "exitCode": 2}) == "$ ls (exit 2)"
    assert cs.describe_codex_item({"type": "commandExecution", "command": "ls"}) == "$ ls"
    assert (
        cs.describe_codex_item({"type": "fileChange", "changes": [{"path": "/out/analysis.md"}]})
        == "wrote /out/analysis.md"
    )
    assert cs.describe_codex_item({"type": "reasoning"}) == ""
    assert cs.describe_codex_item({"type": "fileChange", "changes": "not-a-list"}) == ""
    assert cs.describe_codex_item({"type": "commandExecution", "exitCode": 0}) == ""


def test_codex_file_changes_keeps_only_real_paths():
    """Malformed change entries are dropped rather than surfaced as paths."""
    item = {"type": "fileChange", "changes": [{"path": "/a"}, {"kind": "add"}, "junk", {"path": 7}]}

    assert cs.codex_file_changes(item) == ("/a",)


# --------------------------------------------------------------------------- #
# Installed-SDK contract
# --------------------------------------------------------------------------- #


def test_installed_codex_sdk_exposes_the_api_this_module_drives():
    """Guard the SDK surface so an upgrade fails here, not mid-session."""
    sdk = pytest.importorskip("openai_codex")

    assert {"read_only", "workspace_write", "full_access"} <= set(sdk.Sandbox.__members__)
    assert "deny_all" in sdk.ApprovalMode.__members__
    config_fields = sdk.CodexConfig.__dataclass_fields__
    for field_name in ("codex_bin", "config_overrides", "cwd", "env", "client_name", "client_title"):
        assert field_name in config_fields, field_name
    for method in ("thread_start", "close"):
        assert hasattr(sdk.AsyncCodex, method), method
    assert hasattr(sdk.AsyncThread, "turn")
    for method in ("run", "interrupt"):
        assert hasattr(sdk.AsyncTurnHandle, method), method
    for attr in ("final_response", "items", "usage", "error"):
        assert attr in sdk.TurnResult.__annotations__, attr
