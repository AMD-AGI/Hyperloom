"""Tests for the DeepSeek Harness backend transport, sandbox and workspace guards."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.reasoning_effort import harness_reasoning_effort
from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.deepseek import (
    DEEPSEEK_PROVIDER_ROUTE,
    DeepSeekHarnessBackend,
    DeepSeekHarnessExecutionError,
    DeepSeekHarnessUnavailableError,
    _normalize_run_result,
    _usage_from_events,
    resolve_deepseek_reasoning_effort,
)
from kernelforge.agent_backends.registry import get_agent_provider
from kernelforge.agent_backends.workspace_guard import WorkspaceSafetyError
from kernelforge.tracker.usage import UsageAccumulator

_GATEWAY = {
    "base_url": "https://gateway.example.invalid/v1",
    "key_env": "FAKE_DEEPSEEK_API_KEY",
}


def _git(cwd: Path, *args: str) -> str:
    """Run one git command for a temporary test repository."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a clean repository with one committed target file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    kernel = repo / "kernel.py"
    kernel.write_text("VALUE = 1\n")
    _git(repo, "config", "user.name", "KernelForge Test")
    _git(repo, "config", "user.email", "kernelforge-test@example.invalid")
    _git(repo, "add", "kernel.py")
    _git(repo, "commit", "-q", "-m", "test baseline")
    return repo, kernel


class _FakeHarness:
    """Record the handshake and replay a scripted result per prompt."""

    instances: list["_FakeHarness"] = []

    def __init__(self, **kwargs: object) -> None:
        """Capture the initialization arguments the backend chose."""
        self.kwargs = kwargs
        self.prompts: list[tuple[str, str]] = []
        self.closed = False
        self.on_run = None
        _FakeHarness.instances.append(self)

    def run(self, prompt: str, *, session_id: str | None = None) -> SimpleNamespace:
        """Return one deterministic activity-interval result."""
        self.prompts.append((prompt, str(session_id)))
        if self.on_run is not None:
            self.on_run(self, prompt, session_id)
        return SimpleNamespace(
            session_id=str(session_id),
            final_response="done",
            finish_reason="completed",
            events=[],
        )

    def close(self) -> None:
        """Mark this runtime shut down."""
        self.closed = True


class _FakeHarnessSdk:
    """Collect the fake classes exposed by the public Python SDK."""

    DeepSeekHarness = _FakeHarness


@pytest.fixture(autouse=True)
def _install_fake_harness_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route backend tests through the deterministic SDK transport fake."""
    _FakeHarness.instances.clear()
    monkeypatch.setattr(
        "kernelforge.agent_backends.deepseek._load_harness_sdk",
        lambda: _FakeHarnessSdk,
    )


def _backend(**runtime_fields: object) -> DeepSeekHarnessBackend:
    """Build a backend with a deterministic fake gateway."""
    fields: dict[str, object] = {"provider": "deepseek", "model": "deepseek-v4-flash"}
    fields.update(runtime_fields)
    from kernelforge.llm import LlmGateway

    return DeepSeekHarnessBackend(
        gateway=LlmGateway.from_mapping(_GATEWAY),
        runtime=AgentRuntimeConfig(**fields),  # type: ignore[arg-type]
    )


def _spec(repo: Path, kernel: Path, **fields: object) -> AgentRunSpec:
    """Build one writable run specification for tests."""
    base: dict[str, object] = {
        "system_prompt": "Optimize the kernel.",
        "user_prompt": "Make the change now.",
        "cwd": str(repo),
        "timeout_sec": 30,
        "target_files": [str(kernel)],
    }
    base.update(fields)
    return AgentRunSpec(**base)  # type: ignore[arg-type]


def test_provider_is_registered_with_the_capabilities_it_implements() -> None:
    """The registry advertises exactly what this backend can be asked to do."""
    provider = get_agent_provider("deepseek")

    assert provider.default_model == "deepseek-v4-flash"
    assert provider.owns_model("deepseek-v4-flash")
    assert not provider.owns_model("claude-opus-5")
    assert not provider.owns_model("gpt-5.6-sol")
    assert provider.capabilities.resumable
    assert provider.capabilities.workspace_guard
    assert provider.capabilities.session_env
    # The harness runs no equivalent of the PreToolUse/PostToolUse/Stop groups and reports no incremental progress, so
    # a caller that builds either must be able to see that before it relies on one.
    assert not provider.capabilities.stop_hooks
    assert not provider.capabilities.native_subagents


def test_registry_capabilities_match_the_backend_class() -> None:
    """One declaration, so a capability cannot be advertised but not implemented."""
    assert get_agent_provider("deepseek").capabilities == DeepSeekHarnessBackend.capabilities


@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ("low", "low"),
        # The route rejects "medium" outright, so it has to become a level that exists. Rounding up keeps a campaign
        # that asked for more deliberation from silently getting less.
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ],
)
def test_reasoning_effort_projects_onto_the_levels_the_route_accepts(asked: str, expected: str) -> None:
    """Every ladder level maps to one the harness will not refuse at initialization."""
    assert harness_reasoning_effort(asked) == expected
    assert resolve_deepseek_reasoning_effort(asked) == expected


def test_an_off_ladder_effort_is_refused_rather_than_rewritten() -> None:
    """A typo must fail loudly instead of travelling into the run as a default."""
    with pytest.raises(DeepSeekHarnessExecutionError, match="unsupported"):
        resolve_deepseek_reasoning_effort("enormous")


@pytest.mark.parametrize(
    ("sandbox_mode", "writable", "expected"),
    [
        ("bypass", True, "danger-full-access"),
        ("bypass", False, "read-only"),
        ("workspace-write", True, "danger-full-access"),
        ("workspace-write", False, "read-only"),
        ("read-only", True, "read-only"),
        ("read-only", False, "read-only"),
    ],
)
def test_permission_mode_confines_a_read_only_session_and_lets_a_writable_one_write(
    sandbox_mode: str,
    writable: bool,
    expected: str,
) -> None:
    """A writable session cannot use workspace-write: its approvals fail closed.

    The harness pairs workspace-write with an ``ask`` approval policy and an SDK
    client registers no answerer, so every write would be refused on a request
    nobody can answer. Confinement for a writable session is WorkspaceGuard's.
    """
    backend = _backend(sandbox_mode=sandbox_mode)
    spec = AgentRunSpec(system_prompt="s", user_prompt="u", cwd=".", writable=writable)

    assert backend._permission_mode(spec) == expected


def test_the_retired_deepseek_variables_are_not_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """This provider must not resurrect a spelling the preflight migrates away from.

    ``DEEPSEEK_BASE_URL``/``DEEPSEEK_API_KEY`` are retired: the preflight
    rewrites them into the two standard lines. Reading them here would make a
    box configured the old way work by accident and stop reporting the
    deprecation it is supposed to.
    """
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://retired.example.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "retired-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = DeepSeekHarnessBackend(runtime=AgentRuntimeConfig(provider="deepseek", model="deepseek-v4-flash"))

    with pytest.raises(DeepSeekHarnessUnavailableError, match="OPENAI_BASE_URL"):
        backend.preflight()


def test_child_environment_carries_the_route_credential_and_the_session_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subprocess gets the endpoint, the key, the sandbox and the caller's overlay."""
    monkeypatch.setenv("FAKE_DEEPSEEK_API_KEY", "secret-value")
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/elsewhere")
    backend = _backend()
    spec = AgentRunSpec(
        system_prompt="s",
        user_prompt="u",
        cwd=".",
        env={"HIP_VISIBLE_DEVICES": "3"},
    )

    env = backend._child_environment(spec)

    assert env["DEEPSEEK_BASE_URL"] == _GATEWAY["base_url"]
    assert env["DEEPSEEK_API_KEY"] == "secret-value"
    assert env["DSH_PERMISSION_MODE"] == "danger-full-access"
    assert env["HIP_VISIBLE_DEVICES"] == "3"
    # These outrank cwd, so a caller that set them to reach one repository would answer every git command the session
    # runs anywhere.
    assert "GIT_DIR" not in env
    assert "GIT_WORK_TREE" not in env


def test_two_roles_in_one_workspace_share_a_runtime(tmp_path: Path) -> None:
    """The system prompt rides the message, so roles need no runtime of their own.

    Each runtime is a Node subprocess. Keying them by system prompt would start
    one per role in a campaign that runs several against the same worktree.
    """
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    implementer = _spec(repo, kernel, system_prompt="Implement.").resolved(backend.runtime)
    analysis = _spec(repo, kernel, system_prompt="Analyse.").resolved(backend.runtime)

    assert backend._runtime_key(implementer) == backend._runtime_key(analysis)


@pytest.mark.parametrize(
    "divergent",
    [
        {"model": "deepseek-v4-pro"},
        {"writable": False},
        {"env": {"HIP_VISIBLE_DEVICES": "1"}},
    ],
)
def test_anything_fixed_at_initialization_gets_its_own_runtime(
    tmp_path: Path,
    divergent: dict[str, object],
) -> None:
    """Values sent in the handshake or the child env cannot be shared across runtimes."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    base = _spec(repo, kernel).resolved(backend.runtime)
    other = _spec(repo, kernel, **divergent).resolved(backend.runtime)

    assert backend._runtime_key(base) != backend._runtime_key(other)


def test_run_sends_the_system_prompt_with_the_task_and_names_the_route(tmp_path: Path) -> None:
    """The instructions reach the model even though the SDK has no field for them."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()

    result = asyncio.run(backend.run(_spec(repo, kernel)))

    assert result.session_id.startswith("forge-")
    harness = _FakeHarness.instances[-1]
    assert harness.kwargs["provider"] == DEEPSEEK_PROVIDER_ROUTE
    # The sandbox root comes from the subprocess working directory, so a runtime rooted anywhere else would confine a
    # read-only session to the wrong tree.
    assert harness.kwargs["runtime_cwd"] == str(repo.resolve())
    assert harness.kwargs["cwd"] == str(repo.resolve())
    prompt, session_id = harness.prompts[0]
    assert "Optimize the kernel." in prompt
    assert "Make the change now." in prompt
    assert session_id == result.session_id


def test_resume_continues_the_same_session_on_the_same_runtime(tmp_path: Path) -> None:
    """A second runtime cannot adopt the session, so the first one has to be reused.

    The SDK answers a session id its own process already ran with ``session
    "<id>" already exists``, so a resume that started a fresh runtime would
    either fail or silently judge the work from a conversation that never saw
    it.
    """
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    spec = _spec(repo, kernel)

    first = asyncio.run(backend.run(spec))
    resumed = asyncio.run(backend.resume(spec, first.session_id, "gate feedback"))

    assert resumed.session_id == first.session_id
    assert len(_FakeHarness.instances) == 1
    assert _FakeHarness.instances[0].prompts[1] == ("gate feedback", first.session_id)


def test_a_read_only_resume_reaches_the_runtime_that_owns_the_session(tmp_path: Path) -> None:
    """The lesson summarizer resumes a writable implementer read-only; a live run got 'session already exists'."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    first = asyncio.run(backend.run(_spec(repo, kernel)))

    from kernelforge.agent_backends.base import AgentToolPolicy

    summary_spec = _spec(
        repo,
        kernel,
        writable=False,
        read_only_resume=True,
        allow_dirty_targets=True,
        allow_untracked=True,
        protected_globs=["*"],
        tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=False, max_turns=4),
    )
    resumed = asyncio.run(backend.resume(summary_spec, first.session_id, "summarize what you explored"))

    assert resumed.session_id == first.session_id
    assert len(_FakeHarness.instances) == 1
    assert _FakeHarness.instances[0].prompts[1] == ("summarize what you explored", first.session_id)


def test_resume_without_a_session_id_is_refused(tmp_path: Path) -> None:
    """An empty id would start a new conversation and call it a continuation."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()

    with pytest.raises(DeepSeekHarnessExecutionError, match="requires a session ID"):
        asyncio.run(backend.resume(_spec(repo, kernel), "  ", "feedback"))


def test_a_runtime_whose_workspace_is_gone_is_shut_down(tmp_path: Path) -> None:
    """Every lane round clones and then deletes its workspaces; a live campaign held ten runtimes for them."""
    import shutil

    lane_root = tmp_path / "lane"
    lane_root.mkdir()
    lane_repo, lane_kernel = _make_repo(lane_root)
    (tmp_path / "main").mkdir()
    repo, kernel = _make_repo(tmp_path / "main")
    backend = _backend()
    asyncio.run(backend.run(_spec(lane_repo, lane_kernel)))
    shutil.rmtree(lane_repo)

    asyncio.run(backend.run(_spec(repo, kernel)))

    lane_runtime, main_runtime = _FakeHarness.instances
    assert lane_runtime.closed
    assert not main_runtime.closed


def test_close_shuts_down_every_runtime_it_started(tmp_path: Path) -> None:
    """A campaign must not leak one Node subprocess per workspace it visited."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    asyncio.run(backend.run(_spec(repo, kernel)))

    backend.close()

    assert [harness.closed for harness in _FakeHarness.instances] == [True]


def test_a_session_that_edits_a_protected_measurement_file_is_rejected_and_restored(
    tmp_path: Path,
) -> None:
    """The shared workspace guard judges this provider's sessions like the others'.

    A session that rewrites the file the speedup is measured with could report
    any gain it liked, so the turn is refused and the baseline restored rather
    than scored.
    """
    repo, kernel = _make_repo(tmp_path)
    measure = repo / "measure.py"
    measure.write_text("BUDGET = 1\n")
    _git(repo, "add", "measure.py")
    _git(repo, "commit", "-q", "-m", "measurement harness")
    backend = _backend()

    def edit_protected_file(harness: _FakeHarness, prompt: str, session_id: str | None) -> None:
        """Rewrite the file the spec declared off limits."""
        measure.write_text("BUDGET = 999\n")

    spec = _spec(repo, kernel, protected_paths=[str(measure)])
    original = DeepSeekHarnessBackend._harness_for

    def patched(self: DeepSeekHarnessBackend, run_spec: AgentRunSpec) -> _FakeHarness:
        harness = original(self, run_spec)
        harness.on_run = edit_protected_file
        return harness

    DeepSeekHarnessBackend._harness_for = patched  # type: ignore[method-assign]
    try:
        with pytest.raises(WorkspaceSafetyError, match="protected tracked files changed"):
            asyncio.run(backend.run(spec))
    finally:
        DeepSeekHarnessBackend._harness_for = original  # type: ignore[method-assign]

    assert measure.read_text() == "BUDGET = 1\n"


def test_usage_sums_per_step_tokens_without_double_counting_the_prefix() -> None:
    """``totalTokens`` is a running turn total, so only the deltas may be summed."""
    events = [
        {
            "type": "assistant/message",
            "data": {"usage": {"inputTokens": 7246, "outputTokens": 54, "totalTokens": 7300}},
        },
        {
            "type": "assistant/message",
            "data": {
                "usage": {
                    "inputTokens": 188,
                    "outputTokens": 18,
                    "totalTokens": 7374,
                    "cacheReadTokens": 7168,
                }
            },
        },
        {"type": "tool/call", "data": {"name": "read"}},
    ]

    usage = _usage_from_events(events)

    assert usage["input_tokens"] == 7434
    assert usage["output_tokens"] == 72
    assert usage["cache_read_input_tokens"] == 7168


def test_a_session_that_reported_no_usage_records_none() -> None:
    """An empty dict keeps an unmeasured session out of the campaign's token totals."""
    assert _usage_from_events([{"type": "turn/end", "data": {}}]) == {}


def test_normalize_reports_tool_calls_and_session_identity() -> None:
    """The loop reads what the session did from the events it published."""
    result = _normalize_run_result(
        SimpleNamespace(
            session_id="forge-1",
            final_response="PLAN: vectorize loads",
            finish_reason="completed",
            events=[
                {"type": "tool/call", "data": {"name": "read", "callId": "call-1", "step": 1}},
                {"type": "tool/call", "data": {"name": "edit", "callId": "call-2", "step": 2}},
            ],
        ),
        "fallback",
    )

    assert result.session_id == "forge-1"
    assert result.text == "PLAN: vectorize loads"
    assert result.subtype == "success"
    assert [name for name, _ in result.tool_calls] == ["read", "edit"]


def test_a_provider_failure_inside_a_completed_run_is_reported_as_one() -> None:
    """The harness reports the failure on turn/end while the run itself returns.

    Reading only ``final_response`` would file a transport failure as a session
    that had nothing to say, and the loop would score it as a real answer.
    """
    result = _normalize_run_result(
        SimpleNamespace(
            session_id="forge-2",
            final_response="",
            finish_reason="error",
            events=[
                {
                    "type": "turn/end",
                    "data": {"reason": {"kind": "error", "error": {"code": "TRANSPORT", "message": "request failed"}}},
                }
            ],
        ),
        "forge-2",
    )

    assert result.subtype == "error"
    assert result.end_reason == "sdk_error"
    assert "TRANSPORT: request failed" in result.findings[0]
    assert "TRANSPORT" in result.text


def test_a_spent_output_cap_stays_a_turn_cap() -> None:
    """A limit the caller chose is an answer, not a provider failure to retry."""
    result = _normalize_run_result(
        SimpleNamespace(session_id="forge-3", final_response="partial", finish_reason="max-tokens", events=[]),
        "forge-3",
    )

    assert result.subtype == "error_max_turns"
    assert result.end_reason == "turn_cap"
    assert result.text == "partial"


def test_usage_is_attributed_to_the_role_that_spent_it(tmp_path: Path) -> None:
    """A campaign's spend has to be splittable by what each session was for."""
    repo, kernel = _make_repo(tmp_path)
    backend = _backend()
    usage = UsageAccumulator()

    def report_usage(harness: _FakeHarness, prompt: str, session_id: str | None) -> None:
        """Nothing to do: the fake result carries the events below."""

    original = DeepSeekHarnessBackend._harness_for

    def patched(self: DeepSeekHarnessBackend, run_spec: AgentRunSpec) -> _FakeHarness:
        harness = original(self, run_spec)
        harness.on_run = report_usage
        harness.run = lambda prompt, *, session_id=None: SimpleNamespace(  # type: ignore[method-assign]
            session_id=str(session_id),
            final_response="done",
            finish_reason="completed",
            events=[
                {
                    "type": "assistant/message",
                    "data": {"usage": {"inputTokens": 10, "outputTokens": 5}},
                }
            ],
        )
        return harness

    DeepSeekHarnessBackend._harness_for = patched  # type: ignore[method-assign]
    try:
        asyncio.run(backend.run(_spec(repo, kernel, role="implementer"), usage=usage))
    finally:
        DeepSeekHarnessBackend._harness_for = original  # type: ignore[method-assign]

    totals = usage.totals()
    assert totals["input_tokens"] == 10
    assert totals["output_tokens"] == 5
    assert "implementer" in totals["by_role"]


def test_a_missing_sdk_reports_provider_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absent extra names itself rather than failing minutes into a campaign."""

    def missing() -> object:
        raise DeepSeekHarnessUnavailableError(
            "DeepSeek Harness Python SDK is not installed; install kernelforge[deepseek]"
        )

    monkeypatch.setattr("kernelforge.agent_backends.deepseek._load_harness_sdk", missing)

    with pytest.raises(DeepSeekHarnessUnavailableError, match="not installed"):
        _backend().preflight()


def test_a_timed_out_session_is_stopped_before_its_edits_are_rolled_back(tmp_path: Path) -> None:
    """The SDK has no per-turn cancel: an abandoned turn keeps editing the workspace the guard just restored."""
    import threading

    repo, kernel = _make_repo(tmp_path)
    release = threading.Event()

    def _stall(harness: _FakeHarness, _prompt: str, _session_id: object) -> None:
        if harness is _FakeHarness.instances[0]:
            release.wait(timeout=5)

    backend = _backend()
    first = backend._harness_for(_spec(repo, kernel))
    first.on_run = _stall
    first.close = lambda: (setattr(first, "closed", True), release.set())

    with pytest.raises(DeepSeekHarnessExecutionError, match="timed out"):
        asyncio.run(backend.run(_spec(repo, kernel, timeout_sec=0.05)))

    assert first.closed
    assert backend._harness_for(_spec(repo, kernel)) is not first


def test_the_relay_never_adds_the_harness_on_its_own(tmp_path):
    """Relay members are the providers auto selection could pick; the harness is reached only by naming it."""
    from kernelforge.cli.relay import resolve_relay_policy
    from kernelforge.config import Config

    claude_first = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="claude"))
    harness_first = resolve_relay_policy(Config(workspace=str(tmp_path), agent_backend="deepseek"))

    assert [member.provider for member in claude_first.profiles] == ["claude", "codex"]
    assert [member.provider for member in harness_first.profiles] == ["deepseek", "claude", "codex"]
