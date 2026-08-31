"""Claude backend session continuation + the read-only lesson summarizer.

GPU-free and SDK-free: the SDK's ``query`` and options type are replaced with
fakes, so these tests pin the contract the lesson summarizer depends on —
capturing a session id, passing ``resume`` through to the SDK, and resuming
under a policy that differs from the session being continued.
"""

from __future__ import annotations

import asyncio
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
from kernelforge.agent_backends.claude import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeBackend,
    ClaudeBackendError,
    _supports_adaptive_thinking,
)
from kernelforge.llm.git import git
from kernelforge.orchestrator.agent import _make_session_summarizer


class _FakeOptions:
    """Stand-in for ClaudeAgentOptions that records what it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _message(**fields):
    return SimpleNamespace(content=[], **fields)


def _result_message(session_id="", subtype="success"):
    return SimpleNamespace(
        content=[SimpleNamespace(text="done")],
        total_cost_usd=0.1,
        subtype=subtype,
        num_turns=3,
        session_id=session_id,
    )


def _backend(messages, captured, stream_error=None):
    """Build a ClaudeBackend whose SDK is replaced by a recording fake."""
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = SimpleNamespace(
        provider="claude",
        model="fake-model",
        executable="",
        timeout_sec=60,
        reasoning_effort="high",
        options={},
    )
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages:
            yield message
        if stream_error is not None:
            raise stream_error

    backend._query = fake_query
    backend._options_type = _FakeOptions
    return backend


@lru_cache(maxsize=1)
def _guarded_workspace() -> str:
    """A real worktree: a writable session is judged against one."""
    root = Path(tempfile.mkdtemp(prefix="forge-claude-resume-"))
    (root / "kernel.py").write_text("VALUE = 0\n")
    git("init", "--quiet", cwd=root)
    git("config", "user.email", "t@test", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-m", "baseline", cwd=root)
    return str(root)


def _spec(**overrides) -> AgentRunSpec:
    base = dict(
        system_prompt="implementer system prompt",
        user_prompt="optimize the kernel",
        cwd=_guarded_workspace(),
        model="fake-model",
        timeout_sec=60,
        tool_policy=AgentToolPolicy(read=True, write=True, shell=True, max_turns=100),
    )
    base.update(overrides)
    return AgentRunSpec(**base)


# ── session id capture ────────────────────────────────────────────────────────


def test_run_captures_the_session_id(monkeypatch):
    captured: dict = {}
    backend = _backend(
        [
            _message(subtype="init", session_id="sess-abc"),
            _result_message(session_id="sess-abc"),
        ],
        captured,
    )

    result = asyncio.run(backend.run(_spec()))

    assert result.session_id == "sess-abc"
    assert result.num_turns == 3


def test_run_without_a_session_id_stays_empty():
    captured: dict = {}
    backend = _backend([_result_message()], captured)
    result = asyncio.run(backend.run(_spec()))
    assert result.session_id == ""


def test_turn_cap_after_session_id_returns_a_resumable_result():
    captured: dict = {}
    backend = _backend(
        [_message(subtype="init", session_id="sess-turn-cap")],
        captured,
        stream_error=RuntimeError("maximum number of turns reached"),
    )

    result = asyncio.run(backend.run(_spec()))

    assert result.session_id == "sess-turn-cap"
    assert result.end_reason == "turn_cap"
    assert result.subtype == "error_max_turns"
    assert "maximum number of turns" in result.stderr_tail


def test_sdk_error_after_session_id_warns_and_preserves_resume(
    caplog,
):
    captured: dict = {}
    backend = _backend(
        [_message(subtype="init", session_id="sess-sdk-error")],
        captured,
        stream_error=ConnectionError("stream disconnected"),
    )

    with caplog.at_level(
        "WARNING",
        logger="kernelforge.agent_backends.claude",
    ):
        result = asyncio.run(backend.run(_spec()))

    assert result.session_id == "sess-sdk-error"
    assert result.end_reason == "sdk_error"
    assert result.subtype == "error"
    assert "stream disconnected" in result.stderr_tail
    assert "preserving the resume handle" in caplog.text


def test_stream_error_before_session_id_is_raised():
    captured: dict = {}
    backend = _backend(
        [],
        captured,
        stream_error=ConnectionError("failed before init"),
    )

    with pytest.raises(ConnectionError, match="failed before init"):
        asyncio.run(backend.run(_spec()))


def test_run_does_not_pass_a_resume_option():
    captured: dict = {}
    backend = _backend([_result_message(session_id="s1")], captured)
    asyncio.run(backend.run(_spec()))
    assert "resume" not in captured["options"].kwargs


def test_opus_5_uses_max_effort_adaptive_thinking_and_fallback():
    captured: dict = {}
    backend = _backend(
        [_result_message(session_id="s-opus-48")],
        captured,
    )
    backend.runtime.fallback_model = "claude-opus-4-8"
    policy = AgentToolPolicy(
        read=True,
        write=True,
        shell=True,
        max_turns=500,
        thinking_budget_tokens=3000,
    )

    asyncio.run(
        backend.run(
            _spec(
                model=DEFAULT_CLAUDE_MODEL,
                reasoning_effort="max",
                tool_policy=policy,
            )
        )
    )

    kwargs = captured["options"].kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["fallback_model"] == "claude-opus-4-8"
    assert kwargs["effort"] == "max"
    assert kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-8-20260101", True),
        ("anthropic/claude-opus-4-7", True),
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-6", True),
        ("claude-haiku-4-8", True),
        ("claude-opus-5", True),
        ("company-current-claude", True),
        ("claude-opus-4", False),
        ("claude-sonnet-4", False),
        ("claude-opus-4-5-20251101", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-3-7-sonnet", False),
        ("", False),
    ],
)
def test_adaptive_thinking_follows_model_capability(model, expected):
    assert _supports_adaptive_thinking(model) is expected


def test_legacy_claude_model_uses_fixed_thinking_budget():
    captured: dict = {}
    backend = _backend(
        [_result_message(session_id="s-opus-45")],
        captured,
    )
    policy = AgentToolPolicy(
        read=True,
        max_turns=10,
        thinking_budget_tokens=3000,
    )

    asyncio.run(
        backend.run(
            _spec(
                model="claude-opus-4-5-20251101",
                reasoning_effort="high",
                tool_policy=policy,
            )
        )
    )

    assert captured["options"].kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 3000,
    }


# ── resume ────────────────────────────────────────────────────────────────────


def test_resume_passes_the_session_id_and_new_prompt():
    captured: dict = {}
    backend = _backend([_result_message(session_id="sess-abc")], captured)

    result = asyncio.run(backend.resume(_spec(), "sess-abc", "write your lesson now"))

    assert captured["options"].kwargs["resume"] == "sess-abc"
    assert captured["prompt"] == "write your lesson now"
    assert result.session_id == "sess-abc"


def test_resume_backfills_a_missing_session_id():
    captured: dict = {}
    backend = _backend([_result_message()], captured)
    result = asyncio.run(backend.resume(_spec(), "sess-xyz", "prompt"))
    assert result.session_id == "sess-xyz"


def test_resume_requires_a_session_id():
    captured: dict = {}
    backend = _backend([_result_message()], captured)
    with pytest.raises(ClaudeBackendError):
        asyncio.run(backend.resume(_spec(), "  ", "prompt"))


def test_resume_honours_this_turns_policy_not_the_original():
    """A writable implementer session must be resumable under a read-only policy."""
    captured: dict = {}
    backend = _backend([_result_message(session_id="s")], captured)

    read_only = _spec(
        tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=False, max_turns=4),
        system_prompt="summarizer role",
    )
    asyncio.run(backend.resume(read_only, "s", "prompt"))

    kwargs = captured["options"].kwargs
    assert kwargs["allowed_tools"] == ["Read", "Grep", "Glob"]
    assert kwargs["max_turns"] == 4
    assert kwargs["system_prompt"] == "summarizer role"


def test_provider_omits_max_turns_for_time_limited_session():
    captured: dict = {}
    backend = _backend([_result_message(session_id="s")], captured)
    time_limited = _spec(
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=None,
        ),
    )

    asyncio.run(backend.run(time_limited))

    kwargs = captured["options"].kwargs
    assert kwargs["allowed_tools"] == ["Read", "Grep", "Glob"]
    assert "max_turns" not in kwargs


# ── the summarizer the agent layer hands back ─────────────────────────────────


class _RecordingBackend:
    capabilities = SimpleNamespace(resumable=True)

    def __init__(self):
        self.calls: list[tuple] = []

    async def resume(self, spec, session_id, feedback, usage=None):
        self.calls.append((spec, session_id, feedback, usage))
        return SimpleNamespace(text="lesson text")


def test_summarizer_resumes_read_only_and_without_hooks():
    backend = _RecordingBackend()
    implementer_spec = _spec(hooks=object(), writable=True, protected_globs=["driver.py"])

    summarize = _make_session_summarizer(backend=backend, spec=implementer_spec, session_id="sess-1", usage="usage-obj")
    reply = asyncio.run(summarize("record your lesson"))
    assert reply == "lesson text"

    spec, session_id, feedback, usage = backend.calls[0]
    assert session_id == "sess-1"
    assert feedback == "record your lesson"
    assert usage == "usage-obj"
    # The in-session gate's Stop hook would otherwise block the summarizing turn
    # and push the agent back into editing the kernel.
    assert spec.hooks is None
    assert spec.writable is False
    assert spec.reasoning_effort == "high"
    assert spec.tool_policy.write is False
    assert spec.tool_policy.shell is False
    assert spec.protected_globs == ["*"]
    # Providers that guard the worktree before resuming must not refuse to start
    # over the pending candidate diff or leftover build artifacts.
    assert spec.allow_dirty_targets is True
    assert spec.allow_untracked is True
    assert spec.read_only_resume is True
    # The implementer's own spec is untouched.
    assert implementer_spec.hooks is not None
    assert implementer_spec.writable is True
    assert implementer_spec.read_only_resume is False


def test_summarizer_is_none_without_a_session_id():
    summarize = _make_session_summarizer(backend=_RecordingBackend(), spec=_spec(), session_id="", usage=None)
    assert summarize is None


def test_summarizer_is_none_for_a_non_resumable_provider():
    backend = _RecordingBackend()
    backend.capabilities = SimpleNamespace(resumable=False)
    summarize = _make_session_summarizer(backend=backend, spec=_spec(), session_id="s", usage=None)
    assert summarize is None


def test_summarizer_is_none_when_the_backend_cannot_resume():
    backend = SimpleNamespace(capabilities=SimpleNamespace(resumable=True))
    summarize = _make_session_summarizer(backend=backend, spec=_spec(), session_id="s", usage=None)
    assert summarize is None


# ── workspace guard ───────────────────────────────────────────────────────────


def _guarded_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    kernel = root / "kernel.py"
    kernel.write_text("VALUE = 0\n")
    (root / "forge_driver.py").write_text("pass\n")
    git("init", "--quiet", cwd=root)
    git("config", "user.email", "t@test", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-m", "baseline", cwd=root)
    return root, kernel


def test_a_session_that_edits_a_protected_file_is_rejected(tmp_path):
    """The default backend now answers the question its sibling always did."""
    root, kernel = _guarded_repo(tmp_path)
    captured: dict = {}

    async def edit_the_driver(prompt, options):
        (root / "forge_driver.py").write_text("print('measurement changed')\n")
        yield _result_message(session_id="sess-guarded")

    backend = _backend([], captured)
    backend._query = edit_the_driver

    with pytest.raises(Exception, match="forge_driver.py"):
        asyncio.run(
            backend.run(
                _spec(
                    cwd=str(root),
                    target_files=[str(kernel)],
                    driver_script=str(root / "forge_driver.py"),
                )
            )
        )

    assert (root / "forge_driver.py").read_text() == "pass\n"


def test_a_session_that_stays_in_its_target_reports_what_it_changed(tmp_path):
    root, kernel = _guarded_repo(tmp_path)
    captured: dict = {}

    async def edit_the_kernel(prompt, options):
        kernel.write_text("VALUE = 1\n")
        yield _result_message(session_id="sess-clean")

    backend = _backend([], captured)
    backend._query = edit_the_kernel

    result = asyncio.run(
        backend.run(
            _spec(
                cwd=str(root),
                target_files=[str(kernel)],
                driver_script=str(root / "forge_driver.py"),
            )
        )
    )

    assert result.file_changes == ["kernel.py"]
    assert result.target_edit_count == 1
    assert kernel.read_text() == "VALUE = 1\n"


def test_the_backend_declares_the_guard_it_now_runs():
    assert ClaudeBackend.capabilities.workspace_guard is True
