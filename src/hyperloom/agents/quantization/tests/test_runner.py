"""Tests for `driver.runner` — prompt assembly + SDK injection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.agents.quantization.driver import runner as runner_module
from hyperloom.agents.quantization.driver.runner import (
    DEFAULT_ALLOWED_TOOLS,
    AttemptContext,
    AttemptResult,
    ClaudeAgentSdk,
    build_attempt_prompt,
    resolve_skill_path,
    run_one_attempt,
)


# ───────────────────────────────────────────────────────────────────────────── resolve_skill_path
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_skill_path_defaults_to_package():
    p = resolve_skill_path()
    assert p.name == "SKILL.md"
    assert p.is_file(), f"SKILL.md must exist next to runner.py (parent of driver/) (got {p})"


def test_resolve_skill_path_respects_override(tmp_path):
    (tmp_path / "SKILL.md").write_text("x", encoding="utf-8")
    p = resolve_skill_path(package_root=tmp_path)
    assert p == tmp_path / "SKILL.md"


def test_quark_py310_compat_dir_is_process_singleton():
    """Repeated attempts reuse one read-only compatibility directory."""
    first = runner_module._prepare_quark_py310_compat()
    second = runner_module._prepare_quark_py310_compat()

    assert first == second
    assert first.is_dir()
    assert (first.stat().st_mode & 0o222) == 0


# ───────────────────────────────────────────────────────────────────────────── build_attempt_prompt
# ─────────────────────────────────────────────────────────────────────────────


def test_build_attempt_prompt_includes_skill_and_user_prompt(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("skill body", encoding="utf-8")
    text = build_attempt_prompt(
        AttemptContext(
            user_prompt="Quantize Qwen/Qwen3-0.5B in fp8",
            workspace=tmp_path / "ws",
            quark_root=tmp_path / "qr",
            attempt_number=1,
            acceptable_eval_gap=0.05,
            interactive=False,
            previous_outcome=None,
        ),
        skill_path=skill,
        fix_hypothesis_path=None,
    )
    assert str(skill) in text
    assert "Quantize Qwen/Qwen3-0.5B in fp8" in text
    assert str(tmp_path / "ws") in text
    assert str(tmp_path / "qr") in text
    assert "0.0500" in text
    assert "off (batch / non-interactive)" in text
    assert "Retry context" not in text


def test_build_attempt_prompt_retry_block_added_on_attempt_2(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    fix_hyp = tmp_path / "fix_hypothesis_attempt_2.md"
    text = build_attempt_prompt(
        AttemptContext(
            user_prompt="do",
            workspace=tmp_path,
            quark_root=tmp_path,
            attempt_number=2,
            acceptable_eval_gap=None,
            interactive=None,
            previous_outcome="exec_oom",
        ),
        skill_path=skill,
        fix_hypothesis_path=fix_hyp,
    )
    assert "Retry context" in text
    assert "exec_oom" in text
    assert "fix_hypothesis_attempt_2.md" in text
    assert str(fix_hyp) in text
    assert "auto" in text  # interactive=None description


def test_build_attempt_prompt_interactive_on(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    text = build_attempt_prompt(
        AttemptContext(
            user_prompt="x",
            workspace=tmp_path,
            quark_root=tmp_path,
            attempt_number=1,
            acceptable_eval_gap=None,
            interactive=True,
            previous_outcome=None,
        ),
        skill_path=skill,
        fix_hypothesis_path=None,
    )
    assert "on (always relay checkpoints to operator)" in text


def test_build_attempt_prompt_default_threshold_message(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    text = build_attempt_prompt(
        AttemptContext(
            user_prompt="x",
            workspace=tmp_path,
            quark_root=tmp_path,
            attempt_number=1,
            acceptable_eval_gap=None,
            interactive=False,
            previous_outcome=None,
        ),
        skill_path=skill,
        fix_hypothesis_path=None,
    )
    assert "caller did not override" in text


# ───────────────────────────────────────────────────────────────────────────── run_one_attempt — SDK injection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_one_attempt_invokes_sdk_with_prompt(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("skill", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    result = await run_one_attempt(
        AttemptContext(user_prompt="my prompt", workspace=tmp_path / "ws", quark_root=qr),
        skill_path=skill,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )

    assert isinstance(result, AttemptResult)
    assert result.sdk_error == ""
    assert len(fake_sdk.received_prompts) == 1
    assert "my prompt" in fake_sdk.received_prompts[0]
    assert str(skill) in fake_sdk.received_prompts[0]


@pytest.mark.asyncio
async def test_run_one_attempt_sets_cwd_to_workspace(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()
    ws = tmp_path / "ws"

    await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=ws, quark_root=qr),
        skill_path=skill,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )
    options = fake_sdk.received_options[0]
    assert options.kwargs.get("cwd") == str(ws)
    assert options.kwargs.get("allowed_tools") == DEFAULT_ALLOWED_TOOLS


@pytest.mark.asyncio
async def test_run_one_attempt_passes_quark_py310_env_to_sdk_options(tmp_path, fake_sdk, fake_options_cls, monkeypatch):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")
    monkeypatch.delenv("PIP_IGNORE_REQUIRES_PYTHON", raising=False)

    await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=qr),
        skill_path=skill,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )

    env = fake_sdk.received_options[0].kwargs["env"]
    pythonpath = env["PYTHONPATH"]
    compat_dir = Path(pythonpath.split(os.pathsep)[0])
    sitecustomize = compat_dir / "sitecustomize.py"
    ws = tmp_path / "ws"
    assert sitecustomize.is_file()
    assert not str(compat_dir).startswith(str(ws.resolve()))
    assert "typing.Self" in sitecustomize.read_text(encoding="utf-8")
    assert env["PYTHONPATH"].startswith(str(compat_dir))
    assert env["PIP_IGNORE_REQUIRES_PYTHON"] == "1"
    assert os.environ["PYTHONPATH"] == "/existing/pythonpath"
    assert "PIP_IGNORE_REQUIRES_PYTHON" not in os.environ
    assert (sitecustomize.stat().st_mode & 0o222) == 0


@pytest.mark.asyncio
async def test_run_one_attempt_captures_sdk_exception(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    fake_sdk.side_effect = RuntimeError("boom from SDK")
    result = await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=qr),
        skill_path=skill,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )
    assert "RuntimeError" in result.sdk_error
    assert "boom from SDK" in result.sdk_error
    # Workspace dir should still be created.
    assert (tmp_path / "ws").is_dir()


@pytest.mark.asyncio
async def test_run_one_attempt_logs_the_sdk_traceback_without_a_log_callback(
    tmp_path, fake_sdk, fake_options_cls, caplog
):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    fake_sdk.side_effect = RuntimeError("boom from SDK")
    with caplog.at_level("WARNING", logger=runner_module.__name__):
        await run_one_attempt(
            AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=tmp_path),
            skill_path=skill,
            sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
        )
    (record,) = caplog.records
    assert record.getMessage() == "quantization-agent SDK attempt 1 failed: RuntimeError: boom from SDK"
    assert record.exc_info[1] is fake_sdk.side_effect


@pytest.mark.asyncio
async def test_run_one_attempt_skill_missing_raises(tmp_path, fake_sdk, fake_options_cls):
    qr = tmp_path / "qr"
    qr.mkdir()
    with pytest.raises(FileNotFoundError):
        await run_one_attempt(
            AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=qr),
            skill_path=tmp_path / "nope.md",
            sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
        )


@pytest.mark.asyncio
async def test_run_one_attempt_retry_picks_up_hypothesis(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    hyp = ws / "fix_hypothesis_attempt_2.md"
    hyp.write_text("retry plan", encoding="utf-8")

    await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=ws, quark_root=qr, attempt_number=2, previous_outcome="exec_oom"),
        skill_path=skill,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )
    prompt = fake_sdk.received_prompts[0]
    assert "exec_oom" in prompt
    assert "fix_hypothesis_attempt_2.md" in prompt
    assert str(hyp) in prompt


@pytest.mark.asyncio
async def test_run_one_attempt_passes_model(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=qr),
        skill_path=skill,
        model="custom-model-id",
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )
    assert fake_sdk.received_options[0].kwargs.get("model") == "custom-model-id"


@pytest.mark.asyncio
async def test_run_one_attempt_log_callback_captures_chunks(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()
    captured: list[str] = []
    fake_sdk.scripted_chunks = ["alpha", "beta"]

    await run_one_attempt(
        AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=qr),
        skill_path=skill,
        log=captured.append,
        sdk=ClaudeAgentSdk(query=fake_sdk, options_cls=fake_options_cls),
    )
    assert any("alpha" in line for line in captured)
    assert any("beta" in line for line in captured)
