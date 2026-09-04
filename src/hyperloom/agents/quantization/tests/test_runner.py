"""Tests for `driver.runner` — prompt assembly + SDK injection.

Uses ``FakeSDK`` / ``FakeOptions`` from conftest to bypass the real SDK.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hyperloom.agents.quantization.driver.runner import (
    DEFAULT_ALLOWED_TOOLS,
    AttemptResult,
    build_attempt_prompt,
    resolve_skill_path,
    run_one_attempt,
)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_skill_path
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_skill_path_defaults_to_package():
    p = resolve_skill_path()
    assert p.name == "SKILL.md"
    assert p.is_file(), f"SKILL.md must exist next to runner.py (parent of driver/) (got {p})"


def test_resolve_skill_path_respects_override(tmp_path):
    (tmp_path / "SKILL.md").write_text("x", encoding="utf-8")
    p = resolve_skill_path(package_root=tmp_path)
    assert p == tmp_path / "SKILL.md"


# ─────────────────────────────────────────────────────────────────────────────
# build_attempt_prompt
# ─────────────────────────────────────────────────────────────────────────────


def test_build_attempt_prompt_includes_skill_and_user_prompt(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("skill body", encoding="utf-8")
    text = build_attempt_prompt(
        user_prompt="Quantize Qwen/Qwen3-0.5B in fp8",
        skill_path=skill,
        workspace=tmp_path / "ws",
        quark_root=tmp_path / "qr",
        attempt_number=1,
        acceptable_eval_gap=0.05,
        interactive=False,
        previous_outcome=None,
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
        user_prompt="do",
        skill_path=skill,
        workspace=tmp_path,
        quark_root=tmp_path,
        attempt_number=2,
        acceptable_eval_gap=None,
        interactive=None,
        previous_outcome="exec_oom",
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
        user_prompt="x",
        skill_path=skill,
        workspace=tmp_path,
        quark_root=tmp_path,
        attempt_number=1,
        acceptable_eval_gap=None,
        interactive=True,
        previous_outcome=None,
        fix_hypothesis_path=None,
    )
    assert "on (always relay checkpoints to operator)" in text


def test_build_attempt_prompt_default_threshold_message(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    text = build_attempt_prompt(
        user_prompt="x",
        skill_path=skill,
        workspace=tmp_path,
        quark_root=tmp_path,
        attempt_number=1,
        acceptable_eval_gap=None,
        interactive=False,
        previous_outcome=None,
        fix_hypothesis_path=None,
    )
    assert "caller did not override" in text


# ─────────────────────────────────────────────────────────────────────────────
# run_one_attempt — SDK injection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_one_attempt_invokes_sdk_with_prompt(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("skill", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    result = await run_one_attempt(
        user_prompt="my prompt",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )

    assert isinstance(result, AttemptResult)
    assert result.sdk_error == ""
    assert len(fake_sdk.received_prompts) == 1
    assert "my prompt" in fake_sdk.received_prompts[0]
    assert str(skill) in fake_sdk.received_prompts[0]


@pytest.mark.asyncio
async def test_run_one_attempt_sets_cwd_to_quark_root(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    await run_one_attempt(
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )
    options = fake_sdk.received_options[0]
    assert options.kwargs.get("cwd") == str(qr)
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
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )

    compat_dir = tmp_path / "ws" / ".hyperloom_quark_py310_compat"
    sitecustomize = compat_dir / "sitecustomize.py"
    env = fake_sdk.received_options[0].kwargs["env"]
    assert sitecustomize.is_file()
    assert "typing.Self" in sitecustomize.read_text(encoding="utf-8")
    assert env["PYTHONPATH"] == f"{compat_dir}{os.pathsep}/existing/pythonpath"
    assert env["PIP_IGNORE_REQUIRES_PYTHON"] == "1"
    assert os.environ["PYTHONPATH"] == "/existing/pythonpath"
    assert "PIP_IGNORE_REQUIRES_PYTHON" not in os.environ


@pytest.mark.asyncio
async def test_run_one_attempt_captures_sdk_exception(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    fake_sdk.side_effect = RuntimeError("boom from SDK")
    result = await run_one_attempt(
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )
    assert "RuntimeError" in result.sdk_error
    assert "boom from SDK" in result.sdk_error
    # Workspace dir should still be created.
    assert (tmp_path / "ws").is_dir()


@pytest.mark.asyncio
async def test_run_one_attempt_aggregates_chunks(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()
    fake_sdk.scripted_chunks = ["part one", "part two", "part three"]

    result = await run_one_attempt(
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )
    assert result.chunks == ["part one", "part two", "part three"]
    assert "part one\npart two\npart three" == result.raw_text


@pytest.mark.asyncio
async def test_run_one_attempt_skill_missing_raises(tmp_path, fake_sdk, fake_options_cls):
    qr = tmp_path / "qr"
    qr.mkdir()
    with pytest.raises(FileNotFoundError):
        await run_one_attempt(
            user_prompt="x",
            workspace=tmp_path / "ws",
            quark_root=qr,
            skill_path=tmp_path / "nope.md",
            sdk_query_factory=fake_sdk,
            sdk_options_cls=fake_options_cls,
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
        user_prompt="x",
        workspace=ws,
        quark_root=qr,
        attempt_number=2,
        previous_outcome="exec_oom",
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
    )
    prompt = fake_sdk.received_prompts[0]
    assert "exec_oom" in prompt
    assert "fix_hypothesis_attempt_2.md" in prompt
    assert str(hyp) in prompt


@pytest.mark.asyncio
async def test_run_one_attempt_falls_back_when_cwd_unsupported(tmp_path, fake_sdk):
    """Older SDK builds without `cwd` kwarg should retry without it."""

    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    class _StrictOptions:
        def __init__(self, **kwargs):
            if "cwd" in kwargs:
                raise TypeError("cwd unsupported")
            self.kwargs = kwargs

    result = await run_one_attempt(
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=_StrictOptions,
    )
    assert result.sdk_error == ""
    # cwd must have been stripped from the retry.
    assert "cwd" not in fake_sdk.received_options[0].kwargs


@pytest.mark.asyncio
async def test_run_one_attempt_passes_model(tmp_path, fake_sdk, fake_options_cls):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    qr = tmp_path / "qr"
    qr.mkdir()

    await run_one_attempt(
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        model="custom-model-id",
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
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
        user_prompt="x",
        workspace=tmp_path / "ws",
        quark_root=qr,
        skill_path=skill,
        sdk_query_factory=fake_sdk,
        sdk_options_cls=fake_options_cls,
        log=captured.append,
    )
    assert any("alpha" in line for line in captured)
    assert any("beta" in line for line in captured)


@pytest.mark.asyncio
async def test_codex_and_hermes_receive_the_same_original_skill_prompt(tmp_path, monkeypatch):
    """Provider selection changes transport only, never the Quark workflow prompt."""

    from hyperloom.agents.quantization.driver import runner

    skill = tmp_path / "SKILL.md"
    skill.write_text("ORIGINAL_QUARK_SKILL", encoding="utf-8")
    quark_root = tmp_path / "Quark"
    quark_root.mkdir()
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-oauth"
    codex_home.mkdir()
    calls: list[tuple[list[str], dict]] = []

    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/runtime/{name}")
    monkeypatch.setattr(runner, "resolve_hermes_executable", lambda: "/runtime/hermes")
    monkeypatch.setattr(runner, "hermes_external_sandbox_enabled", lambda _env: True)
    monkeypatch.setenv("HYPERLOOM_HERMES_PROFILE", "hyperloomfaithful")
    monkeypatch.setenv("HYPERLOOM_HERMES_PROVIDER", "openai-codex")
    monkeypatch.setenv("HYPERLOOM_HERMES_EXTERNAL_SANDBOX", "1")
    monkeypatch.setenv("HYPERLOOM_CODEX_HOME", str(codex_home))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-native-codex")

    def _fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="provider complete", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    codex_result = await run_one_attempt(
        user_prompt="Quantize SOURCE_MODEL with the original Quark skills",
        workspace=workspace,
        quark_root=quark_root,
        skill_path=skill,
        model="provider-model",
        provider="codex",
    )
    hermes_result = await run_one_attempt(
        user_prompt="Quantize SOURCE_MODEL with the original Quark skills",
        workspace=workspace,
        quark_root=quark_root,
        skill_path=skill,
        model="provider-model",
        provider="hermes",
    )

    assert codex_result.sdk_error == hermes_result.sdk_error == ""
    assert len(calls) == 2
    codex_argv, codex_kwargs = calls[0]
    hermes_argv, hermes_kwargs = calls[1]
    codex_prompt = codex_kwargs["input"]
    hermes_prompt = hermes_argv[-1]
    assert codex_prompt == hermes_prompt
    for marker in (
        str(skill),
        str(workspace),
        str(quark_root),
        "Quantize SOURCE_MODEL with the original Quark skills",
    ):
        assert marker in codex_prompt
    assert codex_kwargs["cwd"] == hermes_kwargs["cwd"] == workspace
    assert codex_kwargs["env"]["CODEX_HOME"] == str(codex_home.resolve())
    assert "OPENAI_BASE_URL" not in codex_kwargs["env"]
    assert "OPENAI_API_KEY" not in codex_kwargs["env"]
    assert codex_argv[:7] == [
        "/runtime/codex",
        "exec",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--ignore-rules",
        "-m",
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex_argv
    assert codex_argv[-2:] == ["-m", "provider-model"]
    assert hermes_argv[:7] == [
        "/runtime/hermes",
        "--profile",
        "hyperloomfaithful",
        "--provider",
        "openai-codex",
        "--model",
        "provider-model",
    ]
    assert "--safe-mode" in hermes_argv
    assert hermes_argv[hermes_argv.index("--toolsets") + 1] == "terminal,file"
    assert "--yolo" not in hermes_argv
    assert hermes_argv[-2] == "-z"


@pytest.mark.asyncio
async def test_cli_attempt_redacts_provider_diagnostics(tmp_path, monkeypatch):
    from hyperloom.agents.quantization.driver import runner

    skill = tmp_path / "SKILL.md"
    skill.write_text("ORIGINAL_QUARK_SKILL", encoding="utf-8")
    quark_root = tmp_path / "Quark"
    quark_root.mkdir()
    workspace = tmp_path / "workspace"
    captured: list[str] = []
    monkeypatch.setattr(runner, "resolve_hermes_executable", lambda: "/runtime/hermes")
    monkeypatch.setattr(runner, "hermes_external_sandbox_enabled", lambda _env: True)
    monkeypatch.setenv("PRIVATE_TOKEN", "top-secret-value")
    monkeypatch.setenv("HYPERLOOM_HERMES_EXTERNAL_SANDBOX", "1")

    def _fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="Authorization: Bearer top-secret-value API_KEY=also-secret",
        )

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    result = await run_one_attempt(
        user_prompt="Quantize SOURCE_MODEL",
        workspace=workspace,
        quark_root=quark_root,
        skill_path=skill,
        provider="hermes",
        log=captured.append,
    )

    combined = "\n".join([*captured, result.sdk_error])
    assert "top-secret-value" not in combined
    assert "also-secret" not in combined
    assert "[REDACTED]" in combined


@pytest.mark.asyncio
async def test_run_one_attempt_rejects_unknown_provider(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("x", encoding="utf-8")
    quark_root = tmp_path / "Quark"
    quark_root.mkdir()

    with pytest.raises(ValueError, match="unsupported quantization-agent provider"):
        await run_one_attempt(
            user_prompt="x",
            workspace=tmp_path / "workspace",
            quark_root=quark_root,
            skill_path=skill,
            provider="invented-provider",
        )
