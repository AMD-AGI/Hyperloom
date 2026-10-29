"""Branch coverage for hyperloom.agents.quantization.driver.runner / result_collector."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.agents.quantization.driver import result_collector as rc
from hyperloom.agents.quantization.driver import runner


# --------------------------------------------------------------------------- # runner._iter_message_text #
# --------------------------------------------------------------------------- #
def test_iter_message_text_shapes() -> None:
    obj_block = SimpleNamespace(text="from-object")
    dict_block = {"text": "from-dict"}
    msg = SimpleNamespace(content=[obj_block, dict_block, SimpleNamespace(text="")], result="final-result")
    out = list(runner._iter_message_text(msg))
    assert out == ["from-object", "from-dict", "final-result"]


async def test_run_one_attempt_captures_sdk_error(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("contract", encoding="utf-8")
    logs: list[str] = []

    class _Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _factory(*, prompt, options):
        async def _gen():
            raise RuntimeError("sdk blew up")
            yield  # pragma: no cover - unreachable, makes this an async generator

        return _gen()

    result = await runner.run_one_attempt(
        runner.AttemptContext(user_prompt="quantize", workspace=tmp_path / "ws", quark_root=tmp_path),
        skill_path=skill,
        log=logs.append,
        sdk=runner.ClaudeAgentSdk(query=_factory, options_cls=_Options),
    )
    # SDK error captured, not raised.
    assert "sdk blew up" in result.sdk_error
    assert any("WARNING" in line for line in logs)


def test_run_one_attempt_missing_skill(tmp_path: Path) -> None:
    import asyncio

    with pytest.raises(FileNotFoundError):
        asyncio.run(
            runner.run_one_attempt(
                runner.AttemptContext(user_prompt="x", workspace=tmp_path / "ws", quark_root=tmp_path),
                skill_path=tmp_path / "nope_SKILL.md",
                sdk=runner.ClaudeAgentSdk(query=lambda **k: iter(()), options_cls=dict),
            )
        )


# --------------------------------------------------------------------------- # result_collector branches #
# --------------------------------------------------------------------------- #
def test_read_text_oserror_on_directory(tmp_path: Path) -> None:
    # Reading a directory raises an OSError that is swallowed.
    assert rc._read_text(tmp_path) is None


def test_resolve_quantized_dir_not_mapping(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.yaml").write_text("just-a-scalar", encoding="utf-8")
    path, present, err = rc._resolve_quantized_dir(tmp_path)
    assert path is None
    assert present is True
    assert err == "manifest_not_mapping"


def test_resolve_quantized_dir_missing_manifest(tmp_path: Path) -> None:
    path, present, err = rc._resolve_quantized_dir(tmp_path)
    assert (path, present, err) == (None, False, None)


def test_resolve_quantized_dir_pyyaml_missing(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "run_manifest.yaml").write_text("outputs: {}", encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path, present, err = rc._resolve_quantized_dir(tmp_path)
    assert (path, present, err) == (None, True, "pyyaml_missing")


async def test_an_attempt_runs_on_the_harness_when_the_platform_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from common import llm_config
    from hyperloom.common.deepseek_task import DeepSeekTaskResult

    skill = tmp_path / "SKILL.md"
    skill.write_text("contract", encoding="utf-8")
    seen: dict[str, object] = {}

    async def _drive(request, owed=None):
        seen["request"], seen["owed"] = request, owed
        return DeepSeekTaskResult(text="quantized", error="")

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(runner, "run_deepseek_task", _drive)
    logs: list[str] = []

    result = await runner.run_one_attempt(
        runner.AttemptContext(user_prompt="quantize", workspace=tmp_path / "ws", quark_root=tmp_path, attempt_number=2),
        skill_path=skill,
        model="deepseek-v4-flash",
        log=logs.append,
    )

    request = seen["request"]
    assert request.cwd == tmp_path / "ws" and request.writable
    assert request.child_env is not None
    assert seen["owed"] is not None
    assert result.sdk_error == ""
    assert any("DeepSeek runner" in line and "attempt=2" in line for line in logs)


def test_an_attempt_is_unfinished_until_it_is_blocked_or_has_an_eval_result(tmp_path: Path) -> None:
    """An attempt that stops after one reply must not read as a quantized model whose path is ``None``.

    ``last_phase.txt`` is written when a phase starts, so reaching eval is not
    an end: the attempt still owes the eval report, or the reason it skipped it.
    """
    assert not runner._attempt_finished(tmp_path)

    (tmp_path / "last_phase.txt").write_text("export\n", encoding="utf-8")
    assert not runner._attempt_finished(tmp_path)

    (tmp_path / "last_phase.txt").write_text("eval\n", encoding="utf-8")
    assert not runner._attempt_finished(tmp_path)

    (tmp_path / "eval_skipped.txt").write_text("no eval environment\n", encoding="utf-8")
    assert runner._attempt_finished(tmp_path)


def test_a_blocked_attempt_is_finished(tmp_path: Path) -> None:
    (tmp_path / "blocked.md").write_text("outcome: must_have_weights_missing\n", encoding="utf-8")

    assert runner._attempt_finished(tmp_path)


async def test_an_interactive_attempt_is_not_continued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Its checkpoints wait for an answer a headless session never gets."""
    from common import llm_config
    from hyperloom.common.deepseek_task import DeepSeekTaskResult

    skill = tmp_path / "SKILL.md"
    skill.write_text("contract", encoding="utf-8")
    seen: dict[str, object] = {}

    async def _drive(request, owed=None):
        seen["owed"] = owed
        return DeepSeekTaskResult()

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(runner, "run_deepseek_task", _drive)

    await runner.run_one_attempt(
        runner.AttemptContext(user_prompt="q", workspace=tmp_path / "ws", quark_root=tmp_path, interactive=True),
        skill_path=skill,
    )

    assert seen["owed"] is None
