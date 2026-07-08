"""Branch coverage for quantization_agent.driver.runner / result_collector."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.agents.quantization.driver import result_collector as rc
from hyperloom.agents.quantization.driver import runner


# --------------------------------------------------------------------------- #
# runner._iter_message_text                                                   #
# --------------------------------------------------------------------------- #
def test_iter_message_text_shapes() -> None:
    obj_block = SimpleNamespace(text="from-object")
    dict_block = {"text": "from-dict"}  # lines 82-83
    msg = SimpleNamespace(content=[obj_block, dict_block, SimpleNamespace(text="")], result="final-result")
    out = list(runner._iter_message_text(msg))
    assert out == ["from-object", "from-dict", "final-result"]  # result string -> line 86


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
        user_prompt="quantize",
        workspace=tmp_path / "ws",
        quark_root=tmp_path,
        skill_path=skill,
        sdk_query_factory=_factory,
        sdk_options_cls=_Options,
        log=logs.append,
    )
    # SDK error captured, not raised (line 303 path).
    assert "sdk blew up" in result.sdk_error
    assert any("WARNING" in line for line in logs)


async def test_run_one_attempt_options_typeerror_fallback(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("contract", encoding="utf-8")

    class _Options:
        def __init__(self, **kwargs):
            if "cwd" in kwargs:
                raise TypeError("older sdk has no cwd")
            self.kwargs = kwargs

    async def _empty_gen():
        if False:  # pragma: no cover
            yield None

    def _factory(*, prompt, options):
        return _empty_gen()

    result = await runner.run_one_attempt(
        user_prompt="quantize",
        workspace=tmp_path / "ws",
        quark_root=tmp_path,
        skill_path=skill,
        model="some-model",
        sdk_query_factory=_factory,
        sdk_options_cls=_Options,
    )
    assert result.sdk_error == ""


def test_run_one_attempt_missing_skill(tmp_path: Path) -> None:
    import asyncio

    with pytest.raises(FileNotFoundError):
        asyncio.run(
            runner.run_one_attempt(
                user_prompt="x",
                workspace=tmp_path / "ws",
                quark_root=tmp_path,
                skill_path=tmp_path / "nope_SKILL.md",
                sdk_query_factory=lambda **k: iter(()),
                sdk_options_cls=dict,
            )
        )


# --------------------------------------------------------------------------- #
# result_collector branches                                                   #
# --------------------------------------------------------------------------- #
def test_read_text_oserror_on_directory(tmp_path: Path) -> None:
    # Reading a directory raises an OSError that is swallowed (lines 147-148).
    assert rc._read_text(tmp_path) is None


def test_resolve_quantized_dir_not_mapping(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.yaml").write_text("just-a-scalar", encoding="utf-8")
    path, present, err = rc._resolve_quantized_dir(tmp_path)
    assert path is None
    assert present is True
    assert err == "manifest_not_mapping"  # line 203


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
    assert (path, present, err) == (None, True, "pyyaml_missing")  # lines 194-195


def test_scan_hypothesis_attempts_missing_dir(tmp_path: Path) -> None:
    # Nonexistent workspace -> empty tuple (lines 270-271).
    assert rc._scan_hypothesis_attempts(tmp_path / "does_not_exist") == ()


def test_scan_hypothesis_attempts_sorted(tmp_path: Path) -> None:
    (tmp_path / "fix_hypothesis_attempt_2.md").write_text("x", encoding="utf-8")
    (tmp_path / "fix_hypothesis_attempt_1.md").write_text("x", encoding="utf-8")
    assert rc._scan_hypothesis_attempts(tmp_path) == (1, 2)
