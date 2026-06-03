"""Tests for the --quantize quantization prelude.

Three groups, all offline (no Quark / Claude SDK / GPU):

* Parser     — ``--quantize`` flag parses (default None / value when passed).
* Adapter    — ``run_quantization_prelude_async`` maps QuantSkillRunResult
               status -> decision (return dir vs SystemExit(3)).
* CLI hook   — ``cli._run_quantization_prelude`` is a no-op without the flag,
               skipped on --resume, and rewrites args.model otherwise.

``quantize_via_prompt`` is monkeypatched so nothing real runs.
"""

from __future__ import annotations

import asyncio
import os
import types
from pathlib import Path

import pytest

from inference_optimizer import cli
from inference_optimizer.orchestrator import quantization_request_handlers as qrh
from inference_optimizer.orchestrator import quantization_schemes as qs


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

def _fake_result(status: str, qdir: str | None, *, final="x", eval_gap=None):
    """Build a stand-in for QuantSkillRunResult."""
    assessment = types.SimpleNamespace(final=final, eval_gap=eval_gap)
    return types.SimpleNamespace(
        status=status,
        quantized_model_dir=Path(qdir) if qdir else None,
        assessment=assessment,
    )


def _patch_quantize(monkeypatch: pytest.MonkeyPatch, result):
    """Replace quantization_agent.quantize_via_prompt with an async stub
    that records its call and returns ``result``."""
    import quantization_agent

    calls: list[dict] = []

    async def _fake(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return result

    monkeypatch.setattr(quantization_agent, "quantize_via_prompt", _fake)
    return calls


# ---------------------------------------------------------------------------
# Group A — parser
# ---------------------------------------------------------------------------

def _parse(argv: list[str]):
    return cli._build_parser().parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_quantize_flag_defaults_none():
    args = _parse([])
    assert getattr(args, "quantize") is None


def test_quantize_flag_captures_prompt():
    args = _parse(["--quantize", "fp8, exclude lm_head"])
    assert args.quantize == "fp8, exclude lm_head"


def test_quantize_scheme_flag_parses():
    args = _parse(["--quantize-scheme", "fp8"])
    assert args.quantize_scheme == "fp8"


def test_quantize_scheme_rejects_unknown_choice():
    with pytest.raises(SystemExit):  # argparse rejects invalid choice
        _parse(["--quantize-scheme", "fp16_made_up"])


# ---------------------------------------------------------------------------
# Group A2 — scheme registry
# ---------------------------------------------------------------------------

def test_resolve_scheme_none_returns_none():
    assert qs.resolve_scheme_prompt(None) is None
    assert qs.resolve_scheme_prompt("none") is None


def test_resolve_scheme_known_returns_prompt():
    p = qs.resolve_scheme_prompt("fp8")
    assert p and "fp8" in p and "lm_head" in p


def test_resolve_scheme_unknown_raises():
    with pytest.raises(ValueError):
        qs.resolve_scheme_prompt("bogus")


# ---------------------------------------------------------------------------
# Group B — adapter status mapping
# ---------------------------------------------------------------------------

def test_adapter_success_returns_quantized_dir(tmp_path, monkeypatch):
    calls = _patch_quantize(
        monkeypatch, _fake_result("success", str(tmp_path / "q"), final=None, eval_gap=0.01)
    )
    out = asyncio.run(
        qrh.run_quantization_prelude_async(
            prompt="fp8", source_model="/models/src", workspace=tmp_path
        )
    )
    assert out == str(tmp_path / "q")
    # source model + export dir folded into the effective prompt
    assert "/models/src" in calls[0]["prompt"]
    assert str(tmp_path / "quantized") in calls[0]["prompt"]
    assert calls[0]["interactive"] is False


def test_adapter_partial_with_model_returns_dir(tmp_path, monkeypatch):
    _patch_quantize(
        monkeypatch, _fake_result("partial", str(tmp_path / "q"), final="eval_gap_exceeded")
    )
    out = asyncio.run(
        qrh.run_quantization_prelude_async(
            prompt="fp8", source_model="/m", workspace=tmp_path
        )
    )
    assert out == str(tmp_path / "q")


def test_adapter_partial_without_model_exits_3(tmp_path, monkeypatch):
    _patch_quantize(monkeypatch, _fake_result("partial", None, final="must_validate_skipped"))
    with pytest.raises(SystemExit) as ei:
        asyncio.run(
            qrh.run_quantization_prelude_async(
                prompt="fp8", source_model="/m", workspace=tmp_path
            )
        )
    assert ei.value.code == 3


def test_adapter_failed_exits_3(tmp_path, monkeypatch):
    _patch_quantize(monkeypatch, _fake_result("failed", None, final="exec_model_load_failed"))
    with pytest.raises(SystemExit) as ei:
        asyncio.run(
            qrh.run_quantization_prelude_async(
                prompt="fp8", source_model="/m", workspace=tmp_path
            )
        )
    assert ei.value.code == 3


# ---------------------------------------------------------------------------
# Group C — cli prelude hook
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, *, model, quantize=None, quantize_scheme=None, resume=False):
        self.model = Path(model)
        self.quantize = quantize
        self.quantize_scheme = quantize_scheme
        self.resume = resume


def test_prelude_noop_without_flag(monkeypatch):
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize=None)
    asyncio.run(cli._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged


def test_prelude_skipped_on_resume(monkeypatch):
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize="fp8", resume=True)
    asyncio.run(cli._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged


def test_prelude_rewrites_model_on_success(tmp_path, monkeypatch):
    import inference_optimizer.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)

    async def _fake_async(*, prompt, source_model, workspace):
        return str(tmp_path / "out" / "quantized")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    args = _Args(model="/models/src", quantize="fp8")
    asyncio.run(cli._run_quantization_prelude(args))

    assert str(args.model) == str(tmp_path / "out" / "quantized")
    assert os.environ["MODEL_PATH"] == str(tmp_path / "out" / "quantized")


def test_prelude_noop_when_scheme_none(monkeypatch):
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize=None, quantize_scheme="none")
    asyncio.run(cli._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"


def test_prelude_uses_scheme_enum_when_no_freetext(tmp_path, monkeypatch):
    import inference_optimizer.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(model="/models/src", quantize=None, quantize_scheme="fp8")
    asyncio.run(cli._run_quantization_prelude(args))
    # the fp8 enum resolved to its curated prompt
    assert "fp8" in seen["prompt"]
    assert str(args.model) == str(tmp_path / "q")


def test_prelude_freetext_takes_priority_over_scheme(tmp_path, monkeypatch):
    import inference_optimizer.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(
        model="/models/src", quantize="custom mxfp4 prompt", quantize_scheme="fp8"
    )
    asyncio.run(cli._run_quantization_prelude(args))
    assert seen["prompt"] == "custom mxfp4 prompt"  # free text wins
