"""Tests for the --quantize quantization prelude."""

from __future__ import annotations

import asyncio
import os
import types
from pathlib import Path

import pytest

from hyperloom_common.env import EnvValueError, env_bool
from hyperloom.inference_optimizer.cli import parser as cli_parser
from hyperloom.inference_optimizer.cli import bootstrap as cli_bootstrap
from hyperloom.inference_optimizer.cli import quantization as cli_quantization
from hyperloom.orchestrator.phases import quantization_request_handlers as qrh
from hyperloom.orchestrator.phases import quantization_schemes as qs
from hyperloom.orchestrator.tests._fixtures import MI325X_CARD, MI355X_CARD, publish_card


@pytest.fixture(autouse=True)
def _enable_quant_by_default(monkeypatch):
    """Default the deterministic master switch ON so CLI-hook tests reach the adapter."""
    monkeypatch.setenv("HYPERLOOM_QUANTIZE_ENABLED", "1")


# fakes


def _fake_result(status: str, qdir: str | None, *, final="x", eval_gap=None, run_dir: str | None = None):
    """Build a stand-in for quantization_agent.QuantSkillRunResult."""
    assessment = types.SimpleNamespace(final=final, eval_gap=eval_gap)
    return types.SimpleNamespace(
        status=status,
        quantized_model_dir=Path(qdir) if qdir else None,
        assessment=assessment,
        run_dir=Path(run_dir) if run_dir else None,
    )


def _patch_quantize(monkeypatch: pytest.MonkeyPatch, result):
    """Replace quantize_via_prompt with an async stub that records its call and returns ``result``."""
    import hyperloom.agents.quantization as quantization_agent

    calls: list[dict] = []

    async def _fake(prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return result

    monkeypatch.setattr(quantization_agent, "quantize_via_prompt", _fake)
    return calls


# Group A — parser


def _parse(argv: list[str]):
    return cli_parser._build_parser().parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_quantize_flag_defaults_none():
    args = _parse([])
    assert getattr(args, "quantize") is None


def test_quantize_flag_captures_prompt():
    args = _parse(["--quantize", "fp8, exclude lm_head"])
    assert args.quantize == "fp8, exclude lm_head"


@pytest.mark.parametrize("scheme", ["none", "fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8"])
def test_quantize_scheme_flag_parses(scheme):
    args = _parse(["--quantize-scheme", scheme])
    assert args.quantize_scheme == scheme


def test_quantize_scheme_rejects_unknown_choice():
    with pytest.raises(SystemExit):  # argparse rejects invalid choice
        _parse(["--quantize-scheme", "fp16_made_up"])


# Group A2 — scheme registry


def test_resolve_scheme_none_returns_none():
    assert qs.resolve_scheme_prompt(None) is None
    assert qs.resolve_scheme_prompt("none") is None


# Group A3 — arch-constrained schemes


def test_an_arch_with_an_fp4_rate_offers_every_scheme():
    """gfx950 is every board of that die, the MI350X as much as the MI355X."""
    assert qs.supported_schemes("gfx950") == list(qs.SUPPORTED_SCHEMES)


@pytest.mark.parametrize("gfx", ["gfx942", "gfx1201", "gfx1100", "gfx1151", "gfx9999"])
def test_an_arch_without_an_fp4_rate_offers_no_fp4_scheme(gfx):
    s = qs.supported_schemes(gfx)
    assert "fp8" in s and "ptpc_fp8" in s
    assert not qs.FP4_SCHEMES & set(s)


@pytest.mark.parametrize("scheme", sorted(qs.FP4_SCHEMES))
def test_validate_scheme_fp4_on_gfx942_raises_naming_the_arch(scheme):
    with pytest.raises(qs.SchemeNotSupportedError, match="gfx942"):
        qs.validate_scheme(scheme, "gfx942")


@pytest.mark.parametrize(
    ("scheme", "gfx"),
    [
        pytest.param("mxfp4", "gfx950", id="mxfp4-on-gfx950"),
        pytest.param("mxfp4_fp8", "gfx950", id="mxfp4_fp8-on-gfx950"),
        pytest.param("fp8", "gfx942", id="fp8-on-gfx942"),
        pytest.param("fp8", "gfx950", id="fp8-on-gfx950"),
        pytest.param("fp8", "", id="fp8-on-empty-arch"),
        pytest.param("fp8", None, id="fp8-on-no-arch"),
        pytest.param(None, "gfx942", id="no-scheme"),
        pytest.param("none", "gfx942", id="none-scheme"),
    ],
)
def test_validate_scheme_accepts(scheme, gfx):
    qs.validate_scheme(scheme, gfx)


@pytest.mark.parametrize("gfx", ["", None])
def test_an_unknown_arch_constrains_nothing(gfx):
    """A session that read no card cannot know the scheme is unsupported, so it does not refuse it."""
    qs.validate_scheme("mxfp4", gfx)
    assert qs.supported_schemes(gfx) == list(qs.SUPPORTED_SCHEMES)


def test_validate_scheme_unknown_raises_valueerror():
    with pytest.raises(ValueError):
        qs.validate_scheme("bogus", "gfx950")


# Group B — adapter status mapping


def test_adapter_success_returns_quantized_dir(tmp_path, monkeypatch):
    from hyperloom.agents.quantization import QuantSkillRunOptions

    calls = _patch_quantize(monkeypatch, _fake_result("success", str(tmp_path / "q"), final=None, eval_gap=0.01))
    out = asyncio.run(qrh.run_quantization_prelude_async(prompt="fp8", source_model="/models/src", workspace=tmp_path))
    assert out == str(tmp_path / "q")
    # source model + export dir folded into the effective prompt; the export dir is inside the run's own directory.
    assert "/models/src" in calls[0]["prompt"]
    assert "`quantized/` in your workspace" in calls[0]["prompt"]
    assert str(tmp_path) not in calls[0]["prompt"]
    assert "fp8" in calls[0]["prompt"]
    assert calls[0]["options"] == QuantSkillRunOptions(interactive=False)
    assert calls[0]["workspace"] == tmp_path


def test_adapter_partial_with_model_returns_dir(tmp_path, monkeypatch):
    _patch_quantize(monkeypatch, _fake_result("partial", str(tmp_path / "q"), final="eval_gap_exceeded"))
    out = asyncio.run(qrh.run_quantization_prelude_async(prompt="fp8", source_model="/m", workspace=tmp_path))
    assert out == str(tmp_path / "q")


def test_adapter_partial_without_model_exits_4(tmp_path, monkeypatch):
    _patch_quantize(monkeypatch, _fake_result("partial", None, final="must_validate_skipped"))
    with pytest.raises(SystemExit) as ei:
        asyncio.run(qrh.run_quantization_prelude_async(prompt="fp8", source_model="/m", workspace=tmp_path))
    assert ei.value.code == 4


def test_adapter_failed_exits_4(tmp_path, monkeypatch, capsys):
    run_dir = str(tmp_path / "runs" / "r1")
    _patch_quantize(monkeypatch, _fake_result("failed", None, final="exec_model_load_failed", run_dir=run_dir))
    with pytest.raises(SystemExit) as ei:
        asyncio.run(qrh.run_quantization_prelude_async(prompt="fp8", source_model="/m", workspace=tmp_path))
    assert ei.value.code == 4
    # The failure names the run directory that holds this run's diagnosis.
    assert run_dir in capsys.readouterr().err


# Group C — cli prelude hook


class _Args:
    def __init__(self, *, model, quantize=None, quantize_scheme=None):
        self.model = Path(model)
        self.quantize = quantize
        self.quantize_scheme = quantize_scheme


def test_prelude_noop_without_flag(monkeypatch):
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize=None)
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged


def test_prelude_rewrites_model_on_success(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.session.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)

    async def _fake_async(*, prompt, source_model, workspace):
        return str(tmp_path / "out" / "quantized")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    args = _Args(model="/models/src", quantize="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))

    assert str(args.model) == str(tmp_path / "out" / "quantized")
    assert os.environ["MODEL_PATH"] == str(tmp_path / "out" / "quantized")


def test_prelude_noop_when_scheme_none(monkeypatch):
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize=None, quantize_scheme="none")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"


def test_prelude_uses_scheme_enum_when_no_freetext(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.session.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(model="/models/src", quantize=None, quantize_scheme="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    # the fp8 enum resolved to its curated prompt.
    assert "fp8" in seen["prompt"]
    assert str(args.model) == str(tmp_path / "q")


def test_prelude_skips_on_gpu_scheme_mismatch(tmp_path, monkeypatch, capsys):
    # mxfp4 on the session's gfx942 card is unsupported: report, skip quantization, continue.
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    # The label is a name; the card decides.
    monkeypatch.setenv("GPU_TYPE", "mi355x")
    publish_card(monkeypatch, MI325X_CARD)
    # Register the key so the marker the prelude writes does not leak into other tests.
    monkeypatch.setenv("HYPERLOOM_QUANTIZATION_SKIPPED", "")
    args = _Args(model="/models/src", quantize_scheme="mxfp4")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged -> downstream un-quantized
    captured = capsys.readouterr()
    # Skip is detectable via stdout marker + env var.
    assert "QUANTIZATION_SKIPPED" in captured.out
    assert "gfx942" in (captured.out + captured.err)
    assert os.environ.get("HYPERLOOM_QUANTIZATION_SKIPPED")


def test_prelude_runs_mxfp4_on_a_gfx950_card(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.session.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    publish_card(monkeypatch, MI355X_CARD)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(model="/models/src", quantize_scheme="mxfp4")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert "mxfp4" in seen["prompt"]
    assert str(args.model) == str(tmp_path / "q")


def test_prelude_freetext_takes_priority_over_scheme(tmp_path, monkeypatch):
    import hyperloom.inference_optimizer.session.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(model="/models/src", quantize="custom mxfp4 prompt", quantize_scheme="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert seen["prompt"] == "custom mxfp4 prompt"  # free text wins


def test_prelude_preserves_source_model_identity(tmp_path, monkeypatch):
    """The prelude rewrites args.model to the generic ``.../quantized`` export dir, but the session / display model
    identity must NOT collapse to ``quantized`` (else runs collide and the report loses the real model name).
    """
    import hyperloom.inference_optimizer.session.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)

    async def _fake_async(*, prompt, source_model, workspace):
        # Mirror the real adapter: export dir basename is always "quantized".
        return str(tmp_path / "quantization" / "google-gemma-4-26B-A4B-it" / "quantized")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    args = _Args(model="/path/models/google-gemma-4-26B-A4B-it", quantize="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))

    # Model path is rewritten to the generic quantized export dir ...
    assert str(args.model).endswith("/quantized")
    # ... but the identity used for session_dir / state / manifest is preserved.
    name = cli_bootstrap.resolve_model_display_name(args)
    assert name != "quantized"
    assert name == "google-gemma-4-26B-A4B-it-quantized"


def test_prelude_no_display_name_without_quantization(monkeypatch):
    """Without quantization the prelude leaves args untouched, so the identity resolver falls back to the plain model-path basename."""
    args = _Args(model="/models/Qwen3-32B", quantize=None)
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert getattr(args, "model_display_name", None) in (None, "")
    assert cli_bootstrap.resolve_model_display_name(args) == "Qwen3-32B"


# Group D — deterministic env switch ($HYPERLOOM_QUANTIZE_ENABLED)


def test_prelude_env_gate_skips_when_disabled(monkeypatch, capsys):
    """With $HYPERLOOM_QUANTIZE_ENABLED off, the prelude skips quantization even when --quantize is present."""
    monkeypatch.setenv("HYPERLOOM_QUANTIZE_ENABLED", "0")
    monkeypatch.setenv("HYPERLOOM_QUANTIZATION_SKIPPED", "")
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))

    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged -> downstream un-quantized
    captured = capsys.readouterr()
    assert "QUANTIZATION_SKIPPED" in captured.out
    assert os.environ.get("HYPERLOOM_QUANTIZATION_SKIPPED")


def test_prelude_env_gate_skips_when_unset(monkeypatch):
    """Unset env => disabled: quantization does not run."""
    monkeypatch.delenv("HYPERLOOM_QUANTIZE_ENABLED", raising=False)
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    args = _Args(model="/models/src", quantize="fp8")
    asyncio.run(cli_quantization._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"


def test_an_unreadable_quantize_switch_is_not_silently_off(monkeypatch):
    """Publishing an unquantized model as quantized is worse than refusing to start."""
    monkeypatch.setenv("HYPERLOOM_QUANTIZE_ENABLED", "bogus")
    with pytest.raises(EnvValueError, match="HYPERLOOM_QUANTIZE_ENABLED"):
        env_bool("HYPERLOOM_QUANTIZE_ENABLED")
