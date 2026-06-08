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
    assert p and "fp8" in p


def test_resolve_scheme_has_no_hardcoded_defaults():
    # Issue #453 §4: the agent must not bake in kv_cache / exclude_layers
    # defaults — Quark's intake + plan skill supplies those.
    p = qs.resolve_scheme_prompt("fp8")
    assert "lm_head" not in p
    assert "kv_cache" not in p
    assert "Calibration" not in p
    assert "Evaluation" not in p


def test_resolve_scheme_unknown_raises():
    with pytest.raises(ValueError):
        qs.resolve_scheme_prompt("bogus")


def test_scheme_choices_match_supported_set():
    assert qs.QUANT_SCHEME_CHOICES == [
        "none", "fp8", "ptpc_fp8", "mxfp4", "mxfp4_fp8"
    ]
    # int8 / int4 dropped per issue #453 §4.
    assert "int8" not in qs.QUANT_SCHEME_CHOICES


# ---------------------------------------------------------------------------
# Group A3 — GPU-constrained schemes
# ---------------------------------------------------------------------------

def test_supported_schemes_mi355x_includes_mxfp4():
    s = qs.supported_schemes("mi355x")
    assert "mxfp4" in s and "mxfp4_fp8" in s and "fp8" in s and "ptpc_fp8" in s


def test_supported_schemes_dcgpu_excludes_mxfp4():
    for gpu in ("mi300x", "mi325x", "", None):
        s = qs.supported_schemes(gpu)
        assert "fp8" in s and "ptpc_fp8" in s
        assert "mxfp4" not in s and "mxfp4_fp8" not in s


def test_validate_scheme_mxfp4_on_mi300x_raises():
    with pytest.raises(qs.SchemeNotSupportedError):
        qs.validate_scheme("mxfp4", "mi300x")


def test_validate_scheme_mxfp4_on_mi355x_ok():
    qs.validate_scheme("mxfp4", "mi355x")  # no raise


def test_validate_scheme_mxfp4_unknown_gpu_not_enforced():
    # GPU resolved later via probe; can't enforce without a concrete target.
    qs.validate_scheme("mxfp4", "")
    qs.validate_scheme("mxfp4", None)


def test_validate_scheme_fp8_anywhere_ok():
    for gpu in ("mi300x", "mi325x", "mi355x", "", None):
        qs.validate_scheme("fp8", gpu)


def test_validate_scheme_none_is_noop():
    qs.validate_scheme(None, "mi300x")
    qs.validate_scheme("none", "mi300x")


def test_validate_scheme_unknown_raises_valueerror():
    with pytest.raises(ValueError):
        qs.validate_scheme("bogus", "mi355x")


# ---------------------------------------------------------------------------
# Group A4 — build_quantization_prompt (no hard-coded defaults; three groups)
# ---------------------------------------------------------------------------

def test_build_prompt_minimal_only_strategy():
    cfg = qs.QuantizationConfig(global_scheme="fp8")
    p = qs.build_quantization_prompt(cfg)
    assert "Apply fp8 as the global quantization scheme." in p
    # Unset optional fields are omitted (no Quark-default literals baked in).
    assert "Calibration" not in p
    assert "Evaluation" not in p
    assert "kv_cache" not in p
    assert "pileval" not in p


def test_build_prompt_full_renders_three_groups():
    cfg = qs.QuantizationConfig(
        global_scheme="mxfp4",
        output_dir="/data/quantized/qwen3_32b_mxfp4",
        layer_overrides={"self_attn": "fp8", "moe/mlp": "ptpc_fp8"},
        kv_cache="fp8",
        exclude_layers=["model.layers.0.mlp.down_proj"],
        calib_dataset="pileval",
        num_calib_data=512,
        seq_len=2048,
        acceptable_eval_gap=0.03,
    )
    p = qs.build_quantization_prompt(
        cfg, model_path="/wekafs/models/Qwen-Qwen3-32B", gpu_type="mi355x",
        skill_path="@/wekafs/HyperloomV2/quantization_agent/SKILL.md",
    )
    assert "Use the skill at @/wekafs/HyperloomV2/quantization_agent/SKILL.md" in p
    assert "on an MI355X target" in p
    assert "Quantization strategy:" in p
    assert "Apply mxfp4 as the global quantization scheme." in p
    assert "Override the self_attn layers with fp8 and the moe/mlp layers with ptpc_fp8." in p
    assert "Quantize the kv_cache with fp8." in p
    assert "exclude model.layers.0.mlp.down_proj from quantization" in p
    assert "Write the quantized model to /data/quantized/qwen3_32b_mxfp4." in p
    assert "Calibration:" in p
    assert "Calibrate with the pileval dataset using 512 samples at a sequence length of 2048." in p
    assert "Evaluation:" in p
    assert "Keep the quantized model's accuracy within 3% of the bf16 baseline." in p


def test_build_prompt_eval_gap_formats_percent():
    p = qs.build_quantization_prompt(
        qs.QuantizationConfig(global_scheme="fp8", acceptable_eval_gap=0.05)
    )
    assert "within 5% of the bf16 baseline" in p


def test_build_prompt_partial_calibration():
    # Only num_calib_data set -> calibration group renders just that part.
    p = qs.build_quantization_prompt(
        qs.QuantizationConfig(global_scheme="fp8", num_calib_data=256)
    )
    assert "Calibration:" in p
    assert "256 samples" in p
    assert "pileval" not in p


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
    def __init__(
        self, *, model, quantize=None, quantize_scheme=None, resume=False, gpu_type=None
    ):
        self.model = Path(model)
        self.quantize = quantize
        self.quantize_scheme = quantize_scheme
        self.resume = resume
        self.gpu_type = gpu_type


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


def test_prelude_skips_on_gpu_scheme_mismatch(tmp_path, monkeypatch, capsys):
    # mxfp4 on mi300x is unsupported: report the error, skip quantization,
    # and continue (model unchanged) rather than aborting the run.
    called = {"n": 0}

    async def _should_not_run(**kwargs):  # pragma: no cover - asserts non-call
        called["n"] += 1
        return "x"

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _should_not_run)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    args = _Args(model="/models/src", quantize_scheme="mxfp4", gpu_type="mi300x")
    asyncio.run(cli._run_quantization_prelude(args))
    assert called["n"] == 0
    assert str(args.model) == "/models/src"  # unchanged -> downstream un-quantized
    err = capsys.readouterr().err
    assert "MI355X" in err and "skipped" in err.lower()


def test_prelude_runs_mxfp4_on_mi355x(tmp_path, monkeypatch):
    import inference_optimizer.paths as paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    seen = {}

    async def _fake_async(*, prompt, source_model, workspace):
        seen["prompt"] = prompt
        return str(tmp_path / "q")

    monkeypatch.setattr(qrh, "run_quantization_prelude_async", _fake_async)
    args = _Args(model="/models/src", quantize_scheme="mxfp4", gpu_type="mi355x")
    asyncio.run(cli._run_quantization_prelude(args))
    assert "mxfp4" in seen["prompt"]
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
