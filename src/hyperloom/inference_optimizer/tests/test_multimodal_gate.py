"""Tests for the unsupported-model preflight gate (whitelist approach)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import model_gate as cli


def _write_config(model_dir: Path, payload) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_dir / "config.json"
    if isinstance(payload, str):
        cfg.write_text(payload, encoding="utf-8")
    else:
        cfg.write_text(json.dumps(payload), encoding="utf-8")


def _args(model: str, *, allow_mm_text_fallback: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        allow_mm_text_fallback=allow_mm_text_fallback,
    )


def _seed_state(session_dir: Path, monkeypatch) -> None:
    """Create a minimal seeded session so the preflight can load/save state."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    from hyperloom.orchestrator.state.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


# 1. classifier — whitelist-based detection
def test_detect_unknown_arch_rejected(tmp_path):
    """An unknown architecture not matching text-generation markers -> rejected."""
    m = tmp_path / "custom"
    _write_config(m, {"architectures": ["SomeCustomArch"], "model_type": "qwen2_vl"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["architecture"] == "SomeCustomArch"


def test_detect_vision_encoder_rejected(tmp_path):
    """CLIPVisionModel is not a text-generation arch -> rejected."""
    m = tmp_path / "clip"
    _write_config(m, {"architectures": ["CLIPVisionModel"], "model_type": "clip"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None


def test_detect_seq2seq_rejected(tmp_path):
    """T5ForConditionalGeneration is not ForCausalLM -> rejected."""
    m = tmp_path / "t5"
    _write_config(m, {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None


def test_detect_unknown_model_type_no_arch_rejected(tmp_path):
    """No architectures field + unknown model_type -> rejected."""
    m = tmp_path / "weird"
    _write_config(m, {"model_type": "some_exotic_type"})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "allowlist" in hit["signal"]


def test_detect_empty_config_rejected(tmp_path):
    """A readable config without identity tags cannot prove text generation."""
    m = tmp_path / "empty"
    _write_config(m, {})
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert "neither architectures nor model_type" in hit["signal"]


@pytest.mark.parametrize(
    ("arch", "model_type"),
    [
        ("MistralForCausalLM", "llama"),
        ("Qwen2ForCausalLM", "llama"),
        ("LlamaForCausalLM", "llama"),
        ("DeepseekV3ForCausalLMNextN", "llama"),
        ("LlamaForCausalLMEagle3", "llama"),
        ("GPT2LMHeadModel", "gpt2"),
    ],
)
def test_detect_causal_lm_allowed(tmp_path, arch, model_type):
    """ForCausalLM architectures, their suffixed variants and GPT2LMHeadModel-style ones pass through."""
    m = tmp_path / "text"
    _write_config(m, {"architectures": [arch], "model_type": model_type})
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_known_model_type_no_arch_allowed(tmp_path):
    """No architectures field but known model_type -> allowed."""
    m = tmp_path / "known_type"
    _write_config(m, {"model_type": "mistral"})
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_known_text_model_type_with_nonstandard_arch_allowed(tmp_path):
    """Known text-generation model_type can allow non-ForCausalLM class names."""
    m = tmp_path / "chatglm"
    _write_config(
        m,
        {
            "architectures": ["ChatGLMForConditionalGeneration"],
            "model_type": "chatglm",
        },
    )
    assert cli._detect_unsupported_model(str(m)) is None


def test_detect_causal_lm_with_vision_config_is_text_coercible(tmp_path):
    """A generic ForCausalLM arch carrying vision_config is text-coercible: a text decoder exists, so we degrade to the
    text path (not fail-fast).
    """
    m = tmp_path / "visioncausal"
    _write_config(
        m,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vision_config": {"hidden_size": 1024},
        },
    )
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["verdict"] == cli._VERDICT_TEXT_COERCIBLE
    assert "vision_config" in hit["signal"]


@pytest.mark.parametrize(
    ("arch", "model_type"),
    [
        ("KimiK25ForConditionalGeneration", "kimi_k25"),
        ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
    ],
)
def test_detect_vision_config_text_moe_is_text_coercible(tmp_path, arch, model_type):
    """Kimi-K2.6 and Qwen3.6 MoE carry vision_config but their text MoE path benchmarks fine -> text_coercible (degrade
    with warning), not fail-fast.
    """
    m = tmp_path / model_type
    _write_config(
        m,
        {
            "architectures": [arch],
            "model_type": model_type,
            "vision_config": {"hidden_size": 1024},
        },
    )
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["verdict"] == cli._VERDICT_TEXT_COERCIBLE


def test_detect_gemma4_wrapper_text_coercible(tmp_path):
    """Gemma4 multimodal wrappers route to the text-coercible degraded path."""
    m = tmp_path / "gemma4"
    _write_config(
        m,
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "vision_config": {"hidden_size": 1024},
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 262144,
                "hidden_size": 4096,
                "num_hidden_layers": 48,
            },
        },
    )
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["verdict"] == cli._VERDICT_TEXT_COERCIBLE


def test_detect_mislabeled_vlm_with_vision_config_is_vision_only(tmp_path):
    """A multimodal config whose model_type is merely in the text allowlist (e.g. a real VLM mislabeled model_type='qwen2') but with NO confirmed text-generation architecture must fail-fast (vision_only), not degrade."""
    m = tmp_path / "mislabeled"
    _write_config(
        m,
        {
            "architectures": ["SomeVisionForConditionalGeneration"],
            "model_type": "qwen2",
            "vision_config": {"hidden_size": 1024},
        },
    )
    hit = cli._detect_unsupported_model(str(m))
    assert hit is not None
    assert hit["verdict"] == cli._VERDICT_VISION_ONLY


def test_detect_missing_config_returns_none(tmp_path):
    assert cli._detect_unsupported_model(str(tmp_path / "nope")) is None


def test_detect_invalid_config_returns_none(tmp_path):
    m = tmp_path / "bad"
    _write_config(m, "{not valid json")
    assert cli._detect_unsupported_model(str(m)) is None


# 2. preflight gate — persists a canonical stop reason and signals exit
def test_preflight_blocks_gemma3(tmp_path, monkeypatch):
    model = tmp_path / "gemma3"
    _write_config(
        model,
        {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
        },
    )
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(_args(str(model)), sd)

    assert blocked is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "unsupported_model_arch"
    detail = final["stop_detail"]
    assert "Gemma3ForConditionalGeneration" in detail
    assert "text-generation" in detail.lower()
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"
    breakdown = json.loads((sd / "session_breakdown.json").read_text())
    assert breakdown["outcome"]["stop_reason"] == "unsupported_model_arch"


def test_preflight_blocks_unknown_arch(tmp_path, monkeypatch):
    model = tmp_path / "custom"
    _write_config(model, {"architectures": ["X"], "model_type": "qwen2_vl"})
    sd = tmp_path / "session_custom"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "unsupported_model_arch"


def test_preflight_allows_plain_text(tmp_path, monkeypatch):
    model = tmp_path / "mistral"
    _write_config(
        model,
        {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
        },
    )
    sd = tmp_path / "session_text"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_allows_missing_config(tmp_path, monkeypatch):
    model = tmp_path / "no_config"
    model.mkdir()
    sd = tmp_path / "session_missing"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


# 3. stop_reason vocabulary registration
def test_preflight_persists_stop_reason_under_strict_env(tmp_path, monkeypatch):
    """Under strict mode the preflight must still persist the stop_reason."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "gemma3strict"
    _write_config(
        model,
        {
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
        },
    )
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)
    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"


# 4. text_coercible — multimodal signal but a text path exists
def _coercible_model(tmp_path: Path) -> Path:
    m = tmp_path / "kimi_k25"
    _write_config(
        m,
        {
            "architectures": ["KimiK25ForConditionalGeneration"],
            "model_type": "kimi_k25",
            "vision_config": {"hidden_size": 1024},
        },
    )
    return m


def test_preflight_text_coercible_fallback_on_proceeds(tmp_path, monkeypatch):
    """Default --allow-mm-text-fallback: a text-coercible model proceeds (no fail-fast), records degraded_mode + a model warning, writes no final.json."""
    model = _coercible_model(tmp_path)
    sd = tmp_path / "session_coerce_on"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(
        _args(str(model), allow_mm_text_fallback=True),
        sd,
    )

    assert blocked is False
    # No fail-fast report written.
    assert not (sd / "reports" / "final.json").exists()
    # Degraded marker persisted to state.json.
    state = json.loads((sd / "state.json").read_text())
    assert state["degraded_mode"] is True
    assert state.get("stop_reason", "") in ("", None)
    warnings = state["model_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["kind"] == "multimodal_text_fallback"
    assert "kimi_k25" in warnings[0]["model_type"]


def test_preflight_text_coercible_fallback_off_fails_fast(tmp_path, monkeypatch):
    """--no-allow-mm-text-fallback turns a text-coercible model back into a fail-fast (stop_reason=unsupported_model_arch)."""
    model = _coercible_model(tmp_path)
    sd = tmp_path / "session_coerce_off"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(
        _args(str(model), allow_mm_text_fallback=False),
        sd,
    )

    assert blocked is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"


def test_preflight_vision_only_ignores_fallback_flag(tmp_path, monkeypatch):
    """A true VLM (vision_only) fail-fasts even with the fallback flag on."""
    model = tmp_path / "llava"
    _write_config(
        model,
        {
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        },
    )
    sd = tmp_path / "session_vision_only"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(
        _args(str(model), allow_mm_text_fallback=True),
        sd,
    )

    assert blocked is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"


# 4b. scriptable diffusion (xDiT) frameworks skip this text-LM gate entirely
def _xdit_args(model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        allow_mm_text_fallback=True,
        framework="xdit",
    )


def test_preflight_scriptable_xdit_skips_gate(tmp_path, monkeypatch):
    """A scriptable (xDiT) session must bypass the causal-LM gate even when its root config.json looks nothing like a text-generation model — diffusion checkpoints legitimately lack architectures/model_type at the root."""
    model = tmp_path / "hunyuan_image"
    _write_config(model, {"_class_name": "HunyuanDiTPipeline"})
    sd = tmp_path / "session_xdit"
    _seed_state(sd, monkeypatch)

    blocked = cli._preflight_unsupported_model_arch(_xdit_args(str(model)), sd)

    assert blocked is False
    assert not (sd / "reports" / "final.json").exists()
    state = json.loads((sd / "state.json").read_text())
    assert state.get("stop_reason", "") in ("", None)


def test_preflight_scriptable_xdit_skips_even_vlm_config(tmp_path, monkeypatch):
    """The scriptable skip is framework-driven, so it fires before detection — even a config that would otherwise read as vision_only must not block xDiT."""
    model = tmp_path / "diffusion_vlm_like"
    _write_config(
        model,
        {
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        },
    )
    sd = tmp_path / "session_xdit_vlm"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_unsupported_model_arch(_xdit_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_serving_framework_still_blocks_vlm(tmp_path, monkeypatch):
    """Regression guard: an explicit serving framework (sglang) must still run the gate and block a true VLM — the skip is scoped to scriptable only."""
    model = tmp_path / "llava_sglang"
    _write_config(
        model,
        {
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        },
    )
    sd = tmp_path / "session_sglang_vlm"
    _seed_state(sd, monkeypatch)

    args = argparse.Namespace(
        model=str(model),
        allow_mm_text_fallback=True,
        framework="sglang",
    )
    assert cli._preflight_unsupported_model_arch(args, sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "unsupported_model_arch"


# 5. report rendering of the degraded-mode section
def test_report_renders_degraded_mode_section():
    from hyperloom.orchestrator.actions.executors import report
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="s", model_name="kimi", model_path="/m/kimi")
    state.degraded_mode = True
    state.model_warnings = [
        {
            "kind": "multimodal_text_fallback",
            "model_name": "kimi",
            "architecture": "KimiK25ForConditionalGeneration",
            "model_type": "kimi_k25",
            "signal": "multimodal config key 'vision_config'",
            "detail": "DEGRADED MODE: ...",
        }
    ]
    summary = report._build_summary_dict(state, {}, [], external_baseline=None)
    assert summary["degraded_mode"] is True
    assert len(summary["model_warnings"]) == 1

    md = report._format_md(summary)
    assert "Degraded mode" in md
    assert "text path only" in md
    assert "KimiK25ForConditionalGeneration" in md


def test_report_no_degraded_section_when_clean():
    from hyperloom.orchestrator.actions.executors import report
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="s", model_name="llama", model_path="/m/llama")
    summary = report._build_summary_dict(state, {}, [], external_baseline=None)
    assert summary["degraded_mode"] is False
    md = report._format_md(summary)
    assert "## Degraded mode" not in md
