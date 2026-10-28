# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A draft the pinned stack ships no kernel for is refused as unsupported, never spent as a measurement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.speculative_capability import (
    CAPABILITY_UNSUPPORTED,
    SpeculativeCapabilityUnsupportedError,
    refuse_unrunnable_speculative_stack,
    speculative_capability_lines,
    speculative_capability_reason,
    speculative_descriptor,
)
from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import (
    ServingStack,
    apply_compatibility_filter,
)
from hyperloom.orchestrator.prompts.speculative_decoding_focus import speculative_decoding_focus


def _draft_model(tmp_path: Path, *, heads: int = 16, nested: bool = False) -> str:
    """A drafter checkpoint whose config.json declares ``heads`` query heads."""
    model = tmp_path / "Kimi-K3-DSpark"
    model.mkdir(parents=True)
    config = {"text_config": {"num_attention_heads": heads}} if nested else {"num_attention_heads": heads}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model)


def _spec_args(model: str, **overrides) -> str:
    """The launch the framework agent authored, with ``overrides`` applied to its speculative config."""
    config = {
        "model": model,
        "num_speculative_tokens": 2,
        "method": "dspark",
        "attention_backend": "ROCM_AITER_MLA",
        "kv_cache_dtype": "fp8",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block",
        **overrides,
    }
    return "--speculative-config " + json.dumps(config) + " --max-num-seqs 128"


def test_descriptor_merges_the_per_key_flags_over_the_json_object(tmp_path):
    model = _draft_model(tmp_path)
    args = (
        _spec_args(model, attention_backend="TRITON_MLA")
        + " --speculative-config.attention_backend ROCM_AITER_MLA --tensor-parallel-size 8"
    )
    descriptor = speculative_descriptor(args)
    assert descriptor is not None
    assert (descriptor.model, descriptor.method, descriptor.draft_length) == (model, "dspark", 2)
    assert (descriptor.attention_backend, descriptor.kv_cache_dtype) == ("ROCM_AITER_MLA", "fp8")
    assert descriptor.tensor_parallel_size == 8
    # A launch that drafts nothing has nothing to judge.
    assert speculative_descriptor("--max-num-seqs 128 --tp 8") is None
    assert speculative_capability_reason("--max-num-seqs 128 --tp 8", framework="vllm") is None


def test_the_same_config_spelled_per_key_is_refused_too(tmp_path):
    args = (
        f"--speculative-config.model {_draft_model(tmp_path)} --speculative-config.method dspark"
        " --speculative-config.num_speculative_tokens 2 --speculative-config.kv_cache_dtype fp8"
        " --speculative-config.attention_backend ROCM_AITER_MLA"
    )
    assert speculative_capability_reason(args, framework="vllm") is not None


def test_a_sharded_draft_is_judged_at_the_tensor_parallel_size_the_engine_keeps(tmp_path):
    # 128 heads over 8 ranks is the same 16-head kernel shape the pinned build does not carry.
    sharded = _spec_args(_draft_model(tmp_path, heads=128)) + " -tp 8"
    assert speculative_capability_reason(sharded, framework="vllm") is not None
    # --tensor-parallel-size, --tp and --tp-size are one destination: the engine keeps the last of them, whichever
    # spelling it is, and a composed launch can carry an inherited one and a variant's own.
    args = _spec_args(_draft_model(tmp_path / "pair", heads=128)) + " --tensor-parallel-size 1 --tp 8"
    descriptor = speculative_descriptor(args)
    assert descriptor is not None and descriptor.tensor_parallel_size == 8
    assert speculative_capability_reason(args, framework="vllm") is not None
    mirrored = _spec_args(_draft_model(tmp_path / "mirror", heads=128)) + " --tp 8 --tensor-parallel-size 1"
    mirror = speculative_descriptor(mirrored)
    assert mirror is not None and mirror.tensor_parallel_size == 1
    # 128 heads on one rank is a shape the pinned build ships: a sound launch is not denied.
    assert speculative_capability_reason(mirrored, framework="vllm") is None


@pytest.mark.parametrize(
    ("framework", "heads", "overrides"),
    [
        ("vllm", 16, {"num_speculative_tokens": 1}),
        ("vllm", 16, {"num_speculative_tokens": 5}),
        ("vllm", 16, {"attention_backend": "TRITON_MLA"}),
        ("vllm", 16, {"kv_cache_dtype": "auto"}),
        ("vllm", 32, {}),
        ("vllm", 128, {}),
        ("sglang", 16, {}),
    ],
    ids=["k1", "k5", "triton-draft", "unquantized-kv-cache", "32-head-draft", "128-head-draft", "other-framework"],
)
def test_drafts_the_pinned_stack_can_run_keep_running(tmp_path, framework, heads, overrides):
    # Only the shape whose kernel is missing is refused: a K curve, another backend, another framework keep running.
    args = _spec_args(_draft_model(tmp_path, heads=heads), **overrides)
    assert speculative_capability_reason(args, framework=framework) is None


def test_the_heads_come_from_the_drafter_checkpoint(tmp_path):
    # A nested text_config declares them too; an unreadable checkpoint is not guessed at, so not refused.
    nested = _spec_args(_draft_model(tmp_path / "nested", nested=True))
    assert speculative_capability_reason(nested, framework="vllm") is not None
    assert speculative_capability_reason(_spec_args(str(tmp_path / "missing")), framework="vllm") is None


def test_the_sixteen_head_fp8_draft_is_refused_with_the_capability_error_class(tmp_path):
    args = _spec_args(_draft_model(tmp_path))
    with pytest.raises(SpeculativeCapabilityUnsupportedError) as excinfo:
        refuse_unrunnable_speculative_stack(args, framework="vllm")
    assert excinfo.value.error_class == CAPABILITY_UNSUPPORTED
    # The refusal names the shape with no kernel and what to run instead, as the engine's own guard does.
    reason = str(excinfo.value)
    assert "16 query heads per rank" in reason and "ROCM_AITER_MLA" in reason and "TRITON_MLA" in reason
    # A runnable draft is not refused at all.
    runnable = _spec_args(_draft_model(tmp_path / "ok"), num_speculative_tokens=5)
    refuse_unrunnable_speculative_stack(runnable, framework="vllm")


def test_the_grid_drops_the_unrunnable_draft_before_it_is_launched(tmp_path):
    model = _draft_model(tmp_path)
    runnable = GridVariant(name="spec_triton", extra_server_args=_spec_args(model, attention_backend="TRITON_MLA"))
    unrunnable = GridVariant(name="spec_aiter_fp8_k2", extra_server_args=_spec_args(model))
    kept, dropped = apply_compatibility_filter([runnable, unrunnable], serving=ServingStack("vllm", ""))
    assert [v.name for v in kept] == ["spec_triton"]
    assert [d["name"] for d in dropped] == ["spec_aiter_fp8_k2"]
    assert dropped[0]["source"] == CAPABILITY_UNSUPPORTED
    assert "ROCM_AITER_MLA" in dropped[0]["reason"]


def test_the_proposal_prompt_states_the_same_coverage():
    lines = speculative_capability_lines()
    assert lines and all("ROCM_AITER_MLA" in line and "TRITON_MLA" in line for line in lines)
    prompt = speculative_decoding_focus(model_name="Kimi-K3", model_path="/models/Kimi-K3", model_info={})
    # The agent is told the constraint it would otherwise have to discover by crashing a launch.
    assert all(any(line in rendered for rendered in prompt) for line in lines)
