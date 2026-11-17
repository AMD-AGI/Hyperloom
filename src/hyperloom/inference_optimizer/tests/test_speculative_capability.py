# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A draft the pinned stack ships no kernel for is refused as unsupported, never spent as a measurement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.inference_optimizer.speculative_capability import (
    CAPABILITY_UNSUPPORTED,
    DRAFT_LIMITS_KEY,
    DraftLimits,
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
    # The checkpoint summary reads a nested text tower as the engine does, so the shape is the decoder's.
    nested = _spec_args(_draft_model(tmp_path / "nested", nested=True))
    assert speculative_capability_reason(nested, framework="vllm") is not None


def test_a_self_drafting_mtp_launch_is_judged_on_the_model_it_drafts_from(tmp_path):
    # MTP drafts with the target's own heads, so a launch that names no drafter is a shape this gate can read.
    target = _draft_model(tmp_path / "target")
    config = {"method": "mtp", "num_speculative_tokens": 2, "kv_cache_dtype": "fp8"}
    args = "--speculative-config " + json.dumps(config) + " --speculative-config.attention_backend ROCM_AITER_MLA"
    descriptor = speculative_descriptor(args + f" --model {target}")
    assert descriptor is not None and descriptor.model == target
    assert speculative_capability_reason(args + f" --model {target}", framework="vllm") is not None
    # The serving model the run was given answers for it just as well as an argv one.
    assert speculative_capability_reason(args, framework="vllm", model_path=target) is not None
    # A target whose heads the pinned build does carry keeps running.
    wide = _draft_model(tmp_path / "wide", heads=64)
    assert speculative_capability_reason(args, framework="vllm", model_path=wide) is None


def test_a_drafter_whose_heads_cannot_be_read_is_left_to_the_engine(tmp_path):
    # No metadata names this shape, so no kernel claim can be made about it: the engine decides, not this gate.
    unreadable = _spec_args(str(tmp_path / "missing"))
    assert speculative_capability_reason(unreadable, framework="vllm") is None
    refuse_unrunnable_speculative_stack(unreadable, {"framework": "vllm"})
    # Admission keeps it too, so a drafter this gate cannot name is still measured.
    unknown = GridVariant(name="spec_aiter_fp8_k2_unknown", extra_server_args=unreadable)
    kept, dropped = apply_compatibility_filter([unknown], serving=ServingStack("vllm", ""))
    assert [v.name for v in kept] == ["spec_aiter_fp8_k2_unknown"] and not dropped


def test_the_sixteen_head_fp8_draft_is_refused_with_the_capability_error_class(tmp_path):
    args = _spec_args(_draft_model(tmp_path))
    with pytest.raises(SpeculativeCapabilityUnsupportedError) as excinfo:
        refuse_unrunnable_speculative_stack(args, {"framework": "vllm"})
    assert excinfo.value.error_class == CAPABILITY_UNSUPPORTED
    # The refusal names the shape with no kernel and what to run instead, as the engine's own guard does.
    reason = str(excinfo.value)
    assert "16 query heads per rank" in reason and "ROCM_AITER_MLA" in reason and "TRITON_MLA" in reason
    # A runnable draft is not refused at all.
    runnable = _spec_args(_draft_model(tmp_path / "ok"), num_speculative_tokens=5)
    refuse_unrunnable_speculative_stack(runnable, {"framework": "vllm"})


def test_the_draft_is_judged_at_the_tensor_parallel_size_the_launch_really_carries(tmp_path):
    # AgentX's launcher spells TP only in the environment and appends --tensor-parallel-size "$TP" itself, so a
    # launch whose argv says nothing about sharding still reaches the engine sharded.
    args = _spec_args(_draft_model(tmp_path, heads=128))
    assert speculative_capability_reason(args, framework="vllm") is None
    assert speculative_capability_reason(args, framework="vllm", envs={"TP": "8"}) is not None
    descriptor = speculative_descriptor(args, envs={"TP": "8"})
    assert descriptor is not None and descriptor.tensor_parallel_size == 8
    # An argv TP is the one the engine parses last, and the draft's own size outranks both.
    assert speculative_descriptor(args + " --tp 1", envs={"TP": "8"}).tensor_parallel_size == 1
    own = _spec_args(_draft_model(tmp_path / "own", heads=128), draft_tensor_parallel_size=8)
    assert speculative_descriptor(own + " --tp 1", envs={"TP": "2"}).tensor_parallel_size == 8
    # The grid judges the variant against the environment the run launches with, not against argv alone.
    variant = GridVariant(name="spec_aiter_fp8_k2_sharded", extra_server_args=args)
    kept, dropped = apply_compatibility_filter(
        [variant], serving=ServingStack("vllm", ""), base_envs={"TP": "8", "CONC": "8"}
    )
    assert not kept and [d["source"] for d in dropped] == [CAPABILITY_UNSUPPORTED]
    assert "16 query heads per rank" in dropped[0]["reason"]


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


# A deployment whose patched-in kernel verifies at most four tokens per step, and a draft spelling six.
SGLANG_OVERLONG_DRAFT = "--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
SGLANG_WITHIN_DRAFT = "--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
CAPPED_RECIPE = {"framework": "sglang", DRAFT_LIMITS_KEY: {"draft_tokens": {"max": 4}}}


def test_an_sglang_draft_is_read_from_its_own_flags():
    descriptor = speculative_descriptor(SGLANG_OVERLONG_DRAFT + " --tp 8")
    assert descriptor is not None
    assert (descriptor.method, descriptor.draft_length, descriptor.draft_tokens, descriptor.topk) == ("eagle", 5, 6, 1)
    assert descriptor.tensor_parallel_size == 8
    # A candidate composed onto a recipe spells only what it changes; the algorithm stays in the recipe.
    partial = speculative_descriptor("--speculative-num-draft-tokens 6 --cuda-graph-max-bs 128")
    assert partial is not None and (partial.method, partial.draft_tokens) == ("", 6)
    assert speculative_descriptor("--cuda-graph-max-bs 128") is None


def test_a_draft_outside_the_bounds_the_recipe_declares_is_refused():
    limits = DraftLimits.declared(CAPPED_RECIPE)
    reason = speculative_capability_reason(SGLANG_OVERLONG_DRAFT, framework="sglang", limits=limits)
    assert reason is not None and "--speculative-num-draft-tokens" in reason
    assert "at most 4" in reason and "spells 6" in reason
    with pytest.raises(SpeculativeCapabilityUnsupportedError) as excinfo:
        refuse_unrunnable_speculative_stack(SGLANG_OVERLONG_DRAFT, CAPPED_RECIPE)
    assert excinfo.value.error_class == CAPABILITY_UNSUPPORTED
    assert speculative_capability_reason(SGLANG_WITHIN_DRAFT, framework="sglang", limits=limits) is None
    refuse_unrunnable_speculative_stack(SGLANG_WITHIN_DRAFT, CAPPED_RECIPE)
    # Without a declaration nothing is known about the deployment's kernels, so the engine decides.
    assert speculative_capability_reason(SGLANG_OVERLONG_DRAFT, framework="sglang") is None
    refuse_unrunnable_speculative_stack(SGLANG_OVERLONG_DRAFT, {"framework": "sglang"})


def test_a_setting_the_launch_leaves_to_the_engine_is_not_judged():
    # SGLang derives this chain draft's verify window (six tokens) itself; the launch does not spell it.
    steps_only = "--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1"
    capped = DraftLimits.declared(CAPPED_RECIPE)
    assert speculative_capability_reason(steps_only, framework="sglang", limits=capped) is None
    # Bounding the draft length is how a recipe covers the window SGLang derives from it.
    both = DraftLimits.declared({DRAFT_LIMITS_KEY: {"draft_length": {"max": 3}, "draft_tokens": {"max": 4}}})
    reason = speculative_capability_reason(steps_only, framework="sglang", limits=both)
    assert reason is not None and "--speculative-num-steps" in reason and "at most 3" in reason


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ("4", "must map draft settings"),
        ({"verify_window": {"max": 4}}, "unknown draft setting"),
        ({"draft_tokens": 4}, "must set min, max or both"),
        ({"draft_tokens": {}}, "must set min, max or both"),
        ({"draft_tokens": {"max": 4, "step": 1}}, "must set min, max or both"),
        ({"draft_tokens": {"max": 0}}, "must be a positive integer"),
        ({"draft_tokens": {"max": True}}, "must be a positive integer"),
        ({"draft_tokens": {"max": "4"}}, "must be a positive integer"),
        ({"draft_tokens": {"min": 5, "max": 4}}, "min 5 exceeds max 4"),
    ],
)
def test_a_malformed_declaration_is_refused(declared, message):
    with pytest.raises(ValueError, match=message):
        DraftLimits.declared({DRAFT_LIMITS_KEY: declared})


def test_the_grid_drops_a_variant_outside_the_recipes_bounds_before_it_is_launched():
    over = GridVariant(name="steps5", extra_server_args=SGLANG_OVERLONG_DRAFT)
    within = GridVariant(name="steps3", extra_server_args=SGLANG_WITHIN_DRAFT)
    kept, dropped = apply_compatibility_filter(
        [over, within], serving=ServingStack.of_recipe(CAPPED_RECIPE, model_path="")
    )
    assert [v.name for v in kept] == ["steps3"]
    assert [(d["name"], d["source"]) for d in dropped] == [("steps5", CAPABILITY_UNSUPPORTED)]
    assert "at most 4" in dropped[0]["reason"]


def test_the_proposal_prompt_states_the_recipes_bounds(tmp_path, monkeypatch):
    lines = speculative_capability_lines(CAPPED_RECIPE)
    assert any("--speculative-num-draft-tokens" in line and "at most 4" in line for line in lines)
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"benchmark": CAPPED_RECIPE}), encoding="utf-8")
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, str(config))
    prompt = speculative_decoding_focus(model_name="Kimi-K3", model_path="/models/Kimi-K3", model_info={})
    assert any("--speculative-num-draft-tokens" in rendered and "at most 4" in rendered for rendered in prompt)
