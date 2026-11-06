# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Direction families: a test's family comes from its change first, its words only after."""

from __future__ import annotations

import pytest

from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta
from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import REPLAY_TARGET, MapDirection
from hyperloom_multibranch.blackboard.families import (
    OTHER_TARGET,
    Component,
    Family,
    Mechanism,
    direction_names_op,
    family_of,
    names_op,
    op_family,
)

CONFIG, SOURCE, KERNEL = IntentLayer.CONFIG, IntentLayer.SOURCE, IntentLayer.KERNEL
TUNED = KernelArtifact(KernelKind.GEMM, "bf16 4096x14336", "kernel/gemm/tuned.csv", "a" * 64)


def _args(server_args: str = "", **envs: str) -> RecipeDelta:
    return RecipeDelta(config=ConfigDelta(server_args=server_args, envs=envs))


@pytest.mark.parametrize(
    ("target", "mechanism", "layer", "delta", "found"),
    [
        # The plan's example: fp8 KV is the KV cache's, whatever op the specialist named.
        (
            "paged_attention_decode",
            "fp8 KV",
            CONFIG,
            _args("--kv-cache-dtype fp8_e4m3"),
            ("KV cache", Mechanism.PRECISION),
        ),
        (
            "prefill chunking",
            "raise it",
            CONFIG,
            _args("--chunked-prefill-size 16384"),
            ("scheduling and batching", Mechanism.TUNING),
        ),
        (
            "decode attention",
            "unified path",
            CONFIG,
            _args(SGLANG_USE_AITER_UNIFIED_ATTN="1"),
            ("attention", Mechanism.BACKEND),
        ),
        (
            "allreduce + rmsnorm",
            "fuse them",
            CONFIG,
            _args("--enable-aiter-allreduce-fusion"),
            ("parallelism and communication", Mechanism.FUSION),
        ),
        (
            "a GEMM",
            "retune the decode tiles",
            KERNEL,
            RecipeDelta(add_kernels=(TUNED,)),
            ("gemm", Mechanism.KERNEL_TUNING),
        ),
        (
            "GDN causal_conv1d_update",
            "route to the aiter FlyDSL kernel",
            SOURCE,
            None,
            ("linear attention", Mechanism.BACKEND),
        ),
        (
            "qkv split + qk-norm + rope",
            "extend the fused qk-norm+rope kernel to prefill",
            SOURCE,
            None,
            ("norm", Mechanism.FUSION),
        ),
        (
            "paged_attention_ll4mi decode",
            "SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d routes decode",
            CONFIG,
            None,
            ("KV cache", Mechanism.LAYOUT),
        ),
        ("aten::mm ragged m", "fuse the ragged mm into one launch", KERNEL, None, ("gemm", Mechanism.FUSION)),
        (
            "aten::_index_put_impl_ decode req_to_token write",
            "skip the per-token write",
            SOURCE,
            None,
            ("KV cache", Mechanism.OVERHEAD),
        ),
        ("aten::copy_ (sdma copies)", "remove the extra copy", SOURCE, None, ("memory", Mechanism.OVERHEAD)),
        ("tokenizer worker count", "raise it to 4", CONFIG, None, ("other runtime settings", Mechanism.TUNING)),
        ("something new", "an idea", SOURCE, None, (OTHER_TARGET, Mechanism.OTHER)),
    ],
)
def test_a_tests_family_reads_its_change_before_its_words(
    target: str, mechanism: str, layer: IntentLayer, delta: RecipeDelta | None, found: tuple[str, Mechanism]
) -> None:
    family = family_of(target, mechanism, layer, delta)

    assert (family.target, family.mechanism, family.layer) == (*found, layer)


def test_a_warm_start_replay_is_a_family_of_its_own_and_a_family_reads_as_a_label() -> None:
    replay = family_of(REPLAY_TARGET, "inference:older", SOURCE, _args("--kv-cache-dtype fp8_e4m3"))

    assert replay == Family(Component.WARM_START.value, Mechanism.OTHER, SOURCE)
    assert Family("KV cache", Mechanism.PRECISION, CONFIG).label == "KV cache · quantization precision · config"
    assert Family("gemm", Mechanism.FUSION, None).label == "gemm · fusion · any layer"


@pytest.mark.parametrize(
    ("op", "found"),
    [
        ("mla_decode_fwd", "attention"),
        ("aiter::mha_varlen_fwd", "attention"),
        ("ck_fmha_fwd_d128", "attention"),
        ("fused_pa_decode", "attention"),
        ("vllm::fused_add_rms_norm", "norm"),
        ("layer_norm_kernel", "norm"),
        ("ck_layernorm2d_fwd", "norm"),
        ("normal_sampler", "sampling"),
        ("block_reduce", None),
    ],
)
def test_an_op_name_joined_by_underscores_reads_its_family(op: str, found: str | None) -> None:
    assert op_family(op) == found


@pytest.mark.parametrize(
    ("op", "found"),
    [
        ("ck_tile::kentry [Qwen3Attention.QKVParallelLinear]", "gemm"),
        ("ck_tile::kentry [Qwen3MLP.MergedColumnParallelLinear]", "gemm"),
        ("paged_attention_ll4mi_qkv_kernel [Qwen3Attention.RadixAttention]", "attention"),
        ("triton_poi_fused_0 [Qwen3DecoderLayer.RMSNorm]", "norm"),
        ("triton_poi_fused_1 [Qwen3Attention.Unnamed]", "attention"),
        ("aten::linear", "gemm"),
    ],
)
def test_an_op_name_with_its_launcher_reads_its_kernel_first_then_its_layer_then_its_module(
    op: str, found: str
) -> None:
    assert op_family(op) == found


@pytest.mark.parametrize(
    "family", [Family("KV cache", Mechanism.PRECISION, CONFIG), Family("moe", Mechanism.OTHER, None)]
)
def test_a_label_reads_back_as_its_family(family: Family) -> None:
    assert Family.from_label(family.label) == family


@pytest.mark.parametrize("label", ["not a family", "gemm · sprinkling · config", "gemm · fusion · nowhere"])
def test_a_label_no_family_has_reads_as_none(label: str) -> None:
    assert Family.from_label(label) is None


@pytest.mark.parametrize(
    ("text", "op", "named"),
    [
        ("root profile: largest rms_norm_kernel 4.0 %", "rms_norm_kernel.kd", True),
        ("rms_norm_kernel.kd takes 4% of the profile", "rms_norm_kernel", True),
        ("aiter::cross_device_reduce_1stage is a quarter of it", "aiter::cross_device_reduce_1stage", True),
        ("fused_rms_norm_kernel takes 4%", "rms_norm_kernel", False),
        ("the rms_norm_kernels take 4%", "rms_norm_kernel", False),
        ("largest rms_norm_kernel 4.0 %", "", False),
    ],
)
def test_a_text_names_an_op_as_a_whole_word_with_or_without_its_kd_suffix(text: str, op: str, named: bool) -> None:
    assert names_op(text, op) is named


@pytest.mark.parametrize(
    ("direction", "op", "named"),
    [
        (
            MapDirection("m-fuse", "Small kernels folded", "open", "largest rms_norm_kernel 4.0 %"),
            "rms_norm_kernel.kd",
            True,
        ),
        (MapDirection("m-rmsnorm", "Small kernels folded", "open"), "rmsnorm", True),
        (MapDirection("m-fuse", "Small kernels folded", "open"), "rms_norm_kernel", False),
        (MapDirection("m-fuse", "rmsnorm into the gemm_a8w8 epilogue", "open", ops=("rmsnorm.kd",)), "rmsnorm", True),
        (MapDirection("m-fuse", "rmsnorm into the gemm_a8w8 epilogue", "open", ops=("rmsnorm",)), "gemm_a8w8", False),
    ],
)
def test_a_direction_names_the_ops_it_lists_and_without_a_list_those_its_id_title_or_why_name(
    direction: MapDirection, op: str, named: bool
) -> None:
    assert direction_names_op(direction, op) is named
