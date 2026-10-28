# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Direction families: a test's family comes from its change first, its words only after."""

from __future__ import annotations

import pytest

from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import REPLAY_TARGET
from hyperloom_multibranch.notebook.families import OTHER_TARGET, Component, Family, Mechanism, family_of

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
    "family", [Family("KV cache", Mechanism.PRECISION, CONFIG), Family("moe", Mechanism.OTHER, None)]
)
def test_a_label_reads_back_as_its_family(family: Family) -> None:
    assert Family.from_label(family.label) == family


@pytest.mark.parametrize("label", ["not a family", "gemm · sprinkling · config", "gemm · fusion · nowhere"])
def test_a_label_no_family_has_reads_as_none(label: str) -> None:
    assert Family.from_label(label) is None
