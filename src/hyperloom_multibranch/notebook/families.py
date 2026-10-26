# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Direction families: what a test acts on, how, and at which layer, so the outcomes of alike tests add up.

A :class:`Family` is a target, a :class:`Mechanism` and a layer. The target is an op family of :data:`OP_FAMILIES`,
read off a profile's op names, or a system :class:`Component` such as the KV cache or the scheduler; a Recipe KB
replay is :attr:`Component.WARM_START`. :func:`family_of` reads the change itself first: a config change by the flags
and environment variables it sets, a kernel artifact by its kind; an archived test, whose change was not kept, by the
flags and variables its mechanism quotes. Only then does it read the intent's words, its target before its
mechanism; the intent contract and its fingerprint stay as they are. A test whose words name nothing is
``other``. For example, fp8 KV (``--kv-cache-dtype fp8_e4m3``) is ``KV cache · quantization precision · config``,
whatever op the specialist named as its target.

What the families of past sessions came to is :mod:`rsi.family_stats`'s.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from hyperloom.contracts.recipe import KernelKind, RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import REPLAY_TARGET

#: The op families a target names, each with what its words or op names look like, the more specific first.
OP_FAMILIES: tuple[tuple[str, str], ...] = (
    ("linear attention", r"gdn|gated_delta|delta_rule|causal_conv1d|conv1d|mamba|linear[ _-]attention"),
    ("moe", r"moe|expert|router|gating|topk_?softmax|softmax_?topk"),
    ("attention", r"attn|attention|paged_attention|\bpa_|\bmla\b|\bmha\b|flash|ll4mi"),
    ("gemm", r"gemm|matmul|blockscale|a8w8|a16w16|qkv_proj|o_proj|gate_up|down_proj|split-?k|\bck\b|hipblaslt"),
    ("norm", r"rmsnorm|layernorm|qk-?norm|\bnorm\b"),
    ("rope", r"rope|rotary"),
    ("activation", r"silu|gelu|swiglu|_and_mul|activation"),
    ("sampling", r"sampl|logits"),
    ("elementwise", r"elementwise|residual"),
)


class Component(str, Enum):
    """A part of the serving system a test acts on when it names no op."""

    KV_CACHE = "KV cache"
    SCHEDULING = "scheduling and batching"
    MEMORY = "memory"
    PARALLEL = "parallelism and communication"
    GRAPH = "graph capture and compile"
    SPECULATIVE = "speculative decoding"
    QUANTIZATION = "weight quantization"
    RUNTIME = "other runtime settings"
    WARM_START = "warm start replay"


#: What each component's words, flags and variables look like, checked in this order.
COMPONENTS: tuple[tuple[Component, str], ...] = (
    (
        Component.PARALLEL,
        r"all-?reduce|all_reduce|all-?gather|nccl|rccl|custom[-_ ]?ar\b|cross_device|expert[-_]parallel"
        r"|dp[-_]attention|--tp\b|--dp\b|--ep\b|a2a",
    ),
    (Component.KV_CACHE, r"kv[-_ ]?cache|kvcache|kv store|kv_layout|page[-_]size|radix|prefix[-_ ]cache"),
    (Component.SPECULATIVE, r"specul|eagle|\bmtp\b|draft|spec[ _-]verify"),
    (Component.GRAPH, r"cuda[-_ ]?graph|torch[-_]compile|compile|enforce[-_]eager|\bgraph\b"),
    (
        Component.SCHEDULING,
        r"chunked[-_]prefill|max[-_]running|max[-_]prefill|schedul|batch|mixed[-_]chunk"
        r"|stream[-_]interval|detokeniz|max[-_]num[-_]seqs|continuous[-_]decode",
    ),
    (Component.MEMORY, r"mem[-_]fraction|gpu[-_]memory|memory|swap[-_]space|offload"),
    (Component.QUANTIZATION, r"--quantization|w8a8|w4a16|int4|awq|gptq|weight quant"),
    (Component.RUNTIME, r"gpu_max_hw_queues|\bhip_|host|launch|runtime|overhead"),
)


class Mechanism(str, Enum):
    """How a test changes what it acts on."""

    TUNING = "parameter tuning"
    BACKEND = "backend or implementation"
    KERNEL_TUNING = "kernel tuning"
    FUSION = "fusion"
    PRECISION = "quantization precision"
    LAYOUT = "cache layout"
    PARTITIONING = "partitioning"
    OVERHEAD = "overhead removal"
    OTHER = "other"


#: What each mechanism's words look like, checked in this order; a kernel-layer test is kernel tuning unless it fuses.
MECHANISMS: tuple[tuple[Mechanism, str], ...] = (
    (Mechanism.FUSION, r"fus|fold|merge .* into|combin"),
    (Mechanism.PRECISION, r"fp8|fp4|int8|int4|bf16|fp16|dtype|quant|precision"),
    (Mechanism.LAYOUT, r"layout|vectorized|\b5d\b|page[-_]size|interleav|transpos"),
    (Mechanism.PARTITIONING, r"--tp\b|--dp\b|expert[-_]parallel|dp[-_]attention|shard|partition"),
    (Mechanism.KERNEL_TUNING, r"tun|tile|\bck id|split-?k|\basm\b|\bbv=|warps|block[-_ ]size|occupancy"),
    (
        Mechanism.BACKEND,
        r"backend|use[-_]aiter|aiter|triton|flydsl|replac|route|switch|instead of|implementation"
        r"|unified",
    ),
    (Mechanism.OVERHEAD, r"overhead|disable|remov|skip|stream[-_]interval|bookkeeping|launch|host|sync|copy"),
    (Mechanism.TUNING, r"raise|lower|increas|decreas|size|=\s*\d|\b\d+\b"),
)
OTHER_TARGET = "other"
_FLAG = re.compile(r"--[a-z0-9][a-z0-9-]*")
_ENV = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
_Named = TypeVar("_Named")


@dataclass(frozen=True)
class Family:
    """A direction family: ``target`` (an op family or a :class:`Component`'s value), ``mechanism``, ``layer``."""

    target: str
    mechanism: Mechanism
    layer: IntentLayer | None

    @property
    def label(self) -> str:
        layer = self.layer.value if self.layer is not None else "any layer"
        return f"{self.target} · {self.mechanism.value} · {layer}"


def _matched(patterns: Iterable[tuple[_Named, str]], text: str) -> _Named | None:
    lowered = text.lower()
    return next((name for name, pattern in patterns if re.search(pattern, lowered)), None)


def op_family(op: str) -> str | None:
    """The op family of :data:`OP_FAMILIES` a profile's op name belongs to; ``None`` for one of none."""

    return _matched(OP_FAMILIES, op)


def _target(text: str) -> str | None:
    """The op family ``text`` names, else the component, else ``None``."""

    component = _matched(COMPONENTS, text)
    return _matched(OP_FAMILIES, text) or (component.value if component is not None else None)


def _mechanism(text: str, layer: IntentLayer | None, kinds: set[KernelKind]) -> Mechanism:
    """How a test changes what it acts on: a kernel artifact or a kernel-layer test tunes a kernel unless it fuses."""

    fuses = KernelKind.FUSION in kinds or _matched(MECHANISMS[:1], text) is Mechanism.FUSION
    if fuses:
        return Mechanism.FUSION
    if kinds or layer is IntentLayer.KERNEL:
        return Mechanism.KERNEL_TUNING
    return _matched(MECHANISMS, text) or Mechanism.OTHER


def _changed_names(delta: RecipeDelta) -> str:
    """The flags and environment variables ``delta`` sets or drops, as one line of words."""

    config = delta.config
    flags = _FLAG.findall(config.server_args) + list(config.remove_args)
    return " ".join([*flags, *config.envs, *config.unset_envs])


def _changed_text(delta: RecipeDelta) -> str:
    """What ``delta`` sets, values included, and the targets of the kernel artifacts it adds."""

    config = delta.config
    settings = [f"{name}={value}" for name, value in config.envs.items()]
    return " ".join([config.server_args, *config.remove_args, *settings, *(item.target for item in delta.add_kernels)])


def _quoted_names(text: str) -> str:
    """The flags and environment variables ``text`` quotes, for a test whose change was not kept."""

    return " ".join([*_FLAG.findall(text), *_ENV.findall(text)])


def family_of(target: str, mechanism: str, layer: IntentLayer | None, delta: RecipeDelta | None = None) -> Family:
    """The family of a test of the intent (``target``, ``mechanism``, ``layer``) that made ``delta``."""

    if target.strip().lower() == REPLAY_TARGET:
        return Family(Component.WARM_START.value, Mechanism.OTHER, layer)
    changed = _changed_names(delta) if delta is not None else _quoted_names(mechanism)
    settings = _changed_text(delta) if delta is not None else changed
    kernels = " ".join(artifact.target for artifact in delta.add_kernels) if delta is not None else ""
    kinds = {artifact.kind for artifact in delta.add_kernels} if delta is not None else set()
    acted = _target(kernels) if kernels else None
    acted = acted or ("gemm" if KernelKind.GEMM in kinds else None) or (_target(changed) if changed else None)
    acted = acted or _target(target) or _target(mechanism) or OTHER_TARGET
    return Family(acted, _mechanism(f"{settings} {mechanism}", layer, kinds), layer)


__all__ = [
    "COMPONENTS",
    "MECHANISMS",
    "OP_FAMILIES",
    "OTHER_TARGET",
    "Component",
    "Family",
    "Mechanism",
    "family_of",
    "op_family",
]
