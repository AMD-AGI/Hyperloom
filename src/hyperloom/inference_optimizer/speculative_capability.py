# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which speculative-decoding launches the pinned serving stack can actually run.

The keys that decide whether a draft has a kernel at all -- its backend, its KV cache dtype, its draft length -- live
inside ``--speculative-config``, so a filter reading flag names never sees them. This module merges both spellings of
that config into one descriptor and states the stack's kernel coverage as data, so the grid can refuse an unrunnable
draft rather than spend a measurement on it, and the prompt can name the same constraint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .grid_server_args import tokenize_server_args_preserving_json
from .model_config_utils import _load_model_config_dict

CAPABILITY_UNSUPPORTED = "capability_unsupported"
_SPEC_CONFIG_FLAG = "speculative-config"
_SPEC_CONFIG_KEY_PREFIX = f"{_SPEC_CONFIG_FLAG}."
# One destination the engines spell three ways, so they are resolved together rather than one spelling at a time.
_TP_FLAGS = frozenset({"tensor-parallel-size", "tp", "tp-size"})


class SpeculativeCapabilityUnsupportedError(ValueError):
    """The pinned stack ships no kernel for the draft a launch composes."""

    error_class = CAPABILITY_UNSUPPORTED


def _is_flag(token: str) -> bool:
    """``token`` is a flag rather than a value (``-tp`` counts, ``-1`` does not)."""
    return token.startswith("-") and not token[1:2].isdigit() and token.strip("-") != ""


def flag_occurrences(server_args: str) -> list[tuple[str, str]]:
    """Every flag a composed argument string spells, in order, as ``(name, value)``; ``""`` for a bare flag."""
    split = tokenize_server_args_preserving_json(server_args)
    tokens = split[1] if split is not None else server_args.split()
    occurrences: list[tuple[str, str]] = []
    for index, token in enumerate(tokens):
        if not _is_flag(token):
            continue
        name, assigned, value = token.partition("=")
        if not assigned and index + 1 < len(tokens) and not _is_flag(tokens[index + 1]):
            value = tokens[index + 1]
        occurrences.append((name.lstrip("-").lower().replace("_", "-"), value))
    return occurrences


def _last_alias_value(occurrences: list[tuple[str, str]], aliases: frozenset[str]) -> str | None:
    """The value of the last flag spelling one destination, which is the one the engines' argparse keeps."""
    return next((value for name, value in reversed(occurrences) if name in aliases), None)


def json_object(text: str) -> dict[str, Any]:
    """``text`` as a JSON object, or ``{}`` when it is neither."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SpeculativeDescriptor:
    """What a composed launch asks the engine to draft with, both spellings of the config merged."""

    model: str
    method: str
    draft_length: int | None
    attention_backend: str
    kv_cache_dtype: str
    tensor_parallel_size: int

    def draft_query_heads_per_rank(self) -> int | None:
        """Query heads the draft's attention kernel sees on one rank, or ``None`` when the model does not say."""
        config = _load_model_config_dict(self.model) or {}
        nested = config.get("text_config")
        heads = _as_int(config.get("num_attention_heads"))
        if heads is None and isinstance(nested, Mapping):
            heads = _as_int(nested.get("num_attention_heads"))
        ranks = self.tensor_parallel_size
        return heads // ranks if heads and heads > 0 and not heads % ranks else None


def speculative_descriptor(server_args: str) -> SpeculativeDescriptor | None:
    """The draft a composed argument string spells, or ``None`` when it configures none.

    The config arrives as a JSON object and as ``--speculative-config.<key>`` flags, the per-key flags winning.
    """
    occurrences = flag_occurrences(server_args)
    flags = dict(occurrences)
    config: dict[str, Any] = {}
    for flag, value in flags.items():
        if flag == _SPEC_CONFIG_FLAG:
            config = {**json_object(value), **config}
        elif flag.startswith(_SPEC_CONFIG_KEY_PREFIX):
            config[flag.removeprefix(_SPEC_CONFIG_KEY_PREFIX).replace("-", "_")] = value
    if not config:
        return None
    # The draft's attention is sharded over its own TP size when it sets one, else over the engine's -- whose three
    # spellings are one destination, so the last of them wins rather than a fixed order between the aliases.
    ranks = [config.get("draft_tensor_parallel_size"), _last_alias_value(occurrences, _TP_FLAGS)]
    return SpeculativeDescriptor(
        model=str(config.get("model") or config.get("draft_model") or ""),
        method=str(config.get("method") or ""),
        draft_length=_as_int(config.get("num_speculative_tokens")),
        attention_backend=str(config.get("attention_backend") or "").strip().upper(),
        kv_cache_dtype=str(config.get("kv_cache_dtype") or "").strip().lower(),
        tensor_parallel_size=next((size for size in map(_as_int, ranks) if size and size > 0), 1),
    )


@dataclass(frozen=True)
class DraftKernelCoverage:
    """The draft shapes one pinned backend ships kernels for, and what to run instead outside them."""

    framework: str
    draft_backend: str
    kv_cache_dtypes: frozenset[str]
    query_heads_by_draft_length: Mapping[int, frozenset[int]]
    remedy: str

    def unsupported_reason(self, descriptor: SpeculativeDescriptor, *, framework: str) -> str | None:
        """Why this stack cannot run ``descriptor``, or ``None`` when the shape is covered or not ours to judge."""
        if framework.strip().lower() != self.framework:
            return None
        if descriptor.attention_backend != self.draft_backend or descriptor.kv_cache_dtype not in self.kv_cache_dtypes:
            return None
        covered = self.query_heads_by_draft_length.get(descriptor.draft_length or -1)
        if covered is None:
            return None
        heads = descriptor.draft_query_heads_per_rank()
        if heads is None or heads in covered:
            return None
        return (
            f"{self.draft_backend} has no {descriptor.kv_cache_dtype} kernel for a "
            f"{descriptor.draft_length}-token draft at {heads} query heads per rank "
            f"(the pinned {self.framework} stack ships {_head_counts(covered)}); {self.remedy}"
        )

    def prompt_line(self) -> str:
        """The same coverage as one bullet for the proposal prompt."""
        shapes = ", ".join(
            f"draft length {length} only at {_head_counts(covered)}"
            for length, covered in sorted(self.query_heads_by_draft_length.items())
        )
        return (
            f"- On {self.framework}, a `{self.draft_backend}` draft with a `{'/'.join(sorted(self.kv_cache_dtypes))}` "
            f"KV cache has kernels for {shapes} query heads per rank: {self.remedy} Other draft lengths are unaffected."
        )


def _head_counts(covered: frozenset[int]) -> str:
    return "/".join(str(count) for count in sorted(covered))


# AITER's fp8 MLA dispatch folds a 2-token query block onto its non-causal kernels, which it builds only for the
# 32/64/96/128-head shapes of full-size models; a smaller draft (a 16-head MLA drafter, say) has no such kernel and
# the engine dies in profiling, before any request is served.
PINNED_DRAFT_KERNELS: tuple[DraftKernelCoverage, ...] = (
    DraftKernelCoverage(
        framework="vllm",
        draft_backend="ROCM_AITER_MLA",
        kv_cache_dtypes=frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"}),
        query_heads_by_draft_length={2: frozenset({32, 64, 96, 128})},
        remedy="pin the draft's own attention backend to TRITON_MLA for this draft length.",
    ),
)


def speculative_capability_reason(server_args: str, *, framework: str) -> str | None:
    """Why the pinned stack cannot launch the draft ``server_args`` composes, or ``None`` when it can."""
    descriptor = speculative_descriptor(server_args)
    if descriptor is None:
        return None
    reasons = (coverage.unsupported_reason(descriptor, framework=framework) for coverage in PINNED_DRAFT_KERNELS)
    return next((reason for reason in reasons if reason), None)


def refuse_unrunnable_speculative_stack(server_args: str, *, framework: str) -> None:
    """Refuse a launch whose draft the pinned stack has no kernel for: it is not applicable, not a measurement."""
    reason = speculative_capability_reason(server_args, framework=framework)
    if reason:
        raise SpeculativeCapabilityUnsupportedError(reason)


def speculative_capability_lines() -> list[str]:
    """The pinned stack's draft-kernel coverage, for the prompt that asks an agent for a speculative proposal."""
    return [coverage.prompt_line() for coverage in PINNED_DRAFT_KERNELS]
