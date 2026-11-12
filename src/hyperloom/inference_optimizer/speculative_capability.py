# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which speculative-decoding launches the pinned serving stack can actually run.

The keys that decide whether a draft has a kernel at all -- its backend, its KV cache dtype, its draft length -- live
inside ``--speculative-config``, so a filter reading flag names never sees them. This module reads a composed launch
with the parser :mod:`inference_suite.argv` owns and states the stack's kernel coverage as data, so the grid can refuse
an unrunnable draft rather than spend a measurement on it, and the prompt can name the same constraint.

A rule refuses only the shape it knows has no kernel: the draft's query heads per rank, read off the model that does
the drafting -- the drafter's own checkpoint, or the target's when the method drafts with the target's layers. A
shape no metadata establishes is left to the engine; what this module answers for is the pairing it can name.

SGLang spells its draft as ``--speculative-*`` flags of its own rather than one config. A deployment can also patch
in kernels whose limits neither the engine nor the checkpoint states; its recipe declares them under
``speculative_draft_limits`` (:class:`DraftLimits`), and a draft outside them is refused the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from inference_suite.argv import flag_occurrences, last_alias_value, speculative_config

from .grid_server_args import SERVED_MODEL_FLAGS, TENSOR_PARALLEL_FLAGS
from .model_config_utils import summarize_model_config

CAPABILITY_UNSUPPORTED = "capability_unsupported"
DRAFT_LIMITS_KEY = "speculative_draft_limits"
# The launcher exports the engine's tensor parallel size here; ``launch_server.sh`` turns it into the argv flag.
_TP_ENV = "TP"
_SGLANG_ALGORITHM_FLAGS = frozenset({"speculative-algorithm"})
_SGLANG_STEPS_FLAGS = frozenset({"speculative-num-steps"})
_SGLANG_DRAFT_TOKENS_FLAGS = frozenset({"speculative-num-draft-tokens"})
_SGLANG_TOPK_FLAGS = frozenset({"speculative-eagle-topk"})
_SGLANG_DRAFT_MODEL_FLAGS = frozenset({"speculative-draft-model-path"})
_SGLANG_DRAFT_FLAGS = _SGLANG_ALGORITHM_FLAGS | _SGLANG_STEPS_FLAGS | _SGLANG_DRAFT_TOKENS_FLAGS | _SGLANG_TOPK_FLAGS
# The settings a recipe may bound, named as a refusal and the proposal prompt name them.
_LIMITED_SETTINGS = {
    "draft_length": "the draft length (`--speculative-num-steps`, vLLM's `num_speculative_tokens`)",
    "draft_tokens": "the tokens verified per step (`--speculative-num-draft-tokens`)",
    "topk": "the draft top-k (`--speculative-eagle-topk`)",
}


class SpeculativeCapabilityUnsupportedError(ValueError):
    """The pinned stack ships no kernel for the draft a launch composes."""

    error_class = CAPABILITY_UNSUPPORTED


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SpeculativeDescriptor:
    """What a composed launch asks the engine to draft with, both spellings of the config merged."""

    # The model whose attention the draft runs: the drafter's checkpoint, or the target's when it self-drafts.
    model: str
    method: str
    draft_length: int | None
    attention_backend: str
    kv_cache_dtype: str
    tensor_parallel_size: int
    # SGLang's verify window and tree width; a vLLM config spells neither.
    draft_tokens: int | None = None
    topk: int | None = None

    def draft_query_heads_per_rank(self) -> int | None:
        """Query heads the draft's attention kernel sees on one rank, or ``None`` when the model does not say.

        The count comes from the checkpoint summary the rest of the tree reads models with, so a nested text tower
        and a path the resolver has to resolve are read exactly as the engine's own launch would read them.
        """
        heads = _as_int(summarize_model_config(self.model).get("num_attention_heads"))
        ranks = self.tensor_parallel_size
        return heads // ranks if heads and heads > 0 and not heads % ranks else None


def _self_drafting(method: str) -> bool:
    """``method`` drafts with layers that ship inside the target checkpoint (``mtp``, ``deepseek_mtp``, ...)."""
    return method.endswith("mtp")


def _drafting_model(config: Mapping[str, Any], method: str, occurrences: list[tuple[str, str]], model_path: str) -> str:
    """The model whose attention shape reaches the draft's kernel.

    A self-drafting method names no drafter -- its heads are the target's, so the launch's own ``--model`` (or the
    model the serving stack was built for) is what the kernel will see.
    """
    drafter = str(config.get("model") or config.get("draft_model") or "")
    if drafter or not _self_drafting(method):
        return drafter
    return str(last_alias_value(occurrences, SERVED_MODEL_FLAGS) or model_path or "")


def speculative_descriptor(
    server_args: str, *, envs: Mapping[str, Any] | None = None, model_path: str = ""
) -> SpeculativeDescriptor | None:
    """The draft a composed launch spells, or ``None`` when it configures none.

    ``envs`` is the environment that launch carries and ``model_path`` the model being served, for the launches that
    settle those two outside the argv.
    """
    occurrences = flag_occurrences(server_args)
    config = speculative_config(occurrences)
    if not config:
        return _sglang_descriptor(occurrences, envs=envs, model_path=model_path)
    method = str(config.get("method") or "").strip().lower()
    return SpeculativeDescriptor(
        model=_drafting_model(config, method, occurrences, model_path),
        method=method,
        draft_length=_as_int(config.get("num_speculative_tokens")),
        attention_backend=str(config.get("attention_backend") or "").strip().upper(),
        kv_cache_dtype=str(config.get("kv_cache_dtype") or "").strip().lower(),
        tensor_parallel_size=_tensor_parallel_size(occurrences, envs, config.get("draft_tensor_parallel_size")),
    )


def _sglang_descriptor(
    occurrences: list[tuple[str, str]], *, envs: Mapping[str, Any] | None, model_path: str
) -> SpeculativeDescriptor | None:
    """The draft an SGLang launch spells as its own ``--speculative-*`` flags, or ``None`` when it spells none.

    A candidate composed onto a recipe spells only the settings it changes, the algorithm staying in the recipe, so any
    one of the flags is a draft. Each setting is read as spelled: a value SGLang resolves for itself (a chain draft
    verifies ``num_steps + 1`` tokens whatever ``--speculative-num-draft-tokens`` says) is the engine's to settle.
    """
    if not any(name in _SGLANG_DRAFT_FLAGS for name, _ in occurrences):
        return None
    drafter = last_alias_value(occurrences, _SGLANG_DRAFT_MODEL_FLAGS) or last_alias_value(
        occurrences, SERVED_MODEL_FLAGS
    )
    return SpeculativeDescriptor(
        model=str(drafter or model_path or ""),
        method=str(last_alias_value(occurrences, _SGLANG_ALGORITHM_FLAGS) or "").strip().lower(),
        draft_length=_as_int(last_alias_value(occurrences, _SGLANG_STEPS_FLAGS)),
        attention_backend="",
        kv_cache_dtype="",
        tensor_parallel_size=_tensor_parallel_size(occurrences, envs, None),
        draft_tokens=_as_int(last_alias_value(occurrences, _SGLANG_DRAFT_TOKENS_FLAGS)),
        topk=_as_int(last_alias_value(occurrences, _SGLANG_TOPK_FLAGS)),
    )


def _tensor_parallel_size(occurrences: list[tuple[str, str]], envs: Mapping[str, Any] | None, own: Any) -> int:
    """The size a draft's attention is sharded over.

    Its ``own`` TP size when it sets one, else the engine's: three spellings of one destination, the last winning, and
    then ``envs.TP``, which the launcher writes into the argv itself -- before the extra args, so an argv spelling
    overrides it.
    """
    ranks = [own, last_alias_value(occurrences, TENSOR_PARALLEL_FLAGS), (envs or {}).get(_TP_ENV)]
    return next((size for size in map(_as_int, ranks) if size and size > 0), 1)


@dataclass(frozen=True)
class DraftLimits:
    """The draft settings a recipe declares its deployment runs, as inclusive ``(min, max)`` bounds.

    A deployment can patch in kernels whose limits neither the engine nor the checkpoint states, so the recipe that
    pins the deployment is where they are said. A setting the recipe leaves unbounded, or a launch leaves to the
    engine, is not judged.
    """

    bounds: Mapping[str, tuple[int | None, int | None]] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def declared(cls, benchmark: Mapping[str, Any]) -> DraftLimits:
        """The limits ``benchmark`` declares under ``speculative_draft_limits``; none when it declares nothing.

        Raises:
            ValueError: When the declaration is not a mapping of known settings to ``min``/``max`` positive integers,
                or a ``min`` exceeds its ``max``.
        """
        declared = benchmark.get(DRAFT_LIMITS_KEY) or {}
        if not isinstance(declared, Mapping):
            raise ValueError(f"benchmark.{DRAFT_LIMITS_KEY} must map draft settings to min/max bounds")
        bounds: dict[str, tuple[int | None, int | None]] = {}
        for setting, bound in declared.items():
            where = f"benchmark.{DRAFT_LIMITS_KEY}.{setting}"
            if setting not in _LIMITED_SETTINGS:
                raise ValueError(f"{where}: unknown draft setting; expected one of {', '.join(_LIMITED_SETTINGS)}")
            if not isinstance(bound, Mapping) or not bound or set(bound) - {"min", "max"}:
                raise ValueError(f"{where} must set min, max or both")
            low, high = (_declared_bound(bound.get(end), f"{where}.{end}") for end in ("min", "max"))
            if low is not None and high is not None and low > high:
                raise ValueError(f"{where}: min {low} exceeds max {high}")
            bounds[setting] = (low, high)
        return cls(MappingProxyType(bounds))

    def unsupported_reason(self, descriptor: SpeculativeDescriptor) -> str | None:
        """Why the recipe's deployment does not run ``descriptor``, or ``None`` when it is within every bound."""
        for setting, (low, high) in self.bounds.items():
            value = getattr(descriptor, setting)
            if value is None or ((low is None or value >= low) and (high is None or value <= high)):
                continue
            return (
                f"{_LIMITED_SETTINGS[setting]} must be {_within(low, high)}, the bound this recipe declares for "
                f"its deployment's kernels; this launch spells {value}"
            )
        return None

    def prompt_line(self) -> str | None:
        """The same bounds as one bullet for the proposal prompt, or ``None`` when the recipe declares none."""
        if not self.bounds:
            return None
        bounds = "; ".join(f"{_LIMITED_SETTINGS[s]} {_within(low, high)}" for s, (low, high) in self.bounds.items())
        return f"- This recipe's deployment runs only drafts with {bounds}."


def _declared_bound(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")
    return value


def _within(low: int | None, high: int | None) -> str:
    if low is None:
        return f"at most {high}"
    return f"at least {low}" if high is None else f"from {low} to {high}"


@dataclass(frozen=True)
class DraftKernelCoverage:
    """The draft shapes one pinned backend ships kernels for, and what to run instead outside them."""

    framework: str
    draft_backend: str
    kv_cache_dtypes: frozenset[str]
    query_heads_by_draft_length: Mapping[int, frozenset[int]]
    remedy: str

    def unsupported_reason(self, descriptor: SpeculativeDescriptor, *, framework: str) -> str | None:
        """Why this coverage has no kernel for ``descriptor``, or ``None`` when it does not stand in the way.

        A draft outside the combination this coverage owns passes: that is this rule saying nothing, not a claim
        about kernels it does not ship. Inside the combination the shape decides, and a shape no metadata gives is
        left to the engine rather than guessed at.
        """
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


def speculative_capability_reason(
    server_args: str,
    *,
    framework: str,
    envs: Mapping[str, Any] | None = None,
    model_path: str = "",
    limits: DraftLimits | None = None,
) -> str | None:
    """Why the pinned stack cannot launch the draft ``server_args`` composes, or ``None`` when it can.

    The judgement is on the launch the engine really gets, so it takes the ``envs`` that launch carries -- the
    tensor parallel size travels there -- and the ``model_path`` it serves, which is the drafter itself when the
    method drafts with the target's own layers. ``limits`` are the bounds the recipe declares for its deployment.
    """
    descriptor = speculative_descriptor(server_args, envs=envs, model_path=model_path)
    if descriptor is None:
        return None
    reasons = (coverage.unsupported_reason(descriptor, framework=framework) for coverage in PINNED_DRAFT_KERNELS)
    return next((reason for reason in reasons if reason), None) or (limits or DraftLimits()).unsupported_reason(
        descriptor
    )


def refuse_unrunnable_speculative_stack(
    server_args: str, benchmark: Mapping[str, Any], *, envs: Mapping[str, Any] | None = None, model_path: str = ""
) -> None:
    """Refuse a launch of ``benchmark``'s recipe whose draft cannot run -- the pinned stack ships no kernel for it, or
    the recipe declares its deployment does not -- since it is not applicable, not a measurement."""
    reason = speculative_capability_reason(
        server_args,
        framework=str(benchmark.get("framework") or ""),
        envs=envs,
        model_path=model_path,
        limits=DraftLimits.declared(benchmark),
    )
    if reason:
        raise SpeculativeCapabilityUnsupportedError(reason)


def speculative_capability_lines(benchmark: Mapping[str, Any] | None = None) -> list[str]:
    """The pinned stack's draft-kernel coverage and the draft bounds ``benchmark``'s recipe declares, for the prompt
    that asks an agent for a speculative proposal."""
    declared = DraftLimits.declared(benchmark or {}).prompt_line()
    coverage = [coverage.prompt_line() for coverage in PINNED_DRAFT_KERNELS]
    return [*coverage, declared] if declared else coverage
