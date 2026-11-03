"""Map kernel names onto named components, and say what you failed to map.

This is the one idea worth taking from SIKL. SIKL's per-op bench hooks module
objects, so its rows are named for the source (``attn.indexer``,
``mlp.experts``) and its unmatched time lands in an explicit ``<rest>`` row you
can read. On DSv4 that row was 16.6% of prefill — a number that was *visible*,
and therefore a known coverage hole rather than a silent understatement.

A torch trace gives mangled device symbols instead. Matching those with
substrings is workable but fails in a specific, dangerous way: a pattern that
matches nothing returns the same small share as a component that is genuinely
cheap. So this module refuses to report a share without also reporting how much
of the step it could not account for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from arbor.profiling.trace import KernelTotals

REST = "<rest>"


@dataclass(frozen=True)
class Component:
    """A named slice of the step, matched by substring or regex on kernel name.

    ``patterns`` are matched case-insensitively as plain substrings unless
    ``regex=True``. ``source`` is free text naming where the component lives in
    the model — it is what makes a row actionable, and it is the part a bare
    kernel name cannot tell you.
    """

    name: str
    patterns: tuple[str, ...]
    source: str = ""
    regex: bool = False

    def matches(self, kernel: str) -> bool:
        k = kernel.lower()
        if self.regex:
            return any(re.search(p, k, re.IGNORECASE) for p in self.patterns)
        return any(p.lower() in k for p in self.patterns)


# Starting table for DSv4.x decode on ROCm/vLLM.
#
# Patterns are deliberately drawn from the same vocabulary the pack's
# evaluators already use (eval_dsa_indexer.INDEXER_OPS, eval_moe_fp4.MOE_OPS,
# eval_mla_absorb_rope.MLA_OPS), so a share computed here lines up with the
# component those evaluators measure. Where a pattern in those lists is a
# *module* name that cannot appear in a device symbol ("absorb", "indexer"),
# it is kept — it costs nothing and will match if a build annotates — but it is
# not what the row relies on.
DEFAULT_COMPONENTS: tuple[Component, ...] = (
    Component(
        name="dsa_indexer",
        patterns=(
            "paged_mqa_logits", "topk_transform", "coop_topk", "dual_gemv",
            "qk_rope_hadamard", "indexer", "catarraybatchedcopy",
            "index_select", "repeat_interleave",
        ),
        source="DSA per-step metadata + top-k select (directive P1, vLLM #51315)",
    ),
    Component(
        name="mla_attention",
        patterns=(
            "mla", "fwd_kvcache", "reshape_and_cache", "cat_and_cache",
            "rope", "absorb",
        ),
        source="MLA q-absorb BMM + RoPE + KV-cache write (directive P0)",
    ),
    Component(
        name="moe",
        patterns=(
            "fused_moe", "moe_", "grouped_gemm", "expert", "silu",
            "topk_softmax", "moe_align",
        ),
        source="aiter FP8xFP4 fused MoE",
    ),
    Component(
        name="dense_gemm",
        patterns=("gemm", "hgemm", "cijk", "cutlass", "hipblaslt", "matmul"),
        source="projection GEMMs (qkv_a, q_b, o_proj, lm_head)",
    ),
    Component(
        name="norm_elementwise",
        patterns=("rms_norm", "layer_norm", "add_rms", "elementwise", "vectorized_"),
        source="RMSNorm and residual adds",
    ),
    Component(
        name="quant",
        patterns=("quant", "scaled_fp8", "per_token_group", "dequant"),
        source="activation quantisation / scale application",
    ),
    Component(
        name="spec_decode",
        patterns=("draft", "dspark", "rejection", "accept", "verify"),
        source="DSpark draft + verification (directive P2)",
    ),
    Component(
        name="collective",
        patterns=("nccl", "rccl", "all_reduce", "allreduce", "ncclDevKernel"),
        source="TP all-reduce",
    ),
    Component(
        name="memory_movement",
        patterns=("memcpy", "memset", "copy_", "cat_", "contiguous"),
        source="transient allocation and copies",
    ),
)


@dataclass
class Attribution:
    """Result of mapping a :class:`KernelTotals` onto components."""

    by_component: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    total_us: float = 0.0
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    rest_kernels: list[tuple[str, float]] = field(default_factory=list)

    @property
    def rest_us(self) -> float:
        return self.by_component.get(REST, 0.0)

    @property
    def coverage(self) -> float:
        """Fraction of step GPU time assigned to a *named* component."""
        if self.total_us <= 0:
            return 0.0
        return (self.total_us - self.rest_us) / self.total_us

    def share(self, component: str) -> float | None:
        """This component's fraction of the whole step, or None if unknown."""
        if self.total_us <= 0 or component not in self.by_component:
            return None
        return self.by_component[component] / self.total_us

    def to_dict(self, rest_detail: int = 10) -> dict:
        ranked = sorted(self.by_component.items(), key=lambda kv: -kv[1])
        return {
            "total_us": round(self.total_us, 3),
            "coverage": round(self.coverage, 4),
            "unattributed_pct": round(100.0 * (1.0 - self.coverage), 2),
            "components": [
                {
                    "name": name,
                    "us": round(us, 3),
                    "pct": round(100.0 * us / self.total_us, 3) if self.total_us else 0.0,
                    "kernels": self.counts.get(name, 0),
                }
                for name, us in ranked
            ],
            # The most expensive things we could not name. This list is the
            # work item: every entry is either a missing pattern or a component
            # nobody has modelled yet.
            "rest_top": [
                {"name": n, "us": round(us, 3)}
                for n, us in sorted(self.rest_kernels, key=lambda kv: -kv[1])[:rest_detail]
            ],
            "ambiguous": {k: v for k, v in sorted(self.ambiguous.items())},
        }


def attribute(
    totals: KernelTotals,
    components: tuple[Component, ...] = DEFAULT_COMPONENTS,
) -> Attribution:
    """Assign every kernel to at most one component, keeping time conserved.

    A kernel matching several components is assigned to the **first** in
    declaration order — deterministic, and it keeps the sum equal to the step
    total so shares stay comparable. Every such collision is recorded in
    ``ambiguous``; a component whose share matters should not be sharing its
    kernels with another, and a populated ``ambiguous`` map is a signal to
    tighten the patterns rather than a detail to ignore.
    """
    out = Attribution(total_us=totals.total_us)
    for kernel, (us, _cnt) in totals.by_name.items():
        hits = [c.name for c in components if c.matches(kernel)]
        if hits:
            owner = hits[0]
            if len(hits) > 1:
                out.ambiguous[kernel] = hits
        else:
            owner = REST
            out.rest_kernels.append((kernel, us))
        out.by_component[owner] = out.by_component.get(owner, 0.0) + us
        out.counts[owner] = out.counts.get(owner, 0) + 1
    return out


def components_from_patterns(patterns: list[str], name: str = "query") -> tuple[Component, ...]:
    """Wrap a bare substring list as a single component.

    Bridges the evaluators' existing ``INDEXER_OPS``-style lists into this
    module without asking them to change.
    """
    return (Component(name=name, patterns=tuple(patterns)),)
