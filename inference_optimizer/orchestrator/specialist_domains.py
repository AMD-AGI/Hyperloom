"""Specialist sub-agent domain catalogue — v0.8 M5/M6.

Specialists are an *LLM* sub-agent form factor (distinct from the
deterministic Python executors in ``action_executors/``). Each specialist
is parameterized by a ``domain`` — a prompt-assembly dimension that maps
to:

* a Cortex KB sub-graph anchor (``kernel.*`` / ``framework.*`` / …),
* a PR Monitor repo subset (M4),
* a default tool-call hint set,
* a stable id used by PolicyGate R2 + breakdown ``specialist_runs``.

The catalogue is intentionally a runtime constant rather than per-domain
yaml: §3.5 §5 says "domain 是 prompt 装配维度, 不是新 IntentType, 不是
新 Role" — adding a domain is a one-line change here plus a prompt
template entry in ``specialist_prompt_builder.py``.

M5 ships only ``serving_specialist`` (per §3.13 M5 §2 scope); the
other five (kernel/comm/compiler/system/pr_intel) are listed here so
PolicyGate R2 already knows their identifiers but the prompt builder
falls back to a *generic* template until M6 lands per-domain prompts.

Field reference:

* ``key`` — canonical id used in ``delegate{params.domain}`` and
  ``specialist_done{payload.domain}``.
* ``layer`` — short human label (analysis layer the specialist cares about).
* ``kb_anchor`` — Cortex KB top-level domain to traverse on prompt
  assembly (M4/M5).
* ``pr_repos`` — repos the PR Monitor (M4) should pull recent PRs from
  for this domain.
* ``available_in`` — ``"M5"`` for serving_specialist, ``"M6"`` for the
  others; PolicyGate R2 currently accepts both groups but
  SpecialistRunner falls back to the generic template for M6-only
  domains until the M6 prompt PR lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecialistDomain:
    key: str
    layer: str
    kb_anchor: str
    pr_repos: tuple[str, ...] = ()
    available_in: str = "M6"
    description: str = ""
    # F2-2 (Roofline-v2 / framework-agent): optional per-domain
    # sub_kind catalogue. When empty (default) the dispatch path
    # accepts only ``params.sub_kind`` ∈ {None, ""}; non-empty values
    # are denied with a structured PolicyGate error. ``serving_specialist``
    # currently lists ``framework_pr_scout`` so the orchestration LLM
    # can spawn a PR-discovery sub-task without inventing a new
    # top-level action (Iron-rule §4: specialist-first). Adding more
    # sub_kinds is a one-line tuple append plus a prompt-template
    # entry in specialist_prompt_builder.
    sub_kinds: tuple[str, ...] = ()


# Canonical catalogue. Adding a new domain is a one-line append plus
# (optionally) a prompt template entry. PolicyGate R2's
# `specialist_unknown_domain` rule reads this set.
SPECIALIST_DOMAINS: tuple[SpecialistDomain, ...] = (
    SpecialistDomain(
        key="serving_specialist",
        layer="sglang / vllm scheduler / cuda_graph / kv_cache",
        kb_anchor="framework",
        pr_repos=("sgl-project/sglang", "ROCm/vllm"),
        available_in="M5",
        description=(
            "Reads sglang/vllm source, focuses on scheduler, cuda graph, "
            "kv cache, batching, chunked prefill, max-num-seqs."
        ),
        # ``framework_pr_scout`` lets the specialist invoke
        # ``fa candidates`` + ``git fetch refs/pull/<N>/head`` to
        # discover an upstream sglang/vllm PR addressing a known gap.
        # Gated end-to-end on ``SharedState.framework_agent_enabled``;
        # PolicyGate denies the dispatch when the toggle is off.
        sub_kinds=("framework_pr_scout",),
    ),
    SpecialistDomain(
        key="kernel_switch_specialist",
        layer="aiter / sglang kernels / triton",
        kb_anchor="kernel",
        pr_repos=("ROCm/aiter", "triton-lang/triton"),
        available_in="M6",
        description=(
            "Reads aiter / sglang kernels / triton source; focuses on "
            "attention, MoE, GEMM, fused attention paths."
        ),
    ),
    SpecialistDomain(
        key="comm_specialist",
        layer="RCCL / NCCL / QuickReduce / AllReduce",
        kb_anchor="communication",
        pr_repos=("ROCm/rccl", "nvidia/nccl"),
        available_in="M6",
        description=(
            "Focuses on collective communication, allreduce algorithms, "
            "QuickReduce, topology."
        ),
    ),
    SpecialistDomain(
        key="compiler_specialist",
        layer="torch.compile / inductor / triton",
        kb_anchor="compiler",
        pr_repos=("triton-lang/triton", "pytorch/pytorch"),
        available_in="M6",
        description=(
            "Focuses on torch.compile, inductor, triton codegen, AMDGCN, "
            "register pressure."
        ),
    ),
    SpecialistDomain(
        key="system_specialist",
        layer="KFD / driver / 内存 / dispatch overhead",
        kb_anchor="systems",
        pr_repos=("ROCm/ROCm",),
        available_in="M6",
        description=(
            "Fixes launch latency, dispatch overhead, device "
            "synchronization and host-blocking calls; tunes KFD/driver "
            "env vars, numactl, HSA_ENABLE_SDMA, memory fragmentation."
        ),
    ),
    SpecialistDomain(
        key="pr_intel_specialist",
        layer="cross-repo PR research",
        kb_anchor="pr_intelligence",
        pr_repos=("ROCm/aiter", "sgl-project/sglang", "ROCm/vllm",
                  "triton-lang/triton", "ROCm/rccl"),
        available_in="M6",
        description=(
            "Surveys PRs across all known repos; provides PR refs to "
            "other specialists. Dispatch sparingly (one every K rounds)."
        ),
    ),
    SpecialistDomain(
        key="session_steward_specialist",
        layer="session strategy / remaining-leverage assessment",
        kb_anchor="pr_intelligence",
        pr_repos=(),
        available_in="M5",
        description=(
            "Honest end-of-EXPLORE assessor. Reads optimization_stack, "
            "explore_search.rejected, gaps[], policy_denial_history; "
            "recommends continue_explore / advance_to_kernel / stop_session. "
            "Dispatched by the Coordinator on plateau (not by the LLM); "
            "constrained to one continuation per session before the "
            "EXPLORE→KERNEL transition becomes mandatory."
        ),
    ),
)


SPECIALIST_DOMAIN_KEYS: frozenset[str] = frozenset(
    d.key for d in SPECIALIST_DOMAINS
)

# M5 active set — domains whose prompt templates are fully wired
# in the specialist_prompt_builder. PR-A6 (Arbor-into-Hyperloom)
# added per-domain focus templates for ``kernel_switch_specialist`` /
# ``comm_specialist`` / ``compiler_specialist`` / ``system_specialist`` /
# ``pr_intel_specialist`` (previously M6-only fallbacks), so the M5
# active set now matches the catalogue and Orchestration can dispatch
# any of the six without falling through to the generic template.
SPECIALIST_DOMAINS_M5: frozenset[str] = frozenset(
    d.key for d in SPECIALIST_DOMAINS
)


def get_domain(key: str) -> SpecialistDomain | None:
    """Return the catalogue entry for ``key`` or None when unknown."""
    for d in SPECIALIST_DOMAINS:
        if d.key == key:
            return d
    return None


# F2-2: sub_kinds that require ``SharedState.framework_agent_enabled=True``.
# Centralised here (not per-rule) so PolicyGate / SpecialistRunner can
# share the membership test without duplicating string literals.
FRAMEWORK_AGENT_GATED_SUB_KINDS: frozenset[str] = frozenset({
    "framework_pr_scout",
})


# Maximum number of LLM turns a specialist may run. KB_design §3.5 §9
# bounds the stale-detection threshold at ``max_turns × per_turn_max_min
# × 1.5`` (default ~10 minutes). 8 is the M5 default per §3.13 M5 §5.
DEFAULT_SPECIALIST_MAX_TURNS: int = 8

# Hard cap so the LLM can't request ridiculous turn counts. PolicyGate
# R2's ``specialist_max_turns_excess`` rule enforces this.
SPECIALIST_MAX_TURNS_HARD_CAP: int = 16


__all__ = [
    "DEFAULT_SPECIALIST_MAX_TURNS",
    "SPECIALIST_DOMAINS",
    "SPECIALIST_DOMAINS_M5",
    "SPECIALIST_DOMAIN_KEYS",
    "SPECIALIST_MAX_TURNS_HARD_CAP",
    "SpecialistDomain",
    "get_domain",
]
