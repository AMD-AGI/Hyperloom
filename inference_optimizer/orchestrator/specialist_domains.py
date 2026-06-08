# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist sub-agent domain catalogue — v0.8 M5/M6.

Specialists are an *LLM* sub-agent form factor (distinct from the
deterministic Python executors in ``action_executors/``). Each specialist
is parameterized by a ``domain`` — a prompt-assembly dimension that maps
to:

* a knowledge-domain tag for advisory RecipeKB / prompt context,
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
* ``kb_anchor`` — legacy knowledge-domain label retained for prompt
  grouping and old data compatibility.
* ``pr_repos`` — repos the PR Monitor (M4) should pull recent PRs from
  for this domain.
* ``available_in`` — ``"M5"`` for serving_specialist, ``"M6"`` for the
  others; PolicyGate R2 currently accepts both groups but
  SpecialistRunner falls back to the generic template for M6-only
  domains until the M6 prompt PR lands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistDomain:
    """A single specialist domain entry in the canonical catalogue.

    Describes one specialist (serving, kernel, comm, compiler, system, etc.)
    that the Orchestrator can dispatch, including which source layer it reads,
    which KB anchor it maps to, and which upstream repos it scouts for PRs.

    Attributes:
        key (str): Stable identifier for the domain (e.g. ``serving_specialist``).
        layer (str): Human-readable description of the source/runtime layer it
            focuses on.
        kb_anchor (str): Knowledge-base anchor the domain is associated with.
        pr_repos (tuple[str, ...]): Upstream repositories scanned for relevant
            PRs. Defaults to an empty tuple.
        available_in (str): Milestone in which the domain becomes available
            (e.g. ``M5`` or ``M6``). Defaults to ``"M6"``.
        description (str): Free-form description of the domain's responsibilities.
            Defaults to an empty string.
        sub_kinds (tuple[str, ...]): Optional per-domain sub_kind catalogue. When
            empty, only ``params.sub_kind`` in {None, ""} is accepted; non-empty
            values are denied with a structured PolicyGate error. Defaults to an
            empty tuple.
    """

    key: str
    layer: str
    kb_anchor: str
    pr_repos: tuple[str, ...] = ()
    available_in: str = "M6"
    description: str = ""
    # Optional per-domain sub_kind catalogue. When empty (default) the
    # dispatch path accepts only ``params.sub_kind`` ∈ {None, ""}; non-
    # empty values are denied with a structured PolicyGate error.
    # Adding a new sub_kind is a one-line tuple append plus a prompt-
    # template entry in specialist_prompt_builder. (The former
    # ``framework_pr_scout`` sub_kind was removed when framework-agent
    # was promoted to the FRAMEWORK_PR phase.)
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
        # Upstream PR discovery for sglang/vllm gaps is no longer a
        # per-domain sub_kind — it runs as the standalone FRAMEWORK_PR
        # phase (PRELUDE → FRAMEWORK_PR → EXPLORE) driven by the
        # Coordinator, gated by ``SharedState.framework_phase_enabled``
        # (``--no-framework`` to skip).
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
            "EXPLORE-phase per-gap PR top-up. Surveys PRs across known "
            "repos and feeds refs to other specialists. The bulk pre-scan "
            "runs in the dedicated FRAMEWORK_PR phase; this domain is for "
            "narrow follow-ups discovered mid-EXPLORE. Dispatch sparingly "
            "(one every K rounds)."
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
    SpecialistDomain(
        key="research_scout_specialist",
        layer="proven-prior research / reference scripts / arch features",
        kb_anchor="research_scout",
        pr_repos=("ROCm/aiter", "sgl-project/sglang", "ROCm/vllm",
                  "triton-lang/triton", "ROCm/rccl", "NVIDIA/TensorRT-LLM"),
        available_in="M5",
        description=(
            "Read-only research collector dispatched at PRELUDE (and "
            "periodically during EXPLORE). Surveys reference launch "
            "scripts, model config.json architecture features, and "
            "cross-framework / NVIDIA PRs+blogs+MLPerf for proven "
            "optimizations, then writes prioritised research_hints with "
            "sources. Never benchmarks, applies patches, or decides "
            "KEEP/REVERT."
        ),
    ),
)


SPECIALIST_DOMAIN_KEYS: frozenset[str] = frozenset(
    d.key for d in SPECIALIST_DOMAINS
)


# Controlled knowledge-domain tag vocabulary. Derived from the distinct
# ``kb_anchor`` values in the catalogue so the tag set and the KB
# traversal anchors never drift. A specialist dispatch carries one or
# more of these tags; each tag selects a KB anchor + PR repo subset +
# focus template for prompt assembly and a stable attribution key for
# session breakdown.
#
# Adding a knowledge domain = give a catalogue entry a new ``kb_anchor``
# (or append ``EXTRA_KNOWLEDGE_DOMAIN_TAGS`` for anchors with no backing
# SpecialistDomain yet, e.g. forward-declared roles).
EXTRA_KNOWLEDGE_DOMAIN_TAGS: tuple[str, ...] = ()


def _derive_knowledge_domain_tags() -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for d in SPECIALIST_DOMAINS:
        anchor = (d.kb_anchor or "").strip()
        if anchor:
            seen.setdefault(anchor, None)
    for extra in EXTRA_KNOWLEDGE_DOMAIN_TAGS:
        tag = (extra or "").strip()
        if tag:
            seen.setdefault(tag, None)
    return tuple(seen.keys())


KNOWLEDGE_DOMAIN_TAGS: tuple[str, ...] = _derive_knowledge_domain_tags()
KNOWLEDGE_DOMAIN_TAG_SET: frozenset[str] = frozenset(KNOWLEDGE_DOMAIN_TAGS)


# Map each knowledge-domain tag back to a representative catalogue entry
# so prompt assembly can recover the focus template / pr_repos for a
# tag. The first catalogue entry that owns the anchor wins.
def _anchor_to_domain_map() -> dict[str, "SpecialistDomain"]:
    out: dict[str, SpecialistDomain] = {}
    for d in SPECIALIST_DOMAINS:
        anchor = (d.kb_anchor or "").strip()
        if anchor and anchor not in out:
            out[anchor] = d
    return out


_ANCHOR_TO_DOMAIN: dict[str, "SpecialistDomain"] = _anchor_to_domain_map()


def domain_for_tag(tag: str) -> "SpecialistDomain | None":
    """Return a representative catalogue entry for a knowledge-domain
    tag (matched first by ``kb_anchor``, then by ``key``)."""
    t = (tag or "").strip()
    if not t:
        return None
    hit = _ANCHOR_TO_DOMAIN.get(t)
    if hit is not None:
        return hit
    return get_domain(t)


def normalize_dispatch_tags(params: dict) -> list[str]:
    """Resolve a dispatch payload's tag list.

    Reads ``params.tags`` (a list of knowledge-domain tags). Falls back
    to the single ``params.domain`` alias (mapped to its ``kb_anchor``
    when it names a catalogue entry, else used verbatim) when ``tags``
    is absent. Order-preserving dedup; empty entries dropped.
    """
    raw = params.get("tags")
    tags: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            t = str(item or "").strip()
            if t:
                tags.append(t)
    if not tags:
        domain = str(params.get("domain") or "").strip()
        if domain:
            dom = get_domain(domain)
            tags.append((dom.kb_anchor or domain) if dom else domain)
    return list(dict.fromkeys(tags))

# Active set — domains whose prompt templates are fully wired in the
# specialist_prompt_builder. All six domains have per-domain focus
# templates, so the active set matches the catalogue and Orchestration
# can dispatch any of them without falling through to the generic
# template.
SPECIALIST_DOMAINS_M5: frozenset[str] = frozenset(
    d.key for d in SPECIALIST_DOMAINS
)


def get_domain(key: str) -> SpecialistDomain | None:
    """Return the catalogue entry for ``key`` or None when unknown.

    Args:
        key (str): The domain key to look up (e.g. ``serving_specialist``).

    Returns:
        SpecialistDomain | None: The matching catalogue entry, or None if no
        domain with that key exists.
    """
    for d in SPECIALIST_DOMAINS:
        if d.key == key:
            return d
    return None


# Maximum number of LLM turns a specialist may run. The stale-detection
# threshold is bounded at ``max_turns × per_turn_max_min × 1.5``
# (default ~10 minutes).
DEFAULT_SPECIALIST_MAX_TURNS: int = 8

# Hard cap so the LLM can't request ridiculous turn counts. PolicyGate
# R2's ``specialist_max_turns_excess`` rule enforces this.
SPECIALIST_MAX_TURNS_HARD_CAP: int = 16


__all__ = [
    "DEFAULT_SPECIALIST_MAX_TURNS",
    "EXTRA_KNOWLEDGE_DOMAIN_TAGS",
    "KNOWLEDGE_DOMAIN_TAGS",
    "KNOWLEDGE_DOMAIN_TAG_SET",
    "SPECIALIST_DOMAINS",
    "SPECIALIST_DOMAINS_M5",
    "SPECIALIST_DOMAIN_KEYS",
    "SPECIALIST_MAX_TURNS_HARD_CAP",
    "SpecialistDomain",
    "domain_for_tag",
    "get_domain",
    "normalize_dispatch_tags",
]
