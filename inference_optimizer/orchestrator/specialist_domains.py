# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist sub-agent domain catalogue — v0.8 M5/M6.

LLM sub-agent form factor parameterized by a ``domain`` (§3.5 §5) — a stable
id used by PolicyGate R2. A runtime constant, not per-domain yaml.

Field reference:

* ``key`` — canonical id used in ``delegate{params.domain}``.
* ``layer`` — short human label (analysis layer covered).
* ``kb_anchor`` — knowledge-domain label for prompt grouping.
* ``pr_repos`` — repos the PR Monitor pulls recent PRs from.
* ``available_in`` — ``"M5"`` / ``"M6"``; PolicyGate R2 accepts both.
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
    # Optional per-domain sub_kind catalogue; empty accepts only ``params.sub_kind`` ∈ {None, ""}.
    sub_kinds: tuple[str, ...] = ()


# Canonical catalogue; PolicyGate R2's `specialist_unknown_domain` rule reads this set.
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
        # Upstream PR discovery runs as the standalone FRAMEWORK_PR phase.
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
        layer="KFD / driver / memory / dispatch overhead",
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


# Controlled knowledge-domain tag vocabulary derived from distinct
# ``kb_anchor`` values so tags and KB anchors never drift. Append here for
# anchors with no backing SpecialistDomain yet.
EXTRA_KNOWLEDGE_DOMAIN_TAGS: tuple[str, ...] = ()


def _derive_knowledge_domain_tags() -> tuple[str, ...]:
    """Collect the distinct knowledge-domain tags from the catalogue.

    Combines the ``kb_anchor`` of each specialist domain with any extra
    standalone tags, preserving first-seen order and dropping blanks.

    Returns:
        Ordered tuple of unique knowledge-domain tags.
    """
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


# Map each knowledge-domain tag back to a representative catalogue entry.
# The first catalogue entry that owns the anchor wins.
def _anchor_to_domain_map() -> dict[str, "SpecialistDomain"]:
    """Build a map from KB anchor to its representative domain entry.

    The first catalogue entry that owns a given anchor wins.

    Returns:
        Mapping of ``kb_anchor`` to the owning :class:`SpecialistDomain`.
    """
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

    Reads ``params.tags``; falls back to the ``params.domain`` alias (mapped
    to its ``kb_anchor`` when it names a catalogue entry, else verbatim) when
    absent. Order-preserving dedup; empty entries dropped.
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

# Active set — domains with fully-wired prompt templates. Matches the
# full catalogue, so Orchestration never falls through to the generic template.
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


# Synthetic domain for ``scope='freeform'`` dispatches (absorbed from the
# retired dynamic_specialist wave channel). It is intentionally NOT part of
# SPECIALIST_DOMAINS / the knowledge-domain vocabulary — it exists only to
# satisfy the runner's Domain contract. The real mandate is carried by
# ``params.task_description`` and rendered by the free-form prompt block.
FREEFORM_DOMAIN: SpecialistDomain = SpecialistDomain(
    key="freeform_specialist",
    layer="(free-form — not bound to the domain catalogue)",
    kb_anchor="framework",
    available_in="M6",
    description=(
        "Free-form specialist: not bound to the domain catalogue. The "
        "Orchestration task_description is the whole mandate."
    ),
)


# Default number of LLM turns a specialist may run.
DEFAULT_SPECIALIST_MAX_TURNS: int = 12

# Hard cap (PolicyGate R2 ``specialist_max_turns_excess`` enforces this).
SPECIALIST_MAX_TURNS_HARD_CAP: int = 16


__all__ = [
    "DEFAULT_SPECIALIST_MAX_TURNS",
    "EXTRA_KNOWLEDGE_DOMAIN_TAGS",
    "FREEFORM_DOMAIN",
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
