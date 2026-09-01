# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist sub-agent domain catalogue.

LLM sub-agent form factor parameterized by a ``domain`` — a stable
id used across dispatch and prompt assembly. A runtime constant, not
per-domain yaml.

Field reference:

* ``key`` — canonical id used in ``delegate{params.domain}``.
* ``layer`` — short human label (analysis layer covered).
* ``kb_anchor`` — knowledge-domain label for prompt grouping.
* ``available_in`` — ``"M5"`` / ``"M6"``; both are dispatchable.
* ``llm_selectable`` — False for a domain only the Coordinator dispatches.

All specialists share the global :data:`PR_QUERY_REPOS` allowlist and query
the PR Monitor directly via ``mcp__pr_monitor__*`` tools.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistDomain:
    """A single specialist domain entry in the canonical catalogue.

    Describes one specialist (serving, kernel, comm, compiler, system, etc.)
    that the Orchestrator can dispatch, including which source layer it reads
    and which KB anchor it maps to.

    Attributes:
        key (str): Stable identifier for the domain (e.g. ``serving_specialist``).
        layer (str): Human-readable description of the source/runtime layer it
            focuses on.
        kb_anchor (str): Knowledge-base anchor the domain is associated with.
        available_in (str): Milestone in which the domain becomes available
            (e.g. ``M5`` or ``M6``). Defaults to ``"M6"``.
        description (str): Free-form description of the domain's responsibilities.
            Defaults to an empty string.
        default_mode (str): Default dispatch mode for this domain, either
            ``"patch"`` (authors patches) or ``"research"`` (investigation only).
            A ``"mode"`` key in the dispatch params still overrides this at call
            time; the field only changes the fall-back when no explicit mode is
            given.
    """

    key: str
    layer: str
    kb_anchor: str
    available_in: str = "M6"
    description: str = ""
    default_mode: str = "patch"
    #: Whether Orchestration may name this domain in a ``delegate``. False for
    #: the domains the Coordinator dispatches itself. The emit hint and its test
    #: both derive from this instead of each carrying a copy of the list.
    llm_selectable: bool = True


# Global allowlist of repos specialists may query via mcp__pr_monitor__*.
# Entries are matched verbatim against the PR Monitor's indexed repo keys
# (`pr_repos_list`), so the owner/name casing here must track the server's.
PR_QUERY_REPOS: tuple[str, ...] = (
    "sgl-project/sglang",
    "ROCm/vllm",
    "ROCm/rccl",
    "NVIDIA/nccl",
    "pytorch/pytorch",
    "ROCm/ROCm",
    "ROCm/hip",
    "NVIDIA/TensorRT-LLM",
    "ROCm/aiter",
    "ROCm/ATOM",
    "ROCm/FlyDSL",
    "triton-lang/triton",
    "vllm-project/vllm",
)


# Canonical catalogue of knowledge-domain anchors.
SPECIALIST_DOMAINS: tuple[SpecialistDomain, ...] = (
    SpecialistDomain(
        key="serving_specialist",
        layer="sglang / vllm scheduler / cuda_graph / kv_cache",
        kb_anchor="framework",
        available_in="M5",
        description=(
            "Reads sglang/vllm source, focuses on scheduler, cuda graph, "
            "kv cache, batching, chunked prefill, max-num-seqs."
        ),
    ),
    SpecialistDomain(
        key="kernel_switch_specialist",
        layer="aiter / sglang kernels / triton",
        kb_anchor="kernel_agent",
        available_in="M6",
        description=(
            "Reads aiter / sglang kernels / triton source; focuses on attention, MoE, GEMM, fused attention paths."
        ),
    ),
    SpecialistDomain(
        key="comm_specialist",
        layer="RCCL / NCCL / QuickReduce / AllReduce",
        kb_anchor="communication",
        available_in="M6",
        description=("Focuses on collective communication, allreduce algorithms, QuickReduce, topology."),
    ),
    SpecialistDomain(
        key="compiler_specialist",
        layer="torch.compile / inductor / triton",
        kb_anchor="compiler",
        available_in="M6",
        description=("Focuses on torch.compile, inductor, triton codegen, AMDGCN, register pressure."),
    ),
    SpecialistDomain(
        key="system_specialist",
        layer="KFD / driver / memory / dispatch overhead",
        kb_anchor="systems",
        available_in="M6",
        description=(
            "Fixes launch latency, dispatch overhead, device "
            "synchronization and host-blocking calls; tunes KFD/driver "
            "env vars, numactl, HSA_ENABLE_SDMA, memory fragmentation."
        ),
    ),
    SpecialistDomain(
        key="candidate_discovery_specialist",
        layer="upstream candidate discovery / ranking / audit",
        kb_anchor="pr_intelligence",
        available_in="M6",
        default_mode="research",
        description=(
            "Finds upstream work worth landing. Surveys PRs across the "
            "allowlisted repos for the live bottleneck, ranks what it finds "
            "against the stack and the already-tried ledger, and judges each "
            "candidate: already present, not applicable, or worth a bench and "
            "by which route. Writes candidates to the ledger; Orchestration "
            "then proposes integrate_patch for the ones it wants. A first-class "
            "lever alongside configuration search, not an occasional top-up."
        ),
    ),
    SpecialistDomain(
        key="research_scout_specialist",
        layer="proven-prior research / reference scripts / arch features",
        kb_anchor="research_scout",
        available_in="M5",
        default_mode="research",
        description=(
            "Read-only research collector dispatched at PRELUDE (and "
            "periodically during the optimisation phase). Surveys reference launch "
            "scripts, model config.json architecture features, and "
            "cross-framework / NVIDIA PRs+blogs+MLPerf for proven "
            "optimizations, then writes prioritised research_hints with "
            "sources. Never benchmarks, applies patches, or decides "
            "KEEP/REVERT."
        ),
    ),
    SpecialistDomain(
        key="static_recon_specialist",
        layer="framework source static reconnaissance / un-bridged switches",
        kb_anchor="static_recon",
        available_in="M6",
        default_mode="research",
        description=(
            "Read-only static-source reconnaissance dispatched at PRELUDE. "
            "Greps the framework source tree (vLLM / SGLang) for un-bridged "
            "capability switches — fast paths that should be enabled for the "
            "current (model, GPU, precision) but are silently disabled by a "
            "predicate (e.g. a CUDA-only *_supported() returning False on "
            "ROCm). Seeded with a curated checklist; emits bridge-patch "
            "candidates as gap seeds. Never benchmarks, applies patches, or "
            "decides KEEP/REVERT — the freeform specialist authors the "
            "actual patch under the normal KEEP gate."
        ),
    ),
    SpecialistDomain(
        key="enablement_specialist",
        llm_selectable=False,
        layer="non-runnable or eval-failing (model, backend) enablement / framework + ROCm/HIP bridging",
        kb_anchor="framework",
        available_in="M6",
        description=(
            "Authoring specialist for the ENABLEMENT objective: makes a "
            "(model, backend) combo that is non-runnable, or that boots but fails "
            "its accuracy eval, *run correctly*. Given a "
            "structured failure signature (missing model arch, unsupported "
            "dtype, missing HIP kernel, import/build error, shape mismatch, "
            "not-implemented, accuracy below floor, eval runtime failure) and "
            "ranked bridging PRs, it authors a bridging "
            "patch into an isolated worktree — editing the framework source, "
            "and, when the framework layer cannot bridge it, /opt/rocm / HIP / "
            "aiter source. Gated on RUNNABILITY (server boots + minimal "
            "correctness) or, for eval-origin, meeting the accuracy floor; NOT "
            "throughput. Distinct from static_recon "
            "(which only finds already-runnable-but-disabled fast paths)."
        ),
    ),
    SpecialistDomain(
        key="framework_rewrite_specialist",
        layer="iterative-model pipeline source rewrites (diffusion / autoregressive video)",
        kb_anchor="framework",
        available_in="M6",
        description=(
            "Authoring specialist for framework-level source rewrites on an "
            "ITERATIVE model pipeline — a diffusion or autoregressive rollout "
            "that runs the same transformer stack once per block per denoising "
            "step per chunk. The wins there are not the serving concerns "
            "serving_specialist targets (there is no scheduler, no continuous "
            "batching, no KV-cache admission policy); they are redundant work "
            "the loop structure creates: step-invariant computations repeated "
            "every step, collectives that round-trip through the host to agree "
            "on a shape, tables rebuilt on the host and re-uploaded, adjacent "
            "same-shape collectives that could be one. Works from measured "
            "host-side evidence plus a rewrite-pattern taxonomy, and must "
            "deliver every rewrite behind a default-off environment switch with "
            "a declared manifest so each one can be attributed and composed "
            "independently. Distinct from serving_specialist (request-serving "
            "frameworks) and kernel_switch_specialist (operator kernels)."
        ),
    ),
)


SPECIALIST_DOMAIN_KEYS: frozenset[str] = frozenset(d.key for d in SPECIALIST_DOMAINS)


# Extra knowledge-domain tags for anchors with no backing SpecialistDomain yet.
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
    tag (matched first by ``kb_anchor``, then by ``key``).

    Args:
        tag: The knowledge-domain tag to look up.

    Returns:
        The matching catalogue entry, or ``None`` when the tag is empty or
        unknown.
    """
    t = (tag or "").strip()
    if not t:
        return None
    hit = _ANCHOR_TO_DOMAIN.get(t)
    if hit is not None:
        return hit
    return get_domain(t)


def _tag_to_kb_anchor(tag: str) -> str:
    """Translate one dispatch tag to its knowledge-domain anchor.

    The catalogue has two vocabularies: the domain ``key``
    (``serving_specialist`` …) and the ``kb_anchor`` the PolicyGate whitelist
    validates against (``framework`` …). A tag naming a domain **key** is
    translated to that domain's anchor; a tag that is already a valid anchor is
    kept as-is; anything else is returned verbatim rather than invented into
    an anchor, so an unresolvable tag stays visible downstream.

    Args:
        tag: A single (already-stripped, non-empty) dispatch tag.

    Returns:
        The tag's ``kb_anchor`` when it names a catalogue key, the tag itself
        when it is already a valid anchor, else the tag verbatim.
    """
    dom = get_domain(tag)
    if dom is not None:
        return dom.kb_anchor or tag
    return tag


def normalize_dispatch_tags(params: dict) -> list[str]:
    """Resolve a dispatch payload's tag list, translating domain keys to anchors.

    Reads ``params.tags`` (each element is translated key→``kb_anchor`` via
    :func:`_tag_to_kb_anchor`); falls back to the ``params.domain`` alias
    (same translation) when absent. Order-preserving dedup; empty entries
    dropped. Genuinely unknown tags pass through untranslated.

    Args:
        params: The dispatch payload (reads ``tags`` then ``domain``).

    Returns:
        The resolved, order-preserving deduped anchor list.
    """
    raw = params.get("tags")
    tags: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            t = str(item or "").strip()
            if t:
                tags.append(_tag_to_kb_anchor(t))
    if not tags:
        domain = str(params.get("domain") or "").strip()
        if domain:
            tags.append(_tag_to_kb_anchor(domain))
    return list(dict.fromkeys(tags))


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


def authoring_domain_for_framework(framework: str | None) -> str:
    """Return the authoring domain that matches a framework's kind.

    A request-serving framework and an iterative model pipeline have almost no
    optimization surface in common: one is scheduling, batching and KV-cache
    admission, the other is redundant work created by running the same stack
    once per block per denoising step. Steering a scriptable pipeline at
    ``serving_specialist`` points it at a hot path that does not exist there, so
    every place that names an authoring domain resolves it through here rather
    than hard-coding one.

    Args:
        framework (str | None): The session's framework name.

    Returns:
        str: ``"framework_rewrite_specialist"`` for a scriptable (server-less)
        framework, else ``"serving_specialist"``.
    """
    name = str(framework or "").strip().lower()
    if not name:
        return "serving_specialist"
    from hyperloom.inference_optimizer import framework_registry

    return "framework_rewrite_specialist" if framework_registry.is_scriptable(name) else "serving_specialist"


# Synthetic domain for ``scope='freeform'`` dispatches. NOT part of
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


# Default number of LLM turns a specialist may run. Practically unbounded so the
# ``max_seconds = max_turns × per_turn`` ceiling is decoupled from turns; the
# real stop is the wall-clock budget.
DEFAULT_SPECIALIST_MAX_TURNS: int = 1000

# Hard cap; PolicyGate denies a dispatch above it because the in-process
# backend's turn loop has no wall-clock bound.
SPECIALIST_MAX_TURNS_HARD_CAP: int = 1000


__all__ = [
    "DEFAULT_SPECIALIST_MAX_TURNS",
    "EXTRA_KNOWLEDGE_DOMAIN_TAGS",
    "FREEFORM_DOMAIN",
    "KNOWLEDGE_DOMAIN_TAGS",
    "KNOWLEDGE_DOMAIN_TAG_SET",
    "PR_QUERY_REPOS",
    "SPECIALIST_DOMAINS",
    "SPECIALIST_DOMAIN_KEYS",
    "SPECIALIST_MAX_TURNS_HARD_CAP",
    "SpecialistDomain",
    "domain_for_tag",
    "get_domain",
    "normalize_dispatch_tags",
]
