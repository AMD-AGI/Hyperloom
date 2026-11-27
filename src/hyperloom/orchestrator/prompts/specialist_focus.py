# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-domain focus blocks the specialist system prompt injects into its identity section."""

from __future__ import annotations

from typing import Callable

from hyperloom.common.perf_metric import is_agentx_mode
from .speculative_decoding_focus import speculative_decoding_focus
from ..specialists.prompt_inputs import SpecialistPromptInputs


# Curated launch-recipe sites the research scout mines for verified serve
# flags / envs, keyed by (model x hardware x quant x strategy). Overridable
# via HYPERLOOM_RECIPE_SITES (comma/space separated); values are advisory
# templates, not fetched by the Coordinator.
DEFAULT_RECIPE_SITES: tuple[str, ...] = (
    "https://recipes.vllm.ai/<org>/<model>?hardware=<gpu>",
    "https://lmsysorg.mintlify.app/cookbook/autoregressive/<family>/<model>",
)

# Operator sentinels (via HYPERLOOM_RECIPE_SITES) that disable recipe-site guidance.
RECIPE_SITES_DISABLED_VALUES: frozenset[str] = frozenset({"none", "off", "disable", "disabled"})


# Per-domain focus templates: each injects a "Domain focus" block into
# Section 1; a missing key falls back to the generic body.


def _is_agentx(inp: SpecialistPromptInputs) -> bool:
    """True when the session replays the AgentX agentic trace corpus.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        True when ``benchmark_mode`` names the agentic workload.
    """
    return is_agentx_mode(inp.benchmark_mode)


def _focus_candidate_discovery_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Build the domain-focus block for the candidate-discovery specialist.

    Frames it as the owner of the whole upstream-candidate funnel: find, rank,
    and judge. It replaced a Coordinator loop that did those three steps with
    a fixed query, a scoring call and an audit call, so the block has to say
    that a candidate is only useful once it carries a verdict and a route.

    Args:
        inp (SpecialistPromptInputs): The prompt inputs for this dispatch.

    Returns:
        list[str]: Markdown lines for the discovery specialist's focus block.
    """
    return [
        "You own the **upstream candidate funnel**: find what is worth landing,",
        "rank it, and judge it. This is a first-class lever alongside",
        "configuration search — not an occasional top-up.",
        "",
        "**What to do**",
        "1. **Find.** Use ``mcp__pr_monitor__*`` + ``WebSearch`` across the",
        "   allowlisted repos for work that addresses the live bottleneck.",
        "   Read the gap and the profile evidence first; a PR that does not",
        "   touch the hot path is not a candidate however good it looks.",
        "2. **Rank.** Order what you found against the current stack, the",
        "   already-tried ledger, and the KB priors you were given. Say why",
        "   the top one is first — the ordering is the deliverable, not a",
        "   by-product.",
        "3. **Judge each one.** Exactly one verdict per candidate:",
        "   - ``already_present`` — the change is already in the installed",
        "     version. Cite the evidence; a guess here silently skips a real win.",
        "   - ``not_applicable`` — wrong framework, wrong arch, or it cannot",
        "     apply to this tree.",
        "   - ``worth_a_bench`` — plus the route: apply the upstream diff",
        "     directly, or have a specialist author against it as inspiration.",
        "     Direct apply needs a git tree and a same-repo candidate; say so",
        "     when either is missing.",
        "",
        "**What to return**",
        "- ``proposal_set`` entries in the upstream-candidate schema of",
        "  Section 8, one per candidate, each with its verdict and route.",
        "  Orchestration reads these and proposes ``integrate_patch`` for the",
        "  ones it wants benched; you do not apply or benchmark anything.",
        "",
        "**Pitfalls**",
        "- Citing a PR without verifying its target framework matches the",
        "  current install.",
        "- An unevidenced ``already_present``: it is the one verdict that",
        "  discards a candidate without ever measuring it.",
        "- Returning candidates with no ordering and no verdicts. A bare list",
        "  puts the work back on the caller that dispatched you.",
    ]


def _recipe_sites_source_lines(inp: SpecialistPromptInputs) -> list[str]:
    """Render the recipe-site research source; the built-in defaults when unset, nothing when disabled via the sentinel."""
    configured = tuple(s for s in inp.recipe_sites if s)
    if configured and all(s.strip().lower() in RECIPE_SITES_DISABLED_VALUES for s in configured):
        return []
    sites = configured or DEFAULT_RECIPE_SITES
    if not sites:
        return []
    lines = [
        "4. **Verified launch-recipe sites** — structured per",
        "   (model x hardware x quant x strategy) recipe pages carrying",
        "   validated serve flags, env vars, and benchmark numbers. Use",
        "   ``WebFetch`` on the page matching THIS model / GPU / precision",
        "   (fall back to ``WebSearch`` if the exact page 404s). Extract only",
        "   the serve flags, env vars, and reported throughput/accuracy;",
        "   emit them as ``proposal_set`` variants with the page URL in",
        "   ``source``. For a near-miss hardware/quant match, still surface it",
        "   but note the mismatch in ``accuracy_risk``. Sites:",
    ]
    lines.extend(f"   - {site}" for site in sites)
    lines.append("")
    return lines


def _focus_research_scout_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the focus section for the research-scout specialist prompt.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines highlighting already-proven priors and steering the
        scout toward net-new findings.
    """
    proven_lines: list[str] = []
    if inp.already_proven:
        proven_lines.append("**Already proven (warm-start recipe) — do NOT re-mine these; focus on net-new priors:**")
        for item in inp.already_proven[:12]:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            src = str(item.get("source") or "").strip()
            proven_lines.append(f"- {name}" + (f" (source={src})" if src else ""))
        proven_lines.append("")
    return [
        "You are the **research scout** — a read-only collector of",
        "*already-proven* priors. You do NOT benchmark, apply patches, or",
        "decide KEEP/REVERT. Your single deliverable is a prioritised list",
        "of source-backed findings and executable variants.",
        "",
        *proven_lines,
        "**Three research sources (cover all that are reachable)**",
        "1. **Reference launch scripts** — look under",
        "   ``$INFERENCEX_PATH/benchmarks/single_node/`` for scripts",
        "   matching this (model, GPU). Extract every validated env/flag",
        "   and the throughput it reached. ``$INFERENCEX_PATH`` may be",
        "   unset — skip this source silently if so.",
        "2. **Model architecture features** — read the model's",
        "   ``config.json`` (MTP ``num_nextn_predict_layers``, MoE expert",
        "   count / routing, attention type MLA/GQA, quantization support)",
        "   and infer optimizations those features unlock.",
        "3. **Cross-framework / NVIDIA research** — survey PRs, blogs, and",
        "   MLPerf results across frameworks and NVIDIA/TRT-LLM via",
        "   ``WebSearch`` / ``mcp__pr_monitor__*`` for proven wins. Avoid",
        "   re-listing PRs the FRAMEWORK_AGENT phase already covered.",
        "",
        *_recipe_sites_source_lines(inp),
        "**Gap computation** — where you find a reference throughput, use",
        "the gap versus our current baseline only to prioritise your hints",
        "(a bigger gap means a higher-priority hint). Do NOT emit competitor",
        "numbers as a structured target: measured competitor baselines are",
        "sourced from InferenceX, never authored by this scout.",
        "",
        "**Output protocol** — use only the top-level ``specialist_done`` fields:",
        "- ``proposal_set``: executable variants using the standard explore",
        "  schema. Put inspected PRs and references in ``pr_evidence`` or",
        "  ``source_evidence`` on each proposal.",
        "- ``new_findings``: list of ``{what, expected_impact, accuracy_risk,",
        "  source, domain_tags[]}``. ``source`` is REQUIRED (PR link / blog /",
        "  MLPerf row / reference script path).",
        "- ``residual_questions``: unanswered questions for the next scout round.",
        "",
        "**Iron rule** — read-only. Never write a patch, never launch a",
        "benchmark, never recommend a phase transition. Turn proven priors",
        "into structured hints and stop.",
    ]


_STATIC_RECON_FOCUS_METHOD: tuple[str, ...] = (
    "**How to hunt (read the LIVE source under the source roots / hint",
    "directories in Section 7):**",
    "1. grep for capability predicates — ``*_supported()`` /",
    "   ``*_enabled()`` / ``is_*()`` guards and feature gates in the",
    "   quantization, linear, fused_moe and attention dispatch paths.",
    "2. For each, determine whether it returns False (or routes to a slow",
    "   fallback) on THIS hardware/precision when a faster path exists",
    "   (e.g. a CUDA-only ``cutlass_*_supported()`` that is always False on",
    "   ROCm and so disqualifies an AITER kernel).",
    "3. Trace the consequence: which GEMM / kernel / backend the code then",
    "   falls back to, and why that is slower.",
    "4. Confirm the bridge is plausible (the faster path exists and only",
    "   the guard / scale-granularity / activation-config blocks it).",
    "",
    "**Output protocol** — emit ONE ``specialist_done`` carrying a",
    "``recon`` block with ``bridge_candidates``: a list of",
    "``{id, predicate_file, predicate_name, why_disabled_here, consequence,",
    "bridge_sketch, domain_hint, confidence}``. ``id`` is a short slug",
    "(reuse the seed checklist id when verifying one), ``predicate_file`` is",
    "the source path you read, ``why_disabled_here`` explains the False",
    "branch on this hardware, ``bridge_sketch`` is the proposed fix (a",
    "sketch — you do NOT write the patch), and ``domain_hint`` is the",
    "specialist that should author it (``freeform`` keeps the whole",
    "mandate). A candidate without ``predicate_file`` + ``why_disabled_here``",
    "is dropped.",
    "",
    "**Iron rule** — read-only. Never write a patch, never launch a",
    "benchmark, never recommend a phase transition. Turn verified source",
    "findings into structured bridge candidates and stop.",
)


def _focus_static_recon_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the focus section for the static-recon specialist prompt.

    Steers a read-only sub-agent to grep the framework source tree for
    un-bridged capability switches (predicates that silently disable a fast
    path for the current GPU/precision), seeded with a curated checklist, and to
    emit structured bridge candidates rather than patches.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines describing the recon task, the seed checklist, and the
        ``recon`` output block schema.
    """
    checklist_lines: list[str] = []
    if inp.static_recon_checklist:
        checklist_lines = [
            "**Seed checklist (known un-bridged switches for this "
            + "(model, GPU, precision)) — verify each against the LIVE source; "
            + "do not assume it still applies:**",
            inp.static_recon_checklist,
            "",
        ]
    model_info_line = ""
    shared_expert_advisory: list[str] = []
    if inp.model_info:
        attn = str(inp.model_info.get("attention_type") or "").strip()
        is_moe = bool(inp.model_info.get("is_moe"))
        quant = str(inp.model_info.get("quantization") or "").strip()
        has_shared = bool(inp.model_info.get("has_shared_expert"))
        num_shared = inp.model_info.get("num_shared_experts")
        features = f"attention={attn or '?'} moe={is_moe}"
        if has_shared:
            n_str = str(int(num_shared)) if num_shared is not None else "?"
            features += f" shared_expert=True n_shared={n_str}"
        features += f" quant={quant or '?'}."
        model_info_line = f"Model features: {features}"
        if has_shared:
            shared_expert_advisory = [
                "**Shared-expert fusion advisory**: this model has always-on shared "
                + "experts. Confirm whether the shared expert still runs as a separate "
                + "dense MLP per layer. If yes, investigate folding it into the routed "
                + "grouped-GEMM path as an always-selected extra expert slot (code-path "
                + "bridge, not just an env flag). Known caveat: expert parallelism (EP) "
                + "is unsupported until the expert-map behaviour is explicitly handled.",
                "",
            ]
    return [
        "You are the **static-recon specialist** — a read-only reconnaissance",
        "agent. You do NOT benchmark, apply patches, build a worktree, or",
        "decide KEEP/REVERT. Your single deliverable is a prioritised list of",
        "**bridge candidates**: fast paths that *should* be enabled for this",
        f"GPU ({inp.gpu_type or '?'}) + precision ({inp.precision or '?'}) but",
        "are silently disabled in the LIVE framework source.",
        "",
        *([model_info_line, ""] if model_info_line else []),
        *shared_expert_advisory,
        *checklist_lines,
        *_STATIC_RECON_FOCUS_METHOD,
    ]


def _focus_enablement_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Stable enablement-specialist identity blurb.

    The per-task mandate (failure context + the ladder book) is rendered
    separately into the user prompt by ``_section_enablement_playbook`` so this
    system-prompt block stays constant across dispatches (cacheable).

    Args:
        inp: Assembled prompt inputs for the current dispatch (unused; the
            identity is task-independent).

    Returns:
        Prompt lines for the enablement-specialist identity.
    """
    return [
        "You are the **enablement specialist** — an AUTHORING sub-agent for a",
        "(model, backend) combo that is non-runnable OR that boots but fails its",
        "accuracy eval. The gate is RUNNABILITY (the server boots and passes a",
        "minimal inference) or, for an eval-origin dispatch, the real model output",
        "meeting the accuracy floor — not throughput.",
        "",
        "Your deliverable is the smallest **runnable delta** that advances the",
        "boot (or the accuracy) — which may be a serve flag, an in-tree source patch, a",
        "``needs_runtime`` request for another version, or a ``needs_targeted_build`` request. Do NOT",
        "stop at a token registration / two-line alias when the diagnosis says the",
        "architecture is genuinely new: advancing one boot step counts, and a",
        "compiled or from-source need should be requested, not faked. Follow the",
        "ENABLEMENT PLAYBOOK in the task context for the tiered methodology.",
    ]


_FRAMEWORK_REWRITE_FOCUS_BODY: tuple[str, ...] = (
    "stack runs once per block, per denoising step, per chunk. Nothing about",
    "request serving applies here (no scheduler, no continuous batching, no",
    "KV-cache admission policy). The wins are the redundant work the loop",
    "structure creates.",
    "",
    "**Where the wins are**",
    "A single step's cost is multiplied by (blocks x steps x chunks), so any",
    "work whose result does not change across that product is dead weight,",
    "and any host round-trip inside it stalls the whole pipeline. Two classes",
    "of cost dominate and neither one owns a GPU kernel, so neither appears in",
    "a kernel breakdown: **redundant recomputation** and **host stalls**.",
    "",
    "**Rewrite pattern taxonomy** (the categories the evidence is labelled with; the backticked name is its id)",
    "- **(a) `memoize_invariant`: memoize a step- or block-invariant computation.** A pure function called",
    "  with arguments it already received. Cache the result.",
    "- **(b) `hoist_loop_invariant`: hoist a loop-invariant computation out of the loop.** The value is",
    "  logically the same each iteration but rebuilt from scratch, so a cache keyed on tensor identity would",
    "  never hit. Compute once per outer iteration and pass it in. **This is usually an enabler**: on its own",
    "  it measures flat, and its value is that it makes (a) start hitting.",
    "- **(c) `eliminate_host_round_trip` / `eliminate_host_sync`: eliminate a host round-trip or a",
    "  device-to-host sync.** An object collective agreeing on a shape the ranks could derive locally; an",
    "  `.item()` / `.tolist()` / `.cpu()` on the hot path.",
    "- **(d) `fuse_collectives`: fuse adjacent collectives, GEMMs or concatenations.** Several same-shape",
    "  payloads issued separately; pack them and issue one.",
    "- **(e) `swap_vendor_kernel`: swap an operator implementation for a vendor kernel.** Read this off the",
    "  GPU kernel breakdown, not the host evidence.",
    "- **(f) `keep_device_resident`: keep a tensor resident on the device.** A table rebuilt on the host and",
    "  re-uploaded on every use.",
    "- **(g) `drop_noop_glue`: drop no-op glue.** A dtype cast to the dtype the tensor already has; an",
    "  intermediate materialised only to be immediately consumed.",
    "",
    "**Cache-key recipe (get this wrong and you ship a correctness bug)**",
    "- Key on the COMPLETE argument identity: for a tensor,",
    "  `(data_ptr, shape, dtype, device, _version)`; plus every scalar that",
    "  changes the result. A key missing one input returns another input's",
    "  answer.",
    "- **Key on EVERY value you cache, not just the first one.** Caching a",
    "  `(k, v)` pair under a key derived from `k` alone returns the wrong `v`",
    "  the moment two calls share a `k` identity. A previous attempt shipped",
    "  exactly this and moved the output past the quality band.",
    "- **Pin the source tensors in the cache entry.** Under a caching",
    "  allocator a freed tensor's address is handed straight back to the next",
    "  allocation, so `data_ptr` alone will report a brand-new tensor as a hit.",
    "  Holding a reference to the keyed tensors prevents that.",
    "  Pinning means storing the SOURCE objects you keyed on. Storing the",
    "  computed result is not pinning: the sources are then unreferenced, free",
    "  to be deallocated, and their addresses recycled under a live key. A",
    "  previous attempt wrote `# Pin source tensors` above a line that stored",
    "  only the result — the comment is not the mechanism.",
    "- Check `_version` so an in-place mutation invalidates the entry.",
    "- Do NOT hash tensor *contents* to build a key: that forces a",
    "  device-to-host sync per call and costs more than it saves.",
    "- Bound the cache (small LRU) and size it for the calling pattern: under",
    "  classifier-free guidance the positive and negative branches alternate,",
    "  so a single-entry cache thrashes to a 0% hit rate.",
    "",
    "**Deliverable contract — every rewrite is a default-off switch**",
    "Each rewrite MUST be gated by its own environment switch that defaults",
    "OFF, so that with no switches set the code path is byte-for-byte the",
    "original. This is not a style preference; it is what makes each rewrite",
    "independently measurable and composable, and it is checked:",
    "- a parity leg runs with every switch unset and must reproduce the",
    "  baseline within its noise band, so a rewrite that changes behaviour",
    "  when disabled is rejected;",
    "- accepted switches become search levers, so the orchestrator measures",
    "  each one's own contribution and searches combinations rather than",
    "  taking your bundle as given.",
    "",
    "**The manifest is not optional and not documentation — it is what makes",
    "the guarantees above run.** With a gate the manifest does not declare,",
    "nothing is turned on for the measurement, no parity leg runs and no lever",
    "is registered: the patch is benched as an ordinary diff and whatever it",
    "does when 'off' is never checked. Integration now refuses a deliverable",
    "whose patch reads an `os.environ` switch the manifest omits, so an",
    "undeclared gate costs you the whole attempt.",
    "",
    "Alongside the patch, emit a `framework_switches` manifest — one entry per",
    "switch, each with: `switch` (the env var name), `category` (a taxonomy id",
    "above), `target` (file and symbol), `evidence` (which measured candidate",
    "it addresses), `depends_on` (switches that must also be on for this one",
    "to pay), and `enables` (switches that only pay once this one is on).",
    "A size, count or threshold switch must also carry `value`, the setting the",
    "rewrite is meant to run at; a flag may omit it and is set to `1`.",
    "",
    "**Declare `depends_on` / `enables` honestly — this is load-bearing.**",
    "An enabler measured alone shows no gain. If you do not declare the",
    "relationship, it is judged on its standalone number, rejected, and every",
    "rewrite that depended on it is silently devalued along with it. Declared,",
    "the whole bundle is benchmarked together and survives on its joint gain.",
    "",
    "**Always keep a fallback path.** Guard the fast path on the shapes and",
    "dtypes it actually requires and fall through to the original code",
    "otherwise, so an unexpected input degrades in speed and not in",
    "correctness.",
    "",
    "**Pitfalls**",
    "- Graph capture (HIP/CUDA graphs) conflicts with lazily populated",
    "  caches: the first call allocates inside the capture. Do not combine.",
    "- Caching across a chunk boundary needs the chunk identity in the key;",
    "  geometry alone repeats between chunks with different contents.",
    "- A switch name that collides with an upstream variable will be honoured",
    "  by upstream code too. Namespace yours.",
)


def _focus_framework_rewrite_specialist(
    inp: SpecialistPromptInputs,
) -> list[str]:
    """Build the domain-focus block for framework-level source rewrites.

    Carries the rewrite-pattern taxonomy as a *prior* — the categories, the
    cache-key recipe, the correctness invariants and the switch-manifest
    contract — while leaving the landing points to be found from the measured
    evidence. That split is deliberate: a prior that named specific functions
    would only ever work on one model, whereas the pattern vocabulary transfers
    to any iterative pipeline.

    Args:
        inp: Assembled prompt inputs for the current dispatch (used only for the
            framework name; the taxonomy itself is workload-independent).

    Returns:
        Markdown lines for the framework-rewrite specialist's focus block.
    """
    framework = (inp.framework or "the framework").strip()
    return [
        "You are the **framework rewrite specialist** — an AUTHORING sub-agent",
        f"for **{framework}**, an iterative model pipeline: the same transformer",
        *_FRAMEWORK_REWRITE_FOCUS_BODY,
    ]


def _focus_custom_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Render the operator-supplied focus verbatim; empty when not configured."""
    text = inp.custom_specialist_prompt.strip()
    return text.splitlines() if text else []


def _focus_speculative_decoding_specialist(inp: SpecialistPromptInputs) -> list[str]:
    """Render the speculative-decoding focus for this session's target model."""
    return speculative_decoding_focus(model_name=inp.model_name, model_path=inp.model_path, model_info=inp.model_info)


_DOMAIN_FOCUS_TEMPLATES: dict[str, "Callable[[SpecialistPromptInputs], list[str]]"] = {
    "framework_rewrite_specialist": _focus_framework_rewrite_specialist,
    "speculative_decoding_specialist": _focus_speculative_decoding_specialist,
    "candidate_discovery_specialist": _focus_candidate_discovery_specialist,
    "research_scout_specialist": _focus_research_scout_specialist,
    "static_recon_specialist": _focus_static_recon_specialist,
    "enablement_specialist": _focus_enablement_specialist,
    "custom_specialist": _focus_custom_specialist,
}
