# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement discovery + authoring operations.

Two halves of the enablement flow that both build on a
:class:`.enablement.FailureSignature`:

* **Discovery** — given a failure signature, decide which repos to scout for an
  enabling PR (the serving framework plus the ROCm / HIP / aiter bridge repos)
  and rank candidate PR titles by enablement intent ("enable / support / add /
  fix / port to ROCm"). See :func:`build_search_plan`, :func:`rank_titles`,
  :func:`score_enablement_title`.
* **Authoring** — turn a request + ranked candidates into the
  :class:`EnablementMandate` (allowed source roots + task description + patch
  invariants) handed to the patch-authoring specialist. See
  :func:`build_mandate`.

Pure-Python, GPU-free: no network, LLM, or filesystem access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .enablement import EnablementRequest, FailureSignature
from .keywords import extract_keywords, score_title_with_anti_signal
from .repo_map import bridge_repo_urls


# ---------------------------------------------------------------------------
# Discovery: repo selection + enablement-intent ranking
# ---------------------------------------------------------------------------


# Words in a PR title that signal it enables something previously broken.
ENABLEMENT_INTENT_TERMS: frozenset[str] = frozenset(
    {
        "enable",
        "enabled",
        "support",
        "supported",
        "add",
        "adds",
        "implement",
        "implements",
        "fix",
        "fixes",
        "port",
        "rocm",
        "hip",
        "register",
        "compat",
        "compatibility",
    }
)

# Per-kind seed keywords appended to the auto-extracted set. Keys are failure
# ``kind`` ids.
_KIND_SEED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "missing_model_arch": ("model", "architecture", "support", "add"),
    "unsupported_dtype": ("dtype", "fp8", "quant", "support"),
    "hip_kernel_missing": ("rocm", "hip", "aiter", "kernel"),
    "import_error": ("build", "import", "compile"),
    "shape_mismatch": ("shape", "reshape", "layout"),
    "not_implemented": ("implement", "support", "rocm"),
    "capability_disabled": ("enable", "rocm", "supported"),
    "unknown": (),
}


@dataclass(frozen=True)
class EnablementSearchPlan:
    """Where to look and what to match for an enablement failure.

    Attributes:
        repos: Repo URLs to enumerate PRs from (framework first, then any
            opted-in bridge repos), order-preserving and deduped.
        keywords: Ranking keywords (auto-extracted + per-kind seeds + the
            offending symbol/model tokens).
    """

    repos: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


def _symbol_tokens(symbol: str) -> list[str]:
    """Split an offending symbol / arch name into lowercase word tokens.

    Handles CamelCase (``Glm5ForCausalLM`` -> glm, for, causal, lm),
    snake_case and ``::`` C++ qualifiers.

    Args:
        symbol: The offending symbol/arch string.

    Returns:
        list[str]: Lowercased 2+ char tokens (may be empty).
    """
    if not symbol:
        return []
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", symbol)
    parts = re.split(r"[^A-Za-z0-9]+", spaced)
    return [p.lower() for p in parts if len(p) >= 2]


def build_search_plan(
    signature: FailureSignature,
    *,
    framework_repo_url: str,
    model: str = "",
) -> EnablementSearchPlan:
    """Build the repo set + ranking keywords for an enablement failure.

    Includes the framework repo plus the bridge repos (ROCm / HIP / aiter) for
    the signature's ``bridge_layer``.

    Args:
        signature: The classified failure.
        framework_repo_url: Canonical serving-framework repo URL.
        model: Model id/path — mined for extra keyword signal.

    Returns:
        EnablementSearchPlan: The deduped repo list and ranking keywords.
    """
    repos: list[str] = []
    if framework_repo_url.strip():
        repos.append(framework_repo_url.strip())
    repos.extend(bridge_repo_urls(signature.bridge_layer))

    keywords: list[str] = []
    keywords.extend(extract_keywords(model))
    keywords.extend(_symbol_tokens(signature.offending_symbol))
    keywords.extend(_KIND_SEED_KEYWORDS.get(signature.kind, ()))

    return EnablementSearchPlan(
        repos=tuple(dict.fromkeys(repos)),
        keywords=tuple(dict.fromkeys(k for k in keywords if k)),
    )


def score_enablement_title(
    title: str,
    plan: EnablementSearchPlan,
    *,
    intent_weight: float = 1.0,
) -> float:
    """Rank a candidate PR title for enablement relevance.

    Combines the anti-signal-aware gap-keyword overlap
    (:func:`.keywords.score_title_with_anti_signal`) with a boost for
    enablement-intent words (:data:`ENABLEMENT_INTENT_TERMS`).

    Args:
        title: The PR title.
        plan: The search plan carrying ranking keywords.
        intent_weight: Weight per enablement-intent token hit.

    Returns:
        float: The combined score (>= 0.0); callers may drop ``0.0``.
    """
    if not title:
        return 0.0
    base = score_title_with_anti_signal(title, plan.keywords)
    title_tokens = set(re.findall(r"[a-z][a-z0-9_]+", title.lower()))
    intent = len(title_tokens & ENABLEMENT_INTENT_TERMS)
    return base + intent_weight * float(intent)


def rank_titles(
    titles: Sequence[str],
    plan: EnablementSearchPlan,
) -> list[tuple[str, float]]:
    """Score and sort candidate titles by enablement relevance, descending.

    Args:
        titles: Candidate PR titles.
        plan: The search plan carrying ranking keywords.

    Returns:
        list[tuple[str, float]]: ``(title, score)`` pairs, highest first;
        ties keep input order (stable sort).
    """
    scored = [(t, score_enablement_title(t, plan)) for t in titles]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


# ---------------------------------------------------------------------------
# Authoring: the mandate handed to the patch-authoring sub-agent
# ---------------------------------------------------------------------------


# Source-root families the authored patch may target (fallback when discovery fails).
_FRAMEWORK_ROOT_HINT = "the serving-framework source tree (e.g. sglang / vllm / atom)"
_ROCM_HIP_ROOT_HINT = "the ROCm / HIP / aiter source tree (/opt/rocm, aiter)"


def _resolve_package_version(package: str) -> str:
    """Return the installed version of *package*, or empty string on failure."""
    try:
        import importlib.metadata as _m

        return _m.version(package)
    except Exception:  # noqa: BLE001
        return ""


def _resolve_actual_root_hints(framework: str) -> list[str]:
    """Return concrete source-root strings for the mandate (never empty).

    Calls probe_framework_source_roots_for_env() and falls back to the generic
    prose hints when discovery yields nothing.  Also appends version info for the
    target framework package.
    """
    try:
        from hyperloom.orchestrator.framework.paths import (
            probe_framework_source_roots_for_env,
            summarise_framework_root_discovery,
        )

        roots_str = probe_framework_source_roots_for_env()
        if roots_str:
            hints: list[str] = []
            summary = summarise_framework_root_discovery(roots_str)
            for root in roots_str.split(":"):
                root = root.strip()
                if root:
                    hints.append(root)
            hints.append(f"(discovery summary: {summary})")
            pkg_map = {"sglang": "sglang", "vllm": "vllm", "xdit": "xfuser", "atom": "atom"}
            pkg_name = pkg_map.get(framework, framework)
            ver = _resolve_package_version(pkg_name)
            if ver:
                hints.append(f"({pkg_name} installed version: {ver})")
            # Always include the ROCm/HIP root hint (authoring sub-agent always
            # has /opt/rocm in scope for ROCm-side fixes, regardless of whether
            # probe discovered it or not).
            if not any(_ROCM_HIP_ROOT_HINT in h for h in hints):
                hints.append(_ROCM_HIP_ROOT_HINT)
            # Keep the generic framework hint as context even when real paths exist.
            if not any(_FRAMEWORK_ROOT_HINT in h for h in hints):
                hints.append(_FRAMEWORK_ROOT_HINT)
            return hints
    except Exception:  # noqa: BLE001 — discovery is best-effort
        pass
    return [_FRAMEWORK_ROOT_HINT, _ROCM_HIP_ROOT_HINT]

# Invariants every enablement patch must respect.
ENABLEMENT_PATCH_INVARIANTS: tuple[str, ...] = (
    "The patch MUST be a valid unified diff that applies cleanly "
    "(`git apply --check` must pass) against the live source tree.",
    "Only *source edits* must stay under the allowed source roots listed below; "
    "touching any other path with a code patch is a hard reject. (Environment "
    "setup via ENVIRONMENT SETUP below is separate and allowed.)",
    "Do NOT fabricate throughput/latency/accuracy numbers — the gate here is "
    "RUNNABILITY (does the server boot + pass a minimal inference), not perf.",
    "Prefer the smallest bridging change that makes the combo run — or, when full "
    "runnability is out of reach this round, the smallest change that ADVANCES the "
    "boot past the current failure (see PROGRESS DELIVERABLE below); do not "
    "refactor unrelated code.",
    "If a discovered PR already implements the fix, adapt/backport it rather than authoring from scratch.",
)

# Environment-setup authorization: the specialist MAY run dependency/tool
# installs during validation, and must record each verbatim in
# ``specialist_done.setup_commands`` so integrate_patch can replay them.
ENABLEMENT_SETUP_GUIDANCE: tuple[str, ...] = (
    "You MAY install missing/stale packages or CLI tools when that is what the "
    "model needs to build or run — e.g. `pip install -U transformers`, "
    "`pip install <dep>`, `apt-get install -y gh`, `npm install -g <tool>`. Use "
    "non-interactive, version-pinned commands where possible.",
    "For EVERY install/setup command you rely on, record it VERBATIM in the "
    "`setup_commands` list of your final `specialist_done` (a JSON array of "
    "shell strings). integrate_patch replays these (allowlisted) before applying "
    "your patch and booting, so an install you depended on is reproduced rather "
    "than lost after your session ends. An unrecorded install will NOT persist.",
    "Keep setup commands minimal and deterministic (pin versions), non-"
    "interactive (`-y` / `--yes`), and limited to package/tool installation — "
    "they are validated against an install-only allowlist on replay.",
    "If NO environment setup is needed (a pure source fix), leave `setup_commands` empty.",
)

# Serial-enablement progress contract. A brand-new architecture or a large
# capability gap rarely becomes fully runnable inside a single budget window.
# The integrate side already REWARDS partial progress: a patch that only
# advances the boot to a *new, deeper* failure is KEPT and stacked as a base for
# the next round (see ``enablement.enablement_made_progress`` and
# ``integrate_patch`` ``status="advanced"``). Historically the specialist was
# only told to make the combo *run*, so on a big gap it judged full runnability
# infeasible and returned ``empty=true`` wholesale — starving that incremental
# machinery of the very patches it stacks. This guidance closes that asymmetry:
# advancing the boot ONE step is an explicit, valid deliverable.
ENABLEMENT_PROGRESS_GUIDANCE: tuple[str, ...] = (
    "INCREMENTAL PROGRESS IS A FIRST-CLASS DELIVERABLE. Enablement gaps are "
    "serial: clearing one boot failure usually reveals a deeper one. You do NOT "
    "have to reach full end-to-end runnability in this one budget window.",
    "If you cannot make the combo fully run, author the SMALLEST patch that "
    "ADVANCES the boot PAST THE CURRENT failure — clear THIS error even if a "
    "new, different failure then appears. A patch that changes the failure "
    "signature is KEPT and stacked as a base; the next round resumes from the "
    "deeper failure. One step forward is strictly better than returning nothing.",
    "Record that patch in ``patches_written`` (and any installs in "
    "``setup_commands``), set ``empty=false``, and in ``summary`` state which "
    "failure you cleared and what the next (deeper) failure now is.",
    "Return ``empty=true`` ONLY when you cannot advance past the CURRENT failure "
    "by even one step — NOT merely because full runnability is out of reach this "
    "round.",
)


# Targeted-build request contract. A pure source patch (a unified diff against
# the installed tree) cannot deliver a *compiled* component (a new AITER
# FP4/MLA/NSA op, sgl-kernel) or a from-source framework build (a newer vLLM
# that natively implements a brand-new architecture). Historically the
# specialist had no way to ask for one — it could only author a patch or return
# empty — so genuinely-new architectures dead-ended at the arch-registry alias.
# This contract lets the specialist REQUEST an off-loop targeted build; the
# Coordinator enqueues it on the isolated, ROCm-safe build lane (isolated venv +
# pinned ROCm torch constraints), gated by the runnable-decision probe.
ENABLEMENT_BUILD_REQUEST_GUIDANCE: tuple[str, ...] = (
    "REQUESTING A COMPILED / FROM-SOURCE BUILD. If clearing this gap needs a "
    "*compiled* component (a new AITER FP4/MLA/NSA op, sgl-kernel) or a "
    "from-source framework build (e.g. a newer vLLM that NATIVELY implements "
    "this architecture, which a source patch against the INSTALLED tree cannot "
    "provide), do NOT fake it with an install command or a stub patch. Emit a "
    "``needs_targeted_build`` object in your final ``specialist_done`` and the "
    "Coordinator runs it off-loop on an isolated, ROCm-safe build lane.",
    "``needs_targeted_build`` schema: ``{component, capability, repo_url, ref, "
    "reason}``. ``component`` is one of ``aiter`` / ``sgl_kernel`` / "
    "``vllm_source`` / ``framework_ext``. ``capability`` names the missing op / "
    "arch (e.g. ``deepseek_v4_nsa`` / ``fp4_moe``). ``repo_url`` + ``ref`` are "
    "OPTIONAL but HIGH-VALUE: if you found (via WebSearch / mcp__pr_monitor__*) "
    "a specific upstream PR / tag / commit that implements the fix, name it "
    "(a GitHub PR URL, ``PR:1234``, a tag, or a sha) so the build checks out "
    "exactly that; leave them empty for tag-descending autoselect. ``reason`` "
    "is a one-line evidence summary.",
    "A build request is COMPLEMENTARY to a source patch, not a replacement: you "
    "MAY both author the smallest patch that advances the boot one step AND "
    "request a build for the compiled/from-source piece the patch cannot cover. "
    "Setting ``needs_targeted_build`` counts as a real deliverable — do NOT set "
    "``empty=true`` when you emit one.",
)


@dataclass(frozen=True)
class EnablementMandate:
    """A fully-specified authoring task for the enablement specialist.

    Attributes:
        framework: Target serving framework.
        model: Model id/path that must become runnable.
        signature: The classified failure driving the fix.
        allowed_root_hints: Human-readable source-root families in scope.
        candidate_refs: Ranked bridging PR/ref hints (best first).
        task_description: The rendered specialist mandate (prompt body).
        invariants: The patch invariants (see :data:`ENABLEMENT_PATCH_INVARIANTS`).
    """

    framework: str
    model: str
    signature: FailureSignature
    allowed_root_hints: tuple[str, ...]
    candidate_refs: tuple[str, ...] = ()
    task_description: str = ""
    invariants: tuple[str, ...] = field(default_factory=lambda: ENABLEMENT_PATCH_INVARIANTS)


def _render_task_description(
    req: EnablementRequest,
    sig: FailureSignature,
    candidate_refs: Sequence[str],
    allowed_root_hints: Sequence[str],
    source_context: str = "",
) -> str:
    """Render the specialist mandate text for an enablement failure.

    Args:
        req: The enablement request (framework/model/opt-in).
        sig: The classified failure signature.
        candidate_refs: Ranked bridging refs (best first).
        allowed_root_hints: Source-root families in scope.
        source_context: Optional snippet of source lines near the offending
            site, injected verbatim to ground the authoring sub-agent. Empty
            omits the block.

    Returns:
        str: A multi-line prompt body for the authoring sub-agent.
    """
    lines: list[str] = []
    lines.append(
        f"GOAL: make model `{req.model}` run under the `{req.framework}` backend. It currently fails to start."
    )
    lines.append("")
    lines.append(f"FAILURE CLASS: {sig.kind} (confidence {sig.confidence:.2f}).")
    if sig.secondary_kinds:
        lines.append(f"SECONDARY FAILURE CLASSES (also matched): {', '.join(sig.secondary_kinds)}")
    if sig.offending_file:
        lines.append(f"OFFENDING FILE (best guess): {sig.offending_file}")
    if sig.offending_symbol:
        lines.append(f"OFFENDING SYMBOL: {sig.offending_symbol}")
    if sig.raw_excerpt:
        lines.append(f"ERROR EXCERPT: {sig.raw_excerpt}")
    if source_context.strip():
        lines.append("")
        lines.append("SOURCE CONTEXT (near offending site):")
        lines.append("```")
        lines.append(source_context.rstrip("\n"))
        lines.append("```")
    lines.append("")
    if candidate_refs:
        lines.append("CANDIDATE BRIDGING PRs / REFS (most relevant first):")
        for ref in candidate_refs:
            lines.append(f"  - {ref}")
        lines.append("")
    lines.append("ALLOWED SOURCE ROOTS (code edits outside these are rejected):")
    for hint in allowed_root_hints:
        lines.append(f"  - {hint}")
    lines.append("")
    lines.append("ENVIRONMENT SETUP (installs are allowed AND must be recorded):")
    for g in ENABLEMENT_SETUP_GUIDANCE:
        lines.append(f"  - {g}")
    lines.append("")
    lines.append("PROGRESS DELIVERABLE (serial enablement — advancing the boot one step counts):")
    for g in ENABLEMENT_PROGRESS_GUIDANCE:
        lines.append(f"  - {g}")
    lines.append("")
    lines.append("TARGETED BUILD (request a compiled / from-source component when a patch cannot deliver it):")
    for g in ENABLEMENT_BUILD_REQUEST_GUIDANCE:
        lines.append(f"  - {g}")
    lines.append("")
    lines.append("INVARIANTS:")
    for inv in ENABLEMENT_PATCH_INVARIANTS:
        lines.append(f"  - {inv}")
    return "\n".join(lines)


def build_mandate(
    req: EnablementRequest,
    *,
    signature: FailureSignature | None = None,
    candidate_refs: Sequence[str] = (),
    source_context: str = "",
    root_hints: Sequence[str] | None = None,
) -> EnablementMandate:
    """Build an :class:`EnablementMandate` from a request + candidates.

    Args:
        req: The enablement request.
        signature: Pre-computed signature; defaults to ``req.signature``.
        candidate_refs: Ranked bridging refs to suggest (best first).
        source_context: Optional source snippet near the offending site to
            ground the authoring sub-agent (best-effort; empty omits it).
        root_hints: Explicit source-root hints; when ``None`` (default) they
            are resolved via :func:`_resolve_actual_root_hints` (which calls
            ``probe_framework_source_roots_for_env()`` and falls back to the
            generic prose constants on failure).

    Returns:
        EnablementMandate: The authoring contract, ready to hand to the
        specialist runner.
    """
    sig = signature if signature is not None else req.signature
    if root_hints is not None:
        hints: list[str] = list(root_hints) or [_FRAMEWORK_ROOT_HINT, _ROCM_HIP_ROOT_HINT]
    else:
        hints = _resolve_actual_root_hints(req.framework)
    refs = tuple(r for r in candidate_refs if r)
    task = _render_task_description(req, sig, refs, hints, source_context)
    return EnablementMandate(
        framework=req.framework,
        model=req.model,
        signature=sig,
        allowed_root_hints=tuple(hints),
        candidate_refs=refs,
        task_description=task,
    )


__all__ = [
    "ENABLEMENT_BUILD_REQUEST_GUIDANCE",
    "ENABLEMENT_INTENT_TERMS",
    "ENABLEMENT_PATCH_INVARIANTS",
    "ENABLEMENT_PROGRESS_GUIDANCE",
    "ENABLEMENT_SETUP_GUIDANCE",
    "EnablementMandate",
    "EnablementSearchPlan",
    "build_mandate",
    "build_search_plan",
    "rank_titles",
    "score_enablement_title",
]
