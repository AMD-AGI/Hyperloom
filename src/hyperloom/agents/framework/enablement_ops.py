# Copyright Advanced Micro Devices, Inc. All rights reserved.

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


# Source-root families the authored patch may target.
_FRAMEWORK_ROOT_HINT = "the serving-framework source tree (e.g. sglang / vllm / atom)"
_ROCM_HIP_ROOT_HINT = "the ROCm / HIP / aiter source tree (/opt/rocm, aiter)"

# Invariants every enablement patch must respect.
ENABLEMENT_PATCH_INVARIANTS: tuple[str, ...] = (
    "The patch MUST be a valid unified diff that applies cleanly "
    "(`git apply --check` must pass) against the live source tree.",
    "Only *source edits* must stay under the allowed source roots listed below; "
    "touching any other path with a code patch is a hard reject. (Environment "
    "setup via ENVIRONMENT SETUP below is separate and allowed.)",
    "Do NOT fabricate throughput/latency/accuracy numbers — the gate here is "
    "RUNNABILITY (does the server boot + pass a minimal inference), not perf.",
    "Prefer the smallest bridging change that makes the combo run; do not "
    "refactor unrelated code.",
    "If a discovered PR already implements the fix, adapt/backport it rather "
    "than authoring from scratch.",
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
    "If NO environment setup is needed (a pure source fix), leave "
    "`setup_commands` empty.",
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
        f"GOAL: make model `{req.model}` run under the `{req.framework}` backend. "
        "It currently fails to start."
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
) -> EnablementMandate:
    """Build an :class:`EnablementMandate` from a request + candidates.

    Args:
        req: The enablement request.
        signature: Pre-computed signature; defaults to ``req.signature``.
        candidate_refs: Ranked bridging refs to suggest (best first).
        source_context: Optional source snippet near the offending site to
            ground the authoring sub-agent (best-effort; empty omits it).

    Returns:
        EnablementMandate: The authoring contract, ready to hand to the
        specialist runner.
    """
    sig = signature if signature is not None else req.signature
    hints: list[str] = [_FRAMEWORK_ROOT_HINT, _ROCM_HIP_ROOT_HINT]
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
    "ENABLEMENT_INTENT_TERMS",
    "ENABLEMENT_PATCH_INVARIANTS",
    "ENABLEMENT_SETUP_GUIDANCE",
    "EnablementMandate",
    "EnablementSearchPlan",
    "build_mandate",
    "build_search_plan",
    "rank_titles",
    "score_enablement_title",
]
