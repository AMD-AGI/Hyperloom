# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Enablement-oriented bridging discovery: which repos to scout and how to rank.

Given a :class:`framework_agent.enablement.FailureSignature`, this module
decides (1) *which repos* to search for an enabling PR — the serving framework
plus, when opted in, the ROCm / HIP / aiter bridge repos — and (2) *how to
rank* candidate PR titles for *enablement* intent ("enable / support / add /
fix / port to ROCm") rather than the perf intent the existing
:mod:`framework_agent.keywords` ranker targets.

Pure-Python, no network: it produces a search *plan* and a title *ranker*;
the actual PR enumeration is done by the existing ``sources`` layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .enablement import FailureSignature
from .keywords import extract_keywords, score_title_with_anti_signal
from .repo_map import bridge_repo_urls


# Words in a PR title that signal it *enables* something previously broken,
# as opposed to merely tuning perf. Used as a positive boost on top of the
# gap-keyword overlap score.
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

# Per-kind seed keywords appended to the auto-extracted set so discovery has
# signal even when the log is terse. Keys are failure ``kind`` ids.
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

    The framework repo is always included, plus the bridge repos (ROCm / HIP /
    aiter) for the signature's ``bridge_layer`` — cross-layer source edits are
    a first-class, default-on capability of the enablement path.

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

    Combines the anti-signal-aware gap-keyword overlap (reusing
    :func:`framework_agent.keywords.score_title_with_anti_signal` so wrong-axis
    PRs are still demoted) with a boost for enablement-intent words
    (:data:`ENABLEMENT_INTENT_TERMS`).

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


__all__ = [
    "ENABLEMENT_INTENT_TERMS",
    "EnablementSearchPlan",
    "build_search_plan",
    "rank_titles",
    "score_enablement_title",
]
