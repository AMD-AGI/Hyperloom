# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Blocker #3 guard — LLM-facing prompts must not instruct the model to propose ``roofline`` / ``profile``.

The Coordinator owns those analysis actions; PolicyGate R1 ``phase_incompatible``
denies any LLM proposal, so prompts that tell the LLM to propose either cause a
denial loop. This scans prompt-facing files and fails on unguarded imperatives.
"""

from __future__ import annotations

import re
from pathlib import Path



# Files whose content is rendered into LLM context.
_PROMPT_ROOTS = (
    Path("inference_optimizer/actions"),
    Path("inference_optimizer/orchestrator/system_prompts"),
)

# Code files whose string literals reach the orchestration prompt via
# ``SharedState.format_for_prompt()``.
_PROMPT_AUX_FILES = (
    Path("inference_optimizer/orchestrator/shared_state.py"),
)

# Allow-list: these files document the denial rule and necessarily quote the
# forbidden pattern.
_DOC_ALLOWLIST = {
    Path("inference_optimizer/actions/roofline.md"),
    Path("inference_optimizer/actions/profile.md"),
}

# Negation guard: lines containing these markers are treated as denial notes and skipped.
_NEGATION_MARKERS = (
    "never",
    "do not",
    "don't",
    "must not",
    "policygate denies",
    "denied",
    "not llm-proposable",
    "not propose",
    "no longer propose",
    "stop proposing",
)

_FORBIDDEN_PATTERNS = (
    re.compile(r"propose\s+(?:`)?(?:profile|roofline)", re.IGNORECASE),
    re.compile(r"propose_action\b[^.\n]*(?:profile|roofline)", re.IGNORECASE),
    re.compile(r"delegate\b[^.\n]*action[^.\n]*['\"`]?(?:profile|roofline)", re.IGNORECASE),
)


def _iter_prompt_files(repo_root: Path):
    for root in _PROMPT_ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() in {".md", ".yaml", ".yml", ".py", ".txt"}:
                yield path
    for aux in _PROMPT_AUX_FILES:
        candidate = repo_root / aux
        if candidate.exists():
            yield candidate


def _line_is_negation(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in _NEGATION_MARKERS)


def test_no_prompt_instructs_llm_to_propose_analysis_action():
    repo_root = Path(__file__).resolve().parents[2]
    hits: list[tuple[Path, int, str]] = []
    for path in _iter_prompt_files(repo_root):
        rel = path.relative_to(repo_root)
        if rel in _DOC_ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if _line_is_negation(line):
                continue
            for pat in _FORBIDDEN_PATTERNS:
                if pat.search(line):
                    hits.append((rel, lineno, line.strip()))
                    break
    assert not hits, (
        "LLM-facing prompts still instruct the model to propose "
        "roofline/profile (PolicyGate will deny on every session). "
        "Offending sites:\n  "
        + "\n  ".join(f"{p}:{ln}: {body!r}" for p, ln, body in hits)
    )


def test_shared_state_empty_analysis_md_does_not_tell_llm_to_propose():
    """The empty-snapshot ``_format_analysis_md_full`` fallback must NOT tell the LLM to propose ``roofline`` / ``profile``."""
    from inference_optimizer.orchestrator.shared_state import SharedState

    state = SharedState()
    state.last_trace_analyze = {}
    rendered = state._format_analysis_md_full()
    low = rendered.lower()
    assert "propose `roofline`" not in low
    assert "propose `profile`" not in low
    assert "propose roofline" not in low
    assert "propose profile" not in low
    assert (
        "auto-enqueued" in low
        or "coordinator" in low
        or "phase_incompatible" in low
    )
