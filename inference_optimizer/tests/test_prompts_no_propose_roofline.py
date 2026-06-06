"""Blocker #3 guard — LLM-facing prompts must not instruct the model to
propose ``roofline`` / ``profile``.

After the single-path refactor the Coordinator owns the analysis
lifecycle. ``roofline`` and ``profile`` are absent from
``PHASE_LLM_PROPOSABLE_ACTIONS``, so PolicyGate R1
``phase_incompatible`` denies any LLM-emitted
``propose_action{action='roofline'|'profile'}`` /
``delegate{action='roofline'|'profile'}``. Prompts that still tell
the LLM to propose either cause a guaranteed denial loop on every
session.

This test scans every prompt-facing file (orchestration / specialist
prompts, action ``*.md`` and ``_meta/*.yaml`` next-step strings) and
fails if any of the forbidden imperatives slip back in. Negations such
as *"never propose `profile`"* are allowed — the regex requires the
verb to be unguarded.
"""

from __future__ import annotations

import re
from pathlib import Path



# Files whose content is rendered into LLM context. Tests / docs /
# code-comment files are excluded — the contract is "what the LLM
# reads", not "what the codebase mentions".
_PROMPT_ROOTS = (
    Path("inference_optimizer/actions"),
    Path("inference_optimizer/orchestrator/system_prompts"),
)

# Files outside the prompt directories whose user-visible string
# literals end up inside the per-tick orchestration prompt via
# ``SharedState.format_for_prompt()`` / runtime hint composition.
# Scanned in addition to ``_PROMPT_ROOTS`` to catch the case where a
# fallback message like "(no snapshot yet — propose `roofline` ...)"
# is rendered straight into LLM context but lives in code rather than
# a prompt template.
_PROMPT_AUX_FILES = (
    Path("inference_optimizer/orchestrator/shared_state.py"),
)

# Allow-list: these files *describe* the denial rule and necessarily
# quote the forbidden pattern in fenced code or inline backticks. They
# are not instructing the LLM to propose — they are documenting that
# proposing is denied.
_DOC_ALLOWLIST = {
    Path("inference_optimizer/actions/roofline.md"),
    Path("inference_optimizer/actions/profile.md"),
}

# Negation guard: any line that contains one of these markers near the
# forbidden phrase is interpreted as a *denial* note rather than an
# instruction, and is skipped.
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
    # Resolve repo root from this test file (tests/<file> → repo root).
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
    """``SharedState._format_analysis_md_full`` renders straight into
    the per-tick orchestration prompt via ``format_for_prompt``. When
    no analysis snapshot exists yet, the fallback string must NOT tell
    the LLM to propose ``roofline`` / ``profile`` — both names are
    absent from ``PHASE_LLM_PROPOSABLE_ACTIONS``, so PolicyGate R1
    rejects any such proposal with ``rule='phase_incompatible'``.
    Pin the exact output so the grep guard above (line-based regex)
    is reinforced by an end-to-end behaviour check."""
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
