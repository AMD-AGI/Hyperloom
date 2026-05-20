"""LLM-loop patch proposer for framework_optimize (P3 PR-G).

Drives an LLM (Claude / Codex via the same backend abstractions as the
critic-agent and kernel-agent skills) to produce a unified diff patch
that targets vllm or sglang source. Inputs:

* :class:`AstScanResult` from PR-E (the flag discovery findings)
* :class:`KbEntry` list from :mod:`kb_priors` (pitfalls / boundaries / perf priors / lessons)
* A target framework + session_dir to drop ``proposal.diff`` under

Output: :class:`ProposedPatch` with ``path`` to ``runs/framework/<task_id>/proposal.diff``,
``predicted_gain_pct``, ``rationale``, ``files_touched``, ``confidence``.

LLM driver indirection: this module never imports a concrete LLM SDK
directly. Callers inject a ``ProposerDriver`` callable so:

* PR-G unit tests pass a deterministic fixture driver.
* Production wires a real Claude/Codex driver (PR-F's
  FrameworkAgentBackend will subclass this in a future iteration).

When ``ast_scan.mode == 'grep_fallback'`` the returned confidence is
downgraded one step per design §9.3.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol

from .ast_scanner import AstScanResult
from .flag_discovery import DiscoveredFlag
from .kb_priors import KbEntry


log = logging.getLogger(__name__)


Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class ProposerInput:
    """Bundle of inputs fed to the LLM driver."""

    ast_findings: AstScanResult
    kb_priors: list[KbEntry] = field(default_factory=list)
    target_framework: str = ""
    task_id: str = ""
    session_dir: Path = field(default_factory=Path)
    max_turns: int = 16


@dataclass(frozen=True)
class ProposedPatch:
    """LLM output rendered as a writable patch artefact.

    Fields:

    * ``path``               -- absolute path to ``proposal.diff``;
      empty string when the LLM decided no patch is warranted
      (flag-discovery-only outcome -- handler returns this as the
      ``patch_path`` field of OptimizeSuccess).
    * ``diff_text``          -- the unified diff body (also written to ``path``).
    * ``predicted_gain_pct`` -- LLM-estimated throughput gain.
    * ``rationale``          -- one-paragraph explanation; surfaces in
      ``OptimizeSuccess.rationale``.
    * ``files_touched``      -- relative paths to framework source
      files modified by the diff.
    * ``confidence``         -- ``high`` / ``medium`` / ``low``; PR-I
      KB write uses this to weight contributing the eventual lesson.
    * ``elapsed_ms``         -- wall-clock for the whole propose call.
    """

    path: str
    diff_text: str
    predicted_gain_pct: float
    rationale: str
    files_touched: tuple[str, ...] = ()
    confidence: Confidence = "medium"
    elapsed_ms: int = 0


class ProposerDriver(Protocol):
    """Pluggable LLM driver. Production hands a real Claude/Codex
    wrapper; tests hand a fixture driver."""

    def __call__(
        self,
        prompt: str,
        *,
        max_turns: int,
    ) -> tuple[str, str, float]: ...  # noqa: E704
    # Returns (diff_text, rationale, predicted_gain_pct).


def _build_prompt(inp: ProposerInput) -> str:
    """Render the LLM prompt from AST findings + KB priors.

    PR-G ships a minimal template; production prompt engineering lands
    in a follow-up commit driven by real-vllm evaluation results.
    """
    sections: list[str] = []
    sections.append(
        f"You are the Framework optimisation agent. Target framework: "
        f"{inp.target_framework}. Task: {inp.task_id or '(unset)'}."
    )
    sections.append(
        "Your job: propose ONE minimal unified diff against the active "
        "framework source. Do not refactor unrelated code. Patch files "
        "must live under the framework source root."
    )
    if inp.kb_priors:
        sections.append("KB priors (respect pitfalls + boundaries first):")
        for e in inp.kb_priors:
            sections.append(
                f"- [{e.category}/{e.target_framework or 'cross'}] "
                f"{e.entry_id}: {e.title}"
            )
    if inp.ast_findings.flags:
        sections.append(
            f"AST scan ({inp.ast_findings.mode}) surfaced "
            f"{len(inp.ast_findings.flags)} flag/field candidates. "
            "Top entries:"
        )
        for f in inp.ast_findings.flags[:25]:
            sections.append(
                f"- {f.via} {f.surface} {f.flag_name} ({f.type_hint})"
                f" @ {Path(f.source_path).name}:{f.line}"
            )
    else:
        sections.append(
            "AST scan returned no candidates. Either propose a flag-"
            "discovery-only outcome (empty diff) or rely on KB priors."
        )
    sections.append(
        "Output format: a single unified diff block, then ONE-LINE "
        "'PREDICTED_GAIN_PCT: <float>' and ONE-LINE "
        "'RATIONALE: <one-paragraph>'."
    )
    return "\n\n".join(sections)


def _confidence_for(scan: AstScanResult, gain_pct: float) -> Confidence:
    """Pick confidence from scan mode + predicted gain."""
    if scan.mode == "grep_fallback":
        return "low"
    if gain_pct >= 8.0:
        return "high"
    if gain_pct >= 3.0:
        return "medium"
    return "low"


def _files_touched(diff_text: str) -> tuple[str, ...]:
    """Extract `+++ b/...` paths from a unified diff."""
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                out.append(path)
    return tuple(out)


def propose_patch(
    inp: ProposerInput,
    *,
    driver: ProposerDriver,
) -> ProposedPatch:
    """Drive the LLM loop. Writes ``proposal.diff`` under
    ``session_dir/runs/framework/<task_id>/`` and returns
    :class:`ProposedPatch`.
    """
    started = time.monotonic()
    prompt = _build_prompt(inp)
    diff_text, rationale, gain_pct = driver(
        prompt, max_turns=inp.max_turns,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if not diff_text.strip():
        # Flag-discovery-only outcome.
        log.info(
            "propose_patch: empty diff (flag-discovery-only); task=%s",
            inp.task_id,
        )
        return ProposedPatch(
            path="",
            diff_text="",
            predicted_gain_pct=float(gain_pct),
            rationale=rationale or "no patch -- flag discovery only",
            files_touched=(),
            confidence=_confidence_for(inp.ast_findings, gain_pct),
            elapsed_ms=elapsed_ms,
        )

    out_dir = (
        Path(inp.session_dir) / "runs" / "framework" / (inp.task_id or "fw-task")
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proposal.diff"
    out_path.write_text(diff_text, encoding="utf-8")
    return ProposedPatch(
        path=str(out_path),
        diff_text=diff_text,
        predicted_gain_pct=float(gain_pct),
        rationale=rationale,
        files_touched=_files_touched(diff_text),
        confidence=_confidence_for(inp.ast_findings, gain_pct),
        elapsed_ms=elapsed_ms,
    )


__all__ = [
    "Confidence",
    "ProposedPatch",
    "ProposerDriver",
    "ProposerInput",
    "propose_patch",
]
