# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Enablement authoring mandate: the contract handed to the patch-authoring sub-agent.

The actual patch synthesis is executed by Hyperloom's existing
``SpecialistRunner`` (an LLM sub-agent writing into an isolated git worktree,
gated by ``specialist_patch_safety`` and ``integrate_patch``). This module does
**not** re-implement that machinery — it builds the *mandate* that parameterises
it for the enablement objective:

* which source roots the patch may touch (framework always; ROCm/HIP only when
  explicitly opted in),
* a focused ``task_description`` derived from the failure signature and the
  ranked bridging candidates,
* the invariants the authored patch must satisfy (grounded diff, no fabricated
  numbers, runnable-gate — not perf-gate).

Keeping this as a pure, testable function means the coordinator wiring (Task #6)
is a thin adapter: classify → discover → ``build_mandate`` → dispatch specialist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .enablement import EnablementRequest, FailureSignature


# Source-root families the authored patch may target. Both the framework's
# own source tree and the ROCm/HIP/aiter tree are always in scope (mirrors
# ``framework_paths.resolve_rocm_hip_source_roots`` on the IO side, which is
# always merged into the allowlist).
_FRAMEWORK_ROOT_HINT = "the serving-framework source tree (e.g. sglang / vllm / atom)"
_ROCM_HIP_ROOT_HINT = "the ROCm / HIP / aiter source tree (/opt/rocm, aiter)"

# Invariants every enablement patch must respect. Surfaced verbatim into the
# specialist prompt so the sub-agent cannot silently drop them.
ENABLEMENT_PATCH_INVARIANTS: tuple[str, ...] = (
    "The patch MUST be a valid unified diff that applies cleanly "
    "(`git apply --check` must pass) against the live source tree.",
    "Only edit files under the allowed source roots listed below; touching any "
    "other path is a hard reject.",
    "Do NOT fabricate throughput/latency/accuracy numbers — the gate here is "
    "RUNNABILITY (does the server boot + pass a minimal inference), not perf.",
    "Prefer the smallest bridging change that makes the combo run; do not "
    "refactor unrelated code.",
    "If a discovered PR already implements the fix, adapt/backport it rather "
    "than authoring from scratch.",
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
    lines.append("ALLOWED SOURCE ROOTS (edits outside these are rejected):")
    for hint in allowed_root_hints:
        lines.append(f"  - {hint}")
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
    "ENABLEMENT_PATCH_INVARIANTS",
    "EnablementMandate",
    "build_mandate",
]
