# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Enablement authoring mandate: the contract handed to the patch-authoring sub-agent.

Builds the mandate that parameterises patch authoring for the enablement
objective:

* which source roots the patch may touch,
* a ``task_description`` derived from the failure signature and the ranked
  bridging candidates,
* the invariants the authored patch must satisfy (grounded diff, no fabricated
  numbers, runnable-gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .enablement import EnablementRequest, FailureSignature


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
    "ENABLEMENT_PATCH_INVARIANTS",
    "ENABLEMENT_SETUP_GUIDANCE",
    "EnablementMandate",
    "build_mandate",
]
