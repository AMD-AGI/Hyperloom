# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Compose the Critic agent's system prompt from typed inputs.

Wraps the ``critic.md`` rules fragment with generated sections (mission,
run context, known actions, default verdict, phase review contract, optional
kernel-owned carve-out, rules, output protocol). Deterministic for given inputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..action_registry import ActionMetadata, ActionRegistry
from .prompt_builder import KERNEL_OWNED_ACTIONS, _filter_actions


# Family ordering for the catalogue section.
_FAMILY_ORDER: tuple[str, ...] = (
    "prep",
    "analysis",
    "shallow",
    "deep_kernel",
    "long",
    "creative",
    "resilience",
)


def _read_rules_fragment(path: Path | None) -> str:
    """Read the ``critic.md`` rules fragment, tolerating absence.

    Args:
        path (Path | None): Path to the rules fragment, or ``None`` to skip.

    Returns:
        str: The stripped fragment text, or an empty string when the path is
        ``None`` or unreadable.
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _section_mission() -> list[str]:
    """Build the MISSION section lines.

    Returns:
        list[str]: Markdown lines describing the Critic's review mandate.
    """
    return [
        "## 1. MISSION",
        "",
        "You are the Critic agent. Review proposals from Orchestration and",
        "emit exactly one `review_verdict` per un-reviewed proposal in",
        "`judge_bundle.proposals`.",
    ]


def _section_run_context(
    *,
    framework: str,
    kernel_enabled: bool,
    max_minutes: int,
) -> list[str]:
    """Build the RUN CONTEXT section lines.

    Args:
        framework (str): The framework name (e.g. ``sglang``) shown verbatim.
        kernel_enabled (bool): Whether kernel-owned actions are enabled.
        max_minutes (int): Wall-clock budget for the run, in minutes.

    Returns:
        list[str]: Markdown lines describing the static run context.
    """
    return [
        "## 2. RUN CONTEXT",
        "",
        f"- framework      : {framework}",
        f"- kernel_enabled : {'true' if kernel_enabled else 'false'}",
        f"- max_minutes    : {max_minutes}",
        "",
        "Per-tick dynamic context (inbox, proposals, KB priors) arrives in",
        "the user message as a judge bundle — not in this system prompt.",
        "",
        "Every `judge_bundle` you receive carries a `phase` field",
        "(PRELUDE / FRAMEWORK_PR / EXPLORE / KERNEL / SWEEP / CLOSE). Use the phase-",
        "specific review rules in §6 to interpret each proposal in",
        "context. Phase fit is strategy: when a proposal looks out-of-",
        "phase or out-of-sequence, prefer `advise` over `reject` and",
        "let PolicyGate R1 / Orchestration handle the actual rerouting.",
        "Reserve `reject` for the SKILL.md Hard Rules safety carve-outs.",
    ]


def _section_phase_review_contract() -> list[str]:
    """Static phase-aware verdict contract (v0.8 §3.3 §4.3); mirrors
    ``phase_state.PHASE_LLM_PROPOSABLE_ACTIONS`` (PolicyGate R1)."""
    from ..phase_state import (
        PHASE_NAMES,
        is_phase_interleave_enabled,
        llm_proposable_actions_for_with_interleave,
    )

    interleave = is_phase_interleave_enabled()
    lines: list[str] = [
        "## 5. PHASE REVIEW CONTRACT (v0.8 §3.3)",
        "",
        "Each `judge_bundle` carries a `phase` (PRELUDE / FRAMEWORK_PR /",
        "EXPLORE / KERNEL / SWEEP / CLOSE). Phase-proposable action sets:",
        "",
    ]
    for phase in PHASE_NAMES:
        proposable = sorted(
            llm_proposable_actions_for_with_interleave(
                phase, interleave=interleave,
            )
        )
        lines.append(f"- **{phase}**: {', '.join(proposable)}")
    lines.extend([
        "",
        "Phase fit is a strategy concern, not a safety concern: the",
        "Coordinator's PolicyGate R1 already blocks any out-of-phase",
        "action before it reaches you. If a proposal somehow slips",
        "through (legacy / resume / interleave), prefer `advise` with",
        "`reasoning='phase_incompatible: action <name> not allowed in",
        "<phase>'` so the LLM can self-correct without a hard reject.",
        "Reserve `reject` for the safety carve-outs in the SKILL.md",
        "Hard Rules (mismatched benchmark, accuracy gate failure,",
        "missing rollback, robustness conflict, payload-shape /",
        "provenance violations).",
        "",
        "``explore`` grids run their variants directly (each is",
        "benchmarked and judged by the KEEP threshold), so they are not",
        "routed to you for pre-review. Review the single-action",
        "proposals you do receive with one verdict each.",
    ])
    if interleave:
        lines.extend([
            "",
            "Phase interleave is ON: EXPLORE additionally accepts kernel-",
            "owned REQUEST kinds and KERNEL additionally accepts explore /",
            "specialist / integrate_patch. The kernel-owned data-dependency",
            "and integrate_patch Critic gates still apply.",
        ])
    return lines


def _actions_by_family(actions: list[ActionMetadata]) -> list[tuple[str, list[ActionMetadata]]]:
    """Group actions by family in stable display order.

    Args:
        actions (list[ActionMetadata]): The enabled actions to group.

    Returns:
        list[tuple[str, list[ActionMetadata]]]: ``(family, sorted_actions)``
        pairs ordered by :data:`_FAMILY_ORDER`, with any unlisted families
        appended alphabetically.
    """
    bucket: dict[str, list[ActionMetadata]] = {}
    for meta in actions:
        bucket.setdefault(meta.family, []).append(meta)
    ordered: list[tuple[str, list[ActionMetadata]]] = []
    seen: set[str] = set()
    for family in _FAMILY_ORDER:
        if family not in bucket:
            continue
        items = sorted(bucket[family], key=lambda a: a.name)
        ordered.append((family, items))
        seen.add(family)
    for family in sorted(bucket.keys()):
        if family not in seen:
            ordered.append((family, sorted(bucket[family], key=lambda a: a.name)))
    return ordered


def _section_known_actions(actions: list[ActionMetadata]) -> list[str]:
    """Build the KNOWN ACTIONS catalogue section, grouped by family.

    Args:
        actions (list[ActionMetadata]): The actions enabled for this run.

    Returns:
        list[str]: Markdown lines listing each action with its accuracy risk,
        family, and description.
    """
    lines: list[str] = [
        "## 3. KNOWN ACTIONS",
        "",
        "Every action name below is known for this run. Default to `approve`",
        "unless §4 or §6 says otherwise.",
        "",
    ]
    for family, items in _actions_by_family(actions):
        lines.append(f"### {family}")
        lines.append("")
        for meta in items:
            lines.append(
                f"- **{meta.name}** "
                f"(acc_risk={meta.accuracy_risk:.2f}  family={meta.family}) "
                f"— {meta.description}"
            )
        lines.append("")
    return lines


def _section_default_verdict(actions: list[ActionMetadata]) -> list[str]:
    """Build the DEFAULT VERDICT section, including the high-risk action list.

    Args:
        actions (list[ActionMetadata]): The actions enabled for this run; those
            with ``accuracy_risk > 0.30`` are surfaced as high-risk.

    Returns:
        list[str]: Markdown lines describing the default verdict rules by
        accuracy risk and family.
    """
    high_risk = sorted(
        a.name for a in actions if a.accuracy_risk > 0.30
    )
    high_risk_line = (
        ", ".join(high_risk) if high_risk else "(none in this run)"
    )
    return [
        "## 4. DEFAULT VERDICT",
        "",
        "Per-action ``accuracy_risk`` / ``crash_risk`` / ``family`` are",
        "prompt-advisory metadata, not hard gates. The Critic",
        "only rejects on the safety carve-outs in §6 (mismatched",
        "benchmark, accuracy gate fail, missing rollback, robustness",
        "conflict, payload / provenance violations). Strategy concerns",
        "surface as ``advise`` so the patch flows through to benchmark +",
        "stack rebench for adjudication.",
        "",
        "- `accuracy_risk == 0` → `approve` unless duplicate proposal or",
        "  `judge_bundle.required_context` is non-empty.",
        "- `0 < accuracy_risk <= 0.30` → `approve` with `predicted_gain_pct` set.",
        "- `accuracy_risk > 0.30` → `advise` and call out the per-action",
        "  risk in `notes`; only escalate to `reject` when the safety",
        "  carve-outs in §6 actually fire.",
        f"  Higher-risk actions this run: {high_risk_line}.",
        "- `family == \"deep_kernel\"` → `approve` (Orchestration sends these",
        "  via REQUEST; you OK the proposal flow).",
        "- Unknown `action_name` (not listed in §3) → `reject`.",
    ]


def _section_kernel_owned_carveout() -> list[str]:
    """Build the KERNEL-OWNED CARVE-OUT section.

    Returns:
        list[str]: Markdown lines listing the kernel-owned actions and noting
        that hard E2E gating is enforced by the Kernel agent, not the verdict.
    """
    owned = ", ".join(sorted(KERNEL_OWNED_ACTIONS))
    return [
        "## 5b. KERNEL-OWNED CARVE-OUT",
        "",
        "These actions use `request{target_agent='kernel', kind=...}`, not",
        "`propose_action`. Your job is to OK the proposal flow:",
        "",
        f"  {owned}",
        "",
        "Hard E2E gating (correctness, 1.20× speedup, accuracy gate) is",
        "enforced by the Kernel agent, not by your verdict.",
    ]


def _section_rules(rules_md: str) -> list[str]:
    """Build the RULES section wrapping the ``critic.md`` fragment.

    Args:
        rules_md (str): The raw rules-fragment markdown; a placeholder is used
            when empty.

    Returns:
        list[str]: Markdown lines for the RULES section.
    """
    body = rules_md.strip() or (
        "(critic.md rules fragment not found — honor judge_bundle constraints.)"
    )
    return ["## 6. RULES", "", body]


def _section_output_protocol() -> list[str]:
    """Build the OUTPUT PROTOCOL section.

    Returns:
        list[str]: Markdown lines documenting the single-proposal and batch
        ``verdict_map`` reply shapes and their rules.
    """
    return [
        "## 7. OUTPUT PROTOCOL",
        "",
        "Every reply MUST include one `review_verdict` per proposal.",
        "",
        "ALLOWED_VERDICTS = approve | reject | redirect | advise | needs_review",
        "",
        "### Single-proposal shape",
        "",
        "Use this for every proposal (kernel_opt / integrate / report /",
        "specialist dispatch / any single-action proposal):",
        "",
        "  emit_intent{intent_type='review_verdict', payload={",
        "    target_proposal_msg_id: '<msg_id>',",
        "    verdict: 'approve' | 'reject' | 'redirect' | 'advise' | 'needs_review',",
        "    reasoning: '<one-paragraph rationale>',",
        "  }}",
        "",
        "Required per verdict:",
        "  - target_proposal_msg_id  (from judge_bundle.proposals[*].msg_id)",
        "  - verdict",
        "  - reasoning",
        "",
        "Optional: confidence, predicted_gain_pct (required for approve/redirect),",
        "kb_evidence[], packet_evidence[], risks[], required_evidence[],",
        "alternative_action (must be a §3 name when set), persist_to_kb, notes.",
    ]


def build_critic_prompt(
    *,
    action_registry: ActionRegistry,
    enabled_actions: Iterable[str],
    framework: str = "sglang",
    kernel_enabled: bool | None = None,
    max_minutes: int = 0,
    rules_fragment_path: Path | None = None,
) -> str:
    """Compose the Critic system prompt (deterministic for given inputs).

    Parameters
    ----------
    action_registry:
        Pre-loaded ``ActionRegistry`` (caller calls ``.load()``).
    enabled_actions:
        Action names enabled for this run (same set as orchestration).
    framework:
        ``sglang`` / ``vllm`` / ``atom`` — printed in RUN CONTEXT verbatim.
    kernel_enabled:
        ``None`` derives from whether any KERNEL_OWNED action is enabled.
    max_minutes:
        Wall-clock budget for the run.
    rules_fragment_path:
        Path to ``critic.md`` rules fragment.
    """
    actions = _filter_actions(action_registry, enabled_actions)
    if kernel_enabled is None:
        kernel_enabled = any(a.name in KERNEL_OWNED_ACTIONS for a in actions)
    framework_norm = (framework or "sglang").strip().lower() or "sglang"
    rules_md = _read_rules_fragment(rules_fragment_path)

    sections: list[list[str]] = [
        _section_mission(),
        _section_run_context(
            framework=framework_norm,
            kernel_enabled=kernel_enabled,
            max_minutes=max_minutes,
        ),
        _section_known_actions(actions),
        _section_default_verdict(actions),
        _section_phase_review_contract(),
    ]
    if kernel_enabled:
        sections.append(_section_kernel_owned_carveout())
    sections.append(_section_rules(rules_md))
    sections.append(_section_output_protocol())

    parts = ["\n".join(sect) for sect in sections]
    return "\n\n".join(parts).rstrip() + "\n"


__all__ = [
    "build_critic_prompt",
]
