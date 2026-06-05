"""Compose the Critic agent's system prompt from typed inputs.

Replaces the hand-maintained action whitelist in ``critic.md``:

* ``critic.md`` is now only the *rules fragment* (deviation cases,
  hard rules) — content that does not depend on which actions are
  enabled for this run.

* This module wraps that fragment with:
    1. MISSION
    2. RUN CONTEXT
    3. KNOWN ACTIONS (from ActionRegistry, filtered by enabled_actions)
    4. DEFAULT VERDICT
    5. KERNEL-OWNED CARVE-OUT (when kernel_enabled)
    6. RULES (critic.md verbatim)
    7. OUTPUT PROTOCOL

Output is deterministic given the same inputs. The CLI snapshots the
result to ``agents/critic/system_prompt.snapshot.md`` for resume / audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..action_registry import ActionMetadata, ActionRegistry
from .prompt_builder import KERNEL_OWNED_ACTIONS, _filter_actions


# Stable family ordering for the catalogue section.
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
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _section_mission() -> list[str]:
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
        "context. Reject proposals that mutate kernel source while the",
        "run is in EXPLORE phase (rule = 'kernel-source-in-explore').",
    ]


def _section_phase_review_contract() -> list[str]:
    """Static phase-aware verdict contract (v0.8 §3.3 §4.3).

    Mirrors the per-phase LLM-proposable map in
    ``phase_state.PHASE_LLM_PROPOSABLE_ACTIONS`` so the Critic verdict
    process stays aligned with PolicyGate R1's phase_incompatible
    rule and only sees what the LLM can actually propose. The dynamic
    *current* phase is in ``judge_bundle.phase`` each tick.
    """
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
        "If the proposal's `action_name` is NOT in the bundle's phase",
        "allowlist, return `reject` with",
        "`reasoning='phase_incompatible: action <name> not allowed in",
        "<phase>'`. PolicyGate R1 will have already blocked most such",
        "proposals before they reach you, but the verdict closes the",
        "loop and surfaces the denial in `policy_denial_history`.",
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
    high_risk = sorted(
        a.name for a in actions if a.accuracy_risk > 0.30
    )
    high_risk_line = (
        ", ".join(high_risk) if high_risk else "(none in this run)"
    )
    return [
        "## 4. DEFAULT VERDICT",
        "",
        "- `accuracy_risk == 0` → `approve` unless duplicate proposal or",
        "  `judge_bundle.required_context` is non-empty.",
        "- `0 < accuracy_risk <= 0.30` → `approve` with `predicted_gain_pct` set.",
        f"- `accuracy_risk > 0.30` → `reject` unless `notes` justify the risk.",
        f"  High-risk actions this run: {high_risk_line}.",
        "- `family == \"deep_kernel\"` → `approve` (Orchestration sends these",
        "  via REQUEST; you OK the proposal flow).",
        "- Unknown `action_name` (not listed in §3) → `reject`.",
    ]


def _section_kernel_owned_carveout() -> list[str]:
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
    body = rules_md.strip() or (
        "(critic.md rules fragment not found — honor judge_bundle constraints.)"
    )
    return ["## 6. RULES", "", body]


def _section_output_protocol() -> list[str]:
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
    """Compose the Critic system prompt.

    Parameters
    ----------
    action_registry:
        Pre-loaded ``ActionRegistry`` (caller should call ``.load()``).
    enabled_actions:
        Action names enabled for this run (same set as orchestration).
    framework:
        ``sglang`` / ``vllm`` / ``atom`` — printed in RUN CONTEXT
        verbatim. The value is rendered as-is into the prompt with no
        framework-specific rule text, so atom candidates are reviewed
        against the same generic rule set as sglang / vllm.
    kernel_enabled:
        When ``None``, derived from whether any KERNEL_OWNED action is
        in ``enabled_actions``.
    max_minutes:
        Wall-clock budget for the run.
    rules_fragment_path:
        Path to ``critic.md`` rules fragment.

    Returns
    -------
    str
        Full system prompt, deterministic for given inputs.
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
        # phase review contract (per-phase allowlist +
        # specialist batch verdict shape).
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
