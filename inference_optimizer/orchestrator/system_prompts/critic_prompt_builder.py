# Copyright Advanced Micro Devices, Inc. All rights reserved.

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
        "context. Reject proposals that mutate kernel source while the",
        "run is in EXPLORE phase (rule = 'kernel-source-in-explore').",
    ]


def _section_phase_review_contract() -> list[str]:
    """Static phase-aware verdict contract (v0.8 §3.3 §4.3).

    Mirrors the per-phase allowed-action map in
    ``phase_state.PHASE_ALLOWED_ACTIONS`` so the Critic verdict
    process stays aligned with PolicyGate R1's phase_incompatible
    rule. The dynamic *current* phase is in ``judge_bundle.phase``
    each tick.

    Returns:
        list[str]: Markdown lines listing each phase's allowed actions plus the
        phase-incompatible reject rule and batch verdict-map note.
    """
    from ..phase_state import PHASE_ALLOWED_ACTIONS, PHASE_NAMES

    lines: list[str] = [
        "## 5. PHASE REVIEW CONTRACT (v0.8 §3.3)",
        "",
        "Each `judge_bundle` carries a `phase` (PRELUDE / FRAMEWORK_PR /",
        "EXPLORE / KERNEL / SWEEP / CLOSE). Phase-allowed action sets:",
        "",
    ]
    for phase in PHASE_NAMES:
        allowed = sorted(PHASE_ALLOWED_ACTIONS.get(phase, frozenset()))
        lines.append(f"- **{phase}**: {', '.join(allowed)}")
    lines.extend([
        "",
        "If the proposal's `action_name` is NOT in the bundle's phase",
        "allowlist, return `reject` with",
        "`reasoning='phase_incompatible: action <name> not allowed in",
        "<phase>'`. PolicyGate R1 will have already blocked most such",
        "proposals before they reach you, but the verdict closes the",
        "loop and surfaces the denial in `policy_denial_history`.",
        "",
        "Specialist proposal_set packets (M5+) arrive bundled as a",
        "single `propose_action='explore'` whose `payload.params.grid`",
        "is a K-entry list. Respond with the legacy ``verdict_map``",
        "shape (§7) so the Coordinator can dispatch only the approved",
        "subset; missing entries are treated as `needs_review` and",
        "skipped.",
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
        "- `accuracy_risk == 0` → `approve` unless duplicate proposal or",
        "  `judge_bundle.required_context` is non-empty.",
        "- `0 < accuracy_risk <= 0.30` → `approve` with `predicted_gain_pct` set.",
        "- `accuracy_risk > 0.30` → `reject` unless `notes` justify the risk.",
        f"  High-risk actions this run: {high_risk_line}.",
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
        "### Single-proposal shape (v0.6, kept for non-grid actions)",
        "",
        "Use this for kernel_opt / integrate / report / specialist dispatch",
        "and any other single-action proposal:",
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
        "",
        "### Batch shape — per-variant verdict_map (v0.8 KB_gaps/Gap-11)",
        "",
        "When the proposal is a multi-variant ``explore`` grid (specialist",
        "proposal_set or LLM-direct), return one verdict *per variant* via",
        "``verdict_map`` so the Coordinator can dispatch only the approved",
        "subset (not the legacy all-or-nothing 'approve' / 'reject'):",
        "",
        "  emit_intent{intent_type='review_verdict', payload={",
        "    target_proposal_msg_id: '<msg_id>',",
        "    verdict_map: {",
        "      '<variant_name_A>': {verdict: 'approve', rationale: '<why>'},",
        "      '<variant_name_B>': {verdict: 'reject',  rationale: 'KB shows similar tried 3x, all failed'},",
        "      '<variant_name_C>': {verdict: 'needs_review', rationale: '<missing context>'},",
        "    },",
        "    reasoning: '<round-level summary>',",
        "  }}",
        "",
        "Rules:",
        "",
        "* ``verdict`` and ``verdict_map`` are MUTUALLY EXCLUSIVE — pick one.",
        "* Each ``verdict_map[name].verdict`` must be in ALLOWED_VERDICTS.",
        "* Keys MUST match a variant from the proposal's original ``grid``;",
        "  unknown names are dropped + surfaced in the policy_denial",
        "  observation log.",
        "* Variants you ``reject`` are immediately recorded as ``refuted``",
        "  in Cortex KB (no need to wait for explore to run). Use this to",
        "  prune known-bad variants pre-dispatch.",
        "* Variants you omit are treated as ``needs_review`` — neither",
        "  dispatched nor refuted; they appear in the unknown bucket if",
        "  also missing from the original grid.",
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

    Assembles the mission, run context, known actions, default verdict, phase
    review contract, optional kernel carve-out, rules fragment, and output
    protocol into a single deterministic prompt.

    Args:
        action_registry (ActionRegistry): Pre-loaded registry (caller should
            call ``.load()``).
        enabled_actions (Iterable[str]): Action names enabled for this run (same
            set as orchestration).
        framework (str): ``sglang`` / ``vllm`` / ``atom`` — printed in RUN
            CONTEXT verbatim with no framework-specific rule text, so atom
            candidates are reviewed against the same generic rules.
        kernel_enabled (bool | None): When ``None``, derived from whether any
            KERNEL_OWNED action is in ``enabled_actions``.
        max_minutes (int): Wall-clock budget for the run.
        rules_fragment_path (Path | None): Path to the ``critic.md`` rules
            fragment.

    Returns:
        str: The full Critic system prompt, deterministic for given inputs.
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
