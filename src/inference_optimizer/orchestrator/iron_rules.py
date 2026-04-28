"""Iron Rules — DESIGN §4.5.

七条不可破规则。注入到所有 reactor system prompt + ActionRegistry 启动校验里。

STATUS (v0.7):
    All seven IR-1..IR-7 predicates land here. ``validate_action`` collects
    every violation rather than fail-fast, so a misconfigured action yields
    a complete report. ``render_for_prompt`` produces the markdown bullet
    block injected into every reactor prompt (DESIGN §6.3).

References:
    - DESIGN §4.5 Iron Rules (IR-1..IR-7)
    - DESIGN §10.5.7 PolicyGate role permissions (uses IronRule)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .execution_mode import ExecutionMode


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class IronRule:
    """One Iron Rule entry — see DESIGN §4.5."""

    id: str  # IR-1 ... IR-7
    description: str
    applies_to_modes: tuple[str, ...]  # "quick" / "guided" / "marathon" / "*"
    severity: Severity = Severity.BLOCK


@dataclass(frozen=True)
class Violation:
    rule_id: str
    reason: str
    severity: Severity = Severity.BLOCK


# ---------------------------------------------------------------------------
# Rule registry — IR-1..IR-7
# ---------------------------------------------------------------------------
IRON_RULES: tuple[IronRule, ...] = (
    IronRule(
        id="IR-1",
        description=(
            "Always submit kernel candidates IN PARALLEL via GEAK MCP "
            "(no sequential single-candidate submissions)."
        ),
        applies_to_modes=("guided", "marathon"),
    ),
    IronRule(
        id="IR-2",
        description=(
            "Never modify kernel source files before GEAK auto-tunes them; "
            "kernel-opt action MUST be the first writer."
        ),
        applies_to_modes=("guided", "marathon"),
    ),
    IronRule(
        id="IR-3",
        description=(
            "After kernel-opt success, MUST run integrate action and "
            "validate via scripts/run_baseline.sh."
        ),
        applies_to_modes=("guided", "marathon"),
    ),
    IronRule(
        id="IR-4",
        description=(
            "Before any server launch: kill_server then check_gpu_memory "
            "(no orphan sglang/vllm processes allowed)."
        ),
        applies_to_modes=("*",),
    ),
    IronRule(
        id="IR-5",
        description=(
            "Forbidden: pkill -f sglang. Use pgrep + targeted kill on "
            "sglang.launch_server / vllm.entrypoints.openai.api_server only."
        ),
        applies_to_modes=("*",),
    ),
    IronRule(
        id="IR-6",
        description=(
            "patch_inductor.py MUST receive --target-file and (when changing "
            "block_size or num_warps) --best-config. --cache-dir is removed."
        ),
        applies_to_modes=("guided", "marathon"),
    ),
    IronRule(
        id="IR-7",
        description=(
            "Never modify GEAK MCP config (except for tracing headers per "
            "exception list)."
        ),
        applies_to_modes=("guided", "marathon"),
    ),
)


def all_rules() -> tuple[IronRule, ...]:
    return IRON_RULES


def _normalize_mode(mode: "ExecutionMode | str") -> str:
    """Accept either ``ExecutionMode`` enum or short keyword (``quick`` /
    ``guided`` / ``marathon``) or a full enum value (``quick_param_sweep`` …)."""
    if hasattr(mode, "value"):
        full = str(mode.value)
    else:
        full = str(mode)
    full = full.strip().lower()
    if full.startswith("quick"):
        return "quick"
    if full.startswith("guided"):
        return "guided"
    if full.startswith("marathon"):
        return "marathon"
    return full


def rules_for_mode(mode: "ExecutionMode | str") -> tuple[IronRule, ...]:
    """Return the IRs that apply to the given mode.

    A rule with ``applies_to_modes=("*",)`` matches every mode.
    """
    target = _normalize_mode(mode)
    out: list[IronRule] = []
    for rule in IRON_RULES:
        if "*" in rule.applies_to_modes or target in rule.applies_to_modes:
            out.append(rule)
    return tuple(out)


# ---------------------------------------------------------------------------
# Per-rule predicates (each returns Violation or None)
# ---------------------------------------------------------------------------
def _ir1_parallel_kernel_submission(meta: dict[str, Any]) -> Violation | None:
    """IR-1: kernel-opt MUST submit candidates in parallel via GEAK."""
    if meta.get("family") != "deep_kernel":
        return None
    name = str(meta.get("name", ""))
    if name not in ("kernel-opt", "kernel_opt"):
        return None
    flags = list(meta.get("execution_flags", []) or [])
    if "parallel_geak_submission" in flags:
        return None
    if meta.get("parallel_geak_submission") is True:
        return None
    return Violation(
        rule_id="IR-1",
        reason=(
            "kernel-opt action must declare parallel GEAK submission "
            "(execution_flags must include 'parallel_geak_submission')"
        ),
    )


def _ir2_no_kernel_source_modification_before_geak(
    meta: dict[str, Any],
) -> Violation | None:
    """IR-2: only kernel-opt action may be the first writer to kernel sources."""
    name = str(meta.get("name", ""))
    side_effects = set(meta.get("side_effects", []) or [])
    family = str(meta.get("family", ""))
    if "patches_kernel_source" not in side_effects:
        return None
    if name in ("kernel-opt", "kernel_opt", "integrate"):
        return None
    if family == "deep_kernel" and "patches_workspace" in side_effects:
        # already covered by integrate / kernel-opt path
        return None
    return Violation(
        rule_id="IR-2",
        reason=(
            f"action {name!r} declares 'patches_kernel_source' before "
            "GEAK has run; only kernel-opt or integrate may do that"
        ),
    )


def _ir3_integrate_after_kernel_opt(meta: dict[str, Any]) -> Violation | None:
    """IR-3: kernel-opt MUST list integrate among its required follow-ups."""
    name = str(meta.get("name", ""))
    if name not in ("kernel-opt", "kernel_opt"):
        return None
    follow_ups = list(meta.get("required_follow_ups", []) or [])
    if "integrate" in follow_ups:
        return None
    return Violation(
        rule_id="IR-3",
        reason=(
            "kernel-opt must declare 'integrate' in required_follow_ups "
            "and validate via scripts/run_baseline.sh"
        ),
    )


def _ir4_kill_then_check_gpu(meta: dict[str, Any]) -> Violation | None:
    """IR-4: any action that owns server_lifecycle must list both
    ``kill_server`` and ``check_gpu_memory`` in its preflight ordering."""
    requires_lanes = set(meta.get("requires_lanes", []) or [])
    if "server_lifecycle" not in requires_lanes:
        return None
    preflight = list(meta.get("preflight", []) or [])
    if "kill_server" in preflight and "check_gpu_memory" in preflight:
        return None
    return Violation(
        rule_id="IR-4",
        reason=(
            "actions touching server_lifecycle must declare a preflight "
            "ordering containing kill_server + check_gpu_memory"
        ),
    )


def _ir5_no_pkill_f_sglang(meta: dict[str, Any]) -> Violation | None:
    """IR-5: forbidden command pattern in ``commands`` / ``shell_snippets``."""
    needles = ("pkill -f sglang", "pkill -9 -f sglang")
    candidates: list[str] = []
    for key in ("commands", "shell_snippets", "preflight_commands"):
        for line in meta.get(key, []) or []:
            candidates.append(str(line))
    for line in candidates:
        norm = line.strip().lower()
        for n in needles:
            if n in norm:
                return Violation(
                    rule_id="IR-5",
                    reason=(
                        f"forbidden pattern {n!r} found in action "
                        f"definition (would kill the conductor itself)"
                    ),
                )
    return None


def _ir6_patch_inductor_args(meta: dict[str, Any]) -> Violation | None:
    """IR-6: any patch_inductor invocation must carry --target-file and
    drop --cache-dir; --best-config required when block_size or num_warps
    appear in the ``tuning_keys`` list."""
    invocations = list(meta.get("patch_inductor_invocations", []) or [])
    if not invocations:
        return None
    for inv in invocations:
        if not isinstance(inv, dict):
            continue
        argv: list[str] = list(inv.get("argv", []) or [])
        joined = " ".join(argv)
        if "--target-file" not in joined:
            return Violation(
                rule_id="IR-6",
                reason="patch_inductor invocation missing --target-file",
            )
        if "--cache-dir" in joined:
            return Violation(
                rule_id="IR-6",
                reason="patch_inductor invocation must not pass --cache-dir",
            )
        tuning_keys = set(inv.get("tuning_keys", []) or [])
        if {"block_size", "num_warps"} & tuning_keys and "--best-config" not in joined:
            return Violation(
                rule_id="IR-6",
                reason=(
                    "patch_inductor invocation tunes block_size/num_warps "
                    "but does not pass --best-config"
                ),
            )
    return None


def _ir7_no_geak_config_mutation(meta: dict[str, Any]) -> Violation | None:
    """IR-7: never modify GEAK MCP config (except tracing-headers exception)."""
    side_effects = set(meta.get("side_effects", []) or [])
    if "modifies_geak_config" not in side_effects:
        return None
    exceptions = set(meta.get("geak_config_exceptions", []) or [])
    if exceptions <= {"tracing_headers"} and exceptions:
        return None
    return Violation(
        rule_id="IR-7",
        reason=(
            "action declares 'modifies_geak_config' without listing only "
            "the allowed 'tracing_headers' exception"
        ),
    )


_PREDICATES: tuple = (
    _ir1_parallel_kernel_submission,
    _ir2_no_kernel_source_modification_before_geak,
    _ir3_integrate_after_kernel_opt,
    _ir4_kill_then_check_gpu,
    _ir5_no_pkill_f_sglang,
    _ir6_patch_inductor_args,
    _ir7_no_geak_config_mutation,
)


def validate_action(
    action_metadata: dict[str, Any],
    mode: "ExecutionMode | str",
) -> list[Violation]:
    """Run every rule that applies to ``mode`` against ``action_metadata``.

    Returns a (possibly empty) list of :class:`Violation`. Callers decide
    whether to ``raise`` (BLOCK severity) or merely log (WARN).

    The action metadata is a plain dict with the same keys as
    ``actions/_meta/<name>.yaml`` plus optional execution-time hints
    (``preflight``, ``required_follow_ups``, ``patch_inductor_invocations``,
    ``commands``, ``shell_snippets`` …) that hot-path code may inject.
    """
    target = _normalize_mode(mode)
    active_rule_ids = {r.id for r in rules_for_mode(target)}
    violations: list[Violation] = []
    for predicate, rule in zip(_PREDICATES, IRON_RULES):
        if rule.id not in active_rule_ids:
            continue
        v = predicate(action_metadata)
        if v is not None:
            violations.append(v)
    return violations


def render_for_prompt(mode: "ExecutionMode | str") -> str:
    """Markdown bullet block for the per-role system prompt (DESIGN §6.3)."""
    rules = rules_for_mode(mode)
    if not rules:
        return "## Iron Rules\n  (none for this mode)\n"
    lines = ["## Iron Rules (NEVER violate)"]
    for r in rules:
        lines.append(f"- **{r.id}** ({r.severity.value}): {r.description}")
    return "\n".join(lines) + "\n"


__all__ = [
    "Severity",
    "IronRule",
    "Violation",
    "IRON_RULES",
    "all_rules",
    "rules_for_mode",
    "validate_action",
    "render_for_prompt",
]
