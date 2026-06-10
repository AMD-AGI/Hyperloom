# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic-health signals (E1 / E2 / E4 / E5).

Critic is the reviewer and no one reviews the reviewer; these detectors catch the
session silently losing its decision gate:

* **E1 ``critic_kb_outage``** — ``judge_bundle.json`` marks
  ``kb_read_skipped_reason="kb_unreachable"`` for ``min_outage_judges`` consecutive turns.
* **E2 ``critic_unavailable_streak``** — ``review_verdict`` events with
  ``source="critic_unavailable"`` for ``min_unavailable_verdicts`` consecutive verdicts.
* **E4 ``critic_prune_stuck``** — ``critic-workdir/`` count past ``max_workdir_count`` (pruner broken).
* **E5 ``critic_runtime_stuck``** — runtime-cli timeout pattern in server logs
  (reuses :data:`local_log_errors`), collapsed into one critic-attributed symptom.

E3 ("critic full-approve drift") is out of scope: it needs critic-agent runtime
invariants Robustness cannot see from coordinator_events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity



@dataclass
class CriticHealthConfig:
    """Tunables for :func:`evaluate_critic_health_signals`.

    Threshold defaults are deliberately permissive — Critic recovers on
    its own most of the time (KB circuit-breaker auto-resets after
    cooldown), so we only escalate when the outage persists.
    """

    # E1 — KB unreachable across N+ consecutive recent turns.
    min_outage_judges: int = 3
    # E2 — critic_unavailable verdicts across N+ consecutive recent items.
    min_unavailable_verdicts: int = 3
    # E4 — ``critic-workdir/`` count above this is a leak. The
    # critic-agent backend caps at 50 by design (CRITIC_AGENT_WORKDIR_KEEP_COUNT)
    # so 100 = 2x the keep count = pruner stuck for >= 50 turns.
    max_workdir_count: int = 100
    # E5 — log-pattern marker that the runtime-cli timed out.
    runtime_stuck_pattern_marker: str = "runtime.cli"
    # E5 needs at least N log lines mentioning the marker AND a
    # ``timed out`` substring to fire (single occurrence may be a one-off).
    min_runtime_stuck_hits: int = 1


def evaluate_critic_health_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: CriticHealthConfig | None = None,
) -> list[Symptom]:
    """Run the E1/E2/E4/E5 critic-health rules and aggregate their symptoms.

    Args:
        ctx (ReactorContext): Reactor context for the current tick.
        data (SourceData): Collected source data including critic-health and
            log telemetry.
        config (CriticHealthConfig | None): Tunables; defaults to
            :class:`CriticHealthConfig` when ``None``.

    Returns:
        list[Symptom]: All critic-health symptoms found this tick, possibly
            empty.
    """
    cfg = config or CriticHealthConfig()
    out: list[Symptom] = []
    out.extend(_kb_outage_symptoms(data, cfg))
    out.extend(_unavailable_streak_symptoms(ctx, data, cfg))
    out.extend(_prune_stuck_symptoms(data, cfg))
    out.extend(_runtime_stuck_symptoms(data, cfg))
    return out


# ---------------------------------------------------------------------------
# E1 — KB outage streak
# ---------------------------------------------------------------------------

def _kb_outage_symptoms(
    data: SourceData, cfg: CriticHealthConfig,
) -> list[Symptom]:
    """E1: fire ``critic_kb_outage`` for a streak of KB-unreachable judges.

    Args:
        data (SourceData): Collected source data including
            ``local_critic_health``.
        cfg (CriticHealthConfig): Tunables (provides the outage streak
            threshold).

    Returns:
        list[Symptom]: A one-element list with the ``critic_kb_outage`` symptom
            when the streak threshold is met, otherwise an empty list.
    """
    critic = data.local_critic_health
    if not isinstance(critic, dict) or not critic:
        return []
    recent = critic.get("recent_judges") or []
    if not isinstance(recent, list) or not recent:
        return []
    # Recent_judges is mtime-desc; walk from the top and count
    # consecutive ``kb_unreachable`` reasons. A single fresh judge
    # without the marker resets the streak.
    streak = 0
    samples: list[str] = []
    for entry in recent:
        if not isinstance(entry, dict):
            break
        reason = entry.get("kb_read_skipped_reason")
        if reason == "kb_unreachable":
            streak += 1
            samples.append(str(entry.get("turn_dir") or ""))
        else:
            break
    if streak < cfg.min_outage_judges:
        return []
    return [
        Symptom(
            name="critic_kb_outage",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"critic KB unreachable across {streak} consecutive turn(s) "
                f"(>= {cfg.min_outage_judges}); verdicts are landing without "
                f"prior recall — decision quality is degraded"
            ),
            evidence={
                "consecutive_judges": streak,
                "threshold": cfg.min_outage_judges,
                "recent_turn_dirs": samples[:5],
                "workdir_root": critic.get("workdir_root"),
            },
            subject={},
            source="local",
            suggestion=(
                "if KB_BASE_URL set, check service health; otherwise "
                "switch to CRITIC_KB_CLIENT_MODE=inmemory or "
                "--critic-mock until KB is restored"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# E2 — critic_unavailable verdict streak
# ---------------------------------------------------------------------------

def _unavailable_streak_symptoms(
    ctx: ReactorContext,
    data: SourceData,
    cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Count consecutive ``critic_unavailable`` verdicts across coordinator_events + inbox."""
    rows: list[dict[str, Any]] = []
    for event in data.coordinator_events:
        if not isinstance(event, dict):
            continue
        if event.get("topic") != "review_verdict":
            continue
        payload = event.get("payload") or {}
        if isinstance(payload, dict):
            rows.append(payload)
    for item in ctx.inbox:
        if item.topic != "review_verdict":
            continue
        if isinstance(item.payload, dict):
            rows.append(item.payload)
    if not rows:
        return []
    streak = 0
    samples: list[str] = []
    for payload in reversed(rows):
        source = str(payload.get("source") or "")
        if source == "critic_unavailable":
            streak += 1
            target = str(payload.get("target_proposal_msg_id") or "")
            if target:
                samples.append(target)
        else:
            break
    if streak < cfg.min_unavailable_verdicts:
        return []
    return [
        Symptom(
            name="critic_unavailable_streak",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"critic returned source='critic_unavailable' for "
                f"{streak} consecutive verdict(s) "
                f"(>= {cfg.min_unavailable_verdicts}); the reviewer "
                f"has fallen into the missing-context fallback"
            ),
            evidence={
                "consecutive_verdicts": streak,
                "threshold": cfg.min_unavailable_verdicts,
                "sample_targets": samples[:5],
            },
            subject={},
            source="coordinator_events",
            suggestion=(
                "switch to --critic-mock; inspect "
                "critic-workdir/<latest>/judge_bundle.json for the "
                "missing required_context and supply it via manifest"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# E4 — workdir prune stuck
# ---------------------------------------------------------------------------

def _prune_stuck_symptoms(
    data: SourceData, cfg: CriticHealthConfig,
) -> list[Symptom]:
    """E4: fire ``critic_prune_stuck`` when the workdir count leaks past cap.

    Args:
        data (SourceData): Collected source data including
            ``local_critic_health``.
        cfg (CriticHealthConfig): Tunables (provides the max workdir count).

    Returns:
        list[Symptom]: A one-element list with the ``critic_prune_stuck`` symptom
            when the workdir count is over cap, otherwise an empty list.
    """
    critic = data.local_critic_health
    if not isinstance(critic, dict) or not critic:
        return []
    count = critic.get("workdir_count")
    if not isinstance(count, int) or count <= cfg.max_workdir_count:
        return []
    return [
        Symptom(
            name="critic_prune_stuck",
            severity=SymptomSeverity.MEDIUM,
            summary=(
                f"critic-workdir has {count} entries (> "
                f"{cfg.max_workdir_count}); the in-process pruner is "
                f"failing — disk leak on the critic side"
            ),
            evidence={
                "workdir_count": count,
                "threshold": cfg.max_workdir_count,
                "workdir_root": critic.get("workdir_root"),
            },
            subject={},
            source="local",
            suggestion=(
                "monitor disk; if it keeps growing, manually delete "
                "older critic-workdir/<turn>/ entries between sessions"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# E5 — runtime-cli timeout
# ---------------------------------------------------------------------------

def _runtime_stuck_symptoms(
    data: SourceData, cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Collapse ``runtime.cli .* timed out`` log hits into a critic-specific symptom."""
    errors = data.local_log_errors
    if not isinstance(errors, list) or not errors:
        return []
    hits: list[dict[str, Any]] = []
    for entry in errors:
        if not isinstance(entry, dict):
            continue
        line = str(entry.get("line") or "")
        if cfg.runtime_stuck_pattern_marker not in line:
            continue
        if "timed out" not in line:
            continue
        hits.append(entry)
    if len(hits) < cfg.min_runtime_stuck_hits:
        return []
    return [
        Symptom(
            name="critic_runtime_stuck",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"critic runtime.cli timed out {len(hits)} time(s); the "
                f"subprocess transport is stuck — every tick is paying "
                f"the timeout budget"
            ),
            evidence={
                "hit_count": len(hits),
                "samples": [str(h.get("line") or "")[:200] for h in hits[:3]],
            },
            subject={},
            source="local",
            suggestion=(
                "switch to --critic-mock to unblock the loop; investigate "
                "whether the codex chat-completion endpoint is hung"
            ),
        )
    ]


__all__ = [
    "CriticHealthConfig",
    "evaluate_critic_health_signals",
]
