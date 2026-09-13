# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic-health signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .event_view import EventRow, build_event_view
from .symptom import Symptom, SymptomSeverity


@dataclass
class CriticHealthConfig:
    """Tunables for :func:`evaluate_critic_health_signals`."""

    # KB unreachable across N+ consecutive recent turns.
    min_outage_judges: int = 3
    # critic_unavailable verdicts across N+ consecutive recent items.
    min_unavailable_verdicts: int = 3
    # ``critic-workdir/`` count above this is a leak (2x the backend keep count).
    max_workdir_count: int = 100
    # Log-pattern marker that the runtime-cli timed out.
    runtime_stuck_pattern_marker: str = "runtime.cli"
    # Needs log lines with the marker AND a ``timed out`` substring to fire.
    min_runtime_stuck_hits: int = 1


def evaluate_critic_health_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: CriticHealthConfig | None = None,
) -> list[Symptom]:
    """Run the critic-health rules and aggregate their symptoms."""
    cfg = config or CriticHealthConfig()
    view = build_event_view(ctx.inbox, data.coordinator_events)
    out: list[Symptom] = []
    out.extend(_kb_outage_symptoms(data, cfg))
    out.extend(_unavailable_streak_symptoms(view, cfg))
    out.extend(_prune_stuck_symptoms(data, cfg))
    out.extend(_runtime_stuck_symptoms(data, cfg))
    return out


# KB outage streak


def _kb_outage_symptoms(
    data: SourceData,
    cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Fire ``critic_kb_outage`` for a streak of KB-unreachable judges."""
    critic = data.local_critic_health
    if not isinstance(critic, dict) or not critic:
        return []
    recent = critic.get("recent_judges") or []
    if not isinstance(recent, list) or not recent:
        return []
    # recent_judges is mtime-desc; count leading consecutive ``kb_unreachable`` reasons.
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


# critic_unavailable verdict streak


def _unavailable_streak_symptoms(
    view: list[EventRow],
    cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Detect a streak of consecutive ``critic_unavailable`` verdicts."""
    streak = 0
    samples: list[str] = []
    for ev in reversed(view):
        if ev.topic != "review_verdict":
            continue
        source = str(ev.payload.get("source") or "")
        if source == "critic_unavailable":
            streak += 1
            target = str(ev.payload.get("target_proposal_msg_id") or "")
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


# workdir prune stuck


def _prune_stuck_symptoms(
    data: SourceData,
    cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Fire ``critic_prune_stuck`` when the workdir count leaks past cap."""
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


# runtime-cli timeout


def _runtime_stuck_symptoms(
    data: SourceData,
    cfg: CriticHealthConfig,
) -> list[Symptom]:
    """Collapse ``runtime.cli .* timed out`` log hits into one symptom."""
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
