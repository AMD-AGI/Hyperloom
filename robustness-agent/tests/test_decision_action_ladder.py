"""Unit tests for :class:`ActionLadder`."""

from __future__ import annotations

import pytest

from robustness_agent.decision.action_ladder import (
    ActionLadder,
    ActionLadderConfig,
)
from robustness_agent.role.envelope import IntentType
from robustness_agent.signals import Symptom, SymptomSeverity

pytestmark = pytest.mark.asyncio


def _sym(
    name: str,
    severity: SymptomSeverity,
    *,
    summary: str = "summary",
    subject: dict | None = None,
    evidence: dict | None = None,
    suggestion: str = "",
) -> Symptom:
    return Symptom(
        name=name,
        severity=severity,
        summary=summary,
        evidence=evidence or {},
        subject=subject or {"k": "v"},
        source="test",
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

async def test_low_severity_yields_observation_send_message():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("pod_no_metrics", SymptomSeverity.LOW)],
        tick_index=0,
        now_unix=1.0,
    )
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.SEND_MESSAGE
    assert out.intents[0].payload["topic"] == "observation"
    assert out.findings and out.findings[0].symptom_name == "pod_no_metrics"


async def test_medium_severity_yields_alert_only():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
    )
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.ALERT
    assert out.intents[0].payload["severity"] == "medium"


async def test_high_crash_emits_alert_plus_escalate():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("crash_count_high", SymptomSeverity.HIGH, suggestion="revert")],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    escalate = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert escalate.payload["next_action_hint"] == "revert"


async def test_high_cluster_fault_emits_alert_plus_escalate():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "cluster_fault",
                SymptomSeverity.HIGH,
                summary="cluster fault on g53",
                subject={"node": "g53", "fault": "g53-gpu_ecc"},
                suggestion="drain g53",
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    escalate = next(
        i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE
    )
    assert escalate.payload["reason"] == "cluster_fault_high"
    assert escalate.payload["next_action_hint"] == "drain g53"
    assert escalate.payload["severity"] == "high"


async def test_medium_cluster_fault_emits_alert_only():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "cluster_fault",
                SymptomSeverity.MEDIUM,
                summary="cluster fault on g53",
                subject={"node": "g53", "fault": "g53-gpu_ecc"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert types == [IntentType.ALERT]
    assert out.intents[0].payload["severity"] == "medium"


async def test_high_agent_stall_emits_escalate_with_default_hint():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("agent_stall", SymptomSeverity.HIGH, subject={"agent": "kernel"})],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_repeated_failure_with_family_triggers_prune_branch():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "repeated_failure",
                SymptomSeverity.HIGH,
                evidence={"family": "kernel_opt", "count": 3},
                subject={"family": "kernel_opt"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


# ---------------------------------------------------------------------------
# Wind-down path: recover_unsuccessful / deadline_imminent → delegate(report)
# ---------------------------------------------------------------------------

async def test_recover_unsuccessful_emits_escalate_plus_delegate_report():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "recover_unsuccessful",
                SymptomSeverity.HIGH,
                summary="recover state=needs_review (gpu_unhealthy_after_gpureset)",
                evidence={
                    "task_id": "tsk-1",
                    "kind": "recover",
                    "error_class": "gpu_unhealthy_after_gpureset",
                    "force_gpu_cleanup": True,
                    "gpureset_attempted": True,
                    "post_free_mb_per_gpu": [{"gpu_id": 0, "free_mb": 12.0}],
                },
                subject={},  # session-wide
                suggestion="delegate(report) to finalize at the last validated gain",
            )
        ],
        tick_index=7,
        now_unix=1700000000.0,
    )
    types = [i.type for i in out.intents]
    # base recommend tier always emits an alert(high); we additionally
    # need an escalate + a delegate(report) carrying the evidence.
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.DELEGATE in types

    delegate = next(i for i in out.intents if i.type is IntentType.DELEGATE)
    assert delegate.payload["action_name"] == "report"
    assert delegate.payload["params"]["reason"] == "recover_unsuccessful"
    assert (
        delegate.payload["params"]["evidence"]["error_class"]
        == "gpu_unhealthy_after_gpureset"
    )
    assert delegate.payload["idempotency_key"] == (
        "report-recover-unsuccessful-tick-7"
    )


async def test_state_json_corrupt_escalates():
    """I1: state.json broken → HIGH escalate only (no prune; can't auto-heal)."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("state_json_corrupt", SymptomSeverity.HIGH,
              evidence={"path": "/p/state.json"}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types
    assert IntentType.DELEGATE not in types
    assert IntentType.KILL_TASK not in types


async def test_coordinator_wal_bloat_high_escalates():
    """I2: HIGH (4 GiB+) escalates with checkpoint hint."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("coordinator_wal_bloat", SymptomSeverity.HIGH,
              evidence={"wal_bytes": 5 * 1024 * 1024 * 1024}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_coordinator_wal_bloat_medium_alert_only():
    """I2 MEDIUM (1-4 GiB) falls to default _diagnose tier."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("coordinator_wal_bloat", SymptomSeverity.MEDIUM,
              evidence={"wal_bytes": 2 * 1024 * 1024 * 1024}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE not in types
    assert IntentType.KILL_TASK not in types


async def test_stale_lease_emits_kill_task_for_owner_lane():
    """I3: HIGH emits kill_task(task_id) + escalate."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("stale_lease", SymptomSeverity.HIGH,
              evidence={"task_id": "tsk-7", "lane": "lane-1",
                        "holder_pid": 9999},
              subject={"task_id": "tsk-7"})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.KILL_TASK in types
    kill = next(i for i in out.intents if i.type is IntentType.KILL_TASK)
    assert kill.payload["task_id"] == "tsk-7"
    assert kill.payload["reason"] == "stale_lease"
    assert kill.payload["scope"] == "task"
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_stale_lease_without_task_id_does_not_kill():
    """If evidence lacks task_id we skip the kill_task to avoid bad payloads."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("stale_lease", SymptomSeverity.HIGH,
              evidence={}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.KILL_TASK not in types
    # Escalate still fires.
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_inbox_bloat_low_emits_observation_only():
    """I4 LOW falls to the observe tier (send_message)."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("inbox_bloat", SymptomSeverity.LOW,
              evidence={"role": "orchestration", "kind": "inbox"},
              subject={"role": "orchestration", "kind": "inbox"})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    # LOW tier → send_message(observation), not alert / escalate.
    msg_types = {
        i.payload.get("topic") for i in out.intents
        if i.type is IntentType.SEND_MESSAGE
    }
    assert "observation" in msg_types
    assert IntentType.KILL_TASK not in types


async def test_coordinator_zombie_escalates_critical():
    """I5: HIGH escalate (cannot self-heal — Robustness lives in the
    same process tree)."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("coordinator_zombie", SymptomSeverity.HIGH,
              evidence={"recorded_pid": 1234}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "operator restart" in esc.payload["next_action_hint"]


async def test_gateway_auth_outage_escalates():
    """J1: HIGH escalate w/ key-rotation hint."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("gateway_auth_outage", SymptomSeverity.HIGH,
              evidence={"status_code": 401, "url": "https://gw/v1/models"},
              subject={})],
        tick_index=0, now_unix=1.0,
    )
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "$SAFE_API_KEY" in esc.payload["next_action_hint"]


async def test_wekafs_degraded_escalates_no_prune():
    """J2: HIGH escalate — operator decides wait vs remount."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("wekafs_degraded", SymptomSeverity.HIGH,
              evidence={"env_name": "TRACELENS_ROOT",
                        "path": "/wekafs/hyperloom"},
              subject={"path": "/wekafs/hyperloom"})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types


async def test_tracelens_cli_missing_escalates_to_install_sh():
    """J3: HIGH escalate — re-run install.sh."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("tracelens_cli_missing", SymptomSeverity.HIGH,
              evidence={"cli_names": ["a", "b"]}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "install.sh" in esc.payload["next_action_hint"]


async def test_critic_kb_outage_escalates_no_prune():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("critic_kb_outage", SymptomSeverity.HIGH, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types
    assert IntentType.DELEGATE not in types


async def test_critic_unavailable_streak_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("critic_unavailable_streak", SymptomSeverity.HIGH, subject={})],
        tick_index=0, now_unix=1.0,
    )
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "--critic-mock" in esc.payload["next_action_hint"]


async def test_critic_runtime_stuck_escalates_to_mock():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("critic_runtime_stuck", SymptomSeverity.HIGH, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_ray_pending_starvation_prunes_kernel_opt():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("ray_pending_starvation", SymptomSeverity.HIGH,
              evidence={"pending_tasks": 50}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


async def test_geak_budget_starvation_escalates_no_prune():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("geak_budget_starvation", SymptomSeverity.HIGH,
              evidence={"kernel_id": "rms_norm"},
              subject={"kernel_id": "rms_norm"})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types


async def test_kernel_opt_no_progress_prunes_and_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_sym("kernel_opt_no_progress", SymptomSeverity.HIGH,
              evidence={"kernel_count": 3}, subject={})],
        tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


async def test_critic_prune_stuck_falls_to_medium_alert():
    """E4 / F4 — MEDIUM severity → alert only, no destructive action."""
    ladder = ActionLadder()
    for name in ("critic_prune_stuck", "cursor_auth_storm"):
        out = await ladder.decide(
            [_sym(name, SymptomSeverity.MEDIUM, subject={})],
            tick_index=0, now_unix=1.0,
        )
        types = [i.type for i in out.intents]
        assert IntentType.ALERT in types
        assert IntentType.PRUNE_BRANCH not in types
        assert IntentType.DELEGATE not in types


async def test_model_gpu_infeasible_prunes_all_server_families():
    """C1: prune every action family that would launch a server."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "model_gpu_infeasible",
                SymptomSeverity.HIGH,
                summary="DSR1 671B FP8 TP=1 on MI300X doesn't fit",
                evidence={
                    "model_name": "DeepSeek-R1-0528-671B",
                    "gpu_type": "mi300x",
                    "tp": 1,
                    "required_gib": 704.0,
                    "hbm_gib": 192.0,
                    "headroom_pct": -266.0,
                },
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    families_pruned = {
        i.payload["family"] for i in out.intents
        if i.type is IntentType.PRUNE_BRANCH
    }
    # Every server-launching family is pruned.
    assert families_pruned >= {
        "baseline", "backends", "params", "sweep",
        "validate_stack", "kernel_opt",
    }
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types


async def test_amdahl_kernel_ceiling_prunes_kernel_opt():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "amdahl_kernel_ceiling_low",
                SymptomSeverity.HIGH,
                evidence={
                    "optimizable_pct": 5.0,
                    "e2e_ceiling_pct": 1.66,
                },
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"
    assert prune.payload["reason"] == "amdahl_kernel_ceiling_low"


async def test_cold_start_budget_exhausted_escalates_no_prune():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "cold_start_budget_exhausted",
                SymptomSeverity.HIGH,
                evidence={
                    "so_count": 5,
                    "remaining_minutes": 30.0,
                    "cold_start_minutes": 60.0,
                },
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    # Cold start doesn't prune — operator may extend timeout instead.
    assert IntentType.PRUNE_BRANCH not in types


async def test_empty_patch_kept_prunes_kernel_opt_and_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "empty_patch_kept",
                SymptomSeverity.HIGH,
                summary="KEEP on k1 with patch_size_bytes=0",
                evidence={
                    "kernel_id": "k1",
                    "decision": "KEEP",
                    "patch_size_bytes": 0,
                    "gain_pct": 0.05,
                },
                subject={"kernel_id": "k1", "patch_path": "/tmp/p.diff"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


async def test_kernel_dispatch_bypassed_prunes_and_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "kernel_dispatch_bypassed",
                SymptomSeverity.HIGH,
                evidence={
                    "kernel_id": "k7",
                    "dispatched_count": 0,
                    "gain_pct": 0.04,
                },
                subject={"kernel_id": "k7"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "k7" in esc.payload["next_action_hint"]


async def test_kernel_negative_delta_kept_escalates_no_prune():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "kernel_negative_delta_kept",
                SymptomSeverity.HIGH,
                evidence={
                    "kernels_optimized": 6,
                    "optimized_kernel_delta_pct": -0.169,
                },
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    # No prune — this is a roll-back recommendation, not a family kill.
    assert IntentType.PRUNE_BRANCH not in types


async def test_ci_metrics_baseline_zero_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "ci_metrics_baseline_zero",
                SymptomSeverity.HIGH,
                evidence={
                    "ci_metrics_path": "/p/ci_metrics.json",
                    "baseline_values": [0.0],
                },
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types


async def test_oob_no_harness_prunes_kernel_opt():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "oob_no_harness",
                SymptomSeverity.HIGH,
                evidence={
                    "kernel_id": "gemm_a8w8",
                    "backend": "oob_claude",
                    "microbench_speedup": None,
                },
                subject={"kernel_id": "gemm_a8w8"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"
    assert prune.payload["reason"] == "oob_no_harness"


async def test_g_medium_signals_fall_to_diagnose_alert():
    """G2 / G6 — MEDIUM severity → alert only, no destructive action."""
    ladder = ActionLadder()
    for name in ("decision_threshold_violated", "ci_metrics_schema_drift"):
        out = await ladder.decide(
            [_sym(name, SymptomSeverity.MEDIUM, subject={})],
            tick_index=0,
            now_unix=1.0,
        )
        types = [i.type for i in out.intents]
        assert IntentType.ALERT in types
        assert IntentType.PRUNE_BRANCH not in types
        assert IntentType.DELEGATE not in types
        assert IntentType.ESCALATE_STRATEGY_CHANGE not in types


async def test_deadline_warning_high_emits_delegate_report():
    """``deadline_warning`` HIGH = absolute-time backstop for the
    no-validated-gain case. Behaves the same as ``deadline_imminent``
    in the ladder: escalate + delegate(report). The MEDIUM branch is
    covered separately because it falls through the default _diagnose
    rung (no destructive action)."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "deadline_warning",
                SymptomSeverity.HIGH,
                summary="only 25.0min remain (<= 30min); wind down now",
                evidence={
                    "elapsed_minutes": 1415.0,
                    "remaining_minutes": 25.0,
                    "budget_minutes": 1440.0,
                    "cumulative_gain_validated": 0.0,
                    "deadline_warning_minutes": 30.0,
                },
                subject={},
            )
        ],
        tick_index=200,
        now_unix=1700001000.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.DELEGATE in types
    delegate = next(i for i in out.intents if i.type is IntentType.DELEGATE)
    assert delegate.payload["action_name"] == "report"
    assert delegate.payload["params"]["reason"] == "deadline_warning"
    assert delegate.payload["idempotency_key"] == (
        "report-deadline-warning-tick-200"
    )


async def test_deadline_warning_medium_only_emits_alert():
    """MEDIUM = validated gain present + 30 min remain → no delegate."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "deadline_warning",
                SymptomSeverity.MEDIUM,
                summary="only 25.0min remain (validated_gain present)",
                evidence={"remaining_minutes": 25.0},
                subject={},
            )
        ],
        tick_index=200,
        now_unix=1700001000.0,
    )
    types = [i.type for i in out.intents]
    # MEDIUM falls through _diagnose: alert only, no delegate.
    assert IntentType.ALERT in types
    assert IntentType.DELEGATE not in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE not in types


async def test_deadline_hard_cutoff_emits_emergency_delegate():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "deadline_hard_cutoff",
                SymptomSeverity.HIGH,
                summary="only 4.0min remain (<= 5min); finalize now",
                evidence={
                    "remaining_minutes": 4.0,
                    "deadline_hard_cutoff_minutes": 5.0,
                },
                subject={},
            )
        ],
        tick_index=355,
        now_unix=1700001500.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.DELEGATE in types
    delegate = next(i for i in out.intents if i.type is IntentType.DELEGATE)
    assert delegate.payload["action_name"] == "report"
    assert delegate.payload["params"]["reason"] == "deadline_hard_cutoff"
    assert delegate.payload["idempotency_key"] == (
        "report-deadline-hard-cutoff-tick-355"
    )


async def test_budget_strategy_drift_falls_to_medium_diagnose():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "budget_strategy_drift",
                SymptomSeverity.MEDIUM,
                summary="50% burnt, no validated gain",
                evidence={"burn_pct": 0.5, "cumulative_gain_validated": 0.0},
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    # MEDIUM-only: alert, no destructive action.
    assert IntentType.ALERT in types
    assert IntentType.DELEGATE not in types
    assert IntentType.PRUNE_BRANCH not in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE not in types


async def test_deadline_imminent_emits_escalate_plus_delegate_report():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "deadline_imminent",
                SymptomSeverity.HIGH,
                summary=(
                    "wall-clock budget 92% consumed with "
                    "cumulative_gain_validated=0.00%"
                ),
                evidence={
                    "elapsed_minutes": 330.0,
                    "remaining_minutes": 30.0,
                    "budget_minutes": 360.0,
                    "burn_pct": 0.917,
                    "cumulative_gain_validated": 0.0,
                    "imminent_pct": 0.85,
                    "productive_gain_pct": 0.5,
                },
                subject={},
                suggestion="delegate(report) to finalize",
            )
        ],
        tick_index=12,
        now_unix=1700000600.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ALERT in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.DELEGATE in types
    delegate = next(i for i in out.intents if i.type is IntentType.DELEGATE)
    assert delegate.payload["action_name"] == "report"
    assert delegate.payload["params"]["reason"] == "deadline_imminent"
    assert delegate.payload["idempotency_key"] == (
        "report-deadline-imminent-tick-12"
    )


async def test_same_payload_loop_prunes_branch_and_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "same_payload_loop",
                SymptomSeverity.HIGH,
                summary="validate_stack repeated 3x",
                evidence={
                    "family": "validate_stack",
                    "count": 3,
                    "last_task_id": "t-3",
                    "last_error_class": "RuntimeError",
                },
                subject={"family": "validate_stack"},
            )
        ],
        tick_index=5,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "validate_stack"
    assert prune.payload["reason"] == "same_payload_loop"


async def test_ray_head_dead_prunes_kernel_opt_and_escalates():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "ray_head_dead",
                SymptomSeverity.HIGH,
                summary="ray status exit=1",
                evidence={"reason": "ray status exit=1"},
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "kernel_opt"


async def test_disk_pressure_high_prunes_profile():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "disk_pressure",
                SymptomSeverity.HIGH,
                summary="/ at 97% used",
                evidence={"mountpoint": "/", "used_pct": 97.0},
                subject={"mountpoint": "/"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH in types
    prune = next(i for i in out.intents if i.type is IntentType.PRUNE_BRANCH)
    assert prune.payload["family"] == "profile"


async def test_shm_pressure_high_escalates_but_no_prune():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "shm_pressure",
                SymptomSeverity.HIGH,
                summary="/dev/shm at 96%",
                evidence={"mountpoint": "/dev/shm", "used_pct": 96.0},
                subject={"mountpoint": "/dev/shm"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    assert IntentType.PRUNE_BRANCH not in types


async def test_no_levers_found_delegates_report():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "no_levers_found",
                SymptomSeverity.HIGH,
                evidence={"elapsed_minutes": 70.0, "tick": 20},
                subject={},
            )
        ],
        tick_index=20,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.DELEGATE in types
    delegate = next(i for i in out.intents if i.type is IntentType.DELEGATE)
    assert delegate.payload["action_name"] == "report"
    assert delegate.payload["params"]["reason"] == "no_levers_found"
    assert delegate.payload["idempotency_key"] == "report-no-levers-tick-20"


async def test_gain_plateau_high_escalates_to_report():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "gain_plateau",
                SymptomSeverity.HIGH,
                evidence={"history": [0.0, 0.1, 0.0, 0.0]},
                subject={},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.ESCALATE_STRATEGY_CHANGE in types
    esc = next(i for i in out.intents if i.type is IntentType.ESCALATE_STRATEGY_CHANGE)
    assert "no levers left" in esc.payload["next_action_hint"]


async def test_idempotency_replay_falls_to_medium_diagnose_tier():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "idempotency_replay",
                SymptomSeverity.MEDIUM,
                evidence={"action_name": "validate_stack", "distinct_keys": ["a", "b"]},
                subject={"action_name": "validate_stack"},
            )
        ],
        tick_index=0,
        now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    # Medium severity emits only an alert; no destructive action.
    assert IntentType.ALERT in types
    assert IntentType.PRUNE_BRANCH not in types
    assert IntentType.DELEGATE not in types


async def test_wind_down_idempotency_key_varies_per_tick():
    """Same symptom across consecutive ticks → distinct idempotency keys.

    The cooldown blocks intra-window re-emission, but post-cooldown
    re-fires (e.g. recover still failing 10 ticks later) MUST carry a
    different idempotency_key so PolicyGate accepts the second
    ``delegate(report)`` instead of de-duping it.
    """
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=1))
    sym = _sym(
        "recover_unsuccessful",
        SymptomSeverity.HIGH,
        subject={},
        evidence={"error_class": "gpu_unhealthy_after_gpureset"},
    )
    first = await ladder.decide([sym], tick_index=3, now_unix=1.0)
    second = await ladder.decide([sym], tick_index=20, now_unix=2.0)
    first_keys = [
        i.payload.get("idempotency_key")
        for i in first.intents
        if i.type is IntentType.DELEGATE
    ]
    second_keys = [
        i.payload.get("idempotency_key")
        for i in second.intents
        if i.type is IntentType.DELEGATE
    ]
    assert first_keys == ["report-recover-unsuccessful-tick-3"]
    assert second_keys == ["report-recover-unsuccessful-tick-20"]


# ---------------------------------------------------------------------------
# Cooldown / heartbeat
# ---------------------------------------------------------------------------

async def test_no_symptoms_falls_back_to_heartbeat():
    ladder = ActionLadder()
    out = await ladder.decide([], tick_index=0, now_unix=1.0)
    assert len(out.intents) == 1
    assert out.intents[0].type is IntentType.SEND_MESSAGE
    assert out.intents[0].payload["topic"] == "heartbeat"
    assert out.findings == []


async def test_cooldown_suppresses_duplicate_within_window():
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=3))
    sym = _sym("crash_count_rising", SymptomSeverity.MEDIUM, subject={"k": "1"})
    first = await ladder.decide([sym], tick_index=0, now_unix=1.0)
    suppressed = await ladder.decide([sym], tick_index=1, now_unix=2.0)
    assert any(i.type is IntentType.ALERT for i in first.intents)
    # Suppressed tick: no symptom-derived intent emitted, only heartbeat.
    assert all(
        i.type is not IntentType.ALERT for i in suppressed.intents
    )
    assert any(i.payload.get("topic") == "heartbeat" for i in suppressed.intents)
    assert suppressed.findings == []


async def test_cooldown_releases_after_window():
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=3))
    sym = _sym("crash_count_rising", SymptomSeverity.MEDIUM, subject={"k": "1"})
    await ladder.decide([sym], tick_index=0, now_unix=1.0)
    out = await ladder.decide([sym], tick_index=3, now_unix=4.0)
    assert any(i.type is IntentType.ALERT for i in out.intents)


async def test_finding_carries_serialised_intents_and_evidence():
    ladder = ActionLadder()
    out = await ladder.decide(
        [
            _sym(
                "crash_count_high",
                SymptomSeverity.HIGH,
                summary="crash_count=5",
                evidence={"crash_count": 5},
                suggestion="revert",
            )
        ],
        tick_index=2,
        now_unix=10.0,
    )
    assert out.findings
    finding = out.findings[0]
    assert finding.symptom_name == "crash_count_high"
    assert finding.tick_index == 2
    assert finding.timestamp_unix == 10.0
    assert any(i["intent_type"] == "escalate_strategy_change" for i in finding.intents)
    assert finding.evidence == {"crash_count": 5}


async def test_rca_provider_is_invoked_when_supplied():
    ladder = ActionLadder()

    class StubRca:
        def summarize(self, sym: Symptom) -> str:
            return f"rca({sym.name})"

    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
        rca_provider=StubRca(),
    )
    assert out.findings[0].rca_text == "rca(crash_count_rising)"


async def test_rca_provider_failure_does_not_break_ladder():
    ladder = ActionLadder()

    class BadRca:
        def summarize(self, sym):
            raise RuntimeError("oops")

    out = await ladder.decide(
        [_sym("crash_count_rising", SymptomSeverity.MEDIUM)],
        tick_index=0,
        now_unix=1.0,
        rca_provider=BadRca(),
    )
    assert out.findings[0].rca_text == ""
    assert any(i.type is IntentType.ALERT for i in out.intents)


# ---------------------------------------------------------------------------
# gpu_memory_leaked — Change B contract
# ---------------------------------------------------------------------------

def _gpu_leak_symptom(*, summary: str = "all 4 GPUs full, no owner") -> Symptom:
    return Symptom(
        name="gpu_memory_leaked",
        severity=SymptomSeverity.HIGH,
        summary=summary,
        evidence={
            "consecutive_hits": 2,
            "gpu_count": 4,
            "per_gpu": [{"gpu_id": i, "free_mb": 108.0} for i in range(4)],
            "owner_patterns": ["EngineCore", "Magpie"],
        },
        subject={},  # session-wide
        source="local",
        suggestion="delegate(recover, params={force_gpu_cleanup: true})",
    )


async def test_gpu_memory_leaked_emits_alert_escalate_and_delegate():
    ladder = ActionLadder()
    out = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=7, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert types == [
        IntentType.ALERT,
        IntentType.ESCALATE_STRATEGY_CHANGE,
        IntentType.DELEGATE,
    ]
    alert = out.intents[0]
    assert alert.payload["severity"] == "high"
    assert "gpu_memory_leaked" in alert.payload["summary"] or alert.payload.get("detail", {}).get("symptom") == "gpu_memory_leaked"

    escalate = out.intents[1]
    assert escalate.payload["reason"] == "gpu_memory_leaked"
    assert escalate.payload["severity"] == "high"
    assert "recover" in escalate.payload["next_action_hint"]
    assert "report" in escalate.payload["next_action_hint"]

    delegate = out.intents[2]
    assert delegate.payload["action_name"] == "recover"
    assert delegate.payload["params"]["force_gpu_cleanup"] is True
    assert delegate.payload["params"]["reason"] == "gpu_memory_leaked"
    assert delegate.payload["params"]["evidence"]["consecutive_hits"] == 2
    # tick-indexed idempotency_key per design.
    assert delegate.payload["idempotency_key"] == "recover-gpu-leak-tick-7"

    # The Finding mirrors all three intent envelopes for the audit log.
    assert out.findings and out.findings[0].symptom_name == "gpu_memory_leaked"
    finding_types = [i["intent_type"] for i in out.findings[0].intents]
    assert "alert" in finding_types
    assert "escalate_strategy_change" in finding_types
    assert "delegate" in finding_types


async def test_gpu_memory_leaked_does_not_emit_prune_branch():
    """Per design decision (no_prune_only_escalate)."""
    ladder = ActionLadder()
    out = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=0, now_unix=1.0,
    )
    types = [i.type for i in out.intents]
    assert IntentType.PRUNE_BRANCH not in types


async def test_gpu_memory_leaked_cooldown_dedups_within_window():
    """Same dedup_key suppressed within ``cooldown_ticks``."""
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=5))
    first = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=0, now_unix=1.0,
    )
    second = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=1, now_unix=2.0,
    )
    # First tick fires the full trio.
    first_types = [i.type for i in first.intents]
    assert IntentType.DELEGATE in first_types
    # Second tick is in the cooldown window: only heartbeat falls through.
    second_types = [i.type for i in second.intents]
    assert IntentType.DELEGATE not in second_types
    assert IntentType.ALERT not in second_types
    assert IntentType.ESCALATE_STRATEGY_CHANGE not in second_types
    assert any(
        i.payload.get("topic") == "heartbeat" for i in second.intents
    )


async def test_gpu_memory_leaked_idempotency_key_advances_with_tick():
    """After cooldown elapses, the next emit carries a fresh tick-indexed key."""
    ladder = ActionLadder(config=ActionLadderConfig(cooldown_ticks=3))
    first = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=0, now_unix=1.0,
    )
    second = await ladder.decide(
        [_gpu_leak_symptom()], tick_index=5, now_unix=2.0,
    )
    first_delegate = next(i for i in first.intents if i.type is IntentType.DELEGATE)
    second_delegate = next(i for i in second.intents if i.type is IntentType.DELEGATE)
    assert first_delegate.payload["idempotency_key"] == "recover-gpu-leak-tick-0"
    assert second_delegate.payload["idempotency_key"] == "recover-gpu-leak-tick-5"
