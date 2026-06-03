"""Translate Symptoms into Coordinator Intents.

The ladder has three tiers, matching:

1. **observe** — low severity: emit ``send_message(topic="observation")``
   so the orchestration agent has visibility but no pause is triggered.
2. **diagnose** — medium severity: emit ``alert(severity="medium")``
   carrying the symptom evidence; the orchestration agent runs a
   focused RCA next tick.
3. **recommend** — high severity: emit ``alert(severity="high")`` plus
   one of the scheduling-police intents (``escalate_strategy_change``,
   ``prune_branch``, ``kill_task``) when the symptom comes with a
   concrete suggestion.

To avoid flooding the inbox the ladder maintains a per-key cooldown:
the same ``Symptom.dedup_key()`` will not produce another intent until
``cooldown_ticks`` ticks have elapsed.

Findings — one persistent record per intent batch — are emitted
alongside the intents and consumed by :class:`FindingSink` (T9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..state_store import DetectorStateView
from ..role.envelope import (
    Intent,
    build_alert,
    build_delegate,
    build_escalate,
    build_heartbeat,
    build_kill_task,
    build_prune_branch,
    build_send_message,
)
from ..signals import Symptom, SymptomSeverity


log = logging.getLogger(__name__)


@dataclass
class Finding:
    """Persistent record describing one ladder firing.

    Stored on disk by the FindingSink for later inspection / reporting
    to robustness-server.
    """

    tick_index: int
    timestamp_unix: float
    symptom_name: str
    severity: str
    summary: str
    intents: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    rca_text: str = ""


@dataclass
class ActionLadderConfig:
    """Tunables for the ladder."""

    cooldown_ticks: int = 5


@dataclass
class _LadderResult:
    """Bundle of intents and findings produced by one ``decide`` call.

    Attributes:
        intents (list[Intent]): Intents to emit this tick.
        findings (list[Finding]): Persistent records describing the firings.
    """

    intents: list[Intent]
    findings: list[Finding]


class ActionLadder:
    """Stateful ladder that maps symptoms onto intents and findings.

    The ladder is deliberately conservative in M1: it emits intents
    only for symptoms whose dedup key is outside the cooldown window
    and falls back to a heartbeat when the symptom set is empty.
    """

    def __init__(
        self,
        *,
        config: ActionLadderConfig | None = None,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the ladder and load persisted cooldown bookkeeping.

        Args:
            config (ActionLadderConfig | None): Ladder tunables; a default
                config is used when ``None``.
            state_view (DetectorStateView | None): Optional disk-backed store
                used to persist per-key cooldown ticks across restarts.
        """
        self._config = config or ActionLadderConfig()
        self._state_view = state_view
        # Cooldown bookkeeping — persisted across subprocess restarts.
        # Without persistence the ladder re-emits the same intent every
        # tick because the in-memory dict resets, defeating the
        # cooldown contract advertised in :class:`ActionLadderConfig`.
        loaded = state_view.load() if state_view is not None else {}
        self._last_emitted_tick: dict[tuple[str, ...], int] = (
            _decode_last_emitted(loaded.get("last_emitted"))
        )
        # Updated at the top of each :meth:`decide` call so per-symptom
        # branches (notably ``gpu_memory_leaked``) can stamp a stable
        # tick-indexed ``idempotency_key`` onto the intents they emit
        # without threading the tick through every helper.
        self._last_tick_index: int = 0

    def _persist_cooldown(self) -> None:
        """Write the per-key cooldown ticks to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({
            "last_emitted": _encode_last_emitted(self._last_emitted_tick),
        })

    async def decide(
        self,
        symptoms: list[Symptom],
        *,
        tick_index: int,
        now_unix: float,
        rca_provider: Any | None = None,
    ) -> _LadderResult:
        """Produce intents (+ findings) for this tick.

        ``rca_provider`` may be ``None`` (no RCA), or any object exposing
        ``async def summarize(symptom) -> str``. When the provider has a
        ``set_tick(int)`` hook (e.g. :class:`LlmRcaEngine`) we call it
        once per tick so per-tick budgets reset deterministically.

        Args:
            symptoms (list[Symptom]): Symptoms detected this tick.
            tick_index (int): Monotonic index of the current tick.
            now_unix (float): Current wall-clock time in Unix seconds.
            rca_provider (Any | None): Optional RCA engine used to attach
                ``rca_text`` to findings.

        Returns:
            _LadderResult: The intents to emit and the findings to persist; a
            lone heartbeat intent when nothing else fired.
        """
        intents: list[Intent] = []
        findings: list[Finding] = []
        any_emit = False
        self._last_tick_index = tick_index
        if rca_provider is not None:
            set_tick = getattr(rca_provider, "set_tick", None)
            if callable(set_tick):
                try:
                    set_tick(tick_index)
                except Exception:
                    log.exception("rca_provider.set_tick failed; ignoring")
        for sym in symptoms:
            key = sym.dedup_key()
            if not self._cooldown_elapsed(key, tick_index):
                continue
            sym_intents = self._intents_for(sym)
            if not sym_intents:
                continue
            any_emit = True
            self._last_emitted_tick[key] = tick_index
            self._persist_cooldown()
            rca_text = await _safe_rca(rca_provider, sym)
            findings.append(
                _build_finding(
                    sym,
                    sym_intents,
                    tick_index=tick_index,
                    now_unix=now_unix,
                    rca_text=rca_text,
                )
            )
            intents.extend(sym_intents)

        if not any_emit:
            intents.append(build_heartbeat())
        return _LadderResult(intents=intents, findings=findings)

    def _cooldown_elapsed(self, key: tuple[str, ...], tick_index: int) -> bool:
        """Report whether a dedup key is outside its cooldown window.

        Args:
            key (tuple[str, ...]): The symptom dedup key.
            tick_index (int): The current tick index.

        Returns:
            bool: ``True`` if the key has never fired or enough ticks have
            elapsed since it last did.
        """
        cooldown = self._config.cooldown_ticks
        last = self._last_emitted_tick.get(key)
        if last is None:
            return True
        return (tick_index - last) >= cooldown

    def _intents_for(self, sym: Symptom) -> list[Intent]:
        """Dispatch a symptom to the ladder tier matching its severity.

        Args:
            sym (Symptom): The symptom to translate.

        Returns:
            list[Intent]: The intents for the observe/diagnose/recommend tier.
        """
        if sym.severity is SymptomSeverity.LOW:
            return self._observe(sym)
        if sym.severity is SymptomSeverity.MEDIUM:
            return self._diagnose(sym)
        return self._recommend(sym)

    def _observe(self, sym: Symptom) -> list[Intent]:
        """Build the low-severity observation intent for a symptom.

        Args:
            sym (Symptom): The low-severity symptom.

        Returns:
            list[Intent]: A single ``send_message`` observation intent.
        """
        return [
            build_send_message(
                "observation",
                body_md=f"{sym.name}: {sym.summary}",
                extras={"detail": _detail(sym)},
            )
        ]

    def _diagnose(self, sym: Symptom) -> list[Intent]:
        """Build the medium-severity diagnostic intent for a symptom.

        Args:
            sym (Symptom): The medium-severity symptom.

        Returns:
            list[Intent]: A single medium-severity ``alert`` intent.
        """
        return [build_alert("medium", sym.summary, detail=_detail(sym))]

    def _recommend(self, sym: Symptom) -> list[Intent]:
        """Build high-severity intents, adding policing intents per symptom.

        Always emits a high-severity alert; depending on ``sym.name`` it may
        append escalate / prune_branch / kill_task / delegate intents that
        encode the concrete remediation for that symptom.

        Args:
            sym (Symptom): The high-severity symptom.

        Returns:
            list[Intent]: The alert plus any symptom-specific policing intents.
        """
        intents: list[Intent] = [build_alert("high", sym.summary, detail=_detail(sym))]
        if sym.name in {"crash_count_emergency", "crash_count_high"}:
            intents.append(
                build_escalate(
                    reason=sym.name,
                    next_action_hint=sym.suggestion or "revert to known-good baseline",
                    severity="high",
                )
            )
        elif sym.name == "agent_stall" and sym.severity is SymptomSeverity.HIGH:
            intents.append(
                build_escalate(
                    reason="agent_stall_high",
                    next_action_hint=sym.suggestion or "force_dispatch head queued task",
                    severity="high",
                )
            )
        elif sym.name == "cluster_fault":
            # Wide-blast-radius cluster faults need an escalate so the
            # orchestration agent reroutes work away from the affected
            # node before the fault sweeps more sessions.
            intents.append(
                build_escalate(
                    reason="cluster_fault_high",
                    next_action_hint=(
                        sym.suggestion
                        or "drain affected node; reschedule away from fault"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "repeated_failure":
            family = sym.evidence.get("family") if isinstance(sym.evidence, dict) else None
            if isinstance(family, str) and family.strip():
                intents.append(build_prune_branch(family=family, reason="repeated_failure"))
        elif sym.name == "same_payload_loop":
            # B1: same-fingerprint retries — directly the smoking gun
            # the 2026-05-18 GPU-leak run hit. Family is always in the
            # evidence; ``prune_branch`` stops new tasks of the same
            # family from queuing while the operator (or Orchestration)
            # changes the payload content.
            family = sym.evidence.get("family") if isinstance(sym.evidence, dict) else None
            if isinstance(family, str) and family.strip():
                intents.append(
                    build_prune_branch(family=family, reason="same_payload_loop")
                )
            intents.append(
                build_escalate(
                    reason="same_payload_loop",
                    next_action_hint=(
                        f"change params content of {family!r} (not just "
                        f"idempotency_key) or propose `report` to wind "
                        f"down — Coordinator dedup is being bypassed by "
                        f"key churn"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "ray_head_dead":
            # A6: Ray dead → kernel_opt cannot dispatch. Prune the
            # branch to stop queuing; escalate so Orchestration can
            # reroute the budget to params/sweep until ray comes back.
            intents.append(
                build_prune_branch(family="kernel_opt", reason="ray_head_dead")
            )
            intents.append(
                build_escalate(
                    reason="ray_head_dead",
                    next_action_hint=(
                        "restart ray head out-of-band; until then route "
                        "budget to params/sweep"
                    ),
                    severity="high",
                )
            )
        elif sym.name in {"disk_pressure", "shm_pressure"}:
            # A4 / A3: capacity pressure. ``disk_pressure`` HIGH prunes
            # profile (single biggest trace consumer); ``shm_pressure``
            # HIGH escalates so the operator restarts the pod with a
            # larger --shm-size before the next server boot.
            if sym.name == "disk_pressure":
                intents.append(
                    build_prune_branch(family="profile", reason="disk_pressure")
                )
                intents.append(
                    build_escalate(
                        reason="disk_pressure",
                        next_action_hint=(
                            "archive older runs/* under $USER_DATA_PATH; "
                            "the next state.json write may partial-fail"
                        ),
                        severity="high",
                    )
                )
            else:  # shm_pressure
                intents.append(
                    build_escalate(
                        reason="shm_pressure",
                        next_action_hint=(
                            "lower TP or restart pod with larger "
                            "--shm-size; SGLang/vLLM SHM exhaustion has "
                            "no graceful degrade"
                        ),
                        severity="high",
                    )
                )
        elif sym.name == "fd_pressure":
            intents.append(
                build_escalate(
                    reason="fd_pressure",
                    next_action_hint=(
                        "Coordinator has leaked file descriptors; resume "
                        "from session_dir to clear them"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "aiter_jit_regressed":
            intents.append(
                build_escalate(
                    reason="aiter_jit_regressed",
                    next_action_hint=(
                        "do NOT relaunch baseline; either skip baseline "
                        "or bump INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC "
                        "above 5400 before the next attempt"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "gain_plateau":
            # B2: validated gain flat across a window — search exhausted.
            # When validated gain > 0 we just nudge towards report;
            # when it is 0 we escalate harder so Orchestration ends
            # the run instead of churning the deadline.
            current_gain = (
                sym.evidence.get("history", [0.0])[-1]
                if isinstance(sym.evidence, dict)
                else 0.0
            )
            hint = (
                "propose `report` to lock in the validated gain"
                if current_gain > 0
                else "propose `report` and end the run — no levers left"
            )
            intents.append(
                build_escalate(
                    reason="gain_plateau",
                    next_action_hint=hint,
                    severity="high",
                )
            )
        elif sym.name == "state_json_corrupt":
            # I1: state.json failed to load. Can't auto-recover; surface
            # HIGH so the operator stops the run before the next resume
            # silently drops baseline / current_best.
            intents.append(
                build_escalate(
                    reason="state_json_corrupt",
                    next_action_hint=(
                        "back up the broken state.json and stop the run; "
                        "Coordinator atomic-writes are supposed to prevent "
                        "this — investigate the underlying filesystem"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "coordinator_wal_bloat":
            # I2: WAL > critical threshold. ``_recommend`` only runs for
            # HIGH severity, so MEDIUM ``coordinator_wal_bloat`` falls
            # through to ``_diagnose`` (alert only). The HIGH branch
            # adds the escalate hint.
            intents.append(
                build_escalate(
                    reason="coordinator_wal_bloat",
                    next_action_hint=(
                        "PRAGMA wal_checkpoint(TRUNCATE) is overdue; "
                        "SQLite read/write latency will degrade until "
                        "Coordinator rolls the WAL forward"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "stale_lease":
            # I3: lease held by dead PID. Robustness owns ``kill_task``
            # with scope='task'; emit one per stale lease so the
            # Coordinator's task registry releases the lane.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            task_id = str(evidence.get("task_id") or "").strip()
            if task_id and task_id != "unknown":
                intents.append(
                    build_kill_task(
                        task_id=task_id,
                        reason="stale_lease",
                    )
                )
            intents.append(
                build_escalate(
                    reason="stale_lease",
                    next_action_hint=(
                        f"kill_task dispatched for task_id={task_id!r}; "
                        f"lane should free up for the next pending proposal"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "coordinator_zombie":
            # I5: Coordinator PID dead but state.json says running.
            # Worst case Robustness can see; cannot self-heal (we live
            # inside the Coordinator process tree). Emit HIGH escalate
            # so the run-launcher / external monitor catches it.
            intents.append(
                build_escalate(
                    reason="coordinator_zombie",
                    next_action_hint=(
                        "operator restart required: the Coordinator "
                        "process is dead but state.json carries no "
                        "stop_reason; check optimizer_runs/run_*.log for "
                        "the crash and resume from session_dir"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "gateway_auth_outage":
            # J1: upstream LLM gateway 401/403.
            intents.append(
                build_escalate(
                    reason="gateway_auth_outage",
                    next_action_hint=(
                        "rotate $SAFE_API_KEY at https://llm.amd.com/ "
                        "and re-export; the upstream gateway key is revoked"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "wekafs_degraded":
            # J2: WekaFS mount slow or dropped. No prune because the
            # operator may decide to wait it out OR restart the mount;
            # escalate gives the action-able hint.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="wekafs_degraded",
                    next_action_hint=(
                        f"WekaFS mount degraded ({evidence.get('env_name')}); "
                        f"trace_analyze / OOB CLI / benchmark scripts "
                        f"will hang or time out until the mount recovers"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "tracelens_cli_missing":
            # J3: boot-time only.
            intents.append(
                build_escalate(
                    reason="tracelens_cli_missing",
                    next_action_hint=(
                        "re-run $REPO_ROOT/inference_optimizer/scripts/install.sh; "
                        "TraceLens editable install is idempotent and "
                        "will restore both perf-report CLI names"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "critic_kb_outage":
            # E1: KB unreachable across N+ recent turns.
            intents.append(
                build_escalate(
                    reason="critic_kb_outage",
                    next_action_hint=(
                        "switch to CRITIC_KB_CLIENT_MODE=inmemory OR "
                        "--critic-mock until KB is restored; verdicts "
                        "are landing without prior recall"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "critic_unavailable_streak":
            # E2: critic verdicts all carry source='critic_unavailable'.
            intents.append(
                build_escalate(
                    reason="critic_unavailable_streak",
                    next_action_hint=(
                        "switch to --critic-mock; inspect "
                        "critic-workdir/<latest>/judge_bundle.json for "
                        "the missing required_context and add it to the "
                        "manifest"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "critic_runtime_stuck":
            # E5: runtime.cli timed out repeatedly.
            intents.append(
                build_escalate(
                    reason="critic_runtime_stuck",
                    next_action_hint=(
                        "switch to --critic-mock; the codex chat-completion "
                        "endpoint or the critic-agent subprocess is hung"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "ray_pending_starvation":
            # F1: cluster quota wedged — prune kernel_opt + escalate.
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="ray_pending_starvation",
                )
            )
            intents.append(
                build_escalate(
                    reason="ray_pending_starvation",
                    next_action_hint=(
                        "shrink concurrency or wait out cluster quota; "
                        "route budget to params/sweep until ray clears"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "geak_budget_starvation":
            # F2: GEAK SIGTERM repeatedly on same kernel — fire escalate
            # but no prune (operator may extend budget).
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="geak_budget_starvation",
                    next_action_hint=(
                        f"GEAK can't finish select_patch within current "
                        f"budget on kernel_id={evidence.get('kernel_id', '?')}; "
                        f"extend --geak-budget-min above 90 OR prune this "
                        f"kernel from rotation"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "kernel_opt_no_progress":
            # F5: every backend attempt lands PARTIAL/REVERT.
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="kernel_opt_no_progress",
                )
            )
            intents.append(
                build_escalate(
                    reason="kernel_opt_no_progress",
                    next_action_hint=(
                        "kernel pipeline is structurally unable to "
                        "optimise this model; budget will land further "
                        "wins on params/sweep instead"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "model_gpu_infeasible":
            # C1: the manifest configuration cannot possibly fit in HBM.
            # Robustness cannot rescue this — only an operator increasing
            # TP or moving to a larger-HBM GPU can. Prune every action
            # family that would try to launch a server so we stop
            # wasting the time budget on guaranteed-OOM baselines.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            for family in ("baseline", "backends", "params", "sweep",
                           "validate_stack", "kernel_opt"):
                intents.append(
                    build_prune_branch(
                        family=family, reason="model_gpu_infeasible"
                    )
                )
            intents.append(
                build_escalate(
                    reason="model_gpu_infeasible",
                    next_action_hint=(
                        f"abort: model={evidence.get('model_name')!r} on "
                        f"{evidence.get('gpu_type')} tp={evidence.get('tp')} "
                        f"needs {evidence.get('required_gib')} GiB but device "
                        f"has only {evidence.get('hbm_gib')} GiB. Operator "
                        f"must increase TP or move to a larger-HBM GPU; "
                        f"Robustness cannot save this run."
                    ),
                    severity="high",
                )
            )
        elif sym.name == "amdahl_kernel_ceiling_low":
            # C2: profile shows kernel_opt cannot move the E2E needle.
            # Prune kernel_opt and let the budget flow to params/sweep.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="amdahl_kernel_ceiling_low"
                )
            )
            intents.append(
                build_escalate(
                    reason="amdahl_kernel_ceiling_low",
                    next_action_hint=(
                        f"kernel_opt is pointless: only "
                        f"{evidence.get('optimizable_pct', '?')}% of GPU "
                        f"time is Triton-optimizable. Allocate budget to "
                        f"params/sweep where the ceiling is higher."
                    ),
                    severity="high",
                )
            )
        elif sym.name == "cold_start_budget_exhausted":
            # C3: aiter cold + remaining < cold_start cycle → next
            # baseline will SIGTERM. Escalate so operator extends timeout
            # or skips baseline; no prune because the user might still
            # want to retry with extended budget.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="cold_start_budget_exhausted",
                    next_action_hint=(
                        f"do NOT relaunch baseline: aiter cache cold + "
                        f"only {evidence.get('remaining_minutes', '?')}min "
                        f"remain. Skip baseline (reuse current_best) or "
                        f"bump INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC."
                    ),
                    severity="high",
                )
            )
        elif sym.name == "empty_patch_kept":
            # G1: a KEEP decision with patch_size_bytes=0 is a noise-
            # floor false positive. Prune the kernel_opt family so the
            # bad pattern is taken out of rotation, plus escalate so
            # Orchestration knows to revert / KB-rule the kernel_id.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="empty_patch_kept"
                )
            )
            intents.append(
                build_escalate(
                    reason="empty_patch_kept",
                    next_action_hint=(
                        f"REVERT kernel_id={evidence.get('kernel_id', '?')} "
                        "and add a KB rule: empty patches MUST NOT be "
                        "accepted regardless of noise-floor gain"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "kernel_dispatch_bypassed":
            # G3: KEEP'd patch likely never executed. Same prune as G1
            # (kernel_opt family is the source) plus escalate hint
            # asking for dispatch evidence on subsequent attempts.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="kernel_dispatch_bypassed"
                )
            )
            intents.append(
                build_escalate(
                    reason="kernel_dispatch_bypassed",
                    next_action_hint=(
                        f"require integrate to attach ROCprof / TraceLens "
                        f"dispatch evidence (dispatched_count > 0) before "
                        f"KEEP; revert kernel_id="
                        f"{evidence.get('kernel_id', '?')}"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "kernel_negative_delta_kept":
            # G4: aggregate ci_metrics shows kernels_optimized > 0 but
            # delta is net-negative. Escalate so Orchestration can roll
            # back the kernel changes and ship without the negative
            # contribution.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="kernel_negative_delta_kept",
                    next_action_hint=(
                        f"roll back kernel changes; "
                        f"optimized_kernel_delta_pct="
                        f"{evidence.get('optimized_kernel_delta_pct', '?')} "
                        f"is net-negative"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "ci_metrics_baseline_zero":
            # G5: partial ci_metrics write looks like a 0-throughput
            # baseline. Escalate so the operator deletes the bad file
            # and re-runs report_back.
            intents.append(
                build_escalate(
                    reason="ci_metrics_baseline_zero",
                    next_action_hint=(
                        "delete the partial ci_metrics file; require "
                        "report_back to write {status: 'baseline_failed'} "
                        "on baseline failure instead of zero rows"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "oob_no_harness":
            # G7: OOB attempt with expected-only speedup, no measurement.
            # Prune the kernel_opt family so the no-harness pattern is
            # taken out of rotation and escalate for a process fix.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_prune_branch(
                    family="kernel_opt", reason="oob_no_harness"
                )
            )
            intents.append(
                build_escalate(
                    reason="oob_no_harness",
                    next_action_hint=(
                        f"OOB attempt on kernel_id="
                        f"{evidence.get('kernel_id', '?')} advertised "
                        f"expected speedup with no microbench result; "
                        f"reject and mark NO_HARNESS"
                    ),
                    severity="high",
                )
            )
        elif sym.name == "deadline_warning":
            # H1 absolute-time warning. ``_deadline_warning_symptom``
            # already picks HIGH (no validated gain) vs MEDIUM (gain
            # exists). The MEDIUM branch never reaches ``_recommend``;
            # only the HIGH branch lands here and gets a ``delegate(report)``
            # backstop — same wind-down contract as ``deadline_imminent``
            # but triggered by the absolute-time axis instead of burn_pct.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="deadline_warning",
                    next_action_hint=(
                        "delegate(report) is being dispatched; validated "
                        "gain is still 0 with <= deadline_warning_minutes "
                        "remaining"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="report",
                    params={
                        "reason": "deadline_warning",
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"report-deadline-warning-tick-"
                        f"{self._last_tick_index}"
                    ),
                )
            )
        elif sym.name == "deadline_hard_cutoff":
            # H1 absolute-time hard cutoff. We do NOT gate on gain —
            # by this point any new task started would be cut by the
            # deadline supervisor. Highest urgency wind-down.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="deadline_hard_cutoff",
                    next_action_hint=(
                        "delegate(report) is being dispatched; <= "
                        "deadline_hard_cutoff_minutes remain. Any new "
                        "task started now will be SIGTERM'd at the wall"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="report",
                    params={
                        "reason": "deadline_hard_cutoff",
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"report-deadline-hard-cutoff-tick-"
                        f"{self._last_tick_index}"
                    ),
                )
            )
        elif sym.name == "no_levers_found":
            # B3: long-running session, nothing on the stack. Wind down
            # via delegate(report); allowlist accepts ``report``.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="no_levers_found",
                    next_action_hint=(
                        "delegate(report) is being dispatched to surface "
                        "the attempted candidates; stop proposing new "
                        "explore rounds"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="report",
                    params={"reason": "no_levers_found", "evidence": evidence},
                    idempotency_key=(
                        f"report-no-levers-tick-{self._last_tick_index}"
                    ),
                )
            )
        elif sym.name == "gpu_memory_leaked":
            # Per design decision (no_prune_only_escalate): do NOT
            # ``prune_branch`` server-launching families — let the
            # recover sub-agent's outcome decide whether the optimizer
            # can keep going. Emit an advisory escalate + the actual
            # recovery delegate; the high-severity ``alert`` from the
            # base ``_recommend`` call already covers operator visibility.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="gpu_memory_leaked",
                    next_action_hint=(
                        "delegate(recover, params={force_gpu_cleanup: true}) "
                        "is being dispatched; if recover returns "
                        "needs_review, propose `report` to finalize at the "
                        "last validated gain"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="recover",
                    params={
                        "reason": "gpu_memory_leaked",
                        "force_gpu_cleanup": True,
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"recover-gpu-leak-tick-{self._last_tick_index}"
                    ),
                )
            )
        elif sym.name == "recover_unsuccessful":
            # Recover already ran and returned ``state=needs_review`` —
            # in-loop cleanup cannot release the leaked VRAM. The only
            # productive use of the remaining time budget is to finalize
            # at the last validated gain. ``delegate(report)`` is
            # allowed for robustness exactly for this wind-down path
            # (see ROBUSTNESS_DELEGATE_ACTIONS); Orchestration is also
            # nudged via escalate so it stops queueing new
            # validate_stack rounds while ``report`` runs.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="recover_unsuccessful",
                    next_action_hint=(
                        "delegate(report) is being dispatched to finalize "
                        "at the last validated gain; stop proposing new "
                        "explore/validate_stack rounds"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="report",
                    params={
                        "reason": "recover_unsuccessful",
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"report-recover-unsuccessful-tick-"
                        f"{self._last_tick_index}"
                    ),
                )
            )
        elif sym.name == "deadline_imminent":
            # Wall-clock budget mostly burnt with no validated gain
            # locked in. Wind the run down now so the deterministic
            # report stage actually runs inside the budget instead of
            # being cut by the deadline supervisor.
            evidence = (
                dict(sym.evidence) if isinstance(sym.evidence, dict) else {}
            )
            intents.append(
                build_escalate(
                    reason="deadline_imminent",
                    next_action_hint=(
                        "delegate(report) is being dispatched to capture "
                        "what we have before the wall-clock deadline; do "
                        "not start new explore rounds"
                    ),
                    severity="high",
                )
            )
            intents.append(
                build_delegate(
                    action_name="report",
                    params={
                        "reason": "deadline_imminent",
                        "evidence": evidence,
                    },
                    idempotency_key=(
                        f"report-deadline-imminent-tick-"
                        f"{self._last_tick_index}"
                    ),
                )
            )
        return intents


def _detail(sym: Symptom) -> dict[str, Any]:
    """Build the structured detail payload carried on alert intents.

    Args:
        sym (Symptom): The symptom whose fields are packed into the detail.

    Returns:
        dict[str, Any]: The symptom metadata and evidence, plus ``suggestion``
        when present.
    """
    body = {
        "symptom": sym.name,
        "severity": sym.severity.value,
        "subject": sym.subject,
        "source": sym.source,
        "evidence": sym.evidence,
    }
    if sym.suggestion:
        body["suggestion"] = sym.suggestion
    return body


def _build_finding(
    sym: Symptom,
    intents: Iterable[Intent],
    *,
    tick_index: int,
    now_unix: float,
    rca_text: str,
) -> Finding:
    """Assemble a persistent :class:`Finding` for one ladder firing.

    Args:
        sym (Symptom): The symptom that fired.
        intents (Iterable[Intent]): The intents emitted for the symptom.
        tick_index (int): The tick on which the firing occurred.
        now_unix (float): Wall-clock time of the firing, in Unix seconds.
        rca_text (str): Optional root-cause text to attach.

    Returns:
        Finding: The fully populated finding record.
    """
    return Finding(
        tick_index=tick_index,
        timestamp_unix=now_unix,
        symptom_name=sym.name,
        severity=sym.severity.value,
        summary=sym.summary,
        intents=[i.to_envelope_item() for i in intents],
        evidence=dict(sym.evidence) if isinstance(sym.evidence, dict) else {},
        rca_text=rca_text,
    )


async def _safe_rca(provider: Any | None, sym: Symptom) -> str:
    """Invoke an RCA provider defensively, awaiting it when needed.

    Args:
        provider (Any | None): Optional object exposing ``summarize(symptom)``
            (sync or async).
        sym (Symptom): The symptom to summarize.

    Returns:
        str: The RCA text, or an empty string when absent or on error.
    """
    if provider is None:
        return ""
    try:
        result = provider.summarize(sym)
        if hasattr(result, "__await__"):
            result = await result
        return result or ""
    except Exception:
        log.exception("rca provider raised; continuing without RCA text")
        return ""


# ---------------------------------------------------------------------------
# State-store (de)serialisation helpers
# ---------------------------------------------------------------------------

# Separator used to pack ``tuple[str, ...]`` dedup keys into a single
# JSON-safe string. A vertical-bar is unlikely to appear in symptom
# names / subject IDs and keeps the encoded key human-readable in the
# detector_state.json file (useful when debugging cooldown behaviour).
_LADDER_KEY_SEP: str = "\x1f"  # ASCII unit separator — safe inside JSON strings


def _encode_last_emitted(
    last_emitted: dict[tuple[str, ...], int],
) -> dict[str, int]:
    """Serialise a tuple-keyed cooldown dict to a JSON-safe dict.

        JSON object keys must be strings; we join the ``Symptom.dedup_key``
        tuple components with ``_LADDER_KEY_SEP`` so reads can recover the
        original tuple verbatim.

    Args:
        last_emitted (dict[tuple[str, ...], int]): Per-key last-emitted ticks.

    Returns:
        dict[str, int]: A dict with string keys safe for JSON storage.
    """
    out: dict[str, int] = {}
    for key, tick in last_emitted.items():
        try:
            encoded = _LADDER_KEY_SEP.join(str(part) for part in key)
        except Exception:  # noqa: BLE001 — best-effort, skip bad keys
            continue
        try:
            out[encoded] = int(tick)
        except (TypeError, ValueError):
            continue
    return out


def _decode_last_emitted(
    payload: Any,
) -> dict[tuple[str, ...], int]:
    """Inverse of :func:`_encode_last_emitted`; tolerant of bad input.

    Args:
        payload (Any): The persisted mapping of encoded keys to ticks.

    Returns:
        dict[tuple[str, ...], int]: The decoded tuple-keyed cooldown dict;
        empty when ``payload`` is not a dict.
    """
    if not isinstance(payload, dict):
        return {}
    out: dict[tuple[str, ...], int] = {}
    for raw_key, raw_tick in payload.items():
        if not isinstance(raw_key, str):
            continue
        try:
            tick = int(raw_tick)
        except (TypeError, ValueError):
            continue
        parts = tuple(raw_key.split(_LADDER_KEY_SEP))
        out[parts] = tick
    return out


__all__ = ["ActionLadder", "ActionLadderConfig", "Finding"]
