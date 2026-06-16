# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Result recording / fact synthesis collaborator extracted from Coordinator.

This is the MEASURE/REASON bookkeeping layer of the orchestrator. These methods
read SharedState + a finished task/result and emit derived artifacts: specialist
result records, per-task / per-variant journal facts, workload tags, recipe
attributes, the cortex recipe+journal finalization, and research-evidence
aggregation/harvest. They were moved verbatim out of the Coordinator God-object.

Design (transitional collaborator)
----------------------------------
Identical pattern to :class:`IntentRouter`: ``ResultRecorder`` holds a
back-reference to its owning ``Coordinator`` and delegates unknown attributes to
it via ``__getattr__``. The moved bodies keep using ``self.shared_state`` /
``self.tasks`` / ``self._journal_entry_phase()`` etc., which resolve back onto
the coordinator. Safe because the extracted methods do **no** ``self.<attr> = ``
rebinding (AST-verified). Calls between moved methods are routed through
``self._coord.<method>`` so they remain overridable by tests that monkeypatch
the coordinator (e.g. ``_harvest_research_scout``).

Coordinator keeps thin forwarding shims so existing tests that call
``coord._record_specialist_result(...)`` / ``coord._record_fact_per_*`` etc.
keep working unchanged.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from .optimization_journal import (
    JournalEntry,
    OUTCOME_KEEP,
    OUTCOME_NO_PROMOTE,
    OUTCOME_REVERT,
    classify_change_kind,
    summarize_change,
)
from .task_registry import Task

if TYPE_CHECKING:
    from .coordinator import Coordinator

log = __import__("logging").getLogger(__name__)


class ResultRecorder:
    """Synthesizes result records and journal facts on behalf of a Coordinator."""

    def __init__(self, coordinator: "Coordinator") -> None:
        self._coord = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_coord"), name)

    async def _record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> None:
        """Common bookkeeping for any specialist task termination (dispatcher loop + intent routing); idempotent on round_id, failures logged not raised."""
        domain = str(done_payload.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task, done_payload=done_payload, source=source,
        )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing. Defensive.
        _scorer = getattr(self, "_proposal_scorer", None)
        if _scorer is not None and proposals:
            try:
                scores = await _scorer.score(
                    gap={
                        "domain": domain,
                        "gap_canonical_id": done_payload.get(
                            "gap_canonical_id", ""
                        ),
                        "gap_symptom": (task.params or {}).get("gap_symptom"),
                        "gap_evidence": (task.params or {}).get("gap_evidence"),
                        "summary": done_payload.get("summary", ""),
                    },
                    proposals=proposals,
                )
                if scores and scores.get("models"):
                    round_entry["ensemble_scores"] = scores
            except Exception:  # noqa: BLE001 — advisory; never block
                log.exception(
                    "specialist bookkeeping: proposal scoring failed for "
                    "task=%s (continuing without scores)", task.task_id,
                )
        try:
            self.shared_state.record_specialist_round(round_entry)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: record_specialist_round failed for "
                "task=%s", task.task_id,
            )

        try:
            self.shared_state.bump_specialist_domain_empty_streak(
                domain, empty=is_empty,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: bump_specialist_domain_empty_streak "
                "failed for task=%s", task.task_id,
            )

        # Per-anchor coverage ledger (point 1): every specialist completion is
        # one "round" — tick all anchors, then zero the one that just ran so a
        # long-idle domain's counter climbs until the hard-trigger forces it.
        try:
            self.shared_state.bump_domain_round_counters()
            self.shared_state.note_specialist_dispatched(domain)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: domain round-counter update failed "
                "for task=%s", task.task_id,
            )

        try:
            self.shared_state.update_last_specialist({
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": str(
                    done_payload.get("gap_canonical_id") or ""
                ),
                "empty": is_empty,
                "proposals_total": len(proposals),
                "confidence": done_payload.get("confidence"),
                "summary": str(done_payload.get("summary") or "")[:480],
                "reason": str(done_payload.get("reason") or "")[:480],
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: update_last_specialist failed for "
                "task=%s", task.task_id,
            )

        # Persist so a resume picks up the bookkeeping without re-running the specialist.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )

        await self._record_observation(
            source or "coordinator", "observation",
            {
                "kind": "specialist_done_recorded",
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                "proposals_total": len(proposals),
                "empty": is_empty,
            },
        )

        # Multi-node only: auto-materialise the proposal_set into a
        # benchmarked explore task. No-op single-node (LLM drives explore
        # directly there) and no-op when the proposal_set is empty / has
        # no applicable variants. See :meth:`_maybe_materialize_mn_explore`.
        try:
            await self._maybe_materialize_mn_explore(
                task=task, domain=domain, proposals=proposals,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: bridge raised for task=%s (continuing)",
                task.task_id,
            )

        # route session_steward_specialist verdicts. Done payload
        # carries extra fields beyond the standard schema; see
        # ``actions/assess_remaining_gaps.md`` and the prompt builder
        # focus template. Coerce out-of-vocab recommendations to
        # ``stop_session`` (defense in depth — the LLM is allowed to
        # write any string but we only honour the closed enum).
        if domain == "session_steward_specialist":
            try:
                await self._route_steward_verdict(
                    task=task, done_payload=done_payload,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "steward routing failed for task=%s; assessment "
                    "left in last_remaining_gaps_assessment but no "
                    "phase-routing change applied",
                    task.task_id,
                )

        # Harvest research-scout output (hints, competitor target, gap seeds, PR dedup). Fail-soft.
        if domain == "research_scout_specialist":
            try:
                self._coord._harvest_research_scout(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout harvest failed for task=%s", task.task_id,
                )

        # Aggregate research evidence from any domain (e.g. pr_intel) that
        # self-reports a ``research`` block, so FRAMEWORK_PR / explore lanes
        # reuse the session-wide seen-set. Idempotent for research_scout
        # (already harvested above). Fail-soft.
        try:
            self._coord._aggregate_research_evidence(done_payload)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "research evidence aggregation failed for task=%s",
                task.task_id,
            )

        # Refresh the gaps ledger after a specialist round closes; record the verdict as a gap attempt.
        gap_cid = str(done_payload.get("gap_canonical_id") or "").strip()
        if gap_cid:
            try:
                self.shared_state.append_gap_attempt(gap_cid, {
                    "action": "specialist",
                    "variant_name": domain,
                    "outcome": "EMPTY" if is_empty else "PROPOSALS",
                    "proposals_total": len(proposals),
                })
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "specialist bookkeeping: append_gap_attempt failed for "
                    "gap=%s", gap_cid,
                )
        try:
            await self._refresh_gaps(reason="specialist_done")
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "specialist bookkeeping: _refresh_gaps failed for task=%s",
                task.task_id,
            )
        # B3: push specialist-authored patches to the Critic so integrate_patch can pass.
        try:
            await self._maybe_autosubmit_specialist_patches(
                task=task, done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: specialist patch autosubmit failed for task=%s",
                task.task_id,
            )

    def _record_intervention_for_task(
        self, task: "Task", result: Any,
    ) -> None:
        """PR-A8: log a completed task's change_type into SharedState.intervention_mix (explore → config; integrate_patch → code_patch_attempt or code_patch when kept). Best-effort."""
        if not isinstance(result, dict):
            return
        kind = (task.kind or "").strip()
        if kind == "explore":
            # Winner surrogate: result.winners present OR best_variant set.
            winners = result.get("winners") or []
            best = result.get("best_variant")
            if not winners and not best:
                # B2: an explore round that KEPT nothing still counts as a config-only attempt.
                self.shared_state.record_intervention(
                    change_type="config_attempt",
                    action="explore",
                    task_id=task.task_id,
                    delta_pct=None,
                )
                return
            delta_pct = None
            if isinstance(best, dict):
                delta_pct = best.get("gain_pct")
            self.shared_state.record_intervention(
                change_type="config",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            status = str(result.get("status") or "").strip().lower()
            if not status:
                return
            if status != "kept":
                self.shared_state.record_intervention(
                    change_type="code_patch_attempt",
                    action="integrate_patch",
                    task_id=task.task_id,
                    delta_pct=result.get("delta_pct"),
                )
                return
            self.shared_state.record_intervention(
                change_type="code_patch",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact (source_session_id is hyperloom-local)."""
        journal = self._ensure_journal()
        gain_raw = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain_raw) if gain_raw is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        tput_raw = result_dict.get("output_throughput")
        try:
            throughput_after = float(tput_raw) if tput_raw is not None else None
        except (TypeError, ValueError):
            throughput_after = None
        kind = classify_change_kind(task.kind, None)
        change = summarize_change(task.kind, None, result_dict)
        if kept:
            outcome = OUTCOME_KEEP
            error_class = None
            reason = None
        else:
            outcome = OUTCOME_REVERT
            error_class = (str(result_dict.get("error_class") or "") or None)
            reason = (str(result_dict.get("reason") or "") or None)
        journal.append_entry(JournalEntry(
            phase=self._journal_entry_phase(),
            iter=int(self.shared_state.tick or 0),
            kind=kind,
            change=change,
            outcome=outcome,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            error_class=error_class,
            reason=reason,
            task_id=task.task_id,
            tick=int(self.shared_state.tick or 0),
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        # evidence_refs (log:task-...) gives traceability since source_session_id lands in attrs.
        evidence_refs = [f"log:task-{task.task_id}"]
        # Workload-shape tags for lesson/pitfall attrs so the warm-start reader filters cross-framework noise.
        workload_tags = self._coord._collect_workload_tags()
        extra = workload_tags if workload_tags else None
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if kept and gain_pct is not None and gain_pct > 0:
            statement = self._coord._build_statement(
                change=change, gain_pct=gain_pct, kind="lesson",
            )
            impact = self._coord._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            # v2: append onto the recipe's lessons[] (no cross-recipe dedup).
            self._kb_amend_recipe(
                append_lesson={
                    "statement":       statement,
                    "measured_impact": impact,
                },
                provenance_details={
                    "source_session_id": source_session_id,
                    "source_task_id":    task.task_id,
                    "evidence":          list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":             dict(extra or {}),
                    "now":               now_iso,
                },
            )
            return

        severity = self._pitfall_severity_for(result_dict)
        if severity is not None:
            description = self._coord._build_statement(
                change=change, severity=severity, kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity":    severity,
                },
                provenance_details={
                    "source_session_id": source_session_id,
                    "source_task_id":    task.task_id,
                    "evidence":          list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":             dict(extra or {}),
                    "now":               now_iso,
                },
            )

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Per-variant fact write — mirror of _record_fact_per_task for explore per-variant decisions."""
        journal = self._ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw == "KEEP":
            outcome = OUTCOME_KEEP
        elif outcome_raw in ("REVERT", "FAILED", "KEEP_UNSTABLE"):
            outcome = OUTCOME_REVERT
        elif outcome_raw == "SKIPPED_DEDUP":
            return  # nothing to journal
        else:
            outcome = OUTCOME_NO_PROMOTE
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = variant_outcome.get("metrics") or {}
        gain_raw = metrics.get("gain_pct") if isinstance(metrics, dict) else None
        try:
            gain_pct = float(gain_raw) if gain_raw is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        tput_raw = metrics.get("output_throughput") if isinstance(metrics, dict) else None
        try:
            throughput_after = float(tput_raw) if tput_raw is not None else None
        except (TypeError, ValueError):
            throughput_after = None
        variant_attrs = variant_outcome.get("variant") or {}
        kind = classify_change_kind(
            task.kind, variant_attrs if isinstance(variant_attrs, dict) else None,
        )
        # Ensure the change summary is variant-specific (else every explore variant writes an identical row).
        change_attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
        if not (
            change_attrs.get("extra_sglang_args")
            or change_attrs.get("extra_envs")
            or change_attrs.get("name")
        ) and variant_name:
            change_attrs["name"] = variant_name
        change = summarize_change(task.kind, change_attrs, None)
        error_class = None
        reason = None
        if outcome == OUTCOME_REVERT:
            error_class = (str(variant_outcome.get("error_class") or "") or None)
            reason = (str(variant_outcome.get("reason") or "") or None)
        # Proposer attribution + per-variant measurement detail, carried from the
        # explore executor's per_variant_outcomes so the decision row records who
        # proposed the change and how it measured (beyond headline gain/tput).
        detail_metrics = {
            k: metrics[k]
            for k in (
                "runtime_sec", "wall_clock_ratio_vs_baseline",
                "stack_rebench_tput", "estimated_output_throughput",
            )
            if isinstance(metrics, dict) and metrics.get(k) is not None
        }
        journal.append_entry(JournalEntry(
            phase=self._journal_entry_phase(),
            iter=int(self.shared_state.tick or 0),
            kind=kind,
            change=change,
            outcome=outcome,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            error_class=error_class,
            reason=reason,
            task_id=task.task_id,
            variant_name=variant_name,
            provenance=str(variant_outcome.get("provenance") or ""),
            scope=str(variant_outcome.get("scope") or ""),
            fingerprint=str(variant_outcome.get("fingerprint") or ""),
            metrics=detail_metrics,
            tick=int(self.shared_state.tick or 0),
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        evidence_refs = [
            f"log:task-{task.task_id}",
            f"variant:{variant_name}",
        ]
        # Workload-shape tags — see _record_fact_per_task.
        workload_tags = self._coord._collect_workload_tags()
        extra = workload_tags if workload_tags else None

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if outcome == OUTCOME_KEEP and gain_pct is not None and gain_pct > 0:
            statement = self._coord._build_statement(
                change=change, gain_pct=gain_pct, kind="lesson",
            )
            impact = self._coord._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            # v2: per-variant lesson append onto recipe.lessons[]
            # (no cross-recipe dedup, see _record_fact_per_task).
            self._kb_amend_recipe(
                append_lesson={
                    "statement":       statement,
                    "measured_impact": impact,
                },
                provenance_details={
                    "source_session_id":   source_session_id,
                    "source_task_id":      task.task_id,
                    "source_variant_name": variant_name,
                    "evidence":            list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":               dict(extra or {}),
                    "now":                 now_iso,
                },
            )
            return

        severity = self._pitfall_severity_for({
            **(metrics if isinstance(metrics, dict) else {}),
            "error_class": variant_outcome.get("error_class"),
            "status":      variant_outcome.get("outcome"),
        })
        if severity is not None:
            description = self._coord._build_statement(
                change=change, severity=severity, kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity":    severity,
                },
                provenance_details={
                    "source_session_id":   source_session_id,
                    "source_task_id":      task.task_id,
                    "source_variant_name": variant_name,
                    "evidence":            list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":               dict(extra or {}),
                    "now":                 now_iso,
                },
            )

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        gain_pct: float | None = None,  # kept for backward call-signature compat
        severity: str | None = None,
    ) -> str:
        """Build the lesson statement / pitfall description hashed into the KB canonical_id; MUST exclude volatile fields (e.g. gain_pct) so N sessions merge instead of producing N rows. Identity = framework + change + model/hw."""
        framework = str(getattr(self.shared_state, "framework", "") or "").strip()
        fw_tag = f"[{framework or '?'}] "
        model = self.shared_state.model_name or "?"
        hw = self.shared_state.gpu_type or "?"
        if kind == "lesson":
            # gain_pct intentionally NOT included — see docstring.
            return f"{fw_tag}{change} on {model}/{hw}"
        # kind == "pitfall"
        return f"{fw_tag}{change} → {severity or '?'} on {model}/{hw}"

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
        throughput_before: float | None = None,
    ) -> dict[str, Any]:
        """GAP 3 — structured ``measured_impact`` payload (dict not legacy string so consumers parse without regex); stack_depth = stack length before this lesson lands."""
        out: dict[str, Any] = {
            "gain_pct": float(gain_pct) if gain_pct is not None else None,
            "stack_depth_at_apply": int(stack_depth),
            "measured_at": measured_at,
        }
        if throughput_after is not None:
            out["throughput_after"] = float(throughput_after)
        if throughput_before is not None:
            out["throughput_before"] = float(throughput_before)
        # Strip None for compactness (prompt section uses .get).
        return {k: v for k, v in out.items() if v is not None}

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Return the workload-shape KB tag dict for the current session (GAP 5); shared by recipe attrs + lesson/pitfall writes so the warm-start reader filters symmetrically."""
        ss = self.shared_state
        out: dict[str, Any] = {}
        framework = str(getattr(ss, "framework", "") or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(getattr(ss, "model_class", "") or "").strip()
        if model_class:
            out["model_class"] = model_class
        # model_family (v1 fallback) no longer stamped: v2 uses the exact 5-tuple canonical_id.
        model_name = str(getattr(ss, "model_name", "") or "").strip()
        if model_name:
            out["model_name"] = model_name
        for src_attr, dst_key in (
            ("precision",     "precision"),
            ("tp",            "tp"),
            ("ep",            "ep"),
            ("conc",          "conc"),
            ("isl",           "isl"),
            ("osl",           "osl"),
            ("max_model_len", "max_model_len"),
        ):
            v = getattr(ss, src_attr, None)
            if v not in (None, "", 0):
                out[dst_key] = v
        # EP env fallback when SharedState.ep is unset (legacy SDK callers).
        if "ep" not in out:
            raw_ep = (os.environ.get("EP") or "").strip()
            try:
                n = int(raw_ep) if raw_ep else 0
            except ValueError:
                n = 0
            if n > 0:
                out["ep"] = n
        # PP — no SharedState field (no CLI surface); env-only.
        raw_pp = (os.environ.get("PP") or "").strip()
        try:
            pp_n = int(raw_pp) if raw_pp else 0
        except ValueError:
            pp_n = 0
        if pp_n > 0:
            out["pp"] = pp_n
        # runtime version tags from stack_fingerprint_meta (cli writes at boot, resume reads verbatim).
        fp_meta = getattr(ss, "stack_fingerprint_meta", None) or {}
        if isinstance(fp_meta, dict):
            # framework_version is whichever of sglang/vllm is active.
            fw_lc = framework.lower()
            if fw_lc in ("sglang", "vllm"):
                v = str(fp_meta.get(fw_lc) or "").strip()
                if v and v != "unknown":
                    out["framework_version"] = v
            for src_key, dst_key in (
                ("rocm",         "rocm_version"),
                ("aiter",        "aiter_version"),
                ("image_digest", "image_digest"),
            ):
                v = str(fp_meta.get(src_key) or "").strip()
                if v and v != "unknown":
                    out[dst_key] = v
        # per-baseline workload extras from materialized YAML; keep bool False (don't drop an "explicitly disabled" signal).
        wl_extra = getattr(ss, "baseline_workload_extra", None) or {}
        if isinstance(wl_extra, dict):
            for k in ("max_running_requests", "max_num_seqs"):
                v = wl_extra.get(k)
                if isinstance(v, int) and v > 0:
                    out[k] = v
            for k in ("chunked_prefill_enabled", "enable_torch_compile"):
                v = wl_extra.get(k)
                if isinstance(v, bool):
                    out[k] = v
            for k in ("quant_scheme", "workload_mode"):
                v = wl_extra.get(k)
                if isinstance(v, str) and v.strip():
                    out[k] = v.strip()
        return out

    def _build_kernel_optimizations_from_state(self) -> list[dict[str, Any]]:
        """Collect KEEP'd kernel optimizations + their E2E verdict by joining kernel_opt_attempts (micro) and kernel_integrate_attempts (E2E) on kernel_id; non-integrated KEEPs surface integrated=False. Returns KernelOptimization-shaped dicts."""
        ss = self.shared_state
        opt_attempts = getattr(ss, "kernel_opt_attempts", {}) or {}
        integ_attempts = getattr(ss, "kernel_integrate_attempts", {}) or {}
        if not isinstance(opt_attempts, dict):
            return []

        # Index integrate results by kernel_id (last write wins; entry carries rolled-up best_gain_pct).
        integ_by_kid: dict[str, dict[str, Any]] = {}
        if isinstance(integ_attempts, dict):
            for entry in integ_attempts.values():
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get("kernel_id") or "")
                if kid:
                    integ_by_kid[kid] = entry

        out: list[dict[str, Any]] = []
        for kid, e in opt_attempts.items():
            if not isinstance(e, dict):
                continue
            if str(e.get("last_decision", "")).upper() != "KEEP":
                continue
            try:
                micro = float(e.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            integ = integ_by_kid.get(str(kid))
            e2e_gain = 0.0
            e2e_tput = 0.0
            e2e_decision = ""
            integrated = False
            if isinstance(integ, dict):
                integrated = True
                # Integrate-layer verdict (E2E); lets warm-start skip a micro-win/E2E-loss kernel.
                e2e_decision = str(integ.get("last_decision") or "").upper()
                try:
                    e2e_gain = float(integ.get("best_gain_pct") or 0.0)
                except (TypeError, ValueError):
                    e2e_gain = 0.0
                # Last attempt's E2E re-bench throughput.
                for att in reversed(list(integ.get("attempts") or [])):
                    if isinstance(att, dict) and att.get("new_tput") is not None:
                        try:
                            e2e_tput = float(att.get("new_tput") or 0.0)
                        except (TypeError, ValueError):
                            e2e_tput = 0.0
                        break
            out.append({
                "kernel_id":     str(kid),
                # source persisted under last_source_file; source_file is a legacy fallback.
                "source_file":   str(
                    e.get("last_source_file") or e.get("source_file") or ""
                ),
                "artifact_path": str(e.get("last_artifact_path") or ""),
                "micro_speedup": micro,
                "decision":      "KEEP",
                "e2e_gain_pct":  e2e_gain,
                "e2e_tput":      e2e_tput,
                "e2e_decision":  e2e_decision,
                "integrated":    integrated,
                "ts":            str(e.get("last_ts") or e.get("ts") or ""),
            })
        return out

    def _build_recipe_attrs_from_state(self) -> dict[str, Any]:
        """Materialise the recipe-shaped view of :class:`SharedState` (kg-usage-guide §7.4; defensive getattr)."""
        ss = self.shared_state
        current_best = getattr(ss, "current_best", {}) or {}
        opt_stack = getattr(ss, "optimization_stack", []) or []
        gain_per_stack = getattr(ss, "gain_per_stack_entry", []) or []
        last_failures = getattr(ss, "last_action_failures", []) or []
        # Read canonical extra_server_args first, but WRITE the legacy extra_sglang_args key (RecipeKB schema +
        # warm-replay reader still key on it; reading the stale name would break warm-replay reproduction).
        best_config: dict[str, Any] = {}
        if isinstance(current_best, dict):
            cb_args = (
                current_best.get("extra_server_args")
                or current_best.get("extra_sglang_args")
            )
            if cb_args:
                best_config["extra_sglang_args"] = str(cb_args)
            for key in ("extra_envs", "args", "envs", "name", "tput", "accuracy"):
                if key in current_best:
                    best_config[key] = current_best[key]
        # Prefer the last validated stack layer for launch args (current_best may carry a corrupted string).
        if opt_stack:
            last_entry = opt_stack[-1]
            if isinstance(last_entry, dict):
                # Read canonical keys first, legacy *_sglang_args as fallback (#332 best_config fix).
                stack_args = str(
                    last_entry.get("candidate_extra_server_args")
                    or last_entry.get("extra_server_args")
                    or last_entry.get("candidate_extra_sglang_args")
                    or last_entry.get("extra_sglang_args")
                    or "",
                ).strip()
                if stack_args:
                    best_config["extra_sglang_args"] = stack_args
        sediment_on = bool(getattr(ss, "recipe_sediment_enabled", True))
        kept_sources, kept_by_gap, reverted_rows = (
            self._collect_attempt_provenance() if sediment_on else ({}, {}, [])
        )
        what_worked: list[dict[str, Any]] = []
        for idx, entry in enumerate(opt_stack):
            if not isinstance(entry, dict):
                continue
            gain_per: float | None = None
            if idx < len(gain_per_stack):
                gain_per = gain_per_stack[idx]
            name = str(
                entry.get("variant_name")
                or entry.get("name")
                or entry.get("kernel_id")
                or ""
            )
            row: dict[str, Any] = {
                "name":              name,
                "extra_sglang_args": str(
                    entry.get("extra_server_args")
                    or entry.get("extra_sglang_args")
                    or ""
                ),
                "extra_envs":        dict(entry.get("extra_envs") or {}),
                "gain_pct":          gain_per,
            }
            # Prefer the entry's gap-id provenance (naming-independent); fall back to name/kernel_id match.
            entry_gap = str(entry.get("gap_canonical_id") or "").strip()
            src = (
                (kept_by_gap.get(entry_gap) if entry_gap else None)
                or kept_sources.get(name)
                or kept_sources.get(str(entry.get("kernel_id") or ""))
            )
            if src:
                row["source"] = src
            what_worked.append(row)
        what_failed: list[dict[str, Any]] = []
        for failure in last_failures[-10:]:
            if isinstance(failure, dict):
                what_failed.append({
                    "name":  str(failure.get("name") or failure.get("action") or ""),
                    "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                })
        for rev in reverted_rows:
            what_failed.append(rev)
        kernel_optimizations = self._coord._build_kernel_optimizations_from_state()
        cumulative_validated = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
        cumulative_total = float(getattr(ss, "cumulative_gain", 0.0) or 0.0)
        validated_stack_len = int(
            getattr(ss, "cumulative_gain_validated_stack_len", 0) or 0
        )
        stack_fingerprint = getattr(ss, "stack_fingerprint", "") or ""
        # Workload-shape tags for shape-filtered warm-start queries (shared via _collect_workload_tags).
        workload_tags = self._coord._collect_workload_tags()
        # framework_version left unset here (manifest-derived); the T0 backfill writes it.
        return {
            "best_config":       best_config,
            "best_throughput":   float(current_best.get("tput", 0.0))
                                  if isinstance(current_best, dict) else 0.0,
            "what_worked":       what_worked,
            "what_failed":       what_failed,
            "kernel_optimizations": kernel_optimizations,
            "stack_fingerprint": {"sha": str(stack_fingerprint)} if stack_fingerprint else {},
            "last_profiled":     str(getattr(ss, "cumulative_gain_validated_ts", "") or ""),
            "workload":          workload_tags,
            "sessions":          [{
                "session_id":   str(getattr(ss, "cortex_session_id", "")
                                    or self.session_dir.name),
                "gain_pct":     cumulative_validated or cumulative_total,
                "stack_len":    validated_stack_len or len(opt_stack),
                # arbor-shape provenance so the session row is self-describing (before/after tput + knobs).
                "throughput_before": float(getattr(ss, "baseline_tput", 0.0) or 0.0),
                "throughput_after":  (
                    float(current_best.get("tput", 0.0))
                    if isinstance(current_best, dict) else 0.0
                ),
                "date":          datetime.now(timezone.utc).isoformat(),
                "actions_taken": [
                    nm for nm in (
                        str(
                            e.get("variant_name") or e.get("name")
                            or e.get("action") or ""
                        ).strip()
                        for e in opt_stack if isinstance(e, dict)
                    ) if nm
                ],
            }],
        }

    def cortex_finalize_recipe_and_journal(self) -> None:
        """CLOSE-time fact finalize: final update_recipe + journal finalize (total_gain_pct + final_throughput); idempotent (CLOSE sequencer + _cortex_t4_hook safety net)."""
        try:
            journal = self._ensure_journal()
            ss = self.shared_state
            cb = getattr(ss, "current_best", {}) or {}
            final_tput = float(cb.get("tput", 0.0)) if isinstance(cb, dict) else 0.0
            total_gain = float(
                getattr(ss, "cumulative_gain_validated", 0.0)
                or getattr(ss, "cumulative_gain", 0.0)
                or 0.0,
            )
            journal.finalize(
                final_throughput=final_tput if final_tput > 0 else None,
                total_gain_pct=total_gain,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("optimization_journal.finalize failed")

        if self.cortex_kb is None:
            return
        ss = self.shared_state
        model_name = getattr(ss, "model_name", "") or ""
        gpu_type = getattr(ss, "gpu_type", "") or ""
        if not model_name or not gpu_type:
            log.info(
                "cortex finalize_recipe: missing model/hardware "
                "(model=%r hardware=%r); skipping update_recipe",
                model_name, gpu_type,
            )
            return
        try:
            attrs = self._coord._build_recipe_attrs_from_state()
            # Hoist workload tags flat into top-level recipe attrs (shallow-merged) for warm-start filters.
            workload_tags = attrs.get("workload") or {}

            # sessions[] read-modify-write: read anchor, drop prior entry with our session_id (resume safety), append ours, write back.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {
                str((s or {}).get("session_id") or "")
                for s in my_sessions if isinstance(s, dict)
            }
            # v2: read-modify-write the recipe row; sessions[] merged in-process under the cid flock so concurrent finalises don't tear.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.cortex_kb is not None:
                try:
                    cid = self._workload_canonical_id()
                    # Read the LOCAL row (authoritative for writes) so the merge + guard compare against it.
                    existing_row = self.cortex_kb.local.get_recipe(canonical_id=cid) or {}
                    existing_sessions: list[dict[str, Any]] = []
                    for row in (existing_row.get("sessions") or []):
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("session_id") or "") in my_session_ids:
                            # Resume/retry of the same session — our new entry supersedes the prior one.
                            continue
                        existing_sessions.append(dict(row))
                    merged_sessions = existing_sessions + my_sessions
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.info(
                        "recipe read failed (%s); finalize will append "
                        "the current session only; the next finalize "
                        "will catch up.",
                        exc,
                    )

            # KEEP'd kernel optimizations ride the extras channel; merge with prior rows, dedup by kernel_id.
            kopts_new = list(attrs.get("kernel_optimizations") or [])
            new_kids = {
                str((k or {}).get("kernel_id") or "")
                for k in kopts_new if isinstance(k, dict)
            }
            merged_kopts: list[dict[str, Any]] = list(kopts_new)
            for prior in (existing_row.get("kernel_optimizations") or []):
                if not isinstance(prior, dict):
                    continue
                if str(prior.get("kernel_id") or "") in new_kids:
                    continue
                merged_kopts.append(dict(prior))

            extras_payload = dict(workload_tags or {})
            if merged_kopts:
                extras_payload["kernel_optimizations"] = merged_kopts

            overrides: dict[str, Any] = {
                "what_worked":   attrs["what_worked"],
                "what_failed":   attrs["what_failed"],
                "last_profiled": attrs["last_profiled"],
                "sessions":      merged_sessions,
                "extras":        extras_payload,
            }
            # Overwrite best_config/best_throughput only on a real improvement (repro 20260531T144553Z: bare baseline clobbered a validated config): requires has_validated_win AND my_tput > live_tput.
            my_tput = float(attrs.get("best_throughput") or 0.0)
            cb_now = getattr(ss, "current_best", {}) or {}
            cb_args_now = (
                str(cb_now.get("extra_sglang_args") or "").strip()
                if isinstance(cb_now, dict) else ""
            )
            validated_gain = float(
                getattr(ss, "cumulative_gain_validated", 0.0) or 0.0
            )
            has_validated_win = bool(
                (getattr(ss, "optimization_stack", []) or [])
                or validated_gain > 0.0
                or cb_args_now
            )
            try:
                live_tput = float(existing_row.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                live_tput = 0.0
            if has_validated_win and my_tput > live_tput:
                overrides["best_config"] = attrs["best_config"]
                overrides["best_throughput"] = my_tput
            # Merge stack_fingerprint rather than replace (CLOSE only has the sha; T0 stamps version keys).
            merged_fp = dict(existing_row.get("stack_fingerprint") or {})
            for fp_key, fp_val in (attrs.get("stack_fingerprint") or {}).items():
                if fp_val not in (None, "", {}):
                    merged_fp[fp_key] = fp_val
            if merged_fp:
                overrides["stack_fingerprint"] = merged_fp

            self._kb_amend_recipe(
                recipe_overrides=overrides,
                provenance_details={
                    "phase": "close_finalize",
                    "evidence": [
                        f"log:session-{getattr(ss, 'cortex_session_id', '') or self.session_dir.name}",
                    ],
                },
            )
        # Catch-all keeps CLOSE step 2.5 defensive against programmer bugs.
        except Exception:  # noqa: BLE001 — defensive
            log.exception("update_recipe raised unexpectedly")

    def _aggregate_research_evidence(self, done_payload: dict[str, Any]) -> None:
        """Aggregate research evidence (PR ids / diffs / NVIDIA refs) into the
        session-wide seen-set, de-duped across the session.

        Applies to every domain that self-reports a ``research`` block
        (``pr_intel`` + ``research_scout``), so FRAMEWORK_PR / explore lanes do
        not re-fetch the same references. Fail-soft: never raises (the caller
        also guards, but keep this self-contained so partial payloads degrade
        gracefully).
        """
        block = done_payload.get("research")
        if not isinstance(block, dict):
            return
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        if not pr_ids:
            return
        try:
            added = self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "depth: register_seen_pr_ids failed during research aggregation",
            )
            return
        if added:
            log.info(
                "depth: aggregated %d new research reference(s) into seen-set",
                added,
            )

    def _harvest_research_scout(self, done_payload: dict[str, Any]) -> None:
        """Persist scout output (hints, competitor target, gap seeds, dedup); all steps fail-soft."""
        from . import research_hints as _research_hints

        block = done_payload.get("research")
        if not isinstance(block, dict):
            block = {}
        hints = block.get("hints") or []
        try:
            added, dropped = _research_hints.append_hints(
                self.session_dir, hints,
            )
            if dropped:
                log.info(
                    "research-scout: dropped %d sourceless hint(s)", dropped,
                )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: append_hints failed")
            added = 0
        try:
            _research_hints.write_competitor_target(
                self.session_dir, block.get("competitor_target"),
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: competitor_target write failed")
        # Share inspected PR ids with the FRAMEWORK_PR dedup set.
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        try:
            self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: register_seen_pr_ids failed")
        # Seed high-priority hints as gaps[] so EXPLORE tries them early.
        try:
            self._seed_gaps_from_research_hints()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: gap seeding failed")
        log.info(
            "research-scout harvested: hints_added=%d seen_pr_ids=%d",
            added, len(self.shared_state.research_scout_seen_pr_ids or []),
        )

