# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from .shared_state import SharedState
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)

from .coordinator import (
    PendingProposal,
)
import logging as _logging
log = _logging.getLogger(__name__)


class ResumeCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    # Resume
    def _detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking).

        Returns:
            A dict with ``is_resume``, ``event_count``, ``state_json_present``
            and ``rebuilt`` (the last set later by :meth:`replay_for_resume`).
        """
        ev_count = self.bus.db.fetchone_sync("SELECT COUNT(*) AS c FROM events")
        events_present = (int(ev_count["c"]) if ev_count else 0) > 0
        state_path = SharedState.state_path(self.session_dir)
        return {
            "is_resume": events_present or state_path.exists(),
            "event_count": int(ev_count["c"]) if ev_count else 0,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }

    async def replay_for_resume(self) -> dict[str, Any]:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it.

        Returns:
            A dict summarising the replay: ``is_resume``, ``event_count``,
            ``state_json_present``, ``pending_restored`` (count rebuilt) and
            ``verdicts_seen``.
        """
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Verdicts with a verdict_map but no summary are treated as needs_review.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)

        rebuilt = 0
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
                continue
            payload = p.payload or {}
            self.state.pending_proposals[p.msg_id] = PendingProposal(
                proposal_msg_id=p.msg_id,
                from_agent=p.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )
            rebuilt += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "verdicts_seen": len(verdicts),
        }

    def _materialize_stack_config_for_resume(self) -> dict[str, Any]:
        """Rebuild cumulative launch args/envs from ``optimization_stack``."""
        stack = [e for e in (getattr(self.shared_state, "optimization_stack", []) or []) if isinstance(e, dict)]
        args = ""
        envs: dict[str, str] = {}
        tput: float | None = None
        variant_name = ""
        action = "resume_reconstructed"
        workspace = None
        for entry in stack:
            candidate = str(entry.get("candidate_extra_server_args") or "").strip()
            full = str(entry.get("extra_server_args") or entry.get("extra_sglang_args") or "").strip()
            args = _merge_cumulative_extra_sglang_args(args, candidate, full)
            raw_envs = entry.get("extra_envs") or {}
            if isinstance(raw_envs, Mapping):
                envs.update({str(k): str(v) for k, v in raw_envs.items()})
            if isinstance(entry.get("tput"), (int, float)) and float(entry["tput"]) > 0:
                tput = float(entry["tput"])
            variant_name = str(entry.get("variant_name") or variant_name or "")
            action = str(entry.get("action") or action)
            workspace = entry.get("workspace") or workspace
        return {
            "action": action,
            "variant_name": variant_name,
            "extra_server_args": args,
            "extra_envs": envs,
            "tput": tput,
            "workspace": workspace,
            "optimization_stack": stack,
        }

    async def _resume_consistency_pass(self) -> dict[str, Any]:
        """One-shot resume audit + recovery for stack/current_best consistency.

        Order matters: recover half-applied / orphaned KEEPs FIRST (they mutate
        the stack), then reconcile ``current_best`` against the resulting stack,
        then compensate the validation watermark by enqueuing a single
        full-stack end-to-end rebench. Idempotent — only runs on a resumed
        session and every recovery step dedupes, so a second pass is a no-op.
        """
        if not self._resumed_from.get("is_resume"):
            return {"skipped": True, "reason": "not_resume"}
        state = self.shared_state
        report: dict[str, Any] = {
            "skipped": False,
            "fixes": [],
            "warnings": [],
        }
        # (1) Half-applied integrate window: replay the
        # missing stack append or roll back the partial patch BEFORE anything
        # reads the stack, so the rest of the pass sees the recovered truth.
        await self._resume_recover_pending_integrate(report)
        # (2) Orphaned KEEPs: replay integrate_patch KEEPs
        # that crashed before the append landed; surface ambiguous ones loudly.
        await self._resume_recover_orphaned_keeps(report)

        # (3) current_best <-> stack reconcile (after 1/2 may have grown stack).
        stack = [e for e in (getattr(state, "optimization_stack", []) or []) if isinstance(e, dict)]
        cb = state.current_best if isinstance(state.current_best, dict) else {}
        if stack:
            rebuilt = self._materialize_stack_config_for_resume()
            cb_args = str(cb.get("extra_server_args") or "")
            cb_envs = {str(k): str(v) for k, v in (cb.get("extra_envs") or {}).items()} if isinstance(cb.get("extra_envs"), Mapping) else {}
            if cb_args != rebuilt["extra_server_args"] or cb_envs != rebuilt["extra_envs"]:
                # The append-only stack is authoritative; a disagreeing
                # current_best is the inconsistency, recorded distinctly from the
                # rebuild fix so operators can see a stale best was detected.
                report["warnings"].append(
                    {
                        "kind": "resume_inconsistent_current_best",
                        "current_best_args": cb_args,
                        "stack_args": rebuilt["extra_server_args"],
                    }
                )
                new_cb = dict(cb)
                new_cb.update(
                    {
                        "action": rebuilt["action"],
                        "variant_name": rebuilt["variant_name"],
                        "extra_server_args": rebuilt["extra_server_args"],
                        "extra_envs": rebuilt["extra_envs"],
                        "optimization_stack": list(stack),
                        "source": "resume_consistency_rebuild_from_stack",
                    }
                )
                if rebuilt["tput"] is not None and not isinstance(new_cb.get("tput"), (int, float)):
                    new_cb["tput"] = rebuilt["tput"]
                if rebuilt["workspace"] and not new_cb.get("workspace"):
                    new_cb["workspace"] = rebuilt["workspace"]
                state.current_best = new_cb
                report["fixes"].append("rebuilt_current_best_config_from_stack")
        elif cb:
            # Legacy sessions before the append-only stack existed are still
            # recoverable; seed once instead of dropping a possibly valid best.
            before = len(getattr(state, "optimization_stack", []) or [])
            state.seed_stack_from_current_best()
            after = len(getattr(state, "optimization_stack", []) or [])
            if after > before:
                report["fixes"].append("seeded_stack_from_legacy_current_best")
            else:
                report["warnings"].append({"kind": "current_best_without_stack"})

        # (4) Validation-watermark compensation: unvalidated
        # KEEPs (claimed gain not yet end-to-end confirmed) → flag + enqueue ONE
        # full-stack rebench. The flag + watermark are reconciled from the
        # measured tput when that rebench promotes (see _promote_to_shared_state).
        stack = [e for e in (getattr(state, "optimization_stack", []) or []) if isinstance(e, dict)]
        vlen = int(getattr(state, "cumulative_gain_validated_stack_len", 0) or 0)
        if vlen < len(stack):
            state.resume_pending_revalidation = True
            report["warnings"].append(
                {
                    "kind": "resume_unvalidated_keeps",
                    "validated_stack_len": vlen,
                    "stack_len": len(stack),
                }
            )
            try:
                fix = await self._enqueue_internal_stack_rebench(reason="resume_unvalidated_keeps")
                report["fixes"].append({"kind": "queued_resume_stack_rebench", **fix})
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: failed to enqueue resume stack rebench")
                report["warnings"].append({"kind": "resume_stack_rebench_enqueue_failed"})

        if os.environ.get("INFERENCE_OPTIMIZER_RESUME_REVERIFY_BEST", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            cb_now = state.current_best if isinstance(state.current_best, dict) else {}
            cb_args = str(cb_now.get("extra_server_args") or "").strip()
            cb_envs = cb_now.get("extra_envs") if isinstance(cb_now.get("extra_envs"), Mapping) else {}
            if cb_args or cb_envs:
                try:
                    tput = cb_now.get("tput")
                    params: dict[str, Any] = {
                        "source": "resume_reverify_best",
                        "reason": "resume_reverify_best",
                        "grid": [
                            {
                                "name": "resume_current_best",
                                "extra_args": cb_args,
                                "extra_envs": dict(cb_envs),
                                "provenance": "resume_reverify_best",
                                "note": "env-requested post-resume current_best recheck",
                            }
                        ],
                        "base_tput": float(tput) if isinstance(tput, (int, float)) and tput > 0 else 0.0,
                        "enable_stack_rebench": False,
                    }
                    if state.baseline_config_path:
                        params["config_path"] = state.baseline_config_path
                    task, existing = await self.tasks.create_or_return_existing(
                        kind="explore",
                        params=params,
                        idempotency_key="resume-reverify-current-best",
                    )
                    report["fixes"].append(
                        {
                            "kind": "queued_resume_reverify_best",
                            "task_id": task.task_id,
                            "existing": bool(existing),
                        }
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Coordinator: failed to queue resume current_best reverify")
                    report["warnings"].append({"kind": "resume_reverify_best_enqueue_failed"})
            else:
                report["warnings"].append({"kind": "resume_reverify_best_no_config"})
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: resume consistency save failed")
        await self._record_observation("coordinator", "observation", {"kind": "resume_consistency", **report})
        return report

    def _replay_keep_from_result(self, kind: str, result: dict[str, Any]) -> bool:
        """Replay a recorded KEEP delegated-result into current_best/stack.

        Reconstructs the winning-variant dict from a persisted ``delegated_result``
        and routes it through :meth:`_lift_to_current_best`, which dedupes by
        ``(action, variant_name)`` — so replay is idempotent. Used by both the
        pending-integrate (Gap C) and orphaned-KEEP (Gap B) resume recovery
        paths. Returns ``True`` only when a new stack entry was appended.

        Args:
            kind: The originating action kind (``integrate_patch`` / ``explore``
                / ``framework``).
            result: The recorded delegated result payload for that KEEP.

        Returns:
            ``True`` when the replay appended a new stack entry, else ``False``.
        """
        if not isinstance(result, dict):
            return False
        tput = result.get("output_throughput")
        if not (isinstance(tput, (int, float)) and float(tput) > 0):
            return False
        if kind == "explore":
            bv_src = result.get("best_variant")
            if not isinstance(bv_src, dict) or not bv_src.get("name"):
                return False
            bv = dict(bv_src)
        elif kind == "integrate_patch":
            sid = str(result.get("specialist_task_id") or "")
            if not sid:
                return False
            bv = {
                "name": sid,
                "candidate_extra_server_args": "",
                "extra_envs": dict(result.get("config_changes_applied") or {}),
                "tput": float(tput),
                "workspace": result.get("workspace"),
                "provenance": "integrate_patch",
                "scope": "source_patch",
            }
        else:
            return False
        before = len(self.shared_state.optimization_stack or [])
        self._lift_to_current_best(kind, float(tput), bv)
        return len(self.shared_state.optimization_stack or []) > before

    def _resume_rollback_pending_integrate(self, pending: dict[str, Any]) -> dict[str, Any]:
        """Reverse-apply a half-applied integrate patch set (Gap C rollback).

        Best-effort ``git apply -R`` of every patch recorded on the
        ``pending_integrate`` sentinel into the framework source tree, so a
        crash AFTER ``git apply`` but BEFORE the bench/KEEP cannot leak a partial
        change into later launches. A patch that is not currently applied simply
        fails the reverse ``--check`` and is reported, not retried.

        Args:
            pending: The ``pending_integrate`` sentinel dict.

        Returns:
            A summary ``{"reversed": [...], "failed": [...]}``.
        """
        from .action_executors.integrate_patch import _git_apply_reverse

        summary: dict[str, Any] = {"reversed": [], "failed": []}
        root = str(pending.get("framework_source_root") or "").strip()
        patches = [str(p) for p in (pending.get("patches") or []) if str(p).strip()]
        if not root or not patches:
            return summary
        root_path = Path(root)
        for patch in patches:
            try:
                ok, err = _git_apply_reverse(root_path, Path(patch))
            except Exception as exc:  # noqa: BLE001 — rollback is best-effort
                summary["failed"].append({"patch": patch, "error": repr(exc)})
                continue
            if ok:
                summary["reversed"].append(patch)
            else:
                summary["failed"].append({"patch": patch, "error": err})
        return summary

    async def _resume_recover_pending_integrate(self, report: dict[str, Any]) -> None:
        """Recover a crashed integrate_patch window from the sentinel (Gap C).

        Three-way decision keyed on whether a ``kept`` delegated-result exists
        for the sentinel's task: replay the missing append (crashed after KEEP),
        roll back the half-applied patch (crashed after apply, before KEEP), or
        clear a stale sentinel. The sentinel is always cleared afterwards.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        pending = getattr(state, "pending_integrate", {}) or {}
        if not (isinstance(pending, dict) and pending):
            return
        task_id = str(pending.get("task_id") or "")
        kept_res: dict[str, Any] | None = None
        try:
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                if task_id and str(payload.get("task_id") or "") != task_id:
                    continue
                res = payload.get("result") or {}
                # Require an explicit integrate_patch kind: an empty-kind wildcard
                # could misclassify a non-integrate event that happens to share
                # this task_id as a kept integrate result, skipping rollback of a
                # half-applied patch.
                if (
                    isinstance(res, dict)
                    and str(res.get("kind") or payload.get("kind") or "") == "integrate_patch"
                    and str(res.get("status") or "").lower() == "kept"
                ):
                    kept_res = res
                    break
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: pending_integrate kept-result scan failed")
        if kept_res is not None:
            appended = self._replay_keep_from_result("integrate_patch", kept_res)
            report["fixes"].append(
                {"kind": "replayed_pending_integrate", "task_id": task_id, "appended": bool(appended)}
            )
        else:
            summary = self._resume_rollback_pending_integrate(pending)
            if summary.get("reversed"):
                report["fixes"].append(
                    {"kind": "rolled_back_pending_integrate", "task_id": task_id, **summary}
                )
            elif summary.get("failed"):
                report["warnings"].append(
                    {"kind": "pending_integrate_rollback_failed", "task_id": task_id, **summary}
                )
            else:
                report["fixes"].append({"kind": "cleared_stale_pending_integrate", "task_id": task_id})
        state.pending_integrate = {}

    async def _resume_recover_orphaned_keeps(self, report: dict[str, Any]) -> None:
        """Recover / surface KEEPs present in the event log but absent from the stack (Gap B).

        ``integrate_patch`` KEEPs are well-defined (a ``kept`` status means the
        single-variant bench + accuracy gate passed and the patch was committed),
        so a kept-but-absent one is a crash before the append landed → replay it
        (idempotent), unless its run workspace is gone → discard + alert. ``explore``
        / ``framework`` KEEPs are ambiguous (KEEP_UNSTABLE eviction can drop a
        kept explore variant from the stack), so they are surfaced as a
        ``medium`` alert rather than resurrected. Whatever the stack ends up as
        is re-validated by the Gap A full-stack rebench.

        Args:
            report: The resume report dict to append fixes/warnings to.
        """
        state = self.shared_state
        try:
            stack_keys = {
                (str(e.get("action") or ""), str(e.get("variant_name") or ""))
                for e in (state.optimization_stack or [])
                if isinstance(e, dict)
            }
            seen: set[tuple[str, str]] = set()
            for msg in await self.bus.tail(topic="delegated_result", n=10_000):
                payload = msg.payload or {}
                kind = str(payload.get("kind") or "")
                res = payload.get("result") or {}
                if not isinstance(res, dict) or str(res.get("status") or "").lower() != "kept":
                    continue
                if kind == "integrate_patch":
                    variant = str(res.get("specialist_task_id") or "")
                elif kind == "framework_agent":
                    cand = res.get("candidate") or {}
                    variant = str(
                        (cand.get("candidate_id") if isinstance(cand, dict) else "")
                        or (cand.get("pr_url") if isinstance(cand, dict) else "")
                        or ""
                    )
                elif kind == "explore":
                    bv = res.get("best_variant") or {}
                    variant = str((bv.get("name") if isinstance(bv, dict) else "") or "")
                else:
                    continue
                key = (kind, variant)
                if not variant or key in stack_keys or key in seen:
                    continue
                seen.add(key)
                if kind == "integrate_patch":
                    workspace = str(res.get("workspace") or "").strip()
                    if workspace and not Path(workspace).exists():
                        report["warnings"].append(
                            {
                                "kind": "orphaned_keep_discarded",
                                "orphan_kind": kind,
                                "variant": variant,
                                "task_id": payload.get("task_id"),
                                "reason": "workspace_missing",
                            }
                        )
                        await self._record_observation(
                            "coordinator",
                            "observation",
                            {
                                "kind": "orphaned_keep_discarded",
                                "severity": "medium",
                                "orphan_kind": kind,
                                "variant": variant,
                            },
                        )
                    elif self._replay_keep_from_result(kind, res):
                        stack_keys.add(key)
                        report["fixes"].append(
                            {"kind": "replayed_orphaned_keep", "orphan_kind": kind, "variant": variant}
                        )
                    else:
                        report["warnings"].append(
                            {"kind": "orphaned_keep_replay_noop", "orphan_kind": kind, "variant": variant}
                        )
                else:
                    # explore / framework: ambiguous vs eviction — never
                    # resurrect; surface for the operator.
                    report["warnings"].append(
                        {
                            "kind": "orphaned_keep",
                            "orphan_kind": kind,
                            "variant": variant,
                            "task_id": payload.get("task_id"),
                        }
                    )
                    await self._record_observation(
                        "coordinator",
                        "observation",
                        {
                            "kind": "orphaned_keep",
                            "severity": "medium",
                            "orphan_kind": kind,
                            "variant": variant,
                        },
                    )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: orphaned KEEP resume recovery failed")

    async def _enqueue_internal_stack_rebench(self, *, reason: str) -> dict[str, Any]:
        """Enqueue one full-stack end-to-end rebench of the cumulative config (Gap A).

        Builds a single-variant ``explore`` task from the stack-materialized
        launch args/envs, benched against ``baseline_tput`` so the measured
        delta becomes the validated cumulative gain. Tagged
        ``source=resume_stack_revalidate`` so ``_promote_to_shared_state``
        reconciles ``cumulative_gain_validated_stack_len`` + clears
        ``resume_pending_revalidation`` from the measured throughput. Idempotent
        via a fixed idempotency key.

        Args:
            reason: Human-readable reason stamped on the task params.

        Returns:
            A summary ``{"task_id", "existing"}`` or ``{"skipped", "reason"}``.
        """
        rebuilt = self._materialize_stack_config_for_resume()
        args = str(rebuilt.get("extra_server_args") or "").strip()
        envs = rebuilt.get("extra_envs") or {}
        if not (args or envs):
            return {"skipped": True, "reason": "empty_stack"}
        params: dict[str, Any] = {
            "source": "resume_stack_revalidate",
            "reason": reason,
            "grid": [
                {
                    "name": "resume_stack_revalidate",
                    "extra_args": args,
                    "extra_envs": dict(envs),
                    "provenance": "resume_stack_revalidate",
                    "note": "post-resume full-stack end-to-end revalidation",
                }
            ],
            "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
            "enable_stack_rebench": False,
        }
        if self.shared_state.baseline_config_path:
            params["config_path"] = self.shared_state.baseline_config_path
        task, existing = await self.tasks.create_or_return_existing(
            kind="explore",
            params=params,
            idempotency_key="resume-stack-revalidate",
        )
        return {"task_id": task.task_id, "existing": bool(existing)}

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns:
            A copy of the resume-detection dict so callers cannot mutate
            internal state.
        """
        return dict(self._resumed_from)

    # Bounded test interface
    async def _replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        await self._resume_consistency_pass()
