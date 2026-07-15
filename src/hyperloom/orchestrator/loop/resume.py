# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from ..state.shared_state import SharedState
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_server_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _geak_sweep_measured_tput,
    _resolve_roofline_watermark_ratio,
    _scrape_resolved_launch_flags,
    _split_env_and_flags,
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
            "rebuilt": False,
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
        overlay = ""
        tput: float | None = None
        variant_name = ""
        action = "resume_reconstructed"
        workspace = None
        for entry in stack:
            candidate = str(entry.get("candidate_extra_server_args") or "").strip()
            full = str(entry.get("extra_server_args") or "").strip()
            args = _merge_cumulative_extra_server_args(args, candidate, full)
            raw_envs = entry.get("extra_envs") or {}
            if isinstance(raw_envs, Mapping):
                for k, v in raw_envs.items():
                    ks = str(k)
                    # A ``-``-prefixed key mis-stored under extra_envs is a server
                    # arg, not an env var; route it back into extra_server_args.
                    if ks.startswith("-"):
                        tok = ks if v in ("", None) else f"{ks}={v}"
                        args = _merge_cumulative_extra_server_args(args, "", tok)
                    else:
                        envs[ks] = str(v)
            # Carry the authored-kernel overlay (PYTHONPATH prefix) so a native
            # rebuild loads the built kernels. Last non-empty wins.
            entry_overlay = str(entry.get("final_overlay") or "").strip()
            if entry_overlay:
                overlay = entry_overlay
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
            "final_overlay": overlay,
            "tput": tput,
            "workspace": workspace,
            "optimization_stack": stack,
        }

    def build_env_spec(self) -> dict[str, Any]:
        """Fully-reproducible descriptor of ``current_best``'s launch environment.

        Layers, in the order a consumer must apply them to reconstruct the exact
        stack ``current_best`` was measured on:

          * ``config``  — cumulative server args + env vars (the reversible layer).
          * ``source_snapshots`` — ordered durable source-layer snapshots
            (``scope=source_patch`` entries), each a self-contained directory
            (see :mod:`source_snapshot`) that reconstructs the patched framework
            tree independent of the mutable live checkout.
          * ``overlay_pythonpath`` — the authored-kernel overlay prefix.
          * ``launch_recipe`` — the baseline Magpie recipe to launch from.

        The single source of truth the GEAK handoff forwards so the baseline ref
        is materialized from the same layers as ``current_best``.
        """
        materialized = self._materialize_stack_config_for_resume()
        stack = [
            e
            for e in (getattr(self.shared_state, "optimization_stack", []) or [])
            if isinstance(e, dict)
        ]
        source_snapshots: list[dict[str, Any]] = []
        for entry in stack:
            if entry.get("scope") != "source_patch":
                continue
            snap = str(entry.get("source_snapshot") or "").strip()
            if not snap:
                # A source_patch with no durable snapshot is surfaced so the
                # consumer can flag an unreproducible baseline.
                source_snapshots.append(
                    {
                        "id": str(entry.get("variant_name") or entry.get("name") or ""),
                        "snapshot_dir": "",
                        "framework_root": str(entry.get("framework_root") or ""),
                        "base_sha": str(entry.get("base_sha") or ""),
                        "reproducible": False,
                    }
                )
                continue
            source_snapshots.append(
                {
                    "id": str(entry.get("variant_name") or entry.get("name") or ""),
                    "snapshot_dir": snap,
                    "framework_root": str(entry.get("framework_root") or ""),
                    "base_sha": str(entry.get("base_sha") or ""),
                    "reproducible": True,
                }
            )
        # FULL resolved engine config (not just the current_best delta): the
        # complete server-launch flag set the orchestrator ran, scraped from the
        # launched argv. ``extra_server_args``/``extra_envs`` remain the
        # current_best delta (a consumer merges the delta on top, delta wins).
        server_launch_flags = ""
        try:
            cb_now = getattr(self.shared_state, "current_best", None)
            _target_tput = (
                float((cb_now or {}).get("tput") or 0.0)
                if isinstance(cb_now, Mapping)
                else 0.0
            )
            server_launch_flags = _scrape_resolved_launch_flags(
                getattr(self, "session_dir", ""),
                str(os.environ.get("FRAMEWORK", "") or "sglang"),
                target_tput=_target_tput,
            )
        except Exception:  # noqa: BLE001
            server_launch_flags = ""
        return {
            "schema_version": 1,
            "config": {
                "extra_server_args": materialized.get("extra_server_args") or "",
                "extra_envs": dict(materialized.get("extra_envs") or {}),
                # Complete engine flags (run-specific stripped); empty => consumer keeps its defaults.
                "server_launch_flags": server_launch_flags,
            },
            "source_snapshots": source_snapshots,
            "overlay_pythonpath": materialized.get("final_overlay") or "",
            "launch_recipe": str(getattr(self.shared_state, "baseline_config_path", "") or ""),
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
        # (1) Half-applied integrate window: replay the missing stack append or
        # roll back the partial patch before anything reads the stack.
        await self._resume_recover_pending_integrate(report)
        # (2) Orphaned KEEPs: replay integrate_patch KEEPs that crashed before
        # the append landed; surface ambiguous ones.
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
                # current_best is the inconsistency.
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
            # Legacy sessions without an append-only stack: seed once instead of
            # dropping a possibly valid best.
            before = len(getattr(state, "optimization_stack", []) or [])
            state.seed_stack_from_current_best()
            after = len(getattr(state, "optimization_stack", []) or [])
            if after > before:
                report["fixes"].append("seeded_stack_from_legacy_current_best")
            else:
                report["warnings"].append({"kind": "current_best_without_stack"})

        # (4) Validation-watermark compensation: unvalidated KEEPs → flag +
        # enqueue one full-stack rebench (reconciled on promote).
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
                # Durable source-layer handles so a source_patch recovered here
                # is equally reproducible in the GEAK baseline.
                "source_snapshot": result.get("source_snapshot") or "",
                "framework_root": result.get("framework_root") or "",
                "base_sha": result.get("base_sha") or "",
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
        from ..actions.executors.integrate_patch import _git_apply_reverse

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
                # Require an explicit integrate_patch kind so a non-integrate
                # event sharing this task_id is not misclassified as a kept patch.
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
                    # explore / framework: ambiguous vs eviction — never resurrect.
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
        # When the win is a GEAK e2e result, source the revalidation config from
        # result.json (not stack materialization) so the same-harness rebench
        # launches byte-for-byte the config GEAK optimized. The consumer asserts
        # config identity + effect before stamping validated, falling back to 2a.
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        ps_cfg = ps.get("accepted_config") or {}
        ps_overlay = str(ps.get("final_overlay") or "").strip()
        if str(ps.get("status") or "") == "ok" and (ps_cfg.get("flags") or ps_cfg.get("env") or ps_overlay):
            from ..actions.executors._canonical_fingerprint import canonical_fingerprint

            ps_flags = str(ps_cfg.get("flags") or "").strip()
            ps_envs, _ps_extra_flags = _split_env_and_flags(str(ps_cfg.get("env") or ""))
            if _ps_extra_flags:
                ps_flags = (ps_flags + " " + _ps_extra_flags).strip()
            if ps_flags or ps_envs or ps_overlay:
                # Identity hash uses the same (args, envs) contract the grid
                # executor fingerprints with (overlay excluded), so expected ==
                # the ran variant's fingerprint by construction.
                expected_cfg_hash = canonical_fingerprint(ps_flags, ps_envs)
                params_ps: dict[str, Any] = {
                    "source": "resume_stack_revalidate",
                    "reason": reason,
                    "geak_fallback": True,
                    "expected_cfg_hash": expected_cfg_hash,
                    "grid": [
                        {
                            "name": "geak_revalidate",
                            "extra_args": ps_flags,
                            "extra_envs": dict(ps_envs),
                            "overlay_pythonpath": ps_overlay,
                            "provenance": "geak_revalidate",
                            "note": "same-harness config-identity revalidation of the geak e2e win",
                        }
                    ],
                    "base_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                    "enable_stack_rebench": False,
                }
                if self.shared_state.baseline_config_path:
                    params_ps["config_path"] = self.shared_state.baseline_config_path
                task, existing = await self.tasks.create_or_return_existing(
                    kind="explore",
                    params=params_ps,
                    idempotency_key="geak-revalidate",
                )
                return {"task_id": task.task_id, "existing": bool(existing), "mode": "geak_2b"}

        rebuilt = self._materialize_stack_config_for_resume()
        args = str(rebuilt.get("extra_server_args") or "").strip()
        envs = rebuilt.get("extra_envs") or {}
        overlay = str(rebuilt.get("final_overlay") or "").strip()
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
                    # Carry the overlay so a native stack rebuild loads the built kernels.
                    "overlay_pythonpath": overlay,
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

    async def _validate_geak_via_geak_harness(self, *, reason: str) -> dict[str, Any]:
        """2a fallback - validate the geak win by replaying it through GEAK's own
        ``bench_e2e.sh``, so the optimized config engages by construction. A
        ``succeeded`` status is itself the engagement proof; the validated gain
        is the same-harness A/B ``hot_geak_speedup``. Used only when 2b
        (orchestrator harness) is inconclusive.

        Args:
            reason: Human-readable reason stamped in logs/return.

        Returns:
            A summary dict describing whether validation succeeded.
        """
        ps = self.shared_state.geak_result if isinstance(getattr(self.shared_state, "geak_result", None), dict) else {}
        if str(ps.get("status") or "") != "ok":
            return {"validated": False, "skipped": True, "reason": "no_geak_result"}
        am = ps.get("alignment_metrics") or {}
        # Use GEAK's own within-harness speedup on the same basis it promoted, so
        # the validated gain equals GEAK's headline number. Falls back to the
        # explicit within-GEAK ratios when throughput_speedup is missing.
        try:
            geak_sp = float(ps.get("throughput_speedup") or 0.0)
        except (TypeError, ValueError):
            geak_sp = 0.0
        if geak_sp <= 0:
            basis = str(am.get("final_basis") or ps.get("final_throughput_basis") or "hot")
            fallback_key = "cold_geak_speedup" if basis == "cold" else "hot_geak_speedup"
            try:
                geak_sp = float(am.get(fallback_key) or am.get("hot_geak_speedup") or 0.0)
            except (TypeError, ValueError):
                geak_sp = 0.0
        regimes = ps.get("validated_regimes") or []
        reg = regimes[0] if regimes and isinstance(regimes[0], dict) else {}
        try:
            conc = int(reg.get("conc") or 64)
            isl = int(reg.get("isl") or 1024)
            osl = int(reg.get("osl") or 1024)
        except (TypeError, ValueError):
            conc, isl, osl = 64, 1024, 1024
        from hyperloom.inference_optimizer.session.session_paths import runs_dir
        from ..actions.executors._geak_sweep import sweep_via_geak

        try:
            timeout = int(os.environ.get("SWEEP_VARIANT_TIMEOUT_SEC", "").strip() or "2400")
        except (TypeError, ValueError):
            timeout = 2400
        res = await sweep_via_geak(
            result=ps,
            conc_values=[conc],
            isl_osl_configs=[f"{isl}:{osl}"],
            output_root=runs_dir(self.session_dir, "sweep", "revalidate_geak"),
            variant_timeout_sec=timeout,
            repeats=3,
            # Pin the headline protocol so the replay is protocol-identical to the reported result.
            pin_num_prompts=True,
        )
        if str(res.get("status") or "") == "succeeded" and geak_sp > 1.0:
            # Write the headline from the GEAK-harness measured throughput,
            # keeping the leaderboard number a same-harness total.
            measured = _geak_sweep_measured_tput(res)
            if measured is None:
                log.warning(
                    "geak 2a: succeeded sweep but no measurable throughput; "
                    "candidate stays pending"
                )
                return {"validated": False, "status": res.get("status"), "reason": reason}
            self._promote_geak_from_candidate(
                ps,
                measured_tput=measured,
                provenance="geak_same_harness_geak",
            )
            base = float(self.shared_state.baseline_tput or 0.0)
            gain_out = ((measured - base) / base * 100.0) if base > 0 else 0.0
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("geak 2a: SharedState.save failed")
            return {"validated": True, "gain": gain_out, "reason": reason}
        log.warning(
            "geak 2a fallback did not validate (status=%r geak_speedup=%r reason=%s)",
            res.get("status"), geak_sp, reason,
        )
        return {"validated": False, "status": res.get("status"), "reason": reason}

    async def _resume_reenter_kernel_if_needed(self) -> None:
        """Idempotently re-fire the KERNEL_AGENT entry hook on resume.

        Phase-entry side effects (the GEAK delegation + its ``result.json``
        crash-recovery) are bound to a phase transition; a resume only restores
        ``phase`` and never re-enters the current phase, so without this a session
        that crashed mid ``KERNEL_AGENT`` sits idle until the budget cap fires.

        Keyed on whether this KERNEL phase's history row already carries a
        ``geak`` completion record:

          * completed-this-phase -> only re-arm (+persist) the ``skip_to_sweep``
            hint so the phase machine winds down to SWEEP with no e2e re-run;
          * not-completed -> re-enter ``_on_enter_kernel``; its entry guard
            promotes an existing OK ``result.json`` and re-runs the e2e only when
            there is nothing to recover.

        No-op unless resumed while parked in ``KERNEL_AGENT`` with GEAK selected.
        """
        from ..phases.machine_state import (
            ESCALATE_HINT_SKIP_TO_SWEEP,
            PHASE_KERNEL_AGENT,
        )

        if not self._resumed_from.get("is_resume"):
            return
        state = self.shared_state
        if (state.phase or "").strip().upper() != PHASE_KERNEL_AGENT:
            return
        if not (self._kernel_enabled() and self._geak_enabled()):
            return
        history = state.phase_history or []
        row = history[-1] if history else {}
        evidence = row.get("evidence") if isinstance(row, dict) else {}
        completed_this_phase = isinstance(evidence, dict) and isinstance(
            evidence.get("geak"), dict
        )
        if completed_this_phase:
            # The delegation landed but the SWEEP transition never persisted.
            # Re-arm the wind-down hint + persist so the phase machine advances.
            cur = str(getattr(state, "pending_escalate_hint", "") or "").strip()
            if cur != ESCALATE_HINT_SKIP_TO_SWEEP:
                state.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)
                try:
                    state.save(self.session_dir)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "resume: save after re-arming skip_to_sweep failed"
                    )
                log.info(
                    "resume: KERNEL GEAK already completed this phase; "
                    "re-armed skip_to_sweep hint (lost before SWEEP transition)."
                )
            return
        log.info(
            "resume: re-entering KERNEL GEAK delegation (no completion "
            "evidence on the current phase row); recover-from-disk or re-run."
        )
        try:
            await self._on_enter_kernel(from_phase="resume")
        except Exception:  # noqa: BLE001
            log.exception("resume: KERNEL re-entry hook failed")

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
        # Re-fire the KERNEL delegation hook when resuming parked in KERNEL_AGENT,
        # after the consistency pass so current_best/stack are already rebuilt.
        await self._resume_reenter_kernel_if_needed()
