# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE warm-replay settlement: grading the replay, ruling on its keep, and adopting, rejecting or rolling it back."""

from __future__ import annotations

import logging as _logging
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from hyperloom_common.coerce import to_float
from hyperloom.common.perf_metric import GradingAnchor, VERDICT_KEEP, constraint_fields, graded_axes_of
from hyperloom_common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    GATE_ACCURACY,
    GATE_KEEP_THRESHOLD,
    GATE_PARAMS_PRESENT,
    GATE_PROMOTION,
    GATE_QUALITY,
    GATE_TPUT_VALID,
    WarmReplayApplyItem,
    WarmReplayMeasurement,
)
from . import machine_state as _phase_state
from hyperloom.inference_optimizer.session.optimization_journal import (
    JournalEntry,
    KIND_OTHER,
    Verdict,
    derive_journal_outcome,
)
from ..actions.executors._accuracy_gate import AccuracyReference
from ..actions.executors.benchmark_run.warm_patches import revert_warm_kernel_patches
from ..state.shared_state import resolve_graded_comparison
from ..state.task_registry import Task
from .warm_replay_stack import (
    replayed_patch_refs as _replayed_patch_refs,
    warm_patch_trees as _warm_patch_trees,
    commit_warm_replay_overlays,
    committing_overlays,
    restore_recipe_trees,
    warm_replay_source_layer,
)
from .warm_replay_eval import read_replay_eval

log = _logging.getLogger(__name__)


def _gain_pct(after: float, before: float) -> float:
    return (after / before - 1.0) * 100.0


class _WarmReplaySettleMixin:
    """Settles a finished warm replay: grades it, judges its accuracy and adopts it onto the stack or rolls it back."""

    def _resolve_promoted_recipe_checkout(
        self, result: dict[str, Any], task: "Task | None"
    ) -> tuple[bool, dict[str, Any]]:
        """Validate every already-patched checkout selected by warm replay."""
        params = task.params if task is not None else {}
        if not params.get("required_patch_timeline") or not params.get("patches"):
            return True, {"status": "not_required"}

        promoted: list[str] = []
        for tree in _warm_patch_trees(result, {}):
            target = str(tree.get("root") or "").strip()
            pre_sha = str(tree.get("pre_sha") or "").strip()
            manifest = tree.get("snapshot_manifest")
            # Promotion needs a way to unwind the tree if the replay is rejected, not a sha. A git checkout
            # answers with a pre_sha and a snapshot manifest; a pip-installed framework -- the tree a KB recipe
            # is usually measured on -- answers with the backups nogit wrote as it applied. A tree this round
            # never wrote to owes neither, and promotion is reached only once every required overlay applied, so
            # it already carries what is being promoted. A row without the flag is read as written-to.
            backups = list(tree.get("nogit_backups") or [])
            snapshotted = bool(pre_sha) and isinstance(manifest, Mapping)
            mutated = bool(tree.get("mutated", True))
            if not target or (mutated and not (snapshotted or backups)):
                return False, {
                    "status": "failed",
                    "failure": "validated_recipe_checkout_incomplete",
                    "target_repo": target,
                }
            try:
                target_path = Path(target).resolve(strict=True)
                manifest_target = (
                    Path(str(manifest.get("repo_path") or "")).resolve(strict=True) if snapshotted else target_path
                )
            except (OSError, ValueError) as exc:
                return False, {"status": "failed", "failure": f"validated_recipe_checkout_invalid:{type(exc).__name__}"}
            if target_path != manifest_target:
                return False, {
                    "status": "failed",
                    "failure": "validated_recipe_checkout_manifest_mismatch",
                    "target_repo": str(target_path),
                }
            promoted.append(str(target_path))
        self.shared_state.warm_replay_pending = committing_overlays(self.shared_state.warm_replay_pending or {})
        self.shared_state.save(self.session_dir)
        if failed := commit_warm_replay_overlays(result):
            failure = "the replayed overlays cannot be committed to the framework tree: " + "; ".join(failed)
            return False, {"status": "failed", "failure": failure}
        return True, {"status": "promoted", "target_repo": promoted[0], "target_repos": promoted}

    def rollback_combined_warm(self, result: dict[str, Any], task: "Task | None") -> dict[str, Any]:
        """Restore both Recipe and Kernel halves of a combined replay."""
        pending = self.shared_state.warm_replay_pending
        restores = restore_recipe_trees(result, pending)
        params = task.params if task is not None else {}

        def _kernel_rows(key: str, pending_key: str) -> list[Any]:
            return list(result.get(key) or params.get(key) or pending.get(pending_key) or [])

        restores.append(
            revert_warm_kernel_patches(
                _kernel_rows("warm_kernel_apply_results", "kernel_apply_results"),
                _kernel_rows("warm_kernel_snapshots", "kernel_snapshots"),
            )
        )
        errors = [
            error
            for restore in restores
            if not restore.get("ok")
            for error in (restore.get("errors") or ["unknown_restore_failure"])
        ]
        if errors:
            self.shared_state.warm_replay_pending = {
                **dict(pending),
                "status": "rollback_failed",
                "rollback_errors": errors,
            }
        else:
            self.shared_state.warm_replay_pending = {}
        return {"ok": not errors, "errors": errors}

    def _book_combined_kernel_keep(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> dict[str, Any]:
        """Record the kernel half as accepted by the one Recipe replay check."""
        params = (task.params if task is not None else {}) or {}
        plan = [dict(entry) for entry in (params.get("warm_kernel_plan") or []) if isinstance(entry, dict)]
        for entry in plan:
            entry["decision"] = "KEEP"
            entry["gain_pct"] = result.get("combined_gain_pct")
        outcome = {
            "status": "kept" if plan else "empty",
            "total": len(plan),
            "kept": len(plan),
            "reverted": 0,
            "validation": "combined_recipe_kernel",
        }
        self.shared_state.warm_kernel_kb_plan = plan
        self._set_warm_kernel_outcome(outcome)
        return outcome

    def _require_combined_warm_rollback(
        self,
        result: dict[str, Any],
        task: "Task | None",
        outcome: dict[str, Any],
        recorder: Any = None,
    ) -> bool:
        """Roll back a rejected replay, settling it as ``rollback_failed`` when a tree did not come back.

        The rollback is recorded here rather than at each of the branches that
        unwind a rejected replay: what a reader needs is whether the trees came
        back. Returns ``True`` when every tree was restored.
        """
        rollback = self.rollback_combined_warm(result, task)
        if recorder is not None:
            recorder.record_rollback(ok=rollback.get("ok"), errors=rollback.get("errors"))
        if rollback.get("ok"):
            return True
        outcome["status"] = "rollback_failed"
        outcome["reason"] = "combined warm replay rollback failed"
        outcome["rollback"] = rollback
        kernel_outcome = {
            "status": "rollback_failed",
            "validation": "combined_recipe_kernel",
            "errors": list(rollback.get("errors") or []),
        }
        outcome["kernel"] = kernel_outcome
        self.shared_state.set_stop_reason("warm_replay_rollback_failed")
        self._conclude_warm_replay(outcome, Verdict.FAILED)
        return False

    def _judge_warm_replay_accuracy(
        self, result: dict, task: "Task | None", outcome: dict, recorder: Any = None
    ) -> Verdict | None:
        """Rule on whether a replayed config may be promoted on accuracy grounds.

        Every replay is judged, not just the ones touching a knob known to be
        risky: a KB recipe is evidence from another session, so reproducing its
        throughput says nothing about whether it still computes correctly here.
        ``eval_ran`` separates the two ways ``replay_accuracy`` can be absent:
        a score of 0.0 means the model answered nothing, while no score at all
        means no evidence either way.

        Returns ``None`` when promotion may proceed; otherwise the verdict the
        caller must stop on (the rollback and outcome have already been recorded).
        """
        from hyperloom.common.gain_math import (
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            accuracy_meets_floor,
            accuracy_passed,
        )

        baseline = AccuracyReference.of_baseline(self.shared_state)
        baseline_accuracy = baseline.score
        replay_eval = read_replay_eval(result, baseline)
        measured, eval_ran, eval_error = replay_eval.accuracy, replay_eval.ran, replay_eval.error

        outcome["eval_ran"] = bool(eval_ran)
        outcome["eval_error"] = eval_error or None
        outcome["replay_accuracy"] = float(measured) if measured is not None else None
        reference = baseline_accuracy if baseline_accuracy > 0 else None
        outcome["baseline_accuracy"] = reference

        if measured is None:
            # A measurement that failed is not evidence the config broke the model, so it must not stop the run: the
            # replay is admitted and the reason it could not be judged is recorded instead. Ran but could not rule is
            # a verdict of its own: recording a pass here would claim evidence there is none.
            log.warning(
                "warm-replay admitted without an accuracy verdict (eval_ran=%s, baseline %.4f): %s",
                eval_ran,
                baseline_accuracy,
                eval_error or "no reason recorded",
            )
            passed, threshold = None, reference
            reason = eval_error or "no accuracy verdict"
        elif reference is not None:
            passed, threshold = accuracy_passed(reference, float(measured)), reference
            reason = (
                "no regression against the baseline reference"
                if passed
                else f"accuracy regression on the replayed config (baseline {reference:.4f}, replay {measured:.4f})"
            )
        else:
            passed = accuracy_meets_floor(measured, DEFAULT_ENABLEMENT_ACCURACY_FLOOR)
            threshold = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
            reason = (
                "clears the absolute floor with no baseline to compare against"
                if passed
                else "accuracy below absolute floor on the replayed config "
                f"(replay {measured:.4f}, "
                f"floor {DEFAULT_ENABLEMENT_ACCURACY_FLOOR:.2f})"
            )
        if recorder is not None:
            recorder.record_gate(GATE_ACCURACY, passed=passed, reason=reason, observed=measured, threshold=threshold)
        if passed is not False:
            return None
        if not self._require_combined_warm_rollback(result, task, outcome, recorder):
            return Verdict.FAILED
        outcome["status"] = "accuracy_failed"
        outcome["reason"] = reason
        log.info("warm-replay REJECTED on accuracy: %s", reason)
        return self._conclude_warm_replay(outcome, Verdict.REVERTED)

    def _record_warm_patch_apply_items(self, recorder: Any, result: Mapping[str, Any]) -> None:
        """State which of the recipe's code patches landed.

        The executor reports one status per patch it reached, keyed by the
        timeline position the recipe gave it. A required timeline stops at its
        first failure, so the patches behind that one carry no status and
        record nothing.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import APPLY_PATCH

        patch_result = result.get("warm_patch_result")
        if not isinstance(patch_result, Mapping):
            return
        for position, row in enumerate(patch_result.get("patches") or []):
            if not isinstance(row, Mapping):
                continue
            recorder.record_apply_item(
                WarmReplayApplyItem(
                    str(row.get("patch_ref") or ""),
                    kind=APPLY_PATCH,
                    position=row.get("timeline_index", position),
                    applied=str(row.get("status") or "") != "failed",
                    reason=str(row.get("reason") or row.get("detail") or ""),
                    target=str(row.get("target_repo") or ""),
                )
            )

    def promote_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> Verdict:
        """Interpret a combined Recipe+Kernel ``replay_warm_recipe`` result.

        Closes the replay's timeline event on whichever outcome the settling
        below persisted. Done here, around the whole arc, rather than at each
        of the ten branches that can end it: a branch that forgot to close
        would leave a settled replay reading as still in flight.

        Returns:
            The settlement verdict the settling ruled.
        """
        recorder = self._warm_replay_timeline(task)
        try:
            verdict = self._settle_warm_replay(result, task=task, recorder=recorder)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        if recorder is not None:
            recorder.finish(self.shared_state.warm_replay_outcome)
        return verdict

    def _conclude_warm_replay(self, outcome: dict, verdict: Verdict, *, clear_pending: bool = False) -> Verdict:
        """Persist the settled outcome, the one way every settlement ends."""
        state = self.shared_state
        if clear_pending:
            state.warm_replay_pending = {}
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)
        return verdict

    def _settle_warm_replay(self, result: dict, *, task: "Task | None", recorder: Any) -> Verdict:
        """Settle a combined Recipe+Kernel replay onto an outcome.

        Measured uplift promotes the warm config onto ``optimization_stack`` and
        ``current_best``. Failures (including a failed required patch timeline)
        roll back both halves fail-closed, set an outcome status, and never
        propagate. ``task`` may be ``None`` (degraded path). Each gate writes
        its own verdict onto ``recorder`` as it rules, which is what lets a
        reader see why the arc ended.
        """
        state = self.shared_state
        outcome = dict(state.warm_replay_outcome or {})
        # Stamped once up front so every terminal branch below carries it; each of them re-persists ``outcome`` before
        # returning.
        outcome["settled_at"] = now_iso(timespec="seconds")
        if not isinstance(result, dict):
            if not self._require_combined_warm_rollback({}, task, outcome, recorder):
                return Verdict.FAILED
            outcome["status"] = "failed"
            outcome["reason"] = "non_dict_result"
            return self._conclude_warm_replay(outcome, Verdict.FAILED)
        status = str(result.get("status") or "")
        if status != "succeeded":
            if not self._require_combined_warm_rollback(result, task, outcome, recorder):
                return Verdict.FAILED
            outcome["status"] = "failed"
            outcome["error_class"] = str(result.get("error_class") or "")
            outcome["reason"] = str(result.get("error") or result.get("reason") or "")[:240]
            log.info("warm-replay failed (status=%s, error_class=%s)", status, outcome.get("error_class"))
            return self._conclude_warm_replay(outcome, Verdict.FAILED)
        return self._grade_warm_replay(result, task, outcome, recorder)

    def _grade_warm_replay(self, result: dict, task: "Task | None", outcome: dict, recorder: Any) -> Verdict:
        state = self.shared_state
        # ``tput`` (output_throughput) is the HOT measure round and the comparison value; the warmup round is retained
        # for audit only.
        tput = to_float(result.get("output_throughput"), 0.0)
        # Use the baseline_tput captured at enqueue time (fall back to live state).
        anchor = task.params.get("baseline_tput_anchor") if task is not None else None
        baseline_tput = to_float(anchor, 0.0)
        if baseline_tput <= 0:
            baseline_tput = to_float(state.baseline_tput, 0.0)
        if tput <= 0 or baseline_tput <= 0:
            if recorder is not None:
                recorder.record_gate(
                    GATE_TPUT_VALID,
                    passed=False,
                    reason=f"invalid_tput tput={tput} baseline={baseline_tput}",
                    observed=tput,
                    threshold=baseline_tput,
                )
            if not self._require_combined_warm_rollback(result, task, outcome, recorder):
                return Verdict.FAILED
            outcome["status"] = "failed"
            outcome["reason"] = f"invalid_tput tput={tput} baseline={baseline_tput}"
            return self._conclude_warm_replay(outcome, Verdict.FAILED)
        if recorder is not None:
            recorder.record_gate(
                GATE_TPUT_VALID,
                passed=True,
                reason="the replay and its anchor both measured",
                observed=tput,
                threshold=baseline_tput,
            )
        # Recorded before the gates below, not after the KEEP ruling: a replay
        # rejected on quality or accuracy still produced a real measurement, and
        # reading what it scored is how a gate rejection is told apart from a
        # replay that never got that far.
        outcome["actual_gain_pct"] = round(_gain_pct(tput, baseline_tput), 3)
        outcome["throughput_after"] = tput
        if recorder is not None:
            # The anchor goes on record here, where it is used: back-solving it
            # from the gain is exact only until a re-baseline moves the number
            # the replay was never judged against.
            self._record_replay_measurement(recorder, result, outcome, baseline_tput)
        # warm_replay is an optimization candidate, so it must clear the
        # image-quality gate against the baseline reference before promotion.
        # ``require=False`` keeps a missing/skipped gate non-blocking.
        from ..actions.executors._accuracy_gate import quality_gate_passed

        qg = result.get("quality_gate")
        if qg is not None and not quality_gate_passed(qg, require=False):
            if recorder is not None:
                recorder.record_gate(
                    GATE_QUALITY, passed=False, reason="image-quality gate failed vs baseline reference"
                )
            if not self._require_combined_warm_rollback(result, task, outcome, recorder):
                return Verdict.FAILED
            outcome["status"] = "quality_failed"
            outcome["reason"] = "image-quality gate failed vs baseline reference"
            outcome["quality_gate"] = qg
            log.info("warm-replay REJECTED by quality gate: %s", qg)
            return self._conclude_warm_replay(outcome, Verdict.REVERTED)
        # A replayed config lands on ``current_best``, so every later
        # measurement in the session is taken against it. Promoting on
        # throughput alone selects for garbage: breaking the numerics is itself
        # a large throughput win.
        if recorder is not None and qg is not None:
            # Only recorded when a gate existed to rule, which is how a reader
            # tells "passed quality" from "quality did not apply".
            recorder.record_gate(GATE_QUALITY, passed=True, reason="no quality regression vs baseline reference")
        if (accuracy_verdict := self._judge_warm_replay_accuracy(result, task, outcome, recorder)) is not None:
            return accuracy_verdict
        return self._rule_on_warm_replay_keep(result, task, outcome, recorder, baseline_tput)

    def _record_replay_measurement(self, recorder: Any, result: dict, outcome: dict, baseline_tput: float) -> None:
        recorder.record_measurement(
            WarmReplayMeasurement(
                before_tput=baseline_tput,
                after_tput=outcome["throughput_after"],
                gain_pct=_gain_pct(outcome["throughput_after"], baseline_tput),
                hot_tput=outcome["throughput_after"],
                cold_tput=to_float(result.get("warmup_round_tput"), 0.0) or None,
                accuracy=outcome.get("replay_accuracy"),
                baseline_accuracy=outcome.get("baseline_accuracy"),
                eval_ran=outcome.get("eval_ran"),
            )
        )

    def _rule_on_warm_replay_keep(
        self, result: dict, task: "Task | None", outcome: dict, recorder: Any, baseline_tput: float
    ) -> Verdict:
        state = self.shared_state
        measured_gain = _gain_pct(outcome["throughput_after"], baseline_tput)
        if recorder is not None:
            # Restated now that the gate has read the scores onto the outcome,
            # so the score sits beside the reference it was judged against.
            self._record_replay_measurement(recorder, result, outcome, baseline_tput)
        result["combined_gain_pct"] = outcome["actual_gain_pct"]
        params = task.params if task is not None else {}
        if recorder is not None:
            # Recorded before the keep ruling, so a replay that measured and
            # lost still states what lost.
            recorder.record_applied(
                extra_server_args=str(params.get("extra_server_args") or ""),
                extra_envs=dict(params.get("extra_envs") or {}),
            )
            self._record_warm_patch_apply_items(recorder, result)
        keep_threshold = _phase_state.resolve_keep_threshold(state)
        graded = resolve_graded_comparison(
            state, result, keep_threshold_pct=keep_threshold, anchor=GradingAnchor(tput=baseline_tput)
        )
        outcome["keep_threshold_pct"] = keep_threshold
        if graded.veto_reason:
            # The grading applies the session's constraints (latency, power), so the drift branch below rolls the
            # replay back rather than leaving a config a budget refused promoted on disk.
            outcome["constraint_veto"] = graded.veto_reason
        reproduced = graded.verdict == VERDICT_KEEP
        if recorder is not None:
            recorder.record_gate(
                GATE_KEEP_THRESHOLD,
                passed=reproduced,
                reason=f"graded on {graded.objective}: {graded.refused_by or graded.verdict}",
                observed=measured_gain,
                threshold=keep_threshold,
            )
        if reproduced:
            return self._adopt_reproduced_warm_replay(result, task, outcome, recorder)
        return self._reject_warm_replay_as_drift(
            result,
            task,
            outcome,
            recorder,
            reason=graded.veto_reason
            or graded.refused_by
            or f"measured {measured_gain:+.2f}% below keep threshold {keep_threshold:+.2f}%",
        )

    def _adopt_reproduced_warm_replay(self, result: dict, task: "Task | None", outcome: dict, recorder: Any) -> Verdict:
        params = task.params if task is not None else {}
        promoted, promotion = self._resolve_promoted_recipe_checkout(result, task)
        if not promoted:
            failure = str(promotion.get("failure") or "validated Recipe checkout promotion failed")
            if recorder is not None:
                recorder.record_gate(GATE_PROMOTION, passed=False, reason=failure)
            if not self._require_combined_warm_rollback(result, task, outcome, recorder):
                return Verdict.FAILED
            outcome["status"] = "promotion_failed"
            outcome["reason"] = failure
            outcome["recipe_checkout_promotion"] = promotion
            outcome["kernel"] = {
                "status": "reverted",
                "reason": "recipe_checkout_promotion_failed",
                "validation": "combined_recipe_kernel",
            }
            return self._conclude_warm_replay(outcome, Verdict.FAILED)
        if recorder is not None:
            recorder.record_gate(
                GATE_PROMOTION, passed=True, reason="every patched checkout the replay measured on was promoted"
            )
        warm_args = str(params.get("extra_server_args") or "").strip()
        warm_envs = dict(params.get("extra_envs") or {})
        replayed_patch_refs = _replayed_patch_refs(result, params)
        if not warm_args and not warm_envs and not replayed_patch_refs and not params.get("warm_kernel_plan"):
            if recorder is not None:
                recorder.record_gate(
                    GATE_PARAMS_PRESENT,
                    passed=False,
                    reason="task params carry no args, no envs, no patch and no kernel plan to replay",
                )
            outcome["status"] = "reproduced_but_no_params"
            outcome["reason"] = "task.params missing extra_server_args/extra_envs and no warm patch was applied"
            log.warning(
                "warm-replay measured +%.2f%% but cannot push stack (task=%r has no warm args/envs)",
                outcome["actual_gain_pct"],
                task,
            )
            return self._conclude_warm_replay(outcome, Verdict.REFUSED)
        if recorder is not None:
            recorder.record_gate(
                GATE_PARAMS_PRESENT, passed=True, reason="the replay carries params that can be pushed onto the stack"
            )
        return self._push_warm_replay_onto_stack(result, task, outcome, recorder, promotion)

    def _warm_replay_stack_entry(
        self,
        result: dict,
        params: Mapping[str, Any],
        outcome: dict,
        promoted_checkout: str,
        replayed_patch_refs: list[str],
    ) -> dict[str, Any]:
        cold_round_tput = to_float(result.get("warmup_round_tput"), 0.0)
        entry_extra: dict[str, Any] = {
            "gain_pct": outcome["actual_gain_pct"],
            "hot_tput": float(outcome["throughput_after"]),
            "cold_tput": cold_round_tput or None,
            # The score this promotion was judged on, recorded alongside the throughput it was judged with.
            "accuracy": outcome.get("replay_accuracy"),
            # source_tier records the warm-recipe tier for breakdown attribution.
            "source_tier": outcome.get("warm_recipe_tier", ""),
            "source_confidence": outcome.get("warm_recipe_conf", 0.0),
        }
        if promoted_checkout:
            entry_extra["framework_source_root"] = promoted_checkout
        kernel_plan = [entry for entry in (params.get("warm_kernel_plan") or []) if isinstance(entry, dict)]
        if kernel_plan:
            entry_extra["kernel_replay"] = {
                "validation": "combined_recipe_kernel",
                "count": len(kernel_plan),
                "columns": sorted({str(entry.get("column") or "") for entry in kernel_plan}),
            }
        patch_result = result.get("warm_patch_result")
        if isinstance(patch_result, dict):
            entry_extra["recipe_patch_statuses"] = list(patch_result.get("patches") or [])
        if replayed_patch_refs:
            entry_extra["replayed_patch_refs"] = replayed_patch_refs
        return {**entry_extra, **warm_replay_source_layer(result, self.session_dir, self.shared_state.framework)}

    def _warm_replay_candidate(self, result: dict, task: "Task | None", params: Mapping[str, Any]) -> dict[str, Any]:
        warm_args = str(params.get("extra_server_args") or "").strip()
        warm_envs = dict(params.get("extra_envs") or {})
        recipe_args = str(
            params["recipe_extra_server_args"] if "recipe_extra_server_args" in params else warm_args
        ).strip()
        return {
            "name": "warm_replay",
            **graded_axes_of(result),
            # The latency and power budgets grade on these and fail closed without them.
            **constraint_fields(result),
            "candidate_extra_server_args": warm_args,
            "candidate_extra_envs": warm_envs,
            "recipe_delta": {
                "extra_server_args": recipe_args,
                "extra_envs": dict(params["recipe_extra_envs"] if "recipe_extra_envs" in params else warm_envs),
                "remove_args": [],
                "unset_envs": [],
                "args_mode": "replace",
            },
            "extra_envs": warm_envs,
            "source_phase": "PRELUDE",
            "task_id": str(getattr(task, "task_id", "") or ""),
            "workspace": str(result.get("workspace") or ""),
        }

    def _push_warm_replay_onto_stack(
        self, result: dict, task: "Task | None", outcome: dict, recorder: Any, promotion: dict[str, Any]
    ) -> Verdict:
        state = self.shared_state
        params = task.params if task is not None else {}
        warm_args = str(params.get("extra_server_args") or "").strip()
        replayed_patch_refs = _replayed_patch_refs(result, params)
        promoted_checkout = promotion.get("target_repo", "")
        entry_extra = self._warm_replay_stack_entry(result, params, outcome, promoted_checkout, replayed_patch_refs)
        # Resume safety: do not clobber existing stack entries.
        state.optimization_stack = list(state.optimization_stack)
        # A prior promote owns the outcome; re-running would re-journal it.
        already_pushed = any(
            isinstance(e, dict) and e.get("action") == "replay_warm_recipe" for e in state.optimization_stack
        )
        if not already_pushed and not self._coord.writeback.lift_to_current_best(
            "replay_warm_recipe",
            float(outcome["throughput_after"]),
            self._warm_replay_candidate(result, task, params),
            entry_extra=entry_extra,
        ):
            # Cleared the keep threshold against the baseline yet did not
            # beat the adopted configuration on its graded objective.
            return self._reject_warm_replay_as_drift(
                result,
                task,
                outcome,
                recorder,
                reason=f"measured {outcome['actual_gain_pct']:+.2f}% over baseline but current_best refused the lift",
            )
        if promoted_checkout:
            outcome["active_framework_root"] = promoted_checkout
        outcome["status"] = "reproduced"
        outcome.pop("replayed_patch_refs", None)
        if replayed_patch_refs:
            outcome["replayed_patch_refs"] = replayed_patch_refs
        kernel_outcome = self._book_combined_kernel_keep(result, task)
        outcome["kernel"] = dict(kernel_outcome)
        if recorder is not None:
            recorder.record_applied(kernel=kernel_outcome)
        if already_pushed:
            log.info(
                "warm-replay promote: stack already carries the entry; "
                "skipping duplicate push (likely resume mid-promote)"
            )
            return self._conclude_warm_replay(outcome, Verdict.ADOPTED, clear_pending=True)
        if recorder is not None:
            recorder.record_promotion(
                promoted_checkout=promoted_checkout, replayed_patch_refs=replayed_patch_refs, stack_entry=entry_extra
            )
        # Publish the reproduced verdict now that the stack entry exists but
        # before the cumulative update (the one step below that can raise and
        # is swallowed by the caller). Placed after the lift on purpose --
        # if the lift raises, the outcome stays in_flight and both the stack
        # and the post-ruling mirror agree there is nothing adopted.
        state.warm_replay_outcome = outcome
        self._coord.writeback.validate(float(outcome["throughput_after"]), result, source="warm_replay")
        log.info(
            "warm-replay REPRODUCED: measured=+%.2f%% (expected=+%.2f%%); pushed warm_replay onto stack",
            outcome["actual_gain_pct"],
            to_float(outcome.get("expected_gain_pct"), 0.0),
        )
        self._journal_warm_replay(result, task, outcome, warm_args)
        return self._conclude_warm_replay(outcome, Verdict.ADOPTED, clear_pending=True)

    def _journal_warm_replay(self, result: dict, task: "Task | None", outcome: dict, warm_args: str) -> None:
        state = self.shared_state
        try:
            journal = self._coord.recipe_journal.ensure_journal()
            journal.append_entry(
                JournalEntry(
                    phase=(state.phase or "PRELUDE").upper() or "PRELUDE",
                    iter=int(state.tick or 0),
                    kind=KIND_OTHER,
                    change=f"warm_replay({outcome.get('warm_recipe_tier', '?')}): {warm_args}",
                    outcome=derive_journal_outcome(Verdict.ADOPTED, result),
                    gain_pct=outcome["actual_gain_pct"],
                    throughput_after=outcome["throughput_after"],
                    task_id=str(task.task_id if task is not None else ""),
                    tick=int(state.tick or 0),
                )
            )
        except (OSError, TypeError, ValueError):
            log.exception("warm-replay journal append failed")

    def _reject_warm_replay_as_drift(
        self,
        result: dict,
        task: "Task | None",
        outcome: dict,
        recorder: Any,
        *,
        reason: str,
    ) -> Verdict:
        """Roll back a measured replay that is not adopted and settle it as ``drift``.

        A rollback that failed settles as ``rollback_failed`` instead.
        """
        if not self._require_combined_warm_rollback(result, task, outcome, recorder):
            return Verdict.FAILED
        kernel_plan = (task.params or {}).get("warm_kernel_plan") if task is not None else []
        kernel_outcome = {
            "status": "reverted",
            "total": len(kernel_plan or []),
            "kept": 0,
            "reverted": len(kernel_plan or []),
            "validation": "combined_recipe_kernel",
        }
        outcome["kernel"] = dict(kernel_outcome)
        if recorder is not None:
            recorder.record_applied(kernel=kernel_outcome)
        outcome["status"] = "drift"
        outcome["reason"] = reason
        log.info("warm-replay DRIFT: %s", reason)
        return self._conclude_warm_replay(outcome, Verdict.REVERTED, clear_pending=True)
