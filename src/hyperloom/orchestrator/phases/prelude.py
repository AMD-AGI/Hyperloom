# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PRELUDE phase handler: warm-recipe replay (KB best_config auto-apply) and
the initial baseline/roofline internal-analysis task enqueue."""

from __future__ import annotations
import logging as _logging
import os
from datetime import datetime, timezone
from typing import Any
from . import machine_state as _phase_state
from ..state.optimization_journal import (
    JournalEntry,
)
from ..state.task_registry import Task
from ..loop.coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
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
from ..loop.coordinator import (
    _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class PreludePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    def _internal_analysis_kind(self) -> str:
        """Pick the kind for the next Coordinator-internal analysis task: roofline (composite) when enable_roofline else profile. Both absent from PHASE_LLM_PROPOSABLE_ACTIONS (PolicyGate R1 denies LLM proposals).

        Returns:
            ``"roofline"`` when roofline is enabled, else ``"profile"``.
        """
        return (
            "roofline"
            if bool(
                getattr(self.shared_state, "enable_roofline", True),
            )
            else "profile"
        )

    def _warm_recipe_proven_items(self) -> list[dict[str, str]]:
        """Summarise warm-start ``what_worked`` items the scout can skip ({name, source}); fail-soft.

        Returns:
            A list of ``{"name", "source"}`` dicts for proven warm-start items;
            empty when no warm recipe is present.
        """
        state = self.shared_state
        warm = getattr(state, "warm_start_recipe", None) or {}
        if not isinstance(warm, dict) or not warm:
            return []
        recipe = warm.get("recipe") or {}
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        what_worked = recipe_attrs.get("what_worked") or []
        if not isinstance(what_worked, list):
            return []
        out: list[dict[str, str]] = []
        for row in what_worked:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append({"name": name, "source": str(row.get("source") or "").strip()})
        return out

    def _inject_warm_recipe_history_into_ledger(self) -> int:
        """Pre-fill ``explore_search.rejected`` with the warm recipe's ``what_failed`` rows (fingerprinted so the dedup gate denies re-tests). Idempotent via warm_history_injected; returns rows added.

        Returns:
            The number of new rejected rows injected into the explore ledger.
        """
        state = self.shared_state
        if getattr(state, "warm_history_injected", False):
            return 0
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_history_injected = True
            return 0
        recipe = warm.get("recipe") or {}
        # v2 arbor keeps what_failed top-level; v1 nested under attrs. Fall back to the recipe itself.
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        what_failed = recipe_attrs.get("what_failed") or []
        if not isinstance(what_failed, list) or not what_failed:
            state.warm_history_injected = True
            return 0

        from ..actions.executors._canonical_fingerprint import (
            canonical_fingerprint,
        )

        es_raw = getattr(state, "explore_search", None) or {}
        es = dict(es_raw) if isinstance(es_raw, dict) else {}
        rejected = list(es.get("rejected") or [])
        existing_fps = {str(r.get("fingerprint") or "") for r in rejected if isinstance(r, dict)}
        existing_fps.discard("")
        added = 0
        tier = str((warm or {}).get("tier") or "")
        for row in what_failed:
            if not isinstance(row, dict):
                continue
            args = str(
                row.get("extra_sglang_args") or row.get("args") or "",
            ).strip()
            envs = row.get("extra_envs") or row.get("envs") or {}
            if not isinstance(envs, dict):
                envs = {}
            if not args and not envs:
                continue
            fp = canonical_fingerprint(args, envs)
            if fp in existing_fps:
                continue
            existing_fps.add(fp)
            rejected.append(
                {
                    "name": str(row.get("name") or "")[:120],
                    "fingerprint": fp,
                    "reason": "warm_recipe_what_failed",
                    "extra_sglang_args": args,
                    "extra_envs": dict(envs),
                    "source": "warm_start_recipe",
                    "source_tier": tier,
                    # Recipe-carried fields preserved for forensics; not used by the dedup gate.
                    "gain_pct": row.get("gain_pct"),
                    "error_class": row.get("error_class") or row.get("reason"),
                }
            )
            added += 1

        if added:
            es["rejected"] = rejected
            state.explore_search = es
            log.info(
                "warm-recipe history: injected %d what_failed rows into explore_search.rejected (tier=%s)",
                added,
                tier,
            )
        state.warm_history_injected = True
        return added

    def _filter_warm_patches_with_kg(
        self,
        patches: list,
        advisory_blocked: list,
        state: Any,
    ) -> list:
        """Filter replay patches using KG advisory blocks, expiry, conflicts.

        Removes patches that are (a) advisory-blocked at/above the
        configurable confidence threshold, (b) flagged ``expired`` by the
        warm-start validity check, or (c) in a ``CONFLICTS_WITH`` relation
        with another patch in the set. Best-effort: any failure returns the
        input patches unchanged so replay never breaks on a KG hiccup.

        Args:
            patches: The candidate replay patches from ``recommended_replay``.
            advisory_blocked: The ``advisory_blocked_patches`` list from the
                warm-start context.
            state: The live SharedState (for hardware/framework conditions).

        Returns:
            The filtered patch list.
        """
        if not patches:
            return patches
        try:
            threshold = float(os.environ.get("WARM_REPLAY_ADVISORY_CONFIDENCE", "0.75"))
        except (TypeError, ValueError):
            threshold = 0.75

        def _norm(value: Any) -> str:
            return str(value or "").strip().replace(" ", "_").replace("/", "_").lower()

        try:
            advisory_drop = {
                _norm(ab.get("patch_file"))
                for ab in (advisory_blocked or [])
                if isinstance(ab, dict) and float(ab.get("confidence") or 0.0) >= threshold
            }
            kept = [
                p
                for p in patches
                if isinstance(p, dict)
                and not p.get("expired")
                and _norm(p.get("patch_file")) not in advisory_drop
            ]
            for p in patches:
                if isinstance(p, dict) and _norm(p.get("patch_file")) in advisory_drop:
                    log.info("warm-replay advisory block (conf>=%.2f): %s", threshold, p.get("patch_file"))

            if len(kept) >= 2:
                from inference_optimizer.recipe_kb.kg_client import get_kg_client

                kg = get_kg_client()
                if kg is not None and kg.is_available():
                    knobs = [str(p.get("patch_file") or "") for p in kept]
                    conflicts = kg.find_conflicts_safe(
                        knobs=knobs,
                        hardware=str(getattr(state, "gpu_type", "") or getattr(state, "hardware", "") or ""),
                        framework=str(getattr(state, "framework", "") or ""),
                    )
                    drop = {_norm(c.get("knob")) for c in conflicts}
                    if drop:
                        for c in conflicts:
                            log.info(
                                "warm-replay conflict: %s conflicts_with %s",
                                c.get("knob"),
                                c.get("conflicts_with"),
                            )
                        kept = [p for p in kept if _norm(p.get("patch_file")) not in drop]
            return kept
        except Exception as exc:  # noqa: BLE001 - filtering is advisory only
            log.warning("warm-replay KG patch filtering degraded: %s", exc)
            return patches

    async def _maybe_enqueue_warm_replay(
        self,
        *,
        baseline_tput: float,
    ) -> "Task | None":
        """Enqueue a one-shot ``replay_warm_recipe`` task for a high-confidence T0 prior.

        Skips on --no-warm-replay/resume/low-confidence/empty best_config; otherwise mints an internal
        task running the baseline workload contract with the KB config applied. Idempotent via warm-replay-prelude.

        Args:
            baseline_tput: The baseline throughput captured at enqueue time,
                carried forward as the replay's comparison anchor.

        Returns:
            The created (or existing) ``replay_warm_recipe`` task, or ``None``
            when the replay is skipped.
        """
        state = self.shared_state
        if not getattr(self, "_warm_replay_enabled", True):
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }
            # Flip the one-shot guard even on disabled-skip so a resume without --no-warm-replay can't
            # retroactively trigger a replay against the operator's original intent.
            state.warm_replay_attempted = True
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued/ran the replay.
            return None
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "no_warm_start_recipe",
            }
            state.warm_replay_attempted = True
            return None
        # tier/conf stamped at T0 by find_recipe_with_fallback.
        tier = str(warm.get("tier") or "").strip()
        try:
            conf = float(warm.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        min_conf = float(
            getattr(self, "_warm_replay_min_confidence", _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE)
            or _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE
        )
        recipe = warm.get("recipe") or {}
        if not isinstance(recipe, dict):
            recipe = {}
        # best_config/sessions may be top-level or nested under attrs; fall back to recipe itself.
        recipe_attrs = recipe.get("attrs") or recipe
        # Resolve the replay config via config-donor decoupling: prefer the
        # WarmStartContext's ready-to-replay champion — whose config may be
        # BORROWED from a same-architecture sibling when this recipe's own
        # best_config is empty — and gate on the donor's TRANSFER confidence
        # rather than the identity-match confidence. Fall back to the identity
        # recipe's own best_config for legacy state.json without a context.
        wsc = getattr(state, "warm_start_context", None) or {}
        replay = wsc.get("recommended_replay") if isinstance(wsc, dict) else {}
        replay = replay if isinstance(replay, dict) else {}
        rep_args = str(replay.get("extra_server_args") or "").strip()
        rep_envs = replay.get("extra_envs") if isinstance(replay.get("extra_envs"), dict) else {}
        if rep_args or rep_envs:
            bc_args = rep_args
            bc_envs = dict(rep_envs)
            # Donor transfer confidence (self-donor == identity confidence).
            replay_conf = float(replay.get("config_confidence") or conf or 0.0)
            config_source = str(replay.get("config_source") or "")
            config_tier = str(replay.get("config_tier") or "self")
            donor_expected_gain = float(replay.get("expected_gain_pct") or 0.0)
        else:
            best_config = recipe_attrs.get("best_config") or {}
            if not isinstance(best_config, dict):
                best_config = {}
            # Read canonical extra_server_args FIRST, then legacy extra_sglang_args/args.
            bc_args = str(
                best_config.get("extra_server_args")
                or best_config.get("extra_sglang_args")
                or best_config.get("args")
                or ""
            ).strip()
            bc_envs = best_config.get("extra_envs") or best_config.get("envs") or {}
            if not isinstance(bc_envs, dict):
                bc_envs = {}
            replay_conf = float(conf or 0.0)
            config_source = str(recipe.get("canonical_id") or "")
            config_tier = "self"
            donor_expected_gain = 0.0
        # Gate on the replay (config-transfer) confidence: a borrowed
        # same_arch_class config (0.95) clears the 0.7 bar; a cross-version
        # (0.5) donor stays advisory-only and is NOT auto-replayed.
        if replay_conf < min_conf:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": f"confidence_below_threshold ({replay_conf:.2f} < {min_conf:.2f})",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
                "config_donor_tier": config_tier,
                "config_source": config_source,
            }
            state.warm_replay_attempted = True
            return None
        # Extract code patches from warm_start_context (populated by T0).
        wsc_patches = (wsc.get("recommended_replay") or {}).get("patches") or [] if isinstance(wsc, dict) else []
        wsc_blocked = wsc.get("blocked_patches") or [] if isinstance(wsc, dict) else []
        wsc_advisory = wsc.get("advisory_blocked_patches") or [] if isinstance(wsc, dict) else []
        # KG-driven filtering: drop high-confidence advisory blocks, expired
        # patches, and graph-detected conflicts before replay. Best-effort —
        # falls back to the unfiltered list on any KG failure.
        wsc_patches = self._filter_warm_patches_with_kg(wsc_patches, wsc_advisory, state)
        if not bc_args and not bc_envs and not wsc_patches:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "best_config_empty",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
            }
            state.warm_replay_attempted = True
            return None
        # Historical gain anchor for _promote_warm_replay: prefer the donor's
        # expected gain (set when the champion config was borrowed); else MAX
        # gain across this recipe's attrs.sessions[]; fallback flat gain_pct.
        expected_gain = donor_expected_gain
        sessions_field = recipe_attrs.get("sessions")
        if expected_gain <= 0 and isinstance(sessions_field, list):
            session_gains: list[float] = []
            for s in sessions_field:
                if not isinstance(s, dict):
                    continue
                try:
                    g = float(s.get("gain_pct") or 0.0)
                except (TypeError, ValueError):
                    continue
                session_gains.append(g)
            if session_gains:
                expected_gain = max(session_gains)
        # Last-chance fallback for offline-ingested seed rows carrying a
        # flat ``gain_pct`` attr.
        if expected_gain <= 0:
            try:
                fallback = float(recipe_attrs.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                fallback = 0.0
            if fallback > 0:
                expected_gain = fallback
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "extra_sglang_args": bc_args,
            "extra_envs": dict(bc_envs),
            # Reuse the baseline's workload contract; else replay renders from YAML smoke defaults.
            "config_path": str(state.baseline_config_path or ""),
            # Carry the historical-gain anchor forward for the promote path's reproduce ratio.
            "warm_expected_gain_pct": expected_gain,
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            # Config provenance: which sibling the champion config was borrowed
            # from (config_tier="self" when the identity match owned it).
            "config_donor_tier": config_tier,
            "config_source": config_source,
            "baseline_tput_anchor": float(baseline_tput),
            # Code patches to apply before server launch (from prs_tested[KEEP]).
            "patches": list(wsc_patches),
            "blocked_patches": list(wsc_blocked),
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="replay_warm_recipe",
            params=params,
            idempotency_key="warm-replay-prelude",
        )
        if not was_existing:
            log.info(
                "PRELUDE: warm-replay enqueued task=%s (tier=%s conf=%.2f expected_gain=%.2f baseline_tput=%.2f)",
                task.task_id,
                tier,
                conf,
                expected_gain,
                baseline_tput,
            )
        state.warm_replay_attempted = True
        state.warm_replay_outcome = {
            "status": "in_flight",
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            "config_donor_tier": config_tier,
            "config_source": config_source,
            "expected_gain_pct": expected_gain,
            "replay_task_id": task.task_id,
        }
        return task

    def _promote_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Interpret a ``replay_warm_recipe`` result: any measured uplift pushes warm config onto optimization_stack + current_best; failures set status and never propagate.

        Args:
            result: The ``replay_warm_recipe`` task result dict (status,
                throughput, workspace, etc.).
            task: The originating task, used to recover the warm args/envs and
                the baseline anchor; may be ``None`` (degraded path).
        """
        state = self.shared_state
        outcome = dict(state.warm_replay_outcome or {})
        expected_gain = float(outcome.get("expected_gain_pct") or 0.0)
        if not isinstance(result, dict):
            outcome["status"] = "failed"
            outcome["reason"] = "non_dict_result"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        status = str(result.get("status") or "")
        if status != "succeeded":
            outcome["status"] = "failed"
            outcome["error_class"] = str(result.get("error_class") or "")
            outcome["reason"] = str(result.get("error") or result.get("reason") or "")[:240]
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info(
                "warm-replay failed (status=%s, error_class=%s)",
                status,
                outcome.get("error_class"),
            )
            return
        tput_raw = result.get("output_throughput")
        try:
            tput = float(tput_raw) if tput_raw is not None else 0.0
        except (TypeError, ValueError):
            tput = 0.0
        # ``tput`` (output_throughput) is the HOT measure round and is the
        # comparison value for gain/current_best. The discarded warmup round is
        # retained only for audit so warm-replay does not reintroduce
        # cold-before/hot-after drift.
        cold_raw = result.get("warmup_round_tput")
        try:
            cold_round_tput = float(cold_raw) if cold_raw is not None else 0.0
        except (TypeError, ValueError):
            cold_round_tput = 0.0
        single_round_tput = tput
        hot_tput = tput
        # Use the baseline_tput captured at enqueue time so a mid-replay baseline rerun can't shift the anchor; fall back to live state.baseline_tput.
        anchor_raw = None
        if task is not None and isinstance(getattr(task, "params", None), dict):
            anchor_raw = task.params.get("baseline_tput_anchor")
        try:
            baseline_tput = float(anchor_raw) if anchor_raw is not None else 0.0
        except (TypeError, ValueError):
            baseline_tput = 0.0
        if baseline_tput <= 0:
            baseline_tput = float(state.baseline_tput or 0.0)
        if single_round_tput <= 0 or baseline_tput <= 0:
            outcome["status"] = "failed"
            outcome["reason"] = f"invalid_tput tput={single_round_tput} baseline={baseline_tput}"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        # R3 — warm_replay reuses the BaselineExecutor but is an optimization
        # candidate, not the baseline, so it must clear the image-quality gate
        # against the pure baseline reference before promotion. A scriptable run
        # whose gate ran and FAILED (passed=False / threshold violation vs the
        # baseline reference) is rejected so a faster but quality-degrading warm
        # config is never pushed onto the stack/current_best. ``require=False``
        # keeps a missing/skipped gate non-blocking (parity with
        # ``is_valid_measurement`` and the serving no-baseline accuracy skip).
        from ..actions.executors._accuracy_gate import quality_gate_passed

        qg = result.get("quality_gate")
        if qg is not None and not quality_gate_passed(qg, require=False):
            outcome["status"] = "quality_failed"
            outcome["reason"] = "image-quality gate failed vs baseline reference"
            outcome["quality_gate"] = qg
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info("warm-replay REJECTED by quality gate: %s", qg)
            return
        measured_gain = (single_round_tput / baseline_tput - 1.0) * 100.0
        min_reproduce = float(
            getattr(self, "_warm_replay_min_reproduce_pct", 0.8) or 0.8,
        )
        # Adopt KB best_config whenever replay beats baseline; expected_gain/min_reproduce kept for audit only, not gating.
        reproduced = measured_gain > 0
        outcome["actual_gain_pct"] = round(measured_gain, 3)
        outcome["throughput_after"] = tput
        if expected_gain > 0:
            historical_bar = expected_gain * min_reproduce
            if measured_gain > 0 and measured_gain < historical_bar:
                outcome["below_historical_reproduce_pct"] = True
                outcome["historical_reproduce_bar_pct"] = round(
                    historical_bar,
                    3,
                )
        if reproduced:
            # An empty stack entry corrupts session_breakdown attribution; degrade gracefully when task is None.
            params = (task.params if task is not None else {}) or {}
            warm_args = str(params.get("extra_sglang_args") or "").strip()
            warm_envs = dict(params.get("extra_envs") or {})
            if not warm_args and not warm_envs:
                outcome["status"] = "reproduced_but_no_params"
                outcome["reason"] = "task.params missing extra_sglang_args/extra_envs"
                log.warning(
                    "warm-replay measured +%.2f%% but cannot push stack (task=%r has no warm args/envs)",
                    measured_gain,
                    task,
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            outcome["status"] = "reproduced"
            # Push warm best_config onto the stack (schema mirrors explore-KEEP); ts/workspace/gain_pct feed session_breakdown attribution.
            stack_entry = {
                "action": "replay_warm_recipe",
                "name": "warm_replay",
                "variant_name": "warm_replay",
                # Canonical key matching EXPLORE-KEEP stack entries so downstream readers key on the same name.
                "extra_server_args": warm_args,
                "extra_envs": warm_envs,
                "tput": float(single_round_tput),
                "hot_tput": float(hot_tput),
                "cold_tput": float(cold_round_tput) if cold_round_tput > 0 else None,
                "gain_pct": round(measured_gain, 3),
                "workspace": str(result.get("workspace") or ""),
                "ts": datetime.now(timezone.utc).isoformat(),
                # source_tier records the warm-recipe tier (exact/relative) for breakdown attribution.
                "source_tier": outcome.get("warm_recipe_tier", ""),
                "source_confidence": outcome.get("warm_recipe_conf", 0.0),
            }
            # Resume safety: DO NOT clobber existing stack entries; recompute cumulative gain from baseline → current tput.
            state.optimization_stack = list(state.optimization_stack or [])
            # Idempotency guard: skip push if a prior promote run already pushed the warm_replay entry.
            already_pushed = any(
                isinstance(e, dict) and e.get("action") == "replay_warm_recipe" for e in state.optimization_stack
            )
            if already_pushed:
                log.info(
                    "warm-replay promote: stack already carries the entry; "
                    "skipping duplicate push (likely resume mid-promote)",
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            state.optimization_stack.append(stack_entry)
            # gain_per_stack_entry runs in lock-step with optimization_stack.
            gp = list(getattr(state, "gain_per_stack_entry", []) or [])
            gp.append(round(measured_gain, 3))
            state.gain_per_stack_entry = gp
            # Cumulative gain is absolute tput vs baseline, not additive stack
            # deltas. Both sides use the hot measure-round contract.
            total_gain = (single_round_tput / baseline_tput - 1.0) * 100.0
            state.cumulative_gain = round(total_gain, 3)
            state.cumulative_gain_validated = round(total_gain, 3)
            state.cumulative_gain_validated_ts = stack_entry["ts"]
            state.cumulative_gain_validated_stack_len = len(state.optimization_stack)
            state.current_best = {
                "action": "warm_replay",
                "name": "warm_replay",
                "tput": single_round_tput,
                "hot_tput": hot_tput,
                "cold_tput": cold_round_tput if cold_round_tput > 0 else None,
                # Canonical key — matches the current_best shape _lift_to_current_best writes for KEEPs.
                "extra_server_args": warm_args,
                "extra_envs": warm_envs,
            }
            log.info(
                "warm-replay REPRODUCED: measured=+%.2f%% (expected=+%.2f%%, "
                "min_required=+%.2f%%); pushed warm_replay onto stack",
                measured_gain,
                expected_gain,
                expected_gain * min_reproduce if expected_gain > 0 else 0.0,
            )
            # Journal warm-replay as a synthetic KEEP; no KB lesson (verification, not a new fact).
            try:
                journal = self._ensure_journal()
                from ..state.optimization_journal import KIND_OTHER, OUTCOME_KEEP

                journal.append_entry(
                    JournalEntry(
                        phase=str(getattr(state, "phase", "PRELUDE")).upper() or "PRELUDE",
                        iter=int(state.tick or 0),
                        kind=KIND_OTHER,
                        change=f"warm_replay({outcome.get('warm_recipe_tier', '?')}): {warm_args}",
                        outcome=OUTCOME_KEEP,
                        gain_pct=round(measured_gain, 3),
                        throughput_after=tput,
                        task_id=str(task.task_id if task is not None else ""),
                        tick=int(state.tick or 0),
                    )
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay journal append failed")
        else:
            outcome["status"] = "drift"
            outcome["reason"] = (
                f"measured +{measured_gain:.2f}% below {min_reproduce * 100:.0f}% of expected +{expected_gain:.2f}%"
            )
            log.info(
                "warm-replay DRIFT: measured=+%.2f%% < expected=+%.2f%% × %.0f%%",
                measured_gain,
                expected_gain,
                min_reproduce * 100,
            )
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)

    async def _maybe_enqueue_prelude_initial_analysis_after_baseline(
        self,
        *,
        baseline_tput: float | None = None,
    ) -> None:
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline; skipped while warm-replay is in_flight (GPU/port contention).

        Args:
            baseline_tput: The baseline throughput; ``None`` reads it from
                SharedState. A non-positive value short-circuits the enqueue.
        """
        state = self.shared_state
        if _phase_state.warm_replay_in_flight(state):
            log.info(
                "PRELUDE: deferring initial %s until warm-replay completes",
                self._internal_analysis_kind(),
            )
            return
        if baseline_tput is None:
            try:
                baseline_tput = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                baseline_tput = 0.0
        if not isinstance(baseline_tput, (int, float)) or baseline_tput <= 0:
            return
        if (state.auto_roofline_pending_task_id or "").strip():
            return
        try:
            rl_task = await self._enqueue_internal_analysis_task(
                reason="prelude_initial",
            )
            state.auto_roofline_pending_task_id = rl_task.task_id
            log.info(
                "PRELUDE: baseline landed (tput=%.2f); auto-enqueued initial %s task=%s",
                float(baseline_tput),
                rl_task.kind,
                rl_task.task_id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "PRELUDE: failed to enqueue initial analysis task after baseline: %r",
                exc,
            )

    async def _enqueue_internal_analysis_task(self, *, reason: str) -> Task:
        """Build + enqueue a Coordinator-internal analysis task (roofline or profile). Kind-agnostic idempotency key internal-analysis-<reason>; omits baseline_config_path so ProfileExecutor enables torch_profiler.

        Args:
            reason: Tag distinguishing the enqueue site; used in the
                idempotency key and to select baseline vs current-best args.

        Returns:
            The created (or existing idempotent) analysis :class:`Task`.
        """
        state = self.shared_state
        kind = self._internal_analysis_kind()
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        if reason != "prelude_initial":
            cb = state.current_best or {}
            if isinstance(cb, dict):
                cb_args = str(cb.get("extra_server_args") or "")
                if cb_args:
                    params["base_extra_args"] = cb_args
        else:
            # PRELUDE roofline profiles the baseline arm: inject baseline's own
            # server args (from its materialized yaml), never current_best's,
            # so a later warm-replay can't swap in compile/fp8 flags that
            # destabilize profiling and skew the baseline ceiling.
            try:
                from ..kernel.roofline_ceiling import read_baseline_server_args

                bl_args = read_baseline_server_args(state).strip()
            except Exception:  # noqa: BLE001 — best-effort; empty falls through
                bl_args = ""
            if bl_args:
                params["base_extra_args"] = bl_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        lanes, ttl = self._registry_lanes_ttl(kind)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=f"internal-analysis-{reason}{self._cycle_idem_suffix()}",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            log.info(
                "internal-analysis task already exists (idempotent: kind=%s task_id=%s, state=%s)",
                kind,
                task.task_id,
                task.state,
            )
        return task
