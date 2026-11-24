# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE phase handler for section-based Recipe replay and initial analysis."""

from __future__ import annotations

import logging as _logging
import sqlite3
from pathlib import Path
from collections.abc import Mapping
from typing import Any
from hyperloom_common.coerce import to_float, to_int
from hyperloom_common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    SKIP_BEST_CONFIG_EMPTY,
    SKIP_CONFIDENCE_BELOW_THRESHOLD,
    SKIP_DISABLED_BY_FLAG,
    SKIP_ENQUEUE_FAILED,
    SKIP_FRAMEWORK_ROOT_MISSING,
    SKIP_KERNEL_ROOT_MISSING,
    SKIP_NO_WARM_START_RECIPE,
    SKIP_RECIPE_NOT_REPLAYABLE,
    SKIP_RECIPE_READ_FAILED,
    SKIP_WORKLOAD_CONFIG_INCOMPATIBLE,
    WarmReplayApplyItem,
    WarmReplayRequest,
)

from . import machine_state as _phase_state
from hyperloom.inference_optimizer.grid_server_args import compose_server_args, merge_cumulative_extra_server_args
from hyperloom.inference_optimizer.roofline_ceiling import read_baseline_server_args
from ..state.analysis_failures import RECIPE_PARAM, repeated_roofline_failure, roofline_recipe
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import Task, TaskSpec, create_on_free_key
from ..loop.time_budget import (
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
)
from ..collaborator import CoordinatorCollaborator
from ..knowledge.remote_recipe import KBStoreError
from .prelude_warm_kernel import _WarmKernelPrepMixin, _merge_named_current_recipe_configs
from .prelude_warm_launch import _WarmLaunchSkipMixin, suppress_recipe_config, warm_replay_task_spec
from .prelude_warm_settle import _WarmReplaySettleMixin

log = _logging.getLogger(__name__)


# The phase segment of the warm-replay event id. A literal rather than a read of
# ``state.phase``: the id must be identical at the tick that enqueues the replay
# and the later tick that harvests it, and the replay only ever runs in PRELUDE.
_WARM_REPLAY_EVENT_PHASE = "prelude"

#: The marker reason a PRELUDE arm refused for budget is filed under. A
#: constant because a consumer asking "was the initial analysis refused, or
#: never considered" selects on it.
PRELUDE_ARM_DROPPED = "prelude_arm_dropped"

# The donor's identity, carried flattened on the outcome and re-nested for the
# event. Listed once so the two directions cannot drift apart.
_WARM_REPLAY_DONOR_FIELDS = (
    "donor_canonical_id",
    "donor_model",
    "donor_session_id",
    "donor_family_tags",
    "donor_gain_pct",
    "donor_breakdown_link",
)


def _warm_recipe_history_rows(warm: Any, key: str) -> list[Any]:
    if not isinstance(warm, dict):
        return []
    recipe = warm.get("recipe") or {}
    attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
    exact_history = attrs.get("exact_history")
    if isinstance(exact_history, dict):
        attrs = exact_history
    rows = attrs.get(key)
    return rows if isinstance(rows, list) else []


def _recorded_apply_roots(provenance: Any) -> dict[str, str]:
    """Map each overlay ref to the checkout the record says it was applied into."""
    roots: dict[str, str] = {}
    for row in provenance or []:
        if not isinstance(row, Mapping):
            continue
        origin = row.get("host_origin")
        if not isinstance(origin, Mapping):
            continue
        for ref, root in (origin.get("apply_roots") or {}).items():
            name = str(ref or "").strip()
            value = str(root or "").strip()
            if name and value:
                roots.setdefault(name, value)
    return roots


def _overlay_provenance_summary(sdk_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise how the overlays this replay is about to apply were captured."""

    def _count(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    rows = [row for row in (sdk_replay.get("provenance") or []) if isinstance(row, Mapping)]
    overlays = len(sdk_replay.get("patches") or [])
    if not overlays and not rows:
        return {}
    return {
        "overlays": overlays,
        "realized": sum(1 for row in rows if row.get("realized")),
        "incomplete": sum(1 for row in rows if row.get("complete") is False),
        "artifacts_outside_root": sum(_count(row.get("artifacts_outside_root")) for row in rows),
    }


def _warm_donor_metadata(replay: Mapping[str, Any]) -> dict[str, Any]:
    """The donor fields a warm replay's record carries, dropping the ones the recipe left empty."""
    return {field: replay.get(field) for field in _WARM_REPLAY_DONOR_FIELDS if replay.get(field) not in (None, "", [])}


class PreludePhase(_WarmReplaySettleMixin, _WarmKernelPrepMixin, _WarmLaunchSkipMixin, CoordinatorCollaborator):
    """PRELUDE phase handler: session boot, target analysis, and phase seeding."""

    def internal_analysis_kind(self) -> str:
        """Pick the kind for the next Coordinator-internal analysis task: roofline when enable_roofline else profile."""
        return (
            "roofline"
            if bool(
                self.shared_state.enable_roofline,
            )
            else "profile"
        )

    def _measured_analysis_cost_sec(self) -> float:
        """Expected cost of the initial roofline/profile arm, in seconds."""
        meta = self.action_registry.get(self.internal_analysis_kind())
        return (
            expected_action_cost_minutes(
                meta,
                measured_baseline_sec=measured_baseline_runtime_sec(self.shared_state),
            )
            * 60.0
        )

    def _record_prelude_arm_dropped(self, arm: str, evidence: dict[str, Any]) -> None:
        """Record a PRELUDE arm the budget refused, as a marker on the phase.

        A marker rather than a key on the entry row's ``evidence``: that row is
        snapshotted into the phase event when the phase is entered, so anything
        appended to it afterwards reaches ``state`` and never the breakdown --
        where a refused arm then reads identically to one never considered.
        """
        _phase_state.append_phase_history_event(
            self.shared_state,
            reason=PRELUDE_ARM_DROPPED,
            evidence={"arm": arm, **evidence},
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("PRELUDE: failed to persist the dropped-arm record for %r", arm)

    def warm_recipe_proven_items(self) -> list[dict[str, str]]:
        """Summarise warm-start ``what_worked`` items the scout can skip ({name, source}); fail-soft."""
        out: list[dict[str, str]] = []
        for row in _warm_recipe_history_rows(self.shared_state.warm_start_recipe, "what_worked"):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append({"name": name, "source": str(row.get("source") or "").strip()})
        return out

    def inject_warm_recipe_history_into_ledger(self) -> int:
        """Pre-fill ``explore_search.rejected`` with the warm recipe's ``what_failed`` rows (fingerprinted so the dedup gate denies re-tests). Idempotent via warm_history_injected; returns rows added."""
        state = self.shared_state
        if state.warm_history_injected:
            return 0
        warm = state.warm_start_recipe
        what_failed = _warm_recipe_history_rows(warm, "what_failed")
        if not what_failed:
            state.warm_history_injected = True
            return 0

        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

        es = dict(state.explore_search or {})
        rejected = list(es.get("rejected") or [])
        existing_fps = {str(r.get("fingerprint") or "") for r in rejected if isinstance(r, dict)}
        existing_fps.discard("")
        added = 0
        tier = str(warm.get("tier") or "")
        for row in what_failed:
            if not isinstance(row, dict):
                continue
            args = str(row.get("extra_server_args") or "").strip()
            envs = row.get("extra_envs")
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
                    "extra_server_args": args,
                    "extra_envs": dict(envs),
                    "source": "warm_start_recipe",
                    "source_tier": tier,
                    # Preserved for forensics; not used by the dedup gate.
                    "gain_pct": row.get("gain_pct"),
                    "error_class": row.get("error_class") or row.get("reason"),
                }
            )
            added += 1

        if added:
            es["rejected"] = rejected
            state.explore_search = es
            log.info(
                "warm-recipe history: injected %d what_failed rows into explore_search.rejected (tier=%s)", added, tier
            )
        state.warm_history_injected = True
        return added

    @staticmethod
    def _is_current_remote_recipe(warm: Any) -> bool:
        from ..knowledge.remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

        recipe = warm.get("recipe") if isinstance(warm, Mapping) else None
        return bool(isinstance(recipe, Mapping) and recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE)

    def _read_current_recipe_replay(self) -> dict[str, Any]:
        """Load current replay data exclusively through the column facades."""
        from ..knowledge.agent_kb import ConfigKB, KernelAgentKB, PatchKB

        config_kb = ConfigKB.open()
        patch_kb = PatchKB.open()
        kernel = KernelAgentKB.open()
        if not all((config_kb.active, patch_kb.active, kernel.active)):
            raise ValueError("current Recipe column facades are unavailable")

        config = config_kb.read()
        args = str(config.get("extra_server_args") or "")
        envs = dict(config.get("extra_envs") or {})
        kernel_overlay = self._kernel_launch_overlay(self._collect_warm_kernel_plan(kernel))
        kernel_config = {"extra_server_args": kernel_overlay.extra_server_args, "extra_envs": kernel_overlay.extra_envs}
        combined_args, combined_envs = _merge_named_current_recipe_configs(
            [
                ("config", config),
                ("kernel", kernel_config),
            ]
        )
        # The column records its overlays in replay order, so the recorded order is the order they are applied in.
        timeline = patch_kb.read_patches()
        if len(timeline) != len(set(timeline)):
            raise ValueError("current Recipe patch refs contain a duplicate")
        provenance = patch_kb.read_provenance()
        apply_roots = _recorded_apply_roots(provenance)
        patches: list[dict[str, Any]] = []
        for index, ref in enumerate(timeline):
            source = patch_kb.prior_file(ref)
            if source is None:
                raise ValueError(f"current Recipe patch artifact is unavailable: {ref!r}")
            entry: dict[str, Any] = {
                "patch_file": ref,
                "patch_ref": str(source),
                "patch_content": "",
                "measured_gain_pct": 1e-6,
                "required": True,
                "timeline_index": index,
            }
            # Each overlay carries the checkout it was applied into, so a Recipe whose KEEPs came from different trees
            # replays against each of them.
            if recorded_root := apply_roots.get(ref, ""):
                entry["framework_root"] = recorded_root
            patches.append(entry)
        return {
            "extra_server_args": args,
            "extra_envs": envs,
            "combined_extra_server_args": combined_args,
            "combined_extra_envs": combined_envs,
            "kernel_config": kernel_config,
            "timeline": timeline,
            "patches": patches,
            "kernel_kb": kernel,
            "provenance": provenance,
        }

    async def maybe_enqueue_warm_replay(self, *, baseline_tput: float) -> "Task | None":
        """Enqueue a one-shot ``replay_warm_recipe`` task for a high-confidence T0 prior."""
        state = self.shared_state
        if not state.warm_replay_enabled:
            # The guard is flipped even on a disabled-skip so a resume without
            # --no-warm-replay cannot retroactively trigger a replay against
            # the operator's original intent.
            self._skip_warm_replay(
                code=SKIP_DISABLED_BY_FLAG, outcome={"status": "skipped", "reason": "disabled_by_flag"}
            )
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued/ran the replay.
            return None
        warm = state.warm_start_recipe
        current_remote = self._is_current_remote_recipe(warm)
        sdk_replay: dict[str, Any] = {}
        if current_remote:
            read = self._read_replayable_current_recipe(warm)
            if read is None:
                return None
            sdk_replay = read
        try:
            kernel = (
                await self._prepare_warm_kernel_kb(sdk_replay.get("kernel_kb"))
                if current_remote
                else await self._prepare_warm_kernel_kb()
            )
        except (KBStoreError, OSError, ValueError) as exc:
            log.warning("PRELUDE: warm-kernel preparation failed", exc_info=True)
            kernel = {"status": "error", "errors": 1, "reason": f"{type(exc).__name__}: {exc}"[:300]}
        if self._refused_dirty_warm_kernel(kernel):
            return None
        kernel_root_block = self._warm_replay_kernel_root_block_reason(state)
        if kernel_root_block is not None:
            self._skip_warm_replay(
                code=SKIP_KERNEL_ROOT_MISSING,
                outcome=kernel_root_block,
                details={
                    "root_kind": "kernel",
                    "recorded_roots": list(kernel_root_block.get("kernel_patch_recorded_roots") or []),
                },
            )
            return None
        overlay = self._warm_kernel_overlay(kernel, sdk_replay, current_remote)
        if not warm and not overlay["pending"]:
            if str(kernel.get("status") or "") in {"empty", "loaded"}:
                state.warm_replay_pending = {}
            self._skip_warm_replay(
                code=SKIP_NO_WARM_START_RECIPE, outcome={"status": "skipped", "reason": "no_warm_start_recipe"}
            )
            return None
        return await self._enqueue_warm_replay(warm, sdk_replay, overlay, baseline_tput)

    def _read_replayable_current_recipe(self, warm: Mapping[str, Any]) -> dict[str, Any] | None:
        recipe_metadata = warm.get("recipe") or {}
        if isinstance(recipe_metadata, Mapping) and recipe_metadata.get("replayable") is False:
            view_source = str(recipe_metadata.get("view_source") or "")
            self._skip_warm_replay(
                code=SKIP_RECIPE_NOT_REPLAYABLE,
                outcome={
                    "status": "skipped",
                    "reason": str(recipe_metadata.get("replay_disabled_reason") or "remote_recipe_view_not_replayable"),
                    "view_source": view_source,
                },
                details={"view_source": view_source},
            )
            return None
        try:
            sdk_replay = self._read_current_recipe_replay()
        except (OSError, ValueError, RuntimeError, AttributeError, TypeError) as exc:  # current replay fails closed
            self._skip_warm_replay(
                code=SKIP_RECIPE_READ_FAILED,
                outcome={
                    "status": "skipped",
                    "reason": (f"current_recipe_sdk_read_failed:{type(exc).__name__}:{exc}")[:500],
                },
                details={"error_class": type(exc).__name__},
            )
            return None
        # Every overlay names the checkout it was measured on.
        recorded = [str((entry or {}).get("framework_root") or "").strip() for entry in sdk_replay.get("patches") or []]
        absent = [root for root in dict.fromkeys(recorded) if root and not Path(root).is_dir()]
        if not all(recorded):
            self._refuse_framework_roots("framework_apply_root_missing", [root for root in recorded if root])
            return None
        if absent:
            self._refuse_framework_roots("framework_apply_root_absent", absent)
            return None
        return sdk_replay

    def _refuse_framework_roots(self, reason: str, roots: list[str], rollback: dict[str, Any] | None = None) -> None:
        self._skip_warm_replay(
            code=SKIP_FRAMEWORK_ROOT_MISSING,
            outcome=self._warm_replay_root_skip_outcome(
                reason=reason, root_kind="framework", roots=roots, rollback=rollback
            ),
            details={"root_kind": "framework", "recorded_roots": list(roots)},
        )

    def _warm_replay_launch_config(
        self, warm: Mapping[str, Any], sdk_replay: Mapping[str, Any], current_remote: bool
    ) -> dict[str, Any]:
        recipe = warm.get("recipe") or {}
        if not isinstance(recipe, dict):
            recipe = {}
        # best_config/sessions may be top-level or nested under attrs.
        recipe_attrs = recipe.get("attrs") or recipe
        replay = self.shared_state.warm_start_context.get("recommended_replay") if not current_remote else None
        replay = replay if isinstance(replay, dict) else {}
        rep_args = str(replay.get("extra_server_args") or "").strip()
        rep_envs = replay.get("extra_envs") if isinstance(replay.get("extra_envs"), dict) else {}
        conf = to_float(warm.get("confidence"), 0.0)
        config: dict[str, Any] = {
            "recipe_attrs": recipe_attrs,
            "replay": replay,
            "confidence": conf,
            "replay_confidence": conf,
            "config_tier": "self",
            "config_source": str(recipe.get("canonical_id") or ""),
            "donor_expected_gain": 0.0,
        }
        if current_remote:
            config["extra_server_args"] = str(sdk_replay.get("extra_server_args") or "").strip()
            config["extra_envs"] = dict(sdk_replay.get("extra_envs") or {})
            config["donor_expected_gain"] = to_float(recipe.get("validated_gain_pct"), 0.0)
        elif rep_args or rep_envs:
            config["extra_server_args"] = rep_args
            config["extra_envs"] = dict(rep_envs)
            config["replay_confidence"] = to_float(replay.get("config_confidence"), conf)
            config["config_source"] = str(replay.get("config_source") or "")
            config["config_tier"] = str(replay.get("config_tier") or "self")
            config["donor_expected_gain"] = to_float(replay.get("expected_gain_pct"), 0.0)
        else:
            best_config = recipe_attrs.get("best_config")
            if not isinstance(best_config, dict):
                best_config = {}
            bc_envs = best_config.get("extra_envs")
            config["extra_server_args"] = str(best_config.get("extra_server_args") or "").strip()
            config["extra_envs"] = bc_envs if isinstance(bc_envs, dict) else {}
        return config

    def _warm_expected_gain(self, recipe_attrs: Mapping[str, Any], donor_expected_gain: float) -> float:
        gain = donor_expected_gain
        sessions_field = recipe_attrs.get("sessions")
        if gain <= 0 and isinstance(sessions_field, list):
            session_gains = [to_float(row.get("gain_pct"), 0.0) for row in sessions_field if isinstance(row, dict)]
            if session_gains:
                gain = max(session_gains)
        # Last-chance fallback for offline-ingested seed rows.
        if gain <= 0:
            fallback = to_float(recipe_attrs.get("gain_pct"), 0.0)
            if fallback > 0:
                gain = fallback
        return gain

    def _replayable_patch_plan(
        self, config: Mapping[str, Any], sdk_replay: Mapping[str, Any], overlay: Mapping[str, Any], current_remote: bool
    ) -> list[Any] | None:
        patches = list((sdk_replay if current_remote else config["replay"]).get("patches") or [])
        if not patches:
            return patches
        # Re-checked here because a persisted plan may be resumed under a different image, and because the kernel
        # half is already applied by now: a root that has since gone means unwinding that too.
        recorded_roots = [str((entry or {}).get("framework_root") or "").strip() for entry in patches]
        unusable = [root for root in dict.fromkeys(recorded_roots) if root and not Path(root).is_dir()]
        if all(recorded_roots) and not unusable:
            return patches
        rollback = self._unwind_warm_kernel_half(overlay)
        self._refuse_framework_roots(
            "framework_apply_root_absent" if unusable else "framework_apply_root_missing",
            unusable or [root for root in recorded_roots if root],
            rollback=rollback,
        )
        return None

    def _combined_warm_launch(
        self,
        config: Mapping[str, Any],
        overlay: Mapping[str, Any],
        sdk_replay: Mapping[str, Any],
        current_remote: bool,
        recipe_suppressed: bool,
    ) -> tuple[str, dict[str, str]]:
        if current_remote and recipe_suppressed:
            return overlay["extra_server_args"], dict(overlay["extra_envs"])
        if current_remote:
            return (
                str(sdk_replay.get("combined_extra_server_args") or ""),
                dict(sdk_replay.get("combined_extra_envs") or {}),
            )
        combined_envs = dict(config["extra_envs"])
        combined_envs.update(overlay["extra_envs"])
        return (
            merge_cumulative_extra_server_args(config["extra_server_args"], overlay["extra_server_args"], ""),
            combined_envs,
        )

    async def _enqueue_warm_replay(
        self, warm: Mapping[str, Any], sdk_replay: dict[str, Any], overlay: dict[str, Any], baseline_tput: float
    ) -> "Task | None":
        state = self.shared_state
        current_remote = self._is_current_remote_recipe(warm)
        config = self._warm_replay_launch_config(warm, sdk_replay, current_remote)
        # tier/conf stamped at T0.
        tier = str(warm.get("tier") or "").strip()
        conf = config["confidence"]
        donor_metadata = _warm_donor_metadata(config["replay"])
        identity = {"warm_recipe_tier": tier, "warm_recipe_conf": conf, **donor_metadata}
        min_conf = float(state.warm_replay_min_confidence)
        replay_conf = config["replay_confidence"]
        # Low-confidence config/patch content is suppressed, but the kernel section from the same Recipe can still
        # receive a combined check.
        recipe_suppressed = replay_conf < min_conf
        if recipe_suppressed and not overlay["pending"]:
            self._skip_warm_replay(
                code=SKIP_CONFIDENCE_BELOW_THRESHOLD,
                outcome={
                    "status": "skipped",
                    "reason": f"confidence_below_threshold ({replay_conf:.2f} < {min_conf:.2f})",
                    "config_donor_tier": config["config_tier"],
                    "config_source": config["config_source"],
                    **identity,
                },
                details={"observed": replay_conf, "threshold": min_conf},
            )
            return None
        if recipe_suppressed:
            suppress_recipe_config(config)
            patches: list[Any] = []
            # Current records derive fail-closed mode from their exact SDK timeline.
            required_timeline = False
        else:
            plan = self._replayable_patch_plan(config, sdk_replay, overlay, current_remote)
            if plan is None:
                return None
            patches = plan
            required_timeline = bool(
                sdk_replay.get("timeline") if current_remote else config["recipe_attrs"].get("required_patch_timeline")
            )
        if not config["extra_server_args"] and not config["extra_envs"] and not patches and not overlay["pending"]:
            self._skip_warm_replay(
                code=SKIP_BEST_CONFIG_EMPTY, outcome={"status": "skipped", "reason": "best_config_empty", **identity}
            )
            return None
        combined_args, combined_envs = self._combined_warm_launch(
            config, overlay, sdk_replay, current_remote, recipe_suppressed
        )
        if self._skip_restated_replay(combined_args, combined_envs, [*patches, *overlay["pending"]], identity):
            return None
        if self._skip_context_copy_replay(combined_args, combined_envs, overlay, identity):
            return None
        workload = self._workload_compatible_launch(combined_args, overlay)
        if workload is None:
            return None
        params = {
            **self._warm_replay_task_params(config, overlay, patches, baseline_tput, required_timeline),
            "extra_server_args": combined_args,
            "extra_envs": combined_envs,
            # Historical-gain anchor for the promote path's reproduce ratio.
            "warm_expected_gain_pct": self._warm_expected_gain(config["recipe_attrs"], config["donor_expected_gain"]),
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            "workload_compatibility": workload,
        }
        return await self._open_warm_replay_task(params, overlay, sdk_replay, recipe_suppressed, donor_metadata)

    def _workload_compatible_launch(self, combined_args: str, overlay: dict[str, Any]) -> dict[str, Any] | None:
        state = self.shared_state
        try:
            from hyperloom.inference_optimizer.grid_server_args import validate_warm_replay_context_length

            workload_compatibility = validate_warm_replay_context_length(
                combined_args, state.framework, int(state.isl or 0), int(state.osl or 0), state.max_model_len or None
            )
        except ValueError as exc:
            rollback = self._unwind_warm_kernel_half(overlay)
            target_workload_shape = {
                "conc": int(state.conc or 0),
                "isl": int(state.isl or 0),
                "osl": int(state.osl or 0),
            }
            self._skip_warm_replay(
                code=SKIP_WORKLOAD_CONFIG_INCOMPATIBLE,
                outcome={
                    "status": ("skipped" if rollback.get("ok") else "rollback_failed"),
                    "reason": (f"workload_config_incompatible:{type(exc).__name__}:{exc}")[:500],
                    "target_workload_shape": target_workload_shape,
                    "rollback": rollback,
                },
                details={"error_class": type(exc).__name__, "target_workload_shape": dict(target_workload_shape)},
            )
            return None
        return workload_compatibility

    def _warm_replay_task_params(
        self,
        config: Mapping[str, Any],
        overlay: Mapping[str, Any],
        patches: list[Any],
        baseline_tput: float,
        required_timeline: bool,
    ) -> dict[str, Any]:
        return {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "recipe_extra_server_args": config["extra_server_args"],
            "recipe_extra_envs": dict(config["extra_envs"]),
            "kernel_extra_server_args": overlay["extra_server_args"],
            "kernel_extra_envs": overlay["extra_envs"],
            # Reuse the baseline's workload contract (else YAML smoke defaults).
            "config_path": str(self.shared_state.baseline_config_path or ""),
            # Config provenance ("self" when the identity match owned it).
            "config_donor_tier": config["config_tier"],
            "config_source": config["config_source"],
            "baseline_tput_anchor": float(baseline_tput),
            # Code patches to apply before server launch.
            "patches": list(patches),
            "required_patch_timeline": required_timeline,
            "warm_kernel_plan": overlay["pending"],
            "warm_kernel_apply_results": overlay["applied"],
            "warm_kernel_snapshots": overlay["snapshots"],
        }

    async def _open_warm_replay_task(
        self,
        params: dict[str, Any],
        overlay: dict[str, Any],
        sdk_replay: Mapping[str, Any],
        recipe_suppressed: bool,
        donor_metadata: Mapping[str, Any],
    ) -> "Task | None":
        state = self.shared_state
        try:
            lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("replay_warm_recipe")
            task, was_existing = await self.tasks.create_or_return_existing(
                warm_replay_task_spec(params, lanes=lanes, ttl=ttl)
            )
        except Exception as exc:
            rollback = self._unwind_warm_kernel_half(overlay)
            self._skip_warm_replay(
                code=SKIP_ENQUEUE_FAILED,
                outcome={
                    "status": ("enqueue_failed" if rollback.get("ok") else "rollback_failed"),
                    "reason": f"warm replay enqueue failed: {type(exc).__name__}",
                    "rollback": rollback,
                },
                details={"error_class": type(exc).__name__},
            )
            raise
        if not was_existing:
            log.info(
                "PRELUDE: warm-replay enqueued task=%s (tier=%s conf=%.2f expected_gain=%.2f baseline_tput=%.2f)",
                task.task_id,
                params["warm_recipe_tier"],
                params["warm_recipe_conf"],
                params["warm_expected_gain_pct"],
                params["baseline_tput_anchor"],
            )
        state.warm_replay_attempted = True
        state.warm_replay_outcome = {
            "status": "in_flight",
            # Bounds the replay for the SBD timeline; the terminal write below stamps ``settled_at`` against this.
            "enqueued_at": now_iso(timespec="seconds"),
            "warm_recipe_tier": params["warm_recipe_tier"],
            "warm_recipe_conf": params["warm_recipe_conf"],
            "config_donor_tier": params["config_donor_tier"],
            "config_source": params["config_source"],
            "expected_gain_pct": params["warm_expected_gain_pct"],
            "replay_task_id": task.task_id,
            "kernel_count": len(overlay["pending"]),
            "recipe_suppressed": recipe_suppressed,
            **({"overlay_provenance": summary} if (summary := _overlay_provenance_summary(sdk_replay)) else {}),
            **donor_metadata,
        }
        state.warm_replay_pending = {
            **state.warm_replay_pending,
            "status": "in_flight",
            "task_id": task.task_id,
            "kernel_apply_results": overlay["applied"],
            "kernel_plan": overlay["pending"],
            "kernel_snapshots": overlay["snapshots"],
        }
        try:
            state.save(self.session_dir)
        except OSError:
            log.debug("combined warm replay pending save failed", exc_info=True)
        # Opened after the outcome is persisted so the request block reads the
        # donor identity the outcome just stamped, and a session killed between
        # the two is recovered as an in-flight replay of a named recipe.
        self._open_warm_replay_timeline(task=task, session_baseline_tput=state.baseline_tput)
        return task

    # ---- warm-replay timeline recording ----------------------------------
    # The replay spans two ticks: one enqueues the task, a later one harvests it.
    # The event id derives from persisted state alone, so the promote seam
    # rebinds to the event the enqueue seam opened rather than holding a
    # recorder across a boundary a resume does not survive.

    def _warm_replay_donor(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Lift the donor block out of a flat ``donor_*`` mapping; empty when not borrowed."""
        return {
            field.removeprefix("donor_"): source[field]
            for field in _WARM_REPLAY_DONOR_FIELDS
            if source.get(field) not in (None, "", [])
        }

    def _open_warm_replay_timeline(self, *, task: "Task | None", session_baseline_tput: Any) -> None:
        """Open the replay's event at the moment its task is dispatched.

        ``session_baseline_tput`` is kept beside the enqueue anchor so a reader
        can see the two diverge across a re-baseline.
        """
        params = dict(getattr(task, "params", None) or {})
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        recorder = make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            request=WarmReplayRequest(
                task_id=str(getattr(task, "task_id", "") or ""),
                tier=str(params.get("warm_recipe_tier") or ""),
                config_source=str(params.get("config_source") or ""),
                config_donor_tier=str(params.get("config_donor_tier") or ""),
                donor=self._warm_replay_donor(dict(self.shared_state.warm_replay_outcome or {})) or None,
                expected_gain_pct=params.get("warm_expected_gain_pct"),
                confidence=params.get("warm_recipe_conf"),
                session_baseline_tput=session_baseline_tput,
                kernel_count=len(list(params.get("warm_kernel_plan") or [])),
                recipe_suppressed=not str(params.get("config_source") or ""),
            ),
        )
        if recorder is not None:
            self._record_warm_kernel_apply_items(recorder)

    def _record_warm_kernel_apply_items(self, recorder: Any) -> None:
        """State which kernel items the replay carried and which of them landed.

        Recorded as the replay is dispatched, because the ruling replaces the
        plan with the subset it kept: by the time the bench has a verdict, the
        items that never applied are gone from state and unrecoverable.

        ``PENDING`` is an item that applied and awaits the bench. ``DEFERRED``
        resolved to no target under this checkout. ``ERROR`` stops the
        sequence, so the items behind it hold no decision and record nothing --
        they were never attempted.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import APPLY_KERNEL

        for position, entry in enumerate(self.shared_state.warm_kernel_kb_plan or []):
            if not isinstance(entry, Mapping) or not str(entry.get("decision") or ""):
                continue
            outcome = entry.get("apply_result") if isinstance(entry.get("apply_result"), Mapping) else {}
            sources = entry.get("source_paths") or []
            recorder.record_apply_item(
                WarmReplayApplyItem(
                    str(entry.get("patch_path") or (sources[0] if sources else "")),
                    kind=APPLY_KERNEL,
                    position=position,
                    applied=str(entry.get("decision") or "") == "PENDING",
                    reason=str(
                        outcome.get("reason")
                        or outcome.get("error")
                        or outcome.get("exception")
                        or entry.get("resolution_error")
                        or ""
                    ),
                    target=str(entry.get("apply_root") or ""),
                )
            )

    def _warm_replay_timeline(self, task: "Task | None" = None):
        """Rebind to the in-flight replay's event, or ``None`` when one could not be built.

        Absent task params degrade the request block, never the gates and the
        measurement the promote seam is here to record.
        """
        params = dict(getattr(task, "params", None) or {})
        outcome = dict(self.shared_state.warm_replay_outcome or {})
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        return make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            request=WarmReplayRequest(
                task_id=str(getattr(task, "task_id", "") or outcome.get("replay_task_id") or ""),
                tier=str(params.get("warm_recipe_tier") or outcome.get("warm_recipe_tier") or ""),
                config_source=str(params.get("config_source") or outcome.get("config_source") or ""),
                config_donor_tier=str(params.get("config_donor_tier") or outcome.get("config_donor_tier") or ""),
                donor=self._warm_replay_donor(outcome) or None,
                expected_gain_pct=params.get("warm_expected_gain_pct", outcome.get("expected_gain_pct")),
                confidence=params.get("warm_recipe_conf", outcome.get("warm_recipe_conf")),
                session_baseline_tput=self.shared_state.baseline_tput or None,
                kernel_count=len(list(params.get("warm_kernel_plan") or [])),
            ),
            # Rebinding, not opening: the enqueue seam already put this event
            # on the timeline, and opening it twice would restate its start.
            open_event_on_timeline=False,
        )

    def _skip_warm_replay(
        self,
        *,
        code: str,
        outcome: Mapping[str, Any],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Settle the one-shot guard for a replay that will not run.

        Every refusal owes the same four things: flip the guard, state the
        outcome, close the timeline event, and persist. The persist is what
        makes the guard mean anything -- a refusal that never reached disk would
        let the next boot replay against the decision just taken.

        Call this after any rollback or stop-reason the branch also sets, so
        that one save carries the whole refusal.
        """
        state = self.shared_state
        state.warm_replay_attempted = True
        state.warm_replay_outcome = dict(outcome)
        self._record_warm_replay_skip(code=code, outcome=state.warm_replay_outcome, details=details)
        state.save(self.session_dir)

    def _record_warm_replay_skip(
        self,
        *,
        code: str,
        outcome: Mapping[str, Any],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the event for a replay refused before it ran.

        A skip is a decision, so it closes an event of its own rather than
        leaving the timeline silent. ``code`` is a ``SKIP_*`` value. The
        earliest refusals have no identity fields on ``outcome`` yet, and state
        an empty request rather than an invented one.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        recorder = make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            request=WarmReplayRequest(
                tier=str(outcome.get("warm_recipe_tier") or ""),
                config_source=str(outcome.get("config_source") or ""),
                config_donor_tier=str(outcome.get("config_donor_tier") or ""),
                donor=self._warm_replay_donor(outcome) or None,
                expected_gain_pct=outcome.get("expected_gain_pct"),
                confidence=outcome.get("warm_recipe_conf"),
            ),
        )
        if recorder is None:
            return
        # A refusal rooted in the kernel preparation has a plan behind it,
        # and which item refused to apply is the whole of what it means.
        self._record_warm_kernel_apply_items(recorder)
        if rollback := (outcome.get("rollback") if isinstance(outcome.get("rollback"), Mapping) else None):
            recorder.record_rollback(ok=rollback.get("ok"), errors=rollback.get("errors"))
        recorder.finish_skipped(code=code, outcome=outcome, details=details)

    async def maybe_enqueue_prelude_initial_analysis_after_baseline(
        self,
        *,
        baseline_tput: float | None = None,
    ) -> None:
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline; skipped while warm-replay is in_flight (GPU/port contention)."""
        state = self.shared_state
        if _phase_state.warm_replay_in_flight(state):
            log.info(
                "PRELUDE: deferring initial %s until warm-replay completes",
                self.internal_analysis_kind(),
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
        affordable, evidence = _phase_state.prelude_can_afford(
            state,
            expected_cost_sec=self._measured_analysis_cost_sec(),
        )
        if not affordable:
            log.warning(
                "PRELUDE: skipping the initial %s — %.0fs of preparation budget "
                "left (bound=%s) against an expected %.0fs. The optimization "
                "phases keep the time instead.",
                self.internal_analysis_kind(),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
                evidence.get("expected_cost_sec", 0.0),
            )
            self._record_prelude_arm_dropped("initial_analysis", evidence)
            return
        try:
            rl_task = await self.enqueue_internal_analysis_task(
                reason="prelude_initial",
            )
            if rl_task is None:
                return
            state.auto_roofline_pending_task_id = rl_task.task_id
            log.info(
                "PRELUDE: baseline landed (tput=%.2f); auto-enqueued initial %s task=%s",
                float(baseline_tput),
                rl_task.kind,
                rl_task.task_id,
            )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            log.exception(
                "PRELUDE: failed to enqueue initial analysis task after baseline: %r",
                exc,
            )

    def _analysis_attempt_suffix(self, kind: str) -> str:
        """Idempotency-key suffix separating a re-armed roofline retry from the attempt that failed."""
        streak = to_int(self.shared_state.roofline_failure_streak, 0) if kind == "roofline" else 0
        return f"-a{streak}" if streak > 0 else ""

    async def enqueue_internal_analysis_task(self, *, reason: str, idem_scope: str = "") -> "Task | None":
        """Build + enqueue a Coordinator-internal analysis task (roofline or profile) keyed internal-analysis-<reason>
        <idem_scope> (``idem_scope`` separates triggers sharing a reason), stepped past cancelled attempts. Returns None
        when the stack cannot produce a GPU trace for it to analyze, or every retry key holds a cancelled attempt.
        """
        params = self.internal_analysis_params(reason=reason)
        if params is None:
            return None
        kind = self.internal_analysis_kind()
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl(kind)
        created = await create_on_free_key(
            self.tasks,
            TaskSpec(
                kind=kind,
                params=params,
                idempotency_key=(
                    f"internal-analysis-{reason}{idem_scope}"
                    f"{self._coord.dispatcher.cycle_idem_suffix()}{self._analysis_attempt_suffix(kind)}"
                ),
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            ),
            spent=lambda existing: existing.state == "cancelled",
        )
        if created is None:
            return None
        task, was_existing = created
        if was_existing:
            log.info(
                "internal-analysis task already exists (idempotent: kind=%s task_id=%s, state=%s)",
                kind,
                task.task_id,
                task.state,
            )
        return task

    def internal_analysis_params(self, *, reason: str, inline_event: str = "") -> dict[str, Any] | None:
        """Params for a Coordinator-internal analysis run, or None when the stack cannot produce a GPU trace."""
        from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM

        state = self.shared_state
        unsupported = str(state.gpu_trace_unsupported_reason or "")
        if unsupported:
            log.error(
                "internal-analysis (%s): not run -- %s; relying on static-source evidence for the rest of the session",
                reason,
                unsupported,
            )
            self._record_prelude_arm_dropped(
                "internal_analysis",
                {"reason": reason, "gpu_trace_unsupported_reason": unsupported},
            )
            return None
        recipe = roofline_recipe(state, reason)
        repeated = repeated_roofline_failure(state, recipe)
        if repeated is not None:
            log.warning(
                "internal-analysis (%s): not run -- the roofline on %s already failed with %s, which recurs on the "
                "same input; a KEEP that changes the recipe re-arms it",
                reason,
                recipe,
                repeated.get("failure"),
            )
            self._record_prelude_arm_dropped(
                "internal_analysis",
                {"reason": reason, "recipe": recipe, "repeated_failure": dict(repeated)},
            )
            return None
        params: dict[str, Any] = {"source": "coordinator_internal", "reason": str(reason), RECIPE_PARAM: recipe}
        if inline_event:
            params[INLINE_EVENT_PARAM] = str(inline_event)
        bl_args = read_baseline_server_args(state).strip()
        if reason != "prelude_initial":
            # The profiler boots from the benchmark config, so the stack delta is layered over the baseline's own args.
            inject_stack_base_params(params, state)
            params["base_extra_args"] = compose_server_args(
                inherited_args=bl_args,
                base_extra_args=str(params.get("base_extra_args") or ""),
                remove_args=params.get("base_remove_args"),
                args_mode=str(params.get("base_args_mode") or "append"),
            )
        else:
            # PRELUDE roofline profiles the baseline arm: inject baseline's own server args (never current_best's) so
            # a later warm-replay can't swap in flags that skew the baseline ceiling.
            params.update(self._coord.enablement_revalidation.baseline_launch_params())
            if bl_args:
                params["base_extra_args"] = bl_args
        if benchmark_script := state.accepted_baseline_script():
            params["benchmark_script"] = benchmark_script
        return params
