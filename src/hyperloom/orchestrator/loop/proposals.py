# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping
from hyperloom.orchestrator.knowledge.recipe_kb import recipe_canonical_id
from hyperloom.inference_optimizer.recipe_snapshot_constants import detect_framework_version
from ..phases import machine_state as _phase_state
from ..bus.message_bus import Message
if TYPE_CHECKING:
    from ..state.task_registry import Task

from .coordinator import (
    PendingProposal,
)
import logging as _logging
log = _logging.getLogger(__name__)


def _extra_server_args(payload: Mapping[str, Any]) -> str:
    """Read canonical ``extra_server_args`` from a payload."""
    value = payload.get("extra_server_args")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


class ProposalsCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _workload_canonical_id(self) -> str:
        """Canonical 5-tuple recipe id for the current workload. Must match cortex_t0.run_t0_anchor's derivation so warm-start and KEEP/REVERT/CLOSE writes target the same row.

        Returns:
            The canonical recipe id derived from model, hardware, framework,
            framework version and precision.
        """
        ss = self.shared_state
        workload = ss.model_name or "unknown_model"
        hw = self._kb_hardware_slug()
        framework = str(getattr(ss, "framework", "") or "")
        framework_version = str(getattr(ss, "framework_version", "") or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(getattr(ss, "precision", "") or "")
        model_type = str(getattr(ss, "model_type", "") or "")
        architectures = getattr(ss, "model_architectures", None) or []
        return recipe_canonical_id(
            model=workload,
            hardware=hw,
            framework_name=framework,
            framework_version=framework_version,
            precision=precision,
            model_type=model_type,
            architectures=architectures,
        )

    def _read_local_recipe_row(self) -> dict[str, Any]:
        """Load the authoritative local recipe row for amend/finalize writes.

        Cached per tick to avoid repeated I/O during multi-variant KEEP batches.
        """
        if self.cortex_kb is None:
            return {}
        tick = int(getattr(self.shared_state, "tick", 0) or 0)
        cache = getattr(self, "_local_recipe_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == tick:
            return cache[1]
        try:
            row = (
                self.cortex_kb.local.get_recipe(
                    canonical_id=self._workload_canonical_id(),
                )
                or {}
            )
        except Exception:  # noqa: BLE001
            row = {}
        self._coord._local_recipe_cache = (tick, row)
        return row

    @staticmethod
    def _extract_kept_best_config(
        *,
        task: "Task",
        variant_attrs: dict[str, Any] | None = None,
        result_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a replayable ``best_config`` from a KEEP'd task or explore variant."""
        params = task.params if isinstance(getattr(task, "params", None), dict) else {}
        attrs = variant_attrs if isinstance(variant_attrs, dict) else {}

        args = _extra_server_args(attrs)
        if not args.strip():
            args = _extra_server_args(params)
        if not args.strip() and isinstance(result_dict, dict):
            args = _extra_server_args(result_dict)

        envs_raw = attrs.get("extra_envs") or params.get("extra_envs") or {}
        if not envs_raw and isinstance(result_dict, dict):
            envs_raw = result_dict.get("extra_envs") or {}
        envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}

        if not args.strip() and not envs:
            return {}

        best_config: dict[str, Any] = {}
        if args.strip():
            best_config["extra_server_args"] = args.strip()
        if envs:
            best_config["extra_envs"] = envs
        return best_config

    @staticmethod
    def _kb_best_config_overrides_for_keep(
        *,
        live: Mapping[str, Any],
        best_config_candidate: Mapping[str, Any],
        throughput_after: float | None,
    ) -> dict[str, Any]:
        """Decide whether a KEEP amend should also stamp ``best_config`` on the recipe row."""
        if not best_config_candidate:
            return {}

        live_bc = live.get("best_config") if isinstance(live.get("best_config"), Mapping) else {}
        live_has_config = bool(
            _extra_server_args(live_bc).strip()
            or (isinstance(live_bc.get("extra_envs"), Mapping) and live_bc.get("extra_envs"))
        )
        try:
            live_tput = float(live.get("best_throughput") or 0.0)
        except (TypeError, ValueError):
            live_tput = 0.0
        try:
            new_tput = float(throughput_after or 0.0)
        except (TypeError, ValueError):
            new_tput = 0.0

        if not live_has_config or (new_tput > 0.0 and new_tput >= live_tput):
            overrides: dict[str, Any] = {
                "best_config": dict(best_config_candidate),
            }
            if new_tput > 0.0:
                overrides["best_throughput"] = new_tput
            return overrides
        return {}

    def _kb_amend_recipe(
        self,
        *,
        append_lesson: dict[str, Any] | None = None,
        append_pitfall: dict[str, Any] | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> None:
        """Read-modify-write helper for the recipe-snapshot KB: load live row, append lesson/pitfall, merge recipe_overrides (unset fields preserved), write back. Best-effort; lesson/pitfall appended without dedup.

        Args:
            append_lesson: Optional lesson dict appended to the recipe.
            append_pitfall: Optional pitfall dict appended to the recipe.
            recipe_overrides: Optional recipe field overrides merged in (unset
                fields preserved).
            provenance_details: Optional provenance metadata recorded with the
                amendment.
        """
        if self.cortex_kb is None:
            return
        try:
            cid = self._workload_canonical_id()
        except Exception:  # noqa: BLE001
            log.exception("_kb_amend_recipe: cid derivation failed")
            return

        ss = self.shared_state
        framework = str(getattr(ss, "framework", "") or "")
        framework_version = str(getattr(ss, "framework_version", "") or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(getattr(ss, "precision", "") or "")

        # Read the LOCAL authoritative row (bypass remote-first read; else a central row clobbers this session's lessons/pitfalls).
        try:
            live = self.cortex_kb.local.get_recipe(canonical_id=cid) or {}
        except Exception as exc:  # noqa: BLE001
            log.info(
                "_kb_amend_recipe: local get_recipe failed (%s); proceeding with empty live",
                exc,
            )
            live = {}

        lessons = list(live.get("lessons") or [])
        if append_lesson is not None:
            lessons.append(append_lesson)
        pitfalls = list(live.get("pitfalls") or [])
        if append_pitfall is not None:
            pitfalls.append(append_pitfall)

        # Build put_recipe kwargs, preserving live fields the caller didn't override.
        overrides = dict(recipe_overrides or {})
        _reserved = {
            "canonical_id",
            "version",
            "created_at",
            "updated_at",
            "model",
            "hardware",
            "framework",
            "framework_name",
            "framework_version",
            "precision",
            "best_config",
            "best_throughput",
            "what_worked",
            "what_failed",
            "remaining_gaps",
            "prs_tested",
            "pitfalls",
            "lessons",
            "last_profiled",
            "stack_fingerprint",
            "sessions",
            "authority",
            "confidence",
            "evidence_refs",
            "provenance",
        }
        prior_extras = {k: v for k, v in live.items() if k not in _reserved}
        merged_extras = {**prior_extras, **(overrides.get("extras") or {})}
        # Re-stamp config.json architecture-identity tags; skipped when unset.
        _arch = getattr(ss, "model_architectures", None) or []
        if isinstance(_arch, list):
            _arch_list = [str(a).strip() for a in _arch if str(a or "").strip()]
            if _arch_list:
                merged_extras["architectures"] = _arch_list
        _mtype = str(getattr(ss, "model_type", "") or "").strip()
        if _mtype:
            merged_extras["model_type"] = _mtype
        put_kwargs: dict[str, Any] = {
            "canonical_id": cid,
            "model": ss.model_name or "unknown_model",
            "hardware": self._kb_hardware_slug(),
            "framework_name": framework,
            "framework_version": framework_version,
            "precision": precision,
            "best_config": overrides.get("best_config")
            if "best_config" in overrides
            else dict(live.get("best_config") or {}),
            "best_throughput": overrides.get("best_throughput")
            if "best_throughput" in overrides
            else float(live.get("best_throughput") or 0.0),
            "what_worked": overrides.get("what_worked")
            if "what_worked" in overrides
            else list(live.get("what_worked") or []),
            "what_failed": overrides.get("what_failed")
            if "what_failed" in overrides
            else list(live.get("what_failed") or []),
            "remaining_gaps": overrides.get("remaining_gaps")
            if "remaining_gaps" in overrides
            else list(live.get("remaining_gaps") or []),
            "prs_tested": overrides.get("prs_tested")
            if "prs_tested" in overrides
            else list(live.get("prs_tested") or []),
            "pitfalls": pitfalls,
            "lessons": lessons,
            "last_profiled": overrides.get("last_profiled")
            if "last_profiled" in overrides
            else str(live.get("last_profiled") or ""),
            "stack_fingerprint": overrides.get("stack_fingerprint")
            if "stack_fingerprint" in overrides
            else dict(live.get("stack_fingerprint") or {}),
            "sessions": overrides.get("sessions") if "sessions" in overrides else list(live.get("sessions") or []),
            "extras": merged_extras,
            # Preserve audit fields across the amend (else put_recipe resets them to defaults).
            "authority": overrides.get("authority")
            if "authority" in overrides
            else str(live.get("authority") or "EXPERIENTIAL"),
            "confidence": overrides.get("confidence")
            if "confidence" in overrides
            else float(live.get("confidence") or 0.85),
            "evidence_refs": overrides.get("evidence_refs")
            if "evidence_refs" in overrides
            else list(live.get("evidence_refs") or []),
            "provenance": {
                "source": "hyperloom-inference-optimizer",
                "generator": "coordinator",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="microseconds",
                ),
                "details": dict(provenance_details or {}),
            },
        }
        try:
            self.cortex_kb.put_recipe(**put_kwargs)
            self._coord._local_recipe_cache = None
        except Exception:  # noqa: BLE001
            log.exception(
                "_kb_amend_recipe: put_recipe failed for cid=%s",
                cid,
            )

    def _inject_explore_runtime_params(self, params: dict) -> None:
        """Inject explore-task operational knobs from SharedState into ``params`` (single source of truth for both propose/Critic and direct-delegate paths). setdefault preserves LLM overrides.

        Args:
            params: The explore-task params dict mutated in place; existing keys
                are preserved (``setdefault``).
        """
        br = float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0)
        if br > 0:
            params.setdefault("baseline_runtime_sec", br)
        # Warm measure-round anchor for the decision-round overtime kill.
        bwr = float(getattr(self.shared_state, "baseline_warm_runtime_sec", 0.0) or 0.0)
        if bwr > 0:
            params.setdefault("baseline_warm_runtime_sec", bwr)
        kill_ratio = float(
            getattr(
                self.shared_state,
                "explore_overtime_kill_ratio",
                0.0,
            )
            or 0.0
        )
        if kill_ratio > 0:
            params.setdefault("explore_overtime_kill_ratio", kill_ratio)
        variant_timeout_override = int(
            getattr(
                self.shared_state,
                "explore_variant_timeout_sec_override",
                0,
            )
            or 0
        )
        if variant_timeout_override > 0:
            params.setdefault("variant_timeout_sec", variant_timeout_override)
        safety_margin_override = float(
            getattr(
                self.shared_state,
                "explore_variant_timeout_safety_margin",
                -1.0,
            )
        )
        if safety_margin_override >= 0:
            params.setdefault(
                "variant_timeout_safety_margin",
                safety_margin_override,
            )
        # Thread the persisted explore_search ledger so ExploreExecutor's dedup has cross-turn memory.
        es = getattr(self.shared_state, "explore_search", None)
        if isinstance(es, dict) and es.get("tested"):
            params.setdefault("explore_search", es)
        keep = self._decaying_keep_threshold_pct()
        if keep is not None:
            params.setdefault("keep_threshold_pct", keep)
            params.setdefault("stack_stable_threshold_pct", keep / 2.0)

    def _decaying_keep_threshold_pct(self) -> float | None:
        """Per-cycle KEEP threshold to inject, or ``None`` to keep executor defaults.

        Only active in cyclic mode: the bar shrinks along the shared decaying
        curve as macro-cycles accrue. Returns ``None`` off cyclic mode so
        non-cyclic runs fall back to the executor's fixed default.

        Returns:
            The decayed per-cycle KEEP threshold percentage, or ``None`` off
            cyclic mode (keep executor defaults).
        """
        if not _phase_state.is_cyclic_phases_enabled():
            return None
        from ..actions.executors._multi_node_env import is_multi_node

        cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        return _phase_state.decaying_keep_threshold_pct(
            cycle,
            multi_node=is_multi_node(),
        )

    async def _materialize_approved_proposal(
        self,
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        """Promote an approved proposal into a TaskRegistry entry. Grid executors get current best tput as base_tput; approved_variant_names filters the explore grid (None keeps full).

        Args:
            pending: The approved proposal to materialise into a task.
            approved_variant_names: When set, restricts an explore grid to these
                Critic-approved variant names; ``None`` keeps the full grid.
        """
        # Route framework_agent candidates to their own materializer.
        if pending.action_name == "framework_agent":
            await self._materialize_framework_agent_candidate(pending)
            return
        params = dict(pending.payload.get("params") or {})
        # Carry the proposer's predicted gain onto the task for predicted-vs-realized calibration.
        if pending.predicted_gain_pct:
            params.setdefault(
                "predicted_gain_pct", float(pending.predicted_gain_pct),
            )
        # Filter the grid to the Critic-approved subset.
        if pending.action_name == "explore" and isinstance(params.get("grid"), list):
            stamped_grid: list[dict[str, Any]] = []
            for variant in params["grid"]:
                if not isinstance(variant, dict):
                    # Non-dict slots can't carry a name: dropped under a filter, pass-through otherwise.
                    if approved_variant_names is None:
                        stamped_grid.append(variant)
                    continue
                vname = str(variant.get("name") or "").strip()
                # Drop variants the Critic rejected before they hit the executor.
                if approved_variant_names is not None and vname not in approved_variant_names:
                    continue
                stamped_grid.append(dict(variant))
            params["grid"] = stamped_grid
            # Audit hint: how many variants the Critic filtered.
            if approved_variant_names is not None:
                original_grid_len = len(
                    [v for v in (pending.payload.get("params") or {}).get("grid", []) if isinstance(v, dict)]
                )
                params["critic_filtered_count"] = max(
                    0,
                    original_grid_len - len(approved_variant_names),
                )
        cb = self.shared_state.current_best or {}
        cb_args = str(cb.get("extra_server_args") or "") if isinstance(cb, dict) else ""
        if pending.action_name == "profile":
            # Stamp the server config that produced this trace.
            params.setdefault("base_extra_args", cb_args)
        if pending.action_name == "sweep":
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
            if self.shared_state.baseline_config_path:
                params.setdefault("config_path", self.shared_state.baseline_config_path)
        if pending.action_name == "explore":
            self._inject_explore_runtime_params(params)
            # Inject base_tput/base_extra_args tied to current_best (or baseline_tput) so _gain_pct resolves.
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
        if pending.action_name == "integrate_patch":
            keep = self._decaying_keep_threshold_pct()
            if keep is not None:
                params.setdefault("keep_threshold_pct", keep)
            # Seed the patched-eval server with the same base args/config every
            # other eval server uses, else it launches on bare framework defaults
            # and crashes at startup regardless of the patch.
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
            if self.shared_state.baseline_config_path:
                params.setdefault(
                    "config_path", self.shared_state.baseline_config_path
                )
        lanes, ttl = self._registry_lanes_ttl(pending.action_name)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=pending.action_name,
            params=params,
            idempotency_key=f"approved-{pending.proposal_msg_id}",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            # Key is unique per proposal; only a resume/replay collides.
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "proposal_materialize_skipped",
                    "reason": "duplicate_idempotency_key",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "task_id": task.task_id,
                    "task_state": task.state,
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                },
            )
            return
        # proposal_msg_id is the resume contract for the deferred queue (see replay_for_resume).
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "decision",
                {
                    "kind": "approved_proposal",
                    "task_id": task.task_id,
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                    "proposal_msg_id": pending.proposal_msg_id,
                },
            )
        )
        # Trace attribution: record proposal_msg_id -> task_id for the decision-trace collector.
        self._record_proposal_task_map(pending.proposal_msg_id, task.task_id)

    def _record_proposal_task_map(self, proposal_msg_id: str, task_id: str) -> None:
        """Append one ``{proposal_msg_id -> task_id}`` row to the trace map.

        Lets the collector recover which decision a Critic review served. No-op
        on empty ids; OSError swallowed so a trace write never breaks the loop.
        """
        if not proposal_msg_id or not task_id:
            return
        try:
            from ..trace.llm_trace import _now_iso
            from hyperloom.common.io import append_jsonl
            from hyperloom.inference_optimizer.session.session_paths import proposal_task_map_path
            path = proposal_task_map_path(self.session_dir)
            row = {
                "ts": _now_iso(),
                "proposal_msg_id": str(proposal_msg_id),
                "task_id": str(task_id),
            }
            append_jsonl(path, row, make_parents=True, sort_keys=True)
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug(
                "full-trace: proposal_task_map append failed for "
                "msg_id=%s task_id=%s", proposal_msg_id, task_id, exc_info=True,
            )
