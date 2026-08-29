# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    patch_owner_phase,
)
from hyperloom.orchestrator.knowledge.recipe_kb import recipe_canonical_id
from hyperloom.inference_optimizer.recipe_snapshot_constants import detect_framework_version
from ..phases import machine_state as _phase_state
from ..bus.message_bus import Message
from .coordinator_helpers import approved_proposal_idempotency_key
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import TERMINAL_STATES

if TYPE_CHECKING:
    from ..state.task_registry import Task

from .coordinator import (
    PendingProposal,
)
import logging as _logging

log = _logging.getLogger(__name__)

_MAX_IDEMPOTENCY_ATTEMPTS: int = 6


def apply_critic_grid_filter(
    params: dict[str, Any],
    *,
    original_grid: list[Any],
    approved_variant_names: set[str] | None,
) -> bool:
    """Restrict ``params['grid']`` to Critic-approved variant names.

    ``None`` leaves the grid untouched. Non-dict slots cannot carry a name:
    they pass through only when there is no filter.

    Args:
        params: Task params mutated in place; must already hold ``grid``.
        original_grid: The proposer's grid, used for the filter audit count.
        approved_variant_names: Names that may run; ``None`` keeps the full grid.

    Returns:
        ``False`` when a filter was set and no variant survived; ``True`` otherwise.
    """
    stamped_grid: list[Any] = []
    for variant in original_grid:
        if not isinstance(variant, dict):
            if approved_variant_names is None:
                stamped_grid.append(variant)
            continue
        vname = str(variant.get("name") or "").strip()
        if approved_variant_names is not None and vname not in approved_variant_names:
            continue
        stamped_grid.append(dict(variant))
    params["grid"] = stamped_grid
    if approved_variant_names is None:
        return True
    original_grid_len = len([v for v in original_grid if isinstance(v, dict)])
    params["critic_filtered_count"] = max(0, original_grid_len - len(stamped_grid))
    return bool(stamped_grid)


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
        """Return the workload's canonical seven-dimension Recipe identity.

        Returns:
            The canonical Recipe id used by warm-start and writeback.
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
        """Load the selected store's exact authority row for writes.

        Cached per tick to avoid repeated I/O during multi-variant KEEP batches.
        """
        if self.recipe_kb is None:
            return {}
        tick = int(getattr(self.shared_state, "tick", 0) or 0)
        cache = getattr(self, "_local_recipe_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == tick:
            return cache[1]
        try:
            row = (
                self.recipe_kb.get_authoritative_recipe(
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
        config = getattr(getattr(self, "knowledge_plane", None), "config", None)
        if getattr(getattr(config, "mode", None), "value", None) == "remote" or self.recipe_kb is None:
            return
        # See agentx_kb_write_blocked for why; this is one of three sinks.
        from hyperloom.orchestrator.actions.executors._workload_envs import (
            agentx_kb_write_blocked,
        )

        if agentx_kb_write_blocked(self.shared_state):
            log.info(
                "_kb_amend_recipe: skipped (AgentX). The recipe KB has no mode or "
                "workload dimension, so an agentic-replay throughput would overwrite "
                "a synthetic best_throughput and be tagged isl/osl=%s/%s.",
                getattr(self.shared_state, "isl", "?"),
                getattr(self.shared_state, "osl", "?"),
            )
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

        # Local mode reads the exact authority row before amending it.
        try:
            live = self.recipe_kb.get_authoritative_recipe(canonical_id=cid) or {}
        except Exception as exc:  # noqa: BLE001
            log.info(
                "_kb_amend_recipe: authority get_recipe failed (%s); proceeding with empty live",
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
            self.recipe_kb.put_recipe(**put_kwargs)
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
        baseline_accuracy = float(getattr(self.shared_state, "baseline_accuracy", 0.0) or 0.0)
        if baseline_accuracy > 0:
            params.setdefault("accuracy_baseline", baseline_accuracy)
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
        # Thread the persisted explore_search ledger so the executor seeds its
        # tested history; it is evidence only, not an eligibility gate.
        es = getattr(self.shared_state, "explore_search", None)
        if isinstance(es, dict) and es.get("tested"):
            params.setdefault("explore_search", es)
        keep = _phase_state.resolve_keep_threshold(self.shared_state)
        params.setdefault("keep_threshold_pct", keep)

    async def _materialize_approved_proposal(
        self,
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        """Promote an approved proposal into a TaskRegistry entry. Stack-aware actions get current_best's anchor and the base config it was measured on; approved_variant_names filters the explore grid (None keeps full).

        Args:
            pending: The approved proposal to materialise into a task.
            approved_variant_names: When set, restricts an explore grid to these
                Critic-approved variant names; ``None`` keeps the full grid.
        """
        # An upstream-PR candidate pre-screen is an ``integrate_patch`` proposal
        # carrying a candidate id at the top level (rather than params). It has
        # its own materializer because approval means "spend a bench on this
        # candidate", and the task params are built from the candidate row.
        if pending.action_name == "integrate_patch" and (pending.payload or {}).get("framework_agent_candidate_id"):
            await self._materialize_framework_agent_candidate(pending)
            return
        params = dict(pending.payload.get("params") or {})
        # Carry the proposer's predicted gain onto the task for predicted-vs-realized calibration.
        if pending.predicted_gain_pct:
            params.setdefault(
                "predicted_gain_pct",
                float(pending.predicted_gain_pct),
            )
        # Filter the grid to the Critic-approved subset.
        if pending.action_name == "explore" and isinstance(params.get("grid"), list):
            original_grid = list(params["grid"])
            if not apply_critic_grid_filter(
                params,
                original_grid=original_grid,
                approved_variant_names=approved_variant_names,
            ):
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_materialize_skipped",
                        "reason": "critic_filter_empty_grid",
                        "proposal_msg_id": pending.proposal_msg_id,
                        "action_name": pending.action_name,
                        "from_agent": pending.from_agent,
                    },
                )
                return
        if pending.action_name == "profile":
            # Stamp the server config that produced this trace.
            inject_stack_base_params(params, self.shared_state)
        if pending.action_name == "sweep":
            inject_stack_base_params(params, self.shared_state, anchor=True)
            if self.shared_state.baseline_config_path:
                params.setdefault("config_path", self.shared_state.baseline_config_path)
        if pending.action_name == "explore":
            self._inject_explore_runtime_params(params)
            inject_stack_base_params(params, self.shared_state, anchor=True)
        if pending.action_name == "integrate_patch":
            owner = patch_owner_phase(params)
            if not owner:
                specialist_task_id = str(params.get("specialist_task_id") or "")
                self.state.pending_proposals.pop(
                    pending.proposal_msg_id,
                    None,
                )
                if specialist_task_id:
                    self.shared_state.record_specialist_patch_verdict(
                        specialist_task_id,
                        "owner_missing",
                    )
                    try:
                        self.shared_state.save(self.session_dir)
                    except Exception:  # noqa: BLE001
                        log.exception(
                            "failed to persist terminal owner-missing verdict for specialist=%s",
                            specialist_task_id,
                        )
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_materialize_skipped",
                        "reason": "integrate_patch_owner_missing",
                        "proposal_msg_id": pending.proposal_msg_id,
                        "action_name": pending.action_name,
                        "from_agent": pending.from_agent,
                        "specialist_task_id": specialist_task_id,
                    },
                )
                return
            params["source_phase"] = owner
            params.setdefault("keep_threshold_pct", _phase_state.resolve_keep_threshold(self.shared_state))
            # Seed the patched-eval server with the same base args/config every
            # other eval server uses, else it launches on bare framework defaults
            # and crashes at startup regardless of the patch.
            inject_stack_base_params(params, self.shared_state, anchor=True)
            if self.shared_state.baseline_config_path:
                params.setdefault("config_path", self.shared_state.baseline_config_path)
        lanes, ttl = self._registry_lanes_ttl(pending.action_name)
        # Content-addressed so a batch of proposals that would launch identical
        # work collapses to one task; a terminated twin still gets a fresh key so
        # a legitimate retry after failure is never locked out.
        raw_key = approved_proposal_idempotency_key(pending.action_name, params)
        task = None
        was_existing = False
        for attempt in range(_MAX_IDEMPOTENCY_ATTEMPTS):
            idempotency_key = raw_key if attempt == 0 else f"{raw_key}-retry{attempt}"
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=pending.action_name,
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            if not was_existing:
                break
            if task.state not in TERMINAL_STATES:
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_materialize_skipped",
                        "reason": "duplicate_proposal_content",
                        "proposal_msg_id": pending.proposal_msg_id,
                        "task_id": task.task_id,
                        "task_state": task.state,
                        "action_name": pending.action_name,
                        "from_agent": pending.from_agent,
                    },
                )
                return
        else:
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "proposal_materialize_skipped",
                    "reason": "idempotency_key_exhausted",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "task_id": task.task_id if task is not None else "",
                    "task_state": task.state if task is not None else "",
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
                "full-trace: proposal_task_map append failed for msg_id=%s task_id=%s",
                proposal_msg_id,
                task_id,
                exc_info=True,
            )
