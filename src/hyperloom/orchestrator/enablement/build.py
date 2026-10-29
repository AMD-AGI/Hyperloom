# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop compiled-build escalation (rung 5) and build-outcome routing."""

from __future__ import annotations

import os
import sqlite3
from typing import TYPE_CHECKING, Any

from common.coerce import first_text
from common.fs_utils import read_text_nofollow
from common.gpu_identity import gfx_arch_for_gpu_type
from hyperloom.inference_optimizer.gpu_types import targeted_build_validated
from hyperloom.orchestrator.lever import LEVER_ENABLEMENT

from ..collaborator import CoordinatorCollaborator
from ..bringup import recorded_verdict, session_root
from .params import _enablement_carrier_params
from .generation_scan import GenerationScan

if TYPE_CHECKING:
    from ..state.task_registry import Task
    from .runtime.build_actions import TargetedBuildAction

import logging as _logging

log = _logging.getLogger(__name__)


def _repo_matches_targeted_build_component(repo_url: str, component: str) -> bool:
    """Return whether a repo name is compatible with a targeted-build recipe."""
    repo = (repo_url or "").strip().rstrip("/")
    if not repo:
        return True
    repo_name = repo.rsplit("/", 1)[-1].removesuffix(".git").lower()
    hints = {
        "aiter": ("aiter",),
        "framework_ext": ("aiter",),
        "sgl_kernel": ("sglang", "sgl-kernel", "sgl_kernel"),
        "vllm_source": ("vllm",),
    }
    return any(hint in repo_name for hint in hints.get(component, ()))


#: What ``_note_build_routed`` stamps on a row. A manifest entry is a routing
#: sentinel only if it carries one of these; a ``BuildResult.to_state`` attempt
#: row never does, and must not answer for a build nobody has routed.
_ROUTING_FIELDS: tuple[str, ...] = ("routed", "probe_task_id")


def _targeted_build_unavailable() -> bool:
    """The operator switched targeted builds off, or the run is multi-node, where none is attempted."""
    if os.environ.get("HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD", "").strip() == "1":
        return True
    from ..actions.executors._multi_node_env import is_multi_node

    return is_multi_node()


def _compiled_gap_component(framework: str, symbol: str, launch_log: str) -> str:
    """The component a compiled-miss build targets, picked from the failure evidence.

    sgl_kernel when the offending symbol or log names sgl-kernel; vllm_source when vLLM's own C extension is
    implicated; aiter for every other compiled-miss gap.
    """
    sym_lower = symbol.lower()
    log_lower = launch_log.lower()
    if "sgl_kernel" in sym_lower or "sgl-kernel" in sym_lower or "sgl_kernel" in log_lower:
        return "sgl_kernel"
    if framework == "vllm" and (
        "vllm/_c" in log_lower or "vllm.extension" in log_lower or "_c.so" in log_lower or "vllm._c" in sym_lower
    ):
        return "vllm_source"
    return "aiter"


class EnablementBuild(CoordinatorCollaborator):
    """Escalates to a compiled build and routes the result back into the lane."""

    def _refuse_build_on_unvalidated_board(self, capability: str | None) -> bool:
        """Refuse a targeted build on a board whose build path is unvalidated, logging it and telling the lane why."""
        gpu_type = (getattr(self.shared_state, "gpu_type", "") or "").strip().lower()
        if targeted_build_validated(gpu_type):
            return False
        log.warning(
            "ENABLEMENT: targeted_build skipped capability=%s gpu_type=%s (build path not validated on this board)",
            capability or "(none)",
            gpu_type,
        )
        self.shared_state.enablement.last_build_failure = {
            "failure_class": "preflight_toolchain",
            "failure_summary": f"targeted builds are not validated on {gpu_type}; no build was attempted",
        }
        return True

    async def _enqueue_on_validated_board(self, action: TargetedBuildAction) -> str:
        """Enqueue ``action`` unless this board's build path is unvalidated; the task id, or ``""``."""
        if self._refuse_build_on_unvalidated_board(action.capability):
            return ""
        return await self._coord.build_lifecycle.enqueue_targeted_build(action)

    async def maybe_escalate_to_targeted_build(
        self,
        launch_log: str,
        *,
        attempt: int = 0,
    ) -> None:
        """Enqueue a targeted build row when the residual gap is a compiled miss."""
        if _targeted_build_unavailable():
            return
        from hyperloom.common.failure_signature import (
            MISSING_MODEL_ARCH,
            MISSING_WEIGHT,
            NOT_IMPLEMENTED,
            is_targeted_build_candidate,
        )
        from .runtime.build_actions import TargetedBuildAction

        state = self.shared_state
        # The round's own filed observation; wrapper text is the fallback,
        # named by ``degraded`` in the log line below.
        verdict, loaded = recorded_verdict(
            state.enablement.launch_observation_path,
            wrapper_text=launch_log,
            session_dir=session_root(self),
        )
        signature = verdict.signature
        framework = first_text(getattr(state, "framework", "")).lower()
        gpu_type = first_text(getattr(state, "gpu_type", "")).lower()

        # Two escalation triggers: 1.
        is_compiled_gap = is_targeted_build_candidate(signature, launch_log)
        arch_stall = (
            signature.kind in (MISSING_MODEL_ARCH, MISSING_WEIGHT, NOT_IMPLEMENTED)
            and framework == "vllm"
            and attempt >= 1
        )
        if not is_compiled_gap and not arch_stall:
            return
        # Past the guard, a gap that is not a compiled miss is an arch stall.
        arch_only = not is_compiled_gap

        # Derive novelty fields from the current failure + session context.
        existing_stack = state.enablement.kept_stack_action or {}
        repo_url = first_text(existing_stack.get("repo_url"))
        ref = first_text(existing_stack.get("ref"))

        if arch_only:
            # Arch/weight deep-failure on vLLM after source patches: the fix is a from-source vLLM build that
            # natively implements the arch, not an aiter/sgl-kernel op build.
            component = "vllm_source"
            reason = (
                f"Rung-5 arch-stall auto-escalation: {signature.kind} persists on vLLM "
                f"after {attempt} source-patch attempt(s) — build vLLM from source"
            )
        else:
            component = _compiled_gap_component(framework, signature.offending_symbol, launch_log)
            reason = f"Rung-5 auto-escalation from {signature.kind}"

        source_pr_url = ""
        # When no operator-pinned/kept ref, try the top discovery candidate.
        if not ref:
            from .runtime.build_actions import resolve_build_ref

            _component_hints = {
                "aiter": ("aiter",),
                "sgl_kernel": ("sglang", "sgl-kernel", "sgl_kernel"),
                "vllm_source": ("vllm",),
            }
            hints = _component_hints[component]
            for _cand in list(state.enablement.candidate_refs):
                _repo, _ref, _pr_url = resolve_build_ref(str(_cand), repo_url)
                if not _ref:
                    continue
                _repo_lower = (_repo or _cand).lower()
                if not any(h in _repo_lower for h in hints):
                    continue
                repo_url = _repo
                ref = _ref
                source_pr_url = _pr_url
                break

        if not _repo_matches_targeted_build_component(repo_url, component):
            log.warning(
                "ENABLEMENT: targeted_build repo/component mismatch component=%s repo_url=%s",
                component,
                repo_url,
            )
            return
        action = TargetedBuildAction(
            gap_id=f"gap.enablement.{signature.kind}",
            framework=framework or "vllm",
            component=component,
            capability=signature.offending_symbol or signature.kind,
            reason=reason,
            repo_url=repo_url,
            ref=ref,
            gpu_arch=gfx_arch_for_gpu_type(gpu_type) or "",
            build_budget_sec=0,
            source_pr_url=source_pr_url,
        )
        task_id = await self._enqueue_on_validated_board(action)
        if task_id:
            from hyperloom.common.bringup import failure_digest

            log.info(
                "ENABLEMENT: enqueued targeted_build kind=%s component=%s arch_stall=%s "
                "gpu_arch=%s digest=%s degraded=%s task=%s",
                signature.kind,
                component,
                arch_only,
                action.gpu_arch,
                failure_digest(verdict.observation)[:12],
                loaded.degraded or "-",
                task_id,
            )

    async def maybe_enqueue_specialist_requested_build(
        self,
        *,
        task_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Enqueue a targeted build the enablement specialist explicitly requested."""
        state = self.shared_state
        marker_task_id = first_text(state.enablement.last_specialist_task_id)
        task_id = first_text(task_id, marker_task_id)
        if not task_id:
            return

        def _consume_marker() -> None:
            if marker_task_id == task_id:
                state.enablement.last_specialist_task_id = ""

        try:
            if _targeted_build_unavailable():
                _consume_marker()
                return
            if payload is None:
                done_path = self.session_dir / "runs" / "specialist" / task_id / "specialist_done.json"
                if not done_path.is_file():
                    return
                import json as _json

                try:
                    payload = _json.loads(read_text_nofollow(done_path))
                except (OSError, ValueError):  # unreadable or partial done file: retry next tick
                    return
            req = payload.get("needs_targeted_build") if isinstance(payload, dict) else None
            if not isinstance(req, dict) or not req or self._refuse_build_on_unvalidated_board(req.get("capability")):
                _consume_marker()
                return

            capability = first_text(req.get("capability"))
            action = self._specialist_build_action(req, capability)
            if action is None:
                _consume_marker()
                return
            build_task_id = await self._coord.build_lifecycle.enqueue_targeted_build(action)
            if build_task_id:
                _consume_marker()
                log.info(
                    "ENABLEMENT: enqueued specialist-requested targeted_build "
                    "component=%s capability=%s ref=%s task=%s",
                    action.component,
                    capability or "(none)",
                    action.ref or "(autoselect)",
                    build_task_id,
                )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            log.debug("enablement: specialist-requested build enqueue failed", exc_info=True)

    def _specialist_build_action(self, req: dict[str, Any], capability: str) -> TargetedBuildAction | None:
        """The targeted build a specialist request names, or ``None`` when its repo cannot build the component."""
        from .runtime.build_actions import (
            _COMPONENTS,
            TargetedBuildAction,
            resolve_build_ref,
        )

        state = self.shared_state
        component = first_text(req.get("component")).lower()
        if component not in _COMPONENTS:
            # A from-source framework build is the safest default for an arch/model request that named no valid
            # compiled component.
            component = "vllm_source"
        framework = first_text(getattr(state, "framework", "")).lower() or "vllm"
        gpu_type = first_text(getattr(state, "gpu_type", "")).lower()
        repo_url = first_text(req.get("repo_url"))
        ref = first_text(req.get("ref"))
        source_pr_url = ""
        # A PR-like ``ref``/``repo_url`` carries its own repo + provenance.
        candidate = ref or repo_url
        if candidate:
            _repo, _ref, _pr = resolve_build_ref(candidate, repo_url)
            repo_url = _repo or repo_url
            # Take the resolved ref verbatim, including empty.
            ref = _ref
            source_pr_url = _pr
        if not _repo_matches_targeted_build_component(repo_url, component):
            log.warning(
                "ENABLEMENT: specialist-requested build repo/component mismatch component=%s repo_url=%s",
                component,
                repo_url,
            )
            return None
        reason = first_text(req.get("reason")) or "specialist-requested targeted build"
        return TargetedBuildAction(
            gap_id=f"gap.enablement.{capability or component}",
            framework=framework,
            component=component,
            capability=capability or component,
            reason=f"specialist request: {reason}",
            repo_url=repo_url,
            ref=ref,
            gpu_arch=gfx_arch_for_gpu_type(gpu_type) or "",
            build_budget_sec=0,
            source_pr_url=source_pr_url,
        )

    async def maybe_route_build_outcomes(self) -> None:
        """Turn the oldest unrouted succeeded targeted_build row into a launch probe.

        A build settles no round; a failed one reaches the lane through ``last_build_failure`` in the next mandate.
        """
        built = [t for t in await self.tasks.by_state("succeeded") if t.kind == "targeted_build"]
        # Oldest first so a newer, already-routed build cannot hide an older one that still needs routing.
        for task in sorted(built, key=lambda t: str(getattr(t, "updated_at", "") or "")):
            task_id = str(getattr(task, "task_id", "") or "")
            # Skip rows already accounted for (tracked by enablement_build_manifest), unless what they were routed
            # to was cancelled before it ran, which leaves the build no more launched than an unrouted one.
            routed = self._build_routing_record(task_id)
            if routed is not None and not await self._build_probe_was_cancelled(routed):
                continue
            await self._route_succeeded_build(task, routed)
            return

    async def _route_succeeded_build(self, task: "Task", routed: dict[str, Any] | None) -> None:
        """Turn a succeeded targeted build into a launch probe."""
        task_id = str(getattr(task, "task_id", "") or "")
        attempt_root = str((getattr(task, "params", {}) or {}).get("attempt_root") or "")
        br = None
        if attempt_root:
            from .runtime.targeted_build import _load_result_json

            br = _load_result_json(attempt_root)

        if br is None or not br.ok or not br.runtime.to_runtime_override():
            log.info("ENABLEMENT: targeted_build artifact-unreadable task=%s; nothing to launch", task_id)
            self._note_build_routed(task_id)
            return

        log.info("ENABLEMENT: targeted_build probe-verified → enqueue launch probe task=%s", task_id)
        probe_tid, generation = await self._enqueue_build_launch_probe(
            task_id,
            br,
            generation=int((routed or {}).get("probe_generation") or 0),
        )
        if not probe_tid:
            return
        self._note_build_routed(task_id, probe_task_id=probe_tid, probe_generation=generation)

    def _build_routing_record(self, build_task_id: str) -> dict[str, Any] | None:
        """The record of what a build's outcome was already routed to, if any.

        Matched on a routing field as well as the id. ``BuildResult.to_state``
        writes no ``task_id`` today, so an attempt row cannot answer here by
        accident -- but that is an invariant of a serializer two packages away,
        and if it ever gains one, every completed build would read as already
        routed and its launch probe would never be enqueued. The lookup says
        what it is looking for instead of relying on what it will not find.
        """
        for entry in reversed(list(self.shared_state.enablement.build_manifest or [])):
            if not isinstance(entry, dict) or str(entry.get("task_id") or "") != build_task_id:
                continue
            if any(field in entry for field in _ROUTING_FIELDS):
                return entry
        return None

    def _note_build_routed(self, build_task_id: str, **fields: Any) -> None:
        """Record that a build's outcome has been routed, and to what.

        ``routed`` is stamped on both the append and the merge path, so a build
        routed with no fields of its own still says it was routed. A reader
        matching on the id alone could not tell that row from an attempt row
        that happens to carry the same id.
        """
        manifest = list(self.shared_state.enablement.build_manifest or [])
        for idx, entry in enumerate(manifest):
            if isinstance(entry, dict) and str(entry.get("task_id") or "") == build_task_id:
                manifest[idx] = {**entry, "routed": True, **fields}
                break
        else:
            manifest.append({"task_id": build_task_id, "routed": True, **fields})
        self.shared_state.enablement.build_manifest = manifest

    async def _build_probe_was_cancelled(self, routed: dict[str, Any]) -> bool:
        """Whether the probe a build was routed to was stopped before it ran."""
        from ..state.task_registry import TaskNotFound

        probe_tid = str(routed.get("probe_task_id") or "").strip()
        if not probe_tid:
            return False
        try:
            probe = await self.tasks.get(probe_tid)
        except TaskNotFound:
            # Pruned rather than cancelled; re-probing on a row that is gone would re-probe on every later tick too.
            return False
        return str(getattr(probe, "state", "") or "") == "cancelled"

    async def _enqueue_build_launch_probe(
        self,
        build_task_id: str,
        br: Any,
        *,
        generation: int = 0,
    ) -> tuple[str, int]:
        """Enqueue an integrate_patch launch probe for a verified build.

        Runs the built runtime through the enablement runnable gate without
        applying any patch.  The probe completes as an ordinary integrate_patch
        task whose enablement:True result is routed by the dispatcher through
        _maybe_rearm_authored_lane → maybe_rearm_enablement, producing a
        genuine KEEP/advanced/reverted outcome.

        The probe is what declares KEEP for a build, so it must not be opened
        into a session that cannot run it: the queue scan drops a queued row the
        wall-clock budget can no longer fit, and a probe cancelled that way
        leaves the build verified but never launched. So the same gate the scan
        asks is asked here first, and a denial enqueues nothing -- the build
        stays unrouted, and the tick or resume that can afford a probe opens one.

        Args:
            build_task_id: The verified build this probe launches.
            br: Its ``BuildResult``, read for the runtime override.
            generation: The probe generation to try first, from what this build
                was routed to before.

        Returns:
            The probe ``task_id`` and the generation it sits on; the id is empty
            when nothing was enqueued.
        """
        denied = self._coord.dispatcher.time_budget_denial_for_action("integrate_patch")
        if denied is not None:
            log.info(
                "ENABLEMENT: build launch probe held for build=%s, not enqueued -- %s",
                build_task_id,
                denied,
            )
            return "", generation
        state = self.shared_state
        runtime_override = br.runtime.to_runtime_override()
        params: dict[str, Any] = {
            "enablement": True,
            "enablement_launch_only": True,
            "lever_kind": LEVER_ENABLEMENT,
            "runtime_override": runtime_override,
            "framework": str(getattr(state, "framework", "") or ""),
            # The pre-patch half of the gate: the observation the round that
            # watched this failure persisted.
            "enablement_before_observation_path": state.enablement.launch_observation_path,
            "source": "coordinator_internal",
            **_enablement_carrier_params(state),
        }
        # Prefer the eval-origin probe config so the re-run keeps the original workload/eval contract; fall back to
        # the promoted baseline config.
        cfg = str(state.enablement.probe_config_path or "") or str(getattr(state, "baseline_config_path", "") or "")
        if cfg:
            params["config_path"] = cfg
        # The probe boots a server and mutates the tree, so it takes the lanes its own kind declares rather than a
        # specialist's research lane.
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("integrate_patch")
        if not lanes:
            raise RuntimeError("integrate_patch resolved to no lanes; the launch probe would run unserialised.")
        probe_task, generation = await self._coord.enablement_revalidation.open_row_past_spent_generations(
            kind="integrate_patch",
            params=params,
            scan=GenerationScan(
                key_for=lambda gen: f"build_launch_probe:{build_task_id}:gen{gen}",
                generation=generation,
            ),
            label="build launch probe",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if probe_task is None:
            return "", generation
        probe_tid = str(getattr(probe_task, "task_id", "") or "")
        log.info(
            "ENABLEMENT: build launch probe task=%s gen%d state=%s (build=%s)",
            probe_tid,
            generation,
            getattr(probe_task, "state", ""),
            build_task_id,
        )
        return probe_tid, generation
