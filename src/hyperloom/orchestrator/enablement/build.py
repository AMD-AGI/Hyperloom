# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop compiled-build escalation (rung 5) and build-outcome routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_ENABLEMENT

from ..collaborator import CoordinatorCollaborator
from .params import _enablement_carrier_params

if TYPE_CHECKING:
    from ..state.task_registry import Task

import logging as _logging

log = _logging.getLogger(__name__)


def _derive_gpu_arch(gpu_type: str) -> str:
    """Map a gpu_type label to an explicit GFX arch (never silent fallback)."""
    _MAP = {
        "mi355x": "gfx950",
        "mi300x": "gfx942",
        "mi308x": "gfx942",
        "mi300": "gfx942",
        "mi250x": "gfx90a",
        "mi250": "gfx90a",
        "mi210": "gfx90a",
        "radeon8060s": "gfx1151",
        "radeon 8060s": "gfx1151",
        "strix halo": "gfx1151",
        "gfx1151": "gfx1151",
    }
    gt = (gpu_type or "").strip().lower()
    for key, arch in _MAP.items():
        if key in gt:
            return arch
    return ""


def _repo_matches_targeted_build_component(repo_url: str, component: str) -> bool:
    """Return whether a repo name is compatible with a targeted-build recipe.

    This deliberately ignores the origin/owner: specialists may discover and
    request forks on any host.  It only prevents routing an unrelated repository
    into a component-specific recipe.  An empty URL is valid because each recipe
    has its own built-in default repository.
    """
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


class EnablementBuild(CoordinatorCollaborator):
    """Escalates to a compiled build and routes the result back into the lane."""

    async def _maybe_escalate_to_targeted_build(
        self,
        launch_log: str,
        *,
        attempt: int = 0,
    ) -> None:
        """Enqueue a targeted build row when the residual gap is a compiled miss.

        No-op when a build is already queued or running (idempotent by novelty
        key), when the env var ``HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD=1``
        is set, on multi-node, or when the gap is not a compiled miss.
        """
        import os as _os

        if _os.environ.get("HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD", "").strip() == "1":
            return
        try:
            from ..actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return
        except Exception:  # noqa: BLE001
            return
        try:
            from hyperloom.agents.framework.enablement import (
                MISSING_MODEL_ARCH,
                MISSING_WEIGHT,
                NOT_IMPLEMENTED,
                classify_failure,
                is_targeted_build_candidate,
            )
            from ..framework.build_actions import TargetedBuildAction

            signature = classify_failure(launch_log)

            state = self.shared_state
            framework = (getattr(state, "framework", "") or "").strip().lower()
            gpu_type = (getattr(state, "gpu_type", "") or "").strip().lower()

            # Two escalation triggers:
            #  1. An inherently *compiled* gap (build / rocm_hip / native-dtype /
            #     hip-kernel) — the original Rung-5 path.
            #  2. A vLLM **arch/weight deep-failure that source patches keep
            #     hitting**: the enablement specialist authored ≥1 source patch
            #     (attempt >= 1) yet the boot still stops at an arch/weight/
            #     not-implemented wall. For a genuinely new architecture, aliasing
            #     to an existing model class in the *installed* vLLM cannot model
            #     the new op set; the correct acquisition is a from-source vLLM
            #     build of a version that natively implements the arch. Only fires
            #     on vLLM (the from-source recipe target) and never on the first
            #     attempt (give the cheap source-patch path a chance first).
            is_compiled_gap = is_targeted_build_candidate(signature, launch_log)
            arch_stall = (
                signature.kind in (MISSING_MODEL_ARCH, MISSING_WEIGHT, NOT_IMPLEMENTED)
                and framework == "vllm"
                and int(attempt or 0) >= 1
            )
            if not is_compiled_gap and not arch_stall:
                return

            # Derive novelty fields from the current failure + session context.
            # Ref and repo_url come from the existing stack action when present;
            # otherwise the top discovery candidate for the component is used;
            # empty ref falls back to tag-descending autoselect.
            existing_stack = state.enablement.kept_stack_action or {}
            repo_url = str(existing_stack.get("repo_url") or "").strip()
            ref = str(existing_stack.get("ref") or "").strip()

            # Pick the component from the failure evidence:
            # sgl_kernel when the offending symbol/log names sgl-kernel;
            # vllm_source when vLLM's own C extension is implicated;
            # aiter (default) for all other compiled-miss gaps.
            sym_lower = (signature.offending_symbol or "").lower()
            log_lower = launch_log.lower()
            if arch_stall and not is_compiled_gap:
                # Arch/weight deep-failure on vLLM after source patches: the fix
                # is a from-source vLLM build that natively implements the arch,
                # not an aiter/sgl-kernel op build.
                component = "vllm_source"
            elif "sgl_kernel" in sym_lower or "sgl-kernel" in sym_lower or "sgl_kernel" in log_lower:
                component = "sgl_kernel"
            elif framework == "vllm" and (
                "vllm/_c" in log_lower
                or "vllm.extension" in log_lower
                or "_c.so" in log_lower
                or "vllm._c" in sym_lower
            ):
                component = "vllm_source"
            else:
                component = "aiter"

            source_pr_url = ""
            # When no operator-pinned/kept ref, try the top discovery candidate.
            if not ref:
                from ..framework.build_actions import resolve_build_ref

                _component_hints = {
                    "aiter": ("aiter",),
                    "sgl_kernel": ("sglang", "sgl-kernel", "sgl_kernel"),
                    "vllm_source": ("vllm",),
                    "framework_ext": (),
                }
                hints = _component_hints.get(component, ())
                default_repo = repo_url or ""
                for _cand in list(state.enablement.candidate_refs or []):
                    _repo, _ref, _pr_url = resolve_build_ref(str(_cand), default_repo)
                    if not _ref:
                        continue
                    if hints:
                        _repo_lower = (_repo or _cand).lower()
                        if not any(h in _repo_lower for h in hints):
                            continue
                    repo_url = _repo or repo_url
                    ref = _ref
                    source_pr_url = _pr_url
                    break

            reason = (
                f"Rung-5 arch-stall auto-escalation: {signature.kind} persists on vLLM "
                f"after {int(attempt or 0)} source-patch attempt(s) — build vLLM from source"
                if (arch_stall and not is_compiled_gap)
                else f"Rung-5 auto-escalation from {signature.kind}"
            )
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
                capability=str(signature.offending_symbol or signature.kind or ""),
                reason=reason,
                repo_url=repo_url,
                ref=ref,
                gpu_arch=_derive_gpu_arch(gpu_type),
                build_budget_sec=0,
                source_pr_url=source_pr_url,
            )
            task_id = await self.enqueue_targeted_build(action)
            if task_id:
                log.info(
                    "ENABLEMENT: enqueued targeted_build kind=%s component=%s arch_stall=%s gpu_arch=%s task=%s",
                    signature.kind,
                    component,
                    bool(arch_stall and not is_compiled_gap),
                    action.gpu_arch,
                    task_id,
                )
        except Exception:  # noqa: BLE001 — escalation is best-effort; never wedge dispatch
            log.debug("enablement: targeted-build escalation failed", exc_info=True)

    async def _maybe_enqueue_specialist_requested_build(
        self,
        *,
        task_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Enqueue a targeted build the enablement specialist explicitly requested.

        The enablement specialist may emit a ``needs_targeted_build`` object in its
        ``specialist_done.json`` (see ``ENABLEMENT_BUILD_REQUEST_GUIDANCE``) when a
        compiled component / from-source framework build is required that a source
        patch against the installed tree cannot deliver. This reads that request
        from the just-finished round's workdir (task id captured at rearm into
        ``enablement_last_specialist_task_id``) and enqueues it on the isolated,
        ROCm-safe build lane. The field is cleared once read so the request is
        consumed at most once; ``enqueue_targeted_build`` is additionally
        idempotent by build-novelty key. Best-effort — never wedges dispatch.
        """
        import os as _os

        state = self.shared_state
        marker_task_id = str(state.enablement.last_specialist_task_id or "").strip()
        task_id = str(task_id or marker_task_id).strip()
        if not task_id:
            return

        def _consume_marker() -> None:
            if marker_task_id == task_id:
                state.enablement.last_specialist_task_id = ""

        try:
            if _os.environ.get("HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD", "").strip() == "1":
                _consume_marker()
                return
            from ..actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                _consume_marker()
                return
            if payload is None:
                done_path = self.session_dir / "runs" / "specialist" / task_id / "specialist_done.json"
                if not done_path.is_file():
                    return
                import json as _json

                try:
                    payload = _json.loads(done_path.read_text())
                except Exception:  # noqa: BLE001 — malformed/partial done is non-fatal
                    return
            req = payload.get("needs_targeted_build") if isinstance(payload, dict) else None
            if not isinstance(req, dict) or not req:
                _consume_marker()
                return

            from ..framework.build_actions import (
                _COMPONENTS,
                TargetedBuildAction,
                resolve_build_ref,
            )

            component = str(req.get("component") or "").strip().lower()
            if component not in _COMPONENTS:
                # A from-source framework build is the safest default for an
                # arch/model request that named no valid compiled component.
                component = "vllm_source"
            framework = (getattr(state, "framework", "") or "").strip().lower() or "vllm"
            gpu_type = (getattr(state, "gpu_type", "") or "").strip().lower()
            repo_url = str(req.get("repo_url") or "").strip()
            ref = str(req.get("ref") or "").strip()
            source_pr_url = ""
            # A PR-like ``ref``/``repo_url`` carries its own repo + provenance.
            candidate = ref or repo_url
            if candidate:
                _repo, _ref, _pr = resolve_build_ref(candidate, repo_url)
                repo_url = _repo or repo_url
                # Take the resolved ref verbatim, including empty. An empty ref
                # means "not checkoutable, autoselect a tag" (an issue citation),
                # so falling back to the raw request here would hand the builder
                # back the very string resolution just rejected.
                ref = _ref
                source_pr_url = _pr
            if not _repo_matches_targeted_build_component(repo_url, component):
                log.warning(
                    "ENABLEMENT: specialist-requested build repo/component mismatch component=%s repo_url=%s",
                    component,
                    repo_url,
                )
                _consume_marker()
                return
            capability = str(req.get("capability") or "").strip()
            reason = str(req.get("reason") or "").strip() or "specialist-requested targeted build"
            action = TargetedBuildAction(
                gap_id=f"gap.enablement.{capability or component}",
                framework=framework,
                component=component,
                capability=capability or component,
                reason=f"specialist request: {reason}",
                repo_url=repo_url,
                ref=ref,
                gpu_arch=_derive_gpu_arch(gpu_type),
                build_budget_sec=0,
                source_pr_url=source_pr_url,
            )
            build_task_id = await self.enqueue_targeted_build(action)
            if build_task_id:
                _consume_marker()
                log.info(
                    "ENABLEMENT: enqueued specialist-requested targeted_build "
                    "component=%s capability=%s ref=%s task=%s",
                    component,
                    capability or "(none)",
                    ref or "(autoselect)",
                    build_task_id,
                )
        except Exception:  # noqa: BLE001 — best-effort; never wedge dispatch
            log.debug("enablement: specialist-requested build enqueue failed", exc_info=True)

    async def _maybe_route_build_outcomes(self) -> None:
        """Route terminal targeted_build rows to _maybe_rearm_enablement.

        Called every tick from _pump_enablement_safely.  Reads succeeded/failed
        rows, synthesises the rearm res dict (status='kept'/'reverted'/'advanced'),
        and delegates to the existing stall-gate machinery.

        Novelty: a 'timeout' or 'preflight_budget' failure_class maps to
        'advanced' (novel attempt, time vs defect distinction — keep going); all
        real defects map to 'reverted' (advance stall streak).

        A succeeded build no longer synthesises status='kept' directly. Instead
        it enqueues an integrate_patch launch probe so the runtime must actually
        boot the model before KEEP is declared. Each row is read once, except that
        a build whose probe was cancelled before it ran is read again: nothing
        launched the runtime, so nothing has decided anything about that build.

        Oldest-unrouted-first avoids starving older builds when a newer row is
        already routed; the tradeoff is that a succeeded build whose probe cannot
        be enqueued yet is retried every tick and can defer newer failed builds
        until its probe opens or the budget recovers.
        """
        try:
            all_tasks = []
            for st in ("succeeded", "failed"):
                all_tasks.extend(t for t in await self.tasks.by_state(st) if t.kind == "targeted_build")
            if not all_tasks:
                return
            # Oldest terminal row first so a newer, already-routed build cannot
            # hide an older build that still needs routing.
            for task in sorted(
                all_tasks,
                key=lambda t: str(getattr(t, "updated_at", "") or ""),
            ):
                task_id = str(getattr(task, "task_id", "") or "")
                # Skip rows already accounted for (tracked by enablement_build_manifest),
                # unless what they were routed to was cancelled before it ran, which
                # leaves the build no more launched than an unrouted one.
                routed = self._build_routing_record(task_id)
                if routed is not None and not await self._build_probe_was_cancelled(routed):
                    continue

                if task.state == "succeeded":
                    await self._route_succeeded_build(task, routed)
                else:
                    await self._route_failed_build(task)
                return
        except Exception:  # noqa: BLE001 — never wedge the tick
            log.debug("enablement: route_build_outcomes failed", exc_info=True)

    async def _route_failed_build(self, task: "Task") -> None:
        """Route a failed targeted_build row through the enablement rearm path."""
        task_id = str(getattr(task, "task_id", "") or "")
        state = self.shared_state
        fc = ""
        # failed row — read failure_class from history or last_build_failure
        history = getattr(task, "history", None) or []
        if isinstance(history, (list, tuple)) and history:
            last_ev = history[-1]
            if isinstance(last_ev, dict):
                fc = str(last_ev.get("evidence", {}).get("failure_class") or "")

        lbf = state.enablement.last_build_failure or {}
        if not fc and isinstance(lbf, dict):
            fc = str(lbf.get("failure_class") or "")

        # Novelty ledger: time-based failures are always advanced; defect
        # failures are advanced when the (component,ref,gpu_arch,cmd) tuple
        # has not been seen before (novel), reverted when it is a repeat.
        time_classes = frozenset({"timeout", "preflight_budget", "preflight_disk", "preflight_toolchain"})
        novelty_key: list[Any] | None = None
        if fc in time_classes:
            new_log = str(state.enablement.launch_log or "")
            res = {
                "enablement": True,
                "status": "advanced",
                "advanced": True,
                "patches_applied": [],
                "enablement_launch_log": new_log,
            }
        else:
            from ..framework.build_actions import TargetedBuildAction as _TBA, build_novelty_key as _bnk

            task_params = getattr(task, "params", None) or {}
            _action = _TBA.from_state(task_params)
            _key = list(_bnk(_action))
            ledger = list(state.enablement.build_novelty or [])
            is_repeat = any(entry == _key for entry in ledger if isinstance(entry, list))
            if is_repeat:
                res = {"enablement": True, "status": "reverted"}
                novelty_key = None
            else:
                new_log = str(state.enablement.launch_log or "")
                res = {
                    "enablement": True,
                    "status": "advanced",
                    "advanced": True,
                    "patches_applied": [],
                    "enablement_launch_log": new_log,
                }
                novelty_key = _key
        log.info(
            "ENABLEMENT: targeted_build %s task=%s failure_class=%r",
            res["status"],
            task_id,
            fc,
        )
        # Rearm, ledger append, and manifest ack must stay together: a failed
        # rearm leaves the build unrouted and the novelty ledger unchanged.
        self._maybe_rearm_enablement(res)
        if novelty_key is not None:
            ledger = list(state.enablement.build_novelty or [])
            ledger.append(novelty_key)
            state.enablement.build_novelty = ledger[-20:]
        self._note_build_routed(task_id)

    async def _route_succeeded_build(self, task: "Task", routed: dict[str, Any] | None) -> None:
        """Turn a succeeded targeted build into a launch probe, or a no-progress round.

        KEEP is declared by the probe, never here: an artifact that builds is not
        a runtime that boots. So this either opens a probe -- and remembers which
        one, because a probe cancelled before it ran leaves the build unlaunched
        and worth another -- or, when the built runtime cannot even be read, ends
        the round as a revert.

        The build is recorded as accounted for only once there is something to
        account for. A probe the budget refused leaves it unrouted on purpose: it
        is still a probed build nothing has launched, and the manifest saying
        otherwise is what would strand it for the rest of the session.

        Args:
            task: The succeeded ``targeted_build`` row.
            routed: What this build was routed to before, when it is being routed
                again after its probe was cancelled; ``None`` on the first pass.
        """
        task_id = str(getattr(task, "task_id", "") or "")
        attempt_root = str((getattr(task, "params", {}) or {}).get("attempt_root") or "")
        # The build's attempt_root is resolved at pump time and is NOT written
        # back into the task params (they keep the enqueue-time default ""). Fall
        # back to the deterministic build path keyed by task_id so a *successful*
        # build is not wrongly rejected as ``artifact_unreadable`` (mirrors
        # BuildLifecycle._attempt_root).
        if not attempt_root and task_id:
            attempt_root = str(self.session_dir / "enablement" / "builds" / task_id)
        br = None
        if attempt_root:
            from ..framework.targeted_build import _load_result_json

            br = _load_result_json(attempt_root)

        # If the runtime can't be read, it can't be launched → reverted.
        if br is None or not br.ok or not br.runtime.to_runtime_override():
            log.info("ENABLEMENT: targeted_build artifact-unreadable task=%s", task_id)
            self._maybe_rearm_enablement({"enablement": True, "status": "reverted", "reason": "artifact_unreadable"})
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

        Only the routing sentinels carry a ``task_id``; the build attempts the
        lifecycle appends to the same manifest carry ``ok`` instead.

        Args:
            build_task_id: The ``targeted_build`` row to look for.

        Returns:
            The sentinel dict, or ``None`` when this build has not been routed.
        """
        for entry in reversed(list(self.shared_state.enablement.build_manifest or [])):
            if isinstance(entry, dict) and str(entry.get("task_id") or "") == build_task_id:
                return entry
        return None

    def _note_build_routed(self, build_task_id: str, **fields: Any) -> None:
        """Record that a build's outcome has been routed, and to what.

        Args:
            build_task_id: The ``targeted_build`` row that was routed.
            fields: What it was routed to, for a build whose outcome is a probe.
        """
        manifest = list(self.shared_state.enablement.build_manifest or [])
        for idx, entry in enumerate(manifest):
            if isinstance(entry, dict) and str(entry.get("task_id") or "") == build_task_id:
                manifest[idx] = {**entry, **fields}
                break
        else:
            manifest.append({"task_id": build_task_id, "routed": True, **fields})
        self.shared_state.enablement.build_manifest = manifest

    async def _build_probe_was_cancelled(self, routed: dict[str, Any]) -> bool:
        """Whether the probe a build was routed to was stopped before it ran.

        A cancelled probe is no evidence about the build: the queue scan drops a
        queued row the wall-clock budget can no longer fit, and a phase boundary
        drops one the new phase does not allow. Either way the built runtime was
        never launched, so the build is still owed a probe -- and without noticing
        that, the manifest entry written when the first one was opened keeps this
        build accounted for permanently, across resumes included.

        Args:
            routed: The build's routing record.

        Returns:
            ``True`` when the recorded probe exists and was cancelled.
        """
        from ..state.task_registry import TaskNotFound

        probe_tid = str(routed.get("probe_task_id") or "").strip()
        if not probe_tid:
            return False
        try:
            probe = await self.tasks.get(probe_tid)
        except TaskNotFound:
            # Pruned rather than cancelled; re-probing on a row that is gone
            # would re-probe on every later tick too.
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
        _maybe_rearm_authored_lane → _maybe_rearm_enablement, producing a
        genuine KEEP/advanced/reverted outcome.  The whole-machine GPU pool is
        acquired via _framework_gpu_params.

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
        from hyperloom.agents.framework.enablement import classify_failure

        denied = self._time_budget_denial_for_action("integrate_patch")
        if denied is not None:
            log.info(
                "ENABLEMENT: build launch probe held for build=%s, not enqueued -- %s",
                build_task_id,
                denied,
            )
            return "", generation
        state = self.shared_state
        runtime_override = br.runtime.to_runtime_override()
        launch_log = str(state.enablement.launch_log or "")
        before_sig = classify_failure(launch_log).to_dict()
        params: dict[str, Any] = {
            "enablement": True,
            "enablement_launch_only": True,
            "lever_kind": LEVER_ENABLEMENT,
            "runtime_override": runtime_override,
            "framework": str(getattr(state, "framework", "") or ""),
            "enablement_before_signature": before_sig,
            "source": "coordinator_internal",
            **self._framework_gpu_params(),
            **_enablement_carrier_params(state),
        }
        # Prefer the eval-origin probe config so the re-run keeps the original
        # workload/eval contract; fall back to the promoted baseline config.
        cfg = str(state.enablement.probe_config_path or "") or str(getattr(state, "baseline_config_path", "") or "")
        if cfg:
            params["config_path"] = cfg
        # The probe boots a server and mutates the tree, so it takes the lanes
        # its own kind declares rather than a specialist's research lane.
        lanes, ttl = self._registry_lanes_ttl("integrate_patch")
        if not lanes:
            raise RuntimeError("integrate_patch resolved to no lanes; the launch probe would run unserialised.")
        probe_task, generation = await self._open_row_past_spent_generations(
            kind="integrate_patch",
            params=params,
            key_for=lambda gen: f"build_launch_probe:{build_task_id}:gen{gen}",
            generation=generation,
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
