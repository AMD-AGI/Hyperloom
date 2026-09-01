# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement round admission, in-flight tracking and re-arm."""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from ..actions.executors._grid_server_args import merge_server_args
from ..collaborator import CoordinatorCollaborator
from ..loop.coordinator import _ENABLEMENT_MAX_STALL
from ..loop.coordinator_helpers import _dedupe_extra_server_args
from ..phases._enablement_artifacts import snapshot_round, write_setting_script

import logging as _logging

log = _logging.getLogger(__name__)


class EnablementLane(CoordinatorCollaborator):
    """Owns one enablement round: admit, track in-flight, re-arm on outcome."""

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        """Dispatch an enablement_specialist when a baseline cannot launch or its
        accuracy eval fails.

        Retries until the combo runs correctly or the run wall-clock deadline
        passes (no attempt-count cap). Guards:

        * ``enablement_mode`` — the lane matching the trigger origin must be
          admitted by ``--enablement``; otherwise no round is ever opened.
        * ``enablement_succeeded`` — terminal: a prior attempt was KEPT.
        * ``enablement_validation_pending`` — an eval-origin KEEP is awaiting
          genuine-baseline revalidation; authoring is paused until it resolves.
        * ``inflight_task_id`` — a round is still resolving (specialist task or
          the integrate that consumes its deliverable); derived, never stored.
        * run deadline passed — stop dispatching new work near the close.

        A non-blank log is always dispatched (the specialist repairs from the raw
        log even when it classifies to ``UNKNOWN``); only a blank log is a no-op,
        recorded once as ``needs_human_review``. No-op on multi-node.

        Returns:
            str: The dispatched specialist ``task_id`` (empty when skipped).
        """
        from ..actions.executors._accuracy_gate import eval_enablement_allowed, launch_enablement_allowed

        state = self.shared_state
        origin = str(state.enablement.origin or "")
        admitted = eval_enablement_allowed(state) if origin == "eval" else launch_enablement_allowed(state)
        if not admitted:
            return ""
        if bool(state.enablement.succeeded):
            return ""
        if bool(state.enablement.validation_pending):
            # A KEEP'd eval-origin patch is awaiting genuine-baseline revalidation;
            # do not dispatch another authoring round until that resolves.
            return ""
        if state.enablement.inflight_task_id:
            if await self._enablement_in_flight():
                return ""
            # Round ended without calling _maybe_rearm_enablement — count as stall.
            self._maybe_rearm_enablement(
                {"enablement": True, "status": "reverted", "reason": "round_finished_without_rearm"}
            )
            if state.stop_reason:
                return ""
        if float(getattr(state, "baseline_tput", 0.0) or 0.0) > 0:
            return ""
        if int(getattr(state, "baseline_failure_streak", 0) or 0) < 1:
            return ""
        # Stop opening new enablement attempts once the run deadline has passed.
        deadline = getattr(self, "_run_deadline", None)
        if deadline is not None and time.monotonic() >= float(deadline):
            return ""
        launch_log = str(state.enablement.launch_log or "")
        attempt = int(state.enablement.attempts or 0)
        params = self._build_enablement_specialist_params(launch_log, attempt=attempt)
        if params is None:
            # A non-blank UNKNOWN log is recorded for human review, once per log.
            await self._maybe_record_enablement_human_review(launch_log)
            return ""
        # Enqueue any build the *previous* round's specialist explicitly requested
        # (``needs_targeted_build`` in its specialist_done), then auto-escalate to
        # a targeted build when the residual gap is a compiled miss or a vLLM
        # arch/weight deep-failure that source patches keep hitting. Both enqueues
        # are no-ops when a matching build is already queued/running (idempotent by
        # novelty key).
        try:
            await self._maybe_enqueue_specialist_requested_build()
        except Exception:  # noqa: BLE001 — best-effort; never wedge dispatch
            log.debug("enablement: specialist-requested build raised", exc_info=True)
        try:
            await self._maybe_escalate_to_targeted_build(launch_log, attempt=attempt)
        except Exception:  # noqa: BLE001 — escalation is best-effort; never wedge dispatch
            log.debug("enablement: targeted-build escalation raised", exc_info=True)
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return ""
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug("enablement: warm specialist params failed", exc_info=True)
        idem = f"enablement_authoring:{params.get('enablement_failure_kind', '')}:{attempt}"
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _existing = await self.tasks.create_or_return_existing(
            kind="specialist",
            params=params,
            idempotency_key=idem,
            requires_lanes=lanes,
            side_effects=["writes_results", "writes_patches"],
            lease_ttl_sec=ttl,
        )
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        state.enablement.attempts = attempt + 1
        state.enablement.inflight_task_id = spec_tid
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("enablement: save after dispatch failed", exc_info=True)
        log.info(
            "ENABLEMENT: dispatched authoring specialist kind=%s attempt=%d task=%s",
            params.get("enablement_failure_kind"),
            attempt + 1,
            spec_tid,
        )
        return spec_tid

    async def _enablement_in_flight(self) -> bool:
        """True while the current enablement round has not settled.

        A round spans the authoring specialist AND the ``integrate_patch`` that
        consumes its deliverable; the specialist goes terminal a tick before the
        Critic sees the integrate proposal, so its task state alone is not enough.
        """
        from ..state.task_registry import TaskNotFound

        tid = self.shared_state.enablement.inflight_task_id
        if not tid:
            return False
        try:
            spec = await self.tasks.get(tid)
        except TaskNotFound:
            spec = None
        if spec is not None and spec.state in ("queued", "running"):
            return True
        # Undecided proposal keeps the round open; once ruled, approve lands the
        # task matched below and reject rearms directly, so it cannot defer forever.
        for p in self.state.pending_proposals.values():
            if p.action_name != "integrate_patch" or p.decided:
                continue
            if (p.payload.get("params") or {}).get("specialist_task_id") == tid:
                return True
        for t in (await self.tasks.queued()) + (await self.tasks.running()):
            if t.kind == "integrate_patch" and t.params.get("specialist_task_id") == tid:
                return True
        return False

    async def _maybe_record_enablement_human_review(self, launch_log: str) -> None:
        """Record a one-shot ``needs_human_review`` for an UNKNOWN launch failure.

        The enablement path only dispatches authoring for *actionable* failure
        signatures; a non-blank log that classifies to ``UNKNOWN`` used to be
        silently dropped. Instead, emit a single observation
        (deduped per distinct log via a stored hash) carrying the classified
        signature (``raw_excerpt`` + ``offending_file``) so an operator can pick
        it up. No sub-agent is dispatched.

        Args:
            launch_log: The captured launch / traceback text.
        """
        text = (launch_log or "").strip()
        if not text:
            return

        from hyperloom.agents.framework.enablement import classify_failure

        signature = classify_failure(text)
        if signature.is_actionable:
            return
        digest = hashlib.sha1(text.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()
        state = self.shared_state
        seen = state.enablement.human_review_logged
        if not isinstance(seen, list):
            seen = []
            state.enablement.human_review_logged = seen
        if digest in seen:
            return
        seen.append(digest)
        framework = (getattr(state, "framework", "") or "").strip().lower()
        model = (getattr(state, "model_name", "") or "").strip()
        try:
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "enablement_needs_human_review",
                    "applicability": "needs_human_review",
                    "framework": framework,
                    "model": model,
                    "failure_kind": signature.kind,
                    "signature": signature.to_dict(),
                    "reason": (
                        "baseline launch failure did not match any actionable enablement signature; needs human triage"
                    ),
                },
            )
        except Exception:  # noqa: BLE001 — observability is best-effort
            log.debug("enablement: human-review record failed", exc_info=True)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("enablement: save after human-review failed", exc_info=True)
        log.info(
            "ENABLEMENT: recorded needs_human_review for UNKNOWN failure kind=%s",
            signature.kind,
        )

    def _maybe_rearm_enablement(self, res: dict[str, Any] | None) -> None:
        """Re-arm, advance, or terminate the enablement retry loop.

        Called on every ``integrate_patch`` completion. For an enablement patch
        there are three outcomes:

        * ``kept`` — the combo is now fully runnable: terminal success
          (``enablement_succeeded=True``).
        * ``advanced`` — the patch cleared the prior crash and the boot now
          stops at a *new, deeper* gap (serial enablement). **Stack** the patch
          (append to ``enablement_kept_patches``), replace
          ``enablement_launch_log`` with the new failure so the next round
          classifies and targets gap #(n+1), reset the stall streak, and clear
          the in-flight guard to dispatch the next round.
        * anything else (``reverted`` / apply / bench failure) — no progress:
          bump ``enablement_stall_streak``; once it reaches
          :data:`_ENABLEMENT_MAX_STALL`, stop the run with
          ``stop_reason='enablement_stalled'`` instead of looping on the same
          gap; otherwise clear the guard so the next round retries a different
          approach.

        Args:
            res: The integrate_patch result dict (may be ``None`` / non-dict).
        """
        if not isinstance(res, dict) or not res.get("enablement"):
            return
        state = self.shared_state
        status = str(res.get("status") or "")
        stop_set = ""
        # Capture the finished round's specialist task id so the async dispatch
        # chokepoint can read its specialist_done.json for a needs_targeted_build
        # request (a build enqueue needs await; rearm is sync). Only overwrite on
        # a real specialist round (targeted_build rearm rows carry no such id).
        _spec_tid = str(res.get("specialist_task_id") or "").strip()
        if _spec_tid:
            state.enablement.last_specialist_task_id = _spec_tid

        def _stack_setup_commands() -> None:
            """Append this round's applied setup commands to the durable stack."""
            cur = list(state.enablement.setup_commands or [])
            for c in res.get("setup_commands_applied") or []:
                sc = str(c)
                if sc and sc not in cur:
                    cur.append(sc)
            state.enablement.setup_commands = cur

        def _push_kept_round(patches_this_round: list[str]) -> None:
            """Append this round to kept_rounds and re-derive the flat projections.

            Artifacts dedupe last-wins per target so a later round supersedes an
            earlier fix to the same file.
            """
            rounds = list(state.enablement.kept_rounds or [])
            rounds.append(
                {
                    "patches": list(patches_this_round),
                    "artifacts": [dict(a) for a in (res.get("artifacts_applied") or []) if isinstance(a, dict)],
                }
            )
            state.enablement.kept_rounds = rounds

            flat_patches: list[str] = []
            artifact_by_target: dict[str, dict] = {}
            for rnd in rounds:
                for p in rnd.get("patches") or []:
                    if p not in flat_patches:
                        flat_patches.append(p)
                for art in rnd.get("artifacts") or []:
                    target = str(art.get("target") or "")
                    if target:
                        artifact_by_target[target] = art
            state.enablement.kept_patches = flat_patches
            state.enablement.kept_artifacts = list(artifact_by_target.values())

        def _reset_baseline_failure_backstop() -> None:
            """Clear the baseline-failure counters on enablement forward progress.

            A serial enablement makes the baseline re-fail on purpose (each round
            clears gap #n and the next boot stops at a deeper gap), so those
            crashes are progress, not a stuck baseline. Reset the backstop
            counters so ``enablement_stalled`` is the sole enablement-phase
            fast-fail.
            """
            state.baseline_failure_streak = 0
            state.baseline_arg_error_streak = 0
            state.baseline_total_failures = 0

        def _stack_kept_runtime() -> None:
            """Persist the KEEP'd attempt runtime + localization manifest so they
            survive rearm."""
            action = res.get("enablement_kept_stack_action")
            if isinstance(action, dict) and action:
                state.enablement.kept_stack_action = action
            runtime = res.get("enablement_active_runtime")
            if isinstance(runtime, dict) and runtime:
                state.enablement.active_runtime = runtime
                # Retain the attempt-runtime record (cap at 5 newest).
                records = list(state.enablement.attempt_runtimes or [])
                records.append(runtime)
                state.enablement.attempt_runtimes = records[-5:]
            # Record the localized closure manifest so it is not re-fetched on
            # the next round.
            manifest = res.get("enablement_localization_manifest")
            if isinstance(manifest, dict) and manifest:
                existing = list(state.enablement.localization_manifest or [])
                existing.append(manifest)
                state.enablement.localization_manifest = existing

        if status == "kept":
            _reset_baseline_failure_backstop()
            _stack_setup_commands()
            _stack_kept_runtime()
            _push_kept_round([str(p) for p in (res.get("patches_applied") or []) if str(p)])
            accepted_cfg = str(res.get("enablement_accepted_config_path") or "").strip()
            if accepted_cfg:
                state.enablement.accepted_config_path = accepted_cfg
            effective = res.get("enablement_effective_config")
            if isinstance(effective, dict) and effective:
                # Replaced, not merged: what the KEEP bench launched already
                # supersedes every advanced round that fed into it.
                state.enablement.accepted_config = dict(effective)
            if str(state.enablement.origin or "") == "eval":
                # eval-origin: the patch boots and re-passed accuracy in the gate,
                # but tput/accuracy only become official once a GENUINE baseline
                # promotes. Hold succeeded; open the revalidation window. Keep the
                # stall streak so repeated KEEP->revalidation-fail cycles still
                # reach the stall cap.
                state.enablement.validation_pending = True
                # Increment generation so the new window gets a fresh idempotency
                # key and cannot reuse a prior terminal TaskRegistry row.
                state.enablement.revalidation_generation = int(state.enablement.revalidation_generation or 0) + 1
                state.enablement.revalidation_task_id = ""
            else:
                state.enablement.succeeded = True
                state.enablement.stall_streak = 0
        elif status == "advanced" or bool(res.get("advanced")):
            # Forward progress on a serial enablement: stack the progressing
            # patches + setup commands and pivot to the newly-revealed gap.
            _push_kept_round([str(p) for p in (res.get("patches_applied") or []) if str(p)])
            _stack_setup_commands()
            _stack_kept_runtime()
            # Accumulated so a later kept round replays every advance, not just patches.
            adv_envs = res.get("extra_envs_applied") or {}
            adv_args = str(res.get("extra_server_args_applied") or "").strip()
            if adv_envs or adv_args:
                cfg = dict(state.enablement.accepted_config or {})
                merged = dict(cfg.get("extra_envs") or {})
                merged.update({str(k): str(v) for k, v in adv_envs.items()})
                cfg["extra_envs"] = merged
                # Folded by flag keeping the last value, so this round overrides an earlier one.
                cfg["extra_server_args"] = _dedupe_extra_server_args(
                    merge_server_args(str(cfg.get("extra_server_args") or ""), adv_args)
                )
                cfg.setdefault("args_mode", "append")
                state.enablement.accepted_config = cfg
            new_log = str(res.get("enablement_launch_log") or "").strip()
            if new_log:
                state.enablement.launch_log = new_log
            state.enablement.stall_streak = 0
            _reset_baseline_failure_backstop()
        else:
            # No progress: count toward the stall cap.
            state.enablement.stall_streak = int(state.enablement.stall_streak or 0) + 1
            if state.enablement.stall_streak >= _ENABLEMENT_MAX_STALL and not state.stop_reason:
                state.set_stop_reason("enablement_stalled")
                stop_set = "enablement_stalled"
        # Set on every round so neither outlives the round it describes.
        state.enablement.last_grounding_drop_reason = [
            str(d) for d in (res.get("patches_dropped_by_grounding") or [])[:8]
        ]
        state.enablement.patches_span_multiple_roots = bool(res.get("patches_span_multiple_roots"))
        # Phase-synthesised rounds carry no framework_root; keep the last real one.
        res_fw_root = str(res.get("framework_root") or "").strip()
        if res_fw_root:
            state.enablement.framework_root = res_fw_root
        try:
            snapshot_round(self.session_dir, res)
            if status in ("kept", "advanced"):
                write_setting_script(
                    self.session_dir,
                    state.enablement,
                    framework=str(state.framework or os.environ.get("FRAMEWORK") or "sglang"),
                    model=os.environ.get("MODEL_PATH") or state.model_path or state.reference_model,
                    tp=int(state.tp or 0) or None,
                    max_model_len=int(state.max_model_len or 0) or None,
                    gpu_type=str(state.gpu_type or os.environ.get("GPU_TYPE") or "") or None,
                )
        except Exception:  # noqa: BLE001 — archiving must not break the rearm
            log.warning("enablement: artifact write failed", exc_info=True)
        # A rearm always ends the round.
        state.enablement.inflight_task_id = ""
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.debug("enablement: save after rearm failed", exc_info=True)
        log.info(
            "ENABLEMENT: rearm from integrate status=%s succeeded=%s advanced=%s "
            "stacked=%d stall_streak=%d next_attempt=%d%s",
            status,
            bool(state.enablement.succeeded),
            status == "advanced" or bool(res.get("advanced")),
            len(state.enablement.kept_patches or []),
            int(state.enablement.stall_streak or 0),
            int(state.enablement.attempts or 0),
            f" stop_reason={stop_set}" if stop_set else "",
        )

    async def _pump_enablement_safely(self, *, caller: str) -> None:
        """Phase-independent enablement pump — runs every tick.

        A baseline that cannot even *launch* traps the run in PRELUDE forever:
        the only PRELUDE exit gate is ``baseline_tput > 0``, which a
        non-runnable (model, backend) combo never reaches. The enablement
        authoring dispatch used to live only inside
        :meth:`_pump_framework_agent_phase` (guarded on
        ``phase == FRAMEWORK_AGENT``), so it could never fire for the exact
        "can't boot at all" scenario it exists to repair — the run instead hit
        the 3-failure ``baseline_failed`` stop.

        This wrapper drives :meth:`_maybe_enqueue_enablement_specialist` from
        every coordinator tick, independent of phase. All dispatch guards
        (dispatched-in-flight, already-succeeded, ``baseline_tput > 0``,
        failure-streak, run deadline, single-node) live inside that method, so
        calling it unconditionally here is safe and idempotent.

        Args:
            caller: Label identifying the caller ("tick" / "run"), for logs.
        """
        try:
            await self._maybe_route_build_outcomes()
        except Exception:  # noqa: BLE001 — never wedge the tick
            log.exception("ENABLEMENT route_build_outcomes (%s) failed", caller)
        try:
            await self._maybe_enqueue_enablement_baseline_revalidation()
        except Exception:  # noqa: BLE001 — never wedge the tick
            log.exception("ENABLEMENT revalidation (%s) failed", caller)
        try:
            await self._maybe_enqueue_enablement_specialist()
        except Exception:  # noqa: BLE001 — never wedge the tick
            log.exception("ENABLEMENT pump (%s) failed", caller)
