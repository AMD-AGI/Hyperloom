# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement round admission, in-flight tracking and re-arm."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.deadline import Deadline

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event

from hyperloom.inference_optimizer.framework_paths import point_session_framework_root
from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.inference_optimizer.reference_script import ServingTarget
from ..bringup import ARGV_INVALID, ENV_FAULT, is_argv_invalid, load_boot_observation, observation_summary
from ..collaborator import CoordinatorCollaborator
from hyperloom.inference_optimizer.breakdown.round_archive import ROLE_LAUNCH_CONFIG, RoundArchive
from ..phases.machine_state import ENABLEMENT_MAX_ATTEMPTS as _ENABLEMENT_MAX_ATTEMPTS, PHASE_ENABLEMENT
from ..loop.offload import offload
from .params import ENABLEMENT_PARAMS_BUDGET_SEC
from .artifacts import snapshot_round, write_setting_script
from ..bringup import recorded_verdict, session_root
from ..state.round_store import ADVANCED, BOOTED, FAILED, Round, RoundRequest
from ..state.task_registry import TaskSpec, create_in_cursor, task_dispatch_origin
from .recipe.section import recipe_for

if TYPE_CHECKING:
    from ..bringup import EnvVerdict
    from ..state.task_registry import Task

import logging as _logging

log = _logging.getLogger(__name__)


#: Shortest lease a renewal may stamp.
_MIN_LEASE_SEC = 300.0

#: The outcome a result that moved the lane settles its round with; any other result is a stall.
_MOVED_ROUND_OUTCOMES = {"kept": BOOTED, "advanced": ADVANCED}


def takes_enablement_round(state: Any) -> bool:
    """Whether an enablement round may run now: in ENABLEMENT, while the lane accepts rounds."""
    return (state.phase or "").strip().upper() == PHASE_ENABLEMENT and state.enablement.accepts_rounds


class EnablementLane(CoordinatorCollaborator):
    """Owns one enablement round: admit, track in-flight, re-arm on outcome."""

    def enablement_admitted(self) -> bool:
        """Whether this run and host admit the enablement lane at all."""
        from ..actions.executors._accuracy_gate import eval_enablement_allowed, launch_enablement_allowed
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return False
        state = self.shared_state
        if state.enablement.origin == "eval":
            return eval_enablement_allowed(state)
        return launch_enablement_allowed(state)

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        """Dispatch an enablement_specialist when a baseline cannot launch or its accuracy eval fails."""
        state = self.shared_state
        if not self.enablement_admitted():
            return ""
        if not state.enablement.accepts_rounds:
            return ""
        round_row = await self.rounds.held()
        if round_row is not None:
            # This pump is the lease's only heartbeat, so that asking whether the lane is in
            # flight stays free of side effects for the terminal paths that only observe it.
            # A round nobody is working on is left to expire; the reconciler, which runs
            # ahead of this pump, is what ends it.
            if await self._round_has_live_work(round_row.holder_task_id):
                await self._renew_enablement_round(round_row)
            return ""
        # Each terminal below writes stop_reason, which routes the phase to CLOSE on the
        # next tick; returning keeps a new round from opening in the meantime.
        if self._check_argv_terminal():
            return ""
        if self._check_environment_terminal():
            return ""
        stalled = await self.rounds.consecutive_stalled()
        if stalled >= _ENABLEMENT_MAX_ATTEMPTS:
            if not state.stop_reason:
                state.set_stop_reason("enablement_attempts_exhausted")
                state.save(self.session_dir)
                # The cap is the lane's own terminal, so the lane event closes
                # here: no round follows it to close on the lane's behalf.
                await self.close_lane_event(
                    outcome=enablement_event.OUTCOME_STALLED,
                    reason="enablement_attempts_exhausted",
                )
                log.warning(
                    "ENABLEMENT: %d consecutive rounds bought no ground (cap %d); stopping the run",
                    stalled,
                    _ENABLEMENT_MAX_ATTEMPTS,
                )
            return ""
        launch_log = state.enablement.launch_log
        deadline = self._coord.run_deadline
        # Reaches the network and stats a checkout on a network mount, so it
        # runs off the tick; discovery degrades to repos-only at the deadline.
        params = await offload(
            lambda: self._coord.enablement_params.build_enablement_specialist_params(launch_log, attempt=stalled),
            deadline=Deadline.after(ENABLEMENT_PARAMS_BUDGET_SEC).tightened_to(deadline),
            label="enablement specialist params",
        )
        if params is None:
            await self._maybe_record_enablement_human_review(launch_log)
            return ""
        # Recorded on the loop rather than by the thread that found them.
        state.enablement.candidate_refs = [str(r) for r in params["enablement_candidate_refs"]]
        # Any build the previous round's specialist asked for, then an
        # auto-escalation when the residual gap is a compiled miss. Both are
        # no-ops when a matching build is already queued or running, and neither
        # may block the authoring dispatch below, which is this method's point.
        try:
            await self._coord.enablement_build.maybe_enqueue_specialist_requested_build()
            await self._coord.enablement_build.maybe_escalate_to_targeted_build(launch_log, attempt=stalled)
        except (OSError, sqlite3.Error):
            log.exception("enablement: build escalation failed")
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_tid = await self._open_authoring_round(
            params=params,
            lanes=lanes,
            lease_ttl_sec=int(ttl),
        )
        if not spec_tid:
            return ""
        # Recorded here rather than derived at export: the classified failure
        # kind exists only in the params this dispatch built, and the log the
        # round was pointed at is replaced by the next advance.
        enablement_event.record_dispatch(
            enablement_event.AuthoringDispatch(
                task_id=spec_tid,
                attempt=stalled + 1,
                failure_kind=str(params.get("enablement_failure_kind") or ""),
                launch_log=launch_log,
                candidate_refs=params.get("enablement_candidate_refs"),
                mode=str(state.enablement_mode or ""),
                origin=state.enablement.origin or enablement_event.ORIGIN_BOOT,
            )
        )
        state.save(self.session_dir)
        log.info(
            "ENABLEMENT: dispatched authoring specialist kind=%s stalled=%d task=%s",
            params["enablement_failure_kind"],
            stalled,
            spec_tid,
        )
        return spec_tid

    def _check_argv_terminal(self) -> bool:
        """Write server_argv_invalid to stop_reason if the last boot was refused by the parser.

        Returns:
            bool: True when a terminal argv fault was detected; the phase machine
            routes to CLOSE on the next tick through ``_global_terminal``.
        """
        state = self.shared_state
        loaded = load_boot_observation(state.enablement.launch_observation_path)
        observation = loaded.observation
        if observation is None or not is_argv_invalid(observation):
            return False
        if not state.stop_reason:
            state.set_stop_reason(ARGV_INVALID)
            state.save(self.session_dir)
            excerpt = observation.excerpt
            log.error(
                "ENABLEMENT: the server argv was refused before launch; stopping the run rather than "
                "authoring a framework patch for it -- %s",
                (excerpt.text if excerpt is not None else "").replace("\n", " | "),
            )
        return True

    def _check_environment_terminal(self) -> bool:
        """Write environment_fault to stop_reason if the host cannot run the combo.

        Returns:
            bool: True when a terminal host fault was detected; the phase machine
            routes to CLOSE on the next tick through ``_global_terminal``.
        """
        state = self.shared_state
        verdict = self._environment_verdict()
        if verdict is None or not verdict.terminal:
            return False
        if not state.stop_reason:
            state.set_stop_reason(ENV_FAULT)
            state.save(self.session_dir)
            log.error(
                "ENABLEMENT: this host cannot run the combo (%s); stopping the run rather than "
                "authoring a framework patch for it -- %s",
                verdict.fault,
                verdict.detail,
            )
        return True

    def _environment_verdict(self) -> EnvVerdict | None:
        """Ask the host, structurally, whether it can run this combo.

        Returns:
            EnvVerdict | None: The host's verdict, or ``None`` when the check
            could not be made -- it stats a checkout on a network mount, and
            that failing is no evidence about the host.
        """
        from hyperloom.common.env_safety import build_benchmark_env

        from ..bringup import check_environment

        try:
            # The session's own resolved checkpoint. The launch env is a
            # scrubbed environ that only the CLI exports MODEL_PATH into,
            # so reading it alone turns this check off everywhere else.
            return check_environment(
                model=self.shared_state.model_path or build_benchmark_env().get("MODEL_PATH", ""),
            )
        except OSError:
            log.debug("enablement: environment preflight could not be made", exc_info=True)
            return None

    async def _open_authoring_round(
        self,
        *,
        params: dict[str, Any],
        lanes: list[str],
        lease_ttl_sec: int,
    ) -> str:
        """Acquire the round and create the specialist that holds it, together.

        The holder's task row is written by the acquiring cursor, so the two
        land together or not at all. ``holder`` is minted here and never reused,
        so the task's idempotency key cannot collide; what keeps a second round
        from opening is the round mutex, not that key.

        Args:
            params: The specialist's task params.
            lanes: Lanes the specialist must hold while running.
            lease_ttl_sec: The specialist's lease, and the round's first one.

        Returns:
            str: The holder task id, or ``""`` when the machine was excluded.
        """
        holder = uuid.uuid4().hex

        def _join(cur: sqlite3.Cursor) -> None:
            create_in_cursor(
                cur,
                TaskSpec(
                    kind="specialist",
                    params=params,
                    idempotency_key=holder,
                    requires_lanes=lanes,
                    lease_ttl_sec=lease_ttl_sec,
                    task_id=holder,
                    dispatch_class="coordinator",
                    dispatch_origin=task_dispatch_origin(self.shared_state),
                ),
            )

        acquired = await self.rounds.open(
            _authoring_round_id(holder),
            RoundRequest(now_unix=time.time(), request_id=holder),
            holder_task_id=holder,
            lease_sec=float(lease_ttl_sec),
            join=_join,
        )
        if not acquired.ok:
            log.info("ENABLEMENT: round not opened (%s)", acquired.reason)
            return ""
        return holder

    async def enablement_in_flight(self) -> bool:
        """True while an open round still has work running under it.

        Returns:
            bool: Whether the round is still resolving.
        """
        round_row = await self.rounds.held()
        if round_row is None:
            return False
        return await self._round_has_live_work(round_row.holder_task_id)

    async def _round_has_live_work(self, holder: str) -> bool:
        """Report whether anything is still running for the round ``holder`` holds.

        Args:
            holder: The round's holder task id.

        Returns:
            bool: Whether a task or an undecided proposal still owes a result.
        """
        from ..state.task_registry import TaskNotFound

        try:
            task = await self.tasks.get(holder)
        except TaskNotFound:
            task = None
        if task is not None and task.state in ("queued", "running"):
            return True
        # Undecided proposal keeps the round open; once ruled, approve lands the task matched below and reject rearms
        # directly, so it cannot defer forever.
        for p in self.state.pending_proposals.values():
            if p.action_name != "integrate_patch":
                continue
            if (p.payload.get("params") or {}).get("specialist_task_id") == holder:
                return True
        for t in (await self.tasks.queued()) + (await self.tasks.running()):
            if t.kind == "integrate_patch" and t.params.get("specialist_task_id") == holder:
                return True
        return False

    async def _renew_enablement_round(self, round_row: Round) -> None:
        """Extend the open round's lease for the leg still running.

        Args:
            round_row: The round as the store last reported it.
        """
        lease = max(_MIN_LEASE_SEC, float(round_row.expires_unix) - float(round_row.renewed_unix))
        now = time.time()
        await self.rounds.renew(
            round_row,
            RoundRequest(now_unix=now, request_id=f"renew:{round_row.round_id}:{round_row.fence}:{now:.0f}"),
            lease_sec=lease,
        )

    async def handoff_enablement_round(self, task: "Task") -> None:
        """Move the open round onto the integrate that consumes its deliverable.

        Handoff is the only fence increment, so every task id that takes over an
        open round must come through here.

        Args:
            task: The freshly materialised task.
        """
        if task.kind != "integrate_patch":
            return
        specialist = str(task.params.get("specialist_task_id") or "")
        if not specialist:
            return
        round_row = await self.rounds.held()
        if round_row is None or round_row.holder_task_id != specialist:
            return
        lease = float(task.lease_ttl_sec)
        if lease <= 0:
            lease = max(_MIN_LEASE_SEC, float(round_row.expires_unix) - float(round_row.renewed_unix))
        moved = await self.rounds.handoff(
            round_row,
            RoundRequest(now_unix=time.time(), request_id=f"handoff:{task.task_id}"),
            new_holder_task_id=task.task_id,
            lease_sec=lease,
        )
        if not moved.ok:
            log.warning(
                "ENABLEMENT: round %s not handed to integrate %s (%s)",
                round_row.round_id,
                task.task_id,
                moved.reason,
            )

    async def settle_enablement_round(self, outcome: str, *, reason: str = "") -> None:
        """End the open round, if one is still open.

        A specialist that holds the round and has not started is cancelled: its
        mandate names a wall this settle has closed, and the next round is
        dispatched against the current one.

        Args:
            outcome: What the round leaves behind; see the round store's
                outcomes.
            reason: Recorded on the settle's outbox row.
        """
        round_row = await self.rounds.held()
        if round_row is None:
            return
        await self.rounds.settle(
            round_row,
            RoundRequest(
                now_unix=time.time(),
                request_id=f"settle:{round_row.round_id}:{round_row.fence}",
                evidence={"reason": reason} if reason else {},
            ),
            outcome=outcome,
        )
        holder = round_row.holder_task_id
        if await self.tasks.cancel_if_queued(holder, kind="specialist", reason="enablement_round_settled"):
            log.info("ENABLEMENT: cancelled specialist %s; its round was settled before it started", holder)

    async def on_baseline_promoted(self) -> None:
        """Take a baseline promoted outside a pending revalidation: an anchor ends a boot-origin lane.

        A boot-origin lane exists to make the combo boot, so an anchor reached by any route leaves it nothing to
        bring up. An eval-origin lane starts from a combo that already boots; only its revalidation ends it.
        """
        state = self.shared_state
        if (
            (state.phase or "").strip().upper() == PHASE_ENABLEMENT
            and state.enablement.origin != "eval"
            and state.enablement.accepts_rounds
            and float(state.baseline_tput or 0.0) > 0.0
        ):
            await self.close_booted(reason="baseline anchored")
            return
        state.enablement.origin = ""
        state.enablement.pending = False

    async def close_booted(self, *, reason: str) -> None:
        """End the lane on a combo that boots: mark it succeeded, settle its open round and close its event."""
        enablement = self.shared_state.enablement
        enablement.succeeded = True
        enablement.validation_pending = False
        enablement.revalidation_task_id = ""
        enablement.origin = ""
        enablement.pending = False
        await self.settle_enablement_round(BOOTED, reason=reason)
        await self.close_lane_event(outcome=enablement_event.OUTCOME_SUCCEEDED, reason=reason)

    async def close_lane_event(self, *, outcome: str, reason: str) -> None:
        """Close the enablement lane's event on the terminal it just reached.

        Args:
            outcome: One of the ``enablement_event.OUTCOME_*`` constants.
            reason: Why the lane ended, recorded on the event.
        """
        _finish_lane_event(
            self.shared_state,
            LaneEnd(outcome=outcome, reason=reason, stall_streak=await self.rounds.consecutive_stalled()),
            str(self.session_dir or ""),
        )

    async def _maybe_record_enablement_human_review(self, launch_log: str) -> None:
        """Record a one-shot ``needs_human_review`` for an UNKNOWN launch failure.

        Deduped per distinct log via a stored hash. No sub-agent is dispatched.
        A blank log records nothing, and neither does one whose signature is
        actionable.

        Args:
            launch_log: The captured launch / traceback text.

        Raises:
            OSError: When the state carrying the dedupe hash cannot be saved.
            sqlite3.Error: When the observation cannot be appended to the bus.
        """
        text = launch_log.strip()
        if not text:
            return

        state = self.shared_state
        # The round's own filed observation, not a reclassification of the
        # wrapper text, so an operator triages by the same digest.
        verdict, loaded = recorded_verdict(
            state.enablement.launch_observation_path,
            wrapper_text=text,
            session_dir=session_root(self),
        )
        signature = verdict.signature
        if signature.is_actionable:
            return
        digest = hashlib.sha1(text.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()
        seen = state.enablement.human_review_logged
        if digest in seen:
            return
        seen.append(digest)
        # Without this the timeline cannot tell a lane that was triggered and
        # then did nothing from one that was never triggered at all.
        enablement_event.record_human_review(
            digest=digest,
            failure_kind=signature.kind,
            reason=("baseline launch failure did not match any actionable enablement signature; needs human triage"),
            signature=signature.to_dict(),
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "enablement_needs_human_review",
                "applicability": "needs_human_review",
                "framework": state.framework.strip().lower(),
                "model": state.model_name.strip(),
                "failure_kind": signature.kind,
                "signature": signature.to_dict(),
                "observation": observation_summary(verdict.observation),
                "observation_path": loaded.path,
                "observation_degraded": loaded.degraded,
                "reason": (
                    "baseline launch failure did not match any actionable enablement signature; needs human triage"
                ),
            },
        )
        state.save(self.session_dir)
        log.info(
            "ENABLEMENT: recorded needs_human_review for UNKNOWN failure kind=%s",
            signature.kind,
        )

    async def maybe_rearm_enablement(self, res: dict[str, Any]) -> None:
        """Re-arm, advance, or terminate the enablement retry loop.

        Called on every ``integrate_patch`` completion. An enablement patch has
        three outcomes:

        * ``kept`` — the combo is now runnable: terminal success.
        * ``advanced`` — the patch cleared the prior crash and the boot now
          stops at a new, deeper gap. The patch is stacked and
          ``enablement.launch_log`` replaced, so the next round targets that gap.
        * anything else — no patch is kept.

        A kept or advanced patch moves the lane whoever authored it; a stall
        from a specialist the lane never dispatched leaves it untouched.

        Args:
            res: The integrate_patch result dict.
        """
        if not res.get("enablement"):
            return
        state = self.shared_state
        status = str(res.get("status") or "")
        # Only a real specialist round carries this; targeted_build rearm rows
        # do not, and must not clear it.
        spec_tid = str(res.get("specialist_task_id") or "").strip()
        if await _is_foreign_stall(self.rounds, spec_tid, status=status):
            return
        if spec_tid:
            state.enablement.last_specialist_task_id = spec_tid

        if status == "kept":
            _rearm_on_kept(state, res)
        elif status == "advanced":
            _rearm_on_advanced(state, res)
        # A round that bought no ground gets no branch of its own: it is charged
        # by the FAILED settle below, which the pre-hoc ``consecutive_stalled``
        # cap reads off the durable ledger. Counting it in state again here
        # would double-charge it.
        state.enablement.mark_setup_round(spec_tid, status or "unreported", accepted=status == "kept")
        state.enablement.record_round_defects(res)
        # Phase-synthesised rounds carry no framework_root; keep the last real one.
        res_fw_root = str(res.get("framework_root") or "").strip()
        if res_fw_root:
            state.enablement.framework_root = res_fw_root
        # The ordinal the dispatch filed this round under, read before the
        # settle scores it, so every row of this round meets on the same number.
        attempt = await self.rounds.consecutive_stalled() + 1
        setting_script = _archive_round(state, res, self.session_dir, attempt)
        outcome = await _answered_round_outcome(self.rounds, spec_tid, status=status)
        if outcome:
            await self.settle_enablement_round(outcome, reason=status)
        # Recorded after the accepted path is settled, because the row carries
        # it, and after the round is settled, because the streak the row states
        # is the one this round's own outcome left behind.
        _record_enablement_round(
            state,
            res,
            RoundSettlement(
                setting_script=setting_script,
                attempt=attempt,
                stall_streak=await self.rounds.consecutive_stalled(),
            ),
            str(self.session_dir or ""),
        )
        state.save(self.session_dir)
        log.info(
            "ENABLEMENT: rearm from integrate status=%s succeeded=%s advanced=%s stacked=%d",
            status,
            bool(state.enablement.succeeded),
            status == "advanced",
            len(state.enablement.kept_patches),
        )

    async def pump_enablement_safely(self) -> None:
        """ENABLEMENT phase pump — called every tick while in ENABLEMENT."""
        if (self.shared_state.phase or "").strip().upper() != PHASE_ENABLEMENT:
            return
        # Independently, because a raise in one pump must not skip the rest: the
        # one that dispatches the next authoring round is the last of them.
        for pump in (
            self._coord.enablement_build.maybe_route_build_outcomes,
            self._coord.enablement_revalidation.maybe_enqueue_enablement_baseline_revalidation,
            self._maybe_enqueue_enablement_specialist,
        ):
            try:
                await pump()
            except Exception as exc:  # isolation boundary: one enablement pump of the tick
                log.exception("ENABLEMENT %s failed", pump.__name__)
                stage = f"enablement_pump:{pump.__name__}"
                # Named here rather than by the coordinator's generic handler:
                # the lane's event is open for the whole session, so only the
                # lane knows which exceptions are its own to answer for.
                enablement_event.record_fault(stage=stage, exc=exc)
                self._coord.record_exception(stage=stage, exc=exc)


def _push_kept_round(state: Any, res: dict[str, Any]) -> None:
    """Stack the round a rearm accepted, as the integrate result reports it."""
    state.enablement.push_kept_round(
        task_id=str(res.get("specialist_task_id") or ""),
        patches=res.get("patches_applied", []),
        artifacts=res.get("artifacts_applied", []),
    )


def _reset_baseline_failure_backstop(state: Any) -> None:
    """Clear the baseline-failure counters on enablement forward progress.

    A serial enablement makes the baseline re-fail on purpose (each round
    clears gap #n and the next boot stops at a deeper gap), so those crashes
    are progress, not a stuck baseline.
    """
    state.baseline_failure_streak = 0
    state.baseline_total_failures = 0


def _stack_kept_runtime(state: Any, res: dict[str, Any]) -> None:
    """Persist the accepted runtime, point the session framework tree at it, and keep the localization manifest."""
    action = res.get("enablement_kept_stack_action")
    if isinstance(action, dict) and action:
        state.enablement.kept_stack_action = action
    runtime = res.get("enablement_active_runtime")
    if isinstance(runtime, dict) and runtime:
        state.enablement.active_runtime = runtime
        # Cap at the 5 newest attempt-runtime records.
        records = list(state.enablement.attempt_runtimes or [])
        records.append(runtime)
        state.enablement.attempt_runtimes = records[-5:]
        source_root = str(runtime.get("source_root") or "")
        if source_root:
            point_session_framework_root(source_root, state.framework)
            state.framework_repo_path = source_root
    # Record the localized closure manifest so it is not re-fetched on the next round.
    manifest = res.get("enablement_localization_manifest")
    if isinstance(manifest, dict) and manifest:
        existing = list(state.enablement.localization_manifest or [])
        existing.append(manifest)
        state.enablement.localization_manifest = existing


def _rearm_on_kept(state: Any, res: dict[str, Any]) -> None:
    """Record the accepted stack and terminate, or open the eval revalidation."""
    _reset_baseline_failure_backstop(state)
    state.enablement.stack_setup_commands(res.get("setup_commands_applied", []))
    _stack_kept_runtime(state, res)
    _push_kept_round(state, res)
    accepted_cfg = str(res.get("enablement_accepted_config_path") or "").strip()
    if accepted_cfg:
        state.enablement.accepted_config_path = accepted_cfg
    effective = res.get("enablement_effective_config")
    if isinstance(effective, dict) and effective:
        # Replaced, not merged: what the KEEP bench launched already supersedes
        # every advanced round that fed into it.
        state.enablement.record_accepted_config(effective, source="kept_bench")
    state.enablement.record_keep(res)
    # Hold succeeded until the revalidation baseline promotes so every KEEP is
    # validated against a real measurement, not the bench the patch round itself
    # ran. Unconditional since upstream stopped exempting the launch origin: a
    # KEEP nobody re-measured is a KEEP on the round's own word.
    state.enablement.validation_pending = True
    # A fresh generation so the new window's idempotency key cannot reuse a
    # prior terminal TaskRegistry row.
    state.enablement.revalidation_generation += 1
    state.enablement.revalidation_task_id = ""


def _rearm_on_advanced(state: Any, res: dict[str, Any]) -> None:
    """Stack the progressing round and pivot the mandate to the new gap."""
    _push_kept_round(state, res)
    state.enablement.record_patch_roots(res.get("enablement_patch_roots", {}))
    state.enablement.stack_setup_commands(res.get("setup_commands_applied", []))
    _stack_kept_runtime(state, res)
    # Accumulated so a later kept round replays every advance, not just patches.
    adv_envs = res.get("extra_envs_applied") or {}
    adv_args = str(res.get("extra_server_args_applied") or "").strip()
    if adv_envs or adv_args:
        cfg = dict(state.enablement.accepted_config or {})
        merged = dict(cfg.get("extra_envs") or {})
        merged.update({str(k): str(v) for k, v in adv_envs.items()})
        cfg["extra_envs"] = merged
        # Folded by flag keeping the last value, so this round overrides an earlier one.
        cfg["extra_server_args"] = merge_server_args(str(cfg.get("extra_server_args") or ""), adv_args)
        cfg.setdefault("args_mode", "append")
        # An advanced round is by construction not booted, so nothing observed
        # this configuration; the tag is what keeps "verified" and "unverified"
        # distinguishable at all.
        state.enablement.record_accepted_config(cfg, source="advanced_merge")
    new_log = str(res.get("enablement_launch_log") or "").strip()
    if new_log:
        # The wall this round advanced to: the next round's before half.
        state.enablement.record_launch_observation(
            log=new_log, observation_path=str(res.get("enablement_observation_path") or "")
        )
    _reset_baseline_failure_backstop(state)


@dataclass(frozen=True)
class LaneEnd:
    """How the enablement lane ended.

    Attributes:
        outcome: One of the ``enablement_event.OUTCOME_*`` constants.
        reason: Why the lane ended, recorded on the event.
        stall_streak: Consecutive stalled rounds when it ended.
        setting_script: Session-relative path of the exported setting script.
    """

    outcome: str
    reason: str
    stall_streak: int
    setting_script: str = ""


@dataclass(frozen=True)
class RoundSettlement:
    """What the rearm knows about the authoring round it records.

    Attributes:
        setting_script: Session-relative path of the exported setting script.
        attempt: The ordinal the dispatch filed the round under.
        stall_streak: Consecutive stalled rounds after this one settled.
    """

    setting_script: str
    attempt: int
    stall_streak: int


def _finish_lane_event(state: Any, end: LaneEnd, session_dir: str) -> None:
    """Assemble and emit the terminal enablement_event.finish call."""
    lane = state.enablement
    enablement_event.finish(
        enablement_event.LaneTerminal(
            outcome=end.outcome,
            reason=end.reason,
            recipe=recipe_for(
                lane,
                session_dir=str(session_dir or ""),
                mode=str(state.enablement_mode or ""),
            ),
            kept_patches=lane.kept_patches,
            kept_artifacts=lane.kept_artifacts,
            setup_commands=lane.setup_commands,
            accepted_config=lane.accepted_config,
            accepted_config_path=str(lane.accepted_config_path or ""),
            setting_script=end.setting_script,
            active_runtime=lane.active_runtime,
            attempt_runtimes=lane.attempt_runtimes,
            framework_root=str(lane.framework_root or ""),
            stall_streak=end.stall_streak,
        )
    )


def _authoring_round_id(holder: str) -> str:
    """The id of the round the lane opened for specialist ``holder``."""
    return f"enablement-{holder}"


async def _is_foreign_stall(rounds: Any, spec_tid: str, *, status: str) -> bool:
    """Whether this result bought nothing for a specialist the lane never dispatched."""
    if not spec_tid or status in _MOVED_ROUND_OUTCOMES or await rounds.get(_authoring_round_id(spec_tid)) is not None:
        return False
    log.info(
        "ENABLEMENT: integrate for specialist %s, which the lane did not dispatch, bought nothing (%s); "
        "the lane's round is left to its holder",
        spec_tid,
        status or "unreported",
    )
    return True


async def _answered_round_outcome(rounds: Any, spec_tid: str, *, status: str) -> str:
    """The outcome to settle the held round with, or ``""`` when this result answers none.

    A kept or advanced patch settles whichever round is held: the wall that
    round was aimed at is behind the boot now. A stall settles only the round
    it was dispatched for, so a late result cannot charge a newer one. Only a
    KEEP booted and was graded, and an advance is the ledger's record that the
    cap must not charge it.
    """
    held = await rounds.held()
    moved = status in _MOVED_ROUND_OUTCOMES
    if held is None or not (moved or not spec_tid or held.round_id == _authoring_round_id(spec_tid)):
        return ""
    return _MOVED_ROUND_OUTCOMES.get(status, FAILED)


def _round_task_id(state: Any, res: dict[str, Any]) -> str:
    """The specialist task id every row of one round is keyed by.

    A phase-synthesised rearm carries no id of its own, so the in-flight one
    the dispatch filed stands in.
    """
    return str(res.get("specialist_task_id") or "").strip() or str(state.enablement.last_specialist_task_id or "")


def _archive_round(state: Any, res: dict[str, Any], session_dir: str | Path, attempt: int) -> str:
    """Archive one round's deliverables, and export the setting script of a round that moved the lane.

    Returns:
        str: The setting script's session-relative path, ``""`` when none was written.
    """
    setting_script = ""
    archive = RoundArchive(session_dir)
    try:
        archive = snapshot_round(session_dir, res)
        # Where the copies come into being, and from the archive rather
        # than from ``res``. Inside the try so that a snapshot which raised
        # leaves the row silent instead of claiming nothing landed.
        enablement_event.record_archive(
            task_id=_round_task_id(state, res),
            attempt=attempt,
            files=archive.to_list(),
        )
        if res.get("status") in _MOVED_ROUND_OUTCOMES:
            setting_script = write_setting_script(
                session_dir,
                state.enablement,
                framework=state.framework or os.environ.get("FRAMEWORK") or "sglang",
                target=ServingTarget(
                    model=os.environ.get("MODEL_PATH") or state.model_path or state.reference_model,
                    tp=state.tp or None,
                    max_model_len=state.max_model_len or None,
                    gpu_type=state.gpu_type or os.environ.get("GPU_TYPE") or None,
                ),
            )
    except (OSError, ValueError):
        log.warning("enablement: artifact write failed", exc_info=True)
    # Absolute and not the session-relative path the archive records: the
    # revalidation baseline opens this file directly. Only a copy that
    # landed overrides the ``runs/`` original the round reported, since a
    # later reader resolves nothing under ``runs/``.
    archived_config = archive.path_for(ROLE_LAUNCH_CONFIG)
    if res.get("status") == "kept" and archived_config:
        state.enablement.accepted_config_path = str(Path(session_dir) / archived_config)
    return setting_script


def _record_enablement_round(
    state: Any,
    res: dict[str, Any],
    settlement: RoundSettlement,
    session_dir: str = "",
) -> None:
    """Record how one authoring round settled, and close a lane that ended.

    A function of the lane state rather than a method, because it needs nothing
    of the coordinator: the round's own facts are in ``res`` and the terminal it
    reached is what the rearm just wrote to ``state``. ``attempt`` is the
    ordinal the dispatch filed it under.
    """
    stall_streak = settlement.stall_streak
    lane = state.enablement
    succeeded = bool(lane.succeeded)
    enablement_event.record_round(
        enablement_event.AuthoringRound(
            task_id=_round_task_id(state, res),
            attempt=int(settlement.attempt or 0),
            result=res,
            stall_streak=int(stall_streak or 0),
            succeeded=succeeded,
            validation_pending=bool(lane.validation_pending),
        )
    )
    # A KEEP opens a revalidation window, so succeeded is only set by the
    # promote path, not here.
    if not succeeded:
        return
    _finish_lane_event(
        state,
        LaneEnd(
            outcome=enablement_event.OUTCOME_SUCCEEDED,
            reason=str(res.get("status") or ""),
            stall_streak=int(stall_streak or 0),
            setting_script=settlement.setting_script,
        ),
        str(session_dir or ""),
    )
