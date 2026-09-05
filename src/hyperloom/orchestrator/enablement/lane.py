# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement round admission, in-flight tracking and re-arm."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.deadline import Deadline

from ..actions.executors._grid_server_args import merge_server_args
from ..actions.executors.boot_probe import probe_would_inform
from ..bringup import ARGV_INVALID, ENV_FAULT, is_argv_invalid, is_env_fault, load_boot_observation, observation_summary
from ..bringup.budget import STALLED_STOP_REASON, ProgressBudget, digest_of, session_budget, stage_of
from ..collaborator import CoordinatorCollaborator
from ..delivery.archive import ROLE_LAUNCH_CONFIG, RoundArchive
from ..loop.coordinator import _ENABLEMENT_MAX_ATTEMPTS
from ..loop.coordinator_helpers import _dedupe_extra_server_args
from ..loop.offload import offload
from .params import ENABLEMENT_PARAMS_BUDGET_SEC
from ..phases._enablement_artifacts import snapshot_round, write_setting_script
from ..bringup import recorded_verdict, session_root
from ..state.round_store import BOOTED, FAILED, Round
from ..state.task_registry import TerminalTaskReuse, create_in_cursor

if TYPE_CHECKING:
    from ..bringup import EnvVerdict
    from ..state.task_registry import Task

import logging as _logging

log = _logging.getLogger(__name__)


#: Shortest lease a renewal may stamp.
_MIN_LEASE_SEC = 300.0


class _DuplicateAuthoring(Exception):
    """The authoring key already names a live specialist, so no round is taken."""


class EnablementLane(CoordinatorCollaborator):
    """Owns one enablement round: admit, track in-flight, re-arm on outcome."""

    async def _maybe_enqueue_enablement_specialist(self) -> str:
        """Dispatch an enablement_specialist when a baseline cannot launch or its
        accuracy eval fails.

        Skipped when the trigger origin's lane is not admitted by
        ``--enablement``, when a prior attempt was KEPT, while an eval-origin
        KEEP awaits revalidation, while a bring-up round is open, when the argv
        or the host is terminal, when :data:`_ENABLEMENT_MAX_ATTEMPTS` is spent,
        past the run deadline, and on multi-node. A log that classifies to
        ``UNKNOWN`` still dispatches: the specialist repairs from the raw log.
        Nothing is dispatched when the params build yields none -- a blank log,
        or a build that outlasted its budget or raised -- and a non-blank log
        of those is recorded once as ``needs_human_review``.

        Returns:
            str: The dispatched specialist ``task_id`` (empty when skipped).
        """
        from ..actions.executors._accuracy_gate import eval_enablement_allowed, launch_enablement_allowed

        state = self.shared_state
        origin = state.enablement.origin
        admitted = eval_enablement_allowed(state) if origin == "eval" else launch_enablement_allowed(state)
        if not admitted:
            return ""
        if state.enablement.succeeded:
            return ""
        if state.enablement.validation_pending:
            return ""
        if self._refused_argv_is_terminal():
            return ""
        if self._environment_fault_is_terminal():
            return ""
        if await self.rounds.held() is not None:
            # Renews the open round's lease as a side effect; the reconciler,
            # which runs ahead of this pump, ends a round nobody is working on.
            await self._enablement_in_flight()
            return ""
        if state.baseline_tput > 0:
            return ""
        if state.baseline_failure_streak < 1:
            return ""
        # A monotonic count, not a progress predicate: each attempt seeds a
        # baseline whose failure seeds the next.
        if state.enablement.attempts >= _ENABLEMENT_MAX_ATTEMPTS:
            if not state.stop_reason:
                state.set_stop_reason("enablement_attempts_exhausted")
                state.save(self.session_dir)
                log.warning(
                    "ENABLEMENT: attempt cap reached (%d); stopping the run",
                    _ENABLEMENT_MAX_ATTEMPTS,
                )
            return ""
        deadline = self._run_deadline
        if deadline is not None and deadline.expired():
            return ""
        launch_log = state.enablement.launch_log
        attempt = state.enablement.attempts
        # Reaches the network and stats a checkout on a network mount, so it
        # runs off the tick; discovery degrades to repos-only at the deadline.
        params = await offload(
            lambda: self._build_enablement_specialist_params(launch_log, attempt=attempt),
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
            await self._maybe_enqueue_specialist_requested_build()
            await self._maybe_escalate_to_targeted_build(launch_log, attempt=attempt)
        except Exception:  # noqa: BLE001 — a build escalation must not cost the round
            log.exception("enablement: build escalation failed")
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return ""
        await self._warm_specialist_params(params)
        idem = f"enablement_authoring:{params['enablement_failure_kind']}:{attempt}"
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_tid = await self._open_authoring_round(
            params=params,
            idempotency_key=idem,
            lanes=lanes,
            lease_ttl_sec=int(ttl),
        )
        if not spec_tid:
            return ""
        state.enablement.attempts = attempt + 1
        state.save(self.session_dir)
        log.info(
            "ENABLEMENT: dispatched authoring specialist kind=%s attempt=%d task=%s",
            params["enablement_failure_kind"],
            attempt + 1,
            spec_tid,
        )
        return spec_tid

    def _refused_argv_is_terminal(self) -> bool:
        """Stop the run when the last failure was an argv the framework refused.

        No patch to framework source fixes an argument the installed parser does
        not have, so this is stopped as infrastructure.

        Returns:
            bool: True when the run was stopped here and no round may open.
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

    def _environment_fault_is_terminal(self) -> bool:
        """Stop the run when the last failure was the host rather than the model.

        A missing framework, an extension with no build for this platform, an
        unresolvable checkpoint path and a bound port are host faults no patch
        this lane could author would change, so this is stopped as
        infrastructure.

        Returns:
            bool: True when the run was stopped here and no round may open.
        """
        state = self.shared_state
        loaded = load_boot_observation(state.enablement.launch_observation_path)
        observation = loaded.observation
        fault, detail = "", ""
        if observation is not None and is_env_fault(observation):
            excerpt = observation.excerpt
            fault = str(observation.env_fault)
            detail = (excerpt.text if excerpt is not None else "").replace("\n", " | ")
        else:
            verdict = self._environment_verdict()
            if verdict is not None and verdict.terminal:
                fault, detail = verdict.fault, verdict.detail
        if not fault:
            return False
        if not state.stop_reason:
            state.set_stop_reason(ENV_FAULT)
            state.save(self.session_dir)
            log.error(
                "ENABLEMENT: this host cannot run the combo (%s); stopping the run rather than "
                "authoring a framework patch for it -- %s",
                fault,
                detail,
            )
        return True

    def _environment_verdict(self) -> EnvVerdict | None:
        """Ask the host, structurally, whether it can run this combo.

        Returns:
            EnvVerdict | None: The host's verdict, or ``None`` when the checks
            could not be made -- they stat a checkout on a network mount and
            open a socket, and neither failing is evidence about the host.
        """
        from hyperloom.common.env_safety import build_benchmark_env

        from ..bringup import check_environment

        state = self.shared_state
        try:
            launch_env = build_benchmark_env()
            return check_environment(
                framework=state.framework or os.environ.get("FRAMEWORK", ""),
                # The session's own resolved checkpoint. The launch env is a
                # scrubbed environ that only the CLI exports MODEL_PATH into,
                # so reading it alone turns this check off everywhere else.
                model=state.model_path or launch_env.get("MODEL_PATH", ""),
                port=int(launch_env.get("PORT") or 0),
                launch_env=launch_env,
            )
        except OSError:
            log.debug("enablement: environment preflight could not be made", exc_info=True)
            return None

    async def _maybe_enqueue_boot_probe(self) -> str:
        """Enqueue a boot-only probe when nothing says where the last boot stopped.

        The boot half of a baseline is the cheap half, and it alone names the
        wall an authoring round would otherwise guess at. Elided whenever
        :func:`~hyperloom.orchestrator.actions.executors.boot_probe.probe_would_inform`
        says the answer is already known.

        Returns:
            str: The dispatched ``task_id``, empty when nothing was dispatched.
        """
        from ..actions.executors._accuracy_gate import eval_enablement_allowed, launch_enablement_allowed
        from ..actions.executors._multi_node_env import is_multi_node

        state = self.shared_state
        origin = state.enablement.origin
        admitted = eval_enablement_allowed(state) if origin == "eval" else launch_enablement_allowed(state)
        if not admitted or state.stop_reason or is_multi_node():
            return ""
        if state.enablement.succeeded or state.baseline_tput > 0:
            return ""
        if state.baseline_failure_streak < 1:
            return ""
        if await self.rounds.held() is not None:
            return ""
        loaded = load_boot_observation(state.enablement.launch_observation_path)
        if not probe_would_inform(loaded.observation):
            return ""
        attempt = state.enablement.attempts
        try:
            task = await self.tasks.create(
                kind="boot_probe",
                params={"framework": state.framework},
                idempotency_key=f"boot_probe:{attempt}",
                requires_lanes=["server_lifecycle"],
                side_effects=["launches_server", "reads_server"],
                lease_ttl_sec=int(self.action_registry["boot_probe"].lease_ttl_sec),
            )
        except TerminalTaskReuse:
            return ""
        log.info("ENABLEMENT: dispatched boot probe attempt=%d task=%s", attempt, task.task_id)
        return task.task_id

    async def _open_authoring_round(
        self,
        *,
        params: dict[str, Any],
        idempotency_key: str,
        lanes: list[str],
        lease_ttl_sec: int,
    ) -> str:
        """Acquire the round and create the specialist that holds it, together.

        The holder's task row is written by the acquiring cursor, so the two
        land together or not at all.

        Args:
            params: The specialist's task params.
            idempotency_key: Key the specialist row is created under; also the
                acquire's request id.
            lanes: Lanes the specialist must hold while running.
            lease_ttl_sec: The specialist's lease, and the round's first one.

        Returns:
            str: The holder task id, or ``""`` when nothing was dispatched --
            the machine was excluded, or the key already names a live round.
        """
        holder = uuid.uuid4().hex

        def _join(cur: sqlite3.Cursor) -> None:
            _task, existing = create_in_cursor(
                cur,
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                side_effects=["writes_results", "writes_patches"],
                lease_ttl_sec=lease_ttl_sec,
                task_id=holder,
            )
            if existing:
                raise _DuplicateAuthoring(f"{idempotency_key!r} already names a live authoring specialist")

        try:
            acquired = await self.rounds.open(
                f"enablement-{holder}",
                holder_task_id=holder,
                lease_sec=float(lease_ttl_sec),
                now_unix=time.time(),
                request_id=idempotency_key,
                probe_origin=params["enablement_failure_kind"],
                join=_join,
                evidence={"idempotency_key": idempotency_key},
            )
        except (_DuplicateAuthoring, TerminalTaskReuse) as exc:
            log.info("ENABLEMENT: no round opened -- %s", exc)
            return ""
        if not acquired.ok:
            log.info("ENABLEMENT: round not opened (%s)", acquired.reason)
            return ""
        return holder

    async def _enablement_in_flight(self) -> bool:
        """True while an open round still has work running under it.

        Returns:
            bool: Whether the round is still resolving.
        """
        round_row = await self.rounds.held()
        if round_row is None:
            return False
        if not await self._round_has_live_work(round_row.holder_task_id):
            return False
        await self._renew_enablement_round(round_row)
        return True

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
        # A proposal the Critic has not ruled on has no task of its own, but the
        # round is still resolving until it does.
        for p in self.state.pending_proposals.values():
            if p.action_name != "integrate_patch" or p.decided:
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
            round_row.round_id,
            holder_task_id=round_row.holder_task_id,
            fence=round_row.fence,
            lease_sec=lease,
            now_unix=now,
            request_id=f"renew:{round_row.round_id}:{round_row.fence}:{now:.0f}",
        )

    async def _handoff_enablement_round(self, task: "Task") -> None:
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
            round_row.round_id,
            holder_task_id=specialist,
            fence=round_row.fence,
            new_holder_task_id=task.task_id,
            lease_sec=lease,
            now_unix=time.time(),
            request_id=f"handoff:{task.task_id}",
        )
        if not moved.ok:
            log.warning(
                "ENABLEMENT: round %s not handed to integrate %s (%s)",
                round_row.round_id,
                task.task_id,
                moved.reason,
            )

    async def _charge_round_observation(self, res: dict[str, Any]) -> ProgressBudget:
        """Append what this round's boot did to the ledger, and re-read the budget.

        Exactly one observation is charged per round, whatever it claimed to
        achieve; a round that recorded nothing readable is charged at stage zero
        with no digest rather than exempted.

        Args:
            res: The integrate_patch result dict the round came back with.

        Returns:
            ProgressBudget: What the session has left after this round.
        """
        loaded = load_boot_observation(
            res.get("enablement_observation_path")
            or res.get("after_observation_path")
            or (res.get("bench_result") or {}).get("boot_observation_path")
        )
        held = await self.rounds.held()
        round_id = held.round_id if held is not None else ""
        now = time.time()
        await self.rounds.observe(
            round_id,
            actor_task_id=held.holder_task_id if held is not None else "",
            stage=stage_of(loaded.observation),
            failure_digest=digest_of(loaded.observation),
            now_unix=now,
            request_id=f"observe:{round_id or 'unheld'}:{now:.3f}",
            evidence={"status": str(res.get("status") or ""), "degraded": loaded.degraded},
        )
        return await session_budget(self.rounds)

    async def _settle_enablement_round(self, outcome: str, *, reason: str = "") -> None:
        """End the open round, if one is still open.

        Args:
            outcome: What the round leaves behind; see the round store's
                outcomes.
            reason: Recorded on the settle's outbox row.
        """
        round_row = await self.rounds.held()
        if round_row is None:
            return
        await self.rounds.settle(
            round_row.round_id,
            holder_task_id=round_row.holder_task_id,
            fence=round_row.fence,
            outcome=outcome,
            now_unix=time.time(),
            request_id=f"settle:{round_row.round_id}:{round_row.fence}",
            correctness_verified=outcome == BOOTED,
            evidence={"reason": reason} if reason else {},
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
        await self._record_observation(
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

    async def _maybe_rearm_enablement(self, res: dict[str, Any] | None) -> None:
        """Re-arm, advance, or terminate the enablement retry loop.

        Called on every ``integrate_patch`` completion. An enablement patch has
        three outcomes:

        * ``kept`` — the combo is now runnable: terminal success.
        * ``advanced`` — the patch cleared the prior crash and the boot now
          stops at a new, deeper gap. The patch is stacked and
          ``enablement.launch_log`` replaced, so the next round targets that gap.
        * anything else — no patch is kept.

        Every outcome charges one observation to the round ledger, stops the run
        when the ledger reports the budget spent, and settles the round.

        Args:
            res: The integrate_patch result dict (may be ``None`` / non-dict).
        """
        if not isinstance(res, dict) or not res.get("enablement"):
            return
        state = self.shared_state
        status = str(res.get("status") or "")
        stop_set = ""
        # Only a real specialist round carries this; targeted_build rearm rows
        # do not, and must not clear it.
        spec_tid = str(res.get("specialist_task_id") or "").strip()
        if spec_tid:
            state.enablement.last_specialist_task_id = spec_tid

        def _stack_setup_commands() -> None:
            """Append this round's applied setup commands to the durable stack."""
            cur = list(state.enablement.setup_commands)
            for c in res.get("setup_commands_applied") or []:
                sc = str(c)
                if sc and sc not in cur:
                    cur.append(sc)
            state.enablement.setup_commands = cur

        def _push_kept_round(patches_this_round: list[str]) -> None:
            """Append this round to kept_rounds and re-derive the flat projections.

            Artifacts dedupe last-wins per target.
            """
            rounds = list(state.enablement.kept_rounds)
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

            ``baseline_arg_error_streak`` is deliberately left alone: progress
            elsewhere says nothing about arguments the server rejected outright.
            """
            state.baseline_failure_streak = 0
            state.baseline_total_failures = 0

        def _stack_kept_runtime() -> None:
            """Persist the KEEP'd attempt runtime and localization manifest."""
            action = res.get("enablement_kept_stack_action")
            if isinstance(action, dict) and action:
                state.enablement.kept_stack_action = action
            runtime = res.get("enablement_active_runtime")
            if isinstance(runtime, dict) and runtime:
                state.enablement.active_runtime = runtime
                # Cap at the 5 newest attempt-runtime records.
                records = list(state.enablement.attempt_runtimes)
                records.append(runtime)
                state.enablement.attempt_runtimes = records[-5:]
            # Recorded so the localized closure is not re-fetched next round.
            manifest = res.get("enablement_localization_manifest")
            if isinstance(manifest, dict) and manifest:
                existing = list(state.enablement.localization_manifest)
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
            if state.enablement.origin == "eval":
                # The patch passed the gate, but tput and accuracy only become
                # official once a genuine baseline promotes: hold ``succeeded``
                # and open the revalidation window instead.
                state.enablement.validation_pending = True
                # A fresh generation so the new window's idempotency key cannot
                # reuse a prior terminal TaskRegistry row.
                state.enablement.revalidation_generation += 1
                state.enablement.revalidation_task_id = ""
            else:
                state.enablement.succeeded = True
        elif status == "advanced" or bool(res.get("advanced")):
            # Stack the progressing patches and pivot to the newly-revealed gap.
            _push_kept_round([str(p) for p in (res.get("patches_applied") or []) if str(p)])
            _stack_setup_commands()
            _stack_kept_runtime()
            # Accumulated so a later kept round replays every advance, not just patches.
            adv_envs = res.get("extra_envs_applied") or {}
            adv_args = str(res.get("extra_server_args_applied") or "").strip()
            if adv_envs or adv_args:
                cfg = dict(state.enablement.accepted_config)
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
                # The wall this round advanced to: the next round's before half.
                state.enablement.launch_observation_path = str(res.get("enablement_observation_path") or "")
            _reset_baseline_failure_backstop()
        budget = await self._charge_round_observation(res)
        if budget.exhausted and not state.stop_reason:
            state.set_stop_reason(STALLED_STOP_REASON)
            stop_set = STALLED_STOP_REASON
            log.warning("ENABLEMENT: progress budget spent — %s", budget.reason)
        # Set on every round so neither outlives the round it describes.
        state.enablement.last_grounding_drop_reason = [
            str(d) for d in (res.get("patches_dropped_by_grounding") or [])[:8]
        ]
        state.enablement.patches_span_multiple_roots = bool(res.get("patches_span_multiple_roots"))
        # Phase-synthesised rounds carry no framework_root; keep the last real one.
        res_fw_root = str(res.get("framework_root") or "").strip()
        if res_fw_root:
            state.enablement.framework_root = res_fw_root
        archive = RoundArchive(self.session_dir)
        try:
            archive = snapshot_round(self.session_dir, res)
            if status in ("kept", "advanced"):
                write_setting_script(
                    self.session_dir,
                    state.enablement,
                    framework=state.framework or os.environ.get("FRAMEWORK") or "sglang",
                    model=os.environ.get("MODEL_PATH") or state.model_path or state.reference_model,
                    tp=state.tp or None,
                    max_model_len=state.max_model_len or None,
                    gpu_type=state.gpu_type or os.environ.get("GPU_TYPE") or None,
                )
        except Exception:  # noqa: BLE001 — the settle below must run or the round leaks the machine
            log.warning("enablement: artifact write failed", exc_info=True)
        # Absolute and not the session-relative path the archive records: the
        # revalidation baseline opens this file directly, and the breakdown
        # collector relativizes it on the way into SBD. Only a copy that landed
        # overrides the ``runs/`` original the round reported, since the archive
        # collector drops ``runs/`` and every later reader would resolve nothing.
        archived_config = archive.path_for(ROLE_LAUNCH_CONFIG)
        if status == "kept" and archived_config:
            state.enablement.accepted_config_path = str(Path(self.session_dir) / archived_config)
        # A rearm always ends the round; only a KEEP booted and was graded.
        await self._settle_enablement_round(BOOTED if status == "kept" else FAILED, reason=status)
        state.save(self.session_dir)
        log.info(
            "ENABLEMENT: rearm from integrate status=%s succeeded=%s advanced=%s "
            "stacked=%d high_water=%d digests_left=%d stall_left=%d next_attempt=%d%s",
            status,
            bool(state.enablement.succeeded),
            status == "advanced" or bool(res.get("advanced")),
            len(state.enablement.kept_patches),
            budget.stage_high_water,
            budget.digest_credits_left,
            budget.stall_credits_left,
            state.enablement.attempts,
            f" stop_reason={stop_set}" if stop_set else "",
        )

    async def _pump_enablement_safely(self, *, caller: str) -> None:
        """Phase-independent enablement pump — runs every tick.

        The only PRELUDE exit gate is ``baseline_tput > 0``, which a
        non-runnable combo never reaches, so this cannot be bound to a phase.
        Every dispatch guard lives inside the pumped methods, so calling them
        unconditionally is safe and idempotent.

        Args:
            caller: Label identifying the caller ("tick" / "run"), for logs.
        """
        # Independently, because a raise in one pump must not skip the rest: the
        # one that dispatches the next authoring round is the last of them.
        for pump in (
            self._maybe_route_build_outcomes,
            self._maybe_enqueue_enablement_baseline_revalidation,
            self._maybe_enqueue_boot_probe,
            self._maybe_enqueue_enablement_specialist,
        ):
            try:
                await pump()
            except Exception:  # noqa: BLE001 — a wedged pump would strand the run in PRELUDE
                log.exception("ENABLEMENT %s (%s) failed", pump.__name__, caller)
