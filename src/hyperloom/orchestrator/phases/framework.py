# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT phase handler: candidate discovery/ranking/audit, authoring
specialist dispatch, enablement repair, and Critic-review submission/reauthor."""

from __future__ import annotations
import asyncio
import logging as _logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from hyperloom.common.git_safety import safe_directory_args

from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..state.shared_state import inject_stack_base_params, resolve_grading_anchor_tput

if TYPE_CHECKING:
    from ..state.task_registry import Task
from ..loop.coordinator import (
    DEFAULT_FRAMEWORK_MAX_CANDIDATES,
    PendingProposal,
    _AUTHORED_LANE_MAX_ATTEMPTS,
    _FRAMEWORK_MIN_PER_REPO_TIMEOUT_SEC,
    _framework_config_levers_from_done,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)

# Specialist attempts a local-exploration candidate gets before the phase moves
# on. A candidate that keeps failing to author is not worth the wall clock, but
# one interrupted run — a process restart reclaims an in-flight specialist as
# failed — must not retire it.
_LOCAL_EXPLORE_MAX_ATTEMPTS: int = 3


#: Progress-row status for a candidate the Critic rejected. The gate writes it
#: and the working-memory and priors readers select on it, so the ledger is the
#: only record of a denial and the three sites agree by construction.
FRAMEWORK_CRITIC_DENIED_STATUS: str = "critic_denied"


class FrameworkPhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    # Marker for the candidate-free local-exploration arm (a synthetic
    # "candidate" whose id is ``local_explore:<n>``): the ranker may pick it,
    # and it routes to a write-capable authoring specialist instead of a PR.
    _LOCAL_EXPLORE_KIND = "local_explore"

    async def _on_enter_framework(self, *, from_phase: str) -> None:
        """FRAMEWORK entry hook: trigger the per-batch pump once on entry (best-effort; later batches driven from the main tick).

        Args:
            from_phase: The phase being left, used only for logging.
        """
        log.info(
            "FRAMEWORK entry (from=%s): pumping initial batch",
            from_phase or "<unknown>",
        )
        try:
            await self._pump_framework_agent_phase()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("FRAMEWORK entry pump failed: %r", exc)

    async def _pump_framework_agent_phase(self) -> None:
        """Drive the FRAMEWORK_AGENT phase: enqueue the next candidate. Idempotent; a discover failure flips framework_agent_phase_done so the phase advances rather than wedging."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        # When the baseline could not launch, dispatch a one-shot
        # enablement_specialist before the perf PR-discovery loop.
        try:
            enablement_tid = await self._maybe_enqueue_enablement_specialist()
        except Exception:  # noqa: BLE001 — never wedge the perf pump
            log.exception("ENABLEMENT: enqueue failed")
            enablement_tid = ""
        if enablement_tid:
            return
        if bool(getattr(state, "framework_agent_phase_done", False)):
            return
        # Skip if a framework task is already queued or running.
        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 — defensive
            queued, running = [], []
        for t in (*queued, *running):
            if getattr(t, "kind", "") == "framework_agent":
                return
        # Serialize one candidate at a time: skip while a candidate proposal
        # awaits its (durable) Critic verdict, resolved on a later tick.
        try:
            if any(
                getattr(p, "action_name", "") == "framework_agent" and not getattr(p, "decided", False)
                for p in self.state.pending_proposals.values()
            ):
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        # An authoring specialist (or its downstream integrate_patch) for the
        # current candidate may still be running; wait only on a live TASK
        # (queued/running), NOT on a pending Critic proposal. The
        # proposal-pending case is covered by the ``next_candidate is None``
        # inflight wait below.
        if getattr(state, "framework_agent_authoring_enabled", False):
            try:
                _q = await self.tasks.queued()
                _r = await self.tasks.running()
            except Exception:  # noqa: BLE001 — defensive
                _q, _r = [], []
            if any(
                getattr(t, "kind", "") in ("specialist", "integrate_patch")
                and bool((getattr(t, "params", None) or {}).get("framework_agent_authoring"))
                for t in (*_q, *_r)
            ):
                return
            # Proposal-window guard: the task check above misses the interval
            # between a specialist completing and its integrate_patch becoming a
            # live TASK (the deliverable exists only as a pending Critic
            # proposal). ``_framework_agent_authoring_inflight`` covers pending
            # integrate_patch proposals, serializing one candidate's
            # author->integrate->KEEP/REVERT lifecycle before the next.
            if await self._framework_agent_authoring_inflight():
                return
        # Pick the most promising un-dispatched candidate, or request a new batch.
        next_candidate = await self._select_best_framework_agent_candidate()
        if next_candidate is None:
            # Hold the phase open while pump-discovered authored patches are still
            # benched/reviewed; only when the pump itself discovered a PR batch
            # (an LLM-proposed integrate_patch must not keep FRAMEWORK open).
            discovered_batch = bool(getattr(self.shared_state, "framework_agent_batches", None) or [])
            if (
                discovered_batch
                and getattr(self.shared_state, "framework_agent_authoring_enabled", False)
                and await self._framework_agent_authoring_inflight()
            ):
                return
            # Discover a fresh batch; only DISCOVER_FAILURE_RETRY_LIMIT
            # consecutive failures or an empty-but-valid payload mark the phase done.
            from ..framework import client as _fa_client

            ok = await self._discover_next_framework_batch()
            if not ok:
                # Prefer self-driven local exploration over skipping the whole
                # phase. Discovery is retried automatically on later ticks (once
                # the local-explore specialist completes and no PR candidate
                # remains), so with the local-explore arm enabled the phase now
                # exits via plateau / budget / force-exit rather than a
                # discover-failure streak. Falls back to the historical
                # discover-exhaustion exit when the arm is disabled.
                if await self._maybe_dispatch_local_explore(reason="discover_exhausted"):
                    state.save(self.session_dir)
                    return
                failures = int(getattr(state, "framework_agent_discover_failures", 0) or 0)
                if failures >= _fa_client.DISCOVER_FAILURE_RETRY_LIMIT:
                    # Transient discover failures exhausted — real exit.
                    self._record_framework_agent_phase_done(
                        reason="discover_retries_exhausted",
                        failure_count=failures,
                    )
                    state.framework_agent_phase_done = True
                    state.save(self.session_dir)
                    return
                if failures == 0:
                    # Empty-but-valid payload: tolerate a bounded number of
                    # consecutive empties (transient upstream blip) before giving up.
                    empties = int(getattr(state, "framework_agent_empty_discoveries", 0) or 0) + 1
                    state.framework_agent_empty_discoveries = empties
                    if empties < _fa_client.DISCOVER_FAILURE_RETRY_LIMIT:
                        log.info(
                            "FRAMEWORK: empty discovery batch (%d/%d) — retrying on a later tick",
                            empties,
                            _fa_client.DISCOVER_FAILURE_RETRY_LIMIT,
                        )
                        state.save(self.session_dir)
                        return
                    self._record_framework_agent_phase_done(
                        reason="discover_empty_payload",
                        failure_count=failures,
                    )
                state.framework_agent_phase_done = True
                state.save(self.session_dir)
                return
            # A non-empty batch cleared any prior empty-discovery streak.
            state.framework_agent_empty_discoveries = 0
            next_candidate = await self._select_best_framework_agent_candidate()
            if next_candidate is None:
                if await self._maybe_dispatch_local_explore(reason="no_new_candidates"):
                    state.save(self.session_dir)
                    return
                self._record_framework_agent_phase_done(
                    reason="discover_returned_no_new_candidates",
                    failure_count=int(
                        getattr(state, "framework_agent_discover_failures", 0) or 0,
                    ),
                )
                state.framework_agent_phase_done = True
                state.save(self.session_dir)
                return
        # Local-exploration arm: a candidate-free authoring specialist chosen by
        # the ranker (resident arm) has no upstream diff to judge, so it skips
        # the PR semantic audit and dispatches directly.
        if str(next_candidate.get("kind") or "") == self._LOCAL_EXPLORE_KIND:
            await self._enqueue_framework_agent_local_explore_specialist(next_candidate)
            state.save(self.session_dir)
            return
        # Run semantic audit before the Critic/apply. A confident verdict routes
        # the candidate; an unknown / unavailable audit falls back to both-tracks.
        audit = await self._audit_framework_agent_candidate(next_candidate)
        audit_step = str((audit or {}).get("recommended_next_step") or "")
        _cand_id_log = self._framework_candidate_key(next_candidate)
        # Only honour a skip when the audit is confident AND evidence-backed.
        if audit_step == "skip" and not self._framework_agent_audit_skip_confident(audit):
            log.info(
                "FRAMEWORK: audit skip downgraded (low confidence / no evidence) candidate=%s conf=%s",
                _cand_id_log,
                (audit or {}).get("confidence"),
            )
            audit_step = "author_via_specialist"
        if audit_step == "skip":
            await self._record_framework_agent_audit_skip(next_candidate, audit)
            state.save(self.session_dir)
            return
        # direct_apply needs a clean git checkout; degrade to authoring on a
        # wheel install (no git tree among the framework source roots).
        if audit_step == "direct_framework" and not self._framework_agent_roots_have_git():
            log.info(
                "FRAMEWORK: direct_apply downgraded to authoring "
                "(no git checkout among framework source roots) candidate=%s",
                _cand_id_log,
            )
            audit_step = "author_via_specialist"
        # A candidate from a different concrete framework cannot be
        # direct-applied; downgrade to authoring. ``aiter`` is shared across
        # frameworks and is never treated as a mismatch.
        if audit_step == "direct_framework":
            session_fw = str(getattr(state, "framework", "") or "").strip().lower()
            cand_fw = str(next_candidate.get("framework") or "").strip().lower()
            if not cand_fw:
                repo_token = str(next_candidate.get("repo") or next_candidate.get("discovered_repo_url") or "").lower()
                for _fw_tok in ("sglang", "vllm", "atom"):
                    if f"/{_fw_tok}" in repo_token or repo_token.endswith(_fw_tok):
                        cand_fw = _fw_tok
                        break
            if cand_fw and cand_fw != "aiter" and session_fw and cand_fw != session_fw:
                log.info(
                    "FRAMEWORK: direct_apply downgraded to authoring "
                    "(candidate framework=%s differs from session framework=%s) candidate=%s",
                    cand_fw,
                    session_fw,
                    _cand_id_log,
                )
                audit_step = "author_via_specialist"
        # Submit the candidate as a proposal; the async Critic verdict drives the
        # apply/author enqueue or the critic_denied row on a later tick.
        await self._submit_framework_agent_candidate_for_review(
            next_candidate,
            audit=audit,
            audit_step=audit_step,
        )

    async def _framework_agent_authoring_inflight(self) -> bool:
        """True while a FRAMEWORK-authored patch for an unprocessed candidate is still in flight.

        Counts only items with the ``framework_agent_authoring`` provenance marker whose candidate
        is still unprocessed (no terminal progress row yet). A KERNEL-phase specialist/integrate or
        an orphaned stale proposal therefore never pins the pump.

        Returns:
            ``True`` if a framework-owned specialist/integrate_patch task is queued or running,
            or a framework-owned undecided proposal targets an unprocessed candidate; else ``False``.
        """
        unprocessed_ids = {self._framework_candidate_key(c) for c in self._unprocessed_framework_agent_candidates()}
        # The local-exploration arm's synthetic candidate id never appears in a
        # PR batch, so it is "in flight" while it lacks a terminal progress row.
        processed_ids = self._framework_processed_candidate_keys()

        def _cand_pins_pump(cand_id: str) -> bool:
            """True when an authoring cand_id keeps the pump serialized."""
            if not cand_id or cand_id in unprocessed_ids:
                return True
            return cand_id.startswith("local_explore:") and cand_id not in processed_ids

        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 — defensive
            queued, running = [], []
        for t in (*queued, *running):
            if getattr(t, "kind", "") not in ("specialist", "integrate_patch"):
                continue
            params = getattr(t, "params", None) or {}
            if not params.get("framework_agent_authoring"):
                continue
            if _cand_pins_pump(str(params.get("framework_agent_candidate_id") or "")):
                return True
        # An authored patch awaiting Critic review (or a candidate awaiting its
        # pre-screen verdict) keeps the phase open, but only while the proposal
        # targets a still-unprocessed candidate.
        try:
            for p in self.state.pending_proposals.values():
                if getattr(p, "decided", False):
                    continue
                action = getattr(p, "action_name", "")
                payload = getattr(p, "payload", None) or {}
                if action == "framework_agent":
                    if _cand_pins_pump(str(payload.get("framework_agent_candidate_id") or "")):
                        return True
                elif action == "integrate_patch":
                    iparams = payload.get("params") or {}
                    if not iparams.get("framework_agent_authoring"):
                        continue
                    if _cand_pins_pump(str(iparams.get("framework_agent_candidate_id") or "")):
                        return True
        except Exception:  # noqa: BLE001 — defensive
            pass
        return False

    @staticmethod
    def _framework_agent_audit_skip_confident(audit: dict[str, Any] | None) -> bool:
        """True iff an ``already_*`` skip is safe: concrete evidence + confidence ≥ floor (G5).

        Floor is ``INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_SKIP_MIN_CONFIDENCE``
        (default 0.8). A low-confidence / evidence-free skip must not silently
        bypass the GPU test.

        Args:
            audit: The semantic-audit verdict.

        Returns:
            ``True`` when the skip is confident and evidence-backed.
        """
        if not isinstance(audit, dict) or not (audit.get("evidence") or []):
            return False
        try:
            conf = float(audit.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            floor = float(os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_SKIP_MIN_CONFIDENCE", "0.8"))
        except (TypeError, ValueError):
            floor = 0.8
        return conf >= floor

    @staticmethod
    def _framework_agent_roots_have_git() -> bool:
        """True iff any resolved framework source root is a git work tree (G3 preflight).

        A wheel install (dist-packages, no ``.git``) yields ``False`` → the pump
        degrades ``direct_apply`` to authoring (the executor's git apply / commit
        / reset would otherwise fail).

        Returns:
            ``True`` when at least one source root is inside a git work tree.
        """

        from ..framework.paths import resolve_source_file_allowlist

        for root in resolve_source_file_allowlist():
            p = Path(str(root))
            if not p.is_dir():
                continue
            try:
                cp = subprocess.run(
                    ["git", *safe_directory_args(["-C", str(p), "rev-parse", "--is-inside-work-tree"])],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # One unusable root says nothing about the others.
                continue
            if cp.returncode == 0 and cp.stdout.strip() == "true":
                return True
        return False

    @staticmethod
    def _framework_audit_use_llm_mode() -> str:
        """Resolve the phase-audit LLM policy from the environment.

        ``INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_USE_LLM`` selects:
          - ``off`` — never run the LLM refine (hermetic static verdict only);
          - ``on`` — always run the evidence-gated LLM refine;
          - ``auto`` (default) — only escalate to the LLM when the static
            verdict is uncertain (see :meth:`_framework_audit_verdict_uncertain`).

        Returns:
            One of ``"on"`` / ``"off"`` / ``"auto"``.
        """
        val = (os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_AUDIT_USE_LLM", "") or "").strip().lower()
        if val in ("on", "1", "true", "yes", "always"):
            return "on"
        if val in ("off", "0", "false", "no", "never"):
            return "off"
        return "auto"

    @staticmethod
    def _framework_audit_verdict_uncertain(audit: dict[str, Any] | None) -> bool:
        """True when a static audit verdict is too weak to route on confidently.

        The ``auto`` LLM policy re-runs the audit with ``use_llm=True`` only in
        this case, so the extra chat-completion is spent only where the cheap
        static pass could not decide (``unknown`` status or ``confidence < 0.5``).

        Args:
            audit: The static-layer verdict dict.

        Returns:
            ``True`` when the verdict is ``unknown`` or low-confidence.
        """
        if not isinstance(audit, dict) or not audit:
            return True
        status = str(audit.get("semantic_status") or "").strip().lower()
        if status in ("", "unknown"):
            return True
        try:
            return float(audit.get("confidence") or 0.0) < 0.5
        except (TypeError, ValueError):
            return True

    async def _audit_framework_agent_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Run ``fa phase-audit`` for a candidate; degrade to ``unknown`` on any failure.

        A cached verdict stamped on the candidate (``_audit``) is reused so a
        re-pump / resume never re-audits. An unavailable audit returns an
        ``unknown`` verdict with an empty ``recommended_next_step`` so the pump
        falls back to legacy both-tracks routing.

        Args:
            candidate: The discovered candidate row.

        Returns:
            The semantic-audit verdict dict.
        """
        cached = candidate.get("_audit") if isinstance(candidate, dict) else None
        if isinstance(cached, dict) and cached.get("recommended_next_step") is not None:
            return cached
        state = self.shared_state
        unknown: dict[str, Any] = {
            "semantic_status": "unknown",
            "applicability": "needs_human_review",
            "recommended_next_step": "",
            "confidence": 0.0,
            "evidence": [],
            "risks": ["audit unavailable"],
            "source": "audit_unavailable",
        }
        try:
            import os

            from ..framework import client as _fa_client
            from ..framework.paths import resolve_source_file_allowlist

            roots = list(resolve_source_file_allowlist())
            session_framework = str(getattr(state, "framework", "") or "").strip().lower()
            cand_framework = str(candidate.get("framework") or "").strip().lower()
            # A candidate from a DIFFERENT framework's repo is judged for
            # portability into this session's framework; a blank cand_framework
            # (the common same-framework case) resolves to session_framework.
            is_cross_fw_candidate = bool(cand_framework) and cand_framework != session_framework
            audit_kwargs: dict[str, Any] = dict(
                candidate=candidate,
                framework=cand_framework or session_framework,
                framework_source_roots=roots,
                target_framework=session_framework if is_cross_fw_candidate else "",
                target_framework_source_roots=roots if is_cross_fw_candidate else None,
                session_dir=self.session_dir,
                repo_url=str(candidate.get("repo") or ""),
                diff_url=str(candidate.get("diff_url") or ""),
                primus_cortex_url=os.environ.get("PRIMUS_CORTEX_PR_API", "").strip(),
                timeout_sec=getattr(
                    self,
                    "framework_audit_timeout_sec",
                    _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC,
                ),
            )
            # Selectable LLM deep-read: "off" keeps the hermetic static verdict;
            # "on" always runs the evidence-gated LLM refine; "auto" (default)
            # only escalates to the LLM when the cheap static pass is uncertain.
            llm_mode = self._framework_audit_use_llm_mode()
            audit = await _fa_client.phase_audit(**audit_kwargs, use_llm=(llm_mode == "on"))
            if llm_mode == "auto" and self._framework_audit_verdict_uncertain(audit):
                try:
                    refined = await _fa_client.phase_audit(**audit_kwargs, use_llm=True)
                    if isinstance(refined, dict) and refined.get("recommended_next_step") is not None:
                        audit = refined
                except Exception as refine_exc:  # noqa: BLE001 — refine is best-effort
                    log.debug(
                        "FRAMEWORK: auto LLM audit refine failed (%r); keeping static verdict",
                        refine_exc,
                    )
        except Exception as exc:  # noqa: BLE001 — audit is advisory; never wedge the pump
            log.warning("FRAMEWORK: phase-audit failed (%r); routing as unknown", exc)
            audit = dict(unknown)
        if not isinstance(audit, dict):
            audit = dict(unknown)
        try:
            candidate["_audit"] = audit
        except Exception:  # noqa: BLE001 — caching is best-effort
            pass
        log.info(
            "FRAMEWORK: audit candidate=%s status=%s appl=%s next=%s",
            self._framework_candidate_key(candidate),
            audit.get("semantic_status"),
            audit.get("applicability"),
            audit.get("recommended_next_step"),
        )
        return audit

    @staticmethod
    def _framework_agent_audit_seed_lines(audit: dict[str, Any] | None) -> list[str]:
        """Render audit evidence as authoring-seed lines (empty when no audit)."""
        if not isinstance(audit, dict) or not audit:
            return []
        lines = [
            "",
            "AUDIT EVIDENCE (from fa phase-audit — author against the LIVE source):",
            f"- semantic_status: {audit.get('semantic_status') or 'unknown'}",
            f"- applicability: {audit.get('applicability') or 'unknown'}"
            " (raw upstream diff likely needs rewriting to fit the local tree)",
        ]
        evidence = audit.get("evidence") or []
        if isinstance(evidence, list):
            for ev in evidence[:8]:
                if not isinstance(ev, dict):
                    continue
                local_file = str(ev.get("local_file") or "").strip()
                symbol = str(ev.get("symbol") or "").strip()
                reason = str(ev.get("reason") or "").strip()
                if local_file or symbol or reason:
                    lines.append(
                        f"  • {local_file or '(file?)'}"
                        + (f" [{symbol}]" if symbol else "")
                        + (f": {reason}" if reason else "")
                    )
        risks = audit.get("risks") or []
        if isinstance(risks, list) and risks:
            lines.append("- risks: " + "; ".join(str(r) for r in risks[:4]))
        return lines

    async def _record_framework_agent_audit_skip(
        self,
        candidate: dict[str, Any],
        audit: dict[str, Any] | None,
    ) -> None:
        """Record a terminal progress row + decision.json (+ KB) for an audit-skipped candidate.

        Called when the audit's ``recommended_next_step == "skip"`` (already
        present / not applicable): no Critic, no GPU, no specialist. Writes the
        candidate's fate so the batch advances and a later session can dedup.

        Args:
            candidate: The skipped candidate row.
            audit: The semantic-audit verdict driving the skip.
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        batch_id = str(candidate.get("batch_id") or "")
        semantic = str((audit or {}).get("semantic_status") or "")
        status = "already_present" if semantic.startswith("already_") else "not_applicable"
        progress = getattr(state, "framework_agent_phase_progress", None)
        if not isinstance(progress, list):
            progress = []
            state.framework_agent_phase_progress = progress
        progress.append(
            {
                "candidate_id": cand_id,
                "pr_url": str(candidate.get("pr_url") or ""),
                "status": status,
                "kept": False,
                "provenance": "audit",
                "semantic_status": semantic,
                "confidence": float((audit or {}).get("confidence") or 0.0),
                "batch_id": batch_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "cycle": int(getattr(state, "macro_cycle", 0) or 0),
            }
        )
        if status == "already_present":
            try:
                from ..knowledge.kb_writeback import OUTCOME_ALREADY_PRESENT, write_framework_record

                gap_keywords = candidate.get("gap_keywords") or []
                if isinstance(gap_keywords, str):
                    gap_keywords = [gap_keywords]
                changed_files = candidate.get("changed_files") or []
                if isinstance(changed_files, str):
                    changed_files = [changed_files]
                await write_framework_record(
                    pr_url=str(candidate.get("pr_url") or ""),
                    pr_sha=str(candidate.get("head_sha") or ""),
                    patch_path="",
                    outcome=OUTCOME_ALREADY_PRESENT,
                    tps_delta_pct=0.0,
                    session_id=str(getattr(state, "session_id", "") or ""),
                    framework=str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower(),
                    gap_canonical_id=str(candidate.get("gap_canonical_id") or "").strip(),
                    gap_keywords=[str(k).strip().lower() for k in gap_keywords if str(k).strip()],
                    model_class=str(getattr(state, "model_class", "") or "").strip(),
                    gpu_type=str(getattr(state, "gpu_type", "") or "").strip(),
                    precision=str(getattr(state, "precision", "") or "").strip(),
                    applicability=str((audit or {}).get("applicability") or "").strip(),
                    provenance="phase_audit",
                    changed_files=[str(f).strip() for f in changed_files if str(f).strip()],
                    session_dir=self.session_dir,
                )
            except Exception:  # noqa: BLE001 — KB writeback is best-effort
                log.debug("FRAMEWORK: audit-skip KB writeback failed", exc_info=True)
        log.info(
            "FRAMEWORK: audit skip candidate=%s status=%s (semantic=%s)",
            cand_id,
            status,
            semantic,
        )

    async def _enqueue_framework_agent_authoring_specialist(
        self,
        candidate: dict[str, Any],
        audit: dict[str, Any] | None = None,
        *,
        reauthor_attempt: int = 0,
        critic_feedback: dict[str, Any] | None = None,
    ) -> str:
        """Dispatch a write-capable specialist seeded with ``candidate`` (flows through autosubmit → Critic → integrate_patch → bench → KEEP/REVERT).

        Args:
            candidate: The discovered FRAMEWORK candidate (PR url, title,
                diff url, batch/candidate ids) used to seed the specialist's
                authoring task and provenance markers.
            audit: Optional ``fa phase-audit`` verdict; its evidence (local
                symbols / why the raw diff doesn't fit / where the change
                should land) is injected into the seed so the specialist
                authors against the live source instead of re-discovering it.
            reauthor_attempt: Re-author round; ``> 0`` adds a ``reauthor:{n}``
                idempotency-key suffix so the round gets a fresh task.
            critic_feedback: Prior-round Critic advisory (``required_evidence`` /
                ``advice_text`` / ``risks``) appended to the authoring seed.

        Returns:
            The dispatched specialist ``task_id`` (empty when a livelock-break
            short-circuits the re-dispatch).
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        batch_id = str(candidate.get("batch_id") or "")
        gap_cid = str(candidate.get("gap_canonical_id") or "").strip() or f"gap.framework.{cand_id}"
        title = str(candidate.get("title") or "").strip()
        pr_url = str(candidate.get("pr_url") or "").strip()
        diff_url = str(candidate.get("diff_url") or "").strip()
        is_cross_framework = isinstance(audit, dict) and str(audit.get("layer") or "") == "cross_framework"
        cf_src_framework = ""
        cf_dst_framework = ""
        notes_lines: list[str] = []
        notes_lines.extend(self._framework_agent_audit_seed_lines(audit))
        if is_cross_framework:
            _cf_metrics = audit.get("metrics") if isinstance(audit.get("metrics"), dict) else {}
            cf_src_framework = str(_cf_metrics.get("src_framework") or candidate.get("framework") or "").strip().lower()
            cf_dst_framework = (
                str(_cf_metrics.get("dst_framework") or getattr(state, "framework", "") or "").strip().lower()
            )
            cf_provenance = f"specialist:serving:framework:cross_framework:{cf_src_framework}->{cf_dst_framework}"
            notes_lines.extend(
                [
                    "",
                    "CROSS-FRAMEWORK PORT (rewrite, NOT git apply):",
                    f"- source framework: {cf_src_framework or '(unknown)'}; "
                    f"target (this session) framework: {cf_dst_framework or '(unknown)'}",
                    "- The upstream diff targets a DIFFERENT framework's repo layout / API,",
                    "  so it can NEVER be applied directly. Re-implement the equivalent",
                    "  logic against the TARGET framework's live source in your worktree.",
                ]
            )
            for hit in (audit.get("evidence") or [])[:8]:
                if not isinstance(hit, dict):
                    continue
                notes_lines.append(
                    f"  • target module candidate: {hit.get('dst_module') or '(none)'} "
                    f"(from {hit.get('src_path') or '?'}; feature={hit.get('feature') or '?'})"
                )
            notes_lines.extend(
                [
                    "- Deliverable MUST be a unified-diff source patch in your worktree",
                    "  (``patches_written``) against the target framework source; a pure",
                    "  config-lever proposal is NOT sufficient for a cross-framework port.",
                    f"- In your proposal, set provenance exactly to: {cf_provenance}",
                    f"- Echo source_framework={cf_src_framework!r} and "
                    f"target_framework={cf_dst_framework!r} in the proposal so the KB",
                    "  ledger records the cross-framework outcome.",
                ]
            )
        if critic_feedback:
            req_ev = [str(x).strip() for x in (critic_feedback.get("required_evidence") or []) if str(x).strip()]
            fb_lines = [
                "",
                "PRIOR CRITIC FEEDBACK (re-author round — supply the evidence below this round):",
            ]
            fb_lines.extend(f"  • required evidence: {ev}" for ev in req_ev[:10])
            advice = str(critic_feedback.get("advice_text") or "").strip()
            if advice:
                fb_lines.append(f"- advice: {advice}")
            risks = [str(r).strip() for r in (critic_feedback.get("risks") or []) if str(r).strip()]
            if risks:
                fb_lines.append("- risks: " + "; ".join(risks[:6]))
            notes_lines.extend(fb_lines)
        notes = "\n".join(notes_lines).strip()
        params: dict[str, Any] = {
            # Cross-framework ports route to a dedicated rewrite domain; the
            # same-framework case follows the session's framework kind.
            "domain": (
                "cross_framework_rewrite_specialist" if is_cross_framework else self._authoring_specialist_domain()
            ),
            "gap_canonical_id": gap_cid,
            "gap_symptom": (title or f"Author a framework source patch inspired by {pr_url or cand_id}"),
            "gap_layer": "framework",
            "framework": str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower(),
            "task_kind": "framework_authoring",
            "pr_lead": {"title": title, "url": pr_url, "diff_url": diff_url},
            # Provenance markers for the dispatcher-side authored-patch bridge.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": batch_id,
            "framework_audit": (audit if isinstance(audit, dict) else {}),
            "source": "coordinator_internal",
            "notes": notes,
            # Whole-machine GPU request. Empty on multi-node / no-GPU hosts.
            **self._framework_gpu_params(),
        }
        if is_cross_framework:
            # Thread cross-framework provenance to the integrate_patch->ledger path.
            params["cross_framework"] = True
            params["source_framework"] = cf_src_framework
            params["target_framework"] = cf_dst_framework
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug(
                "FRAMEWORK authoring: warm specialist params failed",
                exc_info=True,
            )
        idem = f"framework_agent_authoring:{batch_id}:{cand_id}"
        if reauthor_attempt > 0:
            idem = f"{idem}:reauthor:{int(reauthor_attempt)}"
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _spec_existing = await self.tasks.create_or_return_existing(
            kind="specialist",
            params=params,
            idempotency_key=idem,
            requires_lanes=lanes,
            side_effects=["writes_results", "writes_patches"],
            lease_ttl_sec=ttl,
        )
        from ..state.task_registry import TERMINAL_STATES as _TERMINAL_STATES

        if _spec_existing and str(getattr(spec_task, "state", "") or "") in _TERMINAL_STATES:
            already_rows = self._framework_processed_candidate_keys()
            authoring_inflight = await self._framework_agent_authoring_inflight()
            if cand_id and cand_id not in already_rows and not authoring_inflight:
                try:
                    recovered = await self._recover_framework_agent_authoring_outcome(
                        specialist_task=spec_task,
                    )
                except Exception:  # noqa: BLE001 — never wedge the pump
                    log.exception(
                        "FRAMEWORK: terminal authoring outcome recovery failed candidate=%s",
                        cand_id,
                    )
                    recovered = False
                if not recovered:
                    log.warning(
                        "FRAMEWORK: terminal authoring outcome unavailable candidate=%s state=%s",
                        cand_id,
                        getattr(spec_task, "state", ""),
                    )
                    # Stamp a terminal row so an unrecoverable outcome cannot make
                    # the pump re-select the same finished specialist forever.
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=batch_id,
                        status="recovery_failed",
                        rationale="authoring outcome unrecoverable from persisted results",
                        provenance="pump",
                    )
                return ""
        # Map specialist task -> candidate so the authored-outcome bridge can
        # resolve the PR-URL candidate id from the downstream integrate_patch.
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        try:
            if spec_tid and cand_id:
                if not isinstance(getattr(state, "framework_agent_specialist_candidate_map", None), dict):
                    state.framework_agent_specialist_candidate_map = {}
                state.framework_agent_specialist_candidate_map[spec_tid] = cand_id
                state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort provenance
            log.debug(
                "FRAMEWORK authoring: specialist->candidate map write failed",
                exc_info=True,
            )
        log.info(
            "FRAMEWORK: dispatched authoring specialist candidate=%s batch=%s gap=%s",
            cand_id,
            batch_id,
            gap_cid,
        )
        return spec_tid

    @staticmethod
    def _coerce_needs_gpu(value: Any) -> bool:
        """Coerce a params ``needs_gpu`` value (bool | str) to bool.

        Matches the truthy set used by ``intent_router`` / the dispatcher so a
        JSON-string ``"true"`` and a real ``True`` route identically.

        Args:
            value: The raw ``needs_gpu`` params value.

        Returns:
            bool: Whether the specialist requests a GPU lease.
        """
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _framework_gpu_params(self) -> dict[str, Any]:
        """Return the ``{needs_gpu, gpu_count}`` params for framework authoring.

        ``gpu_count`` defaults to the whole-machine pool capacity.

        Returns:
            dict: ``{"needs_gpu": True, "gpu_count": <n>}`` on single-node hosts
            with a non-empty whole-machine pool, else ``{}`` (authoring falls
            back to the research-lane-only path).
        """
        try:
            from ..actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return {}
        except Exception:  # noqa: BLE001 — treat probe failure as single-node
            log.debug("framework GPU: multi-node probe failed", exc_info=True)
        cap = int(getattr(self.framework_gpu_pool, "capacity", 0) or 0)
        if cap <= 0:
            return {}
        return {"needs_gpu": True, "gpu_count": cap}

    def _framework_authoring_lanes_ttl(self, params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        """Resolve lanes + lease TTL for an internally-dispatched framework specialist.

        When ``needs_gpu`` is set the task acquires the cap-1
        ``gpu_research_lane`` (in addition to ``research_lane``) and its lease
        TTL is re-sourced from the GPU wall budget.

        Args:
            params: The specialist params (checked for ``needs_gpu``).
            base_ttl_sec: The default lane lease TTL (raised, never lowered, for
                a GPU task).

        Returns:
            ``(lanes, ttl_sec)`` — ``["research_lane"]`` (+ ``gpu_research_lane``
            when GPU) and the (possibly budget-raised) lease TTL.
        """
        lanes = ["research_lane"]
        ttl = int(base_ttl_sec or 0)
        if self._coerce_needs_gpu(params.get("needs_gpu")):
            lanes.append("gpu_research_lane")
            try:
                ttl = self._gpu_lease_ttl_sec(ttl, params=params)
            except Exception:  # noqa: BLE001 — fall back to the base TTL
                log.exception(
                    "framework GPU: gpu_research_lane TTL re-source failed; using base TTL",
                )
        return lanes, ttl

    def _maybe_rearm_authored_lane(self, res: dict[str, Any] | None) -> None:
        """Unified rearm dispatcher for all authored lanes.

        Routes to the lane-specific handler:

        * ``enablement`` lane → :meth:`_maybe_rearm_enablement` (unchanged
          semantics; ``advanced`` / ``kept`` / stall logic preserved).
        * ``perf_framework`` / ``perf_explore`` lanes with ``apply_failed``
          status → increment per-candidate apply-fail retry counter; below cap
          clear the in-flight guard so :meth:`_enqueue_author_specialist` can
          be called from the dispatcher; at/above cap stamp a terminal
          progress row.

        All other statuses for perf lanes are not handled here (they go through
        the existing writeback / progress-stamp paths).

        Args:
            res: The ``integrate_patch`` or ``framework_agent`` result dict.
        """
        if not isinstance(res, dict):
            return
        lane = str(res.get("lane") or "")
        status = str(res.get("status") or "")

        if lane == "enablement" or res.get("enablement"):
            self._maybe_rearm_enablement(res)
            return

        if status != "apply_failed":
            # Non-apply-failed perf-lane results go through the writeback path.
            return

        if lane not in ("perf_framework", "perf_explore"):
            return

        # Determine the candidate key for tracking retry attempts.
        candidate = res.get("candidate")
        if not isinstance(candidate, dict):
            candidate = {}
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            cand_id = str(res.get("specialist_task_id") or "").strip()
        if not cand_id:
            return

        batch_id = str(
            (candidate.get("batch_id") if isinstance(candidate, dict) else None) or res.get("batch_id") or ""
        )

        state = self.shared_state

        existing = state.apply_fail_reauthor_attempts
        apply_fail_attempts: dict[str, int] = existing if isinstance(existing, dict) else {}
        prior = int(apply_fail_attempts.get(cand_id, 0) or 0)
        attempt = prior + 1
        apply_fail_attempts[cand_id] = attempt
        state.apply_fail_reauthor_attempts = apply_fail_attempts

        log.info(
            "AUTHORED_LANE rearm: lane=%s cand_id=%s apply_fail_attempt=%d cap=%d",
            lane,
            cand_id,
            attempt,
            _AUTHORED_LANE_MAX_ATTEMPTS,
        )

        if attempt > _AUTHORED_LANE_MAX_ATTEMPTS:
            # Cap reached: stamp a terminal progress row.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="apply_fail_cap",
                kept=False,
                rationale=f"apply_failed {attempt} times (cap={_AUTHORED_LANE_MAX_ATTEMPTS})",
                provenance="apply_fail_retry",
            )
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.debug("authored_lane: save after cap stamp failed", exc_info=True)
            return

        # Under cap: store the retry context for the dispatcher to pick up.
        vetting_drops_raw = res.get("patches_dropped_by_grounding")
        retry_ctx: dict[str, Any] = {
            "cand_id": cand_id,
            "batch_id": batch_id,
            "lane": lane,
            "attempt": attempt,
            "retry_feedback": res.get("retry_feedback") or [],
            "prior_patches": res.get("prior_patches") or [],
            "candidate": candidate,
            "specialist_task_id": str(res.get("specialist_task_id") or ""),
        }
        if isinstance(vetting_drops_raw, list) and vetting_drops_raw:
            retry_ctx["vetting_drops"] = [str(d) for d in vetting_drops_raw[:8]]
        pending = state.apply_fail_retry_pending or []
        if not isinstance(pending, list):
            pending = []
        pending.append(retry_ctx)
        state.apply_fail_retry_pending = pending
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.debug("authored_lane: save after retry-pending failed", exc_info=True)

    async def _enqueue_author_specialist(
        self,
        *,
        lane: str,
        candidate: "dict[str, Any] | None" = None,
        specialist_task_id: str = "",
        attempt: int = 1,
        retry_feedback: "list[dict[str, Any]] | None" = None,
        critic_feedback: "dict[str, Any] | None" = None,
        vetting_drops: "list[str] | None" = None,
    ) -> str:
        """Dispatch a fresh authoring specialist for an apply-failure retry.

        Handles the ``perf_framework`` and ``perf_explore`` lanes only
        (enablement uses :meth:`_maybe_enqueue_enablement_specialist`).
        Injects structured apply-failure feedback into the specialist mandate
        and uses a ``:retry:{n}`` idempotency suffix to get a fresh task that
        reuses the existing worktree via the idempotency-based worktree lookup.

        Args:
            lane: ``"perf_framework"`` or ``"perf_explore"``.
            candidate: The candidate dict (for perf_framework lane).
            specialist_task_id: The original specialist task that produced the
                failing patch (for worktree reuse + provenance).
            attempt: Retry attempt number (1-based; appended to idempotency key).
            retry_feedback: List of :class:`~._apply_feedback.ApplyFeedback`
                dicts from the failed apply, injected into the specialist mandate.
            critic_feedback: Optional prior Critic advisory (for reauthor retries
                that also had a Critic ``needs_review`` verdict).

        Returns:
            The dispatched specialist ``task_id`` (empty on failure / skip).
        """
        if lane not in ("perf_framework", "perf_explore"):
            log.warning("_enqueue_author_specialist: unsupported lane=%s — skipping", lane)
            return ""

        candidate = candidate or {}
        retry_feedback = retry_feedback or []
        state = self.shared_state

        feedback_lines: list[str] = []
        if vetting_drops:
            feedback_lines.append("")
            feedback_lines.append(
                "PATCH GROUNDING FAILURE: the prior round's patches were dropped by "
                "the safety gate before reaching integration. The worktree contains "
                "the correct framework tree — edit files there and the diff is "
                "harvested automatically. Do not switch to the artifacts_written "
                "channel to avoid the gate."
            )
            feedback_lines.append("Dropped targets: " + "; ".join(vetting_drops[:4]))
            feedback_lines.append("")
        if retry_feedback:
            feedback_lines.append("")
            feedback_lines.append(
                "APPLY FAILURE FEEDBACK (previous patch failed to apply; "
                "study the errors below and produce a corrected patch):"
            )
            for fb_dict in retry_feedback[:5]:  # cap at 5 entries for prompt brevity
                from ..actions.executors._apply_feedback import ApplyFeedback

                fb = ApplyFeedback.from_dict(fb_dict) if isinstance(fb_dict, dict) else None
                if fb is not None:
                    feedback_lines.append("")
                    feedback_lines.append(fb.format_for_mandate())
            feedback_lines.append("")

        if lane == "perf_framework":
            # Re-dispatch a framework_agent authoring specialist with apply
            # failure context injected via a critic_feedback-style note.
            cand_id = self._framework_candidate_key(candidate)
            if not cand_id:
                log.warning("_enqueue_author_specialist: perf_framework missing cand_id")
                return ""
            # Look up original audit from the specialist task params if available.
            audit: dict[str, Any] = {}
            if specialist_task_id:
                try:
                    spec_task = await self.tasks.get(specialist_task_id)
                    spec_params = dict(getattr(spec_task, "params", None) or {})
                    raw_audit = spec_params.get("framework_audit")
                    if isinstance(raw_audit, dict):
                        audit = raw_audit
                except Exception:  # noqa: BLE001
                    pass
            # Merge apply feedback + critic feedback into a single note block.
            merged_feedback = dict(critic_feedback or {})
            if feedback_lines:
                existing_advice = str(merged_feedback.get("advice_text") or "")
                apply_advice = "\n".join(feedback_lines)
                merged_feedback["advice_text"] = apply_advice + ("\n\n" + existing_advice if existing_advice else "")
            try:
                new_task_id = await self._enqueue_framework_agent_authoring_specialist(
                    candidate,
                    audit=audit,
                    reauthor_attempt=attempt,
                    critic_feedback=merged_feedback if merged_feedback else None,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "_enqueue_author_specialist: perf_framework dispatch failed cand=%s attempt=%d",
                    cand_id,
                    attempt,
                )
                return ""
            log.info(
                "AUTHORED_LANE: dispatched perf_framework retry specialist cand=%s attempt=%d task=%s",
                cand_id,
                attempt,
                new_task_id,
            )
            return new_task_id

        # perf_explore lane: reauthor from original specialist worktree.
        # Build a minimal candidate proxy from the specialist task params.
        gap_cid = ""
        gap_symptom = ""
        framework_name = str(getattr(state, "framework", "") or "").strip().lower()
        if specialist_task_id:
            try:
                spec_task = await self.tasks.get(specialist_task_id)
                spec_params = dict(getattr(spec_task, "params", None) or {})
                gap_cid = str(spec_params.get("gap_canonical_id") or "").strip()
                gap_symptom = str(spec_params.get("gap_symptom") or "").strip()
                framework_name = str(spec_params.get("framework") or framework_name).strip().lower()
            except Exception:  # noqa: BLE001
                pass
        if not gap_cid:
            gap_cid = f"gap.explore.retry.{specialist_task_id or 'unknown'}"
        notes_lines: list[str] = list(feedback_lines)
        if critic_feedback:
            req_ev = [str(x).strip() for x in (critic_feedback.get("required_evidence") or []) if str(x).strip()]
            if req_ev:
                notes_lines.append("")
                notes_lines.append("PRIOR CRITIC FEEDBACK (also address this):")
                notes_lines.extend(f"  • {ev}" for ev in req_ev[:10])
            advice = str(critic_feedback.get("advice_text") or "").strip()
            if advice:
                notes_lines.append(f"- advice: {advice}")
        notes = "\n".join(notes_lines).strip()
        params: dict[str, Any] = {
            "domain": "serving_specialist",
            "source_phase": "EXPLORE",
            "gap_canonical_id": gap_cid,
            "gap_symptom": gap_symptom or f"Retry apply-failed patch for {gap_cid}",
            "gap_layer": "perf_explore",
            "framework": framework_name,
            "task_kind": "explore_apply_retry",
            "source": "coordinator_internal",
            "notes": notes,
            "apply_retry_attempt": attempt,
            "prior_patches": retry_feedback[0].get("patch") if retry_feedback else "",
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001
            pass
        idem = f"perf_explore_authoring:{gap_cid}:retry:{attempt}"
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        try:
            spec_task, _ = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                side_effects=["writes_results", "writes_patches"],
                lease_ttl_sec=ttl,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "_enqueue_author_specialist: perf_explore dispatch failed gap=%s attempt=%d",
                gap_cid,
                attempt,
            )
            return ""
        new_tid = str(getattr(spec_task, "task_id", "") or "")
        log.info(
            "AUTHORED_LANE: dispatched perf_explore retry specialist gap=%s attempt=%d task=%s",
            gap_cid,
            attempt,
            new_tid,
        )
        return new_tid

    async def _drain_apply_fail_retry_pending(self) -> None:
        """Dispatch authoring specialists for any queued apply-failure retries.

        Called from the dispatcher after every ``integrate_patch`` completion.
        Drains :attr:`SharedState.apply_fail_retry_pending` (a list of retry
        context dicts populated by :meth:`_maybe_rearm_authored_lane`) by
        calling :meth:`_enqueue_author_specialist` for each entry, then
        clearing the list.

        Best-effort: errors are logged and the list is cleared regardless so
        stale contexts cannot accumulate.
        """
        state = self.shared_state
        pending: list[dict[str, Any]] = state.apply_fail_retry_pending or []
        if not isinstance(pending, list) or not pending:
            return
        # Consume the list atomically so a concurrent call doesn't double-fire.
        to_dispatch = list(pending)
        state.apply_fail_retry_pending = []
        for ctx in to_dispatch:
            if not isinstance(ctx, dict):
                continue
            lane = str(ctx.get("lane") or "")
            attempt = int(ctx.get("attempt") or 1)
            candidate = ctx.get("candidate") or {}
            specialist_task_id = str(ctx.get("specialist_task_id") or "")
            retry_feedback = list(ctx.get("retry_feedback") or [])
            vetting_drops = list(ctx.get("vetting_drops") or [])
            try:
                await self._enqueue_author_specialist(
                    lane=lane,
                    candidate=candidate,
                    specialist_task_id=specialist_task_id,
                    attempt=attempt,
                    retry_feedback=retry_feedback,
                    vetting_drops=vetting_drops or None,
                )
            except Exception:  # noqa: BLE001 — never wedge the dispatcher
                log.exception(
                    "_drain_apply_fail_retry_pending: dispatch failed lane=%s attempt=%d",
                    lane,
                    attempt,
                )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.debug("drain_apply_fail: save failed", exc_info=True)

    @staticmethod
    def _framework_candidate_key(row: dict[str, Any] | None) -> str:
        """Canonical FRAMEWORK candidate dedup/progress key (see ``candidate_key``).

        Thin wrapper over
        :func:`~hyperloom.orchestrator.framework.artifacts.candidate_key` so every candidate
        selection / dedup / progress-row / idempotency site derives the key
        from one place (``candidate_id or pr_url or ref``). Prevents the
        asymmetry where a candidate carrying only a ``pr_url`` failed to dedup
        against its own progress row.

        Args:
            row: A candidate dict or ``framework_agent_phase_progress`` row.

        Returns:
            The candidate key, or ``""`` when no identity field is set.
        """
        from ..framework.artifacts import candidate_key

        return candidate_key(row)

    def _framework_processed_candidate_keys(self) -> set[str]:
        """Set of candidate keys that already carry a terminal progress row.

        A candidate is "processed" once any ``framework_agent_phase_progress``
        row is keyed on it; such a candidate must never be re-selected. Progress
        rows store the key in their ``candidate_id`` field, so this reuses
        :meth:`_framework_candidate_key`.

        Returns:
            The set of processed candidate keys (possibly empty).
        """
        return {
            self._framework_candidate_key(p)
            for p in (getattr(self.shared_state, "framework_agent_phase_progress", None) or [])
            if isinstance(p, dict) and self._framework_candidate_key(p)
        }

    def _unprocessed_framework_agent_candidates(self) -> list[dict[str, Any]]:
        """Return all not-yet-processed candidates in the latest batch (order preserved).

        Processed = the candidate key already appears in
        ``framework_agent_phase_progress`` (see
        :meth:`_framework_processed_candidate_keys`).

        Returns:
            The list of candidate dicts lacking a progress row (possibly empty).
        """
        state = self.shared_state
        batches = getattr(state, "framework_agent_batches", None) or []
        if not batches:
            return []
        latest = batches[-1]
        if not isinstance(latest, dict):
            return []
        candidates = latest.get("candidates") or []
        if not isinstance(candidates, list):
            return []
        processed = self._framework_processed_candidate_keys()
        out: list[dict[str, Any]] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_id = self._framework_candidate_key(cand)
            if cand_id and cand_id not in processed:
                out.append(cand)
        return out

    def _select_next_framework_agent_candidate(self) -> dict[str, Any] | None:
        """Return the next unprocessed candidate in the latest batch (linear order).

        Used for existence checks and as the deterministic fallback when the
        agent-driven ranker (:meth:`_select_best_framework_agent_candidate`) is
        unavailable.

        Returns:
            The first candidate dict not yet recorded in the phase progress, or
            ``None`` when no batch exists or all are processed.
        """
        unprocessed = self._unprocessed_framework_agent_candidates()
        return unprocessed[0] if unprocessed else None

    async def _select_best_framework_agent_candidate(self) -> dict[str, Any] | None:
        """Pick the single most promising unprocessed candidate via the agent.

        The batch is discovered once; each FRAMEWORK exploration then asks the
        agent (LLM) to choose — among the *currently available* candidates — the
        one most likely to improve serving throughput for this workload, instead
        of grinding through the batch in discovery order. Degrades safely: any
        ranker failure falls back to the first unprocessed candidate so the pump
        never wedges.

        Returns:
            The chosen candidate dict, or ``None`` when none remain.
        """
        unprocessed = self._unprocessed_framework_agent_candidates()
        if not unprocessed:
            return None
        # Resident local-exploration arm: offer a candidate-free "author from
        # live source" option alongside the discovered PRs so the ranker can
        # choose it when the PR leads look weak / already-present / off the
        # current bottleneck. Only injected when a PR batch already exists (the
        # no-batch path stays the discovery trigger).
        ranking_set = list(unprocessed)
        pseudo = self._make_local_explore_pseudo_candidate()
        if pseudo is not None:
            ranking_set.append(pseudo)
        if len(ranking_set) == 1:
            return ranking_set[0]
        try:
            chosen = await self._rank_framework_agent_candidates_llm(ranking_set)
        except Exception:  # noqa: BLE001 — ranking is advisory; never wedge the pump
            log.debug("FRAMEWORK: agent candidate ranking failed", exc_info=True)
            chosen = None
        if chosen is not None:
            return chosen
        # Deterministic fallback: discovery order (never the pseudo-candidate).
        return unprocessed[0]

    def _authoring_specialist_domain(self) -> str:
        """Pick the authoring domain that matches the session's framework kind.

        Returns:
            ``"framework_rewrite_specialist"`` for a scriptable (server-less)
            framework, else ``"serving_specialist"``.
        """
        from ..specialists.domains import authoring_domain_for_framework

        return authoring_domain_for_framework(getattr(self.shared_state, "framework", ""))

    def _render_rewrite_evidence_for_prompt(self) -> str:
        """Render the measured host-side rewrite evidence as prompt lines.

        Returns:
            The evidence block, or ``""`` when no profile has produced one yet.
            Empty is normal on the first FRAMEWORK pass (the arm can run before
            any profile has landed), and the specialist's own reading of the
            source is the fallback.
        """
        path = str(getattr(self.shared_state, "last_framework_rewrite_evidence", "") or "").strip()
        if not path:
            return ""
        try:
            import json as _json

            from ..actions.executors._framework_rewrite_evidence import summarize_for_prompt

            document = _json.loads(Path(path).read_text(encoding="utf-8"))
            return summarize_for_prompt(document)
        except Exception:  # noqa: BLE001 — advisory only
            log.warning("FRAMEWORK: rewrite-evidence render failed path=%s", path, exc_info=True)
            return ""

    def _rewrite_evidence_absence_note(self) -> str:
        """Explain an empty evidence block instead of implying there is nothing to find.

        "No candidates" and "the probe never produced any" read identically to a
        specialist, and only the first one means the source is already clean.
        Saying which it is decides whether the specialist should trust the
        silence or go read the loop itself.

        Returns:
            str: The prompt note describing why no evidence is present.
        """
        read_the_source = (
            "Locate the candidates by reading the source: find the denoising / "
            "rollout loop and ask, for each call inside it, whether the result "
            "can change across iterations."
        )
        status = str(getattr(self.shared_state, "last_framework_rewrite_evidence_status", "") or "").strip()
        if status == "no_candidates":
            return (
                "The host-side probe ran and found no rewrite candidates. Treat "
                "that as a measured negative for the patterns it covers "
                "(host round-trips, host syncs, device residency, collective "
                "fusion, memoization, loop hoisting) and look elsewhere. " + read_the_source
            )
        if status and status != "ok":
            return (
                "No host-side rewrite evidence is available because the probe did "
                f"not deliver any: {status}. This is a broken instrument, NOT a "
                "measured negative -- do not conclude the loop is clean. " + read_the_source
            )
        if str(getattr(self.shared_state, "last_framework_rewrite_evidence", "") or "").strip():
            # Reached only when a document is on record but rendering it produced
            # nothing, so the evidence exists and this prompt cannot show it.
            return (
                "Host-side rewrite evidence was collected but could not be "
                "rendered here, so its absence below means nothing. " + read_the_source
            )
        return "No host-side rewrite evidence has been collected yet for this session. " + read_the_source

    def _framework_local_explore_arm_enabled(self) -> bool:
        """True when the candidate-free local-exploration arm may run.

        Requires the authoring capability (``framework_agent_authoring_enabled``)
        and the dedicated toggle (``framework_local_explore_enabled``, default
        on; ``--no-framework-local-explore`` opts out).

        Returns:
            ``True`` when both toggles allow the arm.
        """
        state = self.shared_state
        return bool(getattr(state, "framework_agent_authoring_enabled", False)) and bool(
            getattr(state, "framework_local_explore_enabled", True)
        )

    def _compose_framework_local_explore_gap(self) -> tuple[str, list[str]]:
        """Compose the ``(gap, keywords)`` steering the local-exploration arm.

        Reuses the same bottleneck-aware composer as PR discovery so the
        specialist attacks the current hot path.

        Returns:
            A ``(gap_description, keywords)`` tuple (``("", [])`` on failure).
        """
        state = self.shared_state
        try:
            from ..actions.executors._framework_gap_composer import compose_gap

            return compose_gap(
                framework=str(getattr(state, "framework", "") or ""),
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                model_class=str(getattr(state, "model_class", "") or ""),
                precision=str(getattr(state, "precision", "") or ""),
                profile_kernel_breakdown_path=getattr(state, "last_profile_kernel_breakdown", None),
                rewrite_evidence_path=getattr(state, "last_framework_rewrite_evidence", None),
            )
        except Exception:  # noqa: BLE001 — advisory only
            log.debug("FRAMEWORK: local-explore gap compose failed", exc_info=True)
            return "", []

    def _next_local_explore_candidate_id(self) -> str:
        """Return the next unique local-exploration candidate id.

        The sequence is the count of local-exploration progress rows already
        recorded, so the id is stable while an attempt is in flight (no new row
        yet) and increments once it reaches a terminal row — yielding a fresh
        attempt each round until the phase plateaus.

        Returns:
            An id of the form ``local_explore:<n>``.
        """
        progress = getattr(self.shared_state, "framework_agent_phase_progress", None) or []
        n = sum(
            1 for p in progress if isinstance(p, dict) and str(p.get("candidate_id") or "").startswith("local_explore:")
        )
        return f"local_explore:{n}"

    def _make_local_explore_pseudo_candidate(self) -> dict[str, Any] | None:
        """Build the synthetic local-exploration candidate, or ``None`` when disabled.

        Returns:
            A candidate dict tagged ``kind="local_explore"`` for the ranker, or
            ``None`` when the arm is disabled.
        """
        if not self._framework_local_explore_arm_enabled():
            return None
        gap, keywords = self._compose_framework_local_explore_gap()
        cand_id = self._next_local_explore_candidate_id()
        title = (
            f"local source exploration ({gap})"
            if gap
            else "local source exploration (author a throughput patch from live source + profile)"
        )
        return {
            "kind": self._LOCAL_EXPLORE_KIND,
            "candidate_id": cand_id,
            "title": title,
            "repo": "(local source)",
            "framework": str(getattr(self.shared_state, "framework", "") or "").strip().lower(),
            "gap_description": gap,
            "gap_keywords": keywords,
            "gap_canonical_id": f"gap.framework.local_explore.{cand_id}",
        }

    async def _maybe_dispatch_local_explore(self, *, reason: str) -> bool:
        """Dispatch a local-exploration specialist when the arm is enabled.

        Used as the discovery-exhaustion fallback: rather than marking the phase
        done, author a patch from the live source. No-op when the arm is disabled.

        Args:
            reason: Short provenance tag recorded on the dispatch log line.

        Returns:
            ``True`` when a specialist was dispatched, else ``False``.
        """
        if not self._framework_local_explore_arm_enabled():
            return False
        pseudo = self._make_local_explore_pseudo_candidate()
        if pseudo is None:
            return False
        tid = await self._enqueue_framework_agent_local_explore_specialist(pseudo, reason=reason)
        return bool(tid)

    async def _enqueue_framework_agent_local_explore_specialist(
        self,
        candidate: dict[str, Any],
        *,
        reason: str = "",
    ) -> str:
        """Dispatch a candidate-free authoring specialist (no upstream PR lead).

        The specialist reads the live framework source + profiling evidence (and
        may web-search the latest upstream code) and authors the best throughput
        win, flowing through the same autosubmit -> Critic -> integrate_patch ->
        bench -> KEEP/REVERT path as the PR-authoring track. Empty deliverables
        stamp an ``author_empty`` progress row (counts as a no-KEEP for plateau).

        Args:
            candidate: The synthetic ``local_explore`` candidate (carries the
                candidate id + composed gap).
            reason: Short provenance tag for the dispatch log line.

        Returns:
            The dispatched specialist ``task_id`` (empty on a livelock-break
            short-circuit).
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate) or self._next_local_explore_candidate_id()
        gap = str(candidate.get("gap_description") or "").strip()
        gap_cid = str(candidate.get("gap_canonical_id") or "").strip() or f"gap.framework.local_explore.{cand_id}"
        framework = str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower()
        # Route by framework kind. An iterative model pipeline and a
        # request-serving framework share almost no optimization surface, so the
        # scriptable case goes to the rewrite domain; everything else resolves to
        # the serving domain this dispatch has always used. The static guidance
        # for each lives in its domain description, and the per-kind brief in
        # ``_TASK_KIND_BRIEFS``; only measured, per-dispatch evidence is passed
        # as notes below.
        domain = self._authoring_specialist_domain()
        rewrite_arm = domain == "framework_rewrite_specialist"
        notes = ""
        if rewrite_arm:
            notes = self._render_rewrite_evidence_for_prompt() or self._rewrite_evidence_absence_note()
        try:
            state.upsert_gap(
                {
                    "canonical_id": gap_cid,
                    "symptom": gap or "Author a throughput patch from live source + profiling evidence",
                    "layer": "framework",
                    "severity": "medium",
                    "domain_hint": domain,
                    "source": "coordinator_internal",
                }
            )
        except Exception:  # noqa: BLE001
            log.debug("FRAMEWORK local-explore: upsert_gap failed", exc_info=True)
        prior_attempts: list[dict[str, Any]] = []
        try:
            memory = self._build_framework_working_memory()
            for t in memory.get("tried_and_why") or []:
                if isinstance(t, dict) and str(t.get("ref") or "").strip():
                    prior_attempts.append(t)
        except Exception:  # noqa: BLE001
            pass
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": "FRAMEWORK_AGENT",
            "gap_canonical_id": gap_cid,
            "gap_symptom": (gap or "Author a framework source patch from live source + profile evidence"),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "framework_local_explore",
            "prior_attempts": prior_attempts,
            "notes": notes,
            # Same provenance markers as the PR-authoring track so the
            # autosubmit -> integrate_patch -> authored-outcome bridge applies.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": "",
            "framework_audit": {},
            "framework_local_explore": True,
            "source": "coordinator_internal",
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug("FRAMEWORK local-explore: warm specialist params failed", exc_info=True)
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        create_kwargs: dict[str, Any] = {
            "kind": "specialist",
            "params": params,
            "requires_lanes": lanes,
            "side_effects": ["writes_results", "writes_patches"],
            "lease_ttl_sec": ttl,
        }
        # The registry de-duplicates by key and hands back whatever row it finds,
        # so a candidate whose specialist failed keeps resolving to that failure:
        # the phase re-selects the candidate every tick, logs a dispatch, and
        # nothing runs. A specialist that was mid-flight when the process died is
        # reclaimed as failed, so one interrupted run retires a candidate for
        # good. Retries take a fresh key, which is what this registry documents.
        base_idem = f"framework_agent_local_explore:{cand_id}{self._cycle_idem_suffix()}"
        spec_task = None
        _spec_existing = False
        for attempt in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
            idem = base_idem if attempt == 0 else f"{base_idem}:r{attempt}"
            spec_task, _spec_existing = await self.tasks.create_or_return_existing(
                idempotency_key=idem,
                **create_kwargs,
            )
            if not (_spec_existing and str(getattr(spec_task, "state", "") or "") == "failed"):
                break
            log.info(
                "FRAMEWORK local-explore: %s already failed under %s; retrying candidate %s",
                getattr(spec_task, "task_id", "?"),
                idem,
                cand_id,
            )
        else:
            log.warning(
                "FRAMEWORK local-explore: candidate %s exhausted %d specialist "
                "attempt(s); leaving it to the phase to select another",
                cand_id,
                _LOCAL_EXPLORE_MAX_ATTEMPTS,
            )
            return ""
        from ..state.task_registry import TERMINAL_STATES as _TERMINAL_STATES

        if _spec_existing and str(getattr(spec_task, "state", "") or "") in _TERMINAL_STATES:
            already_rows = self._framework_processed_candidate_keys()
            authoring_inflight = await self._framework_agent_authoring_inflight()
            if cand_id and cand_id not in already_rows and not authoring_inflight:
                try:
                    recovered = await self._recover_framework_agent_authoring_outcome(
                        specialist_task=spec_task,
                    )
                except Exception:  # noqa: BLE001 — never wedge the pump
                    log.exception(
                        "FRAMEWORK local-explore: terminal outcome recovery failed candidate=%s",
                        cand_id,
                    )
                    recovered = False
                if not recovered:
                    log.warning(
                        "FRAMEWORK local-explore: terminal outcome unavailable candidate=%s state=%s",
                        cand_id,
                        getattr(spec_task, "state", ""),
                    )
                    # Stamp a terminal row so an unrecoverable outcome cannot make
                    # the pump re-select the same finished specialist forever.
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=str(candidate.get("batch_id") or ""),
                        status="recovery_failed",
                        rationale="local-explore outcome unrecoverable from persisted results",
                        provenance="pump",
                    )
                return ""
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        try:
            if spec_tid and cand_id:
                if not isinstance(getattr(state, "framework_agent_specialist_candidate_map", None), dict):
                    state.framework_agent_specialist_candidate_map = {}
                state.framework_agent_specialist_candidate_map[spec_tid] = cand_id
                state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort provenance
            log.debug("FRAMEWORK local-explore: specialist->candidate map write failed", exc_info=True)
        log.info(
            "FRAMEWORK: dispatched local-exploration specialist candidate=%s gap=%s reason=%s",
            cand_id,
            gap_cid,
            reason or "resident",
        )
        return spec_tid

    async def _rank_framework_agent_candidates_llm(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Ask the agent to choose the most promising candidate; ``None`` on any failure.

        Builds a compact workload-context + candidate-list prompt and requests a
        single ``candidate_id``. Matches the reply back to a candidate (exact id
        or PR number). Best-effort: returns ``None`` (caller falls back) when the
        client/model is unavailable, the call errors/times out, or the reply
        can't be parsed.

        Args:
            candidates: The unprocessed candidates to rank.

        Returns:
            The chosen candidate dict, or ``None``.
        """
        import json as _json

        from hyperloom.common.llm_config import astream_chat_completion_text
        from hyperloom.orchestrator.prompts.framework_ranker_prompt import build_framework_ranker_prompt

        client = self._framework_agent_ranker_client()
        if client is None:
            return None
        model = self._framework_agent_ranker_model()
        if not model:
            return None
        state = self.shared_state
        best = resolve_grading_anchor_tput(state)
        cap = 60
        listed = candidates[:cap]
        candidate_rows: list[str] = []
        for i, c in enumerate(listed):
            cid = self._framework_candidate_key(c)
            title = str(c.get("title") or "").strip()
            repo = str(c.get("repo") or c.get("discovered_repo_url") or "").strip()
            audit = c.get("_audit") if isinstance(c.get("_audit"), dict) else None
            appl = str((audit or {}).get("applicability") or "") if audit else ""
            extra = f" [audit_applicability={appl}]" if appl else ""
            candidate_rows.append(f"{i}. id={cid} repo={repo} title={title!r}{extra}")
        has_local_explore = any(str(c.get("kind") or "") == self._LOCAL_EXPLORE_KIND for c in listed)
        # Fold already-tried / failed candidates as negative samples (best-effort).
        try:
            memory_block = self._render_framework_memory_for_prompt(
                self._build_framework_working_memory(),
            )
        except Exception:  # noqa: BLE001 — advisory only
            log.debug("FRAMEWORK: working-memory render for ranker failed", exc_info=True)
            memory_block = ""
        prompt = build_framework_ranker_prompt(
            model=getattr(state, "model", "") or getattr(state, "model_path", ""),
            framework=getattr(state, "framework", ""),
            gpu_type=getattr(state, "gpu_type", ""),
            precision=getattr(state, "precision", ""),
            tp=getattr(state, "tp", ""),
            best_throughput=best,
            candidate_rows=candidate_rows,
            has_local_explore=has_local_explore,
            memory_block=memory_block,
        )

        async def _read_stream() -> str:
            text, _ = await astream_chat_completion_text(
                client,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=400,
            )
            return text

        try:
            text = (
                await asyncio.wait_for(
                    _read_stream(),
                    timeout=float(getattr(self, "framework_ranker_timeout_sec", 60.0) or 60.0),
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001 — degrade to fallback
            log.debug("FRAMEWORK: ranker LLM call failed (%r)", exc)
            return None
        if not text:
            return None
        # Extract the JSON object (tolerate code fences / surrounding prose).
        chosen_id = ""
        reason = ""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = _json.loads(text[start : end + 1])
                chosen_id = str(obj.get("candidate_id") or "").strip()
                reason = str(obj.get("reason") or "").strip()
        except Exception:  # noqa: BLE001
            chosen_id = ""
        if not chosen_id:
            return None
        match = self._match_framework_agent_candidate(chosen_id, candidates)
        if match is None:
            log.debug("FRAMEWORK: ranker chose unknown id=%s; falling back", chosen_id)
            return None
        log.info(
            "FRAMEWORK: agent selected candidate=%s (of %d) reason=%s",
            str(match.get("candidate_id") or match.get("pr_url") or ""),
            len(candidates),
            reason[:160],
        )
        return match

    @staticmethod
    def _match_framework_agent_candidate(
        chosen_id: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Map a model-chosen id back to a candidate (exact id, url, ref, or PR number)."""
        chosen = (chosen_id or "").strip()
        if not chosen:
            return None
        for c in candidates:
            for key in ("candidate_id", "pr_url", "ref"):
                if str(c.get(key) or "").strip() == chosen:
                    return c
        # Fallback: bare PR number match (e.g. model replied "1015" or "PR:1015").
        digits = "".join(ch for ch in chosen if ch.isdigit())
        if digits:
            for c in candidates:
                if str(c.get("pr_number") or "").strip() == digits:
                    return c
        return None

    def _framework_agent_ranker_model(self) -> str:
        """Model slug for the candidate ranker (env override → orchestration model)."""
        import os

        env_model = os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_RANKER_MODEL", "").strip()
        if env_model:
            return env_model
        backend = self.backends.get("orchestration")
        return str(getattr(backend, "model", "") or "").strip()

    def _framework_agent_ranker_client(self) -> Any:
        """Return an OpenAI-compatible async client for ranking, or ``None``.

        Reuses the ProposalScorer's client when present (same gateway/auth),
        then the orchestration backend's own client (so the LLM ranker is on by
        default whenever orchestration has LLM credentials); otherwise asks
        :func:`hyperloom.common.llm_config.get_async_openai_client` -- the only
        sanctioned owner of provider client construction -- for one built from
        ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``. Returns ``None`` when the
        OpenAI side is unconfigured, which leaves the LLM ranker disabled.
        Cached on first successful build.
        """
        import os

        cached = getattr(self, "_fa_ranker_client", None)
        if cached is not None:
            return cached
        scorer = getattr(self, "_proposal_scorer", None)
        ensure = getattr(scorer, "_ensure_client", None)
        if callable(ensure):
            try:
                client = ensure()
                if client is not None:
                    self._coord._fa_ranker_client = client
                    return client
            except Exception:  # noqa: BLE001 — fall through to direct build
                log.debug("FRAMEWORK: scorer client unavailable for ranker", exc_info=True)
        # Reuse the orchestration backend's own OpenAI-compatible client when it
        # exposes one: same gateway + auth as the running session, so the ranker
        # is default-on without extra configuration. Agent-runtime backends own
        # no client of their own, so those fall through to the build below.
        backend = self.backends.get("orchestration")
        backend_client = getattr(backend, "_client", None)
        if backend_client is not None and hasattr(backend_client, "chat"):
            self._coord._fa_ranker_client = backend_client
            return backend_client
        from hyperloom.common import llm_config as _llm_cfg

        # This client speaks the OpenAI protocol, so it authenticates from the
        # OpenAI side only. The orchestration backend's own ``api_key_env`` is not
        # consulted: the orchestration role is Claude, so it names the Anthropic
        # key, which must never reach an OpenAI-protocol endpoint. The explicit
        # gate keeps ``llm_config``'s ``LLM_GATEWAY_KEY`` fallback out of play.
        if not os.environ.get("OPENAI_API_KEY"):
            log.debug("FRAMEWORK: LLM ranker disabled (OPENAI_API_KEY not set; ranker needs the OpenAI side)")
            return None
        try:
            client = _llm_cfg.get_async_openai_client()
        except Exception:  # noqa: BLE001 — the ranker is optional, so it degrades to off
            log.debug("FRAMEWORK: LLM ranker disabled (async OpenAI client unavailable)", exc_info=True)
            return None
        self._coord._fa_ranker_client = client
        return client

    def _framework_known_candidate_ids(self) -> set[str]:
        """All candidate ids already discovered into any prior batch (dedup for new batches).

        Returns:
            A set of candidate ids drawn from every prior batch plus PR ids the
            research scout has already mined.
        """
        state = self.shared_state
        ids: set[str] = set()
        batches = getattr(state, "framework_agent_batches", None) or []
        if not isinstance(batches, list):
            return ids
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            for cand in batch.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                # Canonical key (candidate_id/pr_url/ref); synthetic repo-PR
                # fallback only when the candidate carries no identity field.
                cid = self._framework_candidate_key(cand) or f"{cand.get('repo', '')}-{cand.get('pr_number', '')}"
                if cid:
                    ids.add(cid)
        # Fold in PR ids the research scout already mined so the two mechanisms never re-process a PR.
        for pid in getattr(state, "research_scout_seen_pr_ids", None) or []:
            pid = str(pid or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def _framework_tried_refs(self) -> list[str]:
        """Refs already discovered this phase (fed to compose_gap to bias away from prior PR categories).

        Returns:
            A list of already-known candidate id refs.
        """
        refs: list[str] = []
        for cid in self._framework_known_candidate_ids():
            if cid:
                refs.append(cid)
        return refs

    def _build_framework_working_memory(self) -> dict[str, Any]:
        """Aggregate the FRAMEWORK working memory from the three ledgers (deterministic, zero-LLM).

        Mirrors ``orchestration_memory`` for the framework layer: a structured,
        purely-derived view of "what this session already tried / excluded /
        learned", so discovery and the candidate ranker can be biased away from
        repeating failed candidates. No new data source — it aggregates
        ``framework_agent_phase_progress`` (terminal rows, reliably populated
        after the P0 fix) and the batch dedup set. Critic denials are part of
        that ledger: the gate stamps them as ``critic_denied`` rows carrying
        the rationale.

        Returns:
            A dict with:
              - ``tried_and_why``: recent terminal candidates as
                ``{ref, status, gain_pct, why}`` (most recent last, capped).
              - ``excluded_refs``: sorted union of known-candidate ids and
                candidates that already carry a terminal progress row (hard
                dedup source for discovery — Step B).
              - ``learnings``: deduped Critic rationales for denied candidates.
              - ``pending``: still-unprocessed candidate refs in the latest batch.
        """
        state = self.shared_state
        progress = getattr(state, "framework_agent_phase_progress", None) or []
        rows = [p for p in progress if isinstance(p, dict) and self._framework_candidate_key(p)]
        tried: list[dict[str, Any]] = []
        for row in rows[-self._FRAMEWORK_TRIED_MEMORY_CAP :]:
            status = str(row.get("status") or "").strip()
            gain = row.get("gain_pct")
            why = str(row.get("rationale") or row.get("reason") or "").strip()
            tried.append(
                {
                    "ref": self._framework_candidate_key(row),
                    "status": status,
                    "gain_pct": (float(gain) if isinstance(gain, (int, float)) else None),
                    "why": why[:200],
                }
            )
        # Learnings: distinct Critic denial rationales (negative priors), capped.
        # Read from the progress ledger, the only place a denial is recorded:
        # ``_record_framework_agent_critic_denied`` stamps a ``critic_denied``
        # row carrying the Critic's reasoning.
        learnings: list[str] = []
        seen_learn: set[str] = set()
        for row in rows:
            if str(row.get("status") or "").strip().lower() != FRAMEWORK_CRITIC_DENIED_STATUS:
                continue
            rationale = str(row.get("rationale") or "").strip()
            if rationale and rationale not in seen_learn:
                seen_learn.add(rationale)
                learnings.append(rationale[:200])
            if len(learnings) >= self._FRAMEWORK_TRIED_MEMORY_CAP:
                break
        pending = [self._framework_candidate_key(c) for c in self._unprocessed_framework_agent_candidates()]
        excluded = self._framework_known_candidate_ids() | self._framework_processed_candidate_keys()
        return {
            "tried_and_why": tried,
            "excluded_refs": sorted(r for r in excluded if r),
            "learnings": learnings,
            "pending": [r for r in pending if r],
        }

    @staticmethod
    def _render_framework_memory_for_prompt(memory: dict[str, Any] | None) -> str:
        """Render the FRAMEWORK working memory into prompt text (mirror of ``render_memory_for_seed``).

        Emits a bounded ``tried_and_why`` / ``learnings`` bullet block framed as
        "already tried this session — avoid proposing the same or equivalent".
        Returns ``""`` when there is nothing tried yet (fresh phase), so callers
        can skip the section cleanly.

        Args:
            memory: A record from :meth:`_build_framework_working_memory`.

        Returns:
            The rendered prompt text, or ``""`` when empty.
        """
        if not isinstance(memory, dict) or not memory:
            return ""
        tried = memory.get("tried_and_why") or []
        learnings = memory.get("learnings") or []
        if not tried and not learnings:
            return ""
        lines: list[str] = [
            "Already tried THIS session (avoid proposing the same PR or an "
            "equivalent change — prefer a candidate that attacks a different "
            "bottleneck):",
        ]
        for t in tried:
            if not isinstance(t, dict):
                continue
            ref = str(t.get("ref") or "").strip()
            if not ref:
                continue
            status = str(t.get("status") or "").strip() or "?"
            gain = t.get("gain_pct")
            gain_str = f" gain={float(gain):+.2f}%" if isinstance(gain, (int, float)) else ""
            why = str(t.get("why") or "").strip()
            why_str = f" — {why}" if why else ""
            lines.append(f"  - {ref} [{status}]{gain_str}{why_str}")
        if learnings:
            lines.append("Learnings (avoid these dead ends):")
            lines.extend(f"  - {str(x)}" for x in learnings)
        return "\n".join(lines)

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        """Repo URLs to query for the FRAMEWORK batch: framework's own repo + global PR_QUERY_REPOS allowlist, dedup preserving order.

        Args:
            framework: The framework name whose own repo seeds the query.

        Returns:
            An order-preserving, deduped list of repo URLs to query.
        """
        from hyperloom.inference_optimizer import framework_registry

        from ..framework import client as _fa_client
        from ..specialists.domains import PR_QUERY_REPOS

        urls: list[str] = []

        def _add(u: str) -> None:
            """Append a trimmed URL to ``urls`` if non-empty and not already present.

            Args:
                u (str): A candidate repo URL.
            """
            u = (u or "").strip()
            if u and u not in urls:
                urls.append(u)

        # Primary: the framework's own repo.
        primary_repo_url = _fa_client.repo_url_for_framework(framework)
        _add(primary_repo_url)

        # Serving/infra PRs cannot be git-applied to scriptable model repos, so
        # a scriptable session queries its own repo and nothing else. Scoping on
        # scriptability alone is deliberate: keying it on also having a repo URL
        # excluded the case that needs it most, since an operator-supplied
        # workload has no upstream repo by construction and so inherited the
        # whole serving allowlist. It queries nothing now rather than everything.
        if not framework_registry.is_scriptable(framework):
            for repo in PR_QUERY_REPOS:
                repo = str(repo or "").strip()
                if repo and "/" in repo:
                    _add(f"https://github.com/{repo}.git")
            if not urls:
                # Last-ditch: let phase_discover resolve from framework itself.
                _add(_fa_client.repo_url_for_framework(framework or "sglang"))
        return urls

    @staticmethod
    def _framework_agent_repo_url_origin_framework(repo_url: str) -> str:
        """Reverse-lookup: which known serving framework (if any) owns ``repo_url``.

        Design #5-P1/discovery-wiring: ``_framework_agent_discover_repo_urls``
        already queries the ``pr_intel_specialist`` cross-repo set (e.g. a
        sglang session's discovery batch already includes ``ROCm/vllm``), but
        candidates returned by ``fa phase-discover`` never carry a
        ``framework`` tag reflecting which repo they actually came from — so
        ``_audit_framework_agent_candidate``'s cross-framework detection had
        no signal to act on even though cross-repo candidates were already
        flowing through the pump. Only the four ``repo_map`` frameworks are
        resolvable (kernel-level repos like aiter/triton/rccl aren't a
        "framework" in the audit sense and correctly resolve to "").

        Args:
            repo_url: A repo URL as returned by
                :meth:`_framework_agent_discover_repo_urls`.

        Returns:
            The lowercase framework name, or ``""`` when ``repo_url`` doesn't
            match any known framework's canonical repo.
        """
        from ..framework import client as _fa_client

        normalized = (repo_url or "").strip().rstrip("/").lower()
        if not normalized:
            return ""
        for fw in ("sglang", "vllm", "atom", "xdit"):
            fw_url = (_fa_client.repo_url_for_framework(fw) or "").strip().rstrip("/").lower()
            if fw_url and fw_url == normalized:
                return fw
        return ""

    def _record_framework_agent_phase_done(
        self,
        *,
        reason: str,
        failure_count: int,
    ) -> None:
        """Append a framework_agent_phase_done row to phase_history describing why the pump gave up.

        Args:
            reason: Human-readable reason the phase ended.
            failure_count: Number of consecutive discover failures recorded.
        """
        state = self.shared_state
        try:
            from ..framework import client as _fa_client
            from ..framework.artifacts import summarize_candidate_outcomes

            # Classify this phase's candidate outcomes so the report /
            # robustness can tell "discovered nothing" (empty_discovery) apart
            # from "tested candidates but none kept" (tested_no_keep).
            summary = summarize_candidate_outcomes(
                getattr(state, "framework_agent_phase_progress", None),
            )
            outcome_class = str(summary.get("outcome_class") or "empty_discovery")

            # Consecutive empty-discovery tracking → advisory ("framework phase
            # ineffective"). Reset the streak the moment a phase tested anything.
            prev_empty = int(getattr(state, "framework_consecutive_empty_discoveries", 0) or 0)
            if outcome_class == "empty_discovery":
                consecutive_empty = prev_empty + 1
            else:
                consecutive_empty = 0
            state.framework_consecutive_empty_discoveries = consecutive_empty

            advisory = ""
            if outcome_class == "empty_discovery" and consecutive_empty >= 2:
                advisory = (
                    f"framework phase ineffective: {consecutive_empty} consecutive "
                    "macro-cycles discovered zero candidates"
                )
            elif outcome_class == "tested_no_keep":
                advisory = (
                    "framework phase tested candidates but none cleared the gate "
                    f"(tested={summary.get('tested')}, keeps=0)"
                )
            if advisory:
                log.warning("FRAMEWORK advisory: %s", advisory)

            state.append_phase_history_event(
                reason=reason,
                evidence={
                    "event": "framework_agent_phase_done",
                    "failure_count": int(failure_count),
                    "retry_limit": int(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT),
                    "batches_discovered": len(getattr(state, "framework_agent_batches", None) or []),
                    "outcome_class": outcome_class,
                    "candidate_outcomes": summary.get("by_status") or {},
                    "keeps": int(summary.get("keeps") or 0),
                    "tested": int(summary.get("tested") or 0),
                    "consecutive_empty_discoveries": consecutive_empty,
                    "advisory": advisory,
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            pass

    async def _discover_next_framework_batch(self) -> bool:
        """Call ``fa phase-discover`` and append a batch to SharedState. Returns True iff a non-empty batch was appended; transient failures return False (see DISCOVER_FAILURE_RETRY_LIMIT).

        Returns:
            ``True`` if a non-empty, deduped batch of new candidates was
            appended to SharedState; ``False`` on transient failure or when no
            new candidates were found.
        """
        from ..framework import client as _fa_client

        state = self.shared_state
        # Directed gap composition: seed search from latest bottleneck + workload taxonomy via compose_gap,
        # then merge structured state.gaps so each batch retargets the current bottleneck.
        framework = ""
        try:
            framework = str(getattr(state, "framework", "") or "").strip().lower()
        except Exception:  # noqa: BLE001
            framework = ""
        directed_keywords: list[str] = []
        directed_gap = ""
        try:
            from ..actions.executors._framework_gap_composer import compose_gap

            directed_gap, directed_keywords = compose_gap(
                framework=framework,
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                model_class=str(getattr(state, "model_class", "") or ""),
                precision=str(getattr(state, "precision", "") or ""),
                profile_kernel_breakdown_path=getattr(
                    state,
                    "last_profile_kernel_breakdown",
                    None,
                ),
                rewrite_evidence_path=getattr(
                    state,
                    "last_framework_rewrite_evidence",
                    None,
                ),
            )
        except Exception:  # noqa: BLE001 — defensive
            directed_gap, directed_keywords = "", []
        gaps: list[dict[str, str]] = []
        try:
            gap_list = getattr(state, "gaps", None) or []
            for g in gap_list:
                if not isinstance(g, dict):
                    continue
                gaps.append(
                    {
                        "gap_canonical_id": str(g.get("canonical_id") or ""),
                        "gap_description": str(g.get("symptom") or g.get("description") or ""),
                    }
                )
        except Exception:  # noqa: BLE001 — defensive
            gaps = []
        if directed_gap:
            # Prepend the directed gap so fa's search leads with bottleneck-aware phrasing; de-dup.
            existing = {str(g.get("gap_description") or "") for g in gaps}
            if directed_gap not in existing:
                gaps.insert(
                    0,
                    {
                        "gap_canonical_id": "directed",
                        "gap_description": directed_gap,
                    },
                )
        if not gaps:
            gaps = [{"gap_canonical_id": "", "gap_description": ""}]
        timeout_sec = float(
            getattr(self, "framework_agent_discover_timeout_sec", 0.0) or _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC
        )
        max_candidates = DEFAULT_FRAMEWORK_MAX_CANDIDATES
        # Cross-repo: query every pr_intel_specialist repo so discovery isn't confined to one framework repo.
        repo_urls = self._framework_agent_discover_repo_urls(framework)
        # Step A/B — feed the session working memory into discovery so fa
        # hard-filters already-seen/terminal candidates and de-prioritises
        # equivalents. Best-effort: a build failure must never wedge discovery.
        try:
            fw_memory = self._build_framework_working_memory()
        except Exception:  # noqa: BLE001 — advisory only
            log.debug("FRAMEWORK: working-memory build for discovery failed", exc_info=True)
            fw_memory = {}
        excluded_candidate_ids = list(fw_memory.get("excluded_refs") or [])
        failed_candidate_context = list(fw_memory.get("tried_and_why") or [])[-10:]
        payload: dict[str, Any] | None = None
        merged_candidates: list[dict[str, Any]] = []
        batch_id = ""
        any_call_ok = False
        last_exc: Exception | None = None
        # ``timeout_sec`` bounds one repo, not the whole fan-out: each call is an
        # independent primus_cortex round-trip, so dividing it starves every repo
        # once the repo list grows (9 repos drove it to the 30s floor while a
        # single discover needs ~20s plus PR-Monitor latency). The whole-phase
        # bound is the caller's budget plus DISCOVER_FAILURE_RETRY_LIMIT.
        per_repo_timeout = max(timeout_sec, _FRAMEWORK_MIN_PER_REPO_TIMEOUT_SEC)
        for repo_url in repo_urls:
            try:
                repo_payload = await _fa_client.phase_discover(
                    model=str(getattr(state, "model", "") or ""),
                    framework=framework or "sglang",
                    gpu_type=str(getattr(state, "gpu_type", "") or ""),
                    gaps=gaps,
                    session_dir=self.session_dir,
                    repo_url=repo_url,
                    keywords=directed_keywords,
                    max_candidates=max_candidates,
                    pr_states=["open"],
                    excluded_candidate_ids=excluded_candidate_ids,
                    failed_candidate_context=failed_candidate_context,
                    timeout_sec=per_repo_timeout,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                last_exc = exc
                log.warning(
                    "fa phase-discover failed for repo_url=%r: %r",
                    repo_url,
                    exc,
                )
                continue
            any_call_ok = True
            if payload is None:
                payload = repo_payload
            if not batch_id:
                batch_id = str((repo_payload or {}).get("batch_id") or "")
            repo_cands = (repo_payload or {}).get("candidates") or []
            if isinstance(repo_cands, list):
                # Cross-framework discovery lane (default ON; kill switch:
                # FRAMEWORK_AGENT_CROSS_DISCOVER_TAG=0). This repo_url loop already
                # discovers from OTHER serving frameworks' repos (pr_intel_specialist
                # cross-repo set — e.g. a sglang session already queries ROCm/vllm),
                # but fa phase-discover never tags candidates with which repo they came
                # from. Untagged, a cross-repo candidate was audited/applied as if it
                # were same-framework — unsafe, since a vllm diff can never be git-
                # applied onto sglang. Stamp candidates from a DIFFERENT framework's
                # own repo with that origin framework so _audit_framework_agent_candidate
                # routes them through the cross-framework PORT (specialist rewrite, never
                # raw apply). Same-framework candidates (incl. kernel-level pr_intel
                # repos with no framework mapping) are untouched, so same-framework audit
                # behaviour is unchanged. Set the env to 0/false/no/off to fully revert.
                cross_on = os.environ.get("FRAMEWORK_AGENT_CROSS_DISCOVER_TAG", "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                origin_fw = self._framework_agent_repo_url_origin_framework(repo_url) if cross_on else ""
                for c in repo_cands:
                    if not isinstance(c, dict):
                        continue
                    if cross_on:
                        # Derive the origin from the candidate's OWN repo first:
                        # fa phase-discover defaults `framework` to the session
                        # framework even for cross-repo pr_intel candidates, so
                        # only filling blanks (the prior behaviour) left a sglang
                        # PR surfaced under a vllm session mis-tagged as vllm and
                        # audited as same-framework — never routed to the cross-
                        # framework port. Override when the candidate's repo maps
                        # to a known framework different from the session's. Fall
                        # back to the queried repo_url's origin when the candidate
                        # carries no repo. Kernel repos (aiter/triton/rccl) resolve
                        # to "" and are left untouched, so same-framework audit
                        # behaviour is unchanged.
                        cand_repo = str(c.get("repo") or "").strip()
                        cand_origin = (
                            self._framework_agent_repo_url_origin_framework(cand_repo) if cand_repo else origin_fw
                        )
                        if cand_origin and cand_origin != framework:
                            c["framework"] = cand_origin
                        elif origin_fw and origin_fw != framework and not c.get("framework"):
                            c["framework"] = origin_fw
                    merged_candidates.append(c)
        if not any_call_ok:
            failures = int(getattr(state, "framework_agent_discover_failures", 0) or 0) + 1
            state.framework_agent_discover_failures = failures
            log.warning(
                "fa phase-discover failed across all %d repo(s) (attempt %d/%d): %r",
                len(repo_urls),
                failures,
                _fa_client.DISCOVER_FAILURE_RETRY_LIMIT,
                last_exc,
            )
            try:
                state.append_phase_history_event(
                    reason="framework_agent_discover_failed",
                    evidence={
                        "event": "framework_agent_discover_failed",
                        "attempt": failures,
                        "limit": _fa_client.DISCOVER_FAILURE_RETRY_LIMIT,
                        "error": repr(last_exc),
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                pass
            state.save(self.session_dir)
            return False
        # Successful call — reset failure counter regardless of whether
        # the payload contained candidates.
        if int(getattr(state, "framework_agent_discover_failures", 0) or 0) != 0:
            state.framework_agent_discover_failures = 0
        if not merged_candidates:
            return False
        batch_id = str((payload or {}).get("batch_id") or "")
        # Cross-batch + cross-repo de-dup so the new batch only carries genuinely
        # new PRs. Coordinator-side hard-dedup backstop (Step B): even if fa
        # forgot to honour ``excluded_candidate_ids``, re-filter here against the
        # full excluded set (known candidate ids ∪ candidates that already carry
        # a terminal progress row) so a failed/tested PR is never re-queued.
        seen_ids = self._framework_known_candidate_ids() | self._framework_processed_candidate_keys()
        primary_repo_url = repo_urls[0] if repo_urls else ""
        # Normalise each candidate for consistent executor fields + a stable progress-ledger id.
        norm: list[dict[str, Any]] = []
        for c in merged_candidates:
            if not isinstance(c, dict):
                continue
            cand_id = str(c.get("pr_url") or c.get("ref") or f"{c.get('repo', '')}-{c.get('pr_number', '')}")
            if cand_id and cand_id in seen_ids:
                continue
            seen_ids.add(cand_id)
            # Stamp the candidate's repo URL so the executor knows same-repo (fetchable) vs foreign (diff_url).
            discovered_repo_url = str(c.get("repo_url") or c.get("discovered_repo_url") or primary_repo_url)
            norm.append(
                {
                    **c,
                    "candidate_id": cand_id,
                    "batch_id": batch_id,
                    "discovered_repo_url": discovered_repo_url,
                }
            )
        if not norm:
            return False
        batch_entry = {
            "batch_id": batch_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": int(getattr(state, "macro_cycle", 0) or 0),
            "candidate_count": len(norm),
            "candidates": norm,
            "max_gain_pct_observed_in_batch": 0.0,
        }
        if not isinstance(state.framework_agent_batches, list):
            state.framework_agent_batches = []
        state.framework_agent_batches.append(batch_entry)
        state.save(self.session_dir)
        log.info(
            "FRAMEWORK: discovered batch=%s with %d candidates",
            batch_id or "<unset>",
            len(norm),
        )
        return True

    async def _enqueue_framework_agent_task(self, candidate: dict[str, Any]) -> None:
        """Enqueue a single ``framework_agent`` task for ``candidate``.

        Builds the task params (candidate, batch id, baseline throughput, KEEP
        threshold, framework) and creates an idempotent ``framework_agent`` task
        whose lanes and lease TTL come from the action catalogue. On enqueue failure,
        records an ``enqueue_failed`` progress row so the pump skips the
        candidate next tick instead of spinning.

        Args:
            candidate (dict[str, Any]): The discovered PR candidate to apply
                and benchmark.
        """
        state = self.shared_state
        params = {
            "candidate": candidate,
            "batch_id": candidate.get("batch_id") or "",
            "base_tput": resolve_grading_anchor_tput(state),
            # Same decaying bar the explore and integrate_patch dispatch paths inject.
            "keep_threshold_pct": _phase_state.resolve_keep_threshold(state),
            "framework": str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower(),
            # Source patches require the accuracy gate for KEEP.
            "require_accuracy_for_keep": True,
            "accuracy_baseline": float(getattr(state, "baseline_accuracy", 0.0) or 0.0),
            # The lane templates from the shipped default config, which
            # materializes RUN_EVAL=true and would override the session's choice.
            "disable_run_eval": bool(getattr(state, "eval_disabled", False)),
        }
        cand_id = self._framework_candidate_key(candidate)
        idem = f"framework:{candidate.get('batch_id', '')}:{cand_id}"
        lanes, ttl = self._registry_lanes_ttl("framework_agent")
        try:
            # A framework candidate rebuilds and benchmarks, so it cannot share
            # the GPU. Enqueueing without lanes would run it unserialised
            # against every other task; the handler below turns this into a
            # warning plus a progress row.
            if not lanes:
                raise RuntimeError("framework_agent resolved to no lanes; the task would run without GPU exclusivity.")
            await self.tasks.create_or_return_existing(
                kind="framework_agent",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            log.info(
                "FRAMEWORK: enqueued candidate=%s batch=%s",
                cand_id,
                candidate.get("batch_id") or "",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "FRAMEWORK: failed to enqueue candidate=%s: %r",
                cand_id,
                exc,
            )
            # Record enqueue_failed progress row so the candidate is skipped next tick (else the loop spins).
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=str(candidate.get("batch_id") or ""),
                status="enqueue_failed",
                kept=False,
                rationale=repr(exc),
                provenance="pump",
                extra={"error": repr(exc)},
            )

    def _collect_framework_agent_candidate_priors(self) -> dict[str, Any]:
        """Return compact session-local priors for the Critic gate.

        Everything the Critic needs about earlier candidates lives in the
        progress ledger, denials included, so the outcomes carry the rationale
        rather than being paired with a separate decision list. Only the rows
        that were stamped with a reason have one — bench results record their
        numbers instead — so the key is omitted rather than sent empty.

        Returns:
            A dict with ``recent_outcomes`` (recent terminal apply/bench
            results, each with the reason recorded for it, where there is one),
            bounded to a short tail.
        """
        raw_progress = getattr(self.shared_state, "framework_agent_phase_progress", None) or []
        terminal = {
            "kept",
            "kept_inert",
            "reverted",
            "no_patch",
            "enqueue_failed",
            FRAMEWORK_CRITIC_DENIED_STATUS,
        }
        tail = [r for r in raw_progress if isinstance(r, dict) and str(r.get("status") or "") in terminal]
        outcomes: list[dict[str, Any]] = []
        for row in tail[-self._CRITIC_PRIORS_OUTCOME_TAIL :]:
            entry: dict[str, Any] = {
                "candidate_id": str(row.get("candidate_id") or ""),
                "status": str(row.get("status") or ""),
                "gain_pct": row.get("gain_pct"),
            }
            rationale = str(row.get("rationale") or "")[:200]
            if rationale:
                entry["rationale"] = rationale
            outcomes.append(entry)
        return {
            "recent_outcomes": outcomes,
        }

    async def _submit_framework_agent_candidate_for_review(
        self,
        candidate: dict[str, Any],
        *,
        audit: dict[str, Any] | None = None,
        audit_step: str = "",
    ) -> None:
        """Submit a FRAMEWORK candidate as a normal ``proposal`` for async Critic review.

        Mirrors :meth:`_maybe_autosubmit_specialist_patches`: writes a
        ``topic="proposal"`` message + registers a :class:`PendingProposal`,
        bypassing ``_handle_intent`` / ``PolicyGate`` (``framework_agent`` is a
        COORDINATOR_INTERNAL action that ``propose_action`` would deny). The
        Critic verdict arrives on a later tick and drives the apply/author
        enqueue (approve/advise) or the critic_denied row (reject) via
        ``_handle_single_verdict``. All context needed by ``replay_for_resume``
        and ``_materialize_framework_agent_candidate`` lives in the payload, so no
        separate persistent candidate map is kept.

        Args:
            candidate: The discovered PR candidate to gate.
            audit: The semantic-audit verdict (carried for the authoring seed).
            audit_step: The resolved route (``direct_framework`` /
                ``author_via_specialist`` / ``""`` for legacy both-tracks).
        """
        cand_id = self._framework_candidate_key(candidate)
        batch_id = str(candidate.get("batch_id") or "")
        # Dedup: a candidate is already awaiting its pre-screen verdict.
        for p in self.state.pending_proposals.values():
            try:
                if getattr(p, "action_name", "") != "framework_agent":
                    continue
                if getattr(p, "decided", False):
                    continue
                pl = getattr(p, "payload", {}) or {}
                if str(pl.get("framework_agent_candidate_id") or "") == cand_id and cand_id:
                    return
            except Exception:  # noqa: BLE001 — defensive
                continue
        # Repeated-review backstop: count how many times this candidate has
        # been sent for review. Under healthy operation a candidate is submitted
        # once (a terminal row makes it "processed"); repeated submissions mean a
        # terminal-row leak let it be re-selected. Past the abort threshold,
        # force a terminal row and stop — no single candidate can burn the phase.
        if cand_id:
            counts = getattr(self.shared_state, "framework_agent_review_counts", None)
            if not isinstance(counts, dict):
                counts = {}
                self.shared_state.framework_agent_review_counts = counts
            count = int(counts.get(cand_id, 0) or 0) + 1
            counts[cand_id] = count
            if count > self._MAX_REPEATED_REVIEW_SUBMISSIONS:
                log.warning(
                    "FRAMEWORK: candidate=%s submitted for review %d times "
                    "(> cap %d); aborting to protect the phase budget",
                    cand_id,
                    count,
                    self._MAX_REPEATED_REVIEW_SUBMISSIONS,
                )
                self._stamp_framework_progress(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="repeated_review_abort",
                    kept=False,
                    rationale=(f"submitted for review {count} times (> cap {self._MAX_REPEATED_REVIEW_SUBMISSIONS})"),
                    provenance="pump",
                    extra={"review_submissions": count},
                )
                return
        propose_payload: dict[str, Any] = {
            "action_name": "framework_agent",
            "provenance": "framework_agent",
            "predicted_gain_pct": 0.0,
            "candidate": dict(candidate),
            "batch_id": batch_id,
            "framework_agent_candidate_id": cand_id,
            "audit": dict(audit) if isinstance(audit, dict) else {},
            "audit_step": str(audit_step or ""),
            "priors": self._collect_framework_agent_candidate_priors(),
        }
        msg = Message.new(
            "coordinator",
            "*",
            "proposal",
            {**propose_payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        self.state.pending_proposals[msg.msg_id] = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent="coordinator",
            action_name="framework_agent",
            predicted_gain_pct=0.0,
            payload=dict(propose_payload),
        )
        log.info(
            "FRAMEWORK: candidate submitted for Critic review msg_id=%s candidate=%s batch=%s audit_step=%s",
            msg.msg_id,
            cand_id,
            batch_id,
            audit_step or "<unknown>",
        )
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "framework_agent_candidate_submitted_for_review",
                "proposal_msg_id": msg.msg_id,
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "audit_step": str(audit_step or ""),
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "save after framework_agent candidate submit failed for candidate=%s",
                cand_id,
            )

    async def _materialize_framework_agent_candidate(
        self,
        pending: "PendingProposal",
    ) -> None:
        """Route a Critic-approved FRAMEWORK candidate to the apply / author tracks.

        Reads the audit route stamped on the proposal payload and reuses the
        existing ``direct_framework`` (raw-diff executor) vs
        ``author_via_specialist`` enqueue helpers, with the same
        authoring-disabled → raw-diff fallback the pump used to apply inline.

        Args:
            pending: The approved framework_agent pending proposal.
        """
        payload = pending.payload or {}
        candidate = dict(payload.get("candidate") or {})
        audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
        audit_step = str(payload.get("audit_step") or "")
        cand_id = str(payload.get("framework_agent_candidate_id") or self._framework_candidate_key(candidate))
        batch_id = str(payload.get("batch_id") or candidate.get("batch_id") or "")
        authoring_enabled = bool(getattr(self.shared_state, "framework_agent_authoring_enabled", False))
        want_raw = audit_step == "direct_framework"
        want_author = audit_step == "author_via_specialist"
        if audit_step not in ("direct_framework", "author_via_specialist"):
            want_raw = True
            want_author = True
        if want_author and not authoring_enabled:
            want_raw = True
            want_author = False
        log.info(
            "FRAMEWORK: critic-approved candidate=%s batch=%s audit_step=%s raw=%s author=%s",
            cand_id,
            batch_id,
            audit_step or "<unknown>",
            want_raw,
            want_author,
        )
        if want_raw:
            # _enqueue_framework_agent_task owns its own enqueue_failed terminal
            # row on failure, so a raw-track candidate always ends up processed.
            await self._enqueue_framework_agent_task(candidate)
        if want_author and authoring_enabled:
            try:
                await self._enqueue_framework_agent_authoring_specialist(
                    candidate,
                    audit=audit if isinstance(audit, dict) else {},
                )
            except Exception as exc:  # noqa: BLE001 — never wedge the phase
                log.warning(
                    "FRAMEWORK: authoring specialist dispatch failed: %r",
                    exc,
                )
                # Author-only route (no raw track to own a terminal row): stamp
                # materialize_failed so an approved-but-undispatchable candidate
                # is not re-selected every tick.
                if not want_raw:
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=batch_id,
                        status="materialize_failed",
                        kept=False,
                        rationale=repr(exc),
                        provenance="pump",
                        extra={"error": repr(exc)},
                    )

    def _stamp_framework_progress(
        self,
        *,
        candidate_id: str,
        batch_id: str = "",
        status: str,
        kept: bool = False,
        rationale: str = "",
        provenance: str = "",
        gain_pct: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Idempotently stamp a terminal ``framework_agent_phase_progress`` row.

        The single terminal-row writer for every FRAMEWORK path that ends a
        candidate's life without a benched executor result (critic denial,
        needs_review dead-ends, materialize/enqueue failures, silent apply/bench
        failures, repeated-review aborts). Guarantees the P0 invariant: any
        candidate that can no longer advance carries exactly one terminal row,
        so the pump's plateau / phase-done early-exit accrues instead of relying
        on the budget-cap backstop.

        Idempotent: a candidate key that already has ANY progress row is left
        untouched (returns ``False``) so a later path can never double-stamp or
        overwrite an earlier verdict. Writes the row + a ``decision.json`` and
        persists SharedState. Best-effort on the artifact/save side (never
        raises into the pump).

        Args:
            candidate_id: The canonical candidate key (see
                :meth:`_framework_candidate_key`).
            batch_id: The discovery batch the candidate belonged to.
            status: The terminal status (e.g. ``critic_denied`` /
                ``no_result_failed`` / ``reauthor_cap`` …).
            kept: Whether the candidate was promoted (terminal rows are almost
                always ``False``).
            rationale: Human-readable reason recorded on the row + decision.json.
            provenance: Origin tag for the decision.json (``critic`` / ``pump``
                / ``executor`` …).
            gain_pct: Measured delta, when one exists.
            extra: Optional additional fields merged into the decision.json.

        Returns:
            ``True`` when a new row was appended; ``False`` when the candidate
            already had a row (idempotent no-op) or the key was empty.
        """
        cand_id = str(candidate_id or "")
        if not cand_id:
            return False
        state = self.shared_state
        progress = getattr(state, "framework_agent_phase_progress", None)
        if not isinstance(progress, list):
            progress = []
            state.framework_agent_phase_progress = progress
        if cand_id in {self._framework_candidate_key(p) for p in progress if isinstance(p, dict)}:
            return False
        row: dict[str, Any] = {
            "candidate_id": cand_id,
            "batch_id": str(batch_id or ""),
            "status": str(status or ""),
            "kept": bool(kept),
            "rationale": str(rationale or ""),
            "gain_pct": (float(gain_pct) if isinstance(gain_pct, (int, float)) else 0.0),
            "provenance": str(provenance or ""),
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": int(getattr(state, "macro_cycle", 0) or 0),
        }
        # Merge caller-supplied extras (e.g. ``error`` / ``review_submissions``)
        # onto the row too, without clobbering the canonical fields above, so
        # downstream consumers see the same detail the decision.json carries.
        if isinstance(extra, dict):
            for k, v in extra.items():
                row.setdefault(str(k), v)
        progress.append(row)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK: save after progress stamp failed candidate=%s status=%s",
                cand_id,
                status,
            )
        log.info(
            "FRAMEWORK: stamped terminal progress candidate=%s batch=%s status=%s",
            cand_id,
            str(batch_id or ""),
            status,
        )
        return True

    async def _record_framework_agent_critic_denied(
        self,
        pending: "PendingProposal",
        reasoning: str,
    ) -> None:
        """Record a ``critic_denied`` FRAMEWORK row when the async gate rejects a candidate.

        Stamps a ``framework_agent_phase_progress`` row + ``decision.json`` keyed
        on the candidate id so ``_select_best_framework_agent_candidate`` treats it
        as processed and the pump advances to the next candidate.

        Args:
            pending: The rejected framework_agent pending proposal.
            reasoning: The Critic's free-text rationale.
        """
        payload = pending.payload or {}
        cand_id = str(
            payload.get("framework_agent_candidate_id")
            or self._framework_candidate_key(
                payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
            )
        )
        batch_id = str(payload.get("batch_id") or "")
        self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=FRAMEWORK_CRITIC_DENIED_STATUS,
            kept=False,
            rationale=str(reasoning or ""),
            provenance="critic",
        )
        log.info(
            "FRAMEWORK: critic rejected candidate=%s batch=%s rationale=%r",
            cand_id,
            batch_id,
            str(reasoning or "")[:200],
        )

    async def _maybe_reauthor_from_critic_feedback(
        self,
        pending: "PendingProposal",
        advisory: dict[str, Any] | None,
    ) -> None:
        """Re-author a framework_agent deliverable once, seeding the next authoring round with the Critic's ``required_evidence``.

        Fires only for a ``needs_review`` verdict carrying non-empty
        ``required_evidence`` on a framework_agent candidate or authoring proposal,
        capped at :attr:`_MAX_REAUTHOR_ATTEMPTS` per candidate.

        Args:
            pending: The proposal the verdict targets.
            advisory: The serialized Critic advisory.
        """
        advisory = advisory or {}
        # Resolve the candidate identity FIRST so the two dead-end returns below
        # (no required_evidence / reauthor cap) can stamp a terminal progress row
        # — a needs_review verdict that neither re-authors nor materializes would
        # otherwise leave the candidate row-less and re-selected forever.
        action_name = str(getattr(pending, "action_name", "") or "")
        payload = getattr(pending, "payload", {}) or {}
        candidate: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        old_sid = ""
        if action_name == "framework_agent":
            candidate = dict(payload.get("candidate") or {})
            raw_audit = payload.get("audit")
            audit = raw_audit if isinstance(raw_audit, dict) else {}
        elif action_name == "integrate_patch":
            # Rebuild the candidate from the originating authoring specialist.
            params = payload.get("params") or {}
            if not bool(params.get("framework_agent_authoring")):
                return
            sid = str(params.get("specialist_task_id") or "").strip()
            old_sid = sid
            spec_params: dict[str, Any] = {}
            if sid:
                try:
                    spec_task = await self.tasks.get(sid)
                    spec_params = dict(getattr(spec_task, "params", None) or {})
                except Exception:  # noqa: BLE001 — best-effort lookup
                    spec_params = {}
            candidate = {
                "candidate_id": str(
                    params.get("framework_agent_candidate_id") or spec_params.get("framework_agent_candidate_id") or ""
                ),
                "batch_id": str(params.get("framework_batch_id") or spec_params.get("framework_batch_id") or ""),
                "title": str(spec_params.get("gap_symptom") or ""),
                "framework": str(spec_params.get("framework") or ""),
                "gap_canonical_id": str(spec_params.get("gap_canonical_id") or ""),
            }
            raw_audit = spec_params.get("framework_audit")
            audit = raw_audit if isinstance(raw_audit, dict) else {}
        else:
            return
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            return
        batch_id = str(candidate.get("batch_id") or payload.get("batch_id") or "")
        required_evidence = [str(x).strip() for x in (advisory.get("required_evidence") or []) if str(x).strip()]
        if not required_evidence:
            # needs_review with nothing to act on: no re-author is possible, so
            # this is terminal for the candidate. Stamp it so the pump advances.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="needs_review_no_evidence",
                kept=False,
                rationale=str(advisory.get("advice_text") or "")[:500],
                provenance="critic",
            )
            return
        # Skip if the candidate is already materializing as a live integrate_patch task.
        try:
            live_tasks = [*await self.tasks.queued(), *await self.tasks.running()]
        except Exception:  # noqa: BLE001 — defensive
            live_tasks = []
        for t in live_tasks:
            if getattr(t, "kind", "") != "integrate_patch":
                continue
            tp = getattr(t, "params", None) or {}
            if str(tp.get("framework_agent_candidate_id") or "") == cand_id:
                return
        attempts = getattr(self.shared_state, "specialist_reauthor_attempts", None)
        if not isinstance(attempts, dict):
            attempts = {}
            self.shared_state.specialist_reauthor_attempts = attempts
        prior = int(attempts.get(cand_id, 0) or 0)
        if prior >= self._MAX_REAUTHOR_ATTEMPTS:
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "reauthor_cap_reached",
                    "candidate_id": cand_id,
                    "attempts": prior,
                    "proposal_msg_id": str(getattr(pending, "proposal_msg_id", "") or ""),
                    "verdict": "needs_review",
                },
            )
            # Re-author budget exhausted: terminal for the candidate. Stamp so
            # the pump stops re-selecting it once this proposal drains.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="reauthor_cap",
                kept=False,
                rationale=f"reauthor attempts >= cap ({self._MAX_REAUTHOR_ATTEMPTS})",
                provenance="pump",
            )
            return
        attempt = prior + 1
        attempts[cand_id] = attempt
        critic_feedback = {
            "required_evidence": required_evidence,
            "advice_text": str(advisory.get("advice_text") or ""),
            "risks": [str(r).strip() for r in (advisory.get("risks") or []) if str(r).strip()],
        }
        new_task_id = ""
        try:
            new_task_id = await self._enqueue_framework_agent_authoring_specialist(
                candidate,
                audit=audit,
                reauthor_attempt=attempt,
                critic_feedback=critic_feedback,
            )
        except Exception:  # noqa: BLE001 — never wedge the verdict handler
            log.exception(
                "re-author dispatch failed candidate=%s attempt=%s",
                cand_id,
                attempt,
            )
            return
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "save after re-author dispatch failed candidate=%s",
                cand_id,
            )
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_reauthor_dispatched",
                "candidate_id": cand_id,
                "attempt": attempt,
                "proposal_msg_id": str(getattr(pending, "proposal_msg_id", "") or ""),
                "old_specialist_task_id": old_sid,
                "new_specialist_task_id": new_task_id,
                "verdict": "needs_review",
                "required_evidence": required_evidence[:6],
            },
        )

    async def _pump_framework_agent_phase_safely(self, *, caller: str) -> None:
        """Best-effort FRAMEWORK pump wrapper shared by tick and run.

        Args:
            caller: Label identifying the caller ("tick" / "run"), used only in
                the failure log.
        """
        try:
            await self._pump_framework_agent_phase()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("FRAMEWORK pump (%s) failed", caller)

    def _record_framework_agent_authored_outcome(
        self,
        *,
        task: "Task",
        result: Any,
    ) -> None:
        """Bridge an authored-patch ``integrate_patch`` outcome into the FRAMEWORK progress ledger (else the gain is invisible). Attributed to the latest batch; every terminal status is recorded (empty/in-progress statuses and lane-owned ``apply_failed`` retries are skipped).

        Args:
            task: The integrate_patch task carrying the FRAMEWORK authoring
                provenance markers.
            result: The task result; any non-empty terminal status is recorded
                except ``apply_failed`` on a perf lane, which the retry loop owns.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_agent_authoring")):
            return
        res = result if isinstance(result, dict) else getattr(result, "result", None)
        if not isinstance(res, dict):
            return
        status = str(res.get("status") or "")
        # Record EVERY terminal integrate_patch outcome — not just keep/revert.
        # A patch that fails to apply / bench (``apply_failed`` /
        # ``bench_reverted`` / ``error`` …) is still a terminal verdict for the
        # candidate; without a progress row the FRAMEWORK pump re-selects the
        # same candidate every tick and livelocks (the authoring specialist's
        # ``patches_written`` is non-empty so the empty-outcome bridge does not
        # fire either). Only an empty / in-progress status is skipped.
        if not status:
            return
        # apply_failed with a lane field means the unified retry loop will handle
        # this result (either re-dispatch or stamp a terminal row at the cap).
        # Do NOT stamp a progress row here — that would block the retry pump.
        if status == "apply_failed" and res.get("lane") in ("perf_framework", "perf_explore"):
            return
        # Resolve the FRAMEWORK candidate id (a PR URL) that this authored
        # patch belongs to. The integrate_patch task carries only
        # ``specialist_task_id``; map that back to the originating candidate via
        # the dispatch-time map so the progress row is keyed on the same PR URL
        # that ``_select_next_framework_agent_candidate`` checks. Falling back to a
        # task_id here would leave the candidate looking unprocessed forever.
        spec_tid = str(params.get("specialist_task_id") or "")
        cand_map = getattr(self.shared_state, "framework_agent_specialist_candidate_map", None)
        mapped_cand = ""
        if isinstance(cand_map, dict) and spec_tid:
            mapped_cand = str(cand_map.get(spec_tid) or "")
        cand_id = str(
            params.get("framework_agent_candidate_id") or mapped_cand or spec_tid or getattr(task, "task_id", "") or ""
        )
        batch_id = str(params.get("framework_batch_id") or "")
        if not batch_id:
            batches = getattr(self.shared_state, "framework_agent_batches", None) or []
            if isinstance(batches, list) and batches and isinstance(batches[-1], dict):
                batch_id = str(batches[-1].get("batch_id") or "")
        delta_pct = res.get("delta_pct")
        new_tput = res.get("output_throughput")
        gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
        progress = getattr(self.shared_state, "framework_agent_phase_progress", None)
        if not isinstance(progress, list):
            progress = []
            self.shared_state.framework_agent_phase_progress = progress
        matching = [row for row in progress if isinstance(row, dict) and self._framework_candidate_key(row) == cand_id]
        if any(str(row.get("provenance") or "") != "authored_empty" for row in matching):
            return
        if matching:
            progress[:] = [
                row for row in progress if not (isinstance(row, dict) and self._framework_candidate_key(row) == cand_id)
            ]
        # Roll the batch max-gain stat the plateau judge reads.
        batches = getattr(self.shared_state, "framework_agent_batches", None) or []
        if isinstance(batches, list) and batch_id:
            for entry in reversed(batches):
                if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                    prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                    if gain > prev:
                        entry["max_gain_pct_observed_in_batch"] = gain
                    break
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=status,
            kept=status == "kept",
            rationale=str(res.get("reason") or ""),
            provenance="authored",
            gain_pct=gain,
            extra={
                "pre_tput": float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                "post_tput": float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
                "accuracy_pass": res.get("accuracy_pass"),
                "specialist_task_id": spec_tid,
                "integrate_task_id": str(getattr(task, "task_id", "") or ""),
            },
        )
        if not recorded:
            return
        log.info(
            "FRAMEWORK: authored patch outcome candidate=%s batch=%s status=%s gain=%.2f%%",
            cand_id,
            batch_id,
            status,
            gain,
        )

    async def _recover_framework_agent_authoring_outcome(
        self,
        *,
        specialist_task: "Task",
    ) -> bool:
        """Recover a missed authoring outcome from persisted delegated results."""
        params = getattr(specialist_task, "params", None) or {}
        specialist_task_id = str(getattr(specialist_task, "task_id", "") or "")
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not specialist_task_id or not cand_id:
            return False
        messages = await self.bus.tail(n=10000, topic="delegated_result")
        for message in messages:
            payload = getattr(message, "payload", None) or {}
            if (
                str(payload.get("task_id") or "") != specialist_task_id
                or str(payload.get("kind") or "") != "specialist"
            ):
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            done_payload = result.get("specialist_done")
            # A run that failed before delivering carries its error on the bus
            # entry rather than in the result, and must reach the recorder for
            # the same reason it does on the live path.
            run_error = str(payload.get("error") or "")
            if isinstance(done_payload, dict) or run_error:
                self._record_framework_agent_authoring_empty_outcome(
                    task=specialist_task,
                    done_payload=done_payload if isinstance(done_payload, dict) else {},
                    run_error=run_error,
                )
                if cand_id in self._framework_processed_candidate_keys():
                    return True
            break
        for message in messages:
            payload = getattr(message, "payload", None) or {}
            if str(payload.get("kind") or "") != "integrate_patch":
                continue
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                continue
            try:
                integrate_task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — stale bus entries are ignored
                continue
            integrate_params = getattr(integrate_task, "params", None) or {}
            if str(integrate_params.get("specialist_task_id") or "") != specialist_task_id or not bool(
                integrate_params.get("framework_agent_authoring")
            ):
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            self._record_framework_agent_authored_outcome(
                task=integrate_task,
                result=result,
            )
            if cand_id in self._framework_processed_candidate_keys():
                return True
        return False

    def _record_framework_agent_dispatch_failure(
        self,
        *,
        task: "Task",
        run_error: str,
    ) -> None:
        """Stamp a terminal row for a candidate whose specialist never ran.

        The row is what stops the pump re-selecting the candidate every tick, so
        it must exist; ``dispatch_failed`` is what keeps the plateau streak from
        counting it as a search that came back empty.

        Args:
            task: The specialist task that failed before delivering.
            run_error: The dispatch error, recorded as the row's rationale.
        """
        params = getattr(task, "params", None) or {}
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=str(params.get("framework_batch_id") or ""),
            status="dispatch_failed",
            kept=False,
            rationale=run_error[:500],
            provenance="dispatch_failed",
            gain_pct=0.0,
            extra={"specialist_task_id": str(getattr(task, "task_id", "") or "")},
        )
        if not recorded:
            return
        log.warning(
            "FRAMEWORK: authoring specialist never delivered candidate=%s: %s",
            cand_id,
            run_error[:300],
        )

    def _record_framework_agent_authoring_empty_outcome(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any] | None,
        run_error: str = "",
    ) -> None:
        """Record a terminal FRAMEWORK row when an authoring specialist finishes WITHOUT a patch.

        The authored-patch bridge (`_record_framework_agent_authored_outcome`)
        only fires on a following ``integrate_patch`` task. An *empty*
        deliverable (specialist judged the PR already-present / not-applicable,
        ``patches_written == []``) never produces an ``integrate_patch``, so
        without this hook the candidate is never marked processed and
        `_select_next_framework_agent_candidate` re-selects it every tick (the
        FRAMEWORK pump livelocks re-dispatching the same candidate). Here we
        stamp a `framework_agent_phase_progress` row + `decision.json` so the pump
        advances. Idempotent: a candidate that already has a row is skipped.

        Args:
            task: The completed authoring specialist task (carries the
                ``framework_*`` provenance markers).
            done_payload: The specialist's ``specialist_done`` payload.
            run_error: The dispatch error when the run failed before delivering
                anything. Together with an absent payload it separates "the
                specialist ran and found nothing" from "the specialist never
                ran", which the plateau streak must not treat alike.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_agent_authoring")):
            return
        payload = done_payload if isinstance(done_payload, dict) else {}
        # No payload at all plus an error means the run never delivered: there is
        # no deliverable to judge, so this is infrastructure, not a search result.
        if run_error and not payload:
            self._record_framework_agent_dispatch_failure(task=task, run_error=run_error)
            return
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        # A downstream integrate_patch (owned by the authored-outcome bridge
        # that writes the terminal row) is created by
        # ``_maybe_autosubmit_specialist_patches`` when the deliverable is
        # routable: ``patches_written`` (post safety-vetting) non-empty, OR a
        # config-lever deliverable, OR a non-diff tuned artifact. This empty-
        # outcome bridge MUST mirror EACH of those signals so it never skip-
        # stamps a deliverable autosubmit will route (double terminal row) nor
        # stamps nothing for one autosubmit will NOT route (livelock). The three
        # guards below (patches / config levers / artifacts) are that mirror. If
        # none hold we stamp the terminal row here, REGARDLESS of
        # ``proposal_set`` — the dangerous case being a specialist that authors a
        # patch (proposal_set non-empty) which safety-vetting then DROPS as
        # unusable (missing_target / forbidden_fields), emptying
        # ``patches_written``: autosubmit creates no integrate_patch, so without
        # stamping here the candidate has no terminal row and the FRAMEWORK pump
        # re-dispatches it forever (gap-5 livelock).
        patches = inner.get("patches_written") or []
        if isinstance(patches, list) and patches:
            return
        # Relaxed FRAMEWORK rule: a config-lever deliverable (proposal_set
        # carrying extra_args / extra_envs) is a FULL result, not "empty". When
        # one exists, ``_maybe_autosubmit_framework_config`` routes it through
        # integrate_patch (which owns the terminal row), so do NOT stamp an
        # authored_empty row here.
        if _framework_config_levers_from_done(inner):
            return
        # Relaxed FRAMEWORK rule (parity with autosubmit): a non-diff tuned
        # artifact deliverable (``artifacts_written`` with a real source file) is
        # a FULL result — autosubmit routes it to integrate_patch, which owns the
        # terminal row. Use the SAME routable-signal as autosubmit so we never
        # skip-stamp a deliverable that autosubmit will NOT route (livelock).
        try:
            from ..loop.coordinator import _resolvable_artifacts_from_done
            from hyperloom.inference_optimizer.session.session_paths import (
                runs_dir as _runs_dir,
            )
            from pathlib import Path as _Path

            _sid_arts = str(getattr(task, "task_id", "") or "")
            if self.session_dir is not None and _sid_arts:
                _spec_root = _runs_dir(_Path(self.session_dir), "specialist", _sid_arts)
                if _resolvable_artifacts_from_done(inner, [_spec_root / "worktree", _spec_root]):
                    return
        except Exception:  # noqa: BLE001 — defensive; fall through to stamp
            log.debug("FRAMEWORK: artifacts routable-check failed", exc_info=True)
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        batch_id = str(params.get("framework_batch_id") or "")
        # Map the cached audit verdict to a terminal status.
        audit = params.get("framework_audit") if isinstance(params.get("framework_audit"), dict) else {}
        sem = str((audit or {}).get("semantic_status") or "").strip().lower()
        if sem in ("already_equivalent", "already_superset"):
            status = "already_present"
        elif sem in ("not_present", "partially_present"):
            status = "not_applicable"
        else:
            status = "author_empty"
        reason = str(inner.get("summary") or "").strip()[:500]
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=status,
            kept=False,
            rationale=reason,
            provenance="authored_empty",
            gain_pct=0.0,
            extra={"specialist_task_id": str(getattr(task, "task_id", "") or "")},
        )
        if not recorded:
            return
        log.info(
            "FRAMEWORK: authoring specialist empty deliverable candidate=%s batch=%s status=%s",
            cand_id,
            batch_id,
            status,
        )

    # Cap on the framework config-exploration grid so a single round cannot
    # monopolise the phase budget (mirrors _MN_AUTO_EXPLORE_GRID_CAP).
    _FRAMEWORK_CONFIG_GRID_CAP = 8
    # Max config-exploration rounds per FRAMEWORK subphase (safety cap; the lane
    # normally terminates earlier once a round yields no new candidates).
    # Overridable via INFERENCE_OPTIMIZER_FRAMEWORK_CONFIG_MAX_ROUNDS.
    _FRAMEWORK_CONFIG_MAX_ROUNDS = 4

    def _build_framework_config_grid(
        self,
        *,
        explicit_grid: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Assemble an explore-schema config grid for the FRAMEWORK_AGENT phase.

        Stage-1 replication of the EXPLORE config-search capability inside
        FRAMEWORK. Candidate sources, in priority order:

        1. ``explicit_grid`` -- a caller/test-supplied list of variant dicts.
        2. The framework's programmatic default seed grid
           (:func:`_default_grid_for_framework`; non-empty for atom / xDiT).

        Specialist config levers and KB recipe best_config are intentionally
        NOT re-sourced here: they already land through
        :meth:`_maybe_autosubmit_framework_config` (integrate_patch) and the
        PRELUDE ``replay_warm_recipe`` path. Every returned variant is stamped
        ``provenance='framework_agent:config'`` so the ledger and report can
        separate framework-driven config search from EXPLORE-phase work.

        Args:
            explicit_grid: Optional caller-supplied variant dicts.

        Returns:
            A (possibly empty) list of variant dicts in the ExploreExecutor
            grid schema, capped at ``_FRAMEWORK_CONFIG_GRID_CAP``.
        """
        provenance = "framework_agent:config"
        grid: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        def _controls(raw: Any) -> dict[str, Any]:
            if not isinstance(raw, dict):
                return {}
            out: dict[str, Any] = {}
            for key in ("remove_args", "unset_envs"):
                value = raw.get(key)
                if isinstance(value, str):
                    vals = [value.strip()] if value.strip() else []
                elif isinstance(value, (list, tuple, set)):
                    vals = [str(v).strip() for v in value if str(v).strip()]
                else:
                    vals = []
                if vals:
                    out[key] = vals
            mode = str(raw.get("args_mode") or "append").strip().lower()
            if mode == "replace":
                out["args_mode"] = "replace"
            return out

        def _add(
            name: str,
            args: str,
            envs: dict[str, str],
            note: str,
            controls: dict[str, Any] | None = None,
        ) -> None:
            nm = (name or "").strip()
            if not nm or nm in seen_names:
                return
            controls = dict(controls or {})
            # A variant with neither a server-arg nor an env override has
            # nothing for the restart to apply unless it removes inherited
            # args/envs.
            if not args and not envs and not controls:
                return
            seen_names.add(nm)
            grid.append(
                {
                    "name": nm,
                    "extra_args": args,
                    "extra_envs": envs,
                    **controls,
                    "provenance": provenance,
                    "note": (note or "")[:200],
                }
            )

        for raw in explicit_grid or []:
            if not isinstance(raw, dict):
                continue
            args = str(raw.get("extra_args") or raw.get("extra_server_args") or "").strip()
            envs_raw = raw.get("extra_envs")
            envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
            _add(
                str(raw.get("name") or ""),
                args,
                envs,
                str(raw.get("note") or raw.get("provenance") or ""),
                _controls(raw),
            )

        # ``explicit_grid=[]`` means the caller harvested an empty set --
        # honour it (no seed fallback); only ``None`` (no grid supplied) seeds.
        if explicit_grid is None and len(grid) < self._FRAMEWORK_CONFIG_GRID_CAP:
            try:
                from ..actions.executors.explore import _default_grid_for_framework

                seeds = _default_grid_for_framework(
                    str(getattr(self.shared_state, "framework", "") or ""),
                    model_class=str(getattr(self.shared_state, "model_class", "") or ""),
                )
            except Exception:  # noqa: BLE001 -- seed grid is best-effort
                log.debug(
                    "framework_config: default seed grid build failed",
                    exc_info=True,
                )
                seeds = []
            for gv in seeds or []:
                gv_envs = {str(k): str(v) for k, v in (getattr(gv, "extra_envs", None) or {}).items()}
                _add(
                    str(getattr(gv, "name", "") or ""),
                    str(getattr(gv, "extra_server_args", "") or "").strip(),
                    gv_envs,
                    str(getattr(gv, "note", "") or ""),
                )

        return grid[: self._FRAMEWORK_CONFIG_GRID_CAP]

    def _framework_config_explore_params(
        self,
        grid: list[dict[str, Any]],
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Build the ``explore`` task params for a framework config round.

        Mirrors the plumbing the LLM / mn-auto explore paths use so the
        ExploreExecutor sees the same base config, base-throughput anchor,
        benchmark script and (via :meth:`_inject_explore_runtime_params`)
        overtime/timeout knobs plus the ``explore_search`` dedup ledger.

        Args:
            grid: The framework config grid (variant dicts).
            reason: Human-readable reason stamped on the task params.

        Returns:
            The explore-task params dict.
        """
        state = self.shared_state
        params: dict[str, Any] = {
            "source": "framework_config_exploration",
            "reason": reason,
            "grid": grid,
        }
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        inject_stack_base_params(params, state, anchor=True)
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        # Thread overtime/timeout knobs + explore_search ledger (dedup memory).
        self._inject_explore_runtime_params(params)
        return params

    async def _run_framework_config_exploration(
        self,
        *,
        explicit_grid: list[dict[str, Any]] | None = None,
        reason: str = "framework_config_lane",
        round_no: int = 0,
    ) -> str:
        """Enqueue one explore-style config-exploration round from FRAMEWORK.

        Stage-1 capability replication: FRAMEWORK reuses the ExploreExecutor
        end-to-end (grid benchmark, overtime kill, throughput + accuracy gate,
        per-KEEP stack rebench, ``explore_search`` dedup) for a multi-variant
        server-arg / env config search. Coordinator-internal; the LLM never
        proposes it and the phase machine is untouched.

        Args:
            explicit_grid: Optional caller/test grid; otherwise the framework
                default seed grid is used.
            reason: Reason stamped on the enqueued task.
            round_no: Round index folded into the idempotency key so each
                config-exploration round dispatches a distinct explore task
                (a shared key would collapse rounds 2..N onto round 1's task).

        Returns:
            The enqueued (or existing) ``explore`` task id, or ``""`` when
            there is no grid to run.
        """
        grid = self._build_framework_config_grid(explicit_grid=explicit_grid)
        if not grid:
            log.info(
                "framework_config: no config grid to run (reason=%s); skipping",
                reason,
            )
            return ""
        params = self._framework_config_explore_params(grid, reason=reason)
        try:
            lanes, ttl = self._registry_lanes_ttl("explore")
            etask, was_existing = await self.tasks.create_or_return_existing(
                kind="explore",
                params=params,
                idempotency_key=f"framework-config-explore-round{int(round_no)}{self._cycle_idem_suffix()}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
        except Exception:  # noqa: BLE001 -- defensive; never wedge the pump
            log.exception("framework_config: failed to enqueue explore round")
            return ""
        log.info(
            "framework_config: enqueued explore task_id=%s (variants=%d reason=%s existing=%s)",
            etask.task_id,
            len(grid),
            reason,
            was_existing,
        )
        return str(getattr(etask, "task_id", "") or "")

    async def _framework_config_exploration_inflight(self) -> bool:
        """True while a framework config-exploration ``explore`` task is live.

        Returns:
            ``True`` when a queued or running ``explore`` task carries the
            ``source='framework_config_exploration'`` marker.
        """
        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 -- defensive
            return False
        for t in (*queued, *running):
            if getattr(t, "kind", "") != "explore":
                continue
            params = getattr(t, "params", None) or {}
            if str(params.get("source") or "") == "framework_config_exploration":
                return True
        return False

    def _framework_config_max_rounds(self) -> int:
        """Resolve the max config-exploration rounds per FRAMEWORK subphase.

        Reads ``INFERENCE_OPTIMIZER_FRAMEWORK_CONFIG_MAX_ROUNDS`` (positive int)
        and otherwise falls back to ``_FRAMEWORK_CONFIG_MAX_ROUNDS``.

        Returns:
            The positive per-subphase round cap.
        """
        try:
            v = int(os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_CONFIG_MAX_ROUNDS", "0") or 0)
        except (TypeError, ValueError):
            v = 0
        return v if v > 0 else self._FRAMEWORK_CONFIG_MAX_ROUNDS

    def _finish_framework_config_lane(self, *, reason: str) -> None:
        """Mark the FRAMEWORK config-exploration subphase done and persist.

        Args:
            reason: Why the lane finished (``no_candidates`` /
                ``generation_empty`` / ``dispatch_skipped`` /
                ``phase_budget_exhausted`` / ``max_rounds``).
        """
        state = self.shared_state
        state.framework_config_lane_state = "done"
        log.info(
            "framework_config: lane done (reason=%s rounds=%d)",
            reason,
            int(getattr(state, "framework_config_lane_round", 0) or 0),
        )
        try:
            state.record_lifecycle_event(
                step="framework_config_lane",
                status=_phase_state.LIFECYCLE_STATUS_END,
                phase=_phase_state.PHASE_FRAMEWORK_AGENT,
                detail=f"reason={reason} rounds={int(getattr(state, 'framework_config_lane_round', 0) or 0)}",
            )
        except Exception:  # noqa: BLE001 -- defensive; observability only
            log.debug("framework_config: lane lifecycle emit failed", exc_info=True)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 -- defensive
            log.exception("framework_config: save after lane finish failed")

    def _framework_config_generation_context_lines(
        self,
        *,
        framework: str,
        direction: str,
        direction_pct: float,
    ) -> list[str]:
        """Build bottleneck + discovered-flag advisory lines for the config
        generation specialist notes (context enrichment).

        Names the live top bottleneck (and dominant roofline direction) plus the
        server/param flags already discovered for ``framework`` so the specialist
        proposes higher-signal, non-redundant variants. All best-effort; returns
        an empty list when no context is available.

        Args:
            framework: The normalized framework key (e.g. ``"sglang"``).
            direction: The dominant roofline direction ("" when unknown).
            direction_pct: The saturation percent for ``direction``.

        Returns:
            A list of advisory note lines (possibly empty).
        """
        state = self.shared_state
        lines: list[str] = []
        try:
            bottleneck = str(state.current_top_bottleneck() or "").strip()
        except Exception:  # noqa: BLE001 -- advisory only; never block dispatch
            bottleneck = ""
        if bottleneck or direction:
            detail = bottleneck or "unknown"
            if direction:
                detail += f" (dominant roofline direction: {direction} {float(direction_pct):.0f}% saturated)"
            lines += [
                "",
                f"CURRENT BOTTLENECK: {detail}.",
                "Prioritise variants that directly relieve THIS bottleneck.",
            ]
        discovered = getattr(state, "discovered_flags", None)
        # Match record_discovered_flags key normalization (blank -> "unknown").
        entry = discovered.get(framework or "unknown") if isinstance(discovered, dict) else None
        if isinstance(entry, dict):
            flag_names = [
                str(f)
                for f in (list(entry.get("backend_flags") or []) + list(entry.get("param_flags") or []))
                if str(f).strip()
            ][:15]
            if flag_names:
                lines.append(
                    "ALREADY-DISCOVERED FLAGS for this framework (build on these, "
                    "avoid re-proposing): " + ", ".join(flag_names)
                )
        return lines

    async def _dispatch_framework_config_generation_specialist(
        self,
        round_no: int,
    ) -> str:
        """Dispatch a read-only, bottleneck-matched specialist that PROPOSES config variants.

        The specialist emits a ``proposal_set`` of ``extra_args`` / ``extra_envs``
        variants (NOT source patches); the config subphase harvests it into a
        grid and benchmarks it via the ExploreExecutor. This is how FRAMEWORK
        replicates EXPLORE's LLM-driven candidate generation. Coordinator-only.

        Args:
            round_no: 1-based round index (for the gap id + idempotency key).

        Returns:
            The dispatched specialist task id, or ``""`` on failure.
        """
        state = self.shared_state
        framework = str(getattr(state, "framework", "") or "").strip().lower()
        gap_cid = f"gap.framework_config.round{int(round_no)}"
        # Increment 1 -- bottleneck-driven domain: mirror EXPLORE's specialist
        # routing so the config-generation specialist matches the current
        # dominant roofline direction (comm/systems/kernel/serving). Falls back
        # to the serving specialist when no roofline snapshot exists yet.
        direction, direction_pct = self._dominant_roofline_direction()
        from ..kernel.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        _hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
        domain = _hint[0] if _hint else "serving_specialist"
        # Increment 2 -- context enrichment: name the live bottleneck + the
        # flags already discovered for this framework so the specialist
        # proposes higher-signal, non-redundant variants.
        context_lines = self._framework_config_generation_context_lines(
            framework=framework,
            direction=direction,
            direction_pct=direction_pct,
        )
        notes = "\n".join(context_lines).strip()
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": "FRAMEWORK_AGENT",
            "gap_canonical_id": gap_cid,
            "gap_symptom": ("Propose runtime config variants (server args / env) for a throughput grid"),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "framework_config_generation",
            # Marker so completion harvest routes the proposal_set into the config
            # subphase (and the mn-explore bridge skips it to avoid double-consume).
            "framework_config_generation": True,
            "source": "coordinator_internal",
            "mode": "research",
            "notes": notes,
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 -- best-effort warmup
            log.debug("framework_config: warm specialist params failed", exc_info=True)
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        idem = f"framework-config-generation:round{int(round_no)}{self._cycle_idem_suffix()}"
        try:
            spec_task, _existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                side_effects=["writes_results"],
                lease_ttl_sec=ttl,
            )
        except Exception:  # noqa: BLE001 -- defensive; never wedge the pump
            log.exception("framework_config: generation specialist dispatch failed")
            return ""
        log.info(
            "framework_config: dispatched generation specialist task_id=%s round=%d",
            getattr(spec_task, "task_id", ""),
            int(round_no),
        )
        return str(getattr(spec_task, "task_id", "") or "")

    async def _framework_config_generation_inflight(self) -> bool:
        """True while a framework config-generation specialist is queued/running."""
        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 -- defensive
            return False
        for t in (*queued, *running):
            if getattr(t, "kind", "") != "specialist":
                continue
            params = getattr(t, "params", None) or {}
            if bool(params.get("framework_config_generation")):
                return True
        return False

    def _framework_config_grid_from_proposals(
        self,
        proposals: list[Any],
    ) -> list[dict[str, Any]]:
        """Convert a specialist ``proposal_set`` into framework config grid dicts.

        Mirrors the mn-explore transform: keep entries carrying a server-arg or
        env override, stamp ``provenance='framework_agent:config'``.

        Args:
            proposals: The specialist ``proposal_set`` entries.

        Returns:
            Variant dicts in the ExploreExecutor grid schema.
        """
        out: list[dict[str, Any]] = []
        for i, p in enumerate(proposals or []):
            if not isinstance(p, dict):
                continue
            args = str(p.get("extra_args") or p.get("extra_server_args") or "").strip()
            envs_raw = p.get("extra_envs")
            envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
            controls: dict[str, Any] = {}
            for key in ("remove_args", "unset_envs"):
                raw = p.get(key)
                if isinstance(raw, str):
                    vals = [raw.strip()] if raw.strip() else []
                elif isinstance(raw, (list, tuple, set)):
                    vals = [str(v).strip() for v in raw if str(v).strip()]
                else:
                    vals = []
                if vals:
                    controls[key] = vals
            mode = str(p.get("args_mode") or "append").strip().lower()
            if mode == "replace":
                controls["args_mode"] = "replace"
            if not args and not envs and not controls:
                continue
            name = str(p.get("name") or "").strip() or f"framework-config-{i}"
            out.append(
                {
                    "name": name,
                    "extra_args": args,
                    "extra_envs": envs,
                    **controls,
                    "provenance": "framework_agent:config",
                    "note": str(p.get("reason") or "")[:200],
                }
            )
        return out

    def _ingest_framework_config_generation(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
    ) -> None:
        """Harvest a config-generation specialist's ``proposal_set`` into the
        pending grid for the config subphase.

        No-op unless the task carries the ``framework_config_generation`` marker.

        Args:
            task: The completed specialist task.
            done_payload: Its ``specialist_done`` payload.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_config_generation")):
            return
        proposals = done_payload.get("proposal_set") if isinstance(done_payload, dict) else None
        grid = self._framework_config_grid_from_proposals(proposals or [])
        state = self.shared_state
        if not isinstance(getattr(state, "framework_config_pending_grid", None), list):
            state.framework_config_pending_grid = []
        state.framework_config_pending_grid = grid
        log.info(
            "framework_config: harvested %d generated config variant(s) from task=%s",
            len(grid),
            getattr(task, "task_id", ""),
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 -- defensive
            log.exception("framework_config: save after generation ingest failed")

    async def _start_framework_config_generation(self, *, round_no: int) -> bool:
        """Start a config-exploration round by dispatching a generation specialist.

        Falls back to the deterministic default seed grid when generation cannot
        be dispatched, so the lane still makes progress (or finishes cleanly).

        Args:
            round_no: 0-based count of rounds already run.

        Returns:
            ``True`` to hold the phase; ``False`` when the lane finished.
        """
        state = self.shared_state
        gen_id = await self._dispatch_framework_config_generation_specialist(round_no + 1)
        if gen_id:
            state.framework_config_lane_state = "generating"
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 -- defensive
                log.exception("framework_config: save after generation dispatch failed")
            return True
        # Generation unavailable: fall back to the deterministic default grid.
        new_variants = self._build_framework_config_grid()
        if not new_variants:
            self._finish_framework_config_lane(reason="no_candidates")
            return False
        task_id = await self._run_framework_config_exploration(
            explicit_grid=new_variants,
            reason=f"framework_config_round_{round_no + 1}",
            round_no=round_no + 1,
        )
        if not task_id:
            self._finish_framework_config_lane(reason="dispatch_skipped")
            return False
        state.framework_config_lane_state = "running"
        state.framework_config_lane_round = round_no + 1
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 -- defensive
            log.exception("framework_config: save after fallback dispatch failed")
        return True

    def _framework_config_lane_should_engage(self, next_phase) -> bool:
        """True when the (default-OFF) FRAMEWORK config-exploration lane should
        engage this tick.

        Engages only when the config-exploration flag is set and the phase
        machine is about to leave FRAMEWORK_AGENT (so the subphase can hold the
        phase and run config rounds first). Extracted from
        :meth:`_advance_phase_if_needed` so the trigger is unit-testable.

        Args:
            next_phase: The ``compute_next_phase`` result (a ``(phase, reason,
                evidence)`` tuple) or ``None``.

        Returns:
            ``True`` when the config subphase should be given a chance to hold
            the phase; ``False`` otherwise (the default flow).
        """
        if not bool(getattr(self.shared_state, "framework_config_exploration_enabled", False)):
            return False
        if next_phase is None:
            return False
        current = str(getattr(self.shared_state, "phase", "") or "").strip().upper()
        target = str(next_phase[0]).strip().upper()
        return current == _phase_state.PHASE_FRAMEWORK_AGENT and target != _phase_state.PHASE_FRAMEWORK_AGENT

    async def _maybe_hold_for_framework_config_lane(self) -> bool:
        """(Default OFF) Drive the FRAMEWORK config-exploration subphase.

        Route B: FRAMEWORK replicates EXPLORE's LLM-driven config search as a
        two-step-per-round loop over ``framework_config_lane_state``
        (``''`` -> ``'generating'`` -> ``'running'`` -> ``'done'``):

        * ``''`` -> dispatch a config-generation specialist (LLM proposes a
          config variant grid); hold.
        * ``'generating'`` -> hold while the specialist runs; on completion its
          harvested ``proposal_set`` (in ``framework_config_pending_grid``) is
          de-duped against the ``explore_search`` tested ledger and benchmarked
          via an explore round; empty -> finish.
        * ``'running'`` -> hold while the explore round runs; on completion start
          the next round (or finish at the round cap).

        When the flag is False (the default) this is a no-op returning ``False``
        so the standard PRELUDE -> FRAMEWORK_AGENT -> EXPLORE flow is unchanged.

        Returns:
            ``True`` to hold the phase inside the config subphase; ``False`` to
            let it advance as usual.
        """
        state = self.shared_state
        if not bool(getattr(state, "framework_config_exploration_enabled", False)):
            return False
        lane = str(getattr(state, "framework_config_lane_state", "") or "")
        if lane == "done":
            return False
        # Phase budget spent: the dispatcher stops spawning new phase-scoped
        # variants (incl. our queued explore round) while the inflight check
        # counts queued tasks, so holding here would livelock the phase past
        # its budget. Yield so the phase-budget-exhausted exit can advance.
        if self._dispatch_paused_for_phase_budget():
            self._finish_framework_config_lane(reason="phase_budget_exhausted")
            return False
        if lane == "":
            return await self._start_framework_config_generation(round_no=0)
        if lane == "generating":
            if await self._framework_config_generation_inflight():
                return True
            pending = list(getattr(state, "framework_config_pending_grid", None) or [])
            state.framework_config_pending_grid = []
            round_no = int(getattr(state, "framework_config_lane_round", 0) or 0)
            new_variants = self._build_framework_config_grid(explicit_grid=pending)
            if not new_variants:
                self._finish_framework_config_lane(reason="generation_empty")
                return False
            task_id = await self._run_framework_config_exploration(
                explicit_grid=new_variants,
                reason=f"framework_config_round_{round_no + 1}",
                round_no=round_no + 1,
            )
            if not task_id:
                self._finish_framework_config_lane(reason="dispatch_skipped")
                return False
            state.framework_config_lane_state = "running"
            state.framework_config_lane_round = round_no + 1
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 -- defensive
                log.exception("framework_config: save after round dispatch failed")
            return True
        if lane == "running":
            if await self._framework_config_exploration_inflight():
                return True
            round_no = int(getattr(state, "framework_config_lane_round", 0) or 0)
            if round_no >= self._framework_config_max_rounds():
                self._finish_framework_config_lane(reason="max_rounds")
                return False
            return await self._start_framework_config_generation(round_no=round_no)
        return False

    def _record_framework_config_exploration_result(
        self,
        *,
        task: "Task",
        result: dict[str, Any],
    ) -> None:
        """Append a compact record of a framework config-exploration round.

        Stored on ``framework_config_exploration_results`` (NOT
        ``framework_agent_phase_progress``) so framework config search never
        perturbs the source-candidate plateau gate. Winners are already
        promoted through the standard explore harvest (current_best /
        optimization_stack / explore_search).

        Args:
            task: The completed framework-config ``explore`` task.
            result: The explore result dict (per_variant_outcomes / winners).
        """
        from ..state.shared_state import _now_iso

        state = self.shared_state
        outcomes = result.get("per_variant_outcomes") or []
        kept = sum(1 for o in outcomes if isinstance(o, dict) and str(o.get("outcome") or "").upper() == "KEEP")
        row = {
            "task_id": str(getattr(task, "task_id", "") or ""),
            "reason": str((getattr(task, "params", None) or {}).get("reason") or ""),
            "variant_count": len(result.get("grid") or []) or len(outcomes),
            "kept": kept,
            "best_gain_pct": result.get("best_gain_pct"),
            "ts": _now_iso(),
        }
        if not isinstance(getattr(state, "framework_config_exploration_results", None), list):
            state.framework_config_exploration_results = []
        state.framework_config_exploration_results.append(row)
        try:
            state.record_lifecycle_event(
                step="framework_config_round",
                status=_phase_state.LIFECYCLE_STATUS_END,
                phase=_phase_state.PHASE_FRAMEWORK_AGENT,
                detail=f"task={row['task_id']} kept={kept} variants={row['variant_count']}",
            )
        except Exception:  # noqa: BLE001 -- defensive; observability only
            log.debug("framework_config: round lifecycle emit failed", exc_info=True)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 -- defensive
            log.exception("framework_config: save after result record failed")
