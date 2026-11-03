# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The supervisor's ruling on a round: when it is asked for, how long it stands, and how a resumed run gets it back."""

from __future__ import annotations

import logging
from pathlib import Path

from kernelforge.workflows.forge_loop.campaign.run_state import (
    CriticRuling,
    apply_supervisor_attempt,
    apply_supervisor_intervention,
    make_event,
)
from kernelforge.workflows.forge_loop.roles.supervisor import clear_latest_supervisor_ruling, persist_supervisor_ruling

log = logging.getLogger(__name__)


class SupervisorReviewMixin:
    """Asking the supervisor to rule, and keeping the ruling."""

    async def _run_supervisor_review(
        self, supervisor_fn, *, iteration, digest, supervisor_due, supervisor_reason
    ) -> None:
        """Ask the supervisor for a ruling when the run has stalled, and record whatever comes back.

        The memo is optional: a supervisor that fails, or returns nothing,
        leaves the run to continue without an active ruling.
        """
        if self.monitor is not None and supervisor_fn is not None:
            if supervisor_due:
                print(f"\n[supervisor] intervening at iteration {iteration}: {supervisor_reason}")
                memo = ""
                try:
                    evidence_context = self._build_supervisor_evidence_context(iteration)
                    # A new review attempt supersedes the prior stall episode's ruling even when the backend
                    # returns empty.
                    self._expire_supervisor_ruling()
                    self.monitor.mark_attempted(iteration)
                    apply_supervisor_attempt(
                        self.run_state,
                        iteration=iteration,
                    )
                    self.state_store.append_event(
                        make_event(
                            "supervisor_attempt",
                            iteration,
                            reason=supervisor_reason,
                        )
                    )
                    self.state_store.save(self.run_state)
                    memo = await supervisor_fn(
                        digest=digest,
                        reason=supervisor_reason,
                        workspace=self.ic.workspace_dir,
                        iteration=iteration,
                        evidence_context=evidence_context,
                    )
                except Exception as e:  # isolation boundary: one stall-triggered supervisor review
                    log.warning("supervisor review failed at iteration %s", iteration, exc_info=True)
                    print(f"  [supervisor] failed ({e}); continuing without a memo")
                finally:
                    self._checkpoint_llm_usage()
                memo = memo or ""
                if memo.strip():
                    self._record_supervisor_ruling(iteration, supervisor_reason, memo)
                else:
                    print("  [supervisor] no new ruling returned; continuing without an active ruling")
                if memo.strip():
                    self.monitor.mark_intervened(iteration)
                    apply_supervisor_intervention(
                        self.run_state,
                        iteration=iteration,
                        stall_threshold=self.ic.supervise_after,
                    )
                    self.state_store.save(self.run_state)

    def _record_supervisor_ruling(self, iteration: int, supervisor_reason: str, memo: str) -> None:
        """Persist a returned ruling, make it the active one, and log its event."""
        interaction_path, ruling_path = persist_supervisor_ruling(
            self.ic.workspace_dir,
            iteration,
            supervisor_reason,
            memo,
        )
        self._supervisor_ruling = memo
        print(f"  [supervisor] injected free-form ruling: {len(self._supervisor_ruling)} chars")
        self.state_store.append_event(
            make_event(
                "supervisor_ruling",
                iteration,
                reason=supervisor_reason,
                ruling_len=len(self._supervisor_ruling),
                interaction_path=(
                    str(interaction_path.relative_to(Path(self.ic.workspace_dir)))
                    if interaction_path is not None
                    else None
                ),
                ruling_path=(
                    str(ruling_path.relative_to(Path(self.ic.workspace_dir))) if ruling_path is not None else None
                ),
            )
        )

    def _record_critic_ruling(self, iteration: int, critic) -> None:
        """Put this round's verdict where the next process can still find it."""
        ruling = CriticRuling()
        if critic is not None and not critic.fail_open:
            ruling = CriticRuling(
                verdict=critic.verdict,
                review_path=str((self._orchestration_root(iteration) / "critic_review.md").resolve()),
            )
        self.run_state.last_critic = ruling

    def _restore_critic_ruling(self) -> None:
        """Resume the ruling whose round ended with the process that bought it."""
        ruling = self.run_state.last_critic
        if not ruling.verdict or not ruling.review_path:
            return
        try:
            review = Path(ruling.review_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            log.warning(
                "critic review at %s is unreadable (%s); resuming without the %s verdict it carried",
                ruling.review_path,
                error,
                ruling.verdict,
            )
            self.run_state.last_critic = CriticRuling()
            return
        if not review:
            self.run_state.last_critic = CriticRuling()
            return
        self._last_critic_verdict = ruling.verdict
        self._last_critic_review = review
        print(f"  [critic] resuming the {ruling.verdict} verdict an earlier process recorded")

    def _expire_supervisor_ruling(self) -> None:
        """Stop injecting a ruling after its stall episode ends."""
        self._supervisor_ruling = ""
        clear_latest_supervisor_ruling(self.ic.workspace_dir)
