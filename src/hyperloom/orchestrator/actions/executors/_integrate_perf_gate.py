# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The performance gate: whether a measured patch is kept, and what a switched-off variant has to match before it is."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from hyperloom.common.gain_math import gain_pct
from hyperloom.common.perf_metric import VERDICT_KEEP, agentx_enabled
from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint
from hyperloom.orchestrator.measurement.runtime_findings import KEEP_REASON_CORRECTNESS_FIX

from ...state.shared_state import resolve_anchor_with_drift, resolve_graded_comparison
from ..stop_attribution import stopped_by_the_run_class
from ._accuracy_gate import accuracy_keep_block, eval_probe_summary
from ._enablement_config import DEFAULT_SWITCH_OFF_PARITY_BAND_PCT
from ._git import _git_head_sha
from ._integrate_attempt import IntegrateAttempt
from ._integrate_refusals import (
    _accuracy_delta_pct,
    _correctness_fix_refusals,
    _fix_accuracy_refusals,
    _fix_axis,
    _gate_failure_reasons,
    _measured_against,
)
from ._integrate_results import (
    _PatchBench,
    _PatchMeasurement,
    _inert_keep_reason,
    _kept_patch,
    _perf_revert,
    _revert_status,
)
from ._integrate_vocabulary import SCREEN_CYCLE, SCREEN_PARAM
from ._patch_git import _uncommitted_revert
from ._patch_sources import _done_field

log = logging.getLogger(__name__)

#: Why an agentic replay keeps a switched patch without a switches-off bench. That leg is a whole replay hour, the
#: replay's own run-to-run drift (6-8.5% on GLM-5.3) is wider than the band it would be judged in, so it would cost
#: every switched patch a round and then revert most of them on noise; an off path that cannot even import already
#: fails the switches-on bench before any of this runs.
_AGENTIC_PARITY_NOT_BENCHED = (
    "switch-off parity not benched under an agentic replay: one leg is a full replay hour and the replay's run-to-run "
    "drift is wider than the parity band, so the leg could only revert on noise"
)


_PARITY_DEFERRED_TO_CYCLE_BASE = "switch-off parity deferred to the next cycle batch, whose base runs this tree"


def _graded_change(graded: Any, delta_pct: float | None) -> float | None:
    """The change on the graded objective; an interactivity pair reports it even when the gate leaves ``delta_pct``."""
    if graded.comparable and graded.graded_on_intvty:
        return gain_pct(graded.candidate, graded.reference)
    return delta_pct


class PerfGateMixin:
    """Decide keep or revert from the measurement, and establish the parity a switched-off variant has to hold."""

    async def _gate_perf(
        self,
        *,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Throughput KEEP / REVERT decision, or no verdict when the run stopped it."""
        specialist_task_id = attempt.specialist_task_id
        shared_state = attempt.shared_state
        done_payload = attempt.done_payload
        assert attempt.output_root is not None
        # Grade against the current live anchor, not a stale task snapshot.
        base_tput, anchor_drifted = resolve_anchor_with_drift(
            float(params.get("base_tput") or 0.0),
            shared_state,
        )
        if anchor_drifted:
            log.warning(
                "integrate_patch: anchor drift; grading against live anchor %.1f",
                base_tput,
            )

        keep_threshold_pct = float(params.get("keep_threshold_pct", self.keep_threshold_pct))

        stopped = stopped_by_the_run_class(bench_result.get("error_class"))
        if stopped is not None:
            return {
                "status": "failed",
                "error_class": stopped.error_class,
                "error": stopped.interrupted,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "bench_result": bench_result,
                "workspace": str(attempt.output_root),
            }

        graded = resolve_graded_comparison(shared_state, bench_result, keep_threshold_pct=keep_threshold_pct)
        if not graded.comparable:
            log.info("integrate_patch: performance comparison unavailable (%s)", graded.degrade_reason)
        if not graded.graded_on_intvty:
            delta_pct = gain_pct(bench_result.get("output_throughput"), base_tput)
        elif graded.verdict == VERDICT_KEEP:
            delta_pct = gain_pct(graded.candidate, graded.reference)
        else:
            log.info(
                "integrate_patch: %s intvty %.1f->%.1f tput %.1f->%.1f",
                graded.verdict,
                graded.reference,
                graded.candidate,
                graded.tput_reference,
                graded.tput_candidate,
            )
            delta_pct = None

        accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
        fw_authored = bool(params.get("framework_agent_authoring") or params.get("framework_agent_candidate_id"))
        required = bool(params.get("require_accuracy_for_keep", fw_authored))
        acc_baseline = params.get("accuracy_baseline")  # filled by _stage_resolve
        acc_block, acc_reason, acc_degraded = accuracy_keep_block(
            accuracy_pass,
            required=required,
            baseline_accuracy=acc_baseline,
        )
        if acc_degraded:
            log.warning(
                "integrate_patch: accuracy gate required but no baseline accuracy; "
                "KEEP allowed on throughput only (task=%s)",
                specialist_task_id,
            )
        acc_delta_pct = _accuracy_delta_pct(gate_evidence.get("accuracy"), acc_baseline)

        gate_pass = (
            graded.comparable
            and delta_pct is not None
            and delta_pct >= keep_threshold_pct
            and not graded.veto_reason
            and not acc_block
        )
        fix_finding = str(_done_field(done_payload, "resolves_finding") or "").strip()
        fix_refusals: list[str] = []
        correctness_fix = False
        if not gate_pass and fix_finding:
            fix_delta_pct = gain_pct(graded.candidate, graded.reference) if graded.graded_on_intvty else delta_pct
            fix_refusals = _correctness_fix_refusals(
                fix_finding,
                shared_state=shared_state,
                bench_result=bench_result,
                graded=graded,
                delta_pct=fix_delta_pct,
            ) + _fix_accuracy_refusals(accuracy_pass)
            if not fix_refusals:
                gate_pass = correctness_fix = True
                delta_pct = fix_delta_pct
        measured = _PatchMeasurement(
            bench_result=bench_result,
            new_tput=bench_result.get("output_throughput"),
            delta_pct=delta_pct,
            base_tput=base_tput,
            keep_threshold_pct=keep_threshold_pct,
            accuracy_pass=accuracy_pass,
            acc_delta_pct=acc_delta_pct,
            cfg_fingerprint=canonical_fingerprint(params.get("extra_server_args"), params.get("extra_envs")),
            measured_against=_measured_against(params, base_tput=base_tput),
            graded_delta_pct=_graded_change(graded, delta_pct),
        )

        switch_manifest: list[dict[str, Any]] = list(attempt.switch_manifest)

        # Switch-off parity. Run before either KEEP verdict, since both of them
        # leave the patch on disk and therefore both depend on it being inert when
        # disabled.
        #
        # It runs on a quality regression too, which looks wasteful and is not: the
        # switches are benched together, so a moved output localises to the bundle
        # rather than to a switch. On a live session a four-switch bundle reached
        # +65.5% and was reverted whole on the gate, discarding three switches that
        # were never implicated along with the one that was. Default-off code costs
        # nothing to keep and explore can bisect it per lever -- but only if the tree
        # is genuinely unchanged with every switch unset, which is exactly what this
        # leg measures. An unswitched patch has no "off" state to fall back to, so
        # it still reverts without spending the leg.
        parity: dict[str, Any] = {"ran": False, "ok": True, "reason": ""}
        if switch_manifest:
            # The parity leg is an additional full bench, so it needs the same
            # session bound the first bench got. Resolved here rather than
            # threaded from the caller because the deadline is an absolute
            # monotonic timestamp: the budget the first bench spent is already
            # reflected in it. Only this leg reads it, so it is resolved only
            # when the leg runs.
            parity = await self._switch_off_parity(
                _PatchBench.for_attempt(attempt, params), switch_manifest=switch_manifest, base_tput=base_tput
            )
        if not parity.get("ok"):
            # An unmeasurable parity leg reverts under its own verdict: the patch was
            # never shown to break, so neither the log line nor the KB lesson may say
            # that it did.
            from ...knowledge import kb_writeback as _kb

            inconclusive = bool(parity.get("inconclusive"))
            log.warning(
                "integrate_patch: switch-off parity %s task=%s: %s",
                "INCONCLUSIVE" if inconclusive else "FAILED",
                specialist_task_id,
                parity.get("reason"),
            )
            await self._maybe_write_framework_kb_record(
                measured.lesson(
                    _kb.OUTCOME_REVERTED_PARITY_INCONCLUSIVE if inconclusive else _kb.OUTCOME_REVERTED_SWITCH_OFF_PARITY
                ),
                done_payload=done_payload,
                params=params,
                extra=extra,
            )
            return _perf_revert(
                attempt,
                measured,
                status="reverted",
                reason=str(parity.get("reason") or "switch-off parity failed"),
                error_class="switch_off_parity_inconclusive" if inconclusive else "switch_off_parity_failed",
                switch_off_parity=parity,
            )

        if gate_pass:
            await self._maybe_write_framework_kb_record(
                measured.lesson("integrated"), done_payload=done_payload, params=params, extra=extra
            )
            commit_failure, keep_committed = self._commit_accepted_work(
                attempt, f"hyperloom KEEP {specialist_task_id} ({delta_pct:+.2f}%)"
            )
            if commit_failure:
                return _uncommitted_revert(
                    attempt,
                    commit_failure,
                    "KEEP",
                    output_throughput=measured.new_tput,
                    delta_pct=delta_pct,
                    measured_against=measured.measured_against,
                    bench_result=bench_result,
                )
            return _kept_patch(
                attempt,
                measured,
                params=params,
                reason=(
                    f"correctness fix {fix_finding} verified at {_fix_axis(graded)} delta {measured.delta_pct:+.2f}%"
                    if correctness_fix
                    else f"throughput delta {measured.delta_pct:+.2f}% >= {measured.keep_threshold_pct:.2f}%"
                ),
                keep_reason=KEEP_REASON_CORRECTNESS_FIX if correctness_fix else "",
                resolves_finding=fix_finding if correctness_fix else "",
                # The bundle cleared the gate, so its switches join the running
                # configuration and are registered as levers that are already on.
                # Attribution from here is leave-one-out.
                framework_levers=switch_manifest,
                framework_lever_outcome=("default_on" if switch_manifest else ""),
                switch_off_parity=parity,
                **self._capture_source_layer(attempt, params, committed=keep_committed),
            )

        # Two-tier verdict for a framework-rewrite patch. Every rewrite in it
        # is behind a switch that defaults OFF, so keeping the code with the
        # switches unset changes nothing at runtime -- which makes reverting
        # it the more expensive choice. The bundle failed as a bundle, but a
        # bundle usually mixes rewrites that pay with one that does not, and
        # some of them are enablers that cannot pay until measured together
        # with what they unlock. So keep the code inert, register the switches
        # as levers, and let the explore phase find the subset that wins.
        #
        # A quality regression does not condemn the bundle either, for the same
        # reason: the switches are benched together, so a moved output says
        # "at least one of these is wrong", not "all of them are". The bundle
        # stays inert and explore bisects it per lever -- the verdict carries
        # ``quality_unverified`` so nothing downstream mistakes it for a clean
        # keep. What still condemns it is failing parity, handled above: code
        # that is not inert when disabled would skew every later measurement,
        # and that check now runs on this path too.
        if switch_manifest and attempt.applied:
            return await self._keep_inert_switches(attempt, measured, params=params, extra=extra, parity=parity)
        await self._maybe_write_framework_kb_record(
            measured.lesson("reverted_smoke_fail"), done_payload=done_payload, params=params, extra=extra
        )
        reasons = _gate_failure_reasons(
            graded,
            measured.delta_pct,
            measured.keep_threshold_pct,
            acc_reason if acc_block else "",
            f"correctness fix {fix_finding} refused: {'; '.join(fix_refusals)}" if fix_refusals else "",
            eval_probe_summary(gate_evidence.get("eval_probe")),
        )
        return _perf_revert(
            attempt,
            measured,
            status=_revert_status(
                measured, accuracy_blocks=acc_block, accuracy_verdict=accuracy_pass, fix_refusals=fix_refusals
            ),
            reason="; ".join(reasons) or "gate failed",
        )

    def _capture_source_layer(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        *,
        committed: bool,
    ) -> dict[str, Any]:
        """Snapshot the source files this KEEP leaves in the tree, so it can be replayed elsewhere."""
        evidence: dict[str, Any] = {
            "source_snapshot": "",
            "source_manifest": "",
            "source_snapshot_complete": False,
            "source_import_root": "",
            "source_realized_patch": "",
            "source_artifacts_outside_root": 0,
            "target_files": [],
            "base_sha": "",
        }
        framework_root = attempt.framework_root
        if framework_root is None:
            return evidence
        try:
            from ...enablement.runtime.adapters import get_adapter
            from ...source_snapshot import MANIFEST_NAME, SourceLayer, snapshot_source_layer
            from ._patch_snapshot import _patch_touched_paths_split, harvest_realized_diff

            base_sha = _git_head_sha(framework_root)
            evidence["base_sha"] = base_sha
            upserted, deleted = _patch_touched_paths_split(framework_root, attempt.applied)
            declared_ops = dict.fromkeys(upserted, "upsert")
            declared_ops.update(dict.fromkeys(deleted, "delete"))
            # An artifact installed into a sibling tree is not addressable by a rel
            # path under this root, so a KEEP whose gain lives outside the tree
            # reads as a known gap rather than as a clean capture.
            here = framework_root.resolve()
            rel_paths = upserted + deleted
            outside = 0
            for entry in attempt.applied_artifacts:
                if Path(str(entry.get("root") or framework_root)).resolve() == here:
                    rel_paths.append(str(entry["rel_target"]))
                else:
                    outside += 1
            evidence["source_artifacts_outside_root"] = outside
            import_root = get_adapter(str(params.get("framework") or "")).source_import_root(str(framework_root))
            evidence["source_import_root"] = import_root
            slot = attempt.specialist_task_id or str(attempt.task_id or "keep")
            snap = snapshot_source_layer(
                SourceLayer(
                    framework_root=framework_root,
                    base_sha=base_sha,
                    rel_paths=rel_paths,
                    declared_ops=declared_ops,
                    import_root=import_root,
                ),
                dest_dir=self.session_dir / "optimization_stack" / "src" / slot,
                provenance="integrate_patch",
                extra={"specialist_task_id": attempt.specialist_task_id},
            )
            if not snap:
                return evidence
            snapshot_dir = str(snap.get("snapshot_dir") or "")
            evidence["source_snapshot"] = snapshot_dir
            if snapshot_dir:
                evidence["source_manifest"] = str(Path(snapshot_dir) / MANIFEST_NAME)
            evidence["target_files"] = [
                str(item.get("rel") or "")
                for item in (snap.get("files") or [])
                if isinstance(item, dict) and item.get("rel")
            ]
            evidence["source_snapshot_complete"] = bool(snap.get("complete"))
            # Only harvest when a real commit landed. Without a new commit
            # ``HEAD^..HEAD`` is the previous KEEP; harvesting it would publish that
            # diff as this KEEP's realized change. Leaving it empty falls back to the
            # delivered patch (realized=False), which is the honest record.
            if committed:
                evidence["source_realized_patch"] = harvest_realized_diff(
                    framework_root, rel_paths, Path(snapshot_dir) / "realized.patch"
                )
        except Exception:
            log.exception("integrate_patch: source-layer snapshot failed")
        return evidence

    async def _switch_off_parity(
        self,
        bench: _PatchBench,
        *,
        switch_manifest: list[dict[str, Any]],
        base_tput: float,
    ) -> dict[str, Any]:
        """Verify the patch is genuinely inert with every rewrite switch unset.

        The whole lever mechanism rests on one invariant: with no switch set, the
        patched tree behaves exactly like the original. That invariant is what
        makes it safe to keep unprofitable rewrite code on disk, what makes a
        per-lever measurement mean anything, and what keeps the baseline
        comparable across a session that has accumulated several rewrite patches.
        It is also the invariant an LLM is most likely to break by accident — by
        reading the switch once at import, inverting a default, or restructuring
        code outside the guard — and nothing else in the pipeline would notice: a
        switches-on bench that improves throughput looks like a success whether or
        not the switches-off path still works.

        So it is measured, not assumed. One extra leg with the switches removed
        must land inside a noise band around the pre-patch base.

        Args:
            bench: The task params, workspace, originating specialist and the
                session bounds the parity bench runs under.
            switch_manifest: Parsed switch manifest.
            base_tput: Pre-patch throughput to compare against.

        Returns:
            ``{"ran", "ok", "tput", "delta_pct", "band_pct", "accuracy_pass",
            "reason"}``. ``ran`` is False when the check was skipped: disabled,
            or an agentic replay, passes and says why; no usable base to
            compare against is inconclusive and reverts.
        """
        params = bench.params
        band_pct = float(params.get("switch_off_parity_band_pct", DEFAULT_SWITCH_OFF_PARITY_BAND_PCT))
        if not bool(params.get("enable_switch_off_parity", True)):
            return {"ran": False, "ok": True, "reason": "switch-off parity check disabled"}
        if agentx_enabled():
            return {"ran": False, "ok": True, "reason": _AGENTIC_PARITY_NOT_BENCHED}
        if base_tput <= 0:
            return {
                "ran": False,
                "ok": False,
                "inconclusive": True,
                "reason": "switch-off parity could not be measured: no positive base throughput to compare against",
            }
        switch_names = [entry["switch"] for entry in switch_manifest]
        try:
            parity_bench, parity_evidence = await self._bench_patch(
                bench,
                extra_server_args_applied="",
                extra_envs_applied={},
                unset_envs=switch_names,
                variant_suffix="-parity",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "ran": True,
                "ok": False,
                "reason": f"switch-off parity leg raised: {exc!r}",
            }
        parity_tput = parity_bench.get("output_throughput")
        accuracy_pass = parity_evidence.get("accuracy_pass")
        if not isinstance(parity_tput, (int, float)) or parity_tput <= 0:
            # No measurement is not evidence of a behavioural change: the patch is
            # still reverted, but the verdict must not claim the invariant was broken.
            return {
                "ran": True,
                "ok": False,
                "inconclusive": True,
                "tput": parity_tput,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity could not be measured: the parity leg returned "
                    "no throughput, so the switches-unset invariant was never tested. "
                    "Reverting because an unverified rewrite must not stay on disk, not "
                    "because the patch was shown to be non-inert"
                ),
            }
        delta_pct = (float(parity_tput) - base_tput) / base_tput * 100.0
        if abs(delta_pct) > band_pct:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    f"switch-off parity leg moved throughput {delta_pct:+.2f}% "
                    f"(band +/-{band_pct:.2f}%): the patch changes behaviour with "
                    f"every switch unset, so it is not a default-off rewrite"
                ),
            }
        if accuracy_pass is False:
            return {
                "ran": True,
                "ok": False,
                "tput": float(parity_tput),
                "delta_pct": delta_pct,
                "band_pct": band_pct,
                "accuracy_pass": accuracy_pass,
                "reason": (
                    "switch-off parity leg failed its correctness gate: the patch "
                    "changes output with every switch unset"
                ),
            }
        return {
            "ran": True,
            "ok": True,
            "tput": float(parity_tput),
            "delta_pct": delta_pct,
            "band_pct": band_pct,
            "accuracy_pass": accuracy_pass,
            "reason": f"switch-off parity within +/-{band_pct:.2f}% ({delta_pct:+.2f}%)",
        }

    async def _admit_dormant(self, attempt: IntegrateAttempt, params: dict[str, Any], extra: dict[str, Any]) -> dict:
        """Commit an applied switched patch dormant without a bench, for the cycle bench to screen its levers.

        Inertness with every switch unset is shown by the next cycle batch, whose base runs this tree.
        """
        measured = _PatchMeasurement(
            bench_result={},
            new_tput=None,
            delta_pct=None,
            base_tput=float(params.get("base_tput") or 0.0),
            keep_threshold_pct=float(params.get("keep_threshold_pct", self.keep_threshold_pct)),
            accuracy_pass=None,
            acc_delta_pct=None,
            cfg_fingerprint=canonical_fingerprint(params.get("extra_server_args"), params.get("extra_envs")),
            measured_against={},
        )
        parity = {"ran": False, "ok": True, "reason": _PARITY_DEFERRED_TO_CYCLE_BASE}
        result = await self._keep_inert_switches(attempt, measured, params=params, extra=extra, parity=parity)
        return {**result, SCREEN_PARAM: SCREEN_CYCLE}

    async def _keep_inert_switches(
        self,
        attempt: IntegrateAttempt,
        measured: _PatchMeasurement,
        *,
        params: dict[str, Any],
        extra: dict[str, Any],
        parity: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep a correct-but-unprofitable rewrite patch dormant and register its levers.

        The bundle passed correctness but not the throughput threshold. Because
        every rewrite is behind a switch that defaults OFF, the applied code is
        inert: leaving it in place costs nothing at runtime, while reverting it
        would discard the rewrites that do pay along with the one that does not,
        and would discard any enabler whose whole purpose is to make another
        rewrite profitable rather than to be profitable itself.

        So the code stays, committed like any KEEP so the next attempt's revert to
        HEAD keeps it, and the switches are registered as search levers, with
        ``extra_envs_applied`` deliberately empty so nothing enters the running
        configuration. The explore phase then turns them on one dependency-closed
        bundle at a time.

        Args:
            attempt: The attempt whose applied patches, artifacts and switch
                manifest are kept.
            measured: The bench with the switches on against the comparison
                base, the throughput threshold it did not meet, its accuracy
                verdict, and the accuracy delta and config fingerprint for the
                KB record.
            params: The task params.
            extra: The runner's extra context.
            parity: The switch-off parity verdict, the inert KEEP's evidence of being inert.

        Returns:
            The ``kept_inert`` result envelope.
        """
        delta_pct = measured.delta_pct
        switch_manifest = attempt.switch_manifest
        commit_failure, _ = self._commit_accepted_work(attempt, f"hyperloom KEEP_INERT {attempt.specialist_task_id}")
        if commit_failure:
            return _uncommitted_revert(
                attempt,
                commit_failure,
                "KEEP_INERT",
                output_throughput=measured.new_tput,
                delta_pct=delta_pct,
                bench_result=measured.bench_result,
            )
        from ...knowledge import kb_writeback as _kb

        enablers = [entry["switch"] for entry in switch_manifest if entry.get("enabler")]
        if params.get(SCREEN_PARAM) != SCREEN_CYCLE:
            await self._maybe_write_framework_kb_record(
                measured.lesson(_kb.OUTCOME_KEPT_INERT),
                done_payload=attempt.done_payload,
                params=params,
                extra=extra,
            )
        log.info(
            "integrate_patch: KEEP_INERT task=%s delta=%s threshold=%.2f%% levers=%d enablers=%d",
            attempt.specialist_task_id,
            f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a",
            measured.keep_threshold_pct,
            len(switch_manifest),
            len(enablers),
        )
        return {
            "status": "kept_inert",
            # True when the bundle moved the output with every switch on. The
            # code is still kept, because the switches are benched together and
            # that verdict does not say which one is at fault — explore bisects
            # per lever from here. The flag exists so nothing downstream reads
            # this as a clean keep.
            "quality_unverified": measured.accuracy_pass is False,
            "specialist_task_id": attempt.specialist_task_id,
            "patches_applied": [str(p) for p in attempt.applied],
            "patches_reverted": [],
            "artifacts_applied": attempt.applied_artifacts,
            # Empty on purpose: the code is present but dormant, so nothing
            # may enter current_best. The levers below are how it gets turned
            # on, one measured bundle at a time.
            "extra_server_args_applied": "",
            "extra_envs_applied": {},
            **measured.envelope(),
            "reason": _inert_keep_reason(
                delta_pct, measured.keep_threshold_pct, len(attempt.applied), switch_manifest, enablers
            ),
            "workspace": str(attempt.output_root),
            "framework_root": str(attempt.framework_root or ""),
            "framework_levers": switch_manifest,
            "framework_lever_outcome": "registered_off",
            "switch_off_parity": parity,
        }
