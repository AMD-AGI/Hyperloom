# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply one discovered framework PR candidate and KEEP or REVERT by benchmark."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from hyperloom.common.env import is_truthy
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ._accuracy_gate import (
    accuracy_keep_block,
    accuracy_passed,
    parse_eval_results,
    require_framework_accuracy_default,
)
from ._patch_source_pr import (
    DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    materialize_candidate_patches,
    _candidate_slug,
    _git_head_sha,
)
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _num_gpus_for_config,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from ._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)
from .integrate_patch import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_VARIANT_TIMEOUT_SEC,
    _accuracy_delta_pct,
    _git_apply_collect_feedback,
    _git_stash_if_dirty,
    _restore_stash_logged,
    _with_stash_restore,
    _resolve_framework_root,
)
from ._apply_feedback import ApplyFeedback
from ._nogit_patch import (
    _apply_patch_no_git,
    _is_git_tree,
    _revert_patches_no_git,
)
from ._patch_snapshot import (
    _create_patch_snapshot,
    _git_commit_kept,
    _patch_touched_paths,
    _restore_patch_snapshot,
)
from ...knowledge.kb_writeback import (
    OUTCOME_INTEGRATED,
    OUTCOME_REVERTED_SMOKE_FAIL,
    write_framework_record,
)
from ...state.shared_state import resolve_anchor_with_drift
from hyperloom.common.gain_math import gain_pct


log = logging.getLogger(__name__)


class FrameworkAgentExecutor:
    """ActionRunner for the ``framework`` action (FRAMEWORK_AGENT phase)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
        diff_fetch_timeout_sec: float = DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    ):
        """Initialize the executor with session + bench defaults.

        Args:
            session_dir (Path | str | None): The session directory used to
                root per-task workspaces; resolved from the environment when
                ``None``.
            default_config_path (Path | str | None): Default baseline config
                path used when ``params.config_path`` is absent.
            variant_timeout_sec (int): Default per-variant bench timeout.
            keep_threshold_pct (float): Default throughput delta (percent)
                required to KEEP a candidate.
            diff_fetch_timeout_sec (float): Default timeout for fetching /
                materializing the candidate diff.
        """
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        self.diff_fetch_timeout_sec = float(diff_fetch_timeout_sec)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Fetch, apply, bench and KEEP/REVERT one approved PR candidate.

        Resolves the patch source (explicit paths, checkout-head worktree
        diff, or curled ``diff_url``), snapshots the live tree HEAD, applies
        the diff, optionally benches it via :meth:`_bench_candidate`, and
        commits (KEEP) or ``git reset --hard`` reverts based on the throughput
        delta and accuracy gate.

        Args:
            ctx: The runner context carrying ``task.params`` (the ``candidate``
                row and bench knobs) and ``extra`` (workspace / shared_state).

        Returns:
            dict[str, Any]: A result dict whose ``status`` is one of ``kept``,
                ``reverted``, ``accuracy_unavailable_reject`` (accuracy gate
                required but never evaluated, so the patch was reverted despite
                an acceptable throughput delta), ``apply_failed``, ``no_patch``,
                ``fetch_failed``, ``applied_no_bench``, ``skipped`` (multi-node
                mode; no patch applied and no failure tallied, see
                ``skipped_reason``) or ``failed``, plus throughput / accuracy
                / patch bookkeeping fields.
        """
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": ("framework requires params.candidate (the PR metadata row produced by `fa phase-discover`)"),
            }
        batch_id = str(params.get("batch_id") or "")
        slug = _candidate_slug(candidate)

        # Multi-node guard (mirrors integrate_patch). This executor git-applies
        # the candidate PR ONLY to the sandbox framework_source_root; in
        # multi-node mode the live sglang/vllm runs on RayJob/Infera pods, not
        # the sandbox, so a sandbox-only apply would NOT reach pod-side serving
        # and the bench would measure the unpatched pod (meaningless KEEP/REVERT
        # verdict). Until a git-diff pod fan-out exists, return a NEUTRAL
        # "skipped" result (no patch touched, no error) so no failure tally is
        # rolled and every other action keeps running. is_multi_node() is False
        # single-node, so the normal path below is reached unchanged.
        from ._multi_node_env import is_multi_node

        if is_multi_node():
            return {
                "status": "skipped",
                "skipped_reason": "multi_node_unsupported",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "reason": (
                    "framework-agent candidate integration is not supported "
                    "in multi-node mode (no git-diff pod fan-out); skipped "
                    "without applying any patch. Other actions "
                    "(baseline/profile/explore/sweep/roofline) continue "
                    "normally. Use the kernel-agent integrate path (which "
                    "fans out via `multi_node apply-patch`) or run single-node."
                ),
            }

        # Per-task workspace under runs/framework_agent/<task_id>/.
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "framework_agent", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        explicit_framework_root = str(params.get("framework_source_root") or "").strip() or None
        framework_root = _resolve_framework_root(explicit_framework_root)
        if framework_root is None:
            if explicit_framework_root:
                _error_class = "framework_source_root_rejected"
                _error = (
                    f"framework_source_root {explicit_framework_root!r} is not under the configured source allowlist"
                )
            else:
                _error_class = "no_framework_agent_root"
                _error = (
                    "no framework_source_root resolved; cannot apply "
                    "candidate PR. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                )
            return {
                "status": "apply_failed",
                "error_class": _error_class,
                "error": _error,
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "workspace": str(output_root),
            }

        # Materialise the candidate into local patch files. Every failure is a
        # terminal verdict rather than an empty list: benching a tree no patch
        # reached measures the baseline and reports it as the candidate's.
        materialized = materialize_candidate_patches(
            candidate=candidate,
            params=params,
            framework_root=framework_root,
            output_root=output_root,
            slug=slug,
            diff_fetch_timeout_sec=self.diff_fetch_timeout_sec,
        )
        patch_source_mode = materialized.mode
        if materialized.failure is not None:
            return {
                **materialized.failure,
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "patch_source_mode": patch_source_mode,
                "workspace": str(output_root),
            }
        patch_paths: list[Path] = list(materialized.patches)

        # Preserve user's uncommitted changes BEFORE applying the candidate so
        # the stash holds only user state and `git stash pop` restores it cleanly.
        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            log.error(
                "framework: cannot stash user changes in %s: %s; aborting candidate to avoid data loss",
                framework_root,
                stash_note,
            )
            return {
                "status": "apply_failed",
                "error_class": "stash_failed",
                "error": f"refusing to proceed: user changes could not be stashed ({stash_note})",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "workspace": str(output_root),
            }

        git_tree = _is_git_tree(framework_root)
        self._nogit_patch_backups: list[dict] = []
        self._git_snapshot_manifest: dict | None = None

        # Stage 1: apply patches (with -3 fallback for git trees;
        # backup-based apply for non-git roots like pip wheel installs).
        applied: list[Path] = []
        apply_errors: list[dict[str, str]] = []
        apply_feedbacks: list[ApplyFeedback] = []

        def _undo_candidate() -> None:
            """Take the candidate back out of the tree and hand the stash back.

            What a stop owes, as opposed to a verdict. The dispatcher cancels
            in-flight actions on shutdown and on a spent wall-clock budget, and
            ``CancelledError`` is not an ``Exception``, so none of the REVERT
            handlers below see one. Unhandled it leaves the candidate applied and
            the operator's uncommitted work in ``git stash`` indefinitely -- the
            budget case does not end the process, so CLOSE would go on to report
            against a tree carrying a patch nothing ever graded.

            Reverting past a KEEP that was already committed is deliberate: the
            result carrying that KEEP never reaches the Coordinator, so leaving
            the commit would leave the tree claiming a win the session does not
            record. Every step is synchronous, so no second cancel can be
            delivered part-way through the undo.
            """
            self._revert_patches(framework_root, applied)
            _restore_stash_logged(framework_root, stash_state, stash_note)

        # Snapshot the patch-touched paths before any mutation so REVERT/REJECT
        # restores exactly those paths and leaves unrelated work in place.
        if git_tree:
            try:
                self._git_snapshot_manifest = _create_patch_snapshot(
                    str(framework_root),
                    [Path(p).read_text(encoding="utf-8", errors="replace") for p in patch_paths],
                    output_root,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "apply_failed",
                        "error_class": "snapshot_failed",
                        "error": f"could not snapshot patch targets in {framework_root}: {exc}",
                        "candidate": candidate,
                        "batch_id": batch_id,
                        "patches_applied": [],
                        "patches_reverted": [],
                        "workspace": str(output_root),
                    },
                )

        # Structural safety gate on the (remote / untrusted) diff before it is
        # applied to the live framework tree: reject non-diff blobs and any
        # header path that escapes the tree (absolute / ``..``). Stale /
        # missing-target diffs are left to git apply's own check so a legitimate
        # candidate is never dropped here.
        from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree

        for patch in patch_paths:
            try:
                _ptext = Path(patch).read_text(encoding="utf-8", errors="replace")
            except OSError as _exc:
                apply_errors.append({"patch": str(patch), "stderr": f"unreadable: {_exc!r}"})
                break
            if not is_unified_diff(_ptext):
                apply_errors.append({"patch": str(patch), "stderr": "not a unified diff"})
                break
            _escape = patch_escapes_tree(_ptext)
            if _escape is not None:
                apply_errors.append({"patch": str(patch), "stderr": f"path escapes tree: {_escape!r}"})
                break
            if git_tree:
                ok, err, fb = _git_apply_collect_feedback(framework_root, patch, three_way=False)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            else:
                nogit_backup_root = output_root / "patch_backups"
                ok, err, backups, fb = _apply_patch_no_git(
                    framework_root,
                    patch,
                    nogit_backup_root,
                    seq_offset=len(self._nogit_patch_backups),
                )
                self._nogit_patch_backups.extend(backups)
                if not ok:
                    apply_errors.append({"patch": str(patch), "stderr": err})
                    if fb is not None:
                        apply_feedbacks.append(fb)
                    break
            applied.append(patch)
        if apply_errors:
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "apply_failed",
                    "error_class": "git_apply_failed",
                    "error": apply_errors,
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "patch_source_mode": patch_source_mode,
                    "reason": "git apply failed (see error)",
                    "workspace": str(output_root),
                    "lane": "perf_framework",
                    "retry_feedback": [fb.to_dict() for fb in apply_feedbacks],
                    "prior_patches": [str(p) for p in patch_paths],
                },
            )

        if params.get("apply_only"):
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "applied_no_bench",
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [str(p) for p in applied],
                    "patches_reverted": [],
                    "patch_source_mode": patch_source_mode,
                    "reason": "apply_only=True; benchmark skipped",
                    "workspace": str(output_root),
                },
            )

        # Bench via run_grid (size=1). Bound it by the session wall-clock, as the
        # sweep and explore arms already are: without it the declared cap is the
        # only limit, and that cap answers "how long before this counts as hung",
        # not "how much budget is left" -- so a candidate benched near the end of
        # a run could outlive the run itself.
        session_deadline_sec, variant_expected_sec = session_grid_bounds(
            extra.get("shared_state") or extra.get("state")
        )
        try:
            bench_result, gate_evidence = await self._bench_candidate(
                params=params,
                output_root=output_root,
                slug=slug,
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
                state_model_path=str(getattr(extra.get("shared_state") or extra.get("state"), "model_path", "") or ""),
            )
        except FrameworkScriptMismatchError as exc:
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "error_class": "framework_script_mismatch",
                    "error": str(exc),
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "reason": str(exc),
                    "workspace": str(output_root),
                },
            )
        except Exception as exc:  # noqa: BLE001
            reverted = self._revert_patches(framework_root, applied)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": "reverted",
                    "error_class": "bench_exception",
                    "error": repr(exc),
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "reason": f"bench raised: {exc!r}",
                    "workspace": str(output_root),
                },
            )
        except BaseException:
            # A stop, not a verdict: let it through rather than grading it, since
            # as a REVERT it would read as the patch having failed a bench that
            # never ran. See :func:`_undo_candidate` for what the stop owes.
            _undo_candidate()
            raise

        # KEEP / REVERT decision.
        base_tput, anchor_drifted = resolve_anchor_with_drift(
            float(params.get("base_tput") or 0.0),
            extra.get("shared_state") or extra.get("state"),
        )
        if anchor_drifted:
            log.warning("framework: anchor drift; grading against live anchor %.1f", base_tput)
        keep_threshold_pct = float(
            params.get("keep_threshold_pct", self.keep_threshold_pct),
        )
        new_tput = bench_result.get("output_throughput")
        delta_pct = gain_pct(new_tput, base_tput)

        accuracy_pass = gate_evidence.get("accuracy_pass")
        # Source patches require the accuracy gate for a KEEP: a measured
        # regression always blocks; a missing verdict blocks only when a
        # baseline accuracy was available (else degrade to throughput-only).
        acc_required = bool(params.get("require_accuracy_for_keep", require_framework_accuracy_default()))
        acc_block, acc_reason, acc_degraded = accuracy_keep_block(
            accuracy_pass,
            required=acc_required,
            baseline_accuracy=params.get("accuracy_baseline"),
        )
        if acc_degraded:
            log.warning(
                "framework: accuracy gate required but no baseline accuracy; "
                "KEEP allowed on throughput only (candidate=%s)",
                slug,
            )
        tput_ok = delta_pct is not None and delta_pct >= keep_threshold_pct
        gate_pass = tput_ok and not acc_block
        acc_delta_pct = _accuracy_delta_pct(gate_evidence.get("accuracy"), params.get("accuracy_baseline"))

        async def _record_outcome(outcome: str) -> None:
            """Write this candidate's KB record, undoing it if the write is stopped.

            Both verdicts record the same measurements and differ only in the
            outcome label, and both record them after the verdict is decided and
            before the ``_with_stash_restore`` that returns it. That await is the
            last one the candidate crosses while the auto-stash is still on the
            stack, and no handler stands between it and the caller: a cancel
            delivered here -- which is what a spent budget delivers, at whatever
            await the action happens to be at -- would strand the operator's work
            in the stash with nothing in the session saying so.

            Args:
                outcome: The KB outcome label for the verdict just decided.
            """
            try:
                await self._write_kb_record(
                    candidate=candidate,
                    outcome=outcome,
                    tps_delta_pct=float(delta_pct or 0.0),
                    patch_path=str(applied[0]) if applied else "",
                    extra=extra,
                    accuracy_delta_pct=acc_delta_pct,
                )
            except BaseException:
                _undo_candidate()
                raise

        if not gate_pass:
            reverted = self._revert_patches(framework_root, applied)
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(f"throughput delta {delta_pct:+.2f}% < keep_threshold {keep_threshold_pct:.2f}%")
            if acc_block and acc_reason:
                reasons.append(acc_reason)
            # Distinguish "accuracy required but unevaluated" (None, not a
            # measured regression) from a throughput/regression revert.
            revert_status = (
                "accuracy_unavailable_reject" if (acc_block and accuracy_pass is None and tput_ok) else "reverted"
            )
            await _record_outcome(OUTCOME_REVERTED_SMOKE_FAIL)
            return _with_stash_restore(
                framework_root,
                stash_state,
                stash_note,
                {
                    "status": revert_status,
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "output_throughput": new_tput,
                    "delta_pct": delta_pct,
                    "accuracy_pass": accuracy_pass,
                    "base_tput": base_tput,
                    "keep_threshold_pct": keep_threshold_pct,
                    "patch_source_mode": patch_source_mode,
                    "reason": "; ".join(reasons) or "gate failed",
                    "bench_result": bench_result,
                    "workspace": str(output_root),
                },
            )

        # KEEP: commit the patch-touched paths so they survive the next
        # candidate's REJECT. Non-git trees keep them as working-tree edits.
        keep_message = f"framework KEEP {slug}"
        keep_sha: str | None = None
        if git_tree:
            touched_paths = _patch_touched_paths(framework_root, list(applied))
            commit_ok, commit_err = _git_commit_kept(framework_root, keep_message, touched_paths)
            if commit_ok:
                keep_sha, _ = _git_head_sha(framework_root)
            else:
                reverted = self._revert_patches(framework_root, applied)
                return _with_stash_restore(
                    framework_root,
                    stash_state,
                    stash_note,
                    {
                        "status": "apply_failed",
                        "error_class": "keep_commit_failed",
                        "error": commit_err or "git commit failed",
                        "candidate": candidate,
                        "batch_id": batch_id,
                        "patches_applied": [],
                        "patches_reverted": [str(p) for p in reverted],
                        "output_throughput": new_tput,
                        "delta_pct": delta_pct,
                        "accuracy_pass": accuracy_pass,
                        "base_tput": base_tput,
                        "keep_threshold_pct": keep_threshold_pct,
                        "reason": f"KEEP commit failed: {commit_err}",
                        "bench_result": bench_result,
                        "workspace": str(output_root),
                    },
                )

        await _record_outcome(OUTCOME_INTEGRATED)
        return _with_stash_restore(
            framework_root,
            stash_state,
            stash_note,
            {
                "status": "kept",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "keep_commit_sha": keep_sha,
                "patch_source_mode": patch_source_mode,
                "reason": (f"throughput delta {delta_pct:+.2f}% >= {keep_threshold_pct:.2f}%"),
                "bench_result": bench_result,
                "workspace": str(output_root),
            },
        )

    # KB writeback: append FRAMEWORK outcome to lessons.jsonl
    async def _write_kb_record(
        self,
        *,
        candidate: dict[str, Any],
        outcome: str,
        tps_delta_pct: float,
        patch_path: str,
        extra: dict[str, Any],
        accuracy_delta_pct: float | None = None,
    ) -> None:
        """Append a FRAMEWORK outcome to ``lessons.jsonl`` so the next
        ``fa phase-discover`` can dedup integrated PRs.

        Best-effort: candidates lacking both ``pr_url`` and ``head_sha``
        (no dedup key) are skipped; write errors are logged + swallowed.

        Args:
            candidate: The PR metadata row (provides the dedup key).
            outcome: The outcome label to record (e.g. integrated / reverted).
            tps_delta_pct: The measured throughput delta percentage.
            patch_path: Path to the applied patch (for provenance).
            extra: The runner ``extra`` mapping (provides the shared state /
                session id).
            accuracy_delta_pct: Measured accuracy delta, when available.
        """
        pr_url = str(candidate.get("pr_url") or "").strip()
        pr_sha = str(candidate.get("head_sha") or "").strip()
        if not pr_url and not pr_sha:
            log.warning(
                "framework: candidate lacks pr_url/head_sha; KB writeback skipped",
            )
            return
        session_id = ""
        ss = extra.get("shared_state") or extra.get("state")
        if ss is not None:
            session_id = str(getattr(ss, "recipe_kb_session_id", "") or "")
        try:
            gap_keywords = candidate.get("gap_keywords") or []
            if isinstance(gap_keywords, str):
                gap_keywords = [gap_keywords]
            changed_files = candidate.get("changed_files") or []
            if isinstance(changed_files, str):
                changed_files = [changed_files]
            written = await write_framework_record(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=patch_path,
                outcome=outcome,
                tps_delta_pct=float(tps_delta_pct),
                session_id=session_id,
                framework=str(candidate.get("framework") or "").strip().lower(),
                gap_canonical_id=str(candidate.get("gap_canonical_id") or "").strip(),
                gap_keywords=[str(k).strip().lower() for k in gap_keywords if str(k).strip()],
                model_class=str(getattr(ss, "model_class", "") if ss is not None else "").strip(),
                gpu_type=str(getattr(ss, "gpu_type", "") if ss is not None else "").strip(),
                precision=str(getattr(ss, "precision", "") if ss is not None else "").strip(),
                applicability=str(candidate.get("applicability") or "").strip(),
                provenance="framework_agent",
                accuracy_delta_pct=float(accuracy_delta_pct or 0.0),
                changed_files=[str(f).strip() for f in changed_files if str(f).strip()],
                session_dir=self.session_dir,
            )
            log.info(
                "framework: wrote KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
                written,
                outcome,
                pr_url,
                float(tps_delta_pct),
            )
        except Exception as exc:  # noqa: BLE001 — KB write is best-effort
            log.warning("framework: KB writeback failed: %r", exc)

    # Helpers
    def _revert_patches(
        self,
        framework_root: Path | None,
        applied: list[Path],
    ) -> list[Path]:
        """Roll back this candidate's changes from the undo ledger.

        Git trees restore the pre-apply snapshot; non-git trees restore the
        per-file backups. Which ledger holds entries — not ``applied`` — decides
        whether a restore is owed: a patch set that fails part-way through its
        first patch has already mutated the tree while ``applied`` is still empty.

        Args:
            framework_root: The source root to revert, or ``None`` (no-op).
            applied: The patches recorded as fully applied; reported back, never
                consulted to decide whether to restore.

        Returns:
            The reverted patches (full ``applied`` on success, ``[]`` on
            failure or no-op).
        """
        if framework_root is None:
            return []
        nogit_backups = getattr(self, "_nogit_patch_backups", None)
        if nogit_backups is not None and not _is_git_tree(framework_root):
            if nogit_backups:
                ok, errors = _revert_patches_no_git(nogit_backups)
                if not ok:
                    log.error("framework: non-git revert incomplete in %s: %s", framework_root, errors)
                    return []
            return list(applied)
        snapshot = getattr(self, "_git_snapshot_manifest", None)
        if not snapshot:
            return []
        result = _restore_patch_snapshot(snapshot)
        if not result["ok"]:
            log.error("framework: snapshot restore incomplete in %s: %s", framework_root, result["errors"])
            return []
        return list(applied)

    async def _bench_candidate(
        self,
        *,
        params: dict[str, Any],
        output_root: Path,
        slug: str,
        session_deadline_sec: float | None = None,
        variant_expected_sec: float | None = None,
        state_model_path: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy
        gate. Mirrors :meth:`IntegratePatchExecutor._bench_patch`.

        Args:
            params: The task params (config / model / bench knobs).
            output_root: The per-task workspace root for the bench.
            slug: The candidate slug used to name the variant.
            session_deadline_sec: Monotonic-clock session budget deadline, or
                ``None`` when unbounded. Resolved by the caller, which owns the
                session context.
            variant_expected_sec: Expected bench runtime used to decide whether
                the remaining budget can fit this bench at all.
            state_model_path: ``SharedState.model_path``, the last fallback in
                the model-path precedence. Passed in because the caller owns the
                session context.

        Returns:
            A ``(bench, gate_evidence)`` tuple: the bench result dict and a
            dict carrying the ``accuracy_pass`` verdict.
        """
        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            raise RuntimeError(f"framework bench: config not found at {config_path}")
        # This bench launches a server, so the value has to be a servable path:
        # the shared resolver walks HL_MODEL_BASE and the hub cache, where the
        # local two-step did not, and handed a bare repo id straight to the
        # server. It falls back to the original string, so an unresolvable
        # value degrades exactly as before rather than emptying.
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=state_model_path,
            for_serving=True,
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        # The session's ``--no-eval`` wins; otherwise force RUN_EVAL=true when the
        # gate is required and a baseline accuracy exists, so a stale config
        # cannot silently disable the eval.
        bench_extra_envs: dict[str, Any] = {}
        acc_required = bool(params.get("require_accuracy_for_keep", require_framework_accuracy_default()))
        try:
            _acc_base = float(params.get("accuracy_baseline") or 0.0)
        except (TypeError, ValueError):
            _acc_base = 0.0
        if is_truthy(params.get("disable_run_eval")):
            bench_extra_envs["RUN_EVAL"] = "false"
        elif acc_required and _acc_base > 0:
            bench_extra_envs["RUN_EVAL"] = "true"
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            extra_envs=bench_extra_envs or None,
            out_name="framework.with_envs.yaml",
        )

        variant = GridVariant(
            name=f"framework-{slug}"[:96],
            extra_server_args=str(params.get("base_extra_args") or "").strip(),
            extra_envs={},
            note=f"framework:{slug}",
        )

        # Ray-managed GPU execution (phase-3 §3.1 / invariant §6.2): the
        # candidate benchmark holds a serving lease (num_gpus=TP + serving_slot)
        # for the whole run_grid, so it serializes against other serving on the
        # whole-machine mutex instead of colliding with a concurrently-running
        # GPU specialist server on the same card. ``None`` keeps the local path
        # (multi-node / RAY_EXEC off / pytest default). Closed right after the
        # grid; the accuracy parse below reads result files and needs no GPU.
        from ._ray_serving import maybe_serving_lease

        serving_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        try:
            results: list[VariantResult] = await run_grid(
                base_yaml_path=config_path,
                base_extra_args=str(params.get("base_extra_args") or "").strip(),
                grid=[variant],
                output_root=output_root,
                magpie_python=params.get("magpie_python") or None,
                variant_timeout_sec=int(
                    params.get("variant_timeout_sec", self.variant_timeout_sec),
                ),
                keep_going_on_failure=False,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                result_dir=override_result_dir,
                serving_lease=serving_lease,
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
        finally:
            if serving_lease is not None:
                serving_lease.close()

        bench: dict[str, Any] = {}
        if results:
            r = results[0]
            bench = {
                "name": r.name,
                "status": r.status,
                "output_throughput": getattr(r, "output_throughput", None),
                # ``VariantResult`` names these ``ttft_mean_ms`` / ``tpot_mean_ms``;
                # the emitted keys stay ``ttft_ms`` / ``itl_ms`` because that is
                # what the breakdown collectors read.
                "ttft_ms": r.ttft_mean_ms,
                "itl_ms": r.tpot_mean_ms,
                # Benchmark dir; the accuracy grade below locates eval artifacts
                # from its parent (the grid slot).
                "workspace": str(getattr(r, "workspace", "") or ""),
                "error": getattr(r, "error", "") or "",
                "error_class": getattr(r, "error_class", "") or "",
                "nonfatal_warnings": list(getattr(r, "nonfatal_warnings", []) or []),
            }

        accuracy_pass: bool | None = None
        measured_accuracy: float | None = None
        baseline_accuracy = params.get("accuracy_baseline")
        # lm-eval writes to ``$EVAL_RESULT_DIR`` under the grid slot, not inside
        # the ``benchmark_*`` workspace, so grade from the slot while honoring an
        # explicit ``result_dir`` override the same way the grid subprocess does.
        # An empty root would resolve to ``Path(".")`` and silently search the
        # process CWD, which reads back as "no eval result" and blocks every KEEP.
        eval_search_root = override_result_dir or (
            str(Path(bench["workspace"]).parent) if bench.get("workspace") else ""
        )
        if (
            bench.get("status") == "succeeded"
            and eval_search_root
            and isinstance(baseline_accuracy, (int, float))
            and float(baseline_accuracy) > 0
        ):
            try:
                eval_results = parse_eval_results(
                    eval_search_root,
                    framework=params.get("framework") or os.environ.get("FRAMEWORK") or None,
                )
                # Read ``accuracy`` and pass (baseline, new) in the order
                # accuracy_passed expects; the framework hint lets scriptable
                # (xDiT) quality-gate reports resolve onto the accuracy contract.
                new_accuracy = eval_results.get("accuracy")
                if new_accuracy is not None:
                    measured_accuracy = float(new_accuracy)
                    accuracy_pass = accuracy_passed(
                        float(baseline_accuracy),
                        measured_accuracy,
                    )
            except Exception:  # noqa: BLE001
                log.exception("framework: accuracy gate parse failed; treating as None (gate skipped)")

        # ``accuracy`` carries the raw measurement for the KB record; the caller
        # reads it for ``accuracy_delta_pct`` and would otherwise always see None.
        return bench, {"accuracy_pass": accuracy_pass, "accuracy": measured_accuracy}


framework_agent_executor = FrameworkAgentExecutor


__all__ = [
    "DEFAULT_DIFF_FETCH_TIMEOUT_SEC",
    "FrameworkAgentExecutor",
    "framework_agent_executor",
]
