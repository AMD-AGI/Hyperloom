"""``framework_pr`` arm executor — fa-driven PR discovery + A/B + rollback.

Replaces the legacy CLI pre-stage hook (``--framework-pr-discover``) with
a regular bandit arm. Each tick the executor runs the following loop:

    1. Compose ``(gap, keywords)`` from SharedState + manifest
       (operator overrides supported via task.params).
    2. Call ``fa candidates`` (read-only) to enumerate top-K PRs.
    3. Filter candidates: drop refs whose head_sha already matches the
       current sglang HEAD (= already applied / KEEP'd this session).
    4. Stash current HEAD; ``git fetch`` + ``git checkout`` the PR head.
    5. Run a sub-baseline benchmark against ``base_tput`` using the same
       workload contract (``config_path`` + ``base_extra_args``).
    6. If new_tput >= base_tput * (1 + min_gain_pct/100): leave applied,
       return ``status=succeeded`` with ``output_throughput`` so the
       Coordinator promotes the PR into ``current_best``.
    7. Else: ``git checkout --detach <prev_head>`` to undo, return
       ``status=succeeded`` with ``decision=discarded`` so the bandit
       arm records a no_promote on the streak counter.

Failure modes (fa subprocess error, git failure, sub-baseline timeout)
return ``status=failed`` with an ``error_class`` so the Coordinator's
:meth:`_apply_action_score_update` fallback streak-penalises the arm
without aborting the session.

All git side effects are confined to ``/sgl-workspace/sglang`` and are
fully reverted on DISCARD; the executor never modifies state on
:exc:`Exception` after a successful checkout (the rollback runs in a
``finally`` block guarded by ``prev_head_sha`` capture).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ..framework_pr_discover import (
    FrameworkPRError,
    apply_to_sglang,
    current_head_sha,
    enumerate_candidates_via_fa,
    rollback_to,
)
from ..sub_agent_runner import RunnerContext
from ._framework_gap_composer import compose_gap
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


_DEFAULT_MAX_CANDIDATES = 5
_DEFAULT_MIN_GAIN_PCT = 1.0
_DEFAULT_REPO_URL = "https://github.com/sgl-project/sglang"
_DEFAULT_SGLANG_PATH = Path("/sgl-workspace/sglang")


def _short(sha: str | None) -> str:
    """12-char SHA prefix or empty string — for log readability."""
    s = (sha or "").strip()
    return s[:12] if s else ""


def _pick_candidate(
    candidates: list[dict[str, Any]],
    *,
    current_head_sha: str,
) -> tuple[dict[str, Any] | None, str]:
    """Pick the first applicable candidate; return ``(cand, skip_reason)``.

    Rules:
      * Skip candidates whose ``head_sha`` exactly matches the current
        sglang HEAD — those are already applied (= prior KEEP this
        session, or the base image already shipped them).
      * Skip candidates whose ``ref`` does not start with ``PR:`` (the
        apply path only supports ``refs/pull/N/head``; tag / branch refs
        would need a separate code path).
      * Otherwise return the first survivor.

    Returns ``(None, reason)`` if every candidate was filtered out so
    the caller can surface ``error_class=no_applicable_candidate``.
    """
    cur = (current_head_sha or "").strip().lower()
    skipped: list[str] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        ref = str(cand.get("ref") or "").strip()
        if not ref.startswith("PR:"):
            skipped.append(f"{ref or '(no ref)'}=non_pr_ref")
            continue
        head = str(cand.get("head_sha") or "").strip().lower()
        if cur and head and head == cur:
            skipped.append(f"{ref}=head_eq_current")
            continue
        return cand, ""
    return None, (
        "all candidates filtered out"
        + (": " + ", ".join(skipped) if skipped else "")
    )


class FrameworkPRExecutor:
    """Stateless arm executor for the framework_pr bandit row.

    Instantiated once at boot (see ``cli.py:_REAL_EXECUTORS_FULL``) and
    invoked per Task by :class:`SubAgentRunner`. The executor itself
    holds no per-session state; all context (base_tput / config_path /
    framework / etc.) flows through ``task.params`` plumbed by
    :meth:`Coordinator._materialize_approved_proposal`.

    ``baseline_executor`` is injected so tests can stub it with a fake
    that returns synthetic ``output_throughput`` without actually
    launching Magpie.
    """

    def __init__(
        self,
        *,
        baseline_executor: BaselineExecutor | None = None,
        sglang_path: Path | str | None = None,
        repo_url: str = _DEFAULT_REPO_URL,
        session_dir: Path | str | None = None,
    ) -> None:
        self.baseline_executor = baseline_executor or BaselineExecutor()
        self.sglang_path = Path(sglang_path) if sglang_path else _DEFAULT_SGLANG_PATH
        self.repo_url = repo_url
        from ._grid_runner import _resolve_session_dir
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )

    def _resolve_workspace(self, ctx: RunnerContext) -> Path:
        """Pick the per-task workspace dir, mirroring BaselineExecutor."""
        params = ctx.task.params or {}
        if params.get("output_dir"):
            return Path(params["output_dir"])
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("workspace"):
            return Path(extra["workspace"])
        return runs_dir(self.session_dir, "framework_pr", ctx.task.task_id)

    async def _run_sub_baseline(
        self,
        *,
        ctx: RunnerContext,
        workspace: Path,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Launch a one-shot Magpie benchmark against the patched source.

        Re-uses :class:`BaselineExecutor` so the cold-start probe + leak
        harvest + accuracy gate behaviour are exactly what ``baseline``
        action does. Returns the executor's raw result dict so the
        framework_pr arm can lift ``output_throughput`` / ``accuracy``
        into its own result without re-implementing the parser.
        """
        from ..task_registry import Task
        sub_workspace = workspace / "pr_baseline"
        sub_workspace.mkdir(parents=True, exist_ok=True)
        sub_params: dict[str, Any] = {
            "output_dir": str(sub_workspace),
        }
        for key in (
            "config_path", "model_path", "gpu_type", "timeout_sec",
            "benchmark_script", "result_dir",
        ):
            if params.get(key) is not None:
                sub_params[key] = params[key]
        extra_args = str(params.get("base_extra_args") or "").strip()
        if extra_args:
            sub_params["extra_sglang_args"] = extra_args
        extra_envs = params.get("extra_envs") or {}
        if isinstance(extra_envs, dict) and extra_envs:
            sub_params["extra_envs"] = dict(extra_envs)
        sub_task = Task(
            task_id=f"{ctx.task.task_id}-prbench",
            kind="baseline",
            state="running",
            params=sub_params,
            idempotency_key=f"{ctx.task.task_id}-prbench",
        )
        sub_ctx = RunnerContext(
            task=sub_task, lease=ctx.lease,
            extra={"workspace": str(sub_workspace)},
        )
        return await self.baseline_executor(sub_ctx)

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:  # noqa: C901
        params = ctx.task.params or {}
        workspace = self._resolve_workspace(ctx)
        workspace.mkdir(parents=True, exist_ok=True)

        base_tput = float(params.get("base_tput") or 0.0)
        min_gain_pct = float(params.get("min_gain_pct") or _DEFAULT_MIN_GAIN_PCT)
        max_candidates = int(params.get("max_candidates") or _DEFAULT_MAX_CANDIDATES)
        dry_run = bool(params.get("dry_run") or False)
        framework = str(params.get("framework") or "sglang").strip() or "sglang"
        if framework != "sglang":
            return {
                "status": "failed",
                "error_class": "unsupported_framework",
                "error": (
                    f"framework_pr arm only supports framework=sglang "
                    f"(got {framework!r}); vllm path is not implemented."
                ),
                "workspace": str(workspace),
            }
        if base_tput <= 0:
            return {
                "status": "failed",
                "error_class": "no_base_tput",
                "error": (
                    "framework_pr requires base_tput > 0 (Coordinator "
                    "should inject baseline_tput / current_best.tput). "
                    "Run `baseline` first."
                ),
                "workspace": str(workspace),
            }

        gap_override = str(params.get("gap_override") or "").strip()
        kw_override = params.get("keyword_override") or []
        if isinstance(kw_override, str):
            kw_override = [kw_override]
        kw_override = [str(k).strip().lower() for k in kw_override if str(k).strip()]

        if gap_override:
            gap = gap_override
            keywords = kw_override
        elif kw_override:
            gap, _composed_kw = compose_gap(
                framework=framework,
                gpu_type=str(params.get("gpu_type") or ""),
                model_class=str(params.get("model_class") or ""),
                precision=str(params.get("precision") or ""),
                profile_kernel_breakdown_path=params.get(
                    "last_profile_kernel_breakdown"
                ),
            )
            keywords = kw_override
        else:
            gap, keywords = compose_gap(
                framework=framework,
                gpu_type=str(params.get("gpu_type") or ""),
                model_class=str(params.get("model_class") or ""),
                precision=str(params.get("precision") or ""),
                profile_kernel_breakdown_path=params.get(
                    "last_profile_kernel_breakdown"
                ),
            )

        repo_url = str(params.get("repo_url") or self.repo_url).strip() or self.repo_url

        log.info(
            "framework_pr_executor: tick start base_tput=%.1f min_gain_pct=%.2f "
            "max_candidates=%d gap=%r keywords=%r dry_run=%s",
            base_tput, min_gain_pct, max_candidates, gap, keywords, dry_run,
        )

        fa_work_dir = workspace / "fa_candidates"
        try:
            candidates = enumerate_candidates_via_fa(
                gap_description=gap,
                repo_url=repo_url,
                framework=framework,
                work_dir=fa_work_dir,
                max_candidates=max(1, max_candidates),
                keywords=keywords or None,
            )
        except FrameworkPRError as exc:
            log.warning("framework_pr_executor: fa candidates failed: %s", exc)
            return {
                "status": "failed",
                "error_class": "fa_candidates_failed",
                "error": str(exc),
                "gap": gap, "keywords": keywords,
                "workspace": str(workspace),
            }

        prev_head_sha = current_head_sha(self.sglang_path)
        if not prev_head_sha:
            return {
                "status": "failed",
                "error_class": "no_prev_head",
                "error": (
                    f"could not read git HEAD of {self.sglang_path}; cannot "
                    "safely apply PR without a rollback target."
                ),
                "workspace": str(workspace),
            }

        cand, skip_reason = _pick_candidate(
            candidates, current_head_sha=prev_head_sha,
        )
        if cand is None:
            log.info(
                "framework_pr_executor: no applicable candidate (%s); "
                "returning no_applicable_candidate",
                skip_reason,
            )
            return {
                "status": "failed",
                "error_class": "no_applicable_candidate",
                "error": skip_reason or "no candidates",
                "candidates_seen": len(candidates),
                "candidate_refs": [
                    str(c.get("ref") or "")
                    for c in candidates if isinstance(c, dict)
                ],
                "gap": gap, "keywords": keywords,
                "prev_head_sha": prev_head_sha,
                "workspace": str(workspace),
            }

        winner_ref = str(cand.get("ref") or "")
        winner_head_sha = str(cand.get("head_sha") or "").strip()
        winner_score = cand.get("score")
        winner_title = str(cand.get("title") or "")

        if dry_run:
            log.info(
                "framework_pr_executor: dry_run=true — would apply %s "
                "(head=%s score=%s); skipping checkout + bench",
                winner_ref, _short(winner_head_sha), winner_score,
            )
            return {
                "status": "succeeded",
                "decision": "dry_run",
                "applied_ref": "",
                "selected_ref": winner_ref,
                "selected_head_sha": winner_head_sha,
                "selected_score": winner_score,
                "selected_title": winner_title,
                "candidates_seen": len(candidates),
                "prev_head_sha": prev_head_sha,
                "gap": gap, "keywords": keywords,
                "workspace": str(workspace),
            }

        from ..framework_pr_discover import _parse_pr_number
        try:
            pr_number = _parse_pr_number(winner_ref)
        except FrameworkPRError as exc:
            return {
                "status": "failed",
                "error_class": "bad_pr_ref",
                "error": str(exc),
                "selected_ref": winner_ref,
                "prev_head_sha": prev_head_sha,
                "workspace": str(workspace),
            }

        apply_started_unix = time.time()
        applied = False
        try:
            apply_to_sglang(
                winner_head_sha,
                pr_number=pr_number,
                sglang_path=self.sglang_path,
                pip_reinstall=False,
                auto_stash=True,
            )
            applied = True
        except FrameworkPRError as exc:
            log.warning(
                "framework_pr_executor: apply_to_sglang failed for %s: %s",
                winner_ref, exc,
            )
            return {
                "status": "failed",
                "error_class": "apply_failed",
                "error": str(exc),
                "selected_ref": winner_ref,
                "selected_head_sha": winner_head_sha,
                "prev_head_sha": prev_head_sha,
                "apply_elapsed_sec": time.time() - apply_started_unix,
                "workspace": str(workspace),
            }

        new_head_sha = current_head_sha(self.sglang_path)
        log.info(
            "framework_pr_executor: applied %s (head %s -> %s); running sub-baseline",
            winner_ref, _short(prev_head_sha), _short(new_head_sha),
        )

        sub_result: dict[str, Any] = {}
        bench_ok = False
        new_tput: float = 0.0
        accuracy: float | None = None
        try:
            sub_result = await self._run_sub_baseline(
                ctx=ctx, workspace=workspace, params=params,
            )
            if isinstance(sub_result, dict) and sub_result.get("status") == "succeeded":
                tput = sub_result.get("output_throughput")
                if isinstance(tput, (int, float)) and tput > 0:
                    new_tput = float(tput)
                    bench_ok = True
                acc = sub_result.get("accuracy")
                if isinstance(acc, (int, float)):
                    accuracy = float(acc)
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "framework_pr_executor: sub-baseline raised on %s: %s",
                winner_ref, exc,
            )
            sub_result = {
                "status": "failed", "error_class": "sub_baseline_exception",
                "error": str(exc),
            }

        if not bench_ok:
            err = (
                str(sub_result.get("error") or "")
                if isinstance(sub_result, dict) else ""
            )
            err_class = (
                str(sub_result.get("error_class") or "")
                if isinstance(sub_result, dict) else ""
            )
            log.warning(
                "framework_pr_executor: sub-baseline failed for %s "
                "(error_class=%s); rolling back to %s",
                winner_ref, err_class or "(unknown)", _short(prev_head_sha),
            )
            rollback_ok, rollback_err = self._rollback(prev_head_sha)
            return {
                "status": "failed",
                "error_class": err_class or "sub_baseline_failed",
                "error": err or "sub-baseline produced no measurement",
                "selected_ref": winner_ref,
                "selected_head_sha": winner_head_sha,
                "prev_head_sha": prev_head_sha,
                "applied_ref": winner_ref,
                "rollback_done": rollback_ok,
                "rollback_error": rollback_err,
                "apply_elapsed_sec": time.time() - apply_started_unix,
                "sub_baseline_result": sub_result,
                "workspace": str(workspace),
            }

        delta_pct = (new_tput - base_tput) / base_tput * 100.0
        promoted = delta_pct >= min_gain_pct

        if not promoted:
            log.info(
                "framework_pr_executor: DISCARD %s — new_tput=%.1f base_tput=%.1f "
                "delta_pct=%.2f%% < min_gain_pct=%.2f%%; rolling back to %s",
                winner_ref, new_tput, base_tput, delta_pct, min_gain_pct,
                _short(prev_head_sha),
            )
            rollback_ok, rollback_err = self._rollback(prev_head_sha)
            return {
                "status": "succeeded",
                "decision": "discarded",
                "applied_ref": winner_ref,
                "selected_ref": winner_ref,
                "selected_head_sha": winner_head_sha,
                "selected_score": winner_score,
                "selected_title": winner_title,
                "prev_head_sha": prev_head_sha,
                "new_head_sha": new_head_sha,
                "rollback_done": rollback_ok,
                "rollback_error": rollback_err,
                "new_tput": new_tput,
                "base_tput": base_tput,
                "delta_pct": delta_pct,
                "min_gain_pct": min_gain_pct,
                "accuracy": accuracy,
                "candidates_seen": len(candidates),
                "gap": gap, "keywords": keywords,
                "apply_elapsed_sec": time.time() - apply_started_unix,
                "sub_baseline_result": sub_result,
                "workspace": str(workspace),
            }

        log.info(
            "framework_pr_executor: KEEP %s — new_tput=%.1f base_tput=%.1f "
            "delta_pct=%.2f%% >= min_gain_pct=%.2f%% (head=%s)",
            winner_ref, new_tput, base_tput, delta_pct, min_gain_pct,
            _short(new_head_sha),
        )
        # KEEP path: leave the worktree on the PR head. Surface
        # output_throughput so Coordinator._promote_to_shared_state
        # lifts the result into current_best.
        return {
            "status": "succeeded",
            "decision": "kept",
            "output_throughput": new_tput,
            "accuracy": accuracy,
            "applied_ref": winner_ref,
            "selected_ref": winner_ref,
            "selected_head_sha": winner_head_sha,
            "selected_score": winner_score,
            "selected_title": winner_title,
            "prev_head_sha": prev_head_sha,
            "new_head_sha": new_head_sha,
            "new_tput": new_tput,
            "base_tput": base_tput,
            "delta_pct": delta_pct,
            "min_gain_pct": min_gain_pct,
            "candidates_seen": len(candidates),
            "gap": gap, "keywords": keywords,
            "apply_elapsed_sec": time.time() - apply_started_unix,
            "sub_baseline_result": sub_result,
            "workspace": str(workspace),
        }

    def _rollback(self, prev_head_sha: str) -> tuple[bool, str]:
        """Try to ``git checkout --detach <prev_head_sha>``.

        Returns ``(ok, err_str)``. Failures are logged but never raised
        so the executor still returns a well-formed result dict (the
        operator sees ``rollback_done=False`` in the audit row and can
        intervene manually before the next arm runs).
        """
        try:
            rollback_to(prev_head_sha, sglang_path=self.sglang_path)
            return True, ""
        except FrameworkPRError as exc:
            log.exception(
                "framework_pr_executor: rollback to %s failed: %s",
                _short(prev_head_sha), exc,
            )
            return False, str(exc)


framework_pr_executor = FrameworkPRExecutor()


__all__ = [
    "FrameworkPRExecutor",
    "framework_pr_executor",
]
