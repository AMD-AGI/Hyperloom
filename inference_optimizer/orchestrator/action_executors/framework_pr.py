"""FrameworkPrExecutor — FRAMEWORK_PR phase per-candidate executor.

Counterpart to :class:`IntegratePatchExecutor` for the FRAMEWORK_PR
phase. The phase loop (Coordinator-side) enumerates PR candidates via
``fa phase-discover``, fetches each one's worktree via
``fa phase-fetch``, runs the Critic gate on a synthesised
specialist_done envelope, and dispatches **this** executor per
approved candidate to:

  1. Apply the PR's unified diff to the live framework_source_roots
     via ``git apply`` (single integration channel, mirrors
     ``integrate_patch``).
  2. Bench the patched server with ``run_grid([GridVariant])`` (size=1,
     same throughput + accuracy gate plumbing as ``integrate_patch``).
  3. KEEP / REVERT decision; on REVERT, reverse-apply the patch so the
     source tree returns to the baseline state.

This is a Coordinator-internal action (``framework_pr_action_not_llm_proposable``
denies any LLM-side delegate / propose_action / request). It is
registered for the FRAMEWORK_PR phase only.

Inputs (``ctx.task.params``):
    candidate (dict, required) — PR metadata row:
        ``{repo, pr_number, ref, title, diff_url, pr_url?, framework?}``
    framework (str, optional) — ``"sglang"`` / ``"vllm"``. Falls back to
        ``candidate["framework"]`` then ``$INFERENCE_OPTIMIZER_FRAMEWORK``.
    batch_id (str, optional) — passed back in the result so the phase
        loop can group ``framework_pr_phase_progress`` entries.
    patches (list[str], optional) — explicit patch paths. When omitted,
        the executor curls ``candidate.diff_url`` into the per-task
        workspace and applies that.
    keep_threshold_pct (float, optional) — default 0.2.
    base_tput (float, optional) — baseline throughput; falls back to
        ``SharedState.baseline_tput``.
    accuracy_baseline (float, optional) — forwarded to the accuracy gate.
    benchmark_script / result_dir / variant_timeout_sec / base_extra_args
        — same semantics as ``integrate_patch``.
    framework_source_root (str, optional) — explicit ``git apply`` target;
        defaults to first existing entry of ``resolve_source_file_allowlist()``.
    apply_only (bool, optional) — skip the bench step (test / smoke).

Outputs (dict returned to the bus as ``delegated_result.result``):
    status: "kept" | "reverted" | "apply_failed" | "no_patch" |
            "fetch_failed" | "applied_no_bench" | "failed"
    output_throughput: float | None
    delta_pct: float | None
    accuracy_pass: bool | None
    candidate: dict (echoes the input row)
    batch_id: str
    patches_applied: list[str]
    patches_reverted: list[str]
    reason: str
    workspace: str
    bench_result: dict | None
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ._accuracy_gate import accuracy_passed, parse_eval_results
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _resolve_session_dir,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
)
from ._workload_envs import default_baseline_config, materialize_config_with_envs
from .integrate_patch import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_VARIANT_TIMEOUT_SEC,
    _git_apply,
    _git_apply_reverse,
    _git_checkout_clean,
    _resolve_framework_root,
)


log = logging.getLogger(__name__)


DEFAULT_DIFF_FETCH_TIMEOUT_SEC: float = 30.0


def _candidate_slug(candidate: dict[str, Any]) -> str:
    """Short, filesystem-safe identifier for the candidate (for variant
    names + workspace paths). Prefer ``repo/pr_number`` when present."""
    repo = str(candidate.get("repo") or "").replace("/", "-")
    pr = candidate.get("pr_number")
    if repo and pr not in (None, "", 0):
        return f"{repo}-pr-{pr}"
    ref = str(candidate.get("ref") or "").replace(":", "-")
    if repo and ref:
        return f"{repo}-{ref}"
    return repo or ref or "candidate"


def _fetch_diff_to_path(
    diff_url: str, dest: Path, *, timeout_sec: float,
) -> tuple[bool, str]:
    """Curl ``diff_url`` into ``dest`` (a .patch file path). Returns
    ``(ok, stderr)``. Uses curl rather than aiohttp because the
    integrate_patch path is also subprocess-based and we want consistent
    behaviour for restricted-network sessions (curl honours the same
    HTTPS_PROXY plumbing as the rest of the runtime)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-fsSL", "--retry", "2", "--max-time",
        str(int(timeout_sec)), "-o", str(dest), diff_url,
    ]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec + 5.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"curl spawn / timeout: {exc!r}"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    if not dest.exists() or dest.stat().st_size == 0:
        return False, "curl wrote empty / missing file"
    return True, ""


class FrameworkPrExecutor:
    """ActionRunner for the ``framework_pr`` action (FRAMEWORK_PR phase)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
        diff_fetch_timeout_sec: float = DEFAULT_DIFF_FETCH_TIMEOUT_SEC,
    ):
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)
        self.diff_fetch_timeout_sec = float(diff_fetch_timeout_sec)

    async def __call__(self, ctx) -> dict[str, Any]:
        params = dict(ctx.task.params or {})
        extra = getattr(ctx, "extra", None) or {}
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "framework_pr requires params.candidate (the PR metadata "
                    "row produced by `fa phase-discover`)"
                ),
            }
        batch_id = str(params.get("batch_id") or "")
        slug = _candidate_slug(candidate)

        # Per-task workspace under runs/framework_pr/<task_id>/.
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "framework_pr", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Resolve patch sources.
        explicit_patches = params.get("patches") or None
        patch_paths: list[Path] = []
        if isinstance(explicit_patches, list) and explicit_patches:
            for p in explicit_patches:
                pp = Path(str(p))
                if pp.exists():
                    patch_paths.append(pp.resolve())
                else:
                    log.warning(
                        "framework_pr: explicit patch %r not found", p,
                    )
        else:
            diff_url = str(candidate.get("diff_url") or "").strip()
            if not diff_url:
                return {
                    "status": "no_patch",
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "reason": (
                        "candidate carries no diff_url and no explicit "
                        "patches were supplied"
                    ),
                    "workspace": str(output_root),
                }
            dest = output_root / f"{slug}.patch"
            ok, err = _fetch_diff_to_path(
                diff_url, dest, timeout_sec=self.diff_fetch_timeout_sec,
            )
            if not ok:
                return {
                    "status": "fetch_failed",
                    "error_class": "diff_fetch_failed",
                    "error": err,
                    "candidate": candidate,
                    "batch_id": batch_id,
                    "patches_applied": [],
                    "patches_reverted": [],
                    "reason": f"failed to fetch {diff_url!r}: {err}",
                    "workspace": str(output_root),
                }
            patch_paths.append(dest.resolve())

        framework_root = _resolve_framework_root(
            params.get("framework_source_root") or None,
        )
        if framework_root is None:
            return {
                "status": "apply_failed",
                "error_class": "no_framework_root",
                "error": (
                    "no framework_source_root resolved; cannot apply "
                    "candidate PR. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                ),
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [],
                "workspace": str(output_root),
            }

        # Stage 1: apply patches (with -3 fallback like integrate_patch).
        applied: list[Path] = []
        apply_errors: list[dict[str, str]] = []
        for patch in patch_paths:
            ok, err = _git_apply(framework_root, patch, three_way=False)
            if not ok:
                ok2, err2 = _git_apply(framework_root, patch, three_way=True)
                if not ok2:
                    apply_errors.append({
                        "patch": str(patch),
                        "stderr": err + " | -3 retry: " + err2,
                    })
                    break
            applied.append(patch)
        if apply_errors:
            reverted = self._revert_patches(framework_root, applied)
            return {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "reason": "git apply failed (see error)",
                "workspace": str(output_root),
            }

        if params.get("apply_only"):
            return {
                "status": "applied_no_bench",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(output_root),
            }

        # Stage 2: bench via run_grid (size=1).
        try:
            bench_result, gate_evidence = await self._bench_candidate(
                params=params,
                output_root=output_root,
                slug=slug,
            )
        except Exception as exc:  # noqa: BLE001
            reverted = self._revert_patches(framework_root, applied)
            return {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            }

        # Stage 3: KEEP / REVERT.
        base_tput = float(params.get("base_tput") or 0.0)
        if base_tput <= 0:
            ss = extra.get("shared_state") or extra.get("state")
            if ss is not None:
                base_tput = float(getattr(ss, "baseline_tput", 0.0) or 0.0)
        keep_threshold_pct = float(
            params.get("keep_threshold_pct", self.keep_threshold_pct),
        )
        new_tput = bench_result.get("output_throughput")
        delta_pct: float | None = None
        if (
            isinstance(new_tput, (int, float)) and new_tput > 0
            and base_tput > 0
        ):
            delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0

        accuracy_pass = gate_evidence.get("accuracy_pass")
        gate_pass = (
            delta_pct is not None
            and delta_pct >= keep_threshold_pct
            and (accuracy_pass is None or accuracy_pass)
        )

        if not gate_pass:
            reverted = self._revert_patches(framework_root, applied)
            reasons: list[str] = []
            if delta_pct is None:
                reasons.append("no measurable throughput")
            elif delta_pct < keep_threshold_pct:
                reasons.append(
                    f"throughput delta {delta_pct:+.2f}% < keep_threshold "
                    f"{keep_threshold_pct:.2f}%"
                )
            if accuracy_pass is False:
                reasons.append("accuracy regression detected")
            return {
                "status": "reverted",
                "candidate": candidate,
                "batch_id": batch_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": "; ".join(reasons) or "gate failed",
                "bench_result": bench_result,
                "workspace": str(output_root),
            }

        return {
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
            "reason": (
                f"throughput delta {delta_pct:+.2f}% >= "
                f"{keep_threshold_pct:.2f}%"
            ),
            "bench_result": bench_result,
            "workspace": str(output_root),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _revert_patches(
        self, framework_root: Path | None, applied: list[Path],
    ) -> list[Path]:
        """Reverse-apply the patches we already applied (mirrors
        :meth:`IntegratePatchExecutor._revert_patches`)."""
        reverted: list[Path] = []
        if framework_root is None or not applied:
            return reverted
        for patch in reversed(applied):
            ok, err = _git_apply_reverse(framework_root, patch)
            if ok:
                reverted.append(patch)
            else:
                log.warning(
                    "framework_pr: git apply -R failed for %s: %s; "
                    "falling back to git checkout",
                    patch, err,
                )
                ok2, err2 = _git_checkout_clean(framework_root)
                if ok2:
                    reverted = list(applied)
                    break
                log.error(
                    "framework_pr: git checkout fallback failed: %s", err2,
                )
                break
        return reverted

    async def _bench_candidate(
        self, *,
        params: dict[str, Any],
        output_root: Path,
        slug: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server +
        evaluate the accuracy gate. Mirrors
        :meth:`IntegratePatchExecutor._bench_patch` so the gain
        bookkeeping is identical across the two integration channels."""
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not config_path.exists():
            raise RuntimeError(
                f"framework_pr bench: config not found at {config_path}"
            )
        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        override_script = sanitize_script_name(params.get("benchmark_script"))
        override_result_dir = sanitize_result_dir(params.get("result_dir"))
        config_path = materialize_config_with_envs(
            config_path,
            output_root,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=override_script,
            out_name="framework_pr.with_envs.yaml",
        )

        variant = GridVariant(
            name=f"framework-pr-{slug}"[:96],
            extra_sglang_args=str(params.get("base_extra_args") or "").strip(),
            extra_envs={},
            note=f"framework_pr:{slug}",
        )

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
        )

        bench: dict[str, Any] = {}
        if results:
            r = results[0]
            bench = {
                "name": r.name,
                "status": r.status,
                "output_throughput": getattr(r, "output_throughput", None),
                "ttft_ms": getattr(r, "ttft_ms", None),
                "itl_ms": getattr(r, "itl_ms", None),
                "result_dir": str(getattr(r, "result_dir", "")),
                "error": getattr(r, "error", "") or "",
                "nonfatal_warnings": list(
                    getattr(r, "nonfatal_warnings", []) or []
                ),
            }

        accuracy_pass: bool | None = None
        baseline_accuracy = params.get("accuracy_baseline")
        if (
            bench.get("status") == "succeeded"
            and isinstance(baseline_accuracy, (int, float))
            and float(baseline_accuracy) > 0
        ):
            try:
                eval_results = parse_eval_results(bench["result_dir"])
                if eval_results.get("score") is not None:
                    accuracy_pass = accuracy_passed(
                        eval_results["score"], float(baseline_accuracy),
                    )
            except Exception:  # noqa: BLE001
                log.exception(
                    "framework_pr: accuracy gate parse failed; "
                    "treating as None (gate skipped)"
                )

        return bench, {"accuracy_pass": accuracy_pass}


framework_pr_executor = FrameworkPrExecutor


__all__ = [
    "DEFAULT_DIFF_FETCH_TIMEOUT_SEC",
    "FrameworkPrExecutor",
    "framework_pr_executor",
]
