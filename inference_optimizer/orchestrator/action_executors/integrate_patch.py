# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""IntegratePatchExecutor — PR-A4 (Arbor-into-Hyperloom).

Serving-lane-locked patch integration: consumes a specialist's worktree
patches, applies them to the live framework source roots, runs a
throughput + optional accuracy gate, then KEEPs (advances the stack) or
REVERTs (rolls back the tree).

Deterministic Python executor (no LLM). Per Inv-5.1, this is the single
allowed ``git apply`` channel against framework_source_roots (specialists
author patches into their isolated worktree only).

Inputs (``ctx.task.params``):
    specialist_task_id (str, required) — completed specialist task
        whose worktree under ``runs/specialist/<task_id>/`` carries
        the patches.
    patches (list[str], optional) — explicit patch paths. Defaults to
        ``specialist_done.patches_written``.
    config_changes (dict[str, str], optional) — env vars layered on
        the variant's launch env. Reverted with the patches on REVERT.
    keep_threshold_pct (float, optional) — KEEP threshold; defaults to
        DEFAULT_KEEP_THRESHOLD_PCT (1.0). No stack rebench here, so the
        sole gate sits at the grid noise floor (1.0%) to avoid committing
        noise-level "gains".
    accuracy_baseline (float | dict, optional) — accuracy gate input;
        forwarded to the existing accuracy gate utilities.
    base_tput (float, optional) — baseline throughput to compare
        against. Falls back to ``SharedState.baseline_tput`` if zero.
    benchmark_script / result_dir / variant_timeout_sec — same
        semantics as the explore executor's params.
    framework_source_root (str, optional) — explicit override for the
        ``git apply`` target. Defaults to the first existing entry of
        ``resolve_source_file_allowlist()``.
    apply_only (bool, optional) — when True, skip the benchmark step
        entirely (used by tests + a future smoke-only mode). The
        executor still applies the patches but returns
        ``status='applied_no_bench'`` so downstream bookkeeping can
        differentiate from a genuine KEEP/REVERT.

Outputs (dict, returned to the bus as ``delegated_result.result``):
    status: "kept" | "reverted" | "apply_failed" | "no_patches" |
            "applied_no_bench" | "failed"
    output_throughput: float | None
    delta_pct: float | None
    accuracy_pass: bool | None
    patches_applied: list[str]
    patches_reverted: list[str]
    config_changes_applied: dict[str, str]
    reason: str
    specialist_task_id: str
    workspace: str
    bench_result: dict | None
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ...session_paths import runs_dir
from ..framework_paths import resolve_source_file_allowlist
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


log = logging.getLogger(__name__)


DEFAULT_KEEP_THRESHOLD_PCT = 1.0  # D1: was 0.2 (below grid 1.0% noise floor; no stack rebench here)
DEFAULT_VARIANT_TIMEOUT_SEC = 7800  # 130 min; aligns with BASELINE_DEFAULT_TIMEOUT_SEC for Qwen3-32B TP=1 long workload


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _resolve_framework_root(explicit: str | None) -> Path | None:
    """Pick the framework source root for patches.

    Precedence: explicit param → first existing
    ``resolve_source_file_allowlist()`` entry. None when nothing resolves.
    """
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            return p
        log.warning(
            "integrate_patch: framework_source_root override %r does not "
            "exist; falling back to allowlist", explicit,
        )
    for root in resolve_source_file_allowlist():
        p = Path(root)
        if p.is_dir() and (p / ".git").exists():
            return p
    # Last resort: a non-git dir (prefer surfacing as clean apply_failed).
    for root in resolve_source_file_allowlist():
        p = Path(root)
        if p.is_dir():
            return p
    return None


def _git_apply(
    framework_root: Path, patch_path: Path, *, three_way: bool = False,
    check_only: bool = False,
) -> tuple[bool, str]:
    """Run ``git apply [-3] -p1 [--check] <patch>`` inside
    ``framework_root``. Returns ``(ok, stderr)``."""
    cmd = ["git", "-C", str(framework_root), "apply", "-p1"]
    if three_way:
        cmd.append("-3")
    if check_only:
        cmd.append("--check")
    cmd.append(str(patch_path))
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"git apply spawn failed: {exc!r}"
    return cp.returncode == 0, cp.stderr.strip()


def _git_apply_reverse(
    framework_root: Path, patch_path: Path,
) -> tuple[bool, str]:
    """Reverse-apply ``patch_path`` (``git apply -R -p1``) as the REVERT
    path; caller falls back to ``git checkout`` on failure."""
    cmd = ["git", "-C", str(framework_root), "apply", "-R", "-p1", str(patch_path)]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"git apply -R spawn failed: {exc!r}"
    if cp.returncode == 0:
        return True, ""
    return False, cp.stderr.strip()


def _git_checkout_clean(framework_root: Path) -> tuple[bool, str]:
    """``git checkout -- .`` to discard every uncommitted change.
    Last-resort REVERT path when individual reverse-apply fails."""
    cmd = ["git", "-C", str(framework_root), "checkout", "--", "."]
    try:
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60.0, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"git checkout spawn failed: {exc!r}"
    return cp.returncode == 0, cp.stderr.strip()


def _resolve_patch_paths(
    *,
    specialist_workspace: Path,
    explicit_patches: list[str] | None,
    done_payload: dict[str, Any] | None,
) -> list[Path]:
    """Resolve the list of patch files to apply.

    Order: ``params.patches`` → ``specialist_done.patches_written`` →
    filesystem scan of ``specialist_workspace/{worktree/,}patches/``.
    Entries normalised to absolute Paths; missing ones logged + dropped.
    """
    candidates: list[str] = []
    if explicit_patches:
        candidates.extend(str(p) for p in explicit_patches)
    elif done_payload and isinstance(done_payload.get("patches_written"), list):
        candidates.extend(
            str(p) for p in done_payload["patches_written"] if p
        )
    else:
        for base in (
            specialist_workspace / "worktree" / "patches",
            specialist_workspace / "patches",
        ):
            if base.is_dir():
                for p in sorted(base.glob("*.patch")):
                    candidates.append(str(p))
                for p in sorted(base.glob("*.diff")):
                    candidates.append(str(p))

    out: list[Path] = []
    for c in candidates:
        p = Path(c)
        # Resolve relative paths against the specialist workspace + worktree.
        if not p.is_absolute():
            for base in (
                specialist_workspace / "worktree",
                specialist_workspace,
            ):
                cand = base / c
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            log.warning(
                "integrate_patch: patch %r not found (specialist_workspace=%s)",
                c, specialist_workspace,
            )
            continue
        out.append(p.resolve())
    return out


def _read_done_payload(workspace: Path) -> dict[str, Any] | None:
    done = workspace / "specialist_done.json"
    if not done.exists():
        return None
    try:
        return json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "integrate_patch: failed to parse %s: %r", done, exc,
        )
        return None


class IntegratePatchExecutor:
    """ActionRunner for the ``integrate_patch`` action (PR-A4)."""

    def __init__(
        self,
        *,
        session_dir: Path | str | None = None,
        default_config_path: Path | str | None = None,
        variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        self.session_dir = (
            Path(session_dir) if session_dir else _resolve_session_dir()
        )
        self.default_config_path = (
            Path(default_config_path) if default_config_path else None
        )
        self.variant_timeout_sec = int(variant_timeout_sec)
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        params = dict(ctx.task.params or {})
        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "integrate_patch requires params.specialist_task_id "
                    "(the completed specialist whose worktree carries "
                    "the patches to integrate)"
                ),
            }
        extra = getattr(ctx, "extra", None) or {}
        # Specialist workspace conventionally at runs/specialist/<id>/.
        specialist_workspace = (
            self.session_dir / "runs" / "specialist" / specialist_task_id
        )
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (
                    f"specialist workspace not found at "
                    f"{specialist_workspace}"
                ),
                "specialist_task_id": specialist_task_id,
            }

        # Read done payload for patches_written + config_changes_default.
        done_payload = _read_done_payload(specialist_workspace)

        # Patch resolution.
        explicit_patches = params.get("patches") or None
        patch_paths = _resolve_patch_paths(
            specialist_workspace=specialist_workspace,
            explicit_patches=(
                list(explicit_patches) if isinstance(explicit_patches, list)
                else None
            ),
            done_payload=done_payload,
        )
        config_changes = dict(params.get("config_changes") or {})
        # Seed config_changes from specialist_done when params didn't.
        if not config_changes and done_payload:
            cc = done_payload.get("config_changes")
            if isinstance(cc, dict):
                config_changes = {str(k): str(v) for k, v in cc.items()}

        if not patch_paths and not config_changes:
            return {
                "status": "no_patches",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
                "reason": (
                    "neither patches nor config_changes were supplied / "
                    "discoverable for this specialist task"
                ),
            }

        framework_root = _resolve_framework_root(
            params.get("framework_source_root") or None,
        )
        # Pure config_changes path works without a framework root.
        if patch_paths and framework_root is None:
            return {
                "status": "apply_failed",
                "error_class": "no_framework_root",
                "error": (
                    "no framework_source_root resolved; cannot apply "
                    "patches. Configure $INFERENCEX_PATH or pass "
                    "params.framework_source_root."
                ),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "config_changes_applied": {},
            }

        # Per-action workspace under runs/integrate_patch/<task_id>/.
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "integrate_patch", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Stage 1: apply patches (best-effort with -3 fallback).
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
                err = err2
            applied.append(patch)
        if apply_errors:
            # Mid-apply failure — reverse the partial set back to clean.
            reverted = self._revert_patches(framework_root, applied)
            await self._maybe_write_framework_pr_kb_record(
                done_payload=done_payload,
                outcome="rejected_apply_fail",
                tps_delta_pct=0.0,
                extra=extra,
            )
            return {
                "status": "apply_failed",
                "error_class": "git_apply_failed",
                "error": apply_errors,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "workspace": str(output_root),
            }

        # Stage 2: layer config_changes onto the launch env (via the
        # variant's ``extra_envs`` knob).
        config_changes_applied = dict(config_changes)

        # Defensive double-check on the Critic verdict. PolicyGate's
        # ``integrate_patch_requires_critic_verdict`` already gates the
        # delegate; this is belt-and-braces for paths that bypass PolicyGate
        # (legacy resume / test injection). No-ops when SharedState is absent.
        shared_state = extra.get("shared_state") or extra.get("state")
        if (
            shared_state is not None
            and not params.get("bypass_critic")
        ):
            try:
                recorded = shared_state.get_specialist_patch_verdict(
                    specialist_task_id,
                )
            except AttributeError:
                recorded = ""
            if recorded and recorded.lower() == "reject":
                reverted = self._revert_patches(framework_root, applied)
                return {
                    "status": "rejected_by_critic",
                    "specialist_task_id": specialist_task_id,
                    "patches_applied": [],
                    "patches_reverted": [str(p) for p in reverted],
                    "config_changes_applied": {},
                    "reason": (
                        f"Critic verdict 'reject' recorded for specialist "
                        f"task {specialist_task_id!r}; integrate_patch "
                        f"refuses to bench. Pass bypass_critic=True to "
                        f"force."
                    ),
                    "workspace": str(output_root),
                }

        # Stage 3: optionally skip the bench (test / smoke).
        if params.get("apply_only"):
            return {
                "status": "applied_no_bench",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [str(p) for p in applied],
                "patches_reverted": [],
                "config_changes_applied": config_changes_applied,
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(output_root),
            }

        # Stage 4: bench the patched config via run_grid (1 variant).
        try:
            bench_result, gate_evidence = await self._bench_patch(
                params=params,
                output_root=output_root,
                config_changes_applied=config_changes_applied,
                specialist_task_id=specialist_task_id,
            )
        except Exception as exc:  # noqa: BLE001
            reverted = self._revert_patches(framework_root, applied)
            return {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            }

        # Stage 5: KEEP / REVERT decision.
        base_tput = float(params.get("base_tput") or 0.0)
        keep_threshold_pct = float(
            params.get("keep_threshold_pct", self.keep_threshold_pct),
        )
        new_tput = bench_result.get("output_throughput")
        delta_pct = None
        if (
            isinstance(new_tput, (int, float)) and new_tput > 0
            and base_tput > 0
        ):
            delta_pct = (float(new_tput) - base_tput) / base_tput * 100.0

        accuracy_pass: bool | None = gate_evidence.get("accuracy_pass")
        # KEEP requires delta_pct ≥ keep_threshold AND accuracy_pass != False.
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
            await self._maybe_write_framework_pr_kb_record(
                done_payload=done_payload,
                outcome="reverted_smoke_fail",
                tps_delta_pct=float(delta_pct or 0.0),
                extra=extra,
            )
            return {
                "status": "reverted",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [str(p) for p in reverted],
                "config_changes_applied": {},
                "output_throughput": new_tput,
                "delta_pct": delta_pct,
                "accuracy_pass": accuracy_pass,
                "base_tput": base_tput,
                "keep_threshold_pct": keep_threshold_pct,
                "reason": "; ".join(reasons) or "gate failed",
                "bench_result": bench_result,
                "workspace": str(output_root),
            }

        await self._maybe_write_framework_pr_kb_record(
            done_payload=done_payload,
            outcome="integrated",
            tps_delta_pct=float(delta_pct or 0.0),
            extra=extra,
        )
        return {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": [str(p) for p in applied],
            "patches_reverted": [],
            "config_changes_applied": config_changes_applied,
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

    # Helpers
    @staticmethod
    def _find_framework_pr_proposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework_pr`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.
        """
        if not isinstance(done_payload, dict):
            return None
        proposal_set = done_payload.get("proposal_set") or []
        if not isinstance(proposal_set, list):
            return None
        for proposal in proposal_set:
            if not isinstance(proposal, dict):
                continue
            provenance = str(proposal.get("provenance") or "")
            if provenance.startswith("specialist:serving:framework_pr"):
                return proposal
        return None

    async def _maybe_write_framework_pr_kb_record(
        self,
        *,
        done_payload: dict[str, Any] | None,
        outcome: str,
        tps_delta_pct: float,
        extra: dict[str, Any],
    ) -> None:
        """F2-5: append a JSONL record to ``lessons.jsonl`` when the patch
        came from the FRAMEWORK_PR phase.

        No-op for other provenance or when both dedup keys (``fa_pr_url`` /
        ``fa_pr_sha``) are missing. Write errors are logged + swallowed.
        """
        proposal = self._find_framework_pr_proposal(done_payload)
        if proposal is None:
            return
        pr_url = str(proposal.get("fa_pr_url") or "").strip()
        pr_sha = str(proposal.get("fa_pr_sha") or "").strip()
        if not pr_url and not pr_sha:
            log.warning(
                "integrate_patch: framework_pr proposal lacks both "
                "fa_pr_url and fa_pr_sha; KB writeback skipped",
            )
            return
        patches_written = proposal.get("patches_written") or []
        patch_path = ""
        if isinstance(patches_written, list) and patches_written:
            patch_path = str(patches_written[0])
        session_id = ""
        shared_state = extra.get("shared_state") or extra.get("state")
        if shared_state is not None:
            session_id = str(
                getattr(shared_state, "cortex_session_id", "") or ""
            )
        try:
            from ..kb_writeback import write_framework_pr_record
            written = await write_framework_pr_record(
                pr_url=pr_url,
                pr_sha=pr_sha,
                patch_path=patch_path,
                outcome=outcome,
                tps_delta_pct=float(tps_delta_pct),
                session_id=session_id,
            )
            log.info(
                "integrate_patch: wrote framework_pr KB record to %s "
                "(outcome=%s pr_url=%s tps_delta=%+.2f%%)",
                written, outcome, pr_url, float(tps_delta_pct),
            )
        except Exception as exc:  # noqa: BLE001 — KB write is best-effort
            log.warning(
                "integrate_patch: framework_pr KB writeback failed: %r",
                exc,
            )

    def _revert_patches(
        self, framework_root: Path | None, applied: list[Path],
    ) -> list[Path]:
        """Reverse-apply the applied patches (best-effort); returns those
        actually reverted."""
        reverted: list[Path] = []
        if framework_root is None or not applied:
            return reverted
        # Reverse order so dependent patches unstick correctly.
        for patch in reversed(applied):
            ok, err = _git_apply_reverse(framework_root, patch)
            if ok:
                reverted.append(patch)
            else:
                log.warning(
                    "integrate_patch: git apply -R failed for %s: %s; "
                    "falling back to git checkout",
                    patch, err,
                )
                # Reverse-apply failed → checkout clears all uncommitted at once.
                ok2, err2 = _git_checkout_clean(framework_root)
                if ok2:
                    reverted = list(applied)
                    break
                log.error(
                    "integrate_patch: git checkout fallback failed: %s",
                    err2,
                )
                break
        return reverted

    async def _bench_patch(
        self, *,
        params: dict[str, Any],
        output_root: Path,
        config_changes_applied: dict[str, str],
        specialist_task_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run a 1-variant Magpie bench under the patched server + accuracy gate.

        Returns ``(bench_result_dict, gate_evidence)`` where gate_evidence
        carries ``accuracy_pass`` (True / False / None).
        """
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or default_baseline_config()
        )
        if not config_path.exists():
            raise RuntimeError(
                f"integrate_patch bench: config not found at {config_path}"
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
            out_name="integrate_patch.with_envs.yaml",
        )

        # Single-variant grid with config_changes_applied as extra_envs.
        variant = GridVariant(
            name=f"integrate-patch-{specialist_task_id[:8]}",
            extra_server_args=str(params.get("base_extra_args") or "").strip(),
            extra_envs=dict(config_changes_applied),
            note=f"integrate_patch:{specialist_task_id}",
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

        # Accuracy gate runs only on a succeeded bench with a baseline;
        # else ``None`` (KEEP gate skips the accuracy check).
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
                    "integrate_patch: accuracy gate parse failed; "
                    "treating as None (gate skipped)"
                )

        return bench, {"accuracy_pass": accuracy_pass}


__all__ = [
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_VARIANT_TIMEOUT_SEC",
    "IntegratePatchExecutor",
    "_git_apply",
    "_git_apply_reverse",
    "_git_checkout_clean",
    "_resolve_framework_root",
    "_resolve_patch_paths",
    "_read_done_payload",
]
