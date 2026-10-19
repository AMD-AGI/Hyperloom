# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE phase handler for section-based Recipe replay and initial analysis."""

from __future__ import annotations
import logging as _logging
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple
from hyperloom.common.perf_metric import VERDICT_KEEP, constraint_fields, graded_axes_of
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
    GATE_ACCURACY,
    GATE_KEEP_THRESHOLD,
    GATE_PARAMS_PRESENT,
    GATE_PROMOTION,
    GATE_QUALITY,
    GATE_TPUT_VALID,
    SKIP_BEST_CONFIG_EMPTY,
    SKIP_CONFIDENCE_BELOW_THRESHOLD,
    SKIP_DISABLED_BY_FLAG,
    SKIP_ENQUEUE_FAILED,
    SKIP_FRAMEWORK_ROOT_MISSING,
    SKIP_KERNEL_PREPARATION_FAILED,
    SKIP_KERNEL_ROOT_MISSING,
    SKIP_NO_WARM_START_RECIPE,
    SKIP_RECIPE_NOT_REPLAYABLE,
    SKIP_RECIPE_READ_FAILED,
    SKIP_RESTATES_BASELINE,
    SKIP_WORKLOAD_CONFIG_INCOMPATIBLE,
)

from . import machine_state as _phase_state
from hyperloom.inference_optimizer.grid_server_args import merge_cumulative_extra_server_args
from hyperloom.inference_optimizer.inferencex_checkout import pick_tree
from hyperloom.inference_optimizer.session.optimization_journal import (
    KIND_OTHER,
    derive_journal_outcome,
    JournalEntry,
    Verdict,
)
from ..actions.executors._launch_restatement import config_launch_restated_by
from ..actions.executors.baseline import _revert_warm_patch_state, revert_warm_kernel_patches
from ..state.analysis_failures import RECIPE_PARAM, repeated_roofline_failure, roofline_recipe
from ..state.shared_state import inject_stack_base_params, resolve_graded_comparison
from ..state.task_registry import Task
from ..loop.time_budget import (
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
)
from ..collaborator import CoordinatorCollaborator
from ._warm_prior import (
    ROOT_ABSENT,
    ROOT_MISSING,
    SUPPRESSED_LOW_CONFIDENCE,
    StagedKernelPlan,
    WarmKernelApply,
    WarmKernelOverlay,
    WarmLaunch,
    WarmReplayConfig,
    WarmReplayRequest,
    add_kernel_envs,
    as_float,
    merge_named_current_recipe_configs,
    overlay_provenance_summary,
    point_envs_at_local_files,
    point_tuner_envs_at_local_files,
    recorded_apply_roots,
    recorded_framework_roots,
    refuse_warm_kernel_entry,
    warm_expected_gain,
    warm_kernel_plan_entry,
    warm_recipe_history_rows,
    warm_recipe_rejection,
    warm_replay_donor_fields,
    warm_replay_launch_config,
    warm_replay_patch_plan,
)

log = _logging.getLogger(__name__)


# The ``parse_eval_results`` reasons that prove an eval produced output: a results file it could not decode, and one
# carrying no metric it recognises.
_EVAL_RAN_BUT_UNSCORABLE = ("parse error:", "no recognized metric in")

# The phase segment of the warm-replay event id. A literal rather than a read of
# ``state.phase``: the id must be identical at the tick that enqueues the replay
# and the later tick that harvests it, and the replay only ever runs in PRELUDE.
_WARM_REPLAY_EVENT_PHASE = "prelude"

#: The marker reason a PRELUDE arm refused for budget is filed under. A
#: constant because a consumer asking "was the initial analysis refused, or
#: never considered" selects on it.
PRELUDE_ARM_DROPPED = "prelude_arm_dropped"


class _KernelHalf(NamedTuple):
    """The kernel items a combined replay applied, and the snapshots taken before they landed."""

    applied: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]


class _PromotableTree(NamedTuple):
    """Where a measured checkout promotes to, or why it cannot be published."""

    root: str
    failure: dict[str, Any] | None


def _replayed_patch_trees(result: Mapping[str, Any], pending: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The checkouts a warm replay applied its overlays into.

    A round that predates the list reports only its primary tree, under the
    flat ``warm_patch_*`` keys.
    """
    trees = result.get("warm_patch_trees") or pending.get("recipe_patch_trees") or []
    listed = [tree for tree in trees if isinstance(tree, Mapping)]
    if listed:
        return listed
    return [
        {
            "root": result.get("warm_patch_target") or pending.get("recipe_patch_target"),
            "pre_sha": result.get("warm_patch_pre_sha") or pending.get("recipe_patch_pre_sha"),
            "snapshot_manifest": (
                result.get("warm_patch_snapshot_manifest") or pending.get("recipe_patch_snapshot_manifest")
            ),
        }
    ]


def _promotable_tree(tree: Mapping[str, Any]) -> _PromotableTree:
    """Rule on whether one replayed checkout may be published as the active one."""
    target = str(tree.get("root") or "").strip()
    pre_sha = str(tree.get("pre_sha") or "").strip()
    manifest = tree.get("snapshot_manifest")
    # Promotion needs a way to unwind the tree if the replay is rejected, not a sha. A git checkout
    # answers with a pre_sha and a snapshot manifest; a pip-installed framework -- the tree a KB recipe
    # is usually measured on -- answers with the backups nogit wrote as it applied. A tree this round
    # never wrote to owes neither, and promotion is reached only once every required overlay applied, so
    # it already carries what is being promoted. Records predating the flag are read as written-to.
    snapshotted = bool(pre_sha) and isinstance(manifest, Mapping)
    unwindable = snapshotted or bool(tree.get("nogit_backups"))
    if not target or (bool(tree.get("mutated", True)) and not unwindable):
        return _PromotableTree(
            "",
            {
                "status": "failed",
                "failure": "validated_recipe_checkout_incomplete",
                "target_repo": target,
            },
        )
    try:
        target_path = Path(target).resolve(strict=True)
        manifest_target = (
            Path(str(manifest.get("repo_path") or "")).resolve(strict=True) if snapshotted else target_path
        )
    except (OSError, ValueError) as exc:
        return _PromotableTree(
            "",
            {"status": "failed", "failure": f"validated_recipe_checkout_invalid:{type(exc).__name__}"},
        )
    if target_path != manifest_target:
        return _PromotableTree(
            "",
            {
                "status": "failed",
                "failure": "validated_recipe_checkout_manifest_mismatch",
                "target_repo": str(target_path),
            },
        )
    return _PromotableTree(str(target_path), None)


def _restore_replayed_patch_tree(tree: Mapping[str, Any]) -> dict[str, Any] | None:
    """Put one replayed checkout back, or None when the tree owes nothing."""
    target = str(tree.get("root") or "")
    if not target:
        return None
    if not bool(tree.get("mutated", True)):
        # An overlay the tree already carried applies as a no-op, writing
        # nothing and so owing nothing. A record predating the flag reads as
        # written-to, and still has to answer with a channel below.
        return None
    manifest = tree.get("snapshot_manifest")
    backups = list(tree.get("nogit_backups") or [])
    if not manifest and not backups:
        return {"ok": False, "errors": [f"recipe:{target}:missing_snapshot_manifest"]}
    return _revert_warm_patch_state(
        target,
        pre_sha=str(tree.get("pre_sha") or ""),
        snapshot_manifest=manifest,
        nogit_backups=backups,
    )


def _recorded_kernel_half(
    result: Mapping[str, Any],
    params: Mapping[str, Any],
    pending: Mapping[str, Any],
) -> _KernelHalf:
    """The kernel items and snapshots a replay recorded, wherever the round left them."""
    return _KernelHalf(
        list(
            result.get("warm_kernel_apply_results")
            or params.get("warm_kernel_apply_results")
            or pending.get("kernel_apply_results")
            or []
        ),
        list(
            result.get("warm_kernel_snapshots")
            or params.get("warm_kernel_snapshots")
            or pending.get("kernel_snapshots")
            or []
        ),
    )


class _ReplayMeasurement(NamedTuple):
    """What a warm-replay round measured, and what it was measured against."""

    tput: float
    cold_round_tput: float
    baseline_tput: float
    measured_gain: float
    expected_gain: float


class _AccuracyReading(NamedTuple):
    """What a replay scored, and whether an eval ran at all to say so."""

    score: float | None
    eval_ran: bool
    error: str


class _WarmReplaySettlement(NamedTuple):
    """The round being settled: what it reported, what asked for it, and where the ruling goes."""

    result: dict[str, Any]
    task: "Task | None"
    outcome: dict[str, Any]
    recorder: Any

    @property
    def params(self) -> Mapping[str, Any]:
        """The params the replay was dispatched with; empty on the degraded path."""
        return (self.task.params if self.task is not None else {}) or {}


class PreludePhase(CoordinatorCollaborator):
    """PRELUDE phase handler: session boot, target analysis, and phase seeding."""

    def internal_analysis_kind(self) -> str:
        """Pick the kind for the next Coordinator-internal analysis task: roofline when enable_roofline else profile."""
        return (
            "roofline"
            if bool(
                self.shared_state.enable_roofline,
            )
            else "profile"
        )

    def _measured_analysis_cost_sec(self) -> float:
        """Expected cost of the initial roofline/profile arm, in seconds."""
        meta = self.action_registry.get(self.internal_analysis_kind())
        return (
            expected_action_cost_minutes(
                meta,
                measured_baseline_sec=measured_baseline_runtime_sec(self.shared_state),
            )
            * 60.0
        )

    def _record_prelude_arm_dropped(self, arm: str, evidence: dict[str, Any]) -> None:
        """Record a PRELUDE arm the budget refused, as a marker on the phase.

        A marker rather than a key on the entry row's ``evidence``: that row is
        snapshotted into the phase event when the phase is entered, so anything
        appended to it afterwards reaches ``state`` and never the breakdown --
        where a refused arm then reads identically to one never considered.
        """
        _phase_state.append_phase_history_event(
            self.shared_state,
            reason=PRELUDE_ARM_DROPPED,
            evidence={"arm": arm, **evidence},
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:
            log.exception("PRELUDE: failed to persist the dropped-arm record for %r", arm)

    def warm_recipe_proven_items(self) -> list[dict[str, str]]:
        """Summarise warm-start ``what_worked`` items the scout can skip ({name, source}); fail-soft."""
        out: list[dict[str, str]] = []
        for row in warm_recipe_history_rows(self.shared_state.warm_start_recipe, "what_worked"):
            name = str(row.get("name") or "").strip()
            if name:
                out.append({"name": name, "source": str(row.get("source") or "").strip()})
        return out

    def inject_warm_recipe_history_into_ledger(self) -> int:
        """Pre-fill ``explore_search.rejected`` with the warm recipe's ``what_failed`` rows (fingerprinted so the dedup gate denies re-tests). Idempotent via warm_history_injected; returns rows added."""
        state = self.shared_state
        if state.warm_history_injected:
            return 0
        warm = state.warm_start_recipe if isinstance(state.warm_start_recipe, dict) else {}
        added = self._reject_warm_recipe_failures(
            warm_recipe_history_rows(warm, "what_failed"),
            str(warm.get("tier") or ""),
        )
        state.warm_history_injected = True
        return added

    def _reject_warm_recipe_failures(self, rows: list[dict[str, Any]], tier: str) -> int:
        """File the prior's failed configs on the explore ledger so the dedup gate denies re-tests."""
        if not rows:
            return 0
        from hyperloom.inference_optimizer.canonical_fingerprint import (
            canonical_fingerprint,
        )

        state = self.shared_state
        es = dict(state.explore_search) if isinstance(state.explore_search, dict) else {}
        rejected = list(es.get("rejected") or [])
        known = {str(r.get("fingerprint") or "") for r in rejected if isinstance(r, dict)}
        known.discard("")
        added = 0
        for row in rows:
            args = str(row.get("extra_server_args") or "").strip()
            envs = row.get("extra_envs") if isinstance(row.get("extra_envs"), dict) else {}
            if not args and not envs:
                continue
            fingerprint = canonical_fingerprint(args, envs)
            if fingerprint in known:
                continue
            known.add(fingerprint)
            rejected.append(warm_recipe_rejection(row, fingerprint, tier))
            added += 1
        if added:
            es["rejected"] = rejected
            state.explore_search = es
            log.info(
                "warm-recipe history: injected %d what_failed rows into explore_search.rejected (tier=%s)",
                added,
                tier,
            )
        return added

    def _collect_warm_kernel_plan(self, kb: Any) -> list[dict[str, Any]]:
        """Resolve the prior-champion kernel columns into a local apply plan."""
        readers = (
            ("gemm", kb.read_gemm, "optimizations"),
            ("fusion", kb.read_fusion, "items"),
            ("rewrite", kb.read_rewrite, "items"),
        )
        plan: list[dict[str, Any]] = []
        for column, reader, list_key in readers:
            data = reader() or {}
            rows = data.get(list_key) if isinstance(data, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                entry = warm_kernel_plan_entry(kb, column, row) if isinstance(row, dict) else None
                if entry is not None:
                    plan.append(entry)
        return plan

    @staticmethod
    def _parse_diff_targets(patch_path: str | None) -> list[str]:
        """Extract every safe repo-relative target from a unified diff."""
        raw = str(patch_path or "").strip()
        if not raw:
            return []
        try:
            text = Path(raw).read_text(errors="replace")
            from ..specialists.patch_safety import parse_patch_targets

            return list(parse_patch_targets(text).all)
        except (OSError, ValueError):
            return []

    @classmethod
    def _parse_diff_target(cls, patch_path: str | None) -> str:
        """Return the shared parser's first safe repo-relative Patch target."""
        targets = cls._parse_diff_targets(patch_path)
        return targets[0] if targets else ""

    def _resolve_kernel_target_paths(self, entry: dict[str, Any]) -> list[str]:
        """Resolve every declared Patch target under the Session active root."""

        def reject(reason: str, *, code: str = "") -> list[str]:
            entry["resolution_error"] = reason
            if code:
                entry["resolution_reason"] = code
            log.warning("Kernel Patch replay rejected: %s", reason)
            return []

        patch_path = Path(str(entry.get("patch_path") or "").strip())
        try:
            from ..specialists.patch_safety import parse_patch_targets

            parsed = parse_patch_targets(patch_path.read_text(errors="replace"))
        except (OSError, ValueError) as exc:
            return reject(f"invalid patch targets: {type(exc).__name__}: {exc}")
        # The recorded root is the only answer.
        root_value = str(entry.get("apply_root") or "").strip().rstrip("/")
        if not root_value:
            return reject(
                "kernel item records no apply root",
                code="kernel_apply_root_missing",
            )
        try:
            root = Path(root_value).resolve(strict=False)
            root_is_dir = root.is_dir()
        except (OSError, RuntimeError) as exc:
            return reject(f"recorded kernel apply root cannot be resolved: {type(exc).__name__}: {exc}")
        if not root_is_dir:
            return reject(
                f"recorded kernel apply root is not present on this host: {root}",
                code="kernel_apply_root_absent",
            )

        resolved: list[str] = []
        for target in parsed.all:
            candidate = (root / target).resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError:
                return reject(f"target escapes active root: {target}")
            if target in parsed.existing and not candidate.is_file():
                return reject(f"existing target is missing: {candidate}")
            if target in parsed.created and candidate.exists():
                return reject(f"create target already exists: {candidate}")
            resolved.append(str(candidate))
        entry.pop("resolution_error", None)
        entry.pop("resolution_reason", None)
        return resolved

    @staticmethod
    def _warm_kernel_extra_envs(entry: dict[str, Any]) -> dict[str, str]:
        """Rebuild a champion's env bundle against this run's downloaded files."""
        meta = entry.get("meta") or {}
        local = [str(path) for path in (entry.get("source_paths") or []) if path]
        envs: dict[str, str] = {}
        for key in ("extra_envs", "env_flags", "recommended_env"):
            add_kernel_envs(envs, meta.get(key))
        e2e = meta.get("e2e")
        if isinstance(e2e, dict):
            add_kernel_envs(envs, e2e.get("extra_envs"))
        point_tuner_envs_at_local_files(envs, meta.get("e2e_results"), local)
        point_envs_at_local_files(envs, local)
        return envs

    def _apply_warm_kernel_patch(self, entry: dict[str, Any], target: str) -> dict[str, Any]:
        """Land one champion's file on disk without measuring it."""
        from ..actions.executors._kernel_agent_tool import (
            _maybe_apply_kernel_patch,
            materialize_unified_patch_snapshot,
        )

        replacement = entry.get("patch_path")
        if not replacement:
            return {
                "status": "failed",
                "error": "warm replay kernel source mutation is missing its Patch",
            }
        kernel_id = str((entry.get("meta") or {}).get("kernel_name") or "warm_kernel")
        payload: dict[str, Any] = {
            "patch_path": replacement,
            "target_file": target,
            "source_file": target,
            "kernel_id": kernel_id,
        }

        patch_path = Path(str(replacement or ""))
        try:
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""
        except OSError:
            patch_text = ""
        from ..specialists.patch_safety import is_unified_diff

        if patch_text and is_unified_diff(patch_text):
            relative_target = self._parse_diff_target(str(patch_path))
            target_path = Path(target).resolve()
            relative_path = Path(relative_target)
            if not relative_target or relative_path.is_absolute() or ".." in relative_path.parts:
                return {
                    "status": "failed",
                    "error": (f"warm replay unified diff has no safe target path: {patch_path}"),
                }
            repo_root = target_path.parents[len(relative_path.parts) - 1]
            try:
                resolved_from_patch = (repo_root / relative_path).resolve()
            except (OSError, RuntimeError):
                resolved_from_patch = Path()
            if resolved_from_patch != target_path:
                return {
                    "status": "failed",
                    "error": (
                        "warm replay diff target does not resolve to target_file: "
                        f"diff={relative_target!r} target={target!r}"
                    ),
                }
            safe_kernel_id = (
                "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in kernel_id)[:96] or "warm_kernel"
            )
            snapshot_dir = Path(self.session_dir) / "runtime" / "warm_kernel_materialized" / safe_kernel_id
            try:
                materialized = materialize_unified_patch_snapshot(
                    patch_path=patch_path,
                    repo_root=repo_root,
                    snapshot_dir=snapshot_dir,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "failed",
                    "error": (f"warm replay unified diff materialization failed: {type(exc).__name__}: {exc}"),
                }
            payload["snapshot_dir"] = materialized
            payload["kernel_repo"] = str(repo_root)

        return _maybe_apply_kernel_patch(
            payload,
            session_dir=self.session_dir,
            kernel_id=kernel_id,
        )

    def _snapshot_warm_kernel_target(
        self,
        target: str,
        index: int,
    ) -> dict[str, Any]:
        """Capture one target before its mutation."""
        path = Path(target)
        if path.is_symlink():
            raise ValueError(f"kernel target must not be a symlink: {path}")
        root = Path(self.session_dir) / "runtime" / "warm_kernel_snapshots"
        root.mkdir(parents=True, exist_ok=True)
        backup = root / f"{index:04d}.bin"
        existed = path.is_file() and not path.is_symlink()
        if existed:
            backup.write_bytes(path.read_bytes())
        return {
            "target": str(path),
            "existed": existed,
            "backup": str(backup) if existed else "",
            "mode": path.stat().st_mode & 0o7777 if existed else None,
        }

    def _warm_kernel_gate_reason(self) -> str:
        """Why this run must not replay kernel champions, or '' when allowed."""
        if not self.shared_state.warm_replay_enabled:
            return "warm_replay_disabled"
        if bool(getattr(self.knowledge_plane, "kb_disabled", False)):
            return "kb_degraded"
        from ..knowledge.config import KnowledgeConfig, KnowledgeStoreMode

        config = getattr(self.knowledge_plane, "config", None) or KnowledgeConfig.from_env()
        if config.mode is not KnowledgeStoreMode.REMOTE:
            return "local_knowledge_mode"
        return ""

    @staticmethod
    def _open_warm_kernel_section() -> Any:
        """Open ``value.kernel`` from the inference Recipe already downloaded."""
        from ..knowledge.agent_kb import KernelAgentKB

        return KernelAgentKB.open()

    @staticmethod
    def _warm_replay_root_skip_outcome(
        *,
        reason: str,
        root_kind: str,
        roots: Sequence[str] = (),
        rollback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record why warm replay stopped, naming the roots the record asked for."""
        outcome: dict[str, Any] = {
            "status": ("skipped" if rollback is None or rollback.get("ok") else "rollback_failed"),
            "reason": reason,
            f"{root_kind}_patch_root_source": "recorded",
            f"{root_kind}_patch_recorded_roots": list(roots),
        }
        if rollback is not None:
            outcome["rollback"] = rollback
        return outcome

    def _warm_replay_kernel_root_block_reason(
        self,
        state: Any,
    ) -> dict[str, Any] | None:
        """Skip combined warm replay when a kernel item's recorded root is unusable."""
        for entry in state.warm_kernel_kb_plan or []:
            if not isinstance(entry, dict):
                continue
            # gemm is parameter-shaped: it re-points env vars at downloaded files and never patches a checkout, so it
            # has no root to resolve.
            if entry.get("column") == "gemm":
                continue
            if not str(entry.get("patch_path") or "").strip():
                continue
            root = str(entry.get("apply_root") or "").strip()
            if not root:
                return self._warm_replay_root_skip_outcome(
                    reason="kernel_apply_root_missing",
                    root_kind="kernel",
                )
            if not Path(root).is_dir():
                return self._warm_replay_root_skip_outcome(
                    reason="kernel_apply_root_absent",
                    root_kind="kernel",
                    roots=[root],
                )
        return None

    def _set_warm_kernel_outcome(
        self,
        kernel_outcome: dict[str, Any],
    ) -> dict[str, Any]:
        """Store kernel diagnostics inside the combined warm replay outcome."""
        combined = dict(self.shared_state.warm_replay_outcome or {})
        combined["kernel"] = dict(kernel_outcome)
        self.shared_state.warm_replay_outcome = combined
        return kernel_outcome

    def _staged_warm_kernel_plan(self, plan: list[dict[str, Any]]) -> StagedKernelPlan:
        """Resolve what each plan item contributes here, and the launch overlay they merge to."""
        item_envs: dict[int, dict[str, str]] = {}
        item_targets: dict[int, list[str]] = {}
        configs: list[tuple[str, Mapping[str, Any]]] = []
        for index, entry in enumerate(plan):
            envs = self._warm_kernel_extra_envs(entry)
            if entry.get("column") == "gemm":
                if not envs:
                    continue
            else:
                targets = self._resolve_kernel_target_paths(entry)
                if not targets:
                    continue
                item_targets[index] = targets
            item_envs[index] = envs
            configs.append(
                (
                    f"kernel[{index}]",
                    {
                        "extra_server_args": str((entry.get("meta") or {}).get("extra_server_args") or "").strip(),
                        "extra_envs": envs,
                    },
                )
            )
        args, merged_envs = merge_named_current_recipe_configs(configs)
        return StagedKernelPlan(item_envs, item_targets, args, merged_envs)

    def _preview_current_kernel_config(self, kb: Any) -> dict[str, Any]:
        """Resolve the kernel launch overlay without mutating target files."""
        staged = self._staged_warm_kernel_plan(self._collect_warm_kernel_plan(kb))
        return {"extra_server_args": staged.extra_server_args, "extra_envs": staged.extra_envs}

    async def _prepare_warm_kernel_kb(self, kb: Any = None) -> dict[str, Any]:
        """Prepare the Recipe's kernel section for the combined replay benchmark."""
        state = self.shared_state
        if state.warm_kernel_kb_attempted:
            return {"status": "skipped", "reason": "already_attempted"}
        if gated := self._warm_kernel_gate_reason():
            return self._set_warm_kernel_outcome({"status": "skipped", "reason": gated})
        state.warm_kernel_kb_attempted = True
        # Persist the one-shot flag now: a crash mid-replay must not re-run the whole (potentially hour-long) set on
        # resume.
        try:
            state.save(self.session_dir)
        except Exception as exc:
            log.warning("warm-kernel KB: one-shot state save failed", exc_info=True)
            return self._set_warm_kernel_outcome(
                {
                    "status": "error",
                    "reason": (f"kernel_attempt_state_persist_failed:{type(exc).__name__}"),
                }
            )
        if kb is None:
            try:
                kb = self._open_warm_kernel_section()
            except Exception as exc:
                log.warning("warm-kernel KB: opening Recipe section failed", exc_info=True)
                return self._set_warm_kernel_outcome(
                    {
                        "status": "skipped",
                        "reason": f"recipe_section_unavailable:{type(exc).__name__}",
                    }
                )
        if kb is None or not kb.active:
            return self._set_warm_kernel_outcome({"status": "skipped", "reason": "no_kernel_section"})
        plan = self._collect_warm_kernel_plan(kb)
        state.warm_kernel_kb_plan = plan
        if not plan:
            return self._settle_empty_warm_kernel_plan()
        staged = self._staged_warm_kernel_plan(plan)
        self._record_warm_kernel_progress([], [])
        try:
            state.save(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            return self._set_warm_kernel_outcome(
                {
                    "status": "error",
                    "reason": f"kernel_pending_state_persist_failed:{type(exc).__name__}",
                }
            )
        run = self._apply_warm_kernel_plan(plan, staged)
        outcome = self._warm_kernel_preparation_outcome(plan, staged, run)
        self._set_warm_kernel_outcome(outcome)
        log.info(
            "PRELUDE warm-kernel KB prepared: columns=%s total=%d pending=%d deferred=%d errors=%d",
            outcome["columns"],
            len(plan),
            len(outcome["pending"]),
            run.deferred,
            run.errors,
        )
        return self._persist_prepared_warm_kernel(outcome)

    def _settle_empty_warm_kernel_plan(self) -> dict[str, Any]:
        """Settle a Recipe whose kernel section named nothing this host can replay."""
        state = self.shared_state
        state.warm_replay_pending = {}
        outcome = self._set_warm_kernel_outcome({"status": "empty"})
        try:
            state.save(self.session_dir)
        except Exception:
            log.debug("warm-kernel KB: empty-state save failed", exc_info=True)
        return outcome

    def _record_warm_kernel_progress(
        self,
        applied: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
    ) -> None:
        """Carry the kernel items landed so far, and the snapshots taken for them, on the pending record."""
        state = self.shared_state
        state.warm_replay_pending = {
            **dict(state.warm_replay_pending or {}),
            "status": "preparing_kernel",
            "kernel_snapshots": list(snapshots),
            "kernel_apply_results": list(applied),
        }

    def _record_kernel_rollback_failure(self, rollback: Mapping[str, Any]) -> None:
        """Carry a kernel unwind that did not come back on the pending record."""
        state = self.shared_state
        state.warm_replay_pending = {
            **dict(state.warm_replay_pending or {}),
            "status": "rollback_failed",
            "rollback_errors": list(rollback.get("errors") or []),
        }

    def _apply_warm_kernel_plan(self, plan: list[dict[str, Any]], staged: StagedKernelPlan) -> WarmKernelApply:
        """Land every staged item in order, stopping at the first one that does not apply."""
        applied: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        errors = 0
        for index, entry in enumerate(plan):
            targets = staged.item_targets.get(index, [])
            if not self._stage_warm_kernel_entry(entry, staged.item_envs.get(index, {}), targets):
                continue
            if not self._land_warm_kernel_entry(entry, targets, applied, snapshots):
                errors = 1
                break
        deferred = sum(1 for entry in plan if entry.get("decision") == "DEFERRED")
        return WarmKernelApply(applied, snapshots, deferred, errors)

    @staticmethod
    def _stage_warm_kernel_entry(entry: dict[str, Any], envs: dict[str, str], targets: list[str]) -> bool:
        """Record what an item contributes, and say whether it still has to land on disk."""
        if not (entry.get("source_paths") or entry.get("patch_path")):
            entry["decision"] = "DEFERRED"
            return False
        # Every column carries its env bundle: a fusion needs its switches to activate the patched path, and a
        # GEMM is nothing but its bundle.
        if envs:
            entry["extra_envs"] = envs
        # GEMM is parameter-shaped: its env bundle is the whole deliverable, so there is nothing to stage on disk.
        if entry.get("column") == "gemm":
            entry["decision"] = "PENDING" if envs else "DEFERRED"
            return False
        if not targets:
            entry["decision"] = "DEFERRED"
            if entry.get("resolution_error"):
                entry["apply_result"] = {"status": "skipped", "reason": str(entry["resolution_error"])[:300]}
            return False
        entry["resolved_patch_targets"] = list(targets)
        return True

    def _land_warm_kernel_entry(
        self,
        entry: dict[str, Any],
        targets: list[str],
        applied: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
    ) -> bool:
        """Snapshot one item's targets and apply it; False once it has failed to land."""
        state = self.shared_state
        try:
            for target_path in targets:
                snapshots.append(self._snapshot_warm_kernel_target(target_path, len(snapshots)))
            self._record_warm_kernel_progress(applied, snapshots)
            state.save(self.session_dir)
        except Exception as exc:
            log.warning("warm-kernel KB: pre-mutation snapshot persist failed for %s", targets, exc_info=True)
            return refuse_warm_kernel_entry(entry, exc)
        try:
            apply_result = self._apply_warm_kernel_patch(entry, targets[0])
        except Exception as exc:
            log.warning("warm-kernel KB: apply failed for %s", targets[0], exc_info=True)
            return refuse_warm_kernel_entry(entry, exc)
        entry["apply_result"] = {
            key: apply_result.get(key) for key in ("status", "reason", "error", "manifest_path") if key in apply_result
        }
        if apply_result.get("status") != "ok":
            entry["decision"] = "ERROR"
            return False
        applied.append(apply_result)
        self._record_warm_kernel_progress(applied, snapshots)
        entry["decision"] = "PENDING"
        return True

    def _warm_kernel_preparation_outcome(
        self,
        plan: list[dict[str, Any]],
        staged: StagedKernelPlan,
        run: WarmKernelApply,
    ) -> dict[str, Any]:
        """What the preparation left on disk, and the launch overlay the pending items earned."""
        state = self.shared_state
        pending = [entry for entry in plan if entry.get("decision") == "PENDING"]
        applied = run.applied
        rollback: dict[str, Any] = {"ok": True, "errors": []}
        if run.errors:
            rollback = revert_warm_kernel_patches(run.applied, run.snapshots)
            for entry in pending:
                entry["decision"] = "ROLLED_BACK"
            if rollback.get("ok"):
                applied = []
                pending = []
                state.warm_replay_pending = {}
            else:
                self._record_kernel_rollback_failure(rollback)
        restored = bool(rollback.get("ok"))
        status = "rollback_failed" if not restored else "error" if run.errors else "prepared" if pending else "loaded"
        if status == "loaded" and not applied and not run.snapshots:
            state.warm_replay_pending = {}
        return {
            "status": status,
            "columns": sorted({str(entry.get("column")) for entry in plan}),
            "total": len(plan),
            "deferred": run.deferred,
            "errors": run.errors,
            "pending": pending,
            "applied": applied,
            "snapshots": run.snapshots,
            "rollback": rollback,
            "dirty": not restored,
            "extra_envs": staged.extra_envs if pending else {},
            "extra_server_args": (staged.extra_server_args if pending else ""),
        }

    def _persist_prepared_warm_kernel(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Persist the prepared kernel half, unwinding it when the record cannot be written."""
        try:
            self.shared_state.save(self.session_dir)
        except Exception as exc:
            log.warning("warm-kernel KB: prepared state save failed", exc_info=True)
            return self._unwind_unpersisted_warm_kernel(outcome, exc)
        return outcome

    def _unwind_unpersisted_warm_kernel(self, outcome: dict[str, Any], exc: BaseException) -> dict[str, Any]:
        """Put back a prepared kernel half whose record could not be written."""
        state = self.shared_state
        applied = list(outcome.get("applied") or [])
        rollback = revert_warm_kernel_patches(applied, list(outcome.get("snapshots") or []))
        for entry in outcome.get("pending") or []:
            entry["decision"] = "ROLLED_BACK"
        restored = bool(rollback.get("ok"))
        if restored:
            state.warm_replay_pending = {}
        else:
            self._record_kernel_rollback_failure(rollback)
        settled = {
            **outcome,
            "status": "error" if restored else "rollback_failed",
            "reason": (f"kernel_prepared_state_persist_failed:{type(exc).__name__}"),
            "pending": [],
            "applied": [] if restored else applied,
            "rollback": rollback,
            "dirty": not restored,
            "extra_envs": {},
            "extra_server_args": "",
        }
        self._set_warm_kernel_outcome(settled)
        try:
            state.save(self.session_dir)
        except Exception:
            log.warning("warm-kernel KB: rollback state save failed", exc_info=True)
        return settled

    @staticmethod
    def _is_current_remote_recipe(warm: Any) -> bool:
        from ..knowledge.remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

        recipe = warm.get("recipe") if isinstance(warm, Mapping) else None
        return bool(isinstance(recipe, Mapping) and recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE)

    def _read_current_recipe_replay(self) -> dict[str, Any]:
        """Load current replay data exclusively through the column facades."""
        from ..knowledge.agent_kb import ConfigKB, KernelAgentKB, PatchKB

        config_kb = ConfigKB.open()
        patch_kb = PatchKB.open()
        kernel = KernelAgentKB.open()
        if not all((config_kb.active, patch_kb.active, kernel.active)):
            raise ValueError("current Recipe column facades are unavailable")

        config = config_kb.read()
        args = str(config.get("extra_server_args") or "")
        envs = dict(config.get("extra_envs") or {})
        kernel_config = self._preview_current_kernel_config(kernel)
        combined_args, combined_envs = merge_named_current_recipe_configs(
            [
                ("config", config),
                ("kernel", kernel_config),
            ]
        )
        # The column records its overlays in replay order, so the recorded order is the order they are applied in.
        timeline = patch_kb.read_patches()
        if len(timeline) != len(set(timeline)):
            raise ValueError("current Recipe patch refs contain a duplicate")
        provenance = patch_kb.read_provenance()
        apply_roots = recorded_apply_roots(provenance)
        patches: list[dict[str, Any]] = []
        for index, ref in enumerate(timeline):
            source = patch_kb.prior_file(ref)
            if source is None:
                raise ValueError(f"current Recipe patch artifact is unavailable: {ref!r}")
            entry: dict[str, Any] = {
                "patch_file": ref,
                "patch_ref": str(source),
                "patch_content": "",
                "measured_gain_pct": 1e-6,
                "required": True,
                "timeline_index": index,
            }
            # Each overlay carries the checkout it was applied into, so a Recipe whose KEEPs came from different trees
            # replays against each of them.
            if recorded_root := apply_roots.get(ref, ""):
                entry["framework_root"] = recorded_root
            patches.append(entry)
        return {
            "extra_server_args": args,
            "extra_envs": envs,
            "combined_extra_server_args": combined_args,
            "combined_extra_envs": combined_envs,
            "kernel_config": kernel_config,
            "timeline": timeline,
            "patches": patches,
            "kernel_kb": kernel,
            "provenance": provenance,
        }

    async def maybe_enqueue_warm_replay(
        self,
        *,
        baseline_tput: float,
    ) -> "Task | None":
        """Enqueue a one-shot ``replay_warm_recipe`` task for a high-confidence T0 prior."""
        state = self.shared_state
        if not state.warm_replay_enabled:
            # The guard is flipped even on a disabled-skip so a resume without
            # --no-warm-replay cannot retroactively trigger a replay against
            # the operator's original intent.
            self._skip_warm_replay(
                code=SKIP_DISABLED_BY_FLAG,
                outcome={"status": "skipped", "reason": "disabled_by_flag"},
            )
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued/ran the replay.
            return None
        warm = state.warm_start_recipe if isinstance(state.warm_start_recipe, dict) else {}
        sdk_replay = self._read_replayable_current_recipe(warm) if self._is_current_remote_recipe(warm) else {}
        if sdk_replay is None:
            return None
        kernel = await self._prepared_warm_kernel_section(sdk_replay)
        if self._refuse_dirty_kernel_preparation(kernel) or self._refuse_absent_kernel_roots():
            return None
        request = self._warm_replay_request(warm, sdk_replay, kernel)
        if request is None:
            return None
        if self._refuse_unusable_framework_roots(request) or self._refuse_empty_warm_launch(request):
            return None
        launch = self._combined_warm_launch(request, sdk_replay)
        if self._skip_restated_replay(
            launch.args,
            launch.envs,
            [*request.patches, *request.overlay.pending],
            request.identity,
        ):
            return None
        compatibility = self._workload_compatible_launch(launch.args, request)
        if compatibility is None:
            return None
        task = await self._enqueue_warm_replay_task(request, launch, compatibility, baseline_tput)
        self._record_enqueued_warm_replay(request, task, sdk_replay)
        return task

    def _read_replayable_current_recipe(self, warm: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read a current Recipe's replay, or None once the refusal has been published."""
        metadata = warm.get("recipe") or {}
        if isinstance(metadata, Mapping) and metadata.get("replayable") is False:
            self._skip_warm_replay(
                code=SKIP_RECIPE_NOT_REPLAYABLE,
                outcome={
                    "status": "skipped",
                    "reason": str(metadata.get("replay_disabled_reason") or "remote_recipe_view_not_replayable"),
                    "view_source": str(metadata.get("view_source") or ""),
                },
                details={"view_source": str(metadata.get("view_source") or "")},
            )
            return None
        try:
            sdk_replay = self._read_current_recipe_replay()
        except Exception as exc:  # noqa: BLE001 — current replay fails closed
            self._skip_warm_replay(
                code=SKIP_RECIPE_READ_FAILED,
                outcome={
                    "status": "skipped",
                    "reason": (f"current_recipe_sdk_read_failed:{type(exc).__name__}:{exc}")[:500],
                },
                details={"error_class": type(exc).__name__},
            )
            return None
        # Every overlay names the checkout it was measured on.
        roots = recorded_framework_roots(sdk_replay.get("patches") or [])
        if not roots.recorded:
            return sdk_replay
        if not roots.complete:
            self._refuse_framework_roots(ROOT_MISSING, roots.named, None)
            return None
        if roots.absent:
            self._refuse_framework_roots(ROOT_ABSENT, roots.absent, None)
            return None
        return sdk_replay

    async def _prepared_warm_kernel_section(self, sdk_replay: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare the prior's kernel half, reporting a preparation that raised as an error outcome.

        A current Recipe hands over the kernel section its own read already
        opened; any other prior leaves the preparation to open one.
        """
        try:
            if opened_section := sdk_replay.get("kernel_kb"):
                return await self._prepare_warm_kernel_kb(opened_section)
            return await self._prepare_warm_kernel_kb()
        except Exception as exc:
            log.warning("PRELUDE: warm-kernel preparation failed", exc_info=True)
            return {"status": "error", "errors": 1, "reason": f"{type(exc).__name__}: {exc}"[:300]}

    def _refuse_dirty_kernel_preparation(self, kernel: Mapping[str, Any]) -> bool:
        """Refuse a replay whose kernel preparation failed with mutable state left behind."""
        pending = dict(self.shared_state.warm_replay_pending or {})
        dirty = kernel.get("dirty") or pending.get("kernel_snapshots") or pending.get("kernel_apply_results")
        if str(kernel.get("status") or "") not in {"error", "rollback_failed"} or not dirty:
            return False
        recorded = kernel.get("rollback")
        rollback = dict(recorded) if isinstance(recorded, Mapping) else {}
        if not rollback:
            rollback = revert_warm_kernel_patches(
                list(pending.get("kernel_apply_results") or []),
                list(pending.get("kernel_snapshots") or []),
            )
        self._settle_kernel_rollback(rollback)
        self._skip_warm_replay(
            code=SKIP_KERNEL_PREPARATION_FAILED,
            outcome={
                "status": ("kernel_preparation_failed" if rollback.get("ok") else "rollback_failed"),
                "reason": str(kernel.get("reason") or "kernel preparation left mutable state"),
                "rollback": rollback,
            },
            details={"kernel_status": str(kernel.get("status") or "")},
        )
        return True

    def _refuse_absent_kernel_roots(self) -> bool:
        """Refuse a replay whose kernel items record a checkout this host does not have."""
        block = self._warm_replay_kernel_root_block_reason(self.shared_state)
        if block is None:
            return False
        self._skip_warm_replay(
            code=SKIP_KERNEL_ROOT_MISSING,
            outcome=block,
            details={
                "root_kind": "kernel",
                "recorded_roots": list(block.get("kernel_patch_recorded_roots") or []),
            },
        )
        return True

    def _settle_kernel_rollback(self, rollback: Mapping[str, Any]) -> None:
        """Record what a kernel unwind left behind, stopping the run when a target did not come back."""
        if rollback.get("ok"):
            self.shared_state.warm_replay_pending = {}
            return
        self._record_kernel_rollback_failure(rollback)
        self.shared_state.set_stop_reason("warm_replay_rollback_failed")

    def _unwind_warm_kernel_half(self, overlay: WarmKernelOverlay) -> dict[str, Any]:
        """Put back whatever the kernel preparation applied, and say whether it came back."""
        rollback = (
            revert_warm_kernel_patches(overlay.applied, overlay.snapshots)
            if overlay.applied or overlay.snapshots
            else {"ok": True, "errors": []}
        )
        self._settle_kernel_rollback(rollback)
        return rollback

    @staticmethod
    def _warm_kernel_overlay(kernel: Mapping[str, Any], launch: Mapping[str, Any]) -> WarmKernelOverlay:
        """The kernel half a combined replay carries, and the launch overlay it brings.

        ``launch`` is the current Recipe's own kernel config where it has one, so
        a Recipe that recorded its overlay replays that rather than re-deriving it.
        """
        if kernel.get("status") != "prepared" or not kernel.get("pending"):
            return WarmKernelOverlay([], [], [], "", {})
        return WarmKernelOverlay(
            list(kernel.get("pending") or []),
            list(kernel.get("applied") or []),
            list(kernel.get("snapshots") or []),
            str(launch.get("extra_server_args") or ""),
            dict(launch.get("extra_envs") or {}),
        )

    def _warm_replay_request(
        self,
        warm: Mapping[str, Any],
        sdk_replay: Mapping[str, Any],
        kernel: Mapping[str, Any],
    ) -> WarmReplayRequest | None:
        """What the prior asks this replay to carry, or None once a refusal has been published."""
        overlay = self._warm_kernel_overlay(kernel, sdk_replay.get("kernel_config") or kernel)
        if not warm and not overlay.pending:
            if str(kernel.get("status") or "") in {"empty", "loaded"}:
                self.shared_state.warm_replay_pending = {}
            self._skip_warm_replay(
                code=SKIP_NO_WARM_START_RECIPE,
                outcome={"status": "skipped", "reason": "no_warm_start_recipe"},
            )
            return None
        recipe = warm.get("recipe") if isinstance(warm.get("recipe"), dict) else {}
        recipe_attrs = recipe.get("attrs") or recipe
        # tier/conf stamped at T0.
        tier = str(warm.get("tier") or "").strip()
        confidence = as_float(warm.get("confidence"))
        donor_replay = self._recommended_replay(sdk_replay)
        identity = {
            "warm_recipe_tier": tier,
            "warm_recipe_conf": confidence,
            **warm_replay_donor_fields(donor_replay),
        }
        config = self._apply_confidence_gate(
            warm_replay_launch_config(recipe, sdk_replay, donor_replay, confidence),
            overlay,
            identity,
        )
        if config is None:
            return None
        patches, required_timeline = warm_replay_patch_plan(sdk_replay, donor_replay, recipe_attrs, config)
        return WarmReplayRequest(
            config=config,
            overlay=overlay,
            patches=patches,
            required_patch_timeline=required_timeline,
            tier=tier,
            confidence=confidence,
            expected_gain=warm_expected_gain(recipe_attrs, config.donor_expected_gain),
            donor=warm_replay_donor_fields(donor_replay),
        )

    def _recommended_replay(self, sdk_replay: Mapping[str, Any]) -> dict[str, Any]:
        """The donor's recommended replay from the warm-start context; empty for a current Recipe."""
        context = self.shared_state.warm_start_context
        if sdk_replay or not isinstance(context, dict):
            return {}
        recommended = context.get("recommended_replay")
        return recommended if isinstance(recommended, dict) else {}

    def _apply_confidence_gate(
        self,
        config: WarmReplayConfig,
        overlay: WarmKernelOverlay,
        identity: Mapping[str, Any],
    ) -> WarmReplayConfig | None:
        """Suppress a low-confidence config, refusing outright when no kernel item would survive it.

        Low-confidence config/patch content is suppressed, but the kernel section
        from the same Recipe can still receive a combined check.
        """
        min_conf = float(self.shared_state.warm_replay_min_confidence)
        if config.confidence >= min_conf:
            return config
        if overlay.pending:
            return config._replace(
                args="",
                envs={},
                source="",
                donor_tier=SUPPRESSED_LOW_CONFIDENCE,
                donor_expected_gain=0.0,
                suppressed=True,
            )
        self._skip_warm_replay(
            code=SKIP_CONFIDENCE_BELOW_THRESHOLD,
            outcome={
                "status": "skipped",
                "reason": f"confidence_below_threshold ({config.confidence:.2f} < {min_conf:.2f})",
                **identity,
                "config_donor_tier": config.donor_tier,
                "config_source": config.source,
            },
            details={"observed": config.confidence, "threshold": min_conf},
        )
        return None

    def _refuse_framework_roots(
        self,
        reason: str,
        roots: Sequence[str],
        rollback: dict[str, Any] | None,
    ) -> None:
        """Publish the refusal of a replay whose overlays name a checkout this host cannot use."""
        self._skip_warm_replay(
            code=SKIP_FRAMEWORK_ROOT_MISSING,
            outcome=self._warm_replay_root_skip_outcome(
                reason=reason,
                root_kind="framework",
                roots=roots,
                rollback=rollback,
            ),
            details={"root_kind": "framework", "recorded_roots": list(roots)},
        )

    def _refuse_unusable_framework_roots(self, request: WarmReplayRequest) -> bool:
        """Refuse a replay whose code overlays name a checkout this host cannot use.

        Re-checked here because a persisted plan may be resumed under a different
        image, and because the kernel half is already applied by now: a root that
        has since gone means unwinding that too.
        """
        roots = recorded_framework_roots(request.patches)
        if roots.complete and not roots.absent:
            return False
        rollback = self._unwind_warm_kernel_half(request.overlay)
        self._refuse_framework_roots(
            ROOT_ABSENT if roots.absent else ROOT_MISSING,
            roots.absent or roots.named,
            rollback,
        )
        return True

    def _refuse_empty_warm_launch(self, request: WarmReplayRequest) -> bool:
        """Refuse a replay that would launch nothing: no config, no overlay, no kernel item."""
        if request.config.args or request.config.envs or request.patches or request.overlay.pending:
            return False
        self._skip_warm_replay(
            code=SKIP_BEST_CONFIG_EMPTY,
            outcome={"status": "skipped", "reason": "best_config_empty", **request.identity},
        )
        return True

    @staticmethod
    def _combined_warm_launch(request: WarmReplayRequest, sdk_replay: Mapping[str, Any]) -> WarmLaunch:
        """The launch the replay runs with: the prior's config and its kernel overlay, merged."""
        overlay = request.overlay
        if sdk_replay:
            if request.config.suppressed:
                return WarmLaunch(overlay.args, dict(overlay.envs))
            return WarmLaunch(
                str(sdk_replay.get("combined_extra_server_args") or ""),
                dict(sdk_replay.get("combined_extra_envs") or {}),
            )
        envs = dict(request.config.envs)
        envs.update(overlay.envs)
        return WarmLaunch(merge_cumulative_extra_server_args(request.config.args, overlay.args, ""), envs)

    def _workload_compatible_launch(self, args: str, request: WarmReplayRequest) -> dict[str, Any] | None:
        """How the launch fits this session's workload, or None once the refusal has been published."""
        from hyperloom.inference_optimizer.grid_server_args import (
            validate_warm_replay_context_length,
        )

        state = self.shared_state
        try:
            validated_args, workload_compatibility = validate_warm_replay_context_length(
                args,
                state.framework,
                int(state.isl or 0),
                int(state.osl or 0),
                state.max_model_len or None,
            )
            if validated_args != args:
                raise RuntimeError("warm replay context preflight must not mutate config")
        except (ValueError, RuntimeError) as exc:
            rollback = self._unwind_warm_kernel_half(request.overlay)
            target_workload_shape = {
                "conc": int(state.conc or 0),
                "isl": int(state.isl or 0),
                "osl": int(state.osl or 0),
            }
            self._skip_warm_replay(
                code=SKIP_WORKLOAD_CONFIG_INCOMPATIBLE,
                outcome={
                    "status": ("skipped" if rollback.get("ok") else "rollback_failed"),
                    "reason": (f"workload_config_incompatible:{type(exc).__name__}:{exc}")[:500],
                    "target_workload_shape": target_workload_shape,
                    "rollback": rollback,
                },
                details={
                    "error_class": type(exc).__name__,
                    "target_workload_shape": dict(target_workload_shape),
                },
            )
            return None
        return workload_compatibility

    def _warm_replay_task_params(
        self,
        request: WarmReplayRequest,
        launch: WarmLaunch,
        workload_compatibility: dict[str, Any],
        baseline_tput: float,
    ) -> dict[str, Any]:
        """The params a ``replay_warm_recipe`` task runs with."""
        config = request.config
        overlay = request.overlay
        return {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "extra_server_args": launch.args,
            "extra_envs": launch.envs,
            "recipe_extra_server_args": config.args,
            "recipe_extra_envs": dict(config.envs),
            "kernel_extra_server_args": overlay.args,
            "kernel_extra_envs": overlay.envs,
            # Reuse the baseline's workload contract (else YAML smoke defaults).
            "config_path": str(self.shared_state.baseline_config_path or ""),
            # Historical-gain anchor for the promote path's reproduce ratio.
            "warm_expected_gain_pct": request.expected_gain,
            "warm_recipe_tier": request.tier,
            "warm_recipe_conf": request.confidence,
            # Config provenance ("self" when the identity match owned it).
            "config_donor_tier": config.donor_tier,
            "config_source": config.source,
            "baseline_tput_anchor": float(baseline_tput),
            # Code patches to apply before server launch.
            "patches": list(request.patches),
            "required_patch_timeline": request.required_patch_timeline,
            "warm_kernel_plan": overlay.pending,
            "warm_kernel_apply_results": overlay.applied,
            "warm_kernel_snapshots": overlay.snapshots,
            "workload_compatibility": workload_compatibility,
        }

    async def _enqueue_warm_replay_task(
        self,
        request: WarmReplayRequest,
        launch: WarmLaunch,
        workload_compatibility: dict[str, Any],
        baseline_tput: float,
    ) -> "Task":
        """Dispatch the replay, unwinding the kernel half if the queue would not take it."""
        params = self._warm_replay_task_params(request, launch, workload_compatibility, baseline_tput)
        try:
            lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("replay_warm_recipe")
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="replay_warm_recipe",
                params=params,
                idempotency_key="warm-replay-prelude",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        except Exception as exc:
            rollback = self._unwind_warm_kernel_half(request.overlay)
            self._skip_warm_replay(
                code=SKIP_ENQUEUE_FAILED,
                outcome={
                    "status": ("enqueue_failed" if rollback.get("ok") else "rollback_failed"),
                    "reason": f"warm replay enqueue failed: {type(exc).__name__}",
                    "rollback": rollback,
                },
                details={"error_class": type(exc).__name__},
            )
            raise
        if not was_existing:
            log.info(
                "PRELUDE: warm-replay enqueued task=%s (tier=%s conf=%.2f expected_gain=%.2f baseline_tput=%.2f)",
                task.task_id,
                request.tier,
                request.confidence,
                request.expected_gain,
                baseline_tput,
            )
        return task

    def _record_enqueued_warm_replay(
        self,
        request: WarmReplayRequest,
        task: "Task",
        sdk_replay: Mapping[str, Any],
    ) -> None:
        """Flip the one-shot guard onto an in-flight replay and open its timeline event."""
        state = self.shared_state
        state.warm_replay_attempted = True
        state.warm_replay_outcome = {
            "status": "in_flight",
            # Bounds the replay for the SBD timeline; the terminal write below stamps ``settled_at`` against this.
            "enqueued_at": now_iso(timespec="seconds"),
            "warm_recipe_tier": request.tier,
            "warm_recipe_conf": request.confidence,
            "config_donor_tier": request.config.donor_tier,
            "config_source": request.config.source,
            "expected_gain_pct": request.expected_gain,
            "replay_task_id": task.task_id,
            "kernel_count": len(request.overlay.pending),
            "recipe_suppressed": request.config.suppressed,
            **({"overlay_provenance": summary} if (summary := overlay_provenance_summary(sdk_replay)) else {}),
            **request.donor,
        }
        state.warm_replay_pending = {
            **dict(state.warm_replay_pending or {}),
            "status": "in_flight",
            "task_id": task.task_id,
            "kernel_apply_results": request.overlay.applied,
            "kernel_plan": request.overlay.pending,
            "kernel_snapshots": request.overlay.snapshots,
        }
        try:
            state.save(self.session_dir)
        except Exception:
            log.debug("combined warm replay pending save failed", exc_info=True)
        # Opened after the outcome is persisted so the request block reads the
        # donor identity the outcome just stamped, and a session killed between
        # the two is recovered as an in-flight replay of a named recipe.
        self._open_warm_replay_timeline(task=task, session_baseline_tput=state.baseline_tput)

    def _resolve_promoted_recipe_checkout(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> tuple[bool, dict[str, Any]]:
        """Validate every already-patched checkout selected by warm replay."""
        params = (task.params if task is not None else {}) or {}
        if not params.get("required_patch_timeline") or not params.get("patches"):
            return True, {"status": "not_required"}

        promoted: list[str] = []
        for tree in _replayed_patch_trees(result, {}):
            resolved = _promotable_tree(tree)
            if resolved.failure is not None:
                return False, resolved.failure
            promoted.append(resolved.root)
        return True, {
            "status": "promoted",
            "target_repo": promoted[0],
            "target_repos": promoted,
        }

    def rollback_combined_warm(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> dict[str, Any]:
        """Restore both Recipe and Kernel halves of a combined replay."""
        pending = self.shared_state.warm_replay_pending or {}
        params = (task.params if task is not None else {}) or {}
        restores = [
            restore
            for tree in _replayed_patch_trees(result, pending)
            if (restore := _restore_replayed_patch_tree(tree)) is not None
        ]
        half = _recorded_kernel_half(result, params, pending)
        restores.append(revert_warm_kernel_patches(half.applied, half.snapshots))
        errors = [
            error
            for restore in restores
            if not restore.get("ok")
            for error in (restore.get("errors") or ["unknown_restore_failure"])
        ]
        if errors:
            self.shared_state.warm_replay_pending = {
                **dict(pending),
                "status": "rollback_failed",
                "rollback_errors": errors,
            }
        else:
            self.shared_state.warm_replay_pending = {}
        return {"ok": not errors, "errors": errors}

    def _book_combined_kernel_keep(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> dict[str, Any]:
        """Record the kernel half as accepted by the one Recipe replay check."""
        params = (task.params if task is not None else {}) or {}
        plan = [dict(entry) for entry in (params.get("warm_kernel_plan") or []) if isinstance(entry, dict)]
        for entry in plan:
            entry["decision"] = "KEEP"
            entry["gain_pct"] = result.get("combined_gain_pct")
        outcome = {
            "status": "kept" if plan else "empty",
            "total": len(plan),
            "kept": len(plan),
            "reverted": 0,
            "validation": "combined_recipe_kernel",
        }
        self.shared_state.warm_kernel_kb_plan = plan
        self._set_warm_kernel_outcome(outcome)
        return outcome

    def _require_combined_warm_rollback(self, settlement: _WarmReplaySettlement) -> bool:
        """Roll back a rejected replay, settling it as ``rollback_failed`` when a tree did not come back.

        The rollback is recorded here rather than at each of the branches that
        unwind a rejected replay: what a reader needs is whether the trees came
        back. Returns ``True`` when every tree was restored.
        """
        outcome = settlement.outcome
        rollback = self.rollback_combined_warm(settlement.result, settlement.task)
        if settlement.recorder is not None:
            settlement.recorder.record_rollback(ok=rollback.get("ok"), errors=rollback.get("errors"))
        if rollback.get("ok"):
            return True
        outcome["status"] = "rollback_failed"
        outcome["reason"] = "combined warm replay rollback failed"
        outcome["rollback"] = rollback
        kernel_outcome = {
            "status": "rollback_failed",
            "validation": "combined_recipe_kernel",
            "errors": list(rollback.get("errors") or []),
        }
        outcome["kernel"] = kernel_outcome
        self.shared_state.set_stop_reason("warm_replay_rollback_failed")
        self._conclude_warm_replay(outcome, Verdict.FAILED)
        return False

    @staticmethod
    def _replay_eval_search_root(result: dict) -> Path | None:
        """Directory holding every round of this replay task."""
        from ..actions.executors.baseline import (
            _MEASURE_ROUND_DIR,
            _WARMUP_ROUND_DIR,
        )

        round_dirs = {_WARMUP_ROUND_DIR, _MEASURE_ROUND_DIR}
        for key in ("output_dir", "workspace"):
            raw = str(result.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw)
            for candidate in (path, *path.parents):
                if candidate.name in round_dirs:
                    return candidate.parent
            return path
        return None

    def _read_replay_accuracy(self, result: dict) -> _AccuracyReading:
        """Read what the replay scored, from the result or from the eval output it left behind."""
        from ..actions.executors._accuracy_gate import parse_eval_results

        reported = result.get("accuracy")
        if isinstance(reported, (int, float)):
            return _AccuracyReading(float(reported), True, "")
        root = self._replay_eval_search_root(result)
        if root is None:
            return _AccuracyReading(None, False, "replay result names no round directory")
        try:
            eval_out = parse_eval_results(root)
        except Exception as exc:  # noqa: BLE001 — an unreadable eval is "no verdict"
            eval_out = {"error": f"eval parse raised: {type(exc).__name__}"}
        parsed = eval_out.get("accuracy")
        if isinstance(parsed, (int, float)):
            return _AccuracyReading(float(parsed), True, "")
        error = str(eval_out.get("error") or "no accuracy in eval output")
        # Only a results file the parser reached can say the eval ran; a parser that never got one says
        # nothing either way.
        return _AccuracyReading(None, error.startswith(_EVAL_RAN_BUT_UNSCORABLE), error)

    def _judge_warm_replay_accuracy(self, settlement: _WarmReplaySettlement) -> Verdict | None:
        """Rule on whether a replayed config may be promoted on accuracy grounds.

        Every replay is judged, not just the ones touching a knob known to be
        risky: a KB recipe is evidence from another session, so reproducing its
        throughput says nothing about whether it still computes correctly here.
        ``eval_ran`` separates the two ways ``replay_accuracy`` can be absent:
        a score of 0.0 means the model answered nothing, while no score at all
        means no evidence either way.

        Returns ``None`` when promotion may proceed; otherwise the verdict the
        caller must stop on (the rollback and outcome have already been recorded).
        """
        outcome = settlement.outcome
        baseline_accuracy = as_float(self.shared_state.baseline_accuracy)
        reading = self._read_replay_accuracy(settlement.result)
        outcome["eval_ran"] = bool(reading.eval_ran)
        outcome["eval_error"] = reading.error or None
        outcome["replay_accuracy"] = reading.score
        outcome["baseline_accuracy"] = baseline_accuracy if baseline_accuracy > 0 else None
        if reading.score is None:
            self._admit_unjudged_replay_accuracy(settlement, reading, baseline_accuracy)
            return None
        reason = self._record_accuracy_gate(settlement.recorder, reading.score, baseline_accuracy)
        if not reason:
            return None
        if not self._require_combined_warm_rollback(settlement):
            return Verdict.FAILED
        outcome["status"] = "accuracy_failed"
        outcome["reason"] = reason
        log.info("warm-replay REJECTED on accuracy: %s", reason)
        return self._conclude_warm_replay(outcome, Verdict.REVERTED)

    @staticmethod
    def _admit_unjudged_replay_accuracy(
        settlement: _WarmReplaySettlement,
        reading: _AccuracyReading,
        baseline_accuracy: float,
    ) -> None:
        """Admit a replay no accuracy verdict could be read for, and say why there is none.

        A measurement that failed is not evidence the config broke the model, so
        it must not stop the run. Recording a pass here would claim evidence
        there is none, so the gate is recorded as having ruled nothing.
        """
        log.warning(
            "warm-replay admitted without an accuracy verdict (eval_ran=%s, baseline %.4f): %s",
            reading.eval_ran,
            baseline_accuracy,
            reading.error or "no reason recorded",
        )
        if settlement.recorder is not None:
            settlement.recorder.record_gate(
                GATE_ACCURACY,
                passed=None,
                reason=reading.error or "no accuracy verdict",
                threshold=baseline_accuracy if baseline_accuracy > 0 else None,
            )

    @staticmethod
    def _record_accuracy_gate(recorder: Any, measured: float, baseline_accuracy: float) -> str:
        """Put the accuracy ruling on the timeline; the reason it rejected, or '' when it passed."""
        from ..actions.executors._accuracy_gate import (
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            accuracy_meets_floor,
            accuracy_passed,
        )

        if baseline_accuracy > 0:
            passed = accuracy_passed(baseline_accuracy, float(measured))
            threshold = baseline_accuracy
            reason = (
                "no regression against the baseline reference"
                if passed
                else f"accuracy regression on the replayed config (baseline {baseline_accuracy:.4f}, replay {measured:.4f})"
            )
        else:
            passed = accuracy_meets_floor(measured, DEFAULT_ENABLEMENT_ACCURACY_FLOOR)
            threshold = DEFAULT_ENABLEMENT_ACCURACY_FLOOR
            reason = (
                "clears the absolute floor with no baseline to compare against"
                if passed
                else "accuracy below absolute floor on the replayed config "
                f"(replay {measured:.4f}, "
                f"floor {DEFAULT_ENABLEMENT_ACCURACY_FLOOR:.2f})"
            )
        if recorder is not None:
            recorder.record_gate(GATE_ACCURACY, passed=passed, reason=reason, observed=measured, threshold=threshold)
        return "" if passed else reason

    # ---- warm-replay timeline recording ----------------------------------
    # The replay spans two ticks: one enqueues the task, a later one harvests it.
    # The event id derives from persisted state alone, so the promote seam
    # rebinds to the event the enqueue seam opened rather than holding a
    # recorder across a boundary a resume does not survive.

    @staticmethod
    def _warm_replay_donor(source: Mapping[str, Any]) -> dict[str, Any]:
        """Lift the donor block out of a flat ``donor_*`` mapping; empty when not borrowed."""
        return {field.removeprefix("donor_"): value for field, value in warm_replay_donor_fields(source).items()}

    def _open_warm_replay_timeline(self, *, task: "Task | None", session_baseline_tput: Any) -> None:
        """Open the replay's event at the moment its task is dispatched.

        ``session_baseline_tput`` is kept beside the enqueue anchor so a reader
        can see the two diverge across a re-baseline.
        """
        params = dict(getattr(task, "params", None) or {})
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        recorder = make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            task_id=str(getattr(task, "task_id", "") or ""),
            tier=str(params.get("warm_recipe_tier") or ""),
            config_source=str(params.get("config_source") or ""),
            config_donor_tier=str(params.get("config_donor_tier") or ""),
            donor=self._warm_replay_donor(dict(self.shared_state.warm_replay_outcome or {})) or None,
            expected_gain_pct=params.get("warm_expected_gain_pct"),
            confidence=params.get("warm_recipe_conf"),
            session_baseline_tput=session_baseline_tput,
            kernel_count=len(list(params.get("warm_kernel_plan") or [])),
            recipe_suppressed=not str(params.get("config_source") or ""),
        )
        if recorder is not None:
            self._record_warm_kernel_apply_items(recorder)

    def _record_warm_kernel_apply_items(self, recorder: Any) -> None:
        """State which kernel items the replay carried and which of them landed.

        Recorded as the replay is dispatched, because the ruling replaces the
        plan with the subset it kept: by the time the bench has a verdict, the
        items that never applied are gone from state and unrecoverable.

        ``PENDING`` is an item that applied and awaits the bench. ``DEFERRED``
        resolved to no target under this checkout. ``ERROR`` stops the
        sequence, so the items behind it hold no decision and record nothing --
        they were never attempted.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import APPLY_KERNEL

        for position, entry in enumerate(self.shared_state.warm_kernel_kb_plan or []):
            if not isinstance(entry, Mapping) or not str(entry.get("decision") or ""):
                continue
            outcome = entry.get("apply_result") if isinstance(entry.get("apply_result"), Mapping) else {}
            sources = entry.get("source_paths") or []
            recorder.record_apply_item(
                str(entry.get("patch_path") or (sources[0] if sources else "")),
                kind=APPLY_KERNEL,
                position=position,
                applied=str(entry.get("decision") or "") == "PENDING",
                reason=str(
                    outcome.get("reason")
                    or outcome.get("error")
                    or outcome.get("exception")
                    or entry.get("resolution_error")
                    or ""
                ),
                target=str(entry.get("apply_root") or ""),
            )

    def _record_warm_patch_apply_items(self, recorder: Any, result: Mapping[str, Any]) -> None:
        """State which of the recipe's code patches landed.

        The executor reports one status per patch it reached, keyed by the
        timeline position the recipe gave it. A required timeline stops at its
        first failure, so the patches behind that one carry no status and
        record nothing.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import APPLY_PATCH

        patch_result = result.get("warm_patch_result")
        if not isinstance(patch_result, Mapping):
            return
        for position, row in enumerate(patch_result.get("patches") or []):
            if not isinstance(row, Mapping):
                continue
            recorder.record_apply_item(
                str(row.get("patch_ref") or ""),
                kind=APPLY_PATCH,
                position=row.get("timeline_index", position),
                applied=str(row.get("status") or "") != "failed",
                reason=str(row.get("reason") or row.get("detail") or ""),
                target=str(row.get("target_repo") or ""),
            )

    def _warm_replay_timeline(self, task: "Task | None" = None):
        """Rebind to the in-flight replay's event, or ``None`` when one could not be built.

        Absent task params degrade the request block, never the gates and the
        measurement the promote seam is here to record.
        """
        params = dict(getattr(task, "params", None) or {})
        outcome = dict(self.shared_state.warm_replay_outcome or {})
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        return make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            task_id=str(getattr(task, "task_id", "") or outcome.get("replay_task_id") or ""),
            tier=str(params.get("warm_recipe_tier") or outcome.get("warm_recipe_tier") or ""),
            config_source=str(params.get("config_source") or outcome.get("config_source") or ""),
            config_donor_tier=str(params.get("config_donor_tier") or outcome.get("config_donor_tier") or ""),
            donor=self._warm_replay_donor(outcome) or None,
            expected_gain_pct=params.get("warm_expected_gain_pct", outcome.get("expected_gain_pct")),
            confidence=params.get("warm_recipe_conf", outcome.get("warm_recipe_conf")),
            session_baseline_tput=self.shared_state.baseline_tput or None,
            kernel_count=len(list(params.get("warm_kernel_plan") or [])),
            # Rebinding, not opening: the enqueue seam already put this event
            # on the timeline, and opening it twice would restate its start.
            open_event_on_timeline=False,
        )

    def _skip_restated_replay(
        self,
        args: str,
        envs: Mapping[str, Any],
        code_changes: Sequence[Any],
        identity: Mapping[str, Any],
    ) -> bool:
        """Skip a replay whose launch would be the baseline's own; True when it did.

        Patches and kernel items change the code, which no argv comparison sees,
        so a replay carrying any of them always runs. So does one under reference
        launch controls: materialization re-applies them to the baseline config,
        so the replay's launch is not that config's argv plus the request.
        """
        state = self.shared_state
        if code_changes or state.reference_launch_controls:
            return False
        if not config_launch_restated_by(str(state.baseline_config_path or ""), args, envs, str(state.framework or "")):
            return False
        log.info("PRELUDE: warm replay skipped: %r %s only restate the baseline launch", args, sorted(envs))
        self._skip_warm_replay(
            code=SKIP_RESTATES_BASELINE,
            outcome={"status": "skipped", "reason": SKIP_RESTATES_BASELINE, **identity},
            details={"extra_server_args": args, "extra_envs": sorted(envs)},
        )
        return True

    def _skip_warm_replay(
        self,
        *,
        code: str,
        outcome: Mapping[str, Any],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Settle the one-shot guard for a replay that will not run.

        Every refusal owes the same four things: flip the guard, state the
        outcome, close the timeline event, and persist. The persist is what
        makes the guard mean anything -- a refusal that never reached disk would
        let the next boot replay against the decision just taken.

        Call this after any rollback or stop-reason the branch also sets, so
        that one save carries the whole refusal.
        """
        state = self.shared_state
        state.warm_replay_attempted = True
        state.warm_replay_outcome = dict(outcome)
        self._record_warm_replay_skip(code=code, outcome=state.warm_replay_outcome, details=details)
        state.save(self.session_dir)

    def _record_warm_replay_skip(
        self,
        *,
        code: str,
        outcome: Mapping[str, Any],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the event for a replay refused before it ran.

        A skip is a decision, so it closes an event of its own rather than
        leaving the timeline silent. ``code`` is a ``SKIP_*`` value. The
        earliest refusals have no identity fields on ``outcome`` yet, and state
        an empty request rather than an invented one.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import (
            make_warm_replay_recorder,
        )

        recorder = make_warm_replay_recorder(
            phase=_WARM_REPLAY_EVENT_PHASE,
            macro_cycle=int(self.shared_state.macro_cycle or 0),
            tier=str(outcome.get("warm_recipe_tier") or ""),
            config_source=str(outcome.get("config_source") or ""),
            config_donor_tier=str(outcome.get("config_donor_tier") or ""),
            donor=self._warm_replay_donor(outcome) or None,
            expected_gain_pct=outcome.get("expected_gain_pct"),
            confidence=outcome.get("warm_recipe_conf"),
        )
        if recorder is None:
            return
        # A refusal rooted in the kernel preparation has a plan behind it,
        # and which item refused to apply is the whole of what it means.
        self._record_warm_kernel_apply_items(recorder)
        if rollback := (outcome.get("rollback") if isinstance(outcome.get("rollback"), Mapping) else None):
            recorder.record_rollback(ok=rollback.get("ok"), errors=rollback.get("errors"))
        recorder.finish_skipped(code=code, outcome=outcome, details=details)

    def promote_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> Verdict:
        """Interpret a combined Recipe+Kernel ``replay_warm_recipe`` result.

        Closes the replay's timeline event on whichever outcome the settling
        below persisted. Done here, around the whole arc, rather than at each
        of the ten branches that can end it: a branch that forgot to close
        would leave a settled replay reading as still in flight.

        Returns:
            The settlement verdict the settling ruled.
        """
        recorder = self._warm_replay_timeline(task)
        try:
            verdict = self._settle_warm_replay(result, task=task, recorder=recorder)
        except BaseException as exc:
            if recorder is not None:
                recorder.finish_crashed(exc)
            raise
        if recorder is not None:
            recorder.finish(self.shared_state.warm_replay_outcome)
        return verdict

    def _conclude_warm_replay(self, outcome: dict, verdict: Verdict, *, clear_pending: bool = False) -> Verdict:
        """Persist the settled outcome, the one way every settlement ends."""
        state = self.shared_state
        if clear_pending:
            state.warm_replay_pending = {}
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)
        return verdict

    def _settle_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None",
        recorder: Any,
    ) -> Verdict:
        """Settle a combined Recipe+Kernel replay onto an outcome.

        Measured uplift promotes the warm config onto ``optimization_stack`` and
        ``current_best``. Failures (including a failed required patch timeline)
        roll back both halves fail-closed, set an outcome status, and never
        propagate. ``task`` may be ``None`` (degraded path). Each gate writes
        its own verdict onto ``recorder`` as it rules, which is what lets a
        reader see why the arc ended.
        """
        outcome = dict(self.shared_state.warm_replay_outcome or {})
        # Stamped once up front so every terminal branch below carries it; each of them re-persists ``outcome`` before
        # returning.
        outcome["settled_at"] = now_iso(timespec="seconds")
        settlement = _WarmReplaySettlement(result if isinstance(result, dict) else {}, task, outcome, recorder)
        if not isinstance(result, dict):
            return self._fail_warm_replay(settlement, "non_dict_result")
        status = str(result.get("status") or "")
        if status != "succeeded":
            outcome["error_class"] = str(result.get("error_class") or "")
            log.info("warm-replay failed (status=%s, error_class=%s)", status, outcome.get("error_class"))
            return self._fail_warm_replay(settlement, str(result.get("error") or result.get("reason") or "")[:240])
        measurement = self._warm_replay_measurement(settlement)
        if measurement.tput <= 0 or measurement.baseline_tput <= 0:
            reason = f"invalid_tput tput={measurement.tput} baseline={measurement.baseline_tput}"
            self._record_tput_gate(settlement, measurement, passed=False, reason=reason)
            return self._fail_warm_replay(settlement, reason)
        self._record_tput_gate(settlement, measurement, passed=True, reason="the replay and its anchor both measured")
        # Recorded before the gates below, not after the KEEP ruling: a replay
        # rejected on quality or accuracy still produced a real measurement, and
        # reading what it scored is how a gate rejection is told apart from a
        # replay that never got that far.
        outcome["actual_gain_pct"] = round(measurement.measured_gain, 3)
        outcome["throughput_after"] = measurement.tput
        self._record_replay_measurement(settlement, measurement)
        if (quality_verdict := self._judge_warm_replay_quality(settlement)) is not None:
            return quality_verdict
        if (accuracy_verdict := self._judge_warm_replay_accuracy(settlement)) is not None:
            return accuracy_verdict
        # Restated now that the gate has read the scores onto the outcome,
        # so the score sits beside the reference it was judged against.
        self._record_replay_measurement(settlement, measurement)
        result["combined_gain_pct"] = round(measurement.measured_gain, 3)
        if recorder is not None:
            # Recorded before the keep ruling, so a replay that measured and
            # lost still states what lost.
            recorder.record_applied(
                extra_server_args=str(settlement.params.get("extra_server_args") or ""),
                extra_envs=dict(settlement.params.get("extra_envs") or {}),
            )
            self._record_warm_patch_apply_items(recorder, result)
        return self._grade_warm_replay(settlement, measurement)

    def _warm_replay_measurement(self, settlement: _WarmReplaySettlement) -> _ReplayMeasurement:
        """What the round measured, against the anchor the replay was enqueued with.

        ``output_throughput`` is the HOT measure round and the comparison value;
        the warmup round is retained for audit only. The anchor falls back to
        live state for a replay whose params did not record one.
        """
        result = settlement.result
        tput = as_float(result.get("output_throughput"))
        baseline_tput = as_float(settlement.params.get("baseline_tput_anchor"))
        if baseline_tput <= 0:
            baseline_tput = as_float(self.shared_state.baseline_tput)
        gain = (tput / baseline_tput - 1.0) * 100.0 if tput > 0 and baseline_tput > 0 else 0.0
        return _ReplayMeasurement(
            tput=tput,
            cold_round_tput=as_float(result.get("warmup_round_tput")),
            baseline_tput=baseline_tput,
            measured_gain=gain,
            expected_gain=as_float(settlement.outcome.get("expected_gain_pct")),
        )

    @staticmethod
    def _record_tput_gate(
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
        *,
        passed: bool,
        reason: str,
    ) -> None:
        """Put the ruling on whether the round and its anchor both measured on the timeline."""
        if settlement.recorder is not None:
            settlement.recorder.record_gate(
                GATE_TPUT_VALID,
                passed=passed,
                reason=reason,
                observed=measurement.tput,
                threshold=measurement.baseline_tput,
            )

    @staticmethod
    def _record_replay_measurement(
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
    ) -> None:
        """Put the numbers the replay was judged on, and any score read onto the outcome, on the timeline.

        The anchor goes on record here, where it is used: back-solving it from
        the gain is exact only until a re-baseline moves the number the replay
        was never judged against.
        """
        if settlement.recorder is None:
            return
        outcome = settlement.outcome
        settlement.recorder.record_measurement(
            before_tput=measurement.baseline_tput,
            after_tput=measurement.tput,
            gain_pct=measurement.measured_gain,
            hot_tput=measurement.tput,
            cold_tput=measurement.cold_round_tput if measurement.cold_round_tput > 0 else None,
            accuracy=outcome.get("replay_accuracy"),
            baseline_accuracy=outcome.get("baseline_accuracy"),
            eval_ran=outcome.get("eval_ran"),
        )

    def _fail_warm_replay(self, settlement: _WarmReplaySettlement, reason: str) -> Verdict:
        """Unwind a replay that produced nothing to judge and settle it as failed."""
        if not self._require_combined_warm_rollback(settlement):
            return Verdict.FAILED
        settlement.outcome["status"] = "failed"
        settlement.outcome["reason"] = reason
        return self._conclude_warm_replay(settlement.outcome, Verdict.FAILED)

    def _judge_warm_replay_quality(self, settlement: _WarmReplaySettlement) -> Verdict | None:
        """Rule on the replayed config's image quality against the baseline reference.

        A replayed config lands on ``current_best``, so every later measurement
        in the session is taken against it. Promoting on throughput alone
        selects for garbage: breaking the numerics is itself a large throughput
        win. ``require=False`` keeps a missing or skipped gate non-blocking.
        """
        from ..actions.executors._accuracy_gate import quality_gate_passed

        recorder = settlement.recorder
        qg = settlement.result.get("quality_gate")
        if qg is None:
            return None
        if quality_gate_passed(qg, require=False):
            if recorder is not None:
                # Only recorded when a gate existed to rule, which is how a reader
                # tells "passed quality" from "quality did not apply".
                recorder.record_gate(GATE_QUALITY, passed=True, reason="no quality regression vs baseline reference")
            return None
        if recorder is not None:
            recorder.record_gate(GATE_QUALITY, passed=False, reason="image-quality gate failed vs baseline reference")
        if not self._require_combined_warm_rollback(settlement):
            return Verdict.FAILED
        outcome = settlement.outcome
        outcome["status"] = "quality_failed"
        outcome["reason"] = "image-quality gate failed vs baseline reference"
        outcome["quality_gate"] = qg
        log.info("warm-replay REJECTED by quality gate: %s", qg)
        return self._conclude_warm_replay(outcome, Verdict.REVERTED)

    def _grade_warm_replay(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
    ) -> Verdict:
        """Rule the measured round against the session's keep threshold and constraints."""
        state = self.shared_state
        keep_threshold = _phase_state.resolve_keep_threshold(state)
        graded = resolve_graded_comparison(
            state,
            settlement.result,
            keep_threshold_pct=keep_threshold,
            anchor_tput=measurement.baseline_tput,
        )
        settlement.outcome["keep_threshold_pct"] = keep_threshold
        if graded.veto_reason:
            # The grading applies the session's constraints (latency, power), so the drift branch below rolls the
            # replay back rather than leaving a config a budget refused promoted on disk.
            settlement.outcome["constraint_veto"] = graded.veto_reason
        if settlement.recorder is not None:
            settlement.recorder.record_gate(
                GATE_KEEP_THRESHOLD,
                passed=graded.verdict == VERDICT_KEEP,
                reason=f"graded on {graded.objective}: {graded.refused_by or graded.verdict}",
                observed=measurement.measured_gain,
                threshold=keep_threshold,
            )
        if graded.verdict == VERDICT_KEEP:
            return self._settle_reproduced_replay(settlement, measurement)
        return self._reject_warm_replay_as_drift(
            settlement,
            reason=graded.veto_reason
            or graded.refused_by
            or f"measured {measurement.measured_gain:+.2f}% below keep threshold {keep_threshold:+.2f}%",
        )

    def _settle_reproduced_replay(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
    ) -> Verdict:
        """Settle a warm replay that reproduced its prior, and say what it earned.

        Every path here ends the settlement: the replay either promotes the
        recipe it reproduced or is concluded against one of the gates, so the
        drift rejection below this call is reached only by a replay that did not
        reproduce at all.
        """
        outcome = settlement.outcome
        recorder = settlement.recorder
        promoted, promotion = self._resolve_promoted_recipe_checkout(settlement.result, settlement.task)
        if not promoted:
            return self._refuse_unpromoted_checkout(settlement, promotion)
        promoted_checkout = (
            str(promotion.get("target_repo") or "").strip() if promotion.get("status") == "promoted" else ""
        )
        patch_refs = self._replayed_patch_refs(settlement.result, settlement.params)
        if recorder is not None:
            recorder.record_gate(
                GATE_PROMOTION,
                passed=True,
                reason="every patched checkout the replay measured on was promoted",
            )
        if (refused := self._refuse_replay_without_params(settlement, measurement, patch_refs)) is not None:
            return refused
        entry_extra = self._replayed_stack_entry(settlement, measurement, promoted_checkout, patch_refs)
        state = self.shared_state
        # Resume safety: do not clobber existing stack entries.
        state.optimization_stack = list(state.optimization_stack or [])
        # A prior promote owns the outcome; re-running would re-journal it.
        already_pushed = any(
            isinstance(entry, dict) and entry.get("action") == "replay_warm_recipe"
            for entry in state.optimization_stack
        )
        if not already_pushed and not self._lift_replayed_config(settlement, measurement, entry_extra):
            # Cleared the keep threshold against the baseline yet did not
            # beat the adopted configuration on its graded objective.
            return self._reject_warm_replay_as_drift(
                settlement,
                reason=f"measured {measurement.measured_gain:+.2f}% over baseline but current_best refused the lift",
            )
        if promoted_checkout:
            outcome["active_framework_root"] = promoted_checkout
            state.active_inferencex_path = pick_tree(promotion["target_repos"], state.active_inferencex_path)
        outcome["status"] = "reproduced"
        outcome.pop("replayed_patch_refs", None)
        if patch_refs:
            outcome["replayed_patch_refs"] = patch_refs
        kernel_outcome = self._book_combined_kernel_keep(settlement.result, settlement.task)
        outcome["kernel"] = dict(kernel_outcome)
        if recorder is not None:
            recorder.record_applied(kernel=kernel_outcome)
        if already_pushed:
            log.info(
                "warm-replay promote: stack already carries the entry; "
                "skipping duplicate push (likely resume mid-promote)",
            )
            return self._conclude_warm_replay(outcome, Verdict.ADOPTED, clear_pending=True)
        if recorder is not None:
            recorder.record_promotion(
                promoted_checkout=promoted_checkout,
                replayed_patch_refs=patch_refs,
                stack_entry=entry_extra,
            )
        return self._adopt_reproduced_replay(settlement, measurement)

    def _refuse_replay_without_params(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
        patch_refs: list[str],
    ) -> Verdict | None:
        """Refuse a replay that reproduced but carries nothing to push onto the stack."""
        params = settlement.params
        recorder = settlement.recorder
        carries = (
            str(params.get("extra_server_args") or "").strip()
            or params.get("extra_envs")
            or patch_refs
            or params.get("warm_kernel_plan")
        )
        if carries:
            if recorder is not None:
                recorder.record_gate(
                    GATE_PARAMS_PRESENT,
                    passed=True,
                    reason="the replay carries params that can be pushed onto the stack",
                )
            return None
        if recorder is not None:
            recorder.record_gate(
                GATE_PARAMS_PRESENT,
                passed=False,
                reason="task params carry no args, no envs, no patch and no kernel plan to replay",
            )
        settlement.outcome["status"] = "reproduced_but_no_params"
        settlement.outcome["reason"] = "task.params missing extra_server_args/extra_envs and no warm patch was applied"
        log.warning(
            "warm-replay measured +%.2f%% but cannot push stack (task=%r has no warm args/envs)",
            measurement.measured_gain,
            settlement.task,
        )
        return self._conclude_warm_replay(settlement.outcome, Verdict.REFUSED)

    def _replayed_stack_entry(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
        promoted_checkout: str,
        patch_refs: list[str],
    ) -> dict[str, Any]:
        """The stack-entry-only metadata a reproduced replay earns; the lift keeps current_best pure config."""
        outcome = settlement.outcome
        return {
            "gain_pct": round(measurement.measured_gain, 3),
            "hot_tput": float(measurement.tput),
            "cold_tput": float(measurement.cold_round_tput) if measurement.cold_round_tput > 0 else None,
            # The score this promotion was judged on, recorded alongside the throughput it was judged with.
            "accuracy": outcome.get("replay_accuracy"),
            # source_tier records the warm-recipe tier for breakdown attribution.
            "source_tier": outcome.get("warm_recipe_tier", ""),
            "source_confidence": outcome.get("warm_recipe_conf", 0.0),
            **self._stack_entry_annotations(
                settlement.params,
                settlement.result,
                promoted_checkout=promoted_checkout,
                replayed_patch_refs=patch_refs,
            ),
        }

    def _lift_replayed_config(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
        entry_extra: dict[str, Any],
    ) -> bool:
        """Push the replayed config onto ``current_best``; False when the adopted one still wins."""
        params = settlement.params
        result = settlement.result
        warm_args = str(params.get("extra_server_args") or "").strip()
        warm_envs = dict(params.get("extra_envs") or {})
        recipe_args = str(
            params["recipe_extra_server_args"] if "recipe_extra_server_args" in params else warm_args
        ).strip()
        recipe_envs = dict(params["recipe_extra_envs"] if "recipe_extra_envs" in params else warm_envs)
        return bool(
            self._coord.writeback.lift_to_current_best(
                "replay_warm_recipe",
                float(measurement.tput),
                {
                    "name": "warm_replay",
                    **graded_axes_of(result),
                    # The latency and power budgets grade on these and fail closed without them.
                    **constraint_fields(result),
                    "candidate_extra_server_args": warm_args,
                    "candidate_extra_envs": warm_envs,
                    "recipe_delta": {
                        "extra_server_args": recipe_args,
                        "extra_envs": recipe_envs,
                        "remove_args": [],
                        "unset_envs": [],
                        "args_mode": "replace",
                    },
                    "extra_envs": warm_envs,
                    "source_phase": "PRELUDE",
                    "task_id": str(getattr(settlement.task, "task_id", "") or ""),
                    "workspace": str(result.get("workspace") or ""),
                },
                entry_extra=entry_extra,
            )
        )

    def _adopt_reproduced_replay(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
    ) -> Verdict:
        """Publish the reproduced verdict, re-validate the session, and journal the promotion.

        The verdict is published now that the stack entry exists but before the
        cumulative update (the one step below that can raise and is swallowed by
        the caller). Placed after the lift on purpose -- if the lift raises, the
        outcome stays in_flight and both the stack and the post-ruling mirror
        agree there is nothing adopted.
        """
        self.shared_state.warm_replay_outcome = settlement.outcome
        self._coord.writeback.validate(measurement.tput, settlement.result, source="warm_replay")
        log.info(
            "warm-replay REPRODUCED: measured=+%.2f%% (expected=+%.2f%%); pushed warm_replay onto stack",
            measurement.measured_gain,
            measurement.expected_gain,
        )
        self._journal_warm_replay(settlement, measurement)
        return self._conclude_warm_replay(settlement.outcome, Verdict.ADOPTED, clear_pending=True)

    @staticmethod
    def _stack_entry_annotations(
        params: Mapping[str, Any],
        result: dict,
        *,
        promoted_checkout: str,
        replayed_patch_refs: list[str],
    ) -> dict[str, Any]:
        """Describe what a replayed stack entry reproduced, for the audit trail.

        None of this is configuration: it is what the replay touched -- the
        framework checkout it ran against, the kernel columns it replayed, and
        the patches that landed -- recorded so a later reader can tell a
        reproduced stack entry from an originally measured one.
        """
        annotations: dict[str, Any] = {}
        if promoted_checkout:
            annotations["framework_source_root"] = promoted_checkout
        kernel_plan = [entry for entry in (params.get("warm_kernel_plan") or []) if isinstance(entry, dict)]
        if kernel_plan:
            annotations["kernel_replay"] = {
                "validation": "combined_recipe_kernel",
                "count": len(kernel_plan),
                "columns": sorted({str(entry.get("column") or "") for entry in kernel_plan}),
            }
        patch_result = result.get("warm_patch_result")
        if isinstance(patch_result, dict):
            annotations["recipe_patch_statuses"] = list(patch_result.get("patches") or [])
        if replayed_patch_refs:
            annotations["replayed_patch_refs"] = replayed_patch_refs
        return annotations

    def _refuse_unpromoted_checkout(
        self,
        settlement: _WarmReplaySettlement,
        promotion: Mapping[str, Any],
    ) -> Verdict:
        """Conclude a reproduced replay whose Recipe checkout would not promote.

        The replay measured what it claimed, but the checkout it measured cannot
        be published, so nothing downstream may read it as adopted: the kernel
        half is reported reverted alongside the recipe half that refused.
        """
        outcome = settlement.outcome
        if settlement.recorder is not None:
            settlement.recorder.record_gate(
                GATE_PROMOTION,
                passed=False,
                reason=str(promotion.get("failure") or "validated Recipe checkout promotion failed"),
            )
        if not self._require_combined_warm_rollback(settlement):
            return Verdict.FAILED
        outcome["status"] = "promotion_failed"
        outcome["reason"] = str(promotion.get("failure") or "validated Recipe checkout promotion failed")
        outcome["recipe_checkout_promotion"] = promotion
        kernel_outcome = {
            "status": "reverted",
            "reason": "recipe_checkout_promotion_failed",
            "validation": "combined_recipe_kernel",
        }
        outcome["kernel"] = kernel_outcome
        return self._conclude_warm_replay(outcome, Verdict.FAILED)

    @staticmethod
    def _replayed_patch_refs(result: dict, params: Mapping[str, Any]) -> list[str]:
        """Name the warm patches this replay actually landed.

        A plan that required its patch timeline counts only the statuses that
        mean the overlay is on disk; without that requirement every reported
        patch file counts, since the apply had nothing to prove.
        """
        return [
            str(item.get("patch_file") or "")
            for item in (result.get("warm_patches_applied") or [])
            if (
                isinstance(item, dict)
                and (
                    not params.get("required_patch_timeline")
                    or item.get("status")
                    in {
                        "applied",
                        "applied_3way",
                        # What a nogit apply reports when it writes; on a pip-installed
                        # framework it is the only status an overlay can land under.
                        "applied_nogit",
                        "present_in_dirty_worktree",
                    }
                )
                and str(item.get("patch_file") or "")
            )
        ]

    def _journal_warm_replay(
        self,
        settlement: _WarmReplaySettlement,
        measurement: _ReplayMeasurement,
    ) -> None:
        """Journal a warm replay as a synthetic KEEP; it carries no KB lesson.

        Best-effort: the replay has already earned its verdict, so a journal
        that cannot be written must not take the verdict down with it.
        """
        warm_args = str(settlement.params.get("extra_server_args") or "").strip()
        try:
            journal = self._coord.recipe_journal.ensure_journal()
            journal.append_entry(
                JournalEntry(
                    phase=(self.shared_state.phase or "PRELUDE").upper() or "PRELUDE",
                    iter=int(self.shared_state.tick or 0),
                    kind=KIND_OTHER,
                    change=f"warm_replay({settlement.outcome.get('warm_recipe_tier', '?')}): {warm_args}",
                    outcome=derive_journal_outcome(Verdict.ADOPTED, settlement.result),
                    gain_pct=round(measurement.measured_gain, 3),
                    throughput_after=measurement.tput,
                    task_id=str(settlement.task.task_id if settlement.task is not None else ""),
                    tick=int(self.shared_state.tick or 0),
                )
            )
        except Exception:
            log.exception("warm-replay journal append failed")

    def _reject_warm_replay_as_drift(self, settlement: _WarmReplaySettlement, *, reason: str) -> Verdict:
        """Roll back a measured replay that is not adopted and settle it as ``drift``.

        A rollback that failed settles as ``rollback_failed`` instead.
        """
        if not self._require_combined_warm_rollback(settlement):
            return Verdict.FAILED
        outcome = settlement.outcome
        kernel_plan = list(settlement.params.get("warm_kernel_plan") or [])
        kernel_outcome = {
            "status": "reverted",
            "total": len(kernel_plan),
            "kept": 0,
            "reverted": len(kernel_plan),
            "validation": "combined_recipe_kernel",
        }
        outcome["kernel"] = dict(kernel_outcome)
        if settlement.recorder is not None:
            settlement.recorder.record_applied(kernel=kernel_outcome)
        outcome["status"] = "drift"
        outcome["reason"] = reason
        log.info("warm-replay DRIFT: %s", reason)
        return self._conclude_warm_replay(outcome, Verdict.REVERTED, clear_pending=True)

    async def maybe_enqueue_prelude_initial_analysis_after_baseline(
        self,
        *,
        baseline_tput: float | None = None,
    ) -> None:
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline; skipped while warm-replay is in_flight (GPU/port contention)."""
        state = self.shared_state
        if _phase_state.warm_replay_in_flight(state):
            log.info(
                "PRELUDE: deferring initial %s until warm-replay completes",
                self.internal_analysis_kind(),
            )
            return
        if baseline_tput is None:
            try:
                baseline_tput = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                baseline_tput = 0.0
        if not isinstance(baseline_tput, (int, float)) or baseline_tput <= 0:
            return
        if (state.auto_roofline_pending_task_id or "").strip():
            return
        affordable, evidence = _phase_state.prelude_can_afford(
            state,
            expected_cost_sec=self._measured_analysis_cost_sec(),
        )
        if not affordable:
            log.warning(
                "PRELUDE: skipping the initial %s — %.0fs of preparation budget "
                "left (bound=%s) against an expected %.0fs. The optimization "
                "phases keep the time instead.",
                self.internal_analysis_kind(),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
                evidence.get("expected_cost_sec", 0.0),
            )
            self._record_prelude_arm_dropped("initial_analysis", evidence)
            return
        try:
            rl_task = await self.enqueue_internal_analysis_task(
                reason="prelude_initial",
            )
            if rl_task is None:
                return
            state.auto_roofline_pending_task_id = rl_task.task_id
            log.info(
                "PRELUDE: baseline landed (tput=%.2f); auto-enqueued initial %s task=%s",
                float(baseline_tput),
                rl_task.kind,
                rl_task.task_id,
            )
        except Exception as exc:
            log.exception(
                "PRELUDE: failed to enqueue initial analysis task after baseline: %r",
                exc,
            )

    def _analysis_attempt_suffix(self, kind: str) -> str:
        """Idempotency-key suffix separating a re-armed roofline retry from the attempt that failed."""
        if kind != "roofline":
            return ""
        try:
            streak = int(self.shared_state.roofline_failure_streak or 0)
        except (TypeError, ValueError):
            streak = 0
        return f"-a{streak}" if streak > 0 else ""

    async def enqueue_internal_analysis_task(
        self, *, reason: str, inline_event: str = "", idem_scope: str = ""
    ) -> "Task | None":
        """Build + enqueue a Coordinator-internal analysis task (roofline or profile). Idempotency key
        internal-analysis-<reason><idem_scope>; ``idem_scope`` separates repeated triggers that share a reason.
        Returns None when the stack cannot produce a GPU trace for it to analyze.
        """
        params = self.internal_analysis_params(reason=reason, inline_event=inline_event)
        if params is None:
            return None
        kind = self.internal_analysis_kind()
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl(kind)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=(
                f"internal-analysis-{reason}{idem_scope}"
                f"{self._coord.dispatcher.cycle_idem_suffix()}{self._analysis_attempt_suffix(kind)}"
            ),
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
            dispatch_class="coordinator",
        )
        if was_existing:
            log.info(
                "internal-analysis task already exists (idempotent: kind=%s task_id=%s, state=%s)",
                kind,
                task.task_id,
                task.state,
            )
        return task

    def internal_analysis_params(self, *, reason: str, inline_event: str = "") -> dict[str, Any] | None:
        """Params for a Coordinator-internal analysis run, or None when the stack cannot produce a GPU trace."""
        from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM

        state = self.shared_state
        unsupported = str(state.gpu_trace_unsupported_reason or "")
        if unsupported:
            log.error(
                "internal-analysis (%s): not run -- %s; relying on static-source evidence for the rest of the session",
                reason,
                unsupported,
            )
            self._record_prelude_arm_dropped(
                "internal_analysis",
                {"reason": reason, "gpu_trace_unsupported_reason": unsupported},
            )
            return None
        recipe = roofline_recipe(state, reason)
        repeated = repeated_roofline_failure(state, recipe)
        if repeated is not None:
            log.warning(
                "internal-analysis (%s): not run -- the roofline on %s already failed with %s, which recurs on the "
                "same input; a KEEP that changes the recipe re-arms it",
                reason,
                recipe,
                repeated.get("failure"),
            )
            self._record_prelude_arm_dropped(
                "internal_analysis",
                {"reason": reason, "recipe": recipe, "repeated_failure": dict(repeated)},
            )
            return None
        params: dict[str, Any] = {"source": "coordinator_internal", "reason": str(reason), RECIPE_PARAM: recipe}
        if inline_event:
            params[INLINE_EVENT_PARAM] = str(inline_event)
        if reason != "prelude_initial":
            inject_stack_base_params(params, state)
        else:
            # PRELUDE roofline profiles the baseline arm: inject baseline's own server args (never current_best's) so
            # a later warm-replay can't swap in flags that skew the baseline ceiling.
            params.update(self._coord.enablement_revalidation.baseline_launch_params())
            try:
                from hyperloom.inference_optimizer.roofline_ceiling import read_baseline_server_args

                bl_args = read_baseline_server_args(state).strip()
            except Exception:  # noqa: BLE001 — best-effort; empty falls through
                bl_args = ""
            if bl_args:
                params["base_extra_args"] = bl_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        return params
