# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE phase handler for section-based Recipe replay and initial analysis.

Merges Explore, Framework, and Kernel AgentKB snapshots, applies their ordered
patch timeline, and enqueues the baseline/roofline internal-analysis tasks.
"""

from __future__ import annotations
import logging as _logging
import math
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
from . import machine_state as _phase_state
from ..state.optimization_journal import (
    JournalEntry,
)
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import Task
from ..loop.coordinator import (
    _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE,
)
from ..loop.coordinator_helpers import (
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
)
from .base import PhaseHandler
from ..knowledge.remote_recipe.sanitize import HOST_ORIGIN_KEY

log = _logging.getLogger(__name__)

# The ``parse_eval_results`` reasons that prove an eval produced output: a
# results file it could not decode, and one carrying no metric it recognises.
# Every other reason -- no file at all, or the parser itself raising -- leaves
# no evidence the eval ran, and calling those "ran" makes an infrastructure
# fault read as a model that answered nothing.
_EVAL_RAN_BUT_UNSCORABLE = ("parse error:", "no recognized metric in")


def _merge_named_current_recipe_configs(
    owners: list[tuple[str, Mapping[str, Any]]],
) -> tuple[str, dict[str, str]]:
    """Merge named config snapshots with exact duplicate conflict checks."""
    from ..actions.executors._grid_server_args import (
        tokenize_server_args_preserving_json,
    )

    pairs: dict[str, tuple[str, ...]] = {}
    order: list[str] = []
    positional = 0
    for owner, config in owners:
        raw = str(config.get("extra_server_args") or "").strip()
        if not raw:
            continue
        parsed = tokenize_server_args_preserving_json(raw)
        if parsed is None:
            raise ValueError(f"{owner} extra_server_args cannot be tokenized safely")
        _normalized, tokens = parsed
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.startswith("--"):
                key, has_equals, inline = token.partition("=")
                values: list[str] = [inline] if has_equals else []
                index += 1
                while not has_equals and index < len(tokens) and not tokens[index].startswith("--"):
                    values.append(tokens[index])
                    index += 1
                rendered = (key, *values)
                prior = pairs.get(key)
                if prior is not None and prior != rendered:
                    raise ValueError(f"current Recipe config conflict for {key}: {prior!r} != {rendered!r}")
                if prior is None:
                    pairs[key] = rendered
                    order.append(key)
                continue
            synthetic = f"__positional_{positional}"
            positional += 1
            pairs[synthetic] = (token,)
            order.append(synthetic)
            index += 1

    envs: dict[str, str] = {}
    for owner, config in owners:
        raw_envs = config.get("extra_envs")
        if not isinstance(raw_envs, Mapping):
            continue
        for raw_key, raw_value in raw_envs.items():
            key = str(raw_key)
            value = str(raw_value)
            if key in envs and envs[key] != value:
                raise ValueError(f"current Recipe env conflict for {key}: {envs[key]!r} != {value!r} ({owner})")
            envs[key] = value
    return " ".join(token for key in order for token in pairs[key]), envs


def _recorded_apply_roots(provenance: Any) -> dict[str, str]:
    """Map each overlay ref to the checkout the record says it was applied into.

    Per ref rather than per record: a session can KEEP a patch against the
    framework tree and another against a sibling one, and each is replayable
    against the tree it came from.
    """
    roots: dict[str, str] = {}
    for row in provenance or []:
        if not isinstance(row, Mapping):
            continue
        origin = row.get("host_origin")
        if not isinstance(origin, Mapping):
            continue
        for ref, root in (origin.get("apply_roots") or {}).items():
            name = str(ref or "").strip()
            value = str(root or "").strip()
            if name and value:
                roots.setdefault(name, value)
    return roots


def _overlay_provenance_summary(sdk_replay: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise how the overlays this replay is about to apply were captured.

    Recorded on the outcome so a reader can tell an overlay set that reproduces
    its session from one that never could: a capture that could not account for
    every path, or a KEEP whose gain partly landed outside the framework root,
    is a known gap rather than a clean replay.
    """

    def _count(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    rows = [row for row in (sdk_replay.get("provenance") or []) if isinstance(row, Mapping)]
    overlays = len(sdk_replay.get("patches") or [])
    if not overlays and not rows:
        return {}
    return {
        "overlays": overlays,
        "realized": sum(1 for row in rows if row.get("realized")),
        "incomplete": sum(1 for row in rows if row.get("complete") is False),
        "artifacts_outside_root": sum(_count(row.get("artifacts_outside_root")) for row in rows),
    }


def _warm_kernel_keep_threshold_pct(state: Any) -> float:
    """Gain a replayed champion set must clear.

    Follows the shared decaying curve so the bar tracks the session's
    macro-cycle; ``HYPERLOOM_WARM_KERNEL_KEEP_PCT`` overrides it.

    Args:
        state: The SharedState the curve reads ``macro_cycle`` from.

    Returns:
        The KEEP threshold percentage for the warm-kernel replay.
    """
    default = _phase_state.resolve_keep_threshold(state)
    raw = str(os.environ.get("HYPERLOOM_WARM_KERNEL_KEEP_PCT", "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isfinite(value):
        return value
    log.warning(
        "warm replay: unusable HYPERLOOM_WARM_KERNEL_KEEP_PCT=%r; using %.2f",
        raw,
        default,
    )
    return default


class PreludePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

    def _internal_analysis_kind(self) -> str:
        """Pick the kind for the next Coordinator-internal analysis task: roofline when enable_roofline else profile.

        Returns:
            ``"roofline"`` when roofline is enabled, else ``"profile"``.
        """
        return (
            "roofline"
            if bool(
                getattr(self.shared_state, "enable_roofline", True),
            )
            else "profile"
        )

    def _measured_analysis_cost_sec(self) -> float:
        """Expected cost of the initial roofline/profile arm, in seconds.

        The analysis arm boots its own server and runs the same benchmark under
        a profiler, so one measured baseline round is a floor on its cost rather
        than a guess at it; :func:`expected_action_cost_minutes` applies that
        floor and falls back to the catalog for the first analysis of a session
        that has no measurement yet.

        Returns:
            float: Expected cost in seconds; ``0.0`` when nothing is on record,
            which :func:`machine_state.prelude_can_afford` reads as free.
        """
        registry = getattr(self, "action_registry", None)
        meta = registry.get(self._internal_analysis_kind()) if registry is not None else None
        return (
            expected_action_cost_minutes(
                meta,
                measured_baseline_sec=measured_baseline_runtime_sec(self.shared_state),
            )
            * 60.0
        )

    def _record_prelude_arm_dropped(self, arm: str, evidence: dict[str, Any]) -> None:
        """Record a PRELUDE arm dropped for budget on the current phase record.

        The phase record is what the session breakdown exports, so a dropped
        arm reads as a decision with numbers behind it rather than as an arm
        that silently never ran.

        Args:
            arm: The arm that was dropped.
            evidence: The affordability numbers behind the decision.
        """
        if not _phase_state.append_phase_evidence_row(
            getattr(self.shared_state, "phase_history", None),
            key="budget_dropped_arms",
            row={"arm": arm, **evidence},
        ):
            return
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort record
            log.exception("PRELUDE: failed to persist the dropped-arm record for %r", arm)

    def _warm_recipe_proven_items(self) -> list[dict[str, str]]:
        """Summarise warm-start ``what_worked`` items the scout can skip ({name, source}); fail-soft.

        Returns:
            A list of ``{"name", "source"}`` dicts for proven warm-start items;
            empty when no warm recipe is present.
        """
        state = self.shared_state
        warm = getattr(state, "warm_start_recipe", None) or {}
        if not isinstance(warm, dict) or not warm:
            return []
        recipe = warm.get("recipe") or {}
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        exact_history = recipe_attrs.get("exact_history")
        if isinstance(exact_history, dict):
            recipe_attrs = exact_history
        what_worked = recipe_attrs.get("what_worked") or []
        if not isinstance(what_worked, list):
            return []
        out: list[dict[str, str]] = []
        for row in what_worked:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append({"name": name, "source": str(row.get("source") or "").strip()})
        return out

    def _inject_warm_recipe_history_into_ledger(self) -> int:
        """Pre-fill ``explore_search.rejected`` with the warm recipe's ``what_failed`` rows (fingerprinted so the dedup gate denies re-tests). Idempotent via warm_history_injected; returns rows added.

        Returns:
            The number of new rejected rows injected into the explore ledger.
        """
        state = self.shared_state
        if getattr(state, "warm_history_injected", False):
            return 0
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_history_injected = True
            return 0
        recipe = warm.get("recipe") or {}
        # what_failed may be top-level or nested under attrs; fall back to the recipe.
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        exact_history = recipe_attrs.get("exact_history")
        if isinstance(exact_history, dict):
            recipe_attrs = exact_history
        what_failed = recipe_attrs.get("what_failed") or []
        if not isinstance(what_failed, list) or not what_failed:
            state.warm_history_injected = True
            return 0

        from ..actions.executors._canonical_fingerprint import (
            canonical_fingerprint,
        )

        es_raw = getattr(state, "explore_search", None) or {}
        es = dict(es_raw) if isinstance(es_raw, dict) else {}
        rejected = list(es.get("rejected") or [])
        existing_fps = {str(r.get("fingerprint") or "") for r in rejected if isinstance(r, dict)}
        existing_fps.discard("")
        added = 0
        tier = str((warm or {}).get("tier") or "")
        for row in what_failed:
            if not isinstance(row, dict):
                continue
            args = str(row.get("extra_server_args") or "").strip()
            envs = row.get("extra_envs") or {}
            if not isinstance(envs, dict):
                envs = {}
            if not args and not envs:
                continue
            fp = canonical_fingerprint(args, envs)
            if fp in existing_fps:
                continue
            existing_fps.add(fp)
            rejected.append(
                {
                    "name": str(row.get("name") or "")[:120],
                    "fingerprint": fp,
                    "reason": "warm_recipe_what_failed",
                    "extra_server_args": args,
                    "extra_envs": dict(envs),
                    "source": "warm_start_recipe",
                    "source_tier": tier,
                    # Preserved for forensics; not used by the dedup gate.
                    "gain_pct": row.get("gain_pct"),
                    "error_class": row.get("error_class") or row.get("reason"),
                }
            )
            added += 1

        if added:
            es["rejected"] = rejected
            state.explore_search = es
            log.info(
                "warm-recipe history: injected %d what_failed rows into explore_search.rejected (tier=%s)",
                added,
                tier,
            )
        state.warm_history_injected = True
        return added

    def _collect_warm_kernel_plan(self, kb: Any) -> list[dict[str, Any]]:
        """Resolve the prior-champion kernel columns into a local apply plan.

        Reads the ``gemm``/``fusion``/``rewrite`` sub-columns the warm-start
        download provided, resolves every recorded file ref to its downloaded
        copy via ``KernelAgentKB.prior_file``, and returns one plan entry per
        item carrying the local Patch or tuned artifact plus non-file metadata.
        Refs that do not resolve are dropped.
        """
        readers = (
            ("gemm", kb.read_gemm, "optimizations"),
            ("fusion", kb.read_fusion, "items"),
            ("rewrite", kb.read_rewrite, "items"),
        )
        plan: list[dict[str, Any]] = []
        for column, reader, list_key in readers:
            try:
                data = reader() or {}
            except Exception:  # noqa: BLE001 — a bad column must not block others
                log.warning("warm-kernel KB: reading %s column failed", column, exc_info=True)
                continue
            rows = data.get(list_key) if isinstance(data, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                meta = {
                    k: v
                    for k, v in row.items()
                    if k
                    not in (
                        "patch",
                        "source_file",
                        "source_files",
                        "target_file",
                        "target_files",
                        "target_path",
                        "tuned_file",
                        "files",
                    )
                }
                entry: dict[str, Any] = {"column": column, "meta": meta}
                # Preserve the exact Recipe row so CLOSE can carry the validated
                # kernel content and its refs forward on the same inference page.
                entry["recipe_row"] = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "source_file",
                        "source_files",
                        "target_file",
                        "target_files",
                        "target_path",
                    }
                }
                # The checkout this item was applied into. Replay places the
                # patch there and nowhere else, so an item that records none is
                # not replayable and says so rather than being searched for.
                host_origin = row.get(HOST_ORIGIN_KEY)
                if isinstance(host_origin, Mapping):
                    entry["apply_root"] = str(host_origin.get("apply_root") or "").strip()
                patch_ref = str(row.get("patch") or "").strip()
                if patch_ref:
                    patch_local = kb.prior_file(patch_ref)
                    if patch_local is not None:
                        entry["patch_path"] = str(patch_local)
                tuned_ref = str(row.get("tuned_file") or "").strip()
                if column == "gemm" and tuned_ref:
                    tuned_local = kb.prior_file(tuned_ref)
                    if tuned_local is not None:
                        entry["source_paths"] = [str(tuned_local)]
                if entry.get("patch_path") or (column == "gemm" and entry.get("source_paths")):
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
        """Return the shared parser's first safe repo-relative Patch target.

        Modify/delete/create targets come from paired pre/post-image headers;
        mode-only and metadata-only rename patches fall back to ``diff --git``.
        No persisted target metadata participates.
        """
        targets = cls._parse_diff_targets(patch_path)
        return targets[0] if targets else ""

    def _resolve_kernel_target_path(self, entry: dict[str, Any]) -> str:
        """Locate the first Patch target under this Session's active root."""
        targets = self._resolve_kernel_target_paths(entry)
        return targets[0] if targets else ""

    def _resolve_kernel_target_paths(self, entry: dict[str, Any]) -> list[str]:
        """Resolve every declared Patch target under the Session active root.

        Existing pre-images must be files, create destinations must be absent,
        and resolved paths must remain beneath the root. Any read, parse, root,
        boundary, or file-state failure records ``entry['resolution_error']``
        and emits a warning before replay is deferred.
        """

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
        # The recorded root is the only answer. Searching an allowlist for a
        # tree the diff happens to fit would replay against code the gain was
        # never measured on, and a record naming no root is simply broken.
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
        """Rebuild a champion's env bundle against this run's downloaded files.

        A champion's deliverable is not always a source file. GEMM is purely
        parameter-shaped — the tuned table travels as a file ref and the env var
        that carries it is recorded per accepted tuner in
        ``e2e_results.kept[].env_var`` — and a fusion carries the env switches
        (``extra_envs``/``env_flags``) that turn the fused path on. Applying the
        patch without them re-measures the unoptimized path and reverts a good
        champion, so replay merges every recorded env and re-points any
        file-valued one at this run's local copy (the producing session's
        absolute paths are scrubbed before publish).
        """
        meta = entry.get("meta") or {}
        local = [str(p) for p in (entry.get("source_paths") or []) if p]
        envs: dict[str, str] = {}

        def _merge(source: Any) -> None:
            if isinstance(source, dict):
                for raw_key, raw_value in source.items():
                    key = str(raw_key).strip()
                    if not key:
                        continue
                    value = str(raw_value)
                    if key in envs and envs[key] != value:
                        raise ValueError(f"current Recipe kernel env conflict for {key}: {envs[key]!r} != {value!r}")
                    envs[key] = value

        for key in ("extra_envs", "env_flags", "recommended_env"):
            _merge(meta.get(key))
        e2e = meta.get("e2e")
        if isinstance(e2e, dict):
            _merge(e2e.get("extra_envs"))
        by_name = {Path(path).name: path for path in local}
        results = meta.get("e2e_results")
        kept = (results.get("kept") or []) if isinstance(results, dict) else []
        for tuner in kept:
            if not isinstance(tuner, dict):
                continue
            _merge(tuner.get("envs"))
            env_var = str(tuner.get("env_var") or "").strip()
            if not env_var:
                continue
            # Each accepted tuner owns its own table, so match by the recorded
            # value's filename. Pointing every tuner at the first download would
            # feed a dense tuner the MoE table (or vice versa) whenever a record
            # carries more than one.
            recorded = str(tuner.get("env_value") or envs.get(env_var) or "").strip()
            local_copy = by_name.get(Path(recorded).name) if recorded else None
            if local_copy is None and len(local) == 1:
                local_copy = local[0]
            if local_copy:
                envs[env_var] = local_copy
        # Re-point any other var that still names a file we downloaded. Only
        # path-shaped values qualify, so a flag like "1" or "candidate" whose
        # text happens to match a filename is left alone.
        for name, value in list(envs.items()):
            if not value or "/" not in value:
                continue
            replacement = by_name.get(Path(value).name)
            if replacement and replacement != value:
                envs[name] = replacement
        return envs

    def _apply_warm_kernel_patch(self, entry: dict[str, Any], target: str) -> dict[str, Any]:
        """Land one champion's file on disk without measuring it.

        The measurement is deliberately not here: a champion set is applied as a
        batch and graded by a single re-baseline, so this only stages the file
        (with a backup manifest that :meth:`_revert_warm_kernel_patches` uses to
        roll the whole set back when the set does not win).
        """
        from ..kernel.request_handlers import (
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
            "allow_unknown_target": True,
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

    @staticmethod
    def _restore_warm_kernel_snapshots(
        snapshots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Restore exact kernel target bytes captured before each mutation."""
        errors: list[str] = []
        for snapshot in reversed(snapshots):
            target = Path(str(snapshot.get("target") or ""))
            try:
                if snapshot.get("existed"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(Path(str(snapshot.get("backup") or "")).read_bytes())
                    if snapshot.get("mode") is not None:
                        target.chmod(int(snapshot["mode"]))
                elif target.exists() or target.is_symlink():
                    target.unlink()
                if snapshot.get("existed"):
                    expected = Path(str(snapshot.get("backup") or "")).read_bytes()
                    if not target.is_file() or target.read_bytes() != expected:
                        raise OSError("kernel restore verification failed")
                elif target.exists() or target.is_symlink():
                    raise OSError("kernel target still exists after restore")
            except OSError as exc:
                errors.append(f"{target}:{type(exc).__name__}:{exc}")
        return {"ok": not errors, "errors": errors}

    @staticmethod
    def _revert_warm_kernel_patches(
        applied: list[dict[str, Any]],
        snapshots: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Rollback kernels exactly, reporting every failure."""
        from ..kernel.request_handlers import _maybe_revert_kernel_patch

        errors: list[str] = []
        for apply_result in reversed(applied):
            if not apply_result.get("manifest_path"):
                continue
            try:
                reverted = _maybe_revert_kernel_patch(apply_result)
                if reverted.get("status") != "ok":
                    raise RuntimeError(
                        str(
                            reverted.get("error")
                            or reverted.get("reason")
                            or f"kernel revert status={reverted.get('status')}"
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("warm-kernel KB: revert failed", exc_info=True)
                errors.append(f"{type(exc).__name__}:{exc}")
        if snapshots:
            restored = PreludePhase._restore_warm_kernel_snapshots(snapshots)
            errors.extend(restored.get("errors") or [])
        return {"ok": not errors, "errors": errors}

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
        """Why this run must not replay kernel champions, or '' when allowed.

        Mirrors the gates the recipe warm-replay honours: an operator opt-out,
        an explicitly disabled KB, and local knowledge mode — which must never
        consult ambient ``KB_STORE_*`` credentials.
        """
        if not getattr(self, "_warm_replay_enabled", True):
            return "warm_replay_disabled"
        if bool(getattr(getattr(self, "knowledge_plane", None), "kb_disabled", False)):
            return "kb_degraded"
        from ..knowledge.config import KnowledgeConfig, KnowledgeStoreMode

        config = getattr(getattr(self, "knowledge_plane", None), "config", None) or KnowledgeConfig.from_env()
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
        """Record why warm replay stopped, naming the roots the record asked for.

        Args:
            reason: The resolution failure to surface in SBD.
            root_kind: ``framework`` or ``kernel``; names the reported keys.
            roots: The recorded roots that could not be used, for a reader
                diagnosing a record produced on another image.
            rollback: Outcome of undoing patches already applied, when any were.

        Returns:
            The ``warm_replay_outcome`` mapping.
        """
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
        """Skip combined warm replay when a kernel item's recorded root is unusable.

        A persisted plan may be restored on resume under a different image, so
        every recorded root is re-checked here rather than trusted from when the
        plan was built.
        """
        for entry in getattr(state, "warm_kernel_kb_plan", []) or []:
            if not isinstance(entry, dict):
                continue
            # gemm is parameter-shaped: it re-points env vars at downloaded files
            # and never patches a checkout, so it has no root to resolve.
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
        combined = dict(getattr(self.shared_state, "warm_replay_outcome", {}) or {})
        combined["kernel"] = dict(kernel_outcome)
        self.shared_state.warm_replay_outcome = combined
        return kernel_outcome

    def _preview_current_kernel_config(self, kb: Any) -> dict[str, Any]:
        """Resolve the kernel launch overlay without mutating target files."""
        configs: list[tuple[str, Mapping[str, Any]]] = []
        for index, entry in enumerate(self._collect_warm_kernel_plan(kb)):
            envs = self._warm_kernel_extra_envs(entry)
            if entry.get("column") == "gemm":
                if not envs:
                    continue
            else:
                targets = self._resolve_kernel_target_paths(entry)
                if not any(targets):
                    continue
            configs.append(
                (
                    f"kernel[{index}]",
                    {
                        "extra_server_args": str((entry.get("meta") or {}).get("extra_server_args") or "").strip(),
                        "extra_envs": envs,
                    },
                )
            )
        args, envs = _merge_named_current_recipe_configs(configs)
        return {"extra_server_args": args, "extra_envs": envs}

    async def _prepare_warm_kernel_kb(self, kb: Any = None) -> dict[str, Any]:
        """Prepare the Recipe's kernel section for the combined replay benchmark.

        This method never benchmarks. It stages Fusion/Rewrite files, resolves
        GEMM/env inputs, and returns the merged launch overlay plus rollback
        manifests. The one ``replay_warm_recipe`` task grades Recipe and Kernel
        together.
        """
        state = self.shared_state
        if getattr(state, "warm_kernel_kb_attempted", False):
            return {"status": "skipped", "reason": "already_attempted"}
        gated = self._warm_kernel_gate_reason()
        if gated:
            return self._set_warm_kernel_outcome({"status": "skipped", "reason": gated})
        state.warm_kernel_kb_attempted = True
        # Persist the one-shot flag now: a crash mid-replay must not re-run the
        # whole (potentially hour-long) set on resume.
        try:
            state.save(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "warm-kernel KB: one-shot state save failed",
                exc_info=True,
            )
            return self._set_warm_kernel_outcome(
                {
                    "status": "error",
                    "reason": (f"kernel_attempt_state_persist_failed:{type(exc).__name__}"),
                }
            )
        if kb is None:
            try:
                kb = self._open_warm_kernel_section()
            except Exception as exc:  # noqa: BLE001 — advisory; never block PRELUDE
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
            state.warm_replay_pending = {}
            outcome = self._set_warm_kernel_outcome({"status": "empty"})
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.debug(
                    "warm-kernel KB: empty-state save failed",
                    exc_info=True,
                )
            return outcome
        deferred = 0
        errors = 0
        applied: list[dict[str, Any]] = []
        kernel_snapshots: list[dict[str, Any]] = []
        prepared_envs: dict[int, dict[str, str]] = {}
        prepared_targets: dict[int, list[str]] = {}
        kernel_configs: list[tuple[str, Mapping[str, Any]]] = []
        for index, entry in enumerate(plan):
            envs = self._warm_kernel_extra_envs(entry)
            if entry.get("column") == "gemm":
                if not envs:
                    continue
            else:
                targets = self._resolve_kernel_target_paths(entry)
                if not targets:
                    continue
                prepared_targets[index] = targets
            prepared_envs[index] = envs
            kernel_configs.append(
                (
                    f"kernel[{index}]",
                    {
                        "extra_server_args": str((entry.get("meta") or {}).get("extra_server_args") or "").strip(),
                        "extra_envs": envs,
                    },
                )
            )
        kernel_args, merged_envs = _merge_named_current_recipe_configs(kernel_configs)
        state.warm_replay_pending = {
            **dict(getattr(state, "warm_replay_pending", {}) or {}),
            "status": "preparing_kernel",
            "kernel_apply_results": [],
            "kernel_snapshots": [],
        }
        try:
            state.save(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            outcome = {
                "status": "error",
                "reason": f"kernel_pending_state_persist_failed:{type(exc).__name__}",
            }
            return self._set_warm_kernel_outcome(outcome)
        for index, entry in enumerate(plan):
            if not (entry.get("source_paths") or entry.get("patch_path")):
                entry["decision"] = "DEFERRED"
                deferred += 1
                continue
            # Every column carries its env bundle: a fusion needs its switches to
            # activate the patched path, and a GEMM is nothing but its bundle.
            envs = prepared_envs.get(index, {})
            if envs:
                entry["extra_envs"] = envs
            # GEMM is parameter-shaped: its env bundle is the whole deliverable,
            # so there is nothing to stage on disk.
            if entry.get("column") == "gemm":
                if not envs:
                    entry["decision"] = "DEFERRED"
                    deferred += 1
                    continue
                entry["decision"] = "PENDING"
                continue
            targets = prepared_targets.get(index, [])
            if not targets:
                entry["decision"] = "DEFERRED"
                if entry.get("resolution_error"):
                    entry["apply_result"] = {
                        "status": "skipped",
                        "reason": str(entry["resolution_error"])[:300],
                    }
                deferred += 1
                continue
            anchor_target = targets[0]
            entry["resolved_patch_targets"] = list(targets)
            try:
                for target_path in targets:
                    snapshot = self._snapshot_warm_kernel_target(
                        target_path,
                        len(kernel_snapshots),
                    )
                    kernel_snapshots.append(snapshot)
                state.warm_replay_pending = {
                    **dict(state.warm_replay_pending or {}),
                    "status": "preparing_kernel",
                    "kernel_snapshots": list(kernel_snapshots),
                    "kernel_apply_results": list(applied),
                }
                state.save(self.session_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "warm-kernel KB: pre-mutation snapshot persist failed for %s",
                    targets,
                    exc_info=True,
                )
                entry["decision"] = "ERROR"
                entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
                errors += 1
                break
            try:
                apply_result = self._apply_warm_kernel_patch(entry, anchor_target)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "warm-kernel KB: apply failed for %s",
                    anchor_target,
                    exc_info=True,
                )
                entry["decision"] = "ERROR"
                entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
                errors += 1
                break
            entry["apply_result"] = {
                k: apply_result.get(k) for k in ("status", "reason", "error", "manifest_path") if k in apply_result
            }
            if apply_result.get("status") != "ok":
                entry["decision"] = "ERROR"
                errors += 1
                break
            applied.append(apply_result)
            state.warm_replay_pending = {
                **dict(state.warm_replay_pending or {}),
                "kernel_snapshots": list(kernel_snapshots),
                "kernel_apply_results": list(applied),
            }
            entry["decision"] = "PENDING"

        pending = [entry for entry in plan if entry.get("decision") == "PENDING"]
        columns = sorted({str(e.get("column")) for e in plan})
        if errors:
            rollback = self._revert_warm_kernel_patches(
                applied,
                kernel_snapshots,
            )
            for entry in pending:
                entry["decision"] = "ROLLED_BACK"
            if rollback.get("ok"):
                applied = []
                pending = []
                state.warm_replay_pending = {}
            else:
                state.warm_replay_pending = {
                    **dict(state.warm_replay_pending or {}),
                    "status": "rollback_failed",
                    "rollback_errors": list(rollback.get("errors") or []),
                }
        status = (
            "rollback_failed"
            if errors and not rollback.get("ok")
            else "error"
            if errors
            else "prepared"
            if pending
            else "loaded"
        )
        outcome = {
            "status": status,
            "columns": columns,
            "total": len(plan),
            "deferred": deferred,
            "errors": errors,
            "pending": pending,
            "applied": applied,
            "snapshots": kernel_snapshots,
            "rollback": rollback if errors else {"ok": True, "errors": []},
            "dirty": bool(errors and not rollback.get("ok")),
            "extra_envs": merged_envs if pending else {},
            "extra_server_args": (kernel_args if pending else ""),
        }
        if status == "loaded" and not applied and not kernel_snapshots:
            state.warm_replay_pending = {}
        self._set_warm_kernel_outcome(outcome)
        log.info(
            "PRELUDE warm-kernel KB prepared: columns=%s total=%d pending=%d deferred=%d errors=%d",
            columns,
            len(plan),
            len(pending),
            deferred,
            errors,
        )
        try:
            state.save(self.session_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "warm-kernel KB: prepared state save failed",
                exc_info=True,
            )
            persist_rollback = self._revert_warm_kernel_patches(
                applied,
                kernel_snapshots,
            )
            for entry in pending:
                entry["decision"] = "ROLLED_BACK"
            if persist_rollback.get("ok"):
                state.warm_replay_pending = {}
                failed_status = "error"
            else:
                state.warm_replay_pending = {
                    **dict(state.warm_replay_pending or {}),
                    "status": "rollback_failed",
                    "rollback_errors": list(persist_rollback.get("errors") or []),
                }
                failed_status = "rollback_failed"
            outcome = {
                **outcome,
                "status": failed_status,
                "reason": (f"kernel_prepared_state_persist_failed:{type(exc).__name__}"),
                "pending": [],
                "applied": [] if persist_rollback.get("ok") else applied,
                "rollback": persist_rollback,
                "dirty": not bool(persist_rollback.get("ok")),
                "extra_envs": {},
                "extra_server_args": "",
            }
            self._set_warm_kernel_outcome(outcome)
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.warning(
                    "warm-kernel KB: rollback state save failed",
                    exc_info=True,
                )
        return outcome

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
        combined_args, combined_envs = _merge_named_current_recipe_configs(
            [
                ("config", config),
                ("kernel", kernel_config),
            ]
        )
        # The column records its overlays in replay order, so the recorded
        # order is the order they are applied in.
        timeline = patch_kb.read_patches()
        if len(timeline) != len(set(timeline)):
            raise ValueError("current Recipe patch refs contain a duplicate")
        provenance = patch_kb.read_provenance()
        apply_roots = _recorded_apply_roots(provenance)
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
            # Each overlay carries the checkout it was applied into, so a Recipe
            # whose KEEPs came from different trees replays against each of them.
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

    async def _maybe_enqueue_warm_replay(
        self,
        *,
        baseline_tput: float,
    ) -> "Task | None":
        """Enqueue a one-shot ``replay_warm_recipe`` task for a high-confidence T0 prior.

        Skips on --no-warm-replay/resume/low-confidence/empty best_config; otherwise
        mints an internal task running the baseline workload contract with the KB
        config applied. Idempotent via warm-replay-prelude.

        Args:
            baseline_tput: The baseline throughput captured at enqueue time,
                carried forward as the replay's comparison anchor.

        Returns:
            The created (or existing) ``replay_warm_recipe`` task, or ``None``
            when the replay is skipped.
        """
        state = self.shared_state
        if not getattr(self, "_warm_replay_enabled", True):
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }
            # Flip the one-shot guard even on disabled-skip so a resume without --no-warm-replay can't
            # retroactively trigger a replay against the operator's original intent.
            state.warm_replay_attempted = True
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued/ran the replay.
            return None
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict):
            warm = {}
        current_remote = self._is_current_remote_recipe(warm)
        sdk_replay: dict[str, Any] = {}
        if current_remote:
            recipe_metadata = warm.get("recipe") or {}
            if isinstance(recipe_metadata, Mapping) and recipe_metadata.get("replayable") is False:
                state.warm_replay_attempted = True
                state.warm_replay_outcome = {
                    "status": "skipped",
                    "reason": str(recipe_metadata.get("replay_disabled_reason") or "remote_recipe_view_not_replayable"),
                    "view_source": str(recipe_metadata.get("view_source") or ""),
                }
                state.save(self.session_dir)
                return None
            try:
                sdk_replay = self._read_current_recipe_replay()
            except Exception as exc:  # noqa: BLE001 — current replay fails closed
                state.warm_replay_attempted = True
                state.warm_replay_outcome = {
                    "status": "skipped",
                    "reason": (f"current_recipe_sdk_read_failed:{type(exc).__name__}:{exc}")[:500],
                }
                state.save(self.session_dir)
                return None
            if patch_entries := list(sdk_replay.get("patches") or []):
                # Every overlay names the checkout it was measured on. One that
                # does not is a broken record, not a case to search a tree for:
                # any tree found that way is one this gain was never measured
                # against. The whole replay is skipped, never part of it.
                recorded = [str((entry or {}).get("framework_root") or "").strip() for entry in patch_entries]
                if not all(recorded):
                    state.warm_replay_attempted = True
                    state.warm_replay_outcome = self._warm_replay_root_skip_outcome(
                        reason="framework_apply_root_missing",
                        root_kind="framework",
                        roots=[root for root in recorded if root],
                    )
                    state.save(self.session_dir)
                    return None
                if absent := [root for root in dict.fromkeys(recorded) if not Path(root).is_dir()]:
                    state.warm_replay_attempted = True
                    state.warm_replay_outcome = self._warm_replay_root_skip_outcome(
                        reason="framework_apply_root_absent",
                        root_kind="framework",
                        roots=absent,
                    )
                    state.save(self.session_dir)
                    return None
        try:
            kernel = (
                await self._prepare_warm_kernel_kb(sdk_replay.get("kernel_kb"))
                if current_remote
                else await self._prepare_warm_kernel_kb()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("PRELUDE: warm-kernel preparation failed", exc_info=True)
            kernel = {
                "status": "error",
                "errors": 1,
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }
        preparation_pending = dict(getattr(state, "warm_replay_pending", {}) or {})
        preparation_dirty = bool(
            kernel.get("dirty")
            or preparation_pending.get("kernel_snapshots")
            or preparation_pending.get("kernel_apply_results")
        )
        if str(kernel.get("status") or "") in {"error", "rollback_failed"} and preparation_dirty:
            rollback = dict(kernel.get("rollback") or {}) if isinstance(kernel.get("rollback"), dict) else {}
            if not rollback:
                rollback = self._revert_warm_kernel_patches(
                    list(preparation_pending.get("kernel_apply_results") or []),
                    list(preparation_pending.get("kernel_snapshots") or []),
                )
            if rollback.get("ok"):
                state.warm_replay_pending = {}
            else:
                state.warm_replay_pending = {
                    **preparation_pending,
                    "status": "rollback_failed",
                    "rollback_errors": list(rollback.get("errors") or []),
                }
                if hasattr(state, "set_stop_reason"):
                    state.set_stop_reason("warm_replay_rollback_failed")
            state.warm_replay_attempted = True
            state.warm_replay_outcome = {
                "status": ("kernel_preparation_failed" if rollback.get("ok") else "rollback_failed"),
                "reason": str(kernel.get("reason") or "kernel preparation left mutable state"),
                "rollback": rollback,
            }
            state.save(self.session_dir)
            return None
        kernel_root_block = self._warm_replay_kernel_root_block_reason(state)
        if kernel_root_block is not None:
            state.warm_replay_attempted = True
            state.warm_replay_outcome = kernel_root_block
            state.save(self.session_dir)
            return None
        kernel_pending = list(kernel.get("pending") or []) if kernel.get("status") == "prepared" else []
        kernel_applied = list(kernel.get("applied") or []) if kernel_pending else []
        kernel_snapshots = list(kernel.get("snapshots") or []) if kernel_pending else []
        if current_remote and kernel_pending:
            kernel_config = sdk_replay.get("kernel_config") or {}
            kernel_envs = dict(kernel_config.get("extra_envs") or {})
            kernel_args = str(kernel_config.get("extra_server_args") or "")
        else:
            kernel_envs = dict(kernel.get("extra_envs") or {}) if kernel_pending else {}
            kernel_args = str(kernel.get("extra_server_args") or "") if kernel_pending else ""

        if not warm and not kernel_pending:
            if str(kernel.get("status") or "") in {"empty", "loaded"}:
                state.warm_replay_pending = {}
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "no_warm_start_recipe",
            }
            state.warm_replay_attempted = True
            return None
        # tier/conf stamped at T0.
        tier = str(warm.get("tier") or "").strip()
        try:
            conf = float(warm.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        min_conf = float(
            getattr(self, "_warm_replay_min_confidence", _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE)
            or _DEFAULT_WARM_REPLAY_MIN_CONFIDENCE
        )
        recipe = warm.get("recipe") or {}
        if not isinstance(recipe, dict):
            recipe = {}
        # best_config/sessions may be top-level or nested under attrs.
        recipe_attrs = recipe.get("attrs") or recipe
        wsc = getattr(state, "warm_start_context", None) or {}
        replay = wsc.get("recommended_replay") if not current_remote and isinstance(wsc, dict) else {}
        replay = replay if isinstance(replay, dict) else {}
        rep_args = str(replay.get("extra_server_args") or "").strip()
        rep_envs = replay.get("extra_envs") if isinstance(replay.get("extra_envs"), dict) else {}
        if current_remote:
            bc_args = str(sdk_replay.get("extra_server_args") or "").strip()
            bc_envs = dict(sdk_replay.get("extra_envs") or {})
            replay_conf = float(conf or 0.0)
            config_source = str(recipe.get("canonical_id") or "")
            config_tier = "self"
            donor_expected_gain = float(recipe.get("validated_gain_pct") or 0.0)
        elif rep_args or rep_envs:
            bc_args = rep_args
            bc_envs = dict(rep_envs)
            raw_replay_conf = replay.get("config_confidence")
            replay_conf = float(conf if raw_replay_conf is None else raw_replay_conf)
            config_source = str(replay.get("config_source") or "")
            config_tier = str(replay.get("config_tier") or "self")
            donor_expected_gain = float(replay.get("expected_gain_pct") or 0.0)
        else:
            best_config = recipe_attrs.get("best_config") or {}
            if not isinstance(best_config, dict):
                best_config = {}
            bc_args = str(best_config.get("extra_server_args") or "").strip()
            bc_envs = best_config.get("extra_envs") or {}
            if not isinstance(bc_envs, dict):
                bc_envs = {}
            replay_conf = float(conf or 0.0)
            config_source = str(recipe.get("canonical_id") or "")
            config_tier = "self"
            donor_expected_gain = 0.0
        donor_metadata = {
            field: replay.get(field)
            for field in (
                "donor_canonical_id",
                "donor_model",
                "donor_session_id",
                "donor_family_tags",
                "donor_gain_pct",
                "donor_breakdown_link",
            )
            if replay.get(field) not in (None, "", [])
        }
        recipe_suppressed = False
        # Low-confidence config/patch content is suppressed, but the kernel
        # section from the same Recipe can still receive a combined check.
        if replay_conf < min_conf:
            if kernel_pending:
                recipe_suppressed = True
                bc_args = ""
                bc_envs = {}
                config_source = ""
                config_tier = "suppressed_low_confidence"
                donor_expected_gain = 0.0
            else:
                state.warm_replay_outcome = {
                    "status": "skipped",
                    "reason": f"confidence_below_threshold ({replay_conf:.2f} < {min_conf:.2f})",
                    "warm_recipe_tier": tier,
                    "warm_recipe_conf": conf,
                    "config_donor_tier": config_tier,
                    "config_source": config_source,
                    **donor_metadata,
                }
                state.warm_replay_attempted = True
                return None
        # Current records derive fail-closed mode from their exact SDK timeline.
        wsc_patches = (
            list(sdk_replay.get("patches") or [])
            if current_remote
            else (wsc.get("recommended_replay") or {}).get("patches") or []
            if isinstance(wsc, dict)
            else []
        )
        required_patch_timeline = bool(
            current_remote
            and sdk_replay.get("timeline")
            or (not current_remote and recipe_attrs.get("required_patch_timeline"))
        )
        if recipe_suppressed:
            wsc_patches = []
            required_patch_timeline = False
        if wsc_patches:
            # Re-checked here because a persisted plan may be resumed under a
            # different image, and because the kernel half is already applied by
            # now: a root that has since gone means unwinding that too.
            recorded_roots = [str((entry or {}).get("framework_root") or "").strip() for entry in wsc_patches]
            unusable = [root for root in dict.fromkeys(recorded_roots) if root and not Path(root).is_dir()]
            if not all(recorded_roots) or unusable:
                rollback = (
                    self._revert_warm_kernel_patches(
                        kernel_applied,
                        kernel_snapshots,
                    )
                    if kernel_applied or kernel_snapshots
                    else {"ok": True, "errors": []}
                )
                if rollback.get("ok"):
                    state.warm_replay_pending = {}
                else:
                    state.warm_replay_pending = {
                        **dict(getattr(state, "warm_replay_pending", {}) or {}),
                        "status": "rollback_failed",
                        "rollback_errors": list(rollback.get("errors") or []),
                    }
                    if hasattr(state, "set_stop_reason"):
                        state.set_stop_reason("warm_replay_rollback_failed")
                state.warm_replay_attempted = True
                state.warm_replay_outcome = self._warm_replay_root_skip_outcome(
                    reason=("framework_apply_root_absent" if unusable else "framework_apply_root_missing"),
                    root_kind="framework",
                    roots=unusable or [root for root in recorded_roots if root],
                    rollback=rollback,
                )
                state.save(self.session_dir)
                return None
        if not bc_args and not bc_envs and not wsc_patches and not kernel_pending:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "best_config_empty",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
                **donor_metadata,
            }
            state.warm_replay_attempted = True
            return None
        # Historical gain anchor: donor's expected gain, else MAX gain across
        # attrs.sessions[], else the flat gain_pct.
        expected_gain = donor_expected_gain
        sessions_field = recipe_attrs.get("sessions")
        if expected_gain <= 0 and isinstance(sessions_field, list):
            session_gains: list[float] = []
            for s in sessions_field:
                if not isinstance(s, dict):
                    continue
                try:
                    g = float(s.get("gain_pct") or 0.0)
                except (TypeError, ValueError):
                    continue
                session_gains.append(g)
            if session_gains:
                expected_gain = max(session_gains)
        # Last-chance fallback for offline-ingested seed rows.
        if expected_gain <= 0:
            try:
                fallback = float(recipe_attrs.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                fallback = 0.0
            if fallback > 0:
                expected_gain = fallback
        if current_remote:
            if recipe_suppressed:
                combined_args = kernel_args
                combined_envs = dict(kernel_envs)
            else:
                combined_args = str(sdk_replay.get("combined_extra_server_args") or "")
                combined_envs = dict(sdk_replay.get("combined_extra_envs") or {})
        else:
            from ..loop.coordinator_helpers import (
                _merge_cumulative_extra_server_args,
            )

            combined_args = _merge_cumulative_extra_server_args(
                bc_args,
                kernel_args,
                "",
            )
            combined_envs = dict(bc_envs)
            combined_envs.update(kernel_envs)
        try:
            from ..actions.executors._grid_server_args import (
                validate_warm_replay_context_length,
            )

            validated_args, workload_compatibility = validate_warm_replay_context_length(
                combined_args,
                getattr(state, "framework", ""),
                int(getattr(state, "isl", 0) or 0),
                int(getattr(state, "osl", 0) or 0),
                getattr(state, "max_model_len", None),
            )
            if validated_args != combined_args:
                raise RuntimeError("warm replay context preflight must not mutate config")
        except (ValueError, RuntimeError) as exc:
            rollback = (
                self._revert_warm_kernel_patches(
                    kernel_applied,
                    kernel_snapshots,
                )
                if kernel_applied or kernel_snapshots
                else {"ok": True, "errors": []}
            )
            if rollback.get("ok"):
                state.warm_replay_pending = {}
            else:
                state.warm_replay_pending = {
                    **dict(getattr(state, "warm_replay_pending", {}) or {}),
                    "status": "rollback_failed",
                    "rollback_errors": list(rollback.get("errors") or []),
                }
                if hasattr(state, "set_stop_reason"):
                    state.set_stop_reason("warm_replay_rollback_failed")
            state.warm_replay_attempted = True
            state.warm_replay_outcome = {
                "status": ("skipped" if rollback.get("ok") else "rollback_failed"),
                "reason": (f"workload_config_incompatible:{type(exc).__name__}:{exc}")[:500],
                "target_workload_shape": {
                    "conc": int(getattr(state, "conc", 0) or 0),
                    "isl": int(getattr(state, "isl", 0) or 0),
                    "osl": int(getattr(state, "osl", 0) or 0),
                },
                "rollback": rollback,
            }
            state.save(self.session_dir)
            return None
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "extra_server_args": combined_args,
            "extra_envs": combined_envs,
            "recipe_extra_server_args": bc_args,
            "recipe_extra_envs": dict(bc_envs),
            "kernel_extra_server_args": kernel_args,
            "kernel_extra_envs": kernel_envs,
            # Reuse the baseline's workload contract (else YAML smoke defaults).
            "config_path": str(state.baseline_config_path or ""),
            # Historical-gain anchor for the promote path's reproduce ratio.
            "warm_expected_gain_pct": expected_gain,
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            # Config provenance ("self" when the identity match owned it).
            "config_donor_tier": config_tier,
            "config_source": config_source,
            "baseline_tput_anchor": float(baseline_tput),
            # Code patches to apply before server launch.
            "patches": list(wsc_patches),
            "required_patch_timeline": required_patch_timeline,
            "warm_kernel_plan": kernel_pending,
            "warm_kernel_apply_results": kernel_applied,
            "warm_kernel_snapshots": kernel_snapshots,
            "combined_current_contract": bool(current_remote or kernel_pending),
            "combined_keep_threshold_pct": _warm_kernel_keep_threshold_pct(self.shared_state),
            "workload_compatibility": workload_compatibility,
        }
        try:
            lanes, ttl = self._registry_lanes_ttl("replay_warm_recipe")
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="replay_warm_recipe",
                params=params,
                idempotency_key="warm-replay-prelude",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
        except Exception as exc:
            rollback = self._revert_warm_kernel_patches(
                kernel_applied,
                kernel_snapshots,
            )
            if rollback.get("ok"):
                state.warm_replay_pending = {}
            else:
                state.warm_replay_pending = {
                    **dict(getattr(state, "warm_replay_pending", {}) or {}),
                    "status": "rollback_failed",
                    "rollback_errors": list(rollback.get("errors") or []),
                }
                if hasattr(state, "set_stop_reason"):
                    state.set_stop_reason("warm_replay_rollback_failed")
            state.warm_replay_attempted = True
            state.warm_replay_outcome = {
                "status": ("enqueue_failed" if rollback.get("ok") else "rollback_failed"),
                "reason": f"warm replay enqueue failed: {type(exc).__name__}",
                "rollback": rollback,
            }
            state.save(self.session_dir)
            raise
        if not was_existing:
            log.info(
                "PRELUDE: warm-replay enqueued task=%s (tier=%s conf=%.2f expected_gain=%.2f baseline_tput=%.2f)",
                task.task_id,
                tier,
                conf,
                expected_gain,
                baseline_tput,
            )
        state.warm_replay_attempted = True
        state.warm_replay_outcome = {
            "status": "in_flight",
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            "config_donor_tier": config_tier,
            "config_source": config_source,
            "expected_gain_pct": expected_gain,
            "replay_task_id": task.task_id,
            "kernel_count": len(kernel_pending),
            "recipe_suppressed": recipe_suppressed,
            **({"overlay_provenance": summary} if (summary := _overlay_provenance_summary(sdk_replay)) else {}),
            **donor_metadata,
        }
        state.warm_replay_pending = {
            **dict(getattr(state, "warm_replay_pending", {}) or {}),
            "status": "in_flight",
            "task_id": task.task_id,
            "kernel_apply_results": kernel_applied,
            "kernel_plan": kernel_pending,
            "kernel_snapshots": kernel_snapshots,
        }
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.debug("combined warm replay pending save failed", exc_info=True)
        return task

    def _resolve_promoted_recipe_checkout(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> tuple[bool, dict[str, Any]]:
        """Validate every already-patched checkout selected by warm replay.

        A replay may have patched more than one tree, because each overlay is
        placed into the checkout it was recorded against. All of them are
        promoted or none is: the measured gain came from the whole set.
        """
        params = (task.params if task is not None else {}) or {}
        if not params.get("required_patch_timeline") or not params.get("patches"):
            return True, {"status": "not_required"}

        trees = [tree for tree in (result.get("warm_patch_trees") or []) if isinstance(tree, Mapping)]
        if not trees:
            # A round that predates the list reports only its primary tree.
            trees = [
                {
                    "root": result.get("warm_patch_target"),
                    "pre_sha": result.get("warm_patch_pre_sha"),
                    "snapshot_manifest": result.get("warm_patch_snapshot_manifest"),
                }
            ]
        promoted: list[str] = []
        for tree in trees:
            target = str(tree.get("root") or "").strip()
            pre_sha = str(tree.get("pre_sha") or "").strip()
            manifest = tree.get("snapshot_manifest")
            if not target or not pre_sha or not isinstance(manifest, Mapping):
                return False, {
                    "status": "failed",
                    "failure": "validated_recipe_checkout_incomplete",
                    "target_repo": target,
                }
            try:
                target_path = Path(target).resolve(strict=True)
                manifest_target = Path(str(manifest.get("repo_path") or "")).resolve(strict=True)
            except (OSError, ValueError) as exc:
                return False, {
                    "status": "failed",
                    "failure": f"validated_recipe_checkout_invalid:{type(exc).__name__}",
                }
            if target_path != manifest_target:
                return False, {
                    "status": "failed",
                    "failure": "validated_recipe_checkout_manifest_mismatch",
                    "target_repo": str(target_path),
                }
            promoted.append(str(target_path))
        return True, {
            "status": "promoted",
            "target_repo": promoted[0],
            "target_repos": promoted,
        }

    def _rollback_combined_warm(
        self,
        result: dict[str, Any],
        task: "Task | None",
    ) -> dict[str, Any]:
        """Restore both Recipe and Kernel halves of a combined replay.

        The Recipe half may span several checkouts, so every tree the replay
        patched is restored -- leaving one behind would bank a mutation from a
        replay that was rejected.
        """
        from ..actions.executors.baseline import _revert_patches

        restores: list[dict[str, Any]] = []
        pending = getattr(self.shared_state, "warm_replay_pending", {}) or {}
        trees = [
            tree
            for tree in (result.get("warm_patch_trees") or pending.get("recipe_patch_trees") or [])
            if isinstance(tree, Mapping)
        ]
        if not trees:
            trees = [
                {
                    "root": result.get("warm_patch_target") or pending.get("recipe_patch_target"),
                    "pre_sha": result.get("warm_patch_pre_sha") or pending.get("recipe_patch_pre_sha"),
                    "snapshot_manifest": (
                        result.get("warm_patch_snapshot_manifest") or pending.get("recipe_patch_snapshot_manifest")
                    ),
                }
            ]
        for tree in trees:
            target = str(tree.get("root") or "")
            if not target:
                continue
            recipe_manifest = tree.get("snapshot_manifest")
            restores.append(
                _revert_patches(target, str(tree.get("pre_sha") or ""), recipe_manifest)
                if recipe_manifest
                else {"ok": False, "errors": [f"recipe:{target}:missing_snapshot_manifest"]}
            )
        params = (task.params if task is not None else {}) or {}
        kernel_applied = (
            result.get("warm_kernel_apply_results")
            or params.get("warm_kernel_apply_results")
            or pending.get("kernel_apply_results")
            or []
        )
        kernel_restore = self._revert_warm_kernel_patches(
            list(kernel_applied),
            list(
                result.get("warm_kernel_snapshots")
                or params.get("warm_kernel_snapshots")
                or pending.get("kernel_snapshots")
                or []
            ),
        )
        restores.append(kernel_restore)
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

    def _require_combined_warm_rollback(
        self,
        result: dict[str, Any],
        task: "Task | None",
        outcome: dict[str, Any],
    ) -> bool:
        """Rollback or persist a terminal recovery failure without clearing it."""
        rollback = self._rollback_combined_warm(result, task)
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
        self.shared_state.warm_replay_outcome = outcome
        if hasattr(self.shared_state, "set_stop_reason"):
            self.shared_state.set_stop_reason("warm_replay_rollback_failed")
        self.shared_state.save(self.session_dir)
        return False

    @staticmethod
    def _replay_eval_search_root(result: dict) -> Path | None:
        """Directory holding every round of this replay task.

        The cold-start guard evaluates only in the warmup round but decides on
        the measure round, so the score lands in a sibling directory rather
        than under the deciding round's own workspace. Searching the workspace
        alone finds nothing on every double-run replay: across 852 recorded
        replays the score sat in ``warmup_round`` 320 times and in
        ``measure_round`` never.

        Args:
            result: The ``replay_warm_recipe`` result envelope.

        Returns:
            The task-level directory containing both round directories, or
            ``None`` when the result names no usable path.
        """
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

    def _warm_replay_accuracy_ok(
        self,
        result: dict,
        task: "Task | None",
        outcome: dict,
    ) -> bool:
        """Whether a replayed config may be promoted on accuracy grounds.

        Every replay is judged, not just the ones touching a knob known to be
        risky: a KB recipe is evidence from another session and another
        machine, so reproducing its throughput says nothing about whether it
        still computes correctly here. The measured score is recorded either
        way — a promotion that was checked and passed is not the same record as
        one that was never checked.

        ``eval_ran`` separates the two ways ``replay_accuracy`` can be absent.
        A score of 0.0 means the model answered nothing; no score at all means
        no evidence either way, and those must not collapse into one state.

        Args:
            result: The ``replay_warm_recipe`` result envelope.
            task: The originating task, carrying the replayed args/envs.
            outcome: The warm-replay outcome dict, stamped either way.

        Returns:
            ``True`` when promotion may proceed; ``False`` when the caller must
            stop (the rollback and outcome have already been recorded).
        """
        from ..actions.executors._accuracy_gate import (
            DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
            accuracy_meets_floor,
            accuracy_passed,
            parse_eval_results,
        )

        state = self.shared_state
        try:
            baseline_accuracy = float(getattr(state, "baseline_accuracy", 0.0) or 0.0)
        except (TypeError, ValueError):
            baseline_accuracy = 0.0

        measured = result.get("accuracy")
        eval_ran = isinstance(measured, (int, float))
        eval_error = ""
        if not eval_ran:
            measured = None
            root = self._replay_eval_search_root(result)
            if root is None:
                eval_error = "replay result names no round directory"
            else:
                try:
                    eval_out = parse_eval_results(root)
                except Exception as exc:  # noqa: BLE001 — an unreadable eval is "no verdict"
                    eval_out = {"error": f"eval parse raised: {type(exc).__name__}"}
                parsed = eval_out.get("accuracy")
                if isinstance(parsed, (int, float)):
                    measured = float(parsed)
                    eval_ran = True
                else:
                    eval_error = str(eval_out.get("error") or "no accuracy in eval output")
                    # Only a results file the parser reached can say the eval
                    # ran; a parser that never got one says nothing either way.
                    eval_ran = eval_error.startswith(_EVAL_RAN_BUT_UNSCORABLE)

        outcome["eval_ran"] = bool(eval_ran)
        outcome["eval_error"] = eval_error or None
        outcome["replay_accuracy"] = float(measured) if measured is not None else None
        outcome["baseline_accuracy"] = baseline_accuracy if baseline_accuracy > 0 else None

        if measured is None:
            # A measurement that failed is not evidence the config broke the
            # model, so it must not stop the run: the replay is admitted and the
            # reason it could not be judged is recorded instead. ``eval_ran``
            # says whether an eval produced nothing or never ran at all.
            log.warning(
                "warm-replay admitted without an accuracy verdict (eval_ran=%s, baseline %.4f): %s",
                eval_ran,
                baseline_accuracy,
                eval_error or "no reason recorded",
            )
            return True
        if baseline_accuracy > 0:
            if accuracy_passed(baseline_accuracy, float(measured)):
                return True
            reason = (
                f"accuracy regression on the replayed config (baseline {baseline_accuracy:.4f}, replay {measured:.4f})"
            )
        elif accuracy_meets_floor(measured, DEFAULT_ENABLEMENT_ACCURACY_FLOOR):
            return True
        else:
            reason = (
                "accuracy below absolute floor on the replayed config "
                f"(replay {measured:.4f}, "
                f"floor {DEFAULT_ENABLEMENT_ACCURACY_FLOOR:.2f})"
            )
        if not self._require_combined_warm_rollback(result, task, outcome):
            return False
        outcome["status"] = "accuracy_failed"
        outcome["reason"] = reason
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)
        log.info("warm-replay REJECTED on accuracy: %s", reason)
        return False

    def _promote_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Interpret a combined Recipe+Kernel ``replay_warm_recipe`` result.

        Measured uplift promotes the warm config onto ``optimization_stack`` and
        ``current_best``. Failures (including a failed required patch timeline)
        roll back both halves fail-closed, set an outcome status, and never
        propagate.

        Args:
            result: The ``replay_warm_recipe`` task result dict (status,
                throughput, workspace, etc.).
            task: The originating task, used to recover the warm args/envs and
                the baseline anchor; may be ``None`` (degraded path).
        """
        state = self.shared_state
        outcome = dict(state.warm_replay_outcome or {})
        expected_gain = float(outcome.get("expected_gain_pct") or 0.0)
        if not isinstance(result, dict):
            if not self._require_combined_warm_rollback({}, task, outcome):
                return
            outcome["status"] = "failed"
            outcome["reason"] = "non_dict_result"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        status = str(result.get("status") or "")
        if status != "succeeded":
            if not self._require_combined_warm_rollback(result, task, outcome):
                return
            outcome["status"] = "failed"
            outcome["error_class"] = str(result.get("error_class") or "")
            outcome["reason"] = str(result.get("error") or result.get("reason") or "")[:240]
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info(
                "warm-replay failed (status=%s, error_class=%s)",
                status,
                outcome.get("error_class"),
            )
            return
        tput_raw = result.get("output_throughput")
        try:
            tput = float(tput_raw) if tput_raw is not None else 0.0
        except (TypeError, ValueError):
            tput = 0.0
        # ``tput`` (output_throughput) is the HOT measure round and the
        # comparison value; the warmup round is retained for audit only.
        cold_raw = result.get("warmup_round_tput")
        try:
            cold_round_tput = float(cold_raw) if cold_raw is not None else 0.0
        except (TypeError, ValueError):
            cold_round_tput = 0.0
        single_round_tput = tput
        hot_tput = tput
        # Use the baseline_tput captured at enqueue time (fall back to live state).
        anchor_raw = None
        if task is not None and isinstance(getattr(task, "params", None), dict):
            anchor_raw = task.params.get("baseline_tput_anchor")
        try:
            baseline_tput = float(anchor_raw) if anchor_raw is not None else 0.0
        except (TypeError, ValueError):
            baseline_tput = 0.0
        if baseline_tput <= 0:
            baseline_tput = float(state.baseline_tput or 0.0)
        if single_round_tput <= 0 or baseline_tput <= 0:
            if not self._require_combined_warm_rollback(result, task, outcome):
                return
            outcome["status"] = "failed"
            outcome["reason"] = f"invalid_tput tput={single_round_tput} baseline={baseline_tput}"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        # warm_replay is an optimization candidate, so it must clear the
        # image-quality gate against the baseline reference before promotion.
        # ``require=False`` keeps a missing/skipped gate non-blocking.
        from ..actions.executors._accuracy_gate import quality_gate_passed

        qg = result.get("quality_gate")
        if qg is not None and not quality_gate_passed(qg, require=False):
            if not self._require_combined_warm_rollback(result, task, outcome):
                return
            outcome["status"] = "quality_failed"
            outcome["reason"] = "image-quality gate failed vs baseline reference"
            outcome["quality_gate"] = qg
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info("warm-replay REJECTED by quality gate: %s", qg)
            return
        # A replayed config lands on ``current_best``, so every later
        # measurement in the session is taken against it. Promoting one on
        # throughput alone is how a config that makes the model emit garbage
        # becomes the session's reference: breaking the numerics is itself a
        # large throughput win, so the objective actively selects for it.
        # Every replay is judged here, not only high-risk knobs: a KB recipe is
        # evidence from another session, so reproducing its throughput says
        # nothing about whether it still computes correctly here.
        if not self._warm_replay_accuracy_ok(result, task, outcome):
            return
        measured_gain = (single_round_tput / baseline_tput - 1.0) * 100.0
        result["combined_gain_pct"] = round(measured_gain, 3)
        decision_params = (task.params if task is not None else {}) or {}
        combined_current_contract = bool(decision_params.get("combined_current_contract"))
        keep_threshold = 0.0
        if combined_current_contract:
            raw_threshold = decision_params.get("combined_keep_threshold_pct")
            try:
                keep_threshold = (
                    float(raw_threshold)
                    if raw_threshold is not None
                    else _warm_kernel_keep_threshold_pct(self.shared_state)
                )
            except (TypeError, ValueError):
                keep_threshold = _warm_kernel_keep_threshold_pct(self.shared_state)
            if not math.isfinite(keep_threshold):
                keep_threshold = _warm_kernel_keep_threshold_pct(self.shared_state)
        min_reproduce = float(
            getattr(self, "_warm_replay_min_reproduce_pct", 0.8) or 0.8,
        )
        # Local legacy replay keeps any positive gain. The current combined
        # contract must clear the approved kernel replay threshold.
        reproduced = measured_gain >= keep_threshold if combined_current_contract else measured_gain > 0
        outcome["actual_gain_pct"] = round(measured_gain, 3)
        outcome["throughput_after"] = tput
        outcome["keep_threshold_pct"] = keep_threshold
        if expected_gain > 0:
            historical_bar = expected_gain * min_reproduce
            if measured_gain > 0 and measured_gain < historical_bar:
                outcome["below_historical_reproduce_pct"] = True
                outcome["historical_reproduce_bar_pct"] = round(
                    historical_bar,
                    3,
                )
        promoted_checkout = ""
        if reproduced:
            params = (task.params if task is not None else {}) or {}
            promoted, promotion = self._resolve_promoted_recipe_checkout(
                result,
                task,
            )
            if not promoted:
                if not self._require_combined_warm_rollback(
                    result,
                    task,
                    outcome,
                ):
                    return
                outcome["status"] = "promotion_failed"
                outcome["reason"] = str(promotion.get("failure") or "validated Recipe checkout promotion failed")
                outcome["recipe_checkout_promotion"] = promotion
                kernel_outcome = {
                    "status": "reverted",
                    "reason": "recipe_checkout_promotion_failed",
                    "validation": "combined_recipe_kernel",
                }
                outcome["kernel"] = kernel_outcome
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            promoted_checkout = (
                str(promotion.get("target_repo") or "").strip() if promotion.get("status") == "promoted" else ""
            )
            if promoted_checkout:
                outcome["active_framework_root"] = promoted_checkout
                # Resume re-points $INFERENCEX_PATH at this checkout, and stops
                # the run when it has since vanished.
                state.active_inferencex_path = promoted_checkout
            warm_args = str(params.get("extra_server_args") or "").strip()
            warm_envs = dict(params.get("extra_envs") or {})
            replayed_patch_refs = [
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
                            "present_in_dirty_worktree",
                        }
                    )
                    and str(item.get("patch_file") or "")
                )
            ]
            has_kernel = bool(params.get("warm_kernel_plan"))
            if not warm_args and not warm_envs and not replayed_patch_refs and not has_kernel:
                outcome["status"] = "reproduced_but_no_params"
                outcome["reason"] = "task.params missing extra_server_args/extra_envs and no warm patch was applied"
                log.warning(
                    "warm-replay measured +%.2f%% but cannot push stack (task=%r has no warm args/envs)",
                    measured_gain,
                    task,
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            outcome["status"] = "reproduced"
            outcome.pop("replayed_patch_refs", None)
            if replayed_patch_refs:
                outcome["replayed_patch_refs"] = replayed_patch_refs
            # Stack-entry-only metadata; the lift keeps current_best pure config.
            entry_extra: dict[str, Any] = {
                "gain_pct": round(measured_gain, 3),
                "hot_tput": float(hot_tput),
                "cold_tput": float(cold_round_tput) if cold_round_tput > 0 else None,
                # The score this promotion was judged on, recorded alongside the
                # throughput it was judged with. ``None`` means no score could be
                # read, not that the model scored nothing — ``eval_ran`` on the
                # outcome separates those.
                "accuracy": outcome.get("replay_accuracy"),
                # source_tier records the warm-recipe tier for breakdown attribution.
                "source_tier": outcome.get("warm_recipe_tier", ""),
                "source_confidence": outcome.get("warm_recipe_conf", 0.0),
            }
            if promoted_checkout:
                entry_extra["framework_source_root"] = promoted_checkout
            kernel_outcome = self._book_combined_kernel_keep(result, task)
            outcome["kernel"] = dict(kernel_outcome)
            if kernel_outcome.get("kept"):
                entry_extra["kernel_replay"] = {
                    "validation": "combined_recipe_kernel",
                    "count": kernel_outcome["kept"],
                    "columns": sorted(
                        {
                            str(entry.get("column") or "")
                            for entry in state.warm_kernel_kb_plan
                            if isinstance(entry, dict)
                        }
                    ),
                }
            patch_result = result.get("warm_patch_result")
            if isinstance(patch_result, dict):
                entry_extra["recipe_patch_statuses"] = list(patch_result.get("patches") or [])
            if replayed_patch_refs:
                entry_extra["replayed_patch_refs"] = replayed_patch_refs
            # Resume safety: do not clobber existing stack entries.
            state.optimization_stack = list(state.optimization_stack or [])
            # A prior promote owns the outcome; re-running would re-journal it.
            already_pushed = any(
                isinstance(e, dict) and e.get("action") == "replay_warm_recipe" for e in state.optimization_stack
            )
            if already_pushed:
                log.info(
                    "warm-replay promote: stack already carries the entry; "
                    "skipping duplicate push (likely resume mid-promote)",
                )
                state.warm_replay_pending = {}
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            recipe_args = str(
                params["recipe_extra_server_args"] if "recipe_extra_server_args" in params else warm_args
            ).strip()
            recipe_envs = dict(params["recipe_extra_envs"] if "recipe_extra_envs" in params else warm_envs)
            self._lift_to_current_best(
                "replay_warm_recipe",
                float(single_round_tput),
                {
                    "name": "warm_replay",
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
                    "task_id": str(getattr(task, "task_id", "") or ""),
                    "workspace": str(result.get("workspace") or ""),
                },
                entry_extra=entry_extra,
            )
            # Publish the reproduced verdict now that the stack entry exists but
            # before the cumulative update (the one step below that can raise and
            # is swallowed by the caller). The post-ruling mirror reads this
            # in-memory outcome: persisting it here keeps the canonical adoption
            # in step with the stack. Placed after the lift on purpose -- if the
            # lift itself raises, the outcome stays in_flight and both the stack
            # and the mirror agree there is nothing adopted.
            state.warm_replay_outcome = outcome
            if baseline_tput > 0:
                self._update_cumulative_gain_validated(single_round_tput)
            log.info(
                "warm-replay REPRODUCED: measured=+%.2f%% (expected=+%.2f%%, "
                "min_required=+%.2f%%); pushed warm_replay onto stack",
                measured_gain,
                expected_gain,
                expected_gain * min_reproduce if expected_gain > 0 else 0.0,
            )
            # Journal warm-replay as a synthetic KEEP; no KB lesson.
            try:
                journal = self._ensure_journal()
                from ..state.optimization_journal import KIND_OTHER, OUTCOME_KEEP

                journal.append_entry(
                    JournalEntry(
                        phase=str(getattr(state, "phase", "PRELUDE")).upper() or "PRELUDE",
                        iter=int(state.tick or 0),
                        kind=KIND_OTHER,
                        change=f"warm_replay({outcome.get('warm_recipe_tier', '?')}): {warm_args}",
                        outcome=OUTCOME_KEEP,
                        gain_pct=round(measured_gain, 3),
                        throughput_after=tput,
                        task_id=str(task.task_id if task is not None else ""),
                        tick=int(state.tick or 0),
                    )
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay journal append failed")
        else:
            if not self._require_combined_warm_rollback(result, task, outcome):
                return
            kernel_plan = (task.params or {}).get("warm_kernel_plan") if task is not None else []
            kernel_outcome = {
                "status": "reverted",
                "total": len(kernel_plan or []),
                "kept": 0,
                "reverted": len(kernel_plan or []),
                "validation": "combined_recipe_kernel",
            }
            outcome["kernel"] = dict(kernel_outcome)
            outcome["status"] = "drift"
            outcome["reason"] = f"measured {measured_gain:+.2f}% below keep threshold {keep_threshold:+.2f}%"
            log.info(
                "warm-replay DRIFT: measured=%+.2f%% threshold=%+.2f%%",
                measured_gain,
                keep_threshold,
            )
        state.warm_replay_pending = {}
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)

    async def _maybe_enqueue_prelude_initial_analysis_after_baseline(
        self,
        *,
        baseline_tput: float | None = None,
    ) -> None:
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline; skipped while warm-replay is in_flight (GPU/port contention).

        Args:
            baseline_tput: The baseline throughput; ``None`` reads it from
                SharedState. A non-positive value short-circuits the enqueue.
        """
        state = self.shared_state
        if _phase_state.warm_replay_in_flight(state):
            log.info(
                "PRELUDE: deferring initial %s until warm-replay completes",
                self._internal_analysis_kind(),
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
                self._internal_analysis_kind(),
                evidence.get("affordable_sec", 0.0),
                evidence.get("bound", ""),
                evidence.get("expected_cost_sec", 0.0),
            )
            self._record_prelude_arm_dropped("initial_analysis", evidence)
            return
        try:
            rl_task = await self._enqueue_internal_analysis_task(
                reason="prelude_initial",
            )
            state.auto_roofline_pending_task_id = rl_task.task_id
            log.info(
                "PRELUDE: baseline landed (tput=%.2f); auto-enqueued initial %s task=%s",
                float(baseline_tput),
                rl_task.kind,
                rl_task.task_id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "PRELUDE: failed to enqueue initial analysis task after baseline: %r",
                exc,
            )

    def _analysis_attempt_suffix(self, kind: str) -> str:
        """Idempotency-key suffix separating a re-armed roofline retry from the
        attempt that failed.

        ``_needs_roofline_for_watermark`` deliberately re-arms once a roofline
        has failed — a failed analysis must not suppress later refreshes. But
        the key it re-enqueued under was a per-cycle singleton, so the registry
        handed back the failed attempt and the retry the system had just armed
        for was deduplicated away. One trace failure therefore blacked out the
        GPU side of the search for a whole macro-cycle: four sessions ran with
        an empty roofline, no kernel hot spots ever reached a specialist, and
        the log said "enqueued" every time.

        The failure streak is the attempt counter and already resets to zero on
        a successful snapshot, so a roofline that worked is still never re-run.

        Args:
            kind: The analysis task kind; only ``roofline`` retries.

        Returns:
            ``"-a<streak>"`` while a roofline retry is outstanding, else ``""``.
        """
        if kind != "roofline":
            return ""
        try:
            streak = int(getattr(self.shared_state, "roofline_failure_streak", 0) or 0)
        except (TypeError, ValueError):
            streak = 0
        return f"-a{streak}" if streak > 0 else ""

    async def _enqueue_internal_analysis_task(self, *, reason: str) -> Task:
        """Build + enqueue a Coordinator-internal analysis task (roofline or profile). Idempotency key internal-analysis-<reason>.

        Args:
            reason: Tag distinguishing the enqueue site; used in the
                idempotency key and to select baseline vs current-best args.

        Returns:
            The created (or existing idempotent) analysis :class:`Task`.
        """
        state = self.shared_state
        kind = self._internal_analysis_kind()
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        if reason != "prelude_initial":
            inject_stack_base_params(params, state)
        else:
            # PRELUDE roofline profiles the baseline arm: inject baseline's own
            # server args (never current_best's) so a later warm-replay can't
            # swap in flags that skew the baseline ceiling.
            try:
                from ..kernel.roofline_ceiling import read_baseline_server_args

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
        lanes, ttl = self._registry_lanes_ttl(kind)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=(
                f"internal-analysis-{reason}{self._cycle_idem_suffix()}{self._analysis_attempt_suffix(kind)}"
            ),
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            log.info(
                "internal-analysis task already exists (idempotent: kind=%s task_id=%s, state=%s)",
                kind,
                task.task_id,
                task.state,
            )
        return task
