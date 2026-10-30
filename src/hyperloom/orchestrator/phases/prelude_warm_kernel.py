# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Warm-kernel preparation for PRELUDE warm replay: plan, land, overlay and unwind the Recipe's kernel section."""

from __future__ import annotations

import logging as _logging
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.inference_optimizer.breakdown.recorder.warm_replay_event import SKIP_KERNEL_PREPARATION_FAILED

from ..actions.executors.benchmark_run.warm_patches import revert_warm_kernel_patches
from ..knowledge.remote_recipe.sanitize import HOST_ORIGIN_KEY

log = _logging.getLogger(__name__)

# Per-row path refs a kernel plan entry resolves itself, dropped from the Recipe row it carries forward, and the
# content refs dropped again from that entry's ``meta`` summary.
_KERNEL_ROW_PATH_KEYS = ("source_file", "source_files", "target_file", "target_files", "target_path")
_KERNEL_ROW_CONTENT_KEYS = ("patch", "tuned_file", "files")


class _KernelLaunchOverlay(NamedTuple):
    """The launch config a kernel plan contributes, with the envs and targets of each contributing entry by index."""

    extra_server_args: str
    extra_envs: dict[str, str]
    entry_envs: dict[int, dict[str, str]]
    entry_targets: dict[int, list[str]]


def _merge_named_current_recipe_configs(owners: list[tuple[str, Mapping[str, Any]]]) -> tuple[str, dict[str, str]]:
    """Merge named config snapshots with exact duplicate conflict checks."""
    from hyperloom.inference_optimizer.grid_server_args import tokenize_server_args_preserving_json

    pairs: dict[str, tuple[str, ...]] = {}
    positional = 0
    for owner, config in owners:
        raw = str(config.get("extra_server_args") or "").strip()
        if not raw:
            continue
        parsed = tokenize_server_args_preserving_json(raw)
        if parsed is None:
            raise ValueError(f"{owner} extra_server_args cannot be tokenized safely")
        tokens = parsed[1]
        index = 0
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if not token.startswith("--"):
                pairs[f"__positional_{positional}"] = (token,)
                positional += 1
                continue
            key, has_equals, inline = token.partition("=")
            values: list[str] = []
            if has_equals:
                values.append(inline)
            else:
                while index < len(tokens) and not tokens[index].startswith("--"):
                    values.append(tokens[index])
                    index += 1
            rendered = (key, *values)
            prior = pairs.get(key)
            if prior is None:
                pairs[key] = rendered
            elif prior != rendered:
                raise ValueError(f"current Recipe config conflict for {key}: {prior!r} != {rendered!r}")

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
    return " ".join(chain.from_iterable(pairs.values())), envs


class _WarmKernelPrepMixin:
    """Prepares, overlays and unwinds the warm-start Recipe's kernel section for the PRELUDE replay."""

    def _collect_warm_kernel_plan(self, kb: Any) -> list[dict[str, Any]]:
        """Resolve the prior-champion kernel columns into a local apply plan."""
        readers = (
            ("gemm", kb.read_gemm, "optimizations"),
            ("fusion", kb.read_fusion, "items"),
            ("rewrite", kb.read_rewrite, "items"),
        )
        plan: list[dict[str, Any]] = []
        for column, reader, list_key in readers:
            rows = reader().get(list_key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # Preserve the exact Recipe row so CLOSE can carry the validated kernel content and its refs forward
                # on the same inference page.
                recipe_row = {k: v for k, v in row.items() if k not in _KERNEL_ROW_PATH_KEYS}
                entry: dict[str, Any] = {
                    "column": column,
                    "meta": {k: v for k, v in recipe_row.items() if k not in _KERNEL_ROW_CONTENT_KEYS},
                    "recipe_row": recipe_row,
                }
                # The checkout this item was applied into.
                host_origin = row.get(HOST_ORIGIN_KEY)
                if isinstance(host_origin, Mapping):
                    entry["apply_root"] = str(host_origin.get("apply_root") or "").strip()
                patch_local = kb.prior_file(str(row.get("patch") or "").strip())
                if patch_local is not None:
                    entry["patch_path"] = str(patch_local)
                if column == "gemm":
                    tuned_local = kb.prior_file(str(row.get("tuned_file") or "").strip())
                    if tuned_local is not None:
                        entry["source_paths"] = [str(tuned_local)]
                if entry.get("patch_path") or entry.get("source_paths"):
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
        meta = entry["meta"]
        local = [str(p) for p in entry.get("source_paths") or []]
        envs: dict[str, str] = {}

        def _merge(container: Any, key: str) -> None:
            source = container.get(key) if isinstance(container, dict) else None
            if not isinstance(source, dict):
                return
            for raw_key, raw_value in source.items():
                name = str(raw_key).strip()
                if not name:
                    continue
                value = str(raw_value)
                if name in envs and envs[name] != value:
                    raise ValueError(f"current Recipe kernel env conflict for {name}: {envs[name]!r} != {value!r}")
                envs[name] = value

        for key in ("extra_envs", "env_flags", "recommended_env"):
            _merge(meta, key)
        _merge(meta.get("e2e"), "extra_envs")
        by_name = {Path(path).name: path for path in local}
        results = meta.get("e2e_results")
        kept = (results.get("kept") or []) if isinstance(results, dict) else []
        for tuner in kept:
            if not isinstance(tuner, dict):
                continue
            _merge(tuner, "envs")
            env_var = str(tuner.get("env_var") or "").strip()
            if not env_var:
                continue
            # Each accepted tuner owns its own table, so match by the recorded value's filename.
            recorded = str(tuner.get("env_value") or envs.get(env_var) or "").strip()
            local_copy = by_name.get(Path(recorded).name)
            if local_copy is None and len(local) == 1:
                local_copy = local[0]
            if local_copy:
                envs[env_var] = local_copy
        # Re-point any other var that still names a file we downloaded.
        for name, value in list(envs.items()):
            if "/" not in value:
                continue
            replacement = by_name.get(Path(value).name)
            if replacement:
                envs[name] = replacement
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
            materialized = materialize_unified_patch_snapshot(
                patch_path=patch_path,
                repo_root=repo_root,
                snapshot_dir=snapshot_dir,
            )
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

    def _kernel_launch_overlay(self, plan: list[dict[str, Any]]) -> _KernelLaunchOverlay:
        """Resolve the launch overlay of the plan entries that contribute one, without mutating target files.

        A GEMM entry contributes only with an env bundle; any other entry only when its patch targets resolve.
        """
        entry_envs: dict[int, dict[str, str]] = {}
        entry_targets: dict[int, list[str]] = {}
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
                entry_targets[index] = targets
            entry_envs[index] = envs
            configs.append(
                (
                    f"kernel[{index}]",
                    {
                        "extra_server_args": str(entry["meta"].get("extra_server_args") or "").strip(),
                        "extra_envs": envs,
                    },
                )
            )
        args, envs = _merge_named_current_recipe_configs(configs)
        return _KernelLaunchOverlay(args, envs, entry_envs, entry_targets)

    async def _prepare_warm_kernel_kb(self, kb: Any = None) -> dict[str, Any]:
        """Prepare the Recipe's kernel section for the combined replay benchmark."""
        state = self.shared_state
        if state.warm_kernel_kb_attempted:
            return {"status": "skipped", "reason": "already_attempted"}
        gated = self._warm_kernel_gate_reason()
        if gated:
            return self._set_warm_kernel_outcome({"status": "skipped", "reason": gated})
        state.warm_kernel_kb_attempted = True
        # Persist the one-shot flag now: a crash mid-replay must not re-run the whole (potentially hour-long) set on
        # resume.
        try:
            state.save(self.session_dir)
        except OSError as exc:
            log.warning("warm-kernel KB: one-shot state save failed", exc_info=True)
            return self._set_warm_kernel_outcome(
                {"status": "error", "reason": (f"kernel_attempt_state_persist_failed:{type(exc).__name__}")}
            )
        if kb is None:
            kb = self._open_warm_kernel_section()
        if kb is None or not kb.active:
            return self._set_warm_kernel_outcome({"status": "skipped", "reason": "no_kernel_section"})
        plan = self._collect_warm_kernel_plan(kb)
        state.warm_kernel_kb_plan = plan
        if not plan:
            state.warm_replay_pending = {}
            outcome = self._set_warm_kernel_outcome({"status": "empty"})
            try:
                state.save(self.session_dir)
            except OSError:
                log.debug("warm-kernel KB: empty-state save failed", exc_info=True)
            return outcome
        return self._apply_warm_kernel_plan(plan)

    def _apply_warm_kernel_plan(self, plan: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.shared_state
        applied: list[dict[str, Any]] = []
        kernel_snapshots: list[dict[str, Any]] = []
        overlay = self._kernel_launch_overlay(plan)
        state.warm_replay_pending = {
            **state.warm_replay_pending,
            "status": "preparing_kernel",
            "kernel_apply_results": [],
            "kernel_snapshots": [],
        }
        state.save(self.session_dir)
        for index, entry in enumerate(plan):
            # Every column carries its env bundle: a fusion needs its switches to activate the patched path, and a
            # GEMM is nothing but its bundle.
            envs = overlay.entry_envs.get(index, {})
            if envs:
                entry["extra_envs"] = envs
            # GEMM is parameter-shaped: its env bundle is the whole deliverable, so there is nothing to stage on disk.
            if entry.get("column") == "gemm":
                entry["decision"] = "PENDING" if envs else "DEFERRED"
                continue
            targets = overlay.entry_targets.get(index, [])
            if not targets:
                entry["decision"] = "DEFERRED"
                if entry.get("resolution_error"):
                    entry["apply_result"] = {"status": "skipped", "reason": str(entry["resolution_error"])[:300]}
                continue
            if not self._land_warm_kernel_entry(entry, targets, kernel_snapshots, applied):
                break
        return self._settle_warm_kernel_preparation(
            plan, applied, kernel_snapshots, overlay.extra_server_args, overlay.extra_envs
        )

    def _land_warm_kernel_entry(
        self, entry: dict[str, Any], targets: list[str], snapshots: list[dict[str, Any]], applied: list[dict[str, Any]]
    ) -> bool:
        state = self.shared_state
        entry["resolved_patch_targets"] = list(targets)
        try:
            for target_path in targets:
                snapshots.append(self._snapshot_warm_kernel_target(target_path, len(snapshots)))
            state.warm_replay_pending = {
                **state.warm_replay_pending,
                "status": "preparing_kernel",
                "kernel_snapshots": list(snapshots),
                "kernel_apply_results": list(applied),
            }
            state.save(self.session_dir)
        except (OSError, ValueError) as exc:
            log.warning("warm-kernel KB: pre-mutation snapshot persist failed for %s", targets, exc_info=True)
            entry["decision"] = "ERROR"
            entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
            return False
        try:
            apply_result = self._apply_warm_kernel_patch(entry, targets[0])
        except Exception as exc:  # isolation boundary: one warm-kernel entry's apply, marked ERROR and unwound
            log.warning("warm-kernel KB: apply failed for %s", targets[0], exc_info=True)
            entry["decision"] = "ERROR"
            entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
            return False
        entry["apply_result"] = {
            k: apply_result.get(k) for k in ("status", "reason", "error", "manifest_path") if k in apply_result
        }
        if apply_result.get("status") != "ok":
            entry["decision"] = "ERROR"
            return False
        applied.append(apply_result)
        state.warm_replay_pending = {
            **state.warm_replay_pending,
            "kernel_snapshots": list(snapshots),
            "kernel_apply_results": list(applied),
        }
        entry["decision"] = "PENDING"
        return True

    def _unwind_prepared_warm_kernel(
        self, applied: list[dict[str, Any]], snapshots: list[dict[str, Any]], pending: list[dict[str, Any]]
    ) -> dict[str, Any]:
        rollback = revert_warm_kernel_patches(applied, snapshots)
        for entry in pending:
            entry["decision"] = "ROLLED_BACK"
        if rollback.get("ok"):
            self.shared_state.warm_replay_pending = {}
        else:
            self.shared_state.warm_replay_pending = {
                **self.shared_state.warm_replay_pending,
                "status": "rollback_failed",
                "rollback_errors": list(rollback.get("errors") or []),
            }
        return rollback

    def _settle_warm_kernel_preparation(
        self,
        plan: list[dict[str, Any]],
        applied: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        kernel_args: str,
        merged_envs: dict[str, str],
    ) -> dict[str, Any]:
        state = self.shared_state
        decisions = [entry.get("decision") for entry in plan]
        errors = decisions.count("ERROR")
        pending = [entry for entry in plan if entry.get("decision") == "PENDING"]
        rollback: dict[str, Any] = {"ok": True, "errors": []}
        if errors:
            rollback = self._unwind_prepared_warm_kernel(applied, snapshots, pending)
            if rollback.get("ok"):
                applied = []
                pending = []
            status = "error" if rollback.get("ok") else "rollback_failed"
        else:
            status = "prepared" if pending else "loaded"
        columns = sorted({str(e.get("column")) for e in plan})
        outcome = {
            "status": status,
            "columns": columns,
            "total": len(plan),
            "deferred": decisions.count("DEFERRED"),
            "errors": errors,
            "pending": pending,
            "applied": applied,
            "snapshots": snapshots,
            "rollback": rollback,
            "dirty": not rollback.get("ok"),
            "extra_envs": merged_envs if pending else {},
            "extra_server_args": (kernel_args if pending else ""),
        }
        if status == "loaded" and not applied and not snapshots:
            state.warm_replay_pending = {}
        self._set_warm_kernel_outcome(outcome)
        log.info(
            "PRELUDE warm-kernel KB prepared: columns=%s total=%d pending=%d deferred=%d errors=%d",
            columns,
            len(plan),
            len(pending),
            outcome["deferred"],
            errors,
        )
        try:
            state.save(self.session_dir)
        except OSError as exc:
            log.warning("warm-kernel KB: prepared state save failed", exc_info=True)
            return self._warm_kernel_persist_failure(outcome, exc, applied, snapshots, pending)
        return outcome

    def _warm_kernel_persist_failure(
        self,
        outcome: dict[str, Any],
        exc: Exception,
        applied: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rollback = self._unwind_prepared_warm_kernel(applied, snapshots, pending)
        restored = bool(rollback.get("ok"))
        failed = {
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
        self._set_warm_kernel_outcome(failed)
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.warning("warm-kernel KB: rollback state save failed", exc_info=True)
        return failed

    def _refused_dirty_warm_kernel(self, kernel: Mapping[str, Any]) -> bool:
        state = self.shared_state
        preparation_pending = dict(state.warm_replay_pending)
        preparation_dirty = bool(
            kernel.get("dirty")
            or preparation_pending.get("kernel_snapshots")
            or preparation_pending.get("kernel_apply_results")
        )
        if str(kernel.get("status") or "") not in {"error", "rollback_failed"} or not preparation_dirty:
            return False
        rollback = dict(kernel.get("rollback") or {}) if isinstance(kernel.get("rollback"), dict) else {}
        if not rollback:
            rollback = revert_warm_kernel_patches(
                list(preparation_pending.get("kernel_apply_results") or []),
                list(preparation_pending.get("kernel_snapshots") or []),
            )
        restored = self._settle_warm_kernel_rollback(rollback)
        self._skip_warm_replay(
            code=SKIP_KERNEL_PREPARATION_FAILED,
            outcome={
                "status": ("kernel_preparation_failed" if restored else "rollback_failed"),
                "reason": str(kernel.get("reason") or "kernel preparation left mutable state"),
                "rollback": rollback,
            },
            details={"kernel_status": str(kernel.get("status") or "")},
        )
        return True

    def _settle_warm_kernel_rollback(self, rollback: Mapping[str, Any]) -> bool:
        state = self.shared_state
        if rollback.get("ok"):
            state.warm_replay_pending = {}
            return True
        state.warm_replay_pending = {
            **state.warm_replay_pending,
            "status": "rollback_failed",
            "rollback_errors": list(rollback.get("errors") or []),
        }
        state.set_stop_reason("warm_replay_rollback_failed")
        return False

    def _unwind_warm_kernel_half(self, overlay: Mapping[str, Any]) -> dict[str, Any]:
        applied, snapshots = overlay["applied"], overlay["snapshots"]
        rollback = (
            revert_warm_kernel_patches(applied, snapshots) if applied or snapshots else {"ok": True, "errors": []}
        )
        self._settle_warm_kernel_rollback(rollback)
        return rollback

    def _warm_kernel_overlay(
        self, kernel: Mapping[str, Any], sdk_replay: Mapping[str, Any], current_remote: bool
    ) -> dict[str, Any]:
        pending = list(kernel.get("pending") or []) if kernel.get("status") == "prepared" else []
        config = (sdk_replay.get("kernel_config") or {}) if current_remote else kernel
        return {
            "pending": pending,
            "applied": list(kernel.get("applied") or []) if pending else [],
            "snapshots": list(kernel.get("snapshots") or []) if pending else [],
            "extra_envs": dict(config.get("extra_envs") or {}) if pending else {},
            "extra_server_args": str(config.get("extra_server_args") or "") if pending else "",
        }
