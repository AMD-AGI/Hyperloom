# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PRELUDE phase handler: warm-recipe replay (KB best_config auto-apply) and
the initial baseline/roofline internal-analysis task enqueue."""

from __future__ import annotations
import logging as _logging
import os
from pathlib import Path
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
from .base import PhaseHandler

log = _logging.getLogger(__name__)


def _warm_kernel_keep_threshold_pct(state: Any = None) -> float:
    """Gain a replayed champion set must clear.

    Reads ``HYPERLOOM_WARM_KERNEL_KEEP_PCT`` as an override; falls back to the
    shared decaying curve so the bar tracks the session's macro-cycle.
    """
    from ..phases.machine_state import resolve_keep_threshold

    raw = str(os.environ.get("HYPERLOOM_WARM_KERNEL_KEEP_PCT", "") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "warm-kernel KB: HYPERLOOM_WARM_KERNEL_KEEP_PCT=%r is not a number; using curve value",
                raw,
            )
    return resolve_keep_threshold(state) if state is not None else 1.0


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

    def _filter_warm_patches_with_kg(
        self,
        patches: list,
        advisory_blocked: list,
        state: Any,
    ) -> list:
        """Filter replay patches using KG advisory blocks, expiry, conflicts.

        Removes patches that are (a) advisory-blocked at/above a fixed 0.75
        confidence threshold, (b) flagged ``expired`` by the
        warm-start validity check, or (c) in a ``CONFLICTS_WITH`` relation
        with another patch in the set. Best-effort: any failure returns the
        input patches unchanged so replay never breaks on a KG hiccup.

        Args:
            patches: The candidate replay patches from ``recommended_replay``.
            advisory_blocked: The ``advisory_blocked_patches`` list from the
                warm-start context.
            state: The live SharedState (for hardware/framework conditions).

        Returns:
            The filtered patch list.
        """
        if not patches:
            return patches
        threshold = 0.75

        def _norm(value: Any) -> str:
            return str(value or "").strip().replace(" ", "_").replace("/", "_").lower()

        try:
            advisory_drop = {
                _norm(ab.get("patch_file"))
                for ab in (advisory_blocked or [])
                if isinstance(ab, dict) and float(ab.get("confidence") or 0.0) >= threshold
            }
            kept = [
                p
                for p in patches
                if isinstance(p, dict) and not p.get("expired") and _norm(p.get("patch_file")) not in advisory_drop
            ]
            for p in patches:
                if isinstance(p, dict) and _norm(p.get("patch_file")) in advisory_drop:
                    log.info("warm-replay advisory block (conf>=%.2f): %s", threshold, p.get("patch_file"))

            if len(kept) >= 2:
                from hyperloom.orchestrator.knowledge.recipe_kb.kg_client import get_kg_client

                kg = get_kg_client()
                if kg is not None and kg.is_available():
                    knobs = [str(p.get("patch_file") or "") for p in kept]
                    conflicts = kg.find_conflicts_safe(
                        knobs=knobs,
                        hardware=str(getattr(state, "gpu_type", "") or getattr(state, "hardware", "") or ""),
                        framework=str(getattr(state, "framework", "") or ""),
                    )
                    drop = {_norm(c.get("knob")) for c in conflicts}
                    if drop:
                        for c in conflicts:
                            log.info(
                                "warm-replay conflict: %s conflicts_with %s",
                                c.get("knob"),
                                c.get("conflicts_with"),
                            )
                        kept = [p for p in kept if _norm(p.get("patch_file")) not in drop]
            return kept
        except Exception as exc:  # noqa: BLE001 - filtering is advisory only
            log.warning("warm-replay KG patch filtering degraded: %s", exc)
            return patches

    def _collect_warm_kernel_plan(self, kb: Any) -> list[dict[str, Any]]:
        """Resolve the prior-champion kernel columns into a local apply plan.

        Reads the ``gemm``/``fusion``/``rewrite`` sub-columns the warm-start
        download provided, resolves every recorded file ref to its downloaded
        copy via ``KernelAgentKB.prior_file``, and returns one plan entry per
        item carrying the local ``patch_path`` / ``source_paths`` plus the
        item's non-file metadata. Refs that do not resolve are dropped.
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
                    if k not in ("patch", "source_file", "source_files", "tuned_file", "files")
                }
                if column == "rewrite":
                    source_refs = [str(r) for r in (row.get("source_files") or []) if str(r or "").strip()]
                elif column == "fusion":
                    source_refs = [str(row.get("source_file"))] if row.get("source_file") else []
                else:  # gemm
                    source_refs = [str(row.get("tuned_file"))] if row.get("tuned_file") else []
                source_paths: list[str] = []
                for ref in source_refs:
                    resolved = kb.prior_file(ref)
                    if resolved is not None:
                        source_paths.append(str(resolved))
                entry: dict[str, Any] = {"column": column, "meta": meta}
                patch_ref = str(row.get("patch") or "").strip()
                if patch_ref:
                    patch_local = kb.prior_file(patch_ref)
                    if patch_local is not None:
                        entry["patch_path"] = str(patch_local)
                if source_paths:
                    entry["source_paths"] = source_paths
                # A portable target path is only honoured when it exists on this
                # host; cross-session records usually omit it, so most entries
                # are loaded/recorded and their apply is left to the kernel phase.
                target = str(row.get("target_path") or meta.get("target_path") or "").strip()
                if target and Path(target).is_file():
                    entry["target_file"] = target
                if entry.get("patch_path") or entry.get("source_paths"):
                    plan.append(entry)
        return plan

    @staticmethod
    def _parse_diff_target(patch_path: str | None) -> str:
        """Extract the patched file's repo-relative path from a unified diff.

        Reads the ``+++ b/<path>`` header (falling back to ``diff --git a/x
        b/y``) so a champion patch can be located in this session's source tree
        even when the KB record did not persist an absolute target path.
        """
        raw = str(patch_path or "").strip()
        if not raw:
            return ""
        try:
            text = Path(raw).read_text(errors="replace")
        except OSError:
            return ""
        git_target = ""
        for line in text.splitlines():
            if line.startswith("+++ "):
                candidate = line[4:].split("\t", 1)[0].strip()
                if candidate.startswith("b/"):
                    candidate = candidate[2:]
                if candidate and candidate != "/dev/null":
                    return candidate
            elif line.startswith("diff --git ") and not git_target:
                parts = line.split()
                if len(parts) >= 4:
                    candidate = parts[3]
                    if candidate.startswith("b/"):
                        candidate = candidate[2:]
                    git_target = candidate
        return git_target

    def _resolve_kernel_target_path(self, entry: dict[str, Any]) -> str:
        """Locate the champion patch's target file in this session's source tree.

        Aggressive resolution: an absolute target that exists is used as-is;
        otherwise the repo-relative path parsed from the diff header is joined
        against every trusted source root (:func:`resolve_patch_target_roots`)
        and the first existing file wins. Returns '' when nothing resolves.
        """
        target = str(entry.get("target_file") or "").strip()
        if target and Path(target).is_file():
            return target
        rel = self._parse_diff_target(entry.get("patch_path"))
        if not rel:
            return ""
        rel = rel.lstrip("/")
        try:
            from ..framework.paths import resolve_patch_target_roots

            roots = resolve_patch_target_roots()
        except Exception:  # noqa: BLE001 — resolution must never raise
            roots = ()
        for root in roots:
            candidate = Path(root) / rel
            if candidate.is_file():
                return str(candidate)
        # Suffix fallback: the diff path may carry a package prefix the root
        # already includes (e.g. ``sglang/foo`` under ``.../sglang/``). Only
        # drop leading components while a package path remains — matching on a
        # bare filename would happily point a whole-file replacement at an
        # unrelated same-named module.
        tail = Path(rel)
        for root in roots:
            base = Path(root.rstrip("/"))
            if not base.is_dir():
                continue
            for depth in range(1, max(1, len(tail.parts) - 1)):
                candidate = base / Path(*tail.parts[depth:])
                if candidate.is_file():
                    return str(candidate)
        return ""

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
                envs.update(
                    {str(k): str(v) for k, v in source.items() if str(k).strip()}
                )

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

    def _apply_warm_kernel_patch(
        self, entry: dict[str, Any], target: str
    ) -> dict[str, Any]:
        """Land one champion's file on disk without measuring it.

        The measurement is deliberately not here: a champion set is applied as a
        batch and graded by a single re-baseline, so this only stages the file
        (with a backup manifest that :meth:`_revert_warm_kernel_patches` uses to
        roll the whole set back when the set does not win).
        """
        from ..kernel.request_handlers import _maybe_apply_kernel_patch

        replacement = (entry.get("source_paths") or [entry.get("patch_path")])[0]
        kernel_id = str((entry.get("meta") or {}).get("kernel_name") or "warm_kernel")
        return _maybe_apply_kernel_patch(
            {
                "patch_path": replacement,
                "target_file": target,
                "source_file": target,
                "kernel_id": kernel_id,
                # Champion targets live in the installed framework tree, which is
                # not a known patch repo root.
                "allow_unknown_target": True,
            },
            session_dir=self.session_dir,
            kernel_id=kernel_id,
        )

    @staticmethod
    def _revert_warm_kernel_patches(applied: list[dict[str, Any]]) -> None:
        """Roll the whole champion set back after a losing measurement."""
        from ..kernel.request_handlers import _maybe_revert_kernel_patch

        for apply_result in reversed(applied):
            try:
                _maybe_revert_kernel_patch(apply_result)
            except Exception:  # noqa: BLE001 — one bad revert must not stop the rest
                log.warning("warm-kernel KB: revert failed", exc_info=True)

    async def _record_warm_kernel_keep(
        self,
        result: dict[str, Any],
        pending: list[dict[str, Any]],
        extra_envs: dict[str, str],
        extra_server_args: str,
        applied: list[dict[str, Any]],
    ) -> None:
        """Promote a winning champion set the same way a kernel integrate is.

        Routes through the shared integrate-KEEP bookkeeping so the replayed set
        lands on ``optimization_stack``, advances ``current_best`` (carrying the
        env bundle forward so later server launches keep the switches), and
        counts toward the validated cumulative gain.
        """
        kernels = [
            str((entry.get("meta") or {}).get("kernel_name") or entry.get("column") or "")
            for entry in pending
        ]
        keep = {
            "decision": "KEEP",
            "kernel_id": "warm_kernel_set",
            "integration_id": "warm_kernel_set",
            "new_tput": result.get("new_tput"),
            "gain_pct": result.get("gain_pct"),
            "workspace": result.get("workspace"),
            "source": "warm_kernel_kb",
            "stack_kernel_ids": [k for k in kernels if k],
            "apply_result": {"stack_apply_results": applied},
        }
        if extra_envs:
            keep["extra_envs"] = dict(extra_envs)
        if extra_server_args:
            keep["extra_server_args"] = extra_server_args
        try:
            await self._record_integrate_keep(keep)
        except Exception:  # noqa: BLE001 — bookkeeping must not fail PRELUDE
            log.warning("warm-kernel KB: recording the KEEP failed", exc_info=True)

    async def _validate_warm_kernel_set(
        self, extra_envs: dict[str, str], extra_server_args: str
    ) -> dict[str, Any]:
        """Grade the whole applied champion set with a single re-baseline.

        The patches are already on disk, so this measures the env bundle plus
        those files in one shot rather than re-benchmarking once per champion.
        """
        from ..kernel.request_handlers import integrate_handler

        ss = self.shared_state
        base_tput = float(getattr(ss, "baseline_tput", 0.0) or 0.0)
        current_best = getattr(ss, "current_best", {}) or {}
        if isinstance(current_best, dict):
            try:
                cb_tput = float(current_best.get("tput") or 0.0)
            except (TypeError, ValueError):
                cb_tput = 0.0
            if cb_tput > 0:
                base_tput = cb_tput
        payload: dict[str, Any] = {
            "kernel_id": "warm_kernel_set",
            "source": "warm_kernel_kb",
            "extra_envs": dict(extra_envs or {}),
            "base_tput": base_tput,
            "keep_threshold_pct": _warm_kernel_keep_threshold_pct(self.shared_state),
        }
        if extra_server_args:
            payload["extra_server_args"] = extra_server_args
        config_path = str(getattr(ss, "baseline_config_path", "") or "").strip()
        if config_path:
            payload["config_path"] = config_path
        return await integrate_handler(payload, session_dir=self.session_dir)

    def _open_warm_kernel_record(self) -> Any:
        """Download this run's independent kernel-agent KB record and open a reader.

        The kernel-agent KB is a standalone ``kernel:`` KB Store record (a
        sibling of the recipe's ``inference:`` record), so PRELUDE reads it
        directly instead of from the recipe warm-start. Returns ``None`` when
        remote KB is not configured or the record does not exist yet.
        """
        from ..knowledge.agent_kb import KernelRecordReader
        from ..knowledge.remote_recipe import (
            kernel_agent_canonical_id,
            read_remote_recipe,
        )

        recipe_cid = str(self._workload_canonical_id() or "").strip()
        if not recipe_cid:
            return None
        kernel_cid = kernel_agent_canonical_id(recipe_cid)
        record_dir = Path(self.session_dir) / "runtime" / "kernel_agent_kb"
        record_dir.mkdir(parents=True, exist_ok=True)
        document = read_remote_recipe(kernel_cid, record_dir)
        if document is None:
            return None
        return KernelRecordReader(record_dir)

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

        config = (
            getattr(getattr(self, "knowledge_plane", None), "config", None)
            or KnowledgeConfig.from_env()
        )
        if config.mode is not KnowledgeStoreMode.REMOTE:
            return "local_knowledge_mode"
        return ""

    async def _maybe_apply_warm_kernel_kb(self) -> dict[str, Any]:
        """Replay this workload's champion kernel set and validate it at PRELUDE.

        The kernel-side mirror of :meth:`_maybe_enqueue_warm_replay`: downloads
        the independent ``kernel:`` KB Store record, reads its
        ``gemm``/``fusion``/``rewrite`` columns, resolves each champion patch's
        target in the live source tree (parsing the diff header when no absolute
        path was stored), stages the whole set — patches on disk, env bundles
        merged — and grades it with a single re-baseline, rolling the set back
        when it does not win. Remote mode only, and skipped when warm replay is
        off or the KB is degraded. One-shot per session (resume-safe) and never
        raises.
        """
        state = self.shared_state
        if getattr(state, "warm_kernel_kb_attempted", False):
            return {"status": "skipped", "reason": "already_attempted"}
        gated = self._warm_kernel_gate_reason()
        if gated:
            return {"status": "skipped", "reason": gated}
        state.warm_kernel_kb_attempted = True
        # Persist the one-shot flag now: a crash mid-replay must not re-run the
        # whole (potentially hour-long) set on resume.
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort persistence
            log.debug("warm-kernel KB: state save failed", exc_info=True)
        try:
            kb = self._open_warm_kernel_record()
        except Exception as exc:  # noqa: BLE001 — advisory; never block PRELUDE
            log.warning("warm-kernel KB: opening the record failed", exc_info=True)
            return {
                "status": "skipped",
                "reason": f"record_unavailable:{type(exc).__name__}",
            }
        if kb is None or not kb.active:
            return {"status": "skipped", "reason": "no_kernel_record"}
        plan = self._collect_warm_kernel_plan(kb)
        if not plan:
            return {"status": "empty"}
        kept = 0
        reverted = 0
        deferred = 0
        errors = 0
        # Stage the whole champion set first: every patch lands on disk and every
        # env bundle is merged, so the set costs one re-baseline instead of one
        # per champion. The trade is all-or-nothing — the set is graded together.
        applied: list[dict[str, Any]] = []
        merged_envs: dict[str, str] = {}
        server_args = ""
        for entry in plan:
            if not (entry.get("source_paths") or entry.get("patch_path")):
                entry["decision"] = "DEFERRED"
                deferred += 1
                continue
            # Every column carries its env bundle: a fusion needs its switches to
            # activate the patched path, and a GEMM is nothing but its bundle.
            envs = self._warm_kernel_extra_envs(entry)
            if envs:
                entry["extra_envs"] = envs
                merged_envs.update(envs)
            candidate_args = str((entry.get("meta") or {}).get("extra_server_args") or "").strip()
            if candidate_args and not server_args:
                server_args = candidate_args
            # GEMM is parameter-shaped: its env bundle is the whole deliverable,
            # so there is nothing to stage on disk.
            if entry.get("column") == "gemm":
                if not envs:
                    entry["decision"] = "DEFERRED"
                    deferred += 1
                    continue
                entry["decision"] = "PENDING"
                continue
            target = self._resolve_kernel_target_path(entry)
            if not target:
                entry["decision"] = "DEFERRED"
                deferred += 1
                continue
            entry["target_file"] = target
            try:
                apply_result = self._apply_warm_kernel_patch(entry, target)
            except Exception as exc:  # noqa: BLE001 — one bad apply must not abort the rest
                log.warning("warm-kernel KB: apply failed for %s", target, exc_info=True)
                entry["decision"] = "ERROR"
                entry["apply_result"] = {"exception": f"{type(exc).__name__}: {exc}"[:300]}
                errors += 1
                continue
            entry["apply_result"] = {
                k: apply_result.get(k)
                for k in ("status", "reason", "error", "manifest_path")
                if k in apply_result
            }
            if apply_result.get("status") != "ok":
                entry["decision"] = "ERROR"
                errors += 1
                continue
            applied.append(apply_result)
            entry["decision"] = "PENDING"

        pending = [entry for entry in plan if entry.get("decision") == "PENDING"]
        if pending:
            # One measurement grades the whole set.
            try:
                result = await self._validate_warm_kernel_set(merged_envs, server_args)
            except Exception as exc:  # noqa: BLE001 — a failed grade must not abort PRELUDE
                log.warning("warm-kernel KB: set validation failed", exc_info=True)
                result = {
                    "status": "failed",
                    "decision": "REVERT",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
            result_dict = result if isinstance(result, dict) else {}
            # Record the verdict verbatim so a REVERT is diagnosable (was it a
            # relaunch failure vs. a measured below-threshold gain).
            verdict = {
                k: result_dict.get(k)
                for k in (
                    "status", "decision", "error_class", "error", "reason",
                    "base_tput", "new_tput", "gain_pct", "accuracy_pass", "workspace",
                )
                if k in result_dict
            }
            decision = str(result_dict.get("decision") or "").strip().upper() or "REVERT"
            if decision != "KEEP":
                self._revert_warm_kernel_patches(applied)
            for entry in pending:
                entry["decision"] = decision
                entry["integrate_result"] = verdict
                entry["gain_pct"] = result_dict.get("gain_pct")
            if decision == "KEEP":
                kept = len(pending)
                # Book the win. Without this the env switches live only inside
                # that one measurement (the next server launch drops them) and
                # CLOSE's scrape never sees the replayed champions, so neither
                # current_best nor cumulative_gain_validated counts them.
                await self._record_warm_kernel_keep(
                    result_dict, pending, merged_envs, server_args, applied
                )
            else:
                reverted = len(pending)
            log.info(
                "warm-kernel KB set: champions=%d decision=%s status=%s "
                "base_tput=%s new_tput=%s gain_pct=%s error_class=%s",
                len(pending), decision, result_dict.get("status"),
                result_dict.get("base_tput"), result_dict.get("new_tput"),
                result_dict.get("gain_pct"), result_dict.get("error_class"),
            )
        columns = sorted({str(e.get("column")) for e in plan})
        if kept:
            status = "kept"
        elif reverted:
            status = "reverted"
        elif errors:
            status = "error"
        else:
            status = "loaded"
        outcome = {
            "status": status,
            "columns": columns,
            "total": len(plan),
            "kept": kept,
            "reverted": reverted,
            "deferred": deferred,
            "errors": errors,
        }
        log.info(
            "PRELUDE warm-kernel KB: columns=%s total=%d kept=%d reverted=%d "
            "deferred=%d errors=%d",
            columns,
            len(plan),
            kept,
            reverted,
            deferred,
            errors,
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort persistence
            log.debug("warm-kernel KB: state save failed", exc_info=True)
        return outcome

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
        if not isinstance(warm, dict) or not warm:
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
        # Prefer the WarmStartContext's ready-to-replay champion (config may be
        # borrowed from a same-arch sibling), gating on the donor's transfer
        # confidence; fall back to the identity recipe's own best_config.
        wsc = getattr(state, "warm_start_context", None) or {}
        replay = wsc.get("recommended_replay") if isinstance(wsc, dict) else {}
        replay = replay if isinstance(replay, dict) else {}
        rep_args = str(replay.get("extra_server_args") or "").strip()
        rep_envs = replay.get("extra_envs") if isinstance(replay.get("extra_envs"), dict) else {}
        if rep_args or rep_envs:
            bc_args = rep_args
            bc_envs = dict(rep_envs)
            # Donor transfer confidence (self-donor == identity confidence).
            replay_conf = float(replay.get("config_confidence") or conf or 0.0)
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
        # Gate on the config-transfer confidence.
        if replay_conf < min_conf:
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
        # Extract code patches from warm_start_context (populated by T0).
        wsc_patches = (wsc.get("recommended_replay") or {}).get("patches") or [] if isinstance(wsc, dict) else []
        wsc_blocked = wsc.get("blocked_patches") or [] if isinstance(wsc, dict) else []
        wsc_advisory = wsc.get("advisory_blocked_patches") or [] if isinstance(wsc, dict) else []
        # KG-driven filtering (best-effort; unfiltered on any KG failure).
        wsc_patches = self._filter_warm_patches_with_kg(wsc_patches, wsc_advisory, state)
        if not bc_args and not bc_envs and not wsc_patches:
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
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "extra_server_args": bc_args,
            "extra_envs": dict(bc_envs),
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
            "blocked_patches": list(wsc_blocked),
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="replay_warm_recipe",
            params=params,
            idempotency_key="warm-replay-prelude",
        )
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
            **donor_metadata,
        }
        return task

    def _promote_warm_replay(
        self,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Interpret a ``replay_warm_recipe`` result: any measured uplift pushes warm config onto optimization_stack + current_best; failures set status and never propagate.

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
            outcome["status"] = "failed"
            outcome["reason"] = "non_dict_result"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        status = str(result.get("status") or "")
        if status != "succeeded":
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
            outcome["status"] = "quality_failed"
            outcome["reason"] = "image-quality gate failed vs baseline reference"
            outcome["quality_gate"] = qg
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info("warm-replay REJECTED by quality gate: %s", qg)
            return
        measured_gain = (single_round_tput / baseline_tput - 1.0) * 100.0
        min_reproduce = float(
            getattr(self, "_warm_replay_min_reproduce_pct", 0.8) or 0.8,
        )
        # Adopt KB best_config whenever replay beats baseline; expected_gain/min_reproduce are audit-only.
        reproduced = measured_gain > 0
        outcome["actual_gain_pct"] = round(measured_gain, 3)
        outcome["throughput_after"] = tput
        if expected_gain > 0:
            historical_bar = expected_gain * min_reproduce
            if measured_gain > 0 and measured_gain < historical_bar:
                outcome["below_historical_reproduce_pct"] = True
                outcome["historical_reproduce_bar_pct"] = round(
                    historical_bar,
                    3,
                )
        if reproduced:
            # Degrade gracefully when task is None (empty stack entry corrupts attribution).
            params = (task.params if task is not None else {}) or {}
            warm_args = str(params.get("extra_server_args") or "").strip()
            warm_envs = dict(params.get("extra_envs") or {})
            if not warm_args and not warm_envs:
                outcome["status"] = "reproduced_but_no_params"
                outcome["reason"] = "task.params missing extra_server_args/extra_envs"
                log.warning(
                    "warm-replay measured +%.2f%% but cannot push stack (task=%r has no warm args/envs)",
                    measured_gain,
                    task,
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            outcome["status"] = "reproduced"
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
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            self._lift_to_current_best(
                "replay_warm_recipe",
                float(single_round_tput),
                {
                    "name": "warm_replay",
                    "candidate_extra_server_args": warm_args,
                    "extra_envs": warm_envs,
                    "source_phase": "PRELUDE",
                    "task_id": str(getattr(task, "task_id", "") or ""),
                    "workspace": str(result.get("workspace") or ""),
                },
                entry_extra={
                    "gain_pct": round(measured_gain, 3),
                    "hot_tput": float(hot_tput),
                    "cold_tput": float(cold_round_tput) if cold_round_tput > 0 else None,
                    # source_tier records the warm-recipe tier for breakdown attribution.
                    "source_tier": outcome.get("warm_recipe_tier", ""),
                    "source_confidence": outcome.get("warm_recipe_conf", 0.0),
                },
            )
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
            outcome["status"] = "drift"
            outcome["reason"] = (
                f"measured +{measured_gain:.2f}% below {min_reproduce * 100:.0f}% of expected +{expected_gain:.2f}%"
            )
            log.info(
                "warm-replay DRIFT: measured=+%.2f%% < expected=+%.2f%% × %.0f%%",
                measured_gain,
                expected_gain,
                min_reproduce * 100,
            )
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
                f"internal-analysis-{reason}"
                f"{self._cycle_idem_suffix()}"
                f"{self._analysis_attempt_suffix(kind)}"
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
