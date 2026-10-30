# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_GemmTuningMixin``: GEMM / FMoE tuned-config coverage, verdicts, integrate and E2E validation for the KERNEL phase."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .kernel_round_budget import integrate_within_phase
from hyperloom.common.coerce import to_float
from hyperloom.common.env import env_int
from hyperloom.common.io import WriteOptions, atomic_write_json
from ..actions.stop_attribution import stopped_by_the_run_class
from hyperloom.inference_optimizer.session.optimization_journal import KIND_GEMM_TUNING, OUTCOME_KEEP, JournalEntry
from ..state.shared_state import resolve_graded_comparison
from ..kernel import kernel_decisions
from ..kernel.lane_budget import INSUFFICIENT_BUDGET

log = logging.getLogger(__name__)

# Last-resort location of the aiter checkout inside the standard serving container.
_CONTAINER_AITER_CONFIG_DIR = Path("/sgl-workspace/aiter/aiter/configs")


def _record_gemm_e2e_envelope(
    result: dict[str, Any],
    stacked_envs: dict[str, str],
    *,
    kept: list[dict[str, Any]],
    reverted: list[dict[str, Any]],
    faults: list[dict[str, Any]],
) -> None:
    # Orchestration must never see the raw combined recommended_env and issue a bundled integrate, so the
    # E2E-validated envs replace it and the tuners' own record moves to the ``_raw`` keys.
    result["e2e_results"] = {"kept": kept, "reverted": reverted, "faults": faults}
    result["recommended_env_raw"] = dict(result.get("recommended_env") or {})
    result["extra_envs_raw"] = dict(result.get("extra_envs") or {})
    result["recommended_env"] = dict(stacked_envs)
    result["extra_envs"] = dict(stacked_envs)


def _first_text(mapping: Mapping[str, Any], *keys: str, default: str = "") -> str:
    # The first of ``keys`` whose value renders as non-empty text.
    for key in keys:
        text = str(mapping.get(key) or "")
        if text:
            return text
    return default


def _read_dispatch_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [dict(row) for row in reader]


def _dispatch_key(row: Mapping[str, str], key_cols: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in key_cols)


def _merged_dispatch_columns(column_sets: list[list[str]]) -> list[str]:
    all_columns = list(column_sets[0])
    for columns in column_sets[1:]:
        for column in columns:
            if column not in all_columns:
                insert_at = all_columns.index("tflops") if "tflops" in all_columns else len(all_columns)
                all_columns.insert(insert_at, column)
    return all_columns


def _normalized_dispatch_rows(rows: list[dict[str, str]], all_columns: list[str]) -> list[dict[str, str]]:
    fill = {"xbf16": "0", "run_1stage": "0", "ksplit": "0"}
    return [
        {column: fill.get(column, "0") if row.get(column) is None else row[column] for column in all_columns}
        for row in rows
    ]


def _dispatch_key_columns(all_columns: list[str], keys_from: Path | None, untuned_stem: str) -> list[str]:
    key_cols: list[str] = []
    untuned_path = keys_from / f"{untuned_stem}.csv" if keys_from is not None else None
    if untuned_path is not None and untuned_path.is_file():
        key_cols = [column for column in _read_dispatch_csv(untuned_path)[0] if column in all_columns]
    if not key_cols:
        key_cols = [column for column in ("M", "N", "K") if column in all_columns]
    for column in ("gfx", "cu_num", "_tag"):
        if column in all_columns and column not in key_cols:
            key_cols.append(column)
    return key_cols


def _fastest_dispatch_row(row: Mapping[str, str]) -> float:
    try:
        return float(row.get("us", ""))
    except (TypeError, ValueError):
        return math.inf


def _deduplicate_dispatch_rows(
    rows: list[dict[str, str]], key_cols: list[str], all_columns: list[str], label: str, env_var: str
) -> list[dict[str, str]] | None:
    keys = [_dispatch_key(row, key_cols) for row in rows]
    if len(set(keys)) == len(keys):
        return rows
    if "us" not in all_columns:
        log.warning(
            "gemm E2E: %s has duplicate dispatch keys for %s but no 'us' column to select the fastest row",
            label,
            env_var,
        )
        return None
    # Keep the fastest (smallest us) row per dispatch key; ties keep the first row seen (stable), NaN-like
    # values sort last.
    best: dict[tuple[str, ...], dict[str, str]] = {}
    for key, row in zip(keys, rows):
        if key not in best or _fastest_dispatch_row(row) < _fastest_dispatch_row(best[key]):
            best[key] = row
    log.info("gemm E2E: removed %d duplicate %s row(s) for %s", len(rows) - len(best), label, env_var)
    return list(best.values())


def _integrate_tput_and_gain(verdict: Mapping[str, Any]) -> tuple[float, float]:
    measurement = verdict.get("bench_result") or verdict
    tput = float(measurement.get("output_throughput", verdict.get("new_tput")) or 0.0)
    return tput, float(verdict.get("gain_pct") or 0.0)


def _candidate_envs(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if str(key).strip() and str(value).strip()}


# Which table each aiter config env var is resolved under at serving time.
_AITER_ENV_TO_TABLE: dict[str, str] = {
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm.csv",
    "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm.csv",
    "AITER_CONFIG_FMOE": "tuned_fmoe.csv",
}


def _integrate_run_name(tuner_name: str) -> str:
    """Name of the integrate run that serves a tuner's candidate."""
    return f"integrate-gemm_tune_{tuner_name}"


def _integrate_server_logs(session_dir: Path, tuner_name: str) -> list[Path]:
    """Server logs for a tuner's integrate run, retries included, oldest first."""
    from ..kernel.gemm_shape_coverage import integrate_server_logs

    return integrate_server_logs(session_dir, _integrate_run_name(tuner_name))


def _candidate_tuned_file(env: Any, env_var: str) -> str:
    """Return the tuned artifact a candidate's env points at."""
    if not isinstance(env, dict):
        return ""
    value = env.get(env_var)
    if value in (None, ""):
        value = next((v for v in env.values() if v not in (None, "")), "")
    return str(value or "")


def _paired_measurement_basis(verdict: Any) -> str:
    """How the promoted gain was measured, so the ledger cannot overstate it."""
    if verdict is None:
        return "e2e_rebench_unpaired"
    if getattr(verdict, "candidate_wins", False):
        return "e2e_paired_entry_reference_to_tuned"
    return f"e2e_paired_entry_reference_to_tuned_{getattr(verdict, 'reason', 'unknown')}"


class _GemmTuningMixin:
    def _runtime_uses_aiter_fused_moe(self) -> bool:
        """Return whether the served model dispatches MoE through aiter."""
        from ..kernel.serving_log_evidence import resolve_forge_server_log

        log_path = resolve_forge_server_log(self.shared_state, self.session_dir)
        if not log_path:
            return False
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return "[aiter] [fused_moe]" in text or "Mxfp4 MoE backend" in text

    def _gemm_tuned_config_coverage(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Report whether the validated aiter CSV was reachable by the server, or ``None`` when undetermined."""
        try:
            return self._gemm_tuned_config_coverage_impl(tuner_name, envs)
        except (OSError, ValueError):  # parses server logs and tuner CSVs whose format varies by aiter version
            log.warning("tuned-config coverage failed for %s; treating it as undetermined", tuner_name, exc_info=True)
            return None

    def _gemm_tuned_config_coverage_impl(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Replay aiter's lookup against the round's log (see the caller)."""
        if tuner_name == "fmoe_ck":
            return self._fmoe_tuned_config_coverage(envs)
        from ..kernel.gemm_shape_coverage import (
            parse_aiter_consulted_tables,
            parse_aiter_shape_lookups,
            parse_aiter_shape_lookups_for_tables,
            tuned_config_coverage,
            tuned_csv_shapes,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        logs = _integrate_server_logs(self.session_dir, tuner_name)
        if not logs:
            return None
        try:
            log_text = logs[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        def _unreadable(kind: str) -> None:
            """Log that the artifact could not be read, so the caller stays out of it."""
            log.warning(
                "gemm E2E: tuner=%s %s tuned CSV yielded no keys from %s; "
                "coverage is undetermined and will not block the KEEP",
                tuner_name,
                kind,
                csv_paths,
            )

        all_missed, all_hit = parse_aiter_shape_lookups(log_text)
        all_requested = all_missed | all_hit
        if not all_requested:
            return None
        wanted = {Path(path).name for path in csv_paths}
        missed, hit = parse_aiter_shape_lookups_for_tables(log_text, wanted)
        requested = missed | hit
        scoped_to_candidate = bool(requested)
        if not scoped_to_candidate:
            # Preserve the existing artifact-not-consulted diagnostic when the server performed lookups, but none
            # against this candidate's table.
            missed, hit = all_missed, set()
            requested = all_requested
        tuned: set[tuple[int, int, int]] = set()
        for path in csv_paths:
            tuned |= tuned_csv_shapes(path)
        if not tuned:
            _unreadable("dense")
            return None
        report = tuned_config_coverage(tuned, requested, known_covered=hit)
        report["server_log"] = str(logs[-1])
        report["runtime_lookup_miss"] = len(missed)
        report["runtime_lookup_hit"] = len(hit)
        report["artifact_applied"] = bool(report.get("covered"))
        consulted = parse_aiter_consulted_tables(log_text)
        report["consulted_tables"] = sorted(consulted)[:8]
        if consulted and not (wanted & {Path(name).name for name in consulted}):
            # The runtime resolved a different quantisation variant's table, so the tuner targeted a kernel this
            # server never dispatches to.
            report["artifact_applied"] = False
            report["not_applied_reason"] = "artifact_table_not_consulted"
        elif not report["artifact_applied"]:
            report["not_applied_reason"] = "no_shape_key_matched"
        return report

    def _fmoe_tuned_config_coverage(
        self,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Report whether a ``tuned_fmoe.csv`` covers logged fused-MoE dispatches."""
        from ..kernel.gemm_shape_coverage import (
            aiter_log_tuned_config_enabled,
            fmoe_tuned_config_coverage,
            log_has_fused_moe_activity,
            parse_aiter_fused_moe_dispatches,
            read_latest_integrate_server_log,
            resolve_fmoe_candidate_csv,
            tuned_fmoe_csv_rows,
        )

        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths:
            return None
        loaded = read_latest_integrate_server_log(self.session_dir, _integrate_run_name("fmoe_ck"))
        if loaded is None:
            return None
        log_path, log_text = loaded
        candidate_path = resolve_fmoe_candidate_csv(csv_paths[0])
        dispatches = parse_aiter_fused_moe_dispatches(log_text)
        hit_logging = aiter_log_tuned_config_enabled(envs)
        report: dict[str, Any] = {
            "server_log": str(log_path),
            "requested": len(dispatches),
        }
        if not dispatches:
            if log_has_fused_moe_activity(log_text):
                report["artifact_applied"] = False
                report["not_applied_reason"] = "fused_moe_parse_inconclusive"
                report["conclusive"] = False
            elif not hit_logging:
                report["artifact_applied"] = False
                report["not_applied_reason"] = "fused_moe_logging_disabled"
                report["conclusive"] = False
            else:
                report["artifact_applied"] = False
                report["not_applied_reason"] = "no_fused_moe_dispatch"
                report["conclusive"] = True
            report["runtime_lookup_miss"] = 0
            report["runtime_lookup_hit"] = 0
            return report
        if candidate_path is None:
            report["artifact_applied"] = False
            report["not_applied_reason"] = "candidate_csv_missing"
            report["runtime_lookup_miss"] = len(dispatches)
            report["runtime_lookup_hit"] = 0
            report["conclusive"] = False
            return report
        candidate_rows = tuned_fmoe_csv_rows(candidate_path)
        report["candidate_csv"] = str(candidate_path)
        report.update(fmoe_tuned_config_coverage(candidate_rows, dispatches))
        report["artifact_applied"] = bool(report.get("covered"))
        report["conclusive"] = True
        report["runtime_lookup_miss"] = report.get("requested", 0) - report.get("covered", 0)
        report["runtime_lookup_hit"] = report.get("covered", 0)
        if not report["artifact_applied"]:
            if report.get("runtime_default"):
                report["not_applied_reason"] = "runtime_default_config"
            elif report.get("kernel_name_mismatch"):
                report["not_applied_reason"] = "kernel_name_mismatch"
            else:
                report["not_applied_reason"] = "no_shape_key_matched"
        return report

    async def _confirm_gemm_gain_paired(
        self,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        *,
        config_path: str,
    ):
        """Re-measure the frozen GEMM entry recipe and tuned stack interleaved."""
        from ..kernel.request_handlers import integrate_handler
        from ..measurement.paired import assess_paired, interleaved_plan

        n_pairs = env_int("HYPERLOOM_GEMM_PAIRED_PAIRS", default=0)
        if n_pairs <= 0 or float(reference.get("tput") or 0.0) <= 0:
            return None

        pairs: list[tuple[float, float]] = []
        pending: float | None = None
        for idx, side in enumerate(interleaved_plan(n_pairs)):
            recipe = reference if side == "A" else candidate
            envs = dict(recipe.get("extra_envs") or {})
            overlay = str(recipe.get("final_overlay") or "")
            if overlay and overlay not in str(envs.get("PYTHONPATH") or "").split(":"):
                envs["PYTHONPATH"] = ":".join(filter(None, (overlay, envs.get("PYTHONPATH"))))
            res = await integrate_handler(
                {
                    "task_id": f"gemm_paired_{side}{idx}",
                    "kernel_id": f"gemm_paired_{side}{idx}",
                    "source": "forge_gemm_paired",
                    "base_tput": reference["tput"],
                    "config_path": config_path,
                    "paired_reference": reference,
                    "extra_server_args": str(recipe.get("extra_server_args") or ""),
                    "extra_envs": envs,
                    "remove_args": recipe.get("remove_args", []),
                    "unset_envs": recipe.get("unset_envs", []),
                    "args_mode": recipe.get("args_mode", "append"),
                    # Measure, do not decide: the verdict comes from the pairs, so a per-round KEEP/REVERT here
                    # would be noise promoted to a decision.
                    "keep_threshold_pct": 100.0,
                    "mode": "env_only",
                },
                session_dir=self.session_dir,
            )
            tput = float(res.get("new_tput") or 0.0)
            if tput <= 0:
                log.warning("forge gemm paired confirmation: %s%d produced no throughput", side, idx)
                break
            if side == "A":
                pending = tput
            elif pending is not None:
                pairs.append((pending, tput))
                pending = None

        verdict = assess_paired(pairs)
        log.info(
            "forge gemm paired confirmation: %d pair(s) -> %s (median delta %s%%)",
            len(pairs),
            verdict.reason,
            verdict.median_delta_pct,
        )
        return verdict

    def _gemm_apply_verdict(
        self,
        tuner_name: str,
        envs: dict[str, str],
    ) -> dict[str, Any] | None:
        """Did the tuned table reach the server's merge list and get read?"""
        csv_paths = [value for key, value in envs.items() if key.startswith("AITER_CONFIG")]
        if not csv_paths or tuner_name == "fmoe_ck":
            return None
        from ..kernel.gemm_shape_coverage import aiter_log_tuned_config_enabled
        from ..measurement.apply_verification import verify_applied

        logs = _integrate_server_logs(self.session_dir, tuner_name)
        if not logs:
            # Say so.
            log.warning(
                "forge gemm E2E: no server.log under %s (retries included); apply verification cannot run for %s",
                self.session_dir / "runs" / "integrate" / _integrate_run_name(tuner_name),
                tuner_name,
            )
            return None

        # The deployed file is named after the candidate, so the runtime's own table name has to travel with it or the
        # arrival check compares merged_tuned_dense_bf16.csv against bf16_tuned_gemm.csv and concludes the artifact
        # never landed.
        table_names = [name for key in envs if (name := _AITER_ENV_TO_TABLE.get(key))]
        # aiter prints a hit line only under this flag; every serving run now sets it by default, but an operator
        # value in the candidate env wins, and then a zero-hit result means nothing.
        hit_logging = aiter_log_tuned_config_enabled(envs)

        return verify_applied(
            logs[-1],
            csv_paths,
            hit_logging=hit_logging,
            runtime_table_names=table_names,
        ).to_dict()

    def _aiter_config_dirs(self) -> list[Path]:
        import importlib.util

        config_dirs: list[Path] = []
        explicit_root = os.environ.get("AITER_ROOT_DIR", "").strip()
        if explicit_root:
            config_dirs.extend((Path(explicit_root) / "aiter" / "configs", Path(explicit_root) / "configs"))
        try:
            spec = importlib.util.find_spec("aiter")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            config_dirs.append(Path(spec.origin).resolve().parent / "configs")
        config_dirs.append(_CONTAINER_AITER_CONFIG_DIR)
        config_dirs.extend(sorted(self.session_dir.glob("runs/specialist/*/worktree/aiter/configs")))
        unique: dict[str, Path] = {}
        for config_dir in config_dirs:
            if config_dir.is_dir():
                unique.setdefault(str(config_dir), config_dir)
        return list(unique.values())

    @staticmethod
    def _aiter_source_tables(
        config_dirs: list[Path], *, runtime_filename: str, tuned_stem: str, untuned_stem: str
    ) -> tuple[list[Path], Path | None]:
        # The complete runtime tables to merge the candidate into, plus the dir that defines their keys.
        cache_dir = os.environ.get("INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR", "/tmp/aiter_configs")
        runtime_path = Path(cache_dir) / runtime_filename
        if runtime_path.is_file():
            keys_from = next((d for d in config_dirs if (d / f"{untuned_stem}.csv").is_file()), None)
            return [runtime_path], keys_from
        for config_dir in config_dirs:
            base_path = config_dir / runtime_filename
            model_paths = sorted(
                path
                for path in (config_dir / "model_configs").glob(f"*{tuned_stem}*.csv")
                if path.is_file() and "untuned" not in path.name
            )
            paths = ([base_path] if base_path.is_file() else []) + model_paths
            if paths:
                return paths, config_dir
        return [], None

    def _merge_gemm_candidate_with_runtime(self, env_var: str, candidate_csv_path: str) -> str | None:
        """Merge a GEMM candidate CSV with the runtime config."""
        import csv
        import re

        candidate_path = Path(candidate_csv_path)
        runtime_filename = _AITER_ENV_TO_TABLE.get(env_var)
        if not runtime_filename or not candidate_path.is_file():
            return None
        tuned_stem = Path(runtime_filename).stem
        untuned_stem = (
            re.sub(r"(?:_)?tuned$", "_untuned", tuned_stem)
            if re.search(r"(?:_)?tuned$", tuned_stem)
            else tuned_stem.replace("tuned", "untuned")
        )
        config_dirs = self._aiter_config_dirs()
        try:
            candidate_columns, candidate_rows = _read_dispatch_csv(candidate_path)
            source_paths, keys_from = self._aiter_source_tables(
                config_dirs, runtime_filename=runtime_filename, tuned_stem=tuned_stem, untuned_stem=untuned_stem
            )
            if not source_paths:
                log.warning(
                    "gemm E2E: no complete aiter config found for %s; candidate-only validation would be unsafe",
                    env_var,
                )
                return None
            sources = [_read_dispatch_csv(path) for path in source_paths]
            all_columns = _merged_dispatch_columns([columns for columns, _ in sources] + [candidate_columns])
            runtime_rows: list[dict[str, str]] = []
            for _, rows in sources:
                runtime_rows.extend(_normalized_dispatch_rows(rows, all_columns))
            candidate_rows = _normalized_dispatch_rows(candidate_rows, all_columns)
            key_cols = _dispatch_key_columns(all_columns, keys_from, untuned_stem)
            if not key_cols:
                log.warning("gemm E2E: cannot derive dispatch keys for %s", env_var)
                return None
            kept = _deduplicate_dispatch_rows(runtime_rows, key_cols, all_columns, "runtime config", env_var)
            tuned = _deduplicate_dispatch_rows(candidate_rows, key_cols, all_columns, "candidate", env_var)
            if kept is None or tuned is None:
                return None
            # Drop rows from runtime that the candidate improves, then concat.
            candidate_keys = {_dispatch_key(row, key_cols) for row in tuned}
            merged_rows = [row for row in kept if _dispatch_key(row, key_cols) not in candidate_keys] + tuned
            merged_path = candidate_path.parent / f"merged_{candidate_path.name}"
            with merged_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(merged_rows)
            log.info(
                "gemm E2E: merged %d candidate rows into %d rows from %d aiter config file(s) -> %d total (%s)",
                len(tuned),
                len(kept),
                len(source_paths),
                len(merged_rows),
                merged_path,
            )
            return str(merged_path)
        except (OSError, ValueError, csv.Error) as exc:
            log.warning("gemm E2E: merge failed (%s); rejecting candidate", exc)
            return None

    def _sync_profile_state_after_gemm_roofline(self, result: dict[str, Any]) -> None:
        """Merge a handler-owned Roofline fallback into the live Coordinator state."""
        shape_capture = result.get("shape_capture") if isinstance(result, dict) else None
        if not isinstance(shape_capture, dict) or shape_capture.get("capture_mode") != "block_fp8_profile":
            return
        source_trace = str(shape_capture.get("source_profile_trace") or "").strip()
        if not source_trace:
            return
        from copy import deepcopy

        from ..state.shared_state import SharedState

        persisted = SharedState.load_or_init(self.session_dir)
        persisted_trace = str((persisted.last_trace_analyze or {}).get("steady_state_trace") or "").strip()
        if persisted_trace != source_trace:
            log.warning(
                "GEMM Roofline state sync skipped: persisted steady trace %r does not match result %r",
                persisted_trace,
                source_trace,
            )
            return
        for field_name in (
            "last_profile_trace",
            "last_profile_launch_evidence_path",
            "last_profile_status",
            "last_profile_args",
            "last_profile_workload",
            "last_profile_workload_action",
            "last_trace_analyze",
            "roofline_snapshots",
            "baseline_eager_fallback",
        ):
            setattr(
                self.shared_state,
                field_name,
                deepcopy(getattr(persisted, field_name)),
            )
        # Lifecycle is append-only telemetry owned by both states; union it so neither the inline Roofline's rows nor
        # the live state's are dropped.
        self.shared_state.merge_lifecycle_events(persisted.lifecycle)

    async def _handle_gemm_tuning_result(self, result: dict[str, Any]) -> None:
        """Record and post-process a run_gemm_tuning result from any entrypoint."""
        if result.get("error_class") == INSUFFICIENT_BUDGET:
            # Never started, so no attempt to record; a recorded ``skipped`` would latch GEMM off for later entries.
            return
        self._sync_profile_state_after_gemm_roofline(result)
        kernel_decisions.record_gemm_tuning(self.shared_state, result)
        try:
            await self._validate_gemm_tuning_e2e(result)
        except Exception as exc:  # isolation boundary: one GEMM campaign's e2e validation, recorded as a REVERT fault
            # Validation spans server restarts, log parsing and CSV merges, and is reached from two entrypoints that
            # only guard the tuning call itself.
            log.exception("gemm E2E validation raised; recording it as a fault")
            e2e = result.setdefault("e2e_results", {})
            if isinstance(e2e, dict):
                faults = e2e.setdefault("faults", [])
                if isinstance(faults, list):
                    faults.append(
                        {
                            "tuner": "*",
                            "error_class": "e2e_validation_exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            # The bridge stamped KEEP + the raw combined env on the micro result; the normal exit rewrites both so
            # Orchestration never bundles an integrate against an unmeasured candidate.
            result["decision"] = "REVERT"
            result["requires_e2e_validation"] = False
            result["e2e_validated"] = False
            result["micro_decision"] = "e2e_validation_exception"
            for stale in ("recommended_env", "extra_envs"):
                if result.get(stale):
                    result[stale] = {}
            # ``record_gemm_tuning`` above stored a SHALLOW COPY, so the scalar rewrites just made
            # (decision/micro_decision/...) do not reach the recorded entry on their own -- only the normal exit
            # re-syncs it.
            self._replace_latest_gemm_tuning_attempt(result)
        self._record_gemm_tuning_timeline(result)
        self.shared_state.save(self.session_dir)

    def _journal_gemm_tuning_keep(
        self,
        entry: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """Mirror an adopted GEMM-tuning stack entry as an optimization_journal KEEP row."""
        try:
            journal = self._coord.recipe_journal.ensure_journal()
            variant_name = str(entry.get("variant_name") or "gemm_tuning")
            backend = str(entry.get("backend") or "").strip().lower()
            try:
                tput = float(entry["tput"]) if entry.get("tput") is not None else None
            except (TypeError, ValueError):
                tput = None
            try:
                gain_pct = float(entry["gain_pct"]) if entry.get("gain_pct") is not None else None
            except (TypeError, ValueError):
                gain_pct = None
            metrics: dict[str, Any] = {}
            if entry.get("tuned_file"):
                metrics["tuned_file"] = str(entry.get("tuned_file"))
            journal.append_entry(
                JournalEntry(
                    phase=self._coord.recipe_journal.journal_entry_phase(),
                    iter=int(self.shared_state.tick or 0),
                    kind=KIND_GEMM_TUNING,
                    change=variant_name,
                    outcome=OUTCOME_KEEP,
                    gain_pct=gain_pct,
                    throughput_after=tput,
                    task_id=str(task_id or ""),
                    variant_name=variant_name,
                    ts=str(entry.get("ts") or ""),
                    provenance=f"gemm_tuning:{backend}" if backend else "gemm_tuning",
                    tick=int(self.shared_state.tick or 0),
                    metrics=metrics,
                )
            )
        except (OSError, TypeError, ValueError):
            log.exception("gemm_tuning journal append failed")

    def _writeback_gemm_result_json(self, entry: dict[str, Any]) -> None:
        """Overwrite ``<workspace>/result.json`` with the E2E-adjudicated envelope."""
        workspace = str(entry.get("workspace") or "").strip()
        if not workspace:
            return
        path = Path(workspace) / "result.json"
        try:
            if not path.parent.is_dir():
                return
            atomic_write_json(path, entry, options=WriteOptions())
        except (OSError, TypeError, ValueError):
            log.warning("gemm result.json writeback failed for %s", path, exc_info=True)

    def _replace_latest_gemm_tuning_attempt(self, result: dict[str, Any]) -> None:
        """Sync the latest GEMM history row, and publish the verdict to disk."""
        if not isinstance(result, dict):
            return
        entry = dict(result)
        attempts = list(self.shared_state.gemm_tuning_attempts or [])
        if attempts and isinstance(attempts[-1], dict):
            entry.setdefault("ts", attempts[-1].get("ts"))
            attempts[-1] = entry
        else:
            entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
            attempts.append(entry)
        self.shared_state.gemm_tuning_attempts = attempts
        self.shared_state.last_gemm_tuning = entry
        self._writeback_gemm_result_json(entry)

    @staticmethod
    def _gemm_canonical_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Read the per-tuner candidates the producer already decided on."""
        rows = result.get("candidates")
        if not isinstance(rows, list):
            return []
        candidates: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            envs = _candidate_envs(row.get("env"))
            if not envs:
                # Nothing to apply, so nothing an e2e run could validate.
                continue
            # The singular pair is what the CK-switch dedup and the promote path read; it is only unambiguous for a
            # single-variable candidate.
            env_var, env_value = next(iter(envs.items())) if len(envs) == 1 else ("", "")
            candidates.append(
                {
                    "tuner": str(row.get("tuner") or "unknown"),
                    "env_var": env_var,
                    "env_value": env_value,
                    "envs": envs,
                    "micro_speedup": to_float(row.get("best_micro_speedup")) or 1.0,
                }
            )
        return candidates

    def _close_gemm_e2e_without_candidates(self, result: dict[str, Any]) -> None:
        log.info("gemm tuning: no candidates to E2E validate")
        result["decision"] = "REVERT"
        result["requires_e2e_validation"] = False
        result["e2e_validated"] = False
        # Only when the tuners left no verdict of their own: ``micro_decision`` is a routing key downstream, not a
        # label, so an existing one stands.
        if not _first_text(result, "micro_decision").strip():
            result["micro_decision"] = "no_e2e_candidates"
        # ``recommended_env``/``extra_envs`` stay as the tuners left them: they are the raw record of what was
        # produced, and the eligibility checks downstream already read them as "must be empty".
        self._replace_latest_gemm_tuning_attempt(result)

    def _gemm_candidate_skip_reason(self, tuner_name: str, *, triton_moe_inert: bool) -> str:
        from hyperloom.inference_optimizer.model_config_utils import model_supports_aiter_ck_fused_moe

        if tuner_name == "vllm_moe_triton" and triton_moe_inert:
            log.warning(
                "gemm E2E: skipping %s -- the server dispatches MoE through "
                "aiter, which never reads VLLM_TUNED_CONFIG_FOLDER, so the tuned "
                "Triton config cannot take effect",
                tuner_name,
            )
            return "aiter_moe_runtime_triton_config_inert"
        # fmoe_ck is only meaningful with --moe-runner-backend aiter, and aiter's CK fused-MoE rejects a
        # non-128-aligned intermediate_size_per_partition.
        if tuner_name == "fmoe_ck" and not model_supports_aiter_ck_fused_moe(
            str(self.shared_state.model_path or ""), int(self.shared_state.tp or 0)
        ):
            log.info(
                "gemm E2E: skipping %s -- aiter CK fused-MoE cannot serve "
                "this model at tp=%s (intermediate size is not 128-aligned)",
                tuner_name,
                self.shared_state.tp or 0,
            )
            return "aiter_ck_moe_shape_unsupported"
        return ""

    def _merged_candidate_envs(self, cand: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any] | None]:
        # Each aiter table merged into the runtime one, plus the refusal row if one of them is unusable.
        env = dict(cand["envs"])
        for env_var, env_value in list(env.items()):
            if not env_var.startswith("AITER_CONFIG"):
                continue
            if Path(env_value).is_file():
                merged_path = self._merge_gemm_candidate_with_runtime(env_var, env_value)
                if merged_path:
                    env[env_var] = merged_path
                    continue
                reason = "complete_aiter_config_unavailable"
            else:
                reason = "candidate_artifact_missing"
            log.error("gemm E2E: refusing aiter candidate for %s (%s: %s)", cand["tuner"], env_var, reason)
            return env, {**cand, "reason": reason, "failed_env_var": env_var}
        return env, None

    def _gemm_candidate_server_args(self, tuner_name: str) -> str:
        if tuner_name == "fmoe_ck" and str(self.shared_state.framework or "").lower() == "sglang":
            return "--moe-runner-backend aiter"
        return ""

    def _gemm_integrate_payload(
        self, tuner_name: str, test_envs: dict[str, str], *, base_tput: float
    ) -> dict[str, Any]:
        from hyperloom.common.model_paths import resolve_session_model_path

        return {
            "task_id": f"gemm_tune_e2e_{tuner_name}",
            "kernel_id": f"gemm_tune_{tuner_name}",
            "source": "forge_gemm_tuning",
            "base_tput": base_tput,
            "model_path": resolve_session_model_path(state_model_path=str(self.shared_state.model_path or "")),
            "extra_server_args": self._gemm_candidate_server_args(tuner_name),
            "extra_envs": test_envs,
            "keep_threshold_pct": 1.0,
            "mode": "env_only",
        }

    async def _integrate_gemm_candidate(
        self,
        cand: dict[str, Any],
        test_envs: dict[str, str],
        jit_backup_dir: Path,
        *,
        base_tput: float,
    ) -> dict[str, Any] | None:
        # The integrate verdict, a row carrying ``fault``, or ``None`` when the run itself was stopped.
        from ..actions.executors._aiter_jit import drop_serving_so_for_envs, prepare_serving_so_for_csvs
        from ..state.kernel_decision_settings import _MAX_INTEGRATE_FAULT_ATTEMPTS

        tuner_name = cand["tuner"]
        payload = self._gemm_integrate_payload(tuner_name, test_envs, base_tput=base_tput)
        # The native handler reloads both the recipe and grading anchor from disk.
        self.shared_state.save(self.session_dir)

        async def _fault(attempt: int, error_class: str, **fields: Any) -> dict[str, Any]:
            await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=jit_backup_dir)
            return {
                **cand,
                "reason": f"integrate_fault:{error_class}",
                "fault": True,
                "error_class": error_class,
                "fault_attempts": attempt,
                **fields,
            }

        for attempt in range(1, _MAX_INTEGRATE_FAULT_ATTEMPTS + 1):
            last = attempt >= _MAX_INTEGRATE_FAULT_ATTEMPTS
            await asyncio.to_thread(prepare_serving_so_for_csvs, test_envs, backup_dir=jit_backup_dir)
            try:
                integrate_result = await integrate_within_phase(
                    payload, state=self.shared_state, session_dir=self.session_dir
                )
            except Exception as exc:  # isolation boundary: one integrate attempt for one GEMM candidate
                log.warning(
                    "gemm E2E: integrate raised for %s (fault attempt %d/%d): %s",
                    tuner_name,
                    attempt,
                    _MAX_INTEGRATE_FAULT_ATTEMPTS,
                    exc,
                    exc_info=True,
                )
                if last:
                    return await _fault(attempt, "handler_exception", error=repr(exc))
                continue
            stopped = stopped_by_the_run_class(integrate_result.get("error_class"))
            if stopped is not None:
                log.info("gemm E2E: %s left unmeasured -- %s", tuner_name, stopped.interrupted)
                return None
            if not self.shared_state._is_integrate_fault(integrate_result):
                return integrate_result
            error_class = str(integrate_result.get("error_class") or "integrate_fault").strip()
            log.warning(
                "gemm E2E: tuner=%s integrate fault %s (attempt %d/%d)",
                tuner_name,
                error_class,
                attempt,
                _MAX_INTEGRATE_FAULT_ATTEMPTS,
            )
            if last:
                return await _fault(
                    attempt,
                    error_class,
                    integrate_status=integrate_result.get("status"),
                    error=integrate_result.get("error"),
                )
        return None

    async def _gemm_apply_blockers(
        self, cand: dict[str, Any], env: dict[str, str], *, gain_pct: float
    ) -> tuple[list[str], dict[str, Any]]:
        # Two independent ways the artifact can fail to take effect, neither of which the throughput delta can
        # see: the keys are unreachable (coverage), and the table never reached the server (apply verdict).
        tuner_name = cand["tuner"]
        blockers: list[str] = []
        # Off the event loop for the same reason: this reads the integrate run's server.log in full and parses
        # every tuned CSV named in the candidate env.
        coverage = await asyncio.to_thread(self._gemm_tuned_config_coverage, tuner_name, env)
        if coverage is not None:
            cand = {**cand, "tuned_config_coverage": coverage}
            if coverage.get("artifact_applied"):
                log.info(
                    "gemm E2E: tuner=%s tuned-config coverage %.2f%% (%d/%d shapes)",
                    tuner_name,
                    coverage.get("coverage_pct") or 0.0,
                    coverage.get("covered") or 0,
                    coverage.get("requested") or 0,
                )
            elif coverage.get("conclusive", True):
                blockers.append(str(coverage.get("not_applied_reason") or "no_shape_key_matched"))
                log.error(
                    "gemm E2E: tuner=%s produced an artifact the runtime never "
                    "applied -- 0 of %d requested shape(s) resolve to a tuned row; "
                    "the %.2f%% e2e delta measures nothing about the tuning",
                    tuner_name,
                    coverage.get("requested") or 0,
                    gain_pct,
                )
            else:
                log.info(
                    "gemm E2E: tuner=%s tuned-config coverage inconclusive -- %s",
                    tuner_name,
                    coverage.get("not_applied_reason"),
                )
        applied = self._gemm_apply_verdict(tuner_name, env)
        if applied is None:
            return blockers, cand
        cand = {**cand, "apply_verdict": applied}
        if applied.get("blocks_keep"):
            blockers.append(str(applied.get("verdict") or "not_applied"))
            log.error(
                "forge gemm E2E: tuner=%s apply verdict=%s -- %s",
                tuner_name,
                applied.get("verdict"),
                applied.get("detail"),
            )
        elif not applied.get("conclusive"):
            # "Cannot tell" is not "did not apply": hit lines need AITER_LOG_TUNED_CONFIG=1, and treating
            # their absence as a failure would revert every arm that ran without it.
            log.info(
                "forge gemm E2E: tuner=%s apply verdict=%s (not conclusive) -- %s",
                tuner_name,
                applied.get("verdict"),
                applied.get("detail"),
            )
        return blockers, cand

    def _keep_gemm_candidate(
        self, cand: dict[str, Any], env: dict[str, str], verdict: Mapping[str, Any], result: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        from ..measurement.integrate_performance import integrate_measurement_fields

        backend = _first_text(result, "backend").strip().lower() or "forge"
        tuner_name = cand["tuner"]
        tput, gain_pct = _integrate_tput_and_gain(verdict)
        tuned_file = _candidate_tuned_file(env, cand["env_var"])
        lifted = self._coord.writeback.lift_to_current_best(
            "gemm_tuning",
            tput,
            {
                "name": f"{backend}_{tuner_name}",
                "candidate_extra_server_args": self._gemm_candidate_server_args(tuner_name),
                "extra_envs": dict(env),
                "source_phase": "KERNEL_AGENT",
                **integrate_measurement_fields(verdict.get("bench_result") or verdict),
            },
            entry_extra={
                "tuned_file": tuned_file,
                "gain_pct": gain_pct,
                "backend": backend,
                "source": "kernel_entry_auto",
            },
        )
        if not lifted:
            return None
        self._journal_gemm_tuning_keep(self.shared_state.optimization_stack[-1], task_id=f"gemm_tune_e2e_{tuner_name}")
        return {**cand, "envs": dict(env), "tput": tput, "gain_pct": gain_pct, "tuned_file": tuned_file}

    async def _settle_gemm_e2e_verdict(
        self,
        result: dict[str, Any],
        accepted_measurement: Mapping[str, Any],
        paired_reference: dict[str, Any],
        *,
        running_tput: float,
        paired_config_path: str,
    ) -> None:
        kept = result["e2e_results"]["kept"]
        reverted = result["e2e_results"]["reverted"]
        faults = result["e2e_results"]["faults"]
        # The watermark covers the whole run, so it waits for the last KEEP.
        result["graded_objective"] = None
        total_gain: float | None = 0.0
        if kept:
            total_gain = None
            # Paired checks isolate the GEMM increment, not cumulative session gain.
            paired = await self._confirm_gemm_gain_paired(
                paired_reference,
                deepcopy(self.shared_state.current_best),
                config_path=paired_config_path,
            )
            if self._coord.writeback.validate(
                running_tput,
                accepted_measurement,
                source="forge_gemm_tuning_e2e",
                measurement_basis=_paired_measurement_basis(paired),
            ):
                total_gain = self.shared_state.cumulative_gain_validated
                result["graded_objective"] = resolve_graded_comparison(
                    self.shared_state,
                    {**accepted_measurement, "output_throughput": running_tput},
                    against_baseline=True,
                ).objective
            # Name the artifact this run adopted, so the breakdown can tell it was.
            if kept[-1]["tuned_file"]:
                result["tuned_file"] = kept[-1]["tuned_file"]
            log.info(
                "gemm E2E: %d tuners KEEP (total gain=%s), %d REVERT",
                len(kept),
                f"{total_gain:+.2f}%" if total_gain is not None else "unavailable",
                len(reverted),
            )
        elif faults:
            log.info("gemm E2E: %d tuner(s) hit integrate fault(s), no E2E verdict", len(faults))
        else:
            log.info("gemm E2E: all %d tuners REVERT, no E2E gain", len(reverted))
        if total_gain is None or (faults and not kept and not reverted):
            result["e2e_gain_pct"] = None
        else:
            result["e2e_gain_pct"] = round(float(total_gain), 4)
        result["e2e_validated"] = True
        result["requires_e2e_validation"] = False
        result["decision"] = "KEEP" if kept else "REVERT"
        if kept:
            result["status"] = "complete"
        elif faults and not reverted:
            result["status"] = "failed"
            result["micro_decision"] = "integrate_fault"
        else:
            result["status"] = "complete"
            result["micro_decision"] = "candidate_no_e2e_gain"
        self._replace_latest_gemm_tuning_attempt(result)

    async def _validate_gemm_tuning_e2e(self, result: dict[str, Any]) -> None:
        """Sequentially E2E-validate each tuning candidate's env independently."""
        from ..actions.executors._aiter_jit import drop_serving_so_for_envs

        candidates = self._gemm_canonical_candidates(result)
        if not candidates:
            self._close_gemm_e2e_without_candidates(result)
            return
        running_tput = float(self.shared_state.current_best.get("tput") or self.shared_state.baseline_tput or 0.0)
        paired_reference = deepcopy(self.shared_state.current_best)
        paired_reference.setdefault("tput", running_tput)
        paired_config_path = str(self.shared_state.baseline_config_path or "")
        jit_backup_dir = self.session_dir / "runs" / "aiter_jit_backup"
        stacked_envs: dict[str, str] = {}
        kept: list[dict[str, Any]] = []
        reverted: list[dict[str, Any]] = []
        faults: list[dict[str, Any]] = []
        accepted_measurement: dict[str, Any] = {}
        # ``_runtime_uses_aiter_fused_moe`` resolves the serving log -- which now byte-scans the whole runs/ tree for
        # aiter evidence -- and then reads it whole, ~17MB on the fleet.
        triton_moe_inert = any(c.get("tuner") == "vllm_moe_triton" for c in candidates) and await asyncio.to_thread(
            self._runtime_uses_aiter_fused_moe
        )
        for cand in candidates:
            tuner_name = cand["tuner"]
            skip_reason = self._gemm_candidate_skip_reason(tuner_name, triton_moe_inert=triton_moe_inert)
            if skip_reason:
                reverted.append({**cand, "reason": skip_reason})
                continue
            # Merge candidate CSV with the runtime config so that shapes NOT in the candidate keep their existing
            # tuned entries.
            env, refusal = self._merged_candidate_envs(cand)
            if refusal is not None:
                reverted.append(refusal)
                continue
            # Merge with previously KEEP'd envs.
            test_envs = {**stacked_envs, **env}
            log.info("gemm E2E: validating tuner=%s env=%s (base_tput=%.1f)", tuner_name, cand["env_var"], running_tput)
            verdict = await self._integrate_gemm_candidate(cand, test_envs, jit_backup_dir, base_tput=running_tput)
            if verdict is None:
                break
            if verdict.get("fault"):
                faults.append(verdict)
                continue
            measurement = verdict.get("bench_result") or verdict
            new_tput, gain_pct = _integrate_tput_and_gain(verdict)
            decision = str(verdict.get("decision") or "").upper()
            log.info(
                "gemm E2E: tuner=%s decision=%s new_tput=%.1f gain=%.2f%%", tuner_name, decision, new_tput, gain_pct
            )
            blockers, cand = await self._gemm_apply_blockers(cand, env, gain_pct=gain_pct)
            if decision == "KEEP" and new_tput > 0 and not blockers:
                keep_row = self._keep_gemm_candidate(cand, env, verdict, result)
                if keep_row is not None:
                    stacked_envs.update(env)
                    running_tput = new_tput
                    accepted_measurement = measurement
                    kept.append(keep_row)
                    continue
            reason = f"decision={decision}, gain={gain_pct:.2f}%"
            if blockers:
                # Distinguish no gain from a tuned artifact the runtime never reached.
                reason = f"tuned_config_never_applied[{'+'.join(blockers)}] ({reason})"
            reverted.append({**cand, "reason": reason})
            await asyncio.to_thread(drop_serving_so_for_envs, test_envs, backup_dir=jit_backup_dir)

        _record_gemm_e2e_envelope(result, stacked_envs, kept=kept, reverted=reverted, faults=faults)
        await self._settle_gemm_e2e_verdict(
            result,
            accepted_measurement,
            paired_reference,
            running_tput=running_tput,
            paired_config_path=paired_config_path,
        )
