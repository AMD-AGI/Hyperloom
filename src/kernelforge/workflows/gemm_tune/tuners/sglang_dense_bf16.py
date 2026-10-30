# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16/FP16 dense GEMM tuner for sglang via aiter's gemm_a16w16 tuner."""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .base import BaseTuner, TuneResult, micro_metrics
from ..dense_shapes import DenseGeometry, compute_dense_nk_shapes, compute_dense_m_values
from ..evidence import demand_for_tuner, demand_shapes, load_demand
from ..script_discovery import discover_tuner_script
from ..script_probe import filter_args, probe_script
from ..utils import resolve_aiter_root, run_subprocess, TUNER_ENV_VARS

log = logging.getLogger(__name__)

# Backwards-compatible aliases: shape derivation now lives in dense_shapes so the fp8 dense tuners can reuse the exact
# same logic (single source of truth).
_compute_nk_shapes = compute_dense_nk_shapes
_compute_m_values = compute_dense_m_values

# aiter moved the bf16 dense GEMM tuner out of gradlib/.
_LEGACY_SCRIPT_RELPATH = ("gradlib", "gradlib", "gemm_tuner.py")

# Printed once when at least one shape produced no row.
_NOT_FINISHED_MARKER = "[Tuning not Finished]"

# argparse's rejection message.
_UNRECOGNIZED_ARG_MARKER = "unrecognized arguments"

# Measured on MI355X (gfx950): one shape costs 58-155s under `--libtype hipblaslt --with-hipblaslt`.
_PER_SHAPE_BUDGET_S = 210
# ~407s/shape measured end to end; keep the same ~1.2x headroom over the worst observed shape that the fast ceiling
# has over its own.
_PER_SHAPE_BUDGET_THOROUGH_S = 600
# Mean observed cost, used to decide *how many* shapes fit in the time budget.
_PER_SHAPE_COST_S = 93
# The same figure for `--libtype all`.
_PER_SHAPE_COST_THOROUGH_S = 420
_MAX_SHAPES_ENV = "FORGE_DEMAND_MAX_SHAPES"
# `--shape_grouped` collapses every shape into ONE task, which makes aiter's `--timeout` a budget for the whole batch
# instead of per shape ("Waiting for 1 tasks to complete (timeout=Ns each)").
_TIMEOUT_RESERVE_S = 120


def _fit_m_values_to_budget(
    m_values: list[int],
    n_nk: int,
    budget: int,
) -> list[int]:
    """Trim the M list so ``n_nk x len(M)`` is something the budget can finish."""
    if n_nk <= 0 or budget <= 0 or n_nk * len(m_values) <= budget:
        return m_values
    per_nk = max(1, budget // n_nk)
    if per_nk >= len(m_values):
        return m_values
    if per_nk == 1:
        return [m_values[-1]]
    step = (len(m_values) - 1) / (per_nk - 1)
    picked = sorted({m_values[round(i * step)] for i in range(per_nk)})
    log.info(
        "Dense BF16: trimming M values %d -> %d so %d NK pairs fit a budget of %d shapes",
        len(m_values),
        len(picked),
        n_nk,
        budget,
    )
    return picked


def _generate_untuned_csv(
    nk_shapes: list[tuple[int, int]],
    m_values: list[int],
    output_path: Path,
    dtype: str = "torch.bfloat16",
) -> Path:
    """Generate untuned CSV in the format expected by aiter's gemm_a16w16 tuner."""
    csv_path = output_path / "untuned_dense_bf16.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n")
        for m in m_values:
            for n, k in nk_shapes:
                f.write(f"{m},{n},{k},False,{dtype},{dtype},False,False\n")
    log.info("Generated %d shapes to %s", len(m_values) * len(nk_shapes), csv_path)
    return csv_path


def _generate_untuned_csv_from_demand(
    shapes: list[dict[str, Any]],
    output_path: Path,
    dtype: str = "torch.bfloat16",
) -> Path:
    """Write the untuned CSV from keys the runtime actually looked up."""
    csv_path = output_path / "untuned_dense_bf16.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n")
        for s in shapes:
            f.write(
                "{M},{N},{K},{bias},{dt},{ot},{scaleAB},{bpre}\n".format(
                    M=s["M"],
                    N=s["N"],
                    K=s["K"],
                    bias=s.get("bias", "False"),
                    dt=s.get("dtype") or dtype,
                    ot=s.get("otype") or dtype,
                    scaleAB=s.get("scaleAB", "False"),
                    bpre=s.get("bpreshuffle", "False"),
                )
            )
    log.info("Generated %d demand-driven shapes to %s", len(shapes), csv_path)
    return csv_path


def _resolve_tuner_script(aiter_root: Path) -> Path | None:
    """Return the bf16 dense tuner script, preferring the direct tuner."""
    return discover_tuner_script("sglang_dense_bf16", aiter_root / "csrc")


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a tuner CSV by column name; never raise."""
    try:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            return [row for row in csv.DictReader(fh) if row]
    except (OSError, csv.Error) as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return []


def _shape_key(row: dict[str, str]) -> tuple[int, int, int] | None:
    try:
        return int(row["M"]), int(row["N"]), int(row["K"])
    except (KeyError, TypeError, ValueError):
        return None


# Shared with the fp8 dense path: the tuned CSV is deployed verbatim there too.
from ._row_accuracy import drop_inaccurate_rows, dropped_row_record  # noqa: E402


def _parse_profile_defaults(profile_csv: Path) -> dict[tuple[int, int, int], float]:
    """Map (M, N, K) to the torch candidate's time from the -o2 profile CSV."""
    defaults: dict[tuple[int, int, int], float] = {}
    for row in _read_rows(profile_csv):
        if (row.get("libtype") or "").strip() != "torch":
            continue
        key = _shape_key(row)
        if key is None:
            continue
        try:
            us = float(row.get("us", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(us) or us <= 0:
            continue
        # Keep the best torch time if the candidate was measured more than once.
        if key not in defaults or us < defaults[key]:
            defaults[key] = us
    return defaults


def _parse_tuner_results(
    tuned_csv: Path,
    defaults: dict[tuple[int, int, int], float] | None = None,
) -> list[dict[str, Any]]:
    """Parse the tuned CSV into per-shape results, one row per shape."""
    defaults = defaults or {}
    results: list[dict[str, Any]] = []
    for row in _read_rows(tuned_csv):
        key = _shape_key(row)
        if key is None:
            continue
        try:
            tuned_us = float(row.get("us", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(tuned_us) or tuned_us <= 0:
            continue
        try:
            tflops = float(row.get("tflops", "") or 0.0)
        except (TypeError, ValueError):
            tflops = 0.0
        m, n, k = key
        entry: dict[str, Any] = {
            "M": m,
            "N": n,
            "K": k,
            "libtype": (row.get("libtype") or "").strip(),
            "tuned_us": tuned_us,
            "tflops": tflops,
        }
        default_us = defaults.get(key)
        if default_us is None:
            entry.update(
                {
                    "default_us": None,
                    "speedup": None,
                    "improved": False,
                    "tuned_unverified": True,
                }
            )
        else:
            speedup = default_us / tuned_us
            entry.update(
                {
                    "default_us": default_us,
                    "speedup": round(speedup, 4),
                    "improved": speedup > 1.0,
                }
            )
        results.append(entry)
    return results


@dataclass(frozen=True)
class _BatchRun:
    """The grouped tuning batch: its artifact, how many shapes it was given, its budget and how it exited."""

    tuned_csv: Path
    n_expected: int
    batch_timeout: int
    rc: int
    not_finished: bool = False


class SglangDenseBf16Tuner(BaseTuner):
    """Tune dense BF16/FP16 GEMM kernels for sglang via aiter's gemm_a16w16 tuner."""

    name = "sglang_dense_bf16"
    env_var = TUNER_ENV_VARS["sglang_dense_bf16"]

    def validate(self) -> str | None:
        aiter_root = resolve_aiter_root()
        if aiter_root is None:
            return "aiter installation not found"
        if _resolve_tuner_script(aiter_root) is None:
            expected = aiter_root / "csrc" / "gemm_a16w16"
            if aiter_root.joinpath(*_LEGACY_SCRIPT_RELPATH).is_file():
                return (
                    f"bf16 GEMM tuner not found under {expected}; this aiter only ships "
                    "the legacy gradlib tuner, which rejects the untuned CSV schema and "
                    "does not support --libtype/--with-hipblaslt"
                )
            return f"bf16 GEMM tuner script not found under {expected}"
        # Shapes come from the config unless demand supplied them.
        if self._has_external_shapes():
            return None
        # Ask the derivation rather than any single config field.
        if not self._nk_shapes():
            return (
                "no dense GEMM shapes can be derived from the model config "
                "(needs hidden_size plus attention head counts, or an MLA rank "
                "layout), and no --demand was supplied to take shapes from instead"
            )
        return None

    def _has_external_shapes(self) -> bool:
        """Whether ``run`` will take its shapes from somewhere other than the config."""
        return bool(getattr(self.ctx, "demand_json", None))

    #: Inputs the FP8 dense path consumes but this tuner does not. Named so a
    #: caller who supplies one is told it went unused instead of being left to
    #: assume the shapes it carried were tuned.
    _UNREAD_SHAPE_INPUTS = ("untuned_csv", "shapes_json", "shapes_manifest")

    def _warn_about_unread_inputs(self) -> None:
        """Say which supplied shape sources this tuner will not read."""
        supplied = [name for name in self._UNREAD_SHAPE_INPUTS if getattr(self.ctx, name, None)]
        if not supplied:
            return
        log.warning(
            "Dense BF16: ignoring %s -- this tuner takes shapes from --demand or "
            "the model config only. Pass --demand to tune the recorded shapes.",
            ", ".join(supplied),
        )

    def _nk_shapes(self) -> list[tuple[int, int]]:
        """The ``(N, K)`` pairs derived from the model config."""
        profile = self.ctx.profile
        num_heads = profile.raw_config.get("num_attention_heads", 32)
        num_kv_heads = profile.raw_config.get("num_key_value_heads", num_heads)
        return _compute_nk_shapes(
            DenseGeometry(
                hidden_size=profile.hidden_size,
                intermediate_size=profile.intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=int(getattr(profile, "head_dim", 0) or 0),
                v_head_dim=int(getattr(profile, "v_head_dim", 0) or 0),
                q_lora_rank=int(getattr(profile, "q_lora_rank", 0) or 0),
                kv_lora_rank=int(getattr(profile, "kv_lora_rank", 0) or 0),
                qk_nope_head_dim=int(getattr(profile, "qk_nope_head_dim", 0) or 0),
                qk_rope_head_dim=int(getattr(profile, "qk_rope_head_dim", 0) or 0),
            ),
            self.ctx.tp,
        )

    def _shape_budget(self) -> int:
        """How many shapes the time budget actually pays for."""
        raw = os.environ.get(_MAX_SHAPES_ENV, "").strip()
        try:
            override = int(raw)
        except ValueError:
            override = 0
        if override > 0:
            return override
        cost = _PER_SHAPE_COST_THOROUGH_S if self.ctx.thorough else _PER_SHAPE_COST_S
        usable = max(self.ctx.timeout_s - _TIMEOUT_RESERVE_S, cost)
        return max(1, usable // cost)

    def _demand_shapes(self) -> list[dict[str, Any]]:
        """Shapes this tuner is asked for, from the serving log. Empty if none."""
        path = getattr(self.ctx, "demand_json", None)
        if not path:
            return []
        report = load_demand(path)
        if report is None:
            return []
        entry = demand_for_tuner(report, self.name)
        if entry is None:
            log.info("demand file has no entry for %s; falling back to derived shapes", self.name)
            return []
        budget = self._shape_budget()
        buckets = demand_shapes(entry)
        shapes = buckets[:budget]
        covered_raw_keys = sum(len(shape.get("observed_M") or []) for shape in shapes)
        log.info(
            "Demand-driven shapes for %s: %d of %d padded-M buckets selected, "
            "covering %d of %d distinct raw keys "
            "(budget %d from %ds timeout, %d misses logged)",
            self.name,
            len(shapes),
            len(buckets),
            covered_raw_keys,
            entry.get("distinct_keys", 0),
            budget,
            self.ctx.timeout_s,
            entry.get("miss_count", 0),
        )
        return shapes

    def _batch_timeout_s(self, n_shapes: int) -> int:
        """aiter --timeout for the whole grouped batch (see _TIMEOUT_RESERVE_S)."""
        per_shape = _PER_SHAPE_BUDGET_THOROUGH_S if self.ctx.thorough else _PER_SHAPE_BUDGET_S
        outer = max(int(self.ctx.timeout_s), 1)
        ceiling = max(outer - _TIMEOUT_RESERVE_S, 1)
        return int(min(max(n_shapes, 1) * per_shape, ceiling))

    def _failure(self, error: str, error_class: str, expected_shapes: int) -> TuneResult:
        """The failed result of a batch that was given ``expected_shapes`` shapes."""
        return TuneResult(
            tuner_name=self.name,
            status="failed",
            error=error,
            error_class=error_class,
            expected_shapes=expected_shapes,
        )

    def run(self) -> TuneResult:
        aiter_root = resolve_aiter_root()
        assert aiter_root is not None

        tuner_script = _resolve_tuner_script(aiter_root)
        assert tuner_script is not None

        # Always bf16, whatever the checkpoint says: sglang is run with --dtype bf16, so an fp16 checkpoint is still
        # served through the bf16 GEMM and tuning it as fp16 would key the table on a dtype the runtime never looks
        # up.
        dtype_str = "torch.bfloat16"

        self._warn_about_unread_inputs()
        # A demand list beats anything derived from config.json: it is the set of keys the runtime actually asked for.
        demand = self._demand_shapes()
        if demand:
            n_expected = len(demand)
            untuned_csv = _generate_untuned_csv_from_demand(demand, self.work_dir, dtype=dtype_str)
        else:
            nk_shapes = self._nk_shapes()
            # Fit the derived cross product to the budget, or a thorough run generates ~20x the shapes its window
            # can pay for.
            budget = self._shape_budget()
            m_values = _fit_m_values_to_budget(
                _compute_m_values(self.ctx.conc, thorough=self.ctx.thorough), len(nk_shapes), budget
            )
            n_expected = len(nk_shapes) * len(m_values)
            log.info(
                "Dense BF16 shapes: %d NK pairs × %d M values = %d total (thorough=%s, budget=%d)",
                len(nk_shapes),
                len(m_values),
                n_expected,
                self.ctx.thorough,
                budget,
            )
            untuned_csv = _generate_untuned_csv(nk_shapes, m_values, self.work_dir, dtype=dtype_str)

        tuned_csv = self.work_dir / "tuned_dense_bf16.csv"
        profile_csv = self.work_dir / "profile_dense_bf16.csv"
        # This tuner judges the run by how many rows landed on disk, precisely because the exit code cannot be
        # trusted.
        for stale in (tuned_csv, profile_csv):
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not clear stale %s: %s", stale, exc)
        batch_timeout = self._batch_timeout_s(n_expected)

        # `hipblaslt` is the only libtype gated on TWO conditions: matching --libtype is not enough, --with-hipblaslt
        # must be set as well (gemm_a16w16_tune.py: `if with_hipblaslt and ("all" in libtype or "hipblaslt" in
        # libtype)`). Outside a thorough run `torch` rides along for measurement, not for winning.
        thorough = self.ctx.thorough
        libtypes = "all" if thorough else "hipblaslt,torch"
        iters = self.ctx.iters if thorough else min(self.ctx.iters, 50)
        warmup = self.ctx.warmup if thorough else min(self.ctx.warmup, 10)
        tail = ["-i", str(untuned_csv), "-o", str(tuned_csv), "-o2", str(profile_csv), "--indtype", "bf16"]
        tail += ["--outdtype", "bf16", "--mp", str(self.ctx.mp), "--iters", str(iters), "--warmup", str(warmup)]
        tail += ["--timeout", str(batch_timeout), "--shape_grouped", "-v", "--libtype", libtypes, "--with-hipblaslt"]

        # Ask the script what it accepts before spending minutes on it.
        filtered = filter_args(tail, probe_script(tuner_script))
        if not filtered.ok:
            return self._failure(
                f"{tuner_script} does not accept {', '.join(filtered.rejected_required)}; without it the tuner has "
                "no candidates to search",
                "unsupported_argument",
                n_expected,
            )

        # Run from the script's directory: its sibling modules are imported by bare name.
        rc, stdout, stderr = run_subprocess(
            ["python3", str(tuner_script), *filtered.args],
            cwd=tuner_script.parent,
            timeout_s=self.ctx.timeout_s,
            log_file=self.work_dir / "tune.log",
        )
        batch = _BatchRun(tuned_csv, n_expected, batch_timeout, rc)
        return self._judge_batch(batch, profile_csv, tuner_script, stdout, stderr)

    def _judge_batch(
        self, batch: _BatchRun, profile_csv: Path, tuner_script: Path, stdout: str, stderr: str
    ) -> TuneResult:
        """The batch's result, judged by the rows it left on disk."""
        tuned_csv, n_expected = batch.tuned_csv, batch.n_expected
        if batch.rc == 124:
            # Killed by the outer timeout -- but the tuner writes rows as it goes, so some shapes may already be on
            # disk.
            accuracy = drop_inaccurate_rows(tuned_csv)
            self._dropped_inaccurate = accuracy.dropped
            salvaged = _parse_tuner_results(tuned_csv, _parse_profile_defaults(profile_csv))
            if salvaged and not accuracy.completed:
                return self._failure(
                    f"Tuning timed out after {self.ctx.timeout_s}s and aiter's accuracy figures "
                    f"were not applied to {tuned_csv}: {accuracy.reason}",
                    "accuracy_filter_incomplete",
                    n_expected,
                )
            if salvaged:
                log.warning(
                    "Dense BF16: timed out after %ds but %d of %d shapes were already written; keeping them",
                    self.ctx.timeout_s,
                    len(salvaged),
                    n_expected,
                )
                return self._build_result(salvaged, batch, forced_status="partial_output")
            return self._failure(
                f"Tuning timed out after {self.ctx.timeout_s}s with no rows written", "timeout", n_expected
            )

        combined = f"{stderr or ''}\n{stdout or ''}"
        if _UNRECOGNIZED_ARG_MARKER in combined:
            rejected = next(
                (ln.strip() for ln in combined.splitlines() if _UNRECOGNIZED_ARG_MARKER in ln),
                _UNRECOGNIZED_ARG_MARKER,
            )
            return self._failure(f"{tuner_script} rejected an argument: {rejected}", "unsupported_argument", n_expected)

        # The exit code cannot decide success here. gemm_a16w16_tune.py returns 1 even when every shape tuned, and the
        # gemm_tuner.py shim rewrites that same 1 into a 0 -- so failing on `rc != 0` throws away good results, while
        # trusting `rc == 0` accepts an empty run.
        defaults = _parse_profile_defaults(profile_csv)
        # Before anything reads the artifact: this file IS what gets deployed, so a row aiter measured as wrong must
        # not survive to serving.
        accuracy = drop_inaccurate_rows(tuned_csv)
        self._dropped_inaccurate = accuracy.dropped
        shape_results = _parse_tuner_results(tuned_csv, defaults)
        not_finished = _NOT_FINISHED_MARKER in (stdout or "") or _NOT_FINISHED_MARKER in (stderr or "")

        if not shape_results:
            detail = f"rc={batch.rc}"
            if not_finished:
                detail += f", aiter reported {_NOT_FINISHED_MARKER}"
            return TuneResult(
                tuner_name=self.name,
                status="empty_output",
                artifact_path=str(tuned_csv) if tuned_csv.is_file() else "",
                total_shapes=0,
                expected_shapes=n_expected,
                error=f"Tuner wrote 0 of {n_expected} shapes to {tuned_csv.name} ({detail}): {stderr[-300:]}",
                error_class="empty_output",
            )

        if not accuracy.completed:
            return self._failure(
                f"aiter's accuracy figures were not applied to {tuned_csv}: {accuracy.reason}",
                "accuracy_filter_incomplete",
                n_expected,
            )

        return self._build_result(shape_results, replace(batch, not_finished=not_finished))

    def _build_result(
        self,
        shape_results: list[dict[str, Any]],
        batch: _BatchRun,
        forced_status: str | None = None,
    ) -> TuneResult:
        """Assemble the TuneResult from the rows that were actually written."""
        n_expected = batch.n_expected
        tuned_csv = batch.tuned_csv
        total = len(shape_results)
        improved = [r for r in shape_results if r.get("improved")]
        unverified = [r for r in shape_results if r.get("tuned_unverified")]
        metrics = micro_metrics(shape_results)

        dropped = list(getattr(self, "_dropped_inaccurate", []) or [])

        # A row the accuracy check removed was tuned and then thrown away; the tuner did reach that shape.
        produced = total + len(dropped)

        if forced_status:
            status = forced_status
        elif produced < n_expected:
            log.warning(
                "Dense BF16: reached %d of %d shapes (rc=%d, not_finished=%s); "
                "the grouped batch budget of %ds was likely exhausted"
                "%s",
                produced,
                n_expected,
                batch.rc,
                batch.not_finished,
                batch.batch_timeout,
                (
                    f". Separately, {len(dropped)} of those rows were dropped as "
                    f"inaccurate, leaving {total} in the artifact"
                    if dropped
                    else ""
                ),
            )
            status = "partial_output"
        else:
            if dropped:
                # Every expected shape was tuned.
                log.warning(
                    "Dense BF16: all %d shapes tuned, but %d row(s) failed aiter's own "
                    "accuracy check and were removed; %d remain in the artifact. This is "
                    "accuracy filtering, not budget exhaustion",
                    n_expected,
                    len(dropped),
                    total,
                )
            status = "ok" if (improved or unverified) else "no_improvement"

        return TuneResult(
            tuner_name=self.name,
            status=status,
            artifact_path=str(tuned_csv),
            env_var=self.env_var,
            env_value=str(tuned_csv),
            # A shape without a torch baseline can never show a micro speedup, so the micro gate would drop it.
            candidate=bool(unverified),
            total_shapes=total,
            expected_shapes=n_expected,
            improved_shapes=metrics.improved,
            unverified_shapes=len(unverified),
            best_micro_speedup=metrics.best,
            avg_micro_speedup=metrics.avg,
            shape_results=shape_results,
            dropped_inaccurate=[dropped_row_record(r) for r in dropped],
        )
