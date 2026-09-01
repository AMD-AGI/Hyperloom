# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BF16/FP16 dense GEMM tuner for sglang via aiter's gemm_a16w16 tuner."""

from __future__ import annotations

import csv
import logging
import math
import os
from pathlib import Path
from typing import Any

from .base import BaseTuner, TuneResult
from ..dense_shapes import compute_dense_nk_shapes, compute_dense_m_values
from ..evidence import demand_for_tuner, demand_shapes, load_demand
from ..script_discovery import discover_tuner_script
from ..script_probe import filter_args, probe_script
from ..utils import resolve_aiter_root, run_subprocess, TUNER_ENV_VARS

log = logging.getLogger(__name__)

# Backwards-compatible aliases: shape derivation now lives in dense_shapes so the
# fp8 dense tuners can reuse the exact same logic (single source of truth).
_compute_nk_shapes = compute_dense_nk_shapes
_compute_m_values = compute_dense_m_values

# aiter moved the bf16 dense GEMM tuner out of gradlib/. Only the new location
# accepts --libtype/--with-hipblaslt, and the gradlib script cannot read the CSV
# schema written below at all: it reads the `dtype` column value
# "torch.bfloat16" as an --indtype key and raises KeyError. So there is no
# usable fallback to gradlib -- an aiter without gemm_a16w16 fails validation.
# Kept only to make that diagnosis explicit in the error message.
_LEGACY_SCRIPT_RELPATH = ("gradlib", "gradlib", "gemm_tuner.py")

# Printed once when at least one shape produced no row. Useful in an error
# message, but not usable as a status: it does not say how many were lost.
_NOT_FINISHED_MARKER = "[Tuning not Finished]"

# argparse's rejection message. A rejected flag voids the whole invocation --
# the search space is not what was asked for -- so it must be reported as a
# failure with the offending argument, never as "ran and produced nothing".
# Reporting it as an empty or unimproved result is what let the previous
# breakage (14 calls rejecting --libtype) read as "this path has no gain".
_UNRECOGNIZED_ARG_MARKER = "unrecognized arguments"

# Measured on MI355X (gfx950): one shape costs 58-155s under
# `--libtype hipblaslt --with-hipblaslt`. `--libtype all` is far more expensive
# than that range suggested: a per-backend breakdown over four shapes on an
# 8-GPU MI355X box measured hipblaslt 127s, asm 19s, triton 17s, skinny 6s,
# opus 83s, torch 19s -- 169s for all six together -- against flydsl alone at
# 1458s, and even that finished only 3 of the 4. flydsl is not droppable: it won
# two of the four shapes, by 37% at M=16 N=1536 K=7168, so the decode range
# depends on it. Thorough mode is simply expensive, and the budget has to say so.
#
# Fast mode also runs `torch`, measured at 19s/shape above, so both fast
# figures below carry it: the ceiling covers 155+19=174 and the mean cost
# covers 74+19=93. Leaving them at the hipblaslt-only numbers would under-size
# the run in exactly the way _PER_SHAPE_COST_THOROUGH_S documents below --
# shapes that do not fit get silently written as nothing, which reads as a
# tuner that found no improvement.
_PER_SHAPE_BUDGET_S = 210
# ~407s/shape measured end to end; keep the same ~1.2x headroom over the worst
# observed shape that the fast ceiling has over its own.
_PER_SHAPE_BUDGET_THOROUGH_S = 600
# Mean observed cost, used to decide *how many* shapes fit in the time budget.
# Distinct from the per-shape timeout above, which is a ceiling with headroom
# over the observed 155s worst case.
_PER_SHAPE_COST_S = 93
# The same figure for `--libtype all`. Sizing a thorough run with the fast cost
# over-commits by 5.5x: an hour "buys" 47 shapes that would need over five, so
# the grouped batch runs out of budget part-way and the rest are written as
# nothing -- which reads as a tuner that found no improvement.
_PER_SHAPE_COST_THOROUGH_S = 420
_MAX_SHAPES_ENV = "FORGE_DEMAND_MAX_SHAPES"
# `--shape_grouped` collapses every shape into ONE task, which makes aiter's
# `--timeout` a budget for the whole batch instead of per shape ("Waiting for 1
# tasks to complete (timeout=Ns each)"). A flat value therefore loses rows as
# soon as the shape count grows: the first shapes spend the entire allowance and
# the rest are silently written as nothing. Scale it by shape count, and leave
# part of the outer timeout for process startup and for aiter to flush the CSV.
_TIMEOUT_RESERVE_S = 120


def _fit_m_values_to_budget(
    m_values: list[int],
    n_nk: int,
    budget: int,
) -> list[int]:
    """Trim the M list so ``n_nk x len(M)`` is something the budget can finish.

    The budget only ever constrained the demand-driven list; the derived one
    took the full cross product. A 1800s thorough run therefore generated 88
    shapes (4 NK pairs x 22 M) worth ~35000s of work -- a 21x over-commit that
    guarantees the grouped batch is cut off part-way, which is how a thorough
    run came back after 3606s with no rows at all.

    M is what gets cut, not NK. aiter resolves a lookup through a *padded* M
    (``found padded_M: 8192`` in the serving log), so a nearby M still serves
    the ones dropped between them; an NK pair that is dropped is a matmul with
    no tuned entry at any token count and no fallback to a neighbour. Values are
    sampled evenly and always keep both ends, so the decode and prefill extremes
    survive the trim rather than losing whichever end is at the tail.
    """
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
    """Write the untuned CSV from keys the runtime actually looked up.

    The log records the full key for bf16 lookups, so bias/scaleAB/bpreshuffle
    are taken from it rather than assumed. Falling back to the assumed value
    only matters for logs that did not print the wide form.
    """
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
    """Return the bf16 dense tuner script, preferring the direct tuner.

    Resolution (hinted path first, then a search) lives in script_discovery so
    another aiter relocation costs nothing here.
    """
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
from ._aiter_dense_common import _row_err_ratio, drop_inaccurate_rows  # noqa: E402


def _parse_profile_defaults(profile_csv: Path) -> dict[tuple[int, int, int], float]:
    """Map (M, N, K) to the torch candidate's time from the -o2 profile CSV.

    torch is the kernel aiter falls back to when a shape has no tuned entry, so
    its row is the only untuned baseline the tuner ever measures. It is present
    only when torch is in the candidate set, which is why both modes ask for it
    (`--libtype all` in thorough, `hipblaslt,torch` in fast). A torch-less
    `--libtype hipblaslt` leaves the profile holding hipblaslt candidates
    exclusively, and every shape then has nothing to compare against -- an
    unmeasurable run, not an unimproved one.

    Rows whose time is not finite are dropped -- aiter writes ``inf`` for a
    candidate that never got to run within the batch budget.
    """
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
    """Parse the tuned CSV into per-shape results, one row per shape.

    A shape is ``improved`` only when the profile CSV supplied a torch baseline
    and the selected kernel beat it. Without a baseline the shape is marked
    ``tuned_unverified``: a tuned row proves aiter picked a kernel, not that the
    kernel is faster than what serving would have used, and claiming otherwise
    would report a win nothing measured.
    """
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


class SglangDenseBf16Tuner(BaseTuner):
    """Tune dense BF16/FP16 GEMM kernels for sglang via aiter's gemm_a16w16 tuner.

    This tuner searches hipblaslt, asm, flydsl, triton, opus and skinny backends
    for the best solution per (M, N, K) shape. The output CSV is loaded by
    aiter's tuned_gemm.py at runtime via AITER_CONFIG_GEMM_BF16.
    """

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
        # Shapes come from the config unless demand supplied them. A demand file
        # lists the keys the runtime actually missed, so the config is not
        # consulted at all -- and requiring it anyway rejected a pure-MoE model
        # whose demand named 122 dense bf16 keys, which is exactly the case
        # demand exists to serve.
        if self._has_external_shapes():
            return None
        # Ask the derivation rather than any single config field. Naming a field
        # got it wrong both ways: a MoE-only config still yields its attention
        # projections (only the FFN pair needs intermediate_size), so refusing it
        # discarded shapes that were derivable and correctly keyed; and a config
        # yielding nothing was let through whenever an unread input happened to
        # be supplied. This also means sparse MLA needs no special case -- it
        # passes because its shapes derive.
        if not self._nk_shapes():
            return (
                "no dense GEMM shapes can be derived from the model config "
                "(needs hidden_size plus attention head counts, or an MLA rank "
                "layout), and no --demand was supplied to take shapes from instead"
            )
        return None

    def _has_external_shapes(self) -> bool:
        """Whether ``run`` will take its shapes from somewhere other than the config.

        Only demand qualifies. Unlike the FP8 dense path, ``run`` never reads
        ``untuned_csv`` / ``shapes_json`` / ``shapes_manifest``, so counting them
        here waived the config requirement for a run that went to the config
        regardless -- and the shapes the caller supplied were dropped without a
        word. Whatever this reports has to be what ``run`` actually consumes.
        """
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
        """The ``(N, K)`` pairs derived from the model config.

        Shared with :meth:`validate` so the check and the run cannot disagree
        about whether anything is derivable.
        """
        profile = self.ctx.profile
        num_heads = profile.raw_config.get("num_attention_heads", 32)
        num_kv_heads = profile.raw_config.get("num_key_value_heads", num_heads)
        return _compute_nk_shapes(
            hidden_size=profile.hidden_size,
            intermediate_size=profile.intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp=self.ctx.tp,
            head_dim=int(getattr(profile, "head_dim", 0) or 0),
            v_head_dim=int(getattr(profile, "v_head_dim", 0) or 0),
            q_lora_rank=int(getattr(profile, "q_lora_rank", 0) or 0),
            kv_lora_rank=int(getattr(profile, "kv_lora_rank", 0) or 0),
            qk_nope_head_dim=int(getattr(profile, "qk_nope_head_dim", 0) or 0),
            qk_rope_head_dim=int(getattr(profile, "qk_rope_head_dim", 0) or 0),
            o_lora_rank=int(getattr(profile, "o_lora_rank", 0) or 0),
            o_groups=int(getattr(profile, "o_groups", 0) or 0),
        )

    def _shape_budget(self) -> int:
        """How many shapes the time budget actually pays for.

        At ~93s per shape an hour buys about 37, while one real arm asks for
        492-849 distinct M values. Trimming is therefore mandatory, not a
        tuning knob -- the only question is what gets cut.

        Thorough mode costs ~5.5x more per shape, so it has to be sized with its
        own figure; sharing the fast one hands aiter a list it cannot finish.
        """
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
        """aiter --timeout for the whole grouped batch (see _TIMEOUT_RESERVE_S).

        Never exceeds the outer kill timeout. ``max(..., per_shape)`` used to
        raise the floor back above it on a small budget, which hands aiter a
        deadline it will be killed before reaching -- so it never gets to flush
        and the run looks like it produced nothing rather than like it ran out
        of time.
        """
        per_shape = _PER_SHAPE_BUDGET_THOROUGH_S if self.ctx.thorough else _PER_SHAPE_BUDGET_S
        outer = max(int(self.ctx.timeout_s), 1)
        ceiling = max(outer - _TIMEOUT_RESERVE_S, 1)
        return int(min(max(n_shapes, 1) * per_shape, ceiling))

    def run(self) -> TuneResult:
        aiter_root = resolve_aiter_root()
        assert aiter_root is not None

        tuner_script = _resolve_tuner_script(aiter_root)
        assert tuner_script is not None

        # Always bf16, whatever the checkpoint says: sglang is run with
        # --dtype bf16, so an fp16 checkpoint is still served through the bf16
        # GEMM and tuning it as fp16 would key the table on a dtype the runtime
        # never looks up.
        dtype_str = "torch.bfloat16"

        self._warn_about_unread_inputs()
        nk_shapes = self._nk_shapes()

        m_values = _compute_m_values(self.ctx.conc, thorough=self.ctx.thorough)

        # A demand list beats anything derived from config.json: it is the set of
        # keys the runtime actually asked for. Derivation stays as the fallback
        # for runs with no serving log to read.
        demand = self._demand_shapes()
        if demand:
            n_expected = len(demand)
            untuned_csv = _generate_untuned_csv_from_demand(
                demand,
                self.work_dir,
                dtype=dtype_str,
            )
        else:
            # The derived cross product used to ignore the budget entirely, so a
            # thorough run generated ~20x the shapes its window could pay for.
            m_values = _fit_m_values_to_budget(
                m_values,
                len(nk_shapes),
                self._shape_budget(),
            )
            n_expected = len(nk_shapes) * len(m_values)
            log.info(
                "Dense BF16 shapes: %d NK pairs × %d M values = %d total (thorough=%s, budget=%d)",
                len(nk_shapes),
                len(m_values),
                n_expected,
                self.ctx.thorough,
                self._shape_budget(),
            )
            untuned_csv = _generate_untuned_csv(
                nk_shapes,
                m_values,
                self.work_dir,
                dtype=dtype_str,
            )

        tuned_csv = self.work_dir / "tuned_dense_bf16.csv"
        profile_csv = self.work_dir / "profile_dense_bf16.csv"
        # This tuner judges the run by how many rows landed on disk, precisely
        # because the exit code cannot be trusted. That only holds if the rows
        # are this run's: a file left by an earlier attempt in the same work dir
        # would be read as output from an invocation that wrote nothing, turning
        # a total failure into "ok" with a full row count. Clear both first so
        # the artifact can only describe the run that just happened.
        for stale in (tuned_csv, profile_csv):
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not clear stale %s: %s", stale, exc)
        batch_timeout = self._batch_timeout_s(n_expected)

        # `hipblaslt` is the only libtype gated on TWO conditions: matching
        # --libtype is not enough, --with-hipblaslt must be set as well
        # (gemm_a16w16_tune.py: `if with_hipblaslt and ("all" in libtype or
        # "hipblaslt" in libtype)`). Without it the candidate set is empty and
        # the tuner exits in seconds having tuned nothing. It is not an optional
        # extra either: hipblaslt is the only backend that produces a result for
        # the large-M shapes at all -- every `--libtype all` variant leaves them
        # untuned. So both modes below enable it.
        if self.ctx.thorough:
            libtype_args = ["--libtype", "all", "--with-hipblaslt"]
            iters, warmup = self.ctx.iters, self.ctx.warmup
        else:
            # `torch` rides along for measurement, not for winning. The only
            # untuned baseline this tuner ever gets is the `torch` row of the
            # -o2 profile CSV (see _parse_profile_defaults), and that row exists
            # only when torch is in the candidate set. Under a torch-less
            # `--libtype hipblaslt` every shape came back with default_us=None,
            # so _parse_tuner_results marked it tuned_unverified and the run
            # reported improved_shapes=0 / best_micro_speedup=1.0 -- "no gain"
            # when the truth was "no measurement". Every sglang_dense_bf16
            # record in CI reads that way for this reason.
            #
            # It is the honest baseline, not a convenient one: aiter's untuned
            # default is hipblaslt/asm only under bpreshuffle and `skinny` for
            # is_skinny_default_shape(), and `torch` for everything else
            # (aiter/tuned_gemm.py:265-296). Checked against Kimi-K3's serving
            # log: 38600/38600 misses and 4756/4756 distinct shapes print
            # "will use default config! using torch", so torch is what the
            # runtime would actually have run for all of them.
            #
            # The comma is legal -- aiter's --libtype takes libtype_list, i.e.
            # string.split(","), and gemm_a16w16_tune.py:974 gates the torch
            # candidates on `"all" in libtype or "torch" in libtype`. A shape
            # torch happens to win dispatches fine at serving time
            # (tuned_gemm.py:389 `solfunc = solMap[libtype]`), and thorough
            # mode has been shipping torch winners via `--libtype all` already.
            libtype_args = ["--libtype", "hipblaslt,torch", "--with-hipblaslt"]
            iters, warmup = min(self.ctx.iters, 50), min(self.ctx.warmup, 10)

        tail = [
            "-i",
            str(untuned_csv),
            "-o",
            str(tuned_csv),
            "-o2",
            str(profile_csv),
            "--indtype",
            "bf16",
            "--outdtype",
            "bf16",
            "--mp",
            str(self.ctx.mp),
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
            "--timeout",
            str(batch_timeout),
            "--shape_grouped",
            "-v",
            *libtype_args,
        ]

        # Ask the script what it accepts before spending minutes on it. A missing
        # --libtype/--with-hipblaslt is not a degraded run, it is a run whose
        # candidate set is empty -- so refuse it here rather than let it finish
        # and report "no improvement" (that reading is what hid the original
        # breakage for a week).
        filtered = filter_args(tail, probe_script(tuner_script))
        if not filtered.ok:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=(
                    f"{tuner_script} does not accept "
                    f"{', '.join(filtered.rejected_required)}; without it the tuner has "
                    "no candidates to search"
                ),
                error_class="unsupported_argument",
                expected_shapes=n_expected,
            )

        cmd = ["python3", str(tuner_script), *filtered.args]

        # Run from the script's directory: its sibling modules are imported by
        # bare name.
        cwd = tuner_script.parent

        rc, stdout, stderr = run_subprocess(
            cmd,
            cwd=cwd,
            timeout_s=self.ctx.timeout_s,
            log_file=self.work_dir / "tune.log",
        )

        if rc == 124:
            # Killed by the outer timeout -- but the tuner writes rows as it goes,
            # so some shapes may already be on disk. Returning "failed" without
            # looking throws those away and reports nothing about how far it got,
            # which is the same mistake as judging by exit code: the artifact,
            # not the manner of exit, says what was produced.
            self._dropped_inaccurate = drop_inaccurate_rows(tuned_csv)
            salvaged = _parse_tuner_results(tuned_csv, _parse_profile_defaults(profile_csv))
            if salvaged:
                log.warning(
                    "Dense BF16: timed out after %ds but %d of %d shapes were already written; keeping them",
                    self.ctx.timeout_s,
                    len(salvaged),
                    n_expected,
                )
                return self._build_result(
                    salvaged,
                    n_expected,
                    tuned_csv,
                    batch_timeout,
                    rc=rc,
                    forced_status="partial_output",
                )
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"Tuning timed out after {self.ctx.timeout_s}s with no rows written",
                error_class="timeout",
                expected_shapes=n_expected,
            )

        combined = f"{stderr or ''}\n{stdout or ''}"
        if _UNRECOGNIZED_ARG_MARKER in combined:
            rejected = next(
                (ln.strip() for ln in combined.splitlines() if _UNRECOGNIZED_ARG_MARKER in ln),
                _UNRECOGNIZED_ARG_MARKER,
            )
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"{tuner_script} rejected an argument: {rejected}",
                error_class="unsupported_argument",
                expected_shapes=n_expected,
            )

        # The exit code cannot decide success here. gemm_a16w16_tune.py returns 1
        # even when every shape tuned, and the gemm_tuner.py shim rewrites that
        # same 1 into a 0 -- so failing on `rc != 0` throws away good results,
        # while trusting `rc == 0` accepts an empty run. Likewise the tuner's own
        # "Tuning Finished. tune N shapes" line reports the shapes it was given,
        # not the rows it wrote. Row count is the only reliable signal.
        defaults = _parse_profile_defaults(profile_csv)
        # Before anything reads the artifact: this file IS what gets deployed,
        # so a row aiter measured as wrong must not survive to serving.
        self._dropped_inaccurate = drop_inaccurate_rows(tuned_csv)
        shape_results = _parse_tuner_results(tuned_csv, defaults)
        total = len(shape_results)
        not_finished = _NOT_FINISHED_MARKER in (stdout or "") or _NOT_FINISHED_MARKER in (stderr or "")

        if total == 0:
            detail = f"rc={rc}"
            if not_finished:
                detail += f", aiter reported {_NOT_FINISHED_MARKER}"
            return TuneResult(
                tuner_name=self.name,
                status="empty_output",
                artifact_path=str(tuned_csv) if tuned_csv.is_file() else "",
                total_shapes=0,
                expected_shapes=n_expected,
                error=(f"Tuner wrote 0 of {n_expected} shapes to {tuned_csv.name} ({detail}): {stderr[-300:]}"),
                error_class="empty_output",
            )

        return self._build_result(
            shape_results,
            n_expected,
            tuned_csv,
            batch_timeout,
            rc=rc,
            not_finished=not_finished,
        )

    def _build_result(
        self,
        shape_results: list[dict[str, Any]],
        n_expected: int,
        tuned_csv: Path,
        batch_timeout: int,
        *,
        rc: int,
        not_finished: bool = False,
        forced_status: str | None = None,
    ) -> TuneResult:
        """Assemble the TuneResult from the rows that were actually written."""
        total = len(shape_results)
        improved = [r for r in shape_results if r.get("improved")]
        unverified = [r for r in shape_results if r.get("tuned_unverified")]
        speedups = [r["speedup"] for r in improved if isinstance(r.get("speedup"), (int, float))]

        dropped = list(getattr(self, "_dropped_inaccurate", []) or [])

        if forced_status:
            status = forced_status
        elif total < n_expected:
            log.warning(
                "Dense BF16: tuned %d of %d shapes (rc=%d, not_finished=%s, "
                "%d dropped as inaccurate); the grouped batch budget of %ds was "
                "likely exhausted",
                total,
                n_expected,
                rc,
                not_finished,
                len(dropped),
                batch_timeout,
            )
            status = "partial_output"
        elif improved or unverified:
            status = "ok"
        else:
            status = "no_improvement"

        return TuneResult(
            tuner_name=self.name,
            status=status,
            artifact_path=str(tuned_csv),
            env_var=self.env_var,
            env_value=str(tuned_csv),
            # A shape without a torch baseline can never show a micro speedup, so
            # the micro gate would drop it. Send it to E2E instead of discarding
            # it, exactly as the fp8 dense path does for split-K and new shapes.
            candidate=bool(unverified),
            total_shapes=total,
            expected_shapes=n_expected,
            improved_shapes=len(improved),
            unverified_shapes=len(unverified),
            best_micro_speedup=max(speedups) if speedups else 1.0,
            avg_micro_speedup=sum(speedups) / len(speedups) if speedups else 1.0,
            shape_results=shape_results,
            dropped_inaccurate=[
                {
                    "M": r.get("M"),
                    "N": r.get("N"),
                    "K": r.get("K"),
                    "libtype": r.get("libtype"),
                    "splitK": r.get("splitK"),
                    "us": r.get("us"),
                    "err_ratio": _row_err_ratio(r),
                }
                for r in dropped
            ],
        )
