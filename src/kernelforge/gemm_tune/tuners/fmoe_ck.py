# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CK MoE GEMM tuner via aiter's gemm_moe_tune.py."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import BaseTuner, TuneResult
from ..utils import find_tuner_script, resolve_aiter_root, run_subprocess, TUNER_ENV_VARS
from .. import tune_robustness as _tr

log = logging.getLogger(__name__)

# Precision -> (q_dtype_a, q_dtype_w, q_type) mapping
_PRECISION_MAP: dict[str, tuple[str, str, str]] = {
    "bf16": ("torch.bfloat16", "torch.bfloat16", "QuantType.No"),
    "fp16": ("torch.float16", "torch.float16", "QuantType.No"),
    "fp8_per_token": ("torch.float8_e4m3fnuz", "torch.float8_e4m3fnuz", "QuantType.per_Token"),
    "fp8_blockscale": ("torch.float8_e4m3fnuz", "torch.float8_e4m3fnuz", "QuantType.per_1x128"),
    "fp4": ("torch.float8_e4m3fnuz", "torch.float8_e4m3fnuz", "QuantType.per_1x32"),
    "mxfp4": ("torch.float8_e4m3fnuz", "torch.float8_e4m3fnuz", "QuantType.per_1x32"),
    "a8w4": ("torch.float8_e4m3fnuz", "torch.float4_e2m1fn_x2", "QuantType.per_1x32"),
}

# Precision -> the aiter ``dtypes`` aliases the tuner expects, as an (activation, weight) pair.
_AITER_DTYPE_ALIAS: dict[str, tuple[str, str]] = {
    "fp8_per_token": ("fp8", "fp8"),
    "fp8_blockscale": ("fp8", "fp8"),
    "fp4": ("fp4x2", "fp4x2"),
    "mxfp4": ("fp4x2", "fp4x2"),
    "a8w4": ("fp8", "fp4x2"),
}

# CSV header for untuned fmoe config
_FMOE_CSV_HEADER = (
    "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
)
_FMOE_CSV_COLUMNS = tuple(_FMOE_CSV_HEADER.split(","))


def _validate_fmoe_csv(path: Path) -> str | None:
    """Return why ``path`` is unusable as an untuned fmoe CSV, or None if it is."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        return f"unreadable ({exc})"
    if not lines:
        return "file is empty"
    header = tuple(col.strip() for col in lines[0].split(","))
    missing = [col for col in _FMOE_CSV_COLUMNS if col not in header]
    if missing:
        return f"header is missing required column(s): {', '.join(missing)}"
    if len(lines) < 2:
        return "header present but no shape rows"
    width = len(header)
    for index, row in enumerate(lines[1:], start=2):
        if len(row.split(",")) != width:
            return f"row {index} has {len(row.split(','))} fields, expected {width}"
    return None


class FmoeCKTuner(BaseTuner):
    """Tune MoE fused GEMM kernels using aiter CK 2-stage codegen tuner."""

    name = "fmoe_ck"
    env_var = TUNER_ENV_VARS["fmoe_ck"]

    def _precision_key(self) -> str:
        """Map CLI precision + quant_type to internal key."""
        p = self.ctx.precision.lower()
        qt = self.ctx.quant_type.lower()
        if qt in ("a8w4", "w4a8", "mxfp4_w4a8"):
            return "a8w4"
        if p in ("bf16", "fp16"):
            return p
        if p == "fp8":
            if qt == "per_token":
                return "fp8_per_token"
            return "fp8_blockscale"
        if p in ("fp4", "mxfp4"):
            return "fp4"
        return "bf16"

    def _per_partition_inter_dim(self) -> int:
        """Return the MoE intermediate width of a single tensor-parallel rank."""
        full = self.ctx.profile.effective_moe_intermediate
        tp = max(1, int(self.ctx.tp or 1))
        return full // tp

    def validate(self) -> str | None:
        script = find_tuner_script("fmoe_ck")
        if script is None:
            return "aiter MoE tuner script not found (gemm_moe_tune.py)"
        profile = self.ctx.profile
        if not profile.is_moe:
            return "Model is not MoE; fmoe_ck tuner not applicable"
        if profile.num_experts < 1:
            return "num_experts < 1"
        if profile.effective_moe_intermediate < 1:
            return "moe_intermediate_size not set in model config"
        tp = max(1, int(self.ctx.tp or 1))
        if profile.effective_moe_intermediate % tp:
            # A non-divisible width means the serving shard size cannot be derived here; emitting a truncated one
            # would key the table on a shape the runtime never asks for.
            return (
                f"moe_intermediate_size {profile.effective_moe_intermediate} is not "
                f"divisible by tp {tp}; cannot derive the per-partition inter_dim"
            )
        if getattr(self.ctx, "moe_untuned_csv", None) is None and not self._demand_key():
            # Refuse rather than tune a guessed key.
            return (
                "no runtime-observed MoE miss available (neither "
                "moe_untuned_csv nor a serving log with a missed aiter "
                "fused_moe dispatch key); refusing to tune a key inferred "
                "from the model config"
            )
        return None

    def _demand_key(self) -> dict[str, Any] | None:
        """The most-missed MoE dispatch key, or None when every lookup hit."""
        if hasattr(self, "_cached_demand_key"):
            return self._cached_demand_key

        path = getattr(self.ctx, "demand_json", None)
        if not path:
            self._cached_demand_key = None
            return None
        # Keep evidence parsing out of module import: CLI registration must not acquire this optional analysis path
        # merely by importing the tuner.
        from ..evidence import load_demand, moe_ck_missed_keys

        report = load_demand(path)
        if report is None:
            self._cached_demand_key = None
            return None
        keys = moe_ck_missed_keys(report)
        if not keys:
            self._cached_demand_key = None
            return None
        if len(keys) > 1:
            # More than one MoE shape in one log means the server changed layout mid-run (or two logs were
            # concatenated).
            log.warning(
                "serving log carries %d distinct MoE dispatch keys; tuning the "
                "most-missed one (inter_dim=%s, q_dtype_w=%s)",
                len(keys),
                keys[0].get("inter_dim"),
                keys[0].get("q_dtype_w"),
            )
        self._cached_demand_key = keys[0]
        return self._cached_demand_key

    def _untuned_csv_from_demand(self, key: dict[str, Any]) -> Path:
        """Write the observed key out as an untuned fmoe CSV."""
        from ..evidence import moe_untuned_csv_text

        tokens = sorted({int(t) for t in (key.get("untuned_tokens") or key.get("tokens") or [])})
        # A token hint is a *set*, not a count.
        hint = getattr(self.ctx, "token_hint", None)
        if hint and tokens:
            allowed = {int(t) for t in hint}
            kept = [t for t in tokens if t in allowed]
            if kept:
                if len(kept) != len(tokens):
                    log.info(
                        "observed %d MoE token count(s); %d of them are served by "
                        "this backend per the log, dropping %s",
                        len(tokens),
                        len(kept),
                        [t for t in tokens if t not in allowed][:8],
                    )
                tokens = kept
            else:
                # Both sets came from the same serving log.
                raise ValueError(
                    "none of the %d observed MoE token count(s) appear in the "
                    "CK 2-stage token hint %s" % (len(tokens), sorted(allowed)[:8])
                )
        # Without a restrictive token hint, honour the caller's token-list length as a budget, the same way the
        # derived path does.
        budget = len(self.ctx.tokens) if self.ctx.tokens else 0
        if budget and len(tokens) > budget:
            observed = len(tokens)
            if budget == 1:
                kept = [tokens[-1]]
            else:
                step = (observed - 1) / (budget - 1)
                kept = sorted({tokens[round(i * step)] for i in range(budget)})
            log.info(
                "observed %d MoE token counts, tuning %d spread across the range %d..%d: %s",
                observed,
                len(kept),
                tokens[0],
                tokens[-1],
                kept,
            )
            tokens = kept
        csv_path = self.work_dir / "untuned_fmoe.csv"
        csv_path.write_text(moe_untuned_csv_text(key, tokens=tokens), encoding="utf-8")
        log.info(
            "Untuned CSV from runtime-observed MoE key at %s: %d token(s), "
            "model_dim=%s inter_dim=%s expert=%s topk=%s %s/%s",
            csv_path,
            len(tokens),
            key.get("model_dim"),
            key.get("inter_dim"),
            key.get("expert"),
            key.get("topk"),
            key.get("q_dtype_a"),
            key.get("q_dtype_w"),
        )
        return csv_path

    def _generate_untuned_csv(self) -> Path:
        """Generate untuned CSV from model profile and token coverage."""
        profile = self.ctx.profile
        prec_key = self._precision_key()
        q_dtype_a, q_dtype_w, q_type = _PRECISION_MAP.get(prec_key, _PRECISION_MAP["bf16"])
        # Every quantized entry in ``_PRECISION_MAP`` hardcodes the CDNA3 (gfx942) fnuz FP8 value.
        alias_pair = _AITER_DTYPE_ALIAS.get(prec_key)
        if alias_pair:
            from ._aiter_dense_common import _aiter_dtype_str

            q_dtype_a = _aiter_dtype_str(alias_pair[0])
            q_dtype_w = _aiter_dtype_str(alias_pair[1])

        inter_dim = self._per_partition_inter_dim()
        rows = []
        for token in self.ctx.tokens:
            rows.append(
                f"{token},{profile.hidden_size},{inter_dim},"
                f"{profile.num_experts},{profile.num_experts_per_tok},"
                f"{profile.activation_type_str},torch.bfloat16,"
                f"{q_dtype_a},{q_dtype_w},{q_type},"
                f"{1 if profile.use_g1u1 else 0},0"
            )

        csv_path = self.work_dir / "untuned_fmoe.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(_FMOE_CSV_HEADER + "\n")
            for row in rows:
                f.write(row + "\n")

        log.info("Generated untuned CSV with %d shapes at %s", len(rows), csv_path)
        return csv_path

    def _resolve_untuned_csv(self) -> tuple[Path, str]:
        """Return the untuned CSV to tune, and where its key came from."""
        external = getattr(self.ctx, "moe_untuned_csv", None)
        if external is None:
            key = self._demand_key()
            if key is not None:
                return self._untuned_csv_from_demand(key), "runtime_observed"
            return self._generate_untuned_csv(), "config_derived"

        path = Path(external)
        if not path.is_file():
            raise FileNotFoundError(f"moe_untuned_csv does not exist: {path}")
        problem = _validate_fmoe_csv(path)
        if problem:
            # Refusing beats silently derived shapes: the caller asked for a specific key, and quietly tuning a
            # different one is what makes a tuned table unreachable at run time.
            raise ValueError(f"unusable moe_untuned_csv {path}: {problem}")
        log.info("Using caller-supplied untuned CSV at %s", path)
        return path, "runtime_observed"

    def _parse_compare_output(self, stdout: str) -> list[dict[str, Any]]:
        """Parse the compare report from tuner stdout."""
        results = []
        # Match table rows: (token, ...) | Pre(us) | Post(us) | Improve% | Action
        pattern = re.compile(
            r"\((\d+),.*?\)\s*\|"
            r"\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)%\s*\|"
            r"\s*(.*)"
        )
        for line in stdout.splitlines():
            m = pattern.search(line)
            if m:
                token = int(m.group(1))
                pre_us = float(m.group(2))
                post_us = float(m.group(3))
                improve_pct = float(m.group(4))
                action = m.group(5).strip()
                speedup = pre_us / post_us if post_us > 0 else 1.0
                results.append(
                    {
                        "token": token,
                        "default_us": pre_us,
                        "tuned_us": post_us,
                        "improve_pct": improve_pct,
                        "speedup": round(speedup, 4),
                        "improved": "UPDATE" in action.upper(),
                    }
                )

        # Also parse the summary line for total counts
        summary_pat = re.compile(r"Total shapes:\s*(\d+)\s*\|\s*Would update:\s*(\d+)")
        for line in stdout.splitlines():
            m = summary_pat.search(line)
            if m:
                log.info("Compare summary: total=%s, would_update=%s", m.group(1), m.group(2))
                break

        return results

    def _find_candidate_csv(self, start_time: float, tuned_stem: str = "tuned_fmoe") -> Path | None:
        """Find the candidate CSV produced by --compare mode for THIS run only."""
        compare_dir = Path("/tmp/aiter_compare")
        if not compare_dir.is_dir():
            return None
        candidates = [
            p for p in compare_dir.glob("*.candidate.csv") if p.stat().st_mtime > start_time and tuned_stem in p.name
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def run(self) -> TuneResult:
        script = find_tuner_script("fmoe_ck")
        assert script is not None  # validated already

        import time

        run_start_time = time.time()

        untuned_csv, key_source = self._resolve_untuned_csv()
        tuned_csv = self.work_dir / "tuned_fmoe.csv"
        profile_csv = self.work_dir / "profile_fmoe.csv"

        # Flags shared by every shape (except -i/-o). aiter --timeout is injected below to activate mp_tuner's
        # per-candidate GPU-fault isolation -- MoE asm candidates fault on gfx950, and without --timeout the run
        # hangs.
        base_args = [
            "-o2",
            str(profile_csv),
            "--mp",
            str(self.ctx.mp),
            "--compare",
            "--iters",
            str(self.ctx.iters),
            "--warmup",
            str(self.ctx.warmup),
            "--min_improvement_pct",
            str(self.ctx.min_improvement_pct),
            "-v",
        ]

        aiter_root = resolve_aiter_root()
        cwd = aiter_root if aiter_root else None

        iso_candidate = None
        if _tr.is_isolation_enabled():
            blocklist = _tr.FaultBlocklist(
                getattr(self.ctx, "faulted_blocklist_path", None),
                {
                    "gpu_type": getattr(self.ctx, "gpu_type", ""),
                    "quant_type": getattr(self.ctx, "quant_type", ""),
                    "tp": getattr(self.ctx, "tp", 1),
                    "tuner": self.name,
                },
            )
            rc, stdout, stderr, iso_candidate = _tr.run_isolated(
                script=str(script),
                base_args=base_args,
                input_csv=untuned_csv,
                tuned_stem=tuned_csv.stem,
                work_dir=self.work_dir,
                aiter_root=aiter_root,
                outer_timeout_s=self.ctx.timeout_s,
                task_timeout_s=_tr.DEFAULT_TASK_TIMEOUT_S,
                gpu_ids=getattr(self.ctx, "gpu_ids", "") or "",
                blocklist=blocklist,
            )
        else:
            cmd = _tr.with_task_timeout(
                ["python3", str(script), "-i", str(untuned_csv), "-o", str(tuned_csv), *base_args]
            )
            rc, stdout, stderr = run_subprocess(
                cmd,
                cwd=cwd,
                timeout_s=self.ctx.timeout_s,
                log_file=self.work_dir / "tune.log",
            )

        if rc == 124:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"Tuning timed out after {self.ctx.timeout_s}s",
                error_class="timeout",
            )

        if rc != 0:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"Tuner exited with code {rc}: {stderr[-500:]}",
                error_class="subprocess_error",
            )

        # Parse compare results from stdout
        shape_results = self._parse_compare_output(stdout)
        if not shape_results:
            # Try parsing from stderr (some versions print there)
            shape_results = self._parse_compare_output(stderr)

        # Find candidate CSV.
        candidate_csv = iso_candidate if iso_candidate is not None else self._find_candidate_csv(run_start_time)
        artifact = str(candidate_csv) if candidate_csv else str(tuned_csv)

        # Copy candidate to output dir for persistence
        if candidate_csv and candidate_csv.is_file():
            dest = self.work_dir / "candidate_fmoe.csv"
            dest.write_bytes(candidate_csv.read_bytes())
            artifact = str(dest)

        # NOTE: The dense candidate-CSV fallback (_parse_candidate_csv in _aiter_dense_common) is intentionally NOT
        # mirrored here.

        # Compute metrics.
        improved = [r for r in shape_results if r.get("improved")]
        # Guard against a present-but-None speedup (mirrors the dense path): a candidate-CSV fallback row has
        # speedup=None, and `None > 1.0` raises TypeError, so filter to real numbers before comparing.
        speedups = [
            r["speedup"] for r in shape_results if isinstance(r.get("speedup"), (int, float)) and r["speedup"] > 1.0
        ]

        total = len(shape_results)
        n_improved = len(improved)
        best_speedup = max(speedups) if speedups else 1.0
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0

        if total == 0:
            status = "empty_output"
        elif n_improved == 0:
            status = "no_improvement"
        else:
            status = "ok"

        return TuneResult(
            tuner_name=self.name,
            status=status,
            artifact_path=artifact,
            env_var=self.env_var,
            env_value=artifact,
            total_shapes=total,
            improved_shapes=n_improved,
            best_micro_speedup=best_speedup,
            avg_micro_speedup=avg_speedup,
            shape_results=shape_results,
            key_source=key_source,
        )
