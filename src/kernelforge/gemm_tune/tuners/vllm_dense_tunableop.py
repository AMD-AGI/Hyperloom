# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""vLLM Dense GEMM tuner via PyTorch TunableOp (hipBLASLt/rocBLAS kernel selection).

Requires pre-recorded GEMM shapes from PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 or
explicit --shapes-json / --tunableop-input.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .base import BaseTuner, TuneResult
from ..utils import TUNER_ENV_VARS, run_subprocess

log = logging.getLogger(__name__)


# PyTorch's own untuned-record format, read back off an MI355X box rather than
# inferred: enabling record_untuned and running three bf16 ``a @ b.t()`` matmuls
# produced, for (M, N, K),
#
#   GemmTunableOp_BFloat16_TN,tn_{N}_{M}_{K}_ld_{K}_{K}_{N}
#
# e.g. (16, 1536, 7168) -> tn_1536_16_7168_ld_7168_7168_1536. Getting this
# wrong would not fail loudly; it would tune shapes nobody asked for.
_TUNABLEOP_OP_BY_PRECISION = {
    "bf16": "GemmTunableOp_BFloat16_TN",
    "fp16": "GemmTunableOp_Half_TN",
    "float16": "GemmTunableOp_Half_TN",
    "bfloat16": "GemmTunableOp_BFloat16_TN",
}

# The activation dtype aiter logs per lookup, which the demand parser carries
# through as ``shape["dtype"]``. This is the authoritative record type for a
# shape: a checkpoint's ``precision`` describes how the WEIGHTS are stored, not
# what the dense GEMM runs in. On a Quark MX-FP4 checkpoint every one of the
# 21056 dense lookups in a real serving log is ``dtype='torch.bfloat16'``, so
# keying off ``ctx.precision`` alone found no record type and discarded the
# whole demand file.
_TUNABLEOP_OP_BY_DTYPE = {
    "bfloat16": "GemmTunableOp_BFloat16_TN",
    "bf16": "GemmTunableOp_BFloat16_TN",
    "float16": "GemmTunableOp_Half_TN",
    "half": "GemmTunableOp_Half_TN",
    "fp16": "GemmTunableOp_Half_TN",
}


def _tunableop_op_for_dtype(raw: Any) -> str | None:
    """Map a logged torch dtype to a TunableOp record type, or ``None``.

    Accepts the ``torch.`` prefix and surrounding quotes as they appear in the
    serving log. Unknown dtypes (fp8, fp4, int4, ...) return ``None`` rather
    than being coerced to a floating-point record type: TunableOp keys on the
    record type, so guessing would tune a shape the runtime never asks for.
    """
    text = str(raw or "").strip().strip("\"'").lower()
    if not text:
        return None
    if text.startswith("torch."):
        text = text[len("torch.") :]
    return _TUNABLEOP_OP_BY_DTYPE.get(text)


# Demand can list thousands of distinct keys; TunableOp times each one against
# every hipBLASLt solution, so the whole run would be spent on one tuner.
_DEMAND_SHAPE_LIMIT = 64


def tunableop_untuned_line(m: int, n: int, k: int, op: str) -> str:
    """One untuned record for a row-major ``A[M,K] @ B[N,K]^T``."""
    return f"{op},tn_{n}_{m}_{k}_ld_{k}_{k}_{n}"


def _is_tunableop_result_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("Validator"):
        return False
    return stripped.count(",") >= 3


def count_tunableop_result_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if _is_tunableop_result_line(line))


def _generate_candidate_sitecustomize() -> str:
    return (
        "import os\n"
        "from pathlib import Path\n\n"
        "_mode = os.environ.get('HL_TUNABLEOP_MODE', '').strip().lower()\n"
        "_file = os.environ.get('HL_TUNABLEOP_FILE', '').strip() or os.environ.get('PYTORCH_TUNABLEOP_FILENAME', '').strip()\n"
        "_verbose = os.environ.get('HL_TUNABLEOP_VERBOSE', '').strip().lower() in {'1', 'true', 'yes', 'on'}\n"
        "if _mode == 'candidate':\n"
        "    try:\n"
        "        if not _file:\n"
        "            raise RuntimeError('HL_TUNABLEOP_FILE or PYTORCH_TUNABLEOP_FILENAME is required in candidate mode')\n"
        "        if not Path(_file).is_file():\n"
        "            raise FileNotFoundError(f'TunableOp candidate file not found: {_file}')\n"
        "        import torch\n"
        "        _t = torch.cuda.tunable\n"
        "        _t.enable(True)\n"
        "        _t.tuning_enable(False)\n"
        "        _t.record_untuned_enable(False)\n"
        "        if hasattr(_t, 'set_filename'):\n"
        "            _t.set_filename(_file)\n"
        "        if not hasattr(_t, 'read_file'):\n"
        "            raise RuntimeError('torch.cuda.tunable.read_file unavailable; cannot load TunableOp candidate')\n"
        "        _t.read_file(_file)\n"
        "    except Exception as exc:\n"
        "        if _verbose:\n"
        "            print(f'HL_TUNABLEOP_READ_FAILED {type(exc).__name__}: {exc}', flush=True)\n"
        "        raise SystemExit(f'HL_TUNABLEOP_READ_FAILED {type(exc).__name__}: {exc}') from exc\n"
    )


def _candidate_pythonpath(site_dir: Path) -> str:
    # Hyperloom currently starts Forge with the same base environment that is later
    # used for E2E validation, then applies recommended_env as an override. Capture
    # and prepend here so candidate sitecustomize is injected without dropping that
    # base PYTHONPATH. If the consumer-side env model changes, move this prepend to
    # the consumer so it can merge against the actual target process environment.
    site = str(site_dir)
    existing = os.environ.get("PYTHONPATH", "").strip()
    return site if not existing else os.pathsep.join([site, existing])


def _generate_tunableop_script(
    work_dir: Path,
    input_file: Path,
    output_file: Path,
    timeout_per_shape: int,
    gpu_id: str,
) -> Path:
    """Generate a standalone script that runs PyTorch TunableOp offline tuning."""
    script_path = work_dir / "tunableop_tune.py"
    script_content = f'''#!/usr/bin/env python3
"""Auto-generated PyTorch TunableOp offline tuning script."""

import inspect
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "{gpu_id}")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "{gpu_id}")
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "{output_file}"

import torch

INPUT_FILE = "{input_file}"
OUTPUT_FILE = "{output_file}"
TIMEOUT_PER_SHAPE = {timeout_per_shape}


def _is_tunableop_result_line(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("Validator"):
        return False
    return stripped.count(",") >= 3


def main():
    if not hasattr(torch.cuda, "tunable"):
        print(json.dumps({{"status": "failed", "error": "torch.cuda.tunable not available"}}))
        return 1

    # Read untuned shapes
    if not os.path.isfile(INPUT_FILE):
        print(json.dumps({{"status": "failed", "error": f"Input file not found: {{INPUT_FILE}}"}}))
        return 1

    start = time.time()
    try:
        # Use PyTorch's built-in file-based tuning. PyTorch builds differ:
        # older APIs accept (input, output), while current ROCm APIs accept
        # input only and keep results in memory. Configure through Python APIs
        # because some ROCm builds ignore PYTORCH_TUNABLEOP_* env vars.
        tunable = torch.cuda.tunable
        tunable.enable(True)
        tunable.tuning_enable(True)
        tunable.record_untuned_enable(False)
        if hasattr(tunable, "set_filename"):
            tunable.set_filename(OUTPUT_FILE)

        sig = inspect.signature(tunable.tune_gemm_in_file)
        if len(sig.parameters) >= 2:
            tunable.tune_gemm_in_file(INPUT_FILE, OUTPUT_FILE)
        else:
            tunable.tune_gemm_in_file(INPUT_FILE)
            results = list(tunable.get_results())
            if results:
                validators = list(tunable.get_validators())
                tmp = OUTPUT_FILE + ".tmp"
                with open(tmp, "w") as f:
                    for key, value in validators:
                        f.write(f"Validator,{{key}},{{value}}\\n")
                    for op, params, solution, elapsed_ms in results:
                        f.write(f"{{op}},{{params}},{{solution}},{{elapsed_ms}}\\n")
                os.replace(tmp, OUTPUT_FILE)
        elapsed = time.time() - start

        # Count tuned shapes
        tuned_count = 0
        if os.path.isfile(OUTPUT_FILE):
            with open(OUTPUT_FILE) as f:
                tuned_count = sum(1 for line in f if _is_tunableop_result_line(line))

        print(json.dumps({{
            "status": "ok",
            "output": OUTPUT_FILE,
            "tuned_shapes": tuned_count,
            "elapsed_s": round(elapsed, 2),
        }}))
        return 0
    except Exception as e:
        elapsed = time.time() - start
        print(json.dumps({{
            "status": "failed",
            "error": str(e),
            "error_class": type(e).__name__,
            "elapsed_s": round(elapsed, 2),
        }}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


class VllmDenseTunableopTuner(BaseTuner):
    """Tune dense GEMM via PyTorch TunableOp (hipBLASLt/rocBLAS kernel selection)."""

    name = "vllm_dense_tunableop"
    env_var = TUNER_ENV_VARS["vllm_dense_tunableop"]

    def validate(self) -> str | None:
        if not (self.ctx.tunableop_input or self.ctx.shapes_json or getattr(self.ctx, "demand_json", None)):
            return (
                "Requires --tunableop-input (from PYTORCH_TUNABLEOP_RECORD_UNTUNED=1), "
                "--shapes-json, or --demand for GEMM shapes"
            )
        return None

    def _input_from_demand(self) -> Path | None:
        """Write an untuned record file from the keys the runtime missed.

        The router counts a demand file as a shape source, which is what lets
        this tuner run on a model that has no recorded TunableOp trace. It has
        to be able to consume one too: selecting it on the strength of demand
        and then failing for want of an input is worse than the honest skip it
        replaced.
        """
        path = getattr(self.ctx, "demand_json", None)
        if not path:
            return None
        from ..evidence import (
            TABLE_KEY_SCHEMA,
            demand_for_tuner,
            demand_shapes,
            load_demand,
        )

        try:
            report = load_demand(path)
            entry = demand_for_tuner(report, self.name) if report else None
        except Exception as exc:  # noqa: BLE001 - a bad demand file is not fatal
            log.warning("%s: could not read demand from %s: %s", self.name, path, exc)
            return None
        if report is None:
            return None

        # bucket=False: the padded-M cover that the aiter tuners want is wrong
        # here. That cover is only reachable because aiter retries a failed
        # lookup at the padded M; TunableOp keys on the exact shape and has no
        # such fallback, so a row written at 512 does nothing for a request at
        # 464. This tuner needs the M values the runtime literally asked for.
        shapes = demand_shapes(entry, limit=_DEMAND_SHAPE_LIMIT, bucket=False) if entry else []
        if not shapes:
            # No demand names this tuner, which is the normal case: the runtime
            # logs lookups against aiter's tables, and TunableOp has no table of
            # its own to miss. But a dense miss is a dense (M, N, K) either way,
            # and on a run where aiter is not serving dense, this tuner is the
            # one that can cover those shapes. Without this the router selects
            # it off the demand and it then fails for want of an input.
            borrowed: list[dict] = []
            for other in report.get("demands") or []:
                table = str(other.get("table") or "")
                if table not in TABLE_KEY_SCHEMA:
                    continue  # MoE and anything else that is not a dense GEMM
                # bucket=False for the same reason as the direct path above:
                # borrowing another table's misses does not borrow aiter's
                # padded-M retry along with them.
                borrowed.extend(demand_shapes(other, limit=_DEMAND_SHAPE_LIMIT, bucket=False))
            if borrowed:
                log.info(
                    "%s: no demand of its own; taking %d dense shape(s) the runtime missed on other dense tables",
                    self.name,
                    len(borrowed[:_DEMAND_SHAPE_LIMIT]),
                )
            shapes = borrowed[:_DEMAND_SHAPE_LIMIT]
        if not shapes:
            return None

        # Per-shape dtype first, ctx.precision only as a fallback for shapes the
        # demand parser recorded without one. A checkpoint precision that has no
        # record type (mxfp4, fp8) is no longer fatal on its own -- the shapes
        # carry the dtype the GEMM actually ran in.
        precision = str(getattr(self.ctx, "precision", "") or "bf16").lower()
        fallback_op = _TUNABLEOP_OP_BY_PRECISION.get(precision)

        lines = []
        unsupported: dict[str, int] = {}
        for shape in shapes:
            try:
                m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
            except (KeyError, TypeError, ValueError):
                continue
            op = _tunableop_op_for_dtype(shape.get("dtype") or shape.get("otype")) or fallback_op
            if op is None:
                label = str(shape.get("dtype") or shape.get("otype") or f"precision={precision}")
                unsupported[label] = unsupported.get(label, 0) + 1
                continue
            lines.append(tunableop_untuned_line(m, n, k, op))
        if unsupported:
            detail = ", ".join(f"{k} x{v}" for k, v in sorted(unsupported.items()))
            log.warning(
                "%s: %d demand shape(s) skipped, no TunableOp record type for %s",
                self.name,
                sum(unsupported.values()),
                detail,
            )
            if not lines:
                # Every shape was an unsupported dtype. The demand file was read
                # and understood, so this is "nothing here for this tuner", not
                # a missing input; run() reports it as skipped.
                self._demand_skip_reason = f"no TunableOp record type for {detail}"
        if not lines:
            return None

        out = self.work_dir / "untuned_from_demand.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info(
            "%s: %d demand shape(s) written as TunableOp records -> %s",
            self.name,
            len(lines),
            out,
        )
        return out

    def _resolve_input(self) -> Path | None:
        """Get the TunableOp input file."""
        if self.ctx.tunableop_input and self.ctx.tunableop_input.is_file():
            return self.ctx.tunableop_input
        if self.ctx.shapes_json and self.ctx.shapes_json.is_file():
            # For TunableOp, we need the native format, not JSON
            # If shapes_json is actually a tunableop format file, use it directly
            return self.ctx.shapes_json
        return self._input_from_demand()

    # Set by _input_from_demand when the demand file parsed fine but carried no
    # shape this tuner has a record type for. Distinguishes "nothing to do" from
    # "the input never arrived".
    _demand_skip_reason: str = ""

    def run(self) -> TuneResult:
        input_file = self._resolve_input()
        if input_file is None and self._demand_skip_reason:
            log.info("%s: skipped -- %s", self.name, self._demand_skip_reason)
            return TuneResult(
                tuner_name=self.name,
                status="skipped",
                error=self._demand_skip_reason,
                error_class="unsupported_precision",
            )
        if input_file is None:
            # Say which sources were offered and why none produced a file. The
            # bare "No valid input file found" cost a real run: the router had
            # selected this tuner off a demand file it could not read, and the
            # log said nothing about which of the three inputs was missing.
            offered = {
                "tunableop_input": str(self.ctx.tunableop_input or ""),
                "shapes_json": str(self.ctx.shapes_json or ""),
                "demand_json": str(getattr(self.ctx, "demand_json", "") or ""),
            }
            detail = ", ".join(f"{k}={v or '(unset)'}" for k, v in offered.items())
            log.error("%s: no usable input file. Sources: %s", self.name, detail)
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"No valid input file found. Sources: {detail}",
                error_class="input_missing",
            )

        output_file = self.work_dir / "tunableop_results.csv"
        gpu_id = self.ctx.gpu_ids.split(",")[0] if self.ctx.gpu_ids else "0"
        timeout_per_shape = max(30, self.ctx.timeout_s // 100)

        script = _generate_tunableop_script(
            work_dir=self.work_dir,
            input_file=input_file,
            output_file=output_file,
            timeout_per_shape=timeout_per_shape,
            gpu_id=gpu_id,
        )

        rc, stdout, stderr = run_subprocess(
            ["python3", str(script)],
            timeout_s=self.ctx.timeout_s,
            log_file=self.work_dir / "tune.log",
        )

        # Parse result
        try:
            result_line = stdout.strip().splitlines()[-1] if stdout.strip() else "{}"
            script_result = json.loads(result_line)
        except (json.JSONDecodeError, IndexError):
            script_result = {}

        if rc == 124:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=f"TunableOp timed out after {self.ctx.timeout_s}s",
                error_class="timeout",
            )

        if script_result.get("status") != "ok":
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=script_result.get("error", f"rc={rc}, stderr={stderr[-300:]}"),
                error_class=script_result.get("error_class", "subprocess_error"),
            )

        tuned_shapes = int(script_result.get("tuned_shapes", 0) or 0)
        if tuned_shapes == 0 and output_file.is_file():
            raw_text = output_file.read_text(encoding="utf-8", errors="replace")
            tuned_shapes = count_tunableop_result_lines(raw_text)
            if tuned_shapes == 0 and raw_text.strip():
                log.warning(
                    "TunableOp output %s is non-empty but no result lines matched "
                    "(expected >=3 comma-separated fields); candidate will be skipped. "
                    "First 200 chars: %r",
                    output_file,
                    raw_text[:200],
                )

        env_vars: dict[str, str] = {}
        if tuned_shapes > 0:
            site_dir = self.work_dir / "runtime_sitecustomize"
            site_dir.mkdir(parents=True, exist_ok=True)
            (site_dir / "sitecustomize.py").write_text(
                _generate_candidate_sitecustomize(),
                encoding="utf-8",
            )
            env_vars = {
                "PYTHONPATH": _candidate_pythonpath(site_dir),
                "HL_TUNABLEOP_MODE": "candidate",
                "HL_TUNABLEOP_FILE": str(output_file),
                "PYTORCH_TUNABLEOP_FILENAME": str(output_file),
            }

        # TunableOp picks the fastest hipBLASLt/rocBLAS solution per shape but
        # never times the untuned dispatch, so there is no baseline to compare
        # against and improved_shapes is 0 by construction, not by measurement.
        # Reporting only "improved 0/N" made completed runs read as "this path
        # has nothing to gain"; unverified_shapes says what actually happened,
        # and candidate sends the artifact to E2E where a real number exists.
        return TuneResult(
            tuner_name=self.name,
            status="ok" if tuned_shapes > 0 else "empty_output",
            artifact_path=str(output_file) if output_file.is_file() else "",
            env_var=self.env_var if tuned_shapes > 0 else "",
            env_value=str(output_file) if tuned_shapes > 0 else "",
            env_vars=env_vars,
            candidate=tuned_shapes > 0,
            total_shapes=tuned_shapes,
            improved_shapes=0,
            unverified_shapes=tuned_shapes,
            best_micro_speedup=1.0,
            avg_micro_speedup=1.0,
        )
