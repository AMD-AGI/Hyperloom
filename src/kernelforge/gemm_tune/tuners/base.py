# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base tuner abstract class and result dataclass."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model_analyzer import ModelProfile


@dataclass
class TuneResult:
    """Result from a single tuner run."""

    tuner_name: str
    # "ok", "skipped", "failed", "no_improvement", "empty_output", "partial_output"
    status: str
    # Artifacts
    artifact_path: str = ""  # Path to the produced tuned CSV/JSON
    env_var: str = ""  # Environment variable name to apply
    env_value: str = ""  # Environment variable value (usually = artifact_path)
    env_vars: dict[str, str] = field(default_factory=dict)  # Additional env vars to apply.
    candidate: bool = False  # True when E2E validation should test this artifact.
    # Metrics
    total_shapes: int = 0
    improved_shapes: int = 0
    # Shapes handed to the tuner. Compared against total_shapes (rows actually
    # produced) to detect a partial run: aiter's own "tune N shapes" line and its
    # exit code both misreport this, so row count is the only reliable signal.
    expected_shapes: int = 0
    # Shapes that were tuned but have no comparable untuned baseline, so
    # improved_shapes cannot count them. Distinguishes "compared, did not win"
    # from "never had anything to compare against".
    unverified_shapes: int = 0
    best_micro_speedup: float = 1.0
    avg_micro_speedup: float = 1.0
    # Per-shape detail (list of dicts with keys: token/M, default_us, tuned_us, speedup)
    shape_results: list[dict[str, Any]] = field(default_factory=list)
    # Rows removed from the deployed artifact because the tuner's own accuracy
    # check found them wrong. Reported rather than silently dropped: "this shape
    # has no tuned entry" and "this shape had one and it computed the wrong
    # answer" are different facts, and only the second says a backend is broken.
    dropped_inaccurate: list[dict[str, Any]] = field(default_factory=list)
    # Timing
    elapsed_s: float = 0.0
    # Error info
    error: str = ""
    error_class: str = ""
    # Skip reason (from router)
    skip_reason: str = ""
    # Where the tuned shapes/keys came from: "runtime_observed" when the caller
    # supplied them from a live dispatch log, "config_derived" when this tuner
    # inferred them from the model config. Recorded because an inferred key can
    # disagree with what the serving framework dispatches, and the resulting
    # unreachable table is otherwise indistinguishable from a tuning that simply
    # did not pay off.
    key_source: str = ""

    @property
    def has_improvement(self) -> bool:
        return self.candidate or (self.improved_shapes > 0 and self.best_micro_speedup > 1.0)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tuner": self.tuner_name,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s, 2),
        }
        if self.artifact_path:
            d["artifact"] = self.artifact_path
        if self.env_var:
            d["env_var"] = self.env_var
            d["env_value"] = self.env_value
        if self.env_vars:
            d["env_vars"] = dict(self.env_vars)
        if self.candidate:
            d["candidate"] = True
        if self.total_shapes:
            d["total_shapes"] = self.total_shapes
            d["improved_shapes"] = self.improved_shapes
            d["best_micro_speedup"] = round(self.best_micro_speedup, 4)
            d["avg_micro_speedup"] = round(self.avg_micro_speedup, 4)
        if self.expected_shapes:
            d["expected_shapes"] = self.expected_shapes
            # A row the accuracy check removed was tuned; it is missing from the
            # artifact but it was not missed by the run. Counting it as
            # "missing" made a completed batch read as a truncated one, which is
            # the same conflation the partial_output gate used to make.
            d["filtered_shapes"] = len(self.dropped_inaccurate)
            d["missing_shapes"] = max(self.expected_shapes - self.total_shapes - len(self.dropped_inaccurate), 0)
        if self.unverified_shapes:
            d["unverified_shapes"] = self.unverified_shapes
        if self.shape_results:
            d["shape_results"] = self.shape_results
        if self.dropped_inaccurate:
            d["dropped_inaccurate"] = self.dropped_inaccurate
        if self.error:
            d["error"] = self.error
            d["error_class"] = self.error_class
        if self.skip_reason:
            d["skip_reason"] = self.skip_reason
        if self.key_source:
            d["key_source"] = self.key_source
        return d


@dataclass
class TuneContext:
    """Runtime context passed to every tuner."""

    profile: ModelProfile
    framework: str
    precision: str
    quant_type: str
    gpu_type: str
    tp: int
    conc: int
    tokens: list[int]
    mp: int  # parallel GPU count for tuning
    output_dir: Path
    iters: int
    warmup: int
    min_improvement_pct: float
    timeout_s: int
    thorough: bool = False  # Full search: all libtypes, more shapes, no per-shape timeout
    # Optional input files
    untuned_csv: Path | None = None
    # MoE shapes are kept in their own field because the dense and MoE untuned
    # CSVs are different schemas (M,N,K versus token,model_dim,inter_dim,...).
    # Sharing one field would hand each tuner family the other's table.
    moe_untuned_csv: Path | None = None
    shapes_json: Path | None = None
    # Weighted, variant-discriminating TraceShapeManifest (Hyperloom WP-1). When
    # supplied it is the preferred dense-shape source (real replay-weighted
    # shapes); see tuners._aiter_dense_common._resolve_input_csv.
    shapes_manifest: Path | None = None
    # demand.json from kernelforge.gemm_tune.evidence: the keys the runtime actually
    # looked up and missed. Preferred over anything derived from config.json,
    # which measured 0.4% coverage of real lookups.
    demand_json: Path | None = None
    tunableop_input: Path | None = None
    kernel_signature_log: Path | None = None
    # The token counts the log shows this particular tuner's kernel actually
    # serving, as opposed to ``tokens``, which is the run's coverage sweep. Set
    # from TunerSpec.token_hint. A tuner that has one should treat it as the
    # allowed set (intersect), not merely as a budget: on a MoE model the
    # 1-stage and Triton paths serve token counts that CK never sees, and
    # tuning those spends the budget on kernels that will not be dispatched.
    token_hint: list[int] | None = None
    gpu_ids: str = ""
    # Additional env overrides from caller
    extra_env: dict[str, str] = field(default_factory=dict)


class BaseTuner(ABC):
    """Abstract base for all tuner backends."""

    # Subclasses must set these
    name: str = ""
    env_var: str = ""

    def __init__(self, ctx: TuneContext):
        self.ctx = ctx
        self.work_dir = ctx.output_dir / "tuners" / self.name
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate(self) -> str | None:
        """Pre-flight validation. Returns error message or None if OK."""

    @abstractmethod
    def run(self) -> TuneResult:
        """Execute tuning. Returns TuneResult."""

    def execute(self) -> TuneResult:
        """Validate then run, converting any failure into a TuneResult.

        ``validate`` is inside the guard because implementations derive shapes
        there, which puts raw config values through ``int()``. A raise outside it
        would leave the CLI with no sentinel JSON for the caller to read.
        """
        started = time.time()
        try:
            err = self.validate()
            if err:
                return TuneResult(
                    tuner_name=self.name,
                    status="failed",
                    error=err,
                    error_class="validation_error",
                )
            result = self.run()
            result.elapsed_s = time.time() - started
            return result
        except Exception as exc:
            return TuneResult(
                tuner_name=self.name,
                status="failed",
                error=repr(exc),
                error_class=type(exc).__name__,
                elapsed_s=time.time() - started,
            )
