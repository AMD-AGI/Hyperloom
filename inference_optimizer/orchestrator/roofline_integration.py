"""Self-contained PMC and roofline integration for inference_optimizer.

This is adapted from TBO's ``marathon_v3.pmc_profiler`` and
``marathon_v3.roofline`` but intentionally kept local to Hyperloom. The profile
flow should not depend on a sibling TBO checkout existing in a developer
workspace.

The module is best-effort: ROCm/rocprofv3 failures return structured
``status=skipped`` results so the existing torch-trace path remains usable.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


PMC_COUNTER_GROUPS = [
    "SQ_INSTS_VALU SQ_INSTS_MFMA SQ_WAVES SQ_BUSY_CU_CYCLES",
    "TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_32B_sum TCC_EA_RDREQ_sum TCC_EA_WRREQ_sum",
    "SQ_ACTIVE_INST_VALU SQ_INSTS_SALU SQ_WAIT_INST_ANY GRBM_COUNT GRBM_GUI_ACTIVE",
]


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_kernel_name(raw: str) -> str:
    """Shorten rocprofv3 kernel names to stable categories."""
    lower = (raw or "").lower()
    if "Cijk_" in raw:
        return "hipblaslt_gemm"
    if "gemm_a16w16_asm" in lower or "A16W16" in raw:
        return "aiter_asm_gemm"
    if "attn_fwd" in lower or "flash_attn" in lower:
        return "attention"
    if "moe_ck2stages" in lower or "moe_ck_tile" in lower:
        return "moe_gemm"
    if "vectorized_layer_norm" in lower or "rms_norm" in lower:
        return "rms_norm"
    if "distribution_elementwise" in lower:
        return "random_init"
    if "fillbuffer" in lower or "fillfunctor" in lower:
        return "fill"
    if "topk" in lower:
        return "topk"
    if "rope" in lower or "rotary" in lower:
        return "rope"
    if "nccl" in lower or "rccl" in lower or "allreduce" in lower:
        return "allreduce"
    if "copy" in lower or "memcpy" in lower:
        return "memcpy"
    if "softmax" in lower:
        return "softmax"
    if "skinny" in lower:
        return "skinny_gemm"
    return raw[:80] if len(raw) > 80 else raw


@dataclass
class PMCKernelResult:
    """PMC counter results for one normalized kernel name."""

    name: str
    dispatches: int = 0
    SQ_INSTS_MFMA: float = 0.0
    SQ_INSTS_VALU: float = 0.0
    SQ_WAVES: float = 0.0
    SQ_BUSY_CU_CYCLES: float = 0.0
    TCC_EA_RDREQ_32B_sum: float = 0.0
    TCC_EA_WRREQ_32B_sum: float = 0.0
    TCC_EA_RDREQ_sum: float = 0.0
    TCC_EA_WRREQ_sum: float = 0.0
    SQ_ACTIVE_INST_VALU: float = 0.0
    SQ_INSTS_SALU: float = 0.0
    SQ_WAIT_INST_ANY: float = 0.0
    duration_ns: float = 0.0
    grid_size: int = 0
    workgroup_size: int = 0

    @property
    def mfma_ratio_pct(self) -> float:
        total = self.SQ_INSTS_MFMA + self.SQ_INSTS_VALU
        return self.SQ_INSTS_MFMA / total * 100.0 if total else 0.0

    @property
    def bytes_transferred(self) -> float | None:
        if self.TCC_EA_RDREQ_32B_sum or self.TCC_EA_WRREQ_32B_sum:
            return (self.TCC_EA_RDREQ_32B_sum + self.TCC_EA_WRREQ_32B_sum) * 32
        if self.TCC_EA_RDREQ_sum or self.TCC_EA_WRREQ_sum:
            return (self.TCC_EA_RDREQ_sum + self.TCC_EA_WRREQ_sum) * 64
        return None

    def to_roofline_dict(self, gpu_pct: float = 0.0) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "gpu_pct": gpu_pct,
            "duration_us": self.duration_ns / 1000.0 if self.duration_ns else 0.0,
            "SQ_INSTS_VALU": self.SQ_INSTS_VALU,
            "SQ_INSTS_MFMA": self.SQ_INSTS_MFMA,
            "SQ_WAVES": self.SQ_WAVES,
            "SQ_BUSY_CU_CYCLES": self.SQ_BUSY_CU_CYCLES,
        }
        for key in (
            "TCC_EA_RDREQ_32B_sum",
            "TCC_EA_WRREQ_32B_sum",
            "TCC_EA_RDREQ_sum",
            "TCC_EA_WRREQ_sum",
            "SQ_ACTIVE_INST_VALU",
            "SQ_INSTS_SALU",
            "SQ_WAIT_INST_ANY",
        ):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    def to_summary_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dispatches": self.dispatches,
            "duration_us": round(self.duration_ns / 1000.0, 1) if self.duration_ns else 0,
            "classification": self.classify(),
        }
        if self.SQ_INSTS_MFMA or self.SQ_INSTS_VALU:
            data.update({
                "SQ_INSTS_MFMA_per_dispatch": self.SQ_INSTS_MFMA / max(self.dispatches, 1),
                "SQ_INSTS_VALU_per_dispatch": self.SQ_INSTS_VALU / max(self.dispatches, 1),
                "SQ_WAVES_per_dispatch": self.SQ_WAVES / max(self.dispatches, 1),
                "SQ_BUSY_CU_CYCLES_per_dispatch": self.SQ_BUSY_CU_CYCLES / max(self.dispatches, 1),
                "mfma_ratio_pct": round(self.mfma_ratio_pct, 1),
                "bytes_transferred": self.bytes_transferred,
            })
        gpu_pct = getattr(self, "_gpu_time_pct", None)
        if gpu_pct is not None:
            data["gpu_time_pct"] = round(float(gpu_pct), 1)
        return data

    def classify(self) -> str:
        if self.SQ_INSTS_MFMA or self.SQ_INSTS_VALU:
            ratio = self.mfma_ratio_pct
            if ratio > 30:
                return "compute_bound"
            if ratio < 15:
                return "memory_bound"
            return "mixed"

        lower = self.name.lower()
        if any(k in lower for k in ("gemm", "hipblaslt", "cijk", "matmul")):
            return "compute_bound"
        if any(k in lower for k in ("attention", "attn", "flash", "norm", "rope", "topk")):
            return "memory_bound"
        if any(k in lower for k in ("allreduce", "nccl", "rccl")):
            return "latency_bound"
        return "unknown"


class PMCProfiler:
    """Collect PMC counters from live inference workers via rocprofv3."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.profile_dir = self.session_dir / "profiles"
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def _write_counter_file(self) -> Path:
        path = self.profile_dir / "pmc_counters.txt"
        path.write_text(
            "".join(f"pmc: {group}\n" for group in PMC_COUNTER_GROUPS),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def server_profiling_env() -> dict[str, str]:
        for candidate in (
            "/opt/rocm/lib/librocprofiler-register.so",
            "/opt/rocm-7.2.1/lib/librocprofiler-register.so",
        ):
            if os.path.exists(candidate):
                existing = os.environ.get("LD_PRELOAD", "")
                if candidate in existing:
                    return {}
                preload = f"{existing}:{candidate}" if existing else candidate
                return {"LD_PRELOAD": preload}
        return {}

    def _find_rocprofv3(self) -> str:
        for candidate in ("rocprofv3", "/opt/rocm/bin/rocprofv3"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise FileNotFoundError("rocprofv3 not found")

    def _find_worker_pid(self, pid_file: str = "/tmp/.marathon_server.pid") -> int | None:
        patterns = (
            "vllm.worker",
            "vllm.entrypoints",
            "sglang.launch_server",
            "sglang.srt",
        )
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            pids = [_to_int(p) for p in result.stdout.strip().split() if p]
            pids = [p for p in pids if p > 0 and p != os.getpid()]
            if pids:
                return pids[0]

        pid_path = Path(pid_file)
        if pid_path.exists():
            try:
                server_pid = _to_int(pid_path.read_text(encoding="utf-8").strip())
                result = subprocess.run(
                    ["pgrep", "-P", str(server_pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                children = [_to_int(p) for p in result.stdout.strip().split() if p]
                children = [p for p in children if p > 0]
                return children[0] if children else server_pid
            except (OSError, subprocess.TimeoutExpired):
                return None
        return None

    def profile_live_server(self, duration_ms: int = 15000) -> list[PMCKernelResult]:
        worker_pid = self._find_worker_pid()
        if not worker_pid:
            raise RuntimeError("Cannot find vLLM/SGLang worker process")
        return self.profile_attach(worker_pid=worker_pid, duration_ms=duration_ms)

    def profile_attach(
        self,
        worker_pid: int,
        duration_ms: int = 15000,
    ) -> list[PMCKernelResult]:
        rocprof = self._find_rocprofv3()
        run_tag = f"attach_{int(time.time())}"
        output_dir = self.profile_dir / "pmc_attach" / run_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        counter_file = self._write_counter_file()
        cmd_pmc = [
            rocprof,
            "--attach", str(worker_pid),
            "--attach-duration-msec", str(duration_ms),
            "-i", str(counter_file),
            "-d", str(output_dir),
            "-f", "csv",
            "--kernel-trace",
        ]
        subprocess.run(
            cmd_pmc,
            capture_output=True,
            text=True,
            timeout=duration_ms / 1000.0 + 30,
        )

        counter_csvs = list(output_dir.rglob("*counter_collection*.csv"))
        if counter_csvs:
            return self._parse_counter_csvs(counter_csvs)

        trace_csvs = list(output_dir.rglob("*kernel_trace*.csv"))
        if not trace_csvs:
            trace_dir = Path(f"{output_dir}_trace")
            trace_dir.mkdir(parents=True, exist_ok=True)
            cmd_trace = [
                rocprof,
                "--attach", str(worker_pid),
                "--attach-duration-msec", str(duration_ms),
                "-d", str(trace_dir),
                "-f", "csv",
                "--kernel-trace",
            ]
            subprocess.run(
                cmd_trace,
                capture_output=True,
                text=True,
                timeout=duration_ms / 1000.0 + 30,
            )
            trace_csvs = list(trace_dir.rglob("*kernel_trace*.csv"))
        return self._parse_trace_only(trace_csvs) if trace_csvs else []

    def save_results(self, results: list[PMCKernelResult], tag: str = "pmc") -> str:
        summary = {result.name: result.to_summary_dict() for result in results}
        path = self.profile_dir / f"{tag}_summary.json"
        path.write_text(
            json.dumps({
                "source": "rocprofv3",
                "tag": tag,
                "counters": [group.split() for group in PMC_COUNTER_GROUPS],
                "kernels": summary,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return str(path)

    def _parse_counter_csvs(self, csv_files: list[Path]) -> list[PMCKernelResult]:
        kernels: dict[str, PMCKernelResult] = {}
        counter_cols = {
            "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_WAVES", "SQ_BUSY_CU_CYCLES",
            "TCC_EA_RDREQ_32B_sum", "TCC_EA_WRREQ_32B_sum",
            "TCC_EA_RDREQ_sum", "TCC_EA_WRREQ_sum",
            "SQ_ACTIVE_INST_VALU", "SQ_INSTS_SALU", "SQ_WAIT_INST_ANY",
        }
        for csv_file in csv_files:
            with csv_file.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = _normalize_kernel_name(row.get("Kernel_Name", ""))
                    if name in ("random_init", "fill", "unknown", ""):
                        continue
                    kernel = kernels.setdefault(
                        name,
                        PMCKernelResult(
                            name=name,
                            grid_size=_to_int(row.get("Grid_Size")),
                            workgroup_size=_to_int(row.get("Workgroup_Size")),
                        ),
                    )
                    counter = row.get("Counter_Name", "")
                    value = _to_float(row.get("Counter_Value"))
                    if counter in counter_cols and hasattr(kernel, counter):
                        setattr(kernel, counter, getattr(kernel, counter) + value)
                    if counter == "SQ_INSTS_MFMA":
                        kernel.dispatches += 1

                    start = _to_float(row.get("Start_Timestamp"))
                    end = _to_float(row.get("End_Timestamp"))
                    if end > start:
                        kernel.duration_ns += end - start
        return list(kernels.values())

    def _parse_trace_only(self, csv_files: list[Path]) -> list[PMCKernelResult]:
        kernels: dict[str, PMCKernelResult] = {}
        for csv_file in csv_files:
            with csv_file.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    name = _normalize_kernel_name(row.get("Kernel_Name", ""))
                    if name in ("random_init", "fill", "unknown", ""):
                        continue
                    kernel = kernels.setdefault(name, PMCKernelResult(name=name))
                    kernel.dispatches += 1
                    start = _to_float(row.get("Start_Timestamp"))
                    end = _to_float(row.get("End_Timestamp"))
                    if end > start:
                        kernel.duration_ns += end - start

        total_ns = sum(kernel.duration_ns for kernel in kernels.values())
        results = sorted(kernels.values(), key=lambda item: item.duration_ns, reverse=True)
        for kernel in results:
            kernel._gpu_time_pct = kernel.duration_ns / total_ns * 100.0 if total_ns else 0.0
        return results


class Bottleneck(Enum):
    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LATENCY_BOUND = "latency_bound"
    UNKNOWN = "unknown"


@dataclass
class GPUSpec:
    name: str
    peak_flops_fp16: float
    peak_flops_fp8: float
    peak_flops_fp4: float
    peak_flops_fp32: float
    peak_bandwidth_tbps: float
    num_cus: int
    lds_per_cu_kb: int
    vgpr_per_simd: int
    waves_per_simd: int
    ridge_point_fp16: float = 0.0
    ridge_point_fp8: float = 0.0

    def __post_init__(self) -> None:
        bandwidth = self.peak_bandwidth_tbps * 1e12
        self.ridge_point_fp16 = (self.peak_flops_fp16 * 1e12) / bandwidth
        self.ridge_point_fp8 = (self.peak_flops_fp8 * 1e12) / bandwidth


MI355X_SPEC = GPUSpec(
    name="MI355X (gfx950, CDNA4)",
    peak_flops_fp16=1311.0,
    peak_flops_fp8=2621.0,
    peak_flops_fp4=5243.0,
    peak_flops_fp32=163.9,
    peak_bandwidth_tbps=8.0,
    num_cus=304,
    lds_per_cu_kb=128,
    vgpr_per_simd=256,
    waves_per_simd=4,
)


@dataclass
class KernelRooflineResult:
    name: str
    gpu_pct: float
    duration_us: float
    flops: float | None = None
    bytes_transferred: float | None = None
    arithmetic_intensity: float | None = None
    bottleneck: Bottleneck = Bottleneck.UNKNOWN
    compute_utilization_pct: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    occupancy_waves: int | None = None
    suggestion: str = ""
    recommended_specialist: str = "kernel"
    recommended_actions: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gpu_pct": self.gpu_pct,
            "duration_us": self.duration_us,
            "flops": self.flops,
            "bytes_transferred": self.bytes_transferred,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bottleneck": self.bottleneck.value,
            "compute_utilization_pct": self.compute_utilization_pct,
            "bandwidth_utilization_pct": self.bandwidth_utilization_pct,
            "occupancy_waves": self.occupancy_waves,
            "suggestion": self.suggestion,
            "recommended_specialist": self.recommended_specialist,
            "recommended_actions": self.recommended_actions,
            "constraints": self.constraints,
        }


class RooflineAnalyzer:
    """Classify kernels against the MI355X roofline."""

    def __init__(self, gpu_spec: GPUSpec = MI355X_SPEC):
        self.spec = gpu_spec

    def analyze_kernels(
        self,
        pmc_data: list[dict[str, Any]],
        precision: str = "fp16",
    ) -> list[KernelRooflineResult]:
        results = [self._analyze_single(kernel, precision) for kernel in pmc_data]
        return sorted(results, key=lambda result: result.gpu_pct, reverse=True)

    def _analyze_single(
        self,
        kernel: dict[str, Any],
        precision: str,
    ) -> KernelRooflineResult:
        result = KernelRooflineResult(
            name=str(kernel.get("name", "unknown")),
            gpu_pct=_to_float(kernel.get("gpu_pct")),
            duration_us=_to_float(kernel.get("duration_us")),
        )
        result.flops = self._estimate_flops(kernel)
        result.bytes_transferred = self._estimate_bytes(kernel)

        duration_s = result.duration_us * 1e-6 if result.duration_us > 0 else 1e-9
        peak_flops = self._peak_flops(precision)
        if result.flops and result.bytes_transferred and result.bytes_transferred > 0:
            result.arithmetic_intensity = result.flops / result.bytes_transferred
            achieved_flops_per_s = result.flops / duration_s
            achieved_bw = result.bytes_transferred / duration_s
            result.compute_utilization_pct = achieved_flops_per_s / (peak_flops * 1e12) * 100.0
            result.bandwidth_utilization_pct = (
                achieved_bw / (self.spec.peak_bandwidth_tbps * 1e12) * 100.0
            )

        waves = _to_int(kernel.get("SQ_WAVES"))
        result.occupancy_waves = waves if waves else None
        result._raw_valu = _to_float(kernel.get("SQ_INSTS_VALU"))
        result.bottleneck = self._classify(result, precision)
        self._suggest(result)
        return result

    def _peak_flops(self, precision: str) -> float:
        return {
            "fp4": self.spec.peak_flops_fp4,
            "fp8": self.spec.peak_flops_fp8,
            "fp16": self.spec.peak_flops_fp16,
            "bf16": self.spec.peak_flops_fp16,
            "fp32": self.spec.peak_flops_fp32,
        }.get(precision, self.spec.peak_flops_fp16)

    def _estimate_flops(self, kernel: dict[str, Any]) -> float | None:
        mfma = _to_float(kernel.get("SQ_INSTS_MFMA"))
        valu = _to_float(kernel.get("SQ_INSTS_VALU"))
        if mfma:
            return mfma * 32768
        if valu:
            return valu * 64
        return None

    def _estimate_bytes(self, kernel: dict[str, Any]) -> float | None:
        rd_32b = _to_float(kernel.get("TCC_EA_RDREQ_32B_sum"))
        wr_32b = _to_float(kernel.get("TCC_EA_WRREQ_32B_sum"))
        if rd_32b or wr_32b:
            return (rd_32b + wr_32b) * 32
        rd = _to_float(kernel.get("TCC_EA_RDREQ_sum"))
        wr = _to_float(kernel.get("TCC_EA_WRREQ_sum"))
        if rd or wr:
            return (rd + wr) * 64
        return None

    def _classify(self, result: KernelRooflineResult, precision: str) -> Bottleneck:
        ai = result.arithmetic_intensity
        if ai is not None:
            ridge = self.spec.ridge_point_fp8 if precision in ("fp8", "fp4") else self.spec.ridge_point_fp16
            if ai < ridge * 0.3:
                return Bottleneck.MEMORY_BOUND
            if ai > ridge * 1.5:
                return Bottleneck.COMPUTE_BOUND
            if result.compute_utilization_pct < 10 and result.bandwidth_utilization_pct < 10:
                return Bottleneck.LATENCY_BOUND
            return (
                Bottleneck.MEMORY_BOUND
                if result.bandwidth_utilization_pct > result.compute_utilization_pct
                else Bottleneck.COMPUTE_BOUND
            )

        if result.flops and result.flops > 0:
            mfma_insts = result.flops / 32768
            valu_insts = getattr(result, "_raw_valu", 0.0)
            total = mfma_insts + valu_insts
            if total > 0:
                ratio = mfma_insts / total
                if ratio > 0.30:
                    return Bottleneck.COMPUTE_BOUND
                if ratio < 0.15:
                    return Bottleneck.MEMORY_BOUND
        return Bottleneck.UNKNOWN

    def _suggest(self, result: KernelRooflineResult) -> None:
        if result.bottleneck == Bottleneck.MEMORY_BOUND:
            result.suggestion = (
                "Memory-bound. Reduce HBM traffic with fusion, tiling for reuse, "
                "layout cleanup, or intermediate quantization."
            )
            result.recommended_specialist = "fusion"
            result.recommended_actions = [
                "fusion-design",
                "graph-optimization",
                "inductor-optimization",
            ]
            result.constraints = {
                "focus": "reduce_hbm_traffic",
                "current_bandwidth_util_pct": result.bandwidth_utilization_pct,
                "arithmetic_intensity": result.arithmetic_intensity,
            }
        elif result.bottleneck == Bottleneck.COMPUTE_BOUND:
            result.suggestion = (
                "Compute-bound. Improve MFMA scheduling, operand reuse, instruction "
                "mix, and occupancy."
            )
            result.recommended_specialist = "kernel"
            result.recommended_actions = ["deep-kernel-analysis", "operator-tuning"]
            result.constraints = {
                "focus": "instruction_efficiency",
                "current_compute_util_pct": result.compute_utilization_pct,
                "arithmetic_intensity": result.arithmetic_intensity,
                "target_vgprs": 64,
            }
        elif result.bottleneck == Bottleneck.LATENCY_BOUND:
            result.suggestion = (
                "Latency-bound. Increase parallelism, fuse adjacent kernels, or use "
                "graph capture to amortize launch overhead."
            )
            result.recommended_specialist = "systems"
            result.recommended_actions = ["graph-optimization", "fusion-design"]
            result.constraints = {
                "focus": "increase_parallelism",
                "occupancy_waves": result.occupancy_waves,
            }
        else:
            result.suggestion = "Bottleneck unknown. Need PMC counters for classification."
            result.recommended_specialist = "kernel"
            result.recommended_actions = ["deep-kernel-analysis"]


def score_boost_from_roofline(
    action_name: str,
    roofline_results: list[KernelRooflineResult],
) -> float:
    total_gpu_pct = sum(
        result.gpu_pct
        for result in roofline_results
        if action_name in result.recommended_actions
    )
    if total_gpu_pct > 20:
        return 2.0
    if total_gpu_pct > 10:
        return 1.5
    if total_gpu_pct > 5:
        return 1.2
    return 1.0


def server_profiling_env() -> dict[str, str]:
    return PMCProfiler.server_profiling_env()


def collect_profile_roofline(
    *,
    session_dir: Path,
    duration_ms: int = 15000,
    precision: str = "fp16",
) -> dict[str, Any]:
    """Collect PMC counters from a live server and write roofline artifacts."""
    profile_dir = Path(session_dir) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        profiler = PMCProfiler(session_dir)
        pmc_results = profiler.profile_live_server(duration_ms=duration_ms)
        if not pmc_results:
            return {"status": "skipped", "reason": "no PMC results produced"}

        pmc_summary_path = profiler.save_results(pmc_results, tag="pmc")
        total_ns = sum(result.duration_ns for result in pmc_results)
        pmc_dicts = []
        for result in pmc_results:
            gpu_pct = getattr(result, "_gpu_time_pct", None)
            if gpu_pct is None:
                gpu_pct = result.duration_ns / total_ns * 100.0 if total_ns else 0.0
            pmc_dicts.append(result.to_roofline_dict(gpu_pct=float(gpu_pct or 0.0)))

        analyzer = RooflineAnalyzer(gpu_spec=MI355X_SPEC)
        roofline_results = analyzer.analyze_kernels(pmc_dicts, precision=precision)
        roofline_payload = [result.to_dict() for result in roofline_results]
        roofline_path = profile_dir / "roofline.json"
        roofline_path.write_text(
            json.dumps(roofline_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return {
            "status": "ok",
            "pmc_summary_path": str(pmc_summary_path),
            "roofline_path": str(roofline_path),
            "roofline_kernel_count": len(roofline_payload),
        }
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": "rocprofv3 unavailable", "error": str(exc)}
    except RuntimeError as exc:
        return {"status": "skipped", "reason": "live server unavailable", "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"status": "skipped", "reason": "rocprofv3 timed out", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": repr(exc)}
