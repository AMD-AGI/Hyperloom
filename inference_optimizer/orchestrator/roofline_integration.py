"""Self-contained PMC and roofline helpers for a dedicated profile run.

ROCm allows only one rocprofiler tool registration per process. Do not use
these helpers in the normal torch-profiler server process. The caller must
launch a separate server process with ``LD_PRELOAD`` and use that process only
for PMC/roofline collection.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.request import urlopen


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
    lower = (raw or "").lower()
    if "cijk_" in lower:
        return "hipblaslt_gemm"
    if "gemm_a16w16_asm" in lower or "a16w16" in lower:
        return "aiter_asm_gemm"
    if "attn_fwd" in lower or "flash_attn" in lower:
        return "attention"
    if "moe_ck2stages" in lower or "moe_ck_tile" in lower:
        return "moe_gemm"
    if "vectorized_layer_norm" in lower or "rms_norm" in lower:
        return "rms_norm"
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
    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.profile_dir = self.session_dir / "profiles"
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def server_profiling_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(base_env or os.environ)
        for candidate in (
            "/opt/rocm/lib/librocprofiler-register.so",
            "/opt/rocm-7.2.1/lib/librocprofiler-register.so",
        ):
            if os.path.exists(candidate):
                existing = env.get("LD_PRELOAD", "")
                env["LD_PRELOAD"] = f"{existing}:{candidate}" if existing else candidate
                return env
        return env

    def _find_rocprofv3(self) -> str:
        for candidate in ("rocprofv3", "/opt/rocm/bin/rocprofv3"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        raise FileNotFoundError("rocprofv3 not found")

    def _write_counter_file(self) -> Path:
        path = self.profile_dir / "pmc_counters.txt"
        path.write_text("".join(f"pmc: {g}\n" for g in PMC_COUNTER_GROUPS), encoding="utf-8")
        return path

    def profile_attach(
        self,
        worker_pid: int,
        *,
        duration_ms: int = 15000,
        benchmark_cmd: list[str] | None = None,
        benchmark_env: dict[str, str] | None = None,
    ) -> list[PMCKernelResult]:
        rocprof = self._find_rocprofv3()
        run_tag = f"attach_{int(time.time())}"
        output_dir = self.profile_dir / "pmc_attach" / run_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        bench_proc: subprocess.Popen | None = None
        if benchmark_cmd:
            bench_proc = subprocess.Popen(
                benchmark_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=benchmark_env,
            )
            time.sleep(2)

        try:
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
        finally:
            if bench_proc and bench_proc.poll() is None:
                bench_proc.terminate()

    def profile_launch(
        self,
        *,
        server_cmd: list[str],
        health_url: str,
        log_path: Path,
        duration_ms: int = 15000,
        benchmark_cmd: list[str] | None = None,
        benchmark_env: dict[str, str] | None = None,
        startup_timeout_s: int = 600,
        server_env: dict[str, str] | None = None,
    ) -> list[PMCKernelResult]:
        """Launch the server under rocprofv3 instead of attaching by ptrace.

        This avoids ``rocprofv3 --attach`` permissions such as CAP_SYS_PTRACE.
        The server is still isolated from the normal torch-profiler trace run.
        """
        rocprof = self._find_rocprofv3()
        run_tag = f"launch_{int(time.time())}"
        output_dir = self.profile_dir / "pmc_launch" / run_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        counter_file = self._write_counter_file()
        cmd = [
            rocprof,
            "-i", str(counter_file),
            "-d", str(output_dir),
            "-f", "csv",
            "--kernel-trace",
            "--",
            *server_cmd,
        ]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=server_env,
                **_process_group_kwargs(),
            )
            try:
                if not _wait_for_health(health_url, proc, startup_timeout_s):
                    return []

                if benchmark_cmd:
                    bench = subprocess.run(
                        benchmark_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=benchmark_env,
                        timeout=max(duration_ms / 1000.0 + 120, 180),
                    )
                    log.write("\n[benchmark output]\n")
                    log.write(bench.stdout or "")
                    log.write(f"\n[benchmark exit_code] {bench.returncode}\n")
                    log.flush()
                else:
                    time.sleep(duration_ms / 1000.0)
            finally:
                _terminate_process_group(proc)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    _kill_process_group(proc)
                    proc.wait(timeout=30)

        counter_csvs = _wait_for_csvs(output_dir, "*counter_collection*.csv")
        if counter_csvs:
            return self._parse_counter_csvs(counter_csvs)
        trace_csvs = _wait_for_csvs(output_dir, "*kernel_trace*.csv")
        return self._parse_trace_only(trace_csvs) if trace_csvs else []

    def save_results(self, results: list[PMCKernelResult], tag: str = "pmc") -> str:
        path = self.profile_dir / f"{tag}_summary.json"
        path.write_text(
            json.dumps({
                "source": "rocprofv3",
                "tag": tag,
                "counters": [group.split() for group in PMC_COUNTER_GROUPS],
                "kernels": {r.name: r.to_summary_dict() for r in results},
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
                for row in csv.DictReader(handle):
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
                    if counter in counter_cols and hasattr(kernel, counter):
                        setattr(kernel, counter, getattr(kernel, counter) + _to_float(row.get("Counter_Value")))
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
                for row in csv.DictReader(handle):
                    name = _normalize_kernel_name(row.get("Kernel_Name", ""))
                    if name in ("random_init", "fill", "unknown", ""):
                        continue
                    kernel = kernels.setdefault(name, PMCKernelResult(name=name))
                    kernel.dispatches += 1
                    start = _to_float(row.get("Start_Timestamp"))
                    end = _to_float(row.get("End_Timestamp"))
                    if end > start:
                        kernel.duration_ns += end - start
        total_ns = sum(k.duration_ns for k in kernels.values())
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


MI355X_SPEC = GPUSpec("MI355X", 1311.0, 2621.0, 5243.0, 163.9, 8.0, 304, 128, 256, 4)


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
    def __init__(self, gpu_spec: GPUSpec = MI355X_SPEC):
        self.spec = gpu_spec

    def analyze_kernels(self, pmc_data: list[dict[str, Any]], precision: str = "fp16") -> list[KernelRooflineResult]:
        results = [self._analyze_single(kernel, precision) for kernel in pmc_data]
        return sorted(results, key=lambda result: result.gpu_pct, reverse=True)

    def _analyze_single(self, kernel: dict[str, Any], precision: str) -> KernelRooflineResult:
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
            result.compute_utilization_pct = result.flops / duration_s / (peak_flops * 1e12) * 100.0
            result.bandwidth_utilization_pct = (
                result.bytes_transferred / duration_s / (self.spec.peak_bandwidth_tbps * 1e12) * 100.0
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
        return (rd + wr) * 64 if rd or wr else None

    def _classify(self, result: KernelRooflineResult, precision: str) -> Bottleneck:
        if result.arithmetic_intensity is not None:
            ridge = self.spec.ridge_point_fp8 if precision in ("fp8", "fp4") else self.spec.ridge_point_fp16
            if result.arithmetic_intensity < ridge * 0.3:
                return Bottleneck.MEMORY_BOUND
            if result.arithmetic_intensity > ridge * 1.5:
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
            result.suggestion = "Memory-bound. Reduce HBM traffic with fusion, reuse, or layout cleanup."
            result.recommended_specialist = "fusion"
            result.recommended_actions = ["fusion-design", "graph-optimization", "inductor-optimization"]
            result.constraints = {"focus": "reduce_hbm_traffic", "arithmetic_intensity": result.arithmetic_intensity}
        elif result.bottleneck == Bottleneck.COMPUTE_BOUND:
            result.suggestion = "Compute-bound. Improve MFMA scheduling, operand reuse, and occupancy."
            result.recommended_specialist = "kernel"
            result.recommended_actions = ["deep-kernel-analysis", "operator-tuning"]
            result.constraints = {"focus": "instruction_efficiency", "target_vgprs": 64}
        elif result.bottleneck == Bottleneck.LATENCY_BOUND:
            result.suggestion = "Latency-bound. Increase parallelism or amortize launch overhead."
            result.recommended_specialist = "systems"
            result.recommended_actions = ["graph-optimization", "fusion-design"]
            result.constraints = {"focus": "increase_parallelism", "occupancy_waves": result.occupancy_waves}
        else:
            result.suggestion = "Bottleneck unknown. Need PMC counters for classification."
            result.recommended_actions = ["deep-kernel-analysis"]


def _as_cmd(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return shlex.split(value)
    return []


def _wait_for_csvs(directory: Path, pattern: str, timeout_s: float = 10.0) -> list[Path]:
    """Wait briefly for rocprofv3 CSV writers to flush after process exit."""
    deadline = time.time() + timeout_s
    while True:
        files = list(directory.rglob(pattern))
        if files or time.time() >= deadline:
            return files
        time.sleep(0.5)


def _process_group_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"preexec_fn": os.setsid}
    return {}


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass


def _kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def _classify_kernel_tier(name: str) -> str:
    """Keep the legacy 5-tier optimization routing alongside roofline data."""
    lower = name.lower()
    if any(x in name for x in ("triton_", "_permute_kernel", "triton_poi_", "triton_red_", "triton_tem_")):
        return "T1_TRITON"
    if any(x in name for x in ("aiter::", "fmha_v3", "mha_fwd", "fused_moe", "moe_ck", "topkGating",
                                "_gemm_a8w8", "_fused_rms")):
        return "T2_AITER_CK"
    if "vectorized_elementwise_kernel" in name:
        return "T2_AITER_CK"
    if any(x in lower for x in ("launch_server", "schedule", "batch", "token_dispatch",
                                 "kv_cache", "radix", "prefix_match")):
        return "T3_FRAMEWORK"
    if any(x in lower for x in ("nccl", "rccl", "allreduce", "broadcast", "all_gather", "reduce_scatter")):
        return "T4_COMM"
    if any(x in name for x in ("Cijk_", "ck::kernel")) or "hipmodule" in lower:
        return "T5_COMPILED"
    # Normalized rocprof names still need useful routing.
    if name in {"hipblaslt_gemm", "skinny_gemm", "aiter_asm_gemm", "moe_gemm"}:
        return "T5_COMPILED" if name == "hipblaslt_gemm" else "T2_AITER_CK"
    return "T2_AITER_CK"


def _write_kernel_breakdown(
    path: Path,
    *,
    pmc_results: list[PMCKernelResult],
    roofline_results: list[KernelRooflineResult],
) -> None:
    roofline_by_name = {result.name: result for result in roofline_results}
    total_ns = sum(result.duration_ns for result in pmc_results)
    rows: list[dict[str, Any]] = []
    for result in sorted(pmc_results, key=lambda item: item.duration_ns, reverse=True):
        roofline = roofline_by_name.get(result.name)
        gpu_pct = getattr(result, "_gpu_time_pct", None)
        if gpu_pct is None:
            gpu_pct = result.duration_ns / total_ns * 100.0 if total_ns else 0.0
        rows.append({
            "name": result.name,
            "gpu_pct": round(float(gpu_pct or 0.0), 3),
            "count": result.dispatches,
            "tier": _classify_kernel_tier(result.name),
            "bottleneck": roofline.bottleneck.value if roofline else "unknown",
            "arithmetic_intensity": roofline.arithmetic_intensity if roofline else None,
            "compute_utilization_pct": roofline.compute_utilization_pct if roofline else 0.0,
            "bandwidth_utilization_pct": roofline.bandwidth_utilization_pct if roofline else 0.0,
            "recommended_actions": roofline.recommended_actions if roofline else [],
            "suggestion": roofline.suggestion if roofline else "",
            "duration_us": round(result.duration_ns / 1000.0, 3) if result.duration_ns else 0.0,
            "dispatches": result.dispatches,
        })
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wait_for_health(url: str, proc: subprocess.Popen, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urlopen(url, timeout=3) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def run_isolated_pmc_roofline(
    *,
    session_dir: Path,
    server_cmd: list[str],
    health_url: str,
    benchmark_cmd: list[str] | None = None,
    duration_ms: int = 15000,
    precision: str = "fp16",
    startup_timeout_s: int = 600,
    env_overrides: dict[str, str] | None = None,
    profile_mode: str = "launch",
) -> dict[str, Any]:
    """Launch a dedicated server, collect PMC, then stop it.

    This function deliberately owns a separate process so the normal
    torch-profiler/TraceLens server never competes with rocprofiler-sdk.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "pmc_roofline_server.log"
    env = os.environ.copy()
    preload_env = PMCProfiler.server_profiling_env(env)
    if "librocprofiler-register.so" in preload_env.get("LD_PRELOAD", ""):
        env = preload_env
    env.update({str(k): str(v) for k, v in (env_overrides or {}).items()})
    mode = (profile_mode or "launch").lower()

    try:
        profiler = PMCProfiler(session_dir)
        if mode == "attach":
            if "librocprofiler-register.so" not in env.get("LD_PRELOAD", ""):
                return {"status": "skipped", "reason": "librocprofiler-register.so unavailable"}
            with log_path.open("w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.Popen(
                    server_cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    **_process_group_kwargs(),
                )
            try:
                if not _wait_for_health(health_url, proc, startup_timeout_s):
                    return {
                        "status": "failed",
                        "reason": "dedicated server did not become healthy",
                        "profile_mode": mode,
                        "server_log": str(log_path),
                        "server_returncode": proc.poll(),
                    }
                pmc_results = profiler.profile_attach(
                    proc.pid,
                    duration_ms=duration_ms,
                    benchmark_cmd=benchmark_cmd,
                    benchmark_env=os.environ.copy(),
                )
            finally:
                _terminate_process_group(proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _kill_process_group(proc)
        else:
            pmc_results = profiler.profile_launch(
                server_cmd=server_cmd,
                health_url=health_url,
                log_path=log_path,
                duration_ms=duration_ms,
                benchmark_cmd=benchmark_cmd,
                benchmark_env=os.environ.copy(),
                startup_timeout_s=startup_timeout_s,
                server_env=env,
            )
        if not pmc_results:
            return {
                "status": "skipped",
                "reason": "no PMC or kernel trace results",
                "profile_mode": mode,
                "server_log": str(log_path),
            }

        pmc_summary_path = profiler.save_results(pmc_results, tag="pmc")
        total_ns = sum(result.duration_ns for result in pmc_results)
        pmc_dicts = []
        for result in pmc_results:
            gpu_pct = getattr(result, "_gpu_time_pct", None)
            if gpu_pct is None:
                gpu_pct = result.duration_ns / total_ns * 100.0 if total_ns else 0.0
            pmc_dicts.append(result.to_roofline_dict(gpu_pct=float(gpu_pct or 0.0)))

        roofline_results = RooflineAnalyzer().analyze_kernels(pmc_dicts, precision=precision)
        roofline_path = session_dir / "profiles" / "roofline.json"
        roofline_path.write_text(
            json.dumps([result.to_dict() for result in roofline_results], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        kernel_breakdown_path = session_dir / "profiles" / "kernel_breakdown.json"
        _write_kernel_breakdown(
            kernel_breakdown_path,
            pmc_results=pmc_results,
            roofline_results=roofline_results,
        )
        return {
            "status": "ok",
            "profile_mode": mode,
            "server_log": str(log_path),
            "pmc_summary_path": str(pmc_summary_path),
            "roofline_path": str(roofline_path),
            "kernel_breakdown_path": str(kernel_breakdown_path),
            "roofline_kernel_count": len(roofline_results),
        }
    except FileNotFoundError as exc:
        return {"status": "skipped", "reason": "rocprofv3 unavailable", "error": str(exc), "server_log": str(log_path)}
    except subprocess.TimeoutExpired as exc:
        return {"status": "skipped", "reason": "rocprofv3 timed out", "error": str(exc), "server_log": str(log_path)}
