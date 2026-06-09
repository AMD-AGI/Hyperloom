#!/usr/bin/env python3
"""Run rocprof-compute and emit per-kernel roofline JSON for Hyperloom."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


FP_METRICS = (
    "VALU FLOPs (F16)",
    "VALU FLOPs (F32)",
    "VALU FLOPs (F64)",
    "MFMA FLOPs (F64)",
    "MFMA FLOPs (F32)",
    "MFMA FLOPs (F16)",
    "MFMA FLOPs (BF16)",
    "MFMA FLOPs (F8)",
)
SATURATION_PCT = 60.0


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bound_type(bound: str) -> str:
    lower = (bound or "").lower()
    if lower.startswith("memory-bound"):
        return "memory"
    if lower.startswith("compute-bound"):
        return "compute"
    if lower.startswith("latency-bound"):
        return "latency"
    return "unknown"


def _resolve_rocprof_compute() -> str | None:
    configured = os.environ.get("HYPERLOOM_ROCPROF_COMPUTE_PATH", "").strip()
    candidates = [configured, shutil.which("rocprof-compute"), "/opt/rocm/bin/rocprof-compute"]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        resolved = shutil.which(raw)
        if resolved:
            return resolved
    return None


def _check_rocprof_compute() -> str | None:
    tool = _resolve_rocprof_compute()
    if not tool:
        return None
    proc = subprocess.run(
        [tool, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    match = re.search(r"rocprofiler-compute\s+version:\s*([0-9]+\.[0-9]+\.[0-9]+)", proc.stdout)
    return match.group(1) if match else "unknown"


class RocprofRooflineAnalyzer:
    """Small parser around rocprof-compute section 4 roofline output."""

    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path or tempfile.mkdtemp(prefix="rocprof_roofline_")).resolve()
        self.content = ""

    def parse_roofline_blocks(self) -> list[tuple[str, dict[str, tuple[float, float, str]]]]:
        blocks: list[tuple[str, dict[str, tuple[float, float, str]]]] = []
        current_name = "Unknown"
        rates: dict[str, tuple[float, float, str]] = {}
        header_re = re.compile(r"Kernel\s+\d+:\s*(.+?)\s*\(\d+(?:\.\d+)?%\)\s*$")
        in_section = False

        for line in self.content.splitlines():
            header = header_re.search(line)
            if header:
                current_name = header.group(1).strip()
                continue
            if "4.1 Roofline Rate Metrics" in line:
                in_section = True
                rates = {}
                continue
            if in_section and "╘═" in line:
                in_section = False
                if rates:
                    blocks.append((current_name, rates))
                continue
            if "4.3 Roofline Plot" in line:
                break
            if in_section and "│" in line and "4.1." in line:
                parts = [p.strip() for p in line.split("│")]
                if len(parts) < 6:
                    continue
                value = _safe_float(parts[3])
                peak = _safe_float(parts[5])
                if value is None or peak is None:
                    continue
                rates[parts[2]] = (value, peak, parts[4])
        if in_section and rates:
            blocks.append((current_name, rates))
        return blocks

    def parse_roofline_ai(self) -> list[dict[str, tuple[float, str]]]:
        out: list[dict[str, tuple[float, str]]] = []
        metrics: dict[str, tuple[float, str]] = {}
        in_section = False
        for line in self.content.splitlines():
            if "4.2 Roofline AI Plot Points" in line:
                in_section = True
                metrics = {}
                continue
            if in_section and "╘═" in line:
                in_section = False
                if metrics:
                    out.append(metrics)
                continue
            if "4.3 Roofline Plot" in line:
                break
            if in_section and "│" in line and "4.2." in line:
                parts = [p.strip() for p in line.split("│")]
                if len(parts) < 5:
                    continue
                value = _safe_float(parts[3])
                if value is None:
                    continue
                metric_name = parts[2]
                unit = parts[4]
                if "Performance" in metric_name and "Gflop" in unit:
                    value = value / 1000.0
                    unit = "TFLOPS"
                    metric_name = "Performance (TFLOPs)"
                if "AI" in metric_name or "Performance" in metric_name:
                    metrics[metric_name] = (value, unit)
        if in_section and metrics:
            out.append(metrics)
        return out

    def parse_real_hbm_peak(self) -> float | None:
        for line in self.content.splitlines():
            if "│" in line and "17.1.5" in line:
                parts = [p.strip() for p in line.split("│")]
                if len(parts) >= 4:
                    value = _safe_float(parts[3])
                    if value is not None:
                        return value
        for line in self.content.splitlines():
            if "│" in line and ("2.1.23" in line or "2.1.24" in line):
                parts = [p.strip() for p in line.split("│")]
                if len(parts) >= 6:
                    value = _safe_float(parts[5])
                    if value is not None:
                        return value
        return None

    def compute_efficiency(
        self,
        rates: dict[str, tuple[float, float, str]],
        ai_metrics: dict[str, tuple[float, str]],
        real_hbm_peak: float | None,
    ) -> dict[str, Any]:
        hbm_actual = hbm_emp_peak = None
        if "HBM Bandwidth" in rates:
            hbm_actual, hbm_emp_peak, _ = rates["HBM Bandwidth"]

        best_actual = 0.0
        compute_peak = 0.0
        compute_metric = None
        for metric in FP_METRICS:
            if metric not in rates:
                continue
            actual, peak, _ = rates[metric]
            if actual > best_actual:
                best_actual = actual
                compute_peak = peak
                compute_metric = metric

        perf_gflops = None
        if "Performance (TFLOPs)" in ai_metrics:
            perf_gflops = ai_metrics["Performance (TFLOPs)"][0] * 1000.0
        elif "Performance (GFLOPs)" in ai_metrics:
            perf_gflops = ai_metrics["Performance (GFLOPs)"][0]
        ai_hbm = ai_metrics.get("AI HBM", (0.0, ""))[0]

        compute_util = (best_actual / compute_peak * 100.0) if compute_peak else None
        hbm_util_emp = (hbm_actual / hbm_emp_peak * 100.0) if hbm_actual and hbm_emp_peak else None
        hbm_peak_for_bound = real_hbm_peak or hbm_emp_peak
        hbm_util_real = (hbm_actual / hbm_peak_for_bound * 100.0) if hbm_actual and hbm_peak_for_bound else None

        roofline_eff_emp = None
        if hbm_emp_peak and compute_peak and ai_hbm > 0 and perf_gflops is not None:
            ceiling = min(compute_peak, ai_hbm * hbm_emp_peak)
            roofline_eff_emp = (perf_gflops / ceiling * 100.0) if ceiling > 0 else None

        roofline_eff_real = None
        if hbm_peak_for_bound and compute_peak and ai_hbm > 0 and perf_gflops is not None:
            ceiling = min(compute_peak, ai_hbm * hbm_peak_for_bound)
            roofline_eff_real = (perf_gflops / ceiling * 100.0) if ceiling > 0 else None

        ai_ridge = (compute_peak / hbm_peak_for_bound) if compute_peak and hbm_peak_for_bound else None
        if not compute_peak or ai_hbm <= 0:
            bound = "latency-bound (no FP work; integer / memcpy kernel)"
        elif compute_util is not None and compute_util >= SATURATION_PCT:
            bound = f"compute-bound ({compute_metric} ~{compute_util:.0f}% of peak)"
        elif hbm_util_real is not None and hbm_util_real >= SATURATION_PCT:
            bound = "memory-bound (HBM bandwidth near peak)"
        else:
            bound = "latency-bound (neither compute nor HBM bandwidth near saturation)"

        return {
            "bound": bound,
            "bound_type": _bound_type(bound),
            "compute_roof_metric": compute_metric,
            "compute_peak_gflops": compute_peak or None,
            "compute_utilization_pct": compute_util,
            "ai_hbm": ai_hbm,
            "ai_ridge": ai_ridge,
            "perf_gflops": perf_gflops,
            "hbm_actual_gbps": hbm_actual,
            "hbm_emp_peak_gbps": hbm_emp_peak,
            "hbm_real_peak_gbps": hbm_peak_for_bound,
            "hbm_util_emp_pct": hbm_util_emp,
            "bandwidth_utilization_pct": hbm_util_real,
            "roofline_efficiency_pct": roofline_eff_emp,
            "roofline_efficiency_basis": "empirical_peak",
            "roofline_efficiency_real_pct": roofline_eff_real,
        }

    def analyze_structured(self) -> dict[str, Any]:
        blocks = self.parse_roofline_blocks()
        ai_list = self.parse_roofline_ai()
        real_hbm_peak = self.parse_real_hbm_peak()
        rows: list[dict[str, Any]] = []
        for idx, (name, rates) in enumerate(blocks):
            ai_metrics = ai_list[idx] if idx < len(ai_list) else {}
            eff = self.compute_efficiency(rates, ai_metrics, real_hbm_peak)
            rows.append({
                "name": name,
                "status": "matched",
                "matched_kernel_name": name,
                "bottleneck": eff["bound_type"],
                "arithmetic_intensity": eff["ai_hbm"],
                "recommended_actions": recommended_actions(eff["bound_type"]),
                "rocprof_roofline": eff,
            })
        return {
            "schema_version": 1,
            "source": "rocprof_roofline",
            "results": rows,
            "real_hbm_peak": real_hbm_peak,
        }

    def run(
        self,
        *,
        workdir: str,
        cmd: str,
        target_kernel: str = "",
        analyze_blocks: str = "0 1 2 4 7 10 11 16 17",
        timeout_sec: int = 21600,
    ) -> tuple[bool, str | None]:
        tool = _resolve_rocprof_compute()
        if tool is None:
            return False, "rocprof-compute is not installed or not on PATH"
        if _check_rocprof_compute() is None:
            return False, "rocprof-compute is not installed or not on PATH"
        self.output_path.mkdir(parents=True, exist_ok=True)
        name = Path(workdir).name or "rocprof_roofline"
        kernel_filter = f" -k {shlex.quote(target_kernel)}" if target_kernel else ""
        profile_cmd = (
            f"{shlex.quote(tool)} profile -n {name}{kernel_filter} "
            f"--path {shlex.quote(str(self.output_path))} -- {cmd}"
        )
        proc = subprocess.run(
            [profile_cmd],
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            return False, proc.stdout.strip()
        analyze_cmd = f"{shlex.quote(tool)} analyze -p {shlex.quote(str(self.output_path))} -b {analyze_blocks}"
        proc = subprocess.run(
            [analyze_cmd],
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            return False, proc.stdout.strip()
        self.content = proc.stdout
        return True, None


def recommended_actions(bound_type: str) -> list[str]:
    if bound_type == "memory":
        return ["Improve memory coalescing/locality", "Increase arithmetic intensity", "Use LDS/cache tiling when applicable"]
    if bound_type == "compute":
        return ["Tune MFMA/tile shape", "Reduce instruction overhead", "Improve occupancy if resources allow"]
    if bound_type == "latency":
        return ["Increase parallelism/occupancy", "Reduce dependency and launch overhead", "Batch small work when possible"]
    return []


def build_text_report(payload: dict[str, Any]) -> str:
    lines = ["Below is the rocprof-compute roofline information of the kernel:"]
    for row in payload.get("results", []):
        roof = row.get("rocprof_roofline") or {}
        lines.append("\nkernel function name:")
        lines.append(f"- {row.get('name')}")
        lines.append("ROOFLINE CLASSIFICATION:")
        if roof.get("ai_hbm") is not None:
            lines.append(f"- AI (HBM): {roof['ai_hbm']} Flops/byte")
        if roof.get("perf_gflops") is not None:
            lines.append(f"- Performance: {roof['perf_gflops']:.2f} Gflop/s")
        if roof.get("bandwidth_utilization_pct") is not None:
            lines.append(f"- HBM util (real peak): {roof['bandwidth_utilization_pct']:.2f}%")
        if roof.get("roofline_efficiency_pct") is not None:
            lines.append(f"- Roofline Eff (empirical peak): {roof['roofline_efficiency_pct']:.1f}%")
        if roof.get("roofline_efficiency_real_pct") is not None:
            lines.append(f"- Roofline Eff (real peak): {roof['roofline_efficiency_real_pct']:.1f}%")
        if roof.get("ai_ridge") is not None:
            lines.append(f"- AI Ridge: {roof['ai_ridge']:.2f} Flops/byte")
        lines.append(f"- Kernel bound: {roof.get('bound', 'unknown')}")
    return "\n".join(lines) + "\n"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _kernel_name_matches(row: dict[str, Any], target_kernel: str) -> bool:
    if not target_kernel:
        return True
    target = target_kernel.strip()
    names = (
        str(row.get("matched_kernel_name") or "").strip(),
        str(row.get("name") or "").strip(),
    )
    return any(name == target for name in names)


def _project_payload_to_row(payload: dict[str, Any], target_kernel: str = "") -> dict[str, Any]:
    """Project the row that matches ``target_kernel`` into ``kernel_roofline.json``."""
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        return {"status": payload.get("status", "failed") if isinstance(payload, dict) else "failed"}
    first = None
    if target_kernel:
        for row in rows:
            if isinstance(row, dict) and _kernel_name_matches(row, target_kernel):
                first = row
                break
        if first is None:
            return {
                "status": "skipped",
                "reason": "target_kernel_not_matched",
                "target_kernel": target_kernel,
                "matched_kernel_names": [
                    str(row.get("matched_kernel_name") or row.get("name") or "")
                    for row in rows
                    if isinstance(row, dict)
                ],
            }
    else:
        first = rows[0] if isinstance(rows[0], dict) else {}
    roof = dict(first.get("rocprof_roofline") or {})
    roof.update({
        "status": first.get("status") or payload.get("status") or "matched",
        "matched_kernel_name": first.get("matched_kernel_name") or first.get("name"),
        "target_kernel": target_kernel,
    })
    return roof


def _generate_harness_for_candidate(
    candidate: dict[str, Any],
    *,
    out_dir: Path,
    log_fn: Any,
) -> tuple[str, str | None]:
    """Best-effort harness generation for a TraceLens hot-kernel candidate.

    Returns ``(test_command, error)``. ``test_command`` is empty when no
    benchmark file resolves; the caller should mark the row as skipped.
    """
    bench_files = candidate.get("benchmark_files") or []
    if not isinstance(bench_files, list) or not bench_files:
        return "", "no_benchmark_files"
    bench_py = ""
    for bf in bench_files:
        if isinstance(bf, str) and bf.endswith(".py") and Path(bf).is_file():
            bench_py = bf
            break
    if not bench_py:
        return "", "no_resolvable_benchmark_file"

    try:
        tools_dir = str(Path(__file__).resolve().parent)
        import sys
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from harness_generator import maybe_generate_harness  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - import safety net
        return "", f"harness_generator_import_error: {type(exc).__name__}"

    try:
        hr = maybe_generate_harness(
            benchmark_file=bench_py,
            candidate=candidate,
            source_file=str(candidate.get("source_file") or ""),
            out_dir=out_dir,
            kernel_repo=str(candidate.get("kernel_repo") or ""),
            log_fn=log_fn,
        )
    except Exception as exc:
        return "", f"harness_generator_error: {type(exc).__name__}: {exc}"

    if hr is None or not getattr(hr, "test_command", ""):
        return "", "harness_unavailable"

    cmd = hr.test_command
    # Profile mode is the right form for rocprof; the auto-generated harness
    # ships with ``--correctness`` for GEAK SaveAndTest by default.
    if "--correctness" in cmd:
        cmd = cmd.replace("--correctness", "--profile", 1)
    return cmd, None


def _profile_workdir(candidate: dict[str, Any], fallback: Path) -> Path:
    for raw in (candidate.get("kernel_repo"), candidate.get("source_file")):
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return path
    return fallback


def enrich_kernel_roofline_sidecar(
    *,
    sidecar_path: str | Path,
    candidates_path: str | Path,
    workdir: str | Path | None = None,
    timeout_sec_per_kernel: int = 1800,
    log_fn: Any = None,
) -> dict[str, Any]:
    """Run ``rocprof_roofline`` on every reusable hot-kernel candidate and
    write the result back into ``reports/kernel_roofline.json``.

    Best-effort:

    * Skips candidates without ``reusable_native_kernel`` (vendor / aten / no source).
    * Skips candidates without ``benchmark_files`` and tags them
      ``status='skipped' reason='no_benchmark_files'`` so dashboards know
      this row was considered (vs. silently ``null``).
    * Per-kernel rocprof timeout / failure marks the row ``status='failed'``
      but never aborts enrichment of remaining rows.
    * Returns a small summary suitable for caller logging.
    """
    sidecar_p = Path(sidecar_path).expanduser()
    cand_p = Path(candidates_path).expanduser()

    summary: dict[str, Any] = {"matched": 0, "skipped": 0, "failed": 0, "rows": 0}

    if not sidecar_p.is_file() or not cand_p.is_file():
        summary["status"] = "missing_inputs"
        return summary

    try:
        sidecar = json.loads(sidecar_p.read_text(encoding="utf-8"))
        cands_raw = json.loads(cand_p.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["status"] = f"json_load_error: {type(exc).__name__}"
        return summary

    rows = sidecar.get("kernels") if isinstance(sidecar, dict) else None
    if not isinstance(rows, list):
        summary["status"] = "sidecar_missing_kernels"
        return summary

    cands = []
    if isinstance(cands_raw, dict):
        cands = cands_raw.get("hot_kernels") or cands_raw.get("kernel_candidates") or []
    if not isinstance(cands, list):
        cands = []

    by_id_cand = {str(c.get("kernel_id") or ""): c for c in cands if isinstance(c, dict)}
    by_id_row = {str(r.get("kernel_id") or ""): r for r in rows if isinstance(r, dict)}

    fallback_workdir = Path(workdir).expanduser() if workdir else sidecar_p.parent

    def _log(msg: str) -> None:
        if callable(log_fn):
            try:
                log_fn(msg)
            except Exception:  # pragma: no cover - log sink failures
                pass

    rocprof_present = _check_rocprof_compute() is not None
    if not rocprof_present:
        _log("[rocprof_enrich] rocprof-compute not available; marking all rows skipped")

    for kid, cand in by_id_cand.items():
        row = by_id_row.get(kid)
        if row is None:
            continue
        summary["rows"] += 1
        # Reusable gate: aten / vendor / unresolved sources are skipped because
        # GEAK can't optimize them and the per-kernel rocprof would not feed any
        # downstream KEEP decision either.
        if not cand.get("reusable_native_kernel"):
            row["rocprof_roofline"] = {
                "before_kernel_opt": {"status": "skipped", "reason": "not_reusable_native_kernel"},
                "after_kernel_opt": None,
            }
            summary["skipped"] += 1
            continue
        if not rocprof_present:
            row["rocprof_roofline"] = {
                "before_kernel_opt": {"status": "skipped", "reason": "rocprof_compute_unavailable"},
                "after_kernel_opt": None,
            }
            summary["skipped"] += 1
            continue
        out_dir = (sidecar_p.parent.parent / "kernel-agent" / "rocprof_roofline" / kid)
        out_dir.mkdir(parents=True, exist_ok=True)
        test_command, harness_err = _generate_harness_for_candidate(
            cand, out_dir=out_dir, log_fn=lambda m: _log(f"[rocprof_enrich:{kid}] {m}"),
        )
        if not test_command:
            row["rocprof_roofline"] = {
                "before_kernel_opt": {"status": "skipped", "reason": harness_err or "no_test_command"},
                "after_kernel_opt": None,
            }
            summary["skipped"] += 1
            continue

        run_workdir = _profile_workdir(cand, fallback_workdir)
        analyzer = RocprofRooflineAnalyzer()
        target_kernel = str(row.get("name") or cand.get("name") or "").strip()
        _log(f"[rocprof_enrich:{kid}] running rocprof-compute (cmd={test_command[:120]}...)")
        try:
            ok, error = analyzer.run(
                workdir=str(run_workdir),
                cmd=test_command,
                target_kernel=target_kernel,
                timeout_sec=timeout_sec_per_kernel,
            )
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        out_json = out_dir / "rocprof_roofline.json"
        out_txt = out_dir / "rocprof_roofline.txt"
        if not ok:
            payload = {
                "schema_version": 1,
                "source": "rocprof_roofline",
                "status": "failed",
                "error": error or "unknown",
                "results": [],
            }
            _atomic_write_json(out_json, payload)
            out_txt.write_text(f"rocprof_roofline failed: {error}\n", encoding="utf-8")
            row["rocprof_roofline"] = {
                "before_kernel_opt": {
                    "status": "failed",
                    "reason": (error or "unknown")[:240],
                    "report_path": str(out_txt),
                    "json_path": str(out_json),
                },
                "after_kernel_opt": None,
            }
            summary["failed"] += 1
            continue

        payload = analyzer.analyze_structured()
        payload.update({
            "status": "ok",
            "profiling_cmd": test_command,
            "target_kernel": target_kernel,
            "rocprof_output_path": str(analyzer.output_path),
        })
        _atomic_write_json(out_json, payload)
        out_txt.write_text(analyzer.content + "\n" + build_text_report(payload), encoding="utf-8")

        row_payload = _project_payload_to_row(payload, target_kernel=target_kernel)
        row_payload.setdefault("status", "matched")
        row_payload["report_path"] = str(out_txt)
        row_payload["json_path"] = str(out_json)
        row["rocprof_roofline"] = {
            "before_kernel_opt": row_payload,
            "after_kernel_opt": None,
        }
        # Mirror key metrics onto the row level for fast dashboard rendering.
        if row_payload.get("bound_type"):
            row.setdefault("bottleneck", row.get("bottleneck") or row_payload["bound_type"])
            row["bound_type"] = row.get("bound_type") or row_payload["bound_type"]
        if row_payload.get("ai_hbm") is not None:
            row.setdefault("arithmetic_intensity", row_payload["ai_hbm"])
        if row_payload.get("compute_utilization_pct") is not None:
            row["compute_utilization_pct"] = row_payload["compute_utilization_pct"]
        if row_payload.get("bandwidth_utilization_pct") is not None:
            row["bandwidth_utilization_pct"] = row_payload["bandwidth_utilization_pct"]
        if row_payload.get("roofline_efficiency_pct") is not None:
            row["efficiency_percent"] = row_payload["roofline_efficiency_pct"]
        summary["matched"] += 1

    if summary["matched"]:
        sidecar["source"] = "tracelens_analysis+rocprof_roofline"
    _atomic_write_json(sidecar_p, sidecar)
    summary["status"] = "ok"
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--target-kernel", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-txt", required=True)
    parser.add_argument("--raw-txt", default="")
    parser.add_argument("--timeout-sec", type=int, default=21600)
    args = parser.parse_args(argv)

    analyzer = RocprofRooflineAnalyzer()
    success, error = analyzer.run(
        workdir=args.workdir,
        cmd=args.cmd,
        target_kernel=args.target_kernel,
        timeout_sec=args.timeout_sec,
    )
    out_json = Path(args.out_json)
    out_txt = Path(args.out_txt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if not success:
        payload = {
            "schema_version": 1,
            "source": "rocprof_roofline",
            "status": "failed",
            "error": error,
            "results": [],
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_txt.write_text(f"rocprof_roofline failed: {error}\n", encoding="utf-8")
        return 1

    payload = analyzer.analyze_structured()
    payload.update({
        "status": "ok",
        "profiling_cmd": args.cmd,
        "target_kernel": args.target_kernel,
        "rocprof_output_path": str(analyzer.output_path),
    })
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_txt.write_text(analyzer.content + "\n" + build_text_report(payload), encoding="utf-8")
    if args.raw_txt:
        Path(args.raw_txt).write_text(analyzer.content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
