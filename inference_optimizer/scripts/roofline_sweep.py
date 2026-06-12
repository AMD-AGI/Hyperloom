#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline concurrency sweep: baseline vs optimized vs theoretical peak.

After an ``optimize`` session finishes, this driver replays a fixed
``baseline`` sglang launch and the session's ``current_best`` launch
across a list of concurrencies, calls the *same* ``roofline_ceiling``
formula used by the orchestrator for each concurrency, and emits a
``csv`` + dual-panel ``svg`` (throughput curves + MBU bars) so the
operator can see how close the optimized stack pushes against the
hardware roof.

Usage::

    python -m inference_optimizer.scripts.roofline_sweep \\
        --session-dir /shared_nfs/.../Qwen-Qwen3-30B-A3B-Base/20260528T134736Z \\
        --concs 1,2,4,8,16,32,64,128 \\
        --isl 1024 --osl 1024 --tp 1 --gpu-type mi355x \\
        --port 30100 \\
        --output-prefix runs/manual_roofline_bench/qwen3_30b_a3b_post_e2e_sweep

Per ``conc`` the script reuses one sglang server (no restart between
concurrencies). OOM rows are recorded as ``status=OOM`` in the csv and
left empty in the svg.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Hyperloom roofline formula (single source of truth for ceiling).
from inference_optimizer.orchestrator.roofline_ceiling import (
    HW_SPECS,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
)


# ---------------------------------------------------------------------------
# State / template extraction
# ---------------------------------------------------------------------------
@dataclass
class LaunchTemplate:
    """sglang launch parameter bundle (one per config: baseline / optimized)."""

    label: str
    extra_server_args: str
    extra_envs: dict[str, str] = field(default_factory=dict)


def load_session_state(session_dir: Path) -> dict[str, Any]:
    """Load a session's ``state.json``.

    Args:
        session_dir: Hyperloom session directory.

    Returns:
        The parsed state mapping.

    Raises:
        SystemExit: If ``state.json`` is missing.
    """
    state_path = session_dir / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"state.json not found at {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def extract_templates(state: dict[str, Any]) -> tuple[LaunchTemplate, LaunchTemplate]:
    """Build the (baseline, optimized) launch templates from state.json.

    Baseline = vanilla sglang (no extra flags / envs). Optimized =
    state.current_best.extra_server_args + extra_envs. The optimized
    template is the hard KPI of this sweep — when empty, the caller
    should refuse to run.
    """
    cb = state.get("current_best") or {}
    opt_args = (cb.get("extra_server_args") or "").strip()
    opt_envs = cb.get("extra_envs") or {}
    if not opt_args and not opt_envs:
        raise SystemExit(
            "current_best.extra_server_args and extra_envs are both empty — "
            "the e2e session did not accept any optimization. Re-run the "
            "session until at least one candidate is accepted."
        )
    return (
        LaunchTemplate(label="baseline", extra_server_args="", extra_envs={}),
        LaunchTemplate(label="optimized", extra_server_args=opt_args, extra_envs=dict(opt_envs)),
    )


# ---------------------------------------------------------------------------
# sglang server lifecycle
# ---------------------------------------------------------------------------
class SglangServer:
    """Spawn ``python -m sglang.launch_server`` in a process group; kill on exit."""

    DEFAULT_FLAGS = ["--mem-fraction-static=0.8", "--disable-radix-cache", "--trust-remote-code"]

    def __init__(
        self,
        model_path: str,
        tp: int,
        port: int,
        extra_args: str,
        extra_envs: dict[str, str],
        log_path: Path,
        gpu_id: int,
        ready_timeout_sec: int = 900,
    ) -> None:
        """Initialize the sglang server wrapper.

        Args:
            model_path: Path to the model to serve.
            tp: Tensor-parallel size.
            port: Port to bind the server on.
            extra_args: Extra server CLI args (space-separated).
            extra_envs: Extra environment variables for the server process.
            log_path: File to capture server stdout/stderr.
            gpu_id: Physical GPU id to pin via ``ROCR_VISIBLE_DEVICES``.
            ready_timeout_sec: Max seconds to wait for the server to be ready.
        """
        self.model_path = model_path
        self.tp = tp
        self.port = port
        self.extra_args = extra_args
        self.extra_envs = extra_envs
        self.log_path = log_path
        self.gpu_id = gpu_id
        self.ready_timeout_sec = ready_timeout_sec
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "SglangServer":
        """Launch the server process and block until it is ready.

        Returns:
            This server instance.

        Raises:
            RuntimeError: If the server dies or never becomes ready.
        """
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            f"--model-path={self.model_path}",
            "--host=0.0.0.0",
            f"--port={self.port}",
            f"--tensor-parallel-size={self.tp}",
            *self.DEFAULT_FLAGS,
        ]
        if self.extra_args.strip():
            cmd.extend(self.extra_args.strip().split())

        env = os.environ.copy()
        # ROCm GPU pinning: set ONLY ROCR_VISIBLE_DEVICES. Do NOT also set
        # CUDA/HIP_VISIBLE_DEVICES — ROCR filters to expose the target GPU as
        # logical index 0, so a stacked CUDA_VISIBLE_DEVICES=<gpu_id> would
        # then select a non-existent index and torch reports "No accelerator".
        # Pop the others so only ROCR is in effect.
        env["ROCR_VISIBLE_DEVICES"] = str(self.gpu_id)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        # MI355X defaults the orchestrator's baseline_executor also injects.
        env.setdefault("SGLANG_USE_AITER", "1")
        env.setdefault("SGLANG_AITER_MLA_PERSIST", "1")
        env.setdefault("HSA_NO_SCRATCH_RECLAIM", "1")
        env.update({k: str(v) for k, v in self.extra_envs.items()})

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        print(f"[server] launching: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        """Poll the health endpoint until the server is ready or times out.

        Raises:
            RuntimeError: If the server process dies or does not become ready
                within ``ready_timeout_sec``.
        """
        import urllib.request
        deadline = time.time() + self.ready_timeout_sec
        url = f"http://127.0.0.1:{self.port}/health_generate"
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(
                    f"sglang server died during startup; see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status < 500:
                        print(f"[server] ready on port {self.port}", flush=True)
                        return
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(
            f"sglang server did not become ready in {self.ready_timeout_sec}s"
        )

    def __exit__(self, *_exc: Any) -> None:
        """Terminate the server process group on context exit.

        Args:
            *_exc: Exception info from the ``with`` block (unused).
        """
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=10)
        except ProcessLookupError:
            pass
        print(f"[server] stopped (port {self.port})", flush=True)

    @property
    def base_url(self) -> str:
        """Return the server's local base URL."""
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# bench_serving
# ---------------------------------------------------------------------------
def run_bench(
    *, base_url: str, model_path: str, conc: int, isl: int, osl: int,
    num_prompts: int, dataset: str, output_dir: Path, port: int,
) -> dict[str, Any]:
    """Invoke ``sglang.bench_serving`` once; return parsed metrics dict.

    Returns ``{}`` on subprocess failure (caller can mark as OOM/ERROR).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"bench_conc{conc}.jsonl"
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--model", model_path,
        "--dataset-name", dataset,
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(conc),
        "--output-file", str(out_file),
        "--disable-tqdm",
    ]
    print(f"[bench] conc={conc} num_prompts={num_prompts} -> {out_file}", flush=True)
    try:
        subprocess.run(cmd, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[bench] FAILED conc={conc}: {exc}", flush=True)
        return {}

    # bench_serving appends one line per run (jsonl); pick the last.
    if not out_file.is_file():
        return {}
    last = out_file.read_text(encoding="utf-8").strip().splitlines()
    if not last:
        return {}
    try:
        return json.loads(last[-1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Ceiling per concurrency
# ---------------------------------------------------------------------------
def compute_ceiling(
    *, model_meta: Any, gpu_type: str, num_gpus: int,
    conc: int, isl: int, osl: int,
) -> float:
    """Compute the theoretical peak output throughput for one concurrency.

    Thin wrapper over :func:`compute_theoretical_peak_output_tok_per_sec` that
    unpacks the model metadata fields.

    Args:
        model_meta: Loaded model metadata (weights, layers, KV shape, ...).
        gpu_type: GPU type key into the hardware specs.
        num_gpus: Number of GPUs (tensor-parallel size).
        conc: Concurrency level.
        isl: Input sequence length.
        osl: Output sequence length.

    Returns:
        The theoretical peak output tokens/second.
    """
    return compute_theoretical_peak_output_tok_per_sec(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        weight_bytes=model_meta.weight_bytes,
        active_weight_bytes=model_meta.active_weight_bytes,
        num_experts=getattr(model_meta, "num_experts", 0),
        experts_per_tok=getattr(model_meta, "experts_per_tok", 0),
        expert_weight_bytes=getattr(model_meta, "expert_weight_bytes", 0),
        num_layers=model_meta.num_layers,
        num_kv_heads=model_meta.num_kv_heads,
        head_dim=model_meta.head_dim,
        kv_dtype_bytes=model_meta.weight_dtype_bytes,
        isl=isl,
        osl=osl,
        concurrency=conc,
    )


# ---------------------------------------------------------------------------
# Sweep one template across all concurrencies (single server reuse)
# ---------------------------------------------------------------------------
def sweep_one_template(
    *, tmpl: LaunchTemplate, model_path: str, model_meta: Any,
    gpu_type: str, tp: int, gpu_id: int, port: int, isl: int, osl: int,
    concs: list[int], dataset: str, num_prompts_factor: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Sweep a launch template across all concurrencies on one reused server.

    Args:
        tmpl: Launch template (baseline or optimized).
        model_path: Path to the model to serve.
        model_meta: Loaded model metadata used for ceiling computation.
        gpu_type: GPU type key.
        tp: Tensor-parallel size.
        gpu_id: Physical GPU id to pin.
        port: Server port.
        isl: Input sequence length.
        osl: Output sequence length.
        concs: Concurrency levels to benchmark.
        dataset: Benchmark dataset name.
        num_prompts_factor: ``num_prompts = max(conc * factor, 16)``.
        output_dir: Directory for server/bench logs.

    Returns:
        One result row per concurrency with measured throughput, ceiling, MBU,
        and status.
    """
    rows: list[dict[str, Any]] = []
    server_log = output_dir / f"server_{tmpl.label}.log"
    bench_out_dir = output_dir / f"bench_{tmpl.label}"
    with SglangServer(
        model_path=model_path, tp=tp, port=port,
        extra_args=tmpl.extra_server_args, extra_envs=tmpl.extra_envs,
        log_path=server_log, gpu_id=gpu_id,
    ) as srv:
        for conc in concs:
            num_prompts = max(conc * num_prompts_factor, 16)
            metrics = run_bench(
                base_url=srv.base_url, model_path=model_path, conc=conc,
                isl=isl, osl=osl, num_prompts=num_prompts, dataset=dataset,
                output_dir=bench_out_dir, port=port,
            )
            ceiling = compute_ceiling(
                model_meta=model_meta, gpu_type=gpu_type, num_gpus=tp,
                conc=conc, isl=isl, osl=osl,
            )
            if not metrics:
                rows.append({
                    "conc": conc, "config": tmpl.label,
                    "measured_tps": None, "ceiling_tps": ceiling,
                    "target_70_tps": ceiling * 0.70, "mbu_pct": None,
                    "status": "FAILED_OR_OOM",
                })
                continue
            measured = float(
                metrics.get("output_throughput")
                or metrics.get("output_token_throughput")
                or 0.0
            )
            rows.append({
                "conc": conc, "config": tmpl.label,
                "measured_tps": measured, "ceiling_tps": ceiling,
                "target_70_tps": ceiling * 0.70,
                "mbu_pct": (measured / ceiling * 100.0) if ceiling > 0 else None,
                "status": "OK",
            })
            print(
                f"[sweep] {tmpl.label} conc={conc:>3} measured={measured:8.1f} "
                f"ceiling={ceiling:8.1f} MBU={(measured / ceiling * 100.0) if ceiling > 0 else 0:5.1f}%",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# Output: csv + svg
# ---------------------------------------------------------------------------
def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """Write sweep result rows to a CSV file.

    Args:
        rows: Result rows from :func:`sweep_one_template`.
        csv_path: Destination CSV path (parents created as needed).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["conc", "config", "measured_tps", "ceiling_tps", "target_70_tps", "mbu_pct", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[out] csv -> {csv_path}", flush=True)


def plot_svg(
    rows: list[dict[str, Any]], svg_path: Path, *,
    model_label: str, gpu_label: str, tp: int, isl: int, osl: int,
    target_ratio: float = 0.70,
) -> None:
    """Render the dual-panel throughput/MBU roofline chart as SVG.

    Args:
        rows: Sweep result rows.
        svg_path: Destination SVG path.
        model_label: Model name shown in the title.
        gpu_label: GPU label shown in the title.
        tp: Tensor-parallel size shown in the title.
        isl: Input sequence length shown in the title.
        osl: Output sequence length shown in the title.
        target_ratio: Roofline fraction drawn as the target line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concs = sorted({r["conc"] for r in rows})
    def _series(label: str, key: str) -> list[float | None]:
        """Return per-concurrency values for one config/metric.

        Args:
            label: Config label (``baseline`` / ``optimized``).
            key: Metric key to extract from each row.

        Returns:
            A list aligned with ``concs``; ``None`` where a value is missing.
        """
        out: list[float | None] = []
        for c in concs:
            row = next((r for r in rows if r["conc"] == c and r["config"] == label), None)
            out.append(row[key] if row and row.get(key) is not None else None)
        return out

    base_meas = _series("baseline", "measured_tps")
    opt_meas = _series("optimized", "measured_tps")
    ceiling = _series("baseline", "ceiling_tps")  # ceiling is config-independent
    target = [c * target_ratio if c is not None else None for c in ceiling]
    base_mbu = _series("baseline", "mbu_pct")
    opt_mbu = _series("optimized", "mbu_pct")
    # Baseline-only sweep (no accepted optimization) -> hide the optimized
    # series entirely so the chart reads as Measured vs Roofline vs target.
    has_opt = any(v is not None for v in opt_meas)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]},
    )

    # Upper panel: throughput.
    ax1.plot(concs, ceiling, "--", color="#d62728", label="Theoretical Peak (roofline)")
    ax1.plot(concs, target, ":", color="#ff7f0e", label=f"Target = roofline × {target_ratio:.0%}")
    ax1.plot(concs, base_meas, "o-", color="#1f77b4", label="Measured (baseline)")
    if has_opt:
        ax1.plot(concs, opt_meas, "s-", color="#2ca02c", label="Measured (optimized)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(concs)
    ax1.set_xticklabels([str(c) for c in concs])
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("output_throughput (tok/s)")
    ax1.set_title(
        f"Throughput: Measured vs Roofline\n"
        f"{model_label} | {gpu_label} (TP={tp}) | ISL={isl} OSL={osl}",
        fontsize=11,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    # Lower panel: MBU bars.
    import numpy as np
    x = np.arange(len(concs))
    bar_w = 0.38
    base_vals = [v or 0 for v in base_mbu]
    opt_vals = [v or 0 for v in opt_mbu]
    if has_opt:
        ax2.bar(x - bar_w / 2, base_vals, bar_w, label="baseline", color="#1f77b4")
        ax2.bar(x + bar_w / 2, opt_vals, bar_w, label="optimized", color="#2ca02c")
    else:
        ax2.bar(x, base_vals, bar_w * 1.4, label="baseline", color="#1f77b4")
    ax2.axhline(70.0, color="#ff7f0e", linestyle=":", label=f"70% target")
    ax2.axhline(100.0, color="#d62728", linestyle="--", label="100% = HBM ceiling")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in concs])
    ax2.set_xlabel("Concurrency")
    ax2.set_ylabel("MBU = measured / roofline (%)")
    ax2.set_title("Memory Bandwidth Utilization")
    ax2.set_ylim(0, max(110, max(base_vals + opt_vals + [70.0]) * 1.1))
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right", fontsize=9)

    # Annotate measured values on top panel.
    for c, v in zip(concs, base_meas):
        if v: ax1.annotate(f"{v:.0f}", (c, v), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=8, color="#1f77b4")
    for c, v in zip(concs, opt_meas):
        if v: ax1.annotate(f"{v:.0f}", (c, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#2ca02c")
    for c, v in zip(concs, ceiling):
        if v: ax1.annotate(f"{v:.0f}", (c, v), textcoords="offset points",
                            xytext=(8, 0), ha="left", fontsize=8, color="#d62728")

    fig.tight_layout()
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    print(f"[out] svg -> {svg_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the roofline concurrency sweep.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code (``0`` on success).
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session-dir", required=True, type=Path,
                    help="Hyperloom session directory (contains state.json)")
    ap.add_argument("--concs", default="1,2,4,8,16,32,64,128",
                    help="Comma-separated concurrency list")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--gpu-type", choices=sorted(HW_SPECS), default="mi355x")
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--dataset", default="random")
    ap.add_argument("--num-prompts-factor", type=int, default=4,
                    help="num_prompts = max(conc × factor, 16)")
    ap.add_argument("--output-prefix", default=None,
                    help="Base path for csv/svg; default = <session_dir>/roofline_sweep")
    ap.add_argument("--skip-bench", action="store_true",
                    help="Compute ceilings + plot only (uses existing csv)")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Sweep only the vanilla baseline template (use when "
                         "the session accepted no optimization, e.g. all "
                         "explore/kernel_opt REVERTed). Skips the optimized "
                         "line; chart shows Measured vs Roofline vs target.")
    args = ap.parse_args(argv)

    session_dir: Path = args.session_dir.expanduser().resolve()
    state = load_session_state(session_dir)
    model_path = str(state.get("model_path") or "")
    if not model_path or not Path(model_path).is_dir():
        raise SystemExit(f"state.model_path not found: {model_path!r}")

    model_meta = load_model_meta(model_path, precision_hint=str(state.get("precision") or ""))
    if model_meta is None:
        raise SystemExit(f"Failed to load model meta from {model_path}")

    concs = [int(c) for c in args.concs.split(",") if c.strip()]
    out_prefix = (
        Path(args.output_prefix).expanduser().resolve()
        if args.output_prefix
        else session_dir / "roofline_sweep"
    )
    csv_path = out_prefix.with_suffix(".csv")
    svg_path = out_prefix.with_suffix(".svg")

    if args.skip_bench:
        if not csv_path.is_file():
            raise SystemExit(f"--skip-bench but no csv at {csv_path}")
        with csv_path.open("r", encoding="utf-8") as f:
            rows = [
                {**r, "conc": int(r["conc"]),
                 "measured_tps": float(r["measured_tps"]) if r["measured_tps"] else None,
                 "ceiling_tps": float(r["ceiling_tps"]) if r["ceiling_tps"] else None,
                 "target_70_tps": float(r["target_70_tps"]) if r["target_70_tps"] else None,
                 "mbu_pct": float(r["mbu_pct"]) if r["mbu_pct"] else None}
                for r in csv.DictReader(f)
            ]
    else:
        if args.baseline_only:
            templates = [LaunchTemplate(label="baseline", extra_server_args="", extra_envs={})]
            print("[sweep] baseline-only mode (no accepted optimization)")
        else:
            tmpl_base, tmpl_opt = extract_templates(state)
            print(f"[sweep] baseline:  args={tmpl_base.extra_server_args!r} envs={tmpl_base.extra_envs}")
            print(f"[sweep] optimized: args={tmpl_opt.extra_server_args!r} envs={tmpl_opt.extra_envs}")
            templates = [tmpl_base, tmpl_opt]
        rows: list[dict[str, Any]] = []
        for tmpl in templates:
            rows.extend(sweep_one_template(
                tmpl=tmpl, model_path=model_path, model_meta=model_meta,
                gpu_type=args.gpu_type, tp=args.tp, gpu_id=args.gpu_id, port=args.port,
                isl=args.isl, osl=args.osl, concs=concs, dataset=args.dataset,
                num_prompts_factor=args.num_prompts_factor,
                output_dir=out_prefix.parent / "_sweep_logs",
            ))
        write_csv(rows, csv_path)

    plot_svg(
        rows, svg_path,
        model_label=state.get("model_name") or Path(model_path).name,
        gpu_label=args.gpu_type.upper(), tp=args.tp,
        isl=args.isl, osl=args.osl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
