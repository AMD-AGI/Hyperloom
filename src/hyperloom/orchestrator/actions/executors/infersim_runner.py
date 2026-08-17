# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferSim benchmark runner (CLI).

A simulate-only, GPU-free stand-in for ``python -m Magpie -v benchmark ...
--run-mode local``. It accepts the same CLI flags and writes the same
Magpie-compatible workspace + ``benchmark_report.json`` (via
:mod:`bypass_report`), but the numbers come from Infera's ``infersim`` serving
projection instead of a real server + client.

Selected with ``HYPERLOOM_BENCHMARK_BACKEND=infersim``. Because it emits the
same report contract as Magpie/bypass, every executor, collector, and the
optimizer's gain math consume simulated runs unchanged -- so an entire
optimization session can run without a GPU, and real GPU time is spent only on
the final validation. That is the GPU-time reduction projected in the deck.

Lifecycle/server flags (``--phase``, ``--server-lifecycle-*``) are accepted for
drop-in compatibility and ignored: a projection has no server to persist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from . import bypass_report
from . import infersim_bridge

_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})


def run_benchmark(config_path: Path, output_dir: Path) -> int:
    """Project a serving config with InferSim and write a Magpie-style report.

    Args:
        config_path: Materialized benchmark config YAML (the Magpie contract).
        output_dir: Output root for the benchmark workspace.

    Returns:
        Process exit code (0 on success).
    """
    start = time.time()
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return _emit_failure(output_dir, "unknown", "", f"cannot read benchmark config: {exc}", start)

    bench = cfg.get("benchmark") or {}
    framework = str(bench.get("framework") or "sglang").lower()
    model = str(bench.get("model") or "")

    try:
        spec = infersim_bridge.spec_from_benchmark(bench)
        metrics = infersim_bridge.project(spec)
    except infersim_bridge.InfersimBridgeError as exc:
        return _emit_failure(output_dir, framework, model, str(exc), start)
    except Exception as exc:  # noqa: BLE001 - never crash the optimizer loop
        return _emit_failure(output_dir, framework, model, f"unexpected InferSim error: {exc}", start)

    workspace = bypass_report.create_workspace(output_dir, framework)
    _snapshot_config(workspace, cfg)
    raw = infersim_bridge.raw_result_from_metrics(spec, metrics)
    # Persist the raw InferenceX-style result so the workspace matches a real
    # bypass/Magpie run (collectors that rescan raw json stay consistent).
    try:
        (workspace / "inferencex_result.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    except OSError:
        pass

    report = bypass_report.build_report(
        raw,
        framework=framework,
        model=model,
        success=True,
        workspace_dir=str(workspace),
        execution_time=time.time() - start,
        errors=[],
        analysis={
            "backend": "infersim",
            "source": "calibrated" if metrics.calibrated else "simulation",
            "extrapolation": list(metrics.extras.get("extrapolation") or []),
            "decode_tps_per_gpu": metrics.decode_tps_per_gpu,
            "memory_per_gpu_gb": metrics.memory_per_gpu_gb,
            "max_concurrency": metrics.max_concurrency,
            "replica_gpus": metrics.replica_gpus,
            "tp": spec.tp,
            "ep": spec.ep,
            "pp": spec.pp,
            "isl": spec.isl,
            "osl": spec.osl,
        },
        profiling_enabled=False,
    )
    bypass_report.write_report(workspace, report)
    return 0


def _snapshot_config(workspace: Path, cfg: dict[str, Any]) -> None:
    """Persist the effective config into the workspace (best-effort)."""
    try:
        (workspace / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except OSError:
        pass


def _emit_failure(output_dir: Path, framework: str, model: str, error: str, start: float) -> int:
    """Emit a failing report + workspace for a pre-projection error."""
    workspace = bypass_report.create_workspace(output_dir, framework)
    report = bypass_report.build_report(
        None,
        framework=framework,
        model=model,
        success=False,
        workspace_dir=str(workspace),
        execution_time=time.time() - start,
        errors=[error],
        profiling_enabled=False,
    )
    bypass_report.write_report(workspace, report)
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the InferSim runner parser (Magpie-compatible flags)."""
    parser = argparse.ArgumentParser(prog="hyperloom-infersim-benchmark")
    sub = parser.add_subparsers(dest="mode", required=True)
    bench = sub.add_parser("benchmark", help="Project a serving config with InferSim")
    bench.add_argument("--benchmark-config", required=True)
    bench.add_argument("--output-dir", required=True)
    bench.add_argument("--run-mode", default="local")
    # Accepted for drop-in parity with Magpie/bypass; ignored (no server).
    bench.add_argument("--phase", default="all", choices=["all", "server", "client"])
    bench.add_argument("--server-lifecycle-pid-dir", default=None)
    bench.add_argument("--server-lifecycle-cleanup", default="true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point mirroring the bypass runner.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_arg_parser().parse_args(argv)
    if args.mode != "benchmark":
        print(f"unsupported mode: {args.mode}", file=sys.stderr)
        return 2
    if args.run_mode != "local":
        print(f"infersim runner supports --run-mode local only, got {args.run_mode}", file=sys.stderr)
        return 2
    if args.phase == "server":
        # No persistent server exists for a projection; a lone server phase is a
        # no-op success so lifecycle-driven callers don't stall.
        return 0
    return run_benchmark(Path(args.benchmark_config), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
