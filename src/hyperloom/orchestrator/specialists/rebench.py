# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optional gate-comparable rebench helper for GPU specialists."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors._workload_envs import prepare_grid_benchmark_config


import argparse
import asyncio
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.perf_metric import constraint_fields
from ..actions.executors._grid_runner import GridLaunch, GridRecipe, GridStop, GridVariant, VariantResult, run_grid
from ..actions.executors._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
)

log = logging.getLogger(__name__)


def _resolve_port(port: int | None) -> int:
    """Resolve the requested port, using an OS-assigned port for ``None`` or ``0``."""
    if port not in (None, 0):
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _current_leased_cards() -> str:
    """Return the cards the subprocess is pinned to (for result reporting)."""
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return ""


@dataclass(frozen=True)
class RebenchBenchmark:
    """Which benchmark a rebench runs: the base config and the overrides applied to it.

    Attributes:
        config_path: Base Magpie YAML; the packaged baseline when ``None``.
        model_path: Override of the benchmark model path.
        gpu_type: Pin of the generic benchmark script's GPU type.
        benchmark_script: Force-pinned benchmark script.
        magpie_python: Interpreter that can import Magpie.
    """

    config_path: str | Path | None = None
    model_path: str | None = None
    gpu_type: str | None = None
    benchmark_script: str | None = None
    magpie_python: str | None = None


@dataclass(frozen=True)
class RebenchServer:
    """How the server the rebench boots is launched.

    Attributes:
        base_extra_args: Server args merged ahead of the variant args.
        extra_envs: Per-run env overrides.
        port: Server port; an OS-assigned one when ``None``.
    """

    base_extra_args: str = ""
    extra_envs: dict[str, str] | None = None
    port: int | None = None


async def run_specialist_rebench(
    *,
    output_dir: str | Path,
    benchmark: RebenchBenchmark = RebenchBenchmark(),
    server: RebenchServer = RebenchServer(),
) -> dict[str, Any]:
    """Run one gate-comparable benchmark on the specialist's leased cards."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_port = _resolve_port(server.port)

    base_config = Path(benchmark.config_path) if benchmark.config_path else Path(default_baseline_config())
    target = BenchmarkTarget(
        model_path=benchmark.model_path or None,
        gpu_type=benchmark.gpu_type or None,
        benchmark_script=benchmark.benchmark_script or None,
    )
    materialized = prepare_grid_benchmark_config(
        base_config,
        out_root,
        target=target,
        options=MaterializeOptions(out_name="specialist_rebench.with_envs.yaml"),
    )

    variant = GridVariant(
        name="specialist-rebench",
        # Identity variant kept empty so helper-provided server flags are injected exactly once.
        extra_server_args="",
        extra_envs=dict(server.extra_envs or {}),
        note="specialist_rebench",
    )

    # run_grid's server lifecycle tears down the server it started on exit.
    server_lifecycle = {
        "cleanup": True,
        "pid_dir": str(out_root / "server"),
        "port": resolved_port,
    }

    gpu_ids = _current_leased_cards()
    try:
        results = await run_grid(
            grid=[variant],
            output_root=out_root,
            recipe=GridRecipe(
                base_yaml_path=materialized,
                base=LaunchLayer(extra_server_args=(server.base_extra_args or "").strip()),
                target=target,
            ),
            launch=GridLaunch(magpie_python=benchmark.magpie_python or None, server_lifecycle=server_lifecycle),
            stop=GridStop(keep_going_on_failure=False),
        )
    except Exception as exc:  # outermost CLI command: the failure is printed as an ok=false result
        log.warning("specialist rebench failed: %s", exc, exc_info=True)
        return {
            "ok": False,
            "error": repr(exc),
            "port": resolved_port,
            "gpu_ids": gpu_ids,
        }

    rb = results[0] if results else None
    if rb is None:
        return {
            "ok": False,
            "error": "rebench produced no result",
            "port": resolved_port,
            "gpu_ids": gpu_ids,
        }
    return _rebench_result(rb, port=resolved_port, gpu_ids=gpu_ids)


def _rebench_result(rb: VariantResult, *, port: int, gpu_ids: str) -> dict[str, Any]:
    """The CLI result of one finished rebench variant."""
    warnings: list[str] = []
    ok = rb.status == "succeeded"
    if not ok:
        warnings.append(f"rebench_failed:{(rb.error or '')[-160:]}")
    return {
        "ok": ok,
        "status": rb.status,
        "output_throughput": getattr(rb, "output_throughput", None),
        # ``VariantResult`` names these ``ttft_mean_ms`` / ``tpot_mean_ms``; the emitted keys stay ``ttft_ms`` /
        # ``itl_ms`` for the collectors.
        "ttft_ms": rb.ttft_mean_ms,
        "itl_ms": rb.tpot_mean_ms,
        # The session's constraints fail closed, so a lane that drops these refuses every KEEP.
        **constraint_fields(rb),
        "workspace": str(getattr(rb, "workspace", "") or ""),
        "port": port,
        "gpu_ids": gpu_ids,
        "error": getattr(rb, "error", "") or "",
        "warnings": warnings + list(getattr(rb, "nonfatal_warnings", []) or []),
    }


def _parse_env_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VAL`` CLI ``--env`` pairs into a dict."""
    out: dict[str, str] = {}
    for item in pairs or []:
        key, sep, val = str(item).partition("=")
        if sep and key.strip():
            out[key.strip()] = val
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run one rebench and print a JSON result to stdout."""
    parser = argparse.ArgumentParser(
        prog="specialist_rebench",
        description=(
            "Optional gate-comparable rebench on the specialist's leased cards using the Magpie serving path."
        ),
    )
    parser.add_argument("--config", default=None, help="Base Magpie YAML (defaults to packaged baseline).")
    parser.add_argument("--output", required=True, help="Output directory (conventionally inside the worktree).")
    parser.add_argument("--extra-args", default="", help="Server args merged ahead of the variant args.")
    parser.add_argument("--port", type=int, default=0, help="Server port (0 uses an OS-assigned port).")
    parser.add_argument("--model-path", default=None, help="Override benchmark model path.")
    parser.add_argument("--gpu-type", default=None, help="Pin the generic benchmark script GPU type.")
    parser.add_argument("--benchmark-script", default=None, help="Force-pin a benchmark script.")
    parser.add_argument("--magpie-python", default=None, help="Interpreter that can import Magpie.")
    parser.add_argument("--env", action="append", default=[], help="Per-run env override KEY=VAL (repeatable).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        result = asyncio.run(
            run_specialist_rebench(
                output_dir=args.output,
                benchmark=RebenchBenchmark(
                    config_path=args.config,
                    model_path=args.model_path,
                    gpu_type=args.gpu_type,
                    benchmark_script=args.benchmark_script,
                    magpie_python=args.magpie_python,
                ),
                server=RebenchServer(
                    base_extra_args=args.extra_args,
                    extra_envs=_parse_env_pairs(args.env),
                    port=args.port,
                ),
            )
        )
    except ValueError as exc:
        result = {"ok": False, "error": str(exc)}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


__all__ = [
    "RebenchBenchmark",
    "RebenchServer",
    "run_specialist_rebench",
]


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
