# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Optional gate-comparable rebench helper for GPU specialists.

This is a CONVENIENCE, not a mandate. A GPU specialist that wants throughput
numbers directly comparable to the Coordinator's ``integrate_patch`` E2E gate
can call this helper to reuse the real Magpie serving + benchmark path
(``run_grid``) on its own leased cards and a non-production port — in one call,
instead of hand-rolling a bench harness. The specialist is equally free to
write its own autotune / profiling / bench scripts (broad ``Bash`` autonomy);
throughput does NOT have to come from this helper, and the ``integrate_patch``
gate stays the single authoritative measure of truth.

Physics boundary (mirrors the specialist iron rules): the helper runs the
server on the cards the subprocess already has pinned via
``ROCR_VISIBLE_DEVICES`` (its serving-disjoint lease) and REFUSES the
production serving port 8888, so it can never collide with production serving.

CLI usage (from inside a specialist subprocess)::

    python -m inference_optimizer.orchestrator.specialist_rebench \
        --config <magpie.yaml> --output <dir> [--port 0] \
        [--extra-args "<server args>"] [--env KEY=VAL ...]

prints a single JSON result object to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from .action_executors._grid_runner import GridVariant, run_grid
from .action_executors._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)


# The production serving process owns this port; a specialist rebench must
# never bind it (co-residing would corrupt both measurements).
PRODUCTION_SERVING_PORT = 8888

# Default per-variant timeout (s) for a one-off specialist rebench. Generous
# enough for a cold server start + one benchmark round on a large model.
DEFAULT_REBENCH_TIMEOUT_SEC = 7800


def _pick_free_port() -> int:
    """Pick a free localhost TCP port (never the production serving port).

    Returns:
        int: An ephemeral port the OS reported free, guaranteed != 8888.
    """
    for _ in range(8):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port != PRODUCTION_SERVING_PORT:
            return port
    # Astronomically unlikely fallthrough; pick a fixed non-8888 high port.
    return 18888


def _resolve_port(port: int | None) -> int:
    """Resolve and validate the rebench server port.

    Args:
        port: Requested port; ``None`` or ``0`` auto-picks a free one.

    Returns:
        int: A concrete port that is not the production serving port.

    Raises:
        ValueError: If the production serving port 8888 is requested explicitly.
    """
    if port in (None, 0):
        return _pick_free_port()
    p = int(port)
    if p == PRODUCTION_SERVING_PORT:
        raise ValueError(
            f"refusing port {PRODUCTION_SERVING_PORT}: that is the production "
            "serving port; a specialist rebench must use its own port."
        )
    return p


def _current_leased_cards() -> str:
    """Return the cards the subprocess is pinned to (for result reporting).

    Returns:
        str: The ``ROCR_VISIBLE_DEVICES`` (then HIP/CUDA) value, or ``""``.
    """
    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return ""


async def run_specialist_rebench(
    *,
    config_path: str | Path | None,
    output_dir: str | Path,
    base_extra_args: str = "",
    extra_envs: dict[str, str] | None = None,
    port: int | None = None,
    variant_timeout_sec: int = DEFAULT_REBENCH_TIMEOUT_SEC,
    model_path: str | None = None,
    gpu_type: str | None = None,
    benchmark_script: str | None = None,
    magpie_python: str | None = None,
) -> dict[str, Any]:
    """Run one gate-comparable benchmark on the specialist's leased cards.

    Reuses the real Magpie serving + benchmark path (``run_grid``) with a
    single identity variant. The server runs on the cards already pinned into
    the subprocess env (``ROCR_VISIBLE_DEVICES``) and on a non-production port.

    Args:
        config_path: Base Magpie YAML to template from; ``None`` falls back to
            the packaged default baseline config.
        output_dir: Directory the rebench writes its config / server pid / and
            benchmark output under (conventionally inside the worktree).
        base_extra_args: Server args merged ahead of the variant's args
            (e.g. the current-best stack args, to compare apples-to-apples).
        extra_envs: Per-run environment overrides (e.g. tuned config env vars).
        port: Server port; ``None``/``0`` auto-picks a free one. 8888 is
            rejected.
        variant_timeout_sec: Per-variant wall-clock timeout.
        model_path: Overrides the benchmark model path when set.
        gpu_type: Pins the generic ``{framework}_{gpu_type}.sh`` benchmark.
        benchmark_script: Force-pins a benchmark script.
        magpie_python: Interpreter that can import Magpie.

    Returns:
        A result dict with ``ok``, ``output_throughput`` (the gate-comparable
        number), latency fields, ``status``, ``workspace``, ``port``,
        ``gpu_ids`` and any ``warnings`` / ``error``.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_port = _resolve_port(port)

    base_config = Path(config_path) if config_path else Path(default_baseline_config())
    materialized = materialize_config_with_envs(
        base_config,
        out_root,
        model_path=model_path or None,
        gpu_type=gpu_type or None,
        benchmark_script=benchmark_script or None,
        out_name="specialist_rebench.with_envs.yaml",
    )

    variant = GridVariant(
        name="specialist-rebench",
        # ``run_grid`` merges base_extra_args ahead of each variant's args.
        # Keep the identity variant empty so helper-provided server flags are
        # injected exactly once.
        extra_server_args="",
        extra_envs=dict(extra_envs or {}),
        note="specialist_rebench",
    )

    # Process-group cleanup is handled by run_grid's server lifecycle: cleanup
    # tears down the server it started (by its own pid in pid_dir) on exit, so
    # an aborted rebench does not orphan a server pinning the leased cards.
    server_lifecycle = {
        "cleanup": True,
        "pid_dir": str(out_root / "server"),
        "port": resolved_port,
    }

    gpu_ids = _current_leased_cards()
    warnings: list[str] = []
    try:
        results = await run_grid(
            base_yaml_path=materialized,
            base_extra_args=(base_extra_args or "").strip(),
            grid=[variant],
            output_root=out_root,
            variant_timeout_sec=int(variant_timeout_sec),
            keep_going_on_failure=False,
            model_path=model_path or None,
            gpu_type=gpu_type or None,
            benchmark_script=benchmark_script or None,
            magpie_python=magpie_python or None,
            server_lifecycle=server_lifecycle,
        )
    except Exception as exc:  # noqa: BLE001 — surface as a structured failure
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
    ok = rb.status == "succeeded"
    if not ok:
        warnings.append(f"rebench_failed:{(rb.error or '')[-160:]}")
    return {
        "ok": ok,
        "status": rb.status,
        "output_throughput": getattr(rb, "output_throughput", None),
        "ttft_ms": getattr(rb, "ttft_ms", None),
        "itl_ms": getattr(rb, "itl_ms", None),
        "workspace": str(getattr(rb, "workspace", "") or ""),
        "port": resolved_port,
        "gpu_ids": gpu_ids,
        "error": getattr(rb, "error", "") or "",
        "warnings": warnings + list(getattr(rb, "nonfatal_warnings", []) or []),
    }


def _parse_env_pairs(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VAL`` CLI ``--env`` pairs into a dict.

    Args:
        pairs: Raw ``KEY=VAL`` strings, or ``None``.

    Returns:
        The parsed environment-override mapping (malformed entries skipped).
    """
    out: dict[str, str] = {}
    for item in pairs or []:
        key, sep, val = str(item).partition("=")
        if sep and key.strip():
            out[key.strip()] = val
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run one rebench and print a JSON result to stdout.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        int: ``0`` when the rebench succeeded, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="specialist_rebench",
        description=(
            "Optional gate-comparable rebench on the specialist's leased "
            "cards + a non-production port (reuses the Magpie serving path)."
        ),
    )
    parser.add_argument("--config", default=None, help="Base Magpie YAML (defaults to packaged baseline).")
    parser.add_argument("--output", required=True, help="Output directory (conventionally inside the worktree).")
    parser.add_argument("--extra-args", default="", help="Server args merged ahead of the variant args.")
    parser.add_argument("--port", type=int, default=0, help="Server port (0 auto-picks a free one; 8888 rejected).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_REBENCH_TIMEOUT_SEC, help="Per-variant timeout (s).")
    parser.add_argument("--model-path", default=None, help="Override benchmark model path.")
    parser.add_argument("--gpu-type", default=None, help="Pin the generic benchmark script GPU type.")
    parser.add_argument("--benchmark-script", default=None, help="Force-pin a benchmark script.")
    parser.add_argument("--magpie-python", default=None, help="Interpreter that can import Magpie.")
    parser.add_argument("--env", action="append", default=[], help="Per-run env override KEY=VAL (repeatable).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        result = asyncio.run(
            run_specialist_rebench(
                config_path=args.config,
                output_dir=args.output,
                base_extra_args=args.extra_args,
                extra_envs=_parse_env_pairs(args.env),
                port=args.port,
                variant_timeout_sec=args.timeout,
                model_path=args.model_path,
                gpu_type=args.gpu_type,
                benchmark_script=args.benchmark_script,
                magpie_python=args.magpie_python,
            )
        )
    except ValueError as exc:
        result = {"ok": False, "error": str(exc)}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


__all__ = [
    "DEFAULT_REBENCH_TIMEOUT_SEC",
    "PRODUCTION_SERVING_PORT",
    "run_specialist_rebench",
]


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
