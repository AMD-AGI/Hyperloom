#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""A/B: torch.compile OFF vs ON — **kernel hot-list usage** from profile traces.

Purpose (design alignment): kernel-opt targets **native** sources; traces taken
with ``torch.compile`` enabled skew toward Inductor/Triton/tmp kernels that are
hard to rewrite and whose patches may not apply to the production (no-compile)
path. This script quantifies overlap/divergence.

Workflow::
  1. Two Magpie **profile** runs (same YAML except ``EXTRA_SGLANG_ARGS``).
  2. Parse ``torch_trace/*.trace.json.gz`` with the same ``analyze_trace_files``
     logic as ``kernel-agent/tools/tracelens_analysis.py``.
  3. Emit JSON: Jaccard on top-N kernel **names**, names unique to each arm,
     and heuristics for "compile-like" kernels (inductor/triton/torchinductor/tmp).

Usage::
    /opt/venv/bin/python scripts/ab_torch_compile_kernels.py \\
      --out-json /tmp/ab_kernel_usage.json

Env: reuse ``ROCR_VISIBLE_DEVICES`` / ``PATH`` like other Magpie scripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# kernel-agent lives next to inference_optimizer under Hyperloom/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TL = _REPO_ROOT / "kernel-agent" / "tools" / "tracelens_analysis.py"
if _TL.is_file():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("tracelens_analysis_ab", _TL)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        analyze_trace_files = _mod.analyze_trace_files
    else:
        analyze_trace_files = None  # type: ignore[misc, assignment]
else:
    analyze_trace_files = None  # type: ignore[misc, assignment]


def _inject_extra_sglang(cfg: dict, extra: str) -> dict:
    """Inject ``EXTRA_SGLANG_ARGS`` into a Magpie benchmark config.

    Args:
        cfg (dict): Mutable Magpie config mapping (modified in place).
        extra (str): Extra SGLang server args; ignored when blank.

    Returns:
        dict: The same ``cfg`` mapping with ``benchmark.envs.EXTRA_SGLANG_ARGS``
        set when ``extra`` is non-empty.
    """
    bench = cfg.setdefault("benchmark", {})
    envs = bench.setdefault("envs", {})
    if extra.strip():
        envs["EXTRA_SGLANG_ARGS"] = extra.strip()
    return cfg


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    cwd: str,
    timeout_sec: int,
) -> tuple[int, str, str]:
    """Run a Magpie benchmark subprocess and capture its output.

    Args:
        magpie_python (str): Python interpreter used to launch Magpie.
        config_path (Path): Path to the benchmark config YAML.
        output_dir (Path): Directory Magpie writes its run output to.
        cwd (str): Working directory for the subprocess.
        timeout_sec (int): Subprocess timeout in seconds.

    Returns:
        tuple[int, str, str]: ``(returncode, stdout, stderr)`` from the run.
    """
    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    cmd = [
        magpie_python,
        "-m",
        "Magpie",
        "-v",
        "benchmark",
        "--benchmark-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-mode",
        "local",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
        cwd=cwd,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _trace_files(workspace: Path) -> list[Path]:
    """List torch profiler trace files in a workspace.

    Args:
        workspace (Path): Magpie benchmark workspace directory.

    Returns:
        list[Path]: Sorted ``torch_trace/*.trace.json.gz`` paths; empty when
        the directory is absent.
    """
    td = workspace / "torch_trace"
    if not td.is_dir():
        return []
    return sorted(td.glob("*.trace.json.gz"))


def _kernel_name_set(top: list[dict[str, Any]]) -> list[str]:
    """Extract non-empty kernel names from a top-kernel list.

    Args:
        top (list[dict[str, Any]]): Top-kernel entries from trace analysis.

    Returns:
        list[str]: Kernel name strings, preserving input order.
    """
    return [str(x.get("name", "")) for x in top if x.get("name")]


def _jaccard(a: set[str], b: set[str]) -> float | None:
    """Compute the Jaccard similarity between two name sets.

    Args:
        a (set[str]): First set of kernel names.
        b (set[str]): Second set of kernel names.

    Returns:
        float | None: Intersection-over-union in [0, 1], ``0.0`` when both are
        non-empty but disjoint with empty union, or ``None`` when both sets are
        empty.
    """
    if not a and not b:
        return None
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _compile_like_fraction(names: list[str], sources: list[str]) -> float:
    """Share of kernels whose name/path hints Inductor/tmp compile artifacts.

    Args:
        names (list[str]): Kernel names.
        sources (list[str]): Kernel source-file strings, aligned with ``names``.

    Returns:
        float: Fraction in [0, 1] of kernels matching compile-artifact markers,
        or ``0.0`` when ``names`` is empty.
    """
    markers = (
        "inductor", "triton", "torchinductor", "/tmp/", "generated",
        "CompiledFunction", "autotune",
    )
    n = len(names)
    if not n:
        return 0.0
    hits = 0
    for nm, src in zip(names, sources):
        blob = f"{nm} {src}".lower()
        if any(m.lower() in blob for m in markers):
            hits += 1
    return hits / n


async def _profile_arm(
    *,
    name: str,
    profile_yaml: Path,
    extra_sglang: str,
    work_root: Path,
    magpie_python: str,
    cwd: str,
    variant_timeout_sec: int,
    top_k: int,
) -> dict[str, Any]:
    """Run one profiling arm and summarize its top kernels.

    Writes a per-arm config, runs Magpie, parses the resulting torch traces,
    and computes the top-kernel names plus their compile-like fraction.

    Args:
        name (str): Arm label and output subdirectory name.
        profile_yaml (Path): Base profile config to clone for this arm.
        extra_sglang (str): Extra SGLang args to inject for this arm.
        work_root (Path): Root directory under which the arm slot is created.
        magpie_python (str): Python interpreter used to launch Magpie.
        cwd (str): Working directory for the Magpie subprocess.
        variant_timeout_sec (int): Per-run timeout in seconds.
        top_k (int): Number of top kernels to retain from trace analysis.

    Returns:
        dict[str, Any]: Arm result with workspace/trace info, top kernel names,
        compile-like fraction, success flag, and a stderr tail.
    """
    slot = work_root / name
    slot.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(profile_yaml.read_text(encoding="utf-8"))
    _inject_extra_sglang(cfg, extra_sglang)
    cfg_path = slot / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rc, stdout, stderr = await asyncio.to_thread(
        _run_magpie,
        magpie_python=magpie_python,
        config_path=cfg_path,
        output_dir=slot,
        cwd=cwd,
        timeout_sec=variant_timeout_sec,
    )
    candidates = sorted(slot.glob("benchmark_*"))
    workspace = candidates[-1] if candidates else None
    traces = _trace_files(workspace) if workspace else []
    top: list[dict[str, Any]] = []
    if analyze_trace_files and traces:
        top = analyze_trace_files(traces, top_k)
    names = _kernel_name_set(top)
    sources = [str(x.get("source_file", "")) for x in top]

    return {
        "arm": name,
        "extra_server_args": extra_sglang.strip(),
        "returncode": rc,
        "workspace": str(workspace) if workspace else None,
        "trace_dir": str(workspace / "torch_trace") if workspace else None,
        "trace_files": [str(p) for p in traces],
        "top_k": top_k,
        "success": rc == 0 and bool(traces) and bool(top),
        "top_kernel_names": names,
        "compile_like_fraction_top_k": round(
            _compile_like_fraction(names, sources), 4,
        ),
        "stderr_tail": (stderr or stdout)[-3500:],
    }


async def main_async() -> int:
    """Run both profiling arms and emit the kernel-comparison report.

    Parses CLI arguments, profiles a torch.compile-off and torch.compile-on
    arm, computes Jaccard overlap and per-arm-unique kernel names, then writes
    and prints a JSON summary.

    Returns:
        int: ``0`` when both arms succeeded, ``1`` on partial/failed arms, or
        ``2`` on configuration/dependency errors.
    """
    ap = argparse.ArgumentParser(
        description="Compare hot GPU kernels: torch.compile off vs on (profile traces)",
    )
    ap.add_argument(
        "--profile-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs"
        / "profile_sglang.yaml",
    )
    ap.add_argument("--base-extra-args", default="")
    ap.add_argument(
        "--arm-b-suffix",
        default="--enable-torch-compile --mem-fraction-static 0.6",
    )
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--magpie-python", default="/opt/venv/bin/python")
    ap.add_argument("--cwd", default="/tmp")
    ap.add_argument("--variant-timeout-sec", type=int, default=2400)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if analyze_trace_files is None:
        print(
            f"tracelens_analysis not found at {_TL} — cannot compare kernels.",
            file=sys.stderr,
        )
        return 2
    if not args.profile_config.exists():
        print(f"config not found: {args.profile_config}", file=sys.stderr)
        return 2

    root = args.output_root or Path(
        f"/tmp/ab_kernel_usage_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    root.mkdir(parents=True, exist_ok=True)

    base = (args.base_extra_args or "").strip()
    b_extra = (
        f"{base} {args.arm_b_suffix}".strip()
        if base
        else args.arm_b_suffix.strip()
    )

    a_res = await _profile_arm(
        name="arm_a_no_torch_compile",
        profile_yaml=args.profile_config,
        extra_sglang=base,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
        top_k=args.top_k,
    )
    b_res = await _profile_arm(
        name="arm_b_torch_compile",
        profile_yaml=args.profile_config,
        extra_sglang=b_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
        top_k=args.top_k,
    )

    sa = set(a_res.get("top_kernel_names") or [])
    sb = set(b_res.get("top_kernel_names") or [])
    jac = _jaccard(sa, sb)
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)

    out: dict[str, Any] = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "intent": (
            "Compare kernel hot-lists for kernel-opt targeting native vs compile-only "
            "artifacts"
        ),
        "profile_config": str(args.profile_config),
        "base_extra_args": base,
        "arm_b_suffix": args.arm_b_suffix,
        "top_k": args.top_k,
        "work_root": str(root),
        "arm_a": a_res,
        "arm_b": b_res,
        "comparison": {
            "jaccard_top_k_names": jac,
            "only_in_arm_a": only_a[:50],
            "only_in_arm_b": only_b[:50],
            "interpretation_hint": (
                "Higher compile_like_fraction on arm_b suggests profiling with compile "
                "ON emphasizes Inductor/tmp kernels; for native kernel-opt prefer "
                "arm_a traces when fractions diverge."
            ),
        },
    }

    out_path = args.out_json or (root / "ab_kernel_usage.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    ok = bool(a_res.get("success") and b_res.get("success"))
    return 0 if ok else 1


def main() -> None:
    """Run the async entry point and exit with its return code.

    Raises:
        SystemExit: Always, carrying the exit code from :func:`main_async`.
    """
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
