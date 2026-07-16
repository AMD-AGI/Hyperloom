#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A/B: torch.compile OFF vs ON via Magpie, selectable by ``--mode``.

Both modes run two Magpie arms that share the same YAML except for
``EXTRA_SGLANG_ARGS`` (arm A = baseline flags, arm B = baseline plus
``--arm-b-suffix``, default ``--enable-torch-compile --mem-fraction-static 0.6``
per inference-optimization/actions/baseline.md), then emit a JSON summary.

``--mode kernels`` — **kernel hot-list usage** from profile traces.
  Kernel-opt targets **native** sources; traces taken with ``torch.compile``
  enabled skew toward Inductor/Triton/tmp kernels that are hard to rewrite and
  whose patches may not apply to the production (no-compile) path. This mode
  parses ``torch_trace/*.trace.json.gz`` with the same ``analyze_trace_files``
  logic as ``src/hyperloom/agents/kernel/tools/tracelens_analysis.py`` and
  reports the Jaccard overlap on top-N kernel **names**, names unique to each
  arm, and a "compile-like" heuristic (inductor/triton/torchinductor/tmp).

``--mode magpie`` — **throughput** comparison.
  Reads each arm's ``benchmark_report.json`` and reports output throughput plus
  the B/A ratio and percent delta. For *which* GPU kernels appear in the
  profile (native vs Inductor), use ``--mode kernels`` instead — that is the
  metric that matters for kernel-agent targeting.

Usage::
    /opt/venv/bin/python experiments/ab_torch_compile.py --mode kernels \\
      --out-json /tmp/ab_kernel_usage.json
    /opt/venv/bin/python experiments/ab_torch_compile.py --mode magpie \\
      --config experiments/configs/baseline_sglang.yaml \\
      --out-json /tmp/ab_torch_compile.json

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

# kernel-agent lives in the in-tree hyperloom src-layout namespace
# (src/hyperloom/agents/kernel). Imported lazily-by-path so ``--mode magpie``
# still works when TraceLens deps are unavailable.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TL = _REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "tools" / "tracelens_analysis.py"
if _TL.is_file():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("tracelens_analysis_ab", _TL)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_mod)
            analyze_trace_files = _mod.analyze_trace_files
        except Exception:  # noqa: BLE001 - optional dep; magpie mode must still load
            # TraceLens (or a transitive dep) unavailable: kernels mode reports
            # this cleanly and returns 2; magpie mode stays fully functional.
            analyze_trace_files = None  # type: ignore[misc, assignment]
    else:
        analyze_trace_files = None  # type: ignore[misc, assignment]
else:
    analyze_trace_files = None  # type: ignore[misc, assignment]


def _config_dir() -> Path:
    """Return the bundled config directory next to this script.

    Returns:
        Path: The ``configs/`` directory holding the shipped Magpie YAMLs.
    """
    return Path(__file__).resolve().parent / "configs"


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


async def _run_arm(
    *,
    name: str,
    base_yaml: Path,
    extra_sglang: str,
    work_root: Path,
    magpie_python: str,
    cwd: str,
    variant_timeout_sec: int,
) -> tuple[int, str, str, Path | None]:
    """Materialize a per-arm config, run Magpie, and locate its workspace.

    Shared by both modes: writes ``config.yaml`` into a per-arm slot with the
    injected ``EXTRA_SGLANG_ARGS``, runs Magpie in a worker thread, then finds
    the latest ``benchmark_*`` workspace it produced.

    Args:
        name (str): Arm label and output subdirectory name.
        base_yaml (Path): Base config to clone for this arm.
        extra_sglang (str): Extra SGLang args to inject for this arm.
        work_root (Path): Root directory under which the arm slot is created.
        magpie_python (str): Python interpreter used to launch Magpie.
        cwd (str): Working directory for the Magpie subprocess.
        variant_timeout_sec (int): Per-run timeout in seconds.

    Returns:
        tuple[int, str, str, Path | None]: ``(returncode, stdout, stderr,
        workspace)``; ``workspace`` is ``None`` when Magpie produced no
        ``benchmark_*`` directory.
    """
    slot = work_root / name
    slot.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
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
    return rc, stdout, stderr, workspace


def _emit(out: dict[str, Any], out_path: Path) -> None:
    """Write the report JSON to disk and echo it to stdout.

    Args:
        out (dict[str, Any]): Report mapping to serialize.
        out_path (Path): Destination file for the JSON report.
    """
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


def _arm_extras(base_extra_args: str, arm_b_suffix: str) -> tuple[str, str, str]:
    """Compute the shared base and per-arm ``EXTRA_SGLANG_ARGS`` strings.

    Args:
        base_extra_args (str): Extra args applied to both arms.
        arm_b_suffix (str): Suffix appended to arm A flags to form arm B.

    Returns:
        tuple[str, str, str]: ``(base, arm_a_extra, arm_b_extra)``.
    """
    base = (base_extra_args or "").strip()
    a_extra = base
    b_extra = f"{base} {arm_b_suffix}".strip() if base else arm_b_suffix.strip()
    return base, a_extra, b_extra


# --------------------------------------------------------------------------- #
# Mode: kernels (profile-trace hot-list comparison)
# --------------------------------------------------------------------------- #


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
        "inductor",
        "triton",
        "torchinductor",
        "/tmp/",
        "generated",
        "CompiledFunction",
        "autotune",
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


async def _kernels_arm(
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

    Runs Magpie via :func:`_run_arm`, parses the resulting torch traces, and
    computes the top-kernel names plus their compile-like fraction.

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
    rc, stdout, stderr, workspace = await _run_arm(
        name=name,
        base_yaml=profile_yaml,
        extra_sglang=extra_sglang,
        work_root=work_root,
        magpie_python=magpie_python,
        cwd=cwd,
        variant_timeout_sec=variant_timeout_sec,
    )
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
            _compile_like_fraction(names, sources),
            4,
        ),
        "stderr_tail": (stderr or stdout)[-3500:],
    }


async def _run_mode_kernels(args: argparse.Namespace) -> int:
    """Run both profiling arms and emit the kernel-comparison report.

    Profiles a torch.compile-off and torch.compile-on arm, computes Jaccard
    overlap and per-arm-unique kernel names, then writes and prints a JSON
    summary.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: ``0`` when both arms succeeded, ``1`` on partial/failed arms, or
        ``2`` on configuration/dependency errors.
    """
    config = args.profile_config or (_config_dir() / "profile_sglang.yaml")

    if analyze_trace_files is None:
        print(
            f"tracelens_analysis not found at {_TL} — cannot compare kernels.",
            file=sys.stderr,
        )
        return 2
    if not config.exists():
        print(f"config not found: {config}", file=sys.stderr)
        return 2

    root = args.output_root or Path(
        f"/tmp/ab_kernel_usage_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    root.mkdir(parents=True, exist_ok=True)

    base, a_extra, b_extra = _arm_extras(args.base_extra_args, args.arm_b_suffix)

    # Sequential: both arms pin the same GPU via YAML; parallel runs would contend.
    a_res = await _kernels_arm(
        name="arm_a_no_torch_compile",
        profile_yaml=config,
        extra_sglang=a_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
        top_k=args.top_k,
    )
    b_res = await _kernels_arm(
        name="arm_b_torch_compile",
        profile_yaml=config,
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
        "intent": ("Compare kernel hot-lists for kernel-opt targeting native vs compile-only artifacts"),
        "profile_config": str(config),
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
    _emit(out, out_path)
    ok = bool(a_res.get("success") and b_res.get("success"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Mode: magpie (throughput comparison)
# --------------------------------------------------------------------------- #


def _read_report(workspace: Path) -> dict | None:
    """Load ``benchmark_report.json`` from a workspace if present.

    Args:
        workspace (Path): Magpie benchmark workspace directory.

    Returns:
        dict | None: Parsed report mapping, or ``None`` when the file is
        absent.
    """
    p = workspace / "benchmark_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


async def _magpie_arm(
    *,
    name: str,
    base_yaml: Path,
    extra_sglang: str,
    work_root: Path,
    magpie_python: str,
    cwd: str,
    variant_timeout_sec: int,
) -> dict:
    """Run one throughput arm and summarize its benchmark report.

    Runs Magpie via :func:`_run_arm`, then reads the output throughput from the
    resulting benchmark report.

    Args:
        name (str): Arm label and output subdirectory name.
        base_yaml (Path): Base config to clone for this arm.
        extra_sglang (str): Extra SGLang args to inject for this arm.
        work_root (Path): Root directory under which the arm slot is created.
        magpie_python (str): Python interpreter used to launch Magpie.
        cwd (str): Working directory for the Magpie subprocess.
        variant_timeout_sec (int): Per-run timeout in seconds.

    Returns:
        dict: Arm result with workspace/report paths, success flag, output
        throughput, and a stderr tail.
    """
    rc, stdout, stderr, workspace = await _run_arm(
        name=name,
        base_yaml=base_yaml,
        extra_sglang=extra_sglang,
        work_root=work_root,
        magpie_python=magpie_python,
        cwd=cwd,
        variant_timeout_sec=variant_timeout_sec,
    )
    report = _read_report(workspace) if workspace else None
    tput = None
    if report:
        tput = (report.get("throughput") or {}).get("output_throughput")

    return {
        "arm": name,
        "extra_server_args": extra_sglang.strip(),
        "returncode": rc,
        "workspace": str(workspace) if workspace else None,
        "report_path": str(workspace / "benchmark_report.json") if workspace else None,
        "success": bool(report and report.get("success")),
        "output_throughput": tput,
        "stderr_tail": (stderr or stdout)[-4000:],
    }


async def _run_mode_magpie(args: argparse.Namespace) -> int:
    """Run both throughput arms and emit the A/B comparison report.

    Runs the torch.compile-off and torch.compile-on arms sequentially (shared
    GPU), computes the throughput ratio and percent delta, then writes and
    prints a JSON summary.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: ``0`` when both arms succeeded, ``1`` on partial/failed arms, or
        ``2`` when the config file is missing.
    """
    config = args.config or (_config_dir() / "baseline_sglang.yaml")

    if not config.exists():
        print(f"config not found: {config}", file=sys.stderr)
        return 2

    root = args.output_root or Path(f"/tmp/ab_torch_compile_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=True)

    base, a_extra, b_extra = _arm_extras(args.base_extra_args, args.arm_b_suffix)

    # Sequential: both arms pin the same GPU via YAML; parallel runs would contend.
    a_res = await _magpie_arm(
        name="arm_a_no_torch_compile",
        base_yaml=config,
        extra_sglang=a_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
    )
    b_res = await _magpie_arm(
        name="arm_b_torch_compile",
        base_yaml=config,
        extra_sglang=b_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
    )

    ta = a_res.get("output_throughput")
    tb = b_res.get("output_throughput")
    ratio = (float(tb) / float(ta)) if (isinstance(ta, (int, float)) and ta and isinstance(tb, (int, float))) else None
    diff_pct = (
        ((float(tb) - float(ta)) / float(ta) * 100.0)
        if (isinstance(ta, (int, float)) and ta and isinstance(tb, (int, float)))
        else None
    )

    out = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "config": str(config),
        "base_extra_args": base,
        "arm_b_suffix": args.arm_b_suffix,
        "work_root": str(root),
        "arm_a": a_res,
        "arm_b": b_res,
        "summary": {
            "tput_a": ta,
            "tput_b": tb,
            "b_over_a_ratio": ratio,
            "b_vs_a_pct": diff_pct,
        },
    }

    out_path = args.out_json or (root / "ab_result.json")
    _emit(out, out_path)
    return 0 if a_res.get("success") and b_res.get("success") else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for both A/B modes.

    Returns:
        argparse.ArgumentParser: Parser exposing ``--mode`` plus the shared and
        mode-specific flags of the original ``kernels`` / ``magpie`` scripts.
    """
    ap = argparse.ArgumentParser(
        description="torch.compile A/B via Magpie (kernel hot-list or throughput)",
    )
    ap.add_argument(
        "--mode",
        choices=("kernels", "magpie"),
        required=True,
        help="kernels: compare profile-trace hot kernels; magpie: compare throughput.",
    )
    # magpie config flag (default experiments/configs/baseline_sglang.yaml).
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="[--mode magpie] Base benchmark config YAML.",
    )
    # kernels config flag (default experiments/configs/profile_sglang.yaml).
    ap.add_argument(
        "--profile-config",
        type=Path,
        default=None,
        help="[--mode kernels] Base profile config YAML.",
    )
    ap.add_argument(
        "--base-extra-args",
        default="",
        help="EXTRA_SGLANG_ARGS applied to BOTH arms before arm-specific flags.",
    )
    ap.add_argument(
        "--arm-b-suffix",
        default="--enable-torch-compile --mem-fraction-static 0.6",
        help="Appended to arm A flags for arm B (skill-style pairing).",
    )
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--magpie-python", default="/opt/venv/bin/python")
    ap.add_argument("--cwd", default="/tmp")
    ap.add_argument("--variant-timeout-sec", type=int, default=2400)
    ap.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="[--mode kernels] Number of top kernels to retain from traces.",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    return ap


async def main_async() -> int:
    """Parse CLI arguments and dispatch to the selected mode.

    Returns:
        int: Exit code from the selected mode runner.
    """
    args = _build_parser().parse_args()
    if args.mode == "kernels":
        return await _run_mode_kernels(args)
    return await _run_mode_magpie(args)


def main() -> None:
    """Run the async entry point and exit with its return code.

    Raises:
        SystemExit: Always, carrying the exit code from :func:`main_async`.
    """
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
