#!/usr/bin/env python3
"""A/B: same Magpie YAML, compare **throughput** without vs with --enable-torch-compile.

For **which GPU kernels appear in the profile** (native vs Inductor), use
``ab_torch_compile_kernels.py`` — that is the metric that matters for
kernel-agent targeting.

Arm A — baseline flags only (optionally pass --base-extra-args).
Arm B — skill-aligned pair from inference-optimization/actions/baseline.md:
        ``--enable-torch-compile --mem-fraction-static 0.6`` so memory matches
        marathon scripts when compile is on.

Outputs JSON with both arms' benchmark_report paths and throughput.

Usage::
    /opt/venv/bin/python scripts/ab_torch_compile_magpie.py \\
      --config scripts/configs/baseline_sglang.yaml \\
      --out-json /tmp/ab_torch_compile.json
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

import yaml


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


async def _arm(
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

    Writes a per-arm config, runs Magpie, locates the latest workspace, and
    reads the output throughput from its benchmark report.

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
    report = _read_report(workspace) if workspace else None
    tput = None
    if report:
        tput = (report.get("throughput") or {}).get("output_throughput")

    return {
        "arm": name,
        "extra_server_args": extra_sglang.strip(),
        "returncode": rc,
        "workspace": str(workspace) if workspace else None,
        "report_path": str(workspace / "benchmark_report.json")
        if workspace
        else None,
        "success": bool(report and report.get("success")),
        "output_throughput": tput,
        "stderr_tail": (stderr or stdout)[-4000:],
    }


async def main_async() -> int:
    """Run both throughput arms and emit the A/B comparison report.

    Parses CLI arguments, runs the torch.compile-off and torch.compile-on arms
    sequentially (shared GPU), computes the throughput ratio and percent delta,
    then writes and prints a JSON summary.

    Returns:
        int: ``0`` when both arms succeeded, ``1`` on partial/failed arms, or
        ``2`` when the config file is missing.
    """
    ap = argparse.ArgumentParser(description="torch.compile A/B via Magpie")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs"
        / "baseline_sglang.yaml",
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
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    root = args.output_root or Path(
        f"/tmp/ab_torch_compile_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    root.mkdir(parents=True, exist_ok=True)

    base = (args.base_extra_args or "").strip()
    a_extra = base
    b_extra = f"{base} {args.arm_b_suffix}".strip() if base else args.arm_b_suffix.strip()

    # Sequential: both arms pin the same GPU via YAML; parallel runs would contend.
    a_res = await _arm(
        name="arm_a_no_torch_compile",
        base_yaml=args.config,
        extra_sglang=a_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
    )
    b_res = await _arm(
        name="arm_b_torch_compile",
        base_yaml=args.config,
        extra_sglang=b_extra,
        work_root=root,
        magpie_python=args.magpie_python,
        cwd=args.cwd,
        variant_timeout_sec=args.variant_timeout_sec,
    )

    ta = a_res.get("output_throughput")
    tb = b_res.get("output_throughput")
    ratio = (float(tb) / float(ta)) if (isinstance(ta, (int, float)) and ta
                                        and isinstance(tb, (int, float))) else None
    diff_pct = ((float(tb) - float(ta)) / float(ta) * 100.0) if (
        isinstance(ta, (int, float)) and ta
        and isinstance(tb, (int, float))
    ) else None

    out = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "config": str(args.config),
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
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0 if a_res.get("success") and b_res.get("success") else 1


def main() -> None:
    """Run the async entry point and exit with its return code.

    Raises:
        SystemExit: Always, carrying the exit code from :func:`main_async`.
    """
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
