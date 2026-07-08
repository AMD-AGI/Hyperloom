#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run forge-fusion as a Hyperloom kernel-agent tool.

Mirrors ``forge_gemm_tuning.py`` at the kernel-agent tool layer: the orchestrator
writes an input JSON and calls this script; the autonomous fusion loop itself
lives in the standalone ``forge_fusion`` package (``forge-fusion run``: diagnose
-> discover(llm) -> author(few-shot) -> kernel-validate -> serving-smoke -> keep).

Unlike the GEMM wrapper (which relays forge-gemm-tune's own result sentinel),
forge-fusion emits a ``fusion_manifest.json``; this wrapper normalizes that into
the Hyperloom kernel-result contract (a ``FORGE_FUSION_RESULT_BEGIN/END`` stdout
sentinel + an on-disk ``result.json``) that ``run_fusion_handler`` parses. A KEPT
fusion already carries kernel-parity AND serving-smoke validation, so
``requires_e2e_validation`` is set for the orchestrator's integrate/re-baseline
gate to confirm the end-to-end gain and apply the patch + env flags.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULT_BEGIN = "FORGE_FUSION_RESULT_BEGIN"
RESULT_END = "FORGE_FUSION_RESULT_END"


def _truthy(val: Any) -> bool:
    """Interpret common truthy spellings from JSON or env strings."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _git_toplevel(path: str) -> str:
    """Best-effort git repo root for a source file (for integrate's patch apply)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(path).parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _load_input_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"--input-json must contain a JSON object: {path}")
    return data


def _add_opt(cmd: list[str], args: dict[str, Any], key: str, flag: str, *, required: bool = False) -> None:
    val = args.get(key)
    if val in (None, ""):
        if required:
            raise ValueError(f"{key} is required")
        return
    cmd.extend([flag, str(val)])


def _build_cmd(args: dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "-m", "forge_fusion.cli", "run"]
    _add_opt(cmd, args, "trace_path", "--trace", required=True)
    _add_opt(cmd, args, "model_path", "--model-path", required=True)
    _add_opt(cmd, args, "framework", "--framework", required=True)
    _add_opt(cmd, args, "output_dir", "--output-dir", required=True)
    _add_opt(cmd, args, "discover_mode", "--discover")
    _add_opt(cmd, args, "llm_model", "--llm-model")
    _add_opt(cmd, args, "max_turns", "--max-turns")
    _add_opt(cmd, args, "gpu", "--gpu")
    _add_opt(cmd, args, "decode_batch", "--decode-batch")
    _add_opt(cmd, args, "ab_isl", "--ab-isl")
    _add_opt(cmd, args, "ab_osl", "--ab-osl")
    _add_opt(cmd, args, "framework_root", "--framework-root")
    # Fusion authors ALL source-confirmed patterns together by default (the T3
    # ZAYA recipe stacks 3 fusions and the A/B measures the combined gain);
    # set fuse_all_confirmed=false to author only the top recipe.
    if bool(args.get("fuse_all_confirmed", True)):
        cmd.append("--fuse-all-confirmed")
    if not bool(args.get("author", True)):
        cmd.append("--no-author")
    if not bool(args.get("validate", True)):
        cmd.append("--no-validate")
    if _truthy(args.get("verbose", False)):
        cmd.append("--verbose")
    return cmd


def _normalize_manifest(output_dir: str, rc: int) -> dict[str, Any]:
    """Map forge-fusion's ``fusion_manifest.json`` -> Hyperloom result contract."""
    result: dict[str, Any] = {
        "status": "failed",
        "engine": "forge_fusion",
        "micro_decision": "failed",
        "decision": "REVERT",
        "kept": False,
        "kernel_speedup": None,
        "env_flags": {},
        "baseline_env_flags": {},
        "artifact_files": [],
        "patch": None,
        "requires_e2e_validation": False,
        "workspace": str(output_dir or ""),
    }
    manifest_path = Path(output_dir or ".") / "fusion_manifest.json"
    if not manifest_path.is_file():
        result["error"] = f"no fusion_manifest.json at {manifest_path} (forge-fusion rc={rc})"
        return result
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"fusion_manifest.json parse error: {exc!r}"
        return result

    loop = m.get("fusion_loop") or {}
    val = m.get("validation") or {}
    best = loop.get("best") or {}
    kept = bool(loop.get("kept") or val.get("kept"))
    speedup = best.get("kernel_speedup") or val.get("kernel_speedup")
    best_flags = str(loop.get("best_env_flag") or "").split()
    artifacts = m.get("artifacts") or {}
    changed = [c.get("path") for c in (artifacts.get("changes") or []) if c.get("path")]
    src_file = str((m.get("fusion") or {}).get("source_file") or "")

    result.update({
        "status": "ok" if kept else "complete",
        "micro_decision": "candidate" if kept else "no_improvement",
        "decision": "KEEP" if kept else "REVERT",
        "kept": kept,
        "kernel_speedup": speedup,
        # Fused arm = all confirmed flags ON; baseline arm = same flags OFF.
        "env_flags": {f: "1" for f in best_flags},
        "baseline_env_flags": {f: "0" for f in best_flags},
        "artifact_files": changed,
        "patch": artifacts.get("patch"),
        # For integrate's patch-apply path: primary edited source + its repo root.
        "source_file": src_file,
        "kernel_repo": _git_toplevel(src_file) if src_file else "",
        "best_pattern": loop.get("best_pattern"),
        "verdict": m.get("verdict"),
        # A KEPT fusion passed kernel parity + serving smoke; the orchestrator
        # still confirms the real e2e gain (and applies patch+env) via integrate.
        "requires_e2e_validation": kept,
    })
    return result


def _emit(result: dict[str, Any], output_dir: str) -> None:
    """Write result.json (disk fallback) + print the stdout sentinel."""
    if output_dir:
        try:
            (Path(output_dir) / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
        except OSError:
            pass
    print(f"\n{RESULT_BEGIN}\n{json.dumps(result, sort_keys=True)}\n{RESULT_END}", flush=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperloom wrapper for forge-fusion")
    p.add_argument("--input-json", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(list(argv or sys.argv[1:]))
        payload = _load_input_json(args.input_json)
        cmd = _build_cmd(payload)
    except Exception as exc:  # noqa: BLE001 - structured wrapper failure
        print(json.dumps({
            "status": "failed", "engine": "forge_fusion", "micro_decision": "failed",
            "decision": "REVERT", "kept": False,
            "error_class": exc.__class__.__name__, "error": repr(exc),
        }, sort_keys=True), flush=True)
        return 2

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    output_dir = str(payload.get("output_dir") or "")
    result = _normalize_manifest(output_dir, proc.returncode)
    _emit(result, output_dir)
    # Exit mirrors the forge-fusion subprocess: 0 when it ran (result carries the
    # KEEP/REVERT verdict); non-zero only when the subprocess itself failed.
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
