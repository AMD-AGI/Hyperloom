#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Straightforward kernel-patch APPLY + warm-serve E2E REMEASURE (no decision gate).

This is the "just apply it and tell me the E2E delta" path. It deliberately BYPASSES the
orchestrator's KEEP / REVERT / NEEDS_REVIEW decision machinery (``kernel_request_handlers``'s
integrate gate, the ``micro_speedup``/dead-band/parity arbitration). It does exactly three things:

  1. measure BASELINE end-to-end serving throughput (pristine source, warm server, N reps),
  2. APPLY the patch via :func:`apply_kernel_patch.apply_kernel_patch` — which already handles the
     aiter/Composable-Kernel ``.cu`` case (AITER_REBUILD, ``jit/build`` + ``cpp_itfs`` cache
     invalidation, fresh-rebuild verification) as well as Python overlays — then measure PATCHED
     throughput the same way,
  3. REVERT via :func:`apply_kernel_patch.revert_kernel_patch` (restore source; keep the patch file),
     and report the A/B delta.

No accept/reject verdict is emitted. The caller decides what to do with the numbers. Used by both
the GEAK v3 and GEAK v4 clean paired runs so every E2E number is measured the SAME way.

The serving stack is pluggable: ``--backend {sglang,vllm}`` selects how the server is launched and
benchmarked; for an aiter ``.cu`` patch the prebuilt fused ``module_aiter_core*.so`` is removed and
``AITER_REBUILD=1`` is exported so the edited kernel actually recompiles on the patched server.

Example:
  apply_and_bench.py \
    --patch-path  <final_patch.diff> \
    --target-file /sgl-workspace/aiter/csrc/kernels/quant_kernels.cu \
    --backup-root <work>/backup \
    --model /wekafs/models/meta-llama-Llama-3.1-8B-Instruct \
    --backend sglang --tp 1 --isl 1024 --osl 1024 --conc 64 --num-prompts 320 --reps 3 \
    --out-dir <work> --aiter-rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from apply_kernel_patch import apply_kernel_patch, revert_kernel_patch  # noqa: E402


def _log(out_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        (out_dir / "apply_and_bench.log").open("a", encoding="utf-8").write(line + "\n")
    except OSError:
        pass


def _find_benchmark_serving() -> str | None:
    """Locate HL's ``benchmark_serving.py`` (the canonical E2E driver)."""
    roots = [
        Path("/root/.cache/hyperloom/inferencex_local"),
        Path(os.environ.get("INFERENCEX_PATH", "/wekafs/InferenceX")),
    ]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("benchmark_serving.py"):
            return str(p)
    return None


def _aiter_prebuilt_so(aiter_root: Path) -> list[Path]:
    return list(aiter_root.rglob("module_aiter_core*.so"))


def _launch_server(backend: str, model: str, tp: int, port: int, gpu: str,
                   extra_env: dict[str, str], log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(extra_env)
    env["ROCR_VISIBLE_DEVICES"] = gpu
    env["HIP_VISIBLE_DEVICES"] = gpu
    if backend == "sglang":
        cmd = [
            sys.executable, "-m", "sglang.launch_server", f"--model-path={model}",
            "--host=0.0.0.0", f"--port={port}", "--trust-remote-code",
            f"--tensor-parallel-size={tp}", "--mem-fraction-static=0.8",
            "--context-length", "8192", "--watchdog-timeout", "1800",
        ]
        cwd = "/sgl-workspace/sglang"
    elif backend == "vllm":
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--host", "0.0.0.0", "--port", str(port), "--trust-remote-code",
            "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", "0.8",
        ]
        cwd = None
    else:
        raise SystemExit(f"unknown backend: {backend}")
    fh = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def _wait_health(proc: subprocess.Popen, port: int, out_dir: Path, tries: int = 70) -> bool:
    import urllib.request
    for _ in range(tries):
        time.sleep(12)
        if proc.poll() is not None:
            _log(out_dir, "server died during startup")
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            _log(out_dir, "server healthy")
            return True
        except Exception:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                _log(out_dir, "server healthy (v1/models)")
                return True
            except Exception:
                continue
    return False


def _bench_once(bs: str, model: str, port: int, isl: int, osl: int, conc: int,
                num_prompts: int, arm: str, rep: int, out_dir: Path) -> float | None:
    cmd = [
        sys.executable, bs, "--model", model, "--backend", "vllm",
        "--base-url", f"http://0.0.0.0:{port}", "--dataset-name", "random",
        "--random-input-len", str(isl), "--random-output-len", str(osl),
        "--random-range-ratio", "1", "--num-prompts", str(num_prompts),
        "--max-concurrency", str(conc), "--request-rate", "inf", "--ignore-eos",
        "--save-result", "--num-warmups", "8",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--result-dir", str(out_dir), "--result-filename", f"{arm}_rep{rep}.json",
    ]
    blog = (out_dir / f"bench_{arm}_rep{rep}.log").open("w", encoding="utf-8")
    subprocess.run(cmd, cwd=str(Path(bs).parent), stdout=blog, stderr=subprocess.STDOUT)
    res = out_dir / f"{arm}_rep{rep}.json"
    try:
        return float(json.loads(res.read_text())["output_throughput"])
    except Exception:
        return None


def _kill_servers(proc: subprocess.Popen | None, backend: str) -> None:
    if proc is not None:
        try:
            proc.send_signal(signal.SIGTERM)
            time.sleep(15)
        except Exception:
            pass
    pat = "sglang.launch_server" if backend == "sglang" else "vllm.entrypoints"
    subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(8)


def _serve_and_bench(arm: str, backend: str, model: str, tp: int, port: int, gpu: str,
                     isl: int, osl: int, conc: int, num_prompts: int, reps: int,
                     extra_env: dict[str, str], bs: str, out_dir: Path) -> dict[str, Any]:
    _log(out_dir, f"=== {arm.upper()} arm: launch server ===")
    proc = _launch_server(backend, model, tp, port, gpu, extra_env, out_dir / f"server_{arm}.log")
    if not _wait_health(proc, port, out_dir):
        _kill_servers(proc, backend)
        return {"arm": arm, "status": "server_failed", "reps": [], "median": None}
    reps_out: list[float] = []
    for r in range(1, reps + 1):
        t = _bench_once(bs, model, port, isl, osl, conc, num_prompts, arm, r, out_dir)
        _log(out_dir, f"{arm} rep{r} tput={t}")
        if t is not None:
            reps_out.append(t)
    _kill_servers(proc, backend)
    med = statistics.median(reps_out) if reps_out else None
    return {"arm": arm, "status": "ok" if reps_out else "no_results", "reps": reps_out, "median": med}


def apply_and_bench(
    *, patch_path: str, target_file: str, backup_root: str, model: str, backend: str,
    tp: int, port: int, gpu: str, isl: int, osl: int, conc: int, num_prompts: int, reps: int,
    out_dir: str, kernel_id: str = "", rebuild_command: str | None = None,
    aiter_rebuild: bool = False, skip_rebuild: bool = False,
) -> dict[str, Any]:
    """Apply a kernel patch and measure E2E throughput A/B — NO keep/revert/needs-review gate.

    Returns a result dict with baseline/patched medians + the throughput delta. Always reverts
    the source at the end (the patch FILE is kept). Handles aiter ``.cu`` rebuild when
    ``aiter_rebuild`` is set (removes the prebuilt fused ``.so`` + exports ``AITER_REBUILD=1``
    so the patched server recompiles the edited kernel; the apply step also invalidates the aiter
    jit/cpp_itfs caches via ``apply_kernel_patch``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bs = _find_benchmark_serving()
    if not bs:
        return {"status": "error", "error": "benchmark_serving.py not found"}
    target = Path(target_file)
    is_aiter_cu = "/aiter/" in str(target) and target.suffix in {".cu", ".cuh"}
    patched_env: dict[str, str] = {}
    if aiter_rebuild or is_aiter_cu:
        patched_env["AITER_REBUILD"] = "1"

    # ---- BASELINE (pristine) ----
    base = _serve_and_bench("baseline", backend, model, tp, port, gpu, isl, osl, conc,
                            num_prompts, reps, {}, bs, out)
    if base["status"] != "ok":
        return {"status": "baseline_failed", "baseline": base}

    # ---- APPLY (reuses apply_kernel_patch: aiter .cu rebuild + cache invalidation) ----
    _log(out, "=== APPLY patch (straightforward; no keep/revert/needs-review gate) ===")
    rb = rebuild_command
    apply_res = apply_kernel_patch(
        patch_path=patch_path, target_file=target_file, backup_root=backup_root,
        kernel_id=kernel_id or "apply_and_bench", rebuild_command=rb,
        skip_rebuild=skip_rebuild, allow_unknown_target=True,
    )
    _log(out, f"apply status={apply_res.get('status')} manifest={apply_res.get('manifest_path')}")
    if apply_res.get("status") != "ok":
        return {"status": "apply_failed", "baseline": base, "apply": apply_res}
    manifest_path = apply_res.get("manifest_path")
    # For aiter .cu: also drop the prebuilt fused module so the patched server re-JITs the edit.
    removed_so = []
    if aiter_rebuild or is_aiter_cu:
        aiter_root = Path("/sgl-workspace/aiter")
        for so in _aiter_prebuilt_so(aiter_root):
            try:
                so.unlink(); removed_so.append(str(so))
            except OSError:
                pass
        if removed_so:
            _log(out, f"removed prebuilt aiter .so ({len(removed_so)}) to force rebuild")

    # ---- PATCHED ----
    try:
        patched = _serve_and_bench("patched", backend, model, tp, port, gpu, isl, osl, conc,
                                   num_prompts, reps, patched_env, bs, out)
    finally:
        # ---- REVERT (restore source; keep the patch file) ----
        if manifest_path:
            rev = revert_kernel_patch(manifest_path)
            _log(out, f"revert status={rev.get('status')}")
        # belt-and-suspenders clean of the aiter source tree
        subprocess.run(["git", "-C", "/sgl-workspace/aiter", "checkout", "--", "csrc/"], check=False)

    b_med, p_med = base.get("median"), patched.get("median")
    delta_pct = None
    if b_med and p_med:
        delta_pct = (p_med - b_med) / b_med * 100.0
    result = {
        "status": "ok" if (b_med and p_med) else "patched_failed",
        "gate": "none (straightforward apply + remeasure; KEEP/REVERT/NEEDS_REVIEW bypassed)",
        "baseline_median_tok_s": b_med, "patched_median_tok_s": p_med,
        "delta_pct": delta_pct, "baseline_reps": base.get("reps"),
        "patched_reps": patched.get("reps"), "removed_prebuilt_so": removed_so,
        "manifest_path": manifest_path, "target_file": target_file, "patch_path": patch_path,
    }
    (out / "apply_and_bench_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(out, f"RESULT baseline={b_med} patched={p_med} delta={delta_pct}%")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply a kernel patch + warm-serve E2E remeasure (no decision gate)")
    ap.add_argument("--patch-path", required=True)
    ap.add_argument("--target-file", required=True)
    ap.add_argument("--backup-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=320)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kernel-id", default="")
    ap.add_argument("--rebuild-command", default="")
    ap.add_argument("--aiter-rebuild", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")
    a = ap.parse_args()
    res = apply_and_bench(
        patch_path=a.patch_path, target_file=a.target_file, backup_root=a.backup_root,
        model=a.model, backend=a.backend, tp=a.tp, port=a.port, gpu=a.gpu, isl=a.isl,
        osl=a.osl, conc=a.conc, num_prompts=a.num_prompts, reps=a.reps, out_dir=a.out_dir,
        kernel_id=a.kernel_id, rebuild_command=a.rebuild_command or None,
        aiter_rebuild=a.aiter_rebuild, skip_rebuild=a.skip_rebuild,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
