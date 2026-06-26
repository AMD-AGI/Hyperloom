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
        # Logging to disk is best-effort; the line already went to stdout above, so a
        # write failure (e.g. read-only/full out_dir) must not abort the measurement.
        pass


def _looks_like_diff(path: Path) -> bool:
    """True if the file is a unified diff / patch (vs a complete source file)."""
    if path.suffix in (".diff", ".patch"):
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return False
    return ("diff --git " in head) or ("\n--- " in head and "\n+++ " in head and "\n@@ " in head)


def _reconstruct_sources_from_diff(diff_path: Path, repo_root: Path, out_dir: Path) -> dict[str, Any]:
    """Reconstruct the BYTE-EXACT optimized source for EVERY file a diff touches.

    Apply the WHOLE diff (all hunks, every file — no scoping, nothing dropped) to a throwaway
    copy of the repo's CURRENT (committed) tree, then read back each resulting file. This yields
    the exact bytes the optimizer's patch produces, which the caller then deploys via
    ``apply_kernel_patch`` (the same full-source replace + backup/manifest path that the
    integration deploy in #681 uses). Two patch-deploy semantics collapse into one, and companion
    edits (generated drivers/headers/codegen shims) are preserved rather than filtered out.

    Done in an isolated ``git worktree`` of ``repo_root`` so the live tree is never mutated during
    reconstruction. Returns ``{status, files: {repo_rel_path: reconstructed_source_path}, error}``.
    """
    files: dict[str, str] = {}
    wt = out_dir / f"_recon_wt_{diff_path.stem}"
    # Use a detached worktree at HEAD so reconstruction is hermetic (live repo untouched).
    rm = subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", "-f", str(wt), "HEAD"],
        capture_output=True,
        text=True,
    )
    if rm.returncode != 0:
        return {"status": "failed", "error": f"git worktree add: {rm.stderr[:200]}"}
    try:
        applied = False
        for lvl in (1, 0, 2):
            chk = subprocess.run(
                ["git", "-C", str(wt), "apply", f"-p{lvl}", "--check", str(diff_path)], capture_output=True, text=True
            )
            if chk.returncode == 0:
                ap = subprocess.run(
                    ["git", "-C", str(wt), "apply", f"-p{lvl}", str(diff_path)], capture_output=True, text=True
                )
                if ap.returncode != 0:
                    return {"status": "failed", "error": f"git apply -p{lvl}: {ap.stderr[:200]}"}
                # numstat lists EVERY touched path (incl. new/companion files) — none dropped.
                ns = subprocess.run(
                    ["git", "-C", str(wt), "apply", f"-p{lvl}", "--numstat", str(diff_path)],
                    capture_output=True,
                    text=True,
                ).stdout
                rels = [ln.split("\t")[-1] for ln in ns.splitlines() if "\t" in ln]
                recon_dir = out_dir / f"_recon_src_{diff_path.stem}"
                recon_dir.mkdir(parents=True, exist_ok=True)
                for rel in rels:
                    src = wt / rel
                    if not src.is_file():  # a pure deletion — record so the caller can handle it
                        files[rel] = ""
                        continue
                    dst = recon_dir / rel.replace("/", "__")
                    dst.write_bytes(src.read_bytes())  # byte-exact
                    files[rel] = str(dst)
                applied = True
                _log(
                    out_dir,
                    f"reconstructed {len(files)} file(s) from diff -p{lvl}: "
                    f"{', '.join(r.split('/')[-1] for r in files)}",
                )
                break
        if not applied:
            return {"status": "failed", "error": "git apply: no -p level (0/1/2) applies this diff cleanly"}
        return {"status": "ok", "files": files}
    finally:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "-f", str(wt)], capture_output=True, text=True
        )


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


def _launch_server(
    backend: str, model: str, tp: int, port: int, gpu: str, extra_env: dict[str, str], log_path: Path
) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(extra_env)
    env["ROCR_VISIBLE_DEVICES"] = gpu
    env["HIP_VISIBLE_DEVICES"] = gpu
    if backend == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            f"--model-path={model}",
            "--host=0.0.0.0",
            f"--port={port}",
            "--trust-remote-code",
            f"--tensor-parallel-size={tp}",
            "--mem-fraction-static=0.8",
            "--context-length",
            "8192",
            "--watchdog-timeout",
            "1800",
        ]
        cwd = "/sgl-workspace/sglang"
    elif backend == "vllm":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--trust-remote-code",
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            "0.8",
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


def _bench_once(
    bs: str,
    model: str,
    port: int,
    isl: int,
    osl: int,
    conc: int,
    num_prompts: int,
    arm: str,
    rep: int,
    out_dir: Path,
    seed: int,
) -> dict[str, float] | None:
    # Fixed --seed so BOTH arms benchmark the IDENTICAL random prompt set: removes
    # dataset-draw variance from the A/B so the delta reflects the kernel, not the prompts.
    cmd = [
        sys.executable,
        bs,
        "--model",
        model,
        "--backend",
        "vllm",
        "--base-url",
        f"http://0.0.0.0:{port}",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--random-range-ratio",
        "1",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups",
        "8",
        "--seed",
        str(seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        str(out_dir),
        "--result-filename",
        f"{arm}_rep{rep}.json",
    ]
    blog = (out_dir / f"bench_{arm}_rep{rep}.log").open("w", encoding="utf-8")
    subprocess.run(cmd, cwd=str(Path(bs).parent), stdout=blog, stderr=subprocess.STDOUT)
    res = out_dir / f"{arm}_rep{rep}.json"
    try:
        d = json.loads(res.read_text())
        # output_throughput is the headline; tpot/itl are the decode-bound-sensitive
        # signals (lower=better) that show a kernel delta when aggregate tput is noisy.
        out: dict[str, float] = {"output_throughput": float(d["output_throughput"])}
        for k in ("median_tpot_ms", "mean_tpot_ms", "median_itl_ms", "mean_itl_ms"):
            if d.get(k) is not None:
                out[k] = float(d[k])
        return out
    except Exception:
        return None


def _kill_servers(proc: subprocess.Popen | None, backend: str) -> None:
    if proc is not None:
        try:
            proc.send_signal(signal.SIGTERM)
            time.sleep(15)
        except (ProcessLookupError, OSError):
            # The server may already be gone (crashed / reaped); the pkill sweep below is the
            # authoritative teardown, so a failed SIGTERM here is non-fatal.
            pass
    pat = "sglang.launch_server" if backend == "sglang" else "vllm.entrypoints"
    subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(8)


def _spread(xs: list[float]) -> dict[str, float | None]:
    """median + p25/p75 + stdev for a sample (None-safe for small n)."""
    if not xs:
        return {"median": None, "p25": None, "p75": None, "stdev": None, "n": 0}
    s = sorted(xs)
    q = statistics.quantiles(s, n=4) if len(s) >= 2 else [s[0], s[0], s[0]]
    return {
        "median": statistics.median(s),
        "p25": q[0],
        "p75": q[2],
        "stdev": statistics.stdev(s) if len(s) >= 2 else 0.0,
        "n": len(s),
    }


def _serve_and_bench(
    arm: str,
    backend: str,
    model: str,
    tp: int,
    port: int,
    gpu: str,
    isl: int,
    osl: int,
    conc: int,
    num_prompts: int,
    reps: int,
    extra_env: dict[str, str],
    bs: str,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    _log(out_dir, f"=== {arm.upper()} arm: launch server ===")
    proc = _launch_server(backend, model, tp, port, gpu, extra_env, out_dir / f"server_{arm}.log")
    if not _wait_health(proc, port, out_dir):
        _kill_servers(proc, backend)
        return {"arm": arm, "status": "server_failed", "reps": [], "median": None}
    # Untimed WARMUP pass: the server JIT-compiles / captures CUDA graphs on its first
    # real load, so the first timed run reads ~10-15% low and skews the median. Run one
    # full benchmark and DISCARD it before the timed reps (open-source-standard warmup).
    _bench_once(bs, model, port, isl, osl, conc, num_prompts, arm, 0, out_dir, seed)
    _log(out_dir, f"{arm} warmup pass done (discarded)")
    reps_out: list[float] = []  # output_throughput per timed rep
    tpot_out: list[float] = []  # median_tpot_ms per timed rep (decode latency)
    for r in range(1, reps + 1):
        m = _bench_once(bs, model, port, isl, osl, conc, num_prompts, arm, r, out_dir, seed)
        tput = m.get("output_throughput") if m else None
        _log(out_dir, f"{arm} rep{r} tput={tput}")
        if tput is not None:
            reps_out.append(tput)
            if m.get("median_tpot_ms") is not None:
                tpot_out.append(m["median_tpot_ms"])
    _kill_servers(proc, backend)
    tput_spread = _spread(reps_out)
    return {
        "arm": arm,
        "status": "ok" if reps_out else "no_results",
        "reps": reps_out,
        "median": tput_spread["median"],
        "tput_spread": tput_spread,
        "tpot_reps_ms": tpot_out,
        "tpot_spread_ms": _spread(tpot_out),
    }


def _engagement_proof(server_log: Path, target: Path, is_aiter_cu: bool) -> dict[str, Any]:
    """Was the patched kernel ACTUALLY on the live serving path? (trust, not policy).

    A throughput delta is meaningless if the patch never engaged. We DO NOT accept/reject on this
    — we only report it so a ~0% result is distinguishable from "patch silently not applied". The
    signal is backend/kernel-generic:
      * aiter `.cu` patch: the patched server must have REBUILT the edited kernel (aiter JIT build
        markers in the server log, e.g. "start build [module_..." / "build ... .so"), since we
        removed the prebuilt `.so` + set AITER_REBUILD=1. A rebuild => the edited source compiled.
      * aiter GEMM-DB tune: "is tuned on cu_num" hits > 0.
      * generic: the kernel source's stem appearing in a build/load line.
    Absence of any marker => engaged=False (uncertain): the delta should be treated with suspicion.
    """
    out: dict[str, Any] = {"engaged": False, "reason": "no server log", "markers": []}
    try:
        text = server_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    markers: list[str] = []
    stem = target.stem  # e.g. "quant_kernels", "attention_ragged"
    # aiter JIT rebuild evidence (the strongest proof for a .cu patch we forced to recompile)
    for pat in ("start build [module_", "] build ", "ninja", ".so", "hipcc"):
        if pat in text:
            markers.append(f"jit_build:{pat.strip()}")
            break
    if "is tuned on cu_num" in text:
        n = text.count("is tuned on cu_num")
        markers.append(f"aiter_db_tuned_hits={n}")
    if stem and stem in text:
        markers.append(f"source_stem:{stem}")
    tuned_hits = text.count("is tuned on cu_num")
    engaged = bool(markers) and (
        not is_aiter_cu or any(m.startswith("jit_build") or "tuned_hits" in m for m in markers)
    )
    out.update(
        {
            "engaged": engaged,
            "reason": (
                "rebuild/tune markers present"
                if engaged
                else "no rebuild/tune marker in server log — patch may not have engaged"
            ),
            "markers": markers,
            "aiter_db_tuned_hits": tuned_hits,
        }
    )
    return out


def apply_and_bench(
    *,
    pairs: list[tuple[str, str]] | None = None,
    patch_path: str | None = None,
    target_file: str | None = None,
    backup_root: str,
    model: str,
    backend: str = "sglang",
    tp: int = 1,
    port: int = 8890,
    gpu: str = "0",
    isl: int = 1024,
    osl: int = 1024,
    conc: int = 64,
    num_prompts: int = 320,
    reps: int = 5,
    out_dir: str,
    kernel_id: str = "",
    rebuild_command: str | None = None,
    aiter_rebuild: bool = False,
    skip_rebuild: bool = False,
    seed: int = 1234,
) -> dict[str, Any]:
    """Apply ONE OR MORE kernel patches together and measure E2E throughput A/B — NO gate.

    Pass either a single (`patch_path`,`target_file`) or a list of `pairs` [(patch, target), ...].
    Multiple pairs are applied to the SAME patched server, so the A/B reports the COMBINED effect of
    all optimized patches (the "apply all patches -> final E2E" number). Always reverts every patched
    source at the end (patch FILES kept). Handles aiter ``.cu`` rebuild (removes prebuilt fused ``.so``
    + AITER_REBUILD=1; apply step also invalidates aiter jit/cpp_itfs caches via ``apply_kernel_patch``).
    No keep/revert/needs-review verdict — that policy is the caller's job.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bs = _find_benchmark_serving()
    if not bs:
        return {"status": "error", "error": "benchmark_serving.py not found"}
    # Normalize to a list of (patch, target) pairs (single-pair back-compat).
    if not pairs:
        if not (patch_path and target_file):
            return {"status": "error", "error": "need pairs=[(patch,target),...] or patch_path+target_file"}
        pairs = [(patch_path, target_file)]
    targets = [Path(t) for _, t in pairs]
    any_aiter_cu = any("/aiter/" in str(t) and t.suffix in {".cu", ".cuh"} for t in targets)
    patched_env: dict[str, str] = {}
    if aiter_rebuild or any_aiter_cu:
        patched_env["AITER_REBUILD"] = "1"

    # ---- BASELINE (pristine) ----
    base = _serve_and_bench(
        "baseline", backend, model, tp, port, gpu, isl, osl, conc, num_prompts, reps, {}, bs, out, seed
    )
    if base["status"] != "ok":
        return {"status": "baseline_failed", "baseline": base}

    # ---- APPLY all pairs. Each pair's patch may be a FULL SOURCE file (apply_kernel_patch:
    # backup + aiter .cu rebuild + cache invalidation) OR a unified DIFF/.patch (git apply,
    # multi-file). We auto-detect so the primitive handles patch/diff/full-source uniformly. ----
    # ONE deploy mechanism for everything: a diff is reconstructed to BYTE-EXACT full source(s)
    # (all files, nothing scoped/dropped), then EVERY file — diff-derived or already-full-source —
    # is deployed via apply_kernel_patch (backup + aiter .cu rebuild + cache invalidation + manifest),
    # the SAME path the integration deploy in #681 uses. Revert is uniformly via the manifests.
    _log(out, f"=== APPLY {len(pairs)} patch(es) (byte-exact via apply_kernel_patch; no gate) ===")
    manifests: list[str] = []  # every deploy -> revert via revert_kernel_patch(manifest)
    applied: list[dict[str, Any]] = []

    def _revert_all() -> None:
        for m in reversed(manifests):
            revert_kernel_patch(m)

    def _deploy_full_source(src_path: str, target: str, tag: str) -> bool:
        """apply_kernel_patch a complete-source file onto target; record manifest. Returns ok."""
        res = apply_kernel_patch(
            patch_path=src_path,
            target_file=target,
            backup_root=backup_root,
            kernel_id=f"{kernel_id or 'apply_and_bench'}_{tag}",
            rebuild_command=rebuild_command,
            skip_rebuild=skip_rebuild,
            allow_unknown_target=True,
        )
        _log(out, f"deploy {Path(target).name} <- {Path(src_path).name} status={res.get('status')}")
        applied.append(
            {"target": target, "source": src_path, "status": res.get("status"), "manifest": res.get("manifest_path")}
        )
        if res.get("status") == "ok" and res.get("manifest_path"):
            manifests.append(res["manifest_path"])
        return res.get("status") == "ok"

    for i, (pp, tf) in enumerate(pairs):
        if _looks_like_diff(Path(pp)):
            # Reconstruct byte-exact full source for EVERY file the diff touches (companion files
            # preserved), then deploy each via apply_kernel_patch. The declared target (tf) is
            # deployed to its real path; any additional reconstructed file is deployed in place
            # (resolved against the repo root that owns the target).
            repo_root = Path("/sgl-workspace/aiter") if "/aiter/" in str(tf) else Path(tf).parents[2]
            rec = _reconstruct_sources_from_diff(Path(pp), repo_root, out)
            if rec.get("status") != "ok":
                _revert_all()
                return {"status": "apply_failed", "baseline": base, "applied": applied, "error": rec.get("error")}
            for rel, src in rec["files"].items():
                if not src:  # pure deletion in the diff — revert-safe skip (rare for kernel patches)
                    _log(out, f"diff deletes {rel} — skipping deploy (not a source replacement)")
                    continue
                tgt = str(tf) if Path(rel).name == Path(tf).name else str(repo_root / rel)
                if not _deploy_full_source(src, tgt, f"{i}_{Path(rel).name}"):
                    _revert_all()
                    return {
                        "status": "apply_failed",
                        "baseline": base,
                        "applied": applied,
                        "error": f"deploy failed for {rel}",
                    }
        else:
            if not _deploy_full_source(pp, tf, str(i)):
                _revert_all()
                return {"status": "apply_failed", "baseline": base, "applied": applied, "error": "deploy failed"}
    # For aiter .cu: drop the prebuilt fused module so the patched server re-JITs the edits.
    removed_so = []
    if aiter_rebuild or any_aiter_cu:
        for so in _aiter_prebuilt_so(Path("/sgl-workspace/aiter")):
            try:
                so.unlink()
                removed_so.append(str(so))
            except OSError:
                # A prebuilt .so we can't remove (already gone / perms) just means aiter may
                # reuse it; AITER_REBUILD=1 + the jit cache invalidation in apply_kernel_patch
                # still force a recompile, so this is non-fatal.
                pass
        if removed_so:
            _log(out, f"removed prebuilt aiter .so ({len(removed_so)}) to force rebuild")

    # ---- PATCHED (all patches live) ----
    try:
        patched = _serve_and_bench(
            "patched", backend, model, tp, port, gpu, isl, osl, conc, num_prompts, reps, patched_env, bs, out, seed
        )
        # ---- ENGAGEMENT PROOF (trust, not policy): were the patched kernels on the live path?
        # Reported per target so a ~0% delta is distinguishable from "patch(es) didn't engage".
        engagement = [
            dict(
                target=str(t),
                **_engagement_proof(out / "server_patched.log", t, "/aiter/" in str(t) and t.suffix in {".cu", ".cuh"}),
            )
            for t in targets
        ]
        _log(out, f"engagement_proof: {[(e['target'].split('/')[-1], e['engaged']) for e in engagement]}")
    finally:
        # ---- REVERT every patched source (full-source manifests + diff touched-paths). Keep patch files. ----
        _revert_all()
        _log(out, f"reverted {len(manifests)} deploy(s) via manifest")

    b_med, p_med = base.get("median"), patched.get("median")
    delta_pct = (p_med - b_med) / b_med * 100.0 if (b_med and p_med) else None
    # Throughput significance: a delta is meaningful only if it clears the measurement
    # noise. Use the arms' [p25,p75] spread — overlapping IQRs => "within noise" (flat),
    # not a real regression/win. Avoids over-reading a sub-% delta as signal.
    bs_sp, ps_sp = base.get("tput_spread", {}), patched.get("tput_spread", {})
    significant = None
    if all(bs_sp.get(k) is not None for k in ("p25", "p75")) and all(ps_sp.get(k) is not None for k in ("p25", "p75")):
        # non-overlapping IQRs => significant
        significant = (ps_sp["p25"] > bs_sp["p75"]) or (ps_sp["p75"] < bs_sp["p25"])
    # TPOT (decode latency, lower=better): delta on the decode-bound-sensitive signal.
    b_tpot = base.get("tpot_spread_ms", {}).get("median")
    p_tpot = patched.get("tpot_spread_ms", {}).get("median")
    tpot_delta_pct = (p_tpot - b_tpot) / b_tpot * 100.0 if (b_tpot and p_tpot) else None
    result = {
        "status": "ok" if (b_med and p_med) else "patched_failed",
        "gate": "none (straightforward apply + remeasure; KEEP/REVERT/NEEDS_REVIEW bypassed; policy is the caller's job)",
        "combined": len(pairs) > 1,
        "baseline_median_tok_s": b_med,
        "patched_median_tok_s": p_med,
        "delta_pct": delta_pct,
        "significant": significant,  # None=insufficient reps; False=within noise (flat); True=clears IQR
        "seed": seed,
        "reps": reps,
        "warmup_discarded": True,
        "baseline_tput_spread": bs_sp,
        "patched_tput_spread": ps_sp,
        "tpot_delta_pct": tpot_delta_pct,  # negative = faster decode (good)
        "baseline_tpot_spread_ms": base.get("tpot_spread_ms"),
        "patched_tpot_spread_ms": patched.get("tpot_spread_ms"),
        "baseline_reps": base.get("reps"),
        "patched_reps": patched.get("reps"),
        "removed_prebuilt_so": removed_so,
        "engagement_proof": engagement,
        "applied": applied,
        "targets": [str(t) for t in targets],
    }
    (out / "apply_and_bench_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(
        out,
        f"RESULT baseline={b_med} patched={p_med} delta={delta_pct}% "
        f"significant={significant} tpot_delta={tpot_delta_pct}% (combined={len(pairs) > 1})",
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply one or more kernel patches + warm-serve E2E remeasure (no decision gate)"
    )
    ap.add_argument("--patch-path", help="single-patch mode: full optimized source (or complete-file patch)")
    ap.add_argument("--target-file", help="single-patch mode: file to replace")
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="PATCH:TARGET",
        help="multi-patch mode (repeatable): 'patch_path:target_file'. Apply ALL together "
        "to one patched server -> COMBINED E2E (the 'apply all optimized patches' number).",
    )
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
    ap.add_argument(
        "--reps",
        type=int,
        default=5,
        help="timed reps per arm (a separate untimed warmup pass is always discarded first)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random-dataset seed; both arms use it so they benchmark identical prompts",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kernel-id", default="")
    ap.add_argument("--rebuild-command", default="")
    ap.add_argument("--aiter-rebuild", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")
    a = ap.parse_args()
    # Build the (patch, target) pair list: --pair entries (rsplit on last ':' so absolute paths
    # with no ':' work) plus the single --patch-path/--target-file for back-compat.
    pairs: list[tuple[str, str]] = []
    for p in a.pair:
        if ":" not in p:
            raise SystemExit(f"--pair must be 'patch:target', got: {p}")
        patch, tgt = p.rsplit(":", 1)
        pairs.append((patch, tgt))
    if a.patch_path and a.target_file:
        pairs.append((a.patch_path, a.target_file))
    if not pairs:
        raise SystemExit("need --pair PATCH:TARGET (repeatable) or --patch-path + --target-file")
    res = apply_and_bench(
        pairs=pairs,
        backup_root=a.backup_root,
        model=a.model,
        backend=a.backend,
        tp=a.tp,
        port=a.port,
        gpu=a.gpu,
        isl=a.isl,
        osl=a.osl,
        conc=a.conc,
        num_prompts=a.num_prompts,
        reps=a.reps,
        out_dir=a.out_dir,
        kernel_id=a.kernel_id,
        rebuild_command=a.rebuild_command or None,
        aiter_rebuild=a.aiter_rebuild,
        skip_rebuild=a.skip_rebuild,
        seed=a.seed,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
