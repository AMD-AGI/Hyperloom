#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-patch APPLY + warm-serve E2E REMEASURE (no decision gate).

Bypasses the orchestrator's KEEP / REVERT / NEEDS_REVIEW machinery and does three things:

  1. measure BASELINE end-to-end serving throughput (pristine source, warm server, N reps),
  2. APPLY the patch via :func:`apply_kernel_patch.apply_kernel_patch`, then measure
     PATCHED throughput the same way,
  3. REVERT via :func:`apply_kernel_patch.revert_kernel_patch` and report the A/B delta.

No accept/reject verdict is emitted; the caller decides. The serving stack is pluggable
via ``--backend {sglang,vllm}``; for an aiter ``.cu`` patch the prebuilt fused
``module_aiter_core*.so`` is removed and ``AITER_REBUILD=1`` is exported so the edited
kernel recompiles on the patched server.
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
        # Logging to disk is best-effort.
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


def _diff_unsupported_ops(diff_text: str) -> list[str]:
    """Detect patch operations that full-source replacement CANNOT represent.

    Full-source replace expresses modify + add but not delete / rename / copy / mode-only /
    binary changes. Returns a list of unsupported-op descriptions (empty = all good).
    """
    bad: list[str] = []
    for ln in diff_text.splitlines():
        if ln.startswith("deleted file mode "):
            bad.append("delete")
        elif ln.startswith("rename from ") or ln.startswith("rename to "):
            bad.append("rename")
        elif ln.startswith("copy from ") or ln.startswith("copy to "):
            bad.append("copy")
        elif ln.startswith("old mode ") or ln.startswith("new mode "):
            bad.append("mode-change")
        elif ln.startswith("GIT binary patch") or ln.startswith("Binary files "):
            bad.append("binary")
    return sorted(set(bad))


def _reconstruct_sources_from_diff(diff_path: Path, repo_root: Path, out_dir: Path) -> dict[str, Any]:
    """Reconstruct the byte-exact optimized source for every file a diff touches.

    Applies the diff to a throwaway copy of the repo's committed tree and reads back each
    resulting file. delete / rename / copy / mode-only / binary ops are detected up front
    (the function fails). Done in an isolated ``git worktree`` so the live tree is never
    mutated. Returns ``{status, files: {repo_rel_path: reconstructed_source_path}, error}``.
    """
    try:
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "failed", "error": f"cannot read diff: {exc}"}
    unsupported = _diff_unsupported_ops(diff_text)
    if unsupported:
        return {
            "status": "failed",
            "error": f"diff contains unsupported op(s) for full-source deploy: {', '.join(unsupported)} "
            f"— kernel patches must be modify/add only (use the integration snapshot path for these)",
        }
    files: dict[str, str] = {}
    wt = out_dir / f"_recon_wt_{diff_path.stem}"
    # Detached worktree at HEAD for hermetic reconstruction.
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
                # numstat lists every touched path (incl. new/companion files).
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
                    if not src.is_file():
                        return {"status": "failed", "error": f"reconstructed path missing after apply: {rel}"}
                    dst = recon_dir / rel.replace("/", "__")
                    dst.write_bytes(src.read_bytes())
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
        Path(os.environ.get("INFERENCEX_PATH", "/opt/InferenceX")),
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
            "0.0.0.0",  # nosec B104 - benchmark server must accept local/container probes.
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
    # Own process group (POSIX) so teardown can killpg just this server's tree.
    session_kwargs = {"start_new_session": True} if os.name == "posix" else {}
    return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, **session_kwargs)


def _wait_health(proc: subprocess.Popen, port: int, out_dir: Path, tries: int = 70) -> bool:
    import urllib.request

    for _ in range(tries):
        time.sleep(12)
        if proc.poll() is not None:
            _log(out_dir, "server died during startup")
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)  # nosec B310 - fixed loopback health check.
            _log(out_dir, "server healthy")
            return True
        except Exception:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)  # nosec B310 - fixed loopback health check.
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
    # Fixed --seed so both arms benchmark the identical random prompt set.
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
        # output_throughput is the headline; tpot/itl are decode-latency signals (lower=better).
        out: dict[str, float] = {"output_throughput": float(d["output_throughput"])}
        # P99 tail latencies (the --percentile-metrics request already emits them)
        # alongside the median/mean decode-latency signals.
        for k in (
            "median_tpot_ms", "mean_tpot_ms", "median_itl_ms", "mean_itl_ms",
            "p99_tpot_ms", "p99_itl_ms", "p99_e2el_ms", "p99_ttft_ms",
        ):
            if d.get(k) is not None:
                out[k] = float(d[k])
        return out
    except Exception:
        # No parseable result: dropped sample (caller filters None).
        return None


# <20 GiB used => the arm's GPUs are considered clean for the next launch.
_VRAM_DRAIN_THRESHOLD_MB = 20_000.0
_VRAM_DRAIN_TIMEOUT_S = 60.0


def _reap_vllm_orphans(out_dir: Path | None = None) -> None:
    """Reap orphaned vLLM engine subprocesses that escape the server's process group.

    vLLM spawns its engine worker as a separate ``VLLM::EngineCore`` (and, under TP,
    ``VLLM::Worker``) process via multiprocessing ``spawn`` in a *fresh session*, so it
    lives OUTSIDE the api_server's process group and survives the ``killpg`` above. Left
    behind, it squats on GPU VRAM (observed >280 GiB) and poisons the next arm's
    measurement. This is a targeted ``pkill`` of those well-known process names only.

    Best-effort, POSIX-only, never raises into the caller. The ``[V]`` bracket makes the
    pattern a regex that matches the live ``VLLM::...`` process while NOT matching pkill's
    own ``argv`` (which contains the literal ``[V]LLM::...``) — otherwise pkill kills
    itself. Disable with APPLY_BENCH_NO_ORPHAN_REAP=1 on shared multi-tenant hosts.
    """
    if os.name != "posix":
        # os.killpg / pkill are POSIX-only; nothing to do off-POSIX.
        return
    if os.environ.get("APPLY_BENCH_NO_ORPHAN_REAP") == "1":
        return
    for pat in ("[V]LLM::EngineCore", "[V]LLM::Worker"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat], check=False, timeout=15)
        except (OSError, subprocess.SubprocessError):
            # Reaping is advisory; a missing/failing pkill must never break teardown.
            if out_dir is not None:
                _log(out_dir, f"orphan reap skipped (pkill unavailable) for {pat}")


def _wait_vram_drain(
    gpu: str,
    out_dir: Path | None = None,
    threshold_mb: float = _VRAM_DRAIN_THRESHOLD_MB,
    timeout_s: float = _VRAM_DRAIN_TIMEOUT_S,
) -> float | None:
    """Poll rocm-smi until the arm's GPUs fall below ``threshold_mb`` used (or timeout).

    Guards the next arm against a slow-releasing (or freshly reaped) server still holding
    VRAM. Best-effort: returns the last observed used-MiB, or ``None`` when rocm-smi is
    unavailable (which short-circuits immediately so non-GPU hosts are unaffected). Never
    raises.
    """
    deadline = time.time() + timeout_s
    used = _gpu_vram_used_mb(gpu)
    while used is not None and used > threshold_mb and time.time() < deadline:
        if out_dir is not None:
            _log(out_dir, f"waiting for VRAM to drain: {used:.0f} MiB used > {threshold_mb:.0f} MiB")
        time.sleep(3)
        used = _gpu_vram_used_mb(gpu)
    return used


def _kill_servers(
    proc: subprocess.Popen | None,
    backend: str,
    gpu: str | None = None,
    out_dir: Path | None = None,
) -> None:
    """Tear down only the server we spawned (its process group) — never a global sweep.

    The server leads its own process group, so ``killpg`` reaps the whole tree without
    touching other tenants: SIGTERM, grace, then SIGKILL survivors. A broad
    ``pkill -f sglang/vllm`` runs only when APPLY_BENCH_PKILL_SWEEP=1.

    vLLM is the exception: its ``VLLM::EngineCore`` / ``VLLM::Worker`` escape the process
    group (spawned in a fresh session), so for the ``vllm`` backend we additionally reap
    those orphans by name (:func:`_reap_vllm_orphans`) and, when ``gpu`` is known, wait for
    VRAM to actually drain before returning — both best-effort and additive.
    """
    if proc is not None:
        if os.name == "posix":
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Group already gone.
                    pass
                for _ in range(15):  # up to ~15s grace for a clean shutdown
                    if proc.poll() is not None:
                        break
                    time.sleep(1)
                try:
                    os.killpg(pgid, signal.SIGKILL)  # reap survivors
                except (ProcessLookupError, OSError):
                    # Whole group already reaped.
                    pass
        else:  # non-POSIX: best-effort single-process teardown
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    # proc already dead.
                    pass
        try:
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            # Final reap is best-effort.
            pass
    # Explicit single-tenant blunt sweep (off by default).
    if os.environ.get("APPLY_BENCH_PKILL_SWEEP") == "1":
        pat = "sglang.launch_server" if backend == "sglang" else "vllm.entrypoints"
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    # vLLM's engine worker escapes the process group — reap it explicitly so it can't
    # squat VRAM and poison the next arm. (No-op for sglang / off-POSIX.)
    if backend == "vllm":
        _reap_vllm_orphans(out_dir)
    time.sleep(5)
    # Confirm the driver actually released VRAM before the next launch (best-effort).
    if backend == "vllm" and gpu:
        _wait_vram_drain(gpu, out_dir)


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


def _gpu_vram_used_mb(gpu_ids: str) -> float | None:
    """Best-effort used VRAM (MiB) summed over the arm's GPUs, via rocm-smi.

    Sampled while the server is up (weights + KV cache resident) as a proxy for
    peak serving VRAM. Returns ``None`` on any probe/parse failure (never raises)
    so the ABBA result degrades gracefully when rocm-smi is unavailable.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    wanted = {g.strip() for g in (gpu_ids or "").split(",") if g.strip()}
    total = 0.0
    found = False
    for key, info in data.items():
        idx = "".join(ch for ch in str(key) if ch.isdigit())
        if wanted and idx not in wanted:
            continue
        if not isinstance(info, dict):
            continue
        used_key = next(
            (k for k in info if "used" in k.lower() and ("vram" in k.lower() or "memory" in k.lower())),
            None,
        )
        if used_key is None:
            continue
        # Only convert byte-denominated fields (e.g. "VRAM Total Used Memory (B)").
        # rocm-smi key/unit naming varies across builds; if the matched field is
        # not clearly in bytes we skip it rather than emit an off-by-1000s MiB
        # number -- a wrong-but-plausible value is worse than a graceful None.
        kl = used_key.lower()
        if "(b)" not in kl and "byte" not in kl:
            continue
        try:
            total += float(str(info[used_key]).strip()) / (1024.0 * 1024.0)  # bytes -> MiB
            found = True
        except (ValueError, TypeError):
            continue
    return round(total, 1) if found else None


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
    # Preclean: reap any orphaned vLLM engine worker left by a prior arm and record the
    # pre-launch VRAM so a leaked EngineCore squatting memory can't silently skew this arm.
    if backend == "vllm":
        _reap_vllm_orphans(out_dir)
        pre_vram = _wait_vram_drain(gpu, out_dir)
        if pre_vram is not None:
            _log(out_dir, f"pre-launch VRAM: {pre_vram:.0f} MiB used")
    proc = _launch_server(backend, model, tp, port, gpu, extra_env, out_dir / f"server_{arm}.log")
    if not _wait_health(proc, port, out_dir):
        _kill_servers(proc, backend, gpu, out_dir)
        return {"arm": arm, "status": "server_failed", "reps": [], "median": None}
    # Untimed warmup pass (server JIT-compiles / captures CUDA graphs on first load).
    _bench_once(bs, model, port, isl, osl, conc, num_prompts, arm, 0, out_dir, seed)
    _log(out_dir, f"{arm} warmup pass done (discarded)")
    reps_out: list[float] = []  # output_throughput per timed rep
    tpot_out: list[float] = []  # median_tpot_ms per timed rep (decode latency)
    p99_tpot_out: list[float] = []  # p99_tpot_ms per timed rep (tail decode latency)
    for r in range(1, reps + 1):
        m = _bench_once(bs, model, port, isl, osl, conc, num_prompts, arm, r, out_dir, seed)
        tput = m.get("output_throughput") if m else None
        _log(out_dir, f"{arm} rep{r} tput={tput}")
        if tput is not None:
            reps_out.append(tput)
            if m.get("median_tpot_ms") is not None:
                tpot_out.append(m["median_tpot_ms"])
            if m.get("p99_tpot_ms") is not None:
                p99_tpot_out.append(m["p99_tpot_ms"])
    # Peak serving VRAM (weights + KV cache), sampled before teardown.
    vram_used_mb = _gpu_vram_used_mb(gpu)
    _kill_servers(proc, backend, gpu, out_dir)
    tput_spread = _spread(reps_out)
    return {
        "arm": arm,
        "status": "ok" if reps_out else "no_results",
        "reps": reps_out,
        "median": tput_spread["median"],
        "tput_spread": tput_spread,
        "tpot_reps_ms": tpot_out,
        "tpot_spread_ms": _spread(tpot_out),
        "p99_tpot_reps_ms": p99_tpot_out,
        "p99_tpot_spread_ms": _spread(p99_tpot_out),
        "vram_used_mb": vram_used_mb,
    }


def _engagement_proof(server_log: Path, target: Path, is_aiter_cu: bool) -> dict[str, Any]:
    """Was the patched kernel actually on the live serving path? (trust, not policy).

    Reported (not enforced) so a ~0% result is distinguishable from "patch not applied". Signals:
      * aiter `.cu` patch: aiter JIT build markers in the server log.
      * aiter GEMM-DB tune: "is tuned on cu_num" hits > 0.
      * generic: the kernel source's stem appearing in a build/load line.
    Absence of any marker => engaged=False (uncertain).
    """
    out: dict[str, Any] = {"engaged": False, "reason": "no server log", "markers": []}
    try:
        text = server_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    markers: list[str] = []
    stem = target.stem
    # aiter JIT rebuild evidence.
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
    """Apply one or more kernel patches together and measure E2E throughput A/B — no gate.

    Pass either a single (`patch_path`,`target_file`) or a list of `pairs` [(patch, target), ...].
    Multiple pairs are applied to the same patched server, so the A/B reports their combined
    effect. Always reverts every patched source at the end (patch files kept). Handles aiter
    ``.cu`` rebuild (removes prebuilt fused ``.so`` + AITER_REBUILD=1). No verdict.
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

    # ---- APPLY all pairs. Each patch is a full-source file or a unified diff (auto-detected);
    # a diff is reconstructed to byte-exact source and deployed via apply_kernel_patch. ----
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
            # Reconstruct byte-exact source for every file the diff touches, then deploy each.
            repo_root = Path("/sgl-workspace/aiter") if "/aiter/" in str(tf) else Path(tf).parents[2]
            rec = _reconstruct_sources_from_diff(Path(pp), repo_root, out)
            if rec.get("status") != "ok":
                _revert_all()
                return {"status": "apply_failed", "baseline": base, "applied": applied, "error": rec.get("error")}
            for rel, src in rec["files"].items():
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
    # aiter .cu: drop the prebuilt fused module so the patched server re-JITs.
    removed_so = []
    if aiter_rebuild or any_aiter_cu:
        for so in _aiter_prebuilt_so(Path("/sgl-workspace/aiter")):
            try:
                so.unlink()
                removed_so.append(str(so))
            except OSError:
                # Non-fatal: AITER_REBUILD=1 + jit cache invalidation still force a recompile.
                pass
        if removed_so:
            _log(out, f"removed prebuilt aiter .so ({len(removed_so)}) to force rebuild")

    # ---- PATCHED (all patches live) ----
    try:
        patched = _serve_and_bench(
            "patched", backend, model, tp, port, gpu, isl, osl, conc, num_prompts, reps, patched_env, bs, out, seed
        )
        # ---- ENGAGEMENT PROOF (trust, not policy), reported per target.
        engagement = [
            dict(
                target=str(t),
                **_engagement_proof(out / "server_patched.log", t, "/aiter/" in str(t) and t.suffix in {".cu", ".cuh"}),
            )
            for t in targets
        ]
        _log(out, f"engagement_proof: {[(e['target'].split('/')[-1], e['engaged']) for e in engagement]}")
    finally:
        # ---- REVERT every patched source (keep patch files). ----
        _revert_all()
        _log(out, f"reverted {len(manifests)} deploy(s) via manifest")

    b_med, p_med = base.get("median"), patched.get("median")
    delta_pct = (p_med - b_med) / b_med * 100.0 if (b_med and p_med) else None
    # Significance: a delta is real only if it clears the arms' [p25,p75] spread.
    bs_sp, ps_sp = base.get("tput_spread", {}), patched.get("tput_spread", {})
    significant = None
    if all(bs_sp.get(k) is not None for k in ("p25", "p75")) and all(ps_sp.get(k) is not None for k in ("p25", "p75")):
        # non-overlapping IQRs => significant
        significant = (ps_sp["p25"] > bs_sp["p75"]) or (ps_sp["p75"] < bs_sp["p25"])
    # TPOT (decode latency, lower=better).
    b_tpot = base.get("tpot_spread_ms", {}).get("median")
    p_tpot = patched.get("tpot_spread_ms", {}).get("median")
    tpot_delta_pct = (p_tpot - b_tpot) / b_tpot * 100.0 if (b_tpot and p_tpot) else None
    # P99 tail decode latency (lower=better) and peak VRAM footprint.
    b_p99 = (base.get("p99_tpot_spread_ms") or {}).get("median")
    p_p99 = (patched.get("p99_tpot_spread_ms") or {}).get("median")
    p99_tpot_delta_pct = (p_p99 - b_p99) / b_p99 * 100.0 if (b_p99 and p_p99) else None
    b_vram, p_vram = base.get("vram_used_mb"), patched.get("vram_used_mb")
    vram_delta_mb = (p_vram - b_vram) if (b_vram is not None and p_vram is not None) else None
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
        "p99_tpot_delta_pct": p99_tpot_delta_pct,  # negative = better tail decode
        "baseline_p99_tpot_spread_ms": base.get("p99_tpot_spread_ms"),
        "patched_p99_tpot_spread_ms": patched.get("p99_tpot_spread_ms"),
        "baseline_vram_used_mb": b_vram,
        "patched_vram_used_mb": p_vram,
        "vram_delta_mb": vram_delta_mb,
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
    # Build the (patch, target) pair list from --pair entries plus --patch-path/--target-file.
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
