# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-level validation of an authored fusion (Phase 4; e2e is out of scope).

forge-fuse validates at the KERNEL level, NOT full serving e2e (that is
Hyperloom's job). The validator this module provides is:

* :func:`validate_recipe` -- the fine-grained, GPU-optional KERNEL validator used
  by the autoloop (see ``loop.py``). Given a :class:`~kernelforge.fusion.models.Recipe`
  and an injectable :class:`KernelValidationRunner`, it runs three gates and
  returns a :class:`~kernelforge.fusion.models.ValidationResult`:

    (a) COMPILE/IMPORT -- the fused kernel module must import and, if Triton,
        JIT-compile on this GPU arch. "Diagnosed headroom that cannot build on
        ROCm" is a hard FAIL (e.g. reusing a framework CUDA-only op such as
        ``fused_qk_norm_rope`` which pulls in ``cuda_bf16.h``).
    (b) NUMERICAL PARITY vs the REAL eager op -- compared with the shared SNR
        gate or an rtol fallback, NEVER strict allclose (bf16 + fp32-accum is
        not bit-exact).
    (c) MICROBENCH speedup -- ``eager_us`` vs ``fused_us``; the speedup must
        clear ``target_speedup`` and stay under this module's own absolute
        plausibility ceiling.
    (d) LAUNCH COUNT -- ``fused_launches`` must be strictly below
        ``eager_launches``. Launches are what a fusion buys, and a chain that
        times faster while adding a scratch-fill or a cast elsewhere has bought
        nothing; counts the harness could not measure leave this unverified
        rather than passed.

``kept`` requires (a) through (d).

The GPU/import work lives entirely behind the injectable ``KernelValidationRunner``
so the orchestration + parity math + ROCm failure-mode classification are unit
testable WITHOUT a GPU (tests pass a fake runner). Known ROCm failure modes are
encoded as first-class classifiers (:func:`classify_compile_error`,
:func:`classify_bench_skip`) so the loop's experience ledger learns the right
lesson (author Triton, not the framework CUDA op; skip the microbench when the
Mamba backend cannot init on ROCm).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import ast
import importlib.util
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

from .models import Recipe, ValidationResult

log = logging.getLogger("forge_fusion")


# ───────────────────────── serving smoke (CUDA-graph-ON) ─────────────────────
# GPU hardware-exception / scheduler-crash signatures a kernel-level microbench
# never triggers, but the REAL sglang decode CUDA-graph loop does when a fused
# kernel uses a data-dependent grid / per-call allocation / OOB access.
# Strips "(EngineCore pid=123) ERROR 08-15 15:32:33 [core.py:1231] " style prefixes.
_EXC_PREFIX_RE = re.compile(
    r"^\s*(?:\([^)]*\)\s*)?(?:(?:ERROR|CRITICAL|WARNING)\s+)?"
    r"(?:\d[\d\-: ]*)?(?:\[[^\]]*\]\s*)?"
)
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit)\b\s*:\s*\S")

_SERVING_CRASH_MARKERS = (
    "HSA_STATUS_ERROR_EXCEPTION",
    "hardware exception",
    "Memory access fault",
    "an illegal memory access",
    "device-side assert",
    "Fatal Python error",
    "SIGQUIT",
    "core dumped",
    "CUDA error",
    "HIP error",
    "aborting with error",
)
# Ready markers cover both frameworks (aligned with Hyperloom _subprocess_kill):
# SGLang "...fired up..." (substring of the full banner) and vLLM's uvicorn/FastAPI
# lines. "Uvicorn running on" is included because some vLLM builds emit it while
# "Application startup complete" can lag.
_SERVER_READY_MARKERS = (
    "The server is fired up",
    "Application startup complete",
    "Uvicorn running on",
)


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    """Match runtime log markers without depending on producer capitalization."""
    folded = (text or "").casefold()
    return any(marker.casefold() in folded for marker in markers)


def _runtime_dir(kind: str) -> Path:
    """Return a writable runtime directory outside the source tree."""
    root = Path(os.environ.get("USER_DATA_PATH") or "/tmp")
    path = root / "forge_fusion" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tail_text(path: str, n: int = 4000) -> str:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()[-n:]
    except OSError:
        return ""


def _full_log_text(path: str, limit: int = 4_000_000) -> str:
    """Whole server log (bounded), for evidence logged long before the tail."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _serving_crash_reason(server_log_tail: str) -> str:
    """Pull the most informative GPU-fault / crash line from the server log.

    A fault marker is the strongest evidence and wins. Failing that, an explicit
    exception line still says what happened -- a server that refuses to start
    because the install needs an env var it was not given reports that plainly,
    and reporting "no explicit GPU-fault line" instead throws the answer away.
    """
    for line in server_log_tail.splitlines():
        if _contains_marker(line, _SERVING_CRASH_MARKERS):
            return " ".join(line.split())[:220]
    fatal = _explicit_fatal_error(server_log_tail)
    if fatal:
        return fatal
    return "server exited unexpectedly (no explicit GPU-fault line)"


def _explicit_fatal_error(server_log_text: str) -> str:
    """The first exception line the server logged, without the process prefixes.

    First, not last: a failing engine logs its own cause and the API server then
    logs a wrapper around it ("Engine core initialization failed. See root cause
    above."), so the last line is reliably the least informative one. Within a
    single traceback the frames do not match, so the first match is still the
    exception rather than something on the way to it.
    """
    for raw in server_log_text.splitlines():
        line = _EXC_PREFIX_RE.sub("", raw).strip()
        if _EXC_LINE_RE.match(line):
            return " ".join(line.split())[:220]
    return ""


def serving_failure_blames_kernel(reason: str) -> bool:
    """Whether a serving failure is evidence against the KERNEL.

    Only a GPU fault is. Everything else -- an engine that will not initialize, a
    missing dependency, a config the install rejects -- is a soft fail that the
    author cannot fix by re-authoring, and telling it otherwise spends the whole
    attempt budget rewriting a kernel that was never at fault.
    """
    return _contains_marker(reason or "", _SERVING_CRASH_MARKERS)


def _is_vllm_framework(framework: str) -> bool:
    return (framework or "").strip().lower() in ("vllm", "vllm-aiter")


KERNEL_KEEP_CHECKPOINT = "kernel_keep_checkpoint.json"

# Which stage of the smoke produced the verdict. The smoke knows this directly;
# recovering it from the reason text cannot separate a boot-time HIP OOM from a
# fused-kernel fault (both say "HIP error") or a transport error from a crash.
SMOKE_STAGE_OK = "ok"
SMOKE_STAGE_FRAMEWORK_MISMATCH = "framework_mismatch"
SMOKE_STAGE_GPU_BUSY = "gpu_busy"
SMOKE_STAGE_STARTUP_CRASH = "startup_crash"
SMOKE_STAGE_BOOT_TIMEOUT = "boot_timeout"
SMOKE_STAGE_DECODE_CRASH = "decode_crash"
SMOKE_STAGE_DECODE_PROBE = "decode_probe"
SMOKE_STAGE_DECODE_BENCH = "decode_bench"
SMOKE_STAGE_DECODE_HANG = "decode_hang"
SMOKE_STAGE_HARNESS_ERROR = "harness_error"

# A GPU that actually faulted. These are the only signatures that mean the fused
# kernel itself is unusable; everything else a server can print on its way down
# (a rejected config, a missing dependency, exhausted memory) is the environment.
_HARD_GPU_FAULT_MARKERS = (
    "HSA_STATUS_ERROR_EXCEPTION",
    "hardware exception",
    "Memory access fault",
    "an illegal memory access",
    "device-side assert",
    "core dumped",
)
# Resource exhaustion. Reported through the SAME "HIP error:" / "CUDA error:"
# channel as a fault, so it must be excluded explicitly or every OOM reads as a
# kernel bug and discards a KEEP that parity and the microbench both passed.
_RESOURCE_EXHAUSTION_MARKERS = (
    "out of memory",
    "outofmemory",
    "hiperroroutofmemory",
    "no available memory for the cache blocks",
    "insufficient memory",
    "cannot allocate memory",
    "memoryerror",
)


@dataclass(frozen=True)
class SmokeVerdict:
    """What the serving smoke observed, and whether it accuses the kernel.

    ``stage`` is where the smoke was when it stopped, and ``blames_kernel`` is
    the attribution made at that point -- with the server log in hand, not
    re-inferred from ``reason`` by a caller.
    """

    ok: bool
    reason: str
    stage: str = SMOKE_STAGE_OK
    blames_kernel: bool = False


def _looks_resource_exhausted(text: str) -> bool:
    """Whether the failure is memory/resource exhaustion rather than a fault."""
    return _contains_marker(text or "", _RESOURCE_EXHAUSTION_MARKERS)


def _is_hard_gpu_fault(text: str) -> bool:
    """Whether the log carries real GPU-fault evidence against the kernel.

    Exhaustion wins the tie: a run that died on memory is not evidence the fused
    kernel is unsafe, whichever error channel reported it.
    """
    if _looks_resource_exhausted(text):
        return False
    return _contains_marker(text or "", _HARD_GPU_FAULT_MARKERS)


def classify_serving_smoke_failure(reason: str) -> str:
    """Reason-only fallback for callers that kept no verdict.

    Prefer :class:`SmokeVerdict` from :func:`serving_smoke_verdict`: this can only
    see the message, so it recognizes explicit GPU-fault evidence and treats
    everything else -- boot failures, OOM, probe/transport errors -- as the
    environment, which Hyperloom's formal e2e serving is the KEEP/REVERT gate for.
    """
    if _is_hard_gpu_fault(reason or ""):
        return "kernel_fault"
    if "decode bench timed out" in (reason or "").casefold():
        return "kernel_fault"
    return "env_or_boot"


def _hip_visible_devices(gpu: str, tp: int) -> str:
    """Devices the smoke server may use.

    ``HIP_VISIBLE_DEVICES=0`` plus ``--tensor-parallel-size 8`` cannot boot a
    session-sized model; expand a scalar GPU id into a contiguous list of ``tp``
    devices. An already-comma-separated ``gpu`` is left as-is.
    """
    raw = str(gpu or "0").strip() or "0"
    n = max(1, int(tp or 1))
    if "," in raw:
        return raw
    try:
        start = int(raw)
    except ValueError:
        return raw
    if n <= 1:
        return str(start)
    return ",".join(str(start + i) for i in range(n))


def _serving_smoke_launch_cmd(
    framework: str,
    model_path: str,
    port: int,
    server_extra: str,
    *,
    launcher_exe: str = "",
    tp: int = 1,
    block_size: Optional[int] = None,
    max_model_len: int = 4096,
) -> list[str]:
    """Framework-specific serve launch command for the serving smoke.

    vLLM and SGLang have different launchers and flags; the smoke must use the one
    matching the target framework (else e.g. a vLLM run tries ``sglang.launch_server``
    and fails with ``ModuleNotFoundError: sglang`` before the fusion is ever tested).

    ``launcher_exe`` pins the exact executable, so a run validates the install it
    probed and edited rather than whichever one ``PATH`` happens to resolve first.
    ``tp`` / ``block_size`` / ``max_model_len`` must match the session serving
    command (sparse vLLM dies on the default block size 16).
    """
    extra = [p for p in (server_extra or "").split() if p]
    tp_n = max(1, int(tp or 1))
    mml = int(max_model_len) if max_model_len else 4096
    if _is_vllm_framework(framework):
        cmd = [
            launcher_exe or "vllm",
            "serve",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp_n),
            "--trust-remote-code",
            "--max-model-len",
            str(mml),
        ]
        if block_size:
            cmd.extend(["--block-size", str(int(block_size))])
        cmd.extend(extra)
        return cmd
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--trust-remote-code",
        "--tp",
        str(tp_n),
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--cuda-graph-max-bs",
        "128",
        "--moe-runner-backend",
        "aiter",
        "--context-length",
        str(mml),
        *extra,
    ]


_SITES_RE = re.compile(r"on\s+(\d+)\s+sites", re.IGNORECASE)


def pass_activation_evidence(log_text: str) -> tuple[Optional[bool], list[str]]:
    """Did a vLLM fusion pass actually rewrite the graph, per the server log?

    Returns ``(activated, lines)``. ``activated`` is ``None`` when the log carries
    no site-count evidence at all (the pass may not report one, or logging is not
    verbose enough) -- that is unknown, not proof of failure. ``False`` means the
    pass ran and matched NOTHING, which is proof the edit bought nothing.
    """
    lines = [ln.strip() for ln in (log_text or "").splitlines() if _SITES_RE.search(ln) or "FusionPass completed" in ln]
    counts = [int(m.group(1)) for ln in lines for m in [_SITES_RE.search(ln)] if m]
    if not counts:
        return None, lines[:20]
    return any(c > 0 for c in counts), lines[:20]


def _vllm_decode_probe(
    port: int,
    *,
    isl: int,
    osl: int,
    num_prompts: int,
    conc: int,
    timeout_s: int,
    metrics: Optional[dict] = None,
) -> tuple[bool, str]:
    """Drive concurrent decode requests against a live vLLM OpenAI server.

    Dependency-free (stdlib ``urllib``) replacement for ``sglang.bench_serving``:
    resolves the served model id, then issues ``/v1/completions`` requests with
    ``max_tokens=osl`` to exercise the CUDA-graph decode loop. Some fused-kernel
    graph crashes only trigger under real batch/concurrency, so send a real batch
    (>=16) with bounded parallelism rather than a couple of serial calls.
    Returns ``(ok, detail)``; ok=False on any HTTP/error or empty output.
    """
    import json as _json
    import time as _time
    import urllib.request as _rq
    from concurrent.futures import ThreadPoolExecutor

    base = f"http://127.0.0.1:{port}"
    try:
        with _rq.urlopen(f"{base}/v1/models", timeout=30) as r:
            models = _json.loads(r.read().decode())
        model_id = (models.get("data") or [{}])[0].get("id")
        if not model_id:
            return False, "no served model id from /v1/models"
    except Exception as e:  # noqa: BLE001
        return False, f"/v1/models probe error: {type(e).__name__}: {e}"

    prompt = "The quick brown fox " * max(1, isl // 4)
    n = max(16, min(int(num_prompts), 64))
    workers = max(1, min(int(conc), 8))

    tokens: list[int] = []

    def _one(i: int) -> tuple[bool, str]:
        payload = _json.dumps(
            {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": int(osl),
                "temperature": 0.0,
            }
        ).encode()
        req = _rq.Request(f"{base}/v1/completions", data=payload, headers={"Content-Type": "application/json"})
        try:
            with _rq.urlopen(req, timeout=timeout_s) as r:
                body = _json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            return False, f"completion {i} error: {type(e).__name__}: {e}"
        text = (body.get("choices") or [{}])[0].get("text") or ""
        if not text:
            return False, f"completion {i} produced no output tokens"
        # Server-reported count when available; max_tokens is the deterministic
        # fallback (temperature 0, fixed max_tokens) so both A/B arms count alike.
        used = (body.get("usage") or {}).get("completion_tokens")
        tokens.append(int(used) if isinstance(used, int) and used > 0 else int(osl))
        return True, ""

    started = _time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok, detail in ex.map(_one, range(n)):
            if not ok:
                return False, detail
    elapsed = max(1e-6, _time.perf_counter() - started)
    total = sum(tokens)
    if metrics is not None:
        metrics.update(
            {
                "output_tokens": total,
                "seconds": round(elapsed, 3),
                "tok_s": round(total / elapsed, 2),
                "num_prompts": n,
                "concurrency": workers,
                "isl": int(isl),
                "osl": int(osl),
            }
        )
    return True, (f"{n} decode completions ok (conc={workers}, {total / elapsed:.1f} tok/s)")


def _framework_package(framework: str) -> str:
    """The import name whose tree the smoke is supposed to be exercising."""
    return "vllm" if _is_vllm_framework(framework) else "sglang"


def framework_tree_is_the_imported_one(framework_root: str, framework: str, *, _finder=None) -> tuple[bool, str]:
    """Whether the tree the loop patched is the tree a server would import.

    The smoke launches the framework's own entry point, which imports the
    installed package -- so when ``--framework-root`` points somewhere else, the
    server runs stock code with the fusion flag set and comes up cleanly. That
    is a PASS reported for a kernel that was never loaded, which is worse than a
    failure: it certifies the one thing the smoke exists to check.

    Unknown roots are not second-guessed; the check only fires when the two
    locations are both known and different.
    """
    if not framework_root:
        return True, ""
    pkg = _framework_package(framework)
    patched = Path(framework_root) / pkg
    if not patched.exists():
        return True, ""
    find = _finder or _installed_package_dir
    installed = find(pkg)
    if not installed:
        return True, ""
    if Path(installed).resolve() == patched.resolve():
        return True, ""
    return False, (
        f"serving smoke would import {pkg} from {installed}, but the fusion was "
        f"applied to {patched} -- the server would run unpatched code and pass "
        f"without ever loading the kernel"
    )


def _installed_package_dir(pkg: str) -> str:
    """Where ``import <pkg>`` resolves, or "" when it does not resolve."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin:
        return ""
    return str(Path(spec.origin).parent)


def _free_vram_fraction(gpu: str, *, _run=None) -> Optional[float]:
    """Fraction of the target GPU's memory that is free, or None if unknown."""
    run = _run or (lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60))
    try:
        out = run("rocm-smi --showmemuse").stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = re.findall(r"\(VRAM%\):\s*(\d+)", out)
    if not used:
        return None
    idx = 0
    with contextlib.suppress(ValueError):
        idx = min(int(gpu), len(used) - 1)
    return 1.0 - int(used[idx]) / 100.0


def gpu_is_free_enough(gpu: str, *, need: float = 0.5, _probe=None) -> tuple[bool, str]:
    """Whether the card has room for a server, before one is launched.

    A card still held by a previous stage fails the launch with an allocator
    error, which reads as the fused kernel crashing the server. Three runs were
    abandoned that way -- one of them the highest-headroom model in the set --
    after being told across every attempt that their kernel was not CUDA-graph
    safe. Checking first costs a subprocess and turns that into a statement
    about the machine.

    An unreadable card is not treated as busy: the probe is advisory, and a
    false alarm here would block runs on any host without ``rocm-smi``.
    """
    probe = _probe or _free_vram_fraction
    free = probe(gpu)
    if free is None:
        return True, ""
    if free >= need:
        return True, ""
    return False, (
        f"GPU {gpu} has only {free * 100:.0f}% of its memory free before the server "
        f"starts, so the launch would fail on allocation regardless of the kernel "
        f"-- something from an earlier stage is still holding the card"
    )


def serving_smoke(
    model_path: str,
    env_flags: dict,
    **kwargs,
) -> tuple[bool, str]:
    """``(ok, reason)`` view of :func:`serving_smoke_verdict`.

    Kept for callers that only ask "did it serve" (the compile-pass A/B arms).
    Anything deciding KEEP/REVERT must use the verdict instead, so the attribution
    comes from the stage that failed rather than from this string.
    """
    verdict = serving_smoke_verdict(model_path, env_flags, **kwargs)
    return verdict.ok, verdict.reason


def serving_smoke_verdict(
    model_path: str,
    env_flags: dict,
    *,
    framework: str = "sglang",
    gpu: str = "0",
    port: int = 8977,
    isl: int = 512,
    osl: int = 64,
    num_prompts: int = 16,
    conc: int = 16,
    server_extra: str = "",
    framework_root: str = "",
    timeout_s: int = 1200,
    log_path: Optional[str] = None,
    launcher_exe: str = "",
    metrics: Optional[dict] = None,
    tp: int = 1,
    block_size: Optional[int] = None,
    max_model_len: int = 4096,
) -> SmokeVerdict:
    """CUDA-graph-ON serving smoke: does the fused kernel survive REAL decode?

    A fused kernel can pass the kernel-level harness (small shapes, no CUDA graph)
    yet crash the real scheduler with a GPU hardware exception
    (HSA_STATUS_ERROR_EXCEPTION) once it runs inside the captured decode CUDA graph
    over varying token counts (e.g. a data-dependent grid or a per-call allocation).
    Launches serving with the fusion env flags ON, runs a short decode probe, and
    returns a :class:`SmokeVerdict` naming the stage that failed and whether the
    kernel is implicated. The reason is fed back into the autoloop experience
    ledger so the NEXT author attempt fixes the CUDA-graph bug -- but only when
    ``blames_kernel``, because re-authoring cannot fix an environment.

    ``framework`` selects the launcher/probe (``vllm`` / ``vllm-aiter`` vs ``sglang``);
    it MUST match the target framework or the server never boots. ``launcher_exe``
    pins WHICH install serves (else the first one on ``PATH`` wins, which need not
    be the install that was probed and edited).

    ``metrics``, when given, receives measured decode throughput plus fusion-pass
    activation evidence, so a caller can compare two arms instead of only asking
    "did it boot".
    """
    import signal
    import time as _time

    is_vllm = _is_vllm_framework(framework)
    fw = (framework or "").strip().lower()
    same_tree, mismatch = framework_tree_is_the_imported_one(framework_root, framework)
    if not same_tree:
        return SmokeVerdict(False, mismatch, SMOKE_STAGE_FRAMEWORK_MISMATCH)

    roomy, busy = gpu_is_free_enough(str(gpu))
    if not roomy:
        return SmokeVerdict(False, busy, SMOKE_STAGE_GPU_BUSY)

    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = _hip_visible_devices(gpu, tp)
    # AITER is opt-in per framework contract: ``vllm-aiter`` := vLLM with AITER on;
    # plain ``vllm`` keeps vLLM's own default (do NOT force AITER, or a plain-vLLM
    # smoke silently runs a non-target path -> false PASS/FAIL). Matches kernelforge.gemm_tune.
    if fw == "vllm-aiter":
        env.setdefault("VLLM_ROCM_USE_AITER", "1")
    elif not is_vllm:
        env.setdefault("SGLANG_USE_AITER", "1")
    env.update({str(k): str(v) for k, v in (env_flags or {}).items()})

    slog = log_path or str(_runtime_dir("serving_smoke") / f"server_{port}.log")
    if metrics is not None:
        metrics["server_log"] = slog
    cmd = _serving_smoke_launch_cmd(
        framework,
        model_path,
        port,
        server_extra,
        launcher_exe=launcher_exe,
        tp=tp,
        block_size=block_size,
        max_model_len=max_model_len,
    )
    # Wrap the WHOLE harness: any failure returns a verdict instead of raising (a
    # serving-check must NEVER crash the loop) and the server is ALWAYS killed. Only
    # a stage that saw a GPU fault sets ``blames_kernel``, so the ledger distills the
    # CUDA-graph lesson exactly when re-authoring can act on it.
    server = None
    fh: object = None
    try:
        try:
            fh = open(slog, "w")
        except OSError:
            fh = subprocess.DEVNULL
        server = subprocess.Popen(
            cmd,
            env=env,
            stdout=fh,  # type: ignore[arg-type]
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(_runtime_dir("serving_smoke")),
        )

        deadline = _time.time() + timeout_s
        ready = False
        while _time.time() < deadline:
            if server.poll() is not None:
                tail = _tail_text(slog)
                # Boot is also where CUDA graphs are captured, so a fault here does
                # implicate the kernel -- but a rejected config or an OOM does not,
                # and both exit through this same path.
                return SmokeVerdict(
                    False,
                    f"server exited rc={server.returncode} before ready: {_serving_crash_reason(tail)}",
                    SMOKE_STAGE_STARTUP_CRASH,
                    _is_hard_gpu_fault(tail),
                )
            tail = _tail_text(slog)
            if _contains_marker(tail, _SERVER_READY_MARKERS):
                ready = True
                break
            if _contains_marker(tail, _SERVING_CRASH_MARKERS):
                return SmokeVerdict(
                    False,
                    f"server crashed at startup: {_serving_crash_reason(tail)}",
                    SMOKE_STAGE_STARTUP_CRASH,
                    _is_hard_gpu_fault(tail),
                )
            _time.sleep(3)
        if not ready:
            return SmokeVerdict(
                False,
                f"server not ready within {timeout_s}s",
                SMOKE_STAGE_BOOT_TIMEOUT,
            )

        # Exercise the fused kernel in the real CUDA-graph decode loop.
        if is_vllm:
            probe_ok, probe_detail = _vllm_decode_probe(
                port, isl=isl, osl=osl, num_prompts=num_prompts, conc=conc, timeout_s=timeout_s, metrics=metrics
            )
            stail = _tail_text(slog)
            if metrics is not None:
                # Read the WHOLE log: pass activation is logged at compile time,
                # long before the tail window.
                activated, evidence = pass_activation_evidence(_full_log_text(slog))
                metrics["pass_activated"] = activated
                metrics["activation_evidence"] = evidence
            if server.poll() is not None or _contains_marker(stail, _SERVING_CRASH_MARKERS):
                return SmokeVerdict(
                    False,
                    f"scheduler crashed during CUDA-graph decode: {_serving_crash_reason(stail)}",
                    SMOKE_STAGE_DECODE_CRASH,
                    _is_hard_gpu_fault(stail),
                )
            if not probe_ok:
                # The server is up and unfaulted, so this is the probe's own
                # transport/response failure, not the kernel misbehaving.
                return SmokeVerdict(
                    False,
                    f"decode probe failed: {probe_detail}",
                    SMOKE_STAGE_DECODE_PROBE,
                )
            return SmokeVerdict(
                True,
                "serving smoke ok: fused kernel survives CUDA-graph decode",
            )
        try:
            bench = subprocess.run(
                [
                    "python3",
                    "-m",
                    "sglang.bench_serving",
                    "--backend",
                    "sglang",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(isl),
                    "--random-output-len",
                    str(osl),
                    "--num-prompts",
                    str(num_prompts),
                    "--max-concurrency",
                    str(conc),
                    "--random-range-ratio",
                    "1.0",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(_runtime_dir("serving_smoke")),
            )
        except subprocess.TimeoutExpired:
            # A server that came up and then stopped answering is the fused kernel
            # hanging in the decode loop; nothing in the environment stalls only here.
            return SmokeVerdict(
                False,
                "decode bench timed out (possible hang in fused kernel)",
                SMOKE_STAGE_DECODE_HANG,
                True,
            )
        bout = (bench.stdout or "") + "\n" + (bench.stderr or "")
        stail = _tail_text(slog)
        if server.poll() is not None or _contains_marker(stail, _SERVING_CRASH_MARKERS):
            return SmokeVerdict(
                False,
                f"scheduler crashed during CUDA-graph decode: {_serving_crash_reason(stail)}",
                SMOKE_STAGE_DECODE_CRASH,
                _is_hard_gpu_fault(stail),
            )
        if bench.returncode != 0 or "Output token throughput" not in bout:
            # The bench itself failed against a live, unfaulted server.
            return SmokeVerdict(
                False,
                f"decode bench failed rc={bench.returncode}: {bout[-300:]}",
                SMOKE_STAGE_DECODE_BENCH,
            )
        return SmokeVerdict(
            True,
            "serving smoke ok: fused kernel survives CUDA-graph decode",
        )
    except Exception as e:  # noqa: BLE001 — a harness error is a soft-fail, never a crash.
        return SmokeVerdict(
            False,
            f"serving smoke harness error: {type(e).__name__}: {e}",
            SMOKE_STAGE_HARNESS_ERROR,
        )
    finally:
        if server is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        if hasattr(fh, "close"):
            fh.close()
        _time.sleep(2)


# ─────────────────────────── kernel-level validation ────────────────────────
# Phase 4 (kernel level). The three gates below are orchestrated by
# ``validate_recipe`` and exercised through an injectable ``KernelValidationRunner``
# so the decision logic is unit-testable without a GPU.

# Absolute-error fallback, used only when SNR is unavailable.
DEFAULT_RTOL = 2e-2
DEFAULT_TARGET_SPEEDUP = 1.03

# Absolute plausibility ceiling for the microbench speedup. The kernel-rewrite
# loop dropped its equivalent bound because it measures every candidate three
# times and can therefore judge a gain against the candidate's own noise. This
# validator has no such luxury: the harness self-reports one ``eager_us`` and
# one ``fused_us``, so there is no spread to compare against and nothing else
# stands between a broken timing path -- a load-independent floor, a fused arm
# that never ran -- and a KEEP. The highest speedup ever produced by a
# legitimate optimization here is 5.72x.
MAX_PLAUSIBLE_SPEEDUP = 20.0

# Known ROCm compile-failure signatures. A framework "fused" op written for CUDA
# pulls in CUDA-only headers/intrinsics and will NOT build on ROCm; the lesson the
# loop must learn is "author a ROCm-native Triton kernel, do not reuse the CUDA op".
_CUDA_ONLY_MARKERS = (
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_runtime",
    "nvcc",
    "__nv_",
    "sm_80",
    "sm_90",
    "cutlass",
    "mma.sync",
    "device_functions.h",
)
# Triton JIT/compile failures on this GPU arch (gfx942) — actionable but distinct
# from the CUDA-only case (the kernel IS ROCm-native, it just doesn't build yet).
_TRITON_BUILD_MARKERS = (
    "out of resource",
    "shared memory",
    "invalid argument",
    "passmanager",
    "llvm error",
    "triton",
    "cannot compile",
    "no kernel image",
)
# The decode microbench relies on ``bench_one_batch``; on ROCm it cannot init the
# Mamba/SSM backend, so for hybrid models the microbench must fall back / be
# skipped with a note rather than counting as a failure.
_MAMBA_MARKERS = ("mamba", "causal_conv1d", "selective_scan", "ssm", "hybrid")


def snr_db(reference: Sequence[float], test: Sequence[float]) -> Optional[float]:
    """Signal-to-noise ratio in dB between a reference and a test signal.

    ``SNR = 10 * log10( sum(ref^2) / sum((ref - test)^2) )``. Higher is better; a
    bit-exact match returns ``+inf``. This is the parity metric of choice because
    bf16 storage with fp32 accumulation is NOT bit-exact — a strict ``allclose``
    would reject numerically correct kernels, whereas the shared SNR gate accepts
    them (fused vs eager parity typically lands at 35-60 dB).

    Returns ``None`` when the inputs are empty or length-mismatched (the caller
    treats a ``None`` metric as "no data", not as a pass).
    """
    ref = list(reference)
    tst = list(test)
    if not ref or len(ref) != len(tst):
        return None
    signal = sum(r * r for r in ref)
    noise = sum((r - t) * (r - t) for r, t in zip(ref, tst))
    if noise <= 0.0:
        return math.inf
    if signal <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal / noise)


def max_abs_err(reference: Sequence[float], test: Sequence[float]) -> Optional[float]:
    """Maximum absolute elementwise error between reference and test.

    Returns ``None`` on empty / length-mismatched inputs.
    """
    ref = list(reference)
    tst = list(test)
    if not ref or len(ref) != len(tst):
        return None
    return max(abs(r - t) for r, t in zip(ref, tst))


@dataclass
class CompileOutcome:
    """Result of the compile/import + JIT gate (gate a)."""

    ok: bool
    is_triton: bool = False
    error: str = ""


@dataclass
class ParitySample:
    """One shape's parity metrics vs the imported real eager op (gate b).

    ``snr_db`` is the primary metric; ``max_abs_err`` is the rtol fallback used
    only when ``snr_db`` is unavailable. A runner computes these ON-DEVICE (where
    the tensors live) so only the scalars cross the boundary; :func:`snr_db` /
    :func:`max_abs_err` are exposed for runners (and tests) to compute them.
    """

    snr_db: Optional[float] = None
    max_abs_err: Optional[float] = None
    label: str = ""


@dataclass
class BenchOutcome:
    """Result of the microbench gate (gate c).

    ``skipped`` marks a benign unavailability (e.g. the Mamba backend cannot init
    on ROCm) — correctness still counts, but the speedup is unverified.

    The launch counts are per decode step, over the whole step rather than the
    replaced chain. ``None`` means the harness could not count them, which leaves
    the launch gate unverified rather than passed.
    """

    eager_us: Optional[float] = None
    fused_us: Optional[float] = None
    skipped: bool = False
    skip_reason: str = ""
    eager_launches: Optional[int] = None
    fused_launches: Optional[int] = None


@runtime_checkable
class KernelValidationRunner(Protocol):
    """Injectable boundary for all GPU/import work in :func:`validate_recipe`.

    Production code passes a runner that actually compiles + runs the kernel on
    the GPU (see :class:`HarnessKernelRunner`); unit tests pass a fake so the
    orchestration, parity math, and ROCm failure-mode classification are exercised
    without a GPU or an LLM.
    """

    def compile_check(self, recipe: Recipe) -> CompileOutcome:
        """Import the fused module and, if Triton, JIT-compile it on this arch."""

    def parity_samples(self, recipe: Recipe) -> list[ParitySample]:
        """Compare fused vs the imported REAL eager op on representative shapes."""

    def microbench(self, recipe: Recipe) -> BenchOutcome:
        """Time eager vs fused on the decode shape (may be skipped, see above)."""


def classify_compile_error(error: str, recipe: Optional[Recipe] = None) -> str:
    """Map a compile/import error to a crisp, reusable lesson (mirrors the
    forge-loop experience ledger's ``_CONSTRAINT_RULES``).

    The CUDA-only case is first-class: a framework "fused" op authored for CUDA
    fails to build on ROCm, and the loop must learn to author a ROCm-native Triton
    kernel instead of reusing it.
    """
    e = (error or "").lower()
    if any(m in e for m in _CUDA_ONLY_MARKERS):
        return (
            "The fused op is CUDA-only (pulls in e.g. cuda_bf16.h, like sglang's "
            "fused_qk_norm_rope) and cannot build on ROCm. Author a ROCm-native "
            "Triton kernel instead of reusing the framework CUDA fused op."
        )
    if any(m in e for m in _TRITON_BUILD_MARKERS):
        return (
            "The Triton kernel failed to JIT-compile on this GPU arch (gfx942). "
            "Reduce BLOCK size / shared-memory usage or fix tl.constexpr shapes so "
            "it builds, and keep an eager fallback when Triton is unavailable."
        )
    return (
        "The fused module failed to import/compile on ROCm. Ensure the kernel is "
        "ROCm-native and falls back to eager when Triton is unavailable."
    )


def classify_bench_skip(reason: str) -> str:
    """Map a microbench skip reason to a reusable note.

    The Mamba/SSM-backend case is first-class: ``bench_one_batch`` cannot init the
    backend on ROCm for hybrid models, so the microbench is unavailable and the
    speedup is treated as unverified (NOT a failure) — parity remains the gate.
    """
    r = (reason or "").lower()
    if any(m in r for m in _MAMBA_MARKERS):
        return (
            "bench_one_batch cannot initialize the Mamba/SSM backend on ROCm, so "
            "the decode microbench is unavailable for hybrid models. Gate on "
            "kernel-level parity and treat the speedup as unverified, not failed."
        )
    return "Microbench unavailable; kernel-level parity is the gate for this attempt."


def implausible_speedup_reason(speedup: float) -> str:
    """Why this microbench speedup cannot be real, or "" when it can be.

    Fusion owns this bound rather than borrowing the rewrite loop's KEEP policy:
    the two answer different questions from different evidence, and the shared
    import made them look like one policy until the loop, which repeats every
    measurement, dropped its ceiling and silently took fusion's only anomaly
    check with it.
    """
    value = float(speedup)
    if not math.isfinite(value) or value > MAX_PLAUSIBLE_SPEEDUP:
        return f"microbench speedup {value:.6f}x exceeds the {MAX_PLAUSIBLE_SPEEDUP}x absolute plausibility ceiling"
    return ""


def launch_regression_reason(eager_launches: Optional[int], fused_launches: Optional[int]) -> str:
    """Why this fusion did not remove launches, or "" when it did.

    The whole lever is launch count, so a candidate that collapses its chain but
    adds a scratch-fill, a cast or a copy to feed the fused kernel can time
    faster on the chain and cost more over the decode step. Unknown counts are
    NOT a regression: the harness may be unable to profile, and failing a correct
    fusion for that would teach the author to game the number instead of the
    kernel. :func:`launch_gate_unverified` reports that case separately.
    """
    if eager_launches is None or fused_launches is None:
        return ""
    eager, fused = int(eager_launches), int(fused_launches)
    if fused < eager:
        return ""
    verdict = "removes no launches" if fused == eager else "ADDS launches"
    return (
        f"the fused path {verdict}: {eager} -> {fused} per decode step. Fusion is "
        f"paid for in launches, not in the timing of the chain alone; find what "
        f"the fused path launches in addition to the kernel you wrote"
    )


def launch_gate_unverified(eager_launches: Optional[int], fused_launches: Optional[int]) -> bool:
    """Whether the harness left the launch gate unmeasured."""
    return eager_launches is None or fused_launches is None


def launch_count(value: Any) -> Optional[int]:
    """One launch count out of harness JSON, or ``None`` when it is not usable.

    A harness that reports a count it did not measure is the failure this guards:
    anything that is not a non-negative integer reads as "not counted", which
    leaves the gate unverified instead of silently passing it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    count = int(value)
    return count if count >= 0 else None


def _tail(text: str, n: int = 400) -> str:
    """Last ``n`` chars of an error blob, single-lined for compact notes."""
    return " ".join((text or "").split())[-n:]


def _kernel_match_key(name: str) -> str:
    """One kernel name reduced to a form two profilers can agree on."""
    return " ".join(str(name or "").split()).lower()


def _same_kernel(expected: str, observed: str) -> bool:
    """Whether an observed launch is the kernel the trace recorded.

    Kineto and the in-process profiler render the same symbol slightly
    differently (a dropped ``void``, a differently spelled template argument), so
    containment either way counts as a match. The names are long mangled
    signatures, which makes an accidental containment match unlikely enough that
    tolerance costs less here than a false mismatch would.
    """
    want, got = _kernel_match_key(expected), _kernel_match_key(observed)
    if not want or not got:
        return False
    return want == got or want in got or got in want


def eager_trace_alignment(
    observed: Sequence[str],
    trace_kernels: Optional[dict[str, Any]],
) -> tuple[Optional[bool], str]:
    """Whether the harness's eager arm ran the code path the trace recorded.

    Everything the campaign reports -- parity, speedup, both launch counts -- is
    measured against whatever the eager arm happens to call, and nothing
    downstream re-checks that choice. A framework that ships two implementations
    of one chain (a Triton path and a JIT C++ path, say) will happily export
    plausible names for the dead one, and a harness built against it produces
    numbers that are internally consistent and describe software nobody runs.
    Comparing the arm's actual launches against the anchor's recorded
    neighbourhood is the one cheap check that catches it.

    Returns ``None`` when there is nothing to check -- unanchored discovery
    records no neighbourhood, and a harness that reports no kernel names leaves
    the question open rather than answering it "no".
    """
    evidence = trace_kernels or {}
    anchor = str(evidence.get("anchor") or "").strip()
    if not anchor:
        return None, "no anchored trace evidence to check the eager arm against"
    seen = [str(n) for n in (observed or []) if str(n).strip()]
    if not seen:
        return None, "harness reported no eager kernel names; alignment unchecked"

    neighbours: list[str] = []
    for key in ("before", "after"):
        for name in evidence.get(key) or []:
            text = str(name).strip()
            if text and text != anchor and text not in neighbours:
                neighbours.append(text)

    if not any(_same_kernel(anchor, name) for name in seen):
        return False, (
            f"eager arm never launched the anchor kernel ({_tail(anchor, 120)}); "
            f"it issued {len(seen)} other kernel(s), so it is a different code path"
        )
    matched = [n for n in neighbours if any(_same_kernel(n, name) for name in seen)]
    if neighbours and not matched:
        return False, (
            f"eager arm launched the anchor but none of its {len(neighbours)} recorded "
            "neighbour kernels; it reproduces the anchor in isolation, not the chain"
        )
    return True, f"eager arm matched the anchor and {len(matched)}/{len(neighbours)} recorded neighbours"


@dataclass(frozen=True)
class WiringEvidence:
    """What a static read of the framework edit proved about the fused call site."""

    verdict: str
    """``"wired"``, ``"not_wired"``, or ``"unchecked"`` when the gate could not judge."""

    reason: str


def _file_invocation_evidence(source_file: str) -> tuple[Optional[bool], str]:
    """One file's verdict: ``True`` wired, ``False`` dead import, ``None`` no evidence."""
    from .emit import _is_fused_module_name

    label = Path(source_file).name
    try:
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError) as exc:
        return None, f"{label} unchecked ({type(exc).__name__}: {exc})"

    # Names the wiring edit binds from a fused-kernel module, at any nesting
    # depth: a lazy import inside ``forward`` is a legitimate wiring style.
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            leaf = (node.module or "").rsplit(".", 1)[-1]
            if leaf and _is_fused_module_name(f"{leaf}.py"):
                bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_fused_module_name(f"{alias.name.rsplit('.', 1)[-1]}.py"):
                    bound.add(alias.asname or alias.name.split(".")[0])
    if not bound:
        # A fusion authored INLINE in the framework file imports nothing, and is
        # wired by construction. Only a bound-but-unused import is provable, so
        # this branch yields no evidence rather than a verdict.
        return None, f"{label} imports no fused-kernel module"

    # An ``import`` statement contributes ast.alias, never ast.Name, so any Name
    # load of a bound identifier is by construction a use outside the import.
    used = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bound
        }
    )
    if used:
        return True, f"{label} references {', '.join(used)}"
    return False, (
        f"{label} imports {', '.join(sorted(bound))} from a fused-kernel "
        f"module and never references it -- the fused kernel is dead code in the served model"
    )


def fused_symbol_invocation_evidence(
    source_file: str,
    extra_files: Sequence[str] = (),
) -> WiringEvidence:
    """Whether the framework edit CALLS the fused module, or only imports it.

    A fusion is delivered as two edits: a new fused-kernel module, and a wiring
    edit that makes the framework's forward path use it. Everything downstream
    measures only the first. The harness imports the fused entry point and times
    it against its own eager reference, so a 37x microbench is fully explained by
    a module that nothing calls; and the serving smoke boots the framework and
    sends real decodes, which succeed exactly as they did before because the
    unwired kernel never runs. Both report success for zero end-to-end gain --
    the failure this module already names elsewhere as "a PASS reported for a
    kernel that was never loaded, which is worse than a failure".

    This is the missing wiring check, and it is deliberately static: an import
    bound by a name that appears nowhere else in the file (the ``# noqa: F401``
    shape an agent produces when it authors the kernel but forgets the call site)
    cannot execute, whatever the runtime does. Everything else is ``unchecked``,
    which does not block a KEEP -- an unreadable or unparseable source, and
    equally a source that imports no fused module at all, which is what an INLINE
    fusion (the fused call written straight into the framework file) legitimately
    looks like. The gate exists to catch one provable defect, not to demote a KEEP
    it could not inspect, and equally not to claim it inspected one it did not.

    ``extra_files`` are the other files a multi-file fusion edits. The fused module
    only has to be called from ONE of them, so a single file's dead import is not
    the fusion's verdict: the check passes as soon as any file references it, and
    fails only when every file that imports it leaves it unused.
    """
    files = [p for p in [source_file, *extra_files] if p]
    if not files:
        return WiringEvidence("unchecked", "unchecked (no framework file recorded)")

    dead: list[str] = []
    silent: list[str] = []
    for path in files:
        verdict, reason = _file_invocation_evidence(path)
        if verdict is True:
            return WiringEvidence("wired", reason)
        (dead if verdict is False else silent).append(reason)
    if dead:
        return WiringEvidence("not_wired", "; ".join(dead))
    return WiringEvidence("unchecked", f"unchecked ({'; '.join(silent)})")



def validate_recipe(
    recipe: Recipe,
    runner: KernelValidationRunner,
    *,
    target_speedup: float = DEFAULT_TARGET_SPEEDUP,
    snr_threshold_db: float = DEFAULT_SNR_THRESHOLD_DB,
    rtol: float = DEFAULT_RTOL,
) -> ValidationResult:
    """Kernel-level validation of one authored fusion (gates a -> b -> c).

    The gates run in order and short-circuit on the first failure, so a compile
    failure never wastes a parity/bench run. ``kept`` is True only when the kernel
    COMPILES, matches the eager reference (parity), AND is at least
    ``target_speedup`` faster than eager.

    Args:
        recipe: The localized fusion plan (source of shapes, env flag, and the
            eager-reference hint). No per-model literals are read here.
        runner: The injectable GPU/import boundary (mock it in unit tests).
        target_speedup: Microbench speedup required to KEEP.
        snr_threshold_db: Numerical-parity SNR floor in dB.
        rtol: Absolute-error fallback used only when SNR is unavailable.

    Returns:
        A :class:`~kernelforge.fusion.models.ValidationResult`. On any failure the
        ``note`` carries a compressed error signature plus a reusable LESSON so
        the loop's experience ledger can inject it into the next attempt.
    """
    # ── gate (a): compile / import (+ Triton JIT on this arch) ───────────────
    comp = runner.compile_check(recipe)
    if not comp.ok:
        lesson = classify_compile_error(comp.error, recipe)
        kind = "triton JIT" if comp.is_triton else "module import"
        return ValidationResult(
            correctness_passed=False,
            correctness_measured=False,
            max_abs_err=None,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=None,
            fused_us=None,
            kept=False,
            note=f"COMPILE FAILED ({kind}): {_tail(comp.error)} | LESSON: {lesson}",
        )

    # ── gate (b): numerical parity vs the imported REAL eager op ─────────────
    samples = runner.parity_samples(recipe)
    if not samples:
        return ValidationResult(
            correctness_passed=False,
            correctness_measured=False,
            max_abs_err=None,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=None,
            fused_us=None,
            kept=False,
            note=(
                "PARITY UNAVAILABLE: the runner returned no samples — the eager "
                "reference could not be imported/executed. LESSON: import the REAL "
                "eager op per the recipe's eager_reference_hint and assert parity."
            ),
        )
    worst_err: Optional[float] = None
    min_snr: Optional[float] = None
    for s in samples:
        if s.max_abs_err is not None:
            worst_err = s.max_abs_err if worst_err is None else max(worst_err, s.max_abs_err)
        if s.snr_db is not None:
            min_snr = s.snr_db if min_snr is None else min(min_snr, s.snr_db)
    # SNR is the primary gate; rtol is the fallback only when SNR is unavailable.
    if min_snr is not None:
        parity_ok = min_snr >= snr_threshold_db
    elif worst_err is not None:
        parity_ok = worst_err <= rtol
    else:
        parity_ok = False
    if not parity_ok:
        snr_txt = f"{min_snr:.1f} dB" if min_snr is not None else "n/a"
        err_txt = f"{worst_err:.3e}" if worst_err is not None else "n/a"
        return ValidationResult(
            correctness_passed=False,
            max_abs_err=worst_err,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=None,
            fused_us=None,
            kept=False,
            note=(
                f"PARITY FAILED: min SNR={snr_txt} (< {snr_threshold_db:.0f} dB), "
                f"max_abs_err={err_txt}. bf16 + fp32-accum is not bit-exact; check "
                f"the accumulation dtype and the fused math against the eager op."
            ),
        )

    # ── gate (c): microbench speedup (eager vs fused) ────────────────────────
    bench = runner.microbench(recipe)
    if bench.skipped:
        note = classify_bench_skip(bench.skip_reason)
        return ValidationResult(
            correctness_passed=True,
            max_abs_err=worst_err,
            rtol=rtol,
            kernel_speedup=None,
            eager_us=bench.eager_us,
            fused_us=bench.fused_us,
            kept=False,
            note=f"PARITY OK; MICROBENCH SKIPPED ({bench.skip_reason}): {note}",
        )
    speedup: Optional[float] = None
    if bench.eager_us and bench.fused_us and bench.fused_us > 0:
        speedup = bench.eager_us / bench.fused_us
    implausible = implausible_speedup_reason(speedup) if speedup is not None else ""
    # ── gate (d): launch count ───────────────────────────────────────────────
    regression = launch_regression_reason(bench.eager_launches, bench.fused_launches)
    unverified = launch_gate_unverified(bench.eager_launches, bench.fused_launches)
    kept = speedup is not None and not implausible and not regression and speedup >= target_speedup
    if speedup is None:
        note = (
            "PARITY OK but microbench produced no timing (eager_us/fused_us "
            "missing) — cannot confirm the speedup; treat as not kept."
        )
    elif implausible:
        note = f"PARITY OK but the microbench is not believable: {implausible}"
    elif regression:
        note = (
            f"PARITY OK and {speedup:.3f}x on the chain, but {regression}. "
            f"LESSON: count the launches of both arms before optimizing the "
            f"schedule; a faster chain that costs a launch is not a fusion."
        )
    elif kept:
        note = (
            f"KEPT: parity OK and {speedup:.3f}x >= {target_speedup:.2f}x target "
            f"(eager={bench.eager_us} us, fused={bench.fused_us} us"
            + (
                "; launches UNVERIFIED — the harness reported no count"
                if unverified
                else f"; launches {bench.eager_launches} -> {bench.fused_launches}"
            )
            + ")."
        )
    else:
        note = (
            f"PARITY OK but only {speedup:.3f}x (< {target_speedup:.2f}x target) — "
            f"correct yet not fast enough; try a cheaper fused schedule."
        )
    return ValidationResult(
        correctness_passed=True,
        max_abs_err=worst_err,
        rtol=rtol,
        kernel_speedup=round(speedup, 4) if speedup is not None else None,
        eager_us=bench.eager_us,
        fused_us=bench.fused_us,
        kept=kept,
        note=note,
        eager_launches=bench.eager_launches,
        fused_launches=bench.fused_launches,
    )


class HarnessKernelRunner:
    """Production :class:`KernelValidationRunner` backed by an author-written harness.

    The GPU/import work is genuinely environment-specific, so it is kept at the
    process boundary: this runner executes a kernel-validation harness script (the
    author is instructed to write a parity self-check) in a subprocess and parses a
    single JSON object from its stdout. Unit tests never touch this class — they
    inject a fake runner — so it is deliberately defensive and NEVER raises: a
    missing or malformed harness degrades to a compile failure / skipped microbench
    with an actionable note.

    Harness JSON contract (one object on stdout)::

        {"compiled": bool, "is_triton": bool, "error": str,
         "parity": [{"snr_db": float|null, "max_abs_err": float|null, "label": str}],
         "eager_us": float|null, "fused_us": float|null,
         "eager_launches": int|null, "fused_launches": int|null,
         "eager_kernels": [str], "eager_matches_trace": bool,
         "skipped": bool, "skip_reason": str}
    """

    def __init__(
        self,
        harness_path: str,
        *,
        workdir: str = ".",
        framework_root: str = "",
        gpu: str = "0",
        env_flags: Optional[dict[str, str]] = None,
        timeout_s: int = 1800,
    ):
        self.harness_path = harness_path
        self.workdir = workdir
        self.framework_root = framework_root
        self.gpu = gpu
        self.env_flags = dict(env_flags or {})
        self.timeout_s = timeout_s
        self._cache: Optional[dict] = None

    def _load(self, recipe: Recipe) -> dict:
        """Run the harness once (cached) and return its parsed JSON dict."""
        if self._cache is not None:
            return self._cache
        result: dict
        if not self.harness_path or not Path(self.harness_path).is_file():
            result = {
                "compiled": False,
                "is_triton": False,
                "error": f"kernel harness not found: {self.harness_path!r}",
            }
            self._cache = result
            return result
        env = dict(os.environ)
        env["HIP_VISIBLE_DEVICES"] = self.gpu
        # The harness is authored inside the framework tree and then published to
        # the run's output directory, so a path the author derived from
        # ``__file__`` points somewhere else by the time it runs here. Name the
        # tree outright rather than leaving it to be inferred.
        if self.framework_root:
            env["FORGE_FUSION_FRAMEWORK_ROOT"] = self.framework_root
        env.update({k: str(v) for k, v in self.env_flags.items()})
        try:
            proc = subprocess.run(
                ["python3", self.harness_path],
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            result = _parse_harness_json(proc.stdout, proc.stderr, proc.returncode)
        except subprocess.TimeoutExpired:
            result = {
                "compiled": False,
                "is_triton": False,
                "error": f"kernel harness timed out after {self.timeout_s}s",
            }
        except OSError as e:
            result = {"compiled": False, "is_triton": False, "error": f"could not run kernel harness: {e}"}
        self._cache = result
        return result

    def report(self, recipe: Recipe) -> dict:
        """The harness's raw JSON, for gates that read fields beyond the protocol."""
        return dict(self._load(recipe))

    def compile_check(self, recipe: Recipe) -> CompileOutcome:
        d = self._load(recipe)
        return CompileOutcome(
            ok=bool(d.get("compiled")),
            is_triton=bool(d.get("is_triton")),
            error=str(d.get("error") or ""),
        )

    def parity_samples(self, recipe: Recipe) -> list[ParitySample]:
        d = self._load(recipe)
        out: list[ParitySample] = []
        for p in d.get("parity") or []:
            if not isinstance(p, dict):
                continue
            out.append(
                ParitySample(
                    snr_db=p.get("snr_db"),
                    max_abs_err=p.get("max_abs_err"),
                    label=str(p.get("label") or ""),
                )
            )
        return out

    def microbench(self, recipe: Recipe) -> BenchOutcome:
        d = self._load(recipe)
        return BenchOutcome(
            eager_us=d.get("eager_us"),
            fused_us=d.get("fused_us"),
            skipped=bool(d.get("skipped")),
            skip_reason=str(d.get("skip_reason") or ""),
            eager_launches=launch_count(d.get("eager_launches")),
            fused_launches=launch_count(d.get("fused_launches")),
        )


def _parse_harness_json(stdout: str, stderr: str, returncode: int) -> dict:
    """Best-effort parse of the LAST JSON object printed by the harness.

    On any parse failure the harness output tail becomes a compile error so the
    loop still learns something instead of crashing.
    """
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = _tail((stdout or "") + "\n" + (stderr or ""))
    return {"compiled": False, "is_triton": False, "error": f"harness produced no JSON (rc={returncode}): {tail}"}
