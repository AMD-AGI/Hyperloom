# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CUDA-graph-ON serving smoke: boot the framework with a fused kernel and drive real decode through it."""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("forge_fusion")

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


@dataclass(frozen=True)
class SmokeServer:
    """The server a serving smoke boots: model, fusion flags, framework install, device, port and session shape.

    ``framework_root`` is the tree the fusion was applied to, which the server must be the one importing.
    """

    model_path: str
    env_flags: dict
    framework: str = "sglang"
    gpu: str = "0"
    port: int = 8977
    server_extra: str = ""
    framework_root: str = ""
    launcher_exe: str = ""
    tp: int = 1
    block_size: Optional[int] = None
    max_model_len: int = 4096


@dataclass(frozen=True)
class DecodeLoad:
    """The decode traffic a serving smoke drives through the live server."""

    isl: int = 512
    osl: int = 64
    num_prompts: int = 16
    conc: int = 16


@dataclass(frozen=True)
class _LiveServer:
    """A smoke server that has been launched: its process, its log, and the port it answers on."""

    process: subprocess.Popen
    log_path: str
    port: int


def _serving_smoke_launch_cmd(server: SmokeServer) -> list[str]:
    """Framework-specific serve launch command for the serving smoke.

    vLLM and SGLang have different launchers and flags; the smoke must use the one
    matching the target framework (else e.g. a vLLM run tries ``sglang.launch_server``
    and fails with ``ModuleNotFoundError: sglang`` before the fusion is ever tested).

    ``launcher_exe`` pins the exact executable, so a run validates the install it
    probed and edited rather than whichever one ``PATH`` happens to resolve first.
    ``tp`` / ``block_size`` / ``max_model_len`` must match the session serving
    command (sparse vLLM dies on the default block size 16).
    """
    extra = [p for p in (server.server_extra or "").split() if p]
    tp_n = max(1, int(server.tp or 1))
    mml = int(server.max_model_len) if server.max_model_len else 4096
    if _is_vllm_framework(server.framework):
        cmd = [
            server.launcher_exe or "vllm",
            "serve",
            server.model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(server.port),
            "--tensor-parallel-size",
            str(tp_n),
            "--trust-remote-code",
            "--max-model-len",
            str(mml),
        ]
        if server.block_size:
            cmd.extend(["--block-size", str(int(server.block_size))])
        cmd.extend(extra)
        return cmd
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        server.model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(server.port),
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
    load: DecodeLoad,
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

    isl, osl = load.isl, load.osl
    base = f"http://127.0.0.1:{port}"
    try:
        with _rq.urlopen(f"{base}/v1/models", timeout=30) as r:
            models = _json.loads(r.read().decode())
        model_id = (models.get("data") or [{}])[0].get("id")
        if not model_id:
            return False, "no served model id from /v1/models"
    except (OSError, http.client.HTTPException, ValueError, AttributeError, KeyError, TypeError) as e:
        return False, f"/v1/models probe error: {type(e).__name__}: {e}"

    prompt = "The quick brown fox " * max(1, isl // 4)
    n = max(16, min(int(load.num_prompts), 64))
    workers = max(1, min(int(load.conc), 8))

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
        except (OSError, http.client.HTTPException, ValueError) as e:
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
    run = _run or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=60))
    try:
        out = run(["rocm-smi", "--showmemuse"]).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = re.findall(r"\(VRAM%\):\s*(\d+)", out)
    if not used:
        return None
    idx = 0
    with contextlib.suppress(ValueError):
        idx = min(int(gpu), len(used) - 1)
    return 1.0 - int(used[idx]) / 100.0


def gpu_is_free_enough(gpu: str, *, _probe=None) -> tuple[bool, str]:
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
    if free >= 0.5:
        return True, ""
    return False, (
        f"GPU {gpu} has only {free * 100:.0f}% of its memory free before the server "
        f"starts, so the launch would fail on allocation regardless of the kernel "
        f"-- something from an earlier stage is still holding the card"
    )


def serving_smoke(server: SmokeServer, **kwargs) -> tuple[bool, str]:
    """``(ok, reason)`` view of :func:`serving_smoke_verdict`.

    Kept for callers that only ask "did it serve" (the compile-pass A/B arms).
    Anything deciding KEEP/REVERT must use the verdict instead, so the attribution
    comes from the stage that failed rather than from this string.
    """
    verdict = serving_smoke_verdict(server, **kwargs)
    return verdict.ok, verdict.reason


def _serving_smoke_env(server: SmokeServer) -> dict[str, str]:
    """The environment the smoke server boots under, with the fusion flags applied.

    AITER is opt-in per framework contract: ``vllm-aiter`` := vLLM with AITER on;
    plain ``vllm`` keeps vLLM's own default (do NOT force AITER, or a plain-vLLM
    smoke silently runs a non-target path -> false PASS/FAIL). Matches
    ``kernelforge.workflows.gemm_tune``.
    """
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = _hip_visible_devices(server.gpu, server.tp)
    fw = (server.framework or "").strip().lower()
    if fw == "vllm-aiter":
        env.setdefault("VLLM_ROCM_USE_AITER", "1")
    elif not _is_vllm_framework(server.framework):
        env.setdefault("SGLANG_USE_AITER", "1")
    env.update({str(key): str(value) for key, value in (server.env_flags or {}).items()})
    return env


def _await_server_ready(server: subprocess.Popen, slog: str, timeout_s: int) -> SmokeVerdict | None:
    """Wait for the server to answer, or the verdict that says why it never did.

    Boot is also where CUDA graphs are captured, so a fault here does implicate
    the kernel -- but a rejected config or an OOM does not, and both exit through
    this same path.
    """
    import time as _time

    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        if server.poll() is not None:
            tail = _tail_text(slog)
            return SmokeVerdict(
                False,
                f"server exited rc={server.returncode} before ready: {_serving_crash_reason(tail)}",
                SMOKE_STAGE_STARTUP_CRASH,
                _is_hard_gpu_fault(tail),
            )
        tail = _tail_text(slog)
        if _contains_marker(tail, _SERVER_READY_MARKERS):
            return None
        if _contains_marker(tail, _SERVING_CRASH_MARKERS):
            return SmokeVerdict(
                False,
                f"server crashed at startup: {_serving_crash_reason(tail)}",
                SMOKE_STAGE_STARTUP_CRASH,
                _is_hard_gpu_fault(tail),
            )
        _time.sleep(3)
    return SmokeVerdict(False, f"server not ready within {timeout_s}s", SMOKE_STAGE_BOOT_TIMEOUT)


def _decode_crash_verdict(server: subprocess.Popen, tail: str) -> SmokeVerdict | None:
    """The verdict for a scheduler that died inside the CUDA-graph decode loop, or None."""
    if server.poll() is None and not _contains_marker(tail, _SERVING_CRASH_MARKERS):
        return None
    return SmokeVerdict(
        False,
        f"scheduler crashed during CUDA-graph decode: {_serving_crash_reason(tail)}",
        SMOKE_STAGE_DECODE_CRASH,
        _is_hard_gpu_fault(tail),
    )


def _vllm_decode_verdict(
    live: _LiveServer,
    load: DecodeLoad,
    timeout_s: int,
    metrics: Optional[dict],
) -> SmokeVerdict:
    """Run the vLLM decode probe against the live server and judge what came back."""
    server, slog = live.process, live.log_path
    probe_ok, probe_detail = _vllm_decode_probe(live.port, load, timeout_s, metrics)
    tail = _tail_text(slog)
    if metrics is not None:
        # Read the WHOLE log: pass activation is logged at compile time, long before the tail window.
        activated, evidence = pass_activation_evidence(_full_log_text(slog))
        metrics["pass_activated"] = activated
        metrics["activation_evidence"] = evidence
    crashed = _decode_crash_verdict(server, tail)
    if crashed is not None:
        return crashed
    if not probe_ok:
        # The server is up and unfaulted, so this is the probe's own transport/response failure, not the kernel
        # misbehaving.
        return SmokeVerdict(False, f"decode probe failed: {probe_detail}", SMOKE_STAGE_DECODE_PROBE)
    return SmokeVerdict(True, "serving smoke ok: fused kernel survives CUDA-graph decode")


def _sglang_decode_verdict(
    live: _LiveServer,
    env: dict[str, str],
    load: DecodeLoad,
    timeout_s: int,
) -> SmokeVerdict:
    """Run the SGLang decode bench against the live server and judge what came back."""
    server, slog = live.process, live.log_path
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
                str(live.port),
                "--dataset-name",
                "random",
                "--random-input-len",
                str(load.isl),
                "--random-output-len",
                str(load.osl),
                "--num-prompts",
                str(load.num_prompts),
                "--max-concurrency",
                str(load.conc),
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
        # A server that came up and then stopped answering is the fused kernel hanging in the decode loop; nothing in
        # the environment stalls only here.
        return SmokeVerdict(
            False,
            "decode bench timed out (possible hang in fused kernel)",
            SMOKE_STAGE_DECODE_HANG,
            True,
        )
    bout = (bench.stdout or "") + "\n" + (bench.stderr or "")
    crashed = _decode_crash_verdict(server, _tail_text(slog))
    if crashed is not None:
        return crashed
    if bench.returncode != 0 or "Output token throughput" not in bout:
        # The bench itself failed against a live, unfaulted server.
        return SmokeVerdict(
            False,
            f"decode bench failed rc={bench.returncode}: {bout[-300:]}",
            SMOKE_STAGE_DECODE_BENCH,
        )
    return SmokeVerdict(True, "serving smoke ok: fused kernel survives CUDA-graph decode")


def serving_smoke_verdict(
    server: SmokeServer,
    load: DecodeLoad = DecodeLoad(),
    *,
    timeout_s: int = 1200,
    log_path: Optional[str] = None,
    metrics: Optional[dict] = None,
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

    ``server.framework`` selects the launcher/probe (``vllm`` / ``vllm-aiter`` vs ``sglang``);
    it MUST match the target framework or the server never boots. ``server.launcher_exe``
    pins WHICH install serves (else the first one on ``PATH`` wins, which need not
    be the install that was probed and edited).

    ``metrics``, when given, receives measured decode throughput plus fusion-pass
    activation evidence, so a caller can compare two arms instead of only asking
    "did it boot".
    """
    import signal
    import time as _time

    is_vllm = _is_vllm_framework(server.framework)
    same_tree, mismatch = framework_tree_is_the_imported_one(server.framework_root, server.framework)
    if not same_tree:
        return SmokeVerdict(False, mismatch, SMOKE_STAGE_FRAMEWORK_MISMATCH)

    roomy, busy = gpu_is_free_enough(str(server.gpu))
    if not roomy:
        return SmokeVerdict(False, busy, SMOKE_STAGE_GPU_BUSY)

    env = _serving_smoke_env(server)

    slog = log_path or str(_runtime_dir("serving_smoke") / f"server_{server.port}.log")
    if metrics is not None:
        metrics["server_log"] = slog
    cmd = _serving_smoke_launch_cmd(server)
    # Wrap the WHOLE harness: any failure returns a verdict instead of raising (a
    # serving-check must NEVER crash the loop) and the server is ALWAYS killed. Only
    # a stage that saw a GPU fault sets ``blames_kernel``, so the ledger distills the
    # CUDA-graph lesson exactly when re-authoring can act on it.
    process = None
    fh: object = None
    try:
        try:
            fh = open(slog, "w")
        except OSError:
            fh = subprocess.DEVNULL
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=fh,  # type: ignore[arg-type]
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(_runtime_dir("serving_smoke")),
        )

        not_ready = _await_server_ready(process, slog, timeout_s)
        if not_ready is not None:
            return not_ready

        # Exercise the fused kernel in the real CUDA-graph decode loop.
        live = _LiveServer(process, slog, server.port)
        if is_vllm:
            return _vllm_decode_verdict(live, load, timeout_s, metrics)
        return _sglang_decode_verdict(live, env, load, timeout_s)
    except Exception as e:  # isolation boundary: one recipe's serving smoke
        log.warning("serving smoke harness error: %s", e, exc_info=True)
        return SmokeVerdict(
            False,
            f"serving smoke harness error: {type(e).__name__}: {e}",
            SMOKE_STAGE_HARNESS_ERROR,
        )
    finally:
        if process is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        if hasattr(fh, "close"):
            fh.close()
        _time.sleep(2)
