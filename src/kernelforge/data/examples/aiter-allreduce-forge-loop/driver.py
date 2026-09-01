# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Forge driver for TP4 custom all-reduce (raw) and fused all-reduce + RMSNorm.

The same file serves two roles, selected purely by the presence of ``RANK`` /
``LOCAL_RANK`` in the environment:

* **self-launch** (no RANK): validates resources, rebuilds the aiter JIT module
  when the source hash changed, then re-executes itself under
  ``torch.distributed.run --standalone --nproc-per-node=N``.
* **worker** (RANK present): binds one GPU, initialises the aiter custom
  all-reduce communicator and runs correctness / benchmark / profile.

The two branches are mutually exclusive, so the driver can be invoked both by
today's single-process Forge (``python driver.py``) and by a future distributed
launcher (``torchrun ... driver.py``) without any code change.

Output contract (rank 0 only):
  correctness -> ``SNR: <db> dB`` / ``allclose: <bool>`` / ``max_diff: <float>``
  benchmark   -> ``case_ms: <case_id> <ms> [unscored]`` per case,
                 ``mean_ms: <ms>``,
                 plus one single-line ``__FORGE_DISTRIBUTED_RESULT__{...}__``.

The loop's authoritative KEEP decision uses the independently measured
``case_ms`` values; ``mean_ms`` is diagnostic.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SENTINEL = "__FORGE_DISTRIBUTED_RESULT__"
def _default_tp() -> int:
    """Rank count to use when the caller did not name one.

    forge invokes the driver for validation and benchmarking with no extra
    arguments, so anything hard-coded here becomes the configuration those
    stages actually measure. A fixed default is therefore wrong twice over: it
    silently benchmarks a rank count nobody asked for, and it disagrees with the
    --nproc-per-node the profiler launches with, which then fails the driver's
    own WORLD_SIZE check.

    Derived instead, most specific first: an explicit rank count from the
    launcher, then the visible device count, then a last-resort constant.
    """
    for var in ("FORGE_NPROC_PER_NODE", "WORLD_SIZE"):
        try:
            value = int(os.environ.get(var) or 0)
        except ValueError:
            value = 0
        if value > 0:
            return value
    visible = 0
    masked = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if masked:
        visible = len([d for d in masked.split(",") if d.strip()])
    if visible < 2:
        try:
            visible = torch.cuda.device_count()
        except Exception:  # noqa: BLE001 - no CUDA context is not fatal here
            visible = 0
    # A collective on one rank measures nothing, so never fall below two.
    return visible if visible > 1 else 2


DEFAULT_TP = _default_tp()
RAW_GROUP = "raw_dispatch"
FUSED_GROUP = "fused_rmsnorm"

# Source files whose content determines whether the JIT module must be rebuilt.
JIT_SOURCE_FILES = (
    "csrc/include/custom_all_reduce.cuh",
    "csrc/kernels/custom_all_reduce.cu",
    "csrc/include/custom_all_reduce.h",
)
JIT_MODULE = "module_custom_all_reduce"
STAMP_NAME = ".forge_ar_source.stamp"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------
# Case model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One measurable shape for either metric group."""

    target: str  # "raw" | "fused"
    rows: int
    hidden: int
    dtype: str = "bf16"
    graph: int = 0
    sensitive: bool = True  # included in diagnostic group aggregation

    @property
    def case_id(self) -> str:
        return f"{self.target}_{self.dtype}_{self.rows}x{self.hidden}"

    @property
    def group(self) -> str:
        return RAW_GROUP if self.target == "raw" else FUSED_GROUP

    @property
    def nbytes(self) -> int:
        return self.rows * self.hidden * torch.tensor([], dtype=DTYPES[self.dtype]).element_size()


def _suite_tp4_thresholds(dtype: str = "bf16") -> list[Case]:
    """Full case set frozen by the design doc.

    ``sensitive`` marks the cases whose dispatch actually changes across the
    swept thresholds; only those feed the diagnostic group aggregate.
    """
    cases: list[Case] = []
    # raw: default crossover is 160 KiB. Any threshold in the swept 128-256 KiB
    # range flips rows 8..16 (128-256 KiB), so all of them are sensitive.
    for rows in (8, 9, 10, 11, 12, 14, 16):
        cases.append(Case("raw", rows, 8192, dtype, sensitive=True))
    # rows=1 (16 KiB) stays 1-stage and rows=32 (512 KiB) stays 2-stage for
    # every candidate threshold: diagnostics only.
    for rows in (1, 32):
        cases.append(Case("raw", rows, 8192, dtype, sensitive=False))
    # fused: crossover at 128 KiB on total_bytes.
    for rows in (7, 8, 9):
        cases.append(Case("fused", rows, 8192, dtype, sensitive=True))
    for rows in (16, 17):
        cases.append(Case("fused", rows, 4096, dtype, sensitive=True))
    cases.append(Case("fused", 32, 4096, dtype, sensitive=False))
    cases.append(Case("fused", 1, 8192, dtype, sensitive=False))
    cases.append(Case("fused", 64, 8192, dtype, sensitive=False))
    return cases


def _suite_tp4_wide(dtype: str = "bf16") -> list[Case]:
    """Wider case set that also scores the kernels themselves, not just the
    crossover threshold.

    ``tp4_thresholds`` deliberately scores only threshold-sensitive cases, which
    is right while the optimization variable is a constant. Once the kernels are
    being rewritten that set is too narrow: small payloads always take the
    1-stage path and large ones always take 2-stage, so neither shows up in the
    score even though both are real inference regimes (decode is small-payload).
    So nearly every case contributes to the equal-weight score.

    Two cases are excluded from the score anyway, on measured grounds: their
    run-to-run spread is physically large and does NOT shrink with more sampling
    (quadrupling the iteration count moved 21%->16% and 13%->12%), because the
    fluctuation outlasts a single measurement window. They remain in ``cases``
    for visibility but are excluded from the KEEP score.
    """
    cases: list[Case] = []
    # 16-32 KiB: always 1-stage. Sensitive to the 1-stage kernel, not the cut.
    for rows in (1, 2):
        cases.append(Case("raw", rows, 8192, dtype, sensitive=True))
    # 64 KiB: measured spread 13%, unresponsive to more iterations.
    cases.append(Case("raw", 4, 8192, dtype, sensitive=False))
    # 128-256 KiB: straddles the crossover.
    for rows in (8, 9, 10, 11, 12, 14, 16):
        cases.append(Case("raw", rows, 8192, dtype, sensitive=True))
    # 512 KiB-1 MiB: always 2-stage.
    for rows in (32, 64):
        cases.append(Case("raw", rows, 8192, dtype, sensitive=True))

    for rows in (1, 2, 4):
        cases.append(Case("fused", rows, 8192, dtype, sensitive=True))
    for rows in (7, 8, 9):
        cases.append(Case("fused", rows, 8192, dtype, sensitive=True))
    for rows in (16, 17):
        cases.append(Case("fused", rows, 4096, dtype, sensitive=True))
    cases.append(Case("fused", 32, 8192, dtype, sensitive=True))
    # 1 MiB: measured spread 21%, ~80% of this group's score noise. Largest
    # payload in the suite, so it most likely rides a power or interconnect
    # limit rather than a sampling artifact.
    cases.append(Case("fused", 64, 8192, dtype, sensitive=False))
    return cases


def _suite_default(dtype: str = "bf16", tp: int = 2) -> list[Case]:
    """Threshold sweep sized for whatever rank count the caller asked for.

    aiter dispatches 1-stage below a byte cut that depends on world size --
    160 KiB at up to 4 ranks, 80 KiB at up to 8 -- and 2-stage above it. The
    cases are placed either side of that cut so the sweep measures the dispatch
    decision itself rather than one arbitrary payload, which is what makes this
    usable at any rank count instead of only the two that were hand-tuned.

    The named suites below stay for the configurations with a measured baseline;
    this one is the default so a two-GPU box can run the example unmodified.
    """
    hidden = 7168
    row_bytes = hidden * DTYPES[dtype].itemsize
    cut_bytes = (160 if tp <= 4 else 80) * 1024
    cross = max(2, cut_bytes // row_bytes)
    # Below, at, and above the cut, plus a large payload that stays 2-stage.
    rows = sorted({1, max(1, cross // 2), cross - 1, cross, cross + 1, cross * 2, 64})
    cases = [Case("raw", r, hidden, dtype, sensitive=True) for r in rows]
    # Fused allreduce+rmsnorm keeps its own cut as diagnostic coverage.
    cases += [Case("fused", r, hidden, dtype, sensitive=False) for r in (max(1, cross), cross * 2)]
    return cases


def _suite_tp8_k3(dtype: str = "bf16") -> list[Case]:
    """TP8 / gfx950 case set sized for Kimi-K3 (hidden=7168, bf16).

    One row is 7168 * 2 = 14 KiB, so the TP8 raw cut
    (``world_size_ <= 8 && bytes < 80*1024``) lands at 5.71 rows. The measured
    baseline on 8xMI355X at commit 36c421f7f shows the 1-stage path peaking just
    below that cut (rows=5) and 2-stage beating it immediately above (rows=6).

    The sweep itself lives in ``program.md`` -- single source of truth, since it
    also carries the measured case suite. Do not restate the
    numbers here; two copies have already drifted apart once.

    Rows 1..5 are the cases a lowered threshold would flip, so they carry the
    score together with the 2-stage band above the cut. Rows 64 is the real
    Kimi-K3 decode payload at conc=64 (896 KiB).
    """
    cases: list[Case] = []
    # 14-70 KiB: forced 1-stage by the 80 KiB cut today; these flip if it drops.
    for rows in (1, 2, 3, 4, 5):
        cases.append(Case("raw", rows, 7168, dtype, sensitive=True))
    # 84-224 KiB: already 2-stage, so this band scores the 2-stage kernel.
    for rows in (6, 7, 8, 12, 16):
        cases.append(Case("raw", rows, 7168, dtype, sensitive=True))
    # 896 KiB: production decode payload at conc=64, well below the 4 MiB
    # write_mode branch this task must not touch.
    cases.append(Case("raw", 64, 7168, dtype, sensitive=True))
    # Fused allreduce+rmsnorm keeps its own 128 KiB cut on total_bytes. Carried
    # as diagnostics only -- the raw dispatch is this task's target, and
    # the fused path has no measured baseline sweep yet.
    for rows in (4, 8, 9, 16):
        cases.append(Case("fused", rows, 7168, dtype, sensitive=False))
    return cases


# Named suites, plus the rank-derived default. Shared by the --shape parser and
# the FORGE_COLLECTIVE_SUITE default so both accept exactly the same names.
_SUITE_BUILDERS = {
    "default": lambda d: _suite_default(d, DEFAULT_TP),
    "tp4_thresholds": _suite_tp4_thresholds,
    "tp4_wide": _suite_tp4_wide,
    "tp8_k3": _suite_tp8_k3,
}
_SUITE_REQUIRED_TP = {
    "tp4_thresholds": 4,
    "tp4_wide": 4,
    "tp8_k3": 8,
}


def _validate_suite_tp(name: str, tp: int) -> None:
    """Reject a named measured suite at a different tensor-parallel size."""
    expected = _SUITE_REQUIRED_TP.get(name)
    if expected is not None and tp != expected:
        raise ValueError(
            f"suite {name!r} requires tp={expected}, got tp={tp}"
        )


def parse_shape(spec: str) -> tuple[list[Case], dict]:
    """Parse a ``key=value,...`` shape string into concrete cases.

    Two callers pass no cases of their own, and they need opposite things:

    * The literal ``default`` comes from the task preflight, which only probes
      that the driver answers at all, before any shape is known. One cheap case
      keeps that probe fast.
    * An empty string comes from validation and benchmarking, which pass no
      driver arguments. Those decide KEEP, so they have to measure the whole
      suite -- the crossover sweep, the production row count and fused diagnostics.
      Treating them like the probe scores the campaign on a
      single 1x7168 case and silently drops everything the task is about.
    """
    spec_norm = (spec or "").strip().lower()
    if spec_norm == "default":
        return [Case("raw", 1, 7168, "bf16")], {"tp": str(DEFAULT_TP)}
    if spec_norm == "":
        # forge passes no shape, so a named suite chosen by the operator has no
        # other way in: without this the campaign always measures the derived
        # default while the launcher's SUITE only affects its own self-check.
        name = (os.environ.get("FORGE_COLLECTIVE_SUITE") or "default").strip()
        builder = _SUITE_BUILDERS.get(name)
        if builder is None:
            raise ValueError(
                f"unknown suite {name!r} in FORGE_COLLECTIVE_SUITE "
                f"(known: {', '.join(sorted(_SUITE_BUILDERS))})"
            )
        _validate_suite_tp(name, DEFAULT_TP)
        return builder("bf16"), {"tp": str(DEFAULT_TP), "suite": name}

    kv: dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad shape token: {part!r}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()

    dtype = kv.get("dtype", "bf16")
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    graph = int(kv.get("graph", 0))

    if "suite" in kv:
        # Named measured suites are valid only at their frozen rank count. The
        # default suite derives its cases from the requested rank count.
        tp = int(kv.get("tp", DEFAULT_TP))
        builders = dict(
            _SUITE_BUILDERS,
            default=lambda d: _suite_default(d, tp),
        )
        if kv["suite"] not in builders:
            raise ValueError(f"unknown suite: {kv['suite']} (known: {', '.join(builders)})")
        _validate_suite_tp(kv["suite"], tp)
        cases = builders[kv["suite"]](dtype)
        if graph:
            cases = [Case(c.target, c.rows, c.hidden, c.dtype, 1, c.sensitive) for c in cases]
        return cases, kv

    target = kv.get("target", "raw")
    if target not in ("raw", "fused"):
        raise ValueError(f"unsupported target: {target}")
    rows = int(kv.get("rows", 1))
    hidden = int(kv.get("hidden", 7168))
    return [Case(target, rows, hidden, dtype, graph)], kv


# --------------------------------------------------------------------------
# Self-launch branch
# --------------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _source_hash(root: str) -> str:
    """Hash the kernel sources that feed the custom all-reduce JIT module."""
    h = hashlib.sha256()
    for rel in JIT_SOURCE_FILES:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError as exc:
            # Folding a missing source into a stable "<missing>" marker keeps the
            # digest constant, so the stamp file matches, the JIT rebuild is
            # skipped, and every run afterwards measures a stale binary while
            # reporting the edit as applied. A source that cannot be read means
            # the tree is wrong, so say so.
            raise RuntimeError(
                f"cannot hash JIT source {path!r}: {exc}. The digest guards the "
                "rebuild stamp, so continuing would measure a stale module."
            ) from exc
        h.update(rel.encode())
    return h.hexdigest()


def _jit_dir() -> str:
    return os.environ.get("AITER_JIT_DIR") or os.path.join(_repo_root(), "aiter", "jit")


def _ensure_jit_built(root: str, verbose: bool = True) -> str:
    """Rebuild the JIT module once per source change, guarded by a stamp file.

    Without this gate every Forge sub-process (5 validation stages, bench,
    baseline, in-session gate) would recompile from scratch because
    ``AITER_REBUILD`` is inherited by the whole process tree.
    """
    digest = _source_hash(root)
    stamp = os.path.join(_jit_dir(), STAMP_NAME)
    module = os.path.join(_jit_dir(), f"{JIT_MODULE}.so")
    try:
        with open(stamp) as f:
            if f.read().strip() == digest and os.path.exists(module):
                return digest
    except OSError:
        # No stamp, or it is unreadable: treat the cache as cold and rebuild.
        # A missing stamp is the normal first-run state, not an error.
        pass

    env = dict(os.environ)
    # 2 keeps the ninja cache (1 would wipe it and force a full relink).
    env["AITER_REBUILD"] = "2"
    if verbose:
        print(f"[forge-ar] rebuilding {JIT_MODULE} (source {digest[:12]})", file=sys.stderr)
    rc = subprocess.run(
        [sys.executable, "-c", "import aiter; print(aiter.meta_size())"],
        env=env,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(f"JIT rebuild failed:\n{rc.stdout[-2000:]}\n{rc.stderr[-2000:]}")
    os.makedirs(_jit_dir(), exist_ok=True)
    with open(stamp, "w") as f:
        f.write(digest)
    return digest


_CHILD_PROC: "subprocess.Popen | None" = None


def _kill_child_group(*_args) -> None:
    """Tear down torchrun and, through it, every rank.

    Deliberately signals the torchrun PID rather than a process group. This
    driver stays in the process group its caller created, because that caller
    kills the whole group on timeout and SIGKILL is neither catchable nor
    deliverable across a session boundary: a torchrun in its own session would
    survive the group kill with four GPUs still allocated, and no handler here
    would ever run to clean it up.

    SIGTERM first so torchrun reaps its own workers, then SIGKILL for the case
    where it is wedged.
    """
    global _CHILD_PROC
    if _CHILD_PROC is None:
        return
    proc, _CHILD_PROC = _CHILD_PROC, None
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        proc.wait(timeout=10)
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        # Already gone, not ours to signal, or unreapable. Nothing further we
        # can do here, and raising would mask the original failure.
        pass


def self_launch(argv: list[str], nproc: int) -> int:
    """Re-exec this driver under torchrun and forward its exit code."""
    global _CHILD_PROC

    visible = torch.cuda.device_count()
    if visible < nproc:
        print(
            f"ERROR: need {nproc} visible GPUs for TP{nproc}, found {visible}",
            file=sys.stderr,
        )
        return 2

    root = _repo_root()
    _ensure_jit_built(root)

    env = dict(os.environ)
    # The candidate module is already built; workers must never rebuild.
    env["AITER_REBUILD"] = "0"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        os.path.abspath(__file__),
        *argv,
    ]
    # No start_new_session: staying in the caller's process group is what makes
    # the caller's group-wide timeout kill reach torchrun and every rank.
    proc = subprocess.Popen(cmd, env=env)
    _CHILD_PROC = proc
    atexit.register(_kill_child_group)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _kill_child_group)
    try:
        return proc.wait()
    finally:
        _kill_child_group()


# --------------------------------------------------------------------------
# Worker helpers
# --------------------------------------------------------------------------


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB between a reference and a candidate."""
    ref32 = ref.float()
    err = ref32 - got.float()
    sig = torch.sum(ref32 * ref32).item()
    noise = torch.sum(err * err).item()
    if noise <= 0.0:
        return 200.0
    if sig <= 0.0:
        return -200.0
    return 10.0 * math.log10(sig / noise)


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent RMSNorm reference computed in fp32."""
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps)).to(x.dtype) * weight


@dataclass
class WorkerCtx:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    tp_group: object
    cpu_group: object = None
    ca_comm: object = None
    notes: list[str] = field(default_factory=list)


def init_worker(tp_size: int) -> WorkerCtx:
    """Bind one GPU per rank and bring up the aiter custom-AR communicator."""
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != tp_size:
        raise RuntimeError(f"WORLD_SIZE={world_size} does not match tp={tp_size}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world_size, rank=rank)
    ensure_model_parallel_initialized(tp_size, 1)

    tp_group = get_tp_group()
    ca_comm = getattr(tp_group.device_communicator, "ca_comm", None)
    if ca_comm is None or getattr(ca_comm, "disabled", True):
        raise RuntimeError("custom all-reduce communicator is unavailable (would fall back to RCCL)")

    # Align all ranks before any measurement.
    dist.all_reduce(torch.zeros(1, device=device), group=tp_group.device_group)
    torch.cuda.synchronize()
    return WorkerCtx(rank, local_rank, world_size, device, tp_group, ca_comm=ca_comm)


# Dispatch-check state for this process, held in one mutable object so the
# checker can rebind fields without `global`.
#
# ``checks`` is how many shapes were confirmed to dispatch to custom all-reduce.
# A negative check raises, so a run that reaches the payload has only positive
# ones — but the count still carries information a hardcoded flag cannot: it
# separates "checked and passed" from "never checked at all".
#
# ``probed`` marks the one live probe per process that proves the communicator
# is really serving the custom path; repeating it would add a collective to
# every case.
_CUSTOM_AR = {"checks": 0, "probed": False}


def _assert_custom_ar(ctx: WorkerCtx, x: torch.Tensor, prefill_support: bool) -> None:
    """Fail loudly instead of silently measuring the RCCL fallback."""
    if not ctx.ca_comm.should_custom_ar(x, prefill_support):
        raise RuntimeError(
            f"should_custom_ar() is False for shape {tuple(x.shape)} "
            f"({x.numel() * x.element_size()} bytes, prefill_support={prefill_support}); "
            "the measurement would not exercise the custom kernel"
        )
    if not _CUSTOM_AR["probed"]:
        # should_custom_ar() only predicts from payload size and contiguity: it
        # stays True on a communicator that failed to initialise, and every
        # sample would then time the RCCL fallback while the run still looks
        # correct. custom_all_reduce() returns None on exactly that path
        # (`self.disabled or not should_custom_ar`), so one probe turns the
        # prediction into an observation. Every rank reaches this together --
        # the driver is SPMD -- so the collective inside the probe is safe.
        if ctx.ca_comm.custom_all_reduce(x.clone()) is None:
            raise RuntimeError(
                "custom_all_reduce() returned None: the communicator is disabled "
                "or refused this input, so the measurement would time the RCCL "
                "fallback rather than the kernel under test"
            )
        _CUSTOM_AR["probed"] = True
    _CUSTOM_AR["checks"] += 1


def _quick_reduce_guard() -> None:
    """QuickReduce sits ahead of custom AR in the dispatch chain."""
    regime = os.environ.get("AITER_QUICK_REDUCE_QUANTIZATION", "")
    if regime:
        raise RuntimeError(
            f"AITER_QUICK_REDUCE_QUANTIZATION={regime!r} is set; QuickReduce would "
            "preempt custom all-reduce for large payloads. Unset it before measuring."
        )


# --------------------------------------------------------------------------
# Per-case candidate / reference
# --------------------------------------------------------------------------


def _make_inputs(case: Case, ctx: WorkerCtx, seed: int, mode: str = "smoke") -> dict:
    """Build rank-distinct inputs so a no-op all-reduce cannot pass.

    ``stability`` scales the inputs up so that a shortcut accumulating in low
    precision overflows or loses the tail; BF16 saturates near 3.4e38, and a
    4-way sum of 1e4-scale values still leaves headroom for the reference.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed + ctx.rank)
    dtype = DTYPES[case.dtype]
    shape = (case.rows, case.hidden)
    scale = 1e4 if mode == "stability" else 1.0
    x = (torch.randn(shape, generator=gen, device=ctx.device, dtype=torch.float32) * scale).to(dtype)
    out = {"x": x}
    if case.target == "fused":
        out["residual"] = torch.randn(
            shape, generator=gen, device=ctx.device, dtype=torch.float32
        ).to(dtype)
        out["weight"] = torch.randn(
            (case.hidden,), generator=gen, device=ctx.device, dtype=torch.float32
        ).to(dtype)
        out["eps"] = 1e-6
    return out


def run_candidate(case: Case, ctx: WorkerCtx, inp: dict):
    """Run the code path under optimisation."""
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )

    if case.target == "raw":
        return tensor_model_parallel_all_reduce(inp["x"])
    return tensor_model_parallel_fused_allreduce_rmsnorm(
        inp["x"], inp["residual"], inp["weight"], inp["eps"]
    )


def run_reference(case: Case, ctx: WorkerCtx, inp: dict):
    """Run an independent reference through RCCL plus a fp32 RMSNorm."""
    group = ctx.tp_group.device_group
    ar = inp["x"].clone()
    dist.all_reduce(ar, group=group)
    if case.target == "raw":
        return ar
    residual_out = ar + inp["residual"]
    out = _rmsnorm_ref(residual_out, inp["weight"], inp["eps"])
    return out, residual_out


# --------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------


def check_case(case: Case, ctx: WorkerCtx, seed: int, mode: str = "smoke") -> dict:
    """Compare candidate against reference on this rank."""
    inp = _make_inputs(case, ctx, seed, mode)
    _assert_custom_ar(ctx, inp["x"], prefill_support=False)

    ref = run_reference(case, ctx, inp)
    got = run_candidate(case, ctx, inp)
    torch.cuda.synchronize()

    if case.target == "raw":
        pairs = [("out", ref, got)]
    else:
        pairs = [("out", ref[0], got[0]), ("residual_out", ref[1], got[1])]

    snr = 200.0
    max_diff = 0.0
    finite = True
    for _name, r, g in pairs:
        snr = min(snr, _snr_db(r, g))
        max_diff = max(max_diff, (r.float() - g.float()).abs().max().item())
        finite = finite and bool(torch.isfinite(g).all().item())
    return {"snr_db": snr, "max_diff": max_diff, "finite": finite}


def check_graph_case(case: Case, ctx: WorkerCtx, seed: int, mode: str = "smoke") -> dict:
    """Same comparison, but with the candidate captured into a CUDA graph.

    ``graph_capture()`` already wraps ``ca_comm.capture()``, which is what
    flushes the IPC buffer registrations on exit; capturing outside it raises
    from the extension.
    """
    from aiter.dist.parallel_state import graph_capture

    static_inp = _make_inputs(case, ctx, seed, mode)
    replay_inputs = (
        static_inp,
        _make_inputs(case, ctx, seed + 1_000_003, mode),
    )
    _assert_custom_ar(ctx, static_inp["x"], prefill_support=False)

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            got = run_candidate(case, ctx, static_inp)

    snr = 200.0
    max_diff = 0.0
    finite = True
    for replay_inp in replay_inputs:
        ref = run_reference(case, ctx, replay_inp)
        for name, value in replay_inp.items():
            if torch.is_tensor(value) and value is not static_inp[name]:
                static_inp[name].copy_(value)
        graph.replay()
        torch.cuda.synchronize()

        # Validate before loading the next input so a stale replay output cannot
        # be overwritten or compared against the previous replay's reference.
        if case.target == "raw":
            pairs = [("out", ref, got)]
        else:
            pairs = [("out", ref[0], got[0]), ("residual_out", ref[1], got[1])]
        for _name, r, g in pairs:
            snr = min(snr, _snr_db(r, g))
            max_diff = max(max_diff, (r.float() - g.float()).abs().max().item())
            finite = finite and bool(torch.isfinite(g).all().item())
    return {"snr_db": snr, "max_diff": max_diff, "finite": finite}


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def bench_case(case: Case, ctx: WorkerCtx, warmup: int, iters: int, seed: int) -> float:
    """Return the median slowest-rank sample latency in ms for one case."""
    inp = _make_inputs(case, ctx, seed)
    _assert_custom_ar(ctx, inp["x"], prefill_support=False)

    for _ in range(warmup):
        run_candidate(case, ctx, inp)
    torch.cuda.synchronize()

    # Capture a chain of collectives into one CUDA graph, then replay it.
    #
    # Two earlier approaches were measured and rejected:
    #   * one call per barrier -> 6-10% run-to-run spread, because a rank leaving
    #     the barrier late makes the collective wait and that jitter lands in the
    #     sample;
    #   * an eager burst -> stable but CPU-bound: every size from 16 KiB to
    #     512 KiB reported the same ~21.7 us, i.e. the Python dispatch cost, not
    #     the kernel.
    # Replaying a captured chain removes the per-call CPU cost while keeping the
    # collectives serialised on one stream, so the result is real device-side
    # latency. graph_capture() also wraps ca_comm.capture(), which is what
    # flushes the IPC buffer registrations on exit.
    from aiter.dist.parallel_state import graph_capture

    chain = max(1, iters)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(chain):
                run_candidate(case, ctx, inp)

    for _ in range(max(1, warmup // 10)):
        graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(5):
        dist.barrier(group=ctx.tp_group.device_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        sample_ms = start.elapsed_time(end) / chain
        samples.append(_reduce_max(sample_ms, ctx))
    del graph
    return statistics.median(samples)


def _reduce_max(value: float, ctx: WorkerCtx) -> float:
    """Slowest rank wins: a collective is only as fast as its laggard."""
    t = torch.tensor([value], device=ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ctx.tp_group.device_group)
    return float(t.item())


def _reduce_min(value: float, ctx: WorkerCtx) -> float:
    t = torch.tensor([value], device=ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=ctx.tp_group.device_group)
    return float(t.item())


def _geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


# --------------------------------------------------------------------------
# Worker entry
# --------------------------------------------------------------------------


def _emit_sentinel(payload: dict) -> None:
    """Emit exactly one single-line sentinel.

    Multi-line JSON would be torn apart by the other ranks writing to the same
    pipe; a compact line under PIPE_BUF is written atomically.
    """
    line = SENTINEL + json.dumps(payload, separators=(",", ":")) + SENTINEL
    if len(line.encode()) >= 4096:
        raise RuntimeError(f"sentinel too large for an atomic write: {len(line)} bytes")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def worker_main(args: argparse.Namespace) -> int:
    _quick_reduce_guard()
    cases, kv = parse_shape(args.shape)
    tp = int(kv.get("tp", DEFAULT_TP))
    ctx = init_worker(tp)
    rank0 = ctx.rank == 0

    try:
        if args.bench_mode or args.profile_run:
            if args.profile_run and args.profile_case:
                cases = [c for c in cases if c.case_id == args.profile_case] or cases[:1]

            # Repeat the whole sweep in-process and keep the per-case MEDIAN of
            # the round medians. Process-to-process variation (fresh IPC buffers,
            # clock state) dominates the run-to-run spread, so repeating inside
            # one launch is far cheaper than relaunching torchrun.
            #
            # Median, not min: min always picks the luckiest round, which biases
            # the estimate low and stays extremal-sensitive no matter how many
            # rounds are added. The keep/revert gate compares two such estimates,
            # so a biased-but-noisy statistic wastes the whole repeat budget.
            per_case: dict[str, float] = {}
            rounds: list[dict[str, float]] = []
            for _ in range(max(1, args.repeat)):
                this_round: dict[str, float] = {}
                for case in cases:
                    this_round[case.case_id] = bench_case(
                        case, ctx, args.warmup, args.iters, args.seed
                    )
                rounds.append(this_round)
            for case in cases:
                per_case[case.case_id] = statistics.median(
                    r[case.case_id] for r in rounds
                )
            if rank0 and args.repeat > 1:
                for case in cases:
                    vals = [r[case.case_id] for r in rounds]
                    spread = 100.0 * (max(vals) - min(vals)) / max(sum(vals) / len(vals), 1e-12)
                    print(
                        f"[forge-ar] in-process spread {case.case_id}: {spread:.2f}%",
                        file=sys.stderr,
                    )

            if args.profile_run:
                # Profiling only needs the kernels to run; no output contract.
                return 0

            groups: dict[str, dict] = {}
            for case in cases:
                g = groups.setdefault(case.group, {"cases": []})
                # ``scored`` travels with the case so consumers do not have to
                # re-derive which cases back the score from a second list that
                # can drift out of step with this one.
                g["cases"].append({
                    "case_id": case.case_id,
                    "median_ms": per_case[case.case_id],
                    "scored": case.sensitive,
                })

            if rank0:
                # Mark the cases outside the score. Profiling picks the slowest
                # case to analyse, and the slowest here is an excluded one, so
                # without the mark the whole profile-and-optimize chain aims at
                # a shape the gate never reads.
                scored_by_id = {c.case_id: c.sensitive for c in cases}
                for cid, ms in per_case.items():
                    tag = "" if scored_by_id.get(cid, True) else " unscored"
                    print(f"case_ms: {cid} {ms:.6f}{tag}")
                print(f"mean_ms: {_geomean(list(per_case.values())):.6f}")
                payload = {
                    "kind": "integrated_bench",
                    "world_size": ctx.world_size,
                    "metrics": groups,
                    # Measured, not asserted: every benched shape passed a
                    # dispatch check, and the count says how many did. A run
                    # that somehow benched nothing reports False rather than
                    # claiming a custom path it never exercised.
                    "custom_ar_active": _CUSTOM_AR["checks"] > 0,
                    "custom_ar_checks": _CUSTOM_AR["checks"],
                    "source_hash": _source_hash(_repo_root())[:16],
                }
                _emit_sentinel(payload)
                if args.dump_json:
                    with open(args.dump_json, "w") as f:
                        json.dump(payload, f, indent=2)
            return 0

        # Correctness modes. The loop's formal validation passes no arguments,
        # so every benchmark case runs both eager and graph correctness. An
        # explicit --mode keeps graph= selection for focused diagnostics.
        #
        # The default, smoke, uses unit-scale inputs, where a rank publishing
        # its buffer before its peers have read the previous one still produces
        # a plausible sum -- the known race in the publish path survives it.
        # stability scales inputs by 1e4 so a shortcut accumulating in low
        # precision, or a read of a half-updated buffer, moves the result far
        # enough to fail SNR. Validating only under smoke is what lets that
        # class of defect reach a KEEP.
        modes = [args.mode] if args.mode else ["smoke", "stability"]
        worst_snr = 200.0
        worst_diff = 0.0
        all_finite = True
        for mode in modes:
            for case in cases:
                if args.mode is None:
                    checks = (check_case, check_graph_case)
                else:
                    checks = (check_graph_case,) if case.graph else (check_case,)
                for fn in checks:
                    res = fn(case, ctx, args.seed, mode)
                    worst_snr = min(worst_snr, res["snr_db"])
                    worst_diff = max(worst_diff, res["max_diff"])
                    all_finite = all_finite and res["finite"]

        worst_snr = _reduce_min(worst_snr, ctx)
        worst_diff = _reduce_max(worst_diff, ctx)
        finite_flag = _reduce_min(1.0 if all_finite else 0.0, ctx)
        passed = bool(finite_flag) and worst_snr >= args.snr_threshold

        if rank0:
            print(f"SNR: {worst_snr:.2f} dB")
            print(f"allclose: {passed}")
            print(f"max_diff: {worst_diff:.6e}")
        # The parser treats SNR as authoritative, so a failure must also be
        # visible in the exit code.
        return 0 if passed else 1
    finally:
        from aiter.dist.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forge TP8 all-reduce driver (Kimi-K3)")
    # The driver owns case selection, so the default must be the full scored
    # suite rather than one probe case.
    p.add_argument("--shape", default="", help="e.g. suite=tp8_k3,tp=8,dtype=bf16; empty derives the rank count from the launcher")
    # default=None distinguishes "caller chose smoke" from "caller said
    # nothing", which decides whether the full correctness matrix runs.
    p.add_argument("--mode", default=None,
                   choices=["smoke", "stability", "determinism"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bench-mode", action="store_true")
    p.add_argument("--profile-run", action="store_true")
    p.add_argument("--profile-case", default="")
    p.add_argument("--snr-threshold", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--repeat", type=int, default=1, help="in-process repeats of the sweep")
    p.add_argument("--dump-json", default="", help="also write the payload to this path")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        return worker_main(args)

    _, kv = parse_shape(args.shape)
    tp = int(kv.get("tp", DEFAULT_TP))
    return self_launch(sys.argv[1:], tp)


if __name__ == "__main__":
    sys.exit(main())
