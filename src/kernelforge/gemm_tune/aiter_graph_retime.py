# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""In-graph, cache-cold timing of tuned GEMM candidates, the way decode serves them.

aiter's tuner times each candidate eagerly with the weights hot in cache, while sglang replays decode inside a CUDA
graph with weights the previous layer evicted; the two rank some decode shapes differently. Like
:mod:`.aiter_splitk_validate`, the timing is per op: :data:`_RETIME_OPS` registers a timer per tuner ``script_key``,
and an unregistered op gets no timer, so the tuner's pick stands.

Every candidate runs through the production op serving dispatches. aiter builds the production ck and cktile modules
from the tuned table, so they hold only that table's instances, and the ck tune module is built without the
production compiler flags (measured up to 5.5% slower on the same instance). The timing therefore runs in a worker
process whose private AITER JIT directory rebuilds those two modules from the candidates and reuses every other one.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.aiter_jit_cache import resolve_jit_build_dir, resolve_package_root

from .aiter_splitk_validate import _operands, _resolve_device

# Weight copies rotated per call so each call reads weights evicted by the others: above MI355X's 256 MiB MALL.
_ROTATE_BYTES = 2 << 30
_MAX_COPIES = 32
_CALLS_PER_COPY = 10
_REPLAYS = 7
# The weight layout sglang's blockscale bpreshuffle path serves.
_SHUFFLE_LAYOUT = (16, 16)
_WORKER_LOG_TAIL = 2000


@dataclass(frozen=True)
class Kernel:
    """One tuned kernel configuration, as a table row names it."""

    libtype: str
    kernel_id: int
    kernel_name: str
    split_k: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Kernel:
        return cls(row["libtype"].strip().lower(), int(row["kernelId"]), row["kernelName"], int(row["splitK"]))


@dataclass(frozen=True)
class Timing:
    """Microseconds per call of one candidate inside a CUDA graph with cold weights, or why it could not run."""

    us: float | None = None
    error: str = ""


@dataclass(frozen=True)
class RetimeTarget:
    """Whose rows are timed and where: ``op``, the tuner's ``gpu_ids``, and the clock time (``time.time``) the tuner's
    budget ends at, which the timing must finish by."""

    op: str
    gpu_ids: str
    deadline: float


class RetimeError(RuntimeError):
    """The timing run as a whole failed, so no candidate was timed."""


#: Tuned-table rows (aiter's tuned CSV schema) -> one Timing per row, in order; raises RetimeError when the run fails.
RetimeFn = Callable[[Sequence[dict[str, str]]], list[Timing]]


@dataclass(frozen=True)
class _ProductionModules:
    """The op's production modules that aiter builds from a tuned table, and the variable naming that table."""

    config_env: str
    modules: tuple[str, ...]


_BPRESHUFFLE_MODULES = _ProductionModules(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    ("module_gemm_a8w8_blockscale_bpreshuffle", "module_gemm_a8w8_blockscale_bpreshuffle_cktile"),
)


class _WorkerRetime:
    """Times rows in a worker process on ``device``, with ``production``'s modules built from the rows themselves."""

    def __init__(self, target: RetimeTarget, production: _ProductionModules, run_dir: Path):
        self._target, self._device, self._production, self._run_dir = (
            target,
            _resolve_device(target.gpu_ids),
            production,
            run_dir,
        )

    def __call__(self, rows: Sequence[dict[str, str]]) -> list[Timing]:
        seconds_left = self._target.deadline - time.time()
        if seconds_left <= 0:
            raise RetimeError("the tuner's time budget is spent")
        jobs, build, results = (self._run_dir / name for name in ("jobs.csv", "build.csv", "timings.json"))
        jit_dir = self._private_jit_dir()
        _write_rows(jobs, rows)
        # The build keys instances by (gfx, cu_num, M, N, K); a distinct M per row keeps every candidate's instance.
        _write_rows(build, [{**row, "M": str(i + 1)} for i, row in enumerate(rows)])
        results.unlink(missing_ok=True)
        env = {**os.environ, "AITER_JIT_DIR": str(jit_dir), self._production.config_env: str(build)}
        command = [sys.executable, "-m", __name__, self._target.op, self._device, str(jobs), str(results)]
        log_path = self._run_dir / "worker.log"
        try:
            with log_path.open("w") as log_file:
                rc = subprocess.run(
                    command, env=env, stdout=log_file, stderr=subprocess.STDOUT, timeout=seconds_left, check=False
                ).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RetimeError(f"timing worker did not finish: {exc}") from exc
        if rc != 0 or not results.is_file():
            raise RetimeError(f"timing worker exited {rc}; {log_path} ends: {_tail(log_path)}")
        return [Timing(**timing) for timing in json.loads(results.read_text())]

    def _private_jit_dir(self) -> Path:
        """A JIT directory linking every built module except the production ones, which the worker rebuilds."""
        build_dir = resolve_jit_build_dir(resolve_package_root())
        if build_dir is None or not build_dir.parent.is_dir():
            raise RetimeError("the AITER JIT directory could not be located")
        jit_dir = self._run_dir / "jit"
        shutil.rmtree(jit_dir, ignore_errors=True)
        jit_dir.mkdir(parents=True)
        rebuilt = {f"{module}.so" for module in self._production.modules}
        for module in build_dir.parent.glob("*.so"):
            if module.name not in rebuilt:
                (jit_dir / module.name).symlink_to(module.resolve())
        return jit_dir


def _write_rows(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tail(path: Path) -> str:
    try:
        return path.read_text(errors="replace")[-_WORKER_LOG_TAIL:]
    except OSError as exc:
        return f"(unreadable: {exc})"


class _BpreshuffleGraphTimer:
    """Times a8w8_blockscale_bpreshuffle kernels on one device, reusing the operands of the last shape."""

    def __init__(self, device: str):
        self._device = device
        self._weights_key: tuple[int, int] | None = None
        self._weights: tuple[list, object] = ([], None)
        self._acts_key: tuple[int, int, int] | None = None
        self._acts: tuple = ()

    def __call__(self, m: int, n: int, k: int, kernel: Kernel) -> float:
        call = self._call(kernel)
        weights, w_scale = self._weights_for(n, k)
        x, x_scale, outs = self._acts_for(m, n, k, len(weights))
        return _time_in_graph(
            lambda: [call(x, w, x_scale, w_scale, out) for w, out in zip(weights, outs)], len(weights)
        )

    @staticmethod
    def _call(kernel: Kernel):
        """The production call aiter's tuned dispatch makes for ``kernel``; asm captures its split-K memset with it."""
        import aiter

        if kernel.libtype == "asm":
            return lambda x, w, xs, ws, out: aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
                x, w, out, xs, ws, splitK=kernel.split_k, kernelName=kernel.kernel_name
            )
        if kernel.libtype == "ck":
            return lambda x, w, xs, ws, out: aiter.gemm_a8w8_blockscale_bpreshuffle_ck(
                x, w, xs, ws, out, kernelName=kernel.kernel_name
            )
        if kernel.libtype == "cktile":
            return lambda x, w, xs, ws, out: aiter.gemm_a8w8_blockscale_bpreshuffle_cktile(
                x, w, xs, ws, out, kernelName=kernel.kernel_name
            )
        raise RuntimeError(f"no in-graph call for libtype {kernel.libtype!r}")

    def _weights_for(self, n: int, k: int):
        if self._weights_key != (n, k):
            from aiter.ops.shuffle import shuffle_weight

            # Release the previous shape's copies before allocating the next.
            self._weights_key, self._weights = None, ([], None)
            _, w, _, w_scale, _ = _operands(1, n, k, self._device)
            copies = max(2, min(_MAX_COPIES, _ROTATE_BYTES // (n * k)))
            self._weights = ([shuffle_weight(w.clone(), layout=_SHUFFLE_LAYOUT) for _ in range(copies)], w_scale)
            self._weights_key = (n, k)
        return self._weights

    def _acts_for(self, m: int, n: int, k: int, copies: int):
        if self._acts_key != (m, n, k):
            import torch

            x, _, x_scale, _, _ = _operands(m, n, k, self._device)
            # sglang passes x_scale column-major.
            x_scale = x_scale.t().contiguous().view(m, -1)
            outs = [torch.empty(m, n, dtype=torch.bfloat16, device=self._device) for _ in range(copies)]
            self._acts, self._acts_key = (x, x_scale, outs), (m, n, k)
        return self._acts


def _time_in_graph(step: Callable[[], object], calls_per_step: int) -> float:
    """Median microseconds per call of ``step`` replayed inside a CUDA graph."""
    import torch

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        step()
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(_CALLS_PER_COPY):
            step()
    times = []
    for _ in range(_REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (_CALLS_PER_COPY * calls_per_step))
    return sorted(times)[len(times) // 2]


@dataclass(frozen=True)
class _RetimeOp:
    """How one op's candidates are timed: the in-worker timer and the production modules the worker rebuilds."""

    timer: Callable[[str], Callable[[int, int, int, Kernel], float]]
    production: _ProductionModules


#: Ops whose candidates this module can time in a graph, keyed by the tuner ``script_key``.
_RETIME_OPS = {"a8w8_blockscale_bpreshuffle": _RetimeOp(_BpreshuffleGraphTimer, _BPRESHUFFLE_MODULES)}


def make_retime_fn(target: RetimeTarget, run_dir: Path) -> RetimeFn | None:
    """The in-graph timer for ``target``'s op on its card, working under ``run_dir``; None when unregistered."""
    if target.op not in _RETIME_OPS:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    return _WorkerRetime(target, _RETIME_OPS[target.op].production, run_dir)


def _time_rows(op: str, device: str, rows: Sequence[dict[str, str]]) -> list[Timing]:
    """Each row's in-graph time; a row whose kernel cannot dispatch records the error and the others still run."""
    timer = _RETIME_OPS[op].timer(device)
    timings = []
    for row in rows:
        try:
            timings.append(Timing(us=timer(int(row["M"]), int(row["N"]), int(row["K"]), Kernel.from_row(row))))
        except RuntimeError as exc:
            timings.append(Timing(error=str(exc)))
    return timings


def _worker_main(argv: Sequence[str]) -> None:
    op, device, jobs, results = argv
    with open(jobs, newline="") as handle:
        rows = list(csv.DictReader(handle))
    timings = _time_rows(op, device, rows)
    Path(results).write_text(json.dumps([{"us": t.us, "error": t.error} for t in timings]))


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
