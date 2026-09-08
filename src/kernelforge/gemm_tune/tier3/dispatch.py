# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turning a proposed candidate into something the referee can time.

The referee deliberately refuses to interpret a config: a candidate only means
anything against the backend it names, and a wrong guess about what it meant
would be timed as if it were right. So somebody has to supply the three
things it cannot write generically -- what the unmodified path is, how to run
a candidate, and how to tell whether the answer is correct.

That is what this module is, for the tables we can actually dispatch today.
A table with no adapter here yields ``None``, and the attempt stops at the
referee with "no dispatch supplied" -- which is the right outcome, because the
alternative is emitting a tuner nobody re-timed.

Two things in here were learned by getting them wrong on real hardware:

* **Time graph replays, not individual calls.** Python dispatch on MI355X
  costs ~12us, and the kernels under test are 5-13us. Handing the referee raw
  single-kernel callables buries every candidate under the same overhead and
  compresses the ratios toward 1.0, which reads as "nothing to tune here".
* **Do not measure error element by element.** Dividing by each reference
  element (however floored) lets any output that lands near zero dominate, and
  a large-K random GEMM produces plenty of those. By that measure the
  unmodified ``torch.matmul`` scores 1.375 against its own fp32 reference, so
  a gate on it rejects the default path. Error is measured against the
  magnitude of the reference as a whole.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Back-to-back invocations inside one captured graph. Large enough that the
# per-replay overhead is small against the kernel, small enough to capture.
GRAPH_INNER = 20

# Fresh-input correctness repeats, worst result counted. One check passes an
# intermittently wrong kernel roughly at random; this is not hypothetical, four
# such kernels were selected as winners on this hardware before it was
# understood.
CORRECTNESS_TRIALS = 8

# See the module docstring for why this is not an element-wise ratio.
MAX_RELATIVE_ERROR = 5e-2

# Tables this module knows how to exercise. Everything else is honestly absent
# rather than approximated.
SUPPORTED_TABLES = ("bf16_tuned_gemm.csv",)


def adapters_for(table: str) -> Any | None:
    """Return the dispatch adapter for a table, or None if we have none.

    None is a real answer: the runner then stops before the referee rather
    than emitting candidates nobody re-timed.
    """
    if table == "bf16_tuned_gemm.csv":
        return _Bf16DenseAdapter()
    log.info(
        "tier3: no dispatch adapter for %s, so a generated tuner for it could not be re-timed; supported today: %s",
        table,
        ", ".join(SUPPORTED_TABLES),
    )
    return None


def parse_config(cfg: Any) -> dict[str, Any]:
    """``a=1;b=True;c=x`` into a dict, recovering ints and bools."""
    out: dict[str, Any] = {}
    for part in str(cfg).split(";"):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if value in ("True", "False"):
            out[key] = value == "True"
            continue
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def shape_key(shape: str) -> tuple[int, int, int]:
    """``"16x1536x7168"`` into ``(16, 1536, 7168)``.

    Raises ``ValueError`` on anything else. It used to return ``()`` instead,
    which did not spare any caller: every one of them unpacks the result into
    three names, so a malformed shape became a bare ``ValueError`` about tuple
    lengths several frames away -- and ``"16x1536"`` parsed "successfully" into
    a 2-tuple that failed the same way. Failing here says which shape and why;
    the caller in ``cli.py`` already treats that as "tier3 attempt failed;
    tuning continues".
    """
    parts = str(shape).split("x")
    if len(parts) != 3:
        raise ValueError(f"tier3 shape must be MxNxK, got {shape!r}")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"tier3 shape must be MxNxK of integers, got {shape!r}") from exc
    return (m, n, k)


class _Bf16DenseAdapter:
    """Dispatch, baseline and correctness for row-major bf16 A[M,K] x B[N,K]^T.

    Holds the operands per shape so timing measures the kernel rather than
    allocation, and rebuilds against fresh ones for every correctness trial.
    """

    def __init__(self) -> None:
        self._operands: dict[tuple[int, ...], tuple[Any, Any]] = {}
        self._in_play: dict[str, dict[str, Any]] = {}
        self._hipb_ready = False

    # -- torch is imported lazily so this module stays importable off-GPU --
    @staticmethod
    def _torch():
        import torch

        return torch

    def _ops(self, key: tuple[int, int, int]):
        if key not in self._operands:
            torch = self._torch()
            m, n, k = key
            torch.manual_seed(0)
            self._operands[key] = (
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
            )
        return self._operands[key]

    def as_graph(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Replay many invocations per call, so dispatch cost is amortised.

        Falls back to the raw callable when capture fails: a kernel that cannot
        be captured is still worth timing, just less precisely.
        """
        torch = self._torch()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(5):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(GRAPH_INNER):
                    fn()
            return graph.replay
        except Exception as exc:  # noqa: BLE001 - capture is an optimisation
            log.debug("tier3: graph capture failed, timing raw: %r", exc)
            return fn

    def make_baseline(self, shape: str) -> Callable[[], Any]:
        torch = self._torch()
        key = shape_key(shape)
        a, b = self._ops(key)
        return self.as_graph(lambda: torch.matmul(a, b.t()))

    def make_dispatch(self, shape: str) -> Callable[[dict[str, Any]], Callable[[], Any] | None]:
        key = shape_key(shape)

        def dispatch(cand: dict[str, Any]) -> Callable[[], Any] | None:
            # The correctness check runs straight after this and needs to know
            # which candidate is in play, because it has to rebuild against
            # fresh inputs rather than reuse this callable's fixed operands.
            self._in_play[shape] = cand
            call = self._build(key, cand)
            return self.as_graph(call) if call is not None else None

        return dispatch

    def make_correctness(self, shape: str) -> Callable[[Callable[[], Any]], bool]:
        key = shape_key(shape)

        def check(_dispatched: Callable[[], Any]) -> bool:
            cand = self._in_play.get(shape)
            if cand is None:
                return True
            return self._is_correct(key, cand)

        return check

    def sync(self) -> Callable[[], Any]:
        return self._torch().cuda.synchronize

    # ------------------------------------------------------------ internals --
    def _hipb_once(self, a, bt) -> None:
        """hipb_mm on a handle nobody created aborts from C++, uncatchably."""
        if self._hipb_ready:
            return
        import aiter

        torch = self._torch()
        aiter.hipb_findallsols(a, bt, None, torch.bfloat16, None, None, None, False, False)
        self._hipb_ready = True

    def _build(self, key: tuple[int, int, int], cand: dict[str, Any]) -> Callable[[], Any] | None:
        """One candidate as a callable, or None when we cannot dispatch it.

        None is recorded by the referee as "not dispatchable", which is a
        result worth having; approximating what the candidate meant is not.
        """
        import aiter

        torch = self._torch()
        backend = str(cand.get("backend", ""))
        cfg = parse_config(cand.get("config", ""))
        m, n, _k = key
        a, b = self._ops(key)

        try:
            if backend == "torch":
                return lambda: torch.matmul(a, b.t())

            if backend == "hipblaslt":
                sol = cfg.get("solidx")
                if sol is None:
                    return None
                bt = b.t()
                self._hipb_once(a, bt)
                return lambda: aiter.hipb_mm(a, bt, sol, None, torch.bfloat16, None, None, None, False, False)

            if backend == "aiter_asm":
                name = cfg.get("kernelName")
                if not name:
                    return None
                split_k = cfg.get("splitK", 0)
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                return lambda: aiter.gemm_a16w16_asm(a, b, out, None, split_k, name, False)

            if backend == "aiter_opus":
                from aiter.ops.opus import gemm_op_a16w16 as opus

                kernel_id = cfg.get("kernelId")
                if kernel_id is None:
                    return None
                init = getattr(opus, "opus_gemm_workspace_init", None)
                if init:
                    init()
                a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
                y = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
                return lambda: opus.opus_gemm_a16w16_tune(
                    a3, b3, y, bias=None, kernelId=kernel_id, splitK=cfg.get("splitK", 1)
                )

            if backend == "aiter_flydsl":
                import aiter.ops.flydsl.gemm_kernels as fly

                return lambda: fly.flydsl_hgemm(
                    a,
                    b,
                    bias=None,
                    kernel_family="hgemm",
                    tile_m=cfg.get("tile_m"),
                    tile_n=cfg.get("tile_n"),
                    tile_k=cfg.get("tile_k"),
                    split_k=cfg.get("split_k", 1),
                    block_m_warps=cfg.get("block_m_warps", 1),
                    block_n_warps=cfg.get("block_n_warps", 1),
                    block_k_warps=cfg.get("block_k_warps", 1),
                    stages=cfg.get("stages", 4),
                    async_copy=cfg.get("async_copy", True),
                    b_to_lds=cfg.get("b_to_lds", True),
                    b_preshuffle=False,
                    c_to_lds=False,
                )
        except Exception as exc:  # noqa: BLE001 - undispatchable is data
            log.info("tier3: cannot dispatch %s: %r", backend, exc)
            return None

        log.info("tier3: unknown backend %r in a candidate", backend)
        return None

    def _is_correct(self, key: tuple[int, int, int], cand: dict[str, Any]) -> bool:
        torch = self._torch()
        m, n, k = key
        saved = self._operands.get(key)
        worst = 0.0
        try:
            for _ in range(CORRECTNESS_TRIALS):
                self._operands[key] = (
                    torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                    torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
                )
                a, b = self._operands[key]
                call = self._build(key, cand)
                if call is None:
                    return False
                got = call()
                torch.cuda.synchronize()
                if got is None:
                    return False
                ref = torch.matmul(a.float(), b.float().t())
                worst = max(worst, relative_error(got, ref))
        except Exception as exc:  # noqa: BLE001 - a kernel that raises is wrong
            log.warning("tier3: correctness check raised, rejecting: %r", exc)
            return False
        finally:
            if saved is not None:
                self._operands[key] = saved
            else:
                self._operands.pop(key, None)

        if worst > MAX_RELATIVE_ERROR:
            log.error(
                "tier3: rejecting %s %s -- worst error over %d fresh inputs was %.4g, above the %.3g limit",
                cand.get("backend"),
                str(cand.get("config"))[:60],
                CORRECTNESS_TRIALS,
                worst,
                MAX_RELATIVE_ERROR,
            )
            return False
        return True


def relative_error(got: Any, ref: Any) -> float:
    """Largest deviation, against the magnitude of the reference as a whole.

    Not element-wise: see the module docstring for the measurement that
    rejected the default path.
    """
    return float((got.float() - ref).abs().max() / ref.abs().mean())
