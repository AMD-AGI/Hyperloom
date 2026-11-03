"""Measurement driver for the triton2flydsl softmax rewrite task (BYOD).

`forge-rewrite-by-flydsl` treats this driver as a black box invoked as
``python driver.py <args>`` and talks to it purely over stdout. It is the single
source of truth for how the FlyDSL port is called + checked and for both
baselines, and it is protected (never edited by the pipeline). It plays two roles
at once — the correctness ORACLE (the original Triton kernel) and the perf
MEASURER for both the source and the FlyDSL candidate:

  * Correctness   ``python driver.py`` -> runs the complete suite, compares the
    FlyDSL candidate against the source Triton output, and prints SNR/allclose.

  * FlyDSL bench  ``python driver.py --warmup <n> --iters <n>
    --bench-mode`` -> times the FLYDSL candidate (graph replay). Prints per-iter
    ``wall_ms`` samples, a ``median_ms`` aggregate, and one ``case_ms`` line.

  * Source bench  ``python driver.py --warmup <n> --iters <n>
    --ref-bench-mode`` -> times the SOURCE Triton kernel (the speedup baseline).
    Prints ``median_ms``.

  * Profiling     ``python driver.py --profile-run`` -> the driver selects the
    profile case and runs only the FlyDSL candidate, with no reference/timing/checks.

Interface the FlyDSL port MUST expose (this driver defines it):
    build_softmax_module(M, N, dtype_str) -> launch_fn
    launch_fn(A, C, m_rows, stream=fx.Stream(...))          # C = softmax(A) rowwise

Stream routing lives HERE (not in the kernel): the launcher takes a ``stream``
kwarg and this driver always passes the CURRENT stream, so under CUDA-graph
capture the launch is recorded into the graph. Keeping it in the protected driver
means the port cannot break capture by editing the kernel.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys

import torch

from graph_harness import cuda_graph_bench

# The SOURCE kernel we port FROM: its host entry is the correctness oracle AND the
# speedup baseline. Protected during the rewrite.
from softmax import softmax as _source_softmax

# Driver-owned scored case.
_DEFAULT_M = 4096
_DEFAULT_N = 1024
_DEFAULT_DTYPE = "f32"

# Fixed seed so every full-suite invocation builds identical inputs.
_SEED = 0

_TORCH_DTYPE = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}

# build_softmax_module JIT-compiles per (M, N, dtype); cache so correctness and
# bench of the same shape do not recompile.
_MODULE_CACHE: dict[tuple[int, int, str], object] = {}


def _case_id(rows: int, cols: int, dtype: str) -> str:
    """Return the opaque token emitted by benchmark mode."""
    return f"M{rows}_N{cols}_{dtype}"


def _build(rows: int, cols: int, dtype: str):
    """Build (and cache) the FlyDSL candidate launch callable for this shape.

    Imported lazily so that source-only paths (``--ref-bench-mode``) still work
    even while the ported ``kernel.py`` is an unimplemented skeleton.
    """
    key = (rows, cols, dtype)
    if key not in _MODULE_CACHE:
        from kernel import build_softmax_module  # the ported FlyDSL kernel

        _MODULE_CACHE[key] = build_softmax_module(rows, cols, dtype)
    return _MODULE_CACHE[key]


def _make_input(rows: int, cols: int, dtype: str, mode: str, device: str) -> torch.Tensor:
    """Build the softmax input for a given validation mode (deterministic)."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, cols, device=device, dtype=_TORCH_DTYPE[dtype])
    if mode == "stability":
        # Large magnitudes stress the max-subtraction; a kernel that skips it
        # overflows exp() and fails here.
        x = x * 50.0
    return x


def _launch_on_current_stream(launch_fn, x: torch.Tensor, out: torch.Tensor, rows: int) -> None:
    """Run the FlyDSL kernel on whatever stream is currently active.

    Queried at call time on purpose: under torch.cuda.graph the active stream is
    the private capture stream, so the launch gets recorded into the graph.
    """
    import flydsl.expr as fx

    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    launch_fn(x, out, rows, stream=stream)


def _reference(x: torch.Tensor) -> torch.Tensor:
    """The SOURCE Triton kernel output — the numbers the FlyDSL port must match."""
    return _source_softmax(x)


def _snr_db(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB between the reference and the kernel output."""
    reference = reference.float()
    test = test.float()
    noise = test - reference
    signal_power = torch.mean(reference * reference).item()
    noise_power = torch.mean(noise * noise).item()
    if noise_power <= 0.0:
        return 100.0
    if signal_power <= 0.0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _run_correctness(rows: int, cols: int, dtype: str, mode: str, device: str) -> int:
    x = _make_input(rows, cols, dtype, mode, device)
    out = torch.empty_like(x)
    launch_fn = _build(rows, cols, dtype)
    _launch_on_current_stream(launch_fn, x, out, rows)
    torch.cuda.synchronize()

    ref = _reference(x)
    print(f"SNR: {_snr_db(ref, out):.2f} dB")
    print(f"allclose: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return 0


def _run_bench(rows: int, cols: int, dtype: str, warmup: int, iters: int, device: str) -> int:
    """Time the FLYDSL candidate under CUDA-graph replay."""
    x = _make_input(rows, cols, dtype, "full", device)
    out = torch.empty_like(x)
    launch_fn = _build(rows, cols, dtype)
    ref = _reference(x)

    def step():
        _launch_on_current_stream(launch_fn, x, out, rows)

    # dirty + verify prove the graph actually captured the kernel (an uncaptured
    # launch would leave `out` at its dirtied value and fail verify -> eager).
    result = cuda_graph_bench(
        step,
        warmup=warmup,
        iters=iters,
        dirty=lambda: out.zero_(),
        verify=lambda: torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
    )

    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    median = times[len(times) // 2] if times else float("nan")
    # median_ms: consumed by forge-rewrite's oracle; wall_ms samples + case_ms:
    # consumed by the forge-loop OPTIMIZE benchmark.
    print(f"median_ms: {median:.6f}")
    print(f"case_ms: {_case_id(rows, cols, dtype)} {median:.6f}")
    return 0


def _run_ref_bench(rows: int, cols: int, dtype: str, warmup: int, iters: int, device: str) -> int:
    """Time the SOURCE Triton kernel — the speedup baseline (eager event timing)."""
    x = _make_input(rows, cols, dtype, "full", device)

    def step():
        _source_softmax(x)

    for _ in range(max(1, warmup)):
        step()
    torch.cuda.synchronize()

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(max(1, iters)):
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = [t for t in times if t > 0]
    median = statistics.median(times) if times else float("nan")
    print(f"median_ms: {median:.6f}")
    print(f"case_ms: {_case_id(rows, cols, dtype)} {median:.6f}")
    return 0


def _run_profile(rows: int, cols: int, dtype: str, device: str) -> int:
    """Warm the FlyDSL candidate, then expose only its dispatches to the profiler."""
    x = _make_input(rows, cols, dtype, "full", device)
    out = torch.empty_like(x)
    launch_fn = _build(rows, cols, dtype)
    for _ in range(3):
        _launch_on_current_stream(launch_fn, x, out, rows)
    torch.cuda.synchronize()
    for _ in range(3):
        _launch_on_current_stream(launch_fn, x, out, rows)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="triton2flydsl softmax rewrite driver")
    parser.add_argument("--bench-mode", action="store_true", help="time the FlyDSL candidate")
    parser.add_argument("--ref-bench-mode", action="store_true", help="time the source Triton kernel")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    # Ignore any extra flags forge may append that this driver does not use.
    args, _unknown = parser.parse_known_args()

    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1

    device = "cuda"
    if args.profile_run:
        return _run_profile(_DEFAULT_M, _DEFAULT_N, _DEFAULT_DTYPE, device)

    if args.ref_bench_mode:
        return _run_ref_bench(
            _DEFAULT_M,
            _DEFAULT_N,
            _DEFAULT_DTYPE,
            args.warmup,
            args.iters,
            device,
        )
    if args.bench_mode:
        return _run_bench(
            _DEFAULT_M,
            _DEFAULT_N,
            _DEFAULT_DTYPE,
            args.warmup,
            args.iters,
            device,
        )
    return _run_correctness(
        _DEFAULT_M,
        _DEFAULT_N,
        _DEFAULT_DTYPE,
        "full",
        device,
    )


if __name__ == "__main__":
    sys.exit(main())
