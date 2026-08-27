"""Measurement driver for the forge-loop FlyDSL softmax example.

forge-loop treats the driver as a black box invoked as ``python driver.py <args>``
and communicates with it purely through stdout. This driver implements the three
modes of that contract:

  * Correctness  ``python driver.py`` -> runs the complete suite and prints
    ``SNR: <db> dB`` (and ``allclose: True/False``).
    forge invokes this once as the driver-owned complete correctness suite.

  * Benchmark    ``python driver.py --warmup <n> --iters <n>
    --bench-mode`` -> prints ``wall_ms`` samples plus one ``case_ms`` aggregate.
    forge takes the median of those samples as the kernel's wall time.

  * Profiling    ``python driver.py --profile-run`` -> the driver selects the
    profile case, runs only the target kernel, and exits without reference/timing.

The driver is the correctness ORACLE and the perf MEASURER; forge never edits it
(it is a protected measurement file). It imports the kernel under optimization by
its stable public builder ``build_softmax_module`` from ``softmax_kernel.py``.

Stream routing lives HERE, not in the kernel: the FlyDSL launcher takes a
``stream`` kwarg, and this driver always passes the CURRENT stream. Under the
CUDA-graph harness the current stream IS the capture stream, so the kernel is
recorded into the graph. Keeping this in the (protected) driver means the agent
cannot accidentally break graph capture by editing the kernel.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import flydsl.expr as fx

from graph_harness import cuda_graph_bench
from softmax_kernel import build_softmax_module

# Driver-owned scored case. Rows x cols of the 2D softmax input.
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
    key = (rows, cols, dtype)
    if key not in _MODULE_CACHE:
        _MODULE_CACHE[key] = build_softmax_module(rows, cols, dtype)
    return _MODULE_CACHE[key]


def _make_input(rows: int, cols: int, dtype: str, device: str) -> torch.Tensor:
    """Build a standard random softmax input tensor."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, cols, device=device, dtype=_TORCH_DTYPE[dtype])
    return x


def _launch_on_current_stream(launch_fn, x: torch.Tensor, out: torch.Tensor, rows: int) -> None:
    """Run the FlyDSL kernel on whatever stream is currently active.

    Queried at call time on purpose: under torch.cuda.graph the active stream is
    the private capture stream, so the launch gets recorded into the graph.
    """
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    launch_fn(x, out, rows, stream=stream)


def _reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


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


def _run_correctness(rows: int, cols: int, dtype: str, device: str) -> int:
    x = _make_input(rows, cols, dtype, device)
    out = torch.empty_like(x)
    launch_fn = _build(rows, cols, dtype)
    _launch_on_current_stream(launch_fn, x, out, rows)
    torch.cuda.synchronize()

    ref = _reference(x)
    print(f"SNR: {_snr_db(ref, out):.2f} dB")
    print(f"allclose: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return 0


def _run_bench(rows: int, cols: int, dtype: str, warmup: int, iters: int, device: str) -> int:
    # Static tensors allocated once; the graph harness replays the op on the same
    # memory so it times GPU execution, not host launch overhead.
    x = _make_input(rows, cols, dtype, device)
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

    # Informational only (does not match forge's wall_ms/median_ms parser).
    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    print(
        f"case_ms: {_case_id(rows, cols, dtype)} "
        f"{times[len(times) // 2]:.6f}"
    )
    return 0


def _run_profile(
    rows: int,
    cols: int,
    dtype: str,
    device: str,
) -> int:
    """Warm the target, then expose only its dispatches to the profiler."""
    x = _make_input(rows, cols, dtype, device)
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
    parser = argparse.ArgumentParser(description="forge-loop FlyDSL softmax example driver")
    parser.add_argument("--bench-mode", action="store_true", help="run the wall-clock benchmark")
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
        device,
    )


if __name__ == "__main__":
    sys.exit(main())
