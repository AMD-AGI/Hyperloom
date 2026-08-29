"""Measurement driver for the forge-loop Gluon softmax example.

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
its stable public name ``softmax`` from ``softmax_kernel.py``.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from graph_harness import cuda_graph_bench
from softmax_kernel import softmax

# Driver-owned scored cases. Rows x cols of each 2D softmax input.
_CASES = (
    (1024, 256),
    (4096, 1024),
)
_DEFAULT_M, _DEFAULT_N = _CASES[-1]

# Fixed seed so every full-suite invocation builds identical inputs.
_SEED = 0


def _case_id(rows: int, cols: int) -> str:
    """Return the opaque token emitted by benchmark mode."""
    return f"M{rows}_N{cols}"


def _make_input(rows: int, cols: int, mode: str, device: str) -> torch.Tensor:
    """Build the softmax input for a given validation mode."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, cols, device=device, dtype=torch.float16)
    if mode == "stability":
        # Large magnitudes stress the max-subtraction; a kernel that skips it
        # overflows exp() and fails here.
        x = x * 50.0
    return x


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


def _run_correctness(rows: int, cols: int, mode: str, device: str) -> int:
    x = _make_input(rows, cols, mode, device)
    out = softmax(x)
    ref = torch.softmax(x, dim=-1)
    print(f"SNR: {_snr_db(ref, out):.2f} dB")
    print(f"allclose: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    return 0


def _run_correctness_suite(device: str) -> int:
    snr_values = []
    allclose_values = []
    for rows, cols in _CASES:
        x = _make_input(rows, cols, "full", device)
        out = softmax(x)
        ref = torch.softmax(x, dim=-1)
        snr = _snr_db(ref, out)
        passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        snr_values.append(snr)
        allclose_values.append(bool(passed))
        print(f"case_snr: {_case_id(rows, cols)} {snr:.2f}")
        print(f"case_allclose: {_case_id(rows, cols)} {passed}")
    print(f"SNR: {min(snr_values):.2f} dB")
    print(f"allclose: {all(allclose_values)}")
    return 0


def _run_bench(rows: int, cols: int, warmup: int, iters: int, device: str) -> int:
    # Static input allocated once; the graph harness replays the op on the same
    # memory so it times GPU execution, not host launch overhead.
    x = _make_input(rows, cols, "full", device)
    ref = torch.softmax(x, dim=-1)

    # softmax(x) allocates and returns its own output, so there is no external
    # buffer to hand the harness. Capture the returned tensor instead: under graph
    # capture it is a fixed graph-pool buffer that every replay recomputes into, so
    # zeroing it (dirty) and checking it (verify) proves the graph actually did the
    # work — rejecting a silently empty / uncaptured graph rather than reporting a
    # fake speedup. Storing into the dict is a trivial host op, so timing is still
    # just the softmax (no extra copy).
    captured: dict = {}

    def step() -> None:
        captured["out"] = softmax(x)

    result = cuda_graph_bench(
        step,
        warmup=warmup,
        iters=iters,
        dirty=lambda: captured["out"].zero_(),
        verify=lambda: torch.allclose(captured["out"], ref, atol=1e-2, rtol=1e-2),
    )

    # Informational only (does not match forge's wall_ms/median_ms parser).
    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    print(f"case_ms: {_case_id(rows, cols)} {times[len(times) // 2]:.6f}")
    return 0


def _run_profile(rows: int, cols: int, device: str) -> int:
    """Warm the target, then expose only its dispatches to the profiler."""
    x = _make_input(rows, cols, "full", device)
    for _ in range(3):
        softmax(x)
    torch.cuda.synchronize()
    for _ in range(3):
        softmax(x)
    torch.cuda.synchronize()
    return 0


def _run_bench_suite(warmup: int, iters: int, device: str) -> int:
    for rows, cols in _CASES:
        result = _run_bench(rows, cols, warmup, iters, device)
        if result != 0:
            return result
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="forge-loop Gluon softmax example driver")
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
        return _run_profile(_DEFAULT_M, _DEFAULT_N, device)

    if args.bench_mode:
        return _run_bench_suite(args.warmup, args.iters, device)
    return _run_correctness_suite(device)


if __name__ == "__main__":
    sys.exit(main())
