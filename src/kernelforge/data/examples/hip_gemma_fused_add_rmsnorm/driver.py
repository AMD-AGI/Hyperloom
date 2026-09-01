"""Measurement driver for the fused residual-add + Gemma RMSNorm task (HIP).

forge-loop treats the driver as a black box invoked as ``python driver.py <args>``
and communicates with it purely through stdout. This driver implements the two
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
its stable public name ``fused_add_rmsnorm`` from ``fused_add_rmsnorm_kernel.py``.

The op has TWO outputs (the normalized activations and the summed residual). Both
are scored, and the reported SNR is the WORSE of the two, so a kernel cannot pass
by getting only one of them right.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from fused_add_rmsnorm_kernel import EPS, fused_add_rmsnorm
from graph_harness import cuda_graph_bench

# Driver-owned scored case using the Gemma-4-26B-A4B-it hidden size.
_DEFAULT_M = 64
_DEFAULT_N = 2816

# Fixed seed so every full-suite invocation builds identical inputs.
_SEED = 3


def _case_id(rows: int, hidden: int) -> str:
    """Return the opaque token emitted by benchmark mode."""
    return f"M{rows}_N{hidden}"


def _make_inputs(
    rows: int, hidden: int, mode: str, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (x, residual, weight) for a given validation mode."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    residual = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    if mode == "stability":
        # Large magnitudes overflow a kernel that squares in bf16 instead of
        # accumulating the mean-of-squares in fp32.
        x = x * 240.0
        residual = residual * 240.0
    return x, residual, weight


def _reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch oracle: residual add, then fp32-reduced Gemma RMSNorm."""
    summed = x + residual
    sf = summed.float()
    inv_rms = torch.rsqrt(sf.square().mean(-1, keepdim=True) + EPS)
    out = (sf * inv_rms * (1.0 + weight.float())).to(x.dtype)
    return out, summed


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


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)


def _run_correctness(rows: int, hidden: int, mode: str, device: str) -> int:
    x, residual, weight = _make_inputs(rows, hidden, mode, device)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    fused_add_rmsnorm(x, residual, weight, out, residual_out)
    torch.cuda.synchronize()

    ref_out, ref_residual = _reference(x, residual, weight)

    # Report the WORSE of the two outputs so one correct tensor cannot mask a
    # broken one.
    snr = min(_snr_db(ref_out, out), _snr_db(ref_residual, residual_out))
    ok = _close(out, ref_out) and _close(residual_out, ref_residual)
    print(f"SNR: {snr:.2f} dB")
    print(f"allclose: {ok}")
    _assert_no_aiter()
    return 0


def _run_bench(rows: int, hidden: int, warmup: int, iters: int, device: str) -> int:
    # Static tensors allocated once; the graph harness replays the op on the same
    # memory so it times GPU execution, not host launch overhead. The kernel never
    # writes to its inputs, so every replay recomputes the same result.
    x, residual, weight = _make_inputs(rows, hidden, "full", device)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    ref_out, ref_residual = _reference(x, residual, weight)

    def step() -> None:
        fused_add_rmsnorm(x, residual, weight, out, residual_out)

    # dirty + verify prove the graph actually captured the kernel (an uncaptured
    # launch would leave the outputs at their dirtied values and fail verify).
    def dirty() -> None:
        out.zero_()
        residual_out.zero_()

    def verify() -> bool:
        return _close(out, ref_out) and _close(residual_out, ref_residual)

    result = cuda_graph_bench(step, warmup=warmup, iters=iters, dirty=dirty, verify=verify)

    # Informational only (does not match forge's wall_ms/median_ms parser).
    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    print(f"case_ms: {_case_id(rows, hidden)} {times[len(times) // 2]:.6f}")
    _assert_no_aiter()
    return 0


def _run_profile(rows: int, hidden: int, device: str) -> int:
    """Warm the target, then expose only its dispatches to the profiler."""
    x, residual, weight = _make_inputs(rows, hidden, "full", device)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    for _ in range(3):
        fused_add_rmsnorm(x, residual, weight, out, residual_out)
    torch.cuda.synchronize()
    for _ in range(3):
        fused_add_rmsnorm(x, residual, weight, out, residual_out)
    torch.cuda.synchronize()
    return 0


def _assert_no_aiter() -> None:
    """This task must be self-contained: no AITER runtime anywhere."""
    loaded = [n for n in list(sys.modules) if n == "aiter" or n.startswith("aiter.")]
    assert not loaded, f"AITER was imported ({loaded}); this task must stay standalone"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fused add + Gemma RMSNorm (HIP) task driver")
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
        return _run_bench(_DEFAULT_M, _DEFAULT_N, args.warmup, args.iters, device)
    return _run_correctness(_DEFAULT_M, _DEFAULT_N, "full", device)


if __name__ == "__main__":
    sys.exit(main())
