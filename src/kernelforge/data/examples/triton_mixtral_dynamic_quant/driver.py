"""Measurement driver for the Mixtral dynamic FP8 quantization task.

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
its stable public name ``dynamic_quant_fp8`` from ``quant_kernel.py``.

Correctness is scored on the DEQUANTIZED output (``fp8 * scale``) against a pure
Torch per-tensor quantization oracle, so the metric measures how faithfully the
kernel reproduces the reference rather than the ~28 dB physical noise floor of
fp8-e4m3 itself.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from graph_harness import cuda_graph_bench
from quant_kernel import FP8_DTYPE, FP8_MAX, dynamic_quant_fp8

# Driver-owned scored case: the real Mixtral-8x7B activation shape.
_DEFAULT_M = 64
_DEFAULT_N = 4096

# Fixed seed so every full-suite invocation builds identical inputs.
_SEED = 1


def _case_id(rows: int, cols: int) -> str:
    """Return the opaque token emitted by benchmark mode."""
    return f"M{rows}_N{cols}"


def _make_input(rows: int, cols: int, mode: str, device: str) -> torch.Tensor:
    """Build the activation tensor for a given validation mode."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    if mode == "stability":
        # A wide dynamic range stresses the amax reduction: a kernel that reduces
        # per-block without a final global pass picks the wrong scale here.
        x = x * 200.0
        x[0, 0] = 60000.0
    return x


def _reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch per-tensor dynamic fp8 quantization (the oracle)."""
    amax = x.float().abs().amax()
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / FP8_MAX)
    y = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return y, scale.reshape(1)


def _dequantize(y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return y.float() * scale


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


def _allocate_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    return out, scale


def _run_correctness(rows: int, cols: int, mode: str, device: str) -> int:
    x = _make_input(rows, cols, mode, device)
    out, scale = _allocate_outputs(x)
    dynamic_quant_fp8(x, out, scale)
    torch.cuda.synchronize()

    ref_y, ref_scale = _reference(x)
    ref_deq = _dequantize(ref_y, ref_scale)
    got_deq = _dequantize(out, scale)

    print(f"SNR: {_snr_db(ref_deq, got_deq):.2f} dB")
    print(f"allclose: {torch.allclose(got_deq, ref_deq, atol=1e-2, rtol=1e-2)}")
    _assert_no_aiter()
    return 0


def _run_bench(rows: int, cols: int, warmup: int, iters: int, device: str) -> int:
    # Static tensors allocated once; the graph harness replays the op on the same
    # memory so it times GPU execution, not host launch overhead.
    x = _make_input(rows, cols, "full", device)
    out, scale = _allocate_outputs(x)
    ref_y, ref_scale = _reference(x)
    ref_deq = _dequantize(ref_y, ref_scale)

    def step() -> None:
        dynamic_quant_fp8(x, out, scale)

    # dirty + verify prove the graph actually captured the kernel (an uncaptured
    # launch would leave the outputs at their dirtied values and fail verify).
    def dirty() -> None:
        out.zero_()
        scale.zero_()

    def verify() -> bool:
        return torch.allclose(_dequantize(out, scale), ref_deq, atol=1e-2, rtol=1e-2)

    result = cuda_graph_bench(step, warmup=warmup, iters=iters, dirty=dirty, verify=verify)

    # Informational only (does not match forge's wall_ms/median_ms parser).
    print(f"# bench mode: {result['mode']}")
    for t in result["times_ms"]:
        print(f"wall_ms: {t:.6f}")
    times = sorted(result["times_ms"])
    print(f"case_ms: {_case_id(rows, cols)} {times[len(times) // 2]:.6f}")
    _assert_no_aiter()
    return 0


def _run_profile(rows: int, cols: int, device: str) -> int:
    """Warm the target, then expose only its dispatches to the profiler."""
    x = _make_input(rows, cols, "full", device)
    out, scale = _allocate_outputs(x)
    for _ in range(3):
        dynamic_quant_fp8(x, out, scale)
    torch.cuda.synchronize()
    for _ in range(3):
        dynamic_quant_fp8(x, out, scale)
    torch.cuda.synchronize()
    return 0


def _assert_no_aiter() -> None:
    """This task must be self-contained: no AITER runtime anywhere."""
    loaded = [n for n in list(sys.modules) if n == "aiter" or n.startswith("aiter.")]
    assert not loaded, f"AITER was imported ({loaded}); this task must stay standalone"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixtral dynamic FP8 quant task driver")
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
