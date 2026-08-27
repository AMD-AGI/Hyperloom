"""Measurement driver for the Gemma RMSNorm task (FlyDSL).

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
its stable public name ``gemma_rmsnorm`` from ``rmsnorm_kernel.py``.

Stream routing lives HERE, not in the kernel: this driver always passes the handle
of the CURRENTLY ACTIVE stream, queried at call time. Under the CUDA-graph harness
that stream is the private capture stream, so a FlyDSL kernel that honors the
handle gets recorded into the graph. Keeping the decision in the protected driver
means the agent cannot break graph capture by editing the kernel.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from graph_harness import cuda_graph_bench
from rmsnorm_kernel import EPS, gemma_rmsnorm

# Driver-owned scored case using the Gemma-4-26B-A4B-it hidden size.
_DEFAULT_M = 64
_DEFAULT_N = 2816

# Fixed seed so every full-suite invocation builds identical inputs.
_SEED = 2


def _case_id(rows: int, hidden: int) -> str:
    """Return the opaque token emitted by benchmark mode."""
    return f"M{rows}_N{hidden}"


def _make_inputs(
    rows: int, hidden: int, mode: str, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, weight) for a given validation mode."""
    torch.manual_seed(_SEED)
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    if mode == "stability":
        # Large magnitudes overflow a kernel that squares in bf16 instead of
        # accumulating the mean-of-squares in fp32.
        x = x * 240.0
    return x, weight


def _launch(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Run the kernel on whatever stream is currently active.

    The handle is queried at call time on purpose: under ``torch.cuda.graph`` the
    active stream is the private capture stream, so the launch gets recorded.
    """
    gemma_rmsnorm(x, weight, out, stream_handle=torch.cuda.current_stream().cuda_stream)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Torch Gemma RMSNorm oracle: fp32 reduction, (1 + weight) scale."""
    xf = x.float()
    inv_rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    return (xf * inv_rms * (1.0 + weight.float())).to(x.dtype)


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


def _run_correctness(rows: int, hidden: int, mode: str, device: str) -> int:
    x, weight = _make_inputs(rows, hidden, mode, device)
    out = torch.empty_like(x)
    _launch(x, weight, out)
    torch.cuda.synchronize()

    ref = _reference(x, weight)
    print(f"SNR: {_snr_db(ref, out):.2f} dB")
    print(f"allclose: {torch.allclose(out, ref, atol=1e-2, rtol=1e-2)}")
    _assert_no_aiter()
    return 0


def _run_bench(rows: int, hidden: int, warmup: int, iters: int, device: str) -> int:
    # Static tensors allocated once; the graph harness replays the op on the same
    # memory so it times GPU execution, not host launch overhead.
    x, weight = _make_inputs(rows, hidden, "full", device)
    out = torch.empty_like(x)
    ref = _reference(x, weight)

    def step() -> None:
        _launch(x, weight, out)

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
    print(f"case_ms: {_case_id(rows, hidden)} {times[len(times) // 2]:.6f}")
    _assert_no_aiter()
    return 0


def _run_profile(rows: int, hidden: int, device: str) -> int:
    """Warm the target, then expose only its dispatches to the profiler."""
    x, weight = _make_inputs(rows, hidden, "full", device)
    out = torch.empty_like(x)
    for _ in range(3):
        _launch(x, weight, out)
    torch.cuda.synchronize()
    for _ in range(3):
        _launch(x, weight, out)
    torch.cuda.synchronize()
    return 0


def _assert_no_aiter() -> None:
    """This task must be self-contained: no AITER runtime anywhere."""
    loaded = [n for n in list(sys.modules) if n == "aiter" or n.startswith("aiter.")]
    assert not loaded, f"AITER was imported ({loaded}); this task must stay standalone"


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma RMSNorm (FlyDSL) task driver")
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
