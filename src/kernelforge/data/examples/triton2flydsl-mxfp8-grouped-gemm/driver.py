"""Measurement driver for rewriting SGLang MXFP8 grouped GEMM to FlyDSL.

The driver owns the complete workload and is protected during rewrite. It times
the two grouped-GEMM calls from one MiniMax-M3 MoE forward:

* GEMM1: shared token activations times gate/up weights, BF16 output.
* GEMM2: routed activations times down weights, FP32 output with top-k weights.

The generated ``kernel.py`` must expose this exact interface::

    build_mxfp8_grouped_gemm_module(
        experts, n_cols, k_cols, num_valid_tokens, num_sorted_tokens,
        top_k, block_m, out_dtype, a_div, mul_weight
    ) -> launch_fn

    launch_fn(
        a_q, a_scale, w, w_scale, out, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        stream=fx.Stream(...)
    )

Inputs use OCP MXFP8 E4M3 values with uint8 E8M0 scales per 1x32 block.
The launch must write ``out`` in place and must use the supplied stream.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

from graph_harness import cuda_graph_bench
from mxfp8_grouped_gemm import _grouped_gemm_mxfp8 as _source_grouped_gemm


WORKSPACE = Path(__file__).resolve().parent
CASES = json.loads((WORKSPACE / "session_cases.json").read_text())["cases"]
BLOCK_M = 64
CORRECTNESS_MAX_TOKENS = 64
_MODULE_CACHE: dict[tuple, object] = {}


def _configure_sglang() -> None:
    """Make the SGLang helpers used to construct the real workload importable."""
    try:
        import sglang  # noqa: F401

        return
    except ImportError:
        pass
    sglang_python = Path(
        os.environ.get("SGLANG_PYTHON", "/sgl-workspace/sglang/python")
    )
    if not (sglang_python / "sglang").is_dir():
        raise RuntimeError(
            "SGLang is not importable; set SGLANG_PYTHON to its python directory"
        )
    sys.path.insert(0, str(sglang_python))


def _candidate_builder(stage: dict):
    key = (
        stage["experts"],
        stage["n_cols"],
        stage["k_cols"],
        stage["num_valid_tokens"],
        stage["num_sorted_tokens"],
        stage["top_k"],
        stage["block_m"],
        stage["out_dtype_name"],
        stage["a_div"],
        stage["mul_weight"],
    )
    if key not in _MODULE_CACHE:
        from kernel import build_mxfp8_grouped_gemm_module

        _MODULE_CACHE[key] = build_mxfp8_grouped_gemm_module(*key)
    return _MODULE_CACHE[key]


def _run_candidate_stage(stage: dict) -> torch.Tensor:
    import flydsl.expr as fx

    launch = _candidate_builder(stage)
    launch(
        stage["a_q"],
        stage["a_scale"],
        stage["w"],
        stage["w_scale"],
        stage["candidate_out"],
        stage["topk_weights"],
        stage["sorted_token_ids"],
        stage["expert_ids"],
        stage["num_tokens_post_padded"],
        stream=fx.Stream(torch.cuda.current_stream().cuda_stream),
    )
    return stage["candidate_out"]


def _run_source_stage(stage: dict) -> torch.Tensor:
    return _source_grouped_gemm(
        stage["a_q"],
        stage["a_scale"],
        stage["w"],
        stage["w_scale"],
        stage["sorted_token_ids"],
        stage["expert_ids"],
        stage["num_tokens_post_padded"],
        stage["num_valid_tokens"],
        stage["top_k"],
        stage["block_m"],
        stage["out_dtype"],
        stage["a_div"],
        stage["topk_weights"] if stage["mul_weight"] else None,
    )


def _make_stage(
    *,
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k: int,
    out_dtype: torch.dtype,
    a_div: int,
    topk_weights: torch.Tensor | None = None,
) -> dict:
    experts, n_cols, k_cols = w.shape
    return {
        "a_q": a_q,
        "a_scale": a_scale,
        "w": w,
        "w_scale": w_scale,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "num_valid_tokens": num_valid_tokens,
        "num_sorted_tokens": sorted_token_ids.numel(),
        "top_k": top_k,
        "block_m": BLOCK_M,
        "out_dtype": out_dtype,
        "out_dtype_name": "f32" if out_dtype == torch.float32 else "bf16",
        "a_div": a_div,
        "mul_weight": topk_weights is not None,
        "topk_weights": topk_weights if topk_weights is not None else a_q,
        "experts": experts,
        "n_cols": n_cols,
        "k_cols": k_cols,
        "candidate_out": torch.empty(
            (num_valid_tokens, n_cols),
            dtype=out_dtype,
            device=a_q.device,
        ),
    }


def _make_case(case: dict, *, correctness: bool) -> tuple[dict, dict]:
    from sglang.kernels.ops.moe.minimax_m3_swiglu import swiglu_oai_split
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        _mxfp8_e4m3_quantize_torch,
        mxfp8_e4m3_quantize,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    params = case["params"]
    tokens = (
        min(params["tokens"], CORRECTNESS_MAX_TOKENS)
        if correctness
        else params["tokens"]
    )
    hidden_size = params["hidden"]
    inter_size = params["inter"]
    experts = params["experts"]
    top_k = params["top_k"]
    torch.manual_seed(case.get("seed", 0))

    hidden = torch.randn(
        tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.5
    w13_bf16 = torch.randn(
        experts,
        2 * inter_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(w13_bf16)
    del w13_bf16
    w2_bf16 = torch.randn(
        experts,
        hidden_size,
        inter_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16)
    del w2_bf16

    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    routed_tokens = tokens * top_k
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, BLOCK_M, experts
    )
    a_q, a_scale = mxfp8_e4m3_quantize(hidden)

    gemm1 = _make_stage(
        a_q=a_q,
        a_scale=a_scale,
        w=w13_fp8,
        w_scale=w13_scale,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post,
        num_valid_tokens=routed_tokens,
        top_k=top_k,
        out_dtype=torch.bfloat16,
        a_div=top_k,
    )
    source_gemm1 = _run_source_stage(gemm1)
    activation = swiglu_oai_split(
        source_gemm1,
        alpha=params["alpha"],
        beta=params["beta"],
        limit=params["limit"],
        out_dtype=torch.bfloat16,
    )
    act_q, act_scale = mxfp8_e4m3_quantize(activation)
    gemm2 = _make_stage(
        a_q=act_q,
        a_scale=act_scale,
        w=w2_fp8,
        w_scale=w2_scale,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_post,
        num_valid_tokens=routed_tokens,
        top_k=top_k,
        out_dtype=torch.float32,
        a_div=1,
        topk_weights=topk_weights.reshape(-1),
    )
    return gemm1, gemm2


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    return float(
        ((actual_f32 - expected_f32).norm() / (expected_f32.norm() + 1e-8)).item()
    )


def _outputs_match(
    outputs: tuple[torch.Tensor, torch.Tensor],
    references: tuple[torch.Tensor, torch.Tensor],
    tolerance: float,
) -> bool:
    errors = [
        _relative_error(actual, expected)
        for actual, expected in zip(outputs, references)
    ]
    return all(math.isfinite(error) and error < tolerance for error in errors)


def _run_correctness() -> int:
    all_ok = True
    for case in CASES:
        stages = _make_case(case, correctness=True)
        references = tuple(_run_source_stage(stage) for stage in stages)
        for stage in stages:
            stage["candidate_out"].fill_(float("nan"))
        outputs = tuple(_run_candidate_stage(stage) for stage in stages)
        torch.cuda.synchronize()
        errors = [
            _relative_error(actual, expected)
            for actual, expected in zip(outputs, references)
        ]
        tolerance = float(case["params"].get("max_relerr", 0.08))
        ok = all(math.isfinite(error) and error < tolerance for error in errors)
        all_ok = all_ok and ok
        print(
            f"# case {case['id']}: gemm1_relerr={errors[0]:.6f} "
            f"gemm2_relerr={errors[1]:.6f} tol={tolerance} ok={ok}"
        )
    print(f"allclose: {all_ok}")
    return 0 if all_ok else 1


def _bench_candidate(case: dict, warmup: int, iters: int) -> tuple[float, str]:
    stages = _make_case(case, correctness=False)
    references = tuple(_run_source_stage(stage) for stage in stages)
    tolerance = float(case["params"].get("max_relerr", 0.08))

    def step() -> None:
        for stage in stages:
            _run_candidate_stage(stage)

    def dirty() -> None:
        for stage in stages:
            stage["candidate_out"].fill_(float("nan"))

    def verify() -> bool:
        outputs = tuple(stage["candidate_out"] for stage in stages)
        return _outputs_match(outputs, references, tolerance)

    result = cuda_graph_bench(
        step,
        warmup=warmup,
        iters=iters,
        dirty=dirty,
        verify=verify,
    )
    return float(result["median_ms"]), str(result["mode"])


def _bench_source(case: dict, warmup: int, iters: int) -> tuple[float, str]:
    stages = _make_case(case, correctness=False)
    holder: list[torch.Tensor | None] = [None, None]

    def step() -> None:
        holder[0] = _run_source_stage(stages[0])
        holder[1] = _run_source_stage(stages[1])

    result = cuda_graph_bench(step, warmup=warmup, iters=iters)
    return float(result["median_ms"]), str(result["mode"])


def _run_benchmark(*, source: bool, warmup: int, iters: int) -> int:
    measurements: list[float] = []
    bench = _bench_source if source else _bench_candidate
    for case in CASES:
        elapsed_ms, mode = bench(case, warmup, iters)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise RuntimeError(f"invalid timing for {case['id']}: {elapsed_ms}")
        measurements.append(elapsed_ms)
        print(f"case_ms: {case['id']} {elapsed_ms:.6f}")
        print(f"# bench {case['id']}: mode={mode}")
    print(f"mean_ms: {statistics.mean(measurements):.6f}")
    return 0


def _run_profile() -> int:
    case = max(CASES, key=lambda item: int(item["params"]["tokens"]))
    stages = _make_case(case, correctness=False)
    for _ in range(3):
        for stage in stages:
            _run_candidate_stage(stage)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--ref-bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _unknown = parser.parse_known_args()

    _configure_sglang()
    if not torch.cuda.is_available():
        print("error: MI355X/gfx950 GPU is required", file=sys.stderr)
        return 1
    if args.profile_run:
        return _run_profile()
    if args.ref_bench_mode:
        return _run_benchmark(
            source=True,
            warmup=args.warmup,
            iters=args.iters,
        )
    if args.bench_mode:
        return _run_benchmark(
            source=False,
            warmup=args.warmup,
            iters=args.iters,
        )
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
