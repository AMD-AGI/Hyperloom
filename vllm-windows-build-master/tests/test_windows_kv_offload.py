"""Native-Windows real-model validation for vLLM CPU KV offloading.

This is an opt-in integration test. It loads a real local Hugging Face model,
stores a completed prompt's KV blocks in the SimpleCPUOffloadConnector, clears
the GPU prefix cache, and verifies that the next request restores the prefix
without changing generated tokens.

Run baseline and offload modes in separate processes so CUDA and vLLM engine
state cannot leak between comparisons.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path


RESULT_MARKER = "===KV_OFFLOAD_RESULT==="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_PATH"),
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "offload"),
        required=True,
    )
    parser.add_argument("--cpu-cache-gib", type=float, default=1.0)
    parser.add_argument("--gpu-cache-gib", type=float, default=3.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-repeats", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--restore-cycles", type=int, default=1)
    return parser.parse_args()


def build_prompt(repeats: int) -> str:
    prefix = (
        "Native Windows KV cache validation records a stable reusable "
        "document prefix. "
    ) * repeats
    return (
        prefix
        + "\nQuestion: What does this document say is being validated?\nAnswer:"
    )


def serialize_output(output, elapsed: float) -> dict:
    metrics = output.metrics
    return {
        "elapsed_seconds": elapsed,
        "first_token_latency_seconds": (
            metrics.first_token_latency if metrics is not None else None
        ),
        "prompt_tokens": len(output.prompt_token_ids or []),
        "num_cached_tokens": output.num_cached_tokens,
        "token_ids": list(output.outputs[0].token_ids),
        "text": output.outputs[0].text,
    }


def main() -> None:
    args = parse_args()
    if not args.model:
        raise SystemExit("Set VLLM_MODEL_PATH or pass --model.")
    if args.restore_cycles < 1:
        raise SystemExit("--restore-cycles must be at least 1.")

    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Not a Hugging Face model directory: {model_path}")

    # Import only after argument validation so configuration errors are fast.
    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    if args.mode == "offload":
        from vllm.v1.simple_kv_offload import cuda_mem_ops

        if os.name == "nt" and not hasattr(cuda_mem_ops, "_copy_blocks_windows"):
            raise SystemExit(
                "The installed wheel lacks the safe Windows KV-offload DMA "
                "fallback; install the current v0.27.1 Windows wheel."
            )
        try:
            cuda_python_version = importlib.metadata.version("cuda-python")
        except importlib.metadata.PackageNotFoundError as e:
            raise SystemExit("Install cuda-python==12.8.0 for offload mode.") from e
        if cuda_python_version != "12.8.0":
            raise SystemExit(
                f"cuda-python {cuda_python_version} is installed; expected 12.8.0."
            )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Visible GPU: {gpu_name}", flush=True)

    llm_kwargs = {
        "model": str(model_path),
        "dtype": "float16",
        "kv_cache_dtype": "auto",
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": 0.85,
        "kv_cache_memory_bytes": int(args.gpu_cache_gib * (1 << 30)),
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }

    if args.mode == "offload":
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": int(args.cpu_cache_gib * (1 << 30)),
                "lazy_offload": False,
            },
        )

    llm = LLM(**llm_kwargs)
    try:
        prompt = build_prompt(args.prompt_repeats)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=0,
        )

        start = time.perf_counter()
        cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        cold_elapsed = time.perf_counter() - start

        cold_result = serialize_output(cold, cold_elapsed)
        restore_results = []
        for cycle in range(args.restore_cycles):
            # Eager offload is asynchronous. Allow its completion to reach the
            # scheduler, then clear only the GPU prefix cache. The upstream
            # vLLM integration test uses the same sequence to force a CPU
            # cache lookup.
            time.sleep(2.0)
            reset_ok = llm.reset_prefix_cache()
            if not reset_ok:
                raise AssertionError(
                    f"GPU prefix-cache reset was rejected in cycle {cycle}"
                )

            start = time.perf_counter()
            restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            restored_elapsed = time.perf_counter() - start
            restored_result = serialize_output(restored, restored_elapsed)
            restore_results.append(restored_result)

            if cold_result["token_ids"] != restored_result["token_ids"]:
                raise AssertionError(
                    "Generated tokens changed after cache reset/restore: "
                    f"cold={cold_result['token_ids']} "
                    f"cycle_{cycle}={restored_result['token_ids']}"
                )

            if args.mode == "offload" and not restored_result["num_cached_tokens"]:
                raise AssertionError(
                    f"Restore cycle {cycle} reported zero cached tokens; "
                    "CPU restore was not demonstrated"
                )
            if args.mode == "baseline" and restored_result["num_cached_tokens"]:
                raise AssertionError(
                    "Baseline retained cached tokens after GPU prefix-cache reset"
                )

        result = {
            "mode": args.mode,
            "model": str(model_path),
            "gpu": gpu_name,
            "cpu_cache_gib": args.cpu_cache_gib if args.mode == "offload" else 0,
            "gpu_cache_gib": args.gpu_cache_gib,
            "reset_ok": True,
            "cold": cold_result,
            "restores": restore_results,
            "tokens_match": True,
        }
        print(RESULT_MARKER, flush=True)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        del llm
        gc.collect()


if __name__ == "__main__":
    main()
