"""Real-model validation for the vLLM 0.27.1 Windows KV offloading release.

Run one mode per process so CUDA, model, and cache state cannot leak between
comparisons. The caller should expose only the intended GPU.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

RESULT_MARKER = "===VLLM_WINDOWS_KV_RESULT==="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=("baseline", "cpu-lru", "cpu-arc", "fs-lru", "fs-arc"),
        required=True,
    )
    parser.add_argument("--fs-root")
    parser.add_argument("--result-json")
    parser.add_argument("--reference-json")
    parser.add_argument("--reuse-existing-cache", action="store_true")
    parser.add_argument("--expected-version", default="0.27.1")
    parser.add_argument("--cpu-cache-mib", type=int, default=512)
    parser.add_argument("--gpu-cache-mib", type=int, default=1536)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--prompt-repeats", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--restore-cycles", type=int, default=1)
    parser.add_argument("--fs-eviction-prompts", type=int, default=2)
    return parser.parse_args()


def build_prompt(repeats: int) -> str:
    prefix = (
        "Native Windows vLLM cache validation uses a stable document prefix "
        "that should be reusable after the GPU prefix cache is cleared. "
    ) * repeats
    return prefix + "\nQuestion: What cache is being validated?\nAnswer:"


def build_eviction_prompt(prompt: str, cycle: int, index: int) -> str:
    """Build a same-sized, non-overlapping prefix to evict the target from RAM."""
    return f"Filesystem eviction namespace {cycle}:{index}.\n" + prompt


def serialize_output(output, elapsed: float) -> dict[str, object]:
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


def wait_for_prefix_cache_reset(llm, timeout: float = 30.0) -> None:
    from vllm import SamplingParams, TokensPrompt

    dummy_params = SamplingParams(max_tokens=1)
    deadline = time.monotonic() + timeout
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed; an async offload may be stuck"
            )
        llm.generate(
            [TokensPrompt(prompt_token_ids=[0])],
            dummy_params,
            use_tqdm=False,
        )


def filesystem_stats(root: Path | None) -> dict[str, int]:
    if root is None or not root.exists():
        return {"block_files": 0, "block_bytes": 0}
    files = list(root.rglob("*.bin"))
    return {
        "block_files": len(files),
        "block_bytes": sum(path.stat().st_size for path in files),
    }


def wait_for_filesystem_count(
    llm, root: Path, minimum_files: int, timeout: float = 30.0
) -> dict[str, int]:
    """Drive completion polling until all expected filesystem blocks exist."""
    from vllm import SamplingParams

    deadline = time.monotonic() + timeout
    stats = filesystem_stats(root)
    while stats["block_files"] < minimum_files and time.monotonic() < deadline:
        llm.generate("cache poll", SamplingParams(max_tokens=1), use_tqdm=False)
        time.sleep(0.05)
        stats = filesystem_stats(root)
    if stats["block_files"] < minimum_files:
        raise TimeoutError(
            "Filesystem tier did not persist all expected KV blocks within the "
            f"timeout: expected={minimum_files}, current={stats['block_files']}"
        )
    # One more scheduler step consumes the final async completion notifications,
    # releasing the primary-tier references before an eviction prompt is stored.
    llm.generate("cache poll", SamplingParams(max_tokens=1), use_tqdm=False)
    return stats


def offload_metric_lines() -> list[str]:
    try:
        from prometheus_client import generate_latest

        lines = generate_latest().decode("utf-8").splitlines()
    except Exception:
        return []
    return [
        line
        for line in lines
        if "kv_offload" in line and line and not line.startswith("#")
    ]


def main() -> None:
    args = parse_args()
    if args.restore_cycles < 1:
        raise SystemExit("--restore-cycles must be at least 1")
    if args.fs_eviction_prompts < 1:
        raise SystemExit("--fs-eviction-prompts must be at least 1")

    model_path = Path(args.model).resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Not a local Hugging Face model directory: {model_path}")

    fs_root = Path(args.fs_root).resolve() if args.fs_root else None
    if args.mode.startswith("fs-") and fs_root is None:
        raise SystemExit("--fs-root is required for filesystem modes")
    if args.reuse_existing_cache and fs_root is None:
        raise SystemExit("--reuse-existing-cache requires a filesystem mode")
    if args.reuse_existing_cache and not args.reference_json:
        raise SystemExit("--reuse-existing-cache requires --reference-json")
    if fs_root is not None:
        fs_root.mkdir(parents=True, exist_ok=True)
        existing_stats = filesystem_stats(fs_root)
        if args.reuse_existing_cache and not existing_stats["block_files"]:
            raise SystemExit(
                f"Persistent-cache validation requires existing .bin files: {fs_root}"
            )
        if not args.reuse_existing_cache and existing_stats["block_files"]:
            raise SystemExit(
                "Filesystem validation requires a fresh root with no .bin files: "
                f"{fs_root}"
            )

    if args.mode.startswith("fs-") and os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "Filesystem modes require PYTHONHASHSEED=0 before Python starts "
            "so cache keys remain stable across processes"
        )
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    gpu_count = torch.cuda.device_count()
    if gpu_count != 1:
        raise AssertionError(f"Expected exactly one visible GPU, found {gpu_count}")
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in gpu_name:
        raise AssertionError(f"Expected the RTX 3090, found {gpu_name}")
    if "site-packages" not in str(Path(vllm.__file__).resolve()):
        raise AssertionError(
            f"vLLM is not imported from the installed wheel: {vllm.__file__}"
        )
    if vllm.__version__ != args.expected_version:
        raise AssertionError(
            f"Expected vLLM {args.expected_version}, found {vllm.__version__}"
        )

    print(f"Visible GPU: {gpu_name}", flush=True)
    print(f"Installed vLLM: {vllm.__version__} from {vllm.__file__}", flush=True)

    llm_kwargs: dict[str, object] = {
        "model": str(model_path),
        "dtype": "float16",
        "kv_cache_dtype": "auto",
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": 0.85,
        "kv_cache_memory_bytes": args.gpu_cache_mib * (1 << 20),
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "trust_remote_code": True,
        "generation_config": "vllm",
    }

    if args.mode != "baseline":
        policy = args.mode.rsplit("-", maxsplit=1)[1]
        extra_config: dict[str, object] = {
            "cpu_bytes_to_use": args.cpu_cache_mib * (1 << 20),
            "eviction_policy": policy,
            "offload_prompt_only": True,
        }
        if args.mode.startswith("fs-"):
            assert fs_root is not None
            extra_config.update(
                {
                    "spec_name": "TieringOffloadingSpec",
                    "secondary_tiers": [
                        {
                            "type": "fs",
                            "root_dir": str(fs_root),
                            "n_read_threads": 4,
                            "n_write_threads": 4,
                        }
                    ],
                }
            )
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        )

    llm = LLM(**llm_kwargs)
    try:
        prompt = build_prompt(args.prompt_repeats)
        tokenizer = llm.get_tokenizer()
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + args.max_tokens > args.max_model_len:
            raise AssertionError(
                f"Prompt has {prompt_tokens} tokens but max_model_len is "
                f"{args.max_model_len}"
            )

        engine_cache_config = llm.llm_engine.vllm_config.cache_config
        # vLLM 0.27 removed CacheConfig.hash_block_size. Older releases expose
        # it separately, while 0.27 uses block_size for the same fallback here.
        cache_block_tokens = int(
            getattr(engine_cache_config, "hash_block_size", None)
            or engine_cache_config.block_size
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=0,
        )

        started = time.perf_counter()
        cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        cold_result = serialize_output(cold, time.perf_counter() - started)

        if args.reuse_existing_cache:
            cached_tokens = int(cold_result["num_cached_tokens"] or 0)
            if cached_tokens == 0:
                raise AssertionError(
                    "The first request in the restarted engine reused zero "
                    "filesystem cache tokens"
                )
            reference_path = Path(args.reference_json).resolve()
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_token_ids = reference["cold"]["token_ids"]
            if cold_result["token_ids"] != reference_token_ids:
                raise AssertionError(
                    "Restarted-engine token IDs differ from the reference run: "
                    f"reference={reference_token_ids}, "
                    f"restarted={cold_result['token_ids']}"
                )
            result = {
                "mode": args.mode,
                "model": str(model_path),
                "gpu": gpu_name,
                "vllm_version": importlib.metadata.version("vllm"),
                "package_path": str(Path(vllm.__file__).resolve()),
                "cpu_cache_mib": args.cpu_cache_mib,
                "gpu_cache_mib": args.gpu_cache_mib,
                "cache_block_tokens": cache_block_tokens,
                "persistent_first_request": cold_result,
                "filesystem": filesystem_stats(fs_root),
                "reference_json": str(reference_path),
                "tokens_match": True,
                "persistent_restart_hit": True,
            }
            print(RESULT_MARKER, flush=True)
            result_text = json.dumps(result, sort_keys=True)
            print(result_text, flush=True)
            if args.result_json:
                result_path = Path(args.result_json).resolve()
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(result_text + "\n", encoding="utf-8")
            return

        restores: list[dict[str, object]] = []
        fs_observations: list[dict[str, object]] = []
        expected_fs_files = prompt_tokens // cache_block_tokens
        for cycle in range(args.restore_cycles):
            wait_for_prefix_cache_reset(llm)

            if fs_root is not None:
                before = wait_for_filesystem_count(llm, fs_root, expected_fs_files)

                eviction_results: list[dict[str, object]] = []
                for index in range(args.fs_eviction_prompts):
                    eviction_prompt = build_eviction_prompt(prompt, cycle, index)
                    eviction_tokens = len(tokenizer.encode(eviction_prompt))
                    if eviction_tokens + 1 > args.max_model_len:
                        raise AssertionError(
                            f"Eviction prompt has {eviction_tokens} tokens but "
                            f"max_model_len is {args.max_model_len}"
                        )
                    started = time.perf_counter()
                    eviction_output = llm.generate(
                        eviction_prompt,
                        SamplingParams(temperature=0.0, max_tokens=1, seed=0),
                        use_tqdm=False,
                    )[0]
                    eviction_results.append(
                        serialize_output(
                            eviction_output, elapsed=time.perf_counter() - started
                        )
                    )
                    wait_for_prefix_cache_reset(llm)
                    expected_fs_files += eviction_tokens // cache_block_tokens
                    wait_for_filesystem_count(llm, fs_root, expected_fs_files)

                after_eviction = filesystem_stats(fs_root)
                fs_observations.append(
                    {
                        "before_eviction": before,
                        "after_eviction": after_eviction,
                        "eviction_prompts": eviction_results,
                    }
                )

            started = time.perf_counter()
            restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
            restored_result = serialize_output(restored, time.perf_counter() - started)
            restores.append(restored_result)

            if cold_result["token_ids"] != restored_result["token_ids"]:
                raise AssertionError(
                    "Generated token IDs changed after cache restoration: "
                    f"cold={cold_result['token_ids']} restored="
                    f"{restored_result['token_ids']}"
                )
            cached_tokens = int(restored_result["num_cached_tokens"] or 0)
            if args.mode == "baseline" and cached_tokens != 0:
                raise AssertionError(
                    "Baseline retained GPU cached tokens after prefix-cache reset"
                )
            if args.mode != "baseline" and cached_tokens == 0:
                raise AssertionError(
                    "Offload mode restored zero cached tokens after GPU reset"
                )

        if fs_root is not None and not any(
            int(observation["after_eviction"]["block_files"]) > 0
            for observation in fs_observations
        ):
            raise AssertionError("Filesystem mode did not write any KV block files")

        result = {
            "mode": args.mode,
            "model": str(model_path),
            "gpu": gpu_name,
            "vllm_version": importlib.metadata.version("vllm"),
            "package_path": str(Path(vllm.__file__).resolve()),
            "cpu_cache_mib": args.cpu_cache_mib if args.mode != "baseline" else 0,
            "gpu_cache_mib": args.gpu_cache_mib,
            "cache_block_tokens": cache_block_tokens,
            "cold": cold_result,
            "restores": restores,
            "fs_observations": fs_observations,
            "offload_metrics": offload_metric_lines(),
            "tokens_match": True,
        }
        print(RESULT_MARKER, flush=True)
        result_text = json.dumps(result, sort_keys=True)
        print(result_text, flush=True)
        if args.result_json:
            result_path = Path(args.result_json).resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(result_text + "\n", encoding="utf-8")
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
