# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bypass benchmark orchestration engine.

Pure, injectable helpers that let the bypass runner drive a benchmark in
Python instead of shelling out to Magpie's scripts:

* resolve the InferenceX checkout (still required — bypass reuses InferenceX's
  benchmark client and lm-eval, the same runtime Magpie uses),
* build the per-framework server launch command,
* build the InferenceX benchmark client command,
* build the lm-eval command,
* poll an HTTP endpoint for server readiness.

Every function here is side-effect free except ``wait_for_server_ready``, whose
HTTP probe is injectable, so the whole layer is unit-testable without a GPU,
a real server, or the Magpie repository.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

# Frameworks whose server this engine can launch. xdit/scriptable and remote
# (BENCHMARK_BASE_URL) flows are handled elsewhere / deferred.
SERVER_FRAMEWORKS = ("sglang", "vllm", "atom")

DEFAULT_PORT = 8888


def resolve_inferencex_root(bench: dict[str, Any]) -> str:
    """Resolve the InferenceX checkout root.

    Precedence: explicit YAML ``inferencex_path`` -> ``MAGPIE_INFERENCEX_PATH``
    -> ``INFERENCEX_PATH``. bypass depends on InferenceX (not Magpie) for the
    benchmark client and lm-eval.

    Args:
        bench: The ``benchmark`` section of the config.

    Returns:
        The resolved path string (possibly empty if unresolved).
    """
    return (
        str(bench.get("inferencex_path") or "").strip()
        or os.environ.get("MAGPIE_INFERENCEX_PATH", "").strip()
        or os.environ.get("INFERENCEX_PATH", "").strip()
    )


def build_server_command(
    *,
    framework: str,
    model: str,
    tp: int,
    port: int,
    max_model_len: int | None,
    extra_args: list[str],
    profile_dir: str | None,
) -> list[str]:
    """Build the per-framework server launch argv.

    Args:
        framework: One of sglang/vllm/atom.
        model: Model path/id.
        tp: Tensor-parallel size.
        port: HTTP serving port.
        max_model_len: Optional max model length (vllm/atom).
        extra_args: Already-tokenized extra server args.
        profile_dir: Torch profiler dir when profiling, else None.

    Returns:
        The server launch argv list.

    Raises:
        ValueError: When the framework has no known server launcher.
    """
    fw = framework.lower()
    if fw == "sglang":
        cmd = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--trust-remote-code",
            "--tensor-parallel-size", str(tp),
        ]
        return cmd + list(extra_args)
    if fw == "vllm":
        cmd = [
            "vllm", "serve", model,
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--trust-remote-code",
        ]
        if max_model_len:
            cmd += ["--max-model-len", str(max_model_len)]
        return cmd + list(extra_args)
    if fw == "atom":
        cmd = [
            "python3", "-m", "atom.entrypoints.openai_server",
            "--model", model,
            "-tp", str(tp),
            "--server-port", str(port),
        ]
        if max_model_len:
            cmd += ["--max-model-len", str(max_model_len)]
        if profile_dir:
            cmd += ["--torch-profiler-dir", profile_dir]
        return cmd + list(extra_args)
    raise ValueError(f"no server launcher for framework={framework!r}")


def build_client_command(
    *,
    inferencex_root: str,
    python_exe: str,
    model: str,
    base_url: str,
    isl: int,
    osl: int,
    conc: int,
    random_range_ratio: float,
    result_dir: str,
    result_filename: str,
    num_prompts: int | None = None,
    num_warmups: int | None = None,
    profile: bool = False,
    trust_remote_code: bool = False,
) -> list[str]:
    """Build the InferenceX benchmark client argv.

    Mirrors the remote-direct path Magpie uses so the same InferenceX
    ``benchmark_serving.py`` produces the same ``inferencex_result.json``.

    Args:
        inferencex_root: InferenceX checkout root.
        python_exe: Interpreter to run the client with.
        model: Model path/id.
        base_url: OpenAI-compatible server base URL.
        isl: Random input length.
        osl: Random output length.
        conc: Max concurrency.
        random_range_ratio: Random range ratio.
        result_dir: Directory the result JSON is written to.
        result_filename: Result basename (without .json).
        num_prompts: Prompt count; defaults to conc*10 (or conc when profiling).
        num_warmups: Warmup count; defaults to 2*conc.
        profile: Whether to pass --profile.
        trust_remote_code: Whether to pass --trust-remote-code.

    Returns:
        The benchmark client argv list.
    """
    bench_py = str(Path(inferencex_root) / "utils" / "bench_serving" / "benchmark_serving.py")
    prompts = num_prompts if num_prompts is not None else (conc if profile else conc * 10)
    warmups = num_warmups if num_warmups is not None else 2 * conc
    cmd = [
        python_exe, bench_py,
        "--model", model,
        "--backend", "vllm",
        "--base-url", base_url,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", str(random_range_ratio),
        "--num-prompts", str(prompts),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups", str(warmups),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--result-dir", result_dir,
        "--result-filename", f"{result_filename}.json",
    ]
    if profile:
        cmd.append("--profile")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def build_eval_command(
    *,
    python_exe: str,
    model: str,
    base_url: str,
    conc: int,
    out_dir: str,
    tasks: str = "gsm8k",
    batch_size: str = "auto",
    limit: str | None = None,
) -> list[str]:
    """Build the lm-eval argv targeting an OpenAI-compatible endpoint.

    Writes results under ``out_dir`` so ``parse_eval_results`` finds the
    standard lm-eval ``results*.json`` schema.

    Args:
        python_exe: Interpreter to run lm-eval with.
        model: Model id for lm-eval + tokenizer.
        base_url: Server base URL (``/v1/completions`` is appended).
        conc: Concurrency cap.
        out_dir: Output directory for lm-eval results.
        tasks: Comma-separated lm-eval tasks.
        batch_size: lm-eval batch size.
        limit: Optional sample cap for smoke runs.

    Returns:
        The lm-eval argv list.
    """
    completions_url = f"{base_url.rstrip('/')}/v1/completions"
    model_args = (
        f"model={model},base_url={completions_url},num_concurrent={conc},"
        "tokenizer_backend=huggingface,trust_remote_code=true"
    )
    cmd = [
        python_exe, "-m", "lm_eval",
        "--model", "local-completions",
        "--tasks", tasks,
        "--model_args", model_args,
        "--batch_size", batch_size,
        "--output_path", out_dir,
    ]
    if limit:
        cmd += ["--limit", limit]
    return cmd


def wait_for_server_ready(
    base_url: str,
    *,
    timeout_s: float,
    poll_s: float = 2.0,
    probe: Callable[[str], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``<base_url>/health`` until ready or timeout.

    Args:
        base_url: Server base URL.
        timeout_s: Max seconds to wait.
        poll_s: Seconds between probes.
        probe: Injectable probe returning an HTTP status code; defaults to a
            real GET. Any exception/non-200 is treated as not-ready.
        sleep: Injectable sleep (tests pass a no-op).
        now: Injectable monotonic clock.

    Returns:
        True when the server became ready within the timeout, else False.
    """
    health_url = f"{base_url.rstrip('/')}/health"

    def _default_probe(url: str) -> int:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - localhost health probe
            return int(getattr(resp, "status", 0) or resp.getcode())

    do_probe = probe or _default_probe
    deadline = now() + timeout_s
    while now() < deadline:
        try:
            if do_probe(health_url) == 200:
                return True
        except Exception:  # noqa: BLE001 - not-ready yet; keep polling
            pass
        sleep(poll_s)
    return False