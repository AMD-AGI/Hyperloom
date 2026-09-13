# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark orchestration engine."""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

# Frameworks whose server this engine can launch. xdit/scriptable and remote (BENCHMARK_BASE_URL) flows are handled
# elsewhere / deferred.
SERVER_FRAMEWORKS = ("sglang", "vllm", "atom")

DEFAULT_PORT = 8888


def resolve_inferencex_root(bench: dict[str, Any]) -> str:
    """Resolve the InferenceX checkout root."""
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
    python_exe: str = "python3",
    framework_python: str = "",
) -> list[str]:
    """Build the per-framework server launch argv."""
    fw = framework.lower()
    interp = framework_python or python_exe
    if fw == "sglang":
        cmd = [
            interp,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "0.0.0.0",  # nosec B104 - bypass server must accept benchmark probes.
            "--port",
            str(port),
            "--trust-remote-code",
            "--tensor-parallel-size",
            str(tp),
        ]
        return cmd + list(extra_args)
    if fw == "vllm":
        if framework_python:
            cmd = [
                framework_python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(tp),
                "--trust-remote-code",
            ]
        else:
            cmd = [
                "vllm",
                "serve",
                model,
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(tp),
                "--trust-remote-code",
            ]
        if max_model_len:
            cmd += ["--max-model-len", str(max_model_len)]
        if profile_dir:
            # vLLM enables the torch profiler via --profiler-config (the legacy VLLM_TORCH_PROFILER_DIR env is
            # ignored), without which /start_profile returns 404 and no trace is written.
            cmd += [
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                profile_dir,
            ]
        return cmd + list(extra_args)
    if fw == "atom":
        cmd = [
            interp,
            "-m",
            "atom.entrypoints.openai_server",
            "--model",
            model,
            "-tp",
            str(tp),
            "--server-port",
            str(port),
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
    """Build the InferenceX benchmark client argv."""
    bench_py = str(Path(inferencex_root) / "utils" / "bench_serving" / "benchmark_serving.py")
    prompts = num_prompts if num_prompts is not None else (conc if profile else conc * 10)
    warmups = num_warmups if num_warmups is not None else 2 * conc
    cmd = [
        python_exe,
        bench_py,
        "--model",
        model,
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--random-range-ratio",
        str(random_range_ratio),
        "--num-prompts",
        str(prompts),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups",
        str(warmups),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        result_dir,
        "--result-filename",
        f"{result_filename}.json",
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
    """Build the lm-eval argv targeting an OpenAI-compatible endpoint."""
    completions_url = f"{base_url.rstrip('/')}/v1/completions"
    model_args = (
        f"model={model},base_url={completions_url},num_concurrent={conc},"
        "tokenizer_backend=huggingface,trust_remote_code=true"
    )
    cmd = [
        python_exe,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        tasks,
        "--model_args",
        model_args,
        "--batch_size",
        batch_size,
        "--output_path",
        out_dir,
    ]
    if limit:
        cmd += ["--limit", limit]
    return cmd


def wait_for_server_ready(
    base_url: str,
    *,
    timeout_s: float,
    server_exited: Callable[[], bool],
    poll_s: float = 2.0,
    probe: Callable[[str], int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> bool:
    """Poll ``<base_url>/health`` until ready or timeout."""
    health_url = f"{base_url.rstrip('/')}/health"

    def _default_probe(url: str) -> int:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310  # nosec B310 - fixed local health probe
            return int(getattr(resp, "status", 0) or resp.getcode())

    do_probe = probe or _default_probe
    deadline = now() + timeout_s
    while now() < deadline:
        try:
            if do_probe(health_url) == 200:
                return True
        except Exception:  # noqa: BLE001 - not-ready yet; keep polling
            pass
        # Checked after the probe, so a server that answered and exited in the
        # same breath is still credited with having come up.
        if server_exited():
            return False
        sleep(poll_s)
    return False


# --- server_lifecycle pid/meta helpers ------------------------------------- Filenames match Hyperloom's
# teardown_lifecycle_server convention so a persistent bypass server can be reused across processes and torn down by
# either side: <pid_dir>/<framework>_<port>.pid ("<pid> <pgid>") + .json meta.


def lifecycle_pid_file(pid_dir: str, framework: str, port: int) -> Path:
    """Return the pid file path for a persistent server."""
    return Path(pid_dir) / f"{framework}_{port}.pid"


def lifecycle_meta_file(pid_dir: str, framework: str, port: int) -> Path:
    """Return the meta file path for a persistent server."""
    return Path(pid_dir) / f"{framework}_{port}.json"


def lifecycle_files_present(pid_dir: str, framework: str, port: int) -> bool:
    """Whether both pid and meta files exist for a persistent server."""
    return (
        lifecycle_pid_file(pid_dir, framework, port).exists() and lifecycle_meta_file(pid_dir, framework, port).exists()
    )


def write_lifecycle_files(
    *,
    pid_dir: str,
    framework: str,
    port: int,
    pid: int,
    pgid: int,
    model: str,
) -> None:
    """Persist pid + meta for a lifecycle server (Hyperloom-compatible)."""
    import json as _json

    Path(pid_dir).mkdir(parents=True, exist_ok=True)
    lifecycle_pid_file(pid_dir, framework, port).write_text(f"{pid} {pgid}\n", encoding="utf-8")
    lifecycle_meta_file(pid_dir, framework, port).write_text(
        _json.dumps(
            {
                "pid": pid,
                "pgid": pgid,
                "framework": framework,
                "port": port,
                "model": model,
                "base_url": f"http://127.0.0.1:{port}",
            }
        ),
        encoding="utf-8",
    )


def server_health_ok(base_url: str, *, probe: Callable[[str], int] | None = None) -> bool:
    """One-shot health probe (no polling); True iff /health returns 200."""
    health_url = f"{base_url.rstrip('/')}/health"

    def _default_probe(url: str) -> int:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310  # nosec B310 - fixed local health probe
            return int(getattr(resp, "status", 0) or resp.getcode())

    do_probe = probe or _default_probe
    try:
        return do_probe(health_url) == 200
    except Exception:  # noqa: BLE001
        return False


def _json_post(url: str, payload: dict[str, Any], timeout_s: float) -> Any:
    """POST ``payload`` as JSON and return the decoded body."""
    import json as _json

    request = urllib.request.Request(  # noqa: S310  # nosec B310 - fixed local serving endpoint
        url,
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310  # nosec B310
        return _json.loads(resp.read().decode("utf-8", "replace"))


def _json_get(url: str, timeout_s: float) -> Any:
    """GET ``url`` and return the decoded JSON body."""
    import json as _json

    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310  # nosec B310
        return _json.loads(resp.read().decode("utf-8", "replace"))
