# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark orchestration engine."""

from __future__ import annotations

import http.client
import os
import time
import urllib.request
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ServerSpec:
    """What a bypass server is launched with; ``profile_dir`` enables the torch profiler."""

    framework: str
    model: str
    tp: int
    port: int
    max_model_len: int | None = None
    profile_dir: str | None = None


def build_server_command(
    spec: ServerSpec,
    extra_args: list[str],
    *,
    python_exe: str = "python3",
    framework_python: str = "",
) -> list[str]:
    """Build the per-framework server launch argv."""
    model = spec.model
    tp = spec.tp
    port = spec.port
    max_model_len = spec.max_model_len
    profile_dir = spec.profile_dir
    fw = spec.framework.lower()
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
        if not profile_dir:
            return cmd + list(extra_args)
        # vLLM enables the torch profiler via --profiler-config (the legacy VLLM_TORCH_PROFILER_DIR env is
        # ignored), without which /start_profile returns 404 and no trace is written.
        #
        # Last, after extra_args, the way Magpie's launcher orders its own: EXTRA_VLLM_ARGS has to carry a
        # torch_profiler_dir for the argv preflight to accept the profile bounds beside it -- ProfilerConfig
        # refuses `profiler=torch` without one -- and that value is a placeholder standing in for whatever the
        # launcher computes. Emitting it first would let the placeholder win vLLM's last-wins dotted-flag merge
        # and send the trace somewhere this backend's discovery never looks.
        return (
            cmd
            + list(extra_args)
            + [
                "--profiler-config.profiler",
                "torch",
                "--profiler-config.torch_profiler_dir",
                profile_dir,
            ]
        )
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
    raise ValueError(f"no server launcher for framework={spec.framework!r}")


@dataclass(frozen=True)
class ClientRun:
    """One InferenceX ``benchmark_serving.py`` run: its checkout and interpreter, target, load and result dir.

    The result is written to ``<result_dir>/inferencex_result.json``.
    """

    inferencex_root: str
    python_exe: str
    model: str
    base_url: str
    isl: int
    osl: int
    conc: int
    random_range_ratio: float
    result_dir: str
    num_prompts: int | None = None
    num_warmups: int | None = None
    profile: bool = False
    trust_remote_code: bool = False


def build_client_command(run: ClientRun) -> list[str]:
    """Build the InferenceX benchmark client argv."""
    conc = run.conc
    bench_py = str(Path(run.inferencex_root) / "utils" / "bench_serving" / "benchmark_serving.py")
    prompts = run.num_prompts if run.num_prompts is not None else (conc if run.profile else conc * 10)
    warmups = run.num_warmups if run.num_warmups is not None else 2 * conc
    cmd = [
        run.python_exe,
        bench_py,
        "--model",
        run.model,
        "--backend",
        "vllm",
        "--base-url",
        run.base_url,
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(run.isl),
        "--random-output-len",
        str(run.osl),
        "--random-range-ratio",
        str(run.random_range_ratio),
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
        run.result_dir,
        "--result-filename",
        "inferencex_result.json",
    ]
    if run.profile:
        cmd.append("--profile")
    if run.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


@dataclass(frozen=True)
class EvalRun:
    """One lm-eval accuracy pass against an OpenAI-compatible endpoint."""

    python_exe: str
    model: str
    base_url: str
    conc: int
    out_dir: str
    tasks: str = "gsm8k"
    limit: str | None = None


def build_eval_command(run: EvalRun) -> list[str]:
    """Build the lm-eval argv targeting an OpenAI-compatible endpoint."""
    completions_url = f"{run.base_url.rstrip('/')}/v1/completions"
    model_args = (
        f"model={run.model},base_url={completions_url},num_concurrent={run.conc},"
        "tokenizer_backend=huggingface,trust_remote_code=true"
    )
    cmd = [
        run.python_exe,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--tasks",
        run.tasks,
        "--model_args",
        model_args,
        "--batch_size",
        "auto",
        "--output_path",
        run.out_dir,
    ]
    if run.limit:
        cmd += ["--limit", run.limit]
    return cmd


@dataclass(frozen=True)
class HealthPoll:
    """How the readiness wait polls: its interval, probe and clock (injected by tests)."""

    poll_s: float = 2.0
    probe: Callable[[str], int] | None = None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic


def wait_for_server_ready(
    base_url: str,
    *,
    timeout_s: float,
    server_exited: Callable[[], bool],
    poll: HealthPoll = HealthPoll(),
) -> bool:
    """Poll ``<base_url>/health`` until ready or timeout."""
    health_url = f"{base_url.rstrip('/')}/health"

    def _default_probe(url: str) -> int:
        with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310 - fixed local health probe
            return int(getattr(resp, "status", 0) or resp.getcode())

    do_probe = poll.probe or _default_probe
    now = poll.now
    deadline = now() + timeout_s
    while now() < deadline:
        try:
            if do_probe(health_url) == 200:
                return True
        except (OSError, http.client.HTTPException, ValueError):
            pass
        # Checked after the probe, so a server that answered and exited in the
        # same breath is still credited with having come up.
        if server_exited():
            return False
        poll.sleep(poll.poll_s)
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


@dataclass(frozen=True)
class LifecycleRecord:
    """The persistent server a lifecycle pid/meta pair records."""

    framework: str
    port: int
    pid: int
    pgid: int
    model: str


def write_lifecycle_files(pid_dir: str, record: LifecycleRecord) -> None:
    """Persist pid + meta for a lifecycle server (Hyperloom-compatible)."""
    import json as _json

    framework = record.framework
    port = record.port
    Path(pid_dir).mkdir(parents=True, exist_ok=True)
    lifecycle_pid_file(pid_dir, framework, port).write_text(f"{record.pid} {record.pgid}\n", encoding="utf-8")
    lifecycle_meta_file(pid_dir, framework, port).write_text(
        _json.dumps(
            {
                "pid": record.pid,
                "pgid": record.pgid,
                "framework": framework,
                "port": port,
                "model": record.model,
                "base_url": f"http://127.0.0.1:{port}",
            }
        ),
        encoding="utf-8",
    )


def server_health_ok(base_url: str, *, probe: Callable[[str], int] | None = None) -> bool:
    """One-shot health probe (no polling); True iff /health returns 200."""
    health_url = f"{base_url.rstrip('/')}/health"

    def _default_probe(url: str) -> int:
        with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310 - fixed local health probe
            return int(getattr(resp, "status", 0) or resp.getcode())

    do_probe = probe or _default_probe
    try:
        return do_probe(health_url) == 200
    except (OSError, http.client.HTTPException, ValueError):
        return False
