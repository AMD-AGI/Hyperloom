# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark orchestration engine.

Pure, injectable helpers that let the bypass runner drive a benchmark in
Python instead of shelling out to Magpie's scripts:

* resolve the InferenceX checkout (still required — bypass reuses InferenceX's
  benchmark client and lm-eval, the same runtime Magpie uses),
* build the per-framework server launch command,
* build the InferenceX benchmark client command,
* build the lm-eval command,
* poll an HTTP endpoint for server readiness.

The command builders are pure. ``wait_for_server_ready`` and
``server_health_ok`` take an injectable HTTP probe; ``resolve_inferencex_root``
reads env vars; ``lifecycle_files_present`` / ``write_lifecycle_files`` touch
the filesystem under a caller-supplied ``pid_dir``. All of it is therefore
unit-testable without a GPU, a real server, or the Magpie repository.
"""

from __future__ import annotations

import http.client
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
    python_exe: str = "python3",
    framework_python: str = "",
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
        python_exe: Interpreter used to launch python-module servers
            (sglang/atom); defaults to a PATH ``python3``. Callers should pass
            the interpreter running the bypass runner so the server loads the
            SAME venv (avoids a PATH ``python3`` that cannot import the
            framework). vllm uses its own ``vllm`` console script and ignores
            this unless ``framework_python`` is set.
        framework_python: When non-empty, the explicit interpreter that built
            the from-source vLLM/sglang.  For vLLM this switches from the
            bare ``vllm serve`` console script to ``python -m
            vllm.entrypoints.openai.api_server`` so the interpreter is
            honored. For sglang/atom it replaces ``python_exe``.

    Returns:
        The server launch argv list.

    Raises:
        ValueError: When the framework has no known server launcher.
    """
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
            # vLLM enables the torch profiler via --profiler-config
            # (the legacy VLLM_TORCH_PROFILER_DIR env is ignored), without
            # which /start_profile returns 404 and no trace is written.
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
    """Wait for ``<base_url>/health`` until ready, the server exits, or timeout.

    Args:
        base_url: Server base URL.
        timeout_s: Max seconds to wait.
        server_exited: Whether the server process has already exited. Required,
            because a server that is gone will never answer ``/health`` and
            waiting out the rest of ``timeout_s`` on it only holds the
            ``server_lifecycle`` lane -- an hour per attempt on the enablement
            rounds, whose whole subject is servers that fail to boot. Injected
            rather than taken as a process handle to keep this module free of
            process machinery, as its own contract promises.
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


# --- server_lifecycle pid/meta helpers -------------------------------------
# Filenames match Hyperloom's teardown_lifecycle_server convention so a
# persistent bypass server can be reused across processes and torn down by
# either side: <pid_dir>/<framework>_<port>.pid ("<pid> <pgid>") + .json meta.


def lifecycle_pid_file(pid_dir: str, framework: str, port: int) -> Path:
    """Return the pid file path for a persistent server."""
    return Path(pid_dir) / f"{framework}_{port}.pid"


def lifecycle_meta_file(pid_dir: str, framework: str, port: int) -> Path:
    """Return the meta file path for a persistent server."""
    return Path(pid_dir) / f"{framework}_{port}.json"


def lifecycle_files_present(pid_dir: str, framework: str, port: int) -> bool:
    """Whether both pid and meta files exist for a persistent server.

    A healthy ``/health`` port WITHOUT these files means the port is held by a
    server this bypass run did not launch (foreign/zombie), so its reuse-key
    would not match. Callers use this to distinguish a genuine reuse target
    from an unrelated listener.
    """
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


#: Tokens the boot probe asks for, and the floor it accepts. The floor is above
#: one because a prefill-only server still answers the first token.
_COMPLETION_PROBE_TOKENS = 8
_COMPLETION_PROBE_MIN_TOKENS = 2


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


def one_short_completion(base_url: str, *, timeout_s: float = 120.0) -> tuple[bool, str]:
    """Ask a served model for one short completion and say whether it produced one.

    A 200 from ``/health`` is answered by a server whose engine has died;
    generating tokens is the only evidence that the thing behind it serves.

    Args:
        base_url: Server base URL.
        timeout_s: Budget for the whole exchange.

    Returns:
        tuple[bool, str]: ``(generated, detail)`` -- ``detail`` names what went
        wrong when nothing was generated, and is empty on success.
    """
    from hyperloom.inference_optimizer.multi_node._internal.serving_probe import generated_tokens

    root = base_url.rstrip("/")
    deadline = time.monotonic() + float(timeout_s)

    def _left() -> float:
        return max(0.1, deadline - time.monotonic())

    try:
        listing = _json_get(f"{root}/v1/models", _left())
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return False, f"/v1/models: {type(exc).__name__}: {exc}"
    entries = listing.get("data") if isinstance(listing, dict) else None
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return False, "/v1/models registered no model"
    model_id = str(entries[0].get("id") or "")
    if not model_id:
        return False, "/v1/models carries no model id"
    try:
        body = _json_post(
            f"{root}/v1/completions",
            {
                "model": model_id,
                "prompt": "hi",
                "max_tokens": _COMPLETION_PROBE_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            },
            _left(),
        )
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return False, f"/v1/completions: {type(exc).__name__}: {exc}"
    produced = generated_tokens(body)
    if produced < _COMPLETION_PROBE_MIN_TOKENS:
        return False, f"/v1/completions generated {produced} tokens (need {_COMPLETION_PROBE_MIN_TOKENS})"
    return True, ""
