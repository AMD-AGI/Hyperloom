"""Derive PMC roofline server/benchmark commands from a materialized Magpie YAML."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def derive_pmc_roofline_params_from_config(
    config_path: Path | str,
    *,
    framework: str = "",
    model_path: str = "",
    gpu_type: str = "",
    output_dir: str | Path | None = None,
    port: int | None = None,
) -> dict[str, Any] | None:
    """Build pmc_roofline task params from a materialized Magpie YAML."""
    path = Path(config_path)
    if not path.is_file():
        log.warning("PMC roofline config missing: %s", path)
        return None
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read PMC roofline workload config %s: %s", path, exc)
        return None

    bench = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    fw = str(bench.get("framework") or framework or "vllm").lower()
    model = str(bench.get("model") or model_path or "").strip()
    if not model:
        log.warning("PMC roofline config has no model path")
        return None

    listen_port = int(
        port if port is not None
        else os.environ.get("HYPERLOOM_PMC_ROOFLINE_PORT", "30001")
    )
    tp = str(envs.get("TP") or os.environ.get("TP") or "1")
    precision = str(
        bench.get("precision") or os.environ.get("PRECISION") or "bf16"
    ).lower()
    max_model_len = str(
        envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN") or "8192"
    )
    extra_key = "EXTRA_VLLM_ARGS" if fw == "vllm" else "EXTRA_SGLANG_ARGS"
    extra_args = shlex.split(str(envs.get(extra_key) or ""))
    resolved_gpu = (
        os.environ.get("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE")
        or gpu_type
        or os.environ.get("GPU_TYPE", "")
    ).strip().lower()

    if fw == "vllm":
        dtype = "bfloat16" if precision in {"bf16", "bfloat16"} else precision
        server_cmd = [
            "vllm", "serve", model,
            "--host", "0.0.0.0",
            "--port", str(listen_port),
            "--tensor-parallel-size", tp,
            "--trust-remote-code",
            "--dtype", dtype,
            "--max-model-len", max_model_len,
        ] + extra_args
        backend = "vllm"
    else:
        server_cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model,
            "--host", "0.0.0.0",
            "--port", str(listen_port),
            "--tensor-parallel-size", tp,
            "--trust-remote-code",
            "--max-model-len", max_model_len,
        ] + extra_args
        backend = "sglang"

    inferencex = os.environ.get("INFERENCEX_PATH", "/hyperloom/InferenceX").rstrip("/")
    bench_script = f"{inferencex}/utils/bench_serving/benchmark_serving.py"
    isl = str(envs.get("ISL") or os.environ.get("ISL") or "256")
    osl = str(envs.get("OSL") or os.environ.get("OSL") or "256")
    conc = int(envs.get("CONC") or os.environ.get("CONC") or 1)
    num_prompts = str(envs.get("NUM_PROMPTS") or max(conc, 1))
    benchmark_cmd = [
        "python", bench_script,
        "--backend", backend,
        "--base-url", f"http://127.0.0.1:{listen_port}",
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", isl,
        "--random-output-len", osl,
        "--num-prompts", num_prompts,
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--ignore-eos",
    ]

    out_dir = str(output_dir) if output_dir else ""
    return {
        "profile_mode": os.environ.get("HYPERLOOM_PMC_ROOFLINE_MODE", "launch"),
        "server_cmd": server_cmd,
        "health_url": f"http://127.0.0.1:{listen_port}/health",
        "benchmark_cmd": benchmark_cmd,
        "output_dir": out_dir,
        "duration_ms": int(os.environ.get("HYPERLOOM_PMC_ROOFLINE_DURATION_MS", "15000")),
        "precision": precision,
        "gpu_type": resolved_gpu,
        "startup_timeout_s": int(
            os.environ.get("HYPERLOOM_PMC_ROOFLINE_STARTUP_TIMEOUT_S", "600")
        ),
        "config_path": str(path),
    }


__all__ = ["derive_pmc_roofline_params_from_config"]
