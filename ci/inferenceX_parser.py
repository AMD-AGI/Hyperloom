"""Parse InferenceX configs and fetch benchmark data via API."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests
import yaml

log = logging.getLogger(__name__)

INFERENCEX_API = "https://inferencex.semianalysis.com/api/v1/benchmarks"


def resolve_var(value: Any, env: dict | None = None) -> Any:
    """Resolve ${VAR} placeholders from environment."""
    if not isinstance(value, str):
        return value
    env = env or os.environ

    def _replace(m: re.Match) -> str:
        return env.get(m.group(1), m.group(0))

    return re.sub(r"\$\{(\w+)}", _replace, value)


# ── InferenceX GitHub config parsing ──

def fetch_amd_master_yaml(
    repo_url: str,
    config_path: str = ".github/configs/amd-master.yaml",
    ref: str = "main",
) -> dict:
    """Clone InferenceX repo (shallow) and parse amd-master.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "clone", "--depth=1", f"--branch={ref}", repo_url, tmpdir],
            check=True, capture_output=True, text=True,
        )
        yaml_path = Path(tmpdir) / config_path
        with open(yaml_path) as f:
            return yaml.safe_load(f)


def get_latest_commit(repo_url: str, ref: str = "main") -> str:
    """Get latest commit SHA from remote without cloning."""
    result = subprocess.run(
        ["git", "ls-remote", repo_url, f"refs/heads/{ref}"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.split()[0] if result.stdout.strip() else ""


def parse_model_entry(entry: dict) -> dict:
    """Extract structured config from an amd-master.yaml model entry."""
    seq_configs = entry.get("seq-len-configs", [])
    first_seq = seq_configs[0] if seq_configs else {}
    first_search = (first_seq.get("search-space") or [{}])[0]

    isl_osl_list = []
    for sc in seq_configs:
        isl_osl_list.append((sc.get("isl", 1024), sc.get("osl", 1024)))

    return {
        "model_hf": entry.get("model", ""),
        "image": entry.get("image", ""),
        "model_prefix": entry.get("model-prefix", ""),
        "runner": entry.get("runner", ""),
        "precision": entry.get("precision", ""),
        "framework": entry.get("framework", "sglang"),
        "multinode": entry.get("multinode", False),
        "tp": first_search.get("tp", 8),
        "ep": first_search.get("ep", 1),
        "conc_start": first_search.get("conc-start", 4),
        "conc_end": first_search.get("conc-end", 64),
        "isl_osl_configs": isl_osl_list,
    }


# ── InferenceX Benchmark API ──

def fetch_benchmarks(model_api_name: str, api_url: str | None = None) -> list[dict]:
    """Fetch benchmark data from InferenceX API."""
    url = api_url or INFERENCEX_API
    resp = requests.get(f"{url}?model={model_api_name}", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        log.warning("InferenceX API error for %s: %s", model_api_name, data["error"])
        return []
    return data


def find_benchmark(
    benchmarks: list[dict],
    hardware: str,
    isl: int,
    osl: int,
    precision: str | None = None,
    image: str | None = None,
) -> dict | None:
    """Find the best benchmark entry matching hardware/ISL/OSL/precision.

    When ``image`` is provided, prefer entries from the same image to
    avoid comparing across frameworks (e.g. sglang vs atom).  Falls
    back to all candidates if no same-image match exists.

    When multiple concurrency levels exist, returns the one with the
    highest tput_per_gpu.
    """
    candidates = []
    for b in benchmarks:
        if (b.get("hardware") == hardware
                and b.get("isl") == isl
                and b.get("osl") == osl):
            if precision and b.get("precision") != precision:
                continue
            candidates.append(b)
    if not candidates:
        return None

    if image:
        same_image = [b for b in candidates if image in (b.get("image") or "")]
        if same_image:
            candidates = same_image

    return max(candidates,
               key=lambda x: (x.get("metrics") or {}).get("tput_per_gpu") or 0)


def find_benchmark_script(
    repo_path: Path | str,
    ifx_key: str,
    scripts_path: str = "benchmarks/single_node",
) -> str | None:
    """Find the InferenceX benchmark script for a given model key.

    Tries exact match first (e.g. minimaxm2.5_fp8_mi355x.sh),
    then falls back to prefix-based glob.
    """
    scripts_dir = Path(repo_path) / scripts_path
    if not scripts_dir.is_dir():
        return None

    normalized = ifx_key.replace("-", "_").replace(".", "")
    for sh in sorted(scripts_dir.glob("*.sh")):
        stem = sh.stem.replace(".", "")
        if stem == normalized:
            return f"{scripts_path}/{sh.name}"

    prefix = normalized.rsplit("_", 1)[0] if "_" in normalized else normalized
    for sh in sorted(scripts_dir.glob("*.sh")):
        stem = sh.stem.replace(".", "")
        if stem.startswith(prefix):
            return f"{scripts_path}/{sh.name}"

    return None


def find_benchmark_script_from_clone(
    repo_url: str,
    ifx_key: str,
    scripts_path: str = "benchmarks/single_node",
    ref: str = "main",
) -> str | None:
    """Clone InferenceX (shallow) and find the benchmark script path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["git", "clone", "--depth=1", f"--branch={ref}", repo_url, tmpdir],
            check=True, capture_output=True, text=True,
        )
        return find_benchmark_script(tmpdir, ifx_key, scripts_path)


def format_benchmark_for_prompt(
    benchmarks: list[dict],
    target_gpu: str,
    isl: int,
    osl: int,
    precision: str,
    image: str | None = None,
) -> str:
    """Format InferenceX benchmark data as text for the Claw prompt.

    API response nests performance data under a 'metrics' sub-object.
    """
    target = find_benchmark(benchmarks, target_gpu, isl, osl, precision, image)
    if not target:
        return f"# No InferenceX data for {target_gpu} at ISL={isl}/OSL={osl}/{precision}"

    m = target.get("metrics", {})
    lines = [
        f"Hardware: {target.get('hardware')}",
        f"ISL/OSL: {target.get('isl')}/{target.get('osl')}",
        f"Precision: {target.get('precision')}",
        f"TP: {target.get('decode_tp', 'N/A')}",
        f"Concurrency: {target.get('conc', 'N/A')}",
        f"Output Throughput/GPU (tok/s): {m.get('output_tput_per_gpu', 'N/A')}",
        f"Input Throughput/GPU (tok/s): {m.get('input_tput_per_gpu', 'N/A')}",
        f"Total Throughput/GPU (tok/s): {m.get('tput_per_gpu', 'N/A')}",
        f"Mean TPOT (s): {m.get('mean_tpot', 'N/A')}",
        f"Mean TTFT (s): {m.get('mean_ttft', 'N/A')}",
        f"Mean E2EL (s): {m.get('mean_e2el', 'N/A')}",
        f"Mean ITL (s): {m.get('mean_itl', 'N/A')}",
        f"Image: {target.get('image', 'N/A')}",
        f"Date: {target.get('date', 'N/A')}",
    ]
    return "\n".join(lines)


# ── Config merging ──

def merge_model_config(
    model_cfg: dict,
    ifx_entry: dict,
    defaults: dict,
    harbor_prefix: str,
    ifx_benchmarks: list[dict],
) -> dict:
    """Merge InferenceX yaml entry + user ci-config into a single execution config."""
    parsed = parse_model_entry(ifx_entry)
    model_hf = parsed["model_hf"]

    geak_ws = resolve_var(
        model_cfg.get("geak_workspace", defaults.get("geak_workspace", "")))
    min_k = model_cfg.get("min_kernels", defaults.get("min_kernels", 5))
    kern_backends = model_cfg.get(
        "kernel_opt_backends", defaults.get("kernel_opt_backends", "geak"))

    model_path = model_cfg.get("model_path_override") or f"/hyperloom/models/{model_hf.replace('/', '-')}"

    return {
        "model_hf": model_hf,
        "model_path": model_path,
        "image": parsed["image"],
        "sandbox_image": f"{harbor_prefix}/{parsed['image']}" if harbor_prefix else parsed["image"],
        "precision": parsed["precision"],
        "framework": parsed["framework"],
        "runner": parsed["runner"],
        "tp": parsed["tp"],
        "ep": model_cfg.get("ep", parsed["ep"]),
        "conc": parsed["conc_end"],
        "isl_osl_configs": parsed["isl_osl_configs"],
        "optimization_depth": model_cfg.get("optimization_depth", "full"),
        "geak_workspace": geak_ws,
        "geak_image": f"{harbor_prefix}/{parsed['image']}" if harbor_prefix else parsed["image"],
        "geak_step_limit": model_cfg.get("geak_step_limit", defaults.get("geak_step_limit", 100)),
        "kernel_opt_backends": kern_backends,
        "min_kernels": min_k,
        "target_gpu": model_cfg.get("target_gpu", "h200"),
        "mode": defaults.get("mode", "local"),
        "gpu_type": parsed["runner"].upper(),
        "inferencex_path": defaults.get("inferencex_path", "/hyperloom/InferenceX"),
        "result_dir": "/workspace/hyperloom",
        "inferenceX_benchmarks": ifx_benchmarks,
        "inferenceX_api_name": model_cfg.get("inferenceX_api_name", ""),
        "inferenceX_key": model_cfg.get("inferenceX_key", ""),
    }
