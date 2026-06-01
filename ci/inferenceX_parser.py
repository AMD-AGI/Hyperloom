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


def _unmangle_msys_path(v: str) -> str:
    """Undo Git-Bash-on-Windows MSYS path conversion of /wekafs/* → C:/.../wekafs/*.

    Git Bash auto-converts POSIX paths starting with / into Windows-style paths
    when passing them as env vars or CLI args. This function detects the
    mangled form and reverses it. Safe no-op on Linux/macOS or unmangled paths.
    """
    if isinstance(v, str) and re.search(r"[A-Za-z]:[/\\].*[/\\]wekafs", v):
        v = re.sub(r"[A-Za-z]:[/\\].*[/\\](wekafs.*)", r"/\1", v).replace("\\", "/")
    return v


def get_nfs_root(default: str = "/wekafs") -> str:
    """Return $NFS_ROOT with Windows Git Bash path-mangling defense applied.

    All ci/* code MUST use this helper instead of os.environ.get("NFS_ROOT")
    directly, otherwise local dry-run on Windows produces malformed paths like
    `C:/Program Files/Git/wekafs/...` that don't match anything on the runner.
    """
    return _unmangle_msys_path(os.environ.get("NFS_ROOT", default))


def resolve_var(value: Any, env: dict | None = None) -> Any:
    """Resolve ${VAR} placeholders from environment."""
    if not isinstance(value, str):
        return value
    env = env or os.environ

    def _replace(m: re.Match) -> str:
        v = env.get(m.group(1), m.group(0))
        return _unmangle_msys_path(v) if isinstance(v, str) else v

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


def synthesize_entry_from_ci_config(model_cfg: dict) -> dict:
    """Build an amd-master.yaml-style entry from a self-contained ci-config entry.

    Used for Hyperloom-internal models that have no InferenceX baseline
    (e.g., GLM-5 multi-node MI300X — InferenceX only publishes MI355X). The
    returned dict matches the shape consumed by ``parse_model_entry()`` so the
    existing ``merge_model_config()`` flow works unchanged.

    Required fields in ``model_cfg``:
      - ``model_hf``           HF repo (e.g., ``zai-org/GLM-5``)
      - ``image``              container image tag (e.g., ``lmsysorg/sglang:v0.5.11-rocm720-mi30x``)
      - ``framework``          ``sglang`` or ``vllm``
      - ``precision``          ``fp8`` / ``fp4`` / ``bf16``
      - ``conc``               concurrency cap
      - ``isl_osl_configs``    list of ``[isl, osl]`` pairs
    Optional:
      - ``ep``                 default 1
      - ``target_gpu``         default ``mi300x``
      - ``tp``                 default 8

    Caller is responsible for setting ``key`` on the ci-config entry so the
    matrix filter and per-task identifier are unique.
    """
    isl_osl = model_cfg.get("isl_osl_configs") or [[1024, 1024]]
    return {
        "model": model_cfg.get("model_hf", ""),
        "image": model_cfg.get("image", ""),
        "model-prefix": (
            model_cfg.get("key", "").split("-")[0]
            if model_cfg.get("key") else ""
        ),
        "runner": model_cfg.get("target_gpu", "mi300x"),
        "precision": model_cfg.get("precision", ""),
        "framework": model_cfg.get("framework", "sglang"),
        "multinode": model_cfg.get("mode") == "remote",
        "scenarios": {
            "fixed-seq-len": [
                {
                    "isl": pair[0],
                    "osl": pair[1],
                    "search-space": [{
                        "tp": model_cfg.get("tp", 8),
                        "ep": model_cfg.get("ep", 1),
                        "conc-start": 4,
                        "conc-end": model_cfg.get("conc", 64),
                    }],
                }
                for pair in isl_osl
            ],
        },
    }


def parse_model_entry(entry: dict) -> dict:
    """Extract structured config from an amd-master.yaml model entry."""
    # Support both old format (seq-len-configs) and new format (scenarios.fixed-seq-len)
    seq_configs = (
        entry.get("seq-len-configs")
        or (entry.get("scenarios") or {}).get("fixed-seq-len")
        or []
    )
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
    tp: int | None = None,
    conc: int | None = None,
) -> dict | None:
    """Find the best benchmark entry matching hardware/ISL/OSL/precision.

    Applies progressive filtering with fallback:
      1. hardware + ISL + OSL + precision  (required)
      2. image   (prefer same image, fall back if none)
      3. tp      (prefer same TP via ``decode_tp``, fall back if none)
      4. conc    (prefer same concurrency, fall back if none)

    Among remaining candidates, returns the one with the highest
    ``output_tput_per_gpu`` (or ``tput_per_gpu`` as fallback).
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

    if tp is not None:
        same_tp = [b for b in candidates if b.get("decode_tp") == tp]
        if same_tp:
            candidates = same_tp

    if conc is not None:
        same_conc = [b for b in candidates if b.get("conc") == conc]
        if same_conc:
            candidates = same_conc

    def _sort_key(x):
        m = x.get("metrics") or {}
        return m.get("output_tput_per_gpu") or m.get("tput_per_gpu") or 0

    return max(candidates, key=_sort_key)


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
    tp: int | None = None,
    conc: int | None = None,
) -> str:
    """Format InferenceX benchmark data as text for the Claw prompt.

    API response nests performance data under a 'metrics' sub-object.
    """
    target = find_benchmark(benchmarks, target_gpu, isl, osl, precision, image,
                            tp=tp, conc=conc)
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

    kernel_opt_ws = resolve_var(
        model_cfg.get("kernel_opt_workspace", defaults.get("kernel_opt_workspace", "")))
    min_k = model_cfg.get("min_kernels", defaults.get("min_kernels", 5))
    kern_backends = model_cfg.get(
        "kernel_opt_backends", defaults.get("kernel_opt_backends", "geak"))

    nfs_root = get_nfs_root()
    model_path = model_cfg.get("model_path_override") or f"{nfs_root}/models/{model_hf.replace('/', '-')}"

    return {
        "model_hf": model_hf,
        "model_path": model_path,
        "image": parsed["image"],
        "sandbox_image": f"{harbor_prefix}/{parsed['image']}" if harbor_prefix else parsed["image"],
        "precision": parsed["precision"],
        "framework": parsed["framework"],
        "runner": parsed["runner"],
        "tp": model_cfg.get("tp", parsed["tp"]),
        "ep": model_cfg.get("ep", parsed["ep"]),
        "conc": parsed["conc_end"],
        "isl_osl_configs": parsed["isl_osl_configs"],
        "optimization_depth": model_cfg.get("optimization_depth", "full"),
        "kernel_opt_workspace": kernel_opt_ws,
        "kernel_opt_image": f"{harbor_prefix}/{parsed['image']}" if harbor_prefix else parsed["image"],
        "geak_step_limit": model_cfg.get("geak_step_limit", defaults.get("geak_step_limit", 100)),
        "kernel_opt_backends": kern_backends,
        "min_kernels": min_k,
        "target_gpu": model_cfg.get("target_gpu", "mi300x"),
        "mode": model_cfg.get("mode", defaults.get("mode", "claw")),
        "gpu_type": model_cfg.get("target_gpu", parsed["runner"]).upper(),
        "inferencex_path": defaults.get("inferencex_path") or (get_nfs_root() + "/InferenceX"),
        "oob_path": defaults.get("oob_path") or (get_nfs_root() + "/OOB"),
        "tracelens_root": defaults.get("tracelens_root") or (get_nfs_root() + "/TraceLens-internal"),
        "result_dir": defaults.get("result_dir", "/workspace/hyperloom"),
        "inferenceX_benchmarks": ifx_benchmarks,
        "inferenceX_api_name": model_cfg.get("inferenceX_api_name", ""),
        "inferenceX_key": model_cfg.get("inferenceX_key", ""),
        "rayjob_image": resolve_var(model_cfg.get("rayjob_image", "")),
        # ── Per-entry Claw pluginId override ──
        # Default behaviour (key absent in ci-config) → plugin_id=4 (legacy
        # Hyperloom plugin, used by all existing entries). To opt a specific
        # entry OUT of the plugin and have the agent talk to the Claw API
        # without a pluginId in the body, set:
        #     claw_plugin_id: null
        # in the ci-config entry. (claw_client.send_message already omits the
        # "pluginId" field from the JSON body when plugin_id is None.)
        # Other integer values (e.g. claw_plugin_id: 5) switch to a different
        # plugin — same hook used by the Inference A/B Test workflow via
        # --plugin-id CLI override.
        "claw_plugin_id": (
            model_cfg["claw_plugin_id"]
            if "claw_plugin_id" in model_cfg
            else 4
        ),
        # ── Hyperloom-skill knobs surfaced to prompt_template.md ──
        # `nodes` triggers the multinode Task-submission block when > 1.
        # `target_gain` / `max_hours` are forwarded as CLI flags to
        # `inference_optimizer optimize`. `random_range_ratio` controls
        # benchmark prompt length jitter (matches InferenceX default 0.8).
        # `kernel_agent_build_geak_rag_index` defaults off to skip the slow
        # GEAK RAG index rebuild on each cold-start. All five fall back to
        # legacy single-node defaults when absent in ci-config, so the 5
        # existing entries are unchanged.
        "nodes": model_cfg.get("nodes", 1),
        "target_gain": model_cfg.get("target_gain", defaults.get("target_gain", 10)),
        "max_hours": model_cfg.get("max_hours", defaults.get("max_hours", 2)),
        "random_range_ratio": model_cfg.get(
            "random_range_ratio", defaults.get("random_range_ratio", 0.8),
        ),
        "kernel_agent_build_geak_rag_index": model_cfg.get(
            "kernel_agent_build_geak_rag_index",
            defaults.get("kernel_agent_build_geak_rag_index", 0),
        ),
    }
