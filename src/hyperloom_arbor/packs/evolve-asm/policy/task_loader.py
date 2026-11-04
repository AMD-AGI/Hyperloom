"""Load the DSv4.1-Flash (and generic) evolve-asm task YAML into orchestrator configs.

The task YAML has a `kernels:` list — each kernel becomes its own independent
orchestrator instance. This loader:

  1. Parses the YAML (or JSON)
  2. Expands {gpu}, {M}, {N}, {K} template variables in bench_cmd / snr_cmd
  3. Produces one flat config dict per kernel, ready for EvolveOrchestrator
  4. Validates required fields early so errors surface before any GPU time is spent
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _expand(template: str, **values: Any) -> str:
    cmd = " ".join(template.split())
    for k, v in values.items():
        cmd = cmd.replace(f"{{{k}}}", str(v))
    return cmd


def _expand_bench_cmd(template: str, gpu: int | str, M: int, N: int, K: int) -> str:
    return _expand(template, gpu=gpu, M=M, N=N, K=K)


def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    with open(path) as f:
        return json.load(f)


def _kernel_config(campaign: dict, kernel: dict) -> dict:
    primary = kernel.get("shapes", {}).get("primary", {})
    scaling = kernel.get("shapes", {}).get("scaling", [])
    gpus    = kernel.get("gpus", [0])
    gpu0    = gpus[0] if gpus else 0

    M = primary.get("M", 1)
    N = primary.get("N", kernel.get("N", 0))
    K = primary.get("K", kernel.get("K", 0))

    bench_template = kernel.get("bench_cmd", "")
    snr_template   = kernel.get("snr_cmd", "")

    bench_cmd = _expand_bench_cmd(bench_template, gpu=gpu0, M=M, N=N, K=K)
    snr_cmd   = _expand_bench_cmd(snr_template,   gpu=gpu0, M=M, N=N, K=K)

    scaling_cmds = []
    for shape in scaling:
        sM   = shape.get("M", M)
        name = shape.get("config_name", f"M{sM}")
        scaling_cmds.append({
            "config_name": name,
            "M": sM, "N": N, "K": K,
            "bench_cmd": _expand_bench_cmd(bench_template, gpu=gpu0, M=sM, N=N, K=K),
            "snr_cmd":   _expand_bench_cmd(snr_template,   gpu=gpu0, M=sM, N=N, K=K),
        })

    return {
        "task_id":      kernel["task_id"],
        "campaign_id":  campaign.get("campaign_id", kernel["task_id"]),
        "container":   campaign.get("container", "evolve-dsv41flash"),
        "workspace":   campaign.get("workspace", "/workspace"),
        "kb_path":     campaign.get("kb_path",
                           "/workspace/knowledge_base/aiter/intellikit-asm-skills"),
        "gpus":        gpus,
        "n_islands":   len(gpus),
        "op_class":    kernel.get("op_class", "gemm"),
        "kernel_lang": kernel.get("kernel_lang", "flydsl"),
        "ref_co":      kernel.get("ref_co", ""),
        "snr_threshold": float(kernel.get("snr_threshold",
                                          campaign.get("snr_threshold", 30.0))),
        "n_bench_runs":  int(campaign.get("n_bench_runs", 3)),
        "bench_cmd": bench_cmd,
        "snr_cmd":   snr_cmd,
        "scaling_cmds": scaling_cmds,
        "primary_shape": primary,
        "deployment_path": kernel.get("deployment_path", ""),
        "deployment_csv":  kernel.get("deployment_csv", ""),
        "max_runtime_h": float(campaign.get("max_runtime_h", 168)),
        "e2e": campaign.get("e2e", {}),
    }


def load_task(path: str) -> list[dict]:
    campaign = _load_yaml(path)
    kernels = campaign.get("kernels", [])
    if not kernels:
        raise ValueError(f"No 'kernels:' list found in {path}")
    configs = []
    for k in kernels:
        if "task_id" not in k:
            raise ValueError(f"Kernel entry missing 'task_id': {k}")
        cfg = _kernel_config(campaign, k)
        _validate(cfg, path)
        configs.append(cfg)
    return configs


def _validate(cfg: dict, source: str) -> None:
    required = ["task_id", "container", "bench_cmd", "snr_cmd"]
    missing  = [f for f in required if not cfg.get(f)]
    if missing:
        raise ValueError(
            f"[{cfg.get('task_id', '?')}] Missing required fields {missing} in {source}"
        )
    for field in ("bench_cmd", "snr_cmd"):
        remaining = re.findall(r"\{(\w+)\}", cfg[field])
        if remaining:
            raise ValueError(
                f"[{cfg['task_id']}] Unexpanded template vars {remaining} in {field}."
            )
