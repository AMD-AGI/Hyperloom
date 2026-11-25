# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A minimal sglang benchmark config for tests that materialize a launch."""

from __future__ import annotations

from pathlib import Path

import yaml

QWEN3_8B = "/path/models/Qwen-Qwen3-8B"


def write_sglang_yaml(path: Path, model: str = QWEN3_8B) -> Path:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path
