# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write a benchmark config, materialize it with workload envs, and clear the envs that leak into it."""

from __future__ import annotations

import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as we

_WORKLOAD_ENV_KEYS = (
    "TP",
    "EP",
    "ISL",
    "OSL",
    "CONC",
    "MAX_MODEL_LEN",
    "PRECISION",
    "RANDOM_RANGE_RATIO",
    "ROCR_VISIBLE_DEVICES",
    "RUN_EVAL",
    "PROFILE",
    "MODEL_PATH",
    "INFERENCEX_PATH",
    "MYFW_REPO",
    "MYFW_REPO_PATH",
    "MYFW_DIR",
    "MYFW_REPO_URL",
    "FRAMEWORK_REPO_PATH",
    "MYFW_BENCH",
    "MYFW_ACTION_CKPT",
    "HYPERLOOM_PROFILE_MAX_ITERS",
    "HYPERLOOM_PROFILE_DELAY_ITERS",
    "HYPERLOOM_PROFILE_MAX_STEPS_CAP",
    "PROFILE_OSL",
    "INFERENCE_OPTIMIZER_VISIBLE_GPU_COUNT",
    "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP",
    "XDIT_QUALITY_REF",
    "XDIT_QUALITY_REF_WRITE",
    "HYPERLOOM_QUALITY_REF",
    "HYPERLOOM_QUALITY_REF_WRITE",
    "XDIT_MODEL_ROOT",
    "XDIT_ATTENTION_BACKEND",
)


def clear_env(monkeypatch):
    for k in _WORKLOAD_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def write_config(path, **bench_extra):
    bench = {"framework": "sglang", "model": "/m", "envs": {}}
    bench.update(bench_extra)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def materialize(src, out, **kw):
    res = we.materialize_config_with_envs(src, out, **kw)
    return yaml.safe_load(res.read_text())["benchmark"]
