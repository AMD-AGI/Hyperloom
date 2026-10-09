# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover the exact native launch snapshot that produced an accepted measurement."""

from __future__ import annotations

import copy
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def native_launch_config(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Retain tokenized launch controls; display strings never become replay identity."""
    if measurement.get("agentx_launch_contract") != 1:
        return {}
    from hyperloom.inference_optimizer.agentx.identity import has_launch_contract, validate_server_launch

    path = str(measurement.get("materialized_config") or "")
    if not path:
        raise ValueError("Native KEEP requires the actual materialized candidate configuration")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    benchmark = config.get("benchmark") if isinstance(config, Mapping) else None
    if not isinstance(benchmark, Mapping) or not has_launch_contract(benchmark):
        raise ValueError("Native KEEP configuration lacks the optimization launch contract")
    errors = validate_server_launch(benchmark, measurement.get("agentx_server_launch"))
    if errors:
        raise ValueError(f"Native KEEP configuration does not match its measured launch: {', '.join(errors)}")
    snapshot = copy.deepcopy(benchmark["agentx"]["launch_overrides"])
    args = shlex.join(snapshot.get("append_args") or [])
    return {
        "native_launch_overrides": snapshot,
        "extra_server_args": args,
        "candidate_extra_server_args": args,
        "effective_extra_server_args": args,
        "extra_envs": dict(snapshot.get("env") or {}),
        "remove_args": list(snapshot.get("remove_args") or []),
        "unset_envs": list(snapshot.get("unset_env") or []),
        "args_mode": "replace" if snapshot.get("replace_args") else "append",
    }
