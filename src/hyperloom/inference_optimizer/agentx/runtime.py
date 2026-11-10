# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deploy the independent MLPerf client; Magpie owns native AgentX setup."""

from pathlib import Path
from typing import Mapping
import yaml

from hyperloom.common.agentx_workload import MLPERF_CLIENT_SCRIPT
from .deploy import deploy_agentx_assets
from .preflight import check_mlperf_harness


def maybe_prepare_agentx(*, env: Mapping[str, str], inferencex_path: str, config_path: str | Path) -> None:
    bench = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["benchmark"]
    if Path(str(bench.get("benchmark_script", ""))).name != MLPERF_CLIENT_SCRIPT:
        return
    deploy_agentx_assets(Path(inferencex_path) / "benchmarks")
    check_mlperf_harness(env)
