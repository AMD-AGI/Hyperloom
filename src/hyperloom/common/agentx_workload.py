# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an AgentX run drives: its client script, backend, and MLPerf workload size.

Lives outside the ``agentx`` package because the default benchmark path is
pinned not to import it, and that path still has to recognise a client script
and size the benchmark cap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.outside_facts import ENV_CONC

MLPERF_CLIENT_SCRIPT = "mlperf_agentic_client.sh"
AGENTX_CLIENT_SCRIPTS = frozenset({MLPERF_CLIENT_SCRIPT})

# Both client scripts write the pid of the server they boot here, under their RESULT_DIR. The server is setsid'd and
# re-parented away from the client, so reaping the round's process tree never reaches it.
AGENTX_SERVER_PIDFILE = "agentx_server.pid"

BACKEND_ENV = "HYPERLOOM_AGENTIC_BACKEND"

# The MLPerf agentic v6 corpus and the two flows ``run_agentic.sh`` accepts.
MLPERF_CORPUS = "agentic_combined_v6"
MLPERF_SMOKE_FLOW = "smoke_test"
MLPERF_SMOKE_TRAJECTORIES = 150
MLPERF_CANONICAL_TRAJECTORIES = 613

# ``run_agentic.sh`` dials this port and requests this served model name.
MLPERF_PORT = 30000
MLPERF_SERVED_MODEL = "kimi-k3"

# Wall-clock a single agentic trajectory costs, measured on 8xMI355X Kimi-K3 at
# concurrency 16: 25 trajectories in 629-666s and 150 in 3849-4046s, i.e. ~26s
# either way, so the workload scales linearly in trajectory count.
MLPERF_SECONDS_PER_TRAJECTORY = 26

# Server boot, aiter JIT rebuild and drain sit outside the measured window.
MLPERF_BOOT_ALLOWANCE_SEC = 1800

# A first, cold round runs materially slower than the steady state (measured:
# 15 req/min against 48 once warm), so the cap is sized for the cold case.
MLPERF_TIMEOUT_SAFETY_FACTOR = 2.0


def agentic_backend(env: Mapping[str, str] | None = None) -> str:
    """Return ``mlperf`` or ``magpie`` from ``HYPERLOOM_AGENTIC_BACKEND``."""
    runtime = os.environ if env is None else env
    return "mlperf" if str(runtime.get(BACKEND_ENV) or "").strip().lower() == "mlperf" else "magpie"


def is_mlperf_backend(env: Mapping[str, str] | None = None) -> bool:
    return agentic_backend(env) == "mlperf"


def pin_mlperf_round_concurrency(envs: dict[str, Any]) -> None:
    """Make ``AGENTIC_CONCURRENCY`` follow this round's ``CONC``.

    The client used to prefer ``AGENTIC_CONCURRENCY``, and the baseline YAML
    carries it. A conc-sweep rung that only changes ``CONC`` would then be
    recorded at one concurrency and measured at another.
    """
    if not is_mlperf_backend(envs):
        return
    conc = envs.get(ENV_CONC)
    if conc not in (None, ""):
        envs["AGENTIC_CONCURRENCY"] = str(conc)


def agentx_client_script(env: Mapping[str, str] | None = None) -> str:
    """The Magpie ``benchmark_script`` this AgentX backend pins."""
    if not is_mlperf_backend(env):
        raise ValueError("AgentX serving and replay are managed by Magpie; no Hyperloom client script exists")
    return MLPERF_CLIENT_SCRIPT


def is_agentx_client_script(name: str | None) -> bool:
    return Path(str(name or "")).name in AGENTX_CLIENT_SCRIPTS


def mlperf_flow(env: Mapping[str, str] | None = None) -> str:
    runtime = os.environ if env is None else env
    return str(runtime.get("MLPERF_AGENTIC_FLOW") or "").strip() or MLPERF_SMOKE_FLOW


def mlperf_trajectories(env: Mapping[str, str] | None = None) -> int:
    """Trajectory count the run issues: the explicit override, else the flow's own set."""
    runtime = os.environ if env is None else env
    raw = str(runtime.get("AGENTIC_NUM_TRAJECTORIES") or "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return MLPERF_SMOKE_TRAJECTORIES if mlperf_flow(runtime) == MLPERF_SMOKE_FLOW else MLPERF_CANONICAL_TRAJECTORIES


def mlperf_benchmark_timeout_sec(env: Mapping[str, str] | None = None) -> float:
    """Benchmark cap a run of this many trajectories needs, boot included."""
    derived = mlperf_trajectories(env) * MLPERF_SECONDS_PER_TRAJECTORY * MLPERF_TIMEOUT_SAFETY_FACTOR
    return derived + MLPERF_BOOT_ALLOWANCE_SEC
