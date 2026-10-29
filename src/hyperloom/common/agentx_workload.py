# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an AgentX run drives: its client script, backend, and workload size.

Lives outside the ``agentx`` package because the default benchmark path is
pinned not to import it, and that path still has to recognise a client script
and size the benchmark cap.

Both backends size that cap from the work the run was composed to do: an MLPerf
run from its trajectory count, a canonical (Magpie/aiperf) run from the
concurrency it measures, because its warmup phase is one set of requests per
lane.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

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

# Server boot, aiter JIT rebuild and drain sit outside the measured window, for
# either backend.
AGENTX_BOOT_ALLOWANCE_SEC = 1800

# A first, cold round runs materially slower than the steady state (measured:
# 15 req/min against 48 once warm), so the cap is sized for the cold case.
MLPERF_TIMEOUT_SAFETY_FACTOR = 2.0

# The canonical phases below are Magpie's, not ours: Hyperloom composes only
# ``envs.CONC``, and the shape of the round that answers it belongs to Magpie at
# the ref native AgentX pins it to (docs/components/magpie.md). Every CANON_
# number here describes that pin and has to be re-measured when it moves.

# Canonical replay warms up this many requests per lane, plus one probe, before
# it opens the profiling window: a sweep measured 155, 221 and 309 warmup
# requests at CONC 14, 20 and 28, so the warmup phase grows with the
# concurrency being measured while the window that follows it does not.
CANON_WARMUP_PER_LANE = 11
CANON_WARMUP_PROBE_REQUESTS = 1

# Wall-clock one warmup request costs, sized to the slowest rung measured with
# about 6% over it. Measured across the rungs of one sweep on 8xMI355X: 6.0s at
# CONC 14, 18.8s at CONC 20 and 15.6s at CONC 28, so the cost does not fall off
# with concurrency and the slowest rung sets the rate. Three rungs on one host
# are a thin sample, and warmup is the cold part of the round: a slower rung
# than any of these moves this number.
CANON_SECONDS_PER_WARMUP_REQUEST = 20.0

# The profiling window canonical mode owns: ``duration_s`` 3600 inside a
# ``metric_window_s`` of 3640. It is a fixed window, independent of CONC.
CANON_PROFILING_WINDOW_SEC = 3640.0

# Writing the window's metrics out happens after it closes, and is part of the
# measurement: a round reaped 1.3s past a completed profiling phase threw away a
# finished measurement.
CANON_FINALISATION_SEC = 300.0


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
    conc = envs.get("CONC")
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
    return derived + AGENTX_BOOT_ALLOWANCE_SEC


def is_canonical_agentx(env: Mapping[str, str] | None = None) -> bool:
    """True for an AgentX run Magpie replays itself, rather than the MLPerf client."""
    from common.env import env_bool

    runtime = os.environ if env is None else env
    return env_bool("HYPERLOOM_AGENTX", env=runtime) and not is_mlperf_backend(runtime)


def agentx_concurrency(env: Mapping[str, str] | None = None) -> int:
    """The concurrency this invocation was composed for, or 0 when it pins none.

    ``CONC`` is the only workload size Hyperloom composes for a canonical run --
    the window and the per-lane warmup are the mode's own -- so an invocation
    without one carries no workload this layer can size a cap from.
    """
    runtime = os.environ if env is None else env
    raw = str(runtime.get("CONC") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 0


def agentx_warmup_requests(conc: int) -> int:
    """Requests canonical replay issues before it opens the profiling window."""
    return CANON_WARMUP_PER_LANE * conc + CANON_WARMUP_PROBE_REQUESTS


def canonical_benchmark_timeout_sec(env: Mapping[str, str] | None = None) -> float:
    """Benchmark cap a canonical AgentX round at this concurrency needs.

    The phases the grant has to cover are boot, a warmup that is linear in
    ``CONC``, the fixed profiling window and the finalisation that writes it out.
    Returns 0.0 for an invocation that pins no concurrency, which names no
    workload to size: the caller keeps its stock default.
    """
    conc = agentx_concurrency(env)
    if conc <= 0:
        return 0.0
    warmup = agentx_warmup_requests(conc) * CANON_SECONDS_PER_WARMUP_REQUEST
    return AGENTX_BOOT_ALLOWANCE_SEC + warmup + CANON_PROFILING_WINDOW_SEC + CANON_FINALISATION_SEC
