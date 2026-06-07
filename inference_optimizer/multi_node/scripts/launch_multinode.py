#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node sglang / vllm server launcher.

Runs INSIDE the RayJob head pod, submitted via Ray Dashboard REST by
``inference_optimizer.multi_node restart-server`` when the workload has
``nodes >= 2``. Single-node restarts use the bash ``launch_server.sh``
path instead — the entry-point dispatch lives in cli.py.

Algorithm:

  1. ``ray.init()`` (no address; we are inside the cluster pod).
  2. Poll ``ray.nodes()`` until ``nnodes`` alive nodes with GPU>0 are
     visible (KubeRay can take a few seconds to register workers).
  3. Pick the node hosting THIS process as rank 0 (it is the head pod
     because the entry-point was submitted to the head's dashboard).
     Sort remaining nodes deterministically by NodeManagerAddress and
     assign ranks 1..N-1.
  4. For each rank K, spawn a ``@ray.remote`` actor pinned to that
     node via ``NodeAffinitySchedulingStrategy(node_id, soft=False)``.
  5. Inside each actor: start the framework launcher via ``bash`` with
     ``nohup`` and ``setsid`` (when available) so the server reparents
     away from the short-lived Ray task worker — avoids zombie PIDs and
     empty logs from an unreaped ``subprocess.Popen`` parent. The actor
     writes the detached child's PID to ``<pid_dir>/rank_<K>.pid``.
  6. (Optional) Wait for ``http://127.0.0.1:8888/health`` on rank 0
     to come up before returning. Cold MoE may exceed the budget; the
     caller can pass ``--no-wait-health`` to skip and probe externally
     via the ClusterIP service.

ADDENDUM-02 sanity: this file runs INSIDE the RayJob pod (sglang/vllm
image), not in the Claw sandbox. ``import ray`` is the standard, in-pod
way to talk to the local GCS — sandbox-side code (``cli.py`` etc.)
still must NOT import ray.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import pathlib
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Hard-coded inference port — matches SaFE Service.targetPort and the
# ClusterIP Service brain wires up. Changing it requires a coordinated
# brain + safe + cli edit, so we keep a single source of truth here.
# In `colocated` PD mode this port is bound by sglang/vllm rank 0
# directly; in `disaggregated` mode it is bound by the router which
# proxies to the internal prefill/decode ports below.
_INFERENCE_PORT = 8888
# Internal ports for sglang/vllm server groups when PD is disaggregated.
# Both are loopback-only on their respective head pods; the router on
# the cluster head pod fronts them at _INFERENCE_PORT for the magpie
# client, so the external service URL never changes between modes.
_PD_PREFILL_PORT = 30000
_PD_DECODE_PORT = 30001
# Default sglang collective port (resolution: $RAYJOB_DIST_INIT_PORT > 29500).
# Changed from 5000 → 29500 because RayJob head pod uses hostNetwork=true
# (amd-ray-job-template ConfigMap `dnsPolicy: ClusterFirstWithHostNet`),
# so the port lives in the host namespace; torch.distributed TCPStore
# leaves an orphan LISTEN socket on the host after a mid-init crash
# (no SO_REUSEADDR), and the next RayJob scheduled on the same host
# inherits EADDRINUSE → baseline_failed cascade. 29500 is the PyTorch
# convention. Operators override via $RAYJOB_DIST_INIT_PORT.
# PD disaggregated mode auto-derives decode port via
# ``_pd_decode_dist_init_port`` (= prefill + 1) so the two rendezvous
# endpoints never collide when both groups land on the same host
# (single source of truth: whatever prefill resolves to).
_DEFAULT_DIST_INIT_PORT = 29500


def _pd_decode_dist_init_port(prefill_dist_init_port: int) -> int:
    """Derive PD-disaggregated decode rendezvous port from prefill port.

    Returns ``prefill + 1`` so an operator override of
    ``$RAYJOB_DIST_INIT_PORT`` automatically shifts both endpoints in
    lock-step (no chance of decode silently colliding with prefill).
    """
    return prefill_dist_init_port + 1
# sglang PD bootstrap server port (KV transfer rendezvous). Default
# matches the sglang docs example. Override per-call via --pd-bootstrap-port.
_PD_DEFAULT_BOOTSTRAP_PORT = 8998
# Max seconds to wait for ray.nodes() to surface every expected pod.
_NODES_DISCOVERY_TIMEOUT_SEC = 120
# /health probe budget on rank 0. Cold MoE startup can exceed this; pass
# --no-wait-health to bypass and probe externally.
_HEALTH_PROBE_TIMEOUT_SEC = int(os.environ.get('SGLANG_HEALTH_PROBE_TIMEOUT_SEC', '1800'))


def _log(msg: str) -> None:
    """Stderr line with timestamp; no logging module to avoid handler
    surprises when this is exec'd as a Ray Dashboard entry-point."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _wait_for_nodes(target_n: int, timeout_s: int) -> list[dict]:
    """Poll ray.nodes() until ``target_n`` alive GPU nodes are visible."""
    started = time.monotonic()
    while True:
        nodes = [
            n for n in ray.nodes()
            if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0
        ]
        if len(nodes) >= target_n:
            return nodes
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            raise RuntimeError(
                f"only {len(nodes)}/{target_n} alive GPU nodes after {elapsed:.0f}s; "
                f"check KubeRay scheduler logs"
            )
        _log(f"waiting for nodes: have={len(nodes)} need={target_n} t={elapsed:.0f}s")
        time.sleep(3)


def _pick_head_first(nodes: list[dict]) -> list[dict]:
    """Reorder so the local node (which submitted this driver) is rank 0,
    and the rest follow in deterministic NodeManagerAddress order."""
    local_addr = ray.util.get_node_ip_address()
    local = [n for n in nodes if n.get("NodeManagerAddress") == local_addr]
    others = sorted(
        (n for n in nodes if n.get("NodeManagerAddress") != local_addr),
        key=lambda n: n.get("NodeManagerAddress", ""),
    )
    if not local:
        # Fallback: sort everything by address — driver is not on a GPU
        # node (rare but possible if KubeRay submitter pod is GPU-less).
        _log(f"WARN local node {local_addr!r} not in alive GPU set; sorting all")
        return sorted(nodes, key=lambda n: n.get("NodeManagerAddress", ""))
    return local + others


def _build_sglang_cmd(
    *,
    model: str,
    tp: int,
    nnodes: int,
    node_rank: int,
    dist_init_addr: str,
    extra_args: list[str],
    ep: int = 1,
    pd_role: str = "",
    pd_port: int = _INFERENCE_PORT,
    pd_transfer_backend: str = "",
    pd_ib_device: str = "",
    pd_bootstrap_port: int = _PD_DEFAULT_BOOTSTRAP_PORT,
) -> list[str]:
    """Compose the sglang multi-node launch command per upstream docs.

    Per https://sgl-project.github.io/references/multi_node_deployment/multi_node.html
    only the head (rank 0) needs ``--host`` / ``--port`` (the HTTP server
    only runs on rank 0). Worker ranks coordinate via the dist-init-addr
    and never serve HTTP — passing ``--port`` to a worker would cause it
    to bind a useless local socket. Match the upstream contract exactly.

    ``--trust-remote-code`` is added by default to mirror Magpie's
    single-node ``sglang_mi*x.sh`` launchers (which always set it).
    Custom modeling.py models (DSr1, future MoEs) require it; harmless
    for stock models. Callers can override via ``extra_args`` if needed
    — duplicate flags resolve to the last occurrence in argparse.

    ``ep`` (expert-parallel size, default 1) controls MoE expert
    distribution. ``ep <= 1``: omit any EP flag (legacy behaviour;
    experts shard along the TP dimension). ``ep > 1``: emit
    ``--expert-parallel-size N`` so each rank holds only
    ``n_experts / ep`` experts and the cluster does true expert
    parallelism (DSr1 / DSv3 best-practice on multi-node).

    NOTE: the older ``--enable-ep-moe --ep-size N`` flag pair was
    REMOVED in current sglang main; the canonical knob is now just
    ``--expert-parallel-size N`` (verified empirically against the
    sglang in the active RayJob image — earlier flag spelling caused
    ``launch_server.py: error: unrecognized arguments: --enable-ep-moe``
    and a hard restart loop). If you also want to switch the a2a
    backend, pass ``--moe-a2a-backend {deepep,mooncake,mori,...}``
    via ``extra_args``.
    """
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model,
        "--trust-remote-code",
        "--tp", str(tp),
        "--nnodes", str(nnodes),
        "--node-rank", str(node_rank),
        "--dist-init-addr", dist_init_addr,
    ]
    if node_rank == 0:
        # In PD-disaggregated mode the rank-0 HTTP port is the internal
        # prefill / decode port (proxied by the router); in colocated
        # mode it is the public inference port.
        cmd.extend(["--host", "0.0.0.0", "--port", str(pd_port)])
    if ep > 1:
        cmd.extend(["--expert-parallel-size", str(ep)])
    role = pd_role.strip().lower()
    if role in ("prefill", "decode"):
        # PD disaggregated: tell sglang which side of the split this
        # rank is, where the bootstrap rendezvous is, and which IB/RoCE
        # device to use for KV transfer. mooncake auto-detects when
        # ib_device is empty.
        cmd.extend(["--disaggregation-mode", role])
        cmd.extend(["--disaggregation-bootstrap-port", str(pd_bootstrap_port)])
        if pd_transfer_backend:
            cmd.extend(["--disaggregation-transfer-backend", pd_transfer_backend])
        if pd_ib_device:
            cmd.extend(["--disaggregation-ib-device", pd_ib_device])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _build_vllm_cmd(
    *,
    model: str,
    tp: int,
    extra_args: list[str],
    ep: int = 1,
    pd_role: str = "",
    pd_port: int = _INFERENCE_PORT,
    pd_transfer_backend: str = "",
    pd_kv_rank: int = 0,
    pd_kv_parallel_size: int = 1,
) -> list[str]:
    """vLLM multi-node command for rank 0.

    Worker ranks have no command — KubeRay already started ``ray start
    --address=<head>:6379`` in every worker pod's container, so the
    cluster is fully connected by the time we launch ``vllm serve`` on
    rank 0. vLLM auto-discovers the workers via the GCS and places its
    actors with ``--distributed-executor-backend ray``.

    ``--tensor-parallel-size`` = total GPUs across the cluster (caller
    passes ``nnodes * gpus_per_node``).

    ``ep`` (expert-parallel size, default 1) controls MoE expert
    distribution. ``ep <= 1``: omit (vllm default = TP-shard experts).
    ``ep > 1``: add ``--enable-expert-parallel`` (vllm 0.6+). vllm
    does not accept an ep-size value separately; the runtime infers
    ``ep_size = tp_size`` when the flag is set.
    """
    cmd = [
        "vllm", "serve", model,
        "--tensor-parallel-size", str(tp),
        "--host", "0.0.0.0",
        "--port", str(pd_port),
        "--distributed-executor-backend", "ray",
    ]
    if ep > 1:
        cmd.append("--enable-expert-parallel")
    role = pd_role.strip().lower()
    if role in ("prefill", "decode"):
        # vllm PD: the kv-transfer-config JSON encodes connector type,
        # the role (kv_producer = prefill, kv_consumer = decode), and
        # this rank's slot in the kv_parallel cluster. Connector default
        # is NixlConnector (recommended in vllm docs as of 2026 for
        # async push/pull). Buffer device fixed at cuda; backends omitted
        # so the connector picks UCX/UCC/NCCL based on the runtime.
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        connector = pd_transfer_backend or "NixlConnector"
        kv_cfg = (
            "{"
            f'"kv_connector":"{connector}",'
            f'"kv_role":"{kv_role}",'
            f'"kv_rank":{int(pd_kv_rank)},'
            f'"kv_parallel_size":{int(pd_kv_parallel_size)},'
            '"kv_buffer_device":"cuda"'
            "}"
        )
        cmd.extend(["--kv-transfer-config", kv_cfg])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _probe_mec_firmware_lt_177() -> bool:
    """Return True iff rocm-smi reports MEC firmware version < 177.

    Mirrors the gate Magpie's ``sglang_mi300x.sh`` uses to decide
    whether ``HSA_NO_SCRATCH_RECLAIM=1`` is required for RCCL memory
    reclaim correctness. Best-effort: any failure (rocm-smi missing,
    parse error, non-MI300 GPU) returns ``False`` so we don't apply
    the workaround when we can't confirm we need it.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showfw"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    for line in (proc.stdout or "").splitlines():
        if "MEC" not in line:
            continue
        token = line.split()[-1].strip() if line.split() else ""
        try:
            return int(token) < 177
        except ValueError:
            return False
    return False


def _subprocess_env() -> dict[str, str]:
    """Build the env passed to the framework launcher subprocess.

    Force ``/opt/venv/bin`` to the front of ``PATH`` so the spawned
    ``python3 -m sglang.launch_server`` / ``vllm serve`` resolves to
    the framework's venv interpreter (which has sglang/vllm/ray
    installed) regardless of what the actor process's PATH looks like.

    Inheriting ``os.environ`` keeps the *_API_KEY / *_BASE_URL and any
    SGLANG_* / VLLM_* tunings the bootstrap layer (or KubeRay) injected.

    MI300X tuning trio (mirrors Magpie's ``sglang_mi300x.sh``):
      * ``SGLANG_USE_AITER=1`` — aiter kernels (large tput uplift on
        MI300X; default-off in stock sglang)
      * ``SGLANG_AITER_MLA_PERSIST=1`` — persist MLA workspace across
        decode calls (DSr1 / DSv3 workloads)
      * ``HSA_NO_SCRATCH_RECLAIM=1`` — only when MEC firmware < 177
        (older firmware leaks scratch on RCCL reclaim)

    Without these, multi-node tput is not comparable to single-node;
    `--no-clobber` semantics: don't overwrite if caller already set
    them via inherited env (KubeRay env injection wins).
    """
    env = dict(os.environ)
    # Ray sets *_VISIBLE_DEVICES='' on actors spawned with num_gpus=0; that
    # empty string masks ALL physical GPUs for the detached framework child,
    # which then dies with "No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS)
    # is available." Strip the mask so the framework re-discovers all 8 GPUs
    # on the pod (it manages its own per-rank pinning via TP/EP). Honour an
    # explicit non-empty override (set deliberately by the caller).
    for _vis in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if _vis in env and env[_vis].strip() == "":
            env.pop(_vis, None)
    # Prevent sglang crash with fused decode MLA on MI300X (aiter
    # ForwardMetadata mismatch). Restored after an earlier edit dropped
    # this line while adding the ROCR_VISIBLE_DEVICES cleanup above.
    env["SGLANG_ROCM_FUSED_DECODE_MLA"] = "0"
    env.setdefault("SGLANG_USE_AITER", "1")
    env.setdefault("SGLANG_AITER_MLA_PERSIST", "1")
    if "HSA_NO_SCRATCH_RECLAIM" not in env and _probe_mec_firmware_lt_177():
        env["HSA_NO_SCRATCH_RECLAIM"] = "1"
    venv_bin = "/opt/venv/bin"
    cur_path = env.get("PATH", "")
    parts = cur_path.split(":") if cur_path else []
    if venv_bin not in parts:
        env["PATH"] = f"{venv_bin}:{cur_path}" if cur_path else venv_bin
    return env


def _detach_framework_launch(
    cmd: list[str],
    log_file: Path,
    pid_file: Path,
    sub_env: dict[str, str],
    node_rank: int,
) -> int:
    """Start ``cmd`` detached from the Ray worker via bash+nohup.

    A plain ``subprocess.Popen`` without ``wait()`` leaves the Ray worker
    as parent of a dead child (zombie) and can leave ``rank_*.log`` empty
    due to stdio block-buffering. This path reparents the server under
    init, enables line-oriented logs with ``PYTHONUNBUFFERED=1``, and
    fails fast with log tail if the child dies immediately.
    """
    sub_env = dict(sub_env)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")
    log_q = shlex.quote(str(log_file))
    pid_q = shlex.quote(str(pid_file))
    inner = " ".join(shlex.quote(c) for c in cmd)
    # ``setsid`` gives a fresh session so ``kill_multinode`` can SIGTERM
    # the whole process group; fall back to plain ``nohup`` if missing.
    launches = (
        f"if command -v setsid >/dev/null 2>&1; then "
        f"nohup setsid {inner} >>{log_q} 2>&1 & "
        f"else nohup {inner} >>{log_q} 2>&1 & fi; "
        f"echo $! > {pid_q}"
    )
    shell_cmd = f"set -euo pipefail; : >{log_q}; {launches}"
    proc = subprocess.run(
        ["/bin/bash", "-lc", shell_cmd],
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        msg = (
            f"[rank {node_rank}] detach spawn shell failed rc={proc.returncode} "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
        sys.stderr.write(msg + "\n")
        raise RuntimeError(msg)
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"[rank {node_rank}] missing or invalid pid file {pid_file}: {exc}"
        ) from exc

    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        tail = ""
        try:
            if log_file.is_file() and log_file.stat().st_size > 0:
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            tail = "<could not read log>"
        sys.stderr.write(
            f"[rank {node_rank}] child pid={pid} not alive after 0.5s: {exc}\n"
            f"[rank {node_rank}] log tail:\n{tail}\n",
        )
        raise RuntimeError(
            f"[rank {node_rank}] framework exited immediately (pid={pid}); "
            f"see log tail on stderr and {log_file}"
        ) from exc
    return pid


def _spawn_remote(
    *,
    framework: str,
    model: str,
    tp: int,
    nnodes: int,
    node_rank: int,
    head_ip: str,
    dist_init_port: int,
    pid_dir: str,
    log_dir: str,
    extra_args: list[str],
    torch_profiler_dir: str = "",
    ep: int = 1,
    pd_role: str = "",
    pd_port: int = _INFERENCE_PORT,
    pd_transfer_backend: str = "",
    pd_ib_device: str = "",
    pd_bootstrap_port: int = _PD_DEFAULT_BOOTSTRAP_PORT,
    pd_kv_rank: int = 0,
    pd_kv_parallel_size: int = 1,
    pid_file_name: str = "",
) -> int:
    """Spawn the framework launcher detached on the local actor's pod.

    Writes the real server PID to ``<pid_dir>/rank_<K>.pid``. Designed for
    ``@ray.remote`` invocation — inputs are all picklable primitives.

    ``torch_profiler_dir`` (multi-node only) — when non-empty, exported
    as ``SGLANG_TORCH_PROFILER_DIR`` so the rank's sglang server emits
    torch traces to a sandbox-readable shared path (typically wekafs).
    Empty string preserves legacy per-pod /tmp behaviour.
    """
    Path(pid_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # PD-disaggregated mode launches two server groups on potentially
    # overlapping rank numbers (prefill ranks 0..P-1, decode ranks
    # 0..D-1 *within their own group*). Caller passes a role-tagged
    # pid_file_name so they don't collide on disk.
    fname = pid_file_name or f"rank_{node_rank}.pid"
    log_fname = (
        f"{Path(fname).stem}.log" if pid_file_name else f"rank_{node_rank}.log"
    )
    pid_file = Path(pid_dir) / fname
    log_file = Path(log_dir) / log_fname

    sub_env = _subprocess_env()
    # Resume-aware fallback: when the orchestrator (profile_executor) reuses
    # an already-running sglang server (resume path in multi_node cli.py),
    # this LAUNCH never re-runs, so any later round-scoped torch_profiler_dir
    # has no chance of reaching sglang's env. Falling back to
    # $HYPERLOOM_MN_PROFILE_TRACE_DIR (cli.py exports this as
    # `/wekafs/.../<rayjob>/torch_trace`, a single base shared by every
    # profile round) lets sglang write trace.json.gz to that shared dir
    # on its FIRST launch, then profile_executor mtime-filters per-round.
    tpd = (
        (torch_profiler_dir or "").strip()
        or os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    )
    if tpd:
        # Pin sglang torch profiler output to a shared dir; mkdir is
        # safe across racing ranks (exist_ok). Failure to mkdir is
        # non-fatal: the server may still bring up; sglang itself will
        # raise on the first profile request, which is the right place
        # for a loud, attributable error.
        try:
            Path(tpd).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            sys.stderr.write(
                f"[rank {node_rank}] WARN cannot mkdir torch profiler dir "
                f"{tpd!r}: {exc}; sglang will fall back to /tmp\n"
            )
        else:
            sub_env["SGLANG_TORCH_PROFILER_DIR"] = tpd

    dist_init_addr = f"{head_ip}:{dist_init_port}"
    fw = framework.lower()
    if fw == "sglang":
        cmd = _build_sglang_cmd(
            model=model, tp=tp, nnodes=nnodes, node_rank=node_rank,
            dist_init_addr=dist_init_addr, extra_args=extra_args, ep=ep,
            pd_role=pd_role, pd_port=pd_port,
            pd_transfer_backend=pd_transfer_backend,
            pd_ib_device=pd_ib_device,
            pd_bootstrap_port=pd_bootstrap_port,
        )
    elif fw == "vllm":
        # vLLM multi-node: KubeRay already wired every worker pod into
        # the GCS via its auto-generated ``ray start --address=<head>:6379``
        # container command. Worker actors here do NOTHING — vLLM on rank
        # 0 discovers the workers through the existing ray cluster and
        # uses ``--distributed-executor-backend ray`` to place its
        # workers. Trying to ``ray start`` again here would either fail
        # ("ray is already running") or spawn a duplicate worker process
        # that confuses GCS scheduling.
        if node_rank != 0:
            sys.stderr.write(
                f"[rank {node_rank}] vllm worker: no-op (KubeRay already "
                f"joined this node to the ray cluster; rank 0 vllm serve "
                f"will discover us via GCS)\n"
            )
            # Write a sentinel PID so kill_multinode finds *something*
            # to clean up; PID 0 is never alive so kill_remote treats it
            # as stale and just removes the file.
            pid_file.write_text("0")
            return 0
        cmd = _build_vllm_cmd(
            model=model, tp=tp, extra_args=extra_args, ep=ep,
            pd_role=pd_role, pd_port=pd_port,
            pd_transfer_backend=pd_transfer_backend,
            pd_kv_rank=pd_kv_rank,
            pd_kv_parallel_size=pd_kv_parallel_size,
        )
    else:
        raise RuntimeError(f"unsupported framework: {framework!r}")

    sys.stderr.write(f"[rank {node_rank}] launching: {' '.join(cmd)}\n")
    sys.stderr.write(f"[rank {node_rank}] log={log_file} pid={pid_file}\n")
    return _detach_framework_launch(cmd, log_file, pid_file, sub_env, node_rank)


def _rank0_pid_from_log(log_dir: str) -> int | None:
    """Read rank 0 PID from /tmp/multi_node_pids/rank_0.pid (best-effort)."""
    try:
        pid_path = pathlib.Path(log_dir).parent / "multi_node_pids" / "rank_0.pid"
        if not pid_path.is_file():
            pid_path = pathlib.Path("/tmp/multi_node_pids/rank_0.pid")
        return int(pid_path.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


# Fatal patterns to scan rank_0.log for. Triggered even when the rank-0
# wrapper PID is still alive (e.g. nohup lingers after the python sglang
# child has traceback-exited). Without this scan, a real framework crash
# such as `KeyError: 'glm_moe_dsa'` (sglang transformers version too old
# for GLM-4.5/4.6 MoE-DSA) would silently consume the full 1800s /health
# budget before the Ray Dashboard job flips to FAILED.
# Patterns are anchored to line start to reduce false positives from
# benign mentions inside JSON / tracebacks of unrelated frames.
_FATAL_LOG_PATTERNS: tuple[str, ...] = (
    "Traceback (most recent call last):",
    "KeyError:",
    "ValueError:",
    "RuntimeError:",
    "AssertionError:",
    "TypeError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "FileNotFoundError:",
    "OSError:",
    "AttributeError:",
    "torch.cuda.OutOfMemoryError",
    "CUDA out of memory",
    "HIP out of memory",
    "RCCL error",
    "NCCL error",
    "CUDA error",
    "HSA_STATUS_ERROR",
    "abort()",
    "Segmentation fault",
)
# How far back from end-of-file we scan. Large enough to catch a full
# Python traceback (typically <8KB) but bounded to keep this cheap.
_FATAL_SCAN_TAIL_BYTES = 256 * 1024


def _scan_rank0_log_for_fatal(log_dir: str) -> str | None:
    """Scan rank_0.log tail for a fatal traceback / framework error.

    Returns the matched line (stripped) on hit, ``None`` otherwise.
    Used by ``_wait_health`` to bail out of the 1800s health wait the
    moment sglang / vllm has clearly crashed, and by ``main()`` after a
    health timeout to distinguish "still loading weights" from a silent
    framework error masked by a lingering wrapper PID.
    """
    try:
        lf = Path(log_dir) / "rank_0.log"
        if not lf.is_file():
            return None
        size = lf.stat().st_size
        if size == 0:
            return None
        with lf.open("rb") as f:
            if size > _FATAL_SCAN_TAIL_BYTES:
                f.seek(size - _FATAL_SCAN_TAIL_BYTES)
                f.readline()  # discard partial line
            tail = f.read().decode("utf-8", errors="replace")
        for raw in tail.splitlines():
            line = raw.strip()
            if not line:
                continue
            for pat in _FATAL_LOG_PATTERNS:
                if pat in line:
                    return line
        return None
    except OSError:
        return None


def _wait_health(
    timeout_s: int,
    rank0_pid: int | None = None,
    log_dir: str | None = None,
) -> bool:
    """Poll http://127.0.0.1:8888/health on rank 0 (this pod). Returns
    True on first 200, False on timeout / rank 0 process death / fatal
    error detected in ``rank_0.log``."""
    import urllib.request
    import urllib.error
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{_INFERENCE_PORT}/health", timeout=3,
            ) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        # Bail early if rank 0 process died (server crash during cuda graph
        # capture, OOM, MoE assertion, etc.). Without this we wait the full
        # 1800s for /health on a corpse.
        if rank0_pid is not None and rank0_pid > 0:
            try:
                os.kill(rank0_pid, 0)
            except ProcessLookupError:
                _log(f"ERROR rank 0 pid={rank0_pid} died while waiting for /health; "
                     f"aborting health wait")
                return False
            except OSError:
                pass  # permission etc; don't kill the wait
        # Bail early on fatal traceback in rank_0.log. The wrapper PID
        # (nohup) often outlives the actual python framework child, so
        # os.kill above misses the crash. Scanning the log catches
        # framework errors (e.g. KeyError on unsupported model_type)
        # in seconds rather than burning the full 1800s budget.
        if log_dir:
            fatal_line = _scan_rank0_log_for_fatal(log_dir)
            if fatal_line:
                _log(f"ERROR rank 0 fatal in rank_0.log: {fatal_line[:300]}; "
                     f"aborting health wait (was: silent 1800s stall)")
                return False
        time.sleep(5)
    return False


def _emit_rank0_log_tail(log_dir: Path) -> None:
    """Append the tail of ``rank_0.log`` to stderr if the file exists."""
    lf = log_dir / "rank_0.log"
    try:
        sz = lf.stat().st_size if lf.is_file() else 0
        _log(f"(tail probe) rank_0.log bytes={sz} path={lf}")
        if sz > 0:
            _log(
                "rank_0.log tail (last 8kiB):\n"
                + lf.read_text(encoding="utf-8", errors="replace")[-8192:]
            )
    except OSError as exc:
        _log(f"(tail probe) cannot read rank_0.log: {exc}")


def _log_rank0_post_spawn(log_dir: Path, rank0_pid: int | None) -> None:
    """Emit diagnostics for rank 0 after a short settle (driver runs on head).

    If the framework exits during weight load, ``/health`` never flips;
    this log gives operators an immediate ``rank_0.log`` tail without
    exec'ing into the pod first.
    """
    if rank0_pid is None or rank0_pid <= 0:
        return
    time.sleep(3)
    try:
        os.kill(rank0_pid, 0)
        _log(f"post-spawn+3s rank0 pid={rank0_pid} still alive")
    except OSError as exc:
        _log(f"ERROR post-spawn+3s rank0 pid={rank0_pid} not alive: {exc}")
    _emit_rank0_log_tail(log_dir)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="launch_multinode.py",
        description="Spawn one sglang/vllm rank per RayJob node via ray actors.",
    )
    p.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--nnodes", type=int, required=True)
    p.add_argument("--pid-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument(
        "--dist-init-port", type=int,
        default=int(os.environ.get("RAYJOB_DIST_INIT_PORT") or _DEFAULT_DIST_INIT_PORT),
        help=f"sglang collective rendezvous port (default {_DEFAULT_DIST_INIT_PORT}, "
             f"resolution: --dist-init-port > $RAYJOB_DIST_INIT_PORT > {_DEFAULT_DIST_INIT_PORT})",
    )
    p.add_argument("--no-wait-health", action="store_true",
                   help="don't poll /health on rank 0 before returning")
    p.add_argument("--extra-args", default="",
                   help="extra args appended verbatim to the framework launcher")
    p.add_argument("--torch-profiler-dir", default="",
                   help="when set, exported as SGLANG_TORCH_PROFILER_DIR on every "
                        "rank's framework subprocess; intended for a wekafs path "
                        "shared with the sandbox (see HYPERLOOM_MN_PROFILE_TRACE_DIR)")
    p.add_argument("--ep", type=int, default=1,
                   help="expert-parallel size; 1 (default) keeps experts "
                        "TP-sharded (legacy). >=2 emits sglang "
                        "`--enable-ep-moe --ep-size N` or vllm "
                        "`--enable-expert-parallel`. Caller (orchestrator "
                        "helper) is responsible for ensuring ep <= tp.")
    # PD disaggregation arguments. Default `colocated` keeps the legacy
    # single-server-group behaviour (rank 0..N-1 form one TP=tp group).
    # `disaggregated` splits nodes[0:pd_prefill_nodes] into a prefill
    # group (TP=pd_prefill_tp) and nodes[pd_prefill_nodes:] into a
    # decode group (TP=pd_decode_tp). Neither group binds the public
    # 8888 port; the router (launched separately by launch_router.py)
    # does that and proxies to the internal 30000/30001 ports.
    p.add_argument("--pd-mode", choices=("colocated", "disaggregated"),
                   default="colocated",
                   help="PD disaggregation mode (default colocated)")
    p.add_argument("--pd-prefill-nodes", type=int, default=0,
                   help="number of prefill nodes (disaggregated only); "
                        "must satisfy pd_prefill_nodes + pd_decode_nodes == nnodes")
    p.add_argument("--pd-decode-nodes", type=int, default=0,
                   help="number of decode nodes (disaggregated only)")
    p.add_argument("--pd-prefill-tp", type=int, default=0,
                   help="TP size for the prefill group (disaggregated only); "
                        "default = --tp")
    p.add_argument("--pd-decode-tp", type=int, default=0,
                   help="TP size for the decode group (disaggregated only); "
                        "default = --tp")
    p.add_argument("--pd-transfer-backend", default="",
                   help="sglang: mooncake|nixl ; vllm: NixlConnector|"
                        "P2pNcclConnector|MooncakeConnector|LMCacheConnectorV1; "
                        "empty = framework default (sglang mooncake / vllm NixlConnector)")
    p.add_argument("--pd-ib-device", default="",
                   help="comma-separated IB/RoCE device list for KV transfer "
                        "(e.g. mlx5_0,mlx5_1). Empty = read $NCCL_IB_HCA "
                        "from this pod's env (RayJob image typically injects "
                        "it); if that's also empty, mooncake auto-detects.")
    p.add_argument("--pd-bootstrap-port", type=int,
                   default=_PD_DEFAULT_BOOTSTRAP_PORT,
                   help=f"sglang PD bootstrap rendezvous port "
                        f"(default {_PD_DEFAULT_BOOTSTRAP_PORT})")
    args = p.parse_args()

    if args.nnodes < 2:
        _log(f"nnodes={args.nnodes} < 2; this script is for multi-node only. "
             f"Use launch_server.sh for single-pod restarts.")
        return 2

    # Validate PD args; populate defaults.
    pd_mode = (args.pd_mode or "colocated").lower()
    if pd_mode == "disaggregated":
        pn = int(args.pd_prefill_nodes or 0)
        dn = int(args.pd_decode_nodes or 0)
        if pn <= 0 or dn <= 0 or pn + dn != args.nnodes:
            _log(f"PD invalid split: pd_prefill_nodes={pn} pd_decode_nodes={dn} "
                 f"nnodes={args.nnodes}; require pn+dn==nnodes and both >0")
            return 2
        ptp = int(args.pd_prefill_tp or args.tp)
        dtp = int(args.pd_decode_tp or args.tp)
        if ptp <= 0 or dtp <= 0:
            _log(f"PD invalid TP: pd_prefill_tp={ptp} pd_decode_tp={dtp}")
            return 2
        # IB device default: pod's $NCCL_IB_HCA (injected by image).
        ib_dev = args.pd_ib_device or os.environ.get("NCCL_IB_HCA", "")
        ib_dev = ib_dev.strip()
    else:
        pn = args.nnodes
        dn = 0
        ptp = args.tp
        dtp = 0
        ib_dev = ""

    extra_args = args.extra_args.split() if args.extra_args else []

    _log(f"framework={args.framework} model={args.model} tp={args.tp} nnodes={args.nnodes}")

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = _wait_for_nodes(args.nnodes, _NODES_DISCOVERY_TIMEOUT_SEC)
    nodes = _pick_head_first(nodes)
    nodes = nodes[:args.nnodes]
    head_ip = ray.util.get_node_ip_address()
    _log(f"discovered {len(nodes)} GPU nodes; head_ip={head_ip}; rank order:")
    for k, n in enumerate(nodes):
        _log(f"  rank {k}: node_id={n.get('NodeID', '?')[:16]}... "
             f"addr={n.get('NodeManagerAddress', '?')} "
             f"gpu={n.get('Resources', {}).get('GPU', 0)}")

    # Spawn one actor per rank. num_gpus=0 because the framework
    # process itself reserves GPUs via its own mechanisms; pinning
    # the actor takes 0 GPU resource so it doesn't conflict with the
    # forked process's hold.
    SpawnActor = ray.remote(num_cpus=1, num_gpus=0)(_spawn_remote)
    pids: dict[str, int] = {}     # key = pid_file_name (e.g. "rank_0.pid"
                                   # or "prefill_0.pid"); value = real PID.
    refs: list[tuple[str, Any]] = []  # noqa: F821  Any imported below

    if pd_mode == "disaggregated":
        # Group A: prefill — nodes[0:pn] form an inner sglang/vllm group
        # of nnodes=pn, TP=ptp; rank-0 (= physical head pod) binds the
        # internal prefill HTTP port.
        prefill_head_ip = nodes[0].get("NodeManagerAddress", head_ip)
        for grp_rank in range(pn):
            node = nodes[grp_rank]
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False,
                ),
            ).remote(
                framework=args.framework, model=args.model,
                tp=ptp, nnodes=pn, node_rank=grp_rank,
                head_ip=prefill_head_ip,
                dist_init_port=args.dist_init_port,
                pid_dir=args.pid_dir, log_dir=args.log_dir,
                extra_args=extra_args,
                torch_profiler_dir=args.torch_profiler_dir,
                ep=int(args.ep or 1),
                pd_role="prefill",
                pd_port=_PD_PREFILL_PORT,
                pd_transfer_backend=args.pd_transfer_backend,
                pd_ib_device=ib_dev,
                pd_bootstrap_port=args.pd_bootstrap_port,
                pd_kv_rank=0,        # vllm-only; sglang ignores
                pd_kv_parallel_size=2,
                pid_file_name=f"prefill_{grp_rank}.pid",
            )
            refs.append((f"prefill_{grp_rank}", actor_ref))

        # Group B: decode — nodes[pn:pn+dn]; uses prefill_port + 1 as
        # its dist-init port so it never clashes with prefill rendezvous
        # when both happen to share a node, regardless of whether the
        # operator overrode the prefill port via $RAYJOB_DIST_INIT_PORT.
        decode_head_ip = nodes[pn].get("NodeManagerAddress", head_ip)
        for grp_rank in range(dn):
            node = nodes[pn + grp_rank]
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False,
                ),
            ).remote(
                framework=args.framework, model=args.model,
                tp=dtp, nnodes=dn, node_rank=grp_rank,
                head_ip=decode_head_ip,
                dist_init_port=_pd_decode_dist_init_port(args.dist_init_port),
                pid_dir=args.pid_dir, log_dir=args.log_dir,
                extra_args=extra_args,
                torch_profiler_dir=args.torch_profiler_dir,
                ep=int(args.ep or 1),
                pd_role="decode",
                pd_port=_PD_DECODE_PORT,
                pd_transfer_backend=args.pd_transfer_backend,
                pd_ib_device=ib_dev,
                pd_bootstrap_port=args.pd_bootstrap_port,
                pd_kv_rank=1,
                pd_kv_parallel_size=2,
                pid_file_name=f"decode_{grp_rank}.pid",
            )
            refs.append((f"decode_{grp_rank}", actor_ref))
    else:
        # Colocated (legacy): single server group spans all nodes.
        for rank, node in enumerate(nodes):
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False,
                ),
            ).remote(
                framework=args.framework, model=args.model,
                tp=args.tp, nnodes=args.nnodes, node_rank=rank,
                head_ip=head_ip,
                dist_init_port=args.dist_init_port,
                pid_dir=args.pid_dir, log_dir=args.log_dir,
                extra_args=extra_args,
                torch_profiler_dir=args.torch_profiler_dir,
                ep=int(args.ep or 1),
            )
            refs.append((f"rank_{rank}", actor_ref))

    for tag, ref in refs:
        try:
            pid = ray.get(ref, timeout=120)
            pids[tag] = pid
            _log(f"{tag}: spawned pid={pid}")
        except Exception as exc:  # noqa: BLE001
            _log(f"{tag}: spawn FAILED: {type(exc).__name__}: {exc}")
            # Roll back already-spawned ranks so the cluster doesn't
            # leak half-started servers between attempts.
            for tag2, p2 in pids.items():
                _log(f"rolling back {tag2} pid={p2}")
                try:
                    os.killpg(os.getpgid(p2), 15)
                except (ProcessLookupError, PermissionError):
                    pass
            return 1

    # Health-tail probe targets the first launched leader. In colocated
    # mode that's `rank_0`; in PD it's `prefill_0` (the prefill head).
    leader_tag = "rank_0" if pd_mode == "colocated" else "prefill_0"
    _log_rank0_post_spawn(Path(args.log_dir), pids.get(leader_tag))

    summary: dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "tp": args.tp,
        "ep": int(args.ep or 1),
        "nnodes": args.nnodes,
        "head_ip": head_ip,
        "dist_init_port": args.dist_init_port,
        "ranks": [{"tag": r, "pid": pids[r]} for r in sorted(pids)],
        "pid_dir": args.pid_dir,
        "log_dir": args.log_dir,
        "inference_port": _INFERENCE_PORT,
        "pd_mode": pd_mode,
    }
    if pd_mode == "disaggregated":
        # The router (launched separately by launch_router.py) needs the
        # internal prefill/decode endpoints and the bootstrap port. Emit
        # them here so the multi_node CLI can read this JSON and submit
        # the router entrypoint without re-discovering nodes.
        summary["pd_prefill_nodes"] = pn
        summary["pd_decode_nodes"] = dn
        summary["pd_prefill_tp"] = ptp
        summary["pd_decode_tp"] = dtp
        summary["pd_transfer_backend"] = (
            args.pd_transfer_backend
            or ("NixlConnector" if args.framework.lower() == "vllm" else "mooncake")
        )
        summary["pd_ib_device"] = ib_dev
        summary["pd_bootstrap_port"] = args.pd_bootstrap_port
        summary["pd_prefill_url"] = (
            f"http://{prefill_head_ip}:{_PD_PREFILL_PORT}"
        )
        summary["pd_decode_url"] = (
            f"http://{decode_head_ip}:{_PD_DECODE_PORT}"
        )
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()

    if args.no_wait_health:
        _log("--no-wait-health set; not probing /health")
        return 0

    # In PD mode the public 8888 is bound by the router (separate
    # entrypoint), not by any rank here. Skip the local /health probe;
    # the caller polls externally once the router is up.
    if pd_mode == "disaggregated":
        _log("pd_mode=disaggregated; /health probe skipped (router owns 8888)")
        return 0

    _log(f"polling rank 0 /health for up to {_HEALTH_PROBE_TIMEOUT_SEC}s")
    _r0_pid = _rank0_pid_from_log(args.log_dir)
    if _wait_health(
        _HEALTH_PROBE_TIMEOUT_SEC, rank0_pid=_r0_pid, log_dir=args.log_dir,
    ):
        _log("rank 0 /health OK")
        return 0
    # _wait_health returned False on either /health timeout OR early rank-0
    # process death. Tail the log so operators see the framework's last
    # words regardless of which path we took.
    _emit_rank0_log_tail(Path(args.log_dir))
    # Distinguish the two failure modes so the upstream caller (hyperloom
    # `_multi_node_server_lifecycle`) reacts correctly:
    #   * rank-0 process died -> framework refused to start (argparse
    #     error on an unknown flag, aiter kernel ABI assert, OOM during
    #     cuda graph capture, RCCL init, ...). Caller's 1800s /health
    #     wait would just stall on a corpse. Exit non-zero with a
    #     MULTI_NODE_FAILURE_SNAPSHOT marker so the Ray Dashboard job
    #     flips to FAILED and `cmd_restart_server` raises
    #     `ServerRestartFailed` in O(seconds), letting the grid runner
    #     skip the broken variant instead of burning 30 min.
    #   * rank-0 process still alive -> server is still loading weights
    #     (slow checkpoint, slow init). Preserve legacy return-0 + WARN
    #     so the caller's external poll can catch up without us turning
    #     a slow boot into a hard failure.
    # Tri-state: True=alive, False=confirmed dead, None=cannot determine
    # (pid file missing / unreadable). Only flip to FAILED on confirmed
    # death; "unknown" stays on the legacy return-0 + WARN path so we
    # never invent a failure when the signal is missing.
    _r0_alive: bool | None = None
    if _r0_pid is not None and _r0_pid > 0:
        try:
            os.kill(_r0_pid, 0)
            _r0_alive = True
        except ProcessLookupError:
            _r0_alive = False
        except PermissionError:
            # pid exists, owned by another uid -> treat as alive
            _r0_alive = True
        except OSError:
            _r0_alive = None
    if _r0_alive is False:
        snap = {
            "kind": "framework_early_exit",
            "rank0_pid": _r0_pid,
            "hint": "framework process exited before /health flipped; "
                    "common causes: unrecognized launcher flag, kernel "
                    "ABI assert (e.g. aiter MLA num_qo_heads%16), OOM "
                    "during cuda graph capture, RCCL init failure. See "
                    "rank_0.log tail above.",
        }
        sys.stderr.write(
            f"MULTI_NODE_FAILURE_SNAPSHOT={json.dumps(snap)}\n"
        )
        sys.stderr.flush()
        _log(f"ERROR rank 0 pid={_r0_pid} dead before /health; returning 2 "
             f"so the Ray Dashboard job reports FAILED and hyperloom "
             f"surfaces ServerRestartFailed immediately (was: silent "
             f"1800s /health stall).")
        return 2
    # Even when the rank-0 wrapper PID is technically alive (typically
    # `nohup`/`setsid` lingers after the framework child has exited), a
    # fatal traceback in rank_0.log proves the framework already crashed.
    # Catches e.g. `KeyError: 'glm_moe_dsa'` (sglang transformers too old
    # for GLM-4.5/4.6 MoE-DSA), `CUDA out of memory`, RCCL init failures,
    # ImportError on a missing wheel, etc. — every one of which would
    # otherwise silently consume the full 1800s health-wait budget and
    # then be reported as SUCCEEDED to the caller (the bug this patch
    # closes; see /wekafs/.../sandbox-65ad7ec0629801-9k9ct incident).
    _fatal_line = _scan_rank0_log_for_fatal(args.log_dir)
    if _fatal_line:
        snap = {
            "kind": "framework_error",
            "rank0_pid": _r0_pid,
            "rank0_alive": _r0_alive,
            "hint": f"rank_0.log contains fatal: {_fatal_line[:500]}",
        }
        sys.stderr.write(
            f"MULTI_NODE_FAILURE_SNAPSHOT={json.dumps(snap)}\n"
        )
        sys.stderr.flush()
        _log(f"ERROR rank 0 framework error in rank_0.log (pid={_r0_pid} "
             f"alive={_r0_alive}); returning 2 so Ray Dashboard job reports "
             f"FAILED in seconds instead of 1800s silent stall.")
        return 2
    _log(f"WARN rank 0 /health did not pass within {_HEALTH_PROBE_TIMEOUT_SEC}s; "
         f"rank-0 pid={_r0_pid} alive={_r0_alive} (None=pid unknown) — "
         f"server likely still loading weights; the caller's external poll "
         f"will catch up.")
    return 0  # Don't fail; caller polls health from sandbox


if __name__ == "__main__":
    sys.exit(main())
