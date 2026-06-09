#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node sglang / vllm server launcher, run INSIDE the RayJob head pod.

Waits for ``nnodes`` alive GPU nodes, makes the local (head) node rank 0
and ranks the rest by NodeManagerAddress, then spawns one NodeAffinity-
pinned actor per rank that launches the framework detached via
bash+nohup+setsid (avoiding zombie PIDs / empty logs) and records
``<pid_dir>/rank_<K>.pid``. Optionally waits on rank-0 ``/health``
(``--no-wait-health`` to skip). Single-node restarts use the bash path.
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

# Hard-coded inference port (single source of truth; matches SaFE
# Service.targetPort). Bound by rank 0 in colocated mode, by the router in
# disaggregated mode (which proxies to the internal prefill/decode ports).
_INFERENCE_PORT = 8888
# Loopback-only internal ports for disaggregated PD server groups (router
# fronts them at _INFERENCE_PORT so the external URL is mode-independent).
_PD_PREFILL_PORT = 30000
_PD_DECODE_PORT = 30001
# Default collective port ($RAYJOB_DIST_INIT_PORT else 29500). 29500 (PyTorch
# convention) avoids the EADDRINUSE cascade a stale TCPStore LISTEN socket
# caused under the head pod's hostNetwork. Decode port = prefill + 1.
_DEFAULT_DIST_INIT_PORT = 29500


def _pd_decode_dist_init_port(prefill_dist_init_port: int) -> int:
    """Derive the decode rendezvous port as ``prefill + 1`` so an override shifts both endpoints in lock-step."""
    return prefill_dist_init_port + 1
# sglang PD bootstrap (KV transfer rendezvous) port; override via --pd-bootstrap-port.
_PD_DEFAULT_BOOTSTRAP_PORT = 8998
# Max seconds to wait for ray.nodes() to surface every expected pod.
_NODES_DISCOVERY_TIMEOUT_SEC = 120
# rank-0 /health probe budget (cold MoE can exceed it; --no-wait-health to bypass).
_HEALTH_PROBE_TIMEOUT_SEC = int(os.environ.get('SGLANG_HEALTH_PROBE_TIMEOUT_SEC', '1800'))


def _log(msg: str) -> None:
    """Stderr line with timestamp (no logging module to avoid handler surprises as a dashboard entry-point)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _wait_for_nodes(target_n: int, timeout_s: int) -> list[dict]:
    """Poll ray.nodes() until ``target_n`` alive GPU nodes are visible.

    Args:
        target_n (int): Number of alive GPU nodes required.
        timeout_s (int): Maximum seconds to wait before giving up.

    Returns:
        list[dict]: The alive GPU node rows from ``ray.nodes()``.

    Raises:
        RuntimeError: If fewer than ``target_n`` nodes are visible before
            the timeout elapses.
    """
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
    """Reorder so the local (driver) node is rank 0 and the rest follow in NodeManagerAddress order."""
    local_addr = ray.util.get_node_ip_address()
    local = [n for n in nodes if n.get("NodeManagerAddress") == local_addr]
    others = sorted(
        (n for n in nodes if n.get("NodeManagerAddress") != local_addr),
        key=lambda n: n.get("NodeManagerAddress", ""),
    )
    if not local:
        # Fallback: driver isn't on a GPU node; sort everything by address.
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
    """Compose the sglang multi-node launch command.

    Only rank 0 gets ``--host``/``--port`` (workers serve no HTTP).
    ``--trust-remote-code`` is default-on (custom modeling.py models need
    it). ``ep > 1`` emits ``--expert-parallel-size N`` for true expert
    parallelism (the older ``--enable-ep-moe`` pair was removed upstream).
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
        # rank-0 HTTP port = internal prefill/decode port in PD mode, public port in colocated.
        cmd.extend(["--host", "0.0.0.0", "--port", str(pd_port)])
    if ep > 1:
        cmd.extend(["--expert-parallel-size", str(ep)])
    role = pd_role.strip().lower()
    if role in ("prefill", "decode"):
        # PD disaggregated: split side + bootstrap port + KV-transfer IB device.
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
    """vLLM multi-node command for rank 0 (workers are KubeRay-joined; vLLM auto-discovers them via the GCS).

    ``--tensor-parallel-size`` = total cluster GPUs; ``ep > 1`` adds
    ``--enable-expert-parallel`` (vllm infers ep_size = tp_size).
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
        # vllm PD: kv-transfer-config JSON with connector + role + kv slot (default NixlConnector, cuda buffer).
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
    """Return True iff rocm-smi reports MEC firmware < 177 (gate for the HSA_NO_SCRATCH_RECLAIM workaround); False on any failure."""
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
    """Build the framework launcher subprocess env.

    Puts ``/opt/venv/bin`` first on PATH (framework venv) and inherits
    ``os.environ`` (keeps injected API keys / SGLANG_* / VLLM_* tunings).
    Sets the MI300X tuning trio (SGLANG_USE_AITER / SGLANG_AITER_MLA_PERSIST,
    and HSA_NO_SCRATCH_RECLAIM when MEC firmware < 177) without clobbering
    caller-set values.
    """
    env = dict(os.environ)
    # Strip Ray's empty *_VISIBLE_DEVICES mask (num_gpus=0 actors) so the
    # detached framework child re-discovers all GPUs; honour non-empty overrides.
    for _vis in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if _vis in env and env[_vis].strip() == "":
            env.pop(_vis, None)
    # Prevent the MI300X fused-decode-MLA crash (aiter ForwardMetadata mismatch).
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
    """Start ``cmd`` detached from the Ray worker via bash+nohup+setsid (reparents under init, PYTHONUNBUFFERED logs, fails fast with log tail)."""
    sub_env = dict(sub_env)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")
    log_q = shlex.quote(str(log_file))
    pid_q = shlex.quote(str(pid_file))
    inner = " ".join(shlex.quote(c) for c in cmd)
    # ``setsid`` gives a fresh session group for kill_multinode to SIGTERM (falls back to plain nohup).
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
    """Spawn the framework launcher detached on the local actor's pod, recording the PID; ``@ray.remote``-friendly.

    ``torch_profiler_dir`` (if set) is exported as
    ``SGLANG_TORCH_PROFILER_DIR`` for shared-path traces.
    """
    Path(pid_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # PD mode runs two groups with overlapping rank numbers; caller passes
    # a role-tagged pid_file_name so they don't collide on disk.
    fname = pid_file_name or f"rank_{node_rank}.pid"
    log_fname = (
        f"{Path(fname).stem}.log" if pid_file_name else f"rank_{node_rank}.log"
    )
    pid_file = Path(pid_dir) / fname
    log_file = Path(log_dir) / log_fname

    sub_env = _subprocess_env()
    # Resume-aware fallback to $HYPERLOOM_MN_PROFILE_TRACE_DIR so traces still
    # reach a shared dir when a reused server skips this launch.
    tpd = (
        (torch_profiler_dir or "").strip()
        or os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    )
    if tpd:
        # Pin profiler output to a shared dir; mkdir failure is non-fatal.
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
        # vLLM multi-node: workers are already KubeRay-joined to the GCS, so
        # worker actors do nothing; rank-0 vllm serve discovers them.
        if node_rank != 0:
            sys.stderr.write(
                f"[rank {node_rank}] vllm worker: no-op (KubeRay already "
                f"joined this node to the ray cluster; rank 0 vllm serve "
                f"will discover us via GCS)\n"
            )
            # Sentinel PID 0 so kill_multinode treats the file as stale.
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
    """Read rank 0 PID from /tmp/multi_node_pids/rank_0.pid (best-effort).

    Args:
        log_dir (str): The log directory; its parent is probed for the
            ``multi_node_pids/rank_0.pid`` file before falling back to the
            default ``/tmp`` location.

    Returns:
        int | None: The rank-0 PID, or ``None`` if it cannot be read.
    """
    try:
        pid_path = pathlib.Path(log_dir).parent / "multi_node_pids" / "rank_0.pid"
        if not pid_path.is_file():
            pid_path = pathlib.Path("/tmp/multi_node_pids/rank_0.pid")
        return int(pid_path.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


# Fatal patterns scanned in rank_0.log (catches crashes the lingering nohup
# wrapper PID hides, which would otherwise burn the full 1800s /health budget).
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
# How far back from EOF we scan (covers a full traceback, bounded for cost).
_FATAL_SCAN_TAIL_BYTES = 256 * 1024


def _scan_rank0_log_for_fatal(log_dir: str) -> str | None:
    """Scan rank_0.log tail for a fatal traceback / framework error; returns the matched line or ``None``."""
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
                f.readline()  # discard partial first line
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
    """Poll rank-0 ``/health``; True on first 200, False on timeout / rank-0 death / fatal ``rank_0.log`` error."""
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
        # Bail early if rank 0 died, else we wait the full 1800s on a corpse.
        if rank0_pid is not None and rank0_pid > 0:
            try:
                os.kill(rank0_pid, 0)
            except ProcessLookupError:
                _log(f"ERROR rank 0 pid={rank0_pid} died while waiting for /health; "
                     f"aborting health wait")
                return False
            except OSError:
                pass  # permission etc; don't kill the wait
        # Bail on a fatal traceback the lingering wrapper PID hides from os.kill above.
        if log_dir:
            fatal_line = _scan_rank0_log_for_fatal(log_dir)
            if fatal_line:
                _log(f"ERROR rank 0 fatal in rank_0.log: {fatal_line[:300]}; "
                     f"aborting health wait (was: silent 1800s stall)")
                return False
        time.sleep(5)
    return False


def _emit_rank0_log_tail(log_dir: Path) -> None:
    """Append the tail of ``rank_0.log`` to stderr if the file exists.

    Args:
        log_dir (Path): Directory containing ``rank_0.log``.
    """
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
    """Emit rank-0 diagnostics after a short settle (a ``rank_0.log`` tail for early weight-load exits)."""
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
    """Parse CLI arguments and launch the multi-node server group(s).

    Connects to the in-pod Ray cluster, discovers and rank-orders nodes,
    spawns one launcher actor per rank (colocated or PD-disaggregated),
    emits a JSON summary to stdout, and optionally waits for rank-0
    ``/health``.

    Returns:
        int: Process exit code; ``0`` on success (or slow-but-alive boot),
        ``1`` on a spawn failure, ``2`` on invalid args or a confirmed
        framework early-exit / fatal log error.
    """
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
    # PD disaggregation args: `colocated` (default) is one TP group;
    # `disaggregated` splits into prefill/decode groups fronted by the router.
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
        # IB device default: pod's $NCCL_IB_HCA.
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

    # Spawn one actor per rank (num_gpus=0; the framework reserves GPUs itself).
    SpawnActor = ray.remote(num_cpus=1, num_gpus=0)(_spawn_remote)
    pids: dict[str, int] = {}     # pid_file_name -> real PID
    refs: list[tuple[str, Any]] = []  # noqa: F821  Any imported below

    if pd_mode == "disaggregated":
        # Group A: prefill — nodes[0:pn], TP=ptp; rank-0 binds the internal prefill port.
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

        # Group B: decode — nodes[pn:pn+dn]; dist-init port = prefill + 1 to avoid rendezvous clashes.
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
            # Roll back already-spawned ranks so no half-started servers leak.
            for tag2, p2 in pids.items():
                _log(f"rolling back {tag2} pid={p2}")
                try:
                    os.killpg(os.getpgid(p2), 15)
                except (ProcessLookupError, PermissionError):
                    pass
            return 1

    # Health-tail probe targets the leader: rank_0 (colocated) / prefill_0 (PD).
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
        # Emit the internal endpoints + bootstrap port so the CLI can submit
        # the router without re-discovering nodes.
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

    # In PD mode the router owns 8888; skip the local probe (caller polls externally).
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
    # Tail the log (timeout or early rank-0 death) for the framework's last words.
    _emit_rank0_log_tail(Path(args.log_dir))
    # Distinguish failure modes for the caller. Tri-state: True=alive,
    # False=confirmed dead, None=unknown. Only confirmed death flips to
    # FAILED (return 2); alive/unknown stay on the legacy return-0 + WARN path.
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
    # A fatal traceback in rank_0.log proves a crash even when the nohup
    # wrapper PID lingers (else the 1800s wait silently reports SUCCEEDED).
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
