#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node sglang / vllm server launcher, run INSIDE the RayJob head pod."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from server_args_safety import find_denied_flags, find_unsafe_flag_values
from sglang_shape_gate import activate_kernel_shape_tool

# Inference port.
_INFERENCE_PORT = 8888
# Loopback-only internal ports for disaggregated PD server groups (router fronts them at _INFERENCE_PORT so the
# external URL is mode-independent).
_PD_PREFILL_PORT = 30000
_PD_DECODE_PORT = 30001
# Default collective port ($RAYJOB_DIST_INIT_PORT else 29500).
_DEFAULT_DIST_INIT_PORT = 29500
# vLLM PD connectors named in ``--pd-transfer-backend`` help.
_VLLM_KV_CONNECTORS: frozenset[str] = frozenset(
    (
        "NixlConnector",
        "P2pNcclConnector",
        "MooncakeConnector",
        "LMCacheConnectorV1",
    )
)


def _pd_decode_dist_init_port(prefill_dist_init_port: int) -> int:
    """Derive the decode rendezvous port as ``prefill + 1``."""
    return prefill_dist_init_port + 1


# sglang PD bootstrap (KV transfer rendezvous) port; override via --pd-bootstrap-port.
_PD_DEFAULT_BOOTSTRAP_PORT = 8998
# Max seconds to wait for ray.nodes() to surface every expected pod; raise it when pods queue behind a busy scheduler
# or a cold image pull.
_NODES_DISCOVERY_TIMEOUT_SEC = int(os.environ.get("RAY_NODES_DISCOVERY_TIMEOUT_SEC", "120"))
# rank-0 /health probe budget (cold MoE can exceed it; --no-wait-health to bypass).
_HEALTH_PROBE_TIMEOUT_SEC = int(os.environ.get("SGLANG_HEALTH_PROBE_TIMEOUT_SEC", "1800"))


def _log(msg: str) -> None:
    """Write a timestamped launcher log line to stderr."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _wait_for_nodes(target_n: int, timeout_s: int) -> list[dict]:
    """Poll ray.nodes() until ``target_n`` alive GPU nodes are visible."""
    started = time.monotonic()
    while True:
        nodes = [n for n in ray.nodes() if n.get("Alive") and float(n.get("Resources", {}).get("GPU", 0)) > 0]
        if len(nodes) >= target_n:
            return nodes
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            raise RuntimeError(
                f"only {len(nodes)}/{target_n} alive GPU nodes after {elapsed:.0f}s; check KubeRay scheduler logs"
            )
        _log(f"waiting for nodes: have={len(nodes)} need={target_n} t={elapsed:.0f}s")
        time.sleep(3)


def _pick_head_first(nodes: list[dict]) -> list[dict]:
    """Reorder so the local (driver) node is rank 0, rest by NodeManagerAddress."""
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


@dataclass(frozen=True)
class _PdRole:
    """A server group's part in a PD split; the defaults describe an aggregated group on the public port."""

    role: str = ""
    port: int = _INFERENCE_PORT
    transfer_backend: str = ""
    ib_device: str = ""
    bootstrap_port: int = _PD_DEFAULT_BOOTSTRAP_PORT
    kv_rank: int = 0
    kv_parallel_size: int = 1


@dataclass(frozen=True)
class _ServerGroup:
    """One framework server group: what it serves, how it is parallelised, and where its ranks rendezvous."""

    framework: str
    model: str
    tp: int
    nnodes: int
    head_ip: str
    dist_init_port: int
    extra_args: list[str]
    ep: int = 1
    pd: _PdRole = _PdRole()


@dataclass(frozen=True)
class _LaunchDirs:
    """Where a launch's ranks write PID files and logs, and the shared torch profiler dir when one is set."""

    pid_dir: str
    log_dir: str
    torch_profiler_dir: str = ""


def _build_sglang_cmd(group: _ServerGroup, node_rank: int) -> list[str]:
    """Compose the sglang multi-node launch command."""
    pd = group.pd
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        group.model,
        "--trust-remote-code",
        "--tp",
        str(group.tp),
        "--nnodes",
        str(group.nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        f"{group.head_ip}:{group.dist_init_port}",
    ]
    if node_rank == 0:
        # rank-0 HTTP port = internal prefill/decode port in PD mode, else public.
        cmd.extend(["--host", "0.0.0.0", "--port", str(pd.port)])  # nosec B104 - rank-0 server must accept pod traffic.
    if group.ep > 1:
        cmd.extend(["--expert-parallel-size", str(group.ep)])
    role = pd.role.strip().lower()
    if role in ("prefill", "decode"):
        cmd.extend(["--disaggregation-mode", role])
        cmd.extend(["--disaggregation-bootstrap-port", str(pd.bootstrap_port)])
        if pd.transfer_backend:
            cmd.extend(["--disaggregation-transfer-backend", pd.transfer_backend])
        if pd.ib_device:
            cmd.extend(["--disaggregation-ib-device", pd.ib_device])
    if group.extra_args:
        cmd.extend(group.extra_args)
    return cmd


def _build_vllm_cmd(
    model: str,
    tp: int,
    extra_args: list[str],
    *,
    ep: int = 1,
    pd: _PdRole = _PdRole(),
) -> list[str]:
    """vLLM multi-node command for rank 0 (workers are KubeRay-joined and vLLM auto-discovers them via the GCS)."""
    cmd = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size",
        str(tp),
        "--host",
        "0.0.0.0",  # nosec B104 - vLLM rank-0 server must accept pod traffic.
        "--port",
        str(pd.port),
        "--distributed-executor-backend",
        "ray",
    ]
    if ep > 1:
        cmd.append("--enable-expert-parallel")
    role = pd.role.strip().lower()
    if role in ("prefill", "decode"):
        # vllm PD: kv-transfer-config JSON with connector + role + kv slot.
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        connector = pd.transfer_backend or "NixlConnector"
        if connector not in _VLLM_KV_CONNECTORS:
            _log(f"WARN unknown vLLM kv connector {connector!r}; known: {sorted(_VLLM_KV_CONNECTORS)}")
        kv_cfg = json.dumps(
            {
                "kv_connector": connector,
                "kv_role": kv_role,
                "kv_rank": int(pd.kv_rank),
                "kv_parallel_size": int(pd.kv_parallel_size),
                "kv_buffer_device": "cuda",
            },
            separators=(",", ":"),
            sort_keys=False,
        )
        cmd.extend(["--kv-transfer-config", kv_cfg])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _probe_mec_firmware_lt_177() -> bool:
    """Return True iff rocm-smi reports MEC firmware < 177 (gates the HSA_NO_SCRATCH_RECLAIM workaround); False on any failure."""
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showfw"],
            capture_output=True,
            text=True,
            timeout=10,
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
    """Build the framework launcher subprocess env."""
    env = dict(os.environ)
    # Strip Ray's empty *_VISIBLE_DEVICES mask so the detached framework child re-discovers all GPUs; honour non-empty
    # overrides.
    for _vis in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        if _vis in env and env[_vis].strip() == "":
            env.pop(_vis, None)
    # Prevent the MI300X fused-decode-MLA crash.
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
    """Start ``cmd`` detached from the Ray worker via bash+nohup+setsid (reparents under init; fails fast with a log tail)."""
    sub_env = dict(sub_env)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")
    log_q = shlex.quote(str(log_file))
    pid_q = shlex.quote(str(pid_file))
    inner = " ".join(shlex.quote(c) for c in cmd)
    # ``setsid`` gives a fresh session group for kill_multinode to SIGTERM.
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
        raise RuntimeError(f"[rank {node_rank}] missing or invalid pid file {pid_file}: {exc}") from exc

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
            f"[rank {node_rank}] child pid={pid} not alive after 0.5s: {exc}\n[rank {node_rank}] log tail:\n{tail}\n",
        )
        raise RuntimeError(
            f"[rank {node_rank}] framework exited immediately (pid={pid}); see log tail on stderr and {log_file}"
        ) from exc
    return pid


def _spawn_remote(group: _ServerGroup, node_rank: int, dirs: _LaunchDirs, *, pid_file_name: str = "") -> int:
    """Spawn the framework launcher detached on the local actor's pod, recording the PID."""
    Path(dirs.pid_dir).mkdir(parents=True, exist_ok=True)
    Path(dirs.log_dir).mkdir(parents=True, exist_ok=True)
    # PD mode runs two groups with overlapping rank numbers; caller passes a role-tagged pid_file_name so they don't
    # collide on disk.
    fname = pid_file_name or f"rank_{node_rank}.pid"
    log_fname = f"{Path(fname).stem}.log" if pid_file_name else f"rank_{node_rank}.log"
    pid_file = Path(dirs.pid_dir) / fname
    log_file = Path(dirs.log_dir) / log_fname

    sub_env = _subprocess_env()
    # Fall back to $HYPERLOOM_MN_PROFILE_TRACE_DIR so traces still reach a shared dir when a reused server skips this
    # launch.
    tpd = (dirs.torch_profiler_dir or "").strip() or os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
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

    fw = group.framework.lower()
    if fw == "sglang":
        activate_kernel_shape_tool(sub_env)
        cmd = _build_sglang_cmd(group, node_rank)
    elif fw == "vllm":
        # vLLM multi-node: workers are already KubeRay-joined to the GCS, so worker actors do nothing; rank-0 vllm
        # serve discovers them.
        if node_rank != 0:
            sys.stderr.write(
                f"[rank {node_rank}] vllm worker: no-op (KubeRay already "
                f"joined this node to the ray cluster; rank 0 vllm serve "
                f"will discover us via GCS)\n"
            )
            # Sentinel PID 0 so kill_multinode treats the file as stale.
            pid_file.write_text("0")
            return 0
        cmd = _build_vllm_cmd(group.model, group.tp, group.extra_args, ep=group.ep, pd=group.pd)
    else:
        raise RuntimeError(f"unsupported framework: {group.framework!r}")

    sys.stderr.write(f"[rank {node_rank}] launching: {' '.join(cmd)}\n")
    sys.stderr.write(f"[rank {node_rank}] log={log_file} pid={pid_file}\n")
    return _detach_framework_launch(cmd, log_file, pid_file, sub_env, node_rank)


# Budget for one node-pinned rollback kill (SIGTERM + report; no death-wait).
_ROLLBACK_TIMEOUT_SEC = 30


def _rollback_remote(pid: int) -> str:
    """SIGTERM a spawned rank's process group, ON the node that owns the PID."""
    import os as _os
    import signal as _signal

    if pid <= 0:
        return f"skipped: sentinel pid={pid}"
    try:
        _os.killpg(_os.getpgid(pid), _signal.SIGTERM)
    except ProcessLookupError:
        return f"no-op: pid={pid} already gone"
    except PermissionError:
        # Not ours to signal -- report rather than pretend it was cleaned up.
        return f"DENIED: pid={pid} not signallable by this uid"
    return f"SIGTERM sent to pgid of pid={pid}"


def _rank0_pid_from_log(pid_dir: str) -> int | None:
    """Read rank 0 PID from ``{pid_dir}/rank_0.pid`` (best-effort)."""
    try:
        return int((Path(pid_dir) / "rank_0.pid").read_text().strip())
    except (OSError, ValueError):
        return None


# Fatal patterns scanned in rank_0.log (catches crashes the lingering nohup wrapper PID hides).
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


def _scan_log_for_fatal(log_file: Path) -> str | None:
    """Scan a log file's tail for a fatal traceback / framework error."""
    try:
        if not log_file.is_file():
            return None
        size = log_file.stat().st_size
        if size == 0:
            return None
        with log_file.open("rb") as f:
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


def _scan_rank0_log_for_fatal(log_dir: str) -> str | None:
    """Scan rank_0.log tail for a fatal traceback / framework error."""
    return _scan_log_for_fatal(Path(log_dir) / "rank_0.log")


def _wait_health(
    timeout_s: int,
    rank0_pid: int | None = None,
    log_dir: str | None = None,
) -> bool:
    """Poll rank-0 ``/health``; True on first 200, else False on timeout, rank-0 death, or a fatal ``rank_0.log`` error."""
    import urllib.request
    import urllib.error

    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{_INFERENCE_PORT}/health",
                timeout=3,
            ) as resp:  # nosec B310 - fixed loopback health check.
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        # Bail early if rank 0 died, else we wait the full timeout on a corpse.
        if rank0_pid is not None and rank0_pid > 0:
            try:
                os.kill(rank0_pid, 0)
            except ProcessLookupError:
                _log(f"ERROR rank 0 pid={rank0_pid} died while waiting for /health; aborting health wait")
                return False
            except OSError:
                pass
        # Bail on a fatal traceback the lingering wrapper PID hides from os.kill.
        if log_dir:
            fatal_line = _scan_rank0_log_for_fatal(log_dir)
            if fatal_line:
                _log(
                    f"ERROR rank 0 fatal in rank_0.log: {fatal_line[:300]}; "
                    f"aborting health wait (was: silent 1800s stall)"
                )
                return False
        time.sleep(5)
    return False


def _wait_pd_legs_health(
    prefill_url: str,
    decode_url: str,
    timeout_s: int,
    log_dir: str | None = None,
) -> bool | None:
    """Poll both PD legs' own ``/health`` until each answers, within one budget."""
    import urllib.error
    import urllib.request

    def _leg_log(role: str) -> Path | None:
        # Legs log to {pid_file_name stem}.log, e.g. decode_0.log / prefill_0.log.
        return Path(log_dir) / f"{role}_0.log" if log_dir else None

    pending = {"prefill": prefill_url, "decode": decode_url}
    started = time.monotonic()
    while pending and time.monotonic() - started < timeout_s:
        for role, url in list(pending.items()):
            try:
                with urllib.request.urlopen(  # nosec B310 - http URL built from this driver's own summary.
                    f"{url.rstrip('/')}/health",
                    timeout=3,
                ) as resp:
                    if 200 <= resp.status < 300:
                        _log(f"{role} leg /health OK ({url})")
                        pending.pop(role)
                        continue
            except (urllib.error.URLError, OSError):
                pass
            # A remote leg's PID is not ours to os.kill; a fatal in its own log is the cross-node-safe proof of death
            # (the nohup wrapper hides crashes).
            leg_log = _leg_log(role)
            if leg_log is not None:
                fatal_line = _scan_log_for_fatal(leg_log)
                if fatal_line:
                    _log(f"ERROR {role} leg fatal in {leg_log.name}: {fatal_line[:300]}; aborting health wait")
                    _emit_log_tail(leg_log)
                    return False
        if pending:
            time.sleep(5)
    if pending:
        _log(f"WARN PD legs still not /health-ready after {timeout_s}s: {sorted(pending)}")
        return None
    return True


_LOG_TAIL_BYTES = 8192


def _emit_log_tail(log_file: Path) -> None:
    """Append the tail of a log file to stderr if it exists."""
    name = log_file.name
    try:
        sz = log_file.stat().st_size if log_file.is_file() else 0
        _log(f"(tail probe) {name} bytes={sz} path={log_file}")
        if sz > 0:
            with log_file.open("rb") as f:
                if sz > _LOG_TAIL_BYTES:
                    f.seek(sz - _LOG_TAIL_BYTES)
                tail = f.read().decode("utf-8", errors="replace")
            _log(f"{name} tail (last 8kiB):\n" + tail)
    except OSError as exc:
        _log(f"(tail probe) cannot read {name}: {exc}")


def _emit_rank0_log_tail(log_dir: Path) -> None:
    """Append the tail of ``rank_0.log`` to stderr if the file exists."""
    _emit_log_tail(log_dir / "rank_0.log")


def _log_rank0_post_spawn(log_dir: Path, rank0_pid: int | None) -> None:
    """Emit rank-0 diagnostics after a short settle (a ``rank_0.log`` tail)."""
    if rank0_pid is None or rank0_pid <= 0:
        return
    time.sleep(3)
    try:
        os.kill(rank0_pid, 0)
        _log(f"post-spawn+3s rank0 pid={rank0_pid} still alive")
    except OSError as exc:
        _log(f"ERROR post-spawn+3s rank0 pid={rank0_pid} not alive: {exc}")
    _emit_rank0_log_tail(log_dir)


def _build_parser() -> argparse.ArgumentParser:
    """The launcher's command line."""
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
        "--dist-init-port",
        type=int,
        default=int(os.environ.get("RAYJOB_DIST_INIT_PORT") or _DEFAULT_DIST_INIT_PORT),
        help=f"sglang collective rendezvous port (default {_DEFAULT_DIST_INIT_PORT}, "
        f"resolution: --dist-init-port > $RAYJOB_DIST_INIT_PORT > {_DEFAULT_DIST_INIT_PORT})",
    )
    p.add_argument("--no-wait-health", action="store_true", help="don't poll /health on rank 0 before returning")
    p.add_argument("--extra-args", default="", help="extra args appended verbatim to the framework launcher")
    p.add_argument(
        "--torch-profiler-dir",
        default="",
        help="when set, exported as SGLANG_TORCH_PROFILER_DIR on every "
        "rank's framework subprocess; intended for a wekafs path "
        "shared with the sandbox (see HYPERLOOM_MN_PROFILE_TRACE_DIR)",
    )
    p.add_argument(
        "--ep",
        type=int,
        default=1,
        help="expert-parallel size; 1 (default) keeps experts "
        "TP-sharded (legacy). >=2 emits sglang "
        "`--enable-ep-moe --ep-size N` or vllm "
        "`--enable-expert-parallel`. Caller (orchestrator "
        "helper) is responsible for ensuring ep <= tp.",
    )
    # PD disaggregation args: `aggregated` is one TP group; `disaggregated` splits into prefill/decode groups fronted
    # by the router.
    p.add_argument(
        "--pd-mode",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="PD disaggregation mode (default aggregated)",
    )
    p.add_argument(
        "--pd-prefill-nodes",
        type=int,
        default=0,
        help="number of prefill nodes (disaggregated only); must satisfy pd_prefill_nodes + pd_decode_nodes == nnodes",
    )
    p.add_argument("--pd-decode-nodes", type=int, default=0, help="number of decode nodes (disaggregated only)")
    p.add_argument(
        "--pd-prefill-tp",
        type=int,
        default=0,
        help="TP size for the prefill group (disaggregated only); default = --tp",
    )
    p.add_argument(
        "--pd-decode-tp", type=int, default=0, help="TP size for the decode group (disaggregated only); default = --tp"
    )
    p.add_argument(
        "--pd-transfer-backend",
        default="",
        help="sglang: mooncake|nixl ; vllm: NixlConnector|"
        "P2pNcclConnector|MooncakeConnector|LMCacheConnectorV1; "
        "empty = framework default (sglang mooncake / vllm NixlConnector)",
    )
    p.add_argument(
        "--pd-ib-device",
        default="",
        help="comma-separated IB/RoCE device list for KV transfer "
        "(e.g. mlx5_0,mlx5_1). Empty = read $NCCL_IB_HCA "
        "from this pod's env (RayJob image typically injects "
        "it); if that's also empty, mooncake auto-detects.",
    )
    p.add_argument(
        "--pd-bootstrap-port",
        type=int,
        default=_PD_DEFAULT_BOOTSTRAP_PORT,
        help=f"sglang PD bootstrap rendezvous port (default {_PD_DEFAULT_BOOTSTRAP_PORT})",
    )
    return p


@dataclass(frozen=True)
class _PdPlan:
    """How one launch divides its nodes, aggregated or split into prefill/decode groups."""

    mode: str
    prefill_nodes: int
    decode_nodes: int
    prefill_tp: int
    decode_tp: int
    ib_device: str

    @property
    def disaggregated(self) -> bool:
        """Whether this launch fronts two server groups with a router."""
        return self.mode == "disaggregated"


def _resolve_pd_plan(args: argparse.Namespace) -> _PdPlan | None:
    """The node/TP split this launch uses, or None when the arguments do not describe one."""
    mode = (args.pd_mode or "aggregated").lower()
    if mode != "disaggregated":
        return _PdPlan(mode, args.nnodes, 0, args.tp, 0, "")
    prefill_nodes = int(args.pd_prefill_nodes or 0)
    decode_nodes = int(args.pd_decode_nodes or 0)
    if prefill_nodes <= 0 or decode_nodes <= 0 or prefill_nodes + decode_nodes != args.nnodes:
        _log(
            f"PD invalid split: pd_prefill_nodes={prefill_nodes} pd_decode_nodes={decode_nodes} "
            f"nnodes={args.nnodes}; require pn+dn==nnodes and both >0"
        )
        return None
    prefill_tp = int(args.pd_prefill_tp or args.tp)
    decode_tp = int(args.pd_decode_tp or args.tp)
    if prefill_tp <= 0 or decode_tp <= 0:
        _log(f"PD invalid TP: pd_prefill_tp={prefill_tp} pd_decode_tp={decode_tp}")
        return None
    return _PdPlan(
        mode,
        prefill_nodes,
        decode_nodes,
        prefill_tp,
        decode_tp,
        (args.pd_ib_device or os.environ.get("NCCL_IB_HCA", "")).strip(),
    )


@dataclass(frozen=True)
class _GroupHeads:
    """The driver node and the address each server group's rank 0 binds (all the driver when aggregated)."""

    head_ip: str
    prefill_head_ip: str
    decode_head_ip: str


def _on_node(spawn_actor, node: dict):
    """Pin one spawn actor to the node whose GPUs its rank will use."""
    return spawn_actor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
    )


def _spawn_rank_actors(
    args: argparse.Namespace,
    *,
    nodes: list,
    head_ip: str,
    plan: _PdPlan,
    extra_args: list[str],
) -> tuple[list, _GroupHeads]:
    """Spawn one actor per rank, and say which address each server group rendezvous on.

    ``num_gpus=0`` on the actors: the framework process reserves the GPUs itself.
    """
    spawn_actor = ray.remote(num_cpus=1, num_gpus=0)(_spawn_remote)
    dirs = _LaunchDirs(pid_dir=args.pid_dir, log_dir=args.log_dir, torch_profiler_dir=args.torch_profiler_dir)
    refs: list = []
    if not plan.disaggregated:
        # Aggregated: single server group spans all nodes.
        group = _ServerGroup(
            framework=args.framework,
            model=args.model,
            tp=args.tp,
            nnodes=args.nnodes,
            head_ip=head_ip,
            dist_init_port=args.dist_init_port,
            extra_args=extra_args,
            ep=int(args.ep or 1),
        )
        for rank, node in enumerate(nodes):
            actor_ref = _on_node(spawn_actor, node).remote(group, rank, dirs)
            refs.append((f"rank_{rank}", node["NodeID"], actor_ref))
        return refs, _GroupHeads(head_ip, head_ip, head_ip)

    # Group A: prefill -- nodes[0:pn], TP=ptp; rank-0 binds the prefill port.
    prefill_head_ip = nodes[0].get("NodeManagerAddress", head_ip)
    prefill = _ServerGroup(
        framework=args.framework,
        model=args.model,
        tp=plan.prefill_tp,
        nnodes=plan.prefill_nodes,
        head_ip=prefill_head_ip,
        dist_init_port=args.dist_init_port,
        extra_args=extra_args,
        ep=int(args.ep or 1),
        pd=_PdRole(
            role="prefill",
            port=_PD_PREFILL_PORT,
            transfer_backend=args.pd_transfer_backend,
            ib_device=plan.ib_device,
            bootstrap_port=args.pd_bootstrap_port,
            kv_rank=0,  # vllm-only; sglang ignores
            kv_parallel_size=2,
        ),
    )
    for grp_rank in range(plan.prefill_nodes):
        node = nodes[grp_rank]
        actor_ref = _on_node(spawn_actor, node).remote(prefill, grp_rank, dirs, pid_file_name=f"prefill_{grp_rank}.pid")
        refs.append((f"prefill_{grp_rank}", node["NodeID"], actor_ref))

    # Group B: decode -- nodes[pn:pn+dn]; dist-init port = prefill + 1.
    decode_head_ip = nodes[plan.prefill_nodes].get("NodeManagerAddress", head_ip)
    decode = _ServerGroup(
        framework=args.framework,
        model=args.model,
        tp=plan.decode_tp,
        nnodes=plan.decode_nodes,
        head_ip=decode_head_ip,
        dist_init_port=_pd_decode_dist_init_port(args.dist_init_port),
        extra_args=extra_args,
        ep=int(args.ep or 1),
        pd=_PdRole(
            role="decode",
            port=_PD_DECODE_PORT,
            transfer_backend=args.pd_transfer_backend,
            ib_device=plan.ib_device,
            bootstrap_port=args.pd_bootstrap_port,
            kv_rank=1,
            kv_parallel_size=2,
        ),
    )
    for grp_rank in range(plan.decode_nodes):
        node = nodes[plan.prefill_nodes + grp_rank]
        actor_ref = _on_node(spawn_actor, node).remote(decode, grp_rank, dirs, pid_file_name=f"decode_{grp_rank}.pid")
        refs.append((f"decode_{grp_rank}", node["NodeID"], actor_ref))
    return refs, _GroupHeads(head_ip, prefill_head_ip, decode_head_ip)


def _rollback_spawned(pids: dict[str, tuple[str, int]]) -> None:
    """Tear down the ranks that did start, so no half-started servers leak."""
    rollback_actor = ray.remote(num_cpus=0, num_gpus=0)(_rollback_remote)
    for tag, (node_id, pid) in pids.items():
        if pid <= 0:
            _log(f"rolling back {tag}: sentinel pid={pid} (no process); skipped")
            continue
        try:
            outcome = ray.get(
                rollback_actor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(pid),
                timeout=_ROLLBACK_TIMEOUT_SEC,
            )
            _log(f"rolling back {tag} pid={pid} on node={node_id[:16]}...: {outcome}")
        except (ray.exceptions.RayError, OSError) as rb_exc:
            # An unreachable rank keeps its GPUs until the platform reclaims the cluster, so it is reported
            # rather than dropped.
            _log(
                f"WARN rollback of {tag} pid={pid} on node={node_id[:16]}... "
                f"FAILED: {type(rb_exc).__name__}: {rb_exc}; that rank may "
                f"still hold GPU memory"
            )


def _collect_spawned_pids(refs: list) -> dict[str, tuple[str, int]] | None:
    """The PID each rank reported, or None once a failure has rolled the launch back."""
    pids: dict[str, tuple[str, int]] = {}
    for tag, node_id, ref in refs:
        try:
            pid = ray.get(ref, timeout=120)
        except (ray.exceptions.RayError, OSError) as exc:
            _log(f"{tag}: spawn FAILED: {type(exc).__name__}: {exc}")
            _rollback_spawned(pids)
            return None
        pids[tag] = (node_id, pid)
        _log(f"{tag}: spawned pid={pid}")
    return pids


def _rank0_liveness(rank0_pid: int | None) -> bool | None:
    """Whether rank 0 is still running: True, False, or None when that cannot be told."""
    if rank0_pid is None or rank0_pid <= 0:
        return None
    try:
        os.kill(rank0_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # pid exists, owned by another uid -> treat as alive
        return True
    except OSError:
        return None
    return True


def _probe_pd_health(args: argparse.Namespace, summary: dict) -> int:
    """Wait on both PD legs, since the router that owns 8888 does not exist yet.

    The caller submits the router once this driver returns.
    """
    _log(f"polling both PD legs' /health for up to {_HEALTH_PROBE_TIMEOUT_SEC}s")
    legs_ready = _wait_pd_legs_health(
        summary["pd_prefill_url"],
        summary["pd_decode_url"],
        _HEALTH_PROBE_TIMEOUT_SEC,
        log_dir=args.log_dir,
    )
    if legs_ready:
        _log("both PD legs /health OK")
        return 0
    # PD has no rank_0.log; tail each leg's own log so the caller sees why.
    _emit_log_tail(Path(args.log_dir) / "prefill_0.log")
    _emit_log_tail(Path(args.log_dir) / "decode_0.log")
    if legs_ready is False:
        _log("ERROR a PD leg logged a fatal crash before serving; marking the launch failed")
        return 2
    # Undetermined, as for rank 0: warn rather than fail a slow-but-live boot.
    _log("WARN PD legs unconfirmed; the caller's serving probe decides from here")
    return 0


def _probe_rank0_health(args: argparse.Namespace) -> int:
    """Wait on rank 0, and read the log when it does not answer."""
    _log(f"polling rank 0 /health for up to {_HEALTH_PROBE_TIMEOUT_SEC}s")
    rank0_pid = _rank0_pid_from_log(args.pid_dir)
    if _wait_health(_HEALTH_PROBE_TIMEOUT_SEC, rank0_pid=rank0_pid, log_dir=args.log_dir):
        _log("rank 0 /health OK")
        return 0
    # Tail the log (timeout or early rank-0 death) for the framework's last words.
    _emit_rank0_log_tail(Path(args.log_dir))
    rank0_alive = _rank0_liveness(rank0_pid)
    if rank0_alive is False:
        _emit_failure_snapshot(
            {
                "kind": "framework_early_exit",
                "rank0_pid": rank0_pid,
                "hint": "framework process exited before /health flipped; "
                "common causes: unrecognized launcher flag, kernel "
                "ABI assert (e.g. aiter MLA num_qo_heads%16), OOM "
                "during cuda graph capture, RCCL init failure. See "
                "rank_0.log tail above.",
            }
        )
        _log(
            f"ERROR rank 0 pid={rank0_pid} dead before /health; returning 2 "
            f"so the Ray Dashboard job reports FAILED and hyperloom "
            f"surfaces ServerRestartFailed immediately (was: silent "
            f"1800s /health stall)."
        )
        return 2
    # A fatal traceback in rank_0.log proves a crash even when the nohup wrapper PID lingers.
    fatal_line = _scan_rank0_log_for_fatal(args.log_dir)
    if fatal_line:
        _emit_failure_snapshot(
            {
                "kind": "framework_error",
                "rank0_pid": rank0_pid,
                "rank0_alive": rank0_alive,
                "hint": f"rank_0.log contains fatal: {fatal_line[:500]}",
            }
        )
        _log(
            f"ERROR rank 0 framework error in rank_0.log (pid={rank0_pid} "
            f"alive={rank0_alive}); returning 2 so Ray Dashboard job reports "
            f"FAILED in seconds instead of 1800s silent stall."
        )
        return 2
    _log(
        f"WARN rank 0 /health did not pass within {_HEALTH_PROBE_TIMEOUT_SEC}s; "
        f"rank-0 pid={rank0_pid} alive={rank0_alive} (None=pid unknown) — "
        f"server likely still loading weights; the caller's external poll "
        f"will catch up."
    )
    return 0  # Don't fail; caller polls health from sandbox


def _emit_failure_snapshot(snapshot: dict) -> None:
    """Hand the caller a machine-readable reason the launch failed."""
    sys.stderr.write(f"MULTI_NODE_FAILURE_SNAPSHOT={json.dumps(snapshot)}\n")
    sys.stderr.flush()


def _launch_summary(
    args: argparse.Namespace,
    *,
    plan: _PdPlan,
    heads: _GroupHeads,
    pids: dict[str, tuple[str, int]],
) -> dict:
    """What the caller needs to reach and manage the servers this launch started."""
    summary: dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "tp": args.tp,
        "ep": int(args.ep or 1),
        "nnodes": args.nnodes,
        "head_ip": heads.head_ip,
        "dist_init_port": args.dist_init_port,
        "ranks": [{"tag": tag, "node_id": pids[tag][0], "pid": pids[tag][1]} for tag in sorted(pids)],
        "pid_dir": args.pid_dir,
        "log_dir": args.log_dir,
        "inference_port": _INFERENCE_PORT,
        "pd_mode": plan.mode,
    }
    if not plan.disaggregated:
        return summary
    # Emit internal endpoints + bootstrap port so the CLI can submit the router without re-discovering nodes.
    summary["pd_prefill_nodes"] = plan.prefill_nodes
    summary["pd_decode_nodes"] = plan.decode_nodes
    summary["pd_prefill_tp"] = plan.prefill_tp
    summary["pd_decode_tp"] = plan.decode_tp
    summary["pd_transfer_backend"] = args.pd_transfer_backend or (
        "NixlConnector" if args.framework.lower() == "vllm" else "mooncake"
    )
    summary["pd_ib_device"] = plan.ib_device
    summary["pd_bootstrap_port"] = args.pd_bootstrap_port
    summary["pd_prefill_url"] = f"http://{heads.prefill_head_ip}:{_PD_PREFILL_PORT}"
    summary["pd_decode_url"] = f"http://{heads.decode_head_ip}:{_PD_DECODE_PORT}"
    return summary


def _discover_ranked_nodes(nnodes: int) -> tuple[list, str]:
    """The GPU nodes this launch uses, in rank order, and the head address."""
    nodes = _pick_head_first(_wait_for_nodes(nnodes, _NODES_DISCOVERY_TIMEOUT_SEC))[:nnodes]
    head_ip = ray.util.get_node_ip_address()
    _log(f"discovered {len(nodes)} GPU nodes; head_ip={head_ip}; rank order:")
    for rank, node in enumerate(nodes):
        _log(
            f"  rank {rank}: node_id={node.get('NodeID', '?')[:16]}... "
            f"addr={node.get('NodeManagerAddress', '?')} "
            f"gpu={node.get('Resources', {}).get('GPU', 0)}"
        )
    return nodes, head_ip


def main() -> int:
    """Parse CLI arguments and launch the multi-node server group(s)."""
    args = _build_parser().parse_args()

    if args.nnodes < 2:
        _log(
            f"nnodes={args.nnodes} < 2; this script is for multi-node only. "
            f"Use launch_server.sh for single-pod restarts."
        )
        return 2

    plan = _resolve_pd_plan(args)
    if plan is None:
        return 2

    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    denied = find_denied_flags(args.extra_args) + find_unsafe_flag_values(args.extra_args)
    if denied:
        _log(f"ERROR denied server flags in --extra-args: {denied}")
        return 2

    _log(f"framework={args.framework} model={args.model} tp={args.tp} nnodes={args.nnodes}")

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes, head_ip = _discover_ranked_nodes(args.nnodes)
    refs, heads = _spawn_rank_actors(
        args,
        nodes=nodes,
        head_ip=head_ip,
        plan=plan,
        extra_args=extra_args,
    )
    pids = _collect_spawned_pids(refs)
    if pids is None:
        return 1

    # Health-tail probe targets the leader: rank_0 (aggregated) / prefill_0 (PD).
    leader_entry = pids.get("prefill_0" if plan.disaggregated else "rank_0")
    _log_rank0_post_spawn(Path(args.log_dir), leader_entry[1] if leader_entry else None)

    summary = _launch_summary(args, plan=plan, heads=heads, pids=pids)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()

    if args.no_wait_health:
        _log("--no-wait-health set; not probing /health")
        return 0
    if plan.disaggregated:
        return _probe_pd_health(args, summary)
    return _probe_rank0_health(args)


if __name__ == "__main__":
    sys.exit(main())
