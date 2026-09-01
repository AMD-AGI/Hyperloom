#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node sglang / vllm server launcher, run INSIDE the RayJob head pod.

Waits for ``nnodes`` alive GPU nodes, makes the local (head) node rank 0
and ranks the rest by NodeManagerAddress, then spawns one NodeAffinity-
pinned actor per rank that launches the framework detached via
bash+nohup+setsid (avoiding zombie PIDs / empty logs) and records
``<pid_dir>/rank_<K>.pid``. Optionally waits on rank-0 ``/health``
(``--no-wait-health`` to skip). Single-node restarts use the bash path.

In ``--pd-mode disaggregated`` two groups are spawned instead — prefill over
``nodes[0:pn]`` and decode over ``nodes[pn:]`` — recording
``prefill_<K>.pid`` / ``decode_<K>.pid`` on internal ports 30000 / 30001. The
rank-0 probe is replaced by a wait on both legs' ``/health``; the caller fronts
them with ``launch_router.py`` on 8888 using the endpoints in the JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import pathlib
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Inference port. Bound by rank 0 in aggregated mode, by the router in
# disaggregated mode (which proxies to the internal prefill/decode ports).
_INFERENCE_PORT = 8888
# Loopback-only internal ports for disaggregated PD server groups (router
# fronts them at _INFERENCE_PORT so the external URL is mode-independent).
_PD_PREFILL_PORT = 30000
_PD_DECODE_PORT = 30001
# Default collective port ($RAYJOB_DIST_INIT_PORT else 29500).
_DEFAULT_DIST_INIT_PORT = 29500
# vLLM PD connectors named in ``--pd-transfer-backend`` help. Unknown names
# still serialize (so a future vLLM connector is not blocked) but warn.
_VLLM_KV_CONNECTORS: frozenset[str] = frozenset(
    (
        "NixlConnector",
        "P2pNcclConnector",
        "MooncakeConnector",
        "LMCacheConnectorV1",
    )
)


def _pd_decode_dist_init_port(prefill_dist_init_port: int) -> int:
    """Derive the decode rendezvous port as ``prefill + 1``.

    Args:
        prefill_dist_init_port: The prefill group's rendezvous port.

    Returns:
        int: The decode group's rendezvous port.
    """
    return prefill_dist_init_port + 1


# sglang PD bootstrap (KV transfer rendezvous) port; override via --pd-bootstrap-port.
_PD_DEFAULT_BOOTSTRAP_PORT = 8998
# Max seconds to wait for ray.nodes() to surface every expected pod; raise it
# when pods queue behind a busy scheduler or a cold image pull.
_NODES_DISCOVERY_TIMEOUT_SEC = int(os.environ.get("RAY_NODES_DISCOVERY_TIMEOUT_SEC", "120"))
# rank-0 /health probe budget (cold MoE can exceed it; --no-wait-health to bypass).
_HEALTH_PROBE_TIMEOUT_SEC = int(os.environ.get("SGLANG_HEALTH_PROBE_TIMEOUT_SEC", "1800"))

# Keep in sync with multi_node/_internal/server_args_safety.py
_DENIED_SERVER_FLAGS = frozenset(
    {
        "--adapter-model-path",
        "--adapter-path",
        "--allowed-local-media-path",
        "--chat-template",
        "--code-revision",
        "--config",
        "--download-dir",
        "--hf-overrides",
        "--lora-dirs",
        "--lora-modules",
        "--lora-path",
        "--lora-paths",
        "--model",
        "--model-id",
        "--model-path",
        "--quantization-param-path",
        "--revision",
        "--tokenizer",
        "--tokenizer-path",
        "--tokenizer-revision",
    }
)
_DENIED_SERVER_FLAG_SUFFIXES = ("-dir", "-file", "-path")
# Tuning knobs exempt from the suffix rule by name only; their values stay
# constrained by _unsafe_path_value_reason.
_SUFFIX_EXEMPT_SERVER_FLAGS = frozenset({"--speculative-draft-model-path"})


def _is_denied_server_flag(flag: str) -> bool:
    """Return whether a single ``--flag`` token is denied at the pod boundary."""
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_SERVER_FLAGS:
        return True
    if name in _SUFFIX_EXEMPT_SERVER_FLAGS:
        return False
    return any(name.endswith(suffix) for suffix in _DENIED_SERVER_FLAG_SUFFIXES)


def _unsafe_path_value_reason(value: str | None) -> str:
    """Return why an exempt flag's path value is unsafe ("" when acceptable)."""
    val = (value or "").strip()
    if not val:
        return "missing value"
    if not val.startswith("/"):
        return "must be an absolute path, not a repo id or URI"
    if ".." in PurePosixPath(val).parts:
        return "must not traverse with '..'"
    return ""


def _flag_value_pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    """Return ``(flag, value)`` pairs for both ``--flag=value`` and ``--flag value``."""
    pairs: list[tuple[str, str | None]] = []
    for idx, tok in enumerate(tokens):
        if not tok.startswith("--"):
            continue
        if "=" in tok:
            name, _, val = tok.partition("=")
            pairs.append((name, val))
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        pairs.append((tok, None if (nxt is None or nxt.startswith("--")) else nxt))
    return pairs


def _denied_extra_args(raw: str) -> list[str]:
    """Return rejected CLI flags in a pod-side extra-args string.

    Args:
        raw: Whitespace-separated server flags.

    Returns:
        list[str]: Denied flag names, plus ``"flag: reason"`` entries for exempt
        flags whose value is outside the allowed path shape (empty when clean).
    """
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return ["<unparseable>"]
    out: list[str] = []
    for flag, value in _flag_value_pairs(tokens):
        if _is_denied_server_flag(flag):
            if flag not in out:
                out.append(flag)
            continue
        if flag not in _SUFFIX_EXEMPT_SERVER_FLAGS:
            continue
        reason = _unsafe_path_value_reason(value)
        entry = f"{flag}: {reason}"
        if reason and entry not in out:
            out.append(entry)
    return out


def _log(msg: str) -> None:
    """Write a timestamped launcher log line to stderr.

    Args:
        msg: The message text to emit.
    """
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
    """Reorder so the local (driver) node is rank 0, rest by NodeManagerAddress.

    Args:
        nodes: The alive GPU node rows from ``ray.nodes()``.

    Returns:
        list[dict]: The nodes reordered with the local node first, or all
        nodes sorted by address when the driver is not on a GPU node.
    """
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
    ``--trust-remote-code`` is default-on. ``ep > 1`` emits
    ``--expert-parallel-size N`` for expert parallelism.

    Args:
        model: Model path passed to the launcher.
        tp: Tensor-parallel size.
        nnodes: Total node count for this server group.
        node_rank: This node's rank within the group.
        dist_init_addr: Collective rendezvous ``host:port``.
        extra_args: Extra args appended verbatim to the launcher.
        ep: Expert-parallel size; ``> 1`` enables expert parallelism.
        pd_role: PD disaggregation role (``prefill``/``decode``) or empty.
        pd_port: HTTP port bound by rank 0.
        pd_transfer_backend: KV-transfer backend for PD mode.
        pd_ib_device: IB/RoCE device list for PD KV transfer.
        pd_bootstrap_port: sglang PD bootstrap rendezvous port.

    Returns:
        list[str]: The sglang launch argv.
    """
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--trust-remote-code",
        "--tp",
        str(tp),
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        dist_init_addr,
    ]
    if node_rank == 0:
        # rank-0 HTTP port = internal prefill/decode port in PD mode, else public.
        cmd.extend(["--host", "0.0.0.0", "--port", str(pd_port)])  # nosec B104 - rank-0 server must accept pod traffic.
    if ep > 1:
        cmd.extend(["--expert-parallel-size", str(ep)])
    role = pd_role.strip().lower()
    if role in ("prefill", "decode"):
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
    """vLLM multi-node command for rank 0 (workers are KubeRay-joined and vLLM auto-discovers them via the GCS).

    ``--tensor-parallel-size`` = total cluster GPUs; ``ep > 1`` adds
    ``--enable-expert-parallel``.

    Args:
        model: Model path passed to ``vllm serve``.
        tp: Tensor-parallel size (total cluster GPUs).
        extra_args: Extra args appended verbatim to the launcher.
        ep: Expert-parallel size; ``> 1`` enables expert parallelism.
        pd_role: PD disaggregation role (``prefill``/``decode``) or empty.
        pd_port: HTTP port bound by the server.
        pd_transfer_backend: KV connector name for PD mode.
        pd_kv_rank: This group's KV rank in PD mode.
        pd_kv_parallel_size: Total KV parallel size in PD mode.

    Returns:
        list[str]: The vLLM launch argv.
    """
    cmd = [
        "vllm",
        "serve",
        model,
        "--tensor-parallel-size",
        str(tp),
        "--host",
        "0.0.0.0",  # nosec B104 - vLLM rank-0 server must accept pod traffic.
        "--port",
        str(pd_port),
        "--distributed-executor-backend",
        "ray",
    ]
    if ep > 1:
        cmd.append("--enable-expert-parallel")
    role = pd_role.strip().lower()
    if role in ("prefill", "decode"):
        # vllm PD: kv-transfer-config JSON with connector + role + kv slot.
        kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
        connector = pd_transfer_backend or "NixlConnector"
        if connector not in _VLLM_KV_CONNECTORS:
            _log(f"WARN unknown vLLM kv connector {connector!r}; known: {sorted(_VLLM_KV_CONNECTORS)}")
        kv_cfg = json.dumps(
            {
                "kv_connector": connector,
                "kv_role": kv_role,
                "kv_rank": int(pd_kv_rank),
                "kv_parallel_size": int(pd_kv_parallel_size),
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
    """Return True iff rocm-smi reports MEC firmware < 177 (gates the HSA_NO_SCRATCH_RECLAIM workaround); False on any failure.

    Returns:
        bool: ``True`` if MEC firmware is below 177; ``False`` on any probe
        failure or higher firmware.
    """
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
    """Build the framework launcher subprocess env.

    Puts ``/opt/venv/bin`` first on PATH (framework venv) and inherits
    ``os.environ`` (keeps injected API keys / SGLANG_* / VLLM_* tunings).
    Sets the MI300X tuning trio (SGLANG_USE_AITER / SGLANG_AITER_MLA_PERSIST,
    and HSA_NO_SCRATCH_RECLAIM when MEC firmware < 177) without clobbering
    caller-set values.

    Returns:
        dict[str, str]: The environment mapping for the launcher subprocess.
    """
    env = dict(os.environ)
    # Strip Ray's empty *_VISIBLE_DEVICES mask so the detached framework child
    # re-discovers all GPUs; honour non-empty overrides.
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
    """Start ``cmd`` detached from the Ray worker via bash+nohup+setsid (reparents under init; fails fast with a log tail).

    Args:
        cmd: The framework launcher argv.
        log_file: Path the launcher's stdout/stderr is appended to.
        pid_file: Path the spawned launcher PID is written to.
        sub_env: Environment passed to the spawn shell.
        node_rank: This node's rank, used in log/error messages.

    Returns:
        int: The PID of the detached framework process.

    Raises:
        RuntimeError: If the spawn shell fails, the PID file is missing or
            invalid, or the process exits within 0.5s of launch.
    """
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
    """Spawn the framework launcher detached on the local actor's pod, recording the PID.

    ``torch_profiler_dir`` (if set) is exported as
    ``SGLANG_TORCH_PROFILER_DIR`` for shared-path traces.

    Args:
        framework: ``sglang`` or ``vllm``.
        model: Model path passed to the launcher.
        tp: Tensor-parallel size.
        nnodes: Total node count for this server group.
        node_rank: This node's rank within the group.
        head_ip: Rendezvous head IP.
        dist_init_port: Collective rendezvous port.
        pid_dir: Directory the PID file is written to.
        log_dir: Directory the log file is written to.
        extra_args: Extra args appended verbatim to the launcher.
        torch_profiler_dir: Optional shared dir exported as
            ``SGLANG_TORCH_PROFILER_DIR``.
        ep: Expert-parallel size.
        pd_role: PD disaggregation role (``prefill``/``decode``) or empty.
        pd_port: HTTP port bound in PD mode.
        pd_transfer_backend: KV-transfer backend for PD mode.
        pd_ib_device: IB/RoCE device list for PD KV transfer.
        pd_bootstrap_port: sglang PD bootstrap rendezvous port.
        pd_kv_rank: vLLM KV rank for PD mode.
        pd_kv_parallel_size: vLLM total KV parallel size for PD mode.
        pid_file_name: Optional role-tagged PID file name.

    Returns:
        int: The spawned framework PID, or ``0`` for a no-op vLLM worker rank.

    Raises:
        RuntimeError: If ``framework`` is unsupported, or the detached launch
            fails.
    """
    Path(pid_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    # PD mode runs two groups with overlapping rank numbers; caller passes
    # a role-tagged pid_file_name so they don't collide on disk.
    fname = pid_file_name or f"rank_{node_rank}.pid"
    log_fname = f"{Path(fname).stem}.log" if pid_file_name else f"rank_{node_rank}.log"
    pid_file = Path(pid_dir) / fname
    log_file = Path(log_dir) / log_fname

    sub_env = _subprocess_env()
    # Fall back to $HYPERLOOM_MN_PROFILE_TRACE_DIR so traces still reach a shared
    # dir when a reused server skips this launch.
    tpd = (torch_profiler_dir or "").strip() or os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
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
            model=model,
            tp=tp,
            nnodes=nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            extra_args=extra_args,
            ep=ep,
            pd_role=pd_role,
            pd_port=pd_port,
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
            model=model,
            tp=tp,
            extra_args=extra_args,
            ep=ep,
            pd_role=pd_role,
            pd_port=pd_port,
            pd_transfer_backend=pd_transfer_backend,
            pd_kv_rank=pd_kv_rank,
            pd_kv_parallel_size=pd_kv_parallel_size,
        )
    else:
        raise RuntimeError(f"unsupported framework: {framework!r}")

    sys.stderr.write(f"[rank {node_rank}] launching: {' '.join(cmd)}\n")
    sys.stderr.write(f"[rank {node_rank}] log={log_file} pid={pid_file}\n")
    return _detach_framework_launch(cmd, log_file, pid_file, sub_env, node_rank)


# Budget for one node-pinned rollback kill (SIGTERM + report; no death-wait).
_ROLLBACK_TIMEOUT_SEC = 30


def _rollback_remote(pid: int) -> str:
    """SIGTERM a spawned rank's process group, ON the node that owns the PID.

    Must run as a NodeAffinity-pinned actor on that rank's own node: the PID is
    resolvable only in the namespace that produced it.

    Args:
        pid: The PID reported by that node's spawn actor.

    Returns:
        str: A short human-readable outcome for the driver's log.
    """
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
    """Read rank 0 PID from ``{pid_dir}/rank_0.pid`` (best-effort).

    The PID dir is passed explicitly rather than derived from the log dir: the
    two are no longer siblings once the logs move to a shared filesystem, and a
    PID is only meaningful on its own node so its dir stays node-local ``/tmp``.

    Args:
        pid_dir (str): The rank PID directory; ``{pid_dir}/rank_0.pid`` is read
            before falling back to the default temp-directory location.

    Returns:
        int | None: The rank-0 PID, or ``None`` if it cannot be read.
    """
    try:
        pid_path = pathlib.Path(pid_dir) / "rank_0.pid"
        if not pid_path.is_file():
            pid_path = pathlib.Path(tempfile.gettempdir()) / "multi_node_pids" / "rank_0.pid"
        return int(pid_path.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


# Fatal patterns scanned in rank_0.log (catches crashes the lingering nohup
# wrapper PID hides).
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
    """Scan a log file's tail for a fatal traceback / framework error.

    Args:
        log_file: Path to the log file to scan.

    Returns:
        str | None: The first matched fatal line, or ``None`` if none found
        or the log cannot be read.
    """
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
    """Scan rank_0.log tail for a fatal traceback / framework error.

    Args:
        log_dir: Directory containing ``rank_0.log``.

    Returns:
        str | None: The first matched fatal line, or ``None`` if none found
        or the log cannot be read.
    """
    return _scan_log_for_fatal(Path(log_dir) / "rank_0.log")


def _wait_health(
    timeout_s: int,
    rank0_pid: int | None = None,
    log_dir: str | None = None,
) -> bool:
    """Poll rank-0 ``/health``; True on first 200, else False on timeout, rank-0 death, or a fatal ``rank_0.log`` error.

    Args:
        timeout_s: Maximum seconds to poll before giving up.
        rank0_pid: Optional rank-0 PID; the wait aborts early if it dies.
        log_dir: Optional directory whose ``rank_0.log`` is scanned for fatal
            errors to abort early.

    Returns:
        bool: ``True`` on a first healthy response; ``False`` on timeout,
        confirmed rank-0 death, or a fatal log error.
    """
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
    """Poll both PD legs' own ``/health`` until each answers, within one budget.

    The router that will own the public port is submitted by the caller only
    after this driver returns, so there is nothing to probe there yet -- but the
    legs answer on their own ports, and they are what "the cluster is up" means.

    Returning before that made this job's terminal status meaningless: it went
    SUCCEEDED seconds after the ranks were spawned, while the weight load still
    had tens of minutes to run, so a caller retrying mid-boot could not tell a
    booting cluster from a dead one.

    Liveness is decided by ``/health`` (reachable over the network) and each
    leg's own log, never by a PID. This driver runs on the head, but the decode
    group's leader runs on a *different* node, so its PID lives in that node's
    namespace: ``os.kill`` here would raise ``ProcessLookupError`` for a
    perfectly healthy remote leg and false-fail the launch. The aggregated path
    keeps its PID check only because rank 0 is co-located with this driver.

    Args:
        prefill_url: Base URL of the prefill group's rank 0.
        decode_url: Base URL of the decode group's rank 0.
        timeout_s: Shared budget for both legs.
        log_dir: Optional directory holding ``{role}_0.log``; each leg's own log
            is scanned for a fatal error to abort early, and its tail is emitted
            on that abort so the caller can see why the leg died.

    Returns:
        bool | None: True once both legs answer, False when a leg's log shows a
        fatal crash, None on timeout (undetermined, as for rank 0).
    """
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
            # A remote leg's PID is not ours to os.kill; a fatal in its own log is
            # the cross-node-safe proof of death (the nohup wrapper hides crashes).
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
    """Append the tail of a log file to stderr if it exists.

    Seeks to the last ``_LOG_TAIL_BYTES`` rather than reading the whole file:
    the logs now live on a shared filesystem, and slurping a multi-hundred-MB
    server log over the network on the failure path (both PD legs) would be a
    heavy read for an 8 KiB tail.

    Args:
        log_file (Path): Path to the log file to tail.
    """
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
    """Append the tail of ``rank_0.log`` to stderr if the file exists.

    Args:
        log_dir (Path): Directory containing ``rank_0.log``.
    """
    _emit_log_tail(log_dir / "rank_0.log")


def _log_rank0_post_spawn(log_dir: Path, rank0_pid: int | None) -> None:
    """Emit rank-0 diagnostics after a short settle (a ``rank_0.log`` tail).

    Args:
        log_dir: Directory containing ``rank_0.log``.
        rank0_pid: The rank-0 PID to probe, or ``None`` to skip.
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
    """Parse CLI arguments and launch the multi-node server group(s).

    Connects to the in-pod Ray cluster, discovers and rank-orders nodes,
    spawns one launcher actor per rank (aggregated or PD-disaggregated),
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
    # PD disaggregation args: `aggregated` is one TP group; `disaggregated` splits
    # into prefill/decode groups fronted by the router.
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
    args = p.parse_args()

    if args.nnodes < 2:
        _log(
            f"nnodes={args.nnodes} < 2; this script is for multi-node only. "
            f"Use launch_server.sh for single-pod restarts."
        )
        return 2

    # Validate PD args; populate defaults.
    pd_mode = (args.pd_mode or "aggregated").lower()
    if pd_mode == "disaggregated":
        pn = int(args.pd_prefill_nodes or 0)
        dn = int(args.pd_decode_nodes or 0)
        if pn <= 0 or dn <= 0 or pn + dn != args.nnodes:
            _log(
                f"PD invalid split: pd_prefill_nodes={pn} pd_decode_nodes={dn} "
                f"nnodes={args.nnodes}; require pn+dn==nnodes and both >0"
            )
            return 2
        ptp = int(args.pd_prefill_tp or args.tp)
        dtp = int(args.pd_decode_tp or args.tp)
        if ptp <= 0 or dtp <= 0:
            _log(f"PD invalid TP: pd_prefill_tp={ptp} pd_decode_tp={dtp}")
            return 2
        ib_dev = args.pd_ib_device or os.environ.get("NCCL_IB_HCA", "")
        ib_dev = ib_dev.strip()
    else:
        pn = args.nnodes
        dn = 0
        ptp = args.tp
        dtp = 0
        ib_dev = ""

    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    denied = _denied_extra_args(args.extra_args)
    if denied:
        _log(f"ERROR denied server flags in --extra-args: {denied}")
        return 2

    _log(f"framework={args.framework} model={args.model} tp={args.tp} nnodes={args.nnodes}")

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = _wait_for_nodes(args.nnodes, _NODES_DISCOVERY_TIMEOUT_SEC)
    nodes = _pick_head_first(nodes)
    nodes = nodes[: args.nnodes]
    head_ip = ray.util.get_node_ip_address()
    _log(f"discovered {len(nodes)} GPU nodes; head_ip={head_ip}; rank order:")
    for k, n in enumerate(nodes):
        _log(
            f"  rank {k}: node_id={n.get('NodeID', '?')[:16]}... "
            f"addr={n.get('NodeManagerAddress', '?')} "
            f"gpu={n.get('Resources', {}).get('GPU', 0)}"
        )

    # Spawn one actor per rank (num_gpus=0; the framework reserves GPUs itself).
    SpawnActor = ray.remote(num_cpus=1, num_gpus=0)(_spawn_remote)
    # tag -> (node_id, real PID). A PID is only meaningful in the namespace that
    # produced it, and this driver shares one only with rank 0 (_pick_head_first),
    # so every PID travels with the node that owns it.
    pids: dict[str, tuple[str, int]] = {}
    refs: list[tuple[str, str, Any]] = []  # (tag, node_id, actor_ref)

    if pd_mode == "disaggregated":
        # Group A: prefill — nodes[0:pn], TP=ptp; rank-0 binds the prefill port.
        prefill_head_ip = nodes[0].get("NodeManagerAddress", head_ip)
        for grp_rank in range(pn):
            node = nodes[grp_rank]
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                ),
            ).remote(
                framework=args.framework,
                model=args.model,
                tp=ptp,
                nnodes=pn,
                node_rank=grp_rank,
                head_ip=prefill_head_ip,
                dist_init_port=args.dist_init_port,
                pid_dir=args.pid_dir,
                log_dir=args.log_dir,
                extra_args=extra_args,
                torch_profiler_dir=args.torch_profiler_dir,
                ep=int(args.ep or 1),
                pd_role="prefill",
                pd_port=_PD_PREFILL_PORT,
                pd_transfer_backend=args.pd_transfer_backend,
                pd_ib_device=ib_dev,
                pd_bootstrap_port=args.pd_bootstrap_port,
                pd_kv_rank=0,  # vllm-only; sglang ignores
                pd_kv_parallel_size=2,
                pid_file_name=f"prefill_{grp_rank}.pid",
            )
            refs.append((f"prefill_{grp_rank}", node["NodeID"], actor_ref))

        # Group B: decode — nodes[pn:pn+dn]; dist-init port = prefill + 1.
        decode_head_ip = nodes[pn].get("NodeManagerAddress", head_ip)
        for grp_rank in range(dn):
            node = nodes[pn + grp_rank]
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                ),
            ).remote(
                framework=args.framework,
                model=args.model,
                tp=dtp,
                nnodes=dn,
                node_rank=grp_rank,
                head_ip=decode_head_ip,
                dist_init_port=_pd_decode_dist_init_port(args.dist_init_port),
                pid_dir=args.pid_dir,
                log_dir=args.log_dir,
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
            refs.append((f"decode_{grp_rank}", node["NodeID"], actor_ref))
    else:
        # Aggregated: single server group spans all nodes.
        for rank, node in enumerate(nodes):
            actor_ref = SpawnActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                ),
            ).remote(
                framework=args.framework,
                model=args.model,
                tp=args.tp,
                nnodes=args.nnodes,
                node_rank=rank,
                head_ip=head_ip,
                dist_init_port=args.dist_init_port,
                pid_dir=args.pid_dir,
                log_dir=args.log_dir,
                extra_args=extra_args,
                torch_profiler_dir=args.torch_profiler_dir,
                ep=int(args.ep or 1),
            )
            refs.append((f"rank_{rank}", node["NodeID"], actor_ref))

    RollbackActor = ray.remote(num_cpus=0, num_gpus=0)(_rollback_remote)
    for tag, node_id, ref in refs:
        try:
            pid = ray.get(ref, timeout=120)
            pids[tag] = (node_id, pid)
            _log(f"{tag}: spawned pid={pid}")
        except Exception as exc:  # noqa: BLE001
            _log(f"{tag}: spawn FAILED: {type(exc).__name__}: {exc}")
            # Roll back already-spawned ranks so no half-started servers leak.
            # Each kill is dispatched to the node that owns the PID; PIDs are
            # namespace-local and are never signalled from here. A pid <= 0 is
            # the vLLM worker sentinel and names no process -- os.getpgid(0)
            # would resolve to this driver's own process group.
            for tag2, (node_id2, p2) in pids.items():
                if p2 <= 0:
                    _log(f"rolling back {tag2}: sentinel pid={p2} (no process); skipped")
                    continue
                try:
                    outcome = ray.get(
                        RollbackActor.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                node_id=node_id2,
                                soft=False,
                            ),
                        ).remote(p2),
                        timeout=_ROLLBACK_TIMEOUT_SEC,
                    )
                    _log(f"rolling back {tag2} pid={p2} on node={node_id2[:16]}...: {outcome}")
                except Exception as rb_exc:  # noqa: BLE001
                    # An unreachable rank keeps its GPUs until the platform
                    # reclaims the cluster, so it is reported rather than dropped.
                    _log(
                        f"WARN rollback of {tag2} pid={p2} on node={node_id2[:16]}... "
                        f"FAILED: {type(rb_exc).__name__}: {rb_exc}; that rank may "
                        f"still hold GPU memory"
                    )
            return 1

    # Health-tail probe targets the leader: rank_0 (aggregated) / prefill_0 (PD).
    leader_tag = "rank_0" if pd_mode == "aggregated" else "prefill_0"
    _leader_entry = pids.get(leader_tag)
    _log_rank0_post_spawn(Path(args.log_dir), _leader_entry[1] if _leader_entry else None)

    summary: dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "tp": args.tp,
        "ep": int(args.ep or 1),
        "nnodes": args.nnodes,
        "head_ip": head_ip,
        "dist_init_port": args.dist_init_port,
        "ranks": [{"tag": r, "node_id": pids[r][0], "pid": pids[r][1]} for r in sorted(pids)],
        "pid_dir": args.pid_dir,
        "log_dir": args.log_dir,
        "inference_port": _INFERENCE_PORT,
        "pd_mode": pd_mode,
    }
    if pd_mode == "disaggregated":
        # Emit internal endpoints + bootstrap port so the CLI can submit the
        # router without re-discovering nodes.
        summary["pd_prefill_nodes"] = pn
        summary["pd_decode_nodes"] = dn
        summary["pd_prefill_tp"] = ptp
        summary["pd_decode_tp"] = dtp
        summary["pd_transfer_backend"] = args.pd_transfer_backend or (
            "NixlConnector" if args.framework.lower() == "vllm" else "mooncake"
        )
        summary["pd_ib_device"] = ib_dev
        summary["pd_bootstrap_port"] = args.pd_bootstrap_port
        summary["pd_prefill_url"] = f"http://{prefill_head_ip}:{_PD_PREFILL_PORT}"
        summary["pd_decode_url"] = f"http://{decode_head_ip}:{_PD_DECODE_PORT}"
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()

    if args.no_wait_health:
        _log("--no-wait-health set; not probing /health")
        return 0

    # PD: the router that owns 8888 does not exist yet (the caller submits it
    # once this driver returns), so wait on the legs themselves. Doing this is
    # what lets a terminal status here mean the cluster served.
    if pd_mode == "disaggregated":
        _log(f"polling both PD legs' /health for up to {_HEALTH_PROBE_TIMEOUT_SEC}s")
        _legs_ready = _wait_pd_legs_health(
            summary["pd_prefill_url"],
            summary["pd_decode_url"],
            _HEALTH_PROBE_TIMEOUT_SEC,
            log_dir=args.log_dir,
        )
        if _legs_ready:
            _log("both PD legs /health OK")
            return 0
        # PD has no rank_0.log; tail each leg's own log so the caller sees why.
        _emit_log_tail(Path(args.log_dir) / "prefill_0.log")
        _emit_log_tail(Path(args.log_dir) / "decode_0.log")
        if _legs_ready is False:
            _log("ERROR a PD leg logged a fatal crash before serving; marking the launch failed")
            return 2
        # Undetermined, as for rank 0: warn rather than fail a slow-but-live boot.
        _log("WARN PD legs unconfirmed; the caller's serving probe decides from here")
        return 0

    _log(f"polling rank 0 /health for up to {_HEALTH_PROBE_TIMEOUT_SEC}s")
    _r0_pid = _rank0_pid_from_log(args.pid_dir)
    if _wait_health(
        _HEALTH_PROBE_TIMEOUT_SEC,
        rank0_pid=_r0_pid,
        log_dir=args.log_dir,
    ):
        _log("rank 0 /health OK")
        return 0
    # Tail the log (timeout or early rank-0 death) for the framework's last words.
    _emit_rank0_log_tail(Path(args.log_dir))
    # Tri-state liveness: True=alive, False=confirmed dead, None=unknown. Only
    # confirmed death flips to FAILED (return 2); alive/unknown return 0 + WARN.
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
        sys.stderr.write(f"MULTI_NODE_FAILURE_SNAPSHOT={json.dumps(snap)}\n")
        sys.stderr.flush()
        _log(
            f"ERROR rank 0 pid={_r0_pid} dead before /health; returning 2 "
            f"so the Ray Dashboard job reports FAILED and hyperloom "
            f"surfaces ServerRestartFailed immediately (was: silent "
            f"1800s /health stall)."
        )
        return 2
    # A fatal traceback in rank_0.log proves a crash even when the nohup
    # wrapper PID lingers.
    _fatal_line = _scan_rank0_log_for_fatal(args.log_dir)
    if _fatal_line:
        snap = {
            "kind": "framework_error",
            "rank0_pid": _r0_pid,
            "rank0_alive": _r0_alive,
            "hint": f"rank_0.log contains fatal: {_fatal_line[:500]}",
        }
        sys.stderr.write(f"MULTI_NODE_FAILURE_SNAPSHOT={json.dumps(snap)}\n")
        sys.stderr.flush()
        _log(
            f"ERROR rank 0 framework error in rank_0.log (pid={_r0_pid} "
            f"alive={_r0_alive}); returning 2 so Ray Dashboard job reports "
            f"FAILED in seconds instead of 1800s silent stall."
        )
        return 2
    _log(
        f"WARN rank 0 /health did not pass within {_HEALTH_PROBE_TIMEOUT_SEC}s; "
        f"rank-0 pid={_r0_pid} alive={_r0_alive} (None=pid unknown) — "
        f"server likely still loading weights; the caller's external poll "
        f"will catch up."
    )
    return 0  # Don't fail; caller polls health from sandbox


if __name__ == "__main__":
    sys.exit(main())
