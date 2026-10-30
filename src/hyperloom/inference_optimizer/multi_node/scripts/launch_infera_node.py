#!/usr/bin/env python3
"""Pod-side launcher for the Infera multi-node backend (idle-pod SSH mode)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from server_args_safety import find_denied_flags, find_unsafe_flag_values
from sglang_shape_gate import activate_kernel_shape_tool

# Rendezvous port for torch.distributed (matches SaFE common.InferaMultinodeDistInitPort = 5000).
_DEFAULT_DIST_INIT_PORT = 5000
# Ray GCS port for the vllm multi-node bootstrap.
_RAY_GCS_PORT = 6379
# Default HTTP port the infera engine binds via --port (no CLI override is declared).
_DEFAULT_ENGINE_PORT = 30000


def _log(msg: str) -> None:
    """Write a timestamped launcher log line to stderr."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_infera_node {ts}] {msg}\n")
    sys.stderr.flush()


# Env-var prefixes/names worth recovering from pid1 so the framework child sees the same rendezvous / discovery /
# tuning config the container was started with.
_ENV_RECOVER_PREFIXES = (
    "LWS_",
    "POD_",
    "NCCL_",
    "GLOO_",
    "RCCL_",
    "INFERA_",
    "SGLANG_",
    "VLLM_",
    "NATS_",
    "HSA_",
    "HIP_",
    "ROCR_",
    "HF_",
    "NATS_",
    "UCX_",
    "NIXL_",
    "MC_",
    # KUBERNETES_* must propagate too, otherwise infera's kubernetes discovery backend fails with "Failed to create
    # Kubernetes client: failed to infer config: in-cluster: (environment variable not found)" immediately on
    # infera.engine.sglang/infera.engine.vllm start, and the SSH-launched server exits in <1s while the Infera
    # frontend (always-up) keeps returning /health 200 — causing baseline_failed with 0 completed requests and no
    # obvious sandbox-side log evidence.
    "KUBERNETES_",
)
_ENV_RECOVER_NAMES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV")


def _recover_container_env() -> dict[str, str]:
    """Merge the current env with pid1's env for the recovered keys."""
    env = dict(os.environ)
    try:
        raw = Path("/proc/1/environ").read_bytes()
    except OSError as exc:
        _log(f"WARN cannot read /proc/1/environ: {exc}; using sshd session env")
        return env
    for chunk in raw.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        k, _, v = chunk.partition(b"=")
        key = k.decode("utf-8", "ignore")
        val = v.decode("utf-8", "ignore")
        if key in _ENV_RECOVER_NAMES or any(key.startswith(p) for p in _ENV_RECOVER_PREFIXES):
            # pid1 wins for rendezvous/discovery; but keep sshd's PATH augmented with /opt/venv/bin so python3
            # resolves to the framework venv.
            env[key] = val
    venv_bin = "/opt/venv/bin"
    parts = env.get("PATH", "").split(":") if env.get("PATH") else []
    if venv_bin not in parts:
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}".rstrip(":")
    return env


def _resolve_pod_ip(env: dict[str, str]) -> str:
    """Return this pod's routable IP (never a loopback address)."""
    import socket

    cand = (env.get("POD_IP") or "").strip()
    if cand and not cand.startswith("127."):
        return cand
    # Egress-route probe: connecting a UDP socket sends no packets but makes the kernel pick the source IP of the
    # default-route interface.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` exists (signal 0 probe). PermissionError => exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _proc_tree(root: int) -> list[int]:
    """Return ``root`` plus all descendant PIDs via /proc ppid links."""
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return [root]
    for e in entries:
        if not e.isdigit():
            continue
        try:
            with open("/proc/" + e + "/stat", "rb") as fh:
                data = fh.read()
            after = data.rsplit(b")", 1)[1].split()
            ppid = int(after[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(e))
    out: list[int] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        stack.extend(children.get(x, []))
    return out


def _group_alive(pgid: int | None) -> bool:
    """True while process group ``pgid`` has a member this process can still signal."""
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _term_then_kill(pids: list[int], kill_wait_s: float, pgid: int | None = None) -> list[int]:
    """SIGTERM, then SIGKILL, ``pids`` and process group ``pgid`` until all of them are gone; return the surviving pids."""

    def _any_alive() -> bool:
        return any(_pid_alive(pid) for pid in pids) or _group_alive(pgid)

    for sig, wait_s in ((signal.SIGTERM, 5.0), (signal.SIGKILL, kill_wait_s)):
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, sig)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and _any_alive():
            time.sleep(0.5)
        if not _any_alive():
            break
    return [pid for pid in pids if _pid_alive(pid)]


def _reap_stale_engines_by_cmdline(kill_wait_s: float) -> None:
    """SIGKILL any residual sglang / infera engine process matched by cmdline via /proc, before a fresh launch."""
    pats = ("sglang.launch_server", "infera.engine.sglang", "infera.engine.vllm", "sglang::sched")
    me = os.getpid()
    victims: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return
    for e in entries:
        if not e.isdigit():
            continue
        xpid = int(e)
        if xpid <= 1 or xpid == me:
            continue
        try:
            with open("/proc/" + e + "/cmdline", "rb") as fh:
                cl = fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except (OSError, ValueError):
            continue
        if any(pat in cl for pat in pats):
            victims.append(xpid)
    if not victims:
        return
    remaining = _term_then_kill(victims, kill_wait_s)
    if remaining:
        _log("reaper: stale engine pids still alive after SIGKILL: " + str(remaining))
    else:
        _log("reaper: killed stale engine pids " + str(victims))


def _kill_prior(pid_file: Path) -> None:
    """SIGTERM then SIGKILL the prior server's whole process tree, then sweep any residual holder of the managed sglang engine ports."""
    kill_wait_s = float(os.environ.get("HYPERLOOM_MN_KILL_WAIT_S", "120") or 120)
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid = 0
    if pid > 0 and _pid_alive(pid):
        targets = _proc_tree(pid)
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None
        alive = _term_then_kill(targets, kill_wait_s, pgid)
        if alive:
            raise RuntimeError(
                f"prior server pids {alive} still alive {int(kill_wait_s)}s after SIGKILL (wedged, likely D-state on "
                "slow weight I/O); aborting relaunch to avoid VRAM double-stack -- node may need a reboot/GPU reset"
            )
        _log(f"killed prior server tree pid={pid} ({len(targets)} procs)")
    pid_file.unlink(missing_ok=True)
    # Kill residual engine processes by cmdline (catches an orphaned launch_server holding port_base that ss/tree-kill
    # miss), then sweep ports.
    _reap_stale_engines_by_cmdline(kill_wait_s)
    _reap_stale_engine_ports(kill_wait_s)


# The infera decode/prefill restart records the infera.engine wrapper PID in the pid-file, but the real
# sglang.launch_server it spawns becomes its own process-group leader and holds the engine ports (30000 HTTP, 30234
# dist port_base, 30001 bootstrap, 32760 kv-events). _kill_prior's pgid kill therefore misses it, so the NEXT restart
# crashes with "port_base at 30234 is not available" and every explore variant aborts at the health timeout.
_REAP_PORTS = (30000, 30001, 30234, 32760)


def _reap_stale_engine_ports(kill_wait_s: float) -> None:
    """Kill any residual process group still holding the sglang engine ports, waiting for the whole group to exit."""
    for port in _REAP_PORTS:
        try:
            out = subprocess.run(
                ["/bin/bash", "-lc", f"ss -ltnp 2>/dev/null | grep -w ':{port}' || true"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for pid in sorted({int(m) for m in re.findall(r"pid=(\d+)", out or "")} - {0, 1}):
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            _log(f"reaper: port {port} held by pid={pid} pgid={pgid}; killing")
            if not _term_then_kill([pid], kill_wait_s, pgid) and not _group_alive(pgid):
                _log(f"reaper: freed port {port} (pid={pid})")


def _build_sglang_cmd(
    a: argparse.Namespace,
    node_rank: int,
    leader: str,
    *,
    advertise_host: str,
) -> list[str]:
    """infera.engine.sglang multi-node command for this pod's rank."""
    cmd = [
        "python3",
        "-m",
        "infera.engine.sglang",
        "--model-path",
        a.model,
        "--tp-size",
        str(a.tp),
        "--trust-remote-code",
        "--host",
        "0.0.0.0",  # nosec B104 - Infera worker must bind for pod-to-pod traffic.
        "--port",
        str(_DEFAULT_ENGINE_PORT),
        "--discovery-backend",
        "kubernetes",
        "--advertise-host",
        advertise_host,
        "--request-transport",
        "nats",
    ]
    # Single-node roles: omit --nnodes/--node-rank/--dist-init-addr to mirror the SaFE native infera.engine.sglang
    # launch.
    if int(a.nnodes) > 1:
        cmd.extend(
            [
                "--nnodes",
                str(a.nnodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                f"{leader}:{a.dist_init_port}",
            ]
        )
    if a.ep and int(a.ep) > 1:
        cmd.extend(["--ep-size", str(a.ep)])
    if a.extra_args:
        extra_tokens = shlex.split(a.extra_args)
        # infera.engine.sglang enables expert-parallel via --ep-size (added just above).
        _ep_flag = "--enable-expert-parallel"
        if _ep_flag in extra_tokens:
            extra_tokens = [t for t in extra_tokens if t != _ep_flag]
            _log(f"dropped vllm-only {_ep_flag} from sglang extra-args (EP via --ep-size)")
        cmd.extend(extra_tokens)
        # sglang ServerArgs.__post_init__ force-disables enable_dp_attention / enable_dp_lm_head when dp_size == 1
        # (its default).
        _dp_enable_flags = ("--enable-dp-attention", "--enable-dp-lm-head")
        has_dp_enable = any(tok in _dp_enable_flags for tok in extra_tokens)
        has_dp_size = any(tok == "--dp-size" or tok.startswith("--dp-size=") for tok in extra_tokens)
        if has_dp_enable and not has_dp_size and int(a.tp) > 1:
            cmd.extend(["--dp-size", str(a.tp)])
    # Skip sglang's post-load warmup ONLY for PD-disaggregated legs.
    _extra_for_warmup = shlex.split(a.extra_args) if a.extra_args else []
    _is_pd_leg = "--disaggregation-mode" in _extra_for_warmup or any(
        tok.startswith("--disaggregation-mode=") for tok in _extra_for_warmup
    )
    if _is_pd_leg and "--skip-server-warmup" not in _extra_for_warmup:
        cmd.append("--skip-server-warmup")
    return cmd


def _build_vllm_cmd(a: argparse.Namespace, *, advertise_host: str) -> list[str]:
    """infera.engine.vllm command (rank 0 only; workers just join the ray cluster)."""
    cmd = [
        "python3",
        "-m",
        "infera.engine.vllm",
        "--model-path",
        a.model,
        "--tensor-parallel-size",
        str(a.tp),
        "--host",
        "0.0.0.0",  # nosec B104 - Infera worker must bind for pod-to-pod traffic.
        "--port",
        str(_DEFAULT_ENGINE_PORT),
        "--discovery-backend",
        "kubernetes",
        "--advertise-host",
        advertise_host,
        "--request-transport",
        "nats",
    ]
    if a.ep and int(a.ep) > 1:
        cmd.append("--enable-expert-parallel")
    if a.extra_args:
        cmd.extend(shlex.split(a.extra_args))
    return cmd


def _detach_launch(cmd: list[str], log_file: Path, pid_file: Path, env: dict[str, str]) -> int:
    """Start ``cmd`` detached (nohup+setsid) and record its PID."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    inner = " ".join(shlex.quote(c) for c in cmd)
    log_q = shlex.quote(str(log_file))
    pid_q = shlex.quote(str(pid_file))
    launch = (
        f": >{log_q}; "
        f"if command -v setsid >/dev/null 2>&1; then "
        f"nohup setsid {inner} >>{log_q} 2>&1 & "
        f"else nohup {inner} >>{log_q} 2>&1 & fi; "
        f"echo $! > {pid_q}"
    )
    proc = subprocess.run(
        ["/bin/bash", "-lc", f"set -euo pipefail; {launch}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"detach spawn failed rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}")
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        tail = ""
        try:
            if log_file.is_file():
                tail = log_file.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        raise RuntimeError(f"server pid={pid} exited within 0.5s: {exc}; log tail:\n{tail}") from exc
    return pid


def _ray_start(role: str, leader: str, env: dict[str, str]) -> None:
    """Bootstrap a Ray cluster across the LWS pods for vllm multi-node."""
    if role == "head":
        ray_cmd = f"ray start --head --port {_RAY_GCS_PORT} --disable-usage-stats"
    else:
        ray_cmd = f"ray start --address={shlex.quote(leader)}:{_RAY_GCS_PORT} --disable-usage-stats"
    # The daemons ``ray start`` leaves behind get their own session and no pipe of ours: in our process group they pin
    # whatever tracks this launcher by group, and a pipe they inherit never reaches EOF for its reader.
    with tempfile.TemporaryFile(mode="w+") as out, tempfile.TemporaryFile(mode="w+") as err:
        cp = subprocess.run(
            ["/bin/bash", "-lc", ray_cmd],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            text=True,
            timeout=180,
            start_new_session=True,
        )
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    _log(f"ray start ({role}) rc={cp.returncode} {(stderr or stdout).strip()[:300]}")


def _start_gpu_sampler(out_csv: Path, pid_file: Path) -> None:
    """Start a detached rocm-smi sampler for this pod's GPUs: the single-node Magpie GPUMonitor's fields, streamed to
    ``out_csv`` on shared FS for the client to harvest."""
    import shutil

    interval = max(1, int(os.environ.get("HYPERLOOM_MN_GPU_SAMPLE_INTERVAL_S", "5") or "5"))
    rocm = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    if not Path(rocm).exists():
        _log("rocm-smi not found; skipping GPU sampler")
        return
    try:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    q_csv = shlex.quote(str(out_csv))
    q_rocm = shlex.quote(rocm)
    # header once, then loop: prepend epoch ts to each rocm-smi --csv data row.
    script = (
        "set +e; "
        f'H="ts,$({q_rocm} --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null | head -1)"; '
        f'[ -s {q_csv} ] || echo "$H" > {q_csv}; '
        "while true; do "
        "TS=$(date +%s); "
        f"{q_rocm} --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null "
        f'| tail -n +2 | sed "s/^/$TS,/" >> {q_csv}; '
        f"sleep {interval}; done"
    )
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        pid_file.write_text(str(proc.pid), encoding="utf-8")
    except OSError:
        pass
    _log(f"started GPU sampler pid={proc.pid} interval={interval}s -> {out_csv}")


def _kill_gpu_sampler(pid_file: Path) -> None:
    """Kill a prior GPU sampler recorded in ``pid_file`` (best-effort)."""
    try:
        spid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return
    try:
        os.killpg(os.getpgid(spid), 9)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        pid_file.unlink()
    except OSError:
        pass


def _parse_args() -> argparse.Namespace:
    """The launcher's command line."""
    p = argparse.ArgumentParser(prog="launch_infera_node.py")
    p.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    p.add_argument("--model", default="")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--dist-init-port", type=int, default=_DEFAULT_DIST_INIT_PORT)
    p.add_argument("--pid-file", default=str(Path(tempfile.gettempdir()) / "mn_infera_server.pid"))
    p.add_argument("--log-file", default=str(Path(tempfile.gettempdir()) / "mn_infera_server.log"))
    p.add_argument("--extra-args", default="")
    p.add_argument("--kill-only", action="store_true", help="kill the prior server via PID file and exit (frees GPU)")
    return p.parse_args()


def main() -> int:
    """Launch (or kill) this pod's Infera multi-node server rank."""
    args = _parse_args()
    env = _recover_container_env()
    node_rank = int(env.get("LWS_WORKER_INDEX", "0") or "0")
    # Multi-pod LWS roles (TP > one pod's GPUs) rendezvous at the controller-injected leader address.
    leader = env.get("LWS_LEADER_ADDRESS", "").strip() or _resolve_pod_ip(env)
    pid_file = Path(args.pid_file)
    log_file = Path(args.log_file)
    # A forwarded shared-FS dir gets per-pod server.log and GPU metrics files the client reads; PD legs do not collide.
    shared_dir = os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip()
    shared = shared_dir.startswith("/") and "$" not in shared_dir
    if shared:
        sampler_host = _resolve_pod_ip(env)
        sampler_csv = Path(shared_dir) / f"gpu_metrics_{sampler_host}.csv"
        sampler_pid_file = Path(shared_dir) / f"gpu_sampler_{sampler_host}.pid"

    try:
        _kill_prior(pid_file)
        if shared:
            _kill_gpu_sampler(sampler_pid_file)
    except RuntimeError as exc:
        _log(f"ERROR {exc}")
        print(json.dumps({"status": "error", "action": "kill", "node_rank": node_rank, "error": str(exc)}))
        return 3
    if args.kill_only:
        if args.framework == "vllm":  # also tear down the local ray node so GPUs are freed
            subprocess.run(
                ["/bin/bash", "-lc", "ray stop --force || true"], env=env, capture_output=True, text=True, timeout=60
            )
        print(json.dumps({"status": "ok", "action": "kill", "node_rank": node_rank}))
        return 0

    if not args.model or args.tp <= 0:
        _log("ERROR --model and --tp are required unless --kill-only")
        return 2

    denied = find_denied_flags(args.extra_args) + find_unsafe_flag_values(args.extra_args)
    if denied:
        _log(f"ERROR denied server flags in --extra-args: {denied}")
        return 2

    _log(
        f"framework={args.framework} model={args.model} tp={args.tp} "
        f"nnodes={args.nnodes} node_rank={node_rank} leader={leader}"
    )

    advertise_host = _resolve_pod_ip(env)
    if shared:
        log_file = Path(shared_dir) / f"mn_infera_server_{advertise_host}_r{node_rank}.log"
    if args.framework == "sglang":
        activate_kernel_shape_tool(env)
        cmd = _build_sglang_cmd(args, node_rank, leader, advertise_host=advertise_host)
    else:
        _ray_start("head" if node_rank == 0 else "worker", leader, env)
        if node_rank != 0:
            pid_file.write_text("0")  # sentinel; nothing to kill but the ray node
            print(json.dumps({"status": "ok", "node_rank": node_rank, "role": "vllm_ray_worker", "pid": 0}))
            return 0
        cmd = _build_vllm_cmd(args, advertise_host=advertise_host)
    pid = _detach_launch(cmd, log_file, pid_file, env)

    summary = {
        "status": "ok",
        "framework": args.framework,
        "node_rank": node_rank,
        "leader": leader,
        "pid": pid,
        "pid_file": str(pid_file),
        "log_file": str(log_file),
    }
    if shared:
        try:
            _start_gpu_sampler(sampler_csv, sampler_pid_file)
            summary["gpu_metrics_csv"] = str(sampler_csv)
        except (OSError, ValueError) as exc:  # sampler is optional telemetry
            _log(f"GPU sampler start failed: {exc}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
