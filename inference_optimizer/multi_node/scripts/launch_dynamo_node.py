#!/usr/bin/env python3
"""Pod-side launcher for the Dynamo multi-node backend (idle-pod SSH mode).

Runs INSIDE one LeaderWorkerSet (LWS) worker pod, shipped + invoked over SSH by
``inference_optimizer.multi_node restart-server --backend dynamo``. Each pod
self-determines its rank from the LWS-injected env, so the sandbox controller
issues the SAME command to every worker pod.

Why this script (vs the RayJob ``launch_multinode.py``):
  * No Ray. sglang multi-node uses torch.distributed
    (``--dist-init-addr <leader>:5000 --nnodes N --node-rank K``); the LWS
    controller already injects ``$LWS_LEADER_ADDRESS`` / ``$LWS_WORKER_INDEX``
    into every pod, so this script just reads them and launches one rank.
  * We launch ``dynamo.sglang`` (not raw ``sglang.launch_server``) so the
    worker registers with the Dynamo frontend over NATS — benchmarks then hit
    ``dynamo.frontend`` (:8000), never sglang rank-0 :8888.

Responsibilities:
  1. Recover the container env from ``/proc/1/environ`` (an sshd session starts
     with a minimal env and would otherwise miss LWS_* / NATS_SERVER / DYN_* /
     NCCL_* / SGLANG_* / PATH).
  2. PID-file kill of any prior server (IR-5: never ``pkill -f``).
  3. Launch ``dynamo.sglang`` (or ``dynamo.vllm``) detached via nohup+setsid,
     wired with ``--nnodes/--node-rank/--dist-init-addr``.
  4. Optional readiness wait on the leader (``LWS_WORKER_INDEX == 0``).

Stdlib only — this runs in the framework pod, not the optimizer venv.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Rendezvous port for torch.distributed (matches SaFE
# common.DynamoMultinodeDistInitPort = 5000). Override via --dist-init-port.
_DEFAULT_DIST_INIT_PORT = 5000
# Ray GCS port for the vllm multi-node bootstrap.
_RAY_GCS_PORT = 6379


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_dynamo_node {ts}] {msg}\n")
    sys.stderr.flush()


# Env-var prefixes/names worth recovering from pid1 so the framework child sees
# the same rendezvous / discovery / tuning config the container was started
# with. An sshd session would otherwise launch with a bare login env.
_ENV_RECOVER_PREFIXES = (
    "LWS_", "POD_", "NCCL_", "GLOO_", "RCCL_", "DYN_", "SGLANG_", "VLLM_",
    "HSA_", "HIP_", "ROCR_", "HF_", "NATS_", "UCX_", "NIXL_", "MC_",
    # Hyperloom patch (operator-local): KUBERNETES_* must propagate too,
    # otherwise dynamo's kubernetes discovery backend fails with
    #   "Failed to create Kubernetes client: failed to infer config:
    #    in-cluster: (environment variable not found)"
    # immediately on dynamo.sglang/dynamo.vllm start, and the SSH-launched
    # server exits in <1s while the Dynamo frontend (always-up) keeps
    # returning /health 200 — causing baseline_failed with 0 completed
    # requests and no obvious sandbox-side log evidence.
    "KUBERNETES_",
)
_ENV_RECOVER_NAMES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV")


def _recover_container_env() -> dict[str, str]:
    """Merge the current env with pid1's env for the recovered keys.

    sshd sessions get a minimal env; the LWS rendezvous vars
    (``LWS_LEADER_ADDRESS`` / ``LWS_WORKER_INDEX``) and discovery vars
    (``NATS_SERVER`` / ``DYN_*``) live only in the container's pid1 env. We
    read ``/proc/1/environ`` (same uid — we SSH as root, pid1 is root) and
    overlay the relevant keys onto ``os.environ``.
    """
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
        if key in _ENV_RECOVER_NAMES or any(
            key.startswith(p) for p in _ENV_RECOVER_PREFIXES
        ):
            # pid1 wins for rendezvous/discovery; but keep sshd's PATH augmented
            # with /opt/venv/bin so python3 resolves to the framework venv.
            env[key] = val
    venv_bin = "/opt/venv/bin"
    parts = env.get("PATH", "").split(":") if env.get("PATH") else []
    if venv_bin not in parts:
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}".rstrip(":")
    return env


def _resolve_pod_ip(env: dict[str, str]) -> str:
    """Return this pod's routable IP (never a loopback address).

    Single-pod PD-disaggregation roles (prefill / decode, nnodes=1) are NOT a
    LeaderWorkerSet, so ``$LWS_LEADER_ADDRESS`` is unset and the caller would
    otherwise fall back to ``127.0.0.1``. sglang derives the disaggregation
    bootstrap host it advertises to peers from ``--dist-init-addr``; a loopback
    value makes the cross-pod decode->prefill KV handshake fail with
    ``NIXL KVReceiver Exception`` (decode dials its own localhost). Resolve the
    real pod IP so the advertised bootstrap host is reachable across pods.

    Resolution order: ``$POD_IP`` (downward API) -> egress-route probe ->
    hostname lookup. Falls back to ``127.0.0.1`` only if every method yields a
    loopback / fails (single-pod aggregated runs still work in that case).
    """
    import socket

    cand = (env.get("POD_IP") or "").strip()
    if cand and not cand.startswith("127."):
        return cand
    # Egress-route probe: connecting a UDP socket sends no packets but makes the
    # kernel pick the source IP of the default-route interface.
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


def _kill_prior(pid_file: Path) -> None:
    """SIGTERM (then SIGKILL) the process group recorded in ``pid_file``.

    IR-5: never ``pkill -f`` — only the PID we launched. Missing / stale PID
    files are a no-op so callers can use this idempotently before launch.
    """
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    if pid <= 0:
        pid_file.unlink(missing_ok=True)
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return
    for sig in (15, 9):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            break
        # Give SIGTERM a moment before escalating.
        for _ in range(20):
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.25)
        else:
            continue
        break
    pid_file.unlink(missing_ok=True)
    _log(f"killed prior server pgid={pgid}")


def _build_sglang_cmd(a: argparse.Namespace, node_rank: int, leader: str) -> list[str]:
    """dynamo.sglang multi-node command for this pod's rank."""
    cmd = [
        "python3", "-m", "dynamo.sglang",
        "--model-path", a.model,
        "--tp-size", str(a.tp),
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--nnodes", str(a.nnodes),
        "--node-rank", str(node_rank),
        "--dist-init-addr", f"{leader}:{a.dist_init_port}",
    ]
    if a.ep and int(a.ep) > 1:
        cmd.extend(["--ep-size", str(a.ep)])
    if a.extra_args:
        cmd.extend(shlex.split(a.extra_args))
    return cmd


def _build_vllm_cmd(a: argparse.Namespace) -> list[str]:
    """dynamo.vllm command (rank 0 only; workers just join the ray cluster)."""
    cmd = [
        "python3", "-m", "dynamo.vllm",
        "--model", a.model,
        "--tensor-parallel-size", str(a.tp),
    ]
    if a.ep and int(a.ep) > 1:
        cmd.append("--enable-expert-parallel")
    if a.extra_args:
        cmd.extend(shlex.split(a.extra_args))
    return cmd


def _detach_launch(cmd: list[str], log_file: Path, pid_file: Path,
                   env: dict[str, str]) -> int:
    """Start ``cmd`` detached (nohup+setsid) and record its PID.

    Reparents the server under init so it survives the SSH session closing,
    and fails fast (with a log tail) if the child dies within 0.5s.
    """
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
        env=env, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"detach spawn failed rc={proc.returncode} "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
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
        raise RuntimeError(
            f"server pid={pid} exited within 0.5s: {exc}; log tail:\n{tail}"
        ) from exc
    return pid


def _ray_start(role: str, leader: str, env: dict[str, str]) -> None:
    """Bootstrap a Ray cluster across the LWS pods for vllm multi-node.

    rank 0 -> ``ray start --head``; workers -> ``ray start --address``. vllm
    on rank 0 then discovers the workers via the GCS
    (``--distributed-executor-backend ray``).
    """
    if role == "head":
        ray_cmd = f"ray start --head --port {_RAY_GCS_PORT} --disable-usage-stats"
    else:
        ray_cmd = f"ray start --address={shlex.quote(leader)}:{_RAY_GCS_PORT} --disable-usage-stats"
    cp = subprocess.run(
        ["/bin/bash", "-lc", ray_cmd], env=env,
        capture_output=True, text=True, timeout=180,
    )
    _log(f"ray start ({role}) rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:300]}")


def _wait_health(port: int, timeout_s: int, pid: int | None) -> bool:
    """Poll http://127.0.0.1:<port>/health until 200 or the pid dies."""
    import urllib.error
    import urllib.request
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3,
            ) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if pid is not None and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                _log(f"server pid={pid} died during health wait")
                return False
            except OSError:
                pass
        time.sleep(5)
    return False


def main() -> int:
    p = argparse.ArgumentParser(prog="launch_dynamo_node.py")
    p.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    p.add_argument("--model", default="")
    p.add_argument("--tp", type=int, default=0)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--dist-init-port", type=int, default=_DEFAULT_DIST_INIT_PORT)
    p.add_argument("--pid-file", default="/tmp/mn_dynamo_server.pid")
    p.add_argument("--log-file", default="/tmp/mn_dynamo_server.log")
    p.add_argument("--extra-args", default="")
    p.add_argument("--health-port", type=int, default=8000,
                   help="leader local readiness probe port (frontend/http)")
    p.add_argument("--health-wait-sec", type=int, default=0,
                   help="leader-only: seconds to wait for local /health (0=skip)")
    p.add_argument("--kill-only", action="store_true",
                   help="kill the prior server via PID file and exit (frees GPU)")
    args = p.parse_args()

    env = _recover_container_env()
    node_rank = int(env.get("LWS_WORKER_INDEX", "0") or "0")
    lws_leader = (env.get("LWS_LEADER_ADDRESS", "") or "").strip()
    if lws_leader:
        # Multi-pod LWS role (TP > one pod's GPUs): the controller-injected
        # leader address is the torch.distributed rendezvous host.
        leader = lws_leader
    else:
        # Single-pod role (no LWS rendezvous). Use this pod's routable IP rather
        # than 127.0.0.1 so PD-disaggregation advertises a cross-pod-reachable
        # bootstrap host (see _resolve_pod_ip).
        leader = _resolve_pod_ip(env)
    pid_file = Path(args.pid_file)
    log_file = Path(args.log_file)

    _kill_prior(pid_file)
    if args.kill_only:
        # vllm: also tear down the local ray node so GPUs are freed.
        if args.framework == "vllm":
            subprocess.run(["/bin/bash", "-lc", "ray stop --force || true"],
                           env=env, capture_output=True, text=True, timeout=60)
        print(json.dumps({"status": "ok", "action": "kill", "node_rank": node_rank}))
        return 0

    if not args.model or args.tp <= 0:
        _log("ERROR --model and --tp are required unless --kill-only")
        return 2

    _log(f"framework={args.framework} model={args.model} tp={args.tp} "
         f"nnodes={args.nnodes} node_rank={node_rank} leader={leader}")

    if args.framework == "sglang":
        cmd = _build_sglang_cmd(args, node_rank, leader)
        pid = _detach_launch(cmd, log_file, pid_file, env)
    else:
        # vllm: every pod joins the ray cluster; only rank 0 runs dynamo.vllm.
        _ray_start("head" if node_rank == 0 else "worker", leader, env)
        if node_rank != 0:
            pid_file.write_text("0")  # sentinel; nothing to kill but the ray node
            print(json.dumps({"status": "ok", "node_rank": node_rank,
                              "role": "vllm_ray_worker", "pid": 0}))
            return 0
        cmd = _build_vllm_cmd(args)
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

    # Only the leader serves a local HTTP endpoint; workers have none.
    if node_rank == 0 and args.health_wait_sec > 0:
        ok = _wait_health(args.health_port, args.health_wait_sec, pid)
        summary["health_ok"] = ok
        if not ok:
            try:
                summary["log_tail"] = log_file.read_text(errors="replace")[-2000:]
            except OSError:
                pass

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
