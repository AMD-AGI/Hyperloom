#!/usr/bin/env python3
"""Pod-side launcher for the Infera multi-node backend (idle-pod SSH mode).

Runs INSIDE one LeaderWorkerSet (LWS) worker pod, shipped + invoked over SSH by
``inference_optimizer.multi_node restart-server --backend infera``. Each pod
self-determines its rank from the LWS-injected env, so the sandbox controller
issues the SAME command to every worker pod.

Why this script (vs the RayJob ``launch_multinode.py``):
  * No Ray. sglang multi-node uses torch.distributed
    (``--dist-init-addr <leader>:5000 --nnodes N --node-rank K``); the LWS
    controller already injects ``$LWS_LEADER_ADDRESS`` / ``$LWS_WORKER_INDEX``
    into every pod, so this script just reads them and launches one rank.
  * We launch ``infera.sglang`` (not raw ``sglang.launch_server``) so the
    worker registers with the Infera frontend over NATS — benchmarks then hit
    ``infera.frontend`` (:8000), never sglang rank-0 :8888.

Responsibilities:
  1. Recover the container env from ``/proc/1/environ`` (an sshd session starts
     with a minimal env and would otherwise miss LWS_* / NATS_SERVER / INFERA_* /
     NCCL_* / SGLANG_* / PATH).
  2. PID-file kill of any prior server (IR-5: never ``pkill -f``).
  3. Launch ``infera.sglang`` (or ``infera.vllm``) detached via nohup+setsid,
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
import tempfile
import time
from pathlib import Path

# Rendezvous port for torch.distributed (matches SaFE
# common.InferaMultinodeDistInitPort = 5000). Override via --dist-init-port.
_DEFAULT_DIST_INIT_PORT = 5000
# Ray GCS port for the vllm multi-node bootstrap.
_RAY_GCS_PORT = 6379
# Default engine HTTP port for local readiness probes on node-rank 0.
_DEFAULT_ENGINE_PORT = 30000

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


def _is_denied_server_flag(flag: str) -> bool:
    """Return whether a single ``--flag`` token is denied at the pod boundary."""
    name = (flag or "").strip()
    if not name.startswith("--"):
        return False
    if name in _DENIED_SERVER_FLAGS:
        return True
    return any(name.endswith(suffix) for suffix in _DENIED_SERVER_FLAG_SUFFIXES)


def _denied_extra_args(raw: str) -> list[str]:
    """Return denied CLI flag tokens in a pod-side extra-args string.

    Args:
        raw: Whitespace-separated server flags.

    Returns:
        list[str]: Denied flag names (empty when clean).
    """
    text = (raw or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text)
    except ValueError:
        return ["<unparseable>"]
    out: list[str] = []
    for tok in tokens:
        flag = tok.split("=", 1)[0]
        if _is_denied_server_flag(flag) and flag not in out:
            out.append(flag)
    return out


def _log(msg: str) -> None:
    """Write a timestamped launcher log line to stderr.

    Args:
        msg: The message to log.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_infera_node {ts}] {msg}\n")
    sys.stderr.flush()


# Env-var prefixes/names worth recovering from pid1 so the framework child sees
# the same rendezvous / discovery / tuning config the container was started
# with. An sshd session would otherwise launch with a bare login env.
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
    # Hyperloom patch (operator-local): KUBERNETES_* must propagate too,
    # otherwise infera's kubernetes discovery backend fails with
    #   "Failed to create Kubernetes client: failed to infer config:
    #    in-cluster: (environment variable not found)"
    # immediately on infera.sglang/infera.vllm start, and the SSH-launched
    # server exits in <1s while the Infera frontend (always-up) keeps
    # returning /health 200 — causing baseline_failed with 0 completed
    # requests and no obvious sandbox-side log evidence.
    "KUBERNETES_",
)
_ENV_RECOVER_NAMES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV")


def _recover_container_env() -> dict[str, str]:
    """Merge the current env with pid1's env for the recovered keys.

    sshd sessions get a minimal env; the LWS rendezvous vars
    (``LWS_LEADER_ADDRESS`` / ``LWS_WORKER_INDEX``) and discovery vars
    (``NATS_SERVER`` / ``INFERA_*``) live only in the container's pid1 env. We
    read ``/proc/1/environ`` (same uid — we SSH as root, pid1 is root) and
    overlay the relevant keys onto ``os.environ``.

    Returns:
        The current environment merged with pid1's recovered keys, with
        ``/opt/venv/bin`` ensured at the front of ``PATH``.
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
        if key in _ENV_RECOVER_NAMES or any(key.startswith(p) for p in _ENV_RECOVER_PREFIXES):
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

    Args:
        env: The (recovered) environment, consulted for ``$POD_IP``.

    Returns:
        The pod's routable IP, or ``127.0.0.1`` when none can be resolved.
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
    """Return ``root`` plus all descendant PIDs via /proc ppid links.

    Collected in a single pass so a child that later re-parents to init (e.g. the
    sglang scheduler subprocess, which escapes the wrapper's process group) is
    still captured as long as it is a descendant at collection time.
    """
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


def _reap_stale_engines_by_cmdline() -> None:
    """SIGKILL any residual sglang / infera engine process matched by cmdline via
    /proc, before a fresh launch.

    Catches an orphaned ``sglang.launch_server`` (re-parented to init) that still
    holds ``port_base`` but which neither the process-tree kill (no longer a
    descendant) nor the ss-based port sweep can find -- sglang binds port_base in
    a way ``ss``/``fuser`` do not surface, yet it blocks the next launch with
    "port_base not available" -> engine exit(1) -> /v1/completions 503. Runs
    before launch, so every matching engine is stale by definition. Scoped to our
    own engine cmdlines (IR-5: we only kill processes we launched). Never raises.
    """
    import signal as _sig
    kill_wait_s = float(os.environ.get("HYPERLOOM_MN_KILL_WAIT_S", "120") or 120)
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
    for sig in (_sig.SIGTERM, _sig.SIGKILL):
        for v in victims:
            try:
                os.kill(v, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + (5.0 if sig == _sig.SIGTERM else kill_wait_s)
        while time.monotonic() < deadline:
            if not any(_pid_alive(v) for v in victims):
                break
            time.sleep(0.5)
        if not any(_pid_alive(v) for v in victims):
            break
    remaining = [v for v in victims if _pid_alive(v)]
    if remaining:
        _log("reaper: stale engine pids still alive after SIGKILL: " + str(remaining))
    else:
        _log("reaper: killed stale engine pids " + str(victims))


def _kill_prior(pid_file: Path) -> None:
    """SIGTERM then SIGKILL the prior server's whole process tree, then sweep any
    residual holder of the managed sglang engine ports.

    IR-5: only the PID we launched and its descendants (plus the managed-port
    sweep). Missing / stale PID files are a no-op so callers can use this
    idempotently before launch.

    Why the tree (not just ``killpg``): the sglang engine (``sglang.launch_server``)
    spawns scheduler subprocesses that escape the wrapper's process group. A bare
    ``killpg`` leaves one holding ``port_base``, so the next launch dies with
    "port_base not available" -> engine exits(1) -> worker deregisters ->
    /v1/completions 503 -> every restarted candidate REVERTs. We collect the full
    descendant set BEFORE signalling (escaped children are still reachable then)
    and SIGKILL all of them, then always sweep the managed ports as a backstop.

    Args:
        pid_file: Path to the pid file recording the prior server's PID.

    Raises:
        RuntimeError: If any target is still alive after the kill window (wedged,
            e.g. D-state on slow weight I/O), so the caller aborts the relaunch
            instead of double-stacking a second server and leaking VRAM.
    """
    kill_wait_s = float(os.environ.get("HYPERLOOM_MN_KILL_WAIT_S", "120") or 120)
    pid = None
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
    targets: list[int] = []
    if pid and pid > 0 and _pid_alive(pid):
        targets = _proc_tree(pid)
    if targets:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None
        for sig in (15, 9):
            for t in targets:
                try:
                    os.kill(t, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            if pgid is not None:
                try:
                    os.killpg(pgid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.monotonic() + (5.0 if sig == 15 else kill_wait_s)
            while time.monotonic() < deadline:
                if not any(_pid_alive(t) for t in targets):
                    break
                time.sleep(0.5)
            if not any(_pid_alive(t) for t in targets):
                break
        alive = [t for t in targets if _pid_alive(t)]
        if alive:
            raise RuntimeError(
                "prior server pids " + str(alive) + " still alive "
                + str(int(kill_wait_s)) + "s after SIGKILL (wedged, likely D-state on "
                "slow weight I/O); aborting relaunch to avoid VRAM double-stack -- "
                "node may need a reboot/GPU reset"
            )
        _log("killed prior server tree pid=" + str(pid) + " (" + str(len(targets)) + " procs)")
    pid_file.unlink(missing_ok=True)
    # Kill residual engine processes by cmdline (catches an orphaned
    # launch_server holding port_base that ss/tree-kill miss), then sweep ports.
    _reap_stale_engines_by_cmdline()
    _reap_stale_engine_ports()


# Hyperloom operator patch (session 517be0c5): the infera decode/prefill
# restart records the infera.engine wrapper PID in the pid-file, but the real
# sglang.launch_server it spawns becomes its own process-group leader and
# holds the engine ports (30000 HTTP, 30234 dist port_base, 30001 bootstrap,
# 32760 kv-events). _kill_prior's pgid kill therefore misses it, so the NEXT
# restart crashes with "port_base at 30234 is not available" and every explore
# variant aborts at the health timeout. This reaper kills any residual process
# that still holds those specific ports right before we relaunch. It targets
# only the exact port owners we manage (not `pkill -f`), preserving IR-5's
# "only kill what we launched" intent.
_REAP_PORTS = (30000, 30001, 30234, 32760)


def _reap_stale_engine_ports() -> None:
    """Kill any residual process still holding the sglang engine ports.

    Uses ``ss -ltnp`` to map each managed port to its owning PID and SIGKILLs
    that process group. Best-effort and idempotent: absent ports or missing
    tooling are a no-op.
    """
    try:
        import re as _re
        import signal as _signal
    except Exception:  # pragma: no cover
        return
    kill_wait_s = float(os.environ.get("HYPERLOOM_MN_KILL_WAIT_S", "120") or 120)
    for port in _REAP_PORTS:
        try:
            out = subprocess.run(
                ["/bin/bash", "-lc", f"ss -ltnp 2>/dev/null | grep -w ':{port}' || true"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout
        except Exception:
            continue
        for m in _re.finditer(r"pid=(\d+)", out or ""):
            try:
                pid = int(m.group(1))
            except ValueError:
                continue
            if pid <= 1:
                continue
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            _log(f"reaper: port {port} held by pid={pid} pgid={pgid}; killing")
            for sig in (_signal.SIGTERM, _signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                except PermissionError:
                    try:
                        os.kill(pid, sig)
                    except OSError:
                        pass
                deadline = time.monotonic() + (5.0 if sig == _signal.SIGTERM else kill_wait_s)
                gone = False
                while time.monotonic() < deadline:
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        gone = True
                        break
                    except PermissionError:
                        break
                    time.sleep(0.5)
                if gone:
                    _log(f"reaper: freed port {port} (pid={pid}, signal {int(sig)})")
                    break


def _build_sglang_cmd(
    a: argparse.Namespace,
    node_rank: int,
    leader: str,
    *,
    advertise_host: str,
) -> list[str]:
    """infera.engine.sglang multi-node command for this pod's rank.

    Args:
        a: Parsed launcher arguments.
        node_rank: This pod's node rank.
        leader: torch.distributed rendezvous leader address.
        advertise_host: Routable pod IP for worker registration.

    Returns:
        The ``infera.engine.sglang`` command argv for this pod's rank.
    """
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
        str(getattr(a, "engine_port", _DEFAULT_ENGINE_PORT)),
        "--discovery-backend",
        "kubernetes",
        "--advertise-host",
        advertise_host,
        "--request-transport",
        "nats",
    ]
    # Single-node roles: omit --nnodes/--node-rank/--dist-init-addr to mirror the
    # SaFE native infera.sglang launch. Passing them for an nnodes=1 disaggregated
    # PD role made decode emit 0 output tokens (finish_reason=stop), while the
    # SaFE deploy (which omits them for single node) generates normally.
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
        # infera.engine.sglang enables expert-parallel via --ep-size (added just
        # above). The vllm-style --enable-expert-parallel is NOT a recognized
        # infera.engine.sglang arg: its argparse aborts with "unrecognized
        # arguments: --enable-expert-parallel", so the server exits on launch and
        # /health never comes up (EP explore variants then score 0 tok/s or burn
        # the full restart timeout). Upstream (explore atom_ep) injects it
        # framework-blind, so strip it here at the infera-sglang boundary; EP
        # stays enabled through --ep-size.
        _ep_flag = "--enable-expert-parallel"
        if _ep_flag in extra_tokens:
            extra_tokens = [t for t in extra_tokens if t != _ep_flag]
            _log(f"dropped vllm-only {_ep_flag} from sglang extra-args (EP via --ep-size)")
        cmd.extend(extra_tokens)
        # sglang ServerArgs.__post_init__ force-disables enable_dp_attention /
        # enable_dp_lm_head when dp_size == 1 (its default). If the caller asked
        # for either DP-attention flag but omitted --dp-size, set it to tp so the
        # flags actually take effect (full DP-attention; tp % dp == 0 holds).
        _dp_enable_flags = ("--enable-dp-attention", "--enable-dp-lm-head")
        has_dp_enable = any(tok in _dp_enable_flags for tok in extra_tokens)
        has_dp_size = any(
            tok == "--dp-size" or tok.startswith("--dp-size=") for tok in extra_tokens
        )
        if has_dp_enable and not has_dp_size and int(a.tp) > 1:
            cmd.extend(["--dp-size", str(a.tp)])
    # Skip sglang's post-load warmup ONLY for PD-disaggregated legs. In PD the
    # warmup generate needs the prefill<->decode pair fully wired, which is not
    # guaranteed during a restart window, so it blocks until SGLANG_WARMUP_TIMEOUT
    # (default 1800s) then kill_process_tree()s the engine -> the leg goes 0/1 and
    # the frontend returns 503, failing the round on a 30-min timeout. Readiness
    # is validated by the optimizer's /v1/completions probe instead. Aggregated
    # (one shared server, no PD dependency) keeps warmup ON for parity with the
    # single-node path. Respect an explicit --skip-server-warmup in extra-args.
    _extra_for_warmup = shlex.split(a.extra_args) if a.extra_args else []
    _is_pd_leg = "--disaggregation-mode" in _extra_for_warmup or any(
        tok.startswith("--disaggregation-mode=") for tok in _extra_for_warmup
    )
    if _is_pd_leg and "--skip-server-warmup" not in _extra_for_warmup:
        cmd.append("--skip-server-warmup")
    return cmd


def _build_vllm_cmd(a: argparse.Namespace, *, advertise_host: str) -> list[str]:
    """infera.engine.vllm command (rank 0 only; workers just join the ray cluster).

    Args:
        a: Parsed launcher arguments.
        advertise_host: Routable pod IP for worker registration.

    Returns:
        The ``infera.engine.vllm`` command argv for rank 0.
    """
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
        str(getattr(a, "engine_port", _DEFAULT_ENGINE_PORT)),
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
    """Start ``cmd`` detached (nohup+setsid) and record its PID.

    Reparents the server under init so it survives the SSH session closing,
    and fails fast (with a log tail) if the child dies within 0.5s.

    Args:
        cmd: The server command argv to launch.
        log_file: Path the detached server's stdout/stderr is written to.
        pid_file: Path the launched PID is recorded in.
        env: Environment for the launched process.

    Returns:
        The PID of the launched server.

    Raises:
        RuntimeError: If the detach spawn fails or the child exits within
            0.5s of launch.
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
    """Bootstrap a Ray cluster across the LWS pods for vllm multi-node.

    rank 0 -> ``ray start --head``; workers -> ``ray start --address``. vllm
    on rank 0 then discovers the workers via the GCS
    (``--distributed-executor-backend ray``).

    Args:
        role: ``"head"`` for rank 0, otherwise a worker that joins the leader.
        leader: Leader address workers connect to.
        env: Environment for the ``ray start`` subprocess.
    """
    if role == "head":
        ray_cmd = f"ray start --head --port {_RAY_GCS_PORT} --disable-usage-stats"
    else:
        ray_cmd = f"ray start --address={shlex.quote(leader)}:{_RAY_GCS_PORT} --disable-usage-stats"
    cp = subprocess.run(
        ["/bin/bash", "-lc", ray_cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    _log(f"ray start ({role}) rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:300]}")


def _wait_health(port: int, timeout_s: int, pid: int | None) -> bool:
    """Poll http://127.0.0.1:<port>/health until 200 or the pid dies.

    Args:
        port: Local health endpoint port to poll.
        timeout_s: Maximum seconds to wait for a healthy response.
        pid: Optional server PID; polling stops early if the process dies.

    Returns:
        ``True`` when the endpoint returned a 2xx status before the timeout,
        else ``False``.
    """
    import urllib.error
    import urllib.request

    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=3,
            ) as resp:  # nosec B310 - fixed loopback health check.
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


def _start_gpu_sampler(out_csv: Path, pid_file: Path, interval_s: int) -> None:
    """Start a detached rocm-smi sampler for this pod's GPUs.

    Mirrors the single-node Magpie GPUMonitor fields (temp / power / util /
    VRAM) but runs on the GPU pod itself, appending timestamped per-card rows
    to ``out_csv`` on shared storage so the client can harvest them. Runs until
    the next restart kills it. No-op if rocm-smi is unavailable.
    """
    import shutil
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
    interval = max(1, int(interval_s))
    # header once, then loop: prepend epoch ts to each rocm-smi --csv data row.
    script = (
        "set +e; "
        f'H="ts,$({q_rocm} --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null | head -1)"; '
        f'[ -s {q_csv} ] || echo "$H" > {q_csv}; '
        "while true; do "
        "TS=$(date +%s); "
        f'{q_rocm} --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null '
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


def main() -> int:
    """Launch (or kill) this pod's Infera multi-node server rank.

    Recovers the container env, resolves this pod's rank and rendezvous
    leader, kills any prior server, then launches ``infera.sglang`` /
    ``infera.vllm`` for this rank (optionally waiting for leader readiness).
    With ``--kill-only`` it just tears down the prior server and exits.

    Returns:
        ``0`` on success, or ``2`` when required ``--model`` / ``--tp`` args
        are missing.
    """
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
    p.add_argument("--health-port", type=int, default=8000, help="leader local readiness probe port (frontend/http)")
    p.add_argument(
        "--health-wait-sec", type=int, default=0, help="leader-only: seconds to wait for local /health (0=skip)"
    )
    p.add_argument("--kill-only", action="store_true", help="kill the prior server via PID file and exit (frees GPU)")
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
    # GPU metrics sampler paths (shared-FS, per-pod). Empty when no
    # shared server-log dir is forwarded.
    _samp_dir = os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip()
    _samp_csv = ""
    _samp_pid_file = ""
    if _samp_dir.startswith("/") and "$" not in _samp_dir:
        _samp_host = _resolve_pod_ip(env)
        _samp_csv = str(Path(_samp_dir) / f"gpu_metrics_{_samp_host}.csv")
        _samp_pid_file = str(Path(_samp_dir) / f"gpu_sampler_{_samp_host}.pid")

    try:
        _kill_prior(pid_file)
        if _samp_pid_file:
            _kill_gpu_sampler(Path(_samp_pid_file))
    except RuntimeError as exc:
        _log(f"ERROR {exc}")
        print(json.dumps({"status": "error", "action": "kill", "node_rank": node_rank, "error": str(exc)}))
        return 3
    if args.kill_only:
        # vllm: also tear down the local ray node so GPUs are freed.
        if args.framework == "vllm":
            subprocess.run(
                ["/bin/bash", "-lc", "ray stop --force || true"], env=env, capture_output=True, text=True, timeout=60
            )
        print(json.dumps({"status": "ok", "action": "kill", "node_rank": node_rank}))
        return 0

    if not args.model or args.tp <= 0:
        _log("ERROR --model and --tp are required unless --kill-only")
        return 2

    denied = _denied_extra_args(args.extra_args)
    if denied:
        _log(f"ERROR denied server flags in --extra-args: {denied}")
        return 2

    _log(
        f"framework={args.framework} model={args.model} tp={args.tp} "
        f"nnodes={args.nnodes} node_rank={node_rank} leader={leader}"
    )

    advertise_host = _resolve_pod_ip(env)
    # When a shared-FS (WekaFS) server-log dir is forwarded, write server.log
    # there with a per-pod suffix so the client can read it and prefill/decode
    # do not collide. Falls back to the passed --log-file (pod-local /tmp).
    _shared_log_dir = os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip()
    if _shared_log_dir.startswith("/") and "$" not in _shared_log_dir:
        log_file = Path(_shared_log_dir) / f"mn_infera_server_{advertise_host}_r{node_rank}.log"
    if args.framework == "sglang":
        cmd = _build_sglang_cmd(args, node_rank, leader, advertise_host=advertise_host)
        pid = _detach_launch(cmd, log_file, pid_file, env)
    else:
        # vllm: every pod joins the ray cluster; only rank 0 runs infera.engine.vllm.
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

    # Self-contained GPU metrics: run rocm-smi sampling on this GPU pod
    # (same fields as the single-node Magpie GPUMonitor) and stream to
    # shared FS for the client to harvest. Never fail launch on metrics.
    if _samp_csv:
        try:
            _start_gpu_sampler(
                Path(_samp_csv),
                Path(_samp_pid_file),
                int(os.environ.get("HYPERLOOM_MN_GPU_SAMPLE_INTERVAL_S", "5") or "5"),
            )
            summary["gpu_metrics_csv"] = _samp_csv
        except Exception as exc:
            _log(f"GPU sampler start failed: {exc}")

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
