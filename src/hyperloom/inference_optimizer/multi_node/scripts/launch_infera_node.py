#!/usr/bin/env python3
"""Pod-side launcher for the Infera multi-node backend (idle-pod SSH mode)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

# Rendezvous port for torch.distributed (matches SaFE common.InferaMultinodeDistInitPort = 5000).
_DEFAULT_DIST_INIT_PORT = 5000
# Ray GCS port for the vllm multi-node bootstrap.
_RAY_GCS_PORT = 6379
# Default HTTP port the infera engine binds via --port (no CLI override is declared); the rank-0 readiness probe uses
# --health-port (default 8000).
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
# Tuning knobs exempt from the suffix rule by name only; their values stay constrained by _unsafe_path_value_reason.
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
    """Return rejected CLI flags in a pod-side extra-args string."""
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


def _reap_stale_engines_by_cmdline() -> None:
    """SIGKILL any residual sglang / infera engine process matched by cmdline via /proc, before a fresh launch."""
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
    """SIGTERM then SIGKILL the prior server's whole process tree, then sweep any residual holder of the managed sglang engine ports."""
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
                "prior server pids "
                + str(alive)
                + " still alive "
                + str(int(kill_wait_s))
                + "s after SIGKILL (wedged, likely D-state on "
                "slow weight I/O); aborting relaunch to avoid VRAM double-stack -- "
                "node may need a reboot/GPU reset"
            )
        _log("killed prior server tree pid=" + str(pid) + " (" + str(len(targets)) + " procs)")
    pid_file.unlink(missing_ok=True)
    # Kill residual engine processes by cmdline (catches an orphaned launch_server holding port_base that ss/tree-kill
    # miss), then sweep ports.
    _reap_stale_engines_by_cmdline()
    _reap_stale_engine_ports()


# The infera decode/prefill restart records the infera.engine wrapper PID in the pid-file, but the real
# sglang.launch_server it spawns becomes its own process-group leader and holds the engine ports (30000 HTTP, 30234
# dist port_base, 30001 bootstrap, 32760 kv-events). _kill_prior's pgid kill therefore misses it, so the NEXT restart
# crashes with "port_base at 30234 is not available" and every explore variant aborts at the health timeout.
_REAP_PORTS = (30000, 30001, 30234, 32760)


def _reap_stale_engine_ports() -> None:
    """Kill any residual process still holding the sglang engine ports."""
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


# SGLang >= 0.5.18 uses the no-patch kernel_shape_tool (PYTHONPATH +
# sitecustomize + TRACELENS_SHAPE_DISCOVERY) for shape discovery. This script is
# standalone (no hyperloom import), so the gate is mirrored inline.
_KERNEL_SHAPE_TOOL_REL = ("TraceLens", "TraceUtils", "kernel_shape_tool")
_SGLANG_SITECUSTOMIZE_MIN_VERSION = (0, 5, 18)


def _sglang_shape_mode() -> str:
    """Pod-side mirror of hyperloom's SGLang shape-mode gate (no hyperloom import)."""
    override = os.environ.get("HYPERLOOM_SGLANG_SHAPE_MODE", "auto").strip().lower()
    if override in {"patch", "patched"}:
        return "patched"
    if override == "sitecustomize":
        return "sitecustomize"
    version = ""
    try:
        import sglang  # type: ignore

        version = (getattr(sglang, "__version__", "") or "").strip()
    except Exception:  # noqa: BLE001
        version = os.environ.get("HYPERLOOM_SGLANG_VERSION_PIN", "").strip()
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", version)
    if not m:
        return "patched"
    vt = tuple(int(p) for p in m.group(1).split("."))
    return "sitecustomize" if vt >= _SGLANG_SITECUSTOMIZE_MIN_VERSION else "patched"


def _maybe_activate_kernel_shape_tool(env: dict[str, str]) -> None:
    """SGLang >= 0.5.18: put the no-patch kernel_shape_tool on PYTHONPATH."""
    root = (env.get("TRACELENS_ROOT") or os.environ.get("TRACELENS_ROOT") or "").strip()
    if not root or _sglang_shape_mode() != "sitecustomize":
        return
    tool = Path(root).joinpath(*_KERNEL_SHAPE_TOOL_REL)
    if not tool.is_dir():
        _log(f"WARN kernel_shape_tool not found at {tool}; SGLang shape discovery disabled")
        return
    existing = (env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = f"{tool}{os.pathsep}{existing}" if existing else str(tool)
    env.setdefault("TRACELENS_SHAPE_DISCOVERY", "1")


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
        str(getattr(a, "engine_port", _DEFAULT_ENGINE_PORT)),
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
    cp = subprocess.run(
        ["/bin/bash", "-lc", ray_cmd],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
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
    """Start a detached rocm-smi sampler for this pod's GPUs."""
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


def main() -> int:
    """Launch (or kill) this pod's Infera multi-node server rank."""
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
        # Multi-pod LWS role (TP > one pod's GPUs): the controller-injected leader address is the torch.distributed
        # rendezvous host.
        leader = lws_leader
    else:
        # Single-pod role (no LWS rendezvous).
        leader = _resolve_pod_ip(env)
    pid_file = Path(args.pid_file)
    log_file = Path(args.log_file)
    # GPU metrics sampler paths (shared-FS, per-pod).
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
    # When a shared-FS (WekaFS) server-log dir is forwarded, write server.log there with a per-pod suffix so the
    # client can read it and prefill/decode do not collide.
    _shared_log_dir = os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip()
    if _shared_log_dir.startswith("/") and "$" not in _shared_log_dir:
        log_file = Path(_shared_log_dir) / f"mn_infera_server_{advertise_host}_r{node_rank}.log"
    if args.framework == "sglang":
        _maybe_activate_kernel_shape_tool(env)
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

    # Self-contained GPU metrics: run rocm-smi sampling on this GPU pod (same fields as the single-node Magpie
    # GPUMonitor) and stream to shared FS for the client to harvest.
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
