#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PD-disaggregation router launcher (head pod, multi-node).

When ``inference_optimizer.multi_node restart-server`` runs in PD mode,
``launch_multinode.py`` spawns the prefill / decode sglang or vllm
server groups but does NOT bind the public 8888 port — every group
listens on internal ports (30000 / 30001) only. This script then runs
on the head pod (via Ray Dashboard submission) and starts the router
that fronts those internal endpoints at the public 8888 port. The
router is detached with ``nohup`` + ``setsid`` so the dashboard job
can exit while the router keeps serving.

Two router implementations are supported:

* **sglang**: ``python3 -m sglang_router.launch_router --pd-disaggregation
  --prefill <url> --decode <url> --host 0.0.0.0 --port 8888``. Requires
  the ``sglang-router`` PyPI package in the RayJob image.

* **vllm**: ``vllm-project/production-stack`` ships an orchestrated
  router supporting disaggregated prefill — at the time of writing it
  is invoked via ``python3 -m vllm.entrypoints.openai.api_server
  --kv-transfer-config ...`` on a dedicated proxy node (no separate
  binary). For the v1 of this script we route vllm PD via a thin
  ``proxy_server.py`` style ASGI app. Concrete vllm command is gated
  by ``--vllm-router-cmd`` so we don't hard-code something the upstream
  may rename in the next release.

Failure semantics: if the router exits within 0.5 s of spawn we read
the last 8 KiB of its log and raise. Otherwise we return 0 and emit
the router PID so callers (cli.py) can persist it in state.json.
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

# Public port the router binds. Matches launch_multinode.py's
# _INFERENCE_PORT and SaFE Service.targetPort. magpie's
# BENCHMARK_BASE_URL points here regardless of pd_mode.
_PUBLIC_PORT = 8888
_DEFAULT_PID_FILE = "/tmp/multi_node_pids/router.pid"
_DEFAULT_LOG_FILE = "/tmp/multi_node_logs/router.log"


def _log(msg: str) -> None:
    """Stderr line with timestamp; mirrors launch_multinode style.

    Args:
        msg (str): The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_router {ts}] {msg}\n")
    sys.stderr.flush()


def _build_sglang_router_cmd(
    prefill_url: str,
    decode_url: str,
    public_port: int,
) -> list[str]:
    """Compose the sglang_router PD-disaggregation launch command.

    sglang_router accepts repeated ``--prefill`` / ``--decode`` flags
    when there are multiple workers per side; we emit one of each for
    the v1 of this launcher (matching the launch_multinode.py default
    of one prefill server group + one decode server group).

    Args:
        prefill_url (str): Internal prefill server HTTP endpoint.
        decode_url (str): Internal decode server HTTP endpoint.
        public_port (int): Public port the router should bind.

    Returns:
        list[str]: The argv list to launch the sglang router.
    """
    return [
        "python3", "-m", "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill", prefill_url,
        "--decode", decode_url,
        "--host", "0.0.0.0",
        "--port", str(public_port),
    ]


def _build_vllm_router_cmd(
    prefill_url: str,
    decode_url: str,
    public_port: int,
    override_cmd: str = "",
) -> list[str]:
    """Compose the vllm router/proxy launch command.

    vllm has not converged on a single official PD router CLI as of
    2026; ``--vllm-router-cmd`` lets the operator override the entire
    command (with ``{prefill}`` / ``{decode}`` / ``{port}`` placeholders).
    Default falls back to the production-stack disagg proxy entrypoint
    name; if your image ships a different binary, supply the override.

    Args:
        prefill_url (str): Internal prefill server HTTP endpoint.
        decode_url (str): Internal decode server HTTP endpoint.
        public_port (int): Public port the router should bind.
        override_cmd (str): Optional full command template with
            ``{prefill}`` / ``{decode}`` / ``{port}`` placeholders. When
            non-empty it replaces the default command entirely.

    Returns:
        list[str]: The argv list to launch the vllm router/proxy.
    """
    if override_cmd:
        rendered = (
            override_cmd
            .replace("{prefill}", prefill_url)
            .replace("{decode}", decode_url)
            .replace("{port}", str(public_port))
        )
        return shlex.split(rendered)
    return [
        "python3", "-m", "vllm.entrypoints.openai.disagg_proxy",
        "--prefill-url", prefill_url,
        "--decode-url", decode_url,
        "--host", "0.0.0.0",
        "--port", str(public_port),
    ]


def _detach_router(
    cmd: list[str],
    log_file: Path,
    pid_file: Path,
) -> int:
    """Run ``cmd`` detached via bash+nohup+setsid.

    Reuses the same pattern as launch_multinode._detach_framework_launch
    so the router survives the ray dashboard job exit and is killed
    cleanly by kill_multinode.py (which SIGTERMs the process group).

    Args:
        cmd (list[str]): The router argv to launch.
        log_file (Path): File to which router stdout/stderr is redirected.
        pid_file (Path): File where the detached router PID is written.

    Returns:
        int: The PID of the detached router process.

    Raises:
        RuntimeError: If the spawn shell fails, the PID file is missing or
            invalid, or the router is not alive 0.5s after spawn.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_q = shlex.quote(str(log_file))
    pid_q = shlex.quote(str(pid_file))
    inner = " ".join(shlex.quote(c) for c in cmd)
    launches = (
        f"if command -v setsid >/dev/null 2>&1; then "
        f"nohup setsid {inner} >>{log_q} 2>&1 & "
        f"else nohup {inner} >>{log_q} 2>&1 & fi; "
        f"echo $! > {pid_q}"
    )
    shell_cmd = f"set -euo pipefail; : >{log_q}; {launches}"

    sub_env = dict(os.environ)
    sub_env.setdefault("PYTHONUNBUFFERED", "1")
    venv_bin = "/opt/venv/bin"
    cur_path = sub_env.get("PATH", "")
    if venv_bin not in (cur_path or "").split(":"):
        sub_env["PATH"] = f"{venv_bin}:{cur_path}" if cur_path else venv_bin

    proc = subprocess.run(
        ["/bin/bash", "-lc", shell_cmd],
        env=sub_env, capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"router detach spawn shell failed rc={proc.returncode} "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"router pid file {pid_file} missing/invalid: {exc}"
        ) from exc

    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        tail = ""
        try:
            if log_file.is_file() and log_file.stat().st_size > 0:
                tail = log_file.read_text(
                    encoding="utf-8", errors="replace",
                )[-8000:]
        except OSError:
            tail = "<could not read log>"
        raise RuntimeError(
            f"router pid={pid} not alive after 0.5s ({exc}); "
            f"log tail:\n{tail}"
        ) from exc
    return pid


def main() -> int:
    """Parse CLI arguments and detach the PD router on the head pod.

    Builds the framework-specific router command, detaches it, prints a JSON
    summary (framework, PID, URLs, file paths) to stdout, and returns.

    Returns:
        int: Process exit code; ``0`` on success, ``1`` if the router failed
        to stay alive after launch.
    """
    p = argparse.ArgumentParser(
        prog="launch_router.py",
        description="Detach the PD-disaggregation router on the head pod.",
    )
    p.add_argument("--framework", required=True, choices=("sglang", "vllm"),
                   help="picks router implementation: sglang_router vs vllm proxy")
    p.add_argument("--prefill-url", required=True,
                   help="internal prefill HTTP endpoint, e.g. http://10.0.0.1:30000")
    p.add_argument("--decode-url", required=True,
                   help="internal decode HTTP endpoint, e.g. http://10.0.0.2:30001")
    p.add_argument("--public-port", type=int, default=_PUBLIC_PORT,
                   help=f"port the router binds for the magpie client "
                        f"(default {_PUBLIC_PORT})")
    p.add_argument("--pid-file", default=_DEFAULT_PID_FILE,
                   help=f"router PID file (default {_DEFAULT_PID_FILE}). "
                        f"kill_multinode.py picks up router*.pid here.")
    p.add_argument("--log-file", default=_DEFAULT_LOG_FILE,
                   help=f"router stdout/stderr log (default {_DEFAULT_LOG_FILE})")
    p.add_argument("--vllm-router-cmd", default="",
                   help="(vllm only) override entire router command; supports "
                        "{prefill} / {decode} / {port} placeholders")
    args = p.parse_args()

    fw = args.framework.lower()
    if fw == "sglang":
        cmd = _build_sglang_router_cmd(
            args.prefill_url, args.decode_url, args.public_port,
        )
    else:
        cmd = _build_vllm_router_cmd(
            args.prefill_url, args.decode_url, args.public_port,
            override_cmd=args.vllm_router_cmd,
        )

    _log(f"framework={fw} cmd={' '.join(cmd)}")
    _log(f"pid_file={args.pid_file} log_file={args.log_file}")

    try:
        pid = _detach_router(cmd, Path(args.log_file), Path(args.pid_file))
    except RuntimeError as exc:
        _log(f"ERROR {exc}")
        return 1

    summary = {
        "framework": fw,
        "router_pid": pid,
        "router_url": f"http://0.0.0.0:{args.public_port}",
        "prefill_url": args.prefill_url,
        "decode_url": args.decode_url,
        "pid_file": args.pid_file,
        "log_file": args.log_file,
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()
    _log(f"router pid={pid} alive; detached. dashboard job may exit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
