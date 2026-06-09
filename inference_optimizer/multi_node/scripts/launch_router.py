#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PD-disaggregation router launcher (head pod, multi-node).

In PD mode the prefill/decode groups listen on internal ports only; this
script starts the router fronting them at the public 8888 port, detached
via ``nohup`` + ``setsid`` so the dashboard job can exit. Supports the
sglang_router and vllm routers (vllm command overridable via
``--vllm-router-cmd``). If the router dies within 0.5 s of spawn it reads
the log tail and raises; otherwise returns 0 and emits the router PID.
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

# Public port the router binds (matches launch_multinode._INFERENCE_PORT and SaFE Service.targetPort).
_PUBLIC_PORT = 8888
_DEFAULT_PID_FILE = "/tmp/multi_node_pids/router.pid"
_DEFAULT_LOG_FILE = "/tmp/multi_node_logs/router.log"


def _log(msg: str) -> None:
    """Stderr line with timestamp."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_router {ts}] {msg}\n")
    sys.stderr.flush()


def _build_sglang_router_cmd(
    prefill_url: str,
    decode_url: str,
    public_port: int,
) -> list[str]:
    """Compose the sglang_router PD-disaggregation launch command (one prefill + one decode group)."""
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
    """Compose the vllm router/proxy launch command; ``--vllm-router-cmd`` overrides it ({prefill}/{decode}/{port} placeholders)."""
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
    """Run ``cmd`` detached via bash+nohup+setsid so it survives the dashboard job exit and dies cleanly under kill_multinode."""
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
