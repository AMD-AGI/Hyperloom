#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Public port the router binds (matches launch_multinode._INFERENCE_PORT).
_PUBLIC_PORT = 8888
_DEFAULT_PID_FILE = str(Path(tempfile.gettempdir()) / "multi_node_pids" / "router.pid")
_DEFAULT_LOG_FILE = str(Path(tempfile.gettempdir()) / "multi_node_logs" / "router.log")


def _log(msg: str) -> None:
    """Stderr line with timestamp.

    Args:
        msg: The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[launch_router {ts}] {msg}\n")
    sys.stderr.flush()


def _router_alive(pid: int) -> bool:
    """Whether ``pid`` is a live, non-zombie router.

    Same reasoning as ``kill_multinode._pid_alive``: the router is spawned under
    ``nohup setsid``, so it is reparented to a PID 1 that does not reap, and a
    dead one lingers as ``<defunct>`` where a bare ``os.kill(pid, 0)`` still
    succeeds. A zombie holds no port, so treating it as running would only stall
    the replacement.

    Args:
        pid: Process id to probe.

    Returns:
        bool: True only when the process exists and is not a zombie.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        rparen = data.rfind(b")")
        if rparen != -1 and data[rparen + 2 : rparen + 3] == b"Z":
            return False
    except OSError:
        return False
    return True


def _retire_previous_router(pid_file: Path, *, grace_s: float = 5.0) -> None:
    """Stop the router this PID file names, if one is still running.

    The spawn below ends with ``echo $! > pid_file``, so without this step the
    previous router is not replaced but orphaned: it goes on holding the public
    port while the file that named it now points at the newcomer, which puts it
    beyond the reach of ``kill_multinode``'s ``router*.pid`` sweep for the rest
    of the pod's life. The new router then finds its port taken.

    Reached on every restart, including the resume paths that skip KILL+LAUNCH
    and so never sweep the pid dir. Nothing is being interrupted mid-flight --
    a restart is the caller -- and a router is a proxy rather than a model
    server, so replacing one costs a moment instead of a weight load.

    Args:
        pid_file: Path the previous router's PID was written to.
        grace_s: Seconds to wait for a clean exit before SIGKILL.
    """
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return
    if pid <= 0 or not _router_alive(pid):
        return
    _log(f"replacing router pid={pid}, which still holds port {_PUBLIC_PORT}")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _router_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        _log(f"router pid={pid} ignored SIGTERM for {grace_s:.0f}s; sent SIGKILL")
    except OSError:
        return


def _build_sglang_router_cmd(
    prefill_url: str,
    decode_url: str,
    public_port: int,
) -> list[str]:
    """Compose the sglang_router PD-disaggregation launch command.

    Args:
        prefill_url: Internal prefill group HTTP endpoint.
        decode_url: Internal decode group HTTP endpoint.
        public_port: Port the router binds for the client.

    Returns:
        list[str]: The argv for launching the sglang router.
    """
    return [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        prefill_url,
        "--decode",
        decode_url,
        "--host",
        "0.0.0.0",  # nosec B104 - router is the public multi-node endpoint.
        "--port",
        str(public_port),
    ]


def _build_vllm_router_cmd(
    prefill_url: str,
    decode_url: str,
    public_port: int,
    override_cmd: str = "",
) -> list[str]:
    """Compose the vllm router/proxy launch command (``override_cmd`` supports {prefill}/{decode}/{port} placeholders).

    Args:
        prefill_url: Internal prefill group HTTP endpoint.
        decode_url: Internal decode group HTTP endpoint.
        public_port: Port the router binds for the client.
        override_cmd: Optional full command template; supports ``{prefill}``,
            ``{decode}``, and ``{port}`` placeholders.

    Returns:
        list[str]: The argv for launching the vllm router/proxy.
    """
    if override_cmd:
        rendered = (
            override_cmd.replace("{prefill}", prefill_url)
            .replace("{decode}", decode_url)
            .replace("{port}", str(public_port))
        )
        return shlex.split(rendered)
    return [
        "python3",
        "-m",
        "vllm.entrypoints.openai.disagg_proxy",
        "--prefill-url",
        prefill_url,
        "--decode-url",
        decode_url,
        "--host",
        "0.0.0.0",  # nosec B104 - router is the public multi-node endpoint.
        "--port",
        str(public_port),
    ]


def _detach_router(
    cmd: list[str],
    log_file: Path,
    pid_file: Path,
) -> int:
    """Run ``cmd`` detached via bash+nohup+setsid so it survives the dashboard job exit.

    Args:
        cmd: The router argv to launch.
        log_file: Path the router's stdout/stderr is appended to.
        pid_file: Path the spawned router PID is written to.

    Returns:
        int: The PID of the detached router process.

    Raises:
        RuntimeError: If the spawn shell fails, the PID file is missing or
            invalid, or the router is not alive 0.5s after launch.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    _retire_previous_router(pid_file)
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
        env=sub_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"router detach spawn shell failed rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"router pid file {pid_file} missing/invalid: {exc}") from exc

    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except OSError as exc:
        tail = ""
        try:
            if log_file.is_file() and log_file.stat().st_size > 0:
                tail = log_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-8000:]
        except OSError:
            tail = "<could not read log>"
        raise RuntimeError(f"router pid={pid} not alive after 0.5s ({exc}); log tail:\n{tail}") from exc
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
    p.add_argument(
        "--framework",
        required=True,
        choices=("sglang", "vllm"),
        help="picks router implementation: sglang_router vs vllm proxy",
    )
    p.add_argument("--prefill-url", required=True, help="internal prefill HTTP endpoint, e.g. http://10.0.0.1:30000")
    p.add_argument("--decode-url", required=True, help="internal decode HTTP endpoint, e.g. http://10.0.0.2:30001")
    p.add_argument(
        "--public-port",
        type=int,
        default=_PUBLIC_PORT,
        help=f"port the router binds for the magpie client (default {_PUBLIC_PORT})",
    )
    p.add_argument(
        "--pid-file",
        default=_DEFAULT_PID_FILE,
        help=f"router PID file (default {_DEFAULT_PID_FILE}). kill_multinode.py picks up router*.pid here.",
    )
    p.add_argument(
        "--log-file", default=_DEFAULT_LOG_FILE, help=f"router stdout/stderr log (default {_DEFAULT_LOG_FILE})"
    )
    p.add_argument(
        "--vllm-router-cmd",
        default="",
        help="(vllm only) override entire router command; supports {prefill} / {decode} / {port} placeholders",
    )
    args = p.parse_args()

    fw = args.framework.lower()
    if fw == "sglang":
        cmd = _build_sglang_router_cmd(
            args.prefill_url,
            args.decode_url,
            args.public_port,
        )
    else:
        cmd = _build_vllm_router_cmd(
            args.prefill_url,
            args.decode_url,
            args.public_port,
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
