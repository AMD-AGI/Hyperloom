"""Server lifecycle management — launch, health-check, and stop serving processes.

This module is framework-agnostic. It requires a launch script that knows
how to start the serving framework. The script receives environment variables
(MODEL_PATH, PORT, TP, GPUS) and must expose an HTTP /health endpoint.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import SessionConfig

log = logging.getLogger(__name__)

HEALTH_TIMEOUT = int(os.environ.get("HYPERLOOM_HEALTH_TIMEOUT", "1800"))  # large MoE models need extended startup
HEALTH_INTERVAL = 5   # poll interval


@dataclass
class ServerProcess:
    """A managed serving framework process."""

    pid: int
    port: int
    process: subprocess.Popen
    log_path: Path

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def health_check(self) -> bool:
        try:
            r = requests.get(
                f"http://localhost:{self.port}/health",
                timeout=3,
            )
            return r.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def stop(self, grace_sec: float = 10.0) -> None:
        if not self.is_alive():
            return
        log.info("Stopping server (PID %d)...", self.pid)
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            log.warning("Server did not exit gracefully, sending SIGKILL")
            self.process.kill()
            self.process.wait(timeout=5)


def _ensure_port_free(port: int) -> None:
    """Kill any existing process on the target port before launching."""
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=2)
        if r.status_code == 200:
            log.warning("Port %d already occupied — killing existing process", port)
            result = subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                subprocess.run(
                    ["bash", "-c", f"lsof -ti:{port} | xargs kill -9"],
                    capture_output=True, timeout=10,
                )
            time.sleep(3)
    except (requests.ConnectionError, requests.Timeout):
        pass
    except Exception as e:
        log.debug("Port check error (non-fatal): %s", e)


def launch_server(config: SessionConfig) -> ServerProcess:
    """Launch the serving framework via the user-provided launch script.

    The launch script must:
      1. Start a serving process that listens on $PORT
      2. Expose an HTTP GET /health endpoint that returns 200 when ready
      3. Run in the foreground (stdout/stderr captured to server.log)

    Environment variables passed to the script:
      MODEL_PATH, PORT, TP, GPUS, SESSION_DIR
    """
    if not config.launch_script:
        raise FileNotFoundError(
            "--launch-script is required. Provide a script that starts your "
            "serving framework (vLLM, sglang, etc). See examples/ for templates."
        )

    script = Path(config.launch_script)
    if not script.exists():
        raise FileNotFoundError(f"Launch script not found: {config.launch_script}")

    _ensure_port_free(config.port)

    session_dir = Path(config.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "server.log"

    env = _build_env(config)
    cmd = ["bash", str(script.resolve())]

    log.info("Launching server via: %s", script)
    log.info("Server log: %s", log_path)

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    server = ServerProcess(
        pid=proc.pid,
        port=config.port,
        process=proc,
        log_path=log_path,
    )

    log.info("Server PID: %d — waiting for health on port %d...", proc.pid, config.port)
    _wait_for_health(server)

    return server


def _wait_for_health(server: ServerProcess) -> None:
    """Block until server is healthy or timeout."""
    start = time.time()
    while time.time() - start < HEALTH_TIMEOUT:
        if not server.is_alive():
            raise RuntimeError(
                f"Server process died (exit code: {server.process.returncode}). "
                f"Check log: {server.log_path}"
            )
        if server.health_check():
            elapsed = time.time() - start
            log.info("Server healthy after %.1fs", elapsed)
            return
        time.sleep(HEALTH_INTERVAL)

    raise RuntimeError(
        f"Server not healthy after {HEALTH_TIMEOUT}s. Check log: {server.log_path}"
    )


def _build_env(config: SessionConfig) -> dict[str, str]:
    """Pass session config to the launch script via environment variables.

    Injects GPU hardware constants (CU_NUM, GPU_ARCHS) so launch scripts
    don't need to hardcode them per GPU family.
    """
    env = os.environ.copy()
    env["MODEL_PATH"] = config.model_path
    env["PORT"] = str(config.port)
    env["TP"] = str(config.tp) if config.tp else ""
    env["GPUS"] = config.gpus
    env["SESSION_DIR"] = config.session_dir
    env["GPU_TYPE"] = config.gpu_type

    from .gpu import detect_gpu
    spec = detect_gpu()
    if spec:
        for k, v in spec.env_vars.items():
            env.setdefault(k, v)

    return env
