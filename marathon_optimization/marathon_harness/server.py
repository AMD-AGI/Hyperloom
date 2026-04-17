"""Inference server lifecycle — health check, kill, restart, GPU availability."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_server_proc: asyncio.subprocess.Process | None = None

SERVER_KILL_WAIT_S = 10
HEALTH_TIMEOUT_S = 900  # Large MoE models can take 10-12 min to load
HEALTH_POLL_S = 3
GPU_BUSY_THRESHOLD = 0.20  # >20% VRAM usage = busy
GPU_MIN_FREE_RATIO = 0.90  # For auto-selection, need >=90% VRAM free


async def _run(cmd: str, timeout: float = 30) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, (stdout or b"").decode(errors="replace")
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return -1, "timeout"


async def health_check(host: str = "0.0.0.0", port: int = 8888) -> bool:
    code, body = await _run(f"curl -sf http://{host}:{port}/health", timeout=5)
    return code == 0


async def deep_health_check(
    host: str = "0.0.0.0", port: int = 8888, model: str = "",
) -> bool:
    """Verify server can produce tokens, not just that /health returns 200."""
    if not await health_check(host, port):
        return False
    completions_url = f"http://{host}:{port}/v1/completions"
    payload = {"prompt": "test", "max_tokens": 1}
    if model:
        payload["model"] = model
    code, body = await _run(
        f"curl -sf --max-time 30 -H 'Content-Type: application/json' "
        f"-d '{json.dumps(payload)}' {completions_url}",
        timeout=35,
    )
    if code != 0:
        log.warning("Deep health check failed: inference probe returned code %d", code)
        return False
    try:
        data = json.loads(body)
        if data.get("choices"):
            return True
        log.warning("Deep health check: no choices in response: %s", body[:200])
    except (json.JSONDecodeError, KeyError):
        log.warning("Deep health check: unparseable response: %s", body[:200])
    return False


async def kill_server() -> bool:
    """Kill sglang / vllm server processes, wait for GPU memory release."""
    global _server_proc
    log.info("Killing inference server …")

    if _server_proc is not None:
        try:
            _server_proc.terminate()
            try:
                await asyncio.wait_for(_server_proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                _server_proc.kill()
        except ProcessLookupError:
            pass
        _server_proc = None

    await _run("pkill -f 'sglang.srt' || true")
    await _run("pkill -f 'vllm.entrypoints' || true")
    await _run("pkill -f 'vllm serve' || true")
    await asyncio.sleep(SERVER_KILL_WAIT_S)

    code, out = await _run("fuser -v /dev/dri/renderD* 2>&1 || true", timeout=10)
    if "python" in out.lower():
        log.warning("GPU processes still running after kill, force-killing …")
        await _run("fuser -k /dev/dri/renderD* 2>/dev/null || true")
        await asyncio.sleep(3)

    zombies = await _find_zombie_gpu_pids()
    if zombies:
        log.warning(
            "Zombie GPU allocations detected after server kill (dead PIDs: %s). "
            "VRAM on those devices may be unusable until pod restart. "
            "serve_tp1.sh will auto-select a clean GPU.",
            zombies,
        )

    return not await health_check()


async def start_server(config: dict[str, Any]) -> bool:
    """Launch server with config and wait until healthy."""
    global _server_proc
    cmd = config.get("launch_command", "")
    if not cmd:
        log.error("No launch_command in server config")
        return False

    log.info("Starting server: %s", cmd[:120])
    _server_proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    return await wait_healthy()


async def restart_server(config: dict[str, Any]) -> bool:
    await kill_server()
    return await start_server(config)


async def wait_healthy(
    host: str = "0.0.0.0",
    port: int = 8888,
    timeout_s: float = HEALTH_TIMEOUT_S,
) -> bool:
    """Poll /health until 200 or timeout."""
    elapsed = 0.0
    while elapsed < timeout_s:
        if await health_check(host, port):
            log.info("Server healthy after %.1fs", elapsed)
            return True
        await asyncio.sleep(HEALTH_POLL_S)
        elapsed += HEALTH_POLL_S
    log.error("Server not healthy after %.0fs", timeout_s)
    return False


async def _find_zombie_gpu_pids() -> list[int]:
    """Find KFD process entries whose owning PID is dead (zombie GPU allocs)."""
    zombies: list[int] = []
    kfd_proc = Path("/sys/class/kfd/kfd/proc")
    if not kfd_proc.is_dir():
        return zombies
    try:
        for entry in kfd_proc.iterdir():
            pid_str = entry.name
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if not Path(f"/proc/{pid}").exists():
                zombies.append(pid)
    except OSError:
        pass
    return zombies


async def _get_per_gpu_vram() -> list[dict[str, float]]:
    """Return [{gpu_id, used_bytes, total_bytes, free_ratio}, ...] for all GPUs."""
    gpus: list[dict[str, float]] = []
    if shutil.which("rocm-smi"):
        code, out = await _run("rocm-smi --showmeminfo vram", timeout=10)
        if code == 0:
            current: dict[str, float] = {}
            for line in out.splitlines():
                if "VRAM Total Memory" in line and "Used" not in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            current["total"] = float(parts[-1].strip())
                        except ValueError:
                            pass
                elif "VRAM Total Used" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            current["used"] = float(parts[-1].strip())
                        except ValueError:
                            pass
                    if "total" in current and "used" in current:
                        total = current["total"]
                        used = current["used"]
                        free_ratio = (1.0 - used / total) if total > 0 else 0.0
                        gpus.append({
                            "gpu_id": len(gpus),
                            "used_bytes": used,
                            "total_bytes": total,
                            "free_ratio": free_ratio,
                        })
                        current = {}
    elif shutil.which("nvidia-smi"):
        code, out = await _run(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            timeout=10,
        )
        if code == 0:
            for i, line in enumerate(out.strip().splitlines()):
                parts = line.split(",")
                if len(parts) == 2:
                    used = float(parts[0].strip()) * 1024 * 1024
                    total = float(parts[1].strip()) * 1024 * 1024
                    free_ratio = (1.0 - used / total) if total > 0 else 0.0
                    gpus.append({
                        "gpu_id": i,
                        "used_bytes": used,
                        "total_bytes": total,
                        "free_ratio": free_ratio,
                    })
    return gpus


async def pick_clean_gpu(min_free_ratio: float = GPU_MIN_FREE_RATIO) -> int | None:
    """Return the ID of the first GPU with sufficient free VRAM, or None."""
    gpus = await _get_per_gpu_vram()
    for g in gpus:
        if g["free_ratio"] >= min_free_ratio:
            log.info("Auto-selected GPU %d (%.1f%% free)",
                     int(g["gpu_id"]), g["free_ratio"] * 100)
            return int(g["gpu_id"])

    zombies = await _find_zombie_gpu_pids()
    if zombies:
        log.warning("No clean GPU found. Zombie GPU PIDs detected: %s", zombies)
    else:
        log.warning("No clean GPU found (all GPUs above %.0f%% VRAM usage)",
                    (1 - min_free_ratio) * 100)
    return None


async def gpu_available() -> bool:
    """True if at least one GPU has memory usage below GPU_BUSY_THRESHOLD.

    Also logs zombie GPU allocations as warnings.
    """
    zombies = await _find_zombie_gpu_pids()
    if zombies:
        log.warning("Zombie GPU allocations from dead PIDs: %s", zombies)

    gpus = await _get_per_gpu_vram()
    if not gpus:
        code, out = await _run("fuser /dev/dri/renderD* 2>/dev/null", timeout=5)
        if out.strip():
            return False
        return code != 0

    for g in gpus:
        if g["free_ratio"] >= (1.0 - GPU_BUSY_THRESHOLD):
            return True
    return False
