"""Local fallback source.

Wraps a small set of best-effort probes that work on any host: a SQLite
read of the Coordinator session DB, a ``shutil.disk_usage`` sample,
optional ``ps``/``rocm-smi``/``nvidia-smi`` invocations, a tail of a
configured log file plus error-pattern extraction, and an HTTP probe
of locally-running inference servers.  Any sub-probe that fails
returns empty data without raising — :class:`LocalProbeSource` only
raises :class:`SourceUnavailable` when *every* sub-probe yields
nothing, so :class:`DegradeRouter` does not get stuck switching back
and forth.

The probes here intentionally only collect data the agent itself can
see on the host. Cluster-wide GPU time-series, workload inference
metrics, and node-level fault detection stay with primus-robust /
robustness-server (see plan §6.1).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx

from .base import Source, SourceData, SourceUnavailable


log = logging.getLogger(__name__)


# Default process patterns we surface in ``local_processes``. Matches
# the existing monitors so refactoring stays drop-in.
_DEFAULT_PROCESS_PATTERNS: tuple[str, ...] = (
    "sglang.srt",
    "vllm.entrypoints",
    "vllm serve",
    "benchmark_serving",
)


# Default error patterns extracted from local logs. Conservative — only
# unambiguous failure markers — to keep false-positive RCA prompts low.
_DEFAULT_LOG_ERROR_PATTERNS: tuple[str, ...] = (
    r"CUDA out of memory",
    r"hipErrorOutOfMemory",
    r"HIP out of memory",
    r"Segmentation fault",
    r"core dumped",
    r"NCCL error",
    r"RuntimeError",
    r"Killed",
    r"OOMKilled",
    r"failed to allocate",
)


@dataclass
class LocalProbeConfig:
    """Inputs the LocalProbe needs from the agent config."""

    session_dir: Path | None = None
    server_log_path: Path | None = None
    log_tail_lines: int = 200
    disk_mountpoints: tuple[str, ...] = ("/",)
    process_patterns: tuple[str, ...] = _DEFAULT_PROCESS_PATTERNS
    coordinator_event_limit: int = 200
    log_error_patterns: tuple[str, ...] = _DEFAULT_LOG_ERROR_PATTERNS
    log_error_window_lines: int = 500
    health_probe_targets: tuple[str, ...] = ()
    health_probe_timeout_s: float = 1.5

    @property
    def conductor_db_path(self) -> Path | None:
        if self.session_dir is None:
            return None
        return self.session_dir / "storage" / "conductor.db"


@dataclass
class _ProbeOutcome:
    success: bool
    detail: str = ""


class LocalProbeSource:
    """Minimum-effort local data source for the reactor.

    Each :meth:`fetch` call runs the configured sub-probes
    sequentially. CPU-bound bits (sqlite, subprocess) are off-loaded
    to a thread pool so the tick stays responsive.
    """

    name = "local-probe"

    def __init__(self, config: LocalProbeConfig | None = None) -> None:
        self._config = config or LocalProbeConfig()

    async def fetch(self, ctx: Any) -> SourceData:
        cfg = self._config
        coordinator_events = await asyncio.to_thread(
            _read_coordinator_events,
            cfg.conductor_db_path,
            cfg.coordinator_event_limit,
        )
        local_disk = await asyncio.to_thread(_sample_disk, cfg.disk_mountpoints)
        local_processes = await asyncio.to_thread(_sample_processes, cfg.process_patterns)
        local_gpu = await asyncio.to_thread(_sample_gpu)
        local_log_tail = await asyncio.to_thread(
            _tail_log,
            cfg.server_log_path,
            cfg.log_tail_lines,
        )
        local_log_errors = _extract_log_errors(
            local_log_tail,
            cfg.log_error_patterns,
            cfg.log_error_window_lines,
        )
        local_server_health = await _probe_local_servers(
            cfg.health_probe_targets,
            cfg.health_probe_timeout_s,
        )

        any_signal = bool(
            coordinator_events
            or local_disk
            or local_processes
            or local_gpu
            or local_log_tail
            or local_server_health
        )
        if not any_signal:
            raise SourceUnavailable(
                "local probe produced no data (no conductor.db, no disk, no ps, no gpu, no log, no server)"
            )

        return SourceData(
            local_gpu=local_gpu,
            local_processes=local_processes,
            local_disk=local_disk,
            local_log_tail=local_log_tail,
            local_log_errors=local_log_errors,
            local_server_health=local_server_health,
            coordinator_events=coordinator_events,
            sources_used=[self.name],
        )


# ---------------------------------------------------------------------------
# Sub-probes (all sync; called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _read_coordinator_events(
    db_path: Path | None,
    limit: int,
) -> list[dict[str, Any]]:
    if db_path is None or not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            timeout=2.0,
        )
    except sqlite3.Error as exc:
        log.debug("local_probe: cannot open %s: %s", db_path, exc)
        return []
    try:
        conn.row_factory = sqlite3.Row
        # The events table uses ``seq`` as monotonic id; some schemas
        # alias it to ``id`` ??? probe both.
        rows = _try_select(
            conn,
            [
                "SELECT seq AS id, from_agent AS agent, topic, payload, ts "
                "FROM events ORDER BY seq DESC LIMIT ?",
                "SELECT id, agent, topic, payload, timestamp AS ts "
                "FROM events ORDER BY id DESC LIMIT ?",
            ],
            (limit,),
        )
        if not rows:
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row["id"] if "id" in row.keys() else None,
                    "agent": row["agent"] if "agent" in row.keys() else "",
                    "topic": row["topic"] if "topic" in row.keys() else "",
                    "payload": _maybe_decode_json(row["payload"]) if "payload" in row.keys() else None,
                    "ts": row["ts"] if "ts" in row.keys() else None,
                }
            )
        return out
    finally:
        conn.close()


def _try_select(
    conn: sqlite3.Connection,
    candidates: Iterable[str],
    params: tuple,
) -> list[sqlite3.Row]:
    last_err: sqlite3.Error | None = None
    for sql in candidates:
        try:
            return list(conn.execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            last_err = exc
            continue
    if last_err is not None:
        log.debug("local_probe: events select failed: %s", last_err)
    return []


def _maybe_decode_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return None
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _sample_disk(mountpoints: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mp in mountpoints:
        try:
            usage = shutil.disk_usage(mp)
        except OSError as exc:
            log.debug("local_probe: disk_usage(%s) failed: %s", mp, exc)
            continue
        total_gb = usage.total / (1024**3)
        used_gb = (usage.total - usage.free) / (1024**3)
        out[mp] = {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "used_pct": round((used_gb / total_gb) * 100.0, 2) if total_gb else 0.0,
        }
    return out


def _sample_processes(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    if not patterns:
        return []
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,rss=,cmd="],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.debug("local_probe: ps failed: %s", exc)
        return []
    if proc.returncode != 0:
        return []
    out: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, rss_str, cmd = parts
        if not any(pat in cmd for pat in patterns):
            continue
        try:
            pid = int(pid_str)
            rss_kb = int(rss_str)
        except ValueError:
            continue
        out.append(
            {"pid": pid, "rss_mb": round(rss_kb / 1024.0, 1), "cmd": cmd}
        )
    return out


def _sample_gpu() -> dict[str, Any]:
    """Best-effort GPU snapshot using rocm-smi or nvidia-smi.

    Returns an empty dict on any failure (binary missing, exits non-zero,
    parse error). M2 will replace this with a robustness-server cluster
    proxy call, so the M1 implementation is intentionally lightweight.
    """
    snap = _sample_rocm_smi()
    if snap:
        return snap
    return _sample_nvidia_smi()


def _sample_rocm_smi() -> dict[str, Any]:
    if not shutil.which("rocm-smi"):
        return {}
    try:
        proc = subprocess.run(
            [
                "rocm-smi",
                "--showuse",
                "--showmemuse",
                "--showtemp",
                "--showpower",
                "--csv",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except subprocess.TimeoutExpired:
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    gpus = _parse_rocm_smi_csv(proc.stdout)
    if not gpus:
        # Keep the raw text in case of parser drift so RCA still has
        # something to look at.
        return {"raw_csv": proc.stdout, "tool": "rocm-smi"}
    return {"gpus": gpus, "tool": "rocm-smi"}


# Mapping rocm-smi column header -> SourceData GPU snapshot field. The
# column names below are stable across ROCm 5.x / 6.x; new metrics map
# to None and stay in ``raw`` for future inspection.
_ROCM_HEADER_MAP: dict[str, str] = {
    "GPU use (%)": "util_gpu_pct",
    "GPU memory use (%)": "util_mem_pct",
    "Temperature (Sensor edge) (C)": "temperature_c",
    "Temperature (Sensor junction) (C)": "temperature_junction_c",
    "Temperature (Sensor memory) (C)": "temperature_memory_c",
    "Average Graphics Package Power (W)": "power_watts",
    "Current Socket Graphics Package Power (W)": "power_watts",
}


def _parse_rocm_smi_csv(text: str) -> list[dict[str, Any]]:
    """Parse rocm-smi CSV output into ``gpus[]``.

    rocm-smi emits one or more blocks separated by blank lines.  Each
    block starts with a header row whose first column is ``device``
    and is followed by per-device rows (``cardN``).  We accumulate all
    metrics into a single dict per device keyed by ``gpu_id``.
    """
    by_id: dict[int, dict[str, Any]] = {}
    current_columns: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            current_columns = None
            continue
        cells = [c.strip() for c in line.split(",")]
        if not cells:
            continue
        if cells[0].lower() == "device":
            current_columns = cells
            continue
        if current_columns is None or len(cells) < 2:
            continue
        device = cells[0]
        if not device.lower().startswith("card"):
            continue
        try:
            gpu_id = int(device[4:])
        except ValueError:
            continue
        snapshot = by_id.setdefault(gpu_id, {"gpu_id": gpu_id})
        for col_idx in range(1, min(len(cells), len(current_columns))):
            header = current_columns[col_idx]
            field = _ROCM_HEADER_MAP.get(header)
            if not field:
                continue
            value = _coerce_float_or_none(cells[col_idx])
            if value is not None:
                snapshot[field] = value
    out: list[dict[str, Any]] = []
    for k in sorted(by_id):
        snap = by_id[k]
        if len(snap) > 1:  # at least one parsed metric beyond ``gpu_id``
            out.append(snap)
    return out


def _coerce_float_or_none(value: str) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _sample_nvidia_smi() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except subprocess.TimeoutExpired:
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                {
                    "gpu_id": int(parts[0]),
                    "util_gpu_pct": float(parts[1]),
                    "util_mem_pct": float(parts[2]),
                    "temperature_c": float(parts[3]),
                    "vram_used_mb": float(parts[4]),
                    "vram_total_mb": float(parts[5]),
                }
            )
        except ValueError:
            continue
    if not gpus:
        return {}
    return {"gpus": gpus, "tool": "nvidia-smi"}


def _tail_log(path: Path | None, max_lines: int) -> list[str]:
    if path is None or not path.exists() or max_lines <= 0:
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            block = 4096
            data = b""
            while file_size > 0 and data.count(b"\n") <= max_lines:
                read_size = min(block, file_size)
                file_size -= read_size
                handle.seek(file_size)
                data = handle.read(read_size) + data
    except OSError as exc:
        log.debug("local_probe: tail %s failed: %s", path, exc)
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def _extract_log_errors(
    tail: list[str],
    patterns: tuple[str, ...],
    window: int,
) -> list[dict[str, Any]]:
    """Scan the last ``window`` log lines for fatal error patterns.

    Returns one entry per matching line with the pattern that matched
    and the line text trimmed to 240 chars (avoid blowing up the
    `Finding.evidence` payload).
    """
    if not tail or not patterns:
        return []
    compiled = []
    for raw in patterns:
        try:
            compiled.append((raw, re.compile(raw, re.IGNORECASE)))
        except re.error:
            log.debug("local_probe: invalid log pattern: %s", raw)
            continue
    if not compiled:
        return []
    candidate = tail[-window:] if len(tail) > window else tail
    out: list[dict[str, Any]] = []
    for line in candidate:
        for pattern, regex in compiled:
            if regex.search(line):
                out.append(
                    {"pattern": pattern, "line": line[:240]}
                )
                break
    return out


async def _probe_local_servers(
    targets: tuple[str, ...],
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Issue a tiny GET against each target URL.

    Used to detect "process is alive but server is wedged" — a common
    failure mode in single-mode dev where the inference server
    deadlocks and stops accepting requests.  Connection refused /
    timeout each carry a distinct ``status`` so signals can act on
    them.
    """
    if not targets:
        return []
    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(max(0.2, float(timeout_s)))
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in targets:
            entry: dict[str, Any] = {"url": url, "reachable": False, "status": "error"}
            try:
                resp = await client.get(url)
            except httpx.TimeoutException:
                entry["error"] = "timeout"
            except httpx.ConnectError as exc:
                entry["error"] = f"connect: {exc.__class__.__name__}"
            except httpx.RequestError as exc:
                entry["error"] = f"request: {exc.__class__.__name__}: {exc}"
            else:
                entry["status_code"] = resp.status_code
                entry["reachable"] = resp.status_code < 500
                entry["status"] = "ok" if resp.status_code < 400 else "http_error"
            results.append(entry)
    return results


__all__ = ["LocalProbeConfig", "LocalProbeSource"]
