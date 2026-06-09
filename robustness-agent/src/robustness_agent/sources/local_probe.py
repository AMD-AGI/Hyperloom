# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local fallback source.

Wraps best-effort host-local probes (Coordinator SQLite read, disk
usage, ``ps``/``rocm-smi``/``nvidia-smi``, log tail + error-pattern
extraction, HTTP server probe). A failing sub-probe returns empty data
without raising; :class:`LocalProbeSource` raises
:class:`SourceUnavailable` only when *every* sub-probe yields nothing,
so :class:`DegradeRouter` does not flap. Cluster-wide metrics and
node-level fault detection stay with robustness-server (plan §6.1).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from .base import SourceData, SourceUnavailable


log = logging.getLogger(__name__)


# Process patterns surfaced in ``local_processes``: every owner that may
# legitimately hold AMD GPU VRAM, so gpu_memory_leaked can distinguish
# leaked VRAM (no live owner) from an active server. Must cover vLLM v1
# ``EngineCore-`` children that held VRAM in the 2026-05-18 leak post-mortem.
_DEFAULT_PROCESS_PATTERNS: tuple[str, ...] = (
    # SGLang
    "sglang.srt",
    "sglang.launch_server",
    # vLLM
    "vllm.entrypoints",
    "vllm serve",
    "vllm.v1.engine.core",
    "vllm.engine.async_llm_engine",
    "EngineCore",            # generic substring; covers ``EngineCore-`` child PIDs
    # Magpie / InferenceX benchmark harness
    "Magpie",
    "inferencex",
    # Ray + per-task workers (kernel_opt and GEAK schedule via Ray)
    "ray::IDLE",
    "raylet",
    # hipcc stuck mid-build holds GPU locks; list it so it doesn't mask as a live owner.
    "hipcc",
    # Generic benchmark serving client (Magpie sub-process)
    "benchmark_serving",
)


# Conservative log error markers; severity tier decided by
# :data:`signals.local_health._HIGH_SEVERITY_PATTERNS`.
# **Order matters** — :func:`_extract_log_errors` short-circuits on the first
# match per line, so MORE SPECIFIC patterns MUST come BEFORE GENERIC ones.
_DEFAULT_LOG_ERROR_PATTERNS: tuple[str, ...] = (
    # D1/E5 specific patterns — first so generic ``RuntimeError`` / ``Killed`` don't shadow them.
    # E5 critic-agent runtime stuck.
    r"runtime\.cli .* timed out after \d+s",
    # D1 vLLM v1 EngineCore subprocess crashes.
    r"RuntimeError: Engine core initialization failed",
    r"Engine core .* died",
    # D1 sglang tokenizer worker death.
    r"tokenizer worker .* died",
    # D1 aiter JIT compile / hipcc signal exits.
    r"aiter .* compilation failed",
    r"hipcc .* signal",
    # D1 model architecture mismatch (DSR1 MTP / MLA paths).
    r"MLA.*not supported",
    r"MTP draft .* unavailable",
    # D1 accuracy gate failures.
    r"accuracy .* gate failed",
    r"MMLU .* below threshold",
    # D1 KFD resource exhaustion (distinct from OOM).
    r"cudaErrorOutOfDevice",
    r"HSA_STATUS_ERROR_OUT_OF_RESOURCES",
    # D1 matrix library errors.
    r"ROCblas.*Status\s*\d+",
    r"hipBLAS.*Error",
    # D1 NCCL communication timeout (multi-GPU paths).
    r"NCCL WARN .* timeout",
    # D1 server port reuse.
    r"Address already in use",
    # D1 model checkpoint load failure.
    r"Failed to load checkpoint",
    # Existing classic patterns — GPU OOM / segfault / NCCL / OOMKilled.
    r"CUDA out of memory",
    r"hipErrorOutOfMemory",
    r"HIP out of memory",
    r"Segmentation fault",
    r"core dumped",
    r"NCCL error",
    r"OOMKilled",
    r"failed to allocate",
    # Generic fallbacks — ordered LAST so they don't shadow the
    # specific patterns above.
    r"RuntimeError",
    r"Killed",
)


@dataclass
class LocalProbeConfig:
    """Inputs the LocalProbe needs from the agent config."""

    session_dir: Path | None = None
    # Single server log path (legacy); ``extra_server_log_globs`` picks up
    # per-variant grid-run logs. Empty default keeps single-log hosts working.
    server_log_path: Path | None = None
    extra_server_log_globs: tuple[str, ...] = (
        "runs/*/*/server.log",
        "runs/*/*/server_log",
        "runs/*/server.log",
        # depth-3/4 grid_runner layout (per-variant logs live deeper than legacy single-run).
        "runs/*/*/*/server.log",
        "runs/*/*/*/*/server.log",
    )
    # Max extra log files per tick (mtime desc); 5 covers recent grid variants cheaply.
    max_extra_server_logs: int = 5
    log_tail_lines: int = 200
    # Surface ``/dev/shm`` (SGLang/vLLM SHM_* queues, can exhaust mid-session)
    # alongside ``/`` so signals fire ``shm_pressure`` separately from ``disk_pressure``.
    disk_mountpoints: tuple[str, ...] = ("/", "/dev/shm")
    process_patterns: tuple[str, ...] = _DEFAULT_PROCESS_PATTERNS
    coordinator_event_limit: int = 200
    log_error_patterns: tuple[str, ...] = _DEFAULT_LOG_ERROR_PATTERNS
    log_error_window_lines: int = 500
    health_probe_targets: tuple[str, ...] = ()
    health_probe_timeout_s: float = 1.5
    # A5/A6/A7 sub-probe knobs. ``ray_probe_enabled`` skips the Ray check on
    # head-less nodes; ``aiter_jit_dir`` falls back to the aiter wheel when unset;
    # ``fd_probe_pid`` defaults to the current Coordinator process (os.getpid()).
    ray_probe_enabled: bool = True
    ray_probe_timeout_s: float = 5.0
    aiter_jit_dir: Path | None = None
    fd_probe_pid: int | None = None
    fd_probe_enabled: bool = True
    # G — decision-audit probe. Scans ``runs/integrate/*/result.json``,
    # ``results/ci_metrics*.json``, ``kernel-agent/runs/*/optimization_attempts.jsonl``
    # into :attr:`SourceData.local_decision_audit`.
    decision_audit_enabled: bool = True
    # Max recent integrate result.json per tick (mtime desc); 20 catches a
    # same-fingerprint KEEP loop within a tick without blowing the IO budget.
    decision_audit_max_integrate: int = 20
    # Max tail entries pulled from ``optimization_attempts.jsonl``.
    decision_audit_max_oob_attempts: int = 50
    # C — preflight probe. Reads ``manifest.json`` into local_manifest and
    # aggregates ``profiles/kernel_breakdown.json`` into local_kernel_breakdown.
    preflight_enabled: bool = True
    # E — critic-health probe. Scans ``critic-workdir/*/judge_bundle.json`` for
    # KB-unreachable markers and counts workdir entries (``critic_prune_stuck``).
    critic_health_enabled: bool = True
    # Max ``critic-workdir/<turn>/`` dirs scanned per tick; recent ones suffice
    # to catch a consecutive-tick KB outage.
    max_critic_judge_bundles: int = 20
    # I — state-integrity probe. Scans state.json / coordinator.db-wal / leases /
    # agent JSONLs / Coordinator PID file.
    state_integrity_enabled: bool = True
    # Optimiser-run dir under ``session_dir`` holding ``run_*.pid``
    # (defaults to ``$USER_DATA_PATH/optimizer_runs/`` per SKILL.md).
    optimizer_runs_dirname: str = "optimizer_runs"
    # J — external-deps probe. Reads $OPENAI_BASE_URL etc. from env.
    external_deps_enabled: bool = True
    # Per-mount stat-latency budget; above this → ``wekafs_degraded`` (5s per SKILL.md).
    external_mount_stat_timeout_s: float = 5.0
    # Override gateway probe URL; empty → derive from ``$OPENAI_BASE_URL`` + ``/models``.
    external_gateway_probe_url: str = ""

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
            _tail_logs,
            cfg.server_log_path,
            cfg.session_dir,
            cfg.extra_server_log_globs,
            cfg.max_extra_server_logs,
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
        local_ray = (
            await asyncio.to_thread(_probe_ray_head, cfg.ray_probe_timeout_s)
            if cfg.ray_probe_enabled
            else {}
        )
        local_fd = (
            await asyncio.to_thread(_sample_fd_usage, cfg.fd_probe_pid)
            if cfg.fd_probe_enabled
            else {}
        )
        local_aiter_jit = await asyncio.to_thread(
            _sample_aiter_jit, cfg.aiter_jit_dir,
        )
        local_decision_audit = (
            await asyncio.to_thread(
                _sample_decision_audit,
                cfg.session_dir,
                cfg.decision_audit_max_integrate,
                cfg.decision_audit_max_oob_attempts,
            )
            if cfg.decision_audit_enabled
            else {}
        )
        local_manifest = (
            await asyncio.to_thread(_load_manifest_extras, cfg.session_dir)
            if cfg.preflight_enabled
            else {}
        )
        local_kernel_breakdown = (
            await asyncio.to_thread(_load_kernel_breakdown, cfg.session_dir)
            if cfg.preflight_enabled
            else {}
        )
        local_critic_health = (
            await asyncio.to_thread(
                _sample_critic_workdir,
                cfg.session_dir,
                cfg.max_critic_judge_bundles,
            )
            if cfg.critic_health_enabled
            else {}
        )
        local_state_integrity = (
            await asyncio.to_thread(
                _sample_state_integrity,
                cfg.session_dir,
                cfg.optimizer_runs_dirname,
            )
            if cfg.state_integrity_enabled
            else {}
        )
        local_external_deps = (
            await _probe_external_deps(
                cfg.external_gateway_probe_url,
                cfg.external_mount_stat_timeout_s,
                cfg.health_probe_timeout_s,
            )
            if cfg.external_deps_enabled
            else {}
        )

        any_signal = bool(
            coordinator_events
            or local_disk
            or local_processes
            or local_gpu
            or local_log_tail
            or local_server_health
            or local_ray
            or local_fd
            or local_aiter_jit
            or local_decision_audit
            or local_manifest
            or local_kernel_breakdown
            or local_critic_health
            or local_state_integrity
            or local_external_deps
        )
        if not any_signal:
            raise SourceUnavailable(
                "local probe produced no data (all sub-probes empty)"
            )

        return SourceData(
            local_gpu=local_gpu,
            local_processes=local_processes,
            local_disk=local_disk,
            local_log_tail=local_log_tail,
            local_log_errors=local_log_errors,
            local_server_health=local_server_health,
            local_ray=local_ray,
            local_fd=local_fd,
            local_aiter_jit=local_aiter_jit,
            local_decision_audit=local_decision_audit,
            local_manifest=local_manifest,
            local_kernel_breakdown=local_kernel_breakdown,
            local_critic_health=local_critic_health,
            local_state_integrity=local_state_integrity,
            local_external_deps=local_external_deps,
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
        # ``events`` uses ``seq`` as monotonic id; some schemas alias it to ``id`` — probe both.
        rows = _try_select(
            conn,
            [
                "SELECT seq AS id, from_agent AS agent, topic, payload, ts "
                + "FROM events ORDER BY seq DESC LIMIT ?",
                "SELECT id, agent, topic, payload, timestamp AS ts "
                + "FROM events ORDER BY id DESC LIMIT ?",
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
                "--showmeminfo",
                "vram",
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
        # Keep raw text on parser drift so RCA still has something to look at.
        return {"raw_csv": proc.stdout, "tool": "rocm-smi"}
    return {"gpus": gpus, "tool": "rocm-smi"}


# rocm-smi column header -> SourceData GPU field (stable across ROCm 5.x/6.x).
# The ``VRAM ... (B)`` columns are bytes, translated to MiB via _ROCM_BYTE_TO_MB_FIELDS.
_ROCM_HEADER_MAP: dict[str, str] = {
    "GPU use (%)": "util_gpu_pct",
    "GPU memory use (%)": "util_mem_pct",
    "Temperature (Sensor edge) (C)": "temperature_c",
    "Temperature (Sensor junction) (C)": "temperature_junction_c",
    "Temperature (Sensor memory) (C)": "temperature_memory_c",
    "Average Graphics Package Power (W)": "power_watts",
    "Current Socket Graphics Package Power (W)": "power_watts",
    "VRAM Total Used Memory (B)": "vram_used_mb",
    "VRAM Total Memory (B)": "vram_total_mb",
}

# Byte-valued rocm-smi fields; parser divides by 1024**2 to match nvidia-smi units.
_ROCM_BYTE_TO_MB_FIELDS: frozenset[str] = frozenset({
    "vram_used_mb",
    "vram_total_mb",
})


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
            if value is None:
                continue
            if field in _ROCM_BYTE_TO_MB_FIELDS:
                value = value / (1024.0 * 1024.0)
            snapshot[field] = value
    out: list[dict[str, Any]] = []
    for k in sorted(by_id):
        snap = by_id[k]
        if len(snap) > 1:  # at least one parsed metric beyond ``gpu_id``
            # Derive ``util_mem_pct`` from VRAM used/total when rocm-smi omits the
            # percentage column (optional on older releases); otherwise GpuLeakDetector's
            # ``util_mem_pct >= 99%`` trigger never fires on AMD, and the strict
            # ``free_mb <= 500MB`` fallback (0.25% of a 192GiB MI300X) misses multi-GB leaks.
            if "util_mem_pct" not in snap:
                used = snap.get("vram_used_mb")
                total = snap.get("vram_total_mb")
                if (
                    isinstance(used, (int, float))
                    and isinstance(total, (int, float))
                    and total > 0
                ):
                    snap["util_mem_pct"] = used / total * 100.0
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


def _tail_logs(
    primary_path: Path | None,
    session_dir: Path | None,
    extra_globs: tuple[str, ...],
    max_extra_logs: int,
    max_lines: int,
) -> list[str]:
    """Multi-source log tail (D2).

    Returns the union of (a) primary ``server_log_path`` lines and
    (b) the most recently-modified files matched by
    ``extra_server_log_globs`` under ``session_dir``. Each source is
    capped at ``max_lines``; the final list is also capped so the
    pattern scanner stays bounded.

    Lines from extra logs are tagged ``[<filename>]`` so
    :data:`signals.local_health._log_error_symptoms` can attribute
    pattern hits to the right variant.
    """
    if max_lines <= 0:
        return []
    primary = _tail_log(primary_path, max_lines) if primary_path else []
    extras: list[tuple[Path, list[str]]] = []
    if session_dir is not None and extra_globs:
        try:
            candidates: list[Path] = []
            for pattern in extra_globs:
                candidates.extend(session_dir.glob(pattern))
            # Dedupe (a path may match multiple globs) and order by mtime desc.
            unique: dict[Path, float] = {}
            for path in candidates:
                if not path.is_file():
                    continue
                try:
                    unique[path] = path.stat().st_mtime
                except (FileNotFoundError, PermissionError, OSError):
                    continue
            sorted_paths = sorted(
                unique.items(), key=lambda kv: kv[1], reverse=True,
            )[:max_extra_logs]
            for path, _ in sorted_paths:
                if primary_path is not None and path == primary_path:
                    # Already covered by ``primary``.
                    continue
                lines = _tail_log(path, max_lines)
                if lines:
                    extras.append((path, lines))
        except OSError as exc:
            log.debug("local_probe: extra log glob failed: %s", exc)

    # Primary first (so its patterns surface first), then extras with a per-file tag.
    out: list[str] = list(primary)
    for path, lines in extras:
        tag = f"[{path.name}]"
        out.extend(f"{tag} {line}" for line in lines)
    return out


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


_RAY_PENDING_RE = re.compile(
    r"(?m)(?:^|:)\s*(\d+)\+?\s+pending\s+(?:task|actor)",
    re.IGNORECASE,
)


def _parse_ray_pending_count(text: str) -> int:
    """Sum the pending-task counts in the ``Demands:`` section of ``ray status``.

    The regex anchors each digit to either a line start or a colon and
    requires the ``pending task[s]`` / ``pending actor[s]`` suffix that
    Ray's autoscaler emits for queued demands. This excludes hex digits
    embedded inside node IDs (line ``1 node_<64-char-hex>``), which
    never satisfy both the colon/line-start anchor *and* the
    ``task|actor`` suffix.
    """
    if not text:
        return 0
    total = 0
    for match in _RAY_PENDING_RE.finditer(text):
        try:
            total += int(match.group(1))
        except ValueError:
            continue
    return total


def _probe_ray_head(timeout_s: float) -> dict[str, Any]:
    """Best-effort ``ray status`` probe for liveness + queued demand.

    Returns:
        ``{}`` when ``ray`` is not on ``$PATH`` (silent on smoke-test pods).
        ``{"healthy": False, "reason": str, "stderr": str,
          "returncode": int|None}`` when ``ray status`` cannot be run or
        exits non-zero.
        ``{"healthy": True, "pending_tasks": int, "stdout_head": str,
          "returncode": 0}`` on success.

    ``pending_tasks`` is taken from the ``Demands:`` section only via
    :func:`_parse_ray_pending_count`. This avoids the Ray dashboard /
    state-API dependency (port 8265) which is not enabled in production
    Hyperloom pods.
    """
    if not shutil.which("ray"):
        return {}
    try:
        proc = subprocess.run(
            ["ray", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_s)),
        )
    except subprocess.TimeoutExpired:
        return {
            "healthy": False,
            "reason": f"ray status timed out after {timeout_s:.1f}s",
            "stderr": "",
            "returncode": None,
        }
    except OSError as exc:
        return {
            "healthy": False,
            "reason": f"ray status launch error: {type(exc).__name__}",
            "stderr": str(exc)[:200],
            "returncode": None,
        }
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # A crashing `ray status` CLI shim (click/import error in ray's own scripts)
        # is NOT evidence the head is down; treat as inconclusive so we don't falsely
        # emit ray_head_dead and prune the kernel_opt branch.
        cli_self_crash = "Traceback (most recent call last)" in stderr and (
            "ray/scripts/scripts.py" in stderr
            or "add_command_alias" in stderr
            or "ImportError" in stderr
            or "ModuleNotFoundError" in stderr
        )
        if cli_self_crash:
            log.warning(
                "local_probe: `ray status` CLI is broken (self-crash, not "
                "head-dead); skipping ray_head_dead. stderr=%s",
                stderr[:200],
            )
            return {}
        return {
            "healthy": False,
            "reason": f"ray status exit={proc.returncode}",
            "stderr": stderr[:400],
            "returncode": proc.returncode,
        }
    stdout = proc.stdout or ""
    head_line = ""
    for line in stdout.splitlines():
        if line.strip():
            head_line = line.strip()
            break
    return {
        "healthy": True,
        "stdout_head": head_line[:200],
        "pending_tasks": _parse_ray_pending_count(stdout),
        "returncode": 0,
    }


def _sample_fd_usage(pid: int | None) -> dict[str, Any]:
    """Read FD usage + hard limit for ``pid`` (defaults to current PID).

    Linux exposes the open-FD count through ``/proc/<pid>/fd/`` (each
    entry is one FD) and the per-process limit through
    ``/proc/<pid>/limits``. Both files are zero-overhead reads.

    Returns ``{}`` when ``/proc`` is unreadable (containers, sandboxes)
    so the signal stays silent there.
    """
    target_pid = pid if pid is not None else os.getpid()
    fd_dir = Path(f"/proc/{target_pid}/fd")
    limits_path = Path(f"/proc/{target_pid}/limits")
    try:
        used = len([_ for _ in fd_dir.iterdir()])
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.debug("local_probe: /proc/%d/fd unreadable: %s", target_pid, exc)
        return {}
    limit: int | None = None
    try:
        text = limits_path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.debug("local_probe: /proc/%d/limits unreadable: %s", target_pid, exc)
        text = ""
    for line in text.splitlines():
        # Format: "Max open files     1024     4096     files"
        if not line.startswith("Max open files"):
            continue
        parts = line.split()
        # 3 numeric columns: soft, hard, unit
        if len(parts) >= 5:
            try:
                limit = int(parts[3])  # hard limit
            except ValueError:
                continue
            break
    if limit is None or limit <= 0:
        return {"pid": target_pid, "used": used, "limit": None, "used_pct": None}
    used_pct = round((used / limit) * 100.0, 2)
    return {
        "pid": target_pid,
        "used": used,
        "limit": limit,
        "used_pct": used_pct,
    }


def _sample_aiter_jit(jit_dir: Path | None) -> dict[str, Any]:
    """Count compiled ``.so`` artefacts under aiter's JIT cache.

    ``baseline.py:_resolve_aiter_jit_dir`` is the source of truth for
    where the cache lives; we mirror its heuristics here so the
    detector stays accurate without forcing a cross-package import.

    Returns ``{}`` when we cannot resolve a directory. Otherwise:

    * ``so_count``         — total ``*.so`` files under ``jit_dir``
                             (excludes ``build/`` staging).
    * ``build_count``      — files under ``jit_dir/build/`` (in-flight
                             compilation; usually 0 on a warm host).
    * ``jit_dir``          — absolute path probed.
    """
    resolved = _resolve_aiter_jit_dir(jit_dir)
    if resolved is None:
        return {}
    try:
        all_so = list(resolved.rglob("*.so"))
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.debug("local_probe: aiter jit scan failed: %s", exc)
        return {}
    build_root = resolved / "build"
    build_so = [p for p in all_so if str(p).startswith(str(build_root))]
    main_so = [p for p in all_so if p not in build_so]
    return {
        "jit_dir": str(resolved),
        "so_count": len(main_so),
        "build_count": len(build_so),
    }


def _resolve_aiter_jit_dir(explicit: Path | None) -> Path | None:
    """Find the aiter JIT cache root.

    Order:
    1. ``explicit`` (caller-supplied) → use if exists.
    2. ``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` env (matches upstream).
    3. ``importlib.util.find_spec("aiter")`` → ``<pkg>/jit``.
    """
    if explicit is not None and Path(explicit).is_dir():
        return Path(explicit)
    env_dir = os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
        if candidate.is_dir():
            return candidate
    try:
        import importlib.util
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    pkg_root = Path(spec.origin).parent / "jit"
    if pkg_root.is_dir():
        return pkg_root
    return None


# ---------------------------------------------------------------------------
# G — decision-audit probe (reads persisted decision artefacts)
# ---------------------------------------------------------------------------

def _sample_decision_audit(
    session_dir: Path | None,
    max_integrate: int,
    max_oob_attempts: int,
) -> dict[str, Any]:
    """Collect persisted decision artefacts for the G-section signals.

    Three independent slices are gathered defensively — any missing
    file becomes an empty value and the rest still surface, so a host
    that runs without the external ``report_back`` pipeline (no
    ``ci_metrics.json``) still gets G1-G3 audit on the integrate
    artefacts.

    Returned shape::

        {
            "recent_integrate": [{kernel_id, decision, gain_pct,
                                  patch_path, patch_size_bytes,
                                  base_tput, new_tput, dispatched_count,
                                  result_path, mtime}, ...],
            "ci_metrics": {raw json} | {},
            "ci_metrics_path": str | "",
            "oob_attempts": [{kernel_id, backend, report_text,
                              microbench_speedup, ts}, ...],
        }
    """
    if session_dir is None:
        return {}
    out: dict[str, Any] = {
        "recent_integrate": _scan_integrate_results(session_dir, max_integrate),
        "oob_attempts": _scan_oob_attempts(session_dir, max_oob_attempts),
    }
    ci_path, ci_data = _load_ci_metrics(session_dir)
    out["ci_metrics_path"] = str(ci_path) if ci_path else ""
    out["ci_metrics"] = ci_data
    return out


def _scan_integrate_results(
    session_dir: Path,
    max_files: int,
) -> list[dict[str, Any]]:
    """Read the most recent ``runs/integrate/*/result.json`` files."""
    integrate_root = session_dir / "runs" / "integrate"
    if not integrate_root.is_dir():
        return []
    try:
        candidates = sorted(
            integrate_root.glob("*/result.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.debug("local_probe: integrate scan failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for path in candidates[:max_files]:
        try:
            raw = path.read_text(encoding="utf-8")
            data = _json_loads_or_none(raw)
        except (OSError, ValueError) as exc:
            log.debug("local_probe: integrate read %s failed: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        entry = _normalise_integrate_entry(data, result_path=path)
        if entry is not None:
            out.append(entry)
    return out


def _normalise_integrate_entry(
    data: dict[str, Any], *, result_path: Path,
) -> dict[str, Any] | None:
    decision = data.get("decision")
    if not isinstance(decision, str):
        return None
    patch_path = data.get("patch_path") or ""
    patch_size_bytes: int | None = None
    if patch_path:
        try:
            patch_size_bytes = Path(patch_path).stat().st_size
        except (FileNotFoundError, PermissionError, OSError):
            patch_size_bytes = None
    base_tput = _coerce_optional_float(data.get("base_tput"))
    new_tput = _coerce_optional_float(data.get("new_tput"))
    gain_pct = _coerce_optional_float(data.get("gain_pct"))
    dispatched_count = data.get("dispatched_count")
    if not isinstance(dispatched_count, int):
        dispatched_count = None
    try:
        mtime = result_path.stat().st_mtime
    except (FileNotFoundError, PermissionError, OSError):
        mtime = 0.0
    return {
        "kernel_id": str(data.get("kernel_id") or ""),
        "task_id": str(data.get("task_id") or ""),
        "decision": decision,
        "gain_pct": gain_pct,
        "base_tput": base_tput,
        "new_tput": new_tput,
        "patch_path": str(patch_path) if patch_path else "",
        "patch_size_bytes": patch_size_bytes,
        "dispatched_count": dispatched_count,
        "result_path": str(result_path),
        "mtime": mtime,
    }


def _scan_oob_attempts(
    session_dir: Path,
    max_entries: int,
) -> list[dict[str, Any]]:
    root = session_dir / "kernel-agent" / "runs"
    if not root.is_dir():
        return []
    try:
        files = sorted(
            root.glob("*/optimization_attempts.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.debug("local_probe: oob_attempts scan failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for path in files[:3]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            log.debug("local_probe: oob_attempts read %s failed: %s", path, exc)
            continue
        for line in text.splitlines()[-max_entries:]:
            line = line.strip()
            if not line:
                continue
            row = _json_loads_or_none(line)
            if not isinstance(row, dict):
                continue
            out.append({
                "kernel_id": str(row.get("kernel_id") or ""),
                "backend": str(row.get("backend") or ""),
                "report_text": str(row.get("report_text") or "")[:500],
                "microbench_speedup": _coerce_optional_float(
                    row.get("microbench_speedup")
                ),
                "ts": row.get("ts"),
                "source_file": str(path),
            })
        if out:
            break
    return out[-max_entries:]


_CI_METRICS_CANDIDATE_RELPATHS: tuple[str, ...] = (
    "results/ci_metrics_final.json",
    "results/ci_metrics.json",
    "ci_metrics_final.json",
    "ci_metrics.json",
)


def _load_ci_metrics(
    session_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    for relpath in _CI_METRICS_CANDIDATE_RELPATHS:
        candidate = session_dir / relpath
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            log.debug("local_probe: ci_metrics read %s failed: %s", candidate, exc)
            continue
        data = _json_loads_or_none(text)
        if isinstance(data, dict):
            return candidate, data
    return None, {}


def _json_loads_or_none(text: str) -> Any:
    if not text:
        return None
    import json

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# C — preflight probe (manifest + kernel_breakdown)
# ---------------------------------------------------------------------------

def _load_manifest_extras(session_dir: Path | None) -> dict[str, Any]:
    """Read ``manifest.json`` for the C-section preflight signals.

    Returns the raw dict so the signal layer can pick fields without
    leaking knowledge of the manifest schema into the probe. Empty when
    the file is absent (resume from a half-init session) or unreadable.
    """
    if session_dir is None:
        return {}
    candidate = session_dir / "manifest.json"
    if not candidate.is_file():
        return {}
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        log.debug("local_probe: manifest read %s failed: %s", candidate, exc)
        return {}
    data = _json_loads_or_none(text)
    if not isinstance(data, dict):
        return {}
    return data


# Private tier mapping so the signal layer computes the Amdahl ceiling
# without importing the inference_optimizer package.
_AMDAHL_TIER_FAMILIES: dict[str, str] = {
    "T1_TRITON":      "triton",
    "T2_AITER_CK":    "vendor",
    "T3_FRAMEWORK":   "framework",
    "T4_COMM":        "comm",
    "T5_COMPILED":    "compiled",
}


def _load_kernel_breakdown(session_dir: Path | None) -> dict[str, Any]:
    """Read ``profiles/kernel_breakdown.json`` and aggregate by tier.

    The full per-kernel list is huge; the C2 detector only needs the
    aggregate so we pre-collapse to ``{tier_pcts: {triton, vendor, ...},
    total_kernels, total_gpu_pct, mtime}``. Tiers fall back to the
    canonical name when no mapping exists, keeping new tiers from
    surfacing as silent drops.
    """
    if session_dir is None:
        return {}
    candidate = session_dir / "profiles" / "kernel_breakdown.json"
    if not candidate.is_file():
        return {}
    try:
        text = candidate.read_text(encoding="utf-8")
        mtime = candidate.stat().st_mtime
    except (OSError, ValueError) as exc:
        log.debug(
            "local_probe: kernel_breakdown read %s failed: %s", candidate, exc
        )
        return {}
    rows = _json_loads_or_none(text)
    if not isinstance(rows, list):
        return {}
    tier_pcts: dict[str, float] = {}
    total_gpu_pct = 0.0
    total_kernels = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        total_kernels += 1
        gpu_pct = row.get("gpu_pct")
        if not isinstance(gpu_pct, (int, float)) or isinstance(gpu_pct, bool):
            continue
        total_gpu_pct += float(gpu_pct)
        tier_raw = str(row.get("tier") or "").strip()
        bucket = _AMDAHL_TIER_FAMILIES.get(tier_raw, tier_raw.lower() or "unknown")
        tier_pcts[bucket] = tier_pcts.get(bucket, 0.0) + float(gpu_pct)
    return {
        "tier_pcts": {k: round(v, 3) for k, v in tier_pcts.items()},
        "total_kernels": total_kernels,
        "total_gpu_pct": round(total_gpu_pct, 3),
        "kernel_breakdown_path": str(candidate),
        "mtime": mtime,
    }


# ---------------------------------------------------------------------------
# E — critic-health probe (judge_bundle.json + workdir count)
# ---------------------------------------------------------------------------

def _sample_critic_workdir(
    session_dir: Path | None,
    max_judges: int,
) -> dict[str, Any]:
    """Scan ``critic-workdir/<turn>/judge_bundle.json`` for E1+E4 signals.

    Returns ``{recent_judges: [...], workdir_count: int}``. Empty when
    the critic-workdir tree doesn't exist (smoke run / critic disabled).
    """
    if session_dir is None:
        return {}
    root = session_dir / "critic-workdir"
    if not root.is_dir():
        return {}
    try:
        turn_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError as exc:
        log.debug("local_probe: critic-workdir scan failed: %s", exc)
        return {}
    workdir_count = len(turn_dirs)
    recent_judges: list[dict[str, Any]] = []
    for turn_dir in turn_dirs[:max_judges]:
        judge_path = turn_dir / "judge_bundle.json"
        if not judge_path.is_file():
            continue
        try:
            text = judge_path.read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            log.debug(
                "local_probe: judge_bundle read %s failed: %s", judge_path, exc
            )
            continue
        data = _json_loads_or_none(text)
        if not isinstance(data, dict):
            continue
        try:
            mtime = judge_path.stat().st_mtime
        except (FileNotFoundError, PermissionError, OSError):
            mtime = 0.0
        # Count from ``proposals``; some bundles emit ``kb_priors_by_proposal`` instead.
        proposals = data.get("proposals")
        if isinstance(proposals, list):
            proposal_count = len(proposals)
        elif isinstance(data.get("kb_priors_by_proposal"), dict):
            proposal_count = len(data["kb_priors_by_proposal"])
        else:
            proposal_count = 0
        recent_judges.append({
            "turn_dir": turn_dir.name,
            "kb_read_skipped_reason": data.get("kb_read_skipped_reason"),
            "required_context": list(data.get("required_context") or []),
            "proposal_count": proposal_count,
            "mtime": mtime,
        })
    return {
        "recent_judges": recent_judges,
        "workdir_count": workdir_count,
        "workdir_root": str(root),
    }


# ---------------------------------------------------------------------------
# I — state-integrity probe (state.json / WAL / leases / agent JSONLs / PID)
# ---------------------------------------------------------------------------

def _sample_state_integrity(
    session_dir: Path | None,
    optimizer_runs_dirname: str,
) -> dict[str, Any]:
    """Aggregate the I1-I5 state-integrity slots into one payload.

    Returns ``{}`` only when ``session_dir`` is missing. Individual
    sub-slots that fail (missing file / unreadable DB / no PID file)
    surface their own error markers so the signal layer can branch
    on absence without mistaking it for healthy state.
    """
    if session_dir is None:
        return {}
    return {
        "state_json": _probe_state_json(session_dir),
        "wal": _probe_wal_size(session_dir),
        "leases": _probe_leases(session_dir),
        "agents": _probe_agent_files(session_dir),
        "coordinator": _probe_coordinator_pid(
            session_dir, optimizer_runs_dirname,
        ),
    }


def _probe_state_json(session_dir: Path) -> dict[str, Any]:
    """Return ``state.json`` health: validity / size / mtime / error."""
    path = session_dir / "state.json"
    if not path.is_file():
        return {"valid": False, "error": "missing", "path": str(path)}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return {
            "valid": False, "error": f"read_failed: {exc}", "path": str(path),
        }
    decoded = _json_loads_or_none(text)
    if decoded is None or not isinstance(decoded, dict):
        # Partial write / corruption.
        return {
            "valid": False,
            "error": "json_parse_failed",
            "path": str(path),
            "size_bytes": len(text),
        }
    try:
        st = path.stat()
        size_bytes = st.st_size
        mtime = st.st_mtime
    except OSError:
        size_bytes = len(text)
        mtime = 0.0
    return {
        "valid": True,
        "path": str(path),
        "size_bytes": size_bytes,
        "mtime": mtime,
        "stop_reason": decoded.get("stop_reason") or "",
    }


def _probe_wal_size(session_dir: Path) -> dict[str, Any]:
    """``storage/coordinator.db-wal`` size — WAL bloat signal source."""
    db_path = session_dir / "storage" / "coordinator.db"
    wal_path = session_dir / "storage" / "coordinator.db-wal"
    out: dict[str, Any] = {
        "db_path": str(db_path),
        "wal_path": str(wal_path),
        "wal_bytes": 0,
        "db_bytes": 0,
    }
    try:
        if wal_path.is_file():
            out["wal_bytes"] = int(wal_path.stat().st_size)
    except OSError:
        pass
    try:
        if db_path.is_file():
            out["db_bytes"] = int(db_path.stat().st_size)
    except OSError:
        pass
    return out


def _probe_leases(session_dir: Path) -> list[dict[str, Any]]:
    """Cross-reference active leases against ``os.kill(pid, 0)``.

    Reads the ``leases`` table from ``storage/coordinator.db``. Each
    row's ``holder_pid`` is liveness-checked; ``alive=False`` means
    the lease is stale (holder process gone but lease still held).
    """
    db_path = session_dir / "storage" / "coordinator.db"
    if not db_path.is_file():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=2.0,
        )
    except sqlite3.Error as exc:
        log.debug("local_probe: cannot open leases db: %s", exc)
        return []
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = list(conn.execute(
                "SELECT task_id, holder_pid, lane, acquired_at FROM leases"
            ).fetchall())
        except sqlite3.Error as exc:
            log.debug("local_probe: leases select failed: %s", exc)
            return []
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        keys = row.keys()
        holder = row["holder_pid"] if "holder_pid" in keys else None
        try:
            holder_int = int(holder) if holder is not None else None
        except (TypeError, ValueError):
            holder_int = None
        alive = _is_pid_alive(holder_int) if holder_int is not None else False
        out.append({
            "task_id": row["task_id"] if "task_id" in keys else None,
            "holder_pid": holder_int,
            "lane": row["lane"] if "lane" in keys else None,
            "acquired_at": (
                row["acquired_at"] if "acquired_at" in keys else None
            ),
            "alive": alive,
        })
    return out


def _is_pid_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` — POSIX existence probe. Returns False on any
    error (PID missing, permission denied, non-Linux)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _probe_agent_files(session_dir: Path) -> dict[str, Any]:
    """Per-agent inbox/outbox sizes for I4 bloat detection."""
    agents_root = session_dir / "agents"
    if not agents_root.is_dir():
        return {}
    out: dict[str, Any] = {}
    try:
        for role_dir in agents_root.iterdir():
            if not role_dir.is_dir():
                continue
            role = role_dir.name
            inbox = role_dir / "inbox.jsonl"
            outbox = role_dir / "outbox.jsonl"
            inbox_bytes = 0
            outbox_bytes = 0
            try:
                if inbox.is_file():
                    inbox_bytes = int(inbox.stat().st_size)
            except OSError:
                pass
            try:
                if outbox.is_file():
                    outbox_bytes = int(outbox.stat().st_size)
            except OSError:
                pass
            if inbox_bytes or outbox_bytes:
                out[role] = {
                    "inbox_bytes": inbox_bytes,
                    "outbox_bytes": outbox_bytes,
                    "inbox_path": str(inbox),
                    "outbox_path": str(outbox),
                }
    except OSError as exc:
        log.debug("local_probe: agents scan failed: %s", exc)
        return out
    return out


def _probe_coordinator_pid(
    session_dir: Path, optimizer_runs_dirname: str,
) -> dict[str, Any]:
    """Cross-reference ``optimizer_runs/run_*.pid`` against ``os.kill(pid, 0)``.

    The PID file is dropped by the SKILL.md launcher template; absence
    is not itself an error (operator may have launched without ``setsid``).
    Mismatch is the I5 ``coordinator_zombie`` signal.
    """
    runs_dir = session_dir / optimizer_runs_dirname
    out: dict[str, Any] = {
        "recorded_pid": None, "alive": None, "pid_file": "",
    }
    if not runs_dir.is_dir():
        return out
    try:
        pid_files = sorted(
            runs_dir.glob("run_*.pid"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError as exc:
        log.debug("local_probe: pid file scan failed: %s", exc)
        return out
    if not pid_files:
        return out
    newest = pid_files[0]
    try:
        text = newest.read_text(encoding="utf-8").strip()
    except OSError as exc:
        log.debug("local_probe: pid file read %s failed: %s", newest, exc)
        return out
    try:
        pid = int(text.splitlines()[0])
    except (ValueError, IndexError):
        return out
    out["recorded_pid"] = pid
    out["pid_file"] = str(newest)
    out["alive"] = _is_pid_alive(pid)
    return out


# ---------------------------------------------------------------------------
# J — external-deps probe (gateway / mounts / TraceLens CLI)
# ---------------------------------------------------------------------------

async def _probe_external_deps(
    gateway_probe_url_override: str,
    mount_timeout_s: float,
    http_timeout_s: float,
) -> dict[str, Any]:
    """Async wrapper that runs J1+J2+J3 probes once per tick."""
    gateway_url = gateway_probe_url_override
    if not gateway_url:
        base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base:
            gateway_url = base.rstrip("/") + "/models"
    gateway = (
        await _probe_gateway_health(gateway_url, http_timeout_s)
        if gateway_url
        else {}
    )
    mounts = await asyncio.to_thread(
        _probe_external_mounts, mount_timeout_s,
    )
    tracelens_cli = await asyncio.to_thread(_probe_tracelens_cli)
    if not gateway and not mounts and not tracelens_cli:
        return {}
    return {
        "gateway": gateway,
        "mounts": mounts,
        "tracelens_cli": tracelens_cli,
    }


async def _probe_gateway_health(
    url: str, timeout_s: float,
) -> dict[str, Any]:
    """GET ``$OPENAI_BASE_URL/models`` with Bearer; classify the response.

    A 401 here (with the same auth token that critic + kernel-agent
    use) means the upstream gateway has revoked / lost the key — J1
    surfaces this distinct from generic local-server unreachable.
    """
    out: dict[str, Any] = {
        "url": url, "reachable": False, "status": "error",
    }
    headers: dict[str, str] = {}
    api_key = os.environ.get("SAFE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(max(0.5, float(timeout_s)))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        out["error"] = "timeout"
        return out
    except httpx.ConnectError as exc:
        out["error"] = f"connect: {exc.__class__.__name__}"
        return out
    except httpx.RequestError as exc:
        out["error"] = f"request: {exc.__class__.__name__}: {exc}"
        return out
    out["status_code"] = resp.status_code
    out["reachable"] = resp.status_code < 500
    if resp.status_code == 401:
        out["status"] = "unauthorized"
    elif resp.status_code < 400:
        out["status"] = "ok"
    elif resp.status_code < 500:
        out["status"] = "http_error"
    else:
        out["status"] = "server_error"
    return out


# J2 mount paths, read from env at probe time. All default to "" (no fallback):
# we only probe what the operator points at. TRACELENS_ROOT is now session-local
# (install.sh clones into $HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens), so it
# is only flagged degraded when explicitly overridden to a shared mount.
_EXTERNAL_MOUNT_ENVS: tuple[tuple[str, str], ...] = (
    ("TRACELENS_ROOT", ""),
    # Optional internal extension; unset means open-source-only, not a degraded mount.
    ("TRACELENS_INTERNAL_ROOT", ""),
    ("INFERENCEX_PATH", ""),
    ("OOB_SRC", ""),
)


def _probe_external_mounts(
    timeout_s: float,
) -> list[dict[str, Any]]:
    """``os.stat`` each external mount, time it, flag slow / failing."""
    out: list[dict[str, Any]] = []
    for env_name, default_path in _EXTERNAL_MOUNT_ENVS:
        raw = os.environ.get(env_name, default_path) or ""
        path = raw.strip()
        if not path:
            continue
        start = time.monotonic()
        ok = False
        error: str | None = None
        try:
            os.stat(path)
            ok = True
        except FileNotFoundError:
            error = "not_found"
        except PermissionError as exc:
            error = f"permission: {exc}"
        except OSError as exc:
            error = f"oserror: {exc.__class__.__name__}: {exc}"
        latency_ms = (time.monotonic() - start) * 1000.0
        out.append({
            "env_name": env_name,
            "path": path,
            "ok": ok,
            "error": error,
            "latency_ms": round(latency_ms, 2),
            "timeout_ms": timeout_s * 1000.0,
        })
    return out


# Both TraceLens CLI names; the ``_inference`` variant is canonical for
# vLLM/SGLang traces per SKILL.md, the legacy name remains valid for older builds.
_TRACELENS_CLI_NAMES: tuple[str, ...] = (
    "TraceLens_generate_perf_report_pytorch_inference",
    "TraceLens_generate_perf_report_pytorch",
)


def _probe_tracelens_cli() -> dict[str, Any]:
    """Detect both TraceLens CLI names — boot-time presence check."""
    found: dict[str, bool] = {}
    for name in _TRACELENS_CLI_NAMES:
        found[name] = shutil.which(name) is not None
    return {
        "cli_names": list(_TRACELENS_CLI_NAMES),
        "found": found,
        "any_present": any(found.values()),
    }


__all__ = ["LocalProbeConfig", "LocalProbeSource"]
