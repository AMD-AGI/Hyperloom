# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Symptoms derived from LocalProbe-only data.

Fire only when :class:`LocalProbeSource` is active; silent when the probe is
disabled, since the SourceData fields are then empty. Covers
``local_server_unreachable`` (HIGH if all targets fail), ``log_error_pattern`` (OOM/NCCL → HIGH),
``gpu_thermal_high``, plus disk/shm/ray-head/fd pressure rules.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


# Load generators that only run while an inference server is expected to answer,
# so their presence turns "no server process" from an idle stretch into an
# outage. Deliberately narrower than the harness patterns the process probe
# matches: the outer Magpie/InferenceX harness is also up while it launches a
# server and while it tears one down, when a refused port is the correct reading.
_BENCHMARK_CLIENT_PATTERNS: tuple[str, ...] = ("benchmark_serving",)


@dataclass
class LocalHealthConfig:
    """Thresholds for the LocalProbe-derived health rules.

    Attributes:
        gpu_temp_warn_c (float): GPU temperature (Celsius) at/above which a
            MEDIUM thermal symptom fires.
        gpu_temp_crit_c (float): GPU temperature (Celsius) at/above which a HIGH
            thermal symptom fires.
        disk_used_warn_pct (float): Non-SHM mountpoint used-percent for a MEDIUM
            disk-pressure symptom.
        disk_used_crit_pct (float): Non-SHM mountpoint used-percent for a HIGH
            disk-pressure symptom.
        shm_mountpoints (tuple[str, ...]): Mountpoints treated as shared memory
            (stricter thresholds, handled separately from disk).
        shm_used_warn_pct (float): SHM used-percent for a MEDIUM symptom.
        shm_used_crit_pct (float): SHM used-percent for a HIGH symptom.
        fd_warn_used_pct (float): File-descriptor used-percent for a MEDIUM
            symptom.
        fd_crit_used_pct (float): File-descriptor used-percent for a HIGH
            symptom.
        benchmark_client_patterns (tuple[str, ...]): Commands whose presence
            means a server is supposed to be answering right now.
        session_dir (Path | None): This session's directory, used to tell its
            own processes from a co-tenant's. ``None`` leaves the benchmark
            client check host-wide, which is only safe on a dedicated node.
    """

    gpu_temp_warn_c: float = 90.0
    gpu_temp_crit_c: float = 100.0
    # disk_pressure thresholds (% used); crit also emits a prune_branch(profile) hint.
    disk_used_warn_pct: float = 85.0
    disk_used_crit_pct: float = 95.0
    # SHM gets stricter thresholds — small (16-64 GiB) and SGLang/vLLM crash hard when it fills.
    shm_mountpoints: tuple[str, ...] = ("/dev/shm",)  # nosec B108 - mountpoint probe, not temp file creation.
    shm_used_warn_pct: float = 75.0
    shm_used_crit_pct: float = 90.0
    fd_warn_used_pct: float = 80.0
    fd_crit_used_pct: float = 95.0
    benchmark_client_patterns: tuple[str, ...] = _BENCHMARK_CLIENT_PATTERNS
    session_dir: Path | None = None


# HIGH-severity log patterns; anything else from ``local_log_errors`` falls back to MEDIUM.
_HIGH_SEVERITY_PATTERNS: frozenset[str] = frozenset(
    {
        # OOM / fatal-signal patterns.
        "CUDA out of memory",
        "hipErrorOutOfMemory",
        "HIP out of memory",
        "NCCL error",
        "OOMKilled",
        "core dumped",
        "Segmentation fault",
        # vLLM v1 EngineCore crashes.
        r"Engine core .* died",
        r"RuntimeError: Engine core initialization failed",
        # model architecture mismatch.
        r"MLA.*not supported",
        r"MTP draft .* unavailable",
        # aiter / hipcc compilation hard fail.
        r"aiter .* compilation failed",
        r"hipcc .* signal",
        # accuracy gate / model load — terminal, do not retry.
        r"accuracy .* gate failed",
        r"MMLU .* below threshold",
        r"Failed to load checkpoint",
        # KFD resource exhaustion.
        r"cudaErrorOutOfDevice",
        r"HSA_STATUS_ERROR_OUT_OF_RESOURCES",
        # matrix library errors.
        r"ROCblas.*Status\s*\d+",
        r"hipBLAS.*Error",
        # critic-agent runtime stuck.
        r"runtime\.cli .* timed out after \d+s",
    }
)


def evaluate_local_health_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: LocalHealthConfig | None = None,
) -> list[Symptom]:
    """Run all LocalProbe-only health rules and aggregate their symptoms.

    Args:
        ctx (ReactorContext): Reactor context for the current tick.
        data (SourceData): Collected LocalProbe source data.
        config (LocalHealthConfig | None): Thresholds; defaults to
            :class:`LocalHealthConfig` when ``None``.

    Returns:
        list[Symptom]: All local-health symptoms found this tick, possibly
            empty.
    """
    cfg = config or LocalHealthConfig()
    out: list[Symptom] = []
    out.extend(_server_unreachable(data, cfg))
    out.extend(_log_error_symptoms(data))
    out.extend(_gpu_thermal_symptoms(data, cfg))
    out.extend(_disk_pressure_symptoms(data, cfg))
    out.extend(_shm_pressure_symptoms(data, cfg))
    out.extend(_ray_head_dead_symptoms(data))
    out.extend(_fd_pressure_symptoms(data, cfg))
    return out


def _server_unreachable(data: SourceData, cfg: LocalHealthConfig) -> list[Symptom]:
    """Emit ``local_server_unreachable`` for each failed local HTTP probe.

    The probe detects "process is alive but the server is wedged", so a refusal
    with no server process behind the port is the expected reading, not a
    fault: a session spends long stretches — preparation, analysis, the gap
    between two variants — with no server up by design, and alerting there
    tells an operator to restart something that was never meant to be running.

    That reasoning needs to know there is no server, which is not the same as
    failing to find one. When the process probe could not answer at all, the
    symptom is emitted with the uncertainty recorded in its evidence, so a
    broken ``ps`` cannot mute an unrelated finding.

    "No server process" also does not always mean no server was wanted. A
    server that died mid-benchmark leaves that exact snapshot while its own
    load generator keeps sending requests into a closed port, so a benchmark
    client of *this session* is treated as proof that something was supposed to
    be answering and the alert stands. A co-tenant's client on a shared node
    proves nothing about this session's port.

    Severity is HIGH when every probed target is unreachable, otherwise MEDIUM.

    Args:
        data (SourceData): Collected source data including
            ``local_server_health``, ``local_processes`` and
            ``local_processes_known``.
        cfg (LocalHealthConfig): Thresholds; provides the benchmark-client
            patterns.

    Returns:
        list[Symptom]: One symptom per unreachable probe target, possibly empty.
    """
    if not data.local_server_health:
        return []
    server_seen = any(proc.get("is_server") for proc in data.local_processes)
    client_seen = _benchmark_client_seen(data, cfg)
    if data.local_processes_known and not server_seen and not client_seen:
        return []
    bad = [entry for entry in data.local_server_health if not entry.get("reachable")]
    if not bad:
        return []
    severity = SymptomSeverity.HIGH if len(bad) == len(data.local_server_health) else SymptomSeverity.MEDIUM
    out: list[Symptom] = []
    for entry in bad:
        url = str(entry.get("url") or "")
        status = str(entry.get("status") or "")
        out.append(
            Symptom(
                name="local_server_unreachable",
                severity=severity,
                summary=f"local server probe {url} status={status}",
                evidence={
                    "url": url,
                    "status": status,
                    "status_code": entry.get("status_code"),
                    "error": entry.get("error"),
                    "server_process_seen": server_seen if data.local_processes_known else None,
                    "benchmark_client_seen": client_seen if data.local_processes_known else None,
                },
                subject={"url": url},
                source="local",
                suggestion=(
                    "delegate(recover, force_gpu_cleanup=True) to restart the inference server"
                    if severity is SymptomSeverity.HIGH
                    else "monitor; alert orchestration if it persists"
                ),
            )
        )
    return out


def _benchmark_client_seen(data: SourceData, cfg: LocalHealthConfig) -> bool:
    """Report whether a load generator that needs *this session's* server runs.

    The process probe reads a whole-host ``ps``, so on a shared node another
    session's load generator is in the snapshot too — and it vouches for a port
    it has never sent a request to, turning this session's idle stretch back
    into an outage. Only a client that can be tied to this session counts.

    Args:
        data (SourceData): Collected source data including ``local_processes``.
        cfg (LocalHealthConfig): Thresholds; provides the benchmark-client
            patterns and the session anchor.

    Returns:
        bool: ``True`` when a probed process matches a configured
            benchmark-client pattern and belongs to this session.
    """
    anchor = os.path.realpath(cfg.session_dir) if cfg.session_dir else ""
    for proc in data.local_processes:
        if not isinstance(proc, dict):
            continue
        cmd = str(proc.get("cmd") or "")
        if not any(pattern in cmd for pattern in cfg.benchmark_client_patterns):
            continue
        if _in_session(proc, anchor):
            return True
    return False


def _in_session(proc: dict[str, Any], anchor: str) -> bool:
    """Report whether a probed process can be tied to the session at ``anchor``.

    The harness is launched with its working directory inside the session and
    children inherit it, so the cwd is the anchor; a client that names a path
    under the session on its command line (``--result-dir``) counts too, for the
    launch paths that chdir elsewhere. Both readings go through
    :func:`_under_session` so neither can drift into accepting a path that only
    starts with the session's.

    Args:
        proc (dict[str, Any]): One ``local_processes`` entry.
        anchor (str): Resolved session directory, or ``""`` when the session is
            unknown — nothing to compare against, so every match counts and the
            check stays host-wide.

    Returns:
        bool: ``True`` when the process belongs to this session.
    """
    if not anchor:
        return True
    if _under_session(str(proc.get("cwd") or ""), anchor):
        return True
    return any(_under_session(path, anchor) for path in _command_line_paths(str(proc.get("cmd") or "")))


def _under_session(path: str, anchor: str) -> bool:
    """Report whether ``path`` is the session directory or something inside it.

    Compared a path component at a time, never as a string prefix: a co-tenant's
    ``<session>-retry`` — a retry, a backup, or any sibling an operator names
    after ours — starts with the session path without being in the session, and
    a prefix test would let it vouch for its own port.

    Args:
        path (str): Candidate path; ``""`` belongs to nobody.
        anchor (str): Resolved session directory.

    Returns:
        bool: ``True`` when ``path`` lies at or under ``anchor``.
    """
    return bool(path) and Path(path).is_relative_to(anchor)


def _command_line_paths(cmd: str) -> Iterator[str]:
    """Yield the path-shaped pieces of a command line.

    Each whitespace-separated token, plus what follows the first ``=`` in it, so
    ``--result-dir /run/x`` and ``--result-dir=/run/x`` read the same. Tokens are
    yielded whole rather than searched for a substring, which is what lets the
    caller apply a directory boundary to them.

    Args:
        cmd (str): The process command line.

    Yields:
        str: One candidate path per token, and its ``key=value`` value.
    """
    for token in cmd.split():
        yield token
        _, sep, value = token.partition("=")
        if sep:
            yield value


def _log_error_symptoms(data: SourceData) -> list[Symptom]:
    """Emit ``log_error_pattern`` symptoms grouped by matched log pattern.

    Patterns in :data:`_HIGH_SEVERITY_PATTERNS` fire HIGH; all others MEDIUM.

    Args:
        data (SourceData): Collected source data including
            ``local_log_errors``.

    Returns:
        list[Symptom]: One symptom per matched pattern, possibly empty.
    """
    if not data.local_log_errors:
        return []
    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for entry in data.local_log_errors:
        pattern = str(entry.get("pattern") or "")
        if not pattern:
            continue
        by_pattern.setdefault(pattern, []).append(entry)

    out: list[Symptom] = []
    for pattern, hits in by_pattern.items():
        severity = SymptomSeverity.HIGH if pattern in _HIGH_SEVERITY_PATTERNS else SymptomSeverity.MEDIUM
        out.append(
            Symptom(
                name="log_error_pattern",
                severity=severity,
                summary=f"log error pattern {pattern!r} matched {len(hits)} times",
                evidence={
                    "pattern": pattern,
                    "count": len(hits),
                    "samples": [h.get("line") for h in hits[:3]],
                },
                subject={"pattern": pattern},
                source="local",
                suggestion=(
                    "delegate(recover, force_gpu_cleanup=True) or escalate strategy"
                    if severity is SymptomSeverity.HIGH
                    else "review log evidence with RCA before further action"
                ),
            )
        )
    return out


def _gpu_thermal_symptoms(
    data: SourceData,
    cfg: LocalHealthConfig,
) -> list[Symptom]:
    """Emit ``gpu_thermal_high`` for GPUs over the warn/crit temperature.

    Args:
        data (SourceData): Collected source data including ``local_gpu``.
        cfg (LocalHealthConfig): Thresholds (provides warn/crit temperatures).

    Returns:
        list[Symptom]: One symptom per over-temperature GPU, possibly empty.
    """
    gpus = data.local_gpu.get("gpus") if isinstance(data.local_gpu, dict) else None
    if not isinstance(gpus, list):
        return []
    out: list[Symptom] = []
    for snap in gpus:
        if not isinstance(snap, dict):
            continue
        temp = snap.get("temperature_c")
        if not isinstance(temp, (int, float)):
            continue
        if temp >= cfg.gpu_temp_crit_c:
            severity = SymptomSeverity.HIGH
        elif temp >= cfg.gpu_temp_warn_c:
            severity = SymptomSeverity.MEDIUM
        else:
            continue
        gpu_id = snap.get("gpu_id")
        out.append(
            Symptom(
                name="gpu_thermal_high",
                severity=severity,
                summary=f"GPU {gpu_id} temperature_c={temp}",
                evidence={
                    "gpu_id": gpu_id,
                    "temperature_c": temp,
                    "warn_threshold": cfg.gpu_temp_warn_c,
                    "crit_threshold": cfg.gpu_temp_crit_c,
                },
                subject={"gpu_id": str(gpu_id)},
                source="local",
                suggestion="escalate strategy change to back off thermal pressure",
            )
        )
    return out


def _disk_pressure_symptoms(
    data: SourceData,
    cfg: LocalHealthConfig,
) -> list[Symptom]:
    """Emit ``disk_pressure`` for non-SHM mountpoints under capacity stress.

    SHM is handled separately with stricter thresholds, so it is skipped
    here to avoid double-firing.

    Args:
        data: Collected source data (per-mountpoint disk stats).
        cfg: Local-health configuration thresholds.

    Returns:
        Symptoms for stressed mountpoints, possibly empty.
    """
    if not isinstance(data.local_disk, dict) or not data.local_disk:
        return []
    out: list[Symptom] = []
    shm_set = frozenset(cfg.shm_mountpoints)
    for mountpoint, stats in data.local_disk.items():
        if mountpoint in shm_set:
            continue
        if not isinstance(stats, dict):
            continue
        used_pct = stats.get("used_pct")
        if not isinstance(used_pct, (int, float)):
            continue
        if used_pct >= cfg.disk_used_crit_pct:
            severity = SymptomSeverity.HIGH
        elif used_pct >= cfg.disk_used_warn_pct:
            severity = SymptomSeverity.MEDIUM
        else:
            continue
        free_gb = stats.get("free_gb")
        used_gb = stats.get("used_gb")
        total_gb = stats.get("total_gb")
        out.append(
            Symptom(
                name="disk_pressure",
                severity=severity,
                summary=(f"disk {mountpoint!r} at {used_pct:.0f}% used ({used_gb}/{total_gb} GiB, free={free_gb} GiB)"),
                evidence={
                    "mountpoint": mountpoint,
                    "used_pct": used_pct,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "warn_pct": cfg.disk_used_warn_pct,
                    "crit_pct": cfg.disk_used_crit_pct,
                },
                subject={"mountpoint": mountpoint},
                source="local",
                suggestion=(
                    "prune_branch(profile) — profile traces dominate "
                    "$USER_DATA_PATH disk; consider archiving older runs"
                    if severity is SymptomSeverity.HIGH
                    else "observe; rotate logs and clean old runs/* if used_pct keeps climbing"
                ),
            )
        )
    return out


def _shm_pressure_symptoms(
    data: SourceData,
    cfg: LocalHealthConfig,
) -> list[Symptom]:
    """Emit ``shm_pressure`` (stricter thresholds) for SHM mountpoints.

    SGLang/vLLM crash hard when /dev/shm fills; this surfaces the pressure
    before the next server start fails with ``shared memory allocation
    failed``.

    Args:
        data: Collected source data (per-mountpoint disk stats).
        cfg: Local-health configuration thresholds.

    Returns:
        Symptoms for stressed SHM mountpoints, possibly empty.
    """
    if not isinstance(data.local_disk, dict) or not data.local_disk:
        return []
    out: list[Symptom] = []
    shm_set = frozenset(cfg.shm_mountpoints)
    for mountpoint, stats in data.local_disk.items():
        if mountpoint not in shm_set:
            continue
        if not isinstance(stats, dict):
            continue
        used_pct = stats.get("used_pct")
        if not isinstance(used_pct, (int, float)):
            continue
        if used_pct >= cfg.shm_used_crit_pct:
            severity = SymptomSeverity.HIGH
        elif used_pct >= cfg.shm_used_warn_pct:
            severity = SymptomSeverity.MEDIUM
        else:
            continue
        free_gb = stats.get("free_gb")
        total_gb = stats.get("total_gb")
        out.append(
            Symptom(
                name="shm_pressure",
                severity=severity,
                summary=(
                    f"{mountpoint!r} at {used_pct:.0f}% used "
                    f"(free={free_gb}/{total_gb} GiB) — "
                    f"SGLang/vLLM SHM exhaustion is a hard failure"
                ),
                evidence={
                    "mountpoint": mountpoint,
                    "used_pct": used_pct,
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "warn_pct": cfg.shm_used_warn_pct,
                    "crit_pct": cfg.shm_used_crit_pct,
                },
                subject={"mountpoint": mountpoint},
                source="local",
                suggestion=(
                    "lower TP or restart pod with --shm-size; the next "
                    "serving will likely fail with shared memory error"
                    if severity is SymptomSeverity.HIGH
                    else "monitor; restart the affected server if SHM keeps climbing"
                ),
            )
        )
    return out


def _ray_head_dead_symptoms(data: SourceData) -> list[Symptom]:
    """Emit ``ray_head_dead`` (HIGH) when the Ray head is unhealthy.

    Fires when ``data.local_ray`` reports ``healthy=False``; silent when the
    probe slot is empty or healthy. Prompts Orchestration to prune
    kernel_opt.

    Args:
        data: Collected source data (Ray head probe).

    Returns:
        A list with one :class:`Symptom` when unhealthy, else empty.
    """
    ray_info = data.local_ray
    if not ray_info:
        return []
    healthy = ray_info.get("healthy")
    if healthy is None or healthy:
        return []
    reason = str(ray_info.get("reason") or "unknown")
    return [
        Symptom(
            name="ray_head_dead",
            severity=SymptomSeverity.HIGH,
            summary=f"ray head unreachable: {reason}",
            evidence={
                "reason": reason,
                "stderr": (str(ray_info.get("stderr") or ""))[:240],
                "returncode": ray_info.get("returncode"),
            },
            subject={},
            source="local",
            suggestion=(
                "prune_branch(kernel_opt); escalate to restart Ray head — "
                "GEAK + kernel submissions are pending until ray is back"
            ),
        )
    ]


def _fd_pressure_symptoms(
    data: SourceData,
    cfg: LocalHealthConfig,
) -> list[Symptom]:
    """Emit ``fd_pressure`` when Coordinator FD usage nears the limit.

    Leaked sockets hitting ``ulimit -n`` surface as agent_stall(kernel)
    whose real cause is here.

    Args:
        data: Collected source data (file-descriptor usage).
        cfg: Local-health configuration thresholds.

    Returns:
        A list with one :class:`Symptom` when FD pressure trips, else empty.
    """
    fd_info = data.local_fd
    if not fd_info:
        return []
    used_pct = fd_info.get("used_pct")
    if not isinstance(used_pct, (int, float)):
        return []
    if used_pct >= cfg.fd_crit_used_pct:
        severity = SymptomSeverity.HIGH
    elif used_pct >= cfg.fd_warn_used_pct:
        severity = SymptomSeverity.MEDIUM
    else:
        return []
    return [
        Symptom(
            name="fd_pressure",
            severity=severity,
            summary=(
                f"file-descriptor usage {used_pct:.0f}% on pid "
                f"{fd_info.get('pid', '?')} "
                f"(used={fd_info.get('used', '?')}, "
                f"limit={fd_info.get('limit', '?')})"
            ),
            evidence={
                "pid": fd_info.get("pid"),
                "used": fd_info.get("used"),
                "limit": fd_info.get("limit"),
                "used_pct": used_pct,
                "warn_pct": cfg.fd_warn_used_pct,
                "crit_pct": cfg.fd_crit_used_pct,
            },
            subject={"pid": str(fd_info.get("pid") or "")},
            source="local",
            suggestion=(
                "escalate_strategy_change: long-running session has leaked "
                "file descriptors; consider restarting Coordinator + "
                "resume to clear them"
                if severity is SymptomSeverity.HIGH
                else "monitor; if used_pct keeps climbing the next agent_stall likely traces back here"
            ),
        )
    ]


__all__ = ["LocalHealthConfig", "evaluate_local_health_signals"]
