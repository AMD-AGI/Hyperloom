# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Symptoms derived from LocalProbe-only data.

Fire only when DegradeRouter hands control to :class:`LocalProbeSource` (robustness-server
unreachable); silent otherwise since the SourceData fields are empty. Covers
``server_unreachable`` (HIGH if all targets fail), ``log_error_pattern`` (OOM/NCCL → HIGH),
``gpu_thermal_high``, plus disk/shm/ray-head/fd pressure rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


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
        ray_head_unreachable_severity (str): Severity label used when the Ray
            head is unreachable.
        fd_warn_used_pct (float): File-descriptor used-percent for a MEDIUM
            symptom.
        fd_crit_used_pct (float): File-descriptor used-percent for a HIGH
            symptom.
    """

    gpu_temp_warn_c: float = 90.0
    gpu_temp_crit_c: float = 100.0
    # disk_pressure thresholds (% used); crit also emits a prune_branch(profile) hint.
    disk_used_warn_pct: float = 85.0
    disk_used_crit_pct: float = 95.0
    # SHM gets stricter thresholds — small (16-64 GiB) and SGLang/vLLM crash hard when it fills.
    shm_mountpoints: tuple[str, ...] = ("/dev/shm",)
    shm_used_warn_pct: float = 75.0
    shm_used_crit_pct: float = 90.0
    # Ray-head severity kept here for a single source of truth (rule lives below).
    ray_head_unreachable_severity: str = "high"
    fd_warn_used_pct: float = 80.0
    fd_crit_used_pct: float = 95.0


# HIGH-severity log patterns; anything else from ``local_log_errors`` falls back to MEDIUM.
_HIGH_SEVERITY_PATTERNS: frozenset[str] = frozenset({
    # Existing OOM / fatal-signal patterns.
    "CUDA out of memory",
    "hipErrorOutOfMemory",
    "HIP out of memory",
    "NCCL error",
    "OOMKilled",
    "core dumped",
    "Segmentation fault",
    # D1 — vLLM v1 EngineCore crashes (you've-seen-this case).
    r"Engine core .* died",
    r"RuntimeError: Engine core initialization failed",
    # D1 — model architecture mismatch.
    r"MLA.*not supported",
    r"MTP draft .* unavailable",
    # D1 — aiter / hipcc compilation hard fail.
    r"aiter .* compilation failed",
    r"hipcc .* signal",
    # D1 — accuracy gate / model load — drop here, do not retry.
    r"accuracy .* gate failed",
    r"MMLU .* below threshold",
    r"Failed to load checkpoint",
    # D1 — KFD resource exhaustion (≠ OOM but equally terminal).
    r"cudaErrorOutOfDevice",
    r"HSA_STATUS_ERROR_OUT_OF_RESOURCES",
    # D1 — matrix library errors.
    r"ROCblas.*Status\s*\d+",
    r"hipBLAS.*Error",
    # E5 — critic-agent runtime stuck → demand operator switch to mock.
    r"runtime\.cli .* timed out after \d+s",
})


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
    out.extend(_server_unreachable(data))
    out.extend(_log_error_symptoms(data))
    out.extend(_gpu_thermal_symptoms(data, cfg))
    out.extend(_disk_pressure_symptoms(data, cfg))
    out.extend(_shm_pressure_symptoms(data, cfg))
    out.extend(_ray_head_dead_symptoms(data))
    out.extend(_fd_pressure_symptoms(data, cfg))
    return out


def _server_unreachable(data: SourceData) -> list[Symptom]:
    """Emit ``local_server_unreachable`` for each failed local HTTP probe.

    Severity is HIGH when every probed target is unreachable, otherwise MEDIUM.

    Args:
        data (SourceData): Collected source data including
            ``local_server_health``.

    Returns:
        list[Symptom]: One symptom per unreachable probe target, possibly empty.
    """
    if not data.local_server_health:
        return []
    bad = [entry for entry in data.local_server_health if not entry.get("reachable")]
    if not bad:
        return []
    severity = (
        SymptomSeverity.HIGH if len(bad) == len(data.local_server_health) else SymptomSeverity.MEDIUM
    )
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
                },
                subject={"url": url},
                source="local",
                suggestion=(
                    "delegate(server_lifecycle) to restart the inference "
                    "server" if severity is SymptomSeverity.HIGH else
                    "monitor; alert orchestration if it persists"
                ),
            )
        )
    return out


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
        severity = (
            SymptomSeverity.HIGH if pattern in _HIGH_SEVERITY_PATTERNS else SymptomSeverity.MEDIUM
        )
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
                    "delegate(server_lifecycle) or escalate strategy"
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
                summary=(
                    f"disk {mountpoint!r} at {used_pct:.0f}% used "
                    f"({used_gb}/{total_gb} GiB, free={free_gb} GiB)"
                ),
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
                    if severity is SymptomSeverity.HIGH else
                    "observe; rotate logs and clean old runs/* if used_pct keeps climbing"
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
                    "validate_stack will likely fail with shared memory error"
                    if severity is SymptomSeverity.HIGH else
                    "monitor; restart the affected server if SHM keeps climbing"
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
    ray_info = getattr(data, "local_ray", None)
    if not isinstance(ray_info, dict) or not ray_info:
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
                "GEAK + OOB submissions are pending until ray is back"
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
    fd_info = getattr(data, "local_fd", None)
    if not isinstance(fd_info, dict) or not fd_info:
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
                f"{fd_info.get('pid','?')} "
                f"(used={fd_info.get('used','?')}, "
                f"limit={fd_info.get('limit','?')})"
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
                if severity is SymptomSeverity.HIGH else
                "monitor; if used_pct keeps climbing the next agent_stall "
                "likely traces back here"
            ),
        )
    ]


__all__ = ["LocalHealthConfig", "evaluate_local_health_signals"]
