"""Symptoms derived from LocalProbe-only data.

These rules fire only when DegradeRouter has handed control to
:class:`LocalProbeSource` (i.e. robustness-server is unreachable),
which is exactly the single-mode dev / disconnected scenario M1.5
exists to support.  When the server is healthy the corresponding
signals come from cluster sources (M2+) and these rules stay silent
because the SourceData fields are empty.

Three sub-rules:

* ``server_unreachable`` — a configured local HTTP probe target is
  down. Medium severity by default; promoted to high if every probed
  target fails.
* ``log_error_pattern`` — :data:`SourceData.local_log_errors` contains
  one or more matches. OOM / NCCL → high; anything else → medium.
* ``gpu_thermal_high`` — any GPU's ``temperature_c`` exceeds the
  configured warn / crit thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity


@dataclass
class LocalHealthConfig:
    gpu_temp_warn_c: float = 90.0
    gpu_temp_crit_c: float = 100.0


# Patterns we treat as high severity (vs the default medium). Anything
# else from ``local_log_errors`` falls back to medium.
_HIGH_SEVERITY_PATTERNS: frozenset[str] = frozenset({
    "CUDA out of memory",
    "hipErrorOutOfMemory",
    "HIP out of memory",
    "NCCL error",
    "OOMKilled",
    "core dumped",
    "Segmentation fault",
})


def evaluate_local_health_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: LocalHealthConfig | None = None,
) -> list[Symptom]:
    cfg = config or LocalHealthConfig()
    out: list[Symptom] = []
    out.extend(_server_unreachable(data))
    out.extend(_log_error_symptoms(data))
    out.extend(_gpu_thermal_symptoms(data, cfg))
    return out


def _server_unreachable(data: SourceData) -> list[Symptom]:
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


__all__ = ["LocalHealthConfig", "evaluate_local_health_signals"]
