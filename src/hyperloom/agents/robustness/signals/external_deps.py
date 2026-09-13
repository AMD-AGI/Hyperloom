# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External-dependency signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData

if TYPE_CHECKING:
    from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity


@dataclass
class ExternalDepsConfig:
    """Tunables for :func:`evaluate_external_deps_signals`."""

    # Mount stat latency budget; ``ok=False`` fires HIGH regardless of latency.
    mount_latency_warn_ms: float = 5000.0
    mount_latency_critical_ms: float = 15000.0


class TraceLensCliFiredOnce:
    """One-shot latch; backed by :class:`DetectorStateView` to survive restarts."""

    def __init__(
        self,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the latch, restoring the fired flag from state if present."""
        self._state_view = state_view
        loaded = state_view.load() if state_view is not None else {}
        self._value: bool = bool(loaded.get("fired", False))

    @property
    def value(self) -> bool:
        """Whether the ``tracelens_cli_missing`` symptom has already fired this session."""
        return self._value

    @value.setter
    def value(self, new_value: bool) -> None:
        """Set the latch flag and persist it to the state view, if any."""
        self._value = bool(new_value)
        if self._state_view is not None:
            self._state_view.save({"fired": self._value})


def evaluate_external_deps_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: ExternalDepsConfig | None = None,
    tracelens_latch: TraceLensCliFiredOnce | None = None,
) -> list[Symptom]:
    """Run the external-dependency rules and aggregate symptoms."""
    cfg = config or ExternalDepsConfig()
    deps = data.local_external_deps
    if not isinstance(deps, dict) or not deps:
        return []
    out: list[Symptom] = []
    out.extend(_gateway_symptoms(deps.get("gateway") or {}))
    out.extend(_mount_symptoms(deps.get("mounts") or [], cfg))
    if tracelens_latch is not None:
        out.extend(
            _tracelens_symptoms(
                deps.get("tracelens_cli") or {},
                tracelens_latch,
            )
        )
    return out


# Upstream gateway 401 / forbidden


def _gateway_symptoms(
    gateway: dict[str, Any],
) -> list[Symptom]:
    """Fire ``gateway_auth_outage`` when the LLM gateway returns 401/403."""
    if not isinstance(gateway, dict) or not gateway:
        return []
    status = str(gateway.get("status") or "")
    status_code = gateway.get("status_code")
    if status == "unauthorized" or (isinstance(status_code, int) and status_code in (401, 403)):
        return [
            Symptom(
                name="gateway_auth_outage",
                severity=SymptomSeverity.HIGH,
                summary=(
                    f"upstream LLM gateway returned {status_code}/{status} "
                    f"on {gateway.get('url')!r}; every claude/codex CLI "
                    f"will now fail at the gateway"
                ),
                evidence={
                    "url": gateway.get("url"),
                    "status_code": status_code,
                    "status": status,
                    "error": gateway.get("error"),
                },
                subject={},
                source="local",
                suggestion=(
                    "rotate $OPENAI_API_KEY / $ANTHROPIC_API_KEY at your LLM gateway and re-export; "
                    "the upstream key is revoked / expired"
                ),
            )
        ]
    return []


# WekaFS / external mount degraded


def _mount_symptoms(
    mounts: list[Any],
    cfg: ExternalDepsConfig,
) -> list[Symptom]:
    """Fire ``wekafs_degraded`` for unreachable or slow external mounts."""
    if not isinstance(mounts, list) or not mounts:
        return []
    out: list[Symptom] = []
    for entry in mounts:
        if not isinstance(entry, dict):
            continue
        env_name = str(entry.get("env_name") or "")
        path = str(entry.get("path") or "")
        latency_ms = entry.get("latency_ms")
        ok = bool(entry.get("ok"))
        error = entry.get("error")
        # stat returned an error (mount disappeared).
        if not ok:
            out.append(
                Symptom(
                    name="wekafs_degraded",
                    severity=SymptomSeverity.HIGH,
                    summary=(f"mount {env_name}={path!r} unreachable: {error or 'unknown error'}"),
                    evidence={
                        "env_name": env_name,
                        "path": path,
                        "error": error,
                        "latency_ms": latency_ms,
                    },
                    subject={"path": path},
                    source="local",
                    suggestion=(
                        "WekaFS mount may have dropped; trace_analyze / "
                        "external CLI / benchmark scripts will hang. Check "
                        "the read-only mount; consider re-mounting"
                    ),
                )
            )
            continue
        # Latency-degraded path.
        if not isinstance(latency_ms, (int, float)):
            continue
        if latency_ms >= cfg.mount_latency_critical_ms:
            severity = SymptomSeverity.HIGH
        elif latency_ms >= cfg.mount_latency_warn_ms:
            severity = SymptomSeverity.MEDIUM
        else:
            continue
        out.append(
            Symptom(
                name="wekafs_degraded",
                severity=severity,
                summary=(
                    f"mount {env_name}={path!r} stat took "
                    f"{float(latency_ms):.0f}ms (warn="
                    f"{cfg.mount_latency_warn_ms:.0f}ms)"
                ),
                evidence={
                    "env_name": env_name,
                    "path": path,
                    "latency_ms": float(latency_ms),
                    "warn_ms": cfg.mount_latency_warn_ms,
                    "critical_ms": cfg.mount_latency_critical_ms,
                },
                subject={"path": path},
                source="local",
                suggestion=(
                    "WekaFS read latency degrading; if it persists, trace_analyze / external CLI requests will time out"
                ),
            )
        )
    return out


# TraceLens CLI missing


def _tracelens_symptoms(
    cli_info: dict[str, Any],
    latch: TraceLensCliFiredOnce,
) -> list[Symptom]:
    """Fire ``tracelens_cli_missing`` once when no TraceLens CLI is on PATH."""
    if not isinstance(cli_info, dict) or not cli_info:
        return []
    if latch.value:
        return []  # one-shot
    if bool(cli_info.get("any_present")):
        return []
    latch.value = True
    found = cli_info.get("found") or {}
    return [
        Symptom(
            name="tracelens_cli_missing",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"neither TraceLens CLI is on PATH "
                f"(checked: {sorted(found.keys())!r}); trace_analyze "
                f"will fail every tick until install.sh is re-run"
            ),
            evidence={
                "cli_names": cli_info.get("cli_names") or [],
                "found": found,
            },
            subject={},
            source="local",
            suggestion=(
                "re-run $REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh; "
                "TraceLens editable install is idempotent and will "
                "restore both perf-report CLI names"
            ),
        )
    ]


__all__ = [
    "ExternalDepsConfig",
    "TraceLensCliFiredOnce",
    "evaluate_external_deps_signals",
]
