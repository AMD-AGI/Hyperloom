"""External-dependency signals (J1 / J2 / J3).

These three signals cover failure modes that **originate outside
Hyperloom proper** but manifest as opaque hangs / 401 storms inside the
session:

* **J1 ``gateway_auth_outage``** — ``OPENAI_BASE_URL/models`` returns
  401 / forbidden when probed with ``$SAFE_API_KEY``. The upstream
  gateway has lost or revoked the key. Every claude/codex CLI will
  now fail with HTTP 401 at the gateway level.

* **J2 ``wekafs_degraded``** — ``stat`` on any of
  ``$TRACELENS_ROOT`` / ``$INFERENCEX_PATH`` / ``$OOB_SRC`` either
  errored or took longer than the configured budget. WekaFS is the
  read-only source mount Hyperloom relies on for source-code, traces,
  and benchmark scripts; ``trace_analyze`` and the OOB CLI hang
  silently when the mount goes slow / drops.

* **J3 ``tracelens_cli_missing``** — neither
  ``TraceLens_generate_perf_report_pytorch_inference`` nor the legacy
  ``TraceLens_generate_perf_report_pytorch`` is on ``PATH``. This is a
  boot-time-only condition (the binaries land via ``install.sh`` and
  don't disappear mid-run), so the detector latches and stays silent
  after the first fire.

J4 ``cluster_gpu_quota_anomaly`` is covered by F1
``ray_pending_starvation`` (see ``signals/kernel_pipeline.py``); the
P2#7 case study shows up there because all sweep tasks land on
``PENDING`` for >10 min when the cluster quota ledger goes wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



@dataclass
class ExternalDepsConfig:
    """Tunables for :func:`evaluate_external_deps_signals`."""

    # J1 — the J1 detector fires immediately on a single 401 because
    # the gateway is the global authentication source; one bad probe
    # means every subsequent CLI call WILL fail too.
    fire_on_first_401: bool = True
    # J2 — mount stat latency budget. The probe layer also tracks
    # ``ok=False`` for FileNotFoundError; that variant fires HIGH
    # regardless of latency.
    mount_latency_warn_ms: float = 5000.0
    mount_latency_critical_ms: float = 15000.0


class TraceLensCliFiredOnce:
    """One-shot latch helper used by :class:`ExternalDepsDetector`.

    Backed by a :class:`DetectorStateView` when supplied so the latch
    survives subprocess restarts; M1 transport would otherwise re-fire
    the J3 symptom every tick.
    """

    def __init__(
        self,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._state_view = state_view
        loaded = state_view.load() if state_view is not None else {}
        self._value: bool = bool(loaded.get("fired", False))

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, new_value: bool) -> None:
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
    cfg = config or ExternalDepsConfig()
    deps = data.local_external_deps
    if not isinstance(deps, dict) or not deps:
        return []
    out: list[Symptom] = []
    out.extend(_gateway_symptoms(deps.get("gateway") or {}, cfg))
    out.extend(_mount_symptoms(deps.get("mounts") or [], cfg))
    if tracelens_latch is not None:
        out.extend(_tracelens_symptoms(
            deps.get("tracelens_cli") or {}, tracelens_latch,
        ))
    return out


# ---------------------------------------------------------------------------
# J1 — Upstream gateway 401 / forbidden
# ---------------------------------------------------------------------------

def _gateway_symptoms(
    gateway: dict[str, Any], cfg: ExternalDepsConfig,
) -> list[Symptom]:
    if not isinstance(gateway, dict) or not gateway:
        return []
    status = str(gateway.get("status") or "")
    status_code = gateway.get("status_code")
    if status == "unauthorized" or (
        isinstance(status_code, int) and status_code in (401, 403)
    ):
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
                    "rotate $SAFE_API_KEY at https://llm.amd.com/ and "
                    "re-export; the upstream key is revoked / expired"
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# J2 — WekaFS / external mount degraded
# ---------------------------------------------------------------------------

def _mount_symptoms(
    mounts: list[Any], cfg: ExternalDepsConfig,
) -> list[Symptom]:
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
        # Fatal-path: stat returned an error (mount disappeared).
        if not ok:
            out.append(
                Symptom(
                    name="wekafs_degraded",
                    severity=SymptomSeverity.HIGH,
                    summary=(
                        f"mount {env_name}={path!r} unreachable: "
                        f"{error or 'unknown error'}"
                    ),
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
                        "OOB CLI / benchmark scripts will hang. Check "
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
                    "WekaFS read latency degrading; if it persists, "
                    "trace_analyze / OOB CLI requests will time out"
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# J3 — TraceLens CLI missing
# ---------------------------------------------------------------------------

def _tracelens_symptoms(
    cli_info: dict[str, Any],
    latch: TraceLensCliFiredOnce,
) -> list[Symptom]:
    if not isinstance(cli_info, dict) or not cli_info:
        return []
    if latch.value:
        return []  # one-shot — already alerted in this session.
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
                "re-run $REPO_ROOT/inference_optimizer/scripts/install.sh; "
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
