# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pre-launch / pre-action feasibility signals (C1 / C2 / C3) — "doomed before it starts".

* **C1 ``model_gpu_infeasible``** — the ``(model_name, precision, tp, gpu_type,
  max_model_len, conc)`` tuple cannot fit in available HBM. Fires once per session.
* **C2 ``amdahl_kernel_ceiling_low``** — Triton-optimizable tier too small for kernel_opt
  to move E2E: ``E2E_ceiling = 1 / ((1 - p) + p / s)``. Re-fires when breakdown mtime changes.
* **C3 ``cold_start_budget_exhausted``** — aiter JIT cache empty AND remaining budget shorter
  than one cold-start, so the next baseline is SIGTERM'd mid-``hipcc``.

All three suppress repeat fires across the same input fingerprint.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateView
from .symptom import Symptom, SymptomSeverity



# ---------------------------------------------------------------------------
# Static physics tables — conservative engineering values, env-overridable.
# ---------------------------------------------------------------------------

# HBM GiB per GPU device (not aggregate); NVIDIA refs included for ``--compare-against-gpu``.
GPU_HBM_GIB: dict[str, float] = {
    "mi300x": 192.0,
    "mi325x": 256.0,
    "mi355x": 288.0,
    "b200":   192.0,
    "h200":   141.0,
    "h100":   80.0,
    "a100":   80.0,
}

# Bytes per parameter, indexed by manifest ``precision`` (incl. int-quant family).
PRECISION_BYTES_PER_PARAM: dict[str, float] = {
    "fp32":  4.0,
    "fp16":  2.0,
    "bf16":  2.0,
    "fp8":   1.0,
    "int8":  1.0,
    "fp4":   0.5,
    "int4":  0.5,
    "awq":   0.5,   # 4-bit AWQ packs 8 weights per 32-bit word.
    "gptq":  0.5,
}

# Per-token KV cache bytes per model class (rough averages; override via $HYPERLOOM_KV_BYTES).
KV_BYTES_PER_TOKEN: dict[str, float] = {
    "dense":        16.0,
    "moe_swa":      4.0,
    "moe_mla":      0.5,
    "moe_mla_nsa":  0.5,
}

# Fixed activation / scratch buffer per GPU (GiB); 8 GiB covers MoE steady-state.
DEFAULT_ACTIVATION_BUF_GIB: float = 8.0

# Model-name regex matching ``-671B`` / ``-7b`` / `` 3.5B`` size tokens.
_PARAM_BILLIONS_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])(?P<n>\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])"
)


def extract_params_billions(model_name: str) -> float | None:
    """Best-effort parse of ``-<N>B`` from a model name (first match; ``None`` if absent)."""
    if not model_name:
        return None
    match = _PARAM_BILLIONS_RE.search(model_name)
    if not match:
        return None
    try:
        return float(match.group("n"))
    except ValueError:
        return None


@dataclass
class HeadroomBreakdown:
    """Per-GPU HBM budget projection — surfaces in evidence verbatim."""

    weights_gib: float
    kv_cache_gib: float
    activation_gib: float
    required_gib: float
    hbm_gib: float
    headroom_gib: float
    headroom_pct: float


def compute_headroom_gib(
    manifest: dict[str, Any], *,
    activation_buf_gib: float = DEFAULT_ACTIVATION_BUF_GIB,
) -> HeadroomBreakdown | None:
    """Project per-GPU HBM headroom from manifest metadata.

    Returns ``None`` when a required field (model size, precision, gpu_type, ``tp``) is
    unresolved; the C1 detector treats ``None`` as "skip — not enough data to judge".
    """
    if not isinstance(manifest, dict) or not manifest:
        return None
    params_b = extract_params_billions(str(manifest.get("model_name") or ""))
    if params_b is None or params_b <= 0:
        return None
    workload = manifest.get("workload") or {}
    if not isinstance(workload, dict):
        workload = {}
    precision = str(workload.get("precision") or "").strip().lower()
    bytes_per_param = PRECISION_BYTES_PER_PARAM.get(precision)
    if bytes_per_param is None:
        return None
    gpu_type = str(manifest.get("gpu_type") or "").strip().lower()
    hbm_gib = GPU_HBM_GIB.get(gpu_type)
    if hbm_gib is None or hbm_gib <= 0:
        return None
    tp = manifest.get("tp")
    if not isinstance(tp, int) or tp <= 0:
        return None
    model_class = str(manifest.get("model_class") or "dense").strip().lower()
    kv_bpt = KV_BYTES_PER_TOKEN.get(model_class, KV_BYTES_PER_TOKEN["dense"])
    max_model_len = workload.get("max_model_len")
    conc = workload.get("conc")
    if not isinstance(max_model_len, int) or max_model_len <= 0:
        max_model_len = 4096
    if not isinstance(conc, int) or conc <= 0:
        conc = 8

    # Weights split across TP; 5% overhead for pool reservation / shards / launch temporaries.
    total_weights_bytes = params_b * 1_000_000_000 * bytes_per_param * 1.05
    weights_gib = (total_weights_bytes / float(tp)) / (1024 ** 3)

    # KV cache for the in-flight batch; shards over TP (heads are sharded).
    total_kv_bytes = kv_bpt * max_model_len * conc
    kv_cache_gib = (total_kv_bytes / float(tp)) / (1024 ** 3)

    required_gib = weights_gib + kv_cache_gib + activation_buf_gib
    headroom_gib = hbm_gib - required_gib
    # hbm_gib > 0 already guaranteed by the early guard above.
    headroom_pct = (headroom_gib / hbm_gib) * 100.0
    return HeadroomBreakdown(
        weights_gib=round(weights_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        activation_gib=round(float(activation_buf_gib), 2),
        required_gib=round(required_gib, 2),
        hbm_gib=round(hbm_gib, 2),
        headroom_gib=round(headroom_gib, 2),
        headroom_pct=round(headroom_pct, 2),
    )


def amdahl_e2e_ceiling(
    *,
    optimizable_pct: float,
    single_kernel_speedup: float,
) -> float:
    """Amdahl's law best-case E2E speedup ratio (1.0 = none); caller converts via ``(ratio - 1.0) * 100``."""
    p = max(0.0, min(1.0, optimizable_pct / 100.0))
    s = max(1.0, float(single_kernel_speedup))
    serial = 1.0 - p
    if s == 0.0 or (serial + p / s) == 0.0:
        return 1.0
    return 1.0 / (serial + p / s)


# ===========================================================================
# C1 — Model-GPU fit detector
# ===========================================================================

@dataclass
class ModelGpuFitConfig:
    """Tunables for :class:`ModelGpuFitDetector`."""

    # ``headroom_pct`` below this → fire.
    min_headroom_pct: float = 5.0
    activation_buf_gib: float = DEFAULT_ACTIVATION_BUF_GIB


class ModelGpuFitDetector:
    """Stateful: emit ``model_gpu_infeasible`` at most once per session, keyed off the
    immutable manifest fingerprint ``(model_name, gpu_type, tp, precision, max_model_len, conc)``.
    """

    def __init__(
        self,
        config: ModelGpuFitConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or ModelGpuFitConfig()
        self._state_view = state_view
        # Disk-backed dedup so "fire once per session" survives the subprocess-per-tick transport.
        loaded = state_view.load() if state_view is not None else {}
        raw_fp = loaded.get("fired_fingerprint")
        if isinstance(raw_fp, list):
            self._fired_fingerprint: tuple[Any, ...] | None = tuple(raw_fp)
        else:
            self._fired_fingerprint = None

    def _persist(self) -> None:
        if self._state_view is None:
            return
        # tuple → list for JSON round-tripping.
        self._state_view.save({
            "fired_fingerprint": (
                list(self._fired_fingerprint)
                if self._fired_fingerprint is not None
                else None
            ),
        })

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        manifest = data.local_manifest
        if not isinstance(manifest, dict) or not manifest:
            return []
        fingerprint = _manifest_fingerprint(manifest)
        if fingerprint == self._fired_fingerprint:
            return []
        breakdown = compute_headroom_gib(
            manifest, activation_buf_gib=self._config.activation_buf_gib,
        )
        if breakdown is None:
            # Insufficient data — record the fingerprint to avoid retrying until manifest changes.
            self._fired_fingerprint = fingerprint
            self._persist()
            return []
        if breakdown.headroom_pct >= self._config.min_headroom_pct:
            self._fired_fingerprint = fingerprint
            self._persist()
            return []
        self._fired_fingerprint = fingerprint
        self._persist()
        return [self._build_symptom(manifest, breakdown)]

    def _build_symptom(
        self, manifest: dict[str, Any], breakdown: HeadroomBreakdown,
    ) -> Symptom:
        cfg = self._config
        return Symptom(
            name="model_gpu_infeasible",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"model {manifest.get('model_name')!r} on "
                f"{manifest.get('gpu_type')} with tp={manifest.get('tp')} "
                f"needs {breakdown.required_gib} GiB but device has "
                f"{breakdown.hbm_gib} GiB "
                f"(headroom={breakdown.headroom_pct:.1f}%); the next "
                f"baseline will OOM at server-start"
            ),
            evidence={
                "model_name": manifest.get("model_name"),
                "model_class": manifest.get("model_class"),
                "gpu_type": manifest.get("gpu_type"),
                "tp": manifest.get("tp"),
                "precision": (manifest.get("workload") or {}).get("precision"),
                "max_model_len": (manifest.get("workload") or {}).get("max_model_len"),
                "conc": (manifest.get("workload") or {}).get("conc"),
                "weights_gib": breakdown.weights_gib,
                "kv_cache_gib": breakdown.kv_cache_gib,
                "activation_gib": breakdown.activation_gib,
                "required_gib": breakdown.required_gib,
                "hbm_gib": breakdown.hbm_gib,
                "headroom_gib": breakdown.headroom_gib,
                "headroom_pct": breakdown.headroom_pct,
                "min_headroom_pct": cfg.min_headroom_pct,
            },
            subject={},  # session-wide, fires once
            source="local",
            suggestion=(
                f"abort the run; increase TP (need >= {_recommend_tp(breakdown)}) "
                f"or move to a higher-HBM GPU. Robustness cannot save this "
                f"session — the model literally does not fit."
            ),
        )


def _manifest_fingerprint(manifest: dict[str, Any]) -> tuple[Any, ...]:
    workload = manifest.get("workload") or {}
    return (
        str(manifest.get("model_name") or ""),
        str(manifest.get("model_class") or ""),
        str(manifest.get("gpu_type") or ""),
        int(manifest.get("tp") or 0),
        str(workload.get("precision") or "") if isinstance(workload, dict) else "",
        int((workload or {}).get("max_model_len") or 0) if isinstance(workload, dict) else 0,
        int((workload or {}).get("conc") or 0) if isinstance(workload, dict) else 0,
    )


def _recommend_tp(breakdown: HeadroomBreakdown) -> int:
    """Smallest power-of-two TP clearing ``required_gib`` (operator hint; assumes weights dominate KV)."""
    if breakdown.hbm_gib <= 0 or breakdown.weights_gib <= 0:
        return 8
    # Weights shrink ~linearly with TP; ignore KV scaling for the hint.
    target_weight = breakdown.hbm_gib - breakdown.activation_gib - breakdown.kv_cache_gib
    if target_weight <= 0:
        return 16
    ratio = breakdown.weights_gib / max(0.1, target_weight)
    out = 1
    while out < int(ratio) + 1:
        out *= 2
    return max(out, 2)


# ===========================================================================
# C2 — Amdahl kernel-ceiling detector
# ===========================================================================

@dataclass
class AmdahlCeilingConfig:
    """Tunables for :class:`AmdahlCeilingDetector`."""

    # Assumed best-case single-kernel speedup (1.5x = GEAK 2026-05 average).
    single_kernel_speedup: float = 1.5
    # E2E ceiling (% gain) below which kernel_opt is judged pointless → HIGH.
    min_e2e_ceiling_pct: float = 5.0
    # Optimizable tiers from local_kernel_breakdown; Triton is the only one Hyperloom moves.
    optimizable_tier_names: tuple[str, ...] = ("triton",)


class AmdahlCeilingDetector:
    """Stateful: re-fires only when ``kernel_breakdown.json`` mtime changes."""

    def __init__(
        self,
        config: AmdahlCeilingConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        self._config = config or AmdahlCeilingConfig()
        self._state_view = state_view
        # Disk-backed dedup; ``fired_mtime`` re-evaluates only on a fresh kernel_breakdown.json.
        loaded = state_view.load() if state_view is not None else {}
        raw_mtime = loaded.get("fired_mtime")
        self._fired_mtime: float | None = (
            float(raw_mtime)
            if isinstance(raw_mtime, (int, float))
            else None
        )

    def _persist(self) -> None:
        if self._state_view is None:
            return
        self._state_view.save({"fired_mtime": self._fired_mtime})

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        breakdown = data.local_kernel_breakdown
        if not isinstance(breakdown, dict) or not breakdown:
            return []
        mtime = breakdown.get("mtime")
        if not isinstance(mtime, (int, float)):
            return []
        if self._fired_mtime is not None and mtime <= self._fired_mtime:
            return []
        tier_pcts = breakdown.get("tier_pcts") or {}
        if not isinstance(tier_pcts, dict):
            self._fired_mtime = mtime
            self._persist()
            return []
        cfg = self._config
        optimizable_pct = sum(
            float(tier_pcts.get(name) or 0.0)
            for name in cfg.optimizable_tier_names
        )
        ceiling_ratio = amdahl_e2e_ceiling(
            optimizable_pct=optimizable_pct,
            single_kernel_speedup=cfg.single_kernel_speedup,
        )
        ceiling_pct = (ceiling_ratio - 1.0) * 100.0
        # Always advance the gate (skip unchanged traces); fire only when the ceiling is low.
        self._fired_mtime = mtime
        self._persist()
        if ceiling_pct >= cfg.min_e2e_ceiling_pct:
            return []
        return [self._build_symptom(
            tier_pcts=tier_pcts,
            optimizable_pct=optimizable_pct,
            ceiling_pct=ceiling_pct,
            breakdown=breakdown,
        )]

    def _build_symptom(
        self,
        *,
        tier_pcts: dict[str, float],
        optimizable_pct: float,
        ceiling_pct: float,
        breakdown: dict[str, Any],
    ) -> Symptom:
        cfg = self._config
        return Symptom(
            name="amdahl_kernel_ceiling_low",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"profile shows {optimizable_pct:.1f}% Triton-optimizable "
                f"GPU time; E2E ceiling at "
                f"{cfg.single_kernel_speedup}x single-kernel speedup is "
                f"only +{ceiling_pct:.2f}% — below {cfg.min_e2e_ceiling_pct}%"
            ),
            evidence={
                "tier_pcts": tier_pcts,
                "optimizable_pct": round(optimizable_pct, 3),
                "single_kernel_speedup": cfg.single_kernel_speedup,
                "e2e_ceiling_pct": round(ceiling_pct, 3),
                "min_e2e_ceiling_pct": cfg.min_e2e_ceiling_pct,
                "kernel_breakdown_path": breakdown.get("kernel_breakdown_path"),
            },
            subject={},  # session-wide
            source="local",
            suggestion=(
                "prune_branch(kernel_opt); allocate remaining budget to "
                "params/sweep where Amdahl ceiling is higher"
            ),
        )


# ===========================================================================
# C3 — Cold-start budget exhaustion (stateless cross-signal)
# ===========================================================================

@dataclass
class ColdStartConfig:
    """Tunables for :func:`evaluate_cold_start_signals`."""

    # ``aiter_jit.so_count`` below this → COLD (mirrors upstream BaselineExecutor).
    cold_so_count: int = 20
    # When cold AND remaining_minutes < this → fire HIGH (defaults to the cold-start timeout).
    cold_start_minutes: float | None = None  # None → read env
    # Min session budget below which evaluation is skipped (pointless on a smoke test).
    min_budget_minutes: float = 30.0


def _resolve_cold_start_minutes(cfg: ColdStartConfig) -> float:
    if cfg.cold_start_minutes is not None:
        return float(cfg.cold_start_minutes)
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC", "3600",
    )
    try:
        return float(raw) / 60.0
    except ValueError:
        return 60.0


def evaluate_cold_start_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: ColdStartConfig | None = None,
) -> list[Symptom]:
    cfg = config or ColdStartConfig()
    snap = ctx.shared_state
    if snap.budget_minutes < cfg.min_budget_minutes:
        return []
    if snap.closing_phase or snap.stop_reason:
        return []
    aiter = data.local_aiter_jit
    if not isinstance(aiter, dict) or not aiter:
        return []
    so_count = aiter.get("so_count")
    if not isinstance(so_count, int):
        return []
    if so_count >= cfg.cold_so_count:
        return []
    cold_minutes = _resolve_cold_start_minutes(cfg)
    if snap.remaining_minutes <= 0 or snap.remaining_minutes >= cold_minutes:
        return []
    return [
        Symptom(
            name="cold_start_budget_exhausted",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"aiter jit cache cold (so_count={so_count} "
                f"< {cfg.cold_so_count}) with only "
                f"{snap.remaining_minutes:.1f}min remaining; a single "
                f"cold-start cycle needs ~{cold_minutes:.0f}min — the "
                f"next baseline will be SIGTERM'd mid-hipcc"
            ),
            evidence={
                "so_count": so_count,
                "cold_so_count_threshold": cfg.cold_so_count,
                "remaining_minutes": snap.remaining_minutes,
                "cold_start_minutes": cold_minutes,
                "jit_dir": aiter.get("jit_dir"),
            },
            subject={},
            source="local",
            suggestion=(
                "skip the next baseline (use cached current_best) OR "
                "extend INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC; do "
                "NOT just retry with the existing budget"
            ),
        )
    ]


__all__ = [
    "AmdahlCeilingConfig",
    "AmdahlCeilingDetector",
    "ColdStartConfig",
    "GPU_HBM_GIB",
    "HeadroomBreakdown",
    "KV_BYTES_PER_TOKEN",
    "ModelGpuFitConfig",
    "ModelGpuFitDetector",
    "PRECISION_BYTES_PER_PARAM",
    "amdahl_e2e_ceiling",
    "compute_headroom_gib",
    "evaluate_cold_start_signals",
    "extract_params_billions",
]
