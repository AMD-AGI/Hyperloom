# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pre-launch / pre-action feasibility signals (C1 / C2 / C3).

Three independent detectors live here, each catching a different
"this run is doomed before it starts" failure mode:

* **C1 ``model_gpu_infeasible``** — at session boot, the requested
  ``(model_name, precision, tp, gpu_type, max_model_len, conc)`` tuple
  cannot possibly fit in available HBM. The 2026-05 DSR1 post-mortem
  showed 5 sessions burning through 30 minutes of baseline before
  failing because the LLM-supplied ``--tp 1`` couldn't load 671B FP8
  weights into a single MI300X (192 GB).

* **C2 ``amdahl_kernel_ceiling_low``** — after profile completes the
  ``T1_TRITON`` (Hyperloom-optimizable) tier is too small for kernel
  optimization to move the E2E needle. DSR1-FP8 case:
  ``optimizable_triton_pct=30.9%`` × ``single_kernel_speedup=1.5`` =
  ``E2E_ceiling = 1 / ((1 - 0.309) + 0.309 / 1.5) ≈ 1.117``, i.e. at
  best 11.7%; in practice each kernel patch is ~1.1× and lands well
  inside the noise floor.

* **C3 ``cold_start_budget_exhausted``** — aiter JIT cache is empty
  AND the remaining wall-clock budget is shorter than one cold-start
  cycle, so the next baseline/validate_stack will be SIGTERM'd
  mid-``hipcc`` and produce nothing.

All three are stateful in the sense that they suppress repeat fires
across the same input fingerprint. The model-GPU rule only fires once
per session (manifest is immutable); the Amdahl rule re-fires when the
breakdown mtime changes (re-profiled); cold-start fires at most once
per (cold_state, remaining_quanta) tuple to avoid spamming during a
slow cooldown.
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
# Static physics tables — conservative engineering values.
# Override-by-env hooks are provided where operators need to tune for
# new GPUs / unusual model classes without code edits.
# ---------------------------------------------------------------------------

# HBM size per GPU type in GiB (per device, not aggregate).
# MI325X is 256 GB but the upstream installer maps mi325x → mi300x; we
# track it separately so the explicit override case stays correct.
# Reference (NVIDIA) GPUs are included so ``--compare-against-gpu`` runs
# also get a feasibility check.
GPU_HBM_GIB: dict[str, float] = {
    "mi300x": 192.0,
    "mi325x": 256.0,
    "mi355x": 288.0,
    "b200":   192.0,
    "h200":   141.0,
    "h100":   80.0,
    "a100":   80.0,
}

# Bytes per parameter, indexed by ``precision`` field from manifest.
# Includes the int-quant family because Hyperloom supports them on
# specific models (AWQ / GPTQ / SmoothQuant).
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

# Per-token KV cache bytes per model class. The numbers are a single
# rough average across the architectures Hyperloom ships priors for.
# Override per-model via the optional ``$HYPERLOOM_KV_BYTES`` env if
# you have a more accurate figure.
KV_BYTES_PER_TOKEN: dict[str, float] = {
    "dense":        16.0,
    "moe_swa":      4.0,
    "moe_mla":      0.5,
    "moe_mla_nsa":  0.5,
}

# Fixed activation / scratch buffer per GPU (GiB). MoE models consume
# more (gate matmul + expert dispatch buffer), so a single conservative
# 8 GiB matches the steady-state allocation seen in the 2026-05 runs.
DEFAULT_ACTIVATION_BUF_GIB: float = 8.0

# Model-name regex: matches ``-671B`` / ``-7b`` / `` 32B`` variants and
# decimal suffixes (`` 3.5B``). Conservative — we want to match common
# upstream conventions, not invent.
_PARAM_BILLIONS_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])(?P<n>\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])"
)


def extract_params_billions(model_name: str) -> float | None:
    """Best-effort parse of ``-<N>B`` from a model name.

    We deliberately accept the first match — most HF-style names put the
    parameter count at the end (``Qwen3-32B`` / ``DeepSeek-R1-671B``)
    so the leading number ambiguity (e.g. ``LLaMA-2-7B``) is rare
    enough not to warrant heuristics.

    Args:
        model_name (str): Model identifier to scan for a size token.

    Returns:
        float | None: Parameter count in billions, or ``None`` when no
            obvious size token is present or it cannot be parsed.
    """
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

    The C1 detector treats ``None`` as "skip — not enough data to
    judge", which is intentional: the caller's fallback is *always* the
    existing failure mode (baseline returns ``bt=0``), so silent-skip is
    no worse than current behaviour.

    Args:
        manifest (dict[str, Any]): Session manifest with model/GPU/workload
            metadata (``model_name``, ``gpu_type``, ``tp``, ``workload``).
        activation_buf_gib (float): Fixed activation/scratch buffer per GPU
            in GiB to reserve on top of weights and KV cache.

    Returns:
        HeadroomBreakdown | None: The per-GPU budget projection, or ``None``
            when any required field cannot be resolved (no model size in the
            name, unknown precision, unknown GPU type, missing ``tp``).
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

    # Weights — split across TP. 5% overhead for KV-cache pool reservation
    # / param shards / launch-time temporaries the framework reserves
    # before serving traffic.
    total_weights_bytes = params_b * 1_000_000_000 * bytes_per_param * 1.05
    weights_gib = (total_weights_bytes / float(tp)) / (1024 ** 3)

    # KV cache — total bytes for the in-flight batch; also shards over
    # TP because the heads are sharded.
    total_kv_bytes = kv_bpt * max_model_len * conc
    kv_cache_gib = (total_kv_bytes / float(tp)) / (1024 ** 3)

    required_gib = weights_gib + kv_cache_gib + activation_buf_gib
    headroom_gib = hbm_gib - required_gib
    headroom_pct = (headroom_gib / hbm_gib) * 100.0 if hbm_gib > 0 else 0.0
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
    """Amdahl's law: best-case E2E speedup if we optimize the optimizable
    fraction by ``single_kernel_speedup``.

    Args:
        optimizable_pct (float): Percentage of GPU time that lives in the
            optimizable (Triton) tier. Clamped to ``[0, 100]``.
        single_kernel_speedup (float): Assumed per-kernel speedup multiple
            applied to the optimizable fraction. Clamped to ``>= 1.0``.

    Returns:
        float: The E2E speedup *ratio* (1.0 = no speedup). The caller
            converts to a percentage gain via ``(ratio - 1.0) * 100.0``.
    """
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
    """Stateful: emit ``model_gpu_infeasible`` at most once per session.

    The manifest is immutable after boot, so the fingerprint we cache
    is the manifest path + ``(model_name, gpu_type, tp, precision,
    max_model_len, conc)`` tuple. Two consecutive ticks see the same
    fingerprint → second tick stays quiet.
    """

    def __init__(
        self,
        config: ModelGpuFitConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore any persisted dedup state.

        Args:
            config (ModelGpuFitConfig | None): Tunables; defaults to
                :class:`ModelGpuFitConfig` when ``None``.
            state_view (DetectorStateView | None): Disk-backed state view
                used to load/persist the fired fingerprint across ticks.
        """
        self._config = config or ModelGpuFitConfig()
        self._state_view = state_view
        # Disk-backed dedup. Without it, the subprocess-per-tick transport fires
        # ``model_gpu_infeasible`` on every tick and the operator inbox
        # gets one row per tick. Persisting the fingerprint keeps the
        # "fire at most once per session" semantics intact.
        loaded = state_view.load() if state_view is not None else {}
        raw_fp = loaded.get("fired_fingerprint")
        if isinstance(raw_fp, list):
            self._fired_fingerprint: tuple[Any, ...] | None = tuple(raw_fp)
        else:
            self._fired_fingerprint = None

    def _persist(self) -> None:
        """Write the current fired fingerprint to the state view, if any."""
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
        """Emit ``model_gpu_infeasible`` once per session if the model won't fit.

        Computes the per-GPU HBM headroom from the manifest and fires when it
        falls below the configured threshold. The fingerprint is recorded so
        subsequent ticks with the same manifest stay quiet.

        Args:
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data including the local
                manifest used for the feasibility check.

        Returns:
            list[Symptom]: A single ``model_gpu_infeasible`` symptom when the
                model is infeasible, otherwise an empty list.
        """
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
            # Insufficient data — don't claim infeasibility from missing
            # fields. Record the fingerprint anyway so we don't re-try
            # endlessly until manifest changes.
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
        """Construct the ``model_gpu_infeasible`` symptom from the projection.

        Args:
            manifest (dict[str, Any]): Session manifest, used to populate
                evidence (model name, GPU type, tp, workload).
            breakdown (HeadroomBreakdown): Computed per-GPU HBM budget.

        Returns:
            Symptom: A HIGH-severity symptom describing the OOM-at-start risk
                with full evidence and a TP/GPU remediation suggestion.
        """
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
    """Build a stable dedup key from the feasibility-relevant manifest fields.

    Args:
        manifest (dict[str, Any]): Session manifest.

    Returns:
        tuple[Any, ...]: A tuple of ``(model_name, model_class, gpu_type, tp,
            precision, max_model_len, conc)`` suitable for equality comparison
            across ticks.
    """
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
    """Smallest TP power-of-two that would clear ``required_gib``.

    Helper for the operator-facing hint; deliberately conservative
    (assumes weights dominate KV, which is typical for big MoE FP8 models)
    and rescales weights linearly with TP.

    Args:
        breakdown (HeadroomBreakdown): Computed per-GPU HBM budget.

    Returns:
        int: A recommended tensor-parallel degree (power of two, ``>= 2``).
    """
    if breakdown.hbm_gib <= 0 or breakdown.weights_gib <= 0:
        return 8
    # Estimate: weights would shrink linearly with TP; pick the smallest
    # TP where weights_gib / extra_factor + kv + activation <= hbm.
    # We approximate by ignoring KV scaling for the hint.
    target_weight = breakdown.hbm_gib - breakdown.activation_gib - breakdown.kv_cache_gib
    if target_weight <= 0:
        return 16
    ratio = breakdown.weights_gib / max(0.1, target_weight)
    # Round up to next power of two of current TP × ratio.
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

    # Assumed best-case single-kernel speedup (multiple of baseline).
    # 1.5x reflects the GEAK 2026-05 average across accepted kernels.
    single_kernel_speedup: float = 1.5
    # E2E ceiling (percent gain) below which we judge kernel_opt
    # pointless and fire HIGH. Default 5% matches the spec C2 example.
    min_e2e_ceiling_pct: float = 5.0
    # Optimizable tier names from local_kernel_breakdown. Triton is
    # historically the only tier Hyperloom moves the needle on.
    optimizable_tier_names: tuple[str, ...] = ("triton",)


class AmdahlCeilingDetector:
    """Stateful: re-fires only when ``kernel_breakdown.json`` mtime changes."""

    def __init__(
        self,
        config: AmdahlCeilingConfig | None = None,
        *,
        state_view: "DetectorStateView | None" = None,
    ) -> None:
        """Initialise the detector and restore the persisted fire mtime.

        Args:
            config (AmdahlCeilingConfig | None): Tunables; defaults to
                :class:`AmdahlCeilingConfig` when ``None``.
            state_view (DetectorStateView | None): Disk-backed state view used
                to load/persist the last fired ``kernel_breakdown.json`` mtime.
        """
        self._config = config or AmdahlCeilingConfig()
        self._state_view = state_view
        # Disk-backed dedup. ``fired_mtime`` ensures we only re-evaluate
        # when ``kernel_breakdown.json`` is regenerated by a fresh
        # profile run, not every Coordinator tick.
        loaded = state_view.load() if state_view is not None else {}
        raw_mtime = loaded.get("fired_mtime")
        self._fired_mtime: float | None = (
            float(raw_mtime)
            if isinstance(raw_mtime, (int, float))
            else None
        )

    def _persist(self) -> None:
        """Write the last fired breakdown mtime to the state view, if any."""
        if self._state_view is None:
            return
        self._state_view.save({"fired_mtime": self._fired_mtime})

    def evaluate(
        self, ctx: ReactorContext, data: SourceData,
    ) -> list[Symptom]:
        """Fire ``amdahl_kernel_ceiling_low`` when kernel opt can't move E2E.

        Re-evaluates only when the kernel breakdown file's mtime advances,
        computes the Amdahl E2E ceiling for the optimizable tier, and fires
        when that ceiling falls below the configured percentage.

        Args:
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data including the local
                kernel breakdown.

        Returns:
            list[Symptom]: A single ``amdahl_kernel_ceiling_low`` symptom when
                the ceiling is too low, otherwise an empty list.
        """
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
        # We always advance the gate so we don't keep re-evaluating an
        # unchanged trace; symptom only fires when the ceiling is low.
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
        """Construct the ``amdahl_kernel_ceiling_low`` symptom.

        Args:
            tier_pcts (dict[str, float]): Per-tier percentage of GPU time.
            optimizable_pct (float): Summed percentage across optimizable tiers.
            ceiling_pct (float): Computed best-case E2E percentage gain.
            breakdown (dict[str, Any]): Raw kernel breakdown, used for evidence.

        Returns:
            Symptom: A HIGH-severity symptom recommending kernel_opt be pruned
                in favour of higher-ceiling branches.
        """
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

    # ``aiter_jit.so_count`` below this → COLD. Mirrors the upstream
    # BaselineExecutor threshold.
    cold_so_count: int = 20
    # When cold AND remaining_minutes < this → fire HIGH. The default
    # mirrors the cold-start timeout (60 min) so we predict SIGTERM
    # exactly when a cold compile cycle won't finish.
    cold_start_minutes: float | None = None  # None → read env
    # Minimum session budget below which we don't bother evaluating
    # (cold-start signals are pointless on a 10-min smoke).
    min_budget_minutes: float = 30.0


def _resolve_cold_start_minutes(cfg: ColdStartConfig) -> float:
    """Resolve the cold-start cycle duration in minutes.

    Uses the explicit config value when set, otherwise reads the
    ``INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC`` env var (default 3600s).

    Args:
        cfg (ColdStartConfig): Cold-start tunables.

    Returns:
        float: Estimated cold-start cycle length in minutes.
    """
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
    """Fire ``cold_start_budget_exhausted`` when a cold JIT cache won't finish.

    Detects the C3 failure mode: the aiter JIT cache is cold and the remaining
    wall-clock budget is shorter than one cold-start compile cycle, so the next
    baseline would be SIGTERM'd mid-``hipcc``.

    Args:
        ctx (ReactorContext): Reactor context, providing budget/phase state.
        data (SourceData): Collected source data including ``local_aiter_jit``.
        config (ColdStartConfig | None): Tunables; defaults to
            :class:`ColdStartConfig` when ``None``.

    Returns:
        list[Symptom]: A single ``cold_start_budget_exhausted`` symptom when the
            cache is cold and budget is insufficient, otherwise an empty list.
    """
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
