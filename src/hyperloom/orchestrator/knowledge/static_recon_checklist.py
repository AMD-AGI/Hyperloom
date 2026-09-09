# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seed checklist for the static-recon specialist."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistEntry:
    """One known un-bridged-capability pattern the static-recon specialist hunts."""

    id: str
    applies_when: dict[str, str]
    detect: str
    consequence: str
    bridge: str
    domain_hint: str = "freeform"
    source_dirs: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


# Curated starter set; keep entries grounded in a validated finding.
_CHECKLIST: tuple[ChecklistEntry, ...] = (
    ChecklistEntry(
        id="rocm.fp8.cutlass_only_guard",
        applies_when={"gpu": "rocm", "precision": "fp8"},
        detect=(
            "grep for `cutlass_fp8_supported` usage in "
            "vllm/model_executor/layers/quantization/. On ROCm it is CUDA-only "
            "(returns False), so `Fp8LinearMethod.__init__` falls to per-tensor "
            "activation + per-tensor weight scales. Confirm the dense Linear "
            "path lands on per-tensor scales rather than per-token/per-channel."
        ),
        consequence=(
            "Per-tensor scales disqualify every AITER fp8 kernel "
            "(AiterHipbMM/AiterPerToken/AiterPreshuffled require per-token act + "
            "per-channel weight), so dense GEMMs fall back to bf16 "
            "rocm_unquantized_gemm / per-tensor torch._scaled_mm."
        ),
        bridge=(
            "On the ROCm+AITER fp8 Linear path select per-token activation "
            "(kFp8DynamicTokenSym) + per-channel weight (kFp8StaticChannelSym), "
            "and make the online weight quant per-channel, so dense Linears "
            "route to the AITER fp8 GEMM."
        ),
        domain_hint="freeform",
        source_dirs=("vllm/model_executor/layers/quantization/",),
        evidence=("vllm#45854", "session:Qwen3-32B/20260622T032133Z"),
    ),
    ChecklistEntry(
        id="rocm.fp8.aiter_linear_disabled",
        applies_when={"gpu": "rocm", "precision": "fp8"},
        detect=(
            "grep for `is_linear_enabled` / `is_linear_fp8_enabled` "
            "(vllm/_aiter_ops.py) and the AITER linear env gates "
            "(VLLM_ROCM_USE_AITER_LINEAR / _LINEAR_HIPBMM). Confirm whether the "
            "AITER dense-linear fp8 path is gated off for the current run."
        ),
        consequence=(
            "With AITER linear disabled the per-token/per-channel fp8 GEMM "
            "selection never triggers even when scales are correct, leaving "
            "dense Linears on the slower scaled_mm / bf16 path."
        ),
        bridge=(
            "Enable the AITER linear path (env/flag) and confirm "
            "AiterHipbMMPerTokenFp8ScaledMMLinearKernel is selected; pair with "
            "the per-channel scale bridge above."
        ),
        domain_hint="freeform",
        source_dirs=(
            "vllm/model_executor/layers/quantization/",
            "vllm/model_executor/kernels/linear/",
        ),
        evidence=("vllm#45854",),
    ),
    ChecklistEntry(
        id="rocm.mxfp8.smallm_dispatch_gap",
        applies_when={"gpu": "rocm", "precision": "mxfp8"},
        detect=(
            "grep for `dot_scaled` / MXFP8 native linear+grouped-GEMM dispatch "
            "(rocm_native.py, mxfp8_native_moe.py). Confirm whether a low-M "
            "(decode) path tries an AITER small-M HIP kernel before falling back "
            "to the Triton dot_scaled kernel."
        ),
        consequence=(
            "Without small-M dispatch, low-concurrency decode MXFP8 GEMMs run "
            "the Triton dot_scaled kernel which is weight-bandwidth/occupancy "
            "bound at small M, leaving decode TPOT on the table."
        ),
        bridge=(
            "Add a try-import dispatch to the AITER small-M MXFP8 GEMM/grouped "
            "GEMM (guarded by the AITER master switch and a None-fallback to "
            "Triton) on the non-EP decode path."
        ),
        domain_hint="kernel_switch_specialist",
        source_dirs=(
            "vllm/model_executor/kernels/linear/mxfp8/",
            "vllm/model_executor/layers/fused_moe/experts/",
        ),
        evidence=("vllm#46063",),
    ),
    ChecklistEntry(
        id="rocm.moe.aiter_backend_activation_gap",
        applies_when={"gpu": "rocm", "precision": "*"},
        detect=(
            "For MoE models, grep the MoE backend selection "
            "(fused_moe/oracle/*.py, rocm_aiter_moe.py) and `_supports_activation`. "
            "Confirm whether the model's activation (e.g. SWIGLUOAI_UNINTERLEAVE) "
            "and pad config are accepted by the AITER MoE backend, or silently "
            "rejected so it falls back to a slower backend."
        ),
        consequence=(
            "An unsupported activation/pad config makes the AITER MoE backend "
            "self-reject, so MoE runs the slower Triton/unfused path even when "
            "--moe-backend aiter is requested."
        ),
        bridge=(
            "Add the model's activation to `_supports_activation` and thread the "
            "required pad / GateMode config so the AITER MoE backend accepts it."
        ),
        domain_hint="kernel_switch_specialist",
        source_dirs=("vllm/model_executor/layers/fused_moe/",),
        evidence=("vllm#46419",),
    ),
    ChecklistEntry(
        id="rocm.moe.shared_expert_fusion",
        applies_when={"gpu": "rocm", "precision": "mxfp8"},
        detect=(
            "For MoE models with always-on shared experts (n_shared_experts / "
            "num_shared_experts in config.json), grep whether the shared expert "
            "still runs as a separate dense MLP per layer. "
            "In vLLM: check vllm/model_executor/models/ for a `shared_experts` "
            "forward call outside FusedMoE, and vllm/model_executor/layers/fused_moe/ "
            "for whether n_shared_experts is passed to FusedMoE or handled separately. "
            "In SGLang: check python/sglang/srt/models/ and python/sglang/srt/layers/moe/ "
            "for equivalent separate shared-expert execution. "
            "Anti-signatures (do NOT proceed): expert parallelism enabled, "
            "non-uniform precision between shared and routed experts, "
            "prefill-only or high-concurrency-only workload."
        ),
        consequence=(
            "A separate shared-expert MLP adds one extra GEMM launch per MoE layer "
            "during decode. At low-to-medium concurrency this makes decode launch-bound, "
            "degrading throughput significantly (validated: up to +20-30% at concurrency 1, "
            "+6-11% at concurrency 64 on MiniMax-M3 MXFP8 MI355X)."
        ),
        bridge=(
            "Fold the shared expert into the routed grouped-GEMM path as an always-selected "
            "extra expert slot: (1) append shared expert ids to the router top-k selection, "
            "(2) pass n_shared_experts to FusedMoE so it adjusts expert count, "
            "(3) load shared expert weights into the routed expert weight tensor at the end, "
            "(4) handle MXFP8 native MoE bin count to match actual weight rows. "
            "A/B gate with <FUSE_FLAG>=0 vs 1 (confirm actual env-flag name from the "
            "framework build or generated patch; do not treat env-only no-op as KEEP). "
            "Require accuracy gate; check for routed scale compensation to avoid "
            "double-counting shared expert output. "
            "Reference: vLLM PR #46545, upstream MiniMax-M3 shared-expert fusion."
        ),
        domain_hint="freeform",
        source_dirs=(
            "vllm/model_executor/layers/fused_moe/",
            "vllm/model_executor/models/",
            "python/sglang/srt/layers/moe/",
            "python/sglang/srt/models/",
        ),
        evidence=("vllm#46545", "MiniMax-M3-shared-expert-fusion-MI355X-mxfp8"),
    ),
)


def _matches(entry_val: str, run_val: str) -> bool:
    """Return True when a checklist ``applies_when`` value matches the run value."""
    entry_val = (entry_val or "").strip().lower()
    if not entry_val or entry_val == "*":
        return True
    run_val = (run_val or "").strip().lower()
    if not run_val:
        return False
    if entry_val == run_val:
        return True
    tokens = {t for t in re.split(r"[^a-z0-9]+", run_val) if t}
    return entry_val in tokens


def _gpu_family(gpu_type: str) -> str:
    """Map a GPU type label to a coarse family token used by ``applies_when``."""
    g = (gpu_type or "").strip().lower()
    if g.startswith("mi") or g.startswith("gfx") or "rocm" in g or "amd" in g:
        return "rocm"
    return g


def workload_precision(state: object) -> str:
    """Return the precision the checklist should match against.

    ``SharedState.precision`` mirrors ``$PRECISION``, which an operator sets by
    hand and which reads ``fp8`` for MXFP8 checkpoints. The model's own
    ``config.json`` is the ground truth, so ``model_info["quantization"]`` wins
    where it is present -- matching on the operator's label handed an MXFP8 run
    the fp8 entries and withheld the mxfp8 ones.

    Args:
        state: The SharedState (or any object carrying ``model_info`` /
            ``precision``).

    Returns:
        str: The resolved precision, lower-cased; ``""`` when neither is set.
    """
    model_info = getattr(state, "model_info", None)
    if isinstance(model_info, dict):
        quant = str(model_info.get("quantization") or "").strip().lower()
        if quant:
            return quant
    return str(getattr(state, "precision", "") or "").strip().lower()


def entries_for(*, model_class: str = "", gpu_type: str = "", precision: str = "") -> list[ChecklistEntry]:
    """Return the checklist entries applicable to a ``(model_class, gpu, precision)``."""
    gpu_fam = _gpu_family(gpu_type)
    out: list[ChecklistEntry] = []
    for e in _CHECKLIST:
        if not _matches(e.applies_when.get("gpu", "*"), gpu_fam):
            continue
        if not _matches(e.applies_when.get("precision", "*"), precision):
            continue
        out.append(e)
    return out


def source_hint_directories_for(*, model_class: str = "", gpu_type: str = "", precision: str = "") -> tuple[str, ...]:
    """Return the de-duplicated source subdirectories to point the specialist at."""
    seen: set[str] = set()
    out: list[str] = []
    for e in entries_for(model_class=model_class, gpu_type=gpu_type, precision=precision):
        for d in e.source_dirs:
            d = (d or "").strip()
            if d and d not in seen:
                seen.add(d)
                out.append(d)
    return tuple(out)


def render_checklist_for_prompt(entries: list[ChecklistEntry]) -> str:
    """Render checklist entries as a Markdown block for the specialist prompt."""
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries:
        lines.append(f"- **{e.id}** (domain_hint=`{e.domain_hint}`)")
        lines.append(f"  - detect: {e.detect}")
        lines.append(f"  - consequence: {e.consequence}")
        lines.append(f"  - bridge: {e.bridge}")
        if e.source_dirs:
            lines.append(f"  - look under: {', '.join(e.source_dirs)}")
        if e.evidence:
            lines.append(f"  - evidence: {', '.join(e.evidence)}")
    return "\n".join(lines)


def checklist_as_dicts(entries: list[ChecklistEntry]) -> list[dict[str, object]]:
    """Serialize checklist entries to plain dicts (for task params / persistence)."""
    out: list[dict[str, object]] = []
    for e in entries:
        out.append(
            {
                "id": e.id,
                "applies_when": dict(e.applies_when),
                "detect": e.detect,
                "consequence": e.consequence,
                "bridge": e.bridge,
                "domain_hint": e.domain_hint,
                "source_dirs": list(e.source_dirs),
                "evidence": list(e.evidence),
            }
        )
    return out


def filter_entries_for_model(entries: list[ChecklistEntry], model_info: dict) -> list[ChecklistEntry]:
    """Filter checklist entries based on model metadata."""
    if model_info.get("has_shared_expert"):
        return list(entries)
    return [e for e in entries if e.id != "rocm.moe.shared_expert_fusion"]


__all__ = [
    "ChecklistEntry",
    "entries_for",
    "workload_precision",
    "filter_entries_for_model",
    "source_hint_directories_for",
    "render_checklist_for_prompt",
    "checklist_as_dicts",
]
