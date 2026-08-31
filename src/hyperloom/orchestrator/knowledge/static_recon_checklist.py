# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seed checklist for the static-recon specialist.

The static-recon specialist is a read-only PRELUDE sub-agent that greps the
framework source tree (vLLM / SGLang) for *un-bridged capability switches* —
fast paths that *should* be enabled for the current ``(model_class, gpu_type,
precision)`` but are silently disabled by a predicate (e.g. a CUDA-only
``*_supported()`` helper returning ``False`` on ROCm). This module seeds the
specialist with a curated list of known patterns to look for.

Each :class:`ChecklistEntry` describes one known "bridge opportunity":
- ``id`` — stable slug, used to build the gap canonical id.
- ``applies_when`` — coarse predicate over ``{gpu, precision}`` (``"*"`` = any)
  so an entry is only handed to a run where it could plausibly fire.
- ``detect`` — what to grep for and how to confirm the path is disabled.
- ``consequence`` — what regression the disabled path causes.
- ``bridge`` — advisory sketch of the fix.
- ``domain_hint`` — which specialist domain should pick up the gap.
- ``evidence`` — provenance (PR / session) the pattern was distilled from.

A small, hand-curated starter set keyed to validated findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChecklistEntry:
    """One known un-bridged-capability pattern the static-recon specialist hunts.

    Attributes:
        id: Stable slug; used to build ``gap.static_recon.<id>``.
        applies_when: Coarse match dict over ``{"gpu": ..., "precision": ...}``.
            Values are lower-cased substrings; ``"*"`` (or absent) matches any.
        detect: Grep target + confirmation instruction handed to the specialist.
        consequence: The regression caused while the path stays disabled.
        bridge: Advisory sketch of the fix.
        domain_hint: Specialist domain to route the seeded gap to.
        source_dirs: Source subdirectories to point the specialist at (relative
            to a framework source root); rendered as ``source_hint_directories``.
        evidence: Provenance refs (PR / session) the pattern was distilled from.
    """

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
    """Return True when a checklist ``applies_when`` value matches the run value.

    ``"*"`` / empty matches anything. Otherwise the run value is tokenized on
    non-alphanumeric boundaries (so ``"fp8_e4m3"`` -> ``{"fp8", "e4m3"}``) and
    the entry value matches iff it equals the whole run value or is one of its
    tokens. This is deliberately NOT a loose substring match, so ``"fp8"`` does
    NOT match ``"mxfp8"`` and vice versa (they are distinct precisions).
    """
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
    """Map a GPU type label to a coarse family token used by ``applies_when``.

    AMD Instinct parts (``MI300X`` / ``MI325X`` / ``MI355X`` / ``gfx94*`` /
    ``gfx95*``) map to ``"rocm"``; everything else passes through lower-cased so
    a CUDA/NVIDIA entry could match in the future.
    """
    g = (gpu_type or "").strip().lower()
    if g.startswith("mi") or g.startswith("gfx") or "rocm" in g or "amd" in g:
        return "rocm"
    return g


def entries_for(*, model_class: str = "", gpu_type: str = "", precision: str = "") -> list[ChecklistEntry]:
    """Return the checklist entries applicable to a ``(model_class, gpu, precision)``.

    ``model_class`` is currently advisory (entries gate on gpu/precision only);
    it is accepted now so the signature is stable when model-class-specific
    entries are added.

    Args:
        model_class: Categorical model class (advisory; reserved for future use).
        gpu_type: GPU type label (e.g. ``"MI300X"``), normalized to a family.
        precision: Workload precision (e.g. ``"fp8"`` / ``"mxfp8"``).

    Returns:
        The matching :class:`ChecklistEntry` list (possibly empty).
    """
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
    """Return the de-duplicated source subdirectories to point the specialist at.

    Built from the ``source_dirs`` of every applicable checklist entry, in first
    -seen order so the prompt's navigation hint is stable.

    Args:
        model_class: Categorical model class (advisory; reserved for future use).
        gpu_type: GPU type label.
        precision: Workload precision.

    Returns:
        Ordered tuple of relative source subdirectories (possibly empty).
    """
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
    """Render checklist entries as a Markdown block for the specialist prompt.

    Args:
        entries: The applicable checklist entries (from :func:`entries_for`).

    Returns:
        A Markdown string, or ``""`` when there are no entries.
    """
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
    """Serialize checklist entries to plain dicts (for task params / persistence).

    Args:
        entries: The applicable checklist entries.

    Returns:
        A JSON-serializable list of dicts mirroring the dataclass fields.
    """
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
    """Filter checklist entries based on model metadata.

    Currently gates ``rocm.moe.shared_expert_fusion`` on
    ``model_info["has_shared_expert"]`` so the entry does not appear for
    ROCm + MXFP8 runs whose model has no always-on shared expert.

    Args:
        entries: Candidate entries (typically from :func:`entries_for`).
        model_info: The session ``model_info`` dict (may be empty).

    Returns:
        Filtered list; entries not requiring model-level gating pass through
        unchanged.
    """
    if model_info.get("has_shared_expert"):
        return list(entries)
    return [e for e in entries if e.id != "rocm.moe.shared_expert_fusion"]


__all__ = [
    "ChecklistEntry",
    "entries_for",
    "filter_entries_for_model",
    "source_hint_directories_for",
    "render_checklist_for_prompt",
    "checklist_as_dicts",
]
