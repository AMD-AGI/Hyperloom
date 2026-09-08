# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuner routing: select which tuner(s) to run based on model, framework, precision."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_analyzer import ModelProfile

log = logging.getLogger(__name__)


@dataclass
class TunerSpec:
    """A selected tuner with its rationale."""

    name: str
    skip_reason: str | None = None  # If set, tuner is skipped with this explanation
    priority: int = 0  # Lower = run first
    estimated_minutes: float = 10.0  # Estimated runtime for budget allocation
    # Token counts this tuner is responsible for, when the log says it serves
    # only part of the range. Two MoE backends can split one run between them,
    # and tuning the half the other backend serves is time spent on a table
    # nothing will read. ``None`` means the run's full token coverage.
    token_hint: list[int] | None = None
    # A fallback tuner runs only when no earlier non-fallback tuner produced a
    # deployable candidate. It exists so the fp8-barren -> bf16-dense retry that
    # Hyperloom used to launch as a second subprocess happens inside this one
    # call instead: the bf16 dense pass is selected up front but executed only if
    # the fp8 tuning came back empty, so a run whose fp8 tuning won never spends
    # budget tuning a bf16 table it may not even dispatch.
    fallback: bool = False

    @property
    def should_run(self) -> bool:
        return self.skip_reason is None


# Kernel signature patterns indicating 1-stage ASM (from server log)
_1STAGE_PATTERN = re.compile(r"using 1stage default", re.IGNORECASE)

# Map known GPU type strings to their gfx architecture. FP4/MXFP4 GEMM is only
# supported by aiter on gfx950 (CDNA4 / MI355X). gfx942 (CDNA3 / MI300X family)
# hard-rejects FP4 GEMM at runtime and its dense/MoE FP4 tuners are gfx950-only.
_GPU_TYPE_TO_GFX = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
    "amd_instinct_mi300x": "gfx942",
    "amd_instinct_mi355x": "gfx950",
}
_GFX_TO_CANONICAL_GPU = {
    "gfx942": "mi300x",
    "gfx950": "mi355x",
}

# Architectures that cannot run FP4/MXFP4 GEMM (aiter requires gfx950).
_FP4_UNSUPPORTED_GFX = {"gfx942"}

_FP4_GFX942_SKIP_REASON = "FP4/MXFP4 GEMM unsupported on gfx942 (aiter requires gfx950)"


def _detect_local_gfx_arch() -> str:
    """Best-effort detect the local AMD gfx arch via ``rocminfo``.

    Returns the first ``gfxNNN`` token reported (e.g. ``gfx942`` / ``gfx950``),
    or ``""`` when ``rocminfo`` is missing/unparseable so callers fail open
    (never skip a tuner on an undetectable host).
    """
    try:
        out = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"\bgfx[0-9a-f]+\b", out, re.IGNORECASE)
    return m.group(0).lower() if m else ""


def resolve_gpu_type(gpu_type: str) -> str:
    """Resolve a CLI GPU value to a stable KB-compatible identifier.

    Automatic detection is intentionally fail-closed so ``auto`` can never
    leak into artifact names, plans, or knowledge fingerprints.
    """
    key = str(gpu_type or "").strip().lower()
    if key in ("", "auto"):
        key = _detect_local_gfx_arch()
        if not key:
            raise ValueError(
                "Unable to detect the local GPU with rocminfo; "
                "pass --gpu-type explicitly (for example, mi300x or mi355x)."
            )
    normalized = re.sub(r"[\s-]+", "_", key)
    if normalized in _GPU_TYPE_TO_GFX:
        return _GFX_TO_CANONICAL_GPU.get(_GPU_TYPE_TO_GFX[normalized], normalized)
    return _GFX_TO_CANONICAL_GPU.get(normalized, normalized)


def _resolve_gfx_arch(gpu_type: str) -> str:
    """Map a GPU type string (e.g. 'mi300x') to its gfx arch (e.g. 'gfx942').

    Accepts a marketing name ('mi300x', 'mi355x'), a raw gfx string ('gfx942'),
    or ``"auto"``/``""`` to probe the local host via ``rocminfo``. Returns ``""``
    for unrecognized inputs (and undetectable hosts) so callers fail open and
    preserve existing behavior rather than skipping tuners by mistake.
    """
    key = gpu_type.strip().lower()
    if key in ("", "auto"):
        return _detect_local_gfx_arch()
    if key.startswith("gfx"):
        return key
    return _GPU_TYPE_TO_GFX.get(key, "")


def _fp4_unsupported_on(gfx_arch: str) -> bool:
    """True if FP4/MXFP4 GEMM is known to be unsupported on this gfx arch."""
    return gfx_arch in _FP4_UNSUPPORTED_GFX


def moe_stage_coverage(log_path: str | None) -> dict[str, Any]:
    """Which MoE stages the runtime dispatched, and over which token counts.

    A model does not pick one stage and keep it: aiter dispatches 1-stage ASM at
    some token counts and CK 2-stage at others, in the same run. Collapsing that
    into "did we see 1stage anywhere?" throws away the token range 2-stage
    actually serves -- observed covering tokens 1-32 -- and skips tuning for all
    of it.

    Returns ``{"stages_seen": [...], "tunable_ck_2stage": bool,
    "tokens_by_stage": {stage: [tokens]}, "missed_ck_keys": int}``; empty when
    there is nothing to read. ``tunable_ck_2stage`` describes dispatch capability;
    ``missed_ck_keys`` says whether that capability actually needs tuning.
    """
    if not log_path:
        return {}
    path = Path(log_path)
    if not path.is_file():
        return {}
    try:
        from .evidence import moe_ck_missed_keys, parse_log_file

        report = parse_log_file(path)
        moe = (report.get("dispatch") or {}).get("moe") or {}
    except Exception:  # noqa: BLE001 - detection must never break routing
        log.debug("MoE stage parse failed for %s", path, exc_info=True)
        return {}
    by_stage = moe.get("by_stage") or {}
    return {
        "stages_seen": moe.get("stages_seen") or [],
        "tunable_ck_2stage": bool(moe.get("tunable_ck_2stage")),
        "tokens_by_stage": {k: v.get("tokens") or [] for k, v in by_stage.items()},
        "missed_ck_keys": len(moe_ck_missed_keys(report)),
    }


def _detect_1stage_from_log(log_path: str | None) -> bool:
    """True only when 1-stage ASM is the *only* MoE path the runtime used.

    Kept as the routing predicate, but no longer a "saw it once" flag: seeing
    1-stage alongside 2-stage means part of the token range is CK-served and
    therefore tunable, so skipping the CK tuner would forfeit it.
    """
    stages = (moe_stage_coverage(log_path) or {}).get("stages_seen") or []
    if stages:
        # 2-stage present anywhere => there is CK work to tune, so do not skip.
        return not any(s.startswith("2stage") for s in stages) and any(s.startswith("1stage") for s in stages)
    # Nothing structured to read (older log format, unreadable file): fall back
    # to the substring probe so behaviour never regresses to "always tune".
    path = Path(log_path) if log_path else None
    if path is None or not path.is_file():
        return False
    try:
        return bool(_1STAGE_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


# Normalize non-canonical quant-type spellings callers may pass (e.g. a runtime
# --quantization value or a tuner name) to the router's canonical vocabulary.
_QUANT_TYPE_ALIASES: dict[str, str] = {
    "w8a8_fp8": "per_token",
    "fp8_w8a8": "per_token",
    "w8a8": "per_token",
    "a8w8": "per_token",
    "per_tensor": "per_token",
    "a8w8_blockscale": "blockscale",
    "per_1x128": "blockscale",
    "block": "blockscale",
    # Hyperloom's untuned-CSV quant keys, kept in sync so its vocabulary resolves here.
    "block_scale": "blockscale",
    "fp8_blockscale": "blockscale",
    "a8w8_bpreshuffle": "bpreshuffle",
    "a8w8_blockscale_bpreshuffle": "blockscale_bpreshuffle",
    "blockscale+bpreshuffle": "blockscale_bpreshuffle",
    "a4w4_blockscale": "fp4",
    "a4w4": "fp4",
}


def _normalize_quant_type(quant_type_arg: str) -> str:
    """Map a caller-supplied quant_type onto the router's canonical vocabulary."""
    qt = (quant_type_arg or "").strip().lower()
    return _QUANT_TYPE_ALIASES.get(qt, qt)


def _profile_can_derive_dense(profile: ModelProfile) -> bool:
    """True when the config carries enough dims to derive dense GEMM shapes."""
    return int(getattr(profile, "hidden_size", 0) or 0) >= 1 and int(getattr(profile, "intermediate_size", 0) or 0) >= 1


def _resolve_quant_type(
    precision: str,
    quant_type_arg: str,
    profile: ModelProfile,
    kernel_signature_log: str | None,
) -> str:
    """Resolve the effective quant type from CLI args, model config, or log.

    Returns one of: none, per_token, blockscale, bpreshuffle,
    blockscale_bpreshuffle, awq, gptq, fp4, mxfp4, auto.
    (per_tensor is accepted as input but normalized to per_token, never returned.)
    """
    if quant_type_arg and quant_type_arg != "auto":
        return _normalize_quant_type(quant_type_arg)

    # Infer from model config
    if profile.quant_method == "awq":
        return "awq"
    if profile.quant_method == "gptq":
        return "gptq"

    # For fp8, try to detect from log or default to blockscale
    if precision == "fp8":
        if kernel_signature_log:
            path = Path(kernel_signature_log)
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                lowered = text.lower()
                if "a8w8_blockscale_bpreshuffle" in lowered or "blockscale_bpreshuffle" in lowered:
                    return "blockscale_bpreshuffle"
                if "QuantType.per_Token" in text:
                    return "per_token"
                if "QuantType.per_1x128" in text or "blockscale" in lowered:
                    return "blockscale"
                if "bpreshuffle" in lowered:
                    return "bpreshuffle"
        # Default for fp8 without further info
        return "blockscale"

    if precision in ("fp4", "mxfp4"):
        return "fp4"

    if precision in ("bf16", "fp16"):
        return "none"

    return "none"


def select_tuners(
    profile: ModelProfile,
    *,
    framework: str,
    precision: str,
    quant_type: str = "auto",
    gpu_type: str = "auto",
    kernel_signature_log: str | None = None,
    has_untuned_csv: bool = False,
    has_shapes_json: bool = False,
    has_tunableop_input: bool = False,
    demand_report: dict[str, Any] | None = None,
) -> list[TunerSpec]:
    """Select which tuner(s) to run based on model + framework + precision.

    Args:
        profile: Analyzed model profile.
        framework: "sglang" or "vllm".
        precision: "bf16", "fp8", "fp4", "int8", "awq", etc.
        quant_type: Explicit quant type or "auto" for inference.
        gpu_type: Target GPU type ("mi300x", "mi355x", ...). Used to gate
            arch-specific tuners (e.g. FP4 GEMM is gfx950-only).
        kernel_signature_log: Optional server log to detect 1-stage ASM.
        has_untuned_csv: Whether --untuned-csv was provided.
        has_shapes_json: Whether --shapes-json was provided.
        has_tunableop_input: Whether --tunableop-input was provided.
        demand_report: Parsed demand.json from the serving run, when one was
            recorded. Names the tables the runtime actually consulted, and
            widens the framework branch when the runtime names another tuner.

    Returns:
        List of TunerSpec in execution order.
    """
    resolved_qt = _resolve_quant_type(precision, quant_type, profile, kernel_signature_log)
    gfx_arch = _resolve_gfx_arch(gpu_type)
    tuners: list[TunerSpec] = []

    if framework in ("sglang", "vllm-aiter"):
        tuners.extend(
            _select_sglang_tuners(
                profile,
                precision,
                resolved_qt,
                kernel_signature_log,
                has_untuned_csv,
                has_shapes_json,
                gfx_arch,
            )
        )
    elif framework == "vllm":
        tuners.extend(
            _select_vllm_tuners(
                profile,
                precision,
                resolved_qt,
                has_shapes_json,
                has_tunableop_input,
            )
        )
        tuners.extend(
            _moe_tuners_the_log_says_are_needed(
                kernel_signature_log,
                tuners,
                profile,
            )
        )
    else:
        log.warning("Unknown framework %r; no tuners selected", framework)

    # Last, so it sees everything the framework branch decided and can only
    # widen it.
    tuners.extend(_tuners_the_demand_says_are_needed(demand_report, tuners))

    # Sort by priority
    tuners.sort(key=lambda t: t.priority)
    return tuners


def _moe_tuners_the_log_says_are_needed(
    kernel_signature_log: str | None,
    already: list[TunerSpec],
    profile: ModelProfile,
) -> list[TunerSpec]:
    """Add CK MoE tuning when a 2-stage key actually missed in the log.

    A vLLM run is routed by framework alone, which assumes the MoE is served by
    vLLM's Triton path. It is not always: aiter's CK fused-MoE can serve some or
    all of the token range in the same process, and its table is written by
    ``fmoe_ck``, not by ``vllm_moe_triton``. When both appear in one log the
    answer is not to pick a side -- each serves the range it serves, and dropping
    either forfeits that range. This is the same "one boolean cannot describe a
    mixed runtime" mistake that made a single 1-stage sighting disable CK tuning
    for the token counts 2-stage was actually serving.

    Only ever adds. Selection stays with the framework branch; this is the log
    saying that branch's assumption did not hold for the whole run.
    """
    if not profile.is_moe or not kernel_signature_log:
        return []
    moe = moe_stage_coverage(kernel_signature_log) or {}
    if not moe.get("tunable_ck_2stage") or not moe.get("missed_ck_keys"):
        return []
    if any(t.name == "fmoe_ck" for t in already):
        return []
    # Only the tokens CK actually served. The rest of the range is Triton's, and
    # a CK table keyed on those token counts is one nothing ever reads.
    by_stage = moe.get("tokens_by_stage") or {}
    ck_tokens = sorted(
        {int(tok) for stage, tokens in by_stage.items() if stage.startswith("2stage") for tok in (tokens or [])}
    )
    log.info(
        "Serving log shows missed aiter CK 2-stage MoE "
        "(stages=%s, tokens=%s) on a vLLM run; adding fmoe_ck, which owns "
        "the table that path reads",
        moe.get("stages_seen"),
        by_stage,
    )
    return [
        TunerSpec(
            "fmoe_ck",
            priority=10,
            estimated_minutes=15,
            token_hint=ck_tokens or None,
        )
    ]


def _tuners_the_demand_says_are_needed(
    demand_report: dict[str, Any] | None,
    already: list[TunerSpec],
) -> list[TunerSpec]:
    """Add the tuners that own tables the serving run actually looked up.

    Every other input to selection is an inference about what the runtime will
    do -- a precision label, a framework name, a config field. A demand report
    is not: ``AITER_LOG_TUNED_CONFIG`` makes the serving process name the tables
    it consulted and the keys it asked for, so ``demands[].tuner`` is the
    runtime's own answer to the question the router is guessing at.

    Where the two disagree, the log wins, for the same reason it wins on the MoE
    side (:func:`_moe_tuners_the_log_says_are_needed`): a table with thousands
    of recorded misses is being read, whatever the precision label implies.

    Only ever adds. Selection stays with the framework branch; this is the run
    saying that branch left out a table it spends its time in. A tuner already
    present keeps its spec -- including a skip_reason, which is a capability
    statement (wrong arch, unsupported combo) that demand does not overturn.
    """
    demands = (demand_report or {}).get("demands") or []
    if not demands:
        return []
    have = {t.name for t in already}
    added: list[TunerSpec] = []
    for entry in demands:
        name = str(entry.get("tuner") or "")
        # A demand with no registered owner is a coverage gap, not a selection:
        # tier3 handles those, and inventing a TunerSpec here would shadow it.
        if not name or name in have:
            continue
        have.add(name)
        added.append(
            TunerSpec(
                name,
                priority=10 if name == "fmoe_ck" else 20,
                estimated_minutes=15 if name == "fmoe_ck" else 20,
            )
        )
        log.info(
            "Serving log consulted %s (%s misses over %s distinct keys) but the "
            "router did not select %s, which owns it; adding it",
            entry.get("table"),
            entry.get("miss_count"),
            entry.get("distinct_keys"),
            name,
        )
    return added


def _select_sglang_tuners(
    profile: ModelProfile,
    precision: str,
    quant_type: str,
    kernel_signature_log: str | None,
    has_untuned_csv: bool,
    has_shapes_json: bool,
    gfx_arch: str = "",
) -> list[TunerSpec]:
    """Select tuners for sglang framework."""
    tuners: list[TunerSpec] = []
    fp4_unsupported = _fp4_unsupported_on(gfx_arch)

    # --- MoE tuning ---
    if profile.is_moe:
        if quant_type == "per_token":
            # 1-stage ASM already optimal (validated in experiments)
            is_1stage = _detect_1stage_from_log(kernel_signature_log)
            if is_1stage or precision == "fp8":
                tuners.append(
                    TunerSpec(
                        "fmoe_ck",
                        skip_reason=(
                            "FP8 per_Token MoE uses 1-stage ASM kernels that are "
                            "already at peak performance. CK 2-stage tuning cannot "
                            "improve and may fail correctness checks."
                        ),
                        priority=10,
                        estimated_minutes=0,
                    )
                )
            else:
                tuners.append(TunerSpec("fmoe_ck", priority=10, estimated_minutes=15))
        elif precision in ("bf16", "fp16") and quant_type == "none":
            tuners.append(TunerSpec("fmoe_ck", priority=10, estimated_minutes=15))
        elif precision in ("fp4", "mxfp4") or quant_type in ("fp4", "mxfp4"):
            if fp4_unsupported:
                tuners.append(
                    TunerSpec(
                        "fmoe_ck",
                        skip_reason=_FP4_GFX942_SKIP_REASON,
                        priority=10,
                        estimated_minutes=0,
                    )
                )
            else:
                tuners.append(TunerSpec("fmoe_ck", priority=10, estimated_minutes=15))
        elif precision == "fp8" and quant_type in (
            "blockscale",
            "bpreshuffle",
            "blockscale_bpreshuffle",
        ):
            tuners.append(TunerSpec("fmoe_ck", priority=10, estimated_minutes=15))
        else:
            tuners.append(
                TunerSpec(
                    "fmoe_ck",
                    skip_reason=f"Unsupported MoE precision/quant combo: {precision}/{quant_type}",
                    priority=10,
                    estimated_minutes=0,
                )
            )

    # --- Dense GEMM tuning ---
    # Dense fp8/fp4 tuners no longer require an externally-recorded CSV: when
    # none is supplied they derive GEMM shapes from the model config (same as
    # the bf16 dense path). A real --untuned-csv / --shapes-json is still
    # preferred when available because recorded shapes are more accurate. Only
    # when NO shape source is obtainable at all (no csv/shapes AND a config
    # without hidden_size+intermediate_size) do we skip gracefully instead of
    # surfacing a hard validation failure.
    def _dense_spec(name: str) -> TunerSpec:
        if has_untuned_csv or has_shapes_json or _profile_can_derive_dense(profile):
            return TunerSpec(name, priority=20, estimated_minutes=20)
        return TunerSpec(
            name,
            skip_reason=(
                "No GEMM shapes available: needs --untuned-csv/--shapes-json or a "
                "model config with hidden_size and intermediate_size."
            ),
            priority=20,
            estimated_minutes=0,
        )

    if precision == "fp8":
        if quant_type == "blockscale":
            tuners.append(_dense_spec("a8w8_blockscale"))
        elif quant_type == "per_token":
            tuners.append(_dense_spec("a8w8"))
        elif quant_type == "bpreshuffle":
            # Per-token bpreshuffle serves via aiter's gemm_a8w8_bpreshuffle op,
            # which reads AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE. The dedicated
            # a8w8_bpreshuffle tuner writes exactly that config table, so the
            # tuned result is picked up at serving time.
            if gfx_arch == "gfx950":
                # On gfx950 the CK a8w8_bpreshuffle tuner crashes on the
                # FNUZ/OCP fp8 dtype mismatch (gfx950 fp8 is e4m3fn/OCP). The
                # blockscale+bpreshuffle tuner *runs* but writes a DIFFERENT
                # table (AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE) that the
                # per-token bpreshuffle serving op never reads — tuning it is
                # silently ineffective. Skip honestly rather than fake success.
                tuners.append(
                    TunerSpec(
                        "a8w8_bpreshuffle",
                        skip_reason=(
                            "Per-token bpreshuffle GEMM tuning is unavailable on "
                            "gfx950: the CK a8w8_bpreshuffle tuner fails on the "
                            "FNUZ/OCP fp8 dtype mismatch, and the "
                            "blockscale+bpreshuffle tuner writes a config table the "
                            "per-token bpreshuffle serving op does not read."
                        ),
                        priority=20,
                        estimated_minutes=0,
                    )
                )
            else:
                tuners.append(_dense_spec("a8w8_bpreshuffle"))
        elif quant_type == "blockscale_bpreshuffle":
            tuners.append(_dense_spec("a8w8_blockscale_bpreshuffle"))
    elif precision in ("fp4", "mxfp4"):
        if fp4_unsupported:
            tuners.append(
                TunerSpec(
                    "a4w4_blockscale",
                    skip_reason=_FP4_GFX942_SKIP_REASON,
                    priority=20,
                    estimated_minutes=0,
                )
            )
        else:
            tuners.append(_dense_spec("a4w4_blockscale"))

    # Deliberately not an ``elif``: bf16 dense is not the alternative to
    # quantized dense, it runs alongside it. See _dense_bf16_is_dispatched.
    # Shapes are computed from config.json (no --untuned-csv needed).
    if _dense_bf16_is_dispatched(profile, precision, quant_type):
        tuners.append(
            TunerSpec(
                "sglang_dense_bf16",
                priority=20,
                estimated_minutes=10,
            )
        )
    elif precision == "fp8" and not any(t.name == "sglang_dense_bf16" for t in tuners):
        # fp8 -> bf16 dense retry, pushed down from Hyperloom's old second
        # subprocess. When the model config gives no positive signal that bf16
        # dense GEMMs are dispatched (fully-quantized checkpoint, or lm_head-only
        # exclusion), the fp8 dense tuner may still come back empty -- and on
        # such a run the excluded projections do run in bf16. Rather than launch
        # a fresh gemm session for that case, select the bf16 dense pass now as a
        # fallback: it runs in this same call only if every fp8 tuner produced
        # no candidate, so the "fp8 won" path pays nothing for it.
        tuners.append(
            TunerSpec(
                "sglang_dense_bf16",
                priority=30,
                estimated_minutes=10,
                fallback=True,
            )
        )

    return tuners


def _dense_bf16_is_dispatched(
    profile: ModelProfile,
    precision: str,
    quant_type: str,
) -> bool:
    """Whether this run issues bf16/fp16 dense GEMMs worth tuning.

    ``precision`` describes the format of the quantized weights, which is a
    different question from which dense GEMM operators get dispatched, and on a
    quantized MoE model the two answers disagree almost completely. The experts
    carry ~99% of the weight bytes and are served by the fused MoE kernel, not
    by a dense GEMM at all; the attention projections and lm_head carry ~1% and
    are excluded from quantization, so they are essentially the entire dense
    GEMM traffic -- in bf16, against ``bf16_tuned_gemm.csv``.

    An exclusion containing only ``lm_head`` is not enough evidence: on a
    dense-only quantized model that is one GEMM per forward, while the quantized
    projections dominate the workload. Actual bf16 misses still override this
    heuristic through the demand report.

    Reading the scalar as if it partitioned the operator set is what left that
    table untuned while a4w4 dense, an operator the model never dispatches, was
    tuned instead.
    """
    if profile.keeps_dense_layers_at_model_dtype:
        # A quantized checkpoint with substantial excluded linear layers still
        # runs those modules at the model dtype. lm_head alone is one GEMM per
        # forward and does not justify competing with the quantized dense tuner
        # for the shared budget; an observed bf16 miss can still add the tuner
        # through demand_report.
        return profile.model_dtype.lower() in ("bfloat16", "bf16", "float16", "fp16")
    return precision in ("bf16", "fp16") and quant_type == "none"


def _select_vllm_tuners(
    profile: ModelProfile,
    precision: str,
    quant_type: str,
    has_shapes_json: bool,
    has_tunableop_input: bool,
) -> list[TunerSpec]:
    """Select tuners for vLLM framework."""
    tuners: list[TunerSpec] = []

    if profile.is_moe:
        tuners.append(TunerSpec("vllm_moe_triton", priority=10, estimated_minutes=30))

    # Dense GEMM via TunableOp
    if has_tunableop_input or has_shapes_json:
        tuners.append(TunerSpec("vllm_dense_tunableop", priority=20, estimated_minutes=45))
    elif not profile.is_moe:
        # Dense-only model without shape input
        tuners.append(
            TunerSpec(
                "vllm_dense_tunableop",
                skip_reason=(
                    "vLLM dense TunableOp requires --tunableop-input or --shapes-json "
                    "from actual GEMM shape recording (PYTORCH_TUNABLEOP_RECORD_UNTUNED=1). "
                    "Cannot reliably infer all shapes from config.json alone."
                ),
                priority=20,
                estimated_minutes=0,
            )
        )

    return tuners
