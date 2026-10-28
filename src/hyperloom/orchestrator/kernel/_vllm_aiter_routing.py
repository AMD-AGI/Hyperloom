# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Routing a vLLM run onto forge's aiter tuner family, from what the serving log shows."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ._aiter_tuner_support import _aiter_ck_moe_tuner_supports
from .serving_log_evidence import _aiter_serving_evidence
from ._shape_capture import _is_vllm_block_fp8

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VllmAiterRouting:
    """Which aiter kernel families a vLLM run's serving log routes to forge's aiter tuners."""

    bf16_dense: bool = False
    fused_moe: bool = False


def _forge_framework_for_vllm(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    tunableop_input: str,
    aiter_routing: VllmAiterRouting = VllmAiterRouting(),
) -> str:
    """Route vLLM runs served by aiter to Forge's AITER tuner family.

    Forge's vLLM branch only offers ``vllm_moe_triton`` and
    ``vllm_dense_tunableop``, which target kernels an aiter-served model never
    executes. Its sglang branch carries the tuners that do write the tables aiter
    reads, and the router already accepts ``vllm-aiter`` as an alias for it.
    """
    if framework != "vllm" or tunableop_input:
        return framework
    if _is_vllm_block_fp8(precision, quant_type):
        return "vllm-aiter"
    if aiter_routing.bf16_dense and precision in ("bf16", "fp16"):
        return "vllm-aiter"
    if aiter_routing.fused_moe:
        return "vllm-aiter"
    return framework


def _resolve_vllm_aiter_routing(
    *,
    model_path: str,
    server_log: str,
    tp: int,
) -> VllmAiterRouting:
    """Resolve the aiter-routing flags for a vLLM run from runtime evidence."""
    evidence = _aiter_serving_evidence(server_log)
    if not evidence:
        return VllmAiterRouting()

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path) or {}
    if not summary:
        return VllmAiterRouting()
    is_moe = bool(summary.get("is_moe"))

    # Dense BF16 routing is for dense checkpoints; a MoE model's dense side
    # rides along with its MoE routing instead.
    bf16_dense = "bf16_dense" in evidence and not is_moe

    fused_moe = False
    if "fused_moe" in evidence and is_moe and _aiter_ck_moe_tuner_supports(server_log):
        # Only route MoE when aiter's CK fused-MoE can actually serve this
        # checkpoint at this TP -- otherwise the tuner has no reachable target.
        from hyperloom.inference_optimizer.model_config_utils import (
            model_supports_aiter_ck_fused_moe,
        )

        fused_moe = model_supports_aiter_ck_fused_moe(model_path, tp)

    routing = VllmAiterRouting(bf16_dense=bf16_dense, fused_moe=fused_moe)
    _warn_if_moe_routing_is_coarser_than_the_log(server_log, routing)
    return routing


def _warn_if_moe_routing_is_coarser_than_the_log(server_log: str, routing: VllmAiterRouting) -> None:
    """Say so when one log shows both MoE backends and routing picks one.

    The decision above is a substring scan: seeing an aiter fused-MoE marker
    anywhere routes the whole run to the aiter tuner family, and
    ``vllm_moe_triton`` then never runs. A run can dispatch both -- aiter CK over
    part of the token range and vLLM's Triton path over the rest -- and forge's
    own parser records exactly that as ``impl="mixed"``. Whichever way the single
    flag falls, the range served by the other backend is left untuned.

    Reported rather than acted on here: changing this routing changes which
    tuners run for every aiter-served vLLM model, which is a bigger step than
    the tuner-side addition that already covers the CK half. Forge adds
    ``fmoe_ck`` from the same evidence, so the gap this warns about is the
    Triton half.
    """
    if not routing.fused_moe:
        return
    try:
        from kernelforge.gemm_tune import parse_log_file
    except ImportError:
        # Same reasoning as apply_verification._parse: kernelforge is in this
        # wheel, so a miss is a broken install, and a bare return makes the
        # missing routing warning indistinguishable from a clean run.
        log.warning(
            "kernelforge.gemm_tune is not importable, so the aiter/vLLM MoE "
            "routing check is skipped -- it ships with Hyperloom, so this means "
            'an incomplete install; reinstall with pip install -e ".[forge]"'
        )
        return
    moe = (parse_log_file(server_log).get("dispatch") or {}).get("moe") or {}
    if moe.get("impl") == "mixed" or moe.get("vllm_config_hit"):
        log.warning(
            "gemm routing: %s shows both aiter CK and vLLM Triton MoE dispatch "
            "(impl=%s, stages=%s); routing sends the whole run to the aiter "
            "tuner family, so the token range Triton serves goes untuned",
            server_log,
            moe.get("impl"),
            moe.get("stages_seen"),
        )
