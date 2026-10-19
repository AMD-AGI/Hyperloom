# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-model work-arounds a launch needs on the ROCm images.

A handful of flagship models SIGABRT during CUDA-graph capture on the sglang
ROCm image because their DEFAULT fused kernels are buggy on gfx942. Each
work-around is keyed on the model name, applied at the single choke point every
benchmark path funnels through, and merged rather than overwritten so a caller's
own pin always wins.
"""

from __future__ import annotations

from typing import Any

from ._server_argv import add_server_arg_unless_pinned


def _pin_mimo_launch(envs: dict[str, str], bench: dict[str, Any]) -> None:
    """Keep MiMo-V2.x off the attention backend and the arch name that abort its boot.

    Its DEFAULT aiter attention backend SIGABRTs during CUDA-graph capture on
    gfx942, so triton is pinned instead; sglang accepts lowercase ``triton``,
    vLLM only knows ``TRITON_ATTN``. vLLM also registers this checkpoint under
    MiMoV2FlashForCausalLM while the HF config declares MiMoV2ForCausalLM, which
    the pod-local build rejects at boot, so the arch is remapped via
    ``--hf-overrides``; the JSON is kept space-free so it survives Magpie's
    unquoted splice.
    """
    is_vllm = "vllm" in str(bench.get("framework") or "").lower()
    attention_backend = "TRITON_ATTN" if is_vllm else "triton"
    add_server_arg_unless_pinned(
        envs,
        bench.get("framework"),
        f"--attention-backend {attention_backend}",
        pinned_by=("attention-backend",),
    )
    if is_vllm:
        add_server_arg_unless_pinned(
            envs,
            bench.get("framework"),
            '--hf-overrides {"architectures":["MiMoV2FlashForCausalLM"]}',
            pinned_by=("hf-overrides", "hf_overrides"),
        )


def pin_model_launch_workarounds(envs: dict[str, str], bench: dict[str, Any], model_basename: str) -> None:
    """Inject the work-arounds the named model needs in order to boot."""
    if "kimi-k2" in model_basename:
        # Kimi K2.x at tp8 takes sglang's ROCm fused-decode-MLA path, whose RoPE
        # kernel aborts during CUDA-graph capture. Disabling the fused decode
        # pipeline keeps tp8 + the clean aiter MLA path.
        envs.setdefault("SGLANG_ROCM_FUSED_DECODE_MLA", "0")
    if "mimo-v2" in model_basename:
        _pin_mimo_launch(envs, bench)
