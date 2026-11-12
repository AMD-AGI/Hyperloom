# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""Static KV-cache scale overrides for fp8 KV on Windows ROCm.

Checkpoints that ship no k_scale/v_scale (e.g. Qwen2.5 GPTQ) run with
scales of 1.0. That is usually fine, but measured amplitudes on
Qwen2.5-7B-Instruct-GPTQ-Int4 sit close to the fp8 e4m3 ceiling (max|V|
reaches 419-426 at 8-16k context vs 448 max), so long-context prefill can
saturate outliers and derail generation.

This module applies operator-provided STATIC scales to every attention
layer after weight loading:

    VLLM_WIN_KV_KSCALE / VLLM_WIN_KV_VSCALE   (float, default: unset)

When set, the plugin wraps each layer's post-load step to overwrite
_k_scale / _v_scale (and their _float mirrors). vLLM then divides K/V by
these scales on cache writes and multiplies them back on attention reads,
exactly like checkpoint-provided scales.

Use `run/kv_scale_probe.py` to measure amplitudes on your model, then set
scale >= max_amplitude / 448 with ~10% headroom (e.g. for max|V|=426:
v_scale >= 1.05). Scales BELOW 1.0 push amplitudes toward the ceiling and
make saturation MORE likely, not less.

All diagnostics go through logging (never print): this code runs inside
the engine process whose stdout must stay machine-readable (issue #17).
The hook is installed from the platform's check_and_update_config, i.e. in
the process that builds the VllmConfig, so it needs the engine in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0, which every run/ script and localserve
already set).
"""

import logging
import os

logger = logging.getLogger(__name__)


def apply_static_kv_scales() -> None:
    """Install a post-load hook that overrides _k_scale/_v_scale from env."""
    k_str = os.environ.get("VLLM_WIN_KV_KSCALE", "")
    v_str = os.environ.get("VLLM_WIN_KV_VSCALE", "")
    if not k_str and not v_str:
        return
    try:
        k_scale = float(k_str) if k_str else 1.0
        v_scale = float(v_str) if v_str else 1.0
    except ValueError as e:
        logger.warning("vllm-win: invalid VLLM_WIN_KV_*SCALE value: %r", e)
        return

    # Import here: called from check_and_update_config, vLLM is fully loaded.
    from vllm.model_executor.layers.attention.attention import Attention

    orig = Attention.process_weights_after_loading
    if getattr(orig, "_kv_scales_wrapped", False):
        return  # check_and_update_config can run more than once per process

    def patched(self, act_dtype) -> None:
        orig(self, act_dtype)
        if getattr(self, "_k_scale", None) is not None:
            # static override: write both the device tensor and the host
            # float mirror the attention impls read
            self._k_scale.fill_(k_scale)
            self._v_scale.fill_(v_scale)
            self._k_scale_float = k_scale
            self._v_scale_float = v_scale
            logger.debug("static KV scales applied on %s", self.layer_name)

    patched._kv_scales_wrapped = True
    Attention.process_weights_after_loading = patched
    logger.info(
        "vllm-win: static KV scale override installed (k=%s v=%s)",
        k_scale, v_scale,
    )


def register() -> None:
    try:
        apply_static_kv_scales()
    except Exception as e:  # noqa: BLE001
        logger.warning("vllm-win: static KV scales not installed: %r", e)
