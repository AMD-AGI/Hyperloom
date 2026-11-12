# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""Per-layer K offset removal for fp8 KV caches (issue #28).

Qwen2.5's K carries a large per-(layer, kv head, channel) offset: the k_proj
bias survives RoPE almost untouched on the lowest-frequency rotary channels
(channels 58-63 / 123-127 of kv heads 0-1; layer 27 peaks at |K| ~ 423).
fp8 has RELATIVE precision, so a value of ~420 is stored with an absolute
error of up to 16 whatever the scale, and that turns into position-dependent
logit noise of several units. No rescaling fixes it (measured: per-tensor,
per-channel, per-token, e5m2 and Hadamard rotation all stay above 0.7 max-rel
on the outlier layers).

The fix is store-side only: subtract a calibrated [num_kv_heads, head_size]
mean from K before the fp8 store and never add it back -- q·mean is the same
for every key of a given query, so it cancels in the softmax. The per-layer
k_scale is set to max|K - mean| / 448 (the residual the cache actually
holds). Measured effect on the outlier layers: max-rel 0.9-1.0 -> ~0.06.

    VLLM_WIN_KV_OFFSETS=<file.pt>   offsets file produced by calibration:
        python run/kv_scale_probe.py  with VLLM_KV_PROBE_MEAN=<file.pt>
        content: {"model": str, "offsets": {layer: [Hk, D] fp32},
                  "k_scales": {layer: float},
                  ["pc_scales": {layer: [Hk, D] fp32}]}

Do NOT combine with VLLM_WIN_KV_KSCALE: both overwrite _k_scale, the offsets
wrapper wins by registration order but the combination is unsupported.

Notes:
- Applies only where it matters: the store wrapper checks the layer's kv
  cache dtype is fp8, so an fp16 run with the env var set is a harmless no-op.
- Layers missing from the file run plain fp8 (warned once). A shape mismatch
  between the file and the model raises during weight loading: a wrong
  model's offsets would otherwise fail obscurely mid-attention.
- Without "pc_scales" (option 1) no kernel change is involved: the read side
  multiplies k_scale back (which returns the residual) and the softmax
  cancels the constant q·mean by itself.
- With "pc_scales" (option 2), each channel of the residual is normalized to
  its own fp8 full range: the store divides by a per-(kv head, channel)
  scale and the fp8 branches of the unified-attention / reshape-and-cache
  Triton kernels multiply it back (constexpr-gated, no-op when unset).
  The per-tensor _k_scale then stays 1.0 -- the per-channel scale replaces
  it, exactly like the issue's "mean removed + per-(head,channel) scale"
  scheme.

All diagnostics go through logging (never print): this code runs inside the
engine process whose stdout must stay machine-readable (issue #17). Like
kv_scales, this needs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0,
which every run/ script and localserve already set).
"""

import logging
import os
import re

logger = logging.getLogger(__name__)


def apply_kv_offsets() -> None:
    """Install post-load + store hooks that subtract calibrated K offsets."""
    path = os.environ.get("VLLM_WIN_KV_OFFSETS", "")
    if not path:
        return
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    offsets = data["offsets"]
    k_scales = data.get("k_scales", {})
    pc_scales = data.get("pc_scales", {})

    # Import here: called from check_and_update_config, vLLM is fully loaded.
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    orig_pw = Attention.process_weights_after_loading
    if getattr(orig_pw, "_kv_offsets_wrapped", False):
        return  # check_and_update_config can run more than once per process
    missing: set = set()

    def patched_pw(self, act_dtype) -> None:
        orig_pw(self, act_dtype)
        m = re.search(r"\.(\d+)\.", str(getattr(self, "layer_name", "")))
        li = int(m.group(1)) if m else -1
        off = offsets.get(li)
        if off is None:
            if li >= 0 and li not in missing:
                missing.add(li)
                logger.warning("vllm-win: no K offset for layer %d in %s; "
                               "plain fp8 store", li, path)
            return
        if tuple(off.shape) != (self.num_kv_heads, self.head_size):
            raise ValueError(
                f"vllm-win: K offset shape {tuple(off.shape)} != "
                f"({self.num_kv_heads}, {self.head_size}) for layer {li} "
                f"in {path} -- wrong model's offsets file?")
        self._kv_k_offset = off.to(device=self._k_scale.device,
                                   dtype=act_dtype)
        ks = float(k_scales.get(li, 0.0))
        if ks > 0.0 and getattr(self, "_k_scale", None) is not None:
            self._k_scale.fill_(ks)
            self._k_scale_float = ks
            logger.debug("K offset + k_scale %.5f applied on %s",
                         ks, self.layer_name)
        pc = pc_scales.get(li)
        if pc is not None:
            if tuple(pc.shape) != (self.num_kv_heads, self.head_size):
                raise ValueError(
                    f"vllm-win: per-channel K scale shape {tuple(pc.shape)} "
                    f"!= ({self.num_kv_heads}, {self.head_size}) for layer "
                    f"{li} in {path} -- wrong model's offsets file?")
            # option 2: the per-channel residual scale REPLACES the per-tensor
            # k_scale (store divides by pc, fp8 read branches multiply it back)
            self._kv_k_pc_scale = pc.to(self._k_scale.device)
            self._k_scale.fill_(1.0)
            self._k_scale_float = 1.0
            logger.debug("per-channel K scale applied on %s", self.layer_name)

    patched_pw._kv_offsets_wrapped = True
    Attention.process_weights_after_loading = patched_pw

    orig_upd = TritonAttentionImpl.do_kv_cache_update

    def patched_upd(self_impl, layer, key, value, kv_cache, slot_mapping):
        off = getattr(layer, "_kv_k_offset", None)
        if off is not None and self_impl.kv_cache_dtype.startswith("fp8"):
            # out-of-place: unified_kv_cache_update declares mutates_args=[]
            key = key - off
        return orig_upd(self_impl, layer, key, value, kv_cache, slot_mapping)

    TritonAttentionImpl.do_kv_cache_update = patched_upd
    logger.info(
        "vllm-win: K offsets installed from %s (%d layers, %d with "
        "per-channel scale, model %r)",
        path, len(offsets), len(pc_scales), data.get("model", "?"),
    )


def register() -> None:
    try:
        apply_kv_offsets()
    except Exception as e:  # noqa: BLE001
        logger.warning("vllm-win: K offsets not installed: %r", e)
