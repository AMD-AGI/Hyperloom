# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""Measure real Q/K/V amplitudes per layer at several context lengths.

Purpose: see what an fp8 (or any low-bit) KV cache has to represent on a model whose
checkpoint ships no scales (Qwen2.5-7B-Instruct-GPTQ-Int4). For each attention layer we
run one forward with a filler prompt of N tokens and record max|Q|, max|K|, max|V|, then
compare K/V against the fp8 e4m3 representable range (448).

Modes:
  VLLM_KV_PROBE_PER_CHANNEL=1   also record max|K| per (kv head, channel), accumulated over
                                layers and positions: median / p95 / count above 100 / top
                                pairs. On Qwen2.5 a handful of channels carry |K| ~ 420 while
                                the median sits around 3 (the k_proj bias, rotated by RoPE).
  VLLM_KV_PROBE_MEAN=<path>     calibrate per-layer K offsets for the fp8 KV fix (issue #28):
                                two sweeps of the context list; sweep 1 accumulates the
                                per-(layer, kv head, channel) mean of K, sweep 2 measures
                                max|K - mean| (the residual the cache actually stores).
                                Saves {"model", "offsets": {layer: [Hk, D]}, "k_scales":
                                {layer: max|residual|/448}} for VLLM_WIN_KV_OFFSETS. The mean
                                is subtracted at store time and never added back: q·mean is
                                constant per query, so it cancels in the softmax (issue #28).
  VLLM_KV_PROBE_DUMP=<path>     save q/k/v (fp16, CPU) of one layer at the LAST context length
                                for offline quantization experiments; the layer index is
                                VLLM_KV_PROBE_DUMP_LAYER (default 0). The tensors are what the
                                attention backend receives for that step, i.e. after RoPE and
                                before any cache quantization; keep that context length at or
                                below max_num_batched_tokens (8192 by default) or only the
                                last prefill chunk is captured.

Usage:
  python kv_scale_probe.py            # uses VLLM_KV_CTXS / VLLM_KV_MODEL / VLLM_KV_UTIL /
                                      # VLLM_KV_TRUST / VLLM_KV_PROBE_OUT
"""

import json
import math
import os
import re
import tempfile

# MUST be set before importing vllm: keep the engine in-process so the
# module-level attention hooks below intercept the real forward path.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm import LLM, SamplingParams as sp

MODEL = os.environ.get("VLLM_KV_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
CTXS = [int(c) for c in os.environ.get("VLLM_KV_CTXS", "128,2048,8192,16384").split(",")]
OUT = os.environ.get("VLLM_KV_PROBE_OUT", os.path.join(tempfile.gettempdir(), "kv_scale_probe.json"))
UTIL = float(os.environ.get("VLLM_KV_UTIL", "0.6"))
TRUST = os.environ.get("VLLM_KV_TRUST", "0") == "1"
PER_CHANNEL = os.environ.get("VLLM_KV_PROBE_PER_CHANNEL", "0") == "1"
MEAN = os.environ.get("VLLM_KV_PROBE_MEAN", "")
DUMP = os.environ.get("VLLM_KV_PROBE_DUMP", "")
DUMP_LAYER = int(os.environ.get("VLLM_KV_PROBE_DUMP_LAYER", "0"))
FP8_MAX = 448.0

FILLER = (
    "The history of GPU computing begins in the late 1990s when fixed function "
    "graphics pipelines gave way to programmable shaders. Researchers noticed "
    "that vertex and pixel shaders could be tricked into running general "
    "arithmetic, and the GPGPU movement was born. In 2006 CUDA exposed a C "
    "language interface to the chip, and deep learning arrived a decade later "
    "to consume every flop the industry could produce. "
)
QUESTION = "\n\nSummarize the text above."


def _layer_index(layer, impl) -> int:
    name = getattr(layer, "layer_name", None) or getattr(impl, "layer_name", "") or ""
    m = re.search(r"\.(\d+)\.", str(name))
    return int(m.group(1)) if m else -1


def main() -> None:
    llm = LLM(
        model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
        tensor_parallel_size=1, gpu_memory_utilization=UTIL,
        max_model_len=max(CTXS) + 256, kv_cache_dtype="auto",
        enable_prefix_caching=False, enforce_eager=True, trust_remote_code=TRUST,
    )
    tok = llm.get_tokenizer()

    # Hook the backend impl itself: TritonAttentionImpl.forward(self, layer, query, key, value,
    # kv_cache, attn_metadata, ...) receives the raw q/k/v for every layer, regardless of how the
    # layer dispatches (direct call vs opaque custom op). The `layer` argument comes first: an
    # earlier version of this probe bound (query, key, value) one slot too early and reported
    # max|Q| as "K" and max|K| as "V" (issue #25).
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    orig_forward = TritonAttentionImpl.forward
    stats: dict = {}          # layer index -> [max|Q|, max|K|, max|V|]
    chan_max: list = [None]   # [Hk, D] running max of |K| over layers and positions
    dump: dict = {}
    # MEAN calibration (issue #28): pass 1 sums K per (layer, head, channel),
    # pass 2 measures the residual max|K - mean| the cache will actually store.
    sums: dict = {}
    cnts: dict = {}
    mus: dict = {}
    res_max: dict = {}
    mean_pass = [0]

    def patched_forward(self_impl, layer, query, key, value, *a, **kw):
        if key is not None and value is not None:
            li = _layer_index(layer, self_impl)
            mp = mean_pass[0]
            # skip the engine's dummy profiling run (attn_metadata is None): its
            # synthetic K would corrupt a mean, the max probes tolerate it
            amd = a[1] if len(a) > 1 else kw.get("attn_metadata")
            if MEAN and mp and li >= 0 and amd is not None:
                Hk, D = key.shape[-2], key.shape[-1]
                kr = key.reshape(-1, Hk, D).float()
                if mp == 1:
                    s = sums.get(li)
                    sums[li] = kr.sum(0) if s is None else s + kr.sum(0)
                    cnts[li] = cnts.get(li, 0) + kr.shape[0]
                else:
                    rm = (kr - mus[li]).abs().amax(dim=0)
                    prev = res_max.get(li)
                    res_max[li] = rm if prev is None else torch.maximum(prev, rm)
            q_m = query.abs().max().item()
            k_m = key.abs().max().item()
            v_m = value.abs().max().item()
            prev = stats.get(li, [0.0, 0.0, 0.0])
            stats[li] = [max(prev[0], q_m), max(prev[1], k_m), max(prev[2], v_m)]
            if PER_CHANNEL:
                Hk, D = key.shape[-2], key.shape[-1]
                cm = key.reshape(-1, Hk, D).abs().amax(dim=0).float()   # [Hk, D]
                chan_max[0] = cm if chan_max[0] is None else torch.maximum(chan_max[0], cm)
            if dump.get("want") and li == DUMP_LAYER:
                dump["q"] = query.detach().to(torch.float16).cpu()
                dump["k"] = key.detach().to(torch.float16).cpu()
                dump["v"] = value.detach().to(torch.float16).cpu()
        return orig_forward(self_impl, layer, query, key, value, *a, **kw)

    results = {}
    texts = {}
    for ctx in CTXS:
        reps = max(1, math.ceil(ctx / 78)) + 1
        text = FILLER * reps + QUESTION
        ids = tok.encode(text)
        if len(ids) > ctx:
            text = tok.decode(ids[: ctx - 8]) + QUESTION
        texts[ctx] = text
    TritonAttentionImpl.forward = patched_forward
    try:
        if MEAN:
            mean_pass[0] = 1
            for ctx in CTXS:
                llm.generate([texts[ctx]], sp(max_tokens=1, ignore_eos=True))
            for li, s in sums.items():
                mus[li] = s / cnts[li]
            sums.clear()
            mean_pass[0] = 2
        for ctx in CTXS:
            text = texts[ctx]
            stats.clear()
            dump.clear()
            if DUMP and ctx == CTXS[-1]:
                dump["want"] = True
            llm.generate([text], sp(max_tokens=1, ignore_eos=True))

            q_max = max(v[0] for v in stats.values())
            k_max = max(v[1] for v in stats.values())
            v_max = max(v[2] for v in stats.values())
            results[ctx] = {
                "q_max": q_max, "k_max": k_max, "v_max": v_max,
                "k_scale_full_range": round(k_max / FP8_MAX, 6),
                "v_scale_full_range": round(v_max / FP8_MAX, 6),
                "layers": len(stats),
                "per_layer_max": {str(li): {"q": round(v[0], 3), "k": round(v[1], 3), "v": round(v[2], 3)}
                                  for li, v in sorted(stats.items())},
            }
            top_layers = sorted(stats.items(), key=lambda kv: -kv[1][1])[:3]
            print("       layers with the largest max|K|:",
                  ", ".join(f"{li} ({v[1]:.1f})" for li, v in top_layers))
            print(f"ctx={ctx:>6}  max|Q|={q_max:8.3f}  max|K|={k_max:8.3f}  max|V|={v_max:8.3f}  "
                  f"(fp8 e4m3 max {FP8_MAX:.0f}; full-range scales k={results[ctx]['k_scale_full_range']} "
                  f"v={results[ctx]['v_scale_full_range']})")
            if dump.get("k") is not None:
                torch.save({"q": dump["q"], "k": dump["k"], "v": dump["v"], "ctx": ctx,
                            "layer": DUMP_LAYER, "model": MODEL}, DUMP)
                print(f"dumped layer {DUMP_LAYER} q{tuple(dump['q'].shape)} k{tuple(dump['k'].shape)} "
                      f"v{tuple(dump['v'].shape)} at ctx {ctx} -> {DUMP}")
    finally:
        TritonAttentionImpl.forward = orig_forward

    if MEAN:
        out = {
            "model": MODEL,
            "offsets": {li: mu.detach().cpu() for li, mu in mus.items()},
            "k_scales": {li: float(res_max[li].amax() / FP8_MAX)
                         for li in mus if li in res_max},
            # option 2: per-(head, channel) full-range scale of the residual;
            # consumed by the fp8 branches of the Triton store/read kernels
            "pc_scales": {li: (res_max[li] / FP8_MAX).detach().cpu()
                          for li in mus if li in res_max},
        }
        torch.save(out, MEAN)
        top = sorted(out["k_scales"].items(), key=lambda kv: -kv[1])[:5]
        print("k_scales (max|K-mean|/448) top layers:",
              ", ".join(f"{li} ({v:.4f})" for li, v in top))
        print(f"saved K offsets for {len(mus)} layers -> {MEAN}")

    if PER_CHANNEL and chan_max[0] is not None:
        cm = chan_max[0]
        flat = cm.flatten()
        srt, idx = torch.sort(flat, descending=True)
        Hk, D = cm.shape
        top = [(int(i) // D, int(i) % D, round(float(v), 1)) for v, i in zip(srt[:12], idx[:12])]
        summary = {
            "kv_heads": Hk, "head_dim": D,
            "median": round(float(flat.median()), 3),
            "p95": round(float(torch.quantile(flat, 0.95)), 3),
            "max": round(float(flat.max()), 3),
            "pairs_above_100": int((flat > 100).sum()),
            "top_pairs_head_channel_max": top,
        }
        results["per_channel_k"] = summary
        print(f"per-(head, channel) max|K| over layers: median {summary['median']}, p95 {summary['p95']}, "
              f"max {summary['max']}, {summary['pairs_above_100']} of {Hk * D} pairs above 100")
        print("top (head, channel, max|K|):", top)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {OUT}")
    print("KV_SCALE_PROBE_OK")


if __name__ == "__main__":
    main()
