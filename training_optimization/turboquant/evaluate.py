"""
TurboQuant end-to-end evaluation matching the paper's methodology.

Evaluates:
  1. Per-layer attention fidelity (cosine similarity of attention scores)
  2. Needle-in-a-Haystack retrieval (paper Section 4.2)
  3. Generation quality (perplexity on sample text)

Key differences from naive KV quantization:
  - Outlier-aware mixed-precision (paper Section 4.3)
  - TurboQuant_prod for keys (unbiased inner products)
  - TurboQuant_mse for values (optimal reconstruction)
  - Per-layer calibration of outlier channels
"""

import sys
import os
import torch
import torch.nn.functional as F
import math
import time
import gc
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turboquant_core import OutlierAwareTurboQuant, UniformTurboQuant

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

# ─── Configuration ───────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
EXPECTED_ANSWER = "AURORA-7749"

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up before the next "
    "meeting cycle.\n\n"
)


# ─── Model utilities ────────────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    mem = torch.cuda.memory_allocated() // 1024 // 1024
    print(f"Loaded. GPU memory: {mem} MB\n", flush=True)
    return model, tokenizer


def get_model_config(model):
    """Extract head_dim and number of layers."""
    config = model.config
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    n_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    return head_dim, n_layers, num_kv_heads


# ─── Quantized KV Cache ─────────────────────────────────────────────────

class TurboQuantDynamicLayer:
    """Drop-in replacement for DynamicLayer that uses TurboQuant compression.

    Implements the full DynamicLayer interface without subclassing (to avoid
    __init__ conflicts with the keys/values property pattern).

    Behavior matching real inference:
    - On update: compress new KV, store compressed, return decompressed-past + exact-current
    - Each KV entry is quantized exactly once
    """
    is_sliding = False
    is_compileable = False

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.compressed_keys_list = []
        self.compressed_values_list = []
        self.keys = None
        self.values = None
        self.is_initialized = False

    @torch.no_grad()
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict = None):
        """Compress and store new KV, return full sequence for attention."""
        k_comp = self.quantizer.compress_keys(key_states)
        v_comp = self.quantizer.compress_values(value_states)
        self.compressed_keys_list.append(k_comp)
        self.compressed_values_list.append(v_comp)

        if not self.is_initialized:
            self.is_initialized = True
            self.keys = key_states
            self.values = value_states
            return key_states, value_states

        past_keys = [self.quantizer.decompress_keys(kc)
                     for kc in self.compressed_keys_list[:-1]]
        past_values = [self.quantizer.decompress_values(vc)
                       for vc in self.compressed_values_list[:-1]]

        if past_keys:
            past_k = torch.cat(past_keys, dim=-2)
            past_v = torch.cat(past_values, dim=-2)
            full_k = torch.cat([past_k, key_states], dim=-2)
            full_v = torch.cat([past_v, value_states], dim=-2)
        else:
            full_k = key_states
            full_v = value_states

        self.keys = full_k
        self.values = full_v
        return full_k, full_v

    def get_seq_length(self):
        if not self.is_initialized or self.keys is None:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self):
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor):
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def reset(self):
        self.compressed_keys_list = []
        self.compressed_values_list = []
        self.keys = None
        self.values = None
        self.is_initialized = False

    def crop(self, max_length: int):
        pass

    def batch_repeat_interleave(self, repeats: int):
        pass

    def batch_select_indices(self, indices):
        pass

    def reorder_cache(self, beam_idx):
        pass

    def lazy_initialization(self, key_states, value_states):
        pass

    def offload(self):
        pass

    def prefetch(self):
        pass


class TurboQuantCache(DynamicCache):
    """DynamicCache subclass that uses TurboQuant for KV compression."""

    def __init__(self, quantizer_factory):
        super().__init__()
        self.quantizer_factory = quantizer_factory
        self._tq_layers = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx not in self._tq_layers:
            quantizer = self.quantizer_factory(layer_idx)
            if quantizer is None:
                tq_layer = DynamicLayer()
            else:
                tq_layer = TurboQuantDynamicLayer(quantizer)
            self._tq_layers[layer_idx] = tq_layer
            while len(self.layers) <= layer_idx:
                self.layers.append(None)
            self.layers[layer_idx] = tq_layer

        return self._tq_layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0):
        if layer_idx in self._tq_layers:
            return self._tq_layers[layer_idx].get_seq_length()
        return 0

    def reset(self):
        for layer in self._tq_layers.values():
            layer.reset()
        self._tq_layers.clear()
        self.layers.clear()


# ─── Calibration ─────────────────────────────────────────────────────────

def calibrate_outlier_channels(model, tokenizer, calibration_text: str = None,
                                max_tokens: int = 512):
    """Run a calibration forward pass to identify outlier channels per layer.

    Returns: dict mapping layer_idx -> (key_outlier_indices, value_outlier_indices)
    """
    if calibration_text is None:
        calibration_text = FILLER * 5

    inputs = tokenizer(
        calibration_text, return_tensors="pt",
        truncation=True, max_length=max_tokens
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    cache = outputs.past_key_values

    head_dim, n_layers, _ = get_model_config(model)
    outlier_info = {}

    for layer_idx in range(n_layers):
        layer = cache.layers[layer_idx]
        keys = layer.keys    # (batch, n_kv_heads, seq, head_dim)
        values = layer.values

        key_var = keys.float().reshape(-1, head_dim).var(dim=0)
        val_var = values.float().reshape(-1, head_dim).var(dim=0)

        outlier_info[layer_idx] = {
            "key_variance": key_var.cpu(),
            "value_variance": val_var.cpu(),
        }

    del outputs, cache
    torch.cuda.empty_cache()
    gc.collect()

    return outlier_info


# ─── Evaluation: Attention Fidelity ─────────────────────────────────────

def evaluate_attention_fidelity(model, tokenizer, quantizer_configs: List[dict],
                                 text: str = None, max_tokens: int = 512):
    """Compare attention scores between full-precision and quantized KV.

    This matches the paper's validation methodology: extract KV from a full-
    precision forward pass, compress, then compare attention scores.
    """
    if text is None:
        text = FILLER * 3

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_tokens).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    cache = outputs.past_key_values

    head_dim, n_layers, num_kv_heads = get_model_config(model)

    results = {}
    for config in quantizer_configs:
        label = config["label"]
        print(f"\n  Evaluating attention fidelity: {label}")

        cosine_sims = []
        top1_matches = 0
        n_checks = 0

        for layer_idx in range(n_layers):
            layer = cache.layers[layer_idx]
            keys = layer.keys     # (1, H, S, D)
            values = layer.values

            quantizer = config["factory"](layer_idx)
            if quantizer is None:
                k_decompressed = keys
            else:
                if hasattr(quantizer, "calibrate") and hasattr(quantizer, "outlier_indices"):
                    if quantizer.outlier_indices is None:
                        quantizer.calibrate(keys)
                k_comp = quantizer.compress_keys(keys)
                k_decompressed = quantizer.decompress_keys(k_comp)

            query = keys[:, :, -1:, :]  # last token as query

            real_scores = torch.matmul(
                query.float(), keys.float().transpose(-2, -1)
            ).squeeze(-2)  # (1, H, S)

            tq_scores = torch.matmul(
                query.float(), k_decompressed.float().transpose(-2, -1)
            ).squeeze(-2)

            for h in range(real_scores.shape[1]):
                rs = real_scores[0, h]
                ts = tq_scores[0, h]
                cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
                cosine_sims.append(cos)
                if rs.argmax().item() == ts.argmax().item():
                    top1_matches += 1
                n_checks += 1

        avg_cos = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0
        top1_pct = 100 * top1_matches / n_checks if n_checks else 0

        results[label] = {
            "cosine_sim": avg_cos,
            "top1_match_pct": top1_pct,
            "n_checks": n_checks,
        }
        print(f"    Score cosine similarity: {avg_cos:.6f}")
        print(f"    Top-1 match: {top1_pct:.1f}% ({top1_matches}/{n_checks})")

    del outputs, cache
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ─── Evaluation: Needle-in-a-Haystack ───────────────────────────────────

def build_needle_prompt(tokenizer, target_tokens: int = 2048,
                        needle_pos: float = 0.5):
    """Build a prompt with a needle hidden in filler text."""
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)

    messages = [
        {"role": "user", "content": f"{haystack}\nQuestion: {QUESTION}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    return prompt


def run_needle_test(model, tokenizer, past_key_values=None,
                    target_tokens: int = 2048, needle_pos: float = 0.5,
                    max_new_tokens: int = 50):
    """Run a single needle-in-a-haystack test.

    Returns: (generated_text, found_needle: bool)
    """
    prompt = build_needle_prompt(tokenizer, target_tokens, needle_pos)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=target_tokens + 256).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=past_key_values,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    found = EXPECTED_ANSWER.lower() in generated.lower()
    return generated, found


def evaluate_needle_in_haystack(model, tokenizer, quantizer_configs: List[dict],
                                 context_lengths: List[int] = None,
                                 needle_positions: List[float] = None):
    """Full needle-in-a-haystack evaluation matching paper Section 4.2."""
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096]
    if needle_positions is None:
        needle_positions = [0.25, 0.5, 0.75]

    all_results = {}

    # Full precision baseline
    print("\n" + "=" * 70)
    print("Full Precision Baseline")
    print("=" * 70)
    fp_results = {}
    for ctx_len in context_lengths:
        for pos in needle_positions:
            generated, found = run_needle_test(
                model, tokenizer, target_tokens=ctx_len, needle_pos=pos
            )
            key = f"ctx={ctx_len}_pos={pos}"
            fp_results[key] = found
            status = "FOUND" if found else "MISSED"
            print(f"  {key}: {status} | '{generated[:80]}...'")
            torch.cuda.empty_cache()

    fp_score = sum(fp_results.values()) / len(fp_results)
    print(f"\n  Full Precision Score: {fp_score:.3f}")
    all_results["Full Precision"] = {"score": fp_score, "details": fp_results}

    # Quantized evaluations
    head_dim, n_layers, num_kv_heads = get_model_config(model)

    for config in quantizer_configs:
        label = config["label"]
        print(f"\n{'=' * 70}")
        print(f"TurboQuant: {label}")
        print(f"{'=' * 70}")

        outlier_info = calibrate_outlier_channels(model, tokenizer)

        tq_results = {}
        for ctx_len in context_lengths:
            for pos in needle_positions:
                def make_cache():
                    def factory(layer_idx):
                        q = config["factory"](layer_idx)
                        if hasattr(q, "calibrate"):
                            key_var = outlier_info[layer_idx]["key_variance"]
                            _, top_idx = key_var.topk(q.n_outlier if hasattr(q, 'n_outlier') else 0)
                            if hasattr(q, 'set_outlier_indices') and hasattr(q, 'n_outlier'):
                                q.set_outlier_indices(top_idx)
                        return q
                    return TurboQuantCache(factory)

                cache = make_cache()

                try:
                    generated, found = run_needle_test(
                        model, tokenizer, past_key_values=cache,
                        target_tokens=ctx_len, needle_pos=pos
                    )
                    key = f"ctx={ctx_len}_pos={pos}"
                    tq_results[key] = found
                    status = "FOUND" if found else "MISSED"
                    print(f"  {key}: {status} | '{generated[:80]}...'")
                except Exception as e:
                    key = f"ctx={ctx_len}_pos={pos}"
                    tq_results[key] = False
                    print(f"  {key}: ERROR - {e}")

                del cache
                torch.cuda.empty_cache()
                gc.collect()

        tq_score = sum(tq_results.values()) / len(tq_results)
        print(f"\n  {label} Score: {tq_score:.3f}")
        all_results[label] = {"score": tq_score, "details": tq_results}

    return all_results


# ─── Evaluation: Perplexity ──────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, text: str = None,
                        past_key_values_factory=None,
                        max_tokens: int = 1024, stride: int = 512):
    """Evaluate perplexity with optional quantized KV cache."""
    if text is None:
        text = FILLER * 20

    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens)
    input_ids = encodings["input_ids"].to(model.device)
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        return float("inf")

    nlls = []
    n_tokens = 0

    if past_key_values_factory is not None:
        past_key_values = past_key_values_factory()
    else:
        past_key_values = DynamicCache()

    with torch.no_grad():
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + stride, seq_len)
            chunk_ids = input_ids[:, begin:end]

            outputs = model(
                chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            nlls.append(loss.item())
            n_tokens += shift_labels.numel()

    ppl = math.exp(sum(nlls) / n_tokens) if n_tokens > 0 else float("inf")
    return ppl


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        print(f"GPU memory: {mem} MB")

    model, tokenizer = load_model(MODEL_NAME, device)
    head_dim, n_layers, num_kv_heads = get_model_config(model)
    print(f"Model config: head_dim={head_dim}, n_layers={n_layers}, num_kv_heads={num_kv_heads}")

    # ── Step 1: Calibrate outlier channels ──
    print("\n" + "=" * 70)
    print("Step 1: Calibrating outlier channels")
    print("=" * 70)

    outlier_info = calibrate_outlier_channels(model, tokenizer)
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        kv = outlier_info[layer_idx]["key_variance"]
        top_vals, top_idx = kv.topk(5)
        print(f"  Layer {layer_idx}: top-5 channel variances: "
              f"{top_vals.tolist()}, indices: {top_idx.tolist()}")

    # ── Build quantizer configurations ──
    # Paper config: 3.5-bit with 32 outlier channels at 5b, 96 regular at 3b
    # Paper config: 2.5-bit with 32 outlier channels at 4b, 96 regular at 2b

    n_outlier = 32
    skip_layers = 2  # skip first/last N layers (keep at full precision)
    configs = []

    def make_outlier_aware_factory(outlier_bits, regular_bits, n_out, skip_n=0):
        def factory(layer_idx):
            if skip_n > 0 and (layer_idx < skip_n or layer_idx >= n_layers - skip_n):
                return None  # signal: no quantization for this layer
            q = OutlierAwareTurboQuant(
                head_dim=head_dim,
                outlier_bits=outlier_bits,
                regular_bits=regular_bits,
                n_outlier=n_out,
                key_mode="mse",
                value_mode="mse",
                device=device,
                seed=42 + layer_idx * 1000,
            )
            key_var = outlier_info[layer_idx]["key_variance"]
            _, top_idx = key_var.topk(n_out)
            q.set_outlier_indices(top_idx)
            return q
        return factory

    def make_uniform_factory(bits, skip_n=0):
        def factory(layer_idx):
            if skip_n > 0 and (layer_idx < skip_n or layer_idx >= n_layers - skip_n):
                return None
            q = UniformTurboQuant(
                head_dim=head_dim,
                bits=bits,
                key_mode="mse",
                value_mode="mse",
                device=device,
                seed=42 + layer_idx * 1000,
            )
            return q
        return factory

    configs.append({
        "label": f"TQ-3.5bit OA (skip first/last {skip_layers})",
        "factory": make_outlier_aware_factory(5, 3, n_outlier, skip_layers),
    })
    configs.append({
        "label": f"TQ-2.5bit OA (skip first/last {skip_layers})",
        "factory": make_outlier_aware_factory(4, 2, n_outlier, skip_layers),
    })
    configs.append({
        "label": f"TQ-4bit uniform (skip first/last {skip_layers})",
        "factory": make_uniform_factory(4, skip_layers),
    })
    configs.append({
        "label": "TQ-3.5bit OA (all layers)",
        "factory": make_outlier_aware_factory(5, 3, n_outlier, 0),
    })

    # ── Step 2: Attention Fidelity ──
    print("\n" + "=" * 70)
    print("Step 2: Attention Fidelity (per-layer score comparison)")
    print("=" * 70)

    attn_results = evaluate_attention_fidelity(model, tokenizer, configs)
    print("\n  Summary:")
    for label, r in attn_results.items():
        print(f"    {label}: cos_sim={r['cosine_sim']:.6f}, top1={r['top1_match_pct']:.1f}%")

    # ── Step 3: Needle-in-a-Haystack ──
    print("\n" + "=" * 70)
    print("Step 3: Needle-in-a-Haystack Retrieval")
    print("=" * 70)

    needle_results = evaluate_needle_in_haystack(
        model, tokenizer, configs,
        context_lengths=[1024, 2048, 4096],
        needle_positions=[0.25, 0.5, 0.75],
    )
    print("\n  Final Scores:")
    for label, r in needle_results.items():
        print(f"    {label}: {r['score']:.3f}")

    # ── Step 4: Perplexity ──
    print("\n" + "=" * 70)
    print("Step 4: Perplexity Evaluation")
    print("=" * 70)

    ppl_text = FILLER * 30

    fp_ppl = evaluate_perplexity(model, tokenizer, text=ppl_text, max_tokens=2048)
    print(f"\n  Full Precision PPL: {fp_ppl:.2f}")

    for config in configs:
        label = config["label"]

        def make_cache_factory(cfg):
            def factory():
                return TurboQuantCache(cfg["factory"])
            return factory

        ppl = evaluate_perplexity(
            model, tokenizer, text=ppl_text,
            past_key_values_factory=make_cache_factory(config),
            max_tokens=2048,
        )
        print(f"  {label} PPL: {ppl:.2f}")
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
