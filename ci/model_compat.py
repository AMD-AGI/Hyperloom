#!/usr/bin/env python3
"""Shared model-compatibility rules for the AMD/ROCm serving stack.

Single source of truth used by both the offline pool filter
(``filter_candidates.py``) and the online per-model pre-flight in
``optimize_submit.py``. ``unrunnable_reason`` is a pure predicate over a HF
``config.json`` dict (plus an optional local model directory for file checks);
``hf_gated`` is the network-based gated/404 probe used by the offline filter.

All rules are config-deterministic except ``missing_tokenizer`` (needs the local
model dir) and gated/404 (network); the config rules are therefore safe to run
both offline (pool build) and online (after prewarm, before submit).
"""
import functools
import json
import os
import re
import time
import urllib.error
import urllib.request

# Context window at or below this is too small to be worth a sandbox slot.
SHORT_CTX_MAX = 2048

# Curated daily-fixed pool: every repo listed there is hand-picked and may
# intentionally include otherwise-filtered models (e.g. multimodal MoE run in
# text mode). Such repos are exempt from ALL compatibility filtering.
DAILY_FIXED_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "candidates", "inferencex_daily_fixed.json")


@functools.lru_cache(maxsize=8)
def load_whitelist(path=None):
    """Return the set of repo ids exempt from filtering (the daily-fixed pool)."""
    path = path or DAILY_FIXED_DEFAULT
    try:
        with open(path) as f:
            d = json.load(f)
        return frozenset(c["repo_id"] for c in d.get("candidates", [])
                         if c.get("repo_id"))
    except Exception:
        return frozenset()

# Explicit vision / multimodal architecture markers. NOTE: we deliberately do
# NOT match the bare ``*ForConditionalGeneration`` suffix — several text-only
# MoE models (e.g. Qwen3_5Moe / KimiK25) use that suffix without being vision
# models. Genuine multimodal models are caught here by an explicit vision
# token, by a vision ``model_type``, or by a ``vision_config`` block (see
# ``unrunnable_reason``).
VISION_ARCH = re.compile(
    r"(Llava|InternVL|Idefics\d?|PaliGemma|Florence|Mllama|Qwen\w*VL|VLForCausalLM|"
    r"VLMoe|Gemma3ForConditional|Gemma4ForConditional|Llama4ForConditional|"
    r"Mistral3ForConditional|Ernie\w*VL|MiniCPMV|GotOcr|Keye|Step3VL)",
    re.I)
VISION_MT = {"llava", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "internvl", "mllama",
             "idefics", "idefics2", "idefics3", "paligemma", "llava_next",
             "got_ocr2", "phi3_v", "phi4mm", "gemma3", "llama4", "mistral3",
             "ernie4_5_vl_moe"}

AMD_UNSUPPORTED_MODEL_TYPES = {"minimax_m1"}
AMD_UNSUPPORTED_ARCHES = {"minimaxm1forcausallm"}

UNRECOGNIZED_MODEL_TYPES = {
    "bailing_hybrid",
    "bailing_moe",
    "ovis2_6_next",
}
UNRECOGNIZED_ARCHES = {
    "bailingmoev2_5forcausallm",
    "bailingmoev2forcausallm",
    "ovis2_6_nextforcausallm",
}

_TOKENIZER_FILES = {"tokenizer.json", "tokenizer.model", "vocab.json",
                    "spiece.model", "tokenizer.model.v3", "merges.txt"}


def _listdir(d):
    import os
    try:
        return set(os.listdir(d))
    except OSError:
        return None


def has_weights(model_dir):
    """True if the local dir contains model weight shards."""
    files = _listdir(model_dir)
    if files is None:
        return False
    return any(f.endswith((".safetensors", ".bin", ".pt")) for f in files)


def has_tokenizer(model_dir):
    """True if the local dir contains a usable tokenizer file (or is unknown)."""
    files = _listdir(model_dir)
    if files is None:
        return True  # cannot tell -> assume present (do not skip)
    return bool(files & _TOKENIZER_FILES)


def unrunnable_reason(config, repo="", model_dir=None, whitelist=None):
    """Return ``(reason, detail)`` if the model cannot run on the ROCm stack,
    else ``None``.

    Args:
        config: Parsed HF ``config.json`` dict.
        repo: Optional repo id (for messages and whitelist match).
        model_dir: Optional local model directory; enables the
            ``missing_tokenizer`` check when weights are present.
        whitelist: Optional set of repo ids exempt from all filtering (e.g. the
            curated daily-fixed pool). Pass ``model_compat.load_whitelist()``.

    Rules: multimodal, short_ctx, phi3_longrope, dual_chunk_attention, gemma2,
    modelopt_fp8, attn_backend (flashinfer), missing_tokenizer.
    """
    if whitelist and repo and repo in whitelist:
        return None  # curated/whitelisted repo -> never filtered
    if not isinstance(config, dict):
        return None
    archs = config.get("architectures") or []
    arch = archs[0] if archs else ""
    mt = (config.get("model_type") or "").lower()
    qc = config.get("quantization_config") or {}
    rope = config.get("rope_scaling") or {}
    blob = json.dumps(config).lower()

    # 1) multimodal / VL — explicit vision arch token, vision model_type, or a
    #    vision_config/vision_tower block. Bare *ForConditionalGeneration is NOT
    #    treated as multimodal on its own (text-only MoE use it too).
    if (VISION_ARCH.search(arch) or mt in VISION_MT
            or isinstance(config.get("vision_config"), dict)
            or "vision_tower" in blob):
        return ("multimodal", f"arch={arch or mt}")

    arch_l = arch.lower()

    # 2) AMD/ROCm-unsupported architectures with confirmed hardware resource
    #    requirements unavailable on MI300X.
    if mt in AMD_UNSUPPORTED_MODEL_TYPES or arch_l in AMD_UNSUPPORTED_ARCHES:
        return ("amd_unsupported_arch", f"arch={arch or mt}")

    # 3) schema/model types that current Transformers/sglang ModelConfig does
    #    not recognize, causing deterministic engine-init validation failures.
    if mt in UNRECOGNIZED_MODEL_TYPES or arch_l in UNRECOGNIZED_ARCHES:
        return ("unrecognized_arch", f"arch={arch or mt}")

    # 4) short context (<= 2048)
    mpe = config.get("max_position_embeddings")
    if mpe is None:
        tc = config.get("text_config")
        mpe = tc.get("max_position_embeddings") if isinstance(tc, dict) else None
    try:
        if mpe is not None and int(mpe) <= SHORT_CTX_MAX:
            return ("short_ctx", f"max_position_embeddings={mpe}<={SHORT_CTX_MAX}")
    except (TypeError, ValueError):
        pass

    # 5) Phi3 longrope
    if ("phi3" in mt or "phi3" in arch_l) and \
            str(rope.get("type", rope.get("rope_type", ""))).lower() == "longrope":
        return ("phi3_longrope", "Phi3 longrope validation")

    # 6) dual chunk attention (NVIDIA sm90+)
    if config.get("dual_chunk_attention_config"):
        return ("dual_chunk_attention", "needs NVIDIA sm90+")

    # 7) Gemma2 config compatibility
    if mt == "gemma2" or arch == "Gemma2ForCausalLM":
        return ("gemma2", "Gemma2 config compat")

    # 8) NVIDIA ModelOpt FP8 (no ROCm loader)
    if str(qc.get("quant_method", "")).lower() == "modelopt" or "modelopt" in blob:
        return ("modelopt_fp8", "NVIDIA ModelOpt quant, no ROCm loader")

    # 9) unsupported attention backend (FlashInfer)
    attn = str(config.get("attn_implementation",
                          config.get("_attn_implementation", ""))).lower()
    if attn == "flashinfer" or "flashinfer" in blob:
        return ("attn_backend", "requires flashinfer (not on ROCm)")

    # 10) missing tokenizer (only when weights are present locally)
    if model_dir and has_weights(model_dir) and not has_tokenizer(model_dir):
        return ("missing_tokenizer", "weights present but no tokenizer files")

    return None


_tok_idx = [0]


def hf_gated(repo, tokens):
    """Return 'gated' | 'not_found' | None via the HF API with token rotation.

    Args:
        repo: HF repo id.
        tokens: list of HF tokens (rotated across attempts / rate limits).
    """
    if not tokens:
        return None
    url = f"https://huggingface.co/api/models/{repo}?expand[]=gated"
    for attempt in range(6):
        tok = tokens[_tok_idx[0] % len(tokens)]
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {tok}", "User-Agent": "ci-gated-check"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
            return "gated" if d.get("gated") in (True, "auto", "manual") else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "not_found"
            if e.code in (401, 403):
                _tok_idx[0] += 1
                if attempt >= 2:
                    return "gated"
                continue
            if e.code == 429:
                _tok_idx[0] += 1
                time.sleep(5 + attempt * 5)
                continue
            return None
        except Exception:
            time.sleep(2)
            continue
    return None
