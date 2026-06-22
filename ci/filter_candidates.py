#!/usr/bin/env python3
"""Filter HF candidate pools by structural un-runnability on the AMD/ROCm stack.

Primary signal is the locally cached model under ``/wekafs/models/<slug>``
(``config.json``, tokenizer files, ``model.safetensors.index.json``); ``gated``
is verified through the HuggingFace API. Models with no local cache and no other
signal are KEPT (cannot judge from config alone).

Filter rules
------------
- multimodal        : architectures contain ``*ForConditionalGeneration`` /
                      ``Llava`` / ``InternVL`` / ``Qwen*VL`` / ``Llama4`` /
                      ``Mistral3`` / ``Gemma3`` (etc.), or model_type is a vision
                      type, or config has ``vision_config`` / ``vision_tower``.
- short_ctx         : ``max_position_embeddings <= 2048``.
- phi3_longrope     : Phi3 architecture with ``rope_scaling.type == "longrope"``.
- dual_chunk_attention : config has ``dual_chunk_attention_config`` (needs NVIDIA sm90+).
- gemma2            : model_type ``gemma2`` / ``Gemma2ForCausalLM`` (config compat).
- modelopt_fp8      : ``quantization_config.quant_method == "modelopt"`` (no ROCm loader).
- attn_backend      : ``attn_implementation == "flashinfer"`` (not on ROCm).
- missing_tokenizer : weights present locally but no tokenizer files.
- gated             : HuggingFace ``gated`` field is ``auto`` / ``manual`` (HF API).
- not_found         : HuggingFace returns 404 for the repo (HF API).

Env
---
- ``HF_TOKENS``         : comma-separated HF tokens for the gated check (rotated).
                          Falls back to ``HF_TOKEN`` / ``HF_TOKEN_2``.
- ``CI_MODELS_DIR``     : local model cache root (default ``/wekafs/models``).
- ``CI_FILTER_OUT_DIR`` : output directory for filtered JSON + reports.

Usage
-----
    HF_TOKENS=hf_a,hf_b python3 ci/filter_candidates.py POOL1.json [POOL2.json ...]

When no pool paths are given, the two production pools are used by default.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_compat  # noqa: E402  (local sibling module)

MODELS_DIR = os.environ.get("CI_MODELS_DIR", "/wekafs/models")
OUT_DIR = os.environ.get(
    "CI_FILTER_OUT_DIR", "/wekafs/weilei/claw-dev/ci-candidates-filtered")
GATED_CACHE = os.path.join(OUT_DIR, "gated_cache.tsv")

DEFAULT_POOLS = {
    "gt100_rotate": "/wekafs/chenyi/ci-candidates/hf_downloads_gt100_rotate.json",
    "sub100_pulse": "/wekafs/chenyi/ci-candidates/sub100_lt12b_pulse_notrun.json",
}


def hf_tokens():
    """Resolve HF tokens from env (comma-separated HF_TOKENS, or HF_TOKEN[_2])."""
    raw = os.environ.get("HF_TOKENS", "")
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    for k in ("HF_TOKEN", "HF_TOKEN_2"):
        v = os.environ.get(k, "").strip()
        if v and v not in toks:
            toks.append(v)
    return toks


HF_TOKENS = hf_tokens()

def slug(repo):
    """Map a HF repo id to the on-disk cache directory name."""
    return repo.replace("/", "-")


def classify_local(repo):
    """Apply the shared config rules to the cached model. Return reason or None."""
    mdir = os.path.join(MODELS_DIR, slug(repo))
    cfg_path = os.path.join(mdir, "config.json")
    if not os.path.isfile(cfg_path):
        return None  # no local cache -> judged only by the HF gated check
    try:
        cfg = json.load(open(cfg_path))
    except Exception:
        return None
    return model_compat.unrunnable_reason(cfg, repo=repo, model_dir=mdir)


def hf_gated(repo):
    """gated/not_found via the shared HF probe (env-provided tokens)."""
    return model_compat.hf_gated(repo, HF_TOKENS)


def load_gated_cache():
    cache = {}
    if os.path.isfile(GATED_CACHE):
        for line in open(GATED_CACHE):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                cache[parts[0]] = parts[1]
    return cache


def gated_check_all(repos):
    """Resolve gated/not_found for repos, caching results to GATED_CACHE."""
    cache = load_gated_cache()
    todo = [r for r in repos if r not in cache]
    print(f"[gated] cached={len(cache)} todo={len(todo)}", flush=True)
    with open(GATED_CACHE, "a") as fh:
        for i, repo in enumerate(todo):
            cache[repo] = hf_gated(repo) or "ok"
            fh.write(f"{repo}\t{cache[repo]}\n")
            fh.flush()
            if (i + 1) % 200 == 0:
                print(f"[gated] {i + 1}/{len(todo)}", flush=True)
            time.sleep(0.05)
    return cache


def main(argv):
    pools = ({os.path.splitext(os.path.basename(p))[0]: p for p in argv}
             if argv else DEFAULT_POOLS)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: local config rules -> tentative keep set per pool. Repos in the
    # curated daily-fixed whitelist are exempt from ALL filtering (kept as-is,
    # no config rule, no gated check).
    whitelist = model_compat.load_whitelist()
    pools_local = {}
    keep_repos = []
    for pname, ppath in pools.items():
        data = json.load(open(ppath))
        cands = data.get("candidates", [])
        local_keep, local_filt = [], []
        for c in cands:
            repo = c.get("repo_id")
            if not repo or repo in whitelist:
                local_keep.append((c, None))  # exempt: keep, skip gated lookup
                continue
            r = classify_local(repo)
            if r:
                local_filt.append((repo, r[0], r[1]))
            else:
                local_keep.append((c, repo))
                keep_repos.append(repo)
        pools_local[pname] = (data, cands, local_keep, local_filt)

    # Phase 2: HF gated check over tentatively-kept repos.
    gated = gated_check_all(sorted(set(keep_repos)))

    # Phase 3: finalize per pool.
    grand = {}
    with open(os.path.join(OUT_DIR, "pool_filter_report.tsv"), "w") as report:
        report.write("pool\trepo\treason\tdetail\n")
        for pname, (data, cands, local_keep, local_filt) in pools_local.items():
            counts = {}
            for repo, reason, detail in local_filt:
                counts[reason] = counts.get(reason, 0) + 1
                report.write(f"{pname}\t{repo}\t{reason}\t{detail}\n")
            kept = []
            for c, repo in local_keep:
                st = gated.get(repo) if repo else None
                if st in ("gated", "not_found"):
                    counts[st] = counts.get(st, 0) + 1
                    report.write(f"{pname}\t{repo}\t{st}\tHF API\n")
                else:
                    kept.append(c)
            out = dict(data)
            out["candidates"] = kept
            outpath = os.path.join(OUT_DIR, f"{pname}_filtered.json")
            json.dump(out, open(outpath, "w"), indent=2)
            grand[pname] = (len(cands), len(kept), len(cands) - len(kept),
                            counts, outpath)

    print("=" * 60)
    for pname, (tot, k, f, counts, outpath) in grand.items():
        print(f"\n## {pname}: total={tot}  kept={k}  filtered={f}")
        for reason in sorted(counts, key=lambda x: -counts[x]):
            print(f"     {reason}: {counts[reason]}")
        print(f"   -> {outpath}")
    print(f"\nreport -> {OUT_DIR}/pool_filter_report.tsv")


if __name__ == "__main__":
    main(sys.argv[1:])
