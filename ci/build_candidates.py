#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ci/build_candidates.py — one-time HF top-N candidate builder.

Fetches HuggingFace top-N text-generation models, applies a hard filter
(min params, framework support, weight footprint), excludes already-run
models, then writes the result to ci/candidates/topN_v<date>.json.

The output file is the source-of-truth list consumed by
``generate_hf_matrix.py --candidates-file`` during batch dispatch.

Filter rules (default, tweakable via CLI flags):
  - HF API ``filter=text-generation`` listing + per-repo verify of
    ``pipeline_tag`` and ``architectures[0]`` generative-ness.
  - >= ``--min-params`` B parameters (default 7).
  - Weight footprint <= ``--max-weight-gb`` (default 600 GB) — skip
    1-TB DeepSeek-V4 / GLM-5 class. Per-precision bytes-per-param uses
    the same heuristic as ``optimize_submit.detect_tp``.
  - Skip repos listed in ``ci/candidates/already_done.json`` (20 done
    or permanent-failed in the 5/8-5/11 runs).
  - Skip NVFP4 (NVIDIA modelopt; ROCm has no kernels for this).
  - Skip non-vLLM-non-sglang quant formats: GGUF, MLX, GPTQ-Int8,
    Q4_K_M, w8a8 — these will fail at server start.

Usage:
  python3 build_candidates.py --top 200 --min-params 7 \\
      --output ci/candidates/top200_2026-05-12.json

The script is idempotent: re-running with the same args reproduces the
same JSON modulo per-repo HF API drift (download counts shift daily).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger("build-candidates")

HF_BASE = "https://huggingface.co"

# Drop repos matching these non-standard quant formats our toolchain
# (vllm/sglang on ROCm) does not support, even if HF reports text-generation.
QUANT_FORMAT_BLOCKLIST = re.compile(
    r"(?i)(NVFP4|"             # NVIDIA modelopt FP4 — ROCm has no kernels
    r"GGUF|"                   # llama.cpp format
    r"MLX|"                    # Apple Silicon
    r"-MLX-|"
    r"-Q[0-9]_K|"              # GGUF Q4_K_M / Q8_0 etc.
    r"-Q[0-9]_[0-9]|"
    r"\.w[0-9]a[0-9]|"         # RedHatAI quantized.w8a8 / w4a16
    r"quantized\.w"
    r")"
)

# Tasks/heads our pipeline can't optimize (vision-only, embedding, etc.) that
# can show pipeline_tag=text-generation via tag pollution but aren't causal LMs.
NON_LM_BLOCKLIST = re.compile(
    r"(?i)(embedding|reranker|rerank|"
    r"-VL-|"                   # Qwen3-VL, vision-language
    r"-Vision-|"
    r"vision-instruct|"
    r"-TTS-|tts-|"
    r"-Speech-|"
    r"paraphraser|"
    r"-Guard-|"                # Qwen3Guard moderation
    r"-Reward-|"
    r"diffusion"
    r")"
)

# Family-level blocklist — fit under max_weight_gb but de-prioritized (extreme
# MoE families costing disproportionate sandbox time/storage, or archs sglang/vllm
# don't support yet on ROCm). Matches inside repo_id, case-insensitive.
FAMILY_BLOCKLIST = re.compile(
    r"(?i)("
    r"DeepSeek-V4|"     # all V4 variants (Pro/Flash/Base/FP8-test)
    r"GLM-5|"           # 1+ TB MoE
    r"DeepSeek-V3$"     # DSV3 base only (V3.2 / V3.0324 stay eligible)
    r")"
)


def precision_bytes_per_param(precision: str) -> float:
    p = (precision or "").upper()
    if p in ("FP4", "INT4", "NVFP4", "MXFP4"):
        return 0.5
    if p == "FP8":
        return 1.0
    return 2.0  # BF16 / FP16 default


def detect_precision_from_config(config: dict) -> str:
    quant = config.get("quantization_config") or {}
    raw = (
        quant.get("quant_algo")
        or quant.get("quant_type")
        or quant.get("quantization_type")
        or quant.get("quant_method")
        or quant.get("method")
        or ""
    ).lower()
    if "fp8" in raw:   return "FP8"
    if "mxfp4" in raw: return "FP4"
    if "nvfp4" in raw: return "FP4"
    if "int4" in raw:  return "INT4"
    if "gptq" in raw:  return "INT4"
    if "awq" in raw:   return "INT4"
    return "BF16"  # most full-precision HF repos default here


def is_generative_arch(arch: str) -> bool:
    if not arch:
        return False
    suffixes = (
        "ForCausalLM",
        "LMHeadModel",
        "ForConditionalGeneration",
        "ForSeq2SeqLM",
    )
    return any(arch.endswith(s) for s in suffixes)


class HFClient:
    def __init__(self, token: str = "", timeout: int = 20):
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers["User-Agent"] = "hyperloom-build-candidates/1.0"
        if token:
            self._sess.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str) -> dict | list:
        r = self._sess.get(f"{HF_BASE}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def listing(self, limit: int) -> list[dict]:
        """Top-N text-generation by downloads (raw HF API records).

        HF caps one page at 1000 entries; follow the ``Link`` header cursor
        until ``limit`` entries are fetched.
        """
        out: list[dict] = []
        page_limit = min(max(limit, 1), 1000)
        path = (f"/api/models?sort=downloads&direction=-1"
                f"&limit={page_limit}&filter=text-generation")
        seen_urls: set[str] = set()
        while path and len(out) < limit:
            url = f"{HF_BASE}{path}" if path.startswith("/") else path
            if url in seen_urls:
                break
            seen_urls.add(url)
            r = self._sess.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            assert isinstance(data, list)
            remaining = limit - len(out)
            out.extend(data[:remaining])
            if len(out) >= limit:
                break
            path = self._next_link_path(r.headers.get("Link") or "")
        return out

    @staticmethod
    def _next_link_path(link_header: str) -> str:
        for chunk in link_header.split(","):
            if 'rel="next"' not in chunk:
                continue
            m = re.search(r"<([^>]+)>", chunk)
            if not m:
                continue
            url = m.group(1)
            parsed = urlparse(url)
            return parsed.path + (("?" + parsed.query) if parsed.query else "")
        return ""

    def model_info(self, repo_id: str) -> dict:
        return self._get(f"/api/models/{repo_id}")  # type: ignore[return-value]

    def model_config(self, repo_id: str) -> dict:
        return self._get(f"/{repo_id}/resolve/main/config.json")  # type: ignore[return-value]


def load_already_done(path: Path) -> set[str]:
    """Return repo_id set from already_done.json (case-sensitive)."""
    if not path.exists():
        log.warning("already_done.json not found at %s — nothing excluded", path)
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m["repo_id"] for m in data.get("models", [])}


def classify_candidate(
    hf: HFClient,
    repo_id: str,
    min_params_b: float,
    max_weight_gb: float,
) -> dict | None:
    """Return a candidate record or None if filtered out.

    Returned record:
      {repo_id, pipeline_tag, arch, params_b, precision, weight_gb,
       tp_estimate, framework_hint}
    """
    try:
        info = hf.model_info(repo_id)
    except Exception as e:
        log.info("skip %s: model_info failed (%s)", repo_id, e)
        return None

    pipeline_tag = (info.get("pipeline_tag") or "").strip()
    if pipeline_tag and pipeline_tag != "text-generation":
        log.info("skip %s: pipeline_tag=%s", repo_id, pipeline_tag)
        return None

    try:
        config = hf.model_config(repo_id)
    except Exception as e:
        log.info("skip %s: config.json unreachable (%s) — likely gated", repo_id, e)
        return None

    arch = (config.get("architectures") or [""])[0]
    if not is_generative_arch(arch):
        log.info("skip %s: arch=%s non-generative", repo_id, arch)
        return None

    total = (info.get("safetensors") or {}).get("total", 0)
    if not total:
        # No safetensors index — can't size-check; skip rather than guess.
        log.info("skip %s: no safetensors.total (likely pt/bin only)", repo_id)
        return None

    params_b = total / 1e9
    if params_b < min_params_b:
        log.info("skip %s: params=%.1fB < %sB", repo_id, params_b, min_params_b)
        return None

    precision = detect_precision_from_config(config)
    weight_gb = params_b * precision_bytes_per_param(precision)
    if weight_gb > max_weight_gb:
        log.info("skip %s: weight_gb=%.0f > %.0f (DSV4/GLM-5 class)",
                 repo_id, weight_gb, max_weight_gb)
        return None

    return {
        "repo_id": repo_id,
        "pipeline_tag": pipeline_tag,
        "arch": arch,
        "params_b": round(params_b, 2),
        "precision": precision,
        "weight_gb": round(weight_gb, 1),
        "downloads": info.get("downloads", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=200,
                   help="HF top-N to fetch (post-verify count will be lower)")
    p.add_argument("--min-params", type=float, default=7.0,
                   help="Minimum params in B (default 7, matches Qwen2.5-7B baseline)")
    p.add_argument("--max-weight-gb", type=float, default=600.0,
                   help="Skip if weight > this GB (default 600, drops DSV4/GLM-5)")
    p.add_argument("--already-done", type=Path,
                   default=Path(__file__).parent / "candidates" / "already_done.json",
                   help="JSON of repos to exclude (default ci/candidates/already_done.json)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path (e.g. ci/candidates/top200_2026-05-12.json)")
    p.add_argument("--hf-token", default="",
                   help="HF token for gated metadata (rarely needed)")
    p.add_argument("--target-count", type=int, default=None,
                   help="Stop after this many candidates pass (None = no cap, use all)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    already_done = load_already_done(args.already_done)
    log.info("already_done set: %d repos", len(already_done))

    hf = HFClient(args.hf_token)
    log.info("fetching HF top-%d text-generation listing...", args.top)
    listing = hf.listing(args.top)
    log.info("got %d raw entries", len(listing))

    accepted: list[dict] = []
    seen = set()
    n_dup = n_done = n_format = n_task = n_family = n_classify = 0
    for m in listing:
        repo = m.get("modelId") or m.get("id") or ""
        if not repo or "/" not in repo:
            continue
        if repo in seen:
            n_dup += 1
            continue
        seen.add(repo)
        if repo in already_done:
            n_done += 1
            log.info("skip %s: already in already_done.json", repo)
            continue
        if QUANT_FORMAT_BLOCKLIST.search(repo):
            n_format += 1
            log.info("skip %s: blocklisted quant format", repo)
            continue
        if NON_LM_BLOCKLIST.search(repo):
            n_task += 1
            log.info("skip %s: non-LM task pattern", repo)
            continue
        if FAMILY_BLOCKLIST.search(repo):
            n_family += 1
            log.info("skip %s: family-blocklisted (DSV4 / GLM-5 class)", repo)
            continue

        record = classify_candidate(
            hf, repo,
            min_params_b=args.min_params,
            max_weight_gb=args.max_weight_gb,
        )
        if record is None:
            n_classify += 1
            continue
        accepted.append(record)
        log.info("[%d/%d] keep %s (%.1fB %s)",
                 len(accepted), args.top, repo,
                 record["params_b"], record["precision"])
        if args.target_count and len(accepted) >= args.target_count:
            log.info("hit target_count=%d, stopping early", args.target_count)
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "_meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "hf_top": args.top,
            "min_params_b": args.min_params,
            "max_weight_gb": args.max_weight_gb,
            "already_done_file": str(args.already_done),
            "accepted_count": len(accepted),
            "rejected": {
                "duplicate": n_dup,
                "already_done": n_done,
                "blocklisted_format": n_format,
                "non_lm_task": n_task,
                "family_blocklist": n_family,
                "classify_filter": n_classify,
            },
        },
        "candidates": accepted,
    }
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(
        "wrote %d candidates to %s "
        "(rejected: dup=%d done=%d fmt=%d task=%d family=%d classify=%d)",
        len(accepted), args.output, n_dup, n_done, n_format, n_task,
        n_family, n_classify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
