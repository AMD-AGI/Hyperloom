"""Manually publish the 26-model summary to luochen's hyperloom-results-service.

Usage:
  python .harvest/manual_publish.py --dry-run                       # print payload, no network
  python .harvest/manual_publish.py --probe                         # POST a single tiny record to /api/import (debug)
  python .harvest/manual_publish.py --send-all                      # POST all 26 records
  python .harvest/manual_publish.py --send-only "Mistral-7B-v0.1"   # POST one record by display_name match

Records:
  20 from .harvest/RESULTS_FINAL.md (the existing batch — replaces what we previously couldn't publish)
  6 from the 2026-05-12 batch=10 run (run id 25749785697) the agent reported in chat
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# These are the same values build_summary.py would have produced if the GHA
# Build summary step had not crashed on the publish_results 500 error. Schema
# matches artifact_normalizer.SCHEMA_VERSION = "hyperloom.ci.normalized.v1"
# (so an older publish_results.py shape is happy with it).
SCHEMA_VERSION = "hyperloom.ci.normalized.v1"

# Default URL mirrors publish_results.DEFAULT_SERVICE_URL.
DEFAULT_URL = os.environ.get(
    "HYPERLOOM_RESULTS_SERVICE_URL",
    "http://core42.primus-safe.amd.com/hyperloom-results",
)
DEFAULT_TOKEN = os.environ.get("HYPERLOOM_RESULTS_SERVICE_TOKEN", "")

NOW_ISO = datetime.now(timezone.utc).isoformat()


def _record(
    *,
    rank: int,
    model: str,
    display_name: str,
    framework: str,
    precision: str,
    tp: int,
    params_b: float,
    baseline: float | None,
    optimized: float | None,
    gain_pct: float | None,
    final_status: str = "Succeeded",
    inferenceX_ref: float | None = None,
    peak_throughput: float | None = None,
    peak_throughput_conc: int | None = None,
    actions: list[str] | None = None,
    note: str = "",
    source_run: str = "",
    task_id: str | None = None,
    claw_session_id: str | None = None,
) -> dict:
    """Build one normalized result record matching the SCHEMA_VERSION shape."""
    metrics = {
        "baseline_throughput": baseline,
        "optimized_throughput": optimized,
        "gain_pct": gain_pct,
        "tok_per_gpu_baseline": baseline,
        "tok_per_gpu_optimized": optimized,
        "peak_throughput": peak_throughput,
        "peak_throughput_conc": peak_throughput_conc,
        "model": model,
        "framework": framework,
        "tp": tp,
        "isl": 1024,
        "osl": 1024,
        "conc": 64,
        "actions": actions or [],
    }
    if (
        gain_pct is None
        and baseline is not None
        and optimized is not None
        and baseline > 0
    ):
        metrics["gain_pct"] = round((optimized - baseline) / baseline * 100, 2)

    extra: dict = {}
    if inferenceX_ref is not None:
        extra["inferenceX_ref_tok_per_gpu"] = inferenceX_ref
        if optimized is not None and inferenceX_ref:
            extra["vs_inferenceX_pct"] = round(
                (optimized - inferenceX_ref) / inferenceX_ref * 100, 2
            )
    if note:
        extra["note"] = note

    rec = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            # NOTE: leaving submitted_at as None — luochen's service has a bug
            # in postgres_store._nullable_timestamp where it returns str but
            # asyncpg requires datetime for timestamptz. Setting None makes the
            # function early-return and the column ends up NULL (acceptable for
            # historical-import use case).
            "submitted_at": None,
            "submitted_at_iso": NOW_ISO,
            "api_url": DEFAULT_URL,
            "source": "manual-publish-2026-05-12",
            "source_run": source_run,
            "rank": rank,
        },
        "task": {
            "task_id": task_id or f"manual-publish-{rank:02d}",
            # Claw session UUID — same value the dashboard uses to deep-link
            # to the chat transcript. None for purely historical records
            # without a SaFE task linkage.
            "claw_session_id": claw_session_id,
            "model": model,
            "display_name": display_name,
            "submit_status": "imported",
            "final_status": final_status,
            "final_phase": None,
            "final_message": "",
            "detected": {
                "framework": framework,
                "precision": precision,
                "tp": tp,
                "params_b": params_b,
            },
            "overrides": {},
        },
        "metrics": metrics,
        "baseline": {
            "baseline_tput_per_gpu": baseline,
            "tp": tp,
            "conc": 64,
            "isl": 1024,
            "osl": 1024,
            "model": model,
        },
        "run_context": {},
        "sweep_points": [],
        "kernel_candidates": [],
        "kernel_optimizations": [],
        "kernel_summary": {},
        "artifacts": [],
        "source_files": {},
        "warnings": [],
        "extra": extra,
    }
    return rec


# ── Existing 20 (from the user's table — the canonical "20 already processed") ──
RECORDS_EXISTING_20: list[dict] = [
    _record(rank=1, model="Qwen/Qwen2.5-14B-Instruct-AWQ", display_name="Qwen2.5-14B-Instruct-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=14.8,
            baseline=825.1, optimized=1280.0, gain_pct=55.13,
            note="Best gain in original 20 — INT4 quant headroom"),
    _record(rank=2, model="deepseek-ai/DeepSeek-R1-0528", display_name="DeepSeek-R1-0528",
            framework="sglang", precision="FP8", tp=8, params_b=684.5,
            baseline=183.9, optimized=265.6, gain_pct=44.42,
            inferenceX_ref=149.3,
            note="per-GPU; +78% vs InferenceX MI300X"),
    _record(rank=3, model="Qwen/Qwen3-Coder-30B-A3B-Instruct", display_name="Qwen3-Coder-30B-A3B-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=30.5,
            baseline=2428.0, optimized=3060.1, gain_pct=26.00),
    _record(rank=4, model="openai/gpt-oss-120b", display_name="gpt-oss-120b",
            framework="vllm", precision="FP4", tp=1, params_b=120.4,
            baseline=4199.1, optimized=5176.7, gain_pct=23.28,
            inferenceX_ref=4172.1,
            note="+24% vs InferenceX MI300X"),
    _record(rank=5, model="Qwen/Qwen2.5-7B-Instruct", display_name="Qwen2.5-7B-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=7.6,
            baseline=5486.6, optimized=6719.8, gain_pct=22.47),
    _record(rank=6, model="Qwen/Qwen2.5-7B", display_name="Qwen2.5-7B",
            framework="sglang", precision="FP8", tp=1, params_b=7.6,
            baseline=5775.8, optimized=6452.8, gain_pct=11.72),
    _record(rank=7, model="Qwen/Qwen2.5-32B-Instruct-AWQ", display_name="Qwen2.5-32B-Instruct-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=32.8,
            baseline=734.97, optimized=779.45, gain_pct=6.05),
    _record(rank=8, model="Qwen/Qwen3-8B", display_name="Qwen3-8B",
            framework="sglang", precision="FP8", tp=1, params_b=8.2,
            baseline=4802.2, optimized=4887.2, gain_pct=1.77),
    _record(rank=9, model="mistralai/Mistral-7B-Instruct-v0.2", display_name="Mistral-7B-Instruct-v0.2",
            framework="sglang", precision="FP8", tp=1, params_b=7.2,
            baseline=4667.6, optimized=4740.0, gain_pct=1.55),
    _record(rank=10, model="Qwen/Qwen2.5-Coder-7B-Instruct", display_name="Qwen2.5-Coder-7B-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=7.6,
            baseline=6547.3, optimized=6613.1, gain_pct=1.01),
    _record(rank=11, model="meta-llama/Llama-3.1-8B", display_name="Llama-3.1-8B (base)",
            framework="sglang", precision="FP8", tp=1, params_b=8.0,
            baseline=4351.2, optimized=4381.6, gain_pct=0.70),
    _record(rank=12, model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", display_name="DeepSeek-R1-Distill-Llama-8B",
            framework="sglang", precision="FP8", tp=1, params_b=8.0,
            baseline=4335.9, optimized=4361.7, gain_pct=0.59),
    _record(rank=13, model="dphn/dolphin-2.9.1-yi-1.5-34b", display_name="dolphin-2.9.1-yi-1.5-34b",
            framework="sglang", precision="FP8", tp=1, params_b=34.4,
            baseline=2021.9, optimized=2030.7, gain_pct=0.44),
    _record(rank=14, model="meta-llama/Llama-3.1-8B-Instruct", display_name="Llama-3.1-8B-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=8.0,
            baseline=4349.0, optimized=4349.0, gain_pct=0.0),
    _record(rank=15, model="Qwen/Qwen2.5-Coder-7B", display_name="Qwen2.5-Coder-7B",
            framework="sglang", precision="FP8", tp=1, params_b=7.6,
            baseline=None, optimized=6170.4, gain_pct=None,
            note="single best — no clean baseline"),
    _record(rank=16, model="openai/gpt-oss-20b", display_name="gpt-oss-20b",
            framework="vllm", precision="FP4", tp=1, params_b=21.5,
            baseline=None, optimized=3393.1, gain_pct=None,
            note="baseline only; agent stopped before sweep"),
    _record(rank=17, model="Qwen/Qwen3-8B-AWQ", display_name="Qwen3-8B-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=8.2,
            baseline=None, optimized=3271.6, gain_pct=None,
            note="single best — no clean baseline"),
    _record(rank=18, model="deepseek-ai/DeepSeek-V3.2", display_name="DeepSeek-V3.2",
            framework="sglang", precision="FP8", tp=8, params_b=685.4,
            baseline=79.3, optimized=None, gain_pct=None,
            note="per-GPU baseline only; agent did not finish optimize"),
    _record(rank=19, model="deepseek-ai/DeepSeek-R1", display_name="DeepSeek-R1",
            framework="sglang", precision="FP8", tp=8, params_b=684.5,
            baseline=None, optimized=None, gain_pct=None,
            final_status="Failed",
            note="R1 register fail / R2 sandbox fail"),
    _record(rank=20, model="nvidia/Gemma-4-31B-IT-NVFP4", display_name="nvidia/Gemma-4-31B-IT-NVFP4",
            framework="vllm", precision="FP4", tp=1, params_b=20.9,
            baseline=None, optimized=None, gain_pct=None,
            final_status="Failed",
            note="NVFP4 = NVIDIA-only kernels; no AMD path"),
]

# ── 6 NEW from 2026-05-12 batch=10 (run 25749785697) ──
SOURCE_RUN = "https://github.com/AMD-AGI/Hyperloom/actions/runs/25749785697"

RECORDS_NEW_6: list[dict] = [
    # rank 21-26: keep the original manual-publish-NN task_id so luochen
    # UPSERTs (ON CONFLICT DO UPDATE) the rows we sent yesterday — only
    # adds claw_session_id without duplicating. The new task_id from SaFE
    # (opt-...) goes into the `safe_task_id` field of the run/task dict
    # for queryability via raw_result JSONB.
    _record(rank=21, model="mistralai/Mistral-7B-v0.1", display_name="Mistral-7B-v0.1",
            framework="sglang", precision="FP8", tp=1, params_b=7.2,
            baseline=4717.80, optimized=4751.69, gain_pct=0.72,
            actions=["--decode-attention-backend aiter"],
            note="validated cumulative gain +0.30%; SaFE task_id opt-aca13f1a-d210-48c5-9913-5e0b0a9fe699",
            source_run=SOURCE_RUN,
            claw_session_id="c06bce9f-cf0b-47a8-8c14-87f0ebbd9f45"),
    _record(rank=22, model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", display_name="DeepSeek-Coder-V2-Lite-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=15.7,
            baseline=4267.08, optimized=4464.96, gain_pct=4.64,
            peak_throughput=10969.47, peak_throughput_conc=256,
            note="peak @ CONC=256; SaFE task_id opt-bc50ef0a-1f7b-4d02-8a57-e92a0f7f34b7",
            source_run=SOURCE_RUN,
            claw_session_id="83998d6a-3efd-40de-a7cf-e991aab4705d"),
    _record(rank=23, model="Qwen/Qwen3-14B-AWQ", display_name="Qwen3-14B-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=14.8,
            baseline=1420.89, optimized=1552.48, gain_pct=9.26,
            peak_throughput=4821.10, peak_throughput_conc=256,
            note="peak @ ISL=512 OSL=512 CONC=256; SaFE task_id opt-ee18ff0f-74d0-447a-ba3a-3260308529a2",
            source_run=SOURCE_RUN,
            claw_session_id="f1d1b206-8601-47b5-aac8-41468ae4b814"),
    _record(rank=24, model="Qwen/Qwen2.5-Coder-14B-Instruct", display_name="Qwen2.5-Coder-14B-Instruct",
            framework="sglang", precision="FP8", tp=1, params_b=14.8,
            baseline=2818.36, optimized=2840.95, gain_pct=0.80,
            peak_throughput=4119.0, peak_throughput_conc=256,
            actions=["--schedule-conservativeness 0.5"],
            note="full Phase 0-10; 7 kernels optimized via Claude; SaFE task_id opt-fa9b0158-2ebb-435f-a96c-f7a9c9285722",
            source_run=SOURCE_RUN,
            claw_session_id="4b7dff5c-2bac-416b-a294-14bccd2e75ef"),
    _record(rank=25, model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", display_name="Qwen2.5-Coder-14B-Instruct-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=14.8,
            baseline=559.06, optimized=1971.41, gain_pct=252.5,
            peak_throughput=5341.15, peak_throughput_conc=256,
            note="GEN tok/s; total 1209.29 -> 4264.32 +252.7%; SaFE task_id opt-0608a9a8-eece-4c70-9127-63aab8e1e437",
            source_run=SOURCE_RUN,
            claw_session_id="9d5d0425-ec21-46c0-930e-cc2274b60177"),
    _record(rank=26, model="deepseek-ai/DeepSeek-V2-Lite-Chat", display_name="DeepSeek-V2-Lite-Chat",
            framework="sglang", precision="FP8", tp=1, params_b=15.7,
            baseline=5434.27, optimized=5697.57, gain_pct=4.85,
            peak_throughput=11984.04, peak_throughput_conc=256,
            note="peak @ CONC=256; SaFE task_id opt-0be44dec-a082-41e2-93b4-00ae18d18ba5",
            source_run=SOURCE_RUN,
            claw_session_id="31c42e52-086f-44c0-ab24-1c42167d3b97"),
    # rank 27 — 7th GHA success that user hadn't pasted chat data for; numbers
    # below come straight from /wekafs/users/697f4f.../qwen25-coder-32b-awq-opt/
    # ci_metrics.json (mtime 2026-05-12T21:00 UTC).
    _record(rank=27, model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", display_name="Qwen2.5-Coder-32B-Instruct-AWQ",
            framework="vllm", precision="INT4", tp=1, params_b=32.8,
            baseline=750.82, optimized=785.39, gain_pct=4.60,
            actions=["--kv-cache-dtype fp8"],
            note="best_config=--kv-cache-dtype fp8; mean_tpot 80.10ms -> 76.40ms",
            source_run=SOURCE_RUN,
            task_id="opt-c45c86c6-0dd4-4f2d-95de-f8cd4ef0581f",
            claw_session_id="de3af835-c5de-4e4e-bd6a-a5acb150f3e6"),
]

# ── 4 NEW from 2026-05-13 batch (run 25789927636 + 25795816178) ──────────────
# Data pulled from SaFE artifact API
# /api/v1/optimization/tasks/<id>/artifacts/download?path=hyperloom/ci_metrics.json
# for the 3 Succeeded tasks. Mistral's data was extracted live from the
# sandbox pod (now deleted) via kubectl exec; numbers match the
# ci_metrics.json the agent wrote to /workspace/hyperloom/ at 14:04 UTC.
SOURCE_RUN_2 = "https://github.com/AMD-AGI/Hyperloom/actions/runs/25789927636"
RECORDS_2026_05_13: list[dict] = [
    _record(rank=28, model="mistralai/Mistral-7B-v0.1", display_name="Mistral-7B-v0.1 (run 25795816178)",
            framework="sglang", precision="FP8", tp=1, params_b=7.2,
            baseline=4753.18, optimized=4758.11, gain_pct=0.10,
            actions=[
                "baseline: 4753.18 tok/s/GPU at TP=1 CONC=64 ISL=1024 OSL=1024 (sglang_mi300x.sh)",
                "profile: 92.4% GPU time in vendor C++ kernels (Tensile GEMM 29.4%, CK paged_attention 29.3%)",
                "kernel_opt k006 activation_kernels.cu: 1.10x micro speedup via exp2-based SiLU",
                "kernel_opt k007 rmsnorm_quant_kernels.cu: timed out after 63 min, no E2E impact",
                "param_sweep --decode-attention-backend aiter: +0.10% (within noise)",
                "model is heavily vendor-kernel-bound — limited optimization headroom",
            ],
            note="extracted live from sandbox pod /workspace/hyperloom before pod GC; SaFE marked Failed (canonical-path detection bug); SaFE task_id opt-7984731a-d1a2-412e-b959-26204ea32972",
            source_run="https://github.com/AMD-AGI/Hyperloom/actions/runs/25795816178",
            task_id="opt-7984731a-d1a2-412e-b959-26204ea32972",
            claw_session_id="33ffbfac",
            final_status="Succeeded"),
    _record(rank=29, model="Qwen/Qwen3-Coder-Next", display_name="Qwen3-Coder-Next",
            framework="vllm", precision="FP8", tp=4, params_b=79.7,
            baseline=4047.11, optimized=4178.16, gain_pct=3.24,
            actions=[
                "baseline: 4047.11 tok/s TP=4 CONC=64 BF16 torch.compile+CUDAGraph",
                "profile: 16.1% compute / 46.9% communication / 36.2% idle — communication-bound",
                "select_kernels: No actionable compute bottlenecks (GEMM 0.11%)",
                "backends_sweep 28/28 variants tested; best=--max-num-batched-tokens 32768",
                "variant_kv_fp8 -19.8% / variant_aiter_on -7.8% / variant_no_prefix_cache -7.6%",
            ],
            note="per-GPU baseline 1011.78, optimized 1044.54; SaFE task_id opt-457df9ae",
            source_run=SOURCE_RUN_2,
            task_id="opt-457df9ae-ddc3-45d2-a187-b187c655b56c",
            claw_session_id="ffad2e62-3cf2-4551-80bd-9466f54a3cdf"),
    _record(rank=30, model="Qwen/Qwen2.5-Coder-32B-Instruct", display_name="Qwen2.5-Coder-32B-Instruct (run 25789927636)",
            framework="sglang", precision="FP8", tp=1, params_b=32.8,
            baseline=2016.17, optimized=2035.69, gain_pct=0.97,
            actions=[
                "FP8 online dynamic quantization (--quantization fp8)",
                "Memory fraction static increased to 0.90",
                "Continuous decode steps set to 4 (--num-continuous-decode-steps 4)",
                "5 Triton kernels optimized via Claude — 0% E2E impact (91.4% vendor)",
                "Validated: KV cache FP8 (--kv-cache-dtype fp8_e4m3) is HARMFUL (-88%)",
            ],
            note="91.4% vendor-kernel-bound; SaFE task_id opt-6ec0eb01",
            source_run=SOURCE_RUN_2,
            task_id="opt-6ec0eb01-dba2-4e6f-bdbb-921fcc8329df",
            claw_session_id="d15821fc-3821-40d2-9024-16ae8c0fdc0a"),
    _record(rank=31, model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            display_name="NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            framework="vllm", precision="FP8", tp=1, params_b=30.0,
            baseline=None, optimized=None, gain_pct=None,
            actions=[
                "aborted: NemotronH (hybrid Mamba2 + Attention + MoE) cannot run on vLLM v0.19.0 + ROCm/MI300X",
                "AMD Triton MLIR CanonicalizePointers assertion failure in ssd_chunk_scan.py during prefill with ISL>=128",
                "VLLM_ROCM_USE_AITER_MOE must be 0 (RELU2_NO_MUL unsupported)",
                "torch.compile/CUDA graphs trigger separate MLIR failures",
                "Short prompts (<128 tokens) work but ISL=1024 fundamentally incompatible",
            ],
            note="partial-report-on-abort; model-stack incompatibility; SaFE task_id opt-26572b61",
            source_run=SOURCE_RUN_2,
            task_id="opt-26572b61-c277-4251-8094-7f45fe20ce84",
            claw_session_id="0e721f40-a6a9-4a9f-a598-0e80955b17be",
            final_status="Failed"),
]

ALL_RECORDS = RECORDS_EXISTING_20 + RECORDS_NEW_6 + RECORDS_2026_05_13


def _post(records: list[dict], url: str, token: str, timeout: int = 60) -> dict:
    import requests, urllib3
    urllib3.disable_warnings()

    endpoint = url.rstrip("/") + "/api/import"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"results": records}
    resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout, verify=False)
    out = {
        "endpoint": endpoint,
        "status_code": resp.status_code,
        "n_records": len(records),
    }
    try:
        out["body"] = resp.json()
    except Exception:
        out["body_text"] = resp.text[:500]
    return out


def _print_table(recs: list[dict]) -> None:
    print(f"{'#':>3} {'Status':10} {'Model':45} {'Frm':6} {'Prec':5} {'TP':>2} "
          f"{'Baseline':>10} {'Optimized':>10} {'Gain%':>8} {'Peak':>10} {'PeakConc':>9}")
    print("-" * 140)
    for r in recs:
        m = r["metrics"]
        t = r["task"]
        rank = r["run"]["rank"]
        ds = t["display_name"]
        if len(ds) > 44:
            ds = ds[:41] + "..."
        st = t["final_status"]
        bl = m["baseline_throughput"]
        op = m["optimized_throughput"]
        gp = m["gain_pct"]
        pk = m.get("peak_throughput")
        pc = m.get("peak_throughput_conc")
        none = "-"
        bls = f"{bl:>10.2f}" if bl is not None else f"{none:>10}"
        ops = f"{op:>10.2f}" if op is not None else f"{none:>10}"
        gps = f"{gp:>+8.2f}" if gp is not None else f"{none:>8}"
        pks = f"{pk:>10.2f}" if pk is not None else f"{none:>10}"
        pcs = f"{pc:>9}" if pc is not None else f"{none:>9}"
        tp_v = t["detected"].get("tp")
        tp_s = f"{tp_v:>2}" if tp_v is not None else f"{none:>2}"
        print(f"{rank:>3} {st:10} {ds:45} {t['detected']['framework']:6} "
              f"{t['detected']['precision']:5} {tp_s} {bls} {ops} {gps} {pks} {pcs}")


def _filter_by_name(recs: list[dict], name: str) -> list[dict]:
    needle = name.lower()
    return [
        r for r in recs
        if needle in r["task"]["display_name"].lower()
        or needle in r["task"]["model"].lower()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print payload only")
    ap.add_argument("--probe", action="store_true",
                    help="POST a tiny single-record payload to validate the endpoint")
    ap.add_argument("--send-all", action="store_true",
                    help="POST all 26 records as one batch")
    ap.add_argument("--send-only", default="",
                    help="POST one record by display_name / model substring")
    ap.add_argument("--write-json", default="",
                    help="Save the full 26-record payload to a JSON file")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    print(f"endpoint = {args.url.rstrip('/')}/api/import")
    print(f"token    = {'<set>' if args.token else '<unset>'}")
    print(f"records  = {len(ALL_RECORDS)} total "
          f"({len(RECORDS_EXISTING_20)} existing + "
          f"{len(RECORDS_NEW_6)} batch1 (5/12) + "
          f"{len(RECORDS_2026_05_13)} batch2 (5/13))")
    print()
    _print_table(ALL_RECORDS)

    if args.write_json:
        Path(args.write_json).write_text(
            json.dumps({"results": ALL_RECORDS}, indent=2), encoding="utf-8",
        )
        print(f"\nWrote payload to {args.write_json}")

    if args.dry_run:
        return 0

    if args.probe:
        # Use a real but tiny single-record probe with the most critical
        # fields filled, to find out whether the endpoint rejects the
        # full normalized shape because of size or because of schema.
        probe = [_record(
            rank=999, model="manual/probe", display_name="manual-probe",
            framework="sglang", precision="FP8", tp=1, params_b=0.1,
            baseline=1.0, optimized=2.0, gain_pct=100.0,
            note="endpoint probe — please ignore",
        )]
        out = _post(probe, args.url, args.token, args.timeout)
        print("\nProbe response:")
        print(json.dumps(out, indent=2))
        return 0 if out["status_code"] < 300 else 1

    if args.send_only:
        subset = _filter_by_name(ALL_RECORDS, args.send_only)
        if not subset:
            print(f"No records matched '{args.send_only}'", file=sys.stderr)
            return 2
        print(f"\nSending {len(subset)} record(s) matching '{args.send_only}'")
        out = _post(subset, args.url, args.token, args.timeout)
        print(json.dumps(out, indent=2))
        return 0 if out["status_code"] < 300 else 1

    if args.send_all:
        print(f"\nPOSTing all {len(ALL_RECORDS)} records to {args.url.rstrip('/')}/api/import")
        out = _post(ALL_RECORDS, args.url, args.token, args.timeout)
        print(json.dumps(out, indent=2))
        return 0 if out["status_code"] < 300 else 1

    print("\nNo --dry-run / --probe / --send-all / --send-only flag — nothing posted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
