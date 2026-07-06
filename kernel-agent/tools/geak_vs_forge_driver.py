#!/usr/bin/env python3
"""Glue driver that runs GEAK or Forge from a post-explore TraceLens report."""

import argparse
import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

# Derive the Hyperloom repo root; HYPERLOOM_REPO_ROOT overrides clone layout.
REPO = (
    Path(os.environ["HYPERLOOM_REPO_ROOT"])
    if os.environ.get("HYPERLOOM_REPO_ROOT")
    else Path(__file__).resolve().parents[2]
)
sys.path.insert(0, str(REPO / "kernel-agent" / "tools"))
sys.path.insert(0, str(REPO))

# Imports must follow sys.path setup above.
import tracelens_skill_runner as tlr  # noqa: E402  # HL
import tracelens_analysis as tla  # noqa: E402  # HL
from inference_optimizer.orchestrator import kernel_request_handlers as krh  # noqa: E402  # HL


def _load_fusion_cues(analysis_dir: Path) -> str:
    """Best-effort summary of fusion candidates from TraceLens category_data, or ''."""
    for rel in ("category_data/fusion_candidates.json", "category_data/kernel_fusion_metrics.json"):
        p = analysis_dir / rel
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        items = data if isinstance(data, list) else (data.get("candidates") or data.get("fusions") or [])
        names = []
        for it in (items if isinstance(items, list) else [])[:8]:
            if isinstance(it, dict):
                nm = it.get("name") or it.get("pattern") or it.get("fusion") or it.get("title")
                if nm:
                    names.append(str(nm))
        if names:
            return "; ".join(names)
    return ""


def _build_workload_context(c: dict, a, all_cands: list, fusion: str) -> str:
    """Assemble ALL workload context for one kernel (necessary + optional + additional).

    Pulls from the serving config (--serving-config), the candidate's own resolved fields
    (gpu_pct, roofline efficiency, AI, bound type, shapes), the sibling hot kernels, and
    fusion cues. Free-form prose; GEAK is told to use it to pin the harness and figure out the rest.
    """
    name = str(c.get("name") or c.get("op_name") or c.get("kernel_id") or "")
    lines: list[str] = []

    # --- Necessary: serving config (resolves the decode-context ambiguity) ---
    if a.serving_config:
        lines.append(f"Serving config (the workload these shapes were captured under): {a.serving_config}.")
        lines.append(
            "The captured tensor shapes encode a max-capacity KV allocation, NOT the live decode "
            "context — use ISL/OSL/concurrency above to pin the real decode regime; do not guess it."
        )

    # --- E2E / Amdahl framing (so a kernel win is judged against the real ceiling) ---
    gpu_pct = c.get("gpu_pct")
    if gpu_pct is not None:
        lines.append(
            f"This kernel is ~{float(gpu_pct):.1f}% of serve GPU time. The workload is host/decode-bound "
            f"at this config (GPU substantially idle), so the realistic E2E ceiling is bounded by Amdahl — "
            f"the win must survive into end-to-end serving throughput, not just isolated kernel latency."
        )

    # --- Roofline specifics already resolved by TraceLens (the exact gap to close) ---
    bound = c.get("bound_type")
    ai = c.get("flops_per_byte")
    eff = c.get("efficiency_percent")
    eff_unit = c.get("efficiency_peak_unit")
    eff_peak = c.get("efficiency_peak_value")
    roof = []
    if bound:
        roof.append(f"bound={bound}")
    if ai not in (None, 0, 0.0):
        roof.append(f"arithmetic_intensity={ai} FLOP/byte")
    if eff not in (None, 0, 0.0):
        roof.append(f"achieved={eff}% of roofline")
    if eff_peak not in (None, 0, 0.0):
        roof.append(f"peak={eff_peak} {eff_unit or ''}".strip())
    if roof:
        lines.append("Roofline (TraceLens-resolved): " + ", ".join(roof) + ".")

    # --- Neighbouring hot kernels (cross-op fusion opportunities), de-duplicated ---
    sibs, seen = [], set()
    for o in all_cands:
        if o is c:
            continue
        nm = str(o.get("name") or o.get("op_name") or "")
        if nm and nm not in seen:
            seen.add(nm)
            sibs.append(nm)
    if sibs:
        lines.append(
            "Neighbouring hot kernels in this workload (consider cross-op fusion, not just in-kernel tuning): "
            + ", ".join(sibs[:8])
            + "."
        )
    if fusion:
        lines.append(f"TraceLens fusion candidates: {fusion}.")

    if not lines:
        return ""
    header = f"Kernel `{name}` — additional context HL does not inject by default. Use it to build the harness on the real regime and to prioritise; figure out the rest from the source + shapes."
    return header + "\n- " + "\n- ".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--csv-dir", default="")
    ap.add_argument("--framework", required=True)
    ap.add_argument("--backend", required=True, choices=["geak", "forge"])
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--target-platform", default="MI300X")
    ap.add_argument(
        "--candidates", default="",
        help="Structured kernel_candidates.json to use DIRECTLY (its hot_kernels) "
             "instead of re-parsing analysis.md. Preserves the precise profiler op "
             "names that op_to_source resolution needs (analysis.md drops them), so "
             "composite ops (fused-MoE, gdn-attn) resolve to the editable hot source.",
    )
    ap.add_argument(
        "--only-op", default="", help="substring filter: keep only candidates whose op name matches (e.g. 'attention')"
    )
    ap.add_argument(
        "--enrich",
        action="store_true",
        help="append authoritative WORKLOAD CONTEXT (serving config + Amdahl + fusion + roofline) to each kernel's dispatch prompt",
    )
    ap.add_argument(
        "--serving-config",
        default="",
        help="free-form serving config string (e.g. 'framework=vllm, TP=8, ISL=1024, OSL=1024, conc=64'); injected verbatim when --enrich",
    )
    a = ap.parse_args()

    session = Path(a.session_dir)
    session.mkdir(parents=True, exist_ok=True)
    # Pin workspace + backend so HL's run-dir resolution + _backend_order agree.
    os.environ["USER_DATA_PATH"] = str(session)
    os.environ["HYPERLOOM_RUNTIME_DIR"] = str(session / "runtime")
    os.environ["KERNEL_OPT_BACKENDS"] = a.backend
    run_dir = session / "kernel-agent-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    md = Path(a.analysis)
    csv_dir = Path(a.csv_dir) if a.csv_dir else None

    # 1) HL: analysis.md -> candidates (parse + finalize with CSV-resolved shapes/source).
    # --candidates: use the STRUCTURED kernel_candidates.json hot_kernels directly so the
    # precise op names survive (analysis.md re-parse loses them -> composite ops can't resolve).
    if a.candidates:
        _cd = json.loads(Path(a.candidates).read_text())
        parsed = [c for c in (_cd.get("hot_kernels") or []) if isinstance(c, dict)][: a.top_k]
        print(f"[driver] loaded {len(parsed)} structured candidates from {a.candidates}", flush=True)
    else:
        parsed = tlr.parse_analysis_md(md, top_k=a.top_k)
    cands = tla._finalize_candidates(
        parsed,
        total_dur=None,
        perf_report_csv_dir=(csv_dir if (csv_dir and csv_dir.exists()) else None),
        framework=a.framework,
    )
    if a.only_op:
        cands = [c for c in cands if a.only_op.lower() in str(c.get("name") or c.get("op_name") or "").lower()]
        print(f"[driver] --only-op='{a.only_op}' -> {len(cands)} candidates kept", flush=True)
    editable = [c for c in cands if c.get("reusable_native_kernel") is True]
    print(f"[driver] candidates={len(cands)} editable={len(editable)}", flush=True)

    # 1b) OPTIONAL enrichment: attach authoritative WORKLOAD CONTEXT per candidate so GEAK's
    # harness-gen pins the TRUE serving regime instead of guessing the decode context (the proven
    # reason kernel wins fail to reach E2E). Pass everything we have; let GEAK figure out the rest.
    # Default-off: with no --enrich the candidate dict is untouched and the prompt is byte-identical.
    if a.enrich:
        analysis_dir = md.parent
        fusion = _load_fusion_cues(analysis_dir)
        for c in cands:
            c["extra_dispatch_context"] = _build_workload_context(c, a, cands, fusion)
        print(f"[driver] --enrich: attached extra_dispatch_context to {len(cands)} candidates", flush=True)

    # 2) HL: write kernel_candidates.json (the dispatch payload artifact)
    wr_args = Namespace(
        model_name=a.model,
        framework=a.framework,
        target_platform=a.target_platform,
        analysis_mode="default",
        runtime_env="local",
        dry_run=False,
        source_root=None,
        trace_input=str(md),
        roofline_json="",
    )
    artifacts = tla.write_reports(
        run_dir,
        trace_input_type="file",
        trace_files=[md],
        candidates=cands,
        args=wr_args,
        existing_report_path=md,
    )
    cand_path = artifacts.get("kernel_candidates") or str(run_dir / "kernel_candidates.json")
    print(f"[driver] kernel_candidates -> {cand_path}", flush=True)

    # 3) HL: batch dispatch (parallel, 1 kernel/GPU) on the chosen backend
    payload = {
        "candidates_path": cand_path,
        "backend_order": a.backend,  # comma-string, NOT a list (HL splits on ",")
        "model_path": a.model,
        "framework": a.framework,
        "target_platform": a.target_platform,
    }
    # Autonomous combined E2E: ask HL to apply ALL optimized patches together and
    # remeasure end-to-end after the batch finishes (GEAK-only; HL guards on
    # backend/model/GPU). Parse the structured serving knobs out of --serving-config
    # so HL serves at the same config the kernels were optimized for.
    if a.backend == "geak":
        payload["combined_e2e"] = True
        sc: dict = {"framework": a.framework}
        for tok in str(a.serving_config or "").split(","):
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "tp":
                sc["tp"] = int(v)
            elif k == "isl":
                sc["isl"] = int(v)
            elif k == "osl":
                sc["osl"] = int(v)
            elif k in ("conc", "concurrency"):
                sc["conc"] = int(v)
            elif k in ("num_prompts", "num-prompts"):
                sc["num_prompts"] = int(v)
            elif k == "framework":
                sc["framework"] = v.lower()
        payload["serving_config"] = sc
    res = asyncio.run(krh.run_optimization_handler(payload, session_dir=session))
    out = session / f"result_{a.backend}.json"
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"[driver] DONE backend={a.backend} -> {out}", flush=True)
    print(
        f"[driver] status={res.get('status')} best_speedup={res.get('best_speedup') or res.get('best_speedup_verified')}",
        flush=True,
    )
    ce = res.get("combined_e2e")
    if isinstance(ce, dict):
        if ce.get("status") == "ok" or "delta_pct" in ce:
            print(
                f"[driver] combined_e2e: baseline={ce.get('baseline_median_tok_s')} "
                f"patched={ce.get('patched_median_tok_s')} delta={ce.get('delta_pct')}% "
                f"(applied all best patches together)",
                flush=True,
            )
        else:
            print(f"[driver] combined_e2e: {ce.get('status')} - {ce.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
