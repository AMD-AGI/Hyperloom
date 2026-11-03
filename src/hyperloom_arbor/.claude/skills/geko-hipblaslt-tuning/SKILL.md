---
name: geko-hipblaslt-tuning
description: >-
  Set up and run geko hipBLASLt GEMM tuning for DeepSeek-V4.1-Flash + vLLM/AgentX serving on AMD gfx950
  (MI355X), then verify and interpret the result. Use when installing geko in a container, applying the
  hipBLASLt 1.5.0 msgpack + scale_type patches, building a merged device library, running geko --tune,
  confirming the tuned kernels actually fire in serving, or deciding whether hipBLASLt tuning is the right
  lever (vs aiter). Triggers: "install geko", "geko --tune", "hipBLASLt tuning", "tune decode GEMMs",
  "verify tuned kernels fire", "geko full_libs", "why is my tune flat".
allowed-tools: Bash(docker *), Bash(sbatch *), Bash(squeue *), Bash(scancel *)
---

# geko hipBLASLt tuning for DSV4.1-Flash serving (gfx950)

Tune the critical decode hipBLASLt GEMMs for DeepSeek-V4.1-Flash on vLLM/AgentX, deploy them, and **prove the
tuned kernels are actually used** — apple-to-apple against the stock library.

## When to use
- Installing geko into a fresh `rocm/vllm-private` container and running `geko --tune`.
- Hitting `geko --tune` end-to-end errors (msgpack / scale_type — see the 3 patches below).
- A tune produced big GEMM-level uplift but **no e2e change** → verify the kernels fire + check the budget.
- Deciding whether hipBLASLt tuning is worth it for this workload (short answer below).

## Decision first: is hipBLASLt tuning even the lever?
Measured GPU-kernel-time budget for DSV4.1-Flash AgentX serving: **hipBLASLt `Cijk_*` ≈ 3.5–3.8%**. The real
cost is the **aiter** umbrella ≈ 56% (MoE experts `mfma_moe*_afp8_wfp4` ~34%, dense `_mxfp8_linear` ~22%) plus
TP all-reduce ~15%. So geko hipBLASLt tuning is **flat end-to-end at every concurrency even with tuned kernels
verified firing (+17–18% GEMM-level)**. If the goal is e2e throughput, tune **aiter** (its native autotuner),
not hipBLASLt. Still proceed below if you specifically want the hipBLASLt tune + a rigorous usage proof.

## The 3 critical `geko --tune` patches (hipBLASLt 1.5.0)
`geko --tune` fails e2e without these. Idempotent; resolve `$TENSILE` (tensilelite `Tensile/` dir) and
`$GEKO` (`python3 -c "import geko,os;print(os.path.dirname(geko.__file__))"`) first.
1. **Tensile/Tensile.py** — `--library-format` default `yaml → msgpack` (1.5.0 loads **msgpack** device libs).
2. **Tensile/GenerateSummations.py** — hardcoded `--library-format=yaml → msgpack` (+ `Common/GlobalParameters.py` default).
3. **geko/library/library.py** — `create_bench_input()` must set `common["scale_type"] = compute_type`
   (1.5.0 bench requires `scale_type`). Already present at ~line 427 in recent geko; verify, add only if missing.

Full copy-paste block + why-each: **[references/install.md](references/install.md)** (Step 3).

## Workflow (one change per step; validate each)
1. **Install** geko in the container — `[references/install.md](references/install.md)` (and run `scripts/geko_install.sh` inside the container; needs `apt install llvm-14-dev libomp5-14` for the client build).
2. **Profile the decode shapes** at your TARGET concurrency: launch with `--enforce-eager` + `HIPBLASLT_LOG_MASK=32`, drive the load, take the dominant M among K=5120 GEMMs. **Copy the exact types verbatim** — DSV4.1 decode is bf16-in / **f32-out** (`c_type=d_type=f32_r`); a bf16-out workload silently never fires (PITFALL #0).
3. **Tune**: `geko --tune --workload-log <shapes.yaml>` → `full_libs`. Merge into the full stock gfx950 logic and rebuild via `TensileCreateLibrary` → a complete `merged_devlib` (deploying `full_libs` alone is a sparse library that regresses). See `[references/tune-run.md](references/tune-run.md)`.
4. **Deploy + serve**: `HIPBLASLT_TENSILE_LIBPATH=<merged_devlib>/library/gfx950` on BOTH baseline(stock) and tuned so only the library differs.
5. **Verify + record** — below.

## Verifying the tuned kernels actually fire (do this every time)
Two independent checks — the deterministic one is complete, the trace one is a cross-check:
- **Deterministic selection (window-independent):** `OUTDIR=<run> scripts/verify_selection.sh` benches each kept shape merged-vs-stock with the exact serving types. Healthy = `tuned-selected = N of N`.
- **Serving-trace firing:** `python3 scripts/verify_kernels_in_trace.py` — the exact `kernel_tuned` names appear as `Cijk_*` events in the tuned trace, 0 in baseline. A partial count is fine (short window). **Do NOT** use the bare `_UserArgs_` tag — stock kernels carry it too.

Worked result (conc16, f32 workload): selection 27/27, trace 9–10/27 → PASS; e2e flat. With the wrong (bf16-out) workload: 0/27 — the canonical signature-mismatch failure.

## Pitfalls & scale gotchas (read before a long run)
- **Signature-exact match** (PITFALL #0): geko kernels are `Equality` — they fire only on an exact
  `(M,N,K,transA/B,a/b/c/d_type,compute)` match. Tune the M of your target concurrency **and** the real
  output dtype. Details + the dtype trap: **[references/pitfalls.md](references/pitfalls.md)**.
- **A tune degrades the node → serve on a FRESH node** (same-node server crashes at cudagraph capture).
- VRAM drain between serving phases; NFS build races / stale `build/`; run scripts from `/tmp`; bad nodes
  (`/dev/dri`); `pkill -f geko` self-kill. All in **[references/pitfalls.md](references/pitfalls.md)**.
- Container must be `--ipc=host`; client build needs LLVM. Known issues: **[references/issues.md](references/issues.md)**.

## Files
- `references/install.md` — container + geko install + the 3 patches (Steps 1–3).
- `references/tune-run.md` — smoke, full tune, merged-lib build, save/deploy (Steps 4–6).
- `references/pitfalls.md` — PITFALL #0 (M + output dtype), verification, scale gotchas, "is geko the lever".
- `references/issues.md` — known setup issues + a reference run.
- `scripts/geko_install.sh`, `scripts/verify_selection.sh`, `scripts/verify_kernels_in_trace.py`
  (set `WS` to the geko_agentx workspace root; run inside the container).
