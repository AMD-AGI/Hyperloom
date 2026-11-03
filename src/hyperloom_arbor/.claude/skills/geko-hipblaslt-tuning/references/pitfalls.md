# geko pitfalls + verifying tuned kernels actually fire

## PITFALL #0 (most important): the tuned M must match your serving decode M

This is the single easiest way to get **zero benefit from a correct tune** — and it fails *silently*. We hit it:
a trace of a real serving run showed only **1 of 42** tuned kernels ever firing, not because the library was
missing, but because we tuned one regime's M and ran the server at another regime's M.

### Why it happens

1. **geko kernels are exact-match.** A geko tune emits `LibraryType: Equality` entries — each tuned kernel is
   bound to one exact `(M, N, K)`. A GEMM whose `(M,N,K)` does not match an entry **silently falls back to the
   stock kernel**. No error, no warning, serving looks completely normal.
2. **For decode GEMMs, M is not a model constant — it is a runtime property of how you serve.** `N` and `K` are
   fixed by the model weights, but `M` (the batched-token dimension) is set by: concurrency → batch size →
   **cudagraph capture bucket** (the batch is padded up to the nearest captured graph size). M belongs to the
   serving regime, not the model.
3. **The concurrency→M map is non-linear — measure it, never guess.** Measured for DSV4.1-Flash:
   `conc1 → M=128–1024`, `conc4 → M=32`, `conc16 → M=384`, `conc32 → M=384`. It jumps between buckets.
4. **cudagraph hides M.** In normal serving the decode runs inside HIP graphs, and **graph replay does not
   re-log hipBLASLt calls**, so you cannot read the decode M from a normal server. Use `--enforce-eager` +
   `HIPBLASLT_LOG_MASK=32` to discover it.

### The concrete failure we saw

Library tuned for **conc4 (M=32)** — shapes like `(M=32, N=768, K=5120)`, `(M=32, N=672, K=5120)` — was loaded
into a server driven at a **conc16 load**, whose decode padded to **M=384** — shapes like `(M=384, N=16380)`,
`(M=384, N=96)`, `(M=384, N=54)`. Because `384 ≠ 32`, every M=384 GEMM fell back to stock; only one stray M=32
GEMM (from a light step) fired a tuned kernel → **1/42**.

### The correct workflow (do this, in order)

1. **Measure the decode M at your TARGET serving concurrency first** (`--enforce-eager` + `HIPBLASLT_LOG_MASK=32`,
   drive the real load, take the most frequent M among the K=5120 decode GEMMs).
2. **Tune that M's shapes** (the Equality kernels will be pinned to exactly that M).
3. **Verify the kernels fire under the SAME concurrency**, two ways:
   - *Deterministic:* `hipblaslt-bench --print_kernel_info` at the serving `(M,N,K)` — merged lib must select the
     tuned kernel (e.g. `MT16x16x256`), stock selects a different one.
   - *In-serving:* capture a torch-profiler trace under that load and confirm the tuned `..._UserArgs_...` kernel
     names appear as dispatched `Cijk_*` events (these kernels exist only in the merged lib — stock never emits
     them, so even their presence proves the lib is live).
4. **Expected healthy result:** a high tuned-kernel hit-rate in the trace. A hit-rate near zero (e.g. 1/42) is
   the signature of this pitfall — check that the trace's concurrency matches the tune's M.

> One tune does not cover all concurrencies. If you serve at multiple concurrencies, either tune each regime's M,
> or accept that the tune only helps at the one concurrency whose decode M it was built for. (For DSV4.1-Flash,
> conc16 and conc32 happen to share M=384, so one tune covers both; conc1 and conc4 each need their own.)

### The trap is the WHOLE signature, not just M — especially the output dtype (c_type/d_type)

M is the most common mismatch, but geko's Equality match keys on the **entire** GEMM signature: `(M, N, K, transA,
transB, a_type, b_type, c_type, d_type, compute_type)`. If **any** field differs from what serving actually
dispatches, the tuned kernel does not fire — stock fallback, silently.

**Hard-won example (2026-10-02):** a tune produced 25 kept kernels with a real +12.6% GEMM uplift, yet a trace of
the tuned serving run showed **0 of 25** firing. Ground truth from the eager `HIPBLASLT_LOG`: every real
DSV4.1-Flash M=384 decode GEMM is `--a_type bf16_r --b_type bf16_r --c_type f32_r --d_type f32_r` — i.e. **bf16
in, f32 OUT**. But the tuning workload yaml had `c_type: bf16_r, d_type: bf16_r` (bf16 out). So geko tuned the
`...BBS...` (bf16-out) kernel family while serving dispatches `...BSS...` (f32-out) — the kernel-name type code
(`BBS` vs `BSS`) differs, the Equality match never hits, and the tune is a silent no-op. This single wrong field
had masqueraded as "geko gives no e2e benefit" across *every* earlier run.

**Rules:**
- **Build the workload from the real `HIPBLASLT_LOG` lines verbatim** — copy `a_type/b_type/c_type/d_type/
  compute_type/scale_type/bias_type` exactly as logged. Do NOT hand-write or assume bf16-in ⇒ bf16-out; MoE/DSA
  decode GEMMs commonly accumulate to **f32 output**.
- **Read the kernel-name type code as the signature fingerprint.** `Cijk_Alik_Bljk_<TYPES>_...`: the `<TYPES>`
  block (e.g. `BBS` vs `BSS`) encodes (inputDtype, outDtype, computeDtype). If the tuned kernels' code differs
  from the ones in the serving trace, you tuned the wrong variant.
- **Verify the tuned-kernel FULL name appears in the serving trace** (not just "some `_UserArgs_` kernel" — that
  tag is NOT geko-specific; stock Tensile kernels carry it too, ~20-30 per trace). The right check is: the exact
  `kernel_tuned` names from `results/final_results.csv` appear as dispatched `Cijk_*` events in the tuned trace
  AND are absent from the baseline trace. 0 overlap = the tune never fired (check M AND dtypes).

---

## Verifying the tuned kernels actually fire in serving (do this every time)

A tune that produces big GEMM-level uplift means nothing if the kernels never run in serving. Two independent
checks — use BOTH:

1. **Deterministic kernel-selection (window-independent, the strong one).** For every kept shape, bench the
   merged lib vs the stock lib with the shape's EXACT serving types and compare the selected kernel:
   `hipblaslt-bench -m <M> -n <N> -k 5120 --transA T --transB N --a_type bf16_r --b_type bf16_r --c_type f32_r
   --d_type f32_r --compute_type f32_r --print_kernel_info`. If merged selects a different (tuned) kernel for a
   shape, that kernel fires every time serving dispatches it — and every tuned shape occurs in serving (the
   workload was extracted from the serving profile). Healthy result: `tuned-selected = N of N`. See
   `experiments/verify_selection.sh` (reads shapes+types from `results/final_results.csv`).
2. **Serving-trace firing (cross-check).** Capture a vLLM torch-profiler trace of the tuned serving run; confirm
   the exact `kernel_tuned` names appear as dispatched `Cijk_*` events AND are absent from the baseline trace.
   Expect a *partial* count (an 8 s window only exercises some shapes) — that is fine; (1) is the complete
   statement. Do NOT use the bare `_UserArgs_` tag as the signal — stock Tensile kernels carry it too (~20-30
   per trace). See `experiments/verify_kernels_in_trace.py`.

Worked result (conc16, f32-corrected workload): selection 27/27, trace firing 9-10/27 → PASS. With the WRONG
(bf16-out) workload the same checks gave 0/27 — the canonical signature-mismatch failure.

## Running the experiment at scale — hard-won gotchas (2026-10)

The cluster now rejects node-holding sbatch (`while true`); use self-contained per-task sbatch that ends. Beyond
that, these cost real hours to find:

- **A geko tune degrades the node's GPUs → a vLLM server launched on the SAME node right after crashes at
  cudagraph capture** (`WorkerProc ... exit code: None`). The merged lib is fine — proven by serving it cleanly
  on a FRESH node. **Fix: split tune and serve onto separate nodes** (tune job writes the merged lib to NFS; a
  second job on a fresh node serves it). A same-node GPU-recovery (drain all VRAM) is unreliable.
- **VRAM does not free instantly when a server is killed.** Between two serving phases on one node, poll
  `rocm-smi --showmeminfo vram` until GPU0 Used < 5 GB before launching the next server, else it OOMs at
  `request_memory` ("Free memory ... less than desired gpu_memory_utilization").
- **Shared-NFS build races / stale artifacts.** rocisa and geko are pip-built in the shared NFS source tree.
  Concurrent builds corrupt each other (`cannot import name 'rocIsa'`), and a prior container's root-owned
  `build/` blocks a later build (`[Errno 17] File exists: .../geko-0.1.0.dist-info`). **Fix: `rm -rf build dist
  *.egg-info` before each pip-install, and run ONE geko/tune build at a time.** Verify rocisa from `/tmp`, not
  the source dir (the local `./rocisa` package shadows the installed one and lacks the compiled `.so`).
- **NFS stale file handle mid-run.** An hours-long orchestration script read from NFS can abort with "Stale file
  handle". Copy it into the container (`cp run_cycle.sh /tmp/rc.sh && bash /tmp/rc.sh`) so execution is local.
- **Bad nodes:** some nodes are missing `/dev/dri` → `docker run` fails instantly. Keep an `--exclude` list.
- **`pkill -f` self-kill:** a pattern like `pkill -f geko` matches your own orchestration process (its cmdline
  contains the `geko_agentx` path). Only pkill SPECIFIC binary names (`TensileCreateLibrary`, `hipblaslt-bench`).
- **geko client build needs LLVM:** the ROCm image has no `LLVMConfig.cmake`; `apt-get update && apt-get install
  -y llvm-14-dev libomp5-14` makes `invoke build-client` succeed. Add a fail-fast `which geko` after install so a
  broken setup aborts in ~2 min instead of after the 1 h baseline.

## Context: is geko even the right lever? (measure first)

For DSV4.1-Flash AgentX serving, geko hipBLASLt tuning is **flat end-to-end at every concurrency** even with the
tuned kernels verified firing (+17-18% GEMM-level). Root cause (vLLM torch-profiler GPU-time breakdown):
hipBLASLt `Cijk_*` is only **~3.5-3.8%** of GPU kernel time. The real cost is the **aiter** umbrella (~56%): MoE
experts `mfma_moe*_afp8_wfp4` ~34%, dense linears `_mxfp8_linear` ~22%; plus TP all-reduce (`cross_device_reduce`)
~15%. Tune aiter (its native autotuner: `csrc/*/*_tune.py` + `aiter/configs/*.csv`), not hipBLASLt, to move e2e.

