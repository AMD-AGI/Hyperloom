# KernelForge examples

Runnable, end-to-end examples of KernelForge's forge-loop and cross-language
rewrite features. Each subdirectory contains its task files and a
`run_example.sh` launcher.

The softmax examples are single-kernel tasks (one file each) only because that is
the smallest thing that exercises the whole loop. Three additional tasks use
**real Hyperloom TraceLens hot kernels** from production serving traces on MI355X
(gfx950) — same layout, shapes and semantics from traced models, no AITER
dependency. forge-loop is **not** limited to single-file kernels — it also
optimizes multi-file operators and whole repositories (e.g. AITER) via
`--task-type repository` + `--source-files` (see §1 and §5). `mori_ep_dispatch_combine/`
goes further still: a genuine distributed 8-GPU multi-rank task, tuning a
launch-config file rather than kernel source, with no `graph_harness.py` (a
real collective can't be captured under one CUDA/HIP graph). Nothing below
assumes your kernel lives in a single file, or that a task is single-process.

```
examples/
├── README.md                          # this file — the shared standard task spec
├── triton-softmax-forge-loop/         # tutorial task (Triton)
├── gluon-softmax-forge-loop/          # tutorial task (Gluon, Triton's low-level dialect)
├── flydsl-softmax-forge-loop/         # tutorial task (FlyDSL)
├── aiter-allreduce-forge-loop/        # production collective (AITER, any rank count, 2 metric groups)
├── triton_mixtral_dynamic_quant/      # production hot kernel (Triton FP8 quant)
├── flydsl_gemma_rmsnorm/              # production hot kernel (FlyDSL RMSNorm)
├── hip_gemma_fused_add_rmsnorm/       # production hot kernel (HIP fused add+RMSNorm)
├── triton2flydsl-softmax-flydsl-rewrite/
├── triton2flydsl-mxfp8-grouped-gemm/  # SGLang MXFP8 MoE grouped GEMM rewrite
└── mori_ep_dispatch_combine/          # distributed op (MoRI-EP dispatch/combine, 8-GPU EP)
```

Each task directory ships the standard files (`<op>_kernel.py`, `driver.py`,
`graph_harness.py`, `program.md`, `run_example.sh` — `mori_ep_dispatch_combine/`
omits `graph_harness.py`, see §3) and **no pre-optimized kernel** — the
optimized code is what the loop produces in your workspace. Rewrite examples
similarly omit `kernel.py`; the pipeline creates and optimizes that FlyDSL
file in the scratch workspace.

## Available tasks

| Task | Backend | What it shows |
|------|---------|---------------|
| `triton-softmax-forge-loop/` | Triton | Complete driver contract: correctness, CUDA-graph benchmark, per-case timing, and kernel-only profiling. Eager ≈ graph because Triton's launch dispatch is light. |
| `gluon-softmax-forge-loop/` | Gluon | The same contract on Triton's low-level dialect, where the tile layout is explicit source rather than a compiler choice. The baseline is a deliberate v0 (`size_per_thread=1`, no vectorization), so the headroom is on the layout itself rather than on a Triton knob — measured 1.56× on the wide case, with the narrow case at the launch floor. Needs a Gluon-capable Triton on CDNA; the script preflights it and reports the per-generation `gl.amd` surface. |
| `flydsl-softmax-forge-loop/` | FlyDSL | The same complete contract plus the stream-routing + capture-validity guard needed to benchmark a self-managed-stream DSL honestly (eager ≈ 1.4–1.7× graph). |
| `triton_mixtral_dynamic_quant/` | Triton | Dynamic per-tensor FP8 quant from Mixtral-8x7B-Instruct-v0.1, shape `(64, 4096)` BF16 → FP8 E4M3FN. |
| `aiter-allreduce-forge-loop/` | AITER (HIP + Python) | A **repository** task on a collective: two independent dispatch thresholds scored as two separate metric groups. `NPROC` selects the rank count (default 2) and the default case suite brackets the 1-stage/2-stage crossover for it; `SUITE=tp4_wide` and `SUITE=tp8_k3` pin the configurations with a measured baseline. |
| `flydsl_gemma_rmsnorm/` | FlyDSL | Gemma RMSNorm from Gemma-4-26B-A4B-it, shape `(64, 2816)` BF16. |
| `hip_gemma_fused_add_rmsnorm/` | HIP | Fused residual-add + Gemma RMSNorm from Gemma-4-26B-A4B-it, shape `(64, 2816)` BF16. |
| `triton2flydsl-softmax-flydsl-rewrite/` | Triton → FlyDSL | Correctness-first softmax port followed by FlyDSL optimization. |
| `triton2flydsl-mxfp8-grouped-gemm/` | Triton → FlyDSL | SGLang MXFP8 grouped GEMM for MiniMax-M3 MoE on MI355X, covering decode and prefill. |
| `mori_ep_dispatch_combine/` | aiter (MoRI-EP) | Distributed 8-GPU multi-rank task: tune MoRI-EP dispatch/combine launch config (block_num, warp_per_block, kernel_type, buffer mode) for EP8 MoE all-to-all. No `graph_harness.py` — a real 8-process collective can't be captured under one CUDA/HIP graph; see `driver.py`'s docstring. |

Production tasks ship a correct-but-slow eager-Torch seed so forge can measure a
real `baseline_ms` before editing anything. That is also why they have obvious
headroom — the baseline materializes full-size fp32 temporaries and launches one
kernel per elementwise step.

## Run a task

```bash
# Prerequisites: Hyperloom installed (kernelforge on PATH), a GPU + the
# backend the task uses, and a configured Claude gateway:
export ANTHROPIC_BASE_URL=...        # your gateway
export ANTHROPIC_AUTH_TOKEN=...      # bearer token
# Only if the gateway wants more than the credential ("Name: value" per line):
# export ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: ${MY_SUB_KEY}'

# From the task directory:
cd flydsl-softmax-forge-loop
./run_example.sh                                    # scratch /tmp workspace

# Pick the workspace + tune the time budget:
MAX_HOURS=2 ./run_example.sh /tmp/my_run
```

Each `run_example.sh` copies its task into a scratch git workspace, `git init`s
it, and launches forge-loop with that task's kernel backend / target functions / task
type. GPU arch is autodetected via `rocminfo` (override with `GPU_TARGET=gfx942`).
The loop leaves the best-kept kernel in the workspace and writes its iteration
archive, profiles, and a machine-readable `forge_result.json` under
`forge_experiments/`. When it finishes:

```bash
cat  /tmp/my_run/forge_experiments/forge_result.json   # baseline_ms, best_ms, improved
ls   /tmp/my_run/forge_experiments/                    # iteration archive + profiles
python /tmp/my_run/driver.py --warmup 10 --iters 200 --bench-mode   # re-measure yourself
```

The best kept kernel is checked out in the workspace as `<op>_kernel.py`. Nothing
is written back into this repository.

### Docker (MI355X)

On an MI355X host with GPU device access, the SGLang ROCm image has torch, Triton,
and FlyDSL already installed:

```bash
export REPO_ROOT="$(pwd -P)"          # a Hyperloom checkout
export CASE=triton_mixtral_dynamic_quant

docker run --rm -it \
  --ipc=host --shm-size=16g \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
  -e ANTHROPIC_BASE_URL -e ANTHROPIC_AUTH_TOKEN \
  -v "$REPO_ROOT:/workspace" -w /workspace \
  docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix \
  bash -lc "
    python -m pip install . &&
    examples/$CASE/run_example.sh /tmp/forge_run &&
    cat /tmp/forge_run/forge_experiments/forge_result.json
  "
```

forge-loop drives a Claude CLI agent, so that CLI must be available inside the
container (install Node.js and the Claude CLI in the image if it is not).

### Reference results (production tasks)

Measured on MI355X with an earlier revision of these same operators (same shapes
and eager-Torch baselines; the timing harness has since been replaced with the
shared `graph_harness.py`). Treat as evidence of headroom, not a guaranteed
outcome — actual results depend on GPU, ROCm/Triton/FlyDSL versions, and the
agent's search path:

| Case | Baseline → best | Speedup | Correctness |
|---|---|---|---|
| Triton FP8 quant | 0.040921 → 0.014440 ms | 2.83× | bit-exact vs oracle |
| FlyDSL RMSNorm | 0.034641 → 0.014540 ms | 2.38× | bit-exact vs oracle |
| HIP fused add+RMSNorm | 0.038121 → 0.017460 ms | 2.18× | 51 dB SNR |

---

# Anatomy of a standard forge-loop task

A task is a directory of files plus a launch script, not a single config file.
A standard task directory contains:

| File | Required | Role | forge edits it? |
|------|----------|------|-----------------|
| kernel source | yes | What forge optimizes — the `--kernel` anchor (one file, or the entry file of a multi-file operator / repo). | **YES** — the edited target(s) |
| `driver.py` | yes | Measurement driver: correctness oracle + perf measurer. | never (protected) |
| `graph_harness.py` | **recommended** | Operator-agnostic CUDA/HIP graph timing harness — the default way to bench (see §3). | never (protected) |
| `program.md` | recommended | Free-form guidance for the agent. | never |
| `run_example.sh` | yes | Prepares a scratch git workspace and launches forge-loop for this task. | never |

Rules each file must satisfy:

## 1. The kernel source — what forge optimizes (`--kernel` anchor)

`--kernel` points at the **anchor** the driver exercises. A task may be a single
file (like the softmax examples) **or** span several files / an existing package.
For a multi-file operator or whole repo (e.g. AITER) pass `--task-type repository`
and list useful implementation entry points with `--source-files a.py,b.hip,...`.
These paths seed orientation, profiling, JIT handling, and KB identity; they are
not an edit allowlist. The anchor is just the entry point, and the agent may edit
any tracked implementation file outside the protected measurement surface.
Either way the same rules hold:

- **Stable public entry point.** Expose a fixed name + signature the driver calls
  (e.g. `softmax(x)`, a builder `build_softmax_module(M,N,dtype)`, or an existing
  package API for a repo task). The canonical driver detects incompatible edits.
- **Correct at baseline.** It must pass the driver's correctness gate as shipped.
  For a demo, start deliberately conservative (obvious headroom) so the loop has
  something to win; for a real operator, start from the current production code.
- **Same backend / language.** Do not let the agent rewrite the operator in a
  different framework (state this in `program.md`).
- If the kernel manages its own **stream** (e.g. a DSL launcher taking a `stream`
  arg), the *driver* must route the active stream in — see §3.

## 2. `driver.py` — the correctness oracle + perf measurer (protected)

forge treats the driver as a **black box** invoked as `python driver.py <args>`
and talks to it purely over **stdout**. It is the source of truth for
correctness, speed, and the workload replayed by hardware profiling, so the loop
**never edits it**. It must:

- Import the kernel by its stable public name and build inputs itself.
- Be **deterministic** (fixed seed) across repeated full-suite invocations.
- Exit `0` on success; any non-zero exit is treated as a crash.
- Handle all three modes and print the agreed lines (nothing else needs to match):

**Correctness** — `python driver.py`

The driver runs every scored correctness case and reports the suite verdict:

```
SNR: 62.13 dB          # preferred; forge gates on this vs the SNR threshold
allclose: True         # optional fallback if you cannot compute an SNR
```

**Benchmark** — `python driver.py --warmup <n> --iters <n> --bench-mode`

The driver runs every scored benchmark case:

```
wall_ms: 0.081920      # one line per timed iteration; forge takes the median
case_ms: case_001 0.081920
```

or a single pre-aggregated line instead of per-iteration samples:

```
median_ms: 0.081920    # (or) mean_ms: 0.081920 — label it honestly
case_ms: case_001 0.081920
```

**Profiling** — `python driver.py --profile-run`

The driver selects one representative profile case and runs only its target
kernel without the reference implementation, correctness checks, or timing
output. Perform enough warmup to settle JIT compilation/autotuning, then 1-3
target launches, synchronize, and exit `0`. Hardware profilers replay the
process once per counter group, so a large loop in this mode multiplies
collection time.

Benchmark and profiling have separate responsibilities:

- benchmark mode prints one `case_ms: <case_id> <ms>` line for every case it
  measured;
- `<case_id>` is an opaque, whitespace-free token owned by the driver;
- profile mode owns its representative-case policy and never accepts a case
  selector from Forge.

With the default `--prepare-task`, this profiling interface is mandatory.
Preflight executes `--profile-run` directly. If task preparation cannot make
correctness, graph timing, benchmark output, and profiling all pass, forge-loop
fails before optimization. A verified prepared driver is used directly by all
baseline and post-KEEP profiling runs.

Notes:
- Forge does not select shapes or cases. The driver owns the complete
  correctness suite, benchmark suite, and representative-case policy.
- Once the baseline emits per-case timings, every candidate must emit all of
  those cases. Missing coverage is rejected rather than scored by raw mean.
- The pristine baseline uses per-case medians from three measurements. Each of
  three candidate runs is scored independently as
  `mean(pristine_case_ms / candidate_case_ms)`, and the *mean* of those three
  scores must beat the current best by at least `t * sigma / sqrt(3)` -- a
  one-sided 95% Student-t test on the candidate's own scatter -- floored at
  0.1% of the current best. A kernel that measures quietly earns a small gain;
  a noisy one has to show more.

## 3. `graph_harness.py` — time under a CUDA/HIP graph (protected, use by default)

**This is the single most important measurement decision — treat graph timing as
the default, not an add-on.** A GPU kernel's wall time, especially a small or
latency-bound one, is dominated by **host-side launch/dispatch overhead** (Python
→ framework → launch), not by the GPU work. Benchmarking in plain eager mode is
actively harmful to the loop:

- the agent optimizes the **wrong thing** — it chases host dispatch cost it cannot
  actually change, while real GPU wins get buried in launch-overhead noise;
- the numbers are **noisy and not comparable** across iterations, so the loop's
  keep/revert decisions (which compare wall times) become unreliable;
- it **does not reflect production** — on AMD serving these ops run under a HIP
  graph, where per-launch host cost is already amortized away.

`graph_harness.py` fixes this by capturing ONE invocation into a CUDA/HIP graph
and timing graph *replays*: CUDA events bracket only the GPU stream, so the host
replay-launch cost is excluded and you measure GPU execution — the quantity that
actually decides whether a change is faster. It is **operator-agnostic**: pass a
zero-arg `step` closure that runs one invocation on pre-allocated tensors (see its
docstring for the replay-safety contract) and reuse the file as-is across tasks.

**Two correctness requirements when a task uses it:**

1. **Launch on the current stream.** `torch.cuda.graph` records work on a private
   *capture stream*. A kernel launched on the default/NULL stream (common for DSLs
   that manage their own stream) is **not recorded** → a silently EMPTY graph
   whose replay takes microseconds regardless of problem size (a fake "speedup").
   The **driver** (not the kernel) must route the active stream into the launch,
   e.g. `launch_fn(..., stream=fx.Stream(torch.cuda.current_stream().cuda_stream))`,
   queried at call time. Keeping this in the protected driver means the agent
   cannot break capture by editing the kernel.
2. **Verify capture.** Pass `dirty`/`verify` closures to `cuda_graph_bench`; after
   capture it corrupts the output, replays, and checks the result is correct. An
   empty/invalid graph fails the check and the harness falls back to eager timing
   with a `# bench mode: eager (...)` line instead of reporting bogus numbers.

Keep graph timing on even when an op looks graph-neutral: it costs nothing and
keeps the numbers honest and comparable. How much it matters scales with host
cost — light for a Triton launch (eager ≈ graph), heavy for a FlyDSL launch
(eager ≈ 1.4–1.7× graph on this softmax), and larger still for multi-launch
operators or repository tasks (e.g. AITER) that fire many kernels per call, where
the host gaps between launches compound into a big eager-vs-graph gap.

**Why there are multiple identical copies of `graph_harness.py`.**  Six of the
example directories contain a byte-for-byte identical `graph_harness.py` (195
lines, same md5).  This duplication is intentional: each example is designed as a
self-contained, copy-as-a-whole reference task.
`loop/task_preparer.py`'s `_materialize_reference()` copies the entire `examples/`
tree into the agent's workspace so it can read real, complete reference tasks
(driver, harness, README contract) rather than truncated prompt text.  The
agent's driver imports `from graph_harness import …` relative to its own
directory, so each task must be self-contained rather than sharing a root-level
file.  Do not de-duplicate these files.

Note: `triton2flydsl-mxfp8-grouped-gemm/` contains a shorter 94-line variant
with a different md5 (it benchmarks a different op shape).
`mori_ep_dispatch_combine/` has no `graph_harness.py` at all — a real 8-process
collective cannot be captured under a single CUDA/HIP graph; see that task's
`driver.py` docstring.

## 4. `program.md` — agent guidance

Free-form markdown handed to the optimizing agent: the objective, optimization
ideas (framed as hypotheses to measure, not prescriptions), and the hard rules —
above all: keep the public entry-point signature, stay in the backend, and do NOT
edit `driver.py` / `graph_harness.py` (the loop blocks edits to them anyway).

## 5. `run_example.sh` — the launch script

A small script that isolates the task in a scratch git workspace and launches
forge-loop for it. Every task's script follows the same shape; only the task-
specific flags differ. It must:

- Copy the task files into a scratch workspace and `git init` + commit it
  (forge-loop's keep/revert relies on git; leave build artifacts and
  `forge_experiments/` untracked so a revert never fails on a dirtied tree).
- Launch `kernelforge forge-loop` with this task's settings — the ones that
  vary per task are:

```sh
--kernel-backend flydsl                          # <backend>
--kernel  .../softmax_kernel.py                 # the edited anchor
--driver  .../driver.py                         # the protected driver
--program-md-file .../program.md                # agent guidance
--target-functions build_softmax_module,softmax_kernel  # PMC hints + shown to agent
--task-type flydsl2flydsl                       # task type; omit for the default
--snr-threshold 30.0                            # correctness gate (dB)
```

For a **repository** task (multi-file operator / whole repo, e.g. AITER) also pass
`--task-type repository` and `--source-files a.py,b.hip,...` with the best-known
implementation entry points. The list is a hint rather than an edit boundary;
`--kernel` stays the anchor/entry point.

`GPU_TARGET` is autodetected via `rocminfo` (override by exporting it); the
campaign budget is controlled by `MAX_HOURS`. Copy an existing task's
`run_example.sh` and change that value for a new task.

---

## Add your own task

1. Create a directory `examples/<name>/`.
2. Add your kernel/operator (single file, or a multi-file operator / repo; keep a
   stable public entry point) and a `driver.py` that meets the stdout contract in
   §2. Bench through `graph_harness.py` (§3) — copy it as-is and route the current
   stream in the driver.
3. Write `program.md` (§4).
4. Copy the closest existing `run_example.sh` (softmax for a tutorial task, or one
   of the production tasks for a real hot kernel) and update the task-specific
   flags (§5; for a multi-file operator add `--task-type repository` +
   `--source-files`), then run it: `./run_example.sh`.
