---
myst:
  html_meta:
    "description": "Fuse launch-bound decode chains in sglang or vLLM with kernelforge forge-fuse, from trace diagnosis through the serving smoke."
    "keywords": "KernelForge, kernel fusion, forge-fuse, decode, launch-bound, sglang, vLLM, CUDA graph, Triton, ROCm"
---

# Fuse a launch-bound decode path

`kernelforge forge-fuse` attacks a different bottleneck from the rest of
KernelForge. The other kernel backends make one kernel faster. Fusion assumes the
kernels are already fast and goes after what is left: a long tail of tiny
operations -- residual adds, RMSNorm, RoPE, activations, cache writes -- each
paying a full launch on every decode step. Collapsing a chain of them into one
Triton kernel buys back the launches.

## When it is worth running

Capture a kineto trace **with CUDA graphs disabled**. With graphs on, replay has
already amortized the launches you are trying to count and the tail vanishes
from the trace.

The diagnosis reports `launch_bound_share`, the fraction of GPU-busy time spent
outside GEMM, attention and MoE. Below 0.10 the workload is compute dominated
and no decode fusion will pay. The predicted end-to-end gain discounts that
share for what CUDA-graph replay already recovers; below 3% an authoring
campaign is not worth its time.

Both numbers rank candidates rather than veto them. A modest share with one
clearly fusible chain beats a large share with nothing fusible in it.

## Running it

```bash
kernelforge forge-fuse \
    --trace decode.trace.json.gz \
    --model-path /models/LFM2-8B \
    --framework sglang \
    --output-dir /work/fusion-lfm2
```

Diagnose without touching a GPU or an agent first:

```bash
kernelforge forge-fuse ... --dry-run
```

That writes the manifest with the localized recipe skeleton so you can see which
chain would be attempted and where in the framework source it lives.

## What happens

1. **Diagnose** the trace into a launch-bound share and a predicted gain.
2. **Discover** which chain to fuse, either by matching the pattern library
   (`--discover patterns`, the default) or by letting an agent read the trace
   and the real source (`--discover llm`).
3. **Claim an existing pass.** If a vLLM compile pass already covers the chain,
   flipping its default on and running a serving A/B is cheaper than authoring
   anything, so that shortcut runs before the loop.
4. **Author and validate** each ranked recipe as one `forge-loop` campaign, using
   the fusion kernel_backend. The loop iterates, gates correctness at SNR >= 30 dB,
   benchmarks three times, and commits or reverts.
5. **Serving smoke** on whatever the loop kept: boot the real framework with
   CUDA graphs **on** and run decode.
6. **Export** a patch and write `fusion_manifest.json`.

The serving smoke exists because the kernel-level gates cannot see the failure
that matters most. Parity and the microbench run on small shapes with no graph
capture, so a kernel that allocates or host-syncs per call passes both and then
takes down the scheduler decode loop. Set `FORGE_FUSION_SERVING_CHECK=0` to skip
it when you only want the kernel-level verdict.

## How the framework tree is handled

The loop keeps and reverts candidates with git, only sees tracked files, and
treats its commits as the deliverable — it expects the caller to hand it a
workspace it may write history into. Fusion cannot hand it a copy, because the
benchmark and the serving smoke both have to import the framework from its real
install path, so it edits the live tree and isolates the git side instead. For
the same reason a fusion campaign always runs `--lanes 1`: a lane is a workspace
copy measured on its own, and a lane's edit would never reach the tree the
benchmark imports.

Every git call the campaign makes is pointed at a repository under the run's
output directory (`shadow.git`) with the framework tree as its work tree. No
`.git` and no `.gitignore` is written into the framework, so a framework that is
your own checkout keeps its history and its branches untouched. Only the
framework package is indexed — not the wheels installed beside it — and the run
restores the tree to the state it found before exporting its patch.

Because the loop can only commit files that were already tracked, the pipeline
also decides where the fused kernel goes: it creates that module empty, commits
it into the baseline, and names it in the task document as the only file the
author may write. A kernel written anywhere else would be scored and then lost.

## What the agent is told

The durable discipline -- CUDA-graph safety, fp32 accumulation inside the
kernel, one launch replacing the chain, importing the real eager op as the
parity oracle -- lives in the fusion kernel backend's prompt and in
`local_knowledge/languages/fusion/`. Only the per-recipe facts are passed per
campaign.

The one rule worth repeating here is the harness warm-up. The agent writes a
validation harness that microbenches both arms; each arm must warm up at least
500 iterations before it is timed. Measured on this hardware, a 25-iteration
warm-up leaves the chip below its steady clock and whichever arm is timed second
comes out about 3% slower from heat alone -- the same size as the keep bar, and
against the fused arm whenever eager is timed first.

## Output

| Artifact | What it holds |
|:--|:--|
| `fusion_manifest.json` | The verdict, diagnosis, recipe, validation and artifacts |
| `fusion_experience.md` | What each attempted recipe taught, carried into the next |
| `driver_<pattern>.py` | The generated driver the loop scored |
| `program_<pattern>.md` | The task document the campaign's implementer received |
| `forge_loop_<pattern>.log` | The campaign transcript |
| `harness_reports_<pattern>.jsonl` | Every harness report the driver recorded |
| `serving_smoke_<pattern>.log` | The server log from the final gate |
| `fusion.patch` | The fusion, exported before the smoke so a killed run still hands one over |
| `kernel_keep_checkpoint.json` | Written after that patch exists; marks a KEEP as salvageable |

The serving gate boots the model once, with the session's own tensor-parallel
size, KV block size and max model length -- a sparse-attention model rejects the
default block size and would otherwise fail for a reason that has nothing to do
with the kernel. The gate acts on the stage the smoke stopped at, not on the
wording of its message: only an actual GPU fault (or a decode that hangs) is
evidence against the kernel and reverts the KEEP. A boot that ran out of memory,
a rejected config or a probe that could not reach a live server leaves the KEEP
and its patch in place for e2e integrate to judge, and the run still exits zero
so the caller does not read a deferral as a failure.

The manifest is the stable machine-readable output; `verdict` is one of
`candidate`, `no_opportunity` or `llm_unavailable`, and exit code 3 means the
run never reached the model. Each history entry carries the `experiment_id` of
the forge-loop run behind it, and `best_experiment_id` names the one that
produced the kept result.

The validation fields have mixed provenance and are not a single measurement:
`kernel_speedup` is the loop's mean over repeated benchmarks, the number its
keep decision was made on, while `max_abs_err`, `eager_us` and `fused_us` come
from the one harness report behind that decision. Dividing `fused_us` by
`eager_us` will not reproduce `kernel_speedup`, and `rtol` is always `null`
because the harness reports SNR and absolute error rather than a relative
tolerance.
