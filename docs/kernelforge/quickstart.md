# Quick Start Guide

This guide walks through a complete kernel development workflow — from installation to a fully optimized kernel with experiment history and extracted lessons.

## Prerequisites

| Requirement | Version | Check |
|------------|---------|-------|
| Python | ≥ 3.10 | `python3 --version` |
| ROCm | ≥ 6.0 | `rocminfo \| head -5` |
| rocprofv3 | (included with ROCm) | `rocprofv3 --version` |
| GPU | MI300X / MI355X | `rocm-smi --showproductname` |
| Claude auth | API key, subscription token, **or** Claude Code Max | `echo $ANTHROPIC_API_KEY` / `echo $CLAUDE_CODE_OAUTH_TOKEN` **or** `claude --version` |

**Billing choice.** `kernelforge forge-loop` drives its agent sessions through `claude-agent-sdk.query()`, which spawns the `claude` CLI as a subprocess, so whatever that CLI authenticates with is what gets billed. A `claude` logged in with Claude Code Max bills against your Max subscription and needs **no `ANTHROPIC_API_KEY`**. Where a login cannot persist — a container, CI — `CLAUDE_CODE_OAUTH_TOKEN` reaches the same subscription. Set `ANTHROPIC_API_KEY` only if you want API-credit billing instead; the CLI reads it ahead of the subscription token, so setting both bills the key.

Optional but recommended:
- [RTK](https://github.com/rtk-ai/rtk) for 60-90% token savings: `cargo install rtk`
- AITER repo cloned at `/work/aiter-amd` (or wherever your kernel workspace is)

## Step 1: Install

forge ships inside Hyperloom, so installing Hyperloom installs forge:

```bash
git clone git@github.com:AMD-AGI/Hyperloom.git
cd Hyperloom
pip install -e ".[forge]"
```

For rocprof-compute hardware profiling (System Speed-of-Light + roofline), add the
`forge-profiling` extra: `pip install -e ".[forge,forge-profiling]"`. Without it,
profiling degrades to the lightweight PMC path. (`install.sh` installs it for you
unless you set `SKIP_FORGE_PROFILING=1`.)

Verify:

```bash
kernelforge --version
kernelforge forge-loop --help
```

## Step 2: Configure

Set your workspace and GPU target, then whichever Anthropic setup you use:

```bash
export KERNEL_WORKSPACE=/work/aiter-amd
export GPU_TARGET=gfx950
```

**Claude Code Max, at a terminal.** Nothing to configure — as above, a `claude`
CLI already logged in supplies its own endpoint and billing. `claude login` puts
those credentials in `~/.claude/.credentials.json`, on the machine you ran it on.

**Claude Code Max, headless.** That file is what a container does not have: its
`claude` is a fresh `npm install` and your host's home is not mounted in. Mint a
long-lived token instead and pass it in the environment, which is the only route
to subscription billing in a container or CI job:

```bash
claude setup-token   # once, at a terminal with a browser
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

The CLI defaults the endpoint here too, so this one variable is the whole
configuration. Leave `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` unset — the
CLI resolves either of them first, and the run would bill to the key instead.

**API-credit billing.** Just the key; the CLI defaults to `api.anthropic.com`, so
`ANTHROPIC_BASE_URL` is optional:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Anthropic-compatible gateway.** Here both halves are required — the endpoint has
no sensible default — and you authenticate with the gateway's bearer token rather
than a console key:

```bash
export ANTHROPIC_BASE_URL=https://your-gateway.example/api/v1/llm-proxy
export ANTHROPIC_AUTH_TOKEN=...
```

Fusion discovery runs through the agent harness, so it uses whichever line the
selected provider reads — the Anthropic one above for a Claude model. Callers
that use `default_llm_fn` directly instead of the CLI get a plain completion,
and that path speaks whichever protocol is configured: the OpenAI line when both
halves are set, otherwise the Anthropic one natively.

The Codex supervisor speaks the OpenAI-compatible protocol, which is a separate
line. Set it only if you use it. Both halves are required here — KernelForge
builds the client itself, so a missing endpoint would silently target
`api.openai.com`:

```bash
export OPENAI_BASE_URL=https://your-gateway.example/api/v1/llm-proxy/v1
export OPENAI_API_KEY=...
```

**Native Codex OAuth.** Keep subscription OAuth separate from gateway mode. Use a dedicated absolute,
non-symlink `CODEX_HOME` that already contains a successful `codex login`, do not combine it with a
gateway override, and disable both fallback axes explicitly:

```bash
kernelforge forge-loop ... \
  --agent-backend codex \
  --agent-options-json '{"auth_mode":"native_oauth","home":"/absolute/codex-oauth-home"}' \
  --agent-fallback-provider none \
  --agent-fallback-model none
```

In native-OAuth mode KernelForge strips inherited OpenAI gateway variables and does not force a custom
model provider; authentication comes only from that `CODEX_HOME`.

**Hermes Agent.** Select Hermes as a peer provider through its existing profile. The profile owns its
provider/model/tools/authentication; KernelForge still owns validation, benchmarking, and KEEP/REVERT:

```bash
kernelforge forge-loop ... \
  --agent-backend hermes \
  --agent-options-json '{"profile":"hyperloomfaithful","provider":"openai-codex","external_sandbox":true}' \
  --agent-fallback-provider none \
  --agent-fallback-model none
```

Hermes has no native filesystem sandbox. `external_sandbox` is therefore required and must be set only
when the whole run already executes inside a container with a detectable runtime marker. The backend fails closed otherwise, applies
the shared WorkspaceGuard, narrows read-only runs to no local tools, and narrows writable runs to the
terminal/file toolsets. The selected profile should have an empty fallback chain.

That is the whole credential contract. Corporate gateways sometimes demand headers
on top of it, an APIM subscription key or a caller id. Set those on the line they
belong to; only that line's headers are sent:

```bash
export ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: ${MY_SUBSCRIPTION_KEY}'
```

One header per line as `Name: value`, or a JSON object, with `${VAR}` expanded from
the environment so the secret lives in one place. `OPENAI_CUSTOM_HEADERS` is the
equivalent for the OpenAI line; both forms work on either. Comma-separated pairs on
a single line are *not* split — a header value may legitimately contain commas — so
`user: alice, x-foo: bar` becomes one header whose value is `alice, x-foo: bar`.

Or create a `.env` file (see `.env.example`).

## Step 3: Run your first campaign

`kernelforge forge-loop` runs one campaign: it proposes ONE change per iteration, validates it with your driver, benchmarks it, and keeps only measured improvements. The tasks ship inside the package, under `src/kernelforge/data/examples/` in a checkout — the Triton softmax one is the smallest task that exercises the whole loop:

```bash
cd src/kernelforge/data/examples/triton-softmax-forge-loop
MAX_HOURS=1 ./run_example.sh /tmp/forge_softmax
```

`run_example.sh` copies the task into a scratch git workspace, commits it, and launches the loop — the isolate-then-run pattern every caller should follow, because forge-loop git-inits its workspace and edits the kernel **in place**. Underneath it is a plain CLI call:

```bash
kernelforge forge-loop \
    --kernel /tmp/forge_softmax/softmax_kernel.py \
    --driver /tmp/forge_softmax/driver.py \
    --workspace /tmp/forge_softmax \
    --program-md-file /tmp/forge_softmax/program.md \
    --experiments-dir /tmp/forge_softmax/forge_experiments \
    --result-json /tmp/forge_softmax/forge_experiments/forge_result.json \
    --kernel-backend triton \
    --gpu-target gfx950 \
    --snr-threshold 30 \
    --max-hours 1
```

`--kernel-backend` picks which kernel backend's domain knowledge is injected into the agent's prompt: `ck`, `flydsl`, `triton`, `gluon`, `aiter`, `hip`, or `hipblaslt`. Omit it and the backend is inferred from the kernel sources.

**In a container.** On a GPU host, a ROCm image with torch and your backend already installed needs nothing else from the environment except the credential line and a `claude` CLI on PATH:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size 64g \
  -e ANTHROPIC_BASE_URL -e ANTHROPIC_AUTH_TOKEN -e PYTHONUNBUFFERED=1 \
  -v "$PWD:/workspace" -w /workspace \
  rocm/primus-training-private:<tag> \
  bash -lc 'pip install -q --break-system-packages -e ".[forge,forge-profiling]" && \
            src/kernelforge/data/examples/triton-softmax-forge-loop/run_example.sh /tmp/forge_softmax' \
  > /tmp/forge.log 2>&1
```

### What happens

Each iteration:

1. Profiling and analysis of the current best produce the measured evidence for ONE executable plan
2. The implementer agent applies that plan to the kernel sources — one change per iteration
3. The driver's complete correctness suite runs; a candidate that fails is reverted and never benchmarked
4. The benchmark scores the candidate against the pristine baseline over three independent measurements
5. A measured improvement is git-committed and becomes the new best; every other candidate is reverted

The injected backend prompt enforces the development loop:
```
READ → PREDICT → BUILD → TEST (SNR gate) → BENCH → PMC → ANALYZE → DECIDE → LOG
```

### Monitor progress

The loop prints one block per iteration to stdout (redirect it to a file and `tail -F` when you launch it in the background):

```
--- Iteration 3 (best mean case speedup: 1.062000x, remaining: 47 min) ---
  [validate] Running full correctness suite...
  [validate] Stage 1 Full suite: PASS SNR=62.1dB
  [bench] pristine-relative scores=[1.081, 1.086, 1.084]; sigma=0.002517; mean score=1.083667x; required=1.066243x; raw mean=0.48 ms
  [registers] VGPR=238
  [KEEP] mean case speedup=1.083667x — NEW BEST (+2.0% vs previous best); raw mean=0.48 SNR=62.1dB (192s)
```

To stop a running campaign, create the stop file in its workspace — the loop checks for it at the next iteration boundary and finalizes normally:

```bash
touch /tmp/forge_softmax/.stop
```

## Step 4: Review results

The best kernel is left checked out in the workspace; everything else lands under the experiments directory:

```bash
# Baseline, best, speedup, best commit — machine-readable
cat /tmp/forge_softmax/forge_experiments/forge_result.json

# Iteration archive, plans, profiles
ls /tmp/forge_softmax/forge_experiments/

# One commit per kept candidate
git -C /tmp/forge_softmax log --oneline

# Re-measure the winner yourself
python /tmp/forge_softmax/driver.py --warmup 10 --iters 200 --bench-mode
```

The result file:

```json
{
  "decision": "KEEP",
  "baseline_ms": 0.040921,
  "best_ms": 0.032640,
  "mean_case_speedup": 1.253676,
  "improved": true,
  "best_iteration": 7,
  "best_commit": "9f3c1ab...",
  "validation_passed": true,
  "snr_db": 62.1
}
```

## Step 5: Learn from the experiment

When a campaign finishes it runs its own postmortem — pitfalls from regressions, optimizations from improvements — and reports what it kept:

```
  Lessons learned: 3
  Transfer rules discovered: 1
```

The lesson documents land under the backend's `learned/` directory in the writable
knowledge base — `$KERNELFORGE_PROJECT_ROOT/knowledge_base`, defaulting to
`~/.cache/hyperloom/kernelforge/knowledge_base` — and the (config, performance)
pairs go to the tuning database:

```
knowledge_base/triton/learned/optimization_BLOCK_N_256.md
knowledge_base/triton/learned/methodology_Plateau_at_0480_ms.md
```

Next time a campaign runs on a similar kernel, these lessons are automatically injected into the agent's prompt.

## Step 6: Optimize your own kernel

For overnight optimization of a specific kernel, point the loop at the kernel and the driver that measures it:

```bash
# Run one campaign (8-hour time budget)
kernelforge forge-loop \
    --workspace /work/aiter-amd \
    --kernel csrc/hk_sla/vsa_sparse_attention_bwd.cpp \
    --driver op_tests/test_sla_bwd.py \
    --kernel-backend ck \
    --gpu-target gfx950 \
    --snr-threshold 30 \
    --max-hours 8

# Resume an interrupted campaign in the same workspace
kernelforge forge-loop --workspace /work/aiter-amd --resume
```

An operator that spans several files, or lives in an existing checkout such as AITER, is the same command plus the paths that seed orientation and profiling — `--kernel` stays the anchor, and the agent may edit any tracked implementation file outside the protected measurement surface:

```bash
kernelforge forge-loop \
    --workspace /work/aiter-amd \
    --kernel csrc/include/custom_all_reduce.cuh \
    --driver op_tests/multigpu_tests/forge_all_reduce_driver.py \
    --task-type repository \
    --source-files csrc/include/custom_all_reduce.cuh,aiter/dist/device_communicators/communicator_cuda.py \
    --target-functions "CustomAllreduce::allreduce" \
    --gpu-target gfx950 \
    --nproc-per-node 4 \
    --max-hours 8
```

The loop:
- Makes ONE change per iteration
- Git commits each change
- Runs the driver's complete correctness suite
- Benchmarks only if validation passes
- Keeps improvements, reverts regressions
- Stops once what remains of the time budget can no longer finish a round;
  nothing caps it at a number of iterations
- Auto-benches the pristine kernel as a baseline anchor when the driver
  doesn't supply one (so iteration 1 isn't kept unconditionally)

When the search stalls, `--supervisor-backend codex|claude` escalates to a
supervisor after three consecutive iterations without a new best. Interventions
remain available for the full time budget. The supervisor adds API
calls only while stuck; bench remains the final gate on every accepted edit.

## Example Tasks

Paths below are relative to `src/kernelforge/data/examples/` in a checkout. From
an installed wheel, `python -c "import kernelforge, pathlib;
print(pathlib.Path(kernelforge.__file__).parent / 'data/examples')"` prints the
same tree — copy a task out of it rather than running in place, because the
package directory is not meant to be written to.

| Task | Backend | What it shows |
|------|---------|---------------|
| `examples/triton-softmax-forge-loop/` | Triton | Tutorial task — the complete driver contract: correctness, CUDA-graph benchmark, per-case timing, kernel-only profiling |
| `examples/flydsl-softmax-forge-loop/` | FlyDSL | The same contract plus the stream-routing guard a self-managed-stream DSL needs to be benchmarked honestly |
| `examples/triton_mixtral_dynamic_quant/` | Triton | Production hot kernel — dynamic per-tensor FP8 quant, `(64, 4096)` BF16 → FP8 E4M3FN |
| `examples/flydsl_gemma_rmsnorm/` | FlyDSL | Production hot kernel — Gemma RMSNorm, `(64, 2816)` BF16 |
| `examples/hip_gemma_fused_add_rmsnorm/` | HIP | Production hot kernel — fused residual-add + Gemma RMSNorm |
| `examples/aiter-allreduce-forge-loop/` | AITER (HIP + Python) | A repository task on a collective: two dispatch thresholds scored as two metric groups |
| `examples/mori_ep_dispatch_combine/` | AITER (MoRI-EP) | Distributed 8-GPU multi-rank task tuning a launch-config file rather than kernel source |
| `examples/triton2flydsl-softmax-flydsl-rewrite/` | Triton → FlyDSL | Correctness-first port followed by FlyDSL optimization |
| `examples/triton2flydsl-mxfp8-grouped-gemm/` | Triton → FlyDSL | SGLang MXFP8 grouped GEMM for MoE decode and prefill |

The two rewrite tasks use the other command, `kernelforge forge-rewrite-by-flydsl`: it ports a source kernel to FlyDSL correctness-first, then hands the correct FlyDSL kernel to the same optimization loop and reports source vs. FlyDSL speedup.

## Writing Your Own Task

A task is a directory of files plus a launch script:

| File | Required | Role | forge edits it? |
|------|----------|------|-----------------|
| kernel source | yes | What forge optimizes — the `--kernel` anchor (one file, or the entry file of a multi-file operator / repo) | **YES** — the edited target(s) |
| `driver.py` | yes | Measurement driver: correctness oracle + perf measurer | never (protected) |
| `graph_harness.py` | recommended | Operator-agnostic CUDA/HIP graph timing harness — the default way to bench | never (protected) |
| `program.md` | recommended | Free-form guidance for the agent, passed with `--program-md-file` | never |
| `run_example.sh` | yes | Prepares a scratch git workspace and launches the loop for this task | never |

The kernel source must expose a stable public entry point the driver calls, and it must pass the driver's correctness gate as shipped — the loop measures the pristine kernel first, so that measurement is the baseline every candidate is scored against.

The driver is a black box invoked as `python driver.py <args>`; forge talks to it over stdout only, and never edits it. It must be deterministic, exit non-zero only on a real crash, and print the agreed lines in each mode:

```bash
python driver.py                                        # SNR: 62.13 dB   (or allclose: True)
python driver.py --warmup 10 --iters 200 --bench-mode   # wall_ms: 0.081920 per timed iteration
                                                        # case_ms: case_001 0.081920
```

The full contract — every mode, every line, and the rules for multi-rank and self-managed-stream tasks — is in [`src/kernelforge/data/examples/README.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/kernelforge/data/examples/README.md).

## Troubleshooting

### `rtk --version` reports nothing

RTK is optional but saves 60-90% of tokens; it is used transparently whenever it is on PATH. Install it:
```bash
cargo install rtk
# or
pip install rtk
```

### Agent doesn't find kernel source files

Set the workspace to the repo root that owns the kernel, and give `--kernel` and `--driver` paths inside it:
```bash
kernelforge forge-loop --workspace /work/aiter-amd \
    --kernel csrc/hk_sla/vsa_sparse_attention_bwd.cpp \
    --driver op_tests/test_sla_bwd.py
```

### Build fails with "stale .cuda.o"

This is a known CK pitfall (header dependencies not tracked). The build tool automatically cleans stale objects, but if you're building manually:
```bash
rm -f aiter/jit/build/module_*/build/*.cuda.o
```

### SNR is low (< 30 dB) but output looks "close"

Common causes:
- FlyDSL: LSE domain mismatch (scaled-log2 vs raw-qk)
- CK: AGPR asm bug (`"+a"` constraint drops reg_idx=0)
- Tensor layout: `.clone()` preserving non-contiguous strides

Check the shipped knowledge base for your language — the `*_traps.md` levers collect the ones that have already cost someone a campaign:

```bash
python3 -c "import kernelforge.resources as r; print(r.resource_path('local_knowledge'))"
less .../local_knowledge/languages/triton/skills/optimize/triton_levers/triton_traps.md
```

### The campaign runs longer than you wanted

The campaign is time-driven: it stops once what remains of `--max-hours` can no longer finish a round. To end one early, create the stop file in its workspace:

```bash
touch /work/aiter-amd/.stop
```

The loop checks for that file at the next iteration boundary, then finalizes normally — the best kept commit, the result JSON, and the lessons are all written. Remove the file to resume with `--resume`.

## Next Steps

- Read the shipped knowledge base under `src/kernelforge/data/local_knowledge/` to understand what the agent knows
- Browse the [runnable examples](https://github.com/AMD-AGI/Hyperloom/tree/main/src/kernelforge/data/examples) for task templates and the full driver contract
- See {doc}`Architecture </kernelforge/conceptual/architecture>` and {doc}`Extending forge </kernelforge/how-to/extending>`
