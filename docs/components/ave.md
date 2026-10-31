<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Ave kernel optimization

Ave is a kernel optimizer alongside KernelForge that executes ordered prompts.
Select it explicitly with `KERNEL_OPT_BACKEND_ORDER=ave`:

```bash
KERNEL_OPT_BACKEND_ORDER=ave CLAUDE_MODEL=sonnet python -m hyperloom optimize --model /models/my-model --framework vllm
```

`--model` identifies the inference model's weights/configuration directory.
Hyperloom runs its inference optimization workflow. When it enters the kernel
phase, it passes the current workload, serving configuration, trace evidence,
source repositories and prompts to Ave under the existing server, workspace
and benchmark leases. Ave selects one existing AveLang kernel, generates an
independent correctness/timing driver and executes the supplied prompts.
It then applies Ave's cumulative patch and runs
the inference workload's E2E performance and accuracy gate. KEEP promotes the
result into the optimization stack; REVERT restores the serving sources.

```text
Hyperloom kernel phase with backend=ave → inspect workload, trace and source call sites
  ├─ Applicable AveLang kernel found
  │    → Prepare its independent correctness/timing driver
  │    → Ave ordered prompt loop → cumulative patch → E2E KEEP/REVERT
  └─ No applicable AveLang kernel → record the reason and skip kernel optimization
```

This version checks for one existing AveLang kernel on a single node per session.
It does not mix Ave and Forge jobs within the same phase. All supplied prompts
must apply to that one selected kernel. If the evidence cannot establish a
suitable kernel, its serving call path, or a valid workload/reference, Hyperloom
records the reason and ends the kernel phase. Preparation failures and timeouts
also end the phase without invoking another optimizer. GEAK and Forge selections
do not run Ave preparation.
Once an Ave target is selected, its optimization and E2E result settle the phase.
Source repositories come from Hyperloom's existing serving-source discovery;
editable AveLang installations are included automatically. Additional source
checkouts use `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`, a path-separated list.
The serving runtime must import the selected source checkout, for example through
an editable installation. Adding a source root only makes it discoverable; it does
not redirect imports from an installed wheel. Ave verifies the workload's calling path before selecting a kernel;
the presence of a repository alone does not establish that the workload uses it.
The selected optimizer, prompts and result are persisted, so resume does not need the original
prompt file or rerun a completed campaign. Ave and KernelForge do not import
each other.

Ave currently does not support Remote Recipe KB warm start or publication.
Its packaged prompt knowledge remains available. Reports include Ave KEEP counts
and E2E gain contributions alongside GEAK and Forge.

## Supply prompts

Ave uses the `DEFAULT_PROMPT` string in `ave/task.py` as one optimization step.
It asks for a concrete optimization based on the source and baseline measurements,
preserving correctness for every workload case. No prompt file is required.
The default goes through the same ordered prompt loop.

To replace it with your own steps, set `AVE_PROMPTS_FILE=/work/prompts.json`
before starting Hyperloom. Workload evidence determines which kernel receives
those steps. With Ave selected, an invalid or empty file fails session initialization.
Other backends and `--no-kernel` ignore it; resume uses the session's saved prompts.

```json
{
  "steps": [
    {
      "id": "prepare_layout",
      "prompt": "Reorganize the accumulator layout for register tiling. Preserve normalization semantics and all supported shapes."
    },
    {
      "id": "register_tiling",
      "prompt": "Use the prepared layout to keep partial sums in registers and reduce redundant loads. Preserve the previous step's layout and numerical tolerances."
    }
  ]
}
```

Each step receives the current sources, its prompt, the independent driver,
completed step IDs and recent measurements or failures. Array order determines
execution order. A step may optionally provide
`files`, a list of repository-relative paths restricting its edits. These must
be a subset of the selected kernel's sources. When omitted, all selected sources
are editable. Step IDs must be nonempty and unique.

No user-authored task manifest or starting commit is needed. Hyperloom captures
the current tracked serving sources, including uncommitted changes, in an
internal Git snapshot. This leaves the serving branch, index and worktree intact.
Untracked files are not part of the snapshot. Ave edits a detached worktree;
drivers, references and tests stay outside its editable source scope.

Within each step's Claude session, the agent edits sources, runs correctness and
benchmarks, and repairs failures without losing its edits. After the session,
Ave independently checks the edit scope and correctness and records three benchmark
runs. A correct intermediate transformation is committed even if it is slower,
because later steps may depend on it. If this host validation fails, Ave restores
the preceding verified revision and retries, up to three sessions per step. Claude
session and candidate driver timeouts also count as failed attempts; retries
continue while the campaign has enough budget for another session. Persistent failure
stops the sequence. The verified prefix remains available for Hyperloom's E2E
gate, including after a timeout or provider failure. No microbenchmark speedup
alone constitutes an E2E KEEP.

## Generate prompts from shared knowledge

`ave.prompts.build_prompts(recipe, context=...)` renders an ordered recipe into the same `PromptStep` objects accepted by the loop. It reads packaged Markdown knowledge and makes no LLM calls.

The [knowledge directory](../../src/ave/knowledge/) separates shared language/hardware rules from individual optimization mechanisms. Each mechanism document describes its applicability, transformation, invariants and tradeoffs.

| Concern | Knowledge topics |
|---|---|
| Language and hardware | `avelang`, `gfx950` |
| Matrix compute and formats | `register_tiling`, `scale_packing`, `quantization`, `compute_helpers` |
| LDS | `lds_staging`, `double_buffering`, `lds_layout` |
| Memory traffic and issue order | `metadata_caching`, `vectorized_memory`, `instruction_scheduling`, `cache_policy` |
| Work assignment and launches | `persistent_scheduling`, `kernel_fusion` |

Compiler limitations are scoped to the version where they were observed. The recipe remains the source of truth for the ordered transformations, exact formulas, tile sizes, thresholds, file scopes and exclusions. Knowledge explains the selected mechanisms; it does not authorize additional transformations or change the loop's acceptance criteria.

A recipe supplies `objective`, shared `knowledge` topic names and an ordered `steps` list. Each step supplies `name`, `files`, additional `knowledge` topics and `required` changes; `notes` and `avoid` are optional. Topic names are Markdown filenames without `.md`. For example, a register-tiling step selects `register_tiling`, while a later buffering step selects `double_buffering` and the applicable hardware rules. The renderer combines only the referenced knowledge with the step requirements and the caller's workload/driver context. It does not select optimizations or infer their applicability to an arbitrary kernel.

The step's readable `name`, such as `register_tiling`, becomes its execution `id`. Array order still defines the sequence. Existing prompt-file input and the default prompt use the same loop as generated prompts.

Changing shared knowledge changes subsequently generated prompts; an existing Hyperloom session continues to use its saved prompts on resume.

## Runtime and driver

Install the AveLang compiler/runtime and kernel dependencies in Hyperloom's
Python environment. Hyperloom creates a detached Git worktree for Ave, so the
driver must import kernels from that tree. Its root, `python/` and `src/`
directories lead `PYTHONPATH`. The compiler must work from a fresh worktree;
untracked compiler bindings or build artifacts are not copied from the serving
checkout. Both the driver and serving runtime must rebuild source-dependent JIT
artifacts when kernel sources change.

The driver contract is:

- With no arguments, validate all workload cases against an independent
  reference, exit nonzero on failure, and print `allclose: True` after success.
- Accept `--bench-mode --warmup 3 --iters 20`. Warm up compilation and use GPU
  graph timing with synchronization, excluding compilation and host overhead.
- Emit finite positive milliseconds per case, for example
  `case_ms: tokens128_hidden4096_bf16 0.012`. Keep case IDs and workloads fixed.

For a multi-GPU kernel, the driver launches and joins every required rank on the
allocated node, validates against the collective reference, and fails if any rank
fails. Only rank zero emits the aggregate correctness and timing results.

Ave uses Claude Code subscription authentication through the shared Claude
Agent SDK configuration and `CLAUDE_MODEL`. For Hyperloom CLI runs, obtain a token
with `claude setup-token` and set `CLAUDE_CODE_OAUTH_TOKEN`, with
`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` unset. See the shared
[authentication configuration](../reference/authentication.md).

The preparation and optimization system instructions are packaged under `src/ave/instructions/`. They define the agent tools and validation workflow; workload recipes supply the ordered transformations.

Ave's preparation session uses only Read/Glob/Grep to investigate the
workload and return the kernel selection and driver source. It has a maximum
five-minute budget within the optimizer budget. Like Forge's opportunity analysis,
it treats TraceLens conclusions as hints and checks the trace, runtime dispatch
and source itself; selection does not require TraceLens to recognize AveLang.
Ave's optimization sessions run
with the selected repository's detached worktree as their working directory.
Claude uses Read/Glob/Grep to inspect source, language APIs and tests, and
Edit/Write/Bash to modify the scoped sources and run the supplied driver.
The agent receives correctness and benchmark commands using the worker's Python
environment, current per-case timings and its remaining session budget.
Each editing session is limited to sixty minutes or the remaining optimizer budget,
with time reserved for host validation.
Its final text report is informational; the host validates the actual worktree
changes and measures them independently before committing.
Put optimization requirements and constraints in the prompts.

Hyperloom calls `ave.preparation.prepare_kernel()` with an `OptimizationRequest`
containing plain workload evidence, source repository paths and prompt steps.
After selection, it hands over the GPUs and calls `ave.optimizer.optimize()`
with the prepared target, instructions and a worker runner. Ave owns kernel preparation,
the disposable worktree, optimization and patch export;
it does not receive Hyperloom's coordinator or session state. Hyperloom owns
scheduling, time budgets, worker process cleanup, E2E integration and recording the result.
Ave uses the same subprocess runner as Forge. Cancellation stops the worker and its
descendants and exports the verified prefix before removing the worktree; it does not start E2E integration.

## Results

Hyperloom reserves an E2E measurement from its kernel phase budget before running
Ave. It persists `ave_result` in session state, including preparation failures and
skips. After optimization, it publishes an `ave_done` response. Artifacts live in
`<session>/ave/run-*/`: `result.json`,
`preparation-context.json`, `preparation.jsonl`, `selection.json`, `run.json`, `driver.py`, `ave-iterations/`,
`console.log`, and, when sources changed, `candidate.patch`,
`snapshot/` and `integrate.json`. `result.json` records completed steps, attempts,
measurements and the final verified commit. Each attempt in `ave-iterations/`
contains `context.json`, a `claude.jsonl` session transcript with tool calls and
results, and any host validation logs. The disposable worktree is removed
after patch export.
