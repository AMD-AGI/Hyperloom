---
name: discover
description: "Write the dossier's reading parts, which need no GPU when the evidence exists: 1 model architecture, 2 framework and system, 3 the served code read method by method, 4 the optimization target and the benchmark's rules, 5 the workload and the CPU work it causes, and 10 earlier optimizations, with the dossier's index. Run by Arbor in the map.discover chapter, or by hand when a dossier is missing or these parts are stale."
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: map.discover
    produces: ["{dossier}/README.md", "{dossier}/01-model.md", "{dossier}/02-system.md", "{dossier}/03-code.md", "{dossier}/04-target.md", "{dossier}/05-workload.md", "{dossier}/10-history.md"]
    reuse_key: [identity, framework]
    gpu: false
---

# Discover: the dossier's reading parts

**In a campaign, Arbor runs this skill in a turn of its own.** Its argument is the path of a brief: read the brief
first. It names the dossier's directory, the files to write, and the campaign's program.
**By hand,** the directory is `docs/dossier/<model>-<workload>-<gpu>/`.

**Before writing:**
- Read [the rules](../../shared/rules.md) (relative to this skill's directory): every number with a source and label,
  every part with a status, and the asks.
- Look the dossier up (`dossier:read`, `bash ../../shared/find-dossier.sh`). Rewrite no part that still holds.
- Start a missing index from [the template](../../shared/template.md).

**Parts that wait on later ones:**
- Part 1's kernel column and part 3's kernel-to-method mapping wait on parts 7 and 8 (`dossier:measure`). Write them
  `partial`, naming what fills them. `dossier:measure` completes them.
- Part 4's `## The benchmark's rules` table comes from the benchmark's rules when the `benchmark` plugin wrote them
  (`benchmarks/<name>/rules.md`, named in the brief). Copy them in with their source.

**End your reply** with one `LEDGER:` line per part written, a FACT naming its file:
`LEDGER: {"id":"dossier-01","type":"FACT","claim":"part 01 written","evidence":"<file>"}`.

## The parts

### 1. Model architecture

- **Answers:** what computation a token goes through, and which parts can
  change without changing the output.
- **Contains:** the layer stack with counts; attention, including any sparse
  or indexed parts; the MoE layout (experts, top-k, shared experts, routing);
  norms, residuals, hyper-connections; any speculative drafter; shapes and
  dtypes per component at the served precision; parameter and KV bytes per
  token; the kernel that serves each component in this build (from part 8).
- **From:** the model's config and code in the pinned tree, read whole (a
  truncated print of `quantization_config` once hid most of a 2,361-entry
  exclude list); the checkpoint's `config.json` as the container sees it, its
  README, and the tensor dtypes in the safetensors headers (a router stored in
  fp32 was reported as BF16). Write the routing function out, and the MoE's
  dataflow in order, collectives included.
- **Done when:** every component in part 7's profile maps to a row, and each
  row names the kernel that ran, not the one expected; and the parameter
  bytes per GPU reconcile with the server's own "Model loading took" line
  within a stated residual.

### 2. Framework and system architecture

- **Answers:** how data moves from disk to the GPU and back, and what the CPU
  does on every step.
- **Contains:** weight format and loading; the KV cache's layout, block size
  and capacity; the parallelism and which GPUs; collectives per layer; the
  scheduler loop (batching, chunked prefill, speculative verify); what graphs
  capture and what stays eager; the processes and how requests arrive. For
  training: the data loader, the optimizer, gradient synchronization, and how
  far the host runs ahead of the GPU.
- **From:** the launch flags and environment the server actually used; its
  boot log (KV size, capture sizes); the pinned code (part 3).
- **Done when:** for any kernel in part 7, a reader can say whether it runs
  inside a graph and what CPU work precedes it.

### 3. Code reading: the served path, method by method

- **Answers:** what the code on the served path does, at the pins, and where
  it can change. Parts 1, 2 and 8 rest on it.
- **Contains:** one report per area (engine loop, scheduler, KV cache, model
  runner, graph capture, sampling and speculative decode, attention backends,
  MoE, linear, quantization and norms, collectives, the model's own files).
  Each says what every in-path method does here, what it costs (measured or
  reasoned, and which), and whether it can be optimized; code off the served
  path is listed, not read. Add a context file every report starts from (the
  exact launch, environment, config, workload and measured numbers), and one
  step assembled end to end.
- **How:** one reader per area, in parallel (subagents). Each reads every
  in-path method in full and follows its calls. Library code outside the
  tree (AITER, CK) is read where it decides dispatch (part 8). Then:
  - **Line numbers come only from numbered output** (`cat -n`, `grep -n`,
    `nl`). A file too large for one read is read in numbered ranges, never in
    unnumbered `sed -n` slices: Model M campaign B's reader did that, and 30 of
    its 33 `amd/model.py` citations were wrong, one past the end of the file.
  - **A reader's report is a lead, not a source.** Before a claim enters the
    dossier, the writer opens the cited range itself and confirms the named
    class, function or line is there. B's writer copied two reports without
    opening a source file; one claimed a missing capability the code already
    had, and it became a ranked direction.
  - **Write the part after the readers return**, not while one is still out.
  - **A partial part lists every area above it did not read**, not some of
    them.
- **From:** the pinned tree, never upstream `main`.
- **Done when:** every kernel in part 8's census maps to the method that
  launches it, and every launch flag maps to the code it changes.

### 4. Optimization target

- **Answers:** what counts as a win, and what does not.
- **Contains:** the primary metric and its estimator (for serving, paired
  per-request ITL and TTFT by input-length bucket); the operating points and
  the regime at each; the seed and what a result at that seed does not
  cover; correctness requirements; what the benchmark cannot show (a
  synthetic acceptance rate, for example); the practical-significance floor;
  how a Pareto result is reported.
- **Also contains, under its own heading `## The benchmark's rules`:** a
  table with one row per rule a change could break, and three columns:
  - **the rule:** what it fixes (workload, dataset, ISL/OSL, concurrency,
    request count, seed, client, duration), forbids (a flag, a variable, a
    technique) or requires (a precision, golden acceptance, an accuracy
    floor, a valid run);
  - **its source:** the benchmark's published rules or validation script,
    the host's rules, or the pack's constraints;
  - **who checks it:** the host's gate, Arbor's gate, a validator, or nobody,
    said as such.
- **Under a host,** the config's directives that start `Host rule` are the
  host's rules: copy each one into the table verbatim. A host refuses or
  rejects what breaks them, whatever Arbor measured.
- **From:** the project's rules and decisions; whoever sponsors the campaign;
  the benchmark's own rules and scripts; the host's `Host rule` directives.
- **Done when:** a result can be classed as a win, a loss, a Pareto point or
  unresolved without anyone deciding afterwards what the target was; and
  every rule a change could break is in the rules table, with its source.

### 5. Workload, and the CPU work it causes

- **Answers:** what the benchmark sends, what reaches the kernels, and what
  the host does meanwhile.
- **Contains:**
  - effective concurrency (requests in flight, not the client's setting);
    rows per step; context behind each generated token; where request time
    goes (prefill, decode, queueing); cache behaviour; what sets the run's
    duration; synthetic parts; regimes, separated; per operating point;
  - **CPU work**: what the host does each step (scheduling, input
    preparation, sampling, detokenization, launches, the frontend) and how
    long it takes, measured untraced; how much of each step the GPU sits idle
    (TraceLens's GPU timeline: busy time over the window) and whether it waits on the
    host; the client's own pacing (think time in a trace replay).
- **From:** the client's per-request export and the server's metrics (the
  runner's `tools/workload/`); TraceLens's kernel summary for steps; untraced timing
  for host work, because a traced host is slower.
- **Done when:** the distributions of rows per step and context that part 7
  profiles at are written down per operating point, and the GPU's idle share
  per step is stated with how it was measured.


### 10. Earlier optimizations

- **Answers:** what AMD, and the reference side, already did for this model
  or workload, so the campaign neither repeats it nor retries a known dead
  end.
- **Inference:** InferenceX, from a clone with full history
  (`SemiAnalysisAI/InferenceX`):
  - `perf-changelog.yaml`: entries whose `config-keys` match the model on AMD
    GPUs (`mi300x`, `mi325x`, `mi355x`) and on the reference platform, with
    their pull requests;
  - the recipe's history: `git log -p` on
    `benchmarks/single_node/<suite>/<model>_<gpu>_*.sh`, for image bumps,
    flags and kernel choices (#3058 chose CK's a8w4 MoE experts for DSV4.1
    on MI355X);
  - the AMD recipe beside the reference platform's, and beside AMD's recipes
    for the same model on other frameworks (InferenceX has ATOM, SGLang and
    vLLM recipes for DSV4 on MI355X);
  - upstream pull requests to vLLM and AITER for the model's ops; the
    project's own ledger and campaign ledgers.
- **Training:** InferenceX has no equivalent. Tell the user we want the
  history of AMD's optimizations for this workload, and ask where it is
  (asks, item 2). If it is not available, write that and skip the part.
- **Contains:** per earlier change: what it did, where (pull request or
  commit), its measured effect and conditions (GPU, pins, operating point),
  and whether the baseline has it; the dead ends, with reasons.
- **Done when:** every change in the baseline recipe's history since the
  model was added is listed and classed (in the baseline, not, or reverted),
  or the part is marked not available.

## Done when

- **Each part you wrote meets its own "Done when",** above, or is marked partial or not available, with what would fill
  it.
- **Its file holds real content:** `python3 -m arbor.dossier.check <dossier dir> --json` lists it `ok`. That means at
  least 300 words and 10 numbers outside headings, template labels and placeholders. Parts the other dossier skills
  write may fail until they run.
- **Every number has its source and label.**
- **The index lists each part you wrote** with its status, sources, date and staleness key, and the open asks.
