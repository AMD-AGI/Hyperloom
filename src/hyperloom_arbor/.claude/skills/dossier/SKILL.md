---
name: dossier
description: Find, or write if it is missing, the dossier an optimization campaign starts from, for one model, workload and GPU target. When a campaign's gains run small, it says so and ranks kernel rewrites, each written up as a kernel plan. It first looks for an existing dossier and uses it, refreshing only stale parts. A dossier covers model and framework architecture, a method-by-method reading of the served code, the workload and the CPU work it causes, the target and the baseline, a profile, which kernels run and whether better ones exist (AITER, KernelForge, kernels in the codebase that go unused, what B200 picks), an optional B200 comparison the user supplies, AMD's earlier optimizations, and ranked directions. Use before starting or re-basing a campaign, when a new model, framework or workload arrives, when asked what to optimize and why, or to update the dossier in rounds (code re-read, baseline, profile, directions) after a long campaign or a merge has moved the served tree.
---

# The dossier: what to know before optimizing

Audience: an agent (or a person) about to optimize a model on a GPU target,
or asked what to optimize. The dossier answers, in order and with sources,
what the campaign would otherwise rediscover. In the runner's DeepSeek V4.1
Flash work, each of these cost a plan its result or its time:

- a knob aimed at a fused kernel that never ran (Plan 11);
- a kernel-boundary win that graph capture had already amortized (X3c);
- "C8" turned out to put 1.1 to 1.5 requests in flight, at 6 to 18 rows per
  step, not 48;
- every C32 profile in Plans 10 to 15 captured the client's warmup;
- Plan 13's best possible gain sat 19.6 times below its noise floor;
- AITER's tuned MoE table had no row for the model's shapes, so every token
  tier ran the untuned fallback.

Kernel usage (part 8) is in [kernels.md](kernels.md); how to write up a kernel
rewrite, part 11's other kind of direction, is in
[kernel-plan.md](kernel-plan.md); the dossier's skeleton is
[template.md](template.md). Profiles come from the `profiler` skill. In an
Arbor campaign, a pack makes this skill available with `skills: [dossier]` in
its `pack.yaml`, or gets it by default if it is on Arbor's default-on list
(`DEFAULT_ON_SKILLS` in `arbor/pack/loader.py`) and the pack does not list it
in `opt-out-skills:`.

## First: look it up

A dossier is written once per model, workload and GPU type, then refreshed
part by part. It is not rewritten for each campaign or each question. Before
anything else:

1. **Find it.** Use a path you were given (by the user, or by the campaign's
   pack or config) first. Otherwise run `bash <this skill>/find-dossier.sh`.
   It lists `docs/dossier/*/README.md` in this repository and, when this
   repository is a submodule (Arbor inside a project), in the project around
   it. Then it lists the dossiers earlier campaigns shared,
   `/shared/arbor/projects/<project>/<user>/<campaign>/knowledge/dossier/`
   (the `knowledge-collector` skill), read in place where the share is
   mounted and through `ssh arbor-share` elsewhere. Each comes with its
   header: project, pins, platform, workload, date.
2. **Match it** on model, workload (benchmark, dataset, seed, concurrencies)
   and GPU type. Another GPU type means another dossier: MI350 never stands
   in for MI355X. Other pins or another node class make parts stale; they do
   not make the dossier missing.
3. **Found: use it.** Read the index and the parts your question needs. Check
   each of those parts' "Stale when" against the current pins, node class,
   workload and harness. Use the fresh parts as they are. Refresh only the
   parts that are stale, missing or partial, and only the ones the work needs;
   then update their status and date in the index. Never rewrite a part that
   still holds. A dossier found only on the share is the copy one campaign
   worked from, as of its collection: copy it into the project's
   `docs/dossier/` and refresh it there, never on the share. When the
   repository has the same dossier, the repository's copy is the one that is
   kept up to date. When the served tree itself has moved (a long
   campaign's KEEPs, or merges from other branches), refreshing parts one at
   a time is not enough: see "Updating the dossier", below.
4. **Missing: write it**, from the parts below. Say so before starting: it
   takes a node (parts 6 and 7) and the user's answers (the asks below).
   The profile run (part 7) is part of writing it, not a later step: see
   "The profile is taken while the dossier is written", below.
   Material that is not in dossier form, such as the runner's `research/` and
   `learn/code/`, is a source to write from, not a dossier.

## Worked example

DeepSeek V4.1 Flash, AgentX, MI355X, in `<fork>/jiaweic-runner`:

| Part | There |
| --- | --- |
| part definitions this skill grew from | `plans/skills/S01_PREFLIGHT.md` |
| system, and where its time goes | `research/dsv41-flash-on-vllm.md` |
| code reading, 16 reports | `learn/code/` (`CONTEXT.md` first) |
| workload | `workload/l01-understand-your-workload-first.md`, `workload/r01-agentx-seed-workload-study.md` |
| kernels and KernelForge | `docs/kernelforge.md` |
| what worked and what did not | `research/optimization-ledger.md` |
| profiles | `research/profiling-inventory.md`; the host's TraceLens profile (`component.profile`), in component mode |

## Ask the user first

Some inputs only a person has. When you write or refresh a part that needs
one, ask for all of them in one message at the start, then continue without
waiting; a part that needs an answer stays pending. In a run with no one to ask (a campaign), write the questions in the
dossier's index under "Open asks" and go on.

1. **Reference-platform traces** (optional; they help a lot): traces of the
   same model and workload on B200 or another NVIDIA GPU, or access to a node
   to take them. Part 9 lists what is needed.
2. **Earlier optimizations, where no public record exists.** For training,
   or an internal framework, ask where AMD's earlier work on this model or
   workload is recorded: pull requests, branches, MLPerf submission notes,
   internal reports. For inference, part 10 reads InferenceX first and asks
   only for what is missing. If nothing is available, part 10 says so and is
   skipped.
3. **The target**, if no one has stated it: the metric, the operating points,
   what counts as a win (part 4).

## Rules

- **Parts in order.** Each states its question, what it contains, where the
  evidence comes from, and when it is done. Parts 1 to 5 and 10 need no GPU
  if the evidence exists; 6 and 7 need a node; 8 needs 3 and 7; 9 needs the
  user; 11 is written last, from the others only.
- **The profile is taken while the dossier is written.** Where the target
  hardware is available (a campaign holds a node), part 7 is a TraceLens profile
  (the `profiler` skill) on the served tree, taken right after parts 1 to 6
  and before parts 8 and 11 are written. A part 7 with a plan and no results
  leaves the dossier unwritten: parts 8 and 11 are drafts until it lands, and
  no optimization starts from them. The profiler's checks run with it; a
  check not run is the next work, not a footnote. Model M campaign B (runner
  plan M00, 2026-10-07) wrote all twelve parts before any capture and ranked
  part 11 by evidence instead of ceilings; its first profile, one iteration
  later, re-ranked it.
- **Re-profile when the tree or the deployment moves.** After a KEEP, and
  after a change of flags or parallelism (expert parallelism, say), part 7 is
  stale; profile again before part 11 is re-ranked from it.
- **Every number has a source and a label**: node or host, GPU, seed,
  concurrency, n. A number without a source is removed.
- **Every part has a status**: written, partial, missing or not available,
  with what would fill it, its date, and the pins and workload it depends on.
  When one of those moves, the part is stale, and it is refreshed before a
  campaign relies on it.
- **One GPU type per claim.** MI350 results never stand in for MI355X, and a
  B200 result is a lead for an AMD claim, never evidence for one.
- **The dossier informs; it never gates.** Gates are the operator's.
- **Where it lives**: in the project's repository, `docs/dossier/<model>-<workload>-<gpu>/`,
  from [template.md](template.md). A campaign gets a digest of it as pack
  knowledge (`packs/<pack>/knowledge/`, which Arbor renders into the agent's
  instructions), citing the dossier; parts 8 and 11 feed the pack's roadmap.
  The campaign's knowledge collection shares it (`--dossier`), which is how
  campaigns run from other repositories find it.

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

### 6. Baseline and noise floor

- **Answers:** what the system does today, and how much it moves when nothing
  changes.
- **Contains:** the pins (framework, kernel library, image digest, harness);
  the node and its health (foreign processes, containers, a connect census);
  the metric at each operating point; the noise floor from repeated control
  runs on the node the campaign will use, with the same estimator as a
  treatment; the validity gate (for AgentX, the duration band).
- **Done when:** the floor is measured on the campaign's node, in the unit
  part 11 compares gains against.

### 7. Profile: plan, then results

Use the `profiler` skill (TraceLens, rocprofv3).

- **Plan, written before anything is captured:** the question each capture
  answers; the mode (the profiler skill's route, and whether it is validated
  here); windows on the client's phase clock, per regime, with the running
  count inside each; the cells of rows and context to cover, from part 5;
  what else must be idle; where raw traces stay.
- **Results:** step time by component per regime and step kind, from the
  stratified tables; collectives by transfer; overlap and host gaps
  (host-side numbers from untraced runs only); outliers, and whether the
  tracer caused them; the checks and what each left open; what the capture
  could not see.
- **Done when:** a capture of the served tree exists (a plan alone is not
  done); the shares add up to the measured step time within a stated
  residual; every share carries its window, regime, mode and validation
  status; and the profiler's checks have run, or each one not run says why.
- **Stale when:** the tree changes (a KEEP) or the deployment does (flags,
  parallelism).

### 8. Kernel usage: what runs, and what could run instead

The procedure is [kernels.md](kernels.md).

- **Answers:** which kernels serve each op at the served shapes, why those
  were chosen, and whether better ones exist: unused in the libraries already
  shipped, documented in AITER and KernelForge, or picked by the reference
  platform.
- **Contains:** one row per op family: the kernels that ran (symbol,
  provider, calls and share per step, shapes, step kinds); the dispatch
  decision (call site, flag, tuned-table hit or fallback); alternatives in the
  codebase that do not run at these shapes, why not, and how to try each;
  KernelForge's pages; what B200 runs for the same op; the gfx950 gap that may
  block a port; the lead, with its ceiling from part 7. The serving playbook's
  `image-manifest.md` says what the image ships but leaves off, and
  `amd-knobs.md` what each vendor knob does and whether it changes numerics.
  The `kernels` skill holds the kernel notes (AITER, FlyDSL, CK, HIP, Triton)
  and the examples a "how to try it" column draws on.
- **Done when:** every kernel with at least 1% of stratified GPU time in any
  regime has a row, with its dispatch decision and its alternatives checked,
  or "none found" and where the search looked.

### 9. Reference platform (B200): optional, with the user

- **Answers:** what a faster platform does differently on the same model and
  workload.
- **Needs the user** (asked up front). Without traces or access, the part is
  "not available", and parts 8 and 11 say what they could not compare.
- **What to get:** the reference recipe (for inference, InferenceX's script
  for that GPU: image, model revision, flags); the same workload settings
  (dataset snapshot, seed, concurrency, duration); a PyTorch-profiler trace on CUDA (TraceLens
  reads it there too) or an Nsight Systems trace; the client's
  metrics.
- **Compare:** at matched operating points (rows per step, context, requests
  in flight), because one concurrency puts a different load on each platform
  (B200 at C8 kept 0.77 requests in flight); time per step by module; which
  kernel serves each op (into part 8); framework choices (graph mode, capture
  sizes, backends, where the KV cache and other tables live, warmup).
- **Hardware gaps:** name the gap before porting an idea: native number
  formats, matrix instruction shapes, wavefront width (64 on gfx950), on-chip
  memory per compute unit, asynchronous copy engines, interconnect.

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

### 11. Directions

- **Answers:** what to try, in what order, and what each could give at most.
- **Contains:** per direction: the component and kernel it targets (parts 7,
  8); its largest gain in the target's unit (part 4) beside the floor (part
  6); what it depends on, what it costs, what would stop it; whether it was
  tried before (part 10) or the reference platform shows it (part 9); the
  rules of part 4's table it touches, and how it stays within each, or
  "none". A direction that only pays by breaking a rule is not a direction:
  list it with the rule, and set it aside.
  Directions whose best case is below the floor are listed, with the number,
  and set aside. Ranked by expected gain over the floor, then by cost.
- **From:** parts 1 to 10 only, ranked by the ceilings part 7 measured. A
  direction that needs a missing fact names the missing part; without part 7
  results the whole part is a draft and says so at its top.
- **Two kinds of direction, ranked apart.**
  - **An increment** changes one thing that a campaign iteration can build and
    measure: a flag, a table row, a dispatch rule, a fusion of two adjacent
    launches. Its ceiling is what its component costs now.
  - **A rewrite** rebuilds a kernel or a fused region: a decode MoE from route
    sort to finalize, an attention block from projection to output norm, a
    layer as one persistent kernel. It takes many iterations before anything
    can be served. Its ceiling is the region's time now minus its floor at the
    served shapes (bytes over bandwidth, FLOPs over peak, the launches it must
    keep), both from part 7.
  - Part 11 lists rewrites in their own table, each with a kernel plan
    ([kernel-plan.md](kernel-plan.md)), even while increments lead.
- **When rewrites go first: the increments are running out.** Watch for these
  signs. They're calibrated on Model M campaigns B4 and C4 (MI350X, 0.5% gate,
  2026-10-13); revisit the numbers after the first rewrite campaign.
  - **The KEEPs are small:** the median of the last five KEEPs is under 3×
    the gate. B4's was 1.27%, C4's 1.39%.
  - **The probes kill their own leads:** five or more probes in a row each
    closed the lead it set out to build, at or below the gate. That was B4's
    iterations 42–46, and C4's 38–43, the last of which "kills the
    campaign's last big build".
  - **The board is thin:** no increment left has a measured ceiling of 3× the
    gate, or what's left is launch-bound or already at its roofline. B4's
    mixed-step all-gather already ran at the XGMI roofline; 86% of C4's
    mixed-step idle time was a dependency stall.
  - **Arbor's `barren_streak` has reached its notice (8).** Any KEEP resets it,
    so on its own it misses a campaign that still keeps a gate-sized change
    now and then. The first three signs don't.

  Two of these signs make the case. A rewrite still needs something to win,
  though: part 7 must show a region whose time now is well above its floor,
  several percent of the step on paper. Without one, the dossier says so: the
  remaining gap is in the system (collectives, scheduling, the host), not in
  a kernel, or the target is met.
- **Then the operator decides; the dossier informs.** An agent that sees two
  signs says so in its verdict and carries on. The usual next step:
  - an update round (below) whose 11.uK ranks rewrites first, with a kernel
    plan for each of the top ones;
  - then a campaign spent on one rewrite:
    - its config sets `"space": {"levels": ["kernel"]}`;
    - its pack raises `policy.max_consecutive_probes`, so build stages that
      can't serve yet aren't cut off;
    - each stage is judged by the plan's gate for it, until the last stage,
      where the campaign's own A/B gate decides.
- **Check the proven big directions before ranking.** The serving playbook
  (`.claude/skills/serving-playbook/`) lists structural changes that won
  large on other stacks, MegaMoE-style fused MoE and mega kernels among them,
  each with its preconditions (`big-directions.md`), and which settings win at
  which operating point (`regime-tables.md`). A campaign that works one
  iteration at a time rarely reaches these on its own, so part 11 has to
  bring them in.
  - **For each one whose preconditions this workload meets** at the target's
    operating point (parts 4 and 5): part 11 carries it, usually as a rewrite
    with a kernel plan, or lists it as set aside with the reason. Reasons: the
    image lacks it (`image-manifest.md`), the regime doesn't favour it, it is
    already on, or its best case is below the floor.
  - **Price it on this node,** from part 7 and the hardware skill's measured
    table, never from the gain the playbook reports elsewhere.
- **The map: what the campaign tracks.** Every direction has a map ID,
  `m-<component>`, which stays the same across update rounds; a direction
  that changes so much it needs a new ID says which one it replaces.
  - **The `## Map` block.** The part ends with one, holding the exact line an
    Arbor campaign reads (`arbor/directions.py`):
    `DIRECTIONS: [{"id": "m-…", "kind": "increment|rewrite", "title": "…", "rank": 1, "ceiling": "…", "status": "open"}, …]`
  - **Statuses:** open, done, killed, below_floor, or plan (a rewrite whose
    kernel plan is written). Every closed direction carries a `why`.
  - **What a parallel run needs** (optional; a map without them is read as
    before):
    - `lane`: increment, kernel-plan, architectural or upstream. By default
      an increment is an increment lane and a rewrite a kernel-plan lane.
    - `plan`: the kernel plan, relative to the dossier. A kernel-plan lane
      needs one, with its `## Items` block (`kernel-plan.md` §5).
    - `depends_on`: the map IDs it builds on, from the same list. Cycles are
      refused; a closed dependency counts as met.
    - `nodes`: the nodes its lane holds at once (default 1).
    - `upstream`: `{"repo": …, "pr": …, "commit": …, "image": …}`, which an
      upstream lane (a port of work done elsewhere) needs.
    - `reopens`: `{"round": K, "why": …}`, for a direction an earlier round
      closed and this one brings back on evidence: 10.uK marks the kill
      reopenable, or the new tree removes the reason it died.
    - `evidence` on a close: `{"iterations": [N], "ledger": [ids],
      "artifacts": [paths]}`. A lane needs it, and the harness checks it.
  - **What the campaign does with it:** it keeps the map as its checklist and
    closes directions with `DIRECTION:` lines. When every one is closed, it
    runs this part again (see "In the campaign, by the harness" below).
  - **Check the line before it goes in:** `python3 -m arbor.directions check`.
    With `--dossier <this dossier>` it also checks each lane's plan and
    prints the order the dependencies allow.
- **Done when:** each direction can become a plan whose evidence section cites
  the dossier, each rewrite has a kernel plan, every big direction from the
  playbook that applies is carried or set aside with its reason, and the
  `## Map` line passes `arbor.directions check`.

### 12. Fast bench (does not block the campaign)

Whether a cheaper bench can screen changes, and how often it disagrees with
the long one: candidates tied to part 5 (a kernel bench at the served shapes,
a step bench at the weighted cells, a compressed replay, a shortened run,
each with what it cannot see); ground truths of known effect (an A/A pair, a
change the long bench already judged, injected delays proven on the device);
per candidate, its cost, its own floor, sign agreement with the long bench and
false positives. Until a candidate is validated, its readings may order work
but never keep or reject a change; validated, it screens, and a claim still
needs the long bench. The runner's Plan 21 is the worked example.

## Updating the dossier: after a long campaign or a merge

A dossier describes one tree. A campaign changes it one KEEP at a time, and a
merge brings in other branches' work at once; after enough of either, the
served code is laid out differently (a path split, a layer re-partitioned, a
collective moved) and the step's time goes elsewhere, sometimes drastically.
Part 7's rule (re-profile after a KEEP) keeps the profile current inside a
campaign. It does not re-read the code or re-rank the directions, so the
campaign goes on optimizing from a map of a tree that no longer exists. Model M
campaign B2 (runner `plans/model-m`, 2026-10-09) kept ten changes in forty
iterations, a sequence-parallel prefill path among them, and its stack was
then merged onto another team's, which partitions the MoE pooler another way;
the dossier B2 started from describes neither tree.

**When.** The operator decides, on signs like these:

- the served tree has taken a merge from another branch;
- KEEPs have re-partitioned a layer, a projection or a collective, or moved
  where the step's time goes, not only swapped a kernel for a faster one;
- the directions have run out: the campaign's top leads are killed and what
  is left sits below the floor, or the increments are running out (part 11's
  signs), so 11.uK should rank rewrites;
- the bar has to be measured again anyway (a new tree, a new node class).

An agent that sees these signs says so in its verdict and goes on with its
plan; it does not stop the campaign.

**In the campaign, by the harness: when the map is explored** (operator,
2026-10-20: "if all the directions were explored already, the campaign
should not explore aimlessly, but should run dossier and profile again, then
continue 'like new'").
- **The trigger.** When every direction of the campaign's map is closed,
  Arbor (`arbor/directions.py`) makes the next iterations this round, in the
  same session and without a STOP.
- **What the round writes:**
  - 07.uK, the profile again;
  - 10.uK, the round's iterations;
  - 11.uK, the directions again, ending in its `## Map` line;
  - 03.uK, brief: what the round's KEEPs changed in the code;
  - 06.uK cites the session's bar, with no new baseline.
- **How each part is run.** As an `INTENT: LEARN` iteration with a LEDGER FACT
  naming the file, with the LEARN cap waived.
- **What the new map starts.** A new map round, with the streak counters
  reset: the campaign carries on like new.
- **When the new map has no open increment** (operator, 2026-10-22: Model M
  day-0 b0 stopped there, and "it should have started like a new campaign"):
  - **a fresh start**: the next iterations write a new dossier as a new
    campaign would, in the same session: 03.uK, the code read again whole;
    07.uK, the profile again; 10.uK; and 11.uK, a map from 03.uK and 07.uK,
    with the old map as history, not a list to carry over;
  - **then a rewrite**: if the fresh map has no open increment either, the
    harness opens the top planned rewrite (a kernel plan first, then by rank)
    as the next round's work, built stage by stage through the gate.
- **Where the operator still decides.** The harness stops the campaign
  (`MAP_STOP.md`):
  - when no planned rewrite is left after a fresh start;
  - when two rounds in a row end without a KEEP;
  - when the round yields no valid map in its iteration limit.
- **The rest of this section is the operator's version:** a stopped campaign,
  every part, and a new session.

**How.** Stop the campaign first (in Arbor, the session's `STOP` or the
chain's `CHAIN_STOP`), so nothing changes the tree while it is read and
measured. Then, on the served tree as it now is, in order:

1. **03.uK, the code, read again.** Not the diff: the model's code read whole,
   method by method, under part 3's rules (one reader per area, line numbers
   from numbered output, a reader's report is a lead). The model's own files
   and the model-specific ops they call are read; the framework's areas keep
   part 3's reading unless the framework's pins moved, and then they are read
   again too. Each in-path method gets, beside what it does and what it costs:
   - **an opportunity label**: none, or a candidate, with what could change,
     the mechanism, what it would need, and whether the cost behind it is
     measured or reasoned;
   - **its history**: unchanged since part 3, or added or changed by which
     KEEP or merged commit.

   A diff misses what moved around code that did not change: a method left
   alone whose inputs now arrive sharded, or that a new path no longer calls.
2. **06.uK, the baseline again**, as part 6, on the node class the campaign
   will run on: the metric at each operating point and a fresh noise floor.
   It is the restarted campaign's bar.
3. **07.uK, the profile again**, as part 7: plan, then results, on this tree,
   before anything is ranked.
4. **10.uK, what the campaigns since the last round learned.** Read every gated
   iteration of the stopped session, and of any campaign whose work the merge
   brought in, not the ledger alone. The `knowledge-collector` skill stages
   both sources:

   ```bash
   python3 -m arbor.collect knowledge --session <session> --config <config> \
       --types outcomes,findings --no-push
   ```

   This writes `OUTCOMES.md`, every iteration kept, reverted, probed or not
   measured, with why, and `FINDINGS.md`, the ledger's facts and dead ends. Each
   KEEP's `iter_N/patch.diff` says what it changed. Per campaign, the part
   records:
   - **its KEEPs**: what each changed, its gate gain, and where it stands on
     this tree: in it, to re-land, or superseded;
   - **its kills**: each dead end, the reason it died, and what that reason
     rests on, either a measurement (which instrument, which floor, which tree)
     or an argument. Some kills are marked **reopenable**, with the reason: one
     resting on a floor measured with the kernel under test, on a tree this
     round replaced, or on an argument nobody measured;
   - **what its probes established**: the facts that move a ceiling or a cost
     (a launch floor, a roofline, a share of the step), each with its source;
   - **the map against the work**: which of the last round's ranked directions
     were worked, which were never touched, and what was worked off the map;
   - **process lessons**: harness and measurement failures, and the iterations
     they cost;
   - **the counts**: gated iterations, probes and KEEPs, and KEEPs by level
     (config, tuning, framework, kernel), so a round that keeps working one
     level sees it.

   The hand-off a restarted campaign reads is cut from this part. Model M's round
   u2 did this step by hand as `11.u2-c2-ledger.md`. Across eight Model M
   campaigns (runner `plans/model-m/BC_PROBES_AND_KERNELS.md`), at least 5 of the
   30 KEEPs that came from a handed ranking reworked something an earlier map
   had killed. The core kernels there were killed on floors that a later
   campaign found the kernel under test had measured itself.
5. **11.uK, the directions again**, as part 11, from 03.uK, 06.uK, 07.uK and
   10.uK, and the parts still current. A direction the campaign killed stays
   killed unless the new tree removes the reason it died or 10.uK marks it
   reopenable; the entry says which. 10.uK's counts and kills tell whether
   the increments are running out (part 11's signs); if they are, rewrites are
   ranked first, each with a kernel plan.
6. **Restart** the campaign on this tree, as a new session: its bar is 06.uK,
   its pack knowledge is re-digested from the round's parts (10.uK's hand-off
   first), and the stopped session stays as the record. After a merge, "this
   tree" is the merged one, the tree the campaign serves from now on.

Parts 1, 2, 5 and 8 are checked against their "Stale when" as always, and
refreshed if stale; part 8 gains a row for every kernel 07.uK shows that it
lacks.

**Files.** Round K writes `NN.uK-<part>.md` beside the originals
(`03.u1-code.md`, `06.u1-baseline.md`, `07.u1-profile.md`,
`10.u1-lessons.md`, `11.u1-directions.md`). The originals stay as they are, the record of the tree
before. The index's parts table points at each part's newest round, and its
updates table lists every round ([template.md](template.md)).

**Done when.** 03.uK labels every in-path method of the model's code; 06.uK's
floor is measured on the campaign's node; 07.uK has results that add up to the
measured step; 10.uK accounts for every gated iteration since the last round
and gives every kill its evidence; 11.uK ranks from them; the index lists the
round; and the evaluation below passes for the round.

## Evaluation: the files exist and hold real content

Before a dossier or a round is called done, and before a campaign ranks
directions from it, run:

```bash
python3 <this skill>/check-dossier.py <dossier dir>                # parts 01-11 and the index
python3 <this skill>/check-dossier.py <dossier dir> --round uK     # 03, 06, 07, 10, 11 .uK and the index
python3 <this skill>/check-dossier.py <reports dir> --round uK --no-index   # a round staged outside the dossier
```

- **What it requires:** every needed file exists and holds real content. That means at least 300 words and 10 numbers outside headings, template labels, `<placeholders>`, table rules and empty cells. Kernel plans (`11.uK-plan-*.md`) are checked the same way when present. A whole dossier's part 4 also needs its `## The benchmark's rules` table, with at least one rule.
- **What fails a file:** missing, a copied template, a status of "missing", or a round the index doesn't list.
- **What it doesn't judge:** whether the content is right. That's what each part's "Done when" is for.

Calibrated on the Model M MI350X dossier: every real part has at least 635 words and 131 numbers, while a template skeleton with one line filled in has 17 words. A FAIL names the file and why. Write or finish that file. A part that genuinely can't be filled says so in its own status line ("not available", with the reason); it isn't left out.

## Done when

- The evaluation passes (`check-dossier.py`, above).
- Every part is written, or marked partial, missing or not available with
  what would fill it; part 12 may be in progress.
- Every number has its source and label.
- A reader who did not write the dossier can class a made-up result as a win,
  a loss, a Pareto point or unresolved from part 4 alone.
- The index lists each part's status, sources, date and staleness key, and
  the open asks.
