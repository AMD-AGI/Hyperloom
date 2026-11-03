# Why agents miss the big optimizations, and what to give Arbor

Date: 2026-10-20. For: the operator, and whoever builds Arbor's next step.

**The question** (operator, 2026-10-20):
- Arbor's agents make good local optimizations, but rarely architectural changes, bigger plans or
  kernel plans. Two ways to fix that:
  - (i) more knowledge of proven big directions (MegaMoE, mega kernels);
  - (ii) the mesh asking for kernel and architectural plans explicitly.
- Dig through InferenceX's history "seriously, one by one", and say which optimizations are hard for
  agents to come up with, and **why**.
- Add a hardware-spec knowledge base to Arbor.

**The short answer:**
- **Most hard optimizations are hard for structural reasons, not for lack of cleverness.** The loop
  searches the wrong space, cannot take steps that lose before they win, and cannot see where many
  effects show up.
- **Knowledge (i) fixes the share that is missing facts.** It cannot fix the rest.
- **The rest needs harness changes, plus the mesh's lanes (ii)** for plans that take many iterations
  and multi-node layouts. That is the mesh decision 0003 already sets out.
- **The hardware facts every prompt carries today are wrong.** That is the first fix.

## 1. The study

**Source:** `xiaobochen-amd/InferenceX`, the SemiAnalysis benchmark's serving recipes.
- Its 789 `perf-changelog.yaml` entries, each traced to its own commit (2025-10 to 2026-08).
- The 467 recipe commits no entry covers (2025-08 to 2026-08), and the fork's 2 own commits.
- **Total: 1,258 changes.**

**How it was read:**
- Each change was read on its own, by 12 readers working in date order.
- Each change got one record: what changed exactly, why it is faster, the effect, a difficulty for an
  Arbor-like agent, and *why*.
- "Why" means the mechanism: what the agent would have needed to know, see or be allowed to do, and
  why its loop does not get there.
- [`method/INSTRUCTIONS.md`](method/INSTRUCTIONS.md) defines the loop, the difficulty scale and the
  ingredient tags.
- **Evidence:** [`records.jsonl`](records.jsonl), one record per change, keyed by commit. The
  manifest and extraction scripts are in [`method/`](method/).

**What it found:**

| | |
| --- | --- |
| Changes | 1,258: 374 optimizations, 282 infra, 237 new recipes, 179 image upgrades, 97 fixes, 45 reverts, 44 sweeps |
| Optimizations by difficulty | 84 easy, 112 medium, 106 hard, 72 very hard |
| Hard or very hard, any kind | 238 (NVIDIA 184, AMD 54) |
| …of which need several nodes | 157 |
| …single-node, so in principle in Arbor's reach | 81 (NVIDIA 46, AMD 35) |
| AMD environment knobs the recipes name | 101 distinct (`VLLM_ROCM_*`, `AITER_*`, `SGLANG_*`, `ATOM_*`, `TRITON_HIP_*`, `HSA_*`, `GPU_MAX_HW_QUEUES`, `MORI_*`, `RCCL_*`) |

**The single-node hard ones** are the set that bears on Arbor today. What they were missing, by tag:
- needs a newer image: 57;
- pays off outside the measured operating point: 38;
- vendor-library knowledge: 38;
- several settings that only work together: 29;
- awareness of upstream PRs: 28;
- workload knowledge: 21;
- the measurement cannot see it: 19;
- trades accuracy: 18;
- hardware knowledge: 15;
- designed across components: 14;
- counterintuitive removal: 13.

**The easy ones** were found by agents too. All three changes co-authored by Claude in the history
(ids 0, 1255, 1257) were easy local flags. One was +23.6% P90 interactivity from default-off AITER
fusions. That matches what we see in Arbor.

## 2. Why they are hard

Eight mechanisms, each with the evidence that shows it. Ids are `records.jsonl` ids.

### 2.1 The loop searches the wrong space

- **What it optimizes:** Arbor edits one tree, on one node, at one operating point. A gate keeps a
  change that raises one number there.
- **What the people optimized:** the frontier, throughput per GPU against interactivity across all
  concurrencies. Their levers were mostly the deployment:
  - the TP/EP/DP layout per concurrency band;
  - the prefill:decode ratio;
  - decode width, the engine itself, and the number of nodes.
- **Two consequences:**
  - A layout that loses at the measured point and wins elsewhere is rejected.
  - "Fewer GPUs per replica" wins under per-GPU normalization but looks slower in absolute tok/s,
    which is what a fixed-TP pack measures.
- **Evidence:**
  - 157 of the 238 hard changes need several nodes.
  - Single-node examples: GLM-5 FP8 MI355X TP8→TP4 (653); Qwen3.5 MI355X TP4 prefill, +24–31% per
    GPU (1210); MiniMax-M3 MI355X P:D retuned per input length, 2.9× (1001).
- **What fixes it:** a different evaluator, not knowledge. The objective has to be a frontier across
  operating points, and the space has to include the layout. That is a mesh-level capability
  (§4.2).

### 2.2 The decisive fact lives outside the agent's world

- **The agent's horizon:** its fixed image and its pack.
- **What engineers did:** they tracked upstream PR streams and knew which nightly carries what, so
  "this does not work" became "this works in nightly X".
- **Upstream PRs:**
  - AITER #2822, #2900 and #2642 for DSV4 FP4 on ATOM (560);
  - vLLM #40889/#40871 (555, 731) and #47287, sparse paged attention (1050);
  - nine AITER/vLLM PRs for Kimi-K3 on MI355X (1171).
- **The failure is worse than not finding the feature.** In the pinned image, the feature *fails*,
  and the agent writes a dead end that steers later iterations away:
  - AITER MLA rejected 8 heads per GPU until vLLM ROCm v0.18 (425–427);
  - `--moe-backend aiter` raised a KeyError on FP4 scales (955).
- **Facts also expire with the image:**
  - Marlin won one week and lost the next (869 → 899/902);
  - assembly paged attention lost to Triton after an image change (926 → 941);
  - workaround disables outlived their bug (448).

### 2.3 One change at a time cannot climb a bundle

- **The gate is a greedy hill-climber.** Many wins are conjunctions where every partial state
  fails, regresses or sits under 1%:
  - MegaMoE in SGLang needs about a dozen undocumented `SGLANG_OPT_*` switches together, and flipping
    the main one alone fails (539/542/549);
  - TP4 needed four memory and batch knobs changed with it (265);
  - attention DP stalls without load balancing (130);
  - two-batch overlap needs `GPU_MAX_HW_QUEUES` raised on ROCm, or its streams serialize (877, 995);
  - vLLM's attention fusion needs `+quant_fp8`, and full decode graphs need empty `splitting_ops`
    (15).
  - The INT4 quick-reduce is a no-op for decode unless its minimum size is also lowered (1089); it
    then gave −12 to −17% time per output token on MiniMax-M3 MI355X.
  - Five `TRITON_HIP_*` codegen knobs are each under the gate (69).
- **The first step loses, so it is rejected and often written as a dead end.** That poisons the
  search.

### 2.4 The measurement cannot see the effect

The agent's evidence is a GPU profile and a throughput number. Many effects live elsewhere:
- **CPU cost:**
  - the scheduler's receive interval, tuned per concurrency to values from 2 to 1,920 (814);
  - the Rust frontend, −8% to −22% TTFT with tok/s neutral (823);
  - GC and allocator stalls (830, 841).
- **Accuracy:**
  - an FP8 combine passed on throughput and failed GSM8K (766, 609);
  - AITER corrupts output under TP-only and is a win under EP (935);
  - a decode-only MTP setting cut GSM8K from 0.96 to 0.88 with no error and no slowdown (1059).
- **Synthetic acceptance:**
  - draft depth is an input of the benchmark, so deeper drafts only add cost (1167);
  - a draft head trained on 32k collapsed on 100k+ agentic traces, 20–63% slower (1082).
- **Workload:** prefix hit rate, KV working set and session affinity (985, 1030, 1039). HiCache
  sized the obvious way OOM-killed 3 TB nodes, because MLA/DSA pools are replicated on all ranks.

### 2.5 The move looks wrong

- **The bias:** agents, and our packs ("principled solutions only"), lean towards adding and
  fixing.
- **Wins that went the other way:**
  - removing hand-written kernels so Inductor can fuse them, +6–7% (208);
  - removing fast-path disables that guarded an old image (448);
  - adding a batching wait (124, 841);
  - delaying prefill (907, 1235: +28% at c128);
  - `--linear-backend emulation` beating native MXFP8, +26% (972).
- **Why the loop misses them:** each needs a model of the system in which the removed thing was a
  constraint from another time, or in which waiting batches better. The tree does not record why a
  disable was added.

### 2.6 Silent no-ops look like "no effect"

- **The cases:**
  - an env var whose name changed was ignored (286, 367);
  - an env var was set but never exported (310);
  - launcher tests and refactors silently flipped EP (405/407, 448, 239→256);
  - config labels did not match the launch (322, 462).
- **Why the loop cannot tell:** its only feedback is the metric. A no-op reads exactly like a change
  that does not help, so the right idea gets written down as a dead end. Runner AGENTS.md §10 has
  the rule (prove the arms differed); the harness does not enforce it.

### 2.7 The best setting is a policy, not a value

- **Examples of the policies:**
  - MTP depth falls as the batch grows;
  - EP is off at small batches and on at large ones;
  - two-batch overlap: −14% at c64, +8–14% at c256 and above (1067);
  - the SWA token ratio depends on input length;
  - MegaMoE only at concurrency ≥2048 (564).
- **Why the loop misses them:** a campaign at one point finds one row of the table, and its ledger
  records a kill without saying where it applies. The next campaign, at another point, inherits a
  false dead end.

### 2.8 The knob is below the framework

- **Where the knobs live:** Triton's AMD backend (`TRITON_HIP_*`, `AMDGCN_USE_BUFFER_OPS`), the HSA
  runtime (`HSA_NO_SCRATCH_RECLAIM`, `GPU_MAX_HW_QUEUES`), AITER's tuned tables (`AITER_CONFIG_FMOE`
  routes token counts to FlyDSL kernels), and which shapes the assembly kernels support (794).
- **Why the agent cannot find them:** none appears in `--help` or `envs.py`, and a profile says
  "this kernel is slow", not which compiler switch fixes it. The recipes name 101 distinct
  AMD-side knobs.
- **A link to our own work:** `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`, named 7 times, is the
  flag that left model-w campaign A's iterations 11 and 19 dormant. A never set it.

## 3. What each fix can reach

| Mechanism | Knowledge (i) | Harness change | Mesh lanes (ii) |
| --- | --- | --- | --- |
| 2.1 wrong space (layout, topology, frontier) | names the layouts | — | **yes**: multi-node lanes judged on the frontier |
| 2.2 outside facts, image-bound features | **yes**: image and capability manifest, PR watch | image-scoped ledger | an upstream-integration lane |
| 2.3 bundles | **yes**: documented bundles | **yes**: a bundle is one claim | — |
| 2.4 blind measurement | the list of numerics knobs | **yes**: TTFT and ITL gated; an accuracy gate on numerics knobs; real acceptance | — |
| 2.5 counterintuitive moves | **yes**: removals as candidates, with each disable's reason | — | — |
| 2.6 silent no-ops | — | **yes**: prove the effective config per arm | — |
| 2.7 policies | **yes**: regime tables per model and GPU | **yes**: a second operating point before a kill; facts carry their scope | a lane per band |
| 2.8 below the framework | **yes**: vendor knob catalog, hardware KB | — | — |

**What our own campaigns add:**
- **Model M:** 272 iterations made one attempt at a core kernel, with 0 KEEPs. Expert, dense,
  attention and collective kernels, about 77% of decode, got no attempt at all
  (`plans/model-m/BC_PROBES_AND_KERNELS.md`).
- **Model M B5 and C5** wrote five kernel plans. C5 landed one stage, the staged peer reduce-scatter,
  for +1.9%, and no rewrite campaign followed.
- **Model W B** was asked for kernel plans and wrote none in 8 iterations.

**So knowledge alone will not produce rewrites.** The loop's 1% gate per iteration, one concern
per iteration, and no credit for a build that takes many iterations are what stop them.

## 4. Scope

### 4.1 Option (i): knowledge of big directions, and how to reach it

Keep it out of the always-rendered prompt, because packs already spend up to 89 KB per iteration.
Make it a reference that dossier part 11 and the map must consult.

1. **A catalog of big directions:** `arbor/knowledge/directions/` plus a `big-directions` skill.
   - **Entries:** MegaMoE (DeepGEMM `mega_moe`; AMD has no equivalent), persistent decode kernels,
     all-reduce + residual + RMSNorm fusion, two-batch overlap, DP attention with EP, decode context
     parallelism (DCP) for MLA KV, KV offload tiers (HiCache, LMCache, Mooncake), speculative-depth
     policies, quantized collectives (INT4/INT6 quick-reduce, FP8 combine), FlyDSL MoE.
   - **Each entry says:** its mechanism, when it pays (the regime), what it needs (image, topology,
     settings that go with it), the evidence (`records.jsonl` ids and measured effects), and its
     status on AMD.
   - **Draft source:** the 238 hard records.
2. **Regime tables, per model and GPU:** the known-best layout and settings per concurrency band,
   taken from the recipes.
   - A campaign at one point then inherits the policy instead of rediscovering one row of it.
   - The dossier says whether the pack's fixed layout is on that frontier.
3. **An image and capability manifest:** which image tag first carries which kernel or fix.
   - Alongside it, an inventory of workaround flags, each with the image that motivated it, to
     re-test on every image change.
   - An agent that hits a missing feature then reports "blocked by image" instead of writing a dead
     end.
4. **A vendor knob catalog** for ROCm, AITER, ATOM, MoRI, SGLang and vLLM-ROCm: the 101 knobs, each
   with its scope, its prerequisites, and whether it changes numerics.
5. **A list of numerics-changing knobs** that need an accuracy gate (GSM8K at the golden score)
   before a KEEP: INT4/INT6 quick-reduce, FP8 combine, FP4 activations, online FP8 quantization,
   index or top-k reuse, TF32, fp16 instead of bf16.

### 4.2 Option (ii): the mesh asks for plans explicitly

This is decision 0003, the mesh as Arbor's parallel mode, with three lane types:

- **A kernel-plan lane.** It runs the dossier skill's `kernel-plan.md` stages, S1 Reference to S5
  A/B.
  - Each stage is judged by its own gate, and only the last one by the campaign's A/B. The 1%
    throughput gate never judges an intermediate stage.
  - It ends when every item of the plan has a verdict with evidence, with no time limit (operator).
- **An architectural-plan lane.** Layout and topology plans: TP/EP/DP per band, the P:D ratio, DCP,
  the KV tiers.
  - It runs on the nodes it needs.
  - It is judged on the frontier across the operating points it names, not at one point.
- **An upstream-integration lane.** It ports a known PR or feature into the pinned tree, with a
  correctness gate and then an A/B.

**Where plans come from:**
- The map already closes a rewrite as `plan` when its kernel plan is written (map rounds).
- The mesh turns those into lanes, and the dossier's next round sees the results.

**Prerequisites:**
- A worked kernel-plan example on `jiawei-dev`. B5 and C5's plans sit on the un-pulled
  `<fork>/b5-results` branch.
- A kernel-plan template that points at real roofline numbers (§4.3).

### 4.3 The hardware knowledge base

**Fix first, now:**
- `arbor/knowledge/mi355x-optimization.md` is rendered into every prompt, and says "304 CUs, 256-bit
  memory bus". MI355X has 256 CUs (8 XCDs × 32) and 8 TB/s of HBM3E; 304 is MI300X.
- `docs/dossier/model-m-mi350x/07-profile.md` repeats the 304.

**Source:**
- KernelForge's `local_knowledge/hardware/` (22 files, about 103 KB, good quality):
  - CDNA4/MI350: architecture, peak tables, memory, block-scaled matrix cores, FP4/FP6
    microscaling, ISA notes, clocks;
  - CDNA3/MI300: the same, plus the XCD chiplet layout and occupancy;
  - shared: HBM and Infinity Fabric, L2/XCD swizzling, MFMA and SMFMAC, the LDS bank model,
    wavefronts and VGPR/AGPR, dtype numerics.
- KernelForge's own `gpu_arch_gfx950.md` is stale ("304 CUs", "32 banks", "5.3 TB/s") and must not
  be copied.

**Contents:** `arbor/knowledge/hardware/`, read through a skill, not rendered always.

| Part | What it holds |
| --- | --- |
| Peaks | Spec: FP4 10, FP8 5, BF16 2.5 PF; 288 GB of HBM at 8 TB/s. Achievable on our nodes, measured, with the method and node: MI350X 5.1–6.1 TB/s cold, 6.4–6.9 hot through the Infinity Cache; MI355X 5.6 TB/s in its dossier. One table, used by the profiler skill and `kernel-plan.md`, which today point at nothing. |
| Memory | 256 MiB Infinity Cache, L2 per XCD, LDS 160 KiB per CU with 64 banks, `global_load_lds_dwordx4`, `ds_read_b64_tr_b{4,8,16}`. |
| Compute | wave64, VGPR and AGPR, and occupancy. MFMA and scaled MFMA (`v_mfma_scale_f32_16x16x128_f8f6f4`), block-scale formats, packed fp32. No TF32 on gfx950, and no FP4 matrix instructions on gfx942. |
| Communication | XGMI and Infinity Fabric bandwidth; RCCL bus bandwidth measured on our nodes (TP8 all-reduce about 378 GB/s, reduce-scatter about 349); quick-reduce and its size thresholds; MoRI; `GPU_MAX_HW_QUEUES` for overlap. |
| Runtime and compiler knobs | `HSA_*`, `TRITON_HIP_*`, `AMDGCN_*`, each with its effect and scope. |

**Rule:** where a measured number disagrees with the datasheet, both are given with their sources,
and plans use the measured one.

### 4.4 Harness changes the study argues for

These help single-node campaigns now, independent of the mesh:
- **A bundle is one claim.** An iteration may declare a documented bundle from the catalog as one
  change.
- **Kills carry their scope.** A DEAD_END records its image digest, concurrency and input length. It
  goes stale when the image changes. For knobs that depend on the operating point (overlap, linear
  backend, chunked prefill, kernel backends), a second concurrency is checked before the kill.
- **Effective configuration per arm.** Dump the server environment and the selected backends, and
  check them against the claim. Runner AGENTS.md §10 states this as a rule; this makes it
  mechanical.
- **An accuracy gate** on the numerics list (§4.1.5). TTFT and per-token decode time are gated as
  well as tok/s (runner §9).
- **Removal as a candidate.** The map lists the tree's fast-path disables and workaround flags as
  directions to re-test.

## 5. Suggested order

1. **Now:**
   - fix the wrong hardware facts in the always-rendered file and the dossier;
   - import KernelForge's hardware files as the hardware knowledge base;
   - write the numerics-knob list and the vendor knob catalog, mostly by extraction from
     `records.jsonl`.
2. **Next:**
   - the big-directions catalog and the regime tables, drafted from the 238 hard records and
     reviewed;
   - the image and capability manifest.
3. **Then:** the harness changes in §4.4.
4. **With the mesh (decision 0003):**
   - the kernel-plan, architectural and upstream-integration lanes;
   - first, a worked kernel-plan example brought onto `jiawei-dev`.
