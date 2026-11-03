# Big directions: the optimizations local iteration rarely finds

A catalog of 22 large serving optimizations (architectural changes, multi-step builds, layout and
topology changes, cross-component fusions, scheduling policies): what each does, where it pays,
what it needs, the evidence, its AMD status, and how an Arbor campaign should take it on.

Built 2026-10-20 from the runner's docs/research/inferencex-study (records.jsonl ids) and
InferenceX origin/main (`6d6d296c6`, 2026-08-30). Also from: the runner's
`plans/model-m/BC_PROBES_AND_KERNELS.md` ("BC_PROBES"); Arbor's dossier skill (`SKILL.md` part 11,
`kernel-plan.md`); the Model M dossier `docs/dossier/model-m-mi350x/` (`03-code.md`,
`11-directions.md`).

**How to read it:**
- Read only the entries your dossier's part 11 or map names, or that match your profile. The
  table at the end is the index.
- `#655` is a record id of the InferenceX study (its records are not carried in this copy). `main:<path>` is a file on
  InferenceX `origin/main`.
- **NUMERICS** marks a fact that changes model output: it needs an accuracy gate (GSM8K at the
  golden score) before a KEEP. **REGIME** marks a fact that holds only at the stated operating
  point.
- "Missed by" names the mechanism in the study's `REPORT.md` §2 that kept agents away: 2.1 wrong
  search space, 2.2 the fact is outside the image, 2.3 a bundle, 2.4 the measurement cannot see
  it, 2.5 the move looks wrong, 2.6 silent no-op, 2.7 a policy not a value, 2.8 below the
  framework.
- GPUs: MI355X is gfx950 (256 CUs, 288 GB HBM3E, 8 TB/s); MI300X and MI325X are gfx942. Model M
  numbers are MI350X (gfx950). NVIDIA numbers never transfer as numbers, only as mechanisms.

## 0. Plan kinds, and the rules every big direction shares

| Kind | What it is | How to run it in Arbor today | Judged by |
| --- | --- | --- | --- |
| **increment** | one flag, table row, dispatch rule or adjacent fusion | one iteration | the campaign gate |
| **bundle** | a documented set whose members fail or sit under the gate alone | one iteration that applies the whole set, cites this file as the source, and proves each member took effect. "A bundle is one claim" is a proposed harness change (REPORT §4.4); until it lands, say so in the claim | the campaign gate, at two operating points |
| **rewrite** | a new kernel or fused region; many iterations before it can serve | dossier `kernel-plan.md` stages S1 Reference, S2 Correct, S3 Fast, S4 In tree (switch off by default), S5 A/B. Config `"space": {"levels": ["kernel"]}`, pack raises `policy.max_consecutive_probes` | each stage by its own gate; only S5 by the A/B |
| **layout** | TP/EP/DP/DCP width, replica count, P:D split | the pack must free the layout. Single node: compare layouts on the same node. Multi-node: the mesh's architectural lane (REPORT §4.2, decision 0003) | the frontier: tok/s/GPU against per-user interactivity, at every band the layout claims |
| **upstream** | a PR stack or newer image carries the feature | the upstream-integration lane: port, correctness gate, then A/B. Record the image digest | correctness, then the A/B |

The map's `kind` field takes only `increment|rewrite` today (`arbor/directions.py`, dossier part
11). Until the mesh lanes exist, a suggestion: record a bundle as an increment titled `bundle: …`,
and a layout or upstream direction as a rewrite whose plan is a layout or integration plan.

**Rules for all of them:**
- **Prove the arm differed.** Dump the server env and the selected backends per arm; check the
  kernel names in a trace. A renamed or misnamed env var stayed a silent no-op until a recipe
  cleanup fixed it (#286, #367); launcher changes silently flipped EP (#448, #239 to #256).
- **Never kill on one operating point.** Most entries below are policies (2.7). Before a
  DEAD_END, measure a second concurrency on the other side of the stated crossover, and record
  the image digest, concurrency and ISL in the kill.
- **Gate TTFT and per-token decode time as well as tok/s** (runner `AGENTS.md` §9). Several
  directions trade one for the other (#823, #1235).
- **Normalize per GPU** for any layout change (#265, #653).
- **NUMERICS entries need the accuracy gate**; a throughput gate alone kept a broken FP8
  combine (#609, #766).
- **A failure in the pinned image is "blocked by image", not a dead end** (#425, #955).

---

## A. MoE and communication

### D1. MegaMoE: dispatch + expert GEMM + combine in one persistent kernel

- **Mechanism.** DeepGEMM's `mega_moe` runs EP dispatch, the grouped expert GEMMs and combine in
  one persistent kernel over symmetric memory (NVSHMEM), overlapping the NVLink all-to-all with
  GEMM and removing per-stage launches and the HBM round trips of permuted tokens (#559, #566).
  W4A4 variants quantize activations to MXFP4: half the dispatched bytes, FP4 tensor cores (#752,
  #802, #969).
- **Pays.** REGIME: DP attention + EP (DEP) at large per-rank batch. SGLang B300 used it at conc
  2048 to 8192 (256 to 1024 sequences per rank, #564, #571); TRT-LLM only at ISL <= 1k and conc
  >= 2048 (#830); vLLM v0.20 only on the DEP path (#559, #568), B300 TP4/EP4 DP at c256 to 1024
  (#655).
- **Loses / off.** TP-only low concurrency runs FlashInfer MXFP4 MoE instead (conc <= 32, #523,
  #542). The B300 MTP DP band switched MegaMoE off (#562). Cross-node symmetric-memory rendezvous
  failures made recipes drop the explicit backend (#618).
- **Needs.** An image with DeepGEMM `mega_moe` (vLLM v0.20.0+, SGLang DSv4 images by digest). In
  SGLang about a dozen switches together: `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`, `FIX_HASH_MEGA_MOE`,
  `FIX_MEGA_MOE_MEMORY`, `FIX_NEXTN_MEGA_MOE`, `MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` sized to per-rank
  tokens; the main switch alone fails (#539, #542, #549). Lower memory utilization for its
  buffers (0.85, #559); `TORCH_SYMMMEM=NVSHMEM` and an EP weight filter (#566, #1119); token caps
  per role (prefill 9216 / decode 4096, #1161).
- **Evidence.** #539, #542, #549, #559, #564, #566, #568, #571, #655, #752, #802, #938, #969,
  #1119, #1161. No record isolates its effect; it always lands inside a layout change.
  NUMERICS for the W4A4 and FP8-combine variants (#752, #969).
- **AMD.** **Missing.** No AMD recipe on main names it. Model M dossier (`03-code.md`, MoE expert
  GEMM row): "AMD has no megakernel MoE. Pattern is portable; NVFP4/DeepGEMM is not." AITER's
  path is about 5 to 6 visible launches per layer plus its own sort and two grouped GEMMs. The
  pieces AMD has: AITER fused MoE, MoRI EP dispatch/combine (#261, #616), two-batch overlap (D5).
- **Arbor.** **Rewrite**, and only where the layout is EP with an all-to-all and per-rank batch
  in the hundreds; at TP-only there is no dispatch to fuse (go to D2). S1 measures the
  dispatch+GEMM+combine region in situ (µs/layer, exposed all-to-all time). S5 runs at two
  concurrencies, one below the DEP crossover to show the switch stays off there. The Model M map
  set `m-moe-megakernel` aside at TP8: decode was in a FULL graph, idle 0.44%, and the collective,
  not inter-stage traffic, dominated (`11-directions.md` §6). A megakernel only wins collective
  time if it overlaps it.
- **Missed by.** 2.2, 2.3, 2.7.

### D2. Persistent decode kernels

- **Mechanism.** Workgroups stay resident and loop over work items: no per-kernel launch and
  tail-wave cost, fewer HBM round trips between stages. Graph capture removes the host's launch
  cost, not the GPU-side per-kernel floor. Model M C8 decode: 1,250 to 1,360 launches per step at a
  1.7 to 2.0 µs floor each; 919 small kernels were 35.5% of the step; each layer a serial chain of
  19 to 21 kernels (BC_PROBES §2).
- **Instances.** AITER persistent MLA decode `SGLANG_AITER_MLA_PERSIST=1` (#279, #282, #711,
  #746), valid only with FP8 KV (SGLang PR 17327, #287). A persistent asm MLA verify kernel for
  spec decode, Kimi-K3 MI355X (#1171). DeepGEMM MegaMoE (D1). TileRT, a batch-1 tile runtime used
  as the decode engine on B200 (#1144, #1230).
- **Pays.** REGIME: decode at a few rows per step, where the step is a chain of short kernels.
  Our DSV4.1-Flash AgentX runs 6 to 18 rows/step at C8 (runner `AGENTS.md` §8). Loses at large
  batch, where kernels are long and occupancy is already full.
- **Needs.** Graph capture under the served mode; per-step counts as runtime arguments (a
  `tl.constexpr` route count compiled 2,218 variants and stalled the engine 30 s, Model M B4 it. 22,
  `kernel-plan.md` §4); FP8 KV for AITER MLA persist (NUMERICS).
- **AMD.** MLA persist: available (SGLang ROCm from v0.5.7, #279; 6 AMD recipes on main). A
  persistent MoE block or layer: **missing**. Model M ranked `m-moe-megakernel` #6 and no campaign
  probed it. BC_PROBES' kernel plan, item 1: a persistent decode MoE block for <= 8 rows (route,
  sort, quant, both expert stages, finalize in one kernel).
- **Arbor.** Increment for MLA persist (with FP8 KV: accuracy gate). **Rewrite** for a new one.
  S1 needs an independent HBM roofline: Model M's "at floor" kills rested on a floor measured with
  the kernel under test (BC_PROBES). Measure launches per step (census on `ProfilerStep#`), decode
  step µs at the served rows/step, paired per-request ITL at C8 and C32.
- **Missed by.** 2.3 (a build of many iterations under a 1% gate), 2.8.

### D3. All-reduce + residual + RMSNorm (+quant) fusion

- **Mechanism.** Fold the TP all-reduce into the next residual add and RMSNorm (and FP8 quant):
  one kernel instead of a collective, a norm and a quant, and no HBM round trip. About 80
  standalone collectives per forward removed on GLM-5.2 (#0).
- **Pays.** REGIME: TP > 1, latency-bound decode messages. FlashInfer's variant is used only up to
  a message size per world size: 32 MB at TP2/TP4, 8 MB at TP8 (#15).
- **Loses / off.** Disabled with the FlyDSL w4a16 MoE on Kimi-K2.5 INT4 MI355X (#867); on H200
  Kimi-K3 (#1127); in a GLM-5.1 FP8 GB200 disagg recipe (#1013). Symmetric-memory fusion breaks
  when TP spans nodes (#1156).
- **Evidence.** GLM-5.2 MXFP4 MI355X SGLang TP4/EP4 c8: `--enable-aiter-allreduce-fusion` with
  fused QK-norm+RoPE and FlyDSL MoE sorting, +23.6% P90 interactivity, 41.9 to 51.8 tok/s/user
  (#0, #1255; a bundle of three). AMD SGLang adds it in #465, #494, #827, #944, #1163. vLLM's pass
  fires only as a set: `enable_fi_allreduce_fusion` + attention fusion + `+quant_fp8` custom op +
  FULL_DECODE_ONLY with empty `splitting_ops` (#15, NVIDIA).
- **AMD.** SGLang ROCm: available (`--enable-aiter-allreduce-fusion`, 16 AMD recipes on main).
  vLLM ROCm: no AMD recipe on main enables `fuse_allreduce_rms`, one disables it (#867). Check that
  the pass matches on ROCm in your image before claiming it.
- **Arbor.** Increment where the flag exists; bundle in vLLM (#15). Rewrite only if your stack
  lacks it. Measure collectives per forward (census) and the decode step; per-request ITL at C8
  and C32.
- **Missed by.** 2.3, 2.8. Easy where the flag exists: the three agent-coauthored changes in the
  history were easy local flags, two of them this one (#0, #1255; REPORT §1).

### D4. Quantized collectives

- **Mechanism.** Fewer bytes on the wire. TP all-reduce quantized to INT4 or INT6 (AMD
  quick-reduce); EP dispatch in FP8 or FP4 and combine in FP8 (MoRI, DeepEP, TRT-LLM
  low-precision combine; NVFP4 dispatch on NVIDIA, #920, #979).
- **Pays.** REGIME: bandwidth-bound collectives, high-concurrency decode and prefill.
  MiniMax-M3 MXFP4 MI355X vLLM TP4: INT4 quick-reduce gave -12 to -17% TPOT at c64/128/256, **only
  with `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256`**: the default INT4 gate for bf16 TP4
  is 16 MB and the decode all-reduce is about 1.5 MB, so without it nothing fires (#1089). INT4
  quick-reduce 2.2x against INT8's 1.17x (#425); INT6 about 2.7x fewer bytes than bf16 (#935).
  #1050 moved INT6 to INT4 at every point and, under a gate (ISL >= 8192 and conc >= 64), added
  `CAST_BF16_TO_FP16=0` and the 256 KB minimum.
- **Loses.** NUMERICS. MoRI FP8 combine passed throughput and failed GSM8K far below 0.95 at high
  concurrency (#609). The fix is a per-phase combine dtype
  (`MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast`, `MORI_COMBINE_DTYPE_DECODE=fp8`, blockwise) in a
  newer image (#766). For MXFP4 checkpoints, FP8 dispatch re-quantizes FP4 activations: turned
  off (#406).
- **AMD.** Available. vLLM ROCm `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`, SGLang ROCm
  `ROCM_QUICK_REDUCE_QUANTIZATION`, ATOM `AITER_QUICK_REDUCE_QUANTIZATION` (27 AMD recipes on main);
  MoRI dispatch/combine dtypes in AMD multi-node recipes. On Model M (MI350X) the QuickReduce INT6
  KEEP is what made sequence parallelism lose (D6).
- **Arbor.** Bundle of two or three env vars (quantization, minimum size, bf16 cast). Prove it
  fired: the quick-reduce kernel appears at decode message sizes in the census. Accuracy gate.
  Measure TPOT at two concurrencies.
- **Missed by.** 2.3, 2.4, 2.8.

### D5. Two-batch overlap (TBO) and communication/compute overlap

- **Mechanism.** Split each step into two micro-batches so one's MoE dispatch/combine or
  all-reduce runs while the other computes.
- **Pays.** REGIME: needs enough tokens per step. DSv4 FP4 MI355X ATOM with MTP: -14% and -10%
  output throughput at c64 and c128, +11%, +8%, +14% at c256 and above (#1067). Gates moved with
  the image: 1k1k c >= 1024 and 8k1k c >= 256 (#877), then c >= 64 without MTP (#1068;
  `main:benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_atom.sh`). With MTP, decode TBO drops
  the spec metadata, so only prefill TBO is used (#1067).
- **Needs.** On ROCm, `GPU_MAX_HW_QUEUES=5`, or the two streams serialize (#877, #995). The DP
  attention + EP layout and its collectives (D8). On MoRI, SDMA engines for prefill overlap
  (`MORI_ENABLE_SDMA` with TBO, #616).
- **Model M evidence (MI350X).** With the shipped RCCL collective, 0% of the collective was hideable
  behind compute (`11-directions.md`, iteration 16, campaign B2). Reopen only when a collective
  yields the device (an SDMA path, or a persistent co-scheduled comm kernel), and measure overlap
  efficiency >= 0.6 before pricing anything. C3's dual-batch overlap was a KEEP that reworked a
  killed item (BC_PROBES §1).
- **AMD.** Available: SGLang ROCm `--enable-two-batch-overlap`
  (`main:…/dsv4_fp4_mi355x_sglang.sh`, with `GPU_MAX_HW_QUEUES=5`), ATOM `--enable-tbo`, MoRI EP
  TBO since SGLang v0.5.9 (#368).
- **Arbor.** Bundle (TBO + HW queues + D8's collectives), applied as a policy by concurrency.
  Check in a trace that the two streams actually run concurrently. Measure tok/s/GPU and ITL on
  both sides of the crossover.
- **Missed by.** 2.3, 2.7, 2.8.

### D6. Sequence parallelism around attention (measured loss on AMD)

- **Mechanism.** Shard the residual stream along tokens: the o_proj collective becomes a
  reduce-scatter, norms and residual cost 1/TP, an all-gather restores tokens for attention.
  Model M's NVIDIA path does it (with a fused GEMM+reduce-scatter gated to SM103, `03-code.md`).
- **Evidence (Model M, MI350X TP8, `11-directions.md` iteration 15).** Mixed step, T = 8192: SP cost
  +419.8 µs/layer against the served QuickReduce INT6 incumbent, and still +190.5 µs/layer
  against plain RCCL all-reduce. Decode, T = 32: +36% of the decode step. What SP would save was
  <= 655 µs/layer, and a column-parallel pooler had already bought a quarter of it. Runner REPORT
  §3: Model M C5 landed one stage, a staged peer reduce-scatter, for +1.9%.
- **Pays.** Prefill and mixed steps with large token counts and much replicated non-expert work.
  Not at decode on MI350X.
- **AMD.** No fused GEMM+collective kernel in AITER; primitives exist (AITER Triton
  reduce-scatter, Iris symmetric memory, `03-code.md`).
- **Arbor.** Rewrite, mixed-step only, priced against the quantized all-reduce actually served.
  Measure each boundary in situ, paired and alternating, decode and mixed separately.
- **Missed by.** 2.4 (pricing against the wrong incumbent).

---

## B. Parallel layout

### D7. Smaller-TP replicas under per-GPU normalization

- **Mechanism.** When the model fits in fewer GPUs and the point is not compute-bound, a TP4
  (or TP2, TP1) replica gives about the same per-user speed on half the GPUs, with fewer
  all-reduce participants. Per-GPU throughput rises while a single replica's tok/s falls.
- **Evidence.** GLM-5 FP8 MI355X SGLang TP8 to TP4, conc to 256 ("Turn to tp=4 for best perf",
  #653). MiniMax-M3 MXFP8 MI355X vLLM: TP4 beats TP8 per GPU and matches TP4/EP4 (#972).
  Kimi-K2.5 FP4 MI355X: TP4 decode equals TP8 per replica; EP over MoRI was 14 to 27% slower than
  dense TP (#1051). Qwen3.5 FP8 MI355X disagg: TP4 prefill + TP8 decode, same interactivity,
  +24 to 31% tok/s/GPU (#1210). Real conversation pacing moved a Qwen3.5 GB300 frontier to
  aggregate TP2 (#1125).
- **Pays / loses.** REGIME: low and mid concurrency, memory-bound decode, model plus KV fits
  (288 GB per MI355X). Loses the lowest c1 latency (TP8 keeps it) and KV room at long context
  (then D10).
- **Needs.** A pack that frees TP and scores per GPU; memory knobs changed with it. DSR1 FP8 B200
  TP4 only booted with KV room when mem-fraction 0.95, max-running 32, graph bs 32 and chunk 8192
  changed together (#265).
- **AMD.** Available everywhere; mainline AMD searches are often TP4-only (#1089).
- **Arbor.** **Layout plan, single node.** Compare TP8 against 2 x TP4 (or TP4 on four GPUs,
  normalized) at three or more concurrencies, on tok/s/GPU against per-user interactivity, plus
  paired per-request ITL and TTFT.
- **Missed by.** 2.1.

### D8. DP attention + EP, and its scheduling bundle

- **Mechanism.** MLA and DSA keep one latent KV head, so TP attention replicates KV and repeats
  the work on every rank. With DP attention each rank holds its own requests and KV and skips the
  attention all-reduce; the MoE becomes an EP all-to-all or a gather. It pays once every rank has
  enough requests (#130, #805).
- **The bundle.** DP attention alone can lose: ranks wait for the slowest at every MoE sync
  (#130).
  - *Balance:* TRT-LLM `attention_dp_config` balance with `batching_wait_iters` (#130; 30 in
    #841). SGLang `--enable-prefill-delayer`, `--schedule-conservativeness 3.33`,
    `--enable-dp-attention-local-control-broadcast`, `--enable-dp-lm-head`,
    `--scheduler-recv-interval 1` (#907, #948). Under spec decode + DP, `--prefill-decode-interval
    20`: +28% output throughput at c128 on B300, with TTFT p50 2.2x and p90 3.3x worse (#1235).
  - *Collectives:* `SGLANG_DP_USE_GATHERV` / `SGLANG_DP_USE_REDUCE_SCATTER` move only real tokens
    instead of padding every rank to the max (#901, #995). Shared expert local at TP1 per rank:
    `SGLANG_SHARED_EXPERT_TP1`, `SGLANG_DP_SHARED_EXPERT_LOCAL` (#995).
  - *Memory:* max-num-seqs = CONC on every DP cell, since each rank holds a full KV pool and the
    default OOMs (#1068); graphs sized to the per-rank batch (#564, #1134).
  - *EP:* EP must equal TP under DP attention, or experts are TP-sharded and replicated (#256).
    `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` keeps idle ranks' padding from hitting the same experts
    (#1146).
- **Regime.** DSV4 FP4 MI355X: ATOM TP8 DP-off c1 to 64, DP-on c >= 64 (#805); SGLang main config:
  TP4 DP-off c1 to 32, TP4 DP-on c16 to 128, TP8 DP-on c64 to 2048
  (`main:configs/amd-master.yaml`, `dsv4-fp4-mi355x-sglang`). DSR1 TRT-LLM B200: DEP at c >= 256
  (#130).
- **AMD.** Available (SGLang ROCm v0.5.14, ATOM nightlies; `main:…/dsv4_fp4_mi355x_sglang.sh`
  carries the full set). Known failures: DSA + DP attention hangs a collective in long-context
  prefill on ROCm (GLM-5.2 #1158, DSV4 agentic #1181); with MoRI, SGLang mis-counts shared-expert
  slots on Qwen3.5 (#1210); EP8 gave garbage on MiniMax-M3 MI300X (#982).
- **Arbor.** **Bundle + layout plan, single node.** Never judge DP attention without its balance
  knobs, or at one concurrency. Measure per-rank batch and idle time (rank skew), tok/s/GPU in the
  high band, the TTFT tail (the delayer trades it), ITL.
- **Missed by.** 2.1, 2.3, 2.5 (delaying prefill helps), 2.7.

### D9. Wide EP with redundant experts (EPLB)

- **Mechanism.** Spread experts over 16 to 56 GPUs: each holds few, weight reads per GPU fall,
  per-expert batches grow. Redundant replicas of hot experts even out the load.
- **Evidence (NVIDIA, multi-node).** GB200 DEP32 with 256 or 288 expert slots (#299); GB300 DEP32
  with 384 EPLB slots (#855); GLM-5 GB300 EP 16 to 40, 32 redundant experts, static dispatch
  (#775), up to DEP56 (#984); an 18-node EP-width frontier from EP12 to EP40 (#822). Single node:
  B200 vLLM DEP8 with `--enable-eplb` (#816).
- **AMD.** Wide EP over MoRI RDMA: DEP16 across two MI355X nodes (#261). No EPLB setting in any
  AMD record. Model M (MI350X): decode expert skew was a fresh per-layer draw, not persistent
  hotness, so static EPLB was predicted null at decode, and its runner raises
  `NotImplementedError` on EPLB (`11-directions.md`, iteration 28).
- **Arbor.** Width is a **layout plan** for the mesh's multi-node lane. EPLB on a single-node DEP
  is an increment, after measuring whether the skew persists over time (per-rank expert load
  histogram, all-to-all wait per rank).
- **Missed by.** 2.1, 2.4.

### D10. Decode context parallelism (DCP) for MLA KV, and PP across nodes

- **Mechanism.** TP replicates MLA's single latent KV head on every rank. DCP shards each
  request's KV across the ranks, with an all-to-all per decode step: about TP times the KV
  capacity, and a split KV read.
- **Evidence.** Kimi-K3 B300 TP8 x DCP8: the sweep reached c70 where the resident arm stalled at
  c16 (#1189). Kimi K2.6 B300 TP4 x DCP4, a2a backend, c32/64 (#1022, #1029). Kimi-K3 MI355X ATOM:
  DCP8 from c8, off at c1 to 4 (#1254). For a model that needs two nodes, PP2 x DCP8 per stage
  sends only stage-boundary activations across nodes and took the frontier to c96, against c44
  for TP16 x DCP16 (#1213, #1216).
- **Pays / loses.** REGIME: long context (100K+), KV capacity binding, enough load to amortize the
  extra collectives. Off at c1 to 4 (#1254).
- **Needs.** Branch or nightly images (vLLM `agentx-k3` at `5894fdf`, #1189);
  `--dcp-comm-backend a2a`; direct-gather toggles measured per image (`VLLM_USE_DIRECT_DCP_*`,
  #1189, #1212).
- **AMD.** ATOM image `kimi_k3_agentic_0821` has `--decode-context-parallel-size` (#1254;
  `main:…/agentic/kimik3_fp4_mi355x_atom_mtp.sh`, dcp-size 8 at c8/12 with LMCache). No vLLM ROCm
  or SGLang ROCm recipe uses it.
- **For our DSV4.1-Flash AgentX** (runner `AGENTS.md` §8): decode is 87 to 95% of request time at
  134K to 221K context, and the KV cache never exceeds 14%. Capacity is not the lever; DCP would
  have to win on the split KV read alone.
- **Arbor.** **Layout plan, single node** (TP8 x DCP8 against TP8), plus upstream integration if
  the image lacks it. Measure KV capacity in tokens, the highest sustainable concurrency, ITL by
  input-length bucket, collective time per step.
- **Missed by.** 2.1, 2.2, 2.4.

### D11. Disaggregated prefill/decode and the P:D ratio

- **Mechanism.** Separate prefill and decode pools: decode steps no longer stall behind prefill
  (steady ITL), each phase gets its own parallelism, and the P:D ratio is matched to ISL/OSL and
  concurrency.
- **Evidence.** Per-segment topologies: TP decode with deep MTP at low concurrency, DEP decode with
  shallow MTP at high (#233, #292, #316). Long inputs need more prefill: 2P3D (#311), 10P8D
  (#328), up to 26P1D (#1252). MiniMax-M3 MXFP8 MI355X vLLM, all-TP4 workers and P:D per ISL:
  8k1k peak about 2085 to 6003 tok/s/GPU (2.9x), 1k1k 1736 to 2586 (1.5x) (#1001).
  **Single-node disaggregation exists on NVIDIA:** four TP1 prefill engines and one TEP4 decode
  engine on one 8-GPU B300, KV over NVLink (#1065); DEP2 prefill + TP4 decode colocated (#903,
  #1015, #1110).
- **Traps.** Decode admitted over-length requests that prefill rejected: a conc-500 straggler
  tail (#1052). Spec flags must go to both roles, or the draft state is never seeded (GSM8K 0.96
  to 0.88 with no error, #1059). RDMA traffic class and QP depth come from the fabric's QoS (#929).
- **AMD.** Multi-node: SGLang + MoRI (KV transfer and EP), vLLM + MoRI-IO, ATOM (#261, #305,
  #866/#912, #929, #996, #1028, #1041). Single-node colocation: no AMD recipe on main.
- **Arbor.** **Layout plan** for the architectural lane, judged on the frontier. A single-node
  colocated P/D needs a router and a KV connector in the stack, so on AMD it starts as upstream
  integration. Measure tok/s/GPU against interactivity per band, TTFT and ITL distributions, KV
  transfer time, prefill queue depth.
- **Missed by.** 2.1.

---

## C. KV cache and workload

### D12. KV offload tiers (HiCache, LMCache, Mooncake, native; state for hybrids)

- **Mechanism.** Keep evicted prefixes in host DRAM or a remote store and reload them instead of
  re-prefilling: multi-turn agentic traces return to long prefixes after idle gaps.
- **Evidence.** GLM-5.2 B300: past c8 (TP8) or c64 (DP8) KV usage pinned at 1.0 and the radix hit
  rate fell from a 0.97 ceiling to 0.04 to 0.06. HiCache at host/device ratio 0.75 (about 1.0 TB
  pinned) restored it; ratio 2 or GB-based sizing OOM-killed 3 TB nodes, because MLA/DSA device
  pools are replicated on all 8 ranks (#1039). LMCache, DSV4 B200 DEP8: peak about 25.7k
  tok/s/GPU at c72 with 96 to 98% hits (#1043).
- **Traps.** cuMem/VMM allocations cannot be exported over CUDA IPC: drop
  `--enable-cumem-allocator` or `expandable_segments` on LMCache arms only (#1043, #1044).
  Hybrids need a state tier too: Kimi-K3 ATOM `OFFLOAD_STATE=1` with
  `OFFLOAD_GPU_STAGING_CHUNKS=16`, or the tier silently declines (a 54.78 MiB state entry against
  an 8 MiB buffer, #1214); its measured CPU-tier hit rate was 0.0% at c8/12 (#1254). Skip HiCache
  at low load (#1167). Sliding-window models: free out-of-window slots (#985, #1181).
- **Pays.** Workload-bound: multi-turn reuse and a working set larger than HBM. Worth zero on
  random prompts (Model M dossier, set aside). For our DSV4.1-Flash AgentX the KV cache stays under
  14% (runner `AGENTS.md` §8): not a lever at today's points.
- **AMD.** Available: SGLang ROCm HiCache (ratio 1.5, write_through, #1158, #1181); vLLM ROCm
  native `OffloadingConnector` (#1073); LMCache built from source with HIP (#997, #1165, #1208);
  ATOM LMCache plus state offload (#1214).
- **Arbor.** Bundle, on a trace with real multi-turn reuse. Measure the prefix hit rate against
  its theoretical ceiling, KV usage, re-prefilled tokens, TTFT per turn, host memory.
- **Missed by.** 2.3, 2.4.

### D13. Prefix-cache-aware routing and session affinity

- **Mechanism.** With several workers or DP ranks, a turn hits cache only on the rank that holds
  its prefix. Route by a session key (consistent hashing on a correlation id) or by KV overlap
  (KV-aware routing fed by KV events).
- **Evidence.** DSV4 MI355X SGLang: `sglang-router` in front of the DP ranks, consistent hashing,
  dp-aware, correlation-id key (#1030). The client must send the key
  (`AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true`, #1035). Dynamo: a router temperature
  of 1e7 made routing near-random; temperature 0 with session headers restored hits (#1150).
  Prefix hashes must match across ranks (`PYTHONHASHSEED`, #1026, #1043). Weighted-random picking
  spreads load at high concurrency (#1146).
- **AMD.** Available: SGLang router on a single MI355X node (#1030).
- **Arbor.** Cross-component bundle: router, client header, server cache. Measure the hit rate per
  rank, load balance across ranks, TTFT.
- **Missed by.** 2.4; it spans components the loop does not edit.

### D14. Prefill admission and chunking policy

- **Mechanism.** Decide when and how much prefill enters a step. Waiting to batch prefills, or
  bounding them, interrupts decode less.
- **Evidence.** Delay batching until 90% of the token budget or 20 iterations (#124); chunk size
  = ISL, so one prompt is one chunk per rank (#901); a long-prefill token threshold and a prefill
  schedule interval (#1000, #1031, #1194); small 8192-token chunks so one 100K agentic prefill
  cannot monopolize the engine (#1181); 131072 to 32768 at low concurrency (#1167); scheduler
  receive interval tuned per concurrency from 2 to 1,920 (#814); ATOM `--scheduler-delay-factor 1`
  (#1003, #1028).
- **Regime.** REGIME: every value is per concurrency, and trades TTFT against throughput (#814).
- **AMD.** Available (SGLang ROCm, ATOM, vLLM ROCm).
- **Arbor.** Policy increments, gated on TTFT and ITL as well as tok/s, at two or more
  concurrencies. Adding a wait is counterintuitive; keep it on the board.
- **Missed by.** 2.4, 2.5, 2.7.

---

## D. Decode policy

### D15. Speculative decoding as a policy

- **Mechanism.** Draft K tokens, verify 1 + K positions per step. In memory-bound decode at small
  batch the extra tokens ride the same weight read; at saturation the verify width is real
  compute.
- **The policy.** MTP depth 3 under TP attention, 1 under DP attention (#269); depth tapered
  3, 2, 1 as concurrency rises (#1090); Kimi-K3 MI355X ATOM K = 7 (AL 3.84) at c1 to 4, K = 3 at
  c8 to 16, none from c32 (#1254); dropped at saturation (#1252); K = 7 loses to no speculation
  once the GPU saturates even at a golden AL of 3.84, while K = 2 clears break-even (#1082); per
  role in disaggregation, 1 draft token on prefill and 3 on decode (#1034, #1110).
- **Real against synthetic acceptance.** Under synthetic rejection sampling the acceptance length
  is an input of the benchmark, so a deeper draft only adds cost unless AL is re-measured (#1167:
  GLM-5.2 MI355X, steps 3 to 5 with `SGLANG_SIMULATE_ACC_LEN` 2.99 to 3.61). A draft head trained
  at 32k collapsed to AL 1.16 to 2.01 on 100K+ agentic traces: 20 to 63% slower with real
  verification (#1082). Our DSV4.1-Flash replay fixes acceptance at 50.2% (runner `AGENTS.md` §8).
- **NUMERICS.** DSV4 MI355X SGLang: EAGLE depth 1 holds GSM8K at about 0.96, depth 2 drops it to
  about 0.85 (#1059).
- **Companion.** vLLM rounds capture sizes to multiples of 1 + K: give sizes in tokens, or graphs
  cover only MAX_NUM_SEQS/3 sequences (#1082).
- **AMD.** Available: SGLang ROCm EAGLE/MTP with `SGLANG_ENABLE_SPEC_V2=1` (#744, #746), ATOM
  `--method mtp` (#807), DSpark in ATOM (#1214, #1254). With TBO, prefill-only TBO (#1067).
- **Arbor.** The deliverable is a table by concurrency, one increment per row. Under synthetic
  acceptance, depth cannot be judged: it needs a real-acceptance run on the target trace, or an
  AL table per depth (#1221). Measure ITL and tok/s/GPU per band, real AL, GSM8K.
- **Missed by.** 2.4, 2.7.

---

## E. Kernels and compilation

### D16. Graph coverage: capture sizes and mixed steps

- **Mechanism.** A decode batch outside the captured sizes runs eager; mixed prefill/decode steps
  need piecewise graphs; FULL_DECODE_ONLY captures the whole decode step.
- **Evidence.** Piecewise graphs for prefill-bearing steps (#297, #303). Graph max batch = CONC
  (#653, MI355X; #995). Per-rank sizing: just above the per-rank batch (4096/8 = 512, so 544, #564;
  1024 per rank, so 1088, #571); per DP engine, not global concurrency (#1134); twice concurrency
  for agent fan-out bursts (#1229). FULL_DECODE_ONLY only on high-concurrency DEP decode (#1112).
  Full decode graphs need empty `splitting_ops` (#15). Caps forced by bugs (1024, #830). Kimi-K3
  MI355X needed vllm#51171 backported for FULL graphs with AITER MLA spec decode (#1171). DSV4
  ATOM went from `--enforce-eager` to graphs once batched sparse attention landed (#652).
- **AMD.** Available: vLLM ROCm FULL_AND_PIECEWISE (`main:…/dsv4_fp4_mi355x_vllm.sh`), ATOM
  `--cudagraph-capture-sizes`, SGLang `--cuda-graph-max-bs`.
- **Arbor.** Increment, verified by counting: the share of steps replayed from a graph at the
  served batch distribution (our C32 runs 30 to 160 rows per step, `AGENTS.md` §8). Mixed steps
  are where coverage is lost.
- **Missed by.** 2.4, 2.6, 2.7.

### D17. Compiler-driven fusion: remove hand kernels so Inductor can fuse

- **Mechanism.** Opaque custom ops block fusion. Traced as plain PyTorch, residual add + RMSNorm
  + FP8 quant and SiLU-mul + quant become single kernels.
- **Evidence.** Llama-3.3-70B FP8 on MI300X, MI325X and MI355X vLLM: `custom_ops ["-rms_norm",
  "-quant_fp8", "-silu_and_mul"]`, +6 to 7% on 6 of 8 configs (#208). Fusion passes need the op
  visible as a custom op (`+quant_fp8`, #15). AMD vLLM v0.17: `fuse_rope_kvcache` with
  `use_inductor_graph_partition` (#396); v0.16 applies add+RMSNorm+pad automatically (#367).
  Counterpoints: DSV4 B200 ran with torch.compile off (mode 0) and FULL_DECODE_ONLY (#1026);
  `fuse_minimax_qk_norm` fails on ROCm, imported only under `is_cuda()` (#794).
- **AMD.** Available in vLLM ROCm; pass coverage on ROCm varies by pass (#794).
- **Arbor.** An increment that is a removal. It needs compile mode on. Measure kernels per layer
  and the decode step; compare against AITER's hand fusions (D3).
- **Missed by.** 2.5.

### D18. Long-context decode attention: split-KV, sparse and indexer kernels

- **Mechanism.** At long context and small batch, attention is a KV read spread over too few CUs.
  Split the KV across more CUs (two-stage), read only selected blocks (DSA/MSA sparse), and make
  the indexer cheaper (FP8/FP4 indexer cache, top-k reused across layers).
- **Evidence.** AITER two-stage split-KV MLA decode in vLLM ROCm `nightly-09663abd`, "reduces
  decode attention latency across all concurrency levels" (#955, #957). AITER Gluon page-16 sparse
  paged attention (vllm#47287), in a bundle with INT4 quick-reduce and the linear backend:
  MiniMax-M3 MI355X 8k1k +3.3% c128, +2.5% c256, +4.7% c512 tok/s/GPU, neutral elsewhere (#1050).
  gfx950 Triton sparse-MLA decode (vllm#52212) and long-context `num_kv_split` retuning (#1201).
  The asm paged attention chosen one week lost to Triton after an image change (#926 to #941).
  Index top-k reuse every 4 layers is NUMERICS (#877, #941). DSV4 runs sparse MLA (#555).
- **Relevance.** Our DSV4.1-Flash decode is at 134K to 221K context and 1 to a few requests
  (runner `AGENTS.md` §8): this is the region the workload says matters most.
- **AMD.** Available only in specific nightlies; the kernels live in AITER (asm, Gluon, Triton).
- **Arbor.** Upstream integration first (port the AITER/vLLM PR), else **rewrite** at the served
  context lengths. Measure attention µs per layer by context bucket and per-request ITL by
  input-length bucket.
- **Missed by.** 2.2, 2.4.

### D19. Vendor kernel families picked by tuned tables

- **Mechanism.** AITER carries several families per op (FlyDSL, CK, asm, Triton, hipBLASLt) and
  dispatches by shape. For a new model's shapes the default heuristic often picks a slower one,
  and the switch is an env var or a table, never a framework flag.
- **Evidence.** DSV4 FP4 MI355X ATOM: a tuned fMoE CSV registered with `AITER_CONFIG_FMOE` routes
  eligible token counts to FlyDSL kernels (#560). `--moe-backend flydsl` on Kimi-K2.5 INT4 MI355X
  (#867); `AITER_FLYDSL_FORCE=1` (#863); `AITER_USE_FLYDSL_MOE_SORTING=1` (#0). The aiter#3832
  gfx950 MXFP4 MoE backend with `AITER_MXFP4_INTERMEDIATE=1` and `VLLM_ROCM_USE_SKINNY_GEMM=0`
  (#1096). `--linear-backend emulation` (dequantize MXFP8, run the tuned BF16 GEMM): +26% tok/s,
  -21% TPOT at 8k1k c1 (#972), but only at high concurrency in a later image (#1050). asm paged
  attention by shape with `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (#794). The `TRITON_HIP_*` and
  `AMDGCN_USE_BUFFER_OPS` codegen knobs, each under the gate alone (#69). The
  `VLLM_ROCM_USE_AITER` master switch: up to +17.5% tok/s/GPU at c64 on MI300X, unchanged at c1-2
  (#982). Shared-expert fusion
  (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, #935, #1001, #1096) is the flag model-w A never
  set (REPORT §2.8); SGLang's DSV4 MI355X recipe runs `--disable-shared-experts-fusion`
  (`main:…/dsv4_fp4_mi355x_sglang.sh`).
- **NUMERICS.** AITER's BF16 tuned table picks numerically lossy kernels for some shapes (Model M
  pins its pooler to `F.linear`, `03-code.md`); AITER under TP-only corrupted MiniMax-M3 output
  and is correct under EP (#935).
- **AMD.** This direction is AMD's. Read KernelForge's AITER operator pages and AITER's tuner
  first (runner `AGENTS.md` §11).
- **Arbor.** Increment or bundle for a selection; a tuned table is a small rewrite (tuner run,
  correctness, A/B). Prove dispatch by the kernel name in a trace. Accuracy gate when the family
  changes.
- **Missed by.** 2.2, 2.8.

### D20. Lower-precision compute paths (NUMERICS)

- **Mechanism.** Fewer bytes and faster MFMA: FP8 KV (half the KV bytes, twice the capacity, #17,
  #210), online FP8 quantization of BF16 non-MoE linears (`ptpc_fp8`, #941, #952), A4W4 MoE
  (#1214), FP4 MoE activations (#752), FP8 activations (#396).
- **Every one changes numerics.** Packs often fix the quantization; an Arbor KEEP needs the
  accuracy gate. AITER's persistent MLA requires FP8 KV (#287), so D2 can pull D20 in.
- **Arbor.** Bundle with an accuracy gate, only where the pack allows it.
- **Missed by.** 2.4.

---

## F. Host and integration

### D21. Host and CPU overhead

- **Mechanism.** HTTP, tokenization, detokenize/stream, the scheduler loop and Python GC stall the
  GPU at high concurrency or short steps, and add to TTFT at low concurrency. A GPU profile cannot
  see it.
- **Evidence.** vLLM Rust frontend on MiniMax-M2.5 MI355X (v0.22 ROCm image): TTFT -8% (1k1k) and
  -22% (8k1k) at every point, tok/s within 1.5% (#823). GC off, mimalloc purge, allocator settings
  (#830, #841). Stream interval 30 to 100 (#571, #841, #983); 16 tokenizer workers (#571);
  scheduler receive interval per concurrency (#814); extra frontends at thousands of streams
  (#246, #890); async scheduling (#69, #731, #1096); decode skips the prefix lookup (#1236).
- **AMD.** Available: the Rust frontend ships in the vLLM ROCm v0.22 image (#823); async
  scheduling; `OMP_NUM_THREADS=1` (#266, #1096).
- **Arbor.** Bundle of knobs each under the gate alone (#841). Measure host busy and idle per step
  (trace gaps), scheduler loop time, TTFT.
- **Missed by.** 2.3, 2.4.

### D22. Upstream PR stacks and reference-engine parity

- **Mechanism.** The decisive kernel or fix sits in an unmerged PR or a newer nightly. In the
  pinned image the feature fails, and the agent records a dead end that steers later iterations
  away.
- **Evidence.** AITER MLA rejected 8 heads per GPU until vLLM ROCm v0.18 (#425 to #427).
  `--moe-backend aiter` raised a KeyError on FP4 scales until `nightly-09663abd` (#955). The DSV4
  FP4 MI355X ATOM "AITER perf stack" (aiter#2822, #2900, #2642 plus a tuned CSV, #560) was folded
  into a nightly about two weeks later (#652). vLLM #40889/#40871 sparse MLA (#555, #731). Kimi-K3
  MI355X needed nine AITER and vLLM PRs backported (#1171). Native 384-expert MXFP4 at TP8 removed
  the padding to 512: about 30% less redundant MoE compute (#1201). ATOM as the reference: GLM-5.2's
  +23.6% matched ATOM's default dispatch (#0); vLLM recipes chase ATOM parity (#1089, #1096).
- **Arbor.** The **upstream-integration lane**: port, correctness gate, A/B, image digest in the
  record. Re-test workaround disables on every image change (#448). For AMD, diff the served path
  against ATOM's dispatch for the same model before inventing.
- **Missed by.** 2.2.

---

## Quick lookup

"Single node?" means the direction can be built and judged on one 8-GPU node. AMD: **yes** =
available in a public AMD image; **part** = some stacks or nightlies; **no** = missing.

| Direction | Single node? | Plan kind | AMD | Decide on |
| --- | --- | --- | --- | --- |
| D1 MegaMoE | only as EP8/DEP8 | rewrite | no | MoE region µs/layer; tok/s/GPU at high band; one point below the crossover |
| D2 Persistent decode kernels | yes | increment (MLA persist) / rewrite | part | launches/step; decode step µs; ITL at C8, C32 |
| D3 AR+residual+RMSNorm fusion | yes | increment / bundle | yes (SGLang), unclear (vLLM) | collectives per forward; ITL |
| D4 Quantized collectives | yes | bundle, NUMERICS | yes | TPOT at 2 concurrencies; kernel fired; GSM8K |
| D5 Two-batch overlap | yes | bundle, policy | yes | concurrent streams in trace; both sides of crossover |
| D6 Sequence parallelism | yes | rewrite (mixed only) | no fused kernel | per-boundary µs vs the served all-reduce |
| D7 Smaller-TP replicas | yes | layout | yes | tok/s/GPU vs interactivity, 3+ concurrencies |
| D8 DP attention + EP bundle | yes | bundle + layout | yes (DSA long-context hangs) | rank skew; tok/s/GPU; TTFT tail |
| D9 Wide EP + EPLB | EPLB only | layout (multi-node) / increment | part (MoRI, no EPLB) | expert load over time; a2a wait |
| D10 DCP (+PP) | yes (DCP) | layout + upstream | part (ATOM) | KV capacity; max concurrency; ITL by bucket |
| D11 Disaggregated P/D | colocated only (NVIDIA today) | layout (multi-node) | part (multi-node) | frontier per band; TTFT/ITL distributions |
| D12 KV offload tiers | yes | bundle, workload-bound | yes | hit rate vs ceiling; re-prefill tokens |
| D13 Prefix-aware routing | yes (DP ranks) | cross-component bundle | yes (SGLang router) | per-rank hit rate; TTFT |
| D14 Prefill admission/chunking | yes | policy increments | yes | TTFT and ITL at 2+ concurrencies |
| D15 Speculative policy | yes | policy table | yes | real AL; ITL per band; GSM8K |
| D16 Graph coverage | yes | increment | yes | share of steps replayed from graphs |
| D17 Compiler fusion | yes | increment (removal) | yes (pass coverage varies) | kernels per layer |
| D18 Long-context attention kernels | yes | upstream / rewrite | part (nightlies) | attention µs by context bucket |
| D19 Kernel families by table | yes | increment / bundle / small rewrite | yes | kernel name in trace; µs at served shapes |
| D20 Lower precision | yes | bundle, NUMERICS | yes | accuracy gate first |
| D21 Host overhead | yes | bundle | yes | host idle per step; TTFT |
| D22 Upstream PR stacks | yes | upstream | yes | correctness, then A/B; image digest |
