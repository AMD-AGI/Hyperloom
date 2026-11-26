<!-- when: choosing or judging an AgentX serving setting (running-request cap, scheduling, chunked prefill, KV pool or offload, memory fraction, CUDA graph sizes, tokenizer or streaming) and you want the first-order model it trades on -->
<!-- phase: FRAMEWORK_AGENT -->
# AgentX serving models

First-order models for what to measure, not what to set. Check a model's signal in this session before acting on it.
Under end-to-end grading, queueing and TTFT count. The benchmark point sets the parallel layout (TP, DP, EP), which
normally stays fixed; these models cover what a session can change on top of it. Engine notes hold only for the engine
they name.

## 1. Decode step

Per rank, a step reads its dense weight share, the touched experts and the running KV. Dense, or MoE with saturated
experts: a smaller running set barely speeds each request; MoE before saturation: it does. With tensor-parallel
attention, KV is replicated when KV heads are fewer than ranks (MLA keeps a full copy per rank); KV precision scales
the read. Long contexts can plateau throughput on KV reads.

Signal: step time and kernel shares vs batch.

## 2. Queueing and caps

For a given mean of outstanding requests, mean latency falls only if completions rise; fan-out raises that count with
latency, above client concurrency. Under a median-based grade neither law pins it, and TTFT already includes
queueing. On the throughput plateau a cap moves time into queueing; below the point where completions stop rising, it
also cuts them; it gains when it avoids a cost jump (retraction, recompute, eviction, uncaptured batches). Who waits
moves the median; shortest-first can starve long turns. Short-output turns hinge on TTFT.

- SGLang: the cap applies per data-parallel replica and is split across attention data-parallel ranks.
- vLLM: the cap applies per data-parallel rank.
- ATOM: the cap applies per data-parallel rank.

Signal: running vs cap, waiting, completions, TTFT share.

## 3. Prefill interference

A chunk costs fixed overhead (on MoE, also the extra experts it touches), linear work that is less efficient when
small, and attention to its prefix. Alternating: a decode overlapping the whole prompt loses about the whole prefill,
slightly less with larger chunks, which also speed TTFT and enlarge ITL spikes. Shared step: chunks that fit the
decode step's spare compute add little; with small fixed costs interference is smallest near that size, and large
fixed costs (including MoE expert reads) favour larger chunks. Batching admissions cuts overhead, delays arrivals.

- SGLang: prefill and decode alternate unless mixed chunking is on.
- vLLM: prefill shares the step and token budget with decode and draft tokens.

Signal: ITL spikes at prefills, uncached tokens per turn.

## 4. KV working set and offload

The working set includes conversations in tool gaps; evicted prefixes are recomputed on the TTFT path, slowing
decodes, or reloaded by offload. Offload pays after evictions when the unhidden reload is below the recompute and
reloads and writes fit each host link per direction and host memory (times replication). A reload hides only behind
work that runs while it loads; short follow-ups often expose it. Write-back writes on the eviction path; write-through
pays a constant cost. Onset follows working set vs pool, not concurrency alone. Reuse also fails from rewritten
history, page alignment, dropped draft tails, window retention.

Signal: computed vs new tokens per follow-up, evictions, reload latency, link load.

## 5. Memory budget

Local weights, pool, activations, graphs and workspaces share each rank's HBM. More pool helps until it holds the
reused working set, possibly in a jump; past that it only takes headroom from chunks and graph capture where the
engine sizes the pool before capture.

- SGLang: the static fraction covers weights plus pool and sets the default running limit.
- vLLM: with the extensible KV cache (the default on ROCm from v0.34), the pool takes the memory measured free after
  graph capture and an unset utilization means all of it, so a lower utilization only shrinks the pool. Without it
  (v0.33 and earlier), utilization is total and the pool gets the profiled remainder.

Signal: pool size, free memory after capture.

## 6. Graph coverage

Eager steps pay launch overhead; padding reuses the weight read but can add attention or expert work. Cover the
occurring batch range where measured misses cost more than padding and memory, densely when small; fan-out can exceed
concurrency.

- SGLang: decode graph sizes count requests; prefill graph sizes (opt-in on ROCm) count tokens.
- vLLM: prefill graph sizes count tokens; from v0.34 decode graph sizes count requests, and a token-count size list is
  converted to them.
- ATOM: sizes count sequences.

Signal: graph misses, batch range vs captures.

## 7. Host path

CPU work (templates, tokenization, scheduling, streaming) partly precedes prefill, partly hides under overlap
scheduling, which some features disable. Full re-tokenization adds context-length time to TTFT; one tokenizer can
serialize long turns; oversubscription and cross-NUMA inflate it; coarser streaming cuts it and coarsens measured ITL.

Signal: TTFT minus prefill GPU time, CPU saturation.

## 8. Configuration validity

Flags change across images; env vars work only if this image reads them; repeated-flag behaviour depends on the
parser; trust the logged effective config.

## 9. Also check against their signals

Scheduling and prefix-aware order, eviction and preemption mode, page size, attention backend, KV precision and
quantized all-reduce (accuracy risk), expert balancing, tokenizer processes.
