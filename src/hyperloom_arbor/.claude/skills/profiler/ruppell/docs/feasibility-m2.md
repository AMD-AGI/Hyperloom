# Feasibility for M2 (plan-m2 §4, M2.0)

Facts from runs on AMD Instinct MI350 (gfx950) nodes of the Cluster A cluster,
2026-09-29, inside the pinned vLLM image (`sha256:960228cf…`) with the pinned
vLLM tree (`8f400a0ac`) deployed. Each answer names the job that showed it;
logs are under `~/ruppell-work/runs/<job>/` on Cluster A.

## The container

| Question | Answer | Evidence |
| --- | --- | --- |
| ROCm, profiler SDK, Python, PyTorch, vLLM | ROCm 7.2.3; SDK 1.3.2; Python 3.12.13; torch 2.12.0+git6bbd260 (HIP 7.2.53211); vllm 0.1.1.dev50+geed1f3d0c with `8f400a0ac` deployed over it | 48312 |
| `ruppell` builds and M1's tests pass in the container | Yes: the probe with `rocprofv3` in the same process, identical dispatches; 100% attributed | 48312 |
| Per-dispatch cost on the probe | about 5.1 µs per dispatch and 192 ms per process (M1 on the host's ROCm 7.0.2: 2.5 µs and 259 ms) | 48312 `overhead.txt` |
| ruppell next to Kineto (after `import vllm`, which promotes `libtorch_cpu.so`) | Both load as SDK clients; the PyTorch demo stays 100% attributed | 48312 |

## vLLM serving, small model (Qwen2.5-0.5B-Instruct, breakable graphs)

| Question | Answer | Evidence |
| --- | --- | --- |
| Which processes initialize the SDK | Every process that loads HIP: API server, engine core, workers, and under `rocprofv3` about ten short-lived helpers (multiprocessing's resource tracker and spawned children). Only the processes that run the model record dispatches; the others write an empty directory. At TP1 the engine core runs the model | 48352, 48395 |
| Served configuration | `cudagraph_mode FULL_AND_PIECEWISE`, breakable graphs on, compilation mode NONE (so module hooks run); `capture_model` runs twice (a memory-profiling pass, then the real one): 2,603 captures, 154 of them replayed under the fixed load | 48395 |
| Module hooks fire during capture | Yes: 273 modules hooked; 138,705 replayed dispatches per rank attributed through the capture registry | 48395 |
| `execute_model` calls agree across ranks | Yes: 460 engine steps on each of 2 ranks, every fingerprint equal | 48395 |
| Timebase across processes | Shared: clock-pair offsets of all processes on a node agree within 0 ns (small model) and 256 ns (DSV4.1, 26 processes) | 48395, 48424 |
| Side streams | 37,707 kernels per rank on a queue other than the compute (marker) queue; all owned by the step whose CPU window launched them; 489–728 run past their step's end | 48395 |

## vLLM serving, DeepSeek V4.1 Flash at TP4 (the knob-ab screen's base arm)

| Question | Answer | Evidence |
| --- | --- | --- |
| Processes | API server, engine core, `Worker_TP0`–`TP3` (one per GPU), and short-lived helpers; only the four workers record dispatches | 48424 |
| Served configuration | `FULL_AND_PIECEWISE`, breakable graphs on; `capture_model` twice: **6,987 captures**, **417 replayed** in a short C8 replay | 48424 |
| Hooks | 961 target modules and 79 drafter modules hooked; the drafter's modules shared with the target keep their target scope | 48424 |
| The drafter | `DSparkSpeculator`, with its own `capture()`: 4 of the 417 replayed graphs are drafter captures (keyed `drafter|passN|gM`), replayed 20,053 times, about once per engine step | 48424 |
| Where engine-step GPU time goes (rank 0, census by op) | MoE 21.9%, `VocabParallelEmbedding` 12.7%, MLA attention 12.7%, sparse indexer 11.9%, row-parallel linear 11.4%, mHC pre 7.5%, merged column-parallel 6.7%, the runner's own work 2.7% | 48424 |
| Why the embedding is 12.7% | 99% of its scope's time is `cross_device_reduce_1stage`, the all-reduce after the vocab-parallel embedding: the step's first collective, which spin-waits until every rank has launched the step, about 1.2 ms per call. It measures the ranks' launch skew, not the embedding | 48424 (windowed trace, 21 steps) |
| Volume | about **37 million dispatches per rank** over 18 minutes (startup, warmup and a 312 s C8 replay), 6.4 GB per rank on disk; 67 million API records; 96,000 records per second during the replay; **0 dropped**; the SDK's buffer callback took 3.6 s in total, 6.5 ms at most | 48424 |
| Compute and side streams | one compute queue per rank; 6.7 million kernels per rank on other queues, all owned by a step's CPU window, none running outside it | 48424 |
| Step boundaries overlap | 72 of 20,116 steps overlap the next by up to 4.4 µs (async scheduling: the next step's begin marker runs before the previous step's last kernel ends) | 48424 |
| Server start | untraced 470 s on a fresh node; traced 291 s (warm caches) | 48424 |

## Surprises worth keeping

- **PyTorch destroys each `hipGraph_t` right after instantiating it**, so
  every capture can get the same graph address, and exec handles freed with
  the memory-profiling capture pass come back in the real one. The first
  small-model run attributed every replay to the last capture (one "graph",
  refused). Handles are now resolved in record order (`engine._join_graphs`).
- **vLLM's warmup drives `execute_model` with made-up batches** (an empty
  one, then one of `max_num_seqs` requests) and launches a few kernels
  directly between them (`vllm/v1/worker/gpu/warmup.py`). Counted as engine
  steps they looked like served steps with 128 requests, and two kernels per
  rank fell between steps. They are now named `warmup` steps.
- **A served process is usually killed, not finalized.** Under `rocprofv3`
  the server's SIGINT is held by `rocprofv3`'s signal handler and the server
  has to be killed; without it, SIGINT stops the server but workers still end
  before the profiler's finalizer runs. The tool flushes every 5 s
  (`RUPPELL_FLUSH_S`), the reader drops a trailing partial record, and the
  serve scripts wait two flush intervals before stopping.
- **`sleep infinity` as a container's PID 1 reaps nothing**: killed server
  processes stayed as zombies that `pgrep` counted as alive, which made a
  clean stop look like a hang. Containers now run with `--init`.
- **The SDK can give a graph's dispatches on its second queue another
  call's correlation id.** The decode graph replays on two queues; once in
  17,503 replays, its 4 MoE kernels on queue 0x3 carried the id of a
  `hipEventRecordWithFlags` call instead of the `hipGraphLaunch` (48942,
  `tests/m2/odd_replay.py`). The engine completes such a replay from the
  dispatch ids it lacks, only when the kernel sequence then matches exactly.
- **DSV4.1's prefill is not reproducible either**: on one untraced server,
  the prompt's own logprobs differ between two passes by up to 20 nats, from
  the second prompt position and from a cold prefix cache, and the first
  generated token differs on 3 of 8 prompts (48946). The small model is
  exactly reproducible. Correctness of tracing is therefore checked on the
  GPU program (every step's kernels, order and grids), not on outputs.
- **The measured DSV4.1 configuration is not deterministic at temperature 0**:
  with probabilistic drafts and synthetic acceptance, two greedy passes on one
  untraced server differ on all 8 prompts (48424). A traced-against-untraced
  comparison is only decidable under standard rejection sampling.
- **The runner's own work is a few percent of GPU time** (input copies,
  fills, index kernels outside any module forward): 0.89–0.93 attributed
  before runner scopes, 0.98 after (small model), 0.99995 on DSV4.1.
