# M3 for vLLM: the plan's gates on DeepSeek V4.1 Flash

M3 settles plan-v0's items 4.5–4.7 against its §5 gates for the vLLM
adapter. The gates are the operator's (plan-v0 §0); this file says how each
is measured here, and marks the choices the plan leaves open, and one
deviation, for the operator to confirm.

## Where and how

- **Cluster.** Cluster A (MI350), not Cluster B's MI355X as plan-v0 §8.1 drafts
  (operator, 2026-09-29: use Cluster A, jobs that run and exit, 4 hours each).
  Every result names its node; points of one A/B share a node.
- **Stack.** Pinned vLLM `8f400a0ac` (v2+v6) in the image by digest, the
  knob-ab screen's base arm (its `vllm serve` flags and its environment minus
  the knob flags; see `docs/milestones/m2.md` ("Launch configuration") on its reconciliation with
  InferenceX's launch flags). AgentX seed 42, `NUM_PROMPTS` 320,
  `DURATION` 600.
- **Arms.** OFF: the untraced server (the plugin installed, inactive without
  `RUPPELL_VLLM`). ON: `ruppell run` with `RUPPELL_VLLM=1` (always-on: every
  dispatch recorded, the census over every step). Arm markers per point: the
  server log's "vllm adapter active" lines, 4 ON and 0 OFF.
- **Points.** A fresh server, a discarded capped warmup replay, one measured
  replay (AGENTS.md §9.2); per point, duration, successes and cache hit rate.

| Job | What | Gates |
| --- | --- | --- |
| `tests/m3/m3_ab.sh`, C8 | OFF ON OFF ON OFF ON | Overhead (C8); the ON points are the C8 full-phase replays for Coverage, Registry |
| `tests/m3/m3_ab.sh`, C32 | OFF ON OFF ON | Overhead (C32, reported with its regime; gated the same way if all four complete); Coverage, Registry at C32 |
| `tests/m3/m3_reanalyze.sh` | per campaign, on the node holding the records: census on aiperf's phases, window export, TraceLens files, checks, host blocks | Coverage, Registry, Output, host blocks |
| `tests/m3/m3_witness.sh`, C8 and C32 | a traced server inside `rocprofv3`, one short replay | Agreement |
| `tests/m2/m2_dsv41_correct.sh` | untraced and traced, first-token steps' GPU program | Correctness (see the deviation) |

## The gates, as measured

- **Registry.** Per graph, replays joined to the capture's sequence over all
  replays, at least 0.99 for every graph (`check_m2.py`).
- **Coverage.** Over the decode steps (`kind` decode) in aiperf's profiling
  phase, GPU kernel time attributed to a model or drafter module scope (the
  "(layer, op) key"; the runner's own scopes are attributed but not to a
  layer, so they do not count), at least 0.95 on every rank
  (`census_summary.json` → `decode_module_share`). *Open choice:* the plan
  says "decode steps"; mixed steps are excluded here.
- **Agreement.** Per kernel (the witness comparison groups by kernel name
  within kernel id), GPU time within 3% of `rocprofv3` in the same process.
  *Open choice:* the plan offers "one window of the same run" or a separate
  replay at the same point; the witness runs a separate short traced replay
  at each concurrency, because `rocprofv3` over a whole 600 s replay would be
  2.2 GB of CSV per rank per minute.
- **Overhead: replaced by the operator (2026-09-30).** The question a gate
  must answer is whether tracing would point a profile at the wrong
  optimization site, not whether it moves ITL at all. The tracer's cost is
  host time per dispatch; it does not change the GPU program (M2.8), and
  kernel times come from GPU timestamps. Two gates:
  1. **Kernel-time parity** (`tests/m3/kernel_parity.py`): over steps doing
     the same work (the first-token phase's prefills of fixed prompts), per
     kernel family, a traced server against a control whose adapter only
     launches the step markers, both under `rocprofv3` (per-kernel times need
     per-dispatch tracing, whose host cost any observer pays, rocprofv3
     included; so this isolates what ruppell adds): counts equal, GPU time
     within 3% for every family with at least 1% of the time, the phase total
     within 3%, and the top 5 families ranked the same.
     **Compute and collectives apart (operator, 2026-09-30).** A collective
     kernel's time is transfer plus waiting for the slowest rank, and tracing
     moves which rank is slowest (jobs 49062, 49794). The rule above applies
     to compute families (ranking included), per family only for families
     with at least 3% of the compute time (operator, 2026-09-30: two untraced
     servers differ by up to 15% on families under 2%, job 50343); totals and
     the top-5 ranking as above. Collectives are gated on their
     transfer estimate: steps and collective launches aligned across ranks by
     order, each launch's shortest duration across ranks, summed per family;
     within 3% per family with launch counts equal, and in total. Waiting per
     rank is reported, not gated, and every output labels it as untrusted
     (`ruppell.limits`, `collective-wait`: in each trace's and census's
     `limitations`, on each collective kernel as `ruppell_limits`, and in
     `LIMITATIONS.md` beside the TraceLens files).
  2. **Operating point** (`tests/m3/op_point.py`): the ON replays' output
     throughput, requests completed, cache hit rate and aiperf's
     time-weighted in-flight count lie within the OFF replays' range widened
     by 1%. A trace describes the load it ran under; if tracing changes how
     requests batch, it profiles another regime.
  The paired per-request ITL contrast (`tests/m3/overhead.py`) is still
  reported, not gated: it measures the tracer's cost, not whether its
  profile misleads. Host blocks over 100 ms caused by ruppell stay a gate
  (`tests/m3/host_blocks.py`).
- **Output.** TraceLens and the Perfetto trace processor open the exported
  windows.
- **Correctness** (operator, 2026-09-30: the GPU-program check stands).
  plan-v0 asks for greedy tokens identical flag off and on. On DSV4.1 that is
  not determinate (an untraced server differs from itself by up to 20 nats
  in prompt logprobs); it is decided by the GPU program instead (every
  first-token step's kernels, order and grids identical traced and untraced
  on every rank, job 49062). On the small model, greedy tokens identical.
  Choices a framework makes at run time by timing (Inductor's block size for
  a pointwise kernel, job 49826) must come out the same traced and untraced:
  making them so is the framework adapter's job (operator, 2026-09-30); the
  check stays strict. It compares the steps that ran kernels; empty steps
  (markers, nothing between) are counted and reported, since one appears in
  one of two untraced servers and not the other (operator, 2026-09-30, job
  50343).
- **Profiling is not benchmarking** (operator, 2026-09-30). A traced server
  that runs slower is acceptable: tracing's purpose is to find where to
  optimize. The operating-point measurement is reported with every profile,
  as how close the traced load is, not as a bar a profiling mode must
  clear. The mode that traces every mixed and prefill step (round e: C8 in
  flight up to 0.7% over the band, C32 throughput −2.5%) is the profiling
  mode for AgentX on DSV4.1.
- **Mode.** The gates are evaluated for the mode a profile would use:
  kernel detail sampled (`RUPPELL_DETAIL=5/1000`), everything else always on.
  Full tracing is reported alongside.

## Known before starting

The C8 window analysed in M2 (docs/analysis/dsv41-c8-tracelens.md) shows the
tracer lengthening the first all-reduce's wait by about 1.1 ms (greedy steps,
job 49194): ruppell's host cost reaches the step's critical path through the
lagging rank. The OFF arm's ITL spread at C8 is about 0.5% in the middle ISL
buckets (four knob-ab baselines on one Cluster A node). The Overhead gate may
well fail at C8; if it does, the next step is to cut the host cost (the
buffer callback, the flush, the Python hooks), not to change the gate.
