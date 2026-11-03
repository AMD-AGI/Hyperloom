# A target without a route: what to build, what to validate, how to contribute

Read this when your target (framework and version, model, workload,
platform) is not in [SKILL.md](SKILL.md)'s routing table, or when a
limitation blocks the question you were asked. The validated route,
[targets/vllm-dsv41-rocm.md](targets/vllm-dsv41-rocm.md), is the worked
example: everything below was done for it, in `docs/milestones/m1.md` to
`m3.md`.

Until a target has been through the steps below, a profile of it is
**unvalidated**. Say so beside every number; do not rank optimization sites
from it as if it were not.

## Start from what differs

| what differs from the validated route | do | effort |
| --- | --- | --- |
| workload, concurrency, seed, duration | "Same stack, different load" | a few GPU-days |
| GPU of the same family (MI355X), ROCm version | "Same stack, different load", plus the platform facts | a few GPU-days |
| model, on the same vLLM | "Another model" | days |
| vLLM version | "Another framework version" | days |
| platform (CUDA) | "Another platform" | weeks for the gates |
| framework (SGLang, ATOM, TensorRT-LLM, a training loop other than DLRMv4) | "Another framework" | weeks |

## Same stack, different load

The adapter and runtime are unchanged. What changes is what tracing costs
and what the sample holds.

1. **Characterize the workload first** (the runner's AGENTS.md §8, its
   `tools/workload/`):
   - what the client sends;
   - what each concurrency actually puts in flight;
   - the regimes (C32 at seed 42 has two);
   - the step kinds and their shares of engine time.
2. **Define the regimes** for `profile_report.py` (`--split`). A load with
   more than two regimes needs a regimes option there (development).
3. **Run the operating-point gate for the mode at this load.** Tracing cost
   is per traced dispatch, so it depends on the step rate and on which kinds
   are traced:
   - Use `tests/m3/m3_ab.sh` and `op_point.py`: arms alternating OFF/ON on
     one node, three of each.
   - A workload other than AgentX needs its own serve and replay script in
     place of `serve_dsv41.sh`'s measured replay (development).
4. **Choose `RUPPELL_DETAIL_KINDS` from the phase coverage** of a first
   5/1000 run: every kind with a real share of time needs samples.
5. **Correctness** (the GPU-program check, `m2_dsv41_correct.sh`) holds for
   the stack. Rerun it if the serve flags change.
6. **Write a route file** in `targets/`, with the mode table, run commands,
   measured limits and quirks, and add a row to SKILL.md's table.

## Another model (same vLLM)

The adapter hooks every module generically; whether that covers the model
has to be shown.

1. **Facts job** (`docs/adapting-frameworks.md`, "Process" step 1):
   - Do module forward hooks fire inside graph captures?
   - Which subsystems capture graphs (target, drafter, encoder)?
   - What runs at startup?
2. **The section 5 checks on a short serve** (`tests/m2/`):
   - Registry: every replay joined.
   - Coverage: decode GPU time on a (layer, op) key ≥ 95%. Custom ops
     outside any hooked module land in `runner` or nowhere.
   - Granularity: decode GPU time on a layer ≥ 80%, and on the largest
     single scope ≤ 50%. A compiled region (`torch.compile`) passes coverage
     with all its time on the scope around it, and fails this.
   - Steps and fingerprints across ranks.
3. **Collectives.** Check `ruppell.limits.COLLECTIVE` recognizes the model's
   communication kernels (FlashInfer, custom all-reduce, all-to-all for
   expert parallelism). Add the names, with a test, if not.
4. **Tuning.** Check the model's run-time tuners are pinned by
   `RUPPELL_TUNING` (Inductor and Triton are); a library's own tuner needs
   its own pin (development in `python/ruppell/adapters/tuning.py`).
5. **Correctness:**
   - identical greedy outputs where the model is reproducible;
   - otherwise the GPU-program check;
   - either way, control against control first, to know what the check
     reads with nothing traced.
6. **Then the gates at the load**, as in "Same stack, different load".

## Another framework version

The adapter checks every signature it wraps and refuses to trace when one
moved (`check_signatures`). Update the wrappers for the new version, keep
the old one working if the pin still uses it, and rerun the facts job and
the section 5 checks (`docs/adapting-frameworks.md`).

## Another platform

- **CUDA** (branch `cuda`, B200). Shown: replays joined by graph node, vLLM
  through the unchanged adapter (`docs/milestones/cuda-c3.md`, `cuda-c4.md`).
  Missing for profiling:
  - Detail sampling is refused, so every step is traced. It costs −5.8% to
    −8.3% of throughput at C8 on a 35B MoE, and no operating-point gate has
    been run.
  - Memory operations are not recorded.
  - The witness (Nsight Systems) runs on a separate process.

  Development: sampling on CUDA (`TODOs.md`, "Detail sampling on CUDA").
  Then the gates.
- **MI355X or another ROCm.** Record the platform facts (ROCm, profiler
  SDK versions), rerun the section 5 checks and the gates. Results from
  MI350 (Cluster A) do not carry over.

## Another framework

Write an adapter: `docs/adapting-frameworks.md` gives the process, what an
adapter provides, and what bit us in vLLM. An adapter provides:
- **Run fields:** model, TP, rank, role.
- **Scopes:** for every kernel.
- **Graph keys:** for every capture.
- **Engine steps:** with the workload's fields; the kind is what kind-aware
  sampling uses.
- **Named startup steps.**
- **Reproducible run-time choices:** the tuning pin.
- **Detail sampling hooks.**

`docs/adapting-frameworks.md` also has starting points for SGLang. Then
every step of "Another model" and "Same stack, different load".

## What would make profiling better (any target)

Open work, where each piece goes, and how it would be validated. Pick what
blocks your question; `TODOs.md` has the full list.

| development | lifts | where | validated by |
| --- | --- | --- | --- |
| **kind-aware sampling at a rate** (for example mixed at 1/10, not every step) | the host cost of tracing every long step; the inflation of their host time | `adapters/vllm` (`_detail_kinds`), `tests/test_sampling_offline.py` | the operating-point gate; phase coverage still ≥ 20 samples per kind |
| **parity in the sampled modes** | whether kernel times in sampled steps match untraced ones (measured at full detail only) | `tests/m2/m2_dsv41_correct.sh` with sampling that covers first-token steps | kernel parity, control against control alongside |
| **host-time inflation per step** | a correction for always-traced kinds' step times | the tool's own intervals (`self.bin`, recorded, not exported: `TODOs.md`) in the census | a sampled run's step times against a lower-rate run's |
| **aten op for Triton kernels launched from Python** | 19–24% of DSV4.1 kernel time has no aten op | a hook on Triton's launcher emitting an op record (`python/ruppell/`, `runtime/src/cpu_ops.cpp`) | `(no op)` share falls; offline test with a fake launcher |
| **finer runner scopes** | coverage (C8 at 95.01%/94.81%) | `adapters/vllm` `_runner_scope` | the coverage check |
| **rank skew without the tracer's distortion** | per-rank collective waiting is untrusted | an untraced measurement design (open question: anything that times kernels is a tracer) | control against control of the skew itself |
| **kernel categories for TraceLens** | 78–79% of DSV4.1 time is `other`; no roofline for MI350/MI355X | a category map for AITER, Triton and custom HIP kernels, and arch files, contributed upstream to TraceLens (or a ruppell-side map in the exporter) | TraceLens's category report on a known run |
| **detail sampling on CUDA** | every step traced on CUDA | `runtime/src/tool_cuda.cpp`, see `TODOs.md` | end times of sampled kernels against Nsight Systems; the gates |
| **regimes from the load** | hand-set `--split` | `tests/m3/profile_report.py` | offline test; agreement with the workload characterization |
| **large traces** | JSON size | Perfetto protobuf output (`TODOs.md`) | viewers open it |

## How to contribute

This repository's rules are in `AGENTS.md`; the ones that matter most:
- **Evidence travels with the change.** A change that affects what ruppell
  records is checked against an independent collector (the witness) on real
  work. Coverage is stated. Cost is measured by a paired A/B, not assumed.
- **Refuse rather than guess.** A join, a sample or a check whose
  preconditions do not hold refuses and says why.
- **One change per commit**, layers separate (the runtime knows nothing of a
  framework; an adapter knows nothing of a workload).

The steps:
1. **Branch** from `main`; open a pull request (the maintainers merge).
2. **Offline tests** pass, and new code has one:
   `PYTHONPATH=python python3 tests/test_offline.py`, and alongside it
   `test_adapter_offline.py`, `test_sampling_offline.py`,
   `test_tuning_offline.py`, `test_kernel_parity.py`,
   `test_program_check.py`, `test_offline_cuda.py`.
3. **GPU validation.** It runs as a job script that exits when done (the
   pattern of `tests/m2/` and `tests/m3/`: provision, run, check, copy
   results home, remove the container). Record it with job ids in
   `docs/milestones/`; a profile goes in `docs/analysis/`.
4. **Update the skill.**
   - A new target gets a route file in `targets/` and a row in SKILL.md.
   - A new limitation goes in `python/ruppell/limits.py`, in reference.md
     and in `docs/limitations.md`.
   - Update `TODOs.md`.
5. **Gates are the operator's.** Propose a gate or a change to one, with its
   evidence; do not set it.
6. **No secrets.** Before committing, scan for keys (`ANTHROPIC_`,
   `Ocp-Apim`, 32-hex strings).
