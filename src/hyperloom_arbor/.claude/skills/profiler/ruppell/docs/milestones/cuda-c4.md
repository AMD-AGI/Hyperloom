# C4 record: vLLM served on B200 under ruppell

Plan: [`../plan-cuda.md`](../plan-cuda.md) §6 C4. C3 (probe, PyTorch demo,
cost): [`cuda-c3.md`](cuda-c3.md).

The vLLM adapter ran unchanged: the same plugin, hooks and signature checks
that M2 used on ROCm. Each configuration was served three times on the same
GPUs by `tests/cuda/run_c4.sh`: untraced (the greedy reference, two passes),
traced, and a control with the adapter in witness mode (step markers only)
inside Nsight Systems, the independent collector. Load: M2's fixed client
(`tests/m2/client.py`): two passes of first-token requests over 8 prompts, 8
greedy completions, then 32 requests 8 at a time.

| | |
| --- | --- |
| Node | `<host>`, 8× B200, driver 580.126.09, 2026-09-30 02:12–03:01 UTC |
| Image | `vllm/vllm-openai@sha256:ed3c505d…`: vLLM 0.30.1rc1.dev48+g7f1a5398e, torch 2.13.0+cu130, `nvidia-cuda-cupti` 13.0.85 (API 130001); `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `cudagraph_mode` FULL_AND_PIECEWISE |
| Witness | Nsight Systems 2025.3.2, `-t cuda --cuda-graph-trace=node`, on the control server |
| Small model | Qwen/Qwen2.5-0.5B-Instruct. TP1: build `d77aee5` (tool `0ad4a9e5…`). TP2: build `57c7369` (tool `92ea6b70…`), after TP2 on `d77aee5` lost rank 0's `meta.json` to a kill at shutdown (fixed in `d8ff498`) |
| MoE model | Qwen-AgentWorld-35B-A3B (`Qwen3_5MoeForConditionalGeneration`, bf16, hybrid linear and full attention), text only (`--limit-mm-per-prompt '{"image":0,"video":0}'`); TP2 and TP4, build `57c7369` (tool `92ea6b70…`); the runs' `run.json` names `bcf3a60`, the tree synced during the run, whose tool is the same digest |

## Exit criteria (per rank; every rank of a configuration alike)

| Criterion | Small, TP1 | Small, TP2 | MoE, TP2 | MoE, TP4 |
| --- | --- | --- | --- | --- |
| Kernels, of them graph-replayed; dropped | 246,948, 133,773; 0 | 270,099, 144,864; 0 | 1,630,601, 942,040; 0 | 1,631,609, 944,882; 0 |
| Graphs: captures; replayed graphs; replays joined (by node id) | 4,235; 154; 1,607 of 1,607 | 4,235; 129; 1,680 of 1,680 | 6,891; 250; 2,415 of 2,415 | 6,891; 250; 2,296 of 2,296 |
| Attributed share of GPU time | 0.9783 | 0.9899, 0.9926 | 0.9987, 0.9983 | 0.9974–0.9991 |
| Engine steps; ids and fingerprints across ranks | 505 | 506; equal | 505; equal | 506; equal |
| Kernels after the first engine step outside every step; in two | 0; 0 | 0; 0 (2 straddle a step end on rank 0, reported) | 0; 0 | 0; 0 |
| Side-stream kernels owned by the step that launched them | 94,330 of 94,330 | 104,938 of 104,939 | 1,285,806 of 1,285,807 | 1,277,132 of 1,277,133 |
| Chrome trace in Perfetto's trace processor | whole run: 3.21 M slices, 245,920 flows, no error stats | whole run: 6.98 M slices, 538,138 flows, none | 21 engine steps: 204,045 slices, none | 21 engine steps: 451,077 slices, none |
| **Greedy outputs**, traced vs untraced (untraced passes agree) | identical, 8 of 8 | identical | identical | identical |
| **First-token logprobs**, traced vs untraced | identical (0.0 nats at 98 positions) | identical | identical | identical |
| **Program check** (the witness): the steps of the first-token phase, traced (ruppell's records) vs control (nsys), per rank | 16 prefill programs, same multiset; 31 empty steps each side | 16, same, both ranks; empty 30/31 and 31/31 | 16 programs of 2,377–2,378 kernels, same, both ranks | 16, same, all 4 ranks |

The unattributed 0.1–2.2% is startup work before the first engine step,
mostly one fill kernel (`FillFunctor<signed char>`, 65–68 launches, 20–23 ms)
outside vLLM's named startup steps; after the first engine step, 0 kernels
are unattributed (checked on rank 0 of every configuration). The one unowned
side-stream kernel per rank also precedes the first engine step.

## What the checks mean on CUDA

- **The witness is the program check.** nsys cannot trace the traced
  process (one CUPTI subscriber, [`../feasibility-cuda.md`](../feasibility-cuda.md)),
  and a served workload does not repeat step for step between runs, so the
  run-level count comparison of C3 does not apply. What does repeat is the
  prefill of fixed prompts: nsys, on the control server, and ruppell, on the
  traced one, see the same kernels in the same order with the same grids
  and blocks in every such step. This is at once the check that tracing does
  not change the GPU program (M2.8's question) and the independent check of
  ruppell's kernel records.
- **Empty steps are counted, not compared.** Steps that run no kernel on the
  compute stream sit at the phase's edges and fall either side of the
  client's realtime mark by a millisecond (small TP2, rank 0: one empty step
  0.3 ms past the end); `program_check_cuda.py` compares steps that ran
  kernels and reports the empty ones on each side.
- **Outputs are reproducible here**, unlike DSV4.1 on ROCm: both models give
  identical greedy text and identical first-token logprobs untraced and
  traced.

## Volume

The MoE at TP2 wrote 1.63 M kernel and 20.4 M API records per rank (the
runtime and driver calls under each launch), 1.5 GB per rank, at up to
367 k records/s; the tool spent 1.35 s in its buffer callbacks over the run,
at most 9.8 ms in one; 0 dropped. Export of 21 engine steps took 15–27 s,
the census of the whole run 10–22 s.

## Cost on a served model

`tests/cuda/run_cost.sh`: the MoE at TP2, each arm a fresh server on the
same GPUs, driven by `vllm bench serve` with a seeded random dataset (64
prompts, 1,024 tokens in, 128 out, `--ignore-eos`, 8 concurrent); arms
alternate within each round. Build `f1a9380` (tool digest as C4's).

| Arm | Output throughput | Mean TPOT | Mean TTFT |
| --- | --- | --- | --- |
| untraced | 1,021–1,036 tok/s | 7.03–7.14 ms | 92–98 ms |
| **witness**: the adapter's hooks and step markers, no tool | +1.2% (median of 3 rounds; −4.0 to +1.2%) | −0.7% (−1.0 to +3.7%) | −3.0% (−5.7 to +8.2%) |
| **traced** | **−5.8%** (−4.0 to −6.8%) | **+4.3%** (+2.5 to +5.2%) | **+24.9%** (+20.3 to +26.6%) |

An earlier run of the same comparison without the witness arm (3 rounds,
build `f4952da`) measured traced at −8.3% throughput, +6.5% TPOT, +33.6%
TTFT: the cost varies by a few points from run to run.

- **The adapter costs nothing measurable**: its Python hooks run on every
  module in eager parts (prefill), yet the witness arm is within the
  untraced server's own spread.
- **The cost is CUPTI's tracing**, as on the probe (C3: the same with kernel
  records alone, and the same for nsys). Decode replays graphs and pays
  ~4% per token; prefill launches most of its kernels eagerly, each with its
  runtime and driver API records, and pays ~25% in time to first token.
- Not yet tried against it: recording fewer API kinds on served runs (the
  probe showed no difference, but its launches are not driver calls), or a
  CUPTI with working hardware timestamps.

## Found and fixed on the way

- **A worker killed while rewriting `meta.json` left it empty** (small TP2 on
  `d77aee5`, rank 0): vLLM kills its workers at shutdown. The tool now
  writes it whole and renames it (`d8ff498`); the engine reads an unreadable
  `meta.json` beside `ext.bin` as an unfinished CUDA run with drops unknown,
  and refuses without it (`34ca631`). The same fix is due in the ROCm tool.
- **The program check compared empty steps at the phase edge** (`bcf3a60`),
  above.
