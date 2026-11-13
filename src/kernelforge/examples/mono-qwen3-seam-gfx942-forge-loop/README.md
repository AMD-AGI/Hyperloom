# Qwen3-14B-FP8 attention seam: a monokernel task on 4 x MI300X

A forge-loop task with the `mono` kernel backend. The kernel under optimization is a monokernel:
one persistent launch per decoder layer and TP rank that runs, from Qwen3-14B-FP8's attention
output, o_proj, the TP all-reduce, the residual add and the post-attention RMSNorm. SGLang runs the
same work in four launches. Here the all-reduce crosses GPUs and the norm needs whole rows whose
columns other CTAs computed, so the one launch waits inside itself, on other CTAs and on the other
ranks.

The launch is written in FlyDSL against the device runtime of
[sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) (`sgl_amd_mono.common`, at `b10f3b7`):
its tagged mailboxes, guarded waits with fail-stop, launch epoch and symmetric peer memory.

## Layout

```
mono-qwen3-seam-gfx942-forge-loop/
├── README.md
├── program.md                    # the task statement the agents read
├── run_example.sh                # workspace, self-check, campaign, export
└── task/                         # copied into a fresh git workspace
    ├── config.yaml               # the acceptance suite forge-loop runs before a KEEP
    ├── qwen3_seam/seam.py        # the launch: the only editable code
    └── benchmark/
        ├── forge_mono/
        │   ├── driver.py         # dist_harness cases at TP4 (dist_harness.py is copied beside it)
        │   ├── runner.py         # buffers, compile, residency, one launch per call
        │   ├── weights.py        # a random layer as SGLang holds it on gfx942
        │   ├── reference.py      # SGLang's path on the launch's tensors
        │   ├── judge.py          # how an answer is held to the reference
        │   └── contract.py       # the pinned runtime, the static rules on qwen3_seam/
        └── acceptance/           # compile, parity, fail-stop, replay (4 GPUs)
```

forge-loop may edit any tracked file outside its protected paths, and add files. `benchmark/` and
`config.yaml` are protected, so `qwen3_seam/` is all an agent can change. The acceptance suite
lives under `benchmark/` rather than in a `test/` directory, so that it ships with the example:
the wheel leaves out every `test`, `tests` and `testing` directory, the repository ignores every
`benchmarks/` (run artifacts), and its pytest collects every `src/**/tests`, while these tests
need four GPUs.

## What a candidate must meet

`program.md` states the contract in full. In short:

- **one launch, all of it resident**: each driver case declares `launches=1` and `resident=True`,
  so `dist_harness` captures a call and refuses anything but a single kernel node whose grid fits the
  GPU at once;
- **guarded waits only**: `contract.py` refuses `while`, `s_sleep`, `_spin` and `spin_until` in
  `qwen3_seam/`, and `benchmark/acceptance/test_seam_failstop.py` stalls each wait and removes a
  rank, requiring every rank to stop within the deadline;
- **a fresh tag per launch**: parity over two back-to-back calls, the last timed graph replay, and
  `benchmark/acceptance/test_seam_replay.py` (interleaved widths, back-to-back launches, the
  epoch's wrap);
- **numerics** (`judge.py`): `res_out` bit for bit, `x` within one bf16 step on at most 0.1% of its
  elements, the guard's error words clear.

## The reference

`reference.py` uses only SGLang, aiter and torch, on the ops SGLang reaches on this container with
`SGLANG_USE_AITER=1`:

- **o_proj**: whatever SGLang's `dispatch_w8a8_block_fp8_linear` returns for 128x128 blocks, required
  to be `aiter_w8a8_block_fp8_linear`: `aiter_per1x128_quant`, then aiter's Triton
  `gemm_a8w8_blockscale`. At TP4 (N 5120, K 1280) it runs the default gfx942 config: 128x128x128
  tiles, 16x16 MFMA, kpack 2. `reference.py` refuses to run if SGLang would dispatch elsewhere,
  including when `SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK` is set.
- **all-reduce**: the four partials summed in rank order in f32 and rounded once. SGLang's custom
  all-reduce has no single order on gfx942: its LL kernel, which serves 1 to 5 rows at TP4, starts
  each rank's sum from that rank's own partial, so the ranks can disagree by a bf16 step. The task
  fixes the order instead.
- **add + RMSNorm**: `aiter.rmsnorm2d_fwd_with_add`, eps 1e-6, which takes the f32 sum before
  rounding it into the residual it returns.

## Run

```bash
./run_example.sh --workspace /path/to/ws --max-hours 2
```

The script checks for four gfx942 GPUs and the imports, builds the workspace and commits it, then
runs the pristine launch through the driver (correctness, a short benchmark, `--profile-run`) and
the whole acceptance suite. Any failure stops it before the campaign. It runs forge-loop from the
checkout it sits in, with one lane, since every candidate takes all four GPUs. FlyDSL's compile
cache for the run sits beside the workspace (`<ws>.flydsl-cache`).

The campaign's outcome is `forge_experiments/best_result.json`. When it names a KEEP, the script
exports that commit's `qwen3_seam/`, its diff against the pristine launch, and the manifest to
`<ws>.delivered/<commit>/`.

The four GPUs must be otherwise idle during the run: every CTA of the launch has to be resident at
once, and kernels from another process on the same GPU can keep some of them from starting.
