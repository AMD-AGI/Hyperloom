# Record: workers killed at shutdown, and telling it from the records

A traced vLLM server stopped right after its load lost the last engine steps
on some ranks, and nothing in the records said so. Found while profiling Model M
Spark on B200 (runner `plans/model-m/03_B200_CUDA_BRINGUP.md` §5), fixed in
`7ca9e83` (the adapter's mark) and `2032cfa` (census and checks).

| | |
| --- | --- |
| Node | `<host>`, 8× B200, driver 580.126.09, 2026-10-07 |
| Image | `vllm/vllm-openai:nightly-29468dde…` (vLLM 0.30.1rc1.dev143+g29468dde8, torch 2.13.0+cu130) with the runner's Model M tree; CUPTI 130001 |
| Build | tool `fbcbfde2…` (runtime unchanged since `706c6eb`); Python at `7ddf9d2` (= `2032cfa` but for the `incomplete-records` text) |
| Model and load | Model M, a 2-layer cut of the checkpoint, TP8+EP8, `FULL_AND_PIECEWISE`; 16 requests of 8,192 in / 128 out at C8; SIGINT with no idle after the load; `ruppell run` as the container's PID 1 |

## What happened

vLLM's API server, asked to stop with `--shutdown-timeout 0` (the default),
sends SIGTERM to the engine core and at once calls `kill_process_tree`,
which SIGKILLs the engine core and its workers. The workers are by then in
their own shutdown: the death pipe has closed and the adapter's flush at
`model_runner.shutdown` is running or about to. A worker killed there loses
what it recorded since the tool's last periodic flush (`RUPPELL_FLUSH_S`,
5 s): about 140 engine steps here. Its `meta.json` stays at the last periodic
flush, so `dropped_records` read 0; the census showed ranks with different
engine-step counts (`merge_ok` false) and no reason.

## What changed

- `ruppell.flush(complete=True)`: wait for the device, flush, and only then
  record the context field `records_complete` (`defs.tsv` is written through
  at once). The vLLM adapter's shutdown hook uses it. A worker killed before
  or during the flush stays unmarked.
- `Proc.records_complete`: the final flush, or that mark. The engine report
  and the census summary carry it; the census lists processes with engine
  steps that are not complete (`incomplete_processes`) and adds the
  limitation `incomplete-records`; `check_m2.py` has a criterion "records
  complete".
- The adapter warns at startup when vLLM's `shutdown_timeout` is 0.

## Runs

| | A: `--shutdown-timeout 0` | B: `--shutdown-timeout 60` |
| --- | --- | --- |
| adapter warning | printed | none |
| workers marked complete | 3 of 8 (ranks 0, 6, 7) | 8 of 8; all also finalized (`final: true`) |
| unmarked workers | ranks 1–5: `meta.json` at the last periodic flush (1,514,545 kernels) against 1,530,459–1,534,086 in the files, partial records at four files' ends | none: files equal `meta.json` (1,537,534) on every rank |
| census | `incomplete_processes` ranks 1–5, limitation `incomplete-records` | none |
| engine steps per rank | 198–282 (`merge_ok` false); the 3 marked ranks have all 282 | 290 on all 8 (`merge_ok` true) |
| stop | 3 s | 18 s |

In A the mark separates exactly the ranks that kept every step from those
that did not. An earlier run with the mark written before the flush (not
committed) marked ranks killed during it: the mark has to follow the flush.

## Not checked

- Whether the pinned vLLM of the ROCm route (`8f400a0ac`) kills the same way
  at `--shutdown-timeout 0`; its stop scripts wait two flush intervals
  before SIGINT, which bounds the loss but cannot show its absence (TODOs).
- Runs traced before `7ca9e83` carry no mark and read as incomplete.
