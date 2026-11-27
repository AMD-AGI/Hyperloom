---
myst:
    html_meta:
        "description": "Compute partitioning variables for Hyperloom on AMD MI300-series: partition mode assertion, runtime hand-off variables, and the partition_mode_sweep.py script for choosing the mode."
        "keywords": "Hyperloom, environment variables, compute partitioning, HYPERLOOM_PARTITION_GPU, HYPERLOOM_PARTITION_MODE, SPX, DPX, QPX, CPX, MI300, partition_mode_sweep"
---
# Hyperloom compute partitioning

An MI300-series card can be split into independent partitions (`SPX`, `DPX`,
`QPX`, `CPX`). Splitting it trades per-request latency for aggregate throughput,
so a partitioned measurement is not comparable with a whole-card one.

The optimizer does not set the mode. Changing it is privileged, disruptive to
every process holding a GPU context, and not something an optimization loop
should be doing between benchmark rounds. The card is put in its mode before
launch — by the operator or the provisioning platform — and `optimize` only
observes what it is in, refuses a session that cannot work in that shape, and
hands the shape to the benchmark entrypoint that places work across partitions.

Two CLI flags configure this, both optional:

- `--compute-partition-mode {SPX,DPX,QPX,CPX}` asserts the mode the card is
  already in. It is a check, not a request: if the card is in a different mode
  the session is refused rather than silently measuring the wrong topology. If
  the card cannot be read at all, a declared mode is also a refusal — the flag
  exists to catch an external set that did not take, and an unverifiable
  assertion is not a satisfied one.
- `--streams-per-partition N` (default `2`) is how many concurrent streams the
  benchmark places on each partition. One stream leaves each partition idle
  through the fixed per-pass cost; beyond two, on the workloads measured so far,
  only queueing is added. A value below `1` is refused rather than replaced by
  the default, so `0` is a usage error instead of a silent `2`.

At launch the per-stream HBM footprint is checked against one partition's
memory. A workload that provably will not fit is refused in milliseconds instead
of failing out of memory hours in. The footprint is the checkpoint's weight bytes
— a lower bound, since each stream holds its own copy of the weights, which is
why a "does not fit" verdict from it is trustworthy and a "fits" verdict proves
nothing. When the checkpoint cannot be sized the session runs with a warning.

The check only applies where streams will actually share a partition: with a
serving framework and no partition flags, the shape is recorded and nothing is
refused, because nothing in that session places work per partition and, since
whole cards enumerate before partitions, its benchmark may not even land on one.

Multi-node sessions (`--nodes >= 2`) record no shape at all. The card this
process can read is not the card the benchmark runs on, and a shape recorded from
the wrong node is the exact mislabelling this feature exists to prevent. A
declared mode there is a usage error rather than a silently unchecked assertion.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_PARTITION_GPU` | `0` | Which GPU's partition state describes this session. A value that is not a GPU id falls back to `0` with a warning rather than silently — the fallback reads a different card, and every number the session files afterwards would carry that card's shape. |

## Runtime hand-off

Published once at launch by the CLI and read by the benchmark entrypoint. Do not
set these by hand: they are overwritten at every launch, and clearing them first
is what stops a second session in the same shell from inheriting a shape that
was not asked for.

The first three describe the card and are published for every session on a
readable card, because `platform_fingerprint()` reads them back from here — it
runs on the crash path, where spawning `amd-smi` is not acceptable. The last two
are instructions to a benchmark that places work on each partition, so they are
published only when one will.

| Variable | Published | Description |
|----------|-----------|-------------|
| `HYPERLOOM_PARTITION_MODE` | Always | The observed mode. Also recorded in the platform fingerprint, so a result is never filed under a topology it was not measured on. |
| `HYPERLOOM_PARTITION_COUNT` | Always | Partitions the card presents in that mode. |
| `HYPERLOOM_PARTITION_CU` | Always | Compute units per partition. The entrypoint selects partition devices by matching this exactly — HIP enumerates whole cards before partitions, so an index list computed at launch would be wrong in the one case that matters, and wrong invisibly. |
| `HYPERLOOM_PARTITION_STREAMS_PER_PARTITION` | Fan-out only | Streams to place on each partition. |
| `HYPERLOOM_PARTITION_TOTAL_STREAMS` | Fan-out only | `count × streams`; the total concurrency the entrypoint should drive if it fans out across every partition. |

Only scriptable frameworks (`xdit`, `custom`) place work per partition. A serving
session is handed no fan-out instruction rather than a concurrency nothing will
drive, and passing the flags with one warns. Its shape is still recorded in the
report and the fingerprint — that is provenance, not a hand-off — and the report
states plainly that the figure cannot be read as an aggregate.

## Choosing the mode: `scripts/partition_mode_sweep.py`

`optimize` treats the mode as fixed and asserts it. Deciding *which* mode to be
in is a separate job, done before the session, by
`python3 scripts/partition_mode_sweep.py`. It sets each mode on one card in
turn, runs the same benchmark on every partition that mode creates, sums the
result, and restores the card's entry mode on the way out.

```bash
# what it would do, nothing set
python3 scripts/partition_mode_sweep.py --suite-config round.yaml --dry-run

# sweep every mode the card reports, skipping any that cannot hold the workload
python3 scripts/partition_mode_sweep.py \
    --suite-config round.yaml --output-dir /shared/sweep \
    --per-stream-gib 20.7 --sudo
```

Each partition runs the `--suite-config` inference_suite round (`inference-suite-config/1`)
into its own run_dir, on that partition's HIP device and on the config's port plus the
partition index, with GPU telemetry off because `amd-smi` indices are not HIP indices.
The config must serve on one GPU, own its server, not profile, and not let the server
inherit a GPU mask; anything else is refused before the card is touched. A
`--benchmark-command` must leave an inference_suite run_dir at `{run_dir}`: every
partition is measured from its `report.json`, and a round that is not a valid
measurement leaves its mode unmeasured.

The fan-out is the point rather than a detail: a benchmark that loads one
partition and ignores the rest measures a fraction of the card, which makes
`CPX` look eight times worse than it is. The sweep therefore launches every
partition at once, and reports a mode only when all of its partitions returned
a measurement — a mode with six of eight reporting is unmeasured, not slow.

It publishes the same `HYPERLOOM_PARTITION_*` variables as a session, so a
benchmark entrypoint written against the table above works unchanged under
either. It pins each process with `ROCR_VISIBLE_DEVICES` and removes any
inherited `HIP_VISIBLE_DEVICES`, because two masks apply in sequence and the
second indexes into the first.

The privileged `amd-smi set` lives here and nowhere else. An operator-run script
between benchmarks is a reasonable place for a card-wide mutation that evicts
every GPU context; an optimization loop that also runs agent-authored code is
not. Before setting anything it refuses if a process holds a context on the card
being swept — and only that card, since no other card is repartitioned. A
neighbour's benchmark on a shared node is not a reason to stop. If `amd-smi`
reports its process list in a shape the script cannot read, that is also a
refusal rather than an assumption that the card is idle: `--allow-busy` is the
way past both, and `--dry-run` never asks.

Exit codes: `0` swept, `1` nothing measurable, `2` refused before anything
changed, `3` swept but the card could not be restored to its entry mode, `4`
stopped on an error it does not model. Every path out of a started sweep goes
through the restore and the report, so a mode that fails unexpectedly costs its
own result and nothing else.
