# Custom workload: the adapter contract, end to end

A minimal, runnable `--framework custom` workload. It has no GPU dependency and
no third-party packages, so you can run it on a laptop and watch the whole
contract work before pointing Hyperloom at something that takes an hour.

The benchmark itself is deliberately boring -- a chunked integer reduction. What
matters is the shape around it, which is identical whether the thing underneath
is a toy loop or a genomics aligner.

## What a custom workload has to provide

Two files in `--benchmark-scripts-dir`:

| File | Required | Purpose |
|---|---|---|
| `custom_<runner-type>.sh` | yes | The entrypoint. A lone `*.sh` also works if you only have one machine. |
| `levers.json` | no | The knobs the optimizer may sweep. Without it you still get profiling, baselining and kernel optimization -- just no configuration search. |

> **Note:** `levers.json` is read by the declared-lever grid added in
> [#1546](https://github.com/AMD-AGI/Hyperloom/pull/1546). Until that lands, the
> entrypoint and result contract below work as described, but the declared
> variants are not seeded -- `--framework custom` starts from an empty grid.

The script writes one file, `$RESULT_DIR/$RESULT_FILENAME.json`, and Hyperloom
reads exactly two things out of it:

```json
{
  "output_throughput": 14784.23,
  "quality_gate": { "passed": true }
}
```

`output_throughput` is the objective, and it is **maximized**. The unit is
whatever you say it is -- kitems/s here, alignments/s or frames/s or steps/s
elsewhere. No core code knows or cares.

`quality_gate.passed` is the only correctness signal Hyperloom has. A missing or
unparseable gate scores 0.0 accuracy, which rejects every candidate, so a run
that "finds nothing" is often really a run whose gate never parsed.

Everything else in the file is yours. Carry whatever makes the run explicable
later -- this example records the checksum it compared, the config it ran, and
the mean latency.

## Run it standalone first

Always do this before involving Hyperloom. It separates "my benchmark is broken"
from "my Hyperloom invocation is wrong", and it takes seconds:

```bash
cd examples/custom-workload
RESULT_DIR=/tmp/example-run ./custom_mi300x.sh
cat /tmp/example-run/inferencex_result.json
```

Then confirm a lever actually moves the number -- if it does not, the sweep has
nothing to find:

```bash
EXAMPLE_WORKERS=8 RESULT_DIR=/tmp/example-run ./custom_mi300x.sh
```

On a 16-core machine the declared levers span roughly 3.7x, which is the kind of
spread that makes a sweep worth running:

| Variant | kitems/s | vs baseline |
|---|---|---|
| *(baseline)* `workers=1 chunk=100000` | 14,784 | 1.00x |
| `workers_2` | 25,937 | 1.75x |
| `workers_4` | 40,945 | 2.77x |
| `workers_8` | 54,959 | **3.72x** |
| `chunk_25k` | 15,282 | 1.03x |
| `chunk_400k` | 15,579 | 1.05x |
| `workers_4_chunk_25k` | 31,042 | 2.10x |

Two things worth noticing, because both show up in real workloads. The chunk
knob barely matters on its own. And `workers_4_chunk_25k` is *slower* than
`workers_4` alone -- combining two individually-harmless settings made things
worse, because smaller chunks mean more pickling per unit of work. Knobs
interact, and that is the argument for searching rather than reasoning.

Your numbers will differ; the ranking is machine-dependent, which is the point.

## Run it under Hyperloom

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_BYPASS_SCRIPTS_DIR="$PWD"

python3 -m hyperloom optimize \
    --framework custom \
    --framework-path "$PWD" \
    --benchmark-scripts-dir "$PWD" \
    --model example-reduction \
    --gpu-type mi300x \
    --max-hours 1
```

Add `--no-framework-agent` for a measure-only pass: it establishes the baseline
and stops, which is the cheapest way to prove the contract is wired up.

## The correctness gate is the interesting part

The reduction is associative and the chunking is deterministic, so **every knob
setting produces the same checksum**. The gate compares against a single-chunk
reduction and fails on any mismatch.

That is not decoration. It is what separates a real speedup from a wrong answer
returned quickly -- and a configuration search with no gate will happily find
you the latter. When you write your own, make the gate something that would
genuinely fail if the optimization broke the workload: numeric parity against a
reference, a checksum, a recovery rate against known-good output.

## Declaring levers

`levers.json` is a JSON **list** of variants. Each needs a `name` and at least
one thing a restart can apply:

```json
[
  {"name": "workers_4", "extra_envs": {"EXAMPLE_WORKERS": "4"}},
  {"name": "chunk_25k", "extra_envs": {"EXAMPLE_CHUNK": "25000"}}
]
```

Besides `extra_envs`, a variant may carry `extra_args`, `remove_args`,
`unset_envs`, or `args_mode: "replace"`. A variant with only a name is rejected
at launch, because it would re-benchmark the baseline under a new label.

Declarations seed *ahead of* the model's own proposals, so the sweep runs even
when the model proposes nothing. A model proposal of the same name wins, and a
declaration the ledger has already benchmarked is not re-seeded on later rounds.

A malformed `levers.json` fails the launch with exit code 2 rather than starting
a run whose grid silently comes back empty.

## Porting this to a real workload

1. Replace `benchmark.py` with your benchmark; keep the result-JSON writer.
2. Decide the objective and its unit, and make sure a bigger number is better.
3. Write a gate that would actually fail if the answer changed.
4. List the knobs you already tune by hand in `levers.json` -- those are the
   ones worth sweeping, and you already know they matter.

See `docs/how-to/optimize-custom-workload.md` for the full reference, including
pinning operator-supplied environment and resuming a session.
