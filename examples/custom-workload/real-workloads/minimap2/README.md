# minimap2 — a real workload adapter

The sibling `examples/custom-workload/` teaches the contract with a toy benchmark. This is the same
contract wrapped around a real one: **minimap2**, the long-read genomics aligner, running its GPU
chaining path on AMD Instinct.

Everything here is buildable and runnable from public sources — no AMD-internal assets.

## Build the workload

```bash
# One work dir, three siblings: the clone Hyperloom may patch, the dataset, and these files.
export WORK=~/minimap2-run
mkdir -p "$WORK/data" "$WORK/adapter"
cp -r . "$WORK/adapter"            # run this from the adapter directory

cd "$WORK"
git clone --branch rocm-enablement https://github.com/AMD-Ecosystem/minimap2.git
cd minimap2 && git submodule update --init --recursive --depth 1

export ROCM_PATH=$(rocm-sdk path --root)      # or /opt/rocm on a system install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU=AMD -DGPUARCH=gfx950 -DROCM_PATH=$ROCM_PATH
cmake --build build -j
```

Two things that will stop you if you skip them:

- **Submodules are required.** Without `git submodule update --init`, CMake fails with
  *"does not contain a CMakeLists.txt file"* on `external/`.
- **`ROCM_PATH` must point at a tree containing `lib/cmake/hip-lang/hip-lang-config.cmake`.** On the
  `rocm/pytorch` images that is the *devel* tree, not the core one — `pip install "rocm[devel]"`,
  then `rocm-sdk path --root` resolves it. Setting `ROCM_PATH` to the core SDK dir configures far
  enough to print the build summary and then fails at `enable_language(HIP)`.

## Make a dataset

The adapter needs a reference and reads. `gen_reads.py` synthesizes both, seeded, so runs are
reproducible and nothing has to be downloaded:

```bash
cd "$WORK/data"
python3 "$WORK/adapter/gen_reads.py" 60 60000   # 60 Mbp reference, 60k reads -> ref.fa, reads.fq
```

The dataset lives beside the clone, not inside it, so the `git clean -fd` that precedes a patch
candidate cannot delete it.

> **Read this before quoting any number from synthetic data.** A random reference has none of the
> repeat structure of a real genome, so chaining is unrealistically easy and the CPU path looks far
> better than it does on real data. Use this to exercise the adapter and the sweep. For a
> performance claim, point `MINIMAP2_REF`/`MINIMAP2_READS` at a real reference and real reads.

## Run it

```bash
export MINIMAP2_BIN="$WORK/minimap2/build/bin/minimap2"
export MINIMAP2_REF="$WORK/data/ref.fa"
export MINIMAP2_READS="$WORK/data/reads.fq"

RESULT_DIR=/tmp/mm2-baseline "$WORK/adapter/custom_mi355x.sh"
cat /tmp/mm2-baseline/inferencex_result.json
```

Then under Hyperloom.

> **Point `--framework-path` somewhere disposable, never at your Hyperloom clone.**
> It is the tree Hyperloom patches: integrating a patch candidate runs
> `git stash push -u`, `git reset --hard` and `git clean -fd` against it, and commits
> KEEPs into it. Aimed at the clone you are reading this in, that discards your own
> uncommitted work — and `stash -u` takes untracked files too, including any dataset
> you generated there.

Here the workload *has* a checkout of its own — the `minimap2` tree you cloned above — so that is
what `--framework-path` names. Keep the generated dataset outside it, or the `stash -u` on the first
patch candidate will take `ref.fa` and `reads.fq` with it:

```bash
# layout: $WORK/minimap2 (the clone), $WORK/data (ref.fa + reads.fq), $WORK/adapter (these files)
export WORK=~/minimap2-run

export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_BYPASS_SCRIPTS_DIR="$WORK/adapter"
export MINIMAP2_BIN="$WORK/minimap2/build/bin/minimap2"
export MINIMAP2_REF="$WORK/data/ref.fa"
export MINIMAP2_READS="$WORK/data/reads.fq"

python3 -m hyperloom optimize \
    --framework custom \
    --framework-path "$WORK/minimap2" \
    --benchmark-scripts-dir "$WORK/adapter" \
    --model minimap2 --gpu-type mi355x --max-hours 1
```

## What the sweep is actually optimizing

`--gpu-chain` is **always on**. The baseline is the GPU path at its *auto-detected* configuration,
and the levers reconfigure it. This is not a CPU-vs-GPU comparison and the adapter makes no such
claim.

Measured on one MI350X (gfx950), ROCm 7.14, 60 Mbp / 60k synthetic reads, 2 timed iterations:

| Config | aln/s | vs baseline |
|---|---|---|
| *(baseline)* auto workers | 4,745.71 | 1.00x |
| `gpu_workers_4` | **9,309.54** | **1.96x** |
| `gpu_workers_16` | 4,751.72 | 1.00x |

The auto-sizer chose 8 workers on this machine and 4 was nearly twice as fast — the kind of thing
worth sweeping rather than assuming. The optimum is machine- and dataset-dependent; expect a
different winner on different silicon.

## The quality gate

Alignment recovery against a baseline PAF, threshold 0.98 by default. Set `MINIMAP2_REF_PAF` to a
PAF from your baseline run and the gate becomes a real comparison — without it the adapter records
`1.0` against the run itself, which is honest bookkeeping but gates nothing. **Set it before you
trust a speedup**: a configuration that drops alignments will otherwise look like a win.

## Files

| File | Purpose |
|---|---|
| `custom_mi355x.sh` | Hyperloom entrypoint; dispatches to the adapter |
| `_minimap2_bench.sh` | Runs the benchmark, writes `inferencex_result.json` |
| `levers.json` | Six declared variants over GPU workers, host threads, flush threshold |
| `gen_reads.py` | Deterministic synthetic reference + reads |

`levers.json` is read by the declared-lever grid in
[#1546](https://github.com/AMD-AGI/Hyperloom/pull/1546); until that lands the adapter still runs,
but the variants are not swept.
