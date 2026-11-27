---
myst:
    html_meta:
        "description": "EXPLORE pre-filter variables for Hyperloom: measurement store, reduced-GPU screen, and InferaSim projection filters that remove variants before benchmarking."
        "keywords": "Hyperloom, environment variables, EXPLORE pre-filters, HYPERLOOM_MEASUREMENT_STORE, HYPERLOOM_EXPLORE_SCREEN, HYPERLOOM_EXPLORE_PROJECTION, HYPERLOOM_INFERASIM_ROOT"
---
# Hyperloom EXPLORE pre-filters

Three opt-in filters can remove variants from an EXPLORE round before any of
them is benchmarked. All three are off by default; with every switch below
unset, a round runs exactly as it did before. They run once, before the first
variant, in this order: measurement store, screen, projection.

A filter only removes variants. It never adds or reorders one, and none of its
numbers reaches KEEP/REVERT, `current_best`, the journal or the knowledge base:
every KEEP comes from a measured decision round. A removed variant is recorded
in the round's `skipped_dup` with a reason (`measured_revert_before`,
`screen_decisively_slower`, `projection_decisively_slower` or
`projection_outranked`). It is not journaled or written to the tested ledger,
so a later round may propose it again.

No filter ever drops a variant in a stack-revalidation round or a config replay
(`geak_revalidate`).

Boolean switches below are on for `1`, `true`, `yes` or `on`
(case-insensitive) and off for anything else. A malformed number falls back to
its default.

## Measurement store

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_MEASUREMENT_STORE` | unset (off) | Directory that records every decision round that REVERTed on performance, across sessions. A later variant whose exact launch REVERTed before is dropped (`measured_revert_before`) instead of measured again. A KEEP, an accuracy REVERT (`accuracy_drop` / `accuracy_unavailable`) and a failed round are never recorded, so they are always measured. The key covers the model, engine, GPU, workload signature, benchmark block, inherited args, warm or cold decision protocol, grading axis, KEEP threshold, image digest, engine / ROCm / aiter versions, the stack, the variant, the launch envs, and the content of every kernel overlay on `PYTHONPATH`. A launch is never keyed, and so always measured, when the image digest or engine version is unknown, a runtime override is in effect, or an overlay cannot be hashed. Safe to share between sessions; writes are atomic. |
| `HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS` | `14` | Records older than this many days are ignored. Negative values are treated as `0`, which ignores every record. |

## Screen (reduced-GPU probe)

A real engine probe on fewer GPUs, using Infera's vLLM benchmark script. The
baseline probe is the round's stack, and each variant is probed on that stack,
with the server args and envs the round would launch it with. It drops a variant only when the probe resolved the same kernels as the deployment
did, read from both sides' logs; otherwise it returns the round untouched. A
variant whose probe fails is kept. It needs at least three variants and
`HYPERLOOM_INFERASIM_ROOT`. On the one grid it was measured against, the order
did not transfer to fewer GPUs, so validate it on your deployment before relying
on it.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_EXPLORE_SCREEN` | off | Turns the screen on. |
| `HYPERLOOM_EXPLORE_SCREEN_MARGIN_PCT` | `10` | Drop a variant that probes more than this many percent behind the probed stack (`screen_decisively_slower`). Measured on eight replicas, a gap this wide is one every replica reproduces. A non-positive value falls back to the default. |
| `HYPERLOOM_EXPLORE_SCREEN_GPUS` | `1` | GPUs the probe runs on, capped at the deployment's TP. |
| `HYPERLOOM_EXPLORE_SCREEN_MODEL` | the config's model | Model the probe loads, for example a smaller checkpoint of the same architecture. |
| `HYPERLOOM_EXPLORE_SCREEN_LAYERS` | all | Truncate the probed model to this many hidden layers. |
| `HYPERLOOM_EXPLORE_SCREEN_ATTENTION_BACKEND` | the deployment's | Attention backend pinned onto the probe. By default it is read from the deployment's `--attention-backend`, else from the kernels the session's own benchmark logged. A variant that sets the backend itself still wins. |
| `HYPERLOOM_EXPLORE_SCREEN_TIMEOUT_SEC` | `1800` | Wall-clock limit for one probe. A probe that times out is kept. |

## Projection (InferaSim)

Projects the stack and each variant with InferaSim and compares the two
projections, never a projection with a measurement. Only a variant that moves
levers the projection models (TP / EP / PP, `CONC`, `--max-num-seqs`, ISL / OSL,
KV-cache dtype) can be dropped. Any other lever (scheduler flags, kernel or env
levers, `remove_args` / `unset_envs`, overlays) means the variant is benchmarked.
So does a variant that fails to project, projects identically to the stack, or
was projected with a different estimator. If the stack itself cannot be
projected, the whole round is benchmarked. The round result carries a
`projection` summary listing every variant's route, the mode, and the GPU it was
priced for.

Infera is an optional dependency: point `HYPERLOOM_INFERASIM_ROOT` at an Infera
checkout, or install the `inferasim` extra (`pip install 'hyperloom[inferasim]'`),
which pins `amd-infera[projection]==0.3.3`. The checkout takes precedence when both
are present. Either way, a projection also needs Origami, InferaSim's GEMM
simulator, which is not on PyPI and is built against ROCm:
`pip install 'git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python'`.
Without it every projection fails, and the round is benchmarked untouched.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_EXPLORE_PROJECTION` | off | Turns the projection on. |
| `HYPERLOOM_EXPLORE_PROJECTION_MARGIN_PCT` | `15` | Simulate mode: drop a variant projected more than this many percent behind the stack (`projection_decisively_slower`). The default sits above the 9.7% median output-throughput error simulate mode scored on held-out AgentX configurations. A non-positive value falls back to the default. |
| `HYPERLOOM_EXPLORE_PROJECTION_CALIBRATED_MARGIN_PCT` | `5` | The margin when both the stack and the variant are calibrated against an in-regime anchor (regime distance 0); otherwise the simulate margin applies. The default is a placeholder until the gap error between two calibrated configurations has been measured. |
| `HYPERLOOM_EXPLORE_PROJECTION_TOP_K` | `5` | Of the variants that are not decisively behind, benchmark only the top K by projected gain; the rest are dropped (`projection_outranked`). `0` turns the cap off. On held-out AgentX stack ladders, the DES top 5 contained the best configuration 94% of the time, against 47% for the single top pick. |
| `HYPERLOOM_INFERASIM_ROOT` | unset | Path to the Infera checkout; it is put first on `sys.path`. Required by the screen and by harvesting, and by the projection unless Infera is installed. |
| `HYPERLOOM_INFERASIM_MODE` | `auto` | `auto` projects the round in benchmark (calibrated) mode when the anchor store already holds an in-regime anchor for the stack, and in simulate mode otherwise. It is resolved once, on the stack, and never boots a server. `simulate` never reads an anchor. `benchmark` requires an anchor, harvests one on a miss, and fails closed (the variant is benchmarked) when none can be had. Any other value is an error, and the round is benchmarked untouched. |
| `HYPERLOOM_INFERASIM_ESTIMATOR` | `des` | `des` replays a closed-loop client at the round's concurrency (10 requests per client, between 64 and 4,000). `analytical` selects the closed form, which ranked held-out configurations far worse. Any other value is an error. |
| `HYPERLOOM_INFERASIM_MODEL` | inferred | InferaSim model preset to project, for example `gpt_oss_120B`. By default it is inferred from the model path: the most specific match wins, and a distilled checkpoint uses its student. With no match and neither this nor `HYPERLOOM_INFERASIM_WORKLOAD` set, the round is benchmarked. |
| `HYPERLOOM_INFERASIM_WORKLOAD` | bundled template | Full InferaSim workload YAML; takes precedence over `HYPERLOOM_INFERASIM_MODEL`. |
| `HYPERLOOM_INFERASIM_GPU_ARCH` | `mi355x` | Part to price when the session's GPU type is unknown. When it is known, the session GPU always wins. |
| `HYPERLOOM_INFERASIM_HBM_GB` | from the part | Per-GPU HBM capacity in GB. By default the part's entry in the shared hardware table (`roofline_ceiling.HW_SPECS`), e.g. 192 (MI300X / MI308X), 256 (MI325X) or 288 (MI355X). |
| `HYPERLOOM_INFERASIM_SERVING_MODEL` | `continuous` | InferaSim serving model: `continuous` or `static` batching. |

## InferaSim anchors

Benchmark mode calibrates a projection against the nearest anchor in the
candidate's regime: the same model, engine and GPU, plus matching attention
backend, CUDA graph, aiter and dtypes. An anchor is used only if its decode
curve passes a sanity check, and only if the image, versions and launch envs it
recorded do not contradict the candidate's.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_INFERASIM_ANCHOR_STORE` | unset | Anchor store directory searched for the nearest in-regime anchor. When set, every successful EXPLORE decision round's TPOT is also recorded in it as a measured, real-weights anchor point under `measured/`. Concurrencies of one launch merge into a ladder. A ladder that is not monotonic is not written, and a launch whose overlays cannot be hashed is not recorded. |
| `HYPERLOOM_INFERASIM_ANCHOR` | unset | One anchor JSON to calibrate every projection against; it outranks the store. A file that fails the sanity check is not used, and the store is not searched in its place, so the projection has no anchor. |
| `HYPERLOOM_INFERASIM_ANCHOR_SCALING` | unset | Comma-separated anchor JSONs for TP scaling, passed to InferaSim in benchmark mode. |
| `HYPERLOOM_INFERASIM_HARVEST` | on | Benchmark mode only: on a miss, boot a reduced-width server with Infera's served-anchor benchmark, measure an anchor for the regime, and index it in the store. `0`, `false`, `no` or `off` turns this off, so a miss fails closed instead. `auto` mode never harvests. |
| `HYPERLOOM_INFERASIM_HARVEST_PYTHON` | the running interpreter | Interpreter that can launch the engine for a harvest. |
| `HYPERLOOM_INFERASIM_HARVEST_GPUS` | `4` | GPUs a harvest may use, capped at TP × PP. |
| `HYPERLOOM_INFERASIM_HARVEST_TIMEOUT_SEC` | `1800` | Wall-clock limit for one harvest. |
