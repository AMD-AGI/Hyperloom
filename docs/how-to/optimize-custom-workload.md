---
myst:
    html_meta:
        "description": "How to optimize your own workload with Hyperloom using --framework custom. Covers the code checkout, the entrypoint contract, the mandatory quality gate, and pinning knobs with --extra-env."
        "keywords": "Hyperloom, custom framework, scriptable workload, quality gate, AMD GPU, ROCm, framework rewrite, host probe, extra-env, optimization"
---
# Optimize your own Hyperloom workload

Every other framework in the registry describes a workload Hyperloom knows: its
upstream, its entrypoint, the knobs worth exploring. `--framework custom`
describes none of that, because the workload is yours. It runs on the
**scriptable** (server-less) path: there is no OpenAI-compatible server, no
`benchmark_serving.py` client, and every measurement is one command.

You supply three things; Hyperloom takes over from there.

| You supply | Flag | What it is |
|------------|------|------------|
| Code checkout | `--framework-path` | The source tree to optimize, separate from the weights |
| Entrypoint directory | `--benchmark-scripts-dir` | Holds `custom_<gpu-type>.sh` |
| Knobs your script reads | `--extra-env NAME=VALUE` | Repeatable; Hyperloom interprets none of them |

If you have not installed Hyperloom yet, follow the [installation
instructions](../install/install.md) first, then return here. For the shipped
frameworks, see [Run a Hyperloom optimization](optimize.md) instead.

## Launch

```bash
python3 -m hyperloom optimize -v \
  --framework custom \
  --framework-path /path/to/my-checkout \
  --benchmark-scripts-dir /path/to/my-scripts \
  --model /path/to/weights \
  --gpu-type mi355x --tp 8 --max-hours 12 \
  --extra-env MYFW_STEPS=50 --extra-env MYFW_CKPT=/path/to/ckpt
```

Neither `--framework-path` nor `--benchmark-scripts-dir` is optional here. With
no shipped entrypoint to fall back on, a missing path fails at launch rather
than at the first benchmark.

## The code checkout

`--framework-path` is the **code** checkout, which is not the same thing as the
weights under `--model`. It does more than tell your script where the code
lives: it registers the tree as the framework source root a session searches and
patches. The source probe discovers pip-installed packages on its own but never
a git checkout, which is why this must be explicit.

The equivalent environment variables resolve in this order:
`<FRAMEWORK>_REPO_PATH` > `<FRAMEWORK>_DIR` > `FRAMEWORK_REPO_PATH`. Prefer the
generic `FRAMEWORK_REPO_PATH`: a session is single-framework by construction
(the CLI locks `$FRAMEWORK` for the run), so the prefix resolves no possible
collision and only forces you to rename variables when you switch frameworks.

```{warning}
Hyperloom **commits into this tree**. Each accepted candidate lands as a commit
whose message looks like `hyperloom KEEP <id> (+12.34%)`. Point the flag at a
checkout dedicated to the run, not at a tree you rely on as a measurement
reference.
```

## The entrypoint

Hyperloom looks for your entrypoint inside `--benchmark-scripts-dir` in this
order:

1. `custom_<gpu-type>.sh`, named after the board itself — with `--gpu-type mi325x`,
   that is `custom_mi325x.sh`.
2. Failing that, the **single** `.sh` file in the directory, which is what an
   operator who wrote one script for one machine actually has.
3. If neither applies, the launch exits 2: there is no script to run.

The session freezes the script it found, with the sha256 of its content, as its
base launch. A round whose script has changed since is refused with
`source_files_changed`; start a new session to measure an edited script.

Your script is invoked with `RESULT_DIR`, `RESULT_FILENAME=inferencex_result`,
`MODEL`, `TP`, `ROCR_VISIBLE_DEVICES` and, when the session holds a quality
reference, `QUALITY_REFERENCE` injected into its environment; a profile round
adds `PROFILE=1` and the trace directory as `VLLM_TORCH_PROFILER_DIR` and
`SGLANG_TORCH_PROFILER_DIR`. Pins named like an acceptance threshold (`*_MAX`,
`*_MIN`, `*threshold*`) are refused: Hyperloom judges the result. It must write an InferenceX-shaped report to
`$RESULT_DIR/inferencex_result.json`:

```json
{
  "framework": "custom",
  "workload_kind": "scriptable",
  "throughput_unit": "fps",
  "output_throughput": 0.334003,
  "quality_gate": { "passed": true, "ssim": 0.748698, "mse": 0.006005 }
}
```

`output_throughput` is the objective and is maximized. Declare its unit in
`throughput_unit`; `fps`, `img/s` and `tokens/s` are all fine.

## The quality gate is mandatory

A scriptable workload has no server, and therefore no accuracy benchmark. The
`quality_gate` block is the only correctness signal Hyperloom has, and it is
fail-closed:

```{important}
A baseline whose report carries no `quality_gate` **stops the run at once** with
`stop_reason: baseline_accuracy_failed`; the log line names the report it read.
A candidate whose report has no gate scores 0.0 accuracy and is rejected.
```

A gate with `passed` is taken as written. A gate without it is judged on the
raw scores it carries, each against the limit the gate names or, when it names
none, against Hyperloom's: `lpips` at most 0.15, `ssim` at least 0.85, `mse` at
most 0.006. A gate with neither `passed` nor a score fails.

Run your entrypoint by hand once and confirm the key is actually present in
`inferencex_result.json` before you spend a budget on it.

What the gate contains is yours to decide, because only you know what
correctness means for your workload. For a chaotic generative pipeline, a fixed
similarity threshold is usually the wrong bar — a numerically faithful kernel
that is not bit-identical still drifts over many steps. A self-calibrating band
(measure the pipeline's own drift under a tiny perturbation, widen it by a
margin, then require later legs to stay inside it) is the pattern that holds up.

## Pinning knobs with `--extra-env`

`--extra-env` carries the knobs your script reads. A pin becomes a real
environment variable for the session, so it also sets a Hyperloom control
variable of the same name. Their semantics matter:

> Whatever you pin becomes part of the **measurement contract**. A variant can
> add keys but must not overwrite a pinned one, because the baseline number was
> measured with it.

So pin what must not move, and leave the rest for exploration. This is also what
decides your baseline. Given one code tree and one entrypoint, pinning a set of
already-validated optimization switches makes the baseline the optimized number
and asks the search for an increment on existing work; pinning none of them
makes the baseline the stock path, so every switch the search finds is its own
discovery.

The CLI exports each pin under its own name, persists them in `state.json`, and
also serializes them into `INFERENCE_OPTIMIZER_EXTRA_ENV` as a JSON object so a
resume can tell which names the previous launch set.

On a fresh launch a pin and an `export` of the same name are equivalent: no name
is withheld. For a knob Hyperloom resolves itself — `ISL`, `OSL`, `CONC`, `TP`,
`EP`, `PRECISION`, `MAX_MODEL_LEN`, `FRAMEWORK` — the environment is one rung of
that knob's ladder: **an explicit flag wins, then the environment (where your
pin lands), then the default**. So `--extra-env ISL=4096` sets the ISL, and
`--isl 2048 --extra-env ISL=4096` runs at 2048. On a resume the two stop being
equivalent for `ISL`, `OSL`, `CONC`, `TP`, `EP` and — when the session recorded
one — `MAX_MODEL_LEN`: the pin is still honoured, but a bare `export` is
overwritten by the session's recorded value, so the ladder reads **flag, then the
pins in effect for this resume (its own `--extra-env`, or the launch's persisted
pins when it passes none), then the recorded value, then the environment, then
the default**. `PRECISION` keeps the fresh-launch ladder on a resume too, with
the recorded value and the checkpoint's dtype below the environment. `FRAMEWORK`
belongs to the session: a resume re-exports the recorded framework and reads
`$FRAMEWORK` only when the session recorded none, so neither a pin nor a flag
changes it.
Everything else a pin names is set directly and is not interpreted.
**Forward every pin as its own `--extra-env` flag** — a dropped pin is lost
silently. On `--resume` the persisted pins are re-exported, so they only need
re-passing when you want to change them.

## Declaring the levers to explore

"Leave the rest for exploration" needs somewhere for the search to start. An LLM
framework has source the model can read to infer its knobs; a `custom` workload
does not, so its seed grid is empty and EXPLORE has nothing to sweep until the
model guesses.

Declare the knob space instead, in `levers.json` beside your benchmark script
(inside `--benchmark-scripts-dir`):

```json
[
  {"name": "workers_1", "extra_envs": {"GPU_CHAIN_WORKERS": "1"}},
  {"name": "workers_2", "extra_envs": {"GPU_CHAIN_WORKERS": "2"}},
  {"name": "threads_4",  "extra_envs": {"OMP_NUM_THREADS": "4"}}
]
```

Each entry is an explore variant: a `name` plus at least one thing a restart can
apply (`extra_envs` as an **object**, `extra_args`, `remove_args`, `unset_envs`,
or `args_mode: "replace"`). Each entry is judged by the same rule the executor
applies when it builds the grid, so a shape that normalizes away — `extra_envs`
given as a string or a list, say — is rejected at launch rather than quietly
benchmarking the baseline again under a new name. The declarations are seeded ahead of the model's own
proposals, so the sweep runs even when the model proposes nothing; a variant the
model proposes under the same name wins, and a declaration whose name already
appears in the explore ledger's `tested` rows is not re-seeded on later rounds.

The file is optional — but if it exists and is malformed, the launch fails with
exit code 2 rather than starting a run whose grid silently comes back empty. It
lives in the scripts dir because that directory is already persisted and
re-exported on `--resume`.

## The FRAMEWORK_AGENT phase

`custom` runs the FRAMEWORK_AGENT phase by default, and it is the only phase
that can restructure the pipeline itself: sequence-parallel all-to-all,
per-step host-to-device copies, repeated work hoisted out of a rollout loop.
`explore` reaches none of that, because an explore variant is only CLI flags
plus environment. Pass `--no-framework-agent` to skip the phase, but do not
assume it is the default.

Because a scriptable framework has no server, the authoring arm dispatches
`framework_rewrite_specialist` rather than `serving_specialist`. The two share no
optimization surface: an autoregressive video rollout has no scheduler, no
continuous batching and no KV-cache admission policy, and its wins are the
redundant work its loop structure creates. The specialist works from the pattern
catalog in
[`framework_rewrite_patterns.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/references/framework_rewrite_patterns.md).

### Evidence: the host probe

A `profile` leg arms a host-side probe that is injected through a `PYTHONPATH`
prefix, so **your entrypoint is untouched**. It writes
`framework_rewrite_evidence.json` beside the workspace and measures what a GPU
kernel breakdown structurally cannot: object collectives round-tripping through
the host, device-to-host syncs, and repeated host-to-device copies.

| Setting | Effect |
|---------|--------|
| (default) | Tier 1 on, and cheap |
| `HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE_DEEP=1` | Tier 2: per-function call counts with argument-repeat rates |
| `HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE=0` | Disables the probe entirely |

Tier 2 is what separates a memoization candidate from a loop-hoist enabler —
whether a value should be cached or lifted out of the loop is exactly what the
argument-repeat rate tells you. It inflates host time enough to skew a
co-collected torch trace, so **give it its own leg**.

### Every rewrite ships behind a switch that defaults off

This is a mechanism, not a style preference. Each rewrite must sit behind its own
environment switch that defaults **off**, declared in a `framework_switches`
manifest with `category`, `target`, `depends_on` and `enables`. That discipline
buys three things:

- A **switch-off parity leg** runs with every switch unset and must reproduce the
  base within ±2%. A patch that is not genuinely inert when disabled is reverted
  rather than silently poisoning every later measurement.
- A bundle that passes correctness but misses the throughput threshold is **kept
  inert** instead of reverted. Default-off code costs nothing, and reverting
  would discard the rewrites that do pay along with the one that does not.
- Accepted switches are registered as **search levers**, so `explore` measures
  each rewrite's own contribution (additive while the levers are dormant,
  leave-one-out once they are on) and searches combinations along the declared
  dependency closure, so an enabler is never judged alone.

That last point is not a nicety. A hoist whose only value is making a downstream
cache hit measures flat on its own; a greedy accept/reject loop rejects it and
then measures every dependent rewrite against a permanently cold cache, losing
the bundle rather than the lever.

## Other requirements

**No concurrency sweep.** A scriptable workload runs one forward pass of your script, so there is no request concurrency to vary: every rung of the ladder would measure the same thing. SWEEP records a terminal skip with `non_serving_workload` rather than spending its budget on a curve whose shape means nothing.

**GPU type.** `--gpu-type` accepts `mi300x`, `mi308x`, `mi325x`, `mi355x`,
`r9700`, `r9600d` and `radeon8060s` (Strix Halo). The entrypoint name uses the board as
given: `mi325x` looks for `custom_mi325x.sh`, not `custom_mi300x.sh`. Without the flag, Hyperloom auto-detects through
`rocm-smi --showproductname`.

**Pin GPUs with `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.** On the known
ROCm stack the latter can make `torch.cuda.is_available()` return false.

**Load workspace settings through the shared loader.** Existing caller values
remain authoritative, including gateway keys and custom headers. The loader reads
`.env` as data rather than executing it; never print credential values. See
[Authentication and credentials](../reference/authentication.md).

```bash
export REPO_ROOT="${REPO_ROOT:-$(pwd -P)}"
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
. "${INSTALL_SH%/*}/runtime_env.sh"
load_dotenv_no_clobber
```

**Sessions.** `USER_DATA_PATH` sets the session root, and each `optimize`
creates a new timestamped subdirectory under it. Use
`--resume-from <subdir>` to continue an existing session; `--force-resume`
pushes past the terminal-state guard. Without `--resume-from` you always get a
fresh session, so an interrupted run is never picked up by accident.

A resumed session keeps the budget it started with. Elapsed time is summed
forward over every leg, so resuming never hands the run another `--max-hours`,
however the previous leg ended. To let a run that has spent its budget carry
on, grant more explicitly with `--extend-hours <n>`; the grant is recorded in
the session state with its reason.

## Monitor the run and read the output

The benchmark output does not go to the launcher log; it lands in the session
directory:

```text
<session>/runs/baseline/<id>/.../suite_run_<n>/client/stdout.log
<session>/runs/baseline/<id>/.../suite_run_<n>/client/inferencex_result.json
<session>/runs/baseline/<id>/.../suite_run_<n>/report.json
<session>/reports/optimization_journal.json
```

Check the baseline `output_throughput` first. If a pin is wrong, that number
shows it immediately, instead of after hours of search against the wrong
denominator. For the full artifact schema, see
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).

## Optimize a single Python module

Everything above asks you to write a benchmark. You can instead keep model loading, inputs, validation and timing in the script you already have, and expose one optimization point:

Import the hook **twice**. Once as the very first thing in the script, before any other import -- side effect only, so accepted settings land as environment variables before anything else can read and cache them. And once again at the documented call site, after the model is loaded, for the patches themselves, which need the model object and so cannot move earlier. Python runs a module's top-level code only on its first import, so the second import is a cached no-op for the side-effecting part; this costs one extra line.

```python
try:
    import hyperloom_optimize  # noqa: F401  -- side effect only; must be first
except ModuleNotFoundError as exc:
    if exc.name != "hyperloom_optimize":
        raise

# ... all your other imports, including torch, go after the block above ...

try:
    from hyperloom_optimize import hyperloom_optimize
except ModuleNotFoundError as exc:
    if exc.name != "hyperloom_optimize":
        raise

    def hyperloom_optimize(model):
        return model

# Load model and weights, construct inputs, and capture any reference first.
model = hyperloom_optimize(model)
outputs = model(**inputs)
```

Skipping the early import does not fail quietly. If the exported module carries accepted settings, it checks at import time whether the GPU is already initialized or torch's compiler stack is already loaded, and raises if so, rather than silently shipping a setting that never took effect.

Point `--framework-path` at a directory holding `hyperloom_optimize.py`, initially an identity function. It does not need to be a Git checkout. In the frozen benchmark directory, declare `optimization_scope.json`:

```json
{"schema_version": 1, "file": "/absolute/workspace/optimization/hyperloom_optimize.py"}
```

An optional `function` names the entry point, so neither the module nor the function need mention Hyperloom: `{"schema_version": 1, "file": ".../encoder_opt.py", "function": "apply"}`.

The declaration is the boundary, and it is enforced everywhere a change could enter the tree. Patch integration is narrowed to that exact file and root, including patches carrying a recorded source root. Direct kernel-patch installation is refused outright: a custom kernel belongs in the optimization module and goes in through the normal patch path. Coordinator and specialist prompts carry the same boundary, so an agent is told the rule rather than discovering it by having a patch rejected. The declaration is checked once at launch, against `--framework-path`, so a campaign aimed at the wrong tree costs nothing instead of failing at its first patch.

A workload with no declaration is unaffected; everything above continues to apply to it.

### Optimize an unchanged script

`hyperloom script prepare` builds that workspace around an inference script you already have, whose only integration is importing the module first and passing its model through it once:

```python
from encoder_opt import apply   # first import: accepted settings land before torch reads them
import torch
...
model = apply(model)
```

```bash
hyperloom script prepare --workspace "$WS" --script encoder.py --python /path/to/model/env/python \
  --calibrate-autocast bfloat16 -- --frames 6   # or --error-limit 0.01; arguments after -- go to the script
```

It runs the script once with the identity module, records the calls of the object `apply` returns as references, and writes `benchmark/custom_<runner>.sh`. Each measurement then runs the script until `apply` returns, replays the recorded calls on fresh perturbed inputs with hidden held-out inputs mixed in, and compares outputs in a process that never imports the candidate. Your script needs no timing or validation code, and is never copied -- its checksum is pinned and rechecked on every measurement.

Preparation refuses a script that imports the module after the GPU is initialized or after `torch._inductor`/`torch._dynamo` are loaded, naming both lines, and warns if it imports it merely after `torch`. Settings baked into the module only reach torch if the import comes first, so an import too late means the deployed result would not behave like the measured one.

`--calibrate-autocast bfloat16` sets the tolerance from the model itself: "allow as much extra error as bfloat16 costs this model", measured on your model rather than guessed. `--error-limit` sets it directly instead.

#### Judging an output whose shape changes

By default two outputs are compared element by element, which needs them to have the same shape. Many models do not: anything that generates text stops at a different length, a detector returns however many objects passed its threshold, a transcriber returns a different number of words. A candidate that is only slightly different can produce one row fewer, and the default comparison then reports an infinite error -- not because the answer is bad, but because it cannot line the two up at all.

Pass `--scorer my_scorer.py` to judge such a workload your own way. The file defines one required function and one optional one:

```python
def score(candidate, reference):
    # How wrong the candidate is. Any number, lower being better.
    return word_error_rate(candidate["text"], reference["text"])


def capture(output):
    # Optional: turn one model output into tensors the comparison can load.
    return {"text": tokens_of(output)}
```

`capture` runs next to the model and decides what is worth keeping -- the number of detections, the transcript, whatever your `score` reads. `score` runs in the separate process that never loads the candidate, so a candidate cannot soften its own mark. Whatever units `score` returns are the units of `--error-limit` and of `--calibrate-autocast`.

The file is copied into the frozen benchmark directory and its checksum recorded, so editing it after preparation stops the run rather than quietly changing what passes. Everything else is unchanged: the clock, the hidden held-out inputs, the input perturbation and the output-buffer check all still apply.

## Use the optimized model without Hyperloom

On CLOSE, a custom session exports what it accepted. A single-file workload produces `deployment/hyperloom_optimize.py` with bookkeeping in `reports/deployment.json` (`kind: single_file`); a workload that declares an adapter through `hyperloom_inference.json` produces a self-contained `deployment/` with its own `inference.py` CLI, its source and assets captured, and runtime paths relocated so the directory can be moved.

Copy the single module beside the original script and run that script in its original runtime. No Hyperloom installation is needed at the destination.

The accepted runtime settings travel inside the module. `reports/deployment.json` records what was carried under `settings`: what became a torch API call, what stayed an environment variable, what was deliberately excluded, and what could not be carried at all. An absolute cache path is in the last group -- it is not portable to another machine -- and surfaces as a `settings_not_carried` warning rather than being dropped silently.

The export is checked against what was benchmarked, and refuses rather than shipping something unproven. The module is compared byte for byte against the source snapshots recorded at each KEEP, so it is refused if it differs from what was measured, if no snapshot covers it at all, or if it imports a file sitting beside it -- only the one module ships, so a sibling import would arrive broken. None of this requires the file to live in a git repository.

A refusal is recorded with its reason on the CLOSE step and written into the session package, so the artifacts never imply an export that did not happen.

`deployment.json` always reports `"validation": "not_run"`, and it can never say anything else. The standalone `--validate` command writes its result to a separate file stamped with a checksum of `deployment.json`; putting the result inside `deployment.json` would change that checksum, so the result could no longer point at the thing it checked. A single-file export has no such check to run anyway -- it ships one module, with no `inference.py` or adapter.

What stands behind the module is everything that happened before it was written out: every accepted change had to pass your own quality gate, and the module is compared against the copy saved at the moment it was accepted. Do not read the field as a check that failed.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| Missing-path error at launch | `--framework-path` or `--benchmark-scripts-dir` not given; `custom` has no shipped entrypoint to fall back on |
| Entrypoint not resolved | The directory has no `custom_<gpu-type>.sh` and does not contain exactly one `.sh` |
| Run stops after the baseline with `baseline_accuracy_failed` | The report carries no `quality_gate`; the stop line names the report |
| Whole budget spent in `PRELUDE` | Credentials not exported from the launching shell; every orchestration turn returns 401 |
| Specialist patches never land | `--framework-path` not given, so PolicyGate does not recognize the tree |
| Baseline higher than expected | `--extra-env` pins optimization switches, so the search is incrementing on already-optimized code |

For failures unrelated to this path, see [Troubleshooting
Hyperloom](../reference/troubleshooting.md).
