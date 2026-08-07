---
myst:
    html_meta:
        "description": "How to optimize your own workload with Hyperloom using --framework custom. Covers the code checkout, the entrypoint contract, the mandatory quality gate, and pinning knobs with --extra-env."
        "keywords": "Hyperloom, custom framework, scriptable workload, quality gate, AMD GPU, ROCm, framework rewrite, host probe, extra-env, optimization"
---
# Optimize your own workload

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
export HYPERLOOM_BENCHMARK_BACKEND=bypass
python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework custom \
  --framework-path /path/to/my-checkout \
  --benchmark-scripts-dir /path/to/my-scripts \
  --model /path/to/weights \
  --gpu-type mi355x --tp 8 --max-hours 12 \
  --extra-env MYFW_STEPS=50 --extra-env MYFW_CKPT=/path/to/ckpt
```

```{important}
`HYPERLOOM_BENCHMARK_BACKEND=bypass` is required. The backend defaults to
Magpie, which cannot run a scriptable framework, so omitting it fails at launch
with `Unsupported framework`. This is easy to lose when you move to a fresh
shell, and it is the most common launch failure on this path.
```

Neither `--framework-path` nor `--benchmark-scripts-dir` is optional here. With
no shipped entrypoint to fall back on, a missing path fails at launch rather
than at the first benchmark.

## The code checkout

`--framework-path` is the **code** checkout, which is not the same thing as the
weights under `--model`. It does more than tell your script where the code
lives: it registers the tree as a framework source root, and PolicyGate requires
that registration before any specialist patch against your code can land. The
source probe discovers pip-installed packages on its own but never a git
checkout, which is why this must be explicit.

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

1. `custom_<runner_type>.sh` — with `--gpu-type mi355x`, that is `custom_mi355x.sh`.
2. Failing that, the **single** `.sh` file in the directory, which is what an
   operator who wrote one script for one machine actually has.
3. If neither applies, the entrypoint is unresolved and the run reports the miss.

Your script is invoked with `RESULT_DIR` and `RESULT_FILENAME=inferencex_result`
injected into its environment. It must write an InferenceX-shaped report to
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
A missing or unparseable `quality_gate` scores 0.0 accuracy, which **rejects
every candidate** the run produces. The symptom is a run that completes, reports
speedups, and keeps nothing.
```

Run your entrypoint by hand once and confirm the key is actually present in
`inferencex_result.json` before you spend a budget on it.

What the gate contains is yours to decide, because only you know what
correctness means for your workload. For a chaotic generative pipeline, a fixed
similarity threshold is usually the wrong bar — a numerically faithful kernel
that is not bit-identical still drifts over many steps. A self-calibrating band
(measure the pipeline's own drift under a tiny perturbation, widen it by a
margin, then require later legs to stay inside it) is the pattern that holds up.

## Pinning knobs with `--extra-env`

`--extra-env` carries the knobs your script reads. Hyperloom interprets none of
them, but their semantics matter:

> Whatever you pin becomes part of the **measurement contract**. A variant may
> add keys but may not overwrite a pinned one, because the baseline number was
> measured with it.

So pin what must not move, and leave the rest for exploration. This is also what
decides your baseline. Given one code tree and one entrypoint, pinning a set of
already-validated optimization switches makes the baseline the optimized number
and asks the search for an increment on existing work; pinning none of them
makes the baseline the stock path, so every switch the search finds is its own
discovery.

The CLI serializes these pins into `INFERENCE_OPTIMIZER_EXTRA_ENV` as a JSON
object. **Forward every pin as its own `--extra-env` flag** — a dropped pin is
lost silently.

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
catalogue in
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

**GPU type.** `--gpu-type` accepts `mi300x`, `mi308x`, `mi325x` and `mi355x`.
Note that **`mi308x` and `mi325x` map to `runner_type=mi300x`** with a warning,
since those GPUs share the MI300X runner family. Without the flag, Hyperloom
auto-detects through `rocm-smi --showproductname`.

**Pin GPUs with `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.** On the known
ROCm stack the latter can make `torch.cuda.is_available()` return false.

**Export credentials from the launching shell.** Preflight refuses to load
`*_CUSTOM_HEADERS` out of `.env`, so a gateway subscription key only reaches the
SDK if the launching shell exports it. Without it every catalog probe and every
orchestration turn returns HTTP 401 and the run idles in `PRELUDE` for the whole
budget. See [Authentication and credentials](../reference/authentication.md).

```bash
set -a
. <(grep -E '^(ANTHROPIC|OPENAI)_(CUSTOM_HEADERS|API_KEY|BASE_URL)=' "$REPO_ROOT/.env")
set +a
```

**Sessions.** `USER_DATA_PATH` sets the session root, and each `optimize`
creates a new timestamped subdirectory under it. Use `--resume` to continue an
existing session, optionally with `--resume-from <subdir>`; `--force-resume`
pushes past the terminal-state guard. Without `--resume` you always get a fresh
session, so an interrupted run is never picked up by accident.

## Monitor the run and read the output

The benchmark output does not go to the launcher log; it lands in the session
directory:

```text
<session>/runs/baseline/<id>/benchmark_custom_<stamp>/scriptable_stdout.log
<session>/runs/baseline/<id>/benchmark_custom_<stamp>/inferencex_result.json
<session>/reports/optimization_journal.json
```

Check the baseline `output_throughput` first. If a pin is wrong, that number
shows it immediately, instead of after hours of search against the wrong
denominator. For the full artifact schema, see
[`session_breakdown.json` integration in Hyperloom](../reference/session-breakdown.md).

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `Unsupported framework` at launch | `HYPERLOOM_BENCHMARK_BACKEND=bypass` not set |
| Missing-path error at launch | `--framework-path` or `--benchmark-scripts-dir` not given; `custom` has no shipped entrypoint to fall back on |
| Entrypoint not resolved | The directory has no `custom_<runner_type>.sh` and does not contain exactly one `.sh` |
| Run completes but keeps no optimization | The report carries no `quality_gate`, so accuracy scores 0.0 and every candidate is rejected |
| Whole budget spent in `PRELUDE` | Credentials not exported from the launching shell; every orchestration turn returns 401 |
| Specialist patches never land | `--framework-path` not given, so PolicyGate does not recognize the tree |
| Baseline higher than expected | `--extra-env` pins optimization switches, so the search is incrementing on already-optimized code |

For failures unrelated to this path, see [Troubleshooting
Hyperloom](../reference/troubleshooting.md).
