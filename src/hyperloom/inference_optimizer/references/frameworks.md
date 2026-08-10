# Framework Selection & GPU Runner Type

## Framework Selection

A session is single-framework. Pick `sglang` (default), `vllm`, `atom`, `xdit`
or `custom` via `--framework` or `$FRAMEWORK`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
python3 -m hyperloom.inference_optimizer.cli optimize --framework atom --model "$MODEL_PATH" --max-hours 2  # IR-8 single-node only
python3 -m hyperloom.inference_optimizer.cli optimize --framework xdit --model "$MODEL_PATH" --max-hours 2  # scriptable diffusion
python3 -m hyperloom.inference_optimizer.cli optimize --framework custom --model "$MODEL_PATH" --max-hours 12  # your own scriptable workload
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to —
  `baseline_{sglang,vllm,atom,xdit,custom}.yaml` and
  `profile_{sglang,vllm,atom,xdit,custom}.yaml`. The per-framework resolver
  `_default_profile_config()` in `src/hyperloom/orchestrator/actions/executors/profile.py` picks the right
  file from `$FRAMEWORK`.
- Which framework-specific seed grid the `explore` action falls back to when no
  `params.grid` is supplied. atom and xdit are the frameworks with programmatic
  seeds today (`_default_grid_for_framework` in
  `src/hyperloom/orchestrator/actions/executors/explore.py` dispatches to
  `_atom_default_grid()` / `_xdit_default_grid()`); sglang
  and vllm continue to rely on the orchestration LLM emitting
  `provenance='default_grid'` variants and will fail with
  `error_class="empty_grid"` on a cold-start with no LLM input.
- Which extra-args env name `_grid_runner` writes (`EXTRA_VLLM_ARGS` /
  `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS` / `EXTRA_XDIT_ARGS` /
  `EXTRA_CUSTOM_ARGS`).
- Which KB partition orchestration reads for hints.

Mixing frameworks in a single session is not supported; the CLI locks
`$FRAMEWORK` for the run. Resume re-reads `$FRAMEWORK` from the shell — set it
when you resume a non-default session.

### `--framework atom` specifics (IR-8)

Single-node only (`--nodes>=2` fails fast). Shipped configs `baseline_atom.yaml`
/ `profile_atom.yaml`; the Magpie atom wrapper bridges `PROFILE=1` to atom's
`--torch-profiler-dir`, and TraceLens consumes the resulting
`*.pt.trace.json.gz` unchanged. atom source roots (`/app/ATOM/atom/`) are in
PolicyGate's allowlist + `_REUSABLE_SOURCE_ROOTS`, and the repo URL
`https://github.com/ROCm/ATOM.git` is in `hyperloom.agents.framework.repo_map`. Unlike
sglang/vllm, atom ships a programmatic cold-start seed grid
(`_atom_default_grid`: `atom_level_{2,3}`, `atom_prefix_cache`, `atom_kv_fp8` on
FP8, model-class-gated `atom_ep` / `atom_dp_attn` / `atom_mtp_{1,3}`,
`atom_cudagraph_bracket`) — as does xdit (`_xdit_default_grid`), while
sglang/vllm fail `error_class="empty_grid"` on a
cold start with no LLM variants.

### `--framework custom` specifics (your own workload)

Every other entry in the registry describes a framework this repository knows:
its upstream, its entrypoint, the knobs worth exploring. `custom` describes none
of that, because the workload is yours. It is scriptable (server-less), and two
paths at launch replace everything the shipped frameworks hardcode:

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework custom \
  --framework-path /path/to/my-framework \
  --benchmark-scripts-dir /path/to/my-scripts \
  --gpu-type mi355x --model /path/to/weights --tp 8 --max-hours 12 \
  --extra-env MYFW_STEPS=50 --extra-env MYFW_CKPT=/path/to/ckpt
```

`--framework-path` is the **code** checkout, separate from the weights under
`--model`. It registers the tree as a framework source root, which PolicyGate
requires before any specialist patch against your code can land — the probe
finds pip-installed packages on its own but never a git checkout. Both flags are
friendlier spellings of `FRAMEWORK_REPO_PATH` and `HYPERLOOM_BYPASS_SCRIPTS_DIR`;
an exported value still wins. Neither is optional here: with no shipped
entrypoint to fall back on, a missing path fails at launch rather than at the
first benchmark.

The entrypoint is taken as `custom_<gpu-type>.sh`, or the single `.sh` in the
directory. It **must** emit a `quality_gate` block in its report: for a
server-less workload that gate is the only correctness signal, and a missing one
scores zero accuracy, which rejects every candidate the run produces.

`--extra-env` carries the knobs your script reads; Hyperloom interprets none of
them. Whatever you pin there becomes part of the measurement contract — a
variant may add keys but may not overwrite one, because the baseline number was
measured with it. Pin what must not move, and leave the rest for exploration.

### Naming the source tree without the framework prefix

`FRAMEWORK_REPO_PATH` is the framework-agnostic spelling of the same thing, and it
works for **any** framework rather than only the scriptable ones. Resolution order
is `<FRAMEWORK>_REPO_PATH` > `<FRAMEWORK>_DIR` > `FRAMEWORK_REPO_PATH`, so a
prefixed value keeps precedence and nothing existing changes behaviour.

Prefer the generic form. A session is single-framework by construction (the CLI
locks `$FRAMEWORK` for the run), so the prefix resolves no possible collision — it
only requires the operator to know the framework's name before setting the right
variable, and to change variable names when switching frameworks.

```bash
export FRAMEWORK_REPO_PATH=/path/to/checkout    # any framework
```

The prefixed and generic forms cover different situations by default rather than by
kind:

- **A pip-installed framework** (`sglang`, `vllm`, `atom`) needs neither. Its source
  root is discovered from the import machinery and the site-packages scan, which is
  why patching those has never required a path.
- **A scriptable framework** (`xdit`, `custom`) runs from a git
  checkout that neither mechanism can see, so one of the two variables is required.
- **An editable checkout of a normally-installed framework** — a `vllm` source tree
  you build yourself rather than the wheel — is equally invisible, and the generic
  variable is the supported way to point at it. Previously that case had to be
  handled by hand through `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`.

The FRAMEWORK phase works here (for a shipped framework the registry carries
the `repo_url`; for `custom` the checkout arrives at launch) and is the only phase that can restructure the pipeline itself —
sequence-parallel all-to-all, per-step host-to-device copies, repeated work
hoisted out of the AR rollout. `explore` cannot reach any of that: a variant is
only CLI flags plus env, and the accepted flag surface is four knobs. Pass
`--no-framework-agent` to skip it, but do not assume it is the default.

#### Framework-level source rewrites (the high-ceiling path)

Because a scriptable framework has no server, the FRAMEWORK phase's authoring arm dispatches
`framework_rewrite_specialist` rather than `serving_specialist`. The two share no
optimization surface: an AR video rollout has no scheduler, no continuous
batching and no KV-cache admission policy, and its wins are the redundant work
its loop structure creates. See `references/framework_rewrite_patterns.md` for
the pattern catalogue the specialist works from.

Evidence. A `profile` leg arms a host-side probe
(`assets/host_probe/hl_host_probe.py`, injected via a `PYTHONPATH` prefix so the
customer's entrypoint is untouched) and writes
`framework_rewrite_evidence.json` next to the workspace. It measures what the GPU
kernel breakdown structurally cannot: object collectives round-tripping through
the host, device-to-host syncs, repeated host-to-device copies, and — with
`HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE_DEEP=1` — per-function call counts with
argument-repeat rates that separate a memoization candidate from a loop-hoist
enabler. Tier 1 is on by default and cheap;
`HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE=0` disables the probe entirely. The deep
tier inflates host time enough to skew a co-collected torch trace, so give it its
own leg.

Deliverable. Every rewrite must sit behind its own environment switch that
defaults OFF, declared in a `framework_switches` manifest with `category`,
`target`, `depends_on` and `enables`. That discipline buys three things:

- a **switch-off parity leg** runs with every switch unset and must reproduce the
  base within ±2%, so a patch that is not genuinely inert when disabled is
  reverted rather than silently poisoning every later measurement;
- a bundle that passes correctness but misses the throughput threshold is
  **kept inert** instead of reverted — default-off code costs nothing, and
  reverting would discard the rewrites that do pay along with the one that does
  not;
- accepted switches are registered as **search levers**, so `explore` measures
  each rewrite's own contribution (additive when the levers are dormant,
  leave-one-out when they are already on) and searches combinations, following
  the declared dependency closure so an enabler is never judged alone.

That last point is not a nicety. A hoist whose only value is making a downstream
cache hit measures flat on its own; a greedy accept/reject loop rejects it and
then measures every dependent rewrite against a permanently cold cache, losing
the bundle rather than the lever.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without either, the
optimizer auto-detects via `rocm-smi --showproductname` (falling back to
`torch.cuda.get_device_properties(0).gcnArchName`).

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi308x`, `mi325x`, `mi355x`. **`mi308x` and
`mi325x` map to `runner_type=mi300x`** with a warning, since the GPUs share the
same runner family and Magpie has not shipped MI308X/MI325X-specific SGLang/vLLM
scripts yet. If you need a true MI308X/MI325X-specific script, uncomment the `benchmark_script:` template in the
relevant YAML and point it at your script under `InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.
