---
myst:
  html_meta:
    "description": "KernelForge CLI reference: the kernelforge forge-loop, forge-rewrite-by-flydsl, forge-fuse and gemm-tune commands."
    "keywords": "KernelForge, CLI, kernelforge, forge-loop, forge-rewrite-by-flydsl, forge-fuse, gemm-tune, FlyDSL"
---

# CLI reference

KernelForge installs exactly one CLI, `kernelforge`; everything below is a
subcommand of it.

Every option each subcommand accepts is listed here. That is a contract, not a
courtesy: an option no subcommand declares is a usage error, and the run exits 2
during argument parsing before any GPU work starts. Nothing is dropped, ignored,
or absorbed into a default, so a misspelled flag costs a message and an exit
code rather than an hour of tuning against a value you did not ask for.

One option lives on `kernelforge` itself rather than on any subcommand:

| Option | Meaning |
|:--|:--|
| `--version` | Print the installed KernelForge version and exit. |

## Core

```bash
kernelforge forge-loop --workspace <W> --kernel <f> --driver <f> [options]
kernelforge forge-rewrite-by-flydsl --source-kernel <f> --driver <f> \
    --logical-op-name <op> --workspace <W> --experiments-dir <D> [options]
kernelforge forge-fuse --trace <t> --model-path <d> --framework sglang \
    --output-dir <d> [options]
kernelforge gemm-tune run --model-path <M> --framework sglang \
    --precision <p> --output-dir <D> [options]
```

See {doc}`Experience store </kernelforge/reference/experience-store>` for the exact
local/remote environment contract and durable local layout.

## forge-loop

Runs one measurement-driven optimization campaign over a single kernel:
baseline → agent → validate → bench → keep. The campaign is resumable — its
immutable inputs are snapshotted into
`<workspace>/forge_experiments/campaign_config.json` and the control state into
`run_state.json`. The result dict (`baseline_ms`, `best_ms`,
`mean_case_speedup`, `improved`, `experiment_id`, `iteration_count`) is printed
to stdout wrapped in `__FORGE_RESULT__` sentinels.

Stop a running campaign with `touch <workspace>/.stop`; the loop checks for that
file at the next iteration boundary.

### Campaign inputs

Everything in this group except `--workspace` and `--resume` belongs to a fresh
campaign. Once the campaign config is snapshotted these values are immutable,
and passing one alongside `--resume` is refused rather than silently ignored.

| Option | Default | Meaning |
|:--|:--|:--|
| `--workspace <dir>` | required | Git workspace the campaign runs in. |
| `--resume` | off | Continue the campaign already stored in that exact workspace. |
| `--kernel <file>` | none | The kernel file to optimize. This is the anchor the loop edits. |
| `--driver <file>` | none | The validation/bench driver. |
| `--auto` | off | Pick the kernel here instead of being handed one. Requires `--nomination-input`, refuses `--kernel` and `--resume`, and makes the result carry a `patches` array plus nomination counts. Off by default, so a run without it is unchanged. |
| `--nomination-input <file>` | `''` | Nomination request JSON: raw trace path, candidate list path, lane budget and target ceiling. Read only under `--auto`. |
| `--git-branch <name>` | none | Development branch to optimize on, checked out before the campaign config is snapshotted. |
| `--program-md-file <file>` | none | Optional task context copied into the campaign. |
| `--invocation-spec-file <file>` | none | Hyperloom invocation-spec JSON used by task preparation. |
| `--task-type <name>` | `''` | `repository` or `image_kernel` enable multi-file / whole-repo handling; any other value keeps single-file behavior. |
| `--source-files <a,b>` | `''` | Comma/newline-separated implementation entry points used for orientation, profiling, JIT hints and KB identity. Not an edit allowlist — `--kernel` remains the anchor. |
| `--target-functions <a,b>` | `''` | Comma-separated target kernel/function hints, used for PMC filtering, source mapping and agent orientation. Does not restrict which functions may be edited. |
| `--operator-name <name>` | `''` | Logical operator identity used by profiling and the experience page key. |
| `--framework <name>` | inferred | Framework identity for the experience KB slug: `vllm`, `sglang`, `aiter`, or `standalone` for a framework-less file. Authoritative when given. |
| `--kernel-backend <name>` | inferred | Kernel backend override. An unsupported backend falls back to `flydsl`. |
| `--commit-new-path <glob>` | none | Workspace-relative path or glob naming a file the agent may CREATE and still have committed with a KEEP. Repeatable. Untracked files are otherwise never staged and never removed by a REVERT. `*` does not cross a directory separator and `**` is rejected; name each level. Protected measurement paths are never admitted. Immutable per campaign. |
| `--snr-threshold <dB>` | `30.0` | SNR pre-filter threshold, stored immutably. A KEEP is decided by the task's own `correctness_command`, not by this value. |
| `--prepare-task` / `--no-prepare-task` | on | Pre-loop preflight of the driver against the loop's stdout contract; on failure one agent authors or repairs the measurement driver (never the kernel), then it is re-checked. Skipped on `--resume`. |

### Budget and deadlines

| Option | Default | Meaning |
|:--|:--|:--|
| `--max-hours <h>` | `1.0` | Runtime budget in hours; the loop is time-driven. Minimum `1.0`. A round is started only when what remains can finish it, so the run ends before the budget does. Above 2 hours this also enables Analysis profiling and, for single-lane rounds, Plan Critic review. |
| `--deadline-unix <t>` | `0` | Absolute UNIX deadline shared by task preparation and optimization. |
| `--session-timeout-sec <s>` | from `--max-hours` | Wall-clock budget for one implementer session. The claude backend cuts the session at this deadline and the session is told about it. |
| `--agent-timeout-sec <s>` | provider default | Timeout for one Agent session. |
| `--profile-timeout-sec <s>` | `7200` | Ceiling for the single complete Analysis Agent session. Phase and case artifacts are persisted for resume when the deadline is reached. |

### Hardware and measurement

| Option | Default | Meaning |
|:--|:--|:--|
| `--gpu-target <arch>` | none | ROCm compilation architecture, e.g. `gfx950`. Also exported to the environment. |
| `--gpu-type <sku>` | `mi355x` | Hardware SKU used in knowledge-base identities. |
| `--nproc-per-node <n>` | `1` | Ranks the driver self-launches via torchrun, for collective tasks such as all-reduce. Above 1, EVERY rank is profiled in its own rocprofv3 session — wrapping the driver would only profile the launcher process, which runs no kernel. |
| `--bench-repeat <n>` | `1` | How many times each bench repeats its measurement in-process, reporting the per-case median. Above 1 shrinks run-to-run spread and requires a driver that accepts `--repeat`; the flag is omitted entirely when this is 1. |
| `--aiter-cache-max-gb <g>` | `4.0` | Per-attempt AITER cache soft limit in GiB. LRU pruning targets 75% of the limit; `0` disables in-run pruning. |
| `--profiling` / `--no-profiling` | on | Allow Analysis hardware profiling and Implementer self-profiling guidance on long-horizon runs (>2 hours). Shorter runs keep Analysis static-only regardless. `--no-profiling` disables collection for every duration. |

### Rounds and lanes

| Option | Default | Meaning |
|:--|:--|:--|
| `--lanes <n>` | `3` | Implementer lanes per round (1–8). Above 1 the round's analysis is partitioned into that many non-overlapping plans, each run concurrently in its own workspace copy and measured on its own. Lanes run concurrently, so a lane costs a session rather than a share of the round's wall clock, and three is what the three specialist analyses divide into. The partition returns fewer when the evidence supports fewer. Needs a provider declaring `stop_hooks` and `session_env`; refused on one that does not. |
| `--merge-stacking` / `--no-merge-stacking` | on | Once consecutive iterations stop producing a new best, spend one iteration measuring two archived rejected gains applied together, chosen for winning on different cases. Costs a measurement but no Implementer session. Applies at every `--lanes` setting; turn it off to compare against a run that predates it. |
| `--specialist-probe` / `--no-specialist-probe` | on | Let the read-only planning specialists measure one variant per probe in a scratch tree instead of only arguing about a dispatch constant. Each probe re-runs the workspace driver for one case with declared constants overridden, queues on the same device lock the fan-out lanes take, and never touches the canonical tree. Falls back to `FORGE_SPECIALIST_PROBE`. |
| `--specialist-probe-max <n>` | `6` | Probes ONE analysis round may make in total, shared across every specialist it dispatches. Every call counts, including a refused one. Falls back to `FORGE_SPECIALIST_PROBE_MAX`. |
| `--specialist-probe-budget-sec <s>` | `600` | Seconds on the device ONE analysis round may spend probing, shared across its specialists, and cut down further at call time so no probe leaves a specialist without time to write its analysis. Falls back to `FORGE_SPECIALIST_PROBE_BUDGET_SEC`. |
| `--specialist-probe-scratch-root <dir>` | under `--experiments-dir` | Where round scratch trees are created. Must be absolute and outside the workspace; defaults to a sibling of the workspace when the default would land inside it. Falls back to `FORGE_SPECIALIST_PROBE_SCRATCH_ROOT`. |

### Agent provider

| Option | Default | Meaning |
|:--|:--|:--|
| `--agent-backend <name>` | provider default | Registered local Agent provider for the Implementer. |
| `--supervisor-backend <name>` | follows Implementer | Registered provider for the stalled-search (AVO) supervisor. Omit it so one `--agent-backend` controls every local agent. |
| `--agent-cli <path>` | provider default | Provider executable path or command. |
| `--model <name>` | provider default | Selected provider model. |
| `--agent-reasoning-effort <v>` | provider default | Provider reasoning effort value. |
| `--agent-sandbox-mode <v>` | provider default | Provider sandbox mode. |
| `--agent-fallback-provider <name>` | provider default | Fallback provider name, or `none` to refuse a fallback. |
| `--agent-precheck` / `--no-agent-precheck` | on | Provider preflight and capability probe. |
| `--agent-options-json <json>` | none | Provider extension options as a JSON object. |
| `--permission-mode <v>` | provider default | Provider permission mode, where the provider supports one. |

### Knowledge base

| Option | Default | Meaning |
|:--|:--|:--|
| `--experience-kb` / `--no-experience-kb` | on | Read and publish forge-loop experience KB entries. Internal callers with their own KB lifecycle, such as `forge-rewrite-by-flydsl`, disable this explicitly. |
| `--kb-warmstart` / `--no-kb-warmstart` | on | Apply the best matching KB solution before iteration 1. Only meaningful with `--experience-kb`. A caller that prepares the workspace itself, such as `forge-fuse`, turns this off to keep publishing while never replaying a stored patch over a tree it already staged. |
| `--return-after-read-kb` / `--return-after-read-KB` | off | Return before iteration 1 when a KB solution applies cleanly, passes current correctness and improves current performance. |
| `--producer <name>` | forge-loop's own | System owning the candidate stream these records belong to. A producer has its own index in the KB identity scheme, so a pipeline driving this command as a subprocess keeps its records out of the kernel campaigns' ranking. |
| `--pr-kb` / `--no-pr-kb` | off | Inject upstream pull-request references from the PR Monitor co-hosted at `${KB_STORE_URL}/pr-monitor/v1`. Falls back to `PR_KB_ENABLE`. |
| `--experience-id <id>` | `''` | Unique KB run identity, independent of the checkpoint experiment ID. |

### Output

| Option | Default | Meaning |
|:--|:--|:--|
| `--experiments-dir <dir>` | `<W>/forge_experiments` | Diagnostics and checkpoint root: profiles, optimization potential, tracker checkpoint. Resume artifacts always live under the workspace regardless. |
| `--experiment-id <id>` | none | Caller-owned experiment ID, recorded for external checkpoint recovery. |
| `--result-json <file>` | none | Write the result dict here as well as printing it. |

## forge-rewrite-by-flydsl

Ports a source kernel (Triton, HIP, CUDA or C++) into an equivalent FlyDSL
kernel in a correctness-only PORT phase, then hands the FlyDSL kernel to
`forge-loop` for optimization. With an existing framework git base, the final
20 minutes are reserved for one session that turns the verified best FlyDSL
kernel into a cumulative framework apply-back patch. The driver uses the original
kernel as a live correctness oracle and baseline, so this works for any
operator. The result (`source_ms`, `flydsl_best_ms`, `speedup`, `correct`) uses
the same `__FORGE_RESULT__` contract as `forge-loop`.

### Source and workspace

| Option | Default | Meaning |
|:--|:--|:--|
| `--source-kernel <file>` | required | The kernel to rewrite (a Triton `.py`, a `.hip`, …). |
| `--driver <file>` | required | Rewrite measurement driver. A conforming driver is used unchanged. |
| `--logical-op-name <name>` | required | Stable logical identity of the workload; a namespace or punctuation is allowed. The FlyDSL factory symbol is derived from it and reported in the result — never re-derive it downstream. `--op-name` is a deprecated alias. |
| `--workspace <dir>` | required | Git workspace directory. |
| `--experiments-dir <dir>` | required | Where to write `forge_experiments`. |
| `--source-entry <fn>` | auto | Host callable in the source that runs the kernel, used as the live oracle and baseline: `ref(x) -> y`. |
| `--source-language <lang>` | inferred | One of the source languages reported by `--capabilities-json`. Inferred from the file when omitted, but a caller whose profiler saw the kernel run should state it: a traced Triton kernel lives in a `.py` that names no language. |
| `--target-functions <a,b>` | `''` | Source kernel entry names: the `@triton.jit` name, or the `__global__` function name for HIP/CUDA. |
| `--shapes-json <json>` | `[]` | JSON list of `{M,N,dtype}` shapes driving correctness and benchmarking. |
| `--flydsl-kernel-name <file>` | `kernel.py` | Filename of the produced FlyDSL kernel in the workspace. |
| `--git-branch <name>` | `forge-rewrite-optimize` | Development branch used by the nested FlyDSL forge-loop. |
| `--invocation-spec-file <file>` | `''` | Invocation evidence JSON, used only by rewrite driver preparation. |
| `--prepare-driver` / `--no-prepare-driver` | on | Author or repair a non-conforming dual-path rewrite driver before PORT. |

### Budget and phases

| Option | Default | Meaning |
|:--|:--|:--|
| `--max-hours <h>` | `1.0` | Total rewrite budget across PORT, OPTIMIZE and apply-back. Minimum `1.0`. |
| `--deadline-unix <t>` | `0` | Absolute UNIX deadline for PORT, OPTIMIZE and apply-back finalization. |
| `--max-port-attempts <n>` | `3` | Correctness-only port sessions before giving up. |
| `--profile-timeout-sec <s>` | `3600` | OPTIMIZE: ceiling for the complete Analysis Agent workflow. |
| `--snr-threshold <dB>` | `30.0` | Correctness gate for the ported kernel. |

### Apply-back

| Option | Default | Meaning |
|:--|:--|:--|
| `--framework <name>` | inferred | Apply-back target: `aiter`, `vllm` or `sglang`. Inferred from the source path when omitted. |
| `--applyback-import-module <mod>` | inferred | Import target required to load before and after apply-back. Repeatable; defaults to the source module inferred from its package. |
| `--max-applyback-attempts <n>` | `2` | Maximum clean-room framework integration sessions. |

### Hardware, provider and output

| Option | Default | Meaning |
|:--|:--|:--|
| `--gpu-target <arch>` | none | ROCm compilation architecture, e.g. `gfx950`. Also exported to the environment. |
| `--gpu-type <sku>` | `mi355x` | Hardware SKU for rewrite KB identities. |
| `--model <name>` | provider default | LLM model; overrides `KERNEL_AGENTS_MODEL`. |
| `--permission-mode <v>` | `acceptEdits` | Claude permission mode. |
| `--supervisor-backend <name>` | `codex` | OPTIMIZE supervisor backend on stall: `codex` or `claude`. |
| `--rewrite-kb` / `--no-rewrite-kb` | on | Read and publish rewrite recipes. |
| `--result-json <file>` | none | Write the result dict here as well as printing it. |

### Handshakes

Both print to stdout and exit without running anything.

| Option | Meaning |
|:--|:--|
| `--capabilities-json` | The machine-readable rewrite capability handshake: protocol version, artifact schema, sentinel, frameworks, source languages and source kinds. Probe this before invoking, rather than inferring support from a run that failed. |
| `--applyback-contract-json` | Producer-authored apply-back manifest and outer-result examples. |

## forge-fuse

Diagnoses a decode trace, locates a launch-bound chain of small kernels, and
authors one fused Triton kernel that survives CUDA-graph capture, A/B-validated
against the framework's own eager op. Writes `fusion_manifest.json` and exits 3
when no fusion is found.

### Inputs

| Option | Default | Meaning |
|:--|:--|:--|
| `--trace <file>` | `''` | Decode kineto trace (`*.trace.json[.gz]`), captured with CUDA graphs disabled. |
| `--model-path <dir>` | `''` | Model directory; must contain `config.json`. |
| `--framework <name>` | none | `sglang`, `vllm` or `vllm-aiter`. |
| `--output-dir <dir>` | `''` | Manifest and logs. |
| `--framework-root <dir>` | auto-detect | Explicit framework source root, else the installed package is located. |
| `--decode-batch <n>` | `16` | Representative decode batch size (T) for shapes. |
| `--decode-steps <n>` | `0` | Decode steps captured in the trace, used to normalize kernels/step. |

### Discovery and authoring

| Option | Default | Meaning |
|:--|:--|:--|
| `--discover <mode>` | `patterns` | `patterns` (template library) or `llm` (the LLM reads trace and source, autonomous). |
| `--dry-run` | off | Diagnose and locate only; emit a manifest with a recipe skeleton, no authoring or validation. |
| `--author` / `--no-author` | on | Author the fused kernel via the LLM. Non-dry-run only. |
| `--fuse-all-confirmed` | off | Author ALL source-confirmed patterns together rather than only the top one, and A/B all their flags. A compile-pass candidate cannot be authored with them, so it is claimed alone and the rest wait for a later round. |
| `--agent-backend <name>` | `auto` | `auto`, `claude` or `codex`, for discovery and authoring. |
| `--agent-sandbox-mode <v>` | `workspace-write` | `workspace-write`, `read-only` or `bypass`. Use `bypass` only when an external sandbox already enforces isolation. |
| `--model <name>` | provider default | Agent model. An explicit value wins; otherwise `$CODEX_MODEL` / `$CLAUDE_MODEL`, then the registered provider default. |
| `--max-turns <n>` | `100` | Max authoring turns. |
| `--max-recipes <n>` | `0` | Cap how many ranked recipes to try. `0` means uncapped, so every discovered recipe is considered; Hyperloom passes the count its fusion lane budget pays for. |
| `--gpu <id>` | `0` | HIP device id for authoring and A/B. |
| `--gpu-target <arch>` | auto-detect | Canonical GPU arch the author writes for, e.g. `gfx950`; detected via `rocminfo` when omitted. |

### A/B validation and serving smoke

| Option | Default | Meaning |
|:--|:--|:--|
| `--validate` / `--no-validate` | on | Run the A/B decode validation. Non-dry-run only. |
| `--ab-isl <n>` | `512` | A/B input length. |
| `--ab-osl <n>` | `128` | A/B output length. |
| `--bench-extra <args>` | `''` | Extra `bench_one_batch` args, e.g. `--attention-backend triton`. |
| `--server-extra <args>` | `''` | Extra serving args for the smoke launch, e.g. `--kv-cache-dtype fp8`. A model whose engine refuses to start without a flag can never reach the kernel the smoke exists to exercise. |
| `--tp <n>` | `1` | Tensor-parallel size for the serving smoke; must match the session. |
| `--block-size <n>` | `0` (omit) | vLLM KV `--block-size` for the serving smoke. Required for sparse-attention models that reject the default block size. |
| `--max-model-len <n>` | `0` (smoke default 4096) | Serving-smoke max model / context length. |

### Diagnostics

| Option | Default | Meaning |
|:--|:--|:--|
| `--harness-noise <name>` | `''` | Repeat one kernel-validation harness and report its variance. |
| `--harness-noise-repeat <n>` | `20` | How many times to repeat that harness. |
| `--harness-noise-env <k=v>` | none | Env flag to set for each harness run. Repeatable. |
| `--verbose` / `-v` | off | Verbose logging. |
| `--version` | — | Print the version and exit. |

## gemm-tune

Tunes vendor GEMM libraries for one model. It reads no knowledge base: every run
tunes from scratch and writes only its own output directory.

```bash
kernelforge gemm-tune run --model-path <M> --framework sglang|vllm|vllm-aiter \
    --precision <p> --output-dir <D>       # Tune, and write the artifacts
kernelforge gemm-tune plan --model-path <M> --framework <F> --precision <p>
                                           # Show which tuners would run, without running them
kernelforge gemm-tune evidence <logs...> --out demand.json
                                           # Derive shape demand from serving logs
```

### gemm-tune run

Model and target:

| Option | Default | Meaning |
|:--|:--|:--|
| `--model-path <dir>` | required | Model directory; must contain `config.json`. |
| `--framework <name>` | required | `sglang`, `vllm` or `vllm-aiter` (vllm with `VLLM_ROCM_USE_AITER=1`). |
| `--precision <p>` | required | `bf16`, `fp8`, `fp4`, `int8` or `awq`. |
| `--quant-type <q>` | `auto` | `auto`, `none`, `per_token`, `blockscale`, `bpreshuffle`, `awq`, `gptq`, `fp4` or `mxfp4`. |
| `--gpu-type <sku>` | `auto` | `auto` detects via `rocminfo`; otherwise `mi300x`, `mi355x`, `gfx942`, … |
| `--output-dir <dir>` | required | Output directory for all artifacts. |

Shape sources, in descending priority — `--demand` wins, then
`--shapes-manifest`, then `--shapes-json`:

| Option | Default | Meaning |
|:--|:--|:--|
| `--demand <file>` | `''` | `demand.json` from `kernelforge gemm-tune evidence`; the highest-priority shape source. |
| `--shapes-manifest <file>` | `''` | Weighted `TraceShapeManifest` JSON; the preferred dense-shape source when set. |
| `--shapes-json <file>` | `''` | Shapes JSON from TraceLens or Hyperloom. |
| `--untuned-csv <file>` | `''` | Input untuned CSV for the dense aiter tuners. |
| `--moe-untuned-csv <file>` | `''` | Input untuned CSV for the MoE tuner, keyed on the tuple aiter dispatched at run time. |
| `--tunableop-input <file>` | `''` | PyTorch TunableOp shape file. |
| `--kernel-signature-log <file>` | `''` | Server log for 1-stage ASM detection. |
| `--tokens <a,b>` | auto | Comma-separated explicit token list, overriding the automatic coverage. |
| `--conc <n>` | `64` | Target serving concurrency, used for token coverage. |
| `--tp <n>` | `1` | Tensor-parallel degree. |

Search and execution:

| Option | Default | Meaning |
|:--|:--|:--|
| `--tuner <name>` | routed | Force a specific tuner, skipping routing. |
| `--thorough` | off | Full search space: all libtypes, more shapes, no per-shape timeout. Slower, but finds the absolute best config. |
| `--iters <n>` | `80` | Benchmark iterations per config. |
| `--warmup <n>` | `20` | Warmup iterations. |
| `--min-improvement-pct <p>` | `3.0` | Minimum improvement threshold, in percent. |
| `--timeout <s>` | `10800` | Per-tuner timeout. The first run includes JIT compilation. |
| `--global-timeout <s>` | `0` (unlimited) | Timeout for the entire session. |
| `--mp <n>` | `1` | Number of GPUs for parallel tuning. |
| `--gpu-ids <a,b>` | all | Comma-separated GPU IDs to use. |
| `--skip-gpu-check` | off | Skip the `rocm-smi` preflight check. |
| `--kb-current-lib <v>` | `''` | Current backend `lib_version`, recorded as artifact provenance. |
| `--verbose` / `-v` | off | Verbose logging. |

### gemm-tune plan

Reports which tuners `run` would dispatch, and on which shapes, without tuning
anything. It takes the model, target and shape-source options of `run` and none
of its search or execution options.

| Option | Default | Meaning |
|:--|:--|:--|
| `--model-path <dir>` | required | Model directory; must contain `config.json`. |
| `--framework <name>` | required | `sglang`, `vllm` or `vllm-aiter`. |
| `--precision <p>` | required | `bf16`, `fp8`, `fp4`, `int8` or `awq`. |
| `--quant-type <q>` | `auto` | As for `run`. |
| `--gpu-type <sku>` | `auto` | As for `run`. |
| `--demand <file>` | `''` | As for `run`; the highest-priority shape source. |
| `--shapes-manifest <file>` | `''` | As for `run`. |
| `--shapes-json <file>` | `''` | As for `run`. |
| `--untuned-csv <file>` | `''` | As for `run`. |
| `--tunableop-input <file>` | `''` | As for `run`. |
| `--kernel-signature-log <file>` | `''` | As for `run`. |

### gemm-tune evidence

Reads serving logs and derives the GEMM shape demand they witness, which `run`
and `plan` then consume as their highest-priority shape source.

| Argument / option | Default | Meaning |
|:--|:--|:--|
| `<logs...>` | required | One or more serving logs to read. |
| `--out <file>` | stdout summary only | Write `demand.json` here. |
| `--verbose` / `-v` | off | Verbose logging. |
