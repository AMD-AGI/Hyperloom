---
myst:
  html_meta:
    "description": "KernelForge CLI reference: the kernelforge forge-loop, forge-rewrite-by-flydsl, forge-fuse and gemm-tune commands."
    "keywords": "KernelForge, CLI, kernelforge, forge-loop, forge-rewrite-by-flydsl, forge-fuse, gemm-tune, FlyDSL"
---

# CLI reference

KernelForge installs exactly one CLI, `kernelforge`; everything below is a
subcommand of it.

## Core

```bash
kernelforge forge-loop --workspace <W> --kernel <f> --driver <f> [options]
kernelforge forge-rewrite-by-flydsl --source-kernel <f> --driver <f> \
    --logical-op-name <op> --workspace <W> --experiments-dir <D> [options]
kernelforge forge-fuse --trace <t> --model-path <d> --framework sglang \
    --output-dir <d> [options]
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

| Option | Default | Meaning |
|:--|:--|:--|
| `--workspace <dir>` | required | Git workspace the campaign runs in. |
| `--kernel <file>` | none | Fresh campaign: the kernel file to optimize. |
| `--driver <file>` | none | Fresh campaign: the validation/bench driver. |
| `--resume` | off | Continue the campaign already stored in that exact workspace. |
| `--max-hours <h>` | `1.0` | Runtime budget in hours; the loop is time-driven. Minimum `1.0`. A round is started only when what remains can finish it, so the run ends before the budget does. |
| `--snr-threshold <dB>` | `30.0` | Fresh-campaign correctness gate, stored immutably; ignored on `--resume`. |
| `--kernel-backend <name>` | inferred | Fresh campaign: kernel backend override. Unsupported kernel backends fall back to `flydsl`. |
| `--gpu-target <arch>` | none | ROCm compilation architecture, e.g. `gfx950` (also exported to the environment). |
| `--gpu-type <sku>` | `mi355x` | Hardware SKU used in knowledge-base identities. |
| `--experiments-dir <dir>` | `<W>/forge_experiments` | Diagnostics and checkpoint root (profiles, optimization potential, tracker checkpoint). |
| `--lanes <n>` | `1` | Implementer lanes per round (1–8). Above 1 the round's analysis is partitioned into that many non-overlapping plans, each run in its own workspace copy and measured on its own. A round the remaining budget cannot plan at this width is narrowed one lane at a time before it is refused. |
| `--prepare-task` / `--no-prepare-task` | on | Pre-loop preparation of the measurement driver against the loop's stdout contract. Fresh campaigns only. |
| `--agent-backend <name>` | provider default | Registered local Agent provider for the Implementer. |
| `--supervisor-backend <name>` | follows Implementer | Registered provider for the stalled-search supervisor. |
| `--result-json <file>` | none | Write the result dict here as well as printing it. |

Stop a running campaign with `touch <workspace>/.stop`; the loop checks for that
file at the next iteration boundary.

## forge-rewrite-by-flydsl

Ports a source kernel (Triton, HIP, CUDA or C++) into an equivalent FlyDSL
kernel in a correctness-only PORT phase, then hands the FlyDSL kernel to
`forge-loop` for optimization. With an existing framework git base, the final
20 minutes are reserved for one session that turns the verified best FlyDSL
kernel into a cumulative framework apply-back patch. The driver uses the original
kernel as a live correctness oracle and baseline, so this works for any
operator. The result (`source_ms`, `flydsl_best_ms`, `speedup`, `correct`) uses
the same `__FORGE_RESULT__` contract as `forge-loop`.

| Option | Default | Meaning |
|:--|:--|:--|
| `--source-kernel <file>` | required | The kernel to rewrite (a Triton `.py`, a `.hip`, …). |
| `--driver <file>` | required | Rewrite measurement driver. A conforming driver is used unchanged. |
| `--logical-op-name <name>` | required | Stable logical identity of the workload; the FlyDSL factory symbol is derived from it and reported in the result. |
| `--workspace <dir>` | required | Git workspace directory. |
| `--experiments-dir <dir>` | required | Where to write `forge_experiments`. |
| `--source-entry <fn>` | auto | Host callable in the source that runs the kernel, used as oracle and baseline. |
| `--target-functions <a,b>` | none | Source kernel entry names (the `@triton.jit` name, or the `__global__` function name). |
| `--source-language <lang>` | inferred | One of the source languages reported by `--capabilities-json`. |
| `--shapes-json <json>` | `[]` | JSON list of `{M,N,dtype}` shapes driving correctness and benchmarking. |
| `--framework <name>` | inferred | Apply-back target: `aiter`, `vllm` or `sglang`. |
| `--snr-threshold <dB>` | `30.0` | Correctness gate for the ported kernel. |
| `--max-hours <h>` | `1.0` | Total rewrite budget across PORT, OPTIMIZE and apply-back. Minimum `1.0`. |
| `--prepare-driver` / `--no-prepare-driver` | on | Author or repair a non-conforming dual-path rewrite driver before PORT. |
| `--capabilities-json` | — | Print the machine-readable rewrite capability handshake and exit. |

## forge-fuse

Diagnoses a decode trace, locates a launch-bound chain of small kernels, and
authors one fused Triton kernel that survives CUDA-graph capture, A/B-validated
against the framework's own eager op.

| Option | Purpose |
|:-------|:--------|
| `--trace` | Decode kineto trace, captured with CUDA graphs disabled. |
| `--model-path` | Model directory (must contain `config.json`). |
| `--framework` | `sglang`, `vllm` or `vllm-aiter`. |
| `--output-dir` | Manifest and logs. |
| `--discover` | `patterns` (template library) or `llm` (reads trace + source). |
| `--dry-run` | Diagnose and locate only; emit a recipe skeleton. |
| `--framework-root` | Explicit framework source root, else auto-detected. |

Writes `fusion_manifest.json` and exits 3 when no fusion is found.

## GEMM tuning

```bash
kernelforge gemm-tune run --model-path <M> --framework sglang|vllm|vllm-aiter \
    --precision <p> --output-dir <D>       # Tune vendor GEMM libraries for a model
kernelforge gemm-tune plan --model-path <M> --framework <F> --precision <p>
                                           # Show which tuners would run, without running them
```

It reads no knowledge base: every run tunes from
scratch and writes only its own output directory.
