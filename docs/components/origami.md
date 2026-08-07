---
myst:
    html_meta:
        "description": "Learn how Hyperloom uses Origami to select and validate AITER A8W8 blockscale fallback kernels before GEAK or Forge tuning."
        "keywords": "Origami, Hyperloom, AITER, A8W8, blockscale, GEMM, kernel selection, ROCm, Composable Kernel"
---
# Origami

[Origami](https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami)
is ROCm's analytical GEMM solution-selection library. Hyperloom uses its Python
API as an optional pre-tuner for plain AITER A8W8 blockscale GEMMs.

The integration is explicit opt-in and does not modify Origami or AITER source.
When disabled, Hyperloom does not import Origami, resolve shapes, create an
Origami workspace, launch the selector, inject an AITER config, or emit Origami
telemetry.

- **Source**: <https://github.com/ROCm/rocm-libraries/tree/develop/shared/origami>
- **License**: MIT

## Role in Hyperloom

Origami runs inside the programmatic `run_gemm_tuning` path before the selected
GEAK or Forge tuner:

```text
runtime profile shapes
  -> AITER config-miss detection
  -> Origami template ranking
  -> selected-vs-default paired GPU benchmark
  -> measured winning rows merged into AITER config
  -> GEAK or Forge tuning
```

Origami never becomes the authoritative tuning backend. GEAK or Forge still
runs afterward and owns the normal end-to-end KEEP/REVERT decision.

The implementation is split across:

- `src/hyperloom/orchestrator/kernel/request_handlers.py`: feature gate,
  shape handoff, subprocess invocation, and tuner environment injection.
- `src/hyperloom/agents/kernel/tools/origami_gemm_select.py`: AITER provenance,
  Origami ranking, paired benchmark, correctness check, and CSV generation.
- `src/hyperloom/agents/kernel/skills/origami-gemm/SKILL.md`: executable
  workflow and safety contract.

## Enable the integration

Set the master feature gate before running the kernel-agent installer:

```bash
export HYPERLOOM_ORIGAMI_GEMM_FALLBACK=1
bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
source "${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh"
```

Unset, `0`, `false`, `no`, and `off` disable the complete integration.

## Installation

When enabled, the kernel-agent installer:

1. Resolves the pinned `ORIGAMI_REF`.
2. Creates a partial, sparse checkout of `shared/origami` under
   `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/rocm-libraries@<sha>`.
3. Builds and installs `shared/origami/python` with the active Python and ROCm
   toolchain.
4. Verifies `import origami`, `config_t`, `compute_total_latency`, and
   `get_hardware_for_device`.
5. Writes the resolved `ORIGAMI_ROOT` and `ORIGAMI_REF` to
   `kernel-agent.env.sh`.

The managed checkout is revision-keyed and idempotent. Re-running the installer
realigns the checkout to the pin and skips the Python reinstall when the active
package already came from that source directory.

Origami's Python extension requires:

- ROCm/HIP, normally under `${ROCM_PATH:-/opt/rocm}`
- CMake 3.25 or newer
- A C++17 compiler
- Python development headers

The build backend installs its Python build requirements through pip.

### Use an existing checkout

Set `ORIGAMI_ROOT` to the `shared/origami` directory, not the
`rocm-libraries` repository root:

```bash
export HYPERLOOM_ORIGAMI_GEMM_FALLBACK=1
export ORIGAMI_ROOT=/path/to/rocm-libraries/shared/origami
bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
```

The installer validates `$ORIGAMI_ROOT/python/pyproject.toml`, installs that
checkout without moving it, and records its Git revision when available.

Use `--check-only` to verify an enabled installation without cloning or
installing, or `--dry-run` to print the managed clone/build actions.

## Fallback detection

For each observed `(M,N,K)` shape, Hyperloom calls AITER's real
`get_CKGEMM_config()` resolver against the active blockscale CSV.

- No row or an empty `kernelName`: true fallback; eligible for Origami.
- A non-empty row selecting kernel ID 7: explicit CSV decision; preserve it.
- Any other configured row: preserve it.
- Invalid CSV data: fail closed.

The launched kernel symbol alone cannot distinguish a fallback from an explicit
CSV selection, so the decision is made before dispatch.

## Selection and benchmark gate

For true fallback shapes, the selector:

1. Loads AITER's 19 plain blockscale CK templates.
2. Maps each template's macro-tile and MFMA geometry into a baseline Origami
   config with occupancy 2 and splitK 0.
3. Ranks feasible templates using Origami's predicted total latency.
4. Preserves the default immediately if Origami selects kernel ID 7.
5. Otherwise benchmarks the selected template against AITER's empty-name
   default using shared inputs and separate outputs.

The paired benchmark warms both kernels, alternates timing order across rounds,
uses GPU events, compares median per-launch latency, and validates sampled
output correctness. Hyperloom emits an override row only when the selected
kernel is correct and strictly faster than the default.

Ties, regressions, unsupported kernels, OOMs, timing failures, or correctness
failures preserve AITER's default.

## Generated artifacts

Artifacts live under the GEMM-tuning workspace's `origami/` directory:

- `origami_a8w8_blockscale.csv`: measured winning fallback rows only.
- `origami_a8w8_blockscale_merged.csv`: complete active AITER configuration
  plus those winning rows.
- `origami_a8w8_blockscale_report.json`: dispatch provenance, ranking,
  benchmark latency, correctness, speedup, and skip reasons.

When at least one measured winner exists, Hyperloom passes the merged file to
the configured tuner through the existing
`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` environment variable.

## Runtime logging

The optimizer log emits stable markers that can be monitored without parsing
the full selector JSON:

- `ORIGAMI_GEMM_START`: selector invocation, workload type, shape source, and
  workspace.
- `ORIGAMI_GEMM_SUMMARY`: observed, fallback, benchmarked, and selected counts,
  plus the report path.
- `ORIGAMI_GEMM_WIN`: exact shape, selected kernel ID, paired median timings,
  and measured speedup.
- `ORIGAMI_GEMM_INJECT`: merged AITER CSV injected before GEAK/Forge.
- `ORIGAMI_GEMM_BACKEND`: authoritative backend continuation and decision.
- `ORIGAMI_GEMM_SKIP` / `ORIGAMI_GEMM_ERROR`: fail-closed reason.

For an overnight run:

```bash
grep -E 'ORIGAMI_GEMM_(START|SUMMARY|WIN|INJECT|BACKEND|SKIP|ERROR)' run.log
```

Detailed provenance and every per-shape decision remain in
`origami_a8w8_blockscale_report.json`; logs intentionally omit tensors,
credentials, and the full ranking payload.

## Safety properties

- Default-off and zero-touch when disabled.
- No source modifications to Origami or AITER.
- Existing AITER CSV choices always win over fallback selection.
- Direct measurement, not a learned size threshold, decides adoption.
- Scope is plain FP8 A8W8 blockscale only; bpreshuffle, per-token A8W8, FP4,
  and non-CK paths are excluded.
- The first profile used to discover shapes can still execute AITER's default;
  the generated overlay affects subsequent launches.

## Verification

Focused tests cover installer gating and pinning, local checkout overrides,
environment persistence, AITER provenance, Origami selection, paired benchmark
decisions, CSV merging, and GEAK/Forge precedence:

```bash
python -m pytest -q \
  src/hyperloom/agents/kernel/tests/test_install_origami.py \
  src/hyperloom/agents/kernel/tests/test_origami_gemm_select.py \
  src/hyperloom/inference_optimizer/tests/test_origami_gemm_handler.py
```
