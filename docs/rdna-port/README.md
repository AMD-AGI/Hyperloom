# RDNA support for Hyperloom (gfx12 / RDNA4)

Community port. **Not an AMD product and not endorsed by AMD** — this is a fork
of [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom) (MIT, © AMD)
adding consumer/workstation Radeon support. Upstream tracks this as
[issue #1196](https://github.com/AMD-AGI/Hyperloom/issues/1196).

Upstream validates against Instinct only; `_GFX_TO_RUNNER` maps consumer arches
to `None`, so a Radeon box resolves to no runner and no dispatch identity. This
branch makes gfx1201 a first-class target.

## Support matrix — what is measured vs. what is not

| Target | Arch | Status |
|---|---|---|
| Radeon AI PRO R9700 | gfx1201 | **MEASURED.** Board autodetects; runner benchmarked end-to-end on 2× R9700 |
| RX 9070 / 9070 XT | gfx1201 | Same arch, board never tested here. Autodetect keys on arch, so it should resolve — unverified |
| RX 9060 XT | gfx1200 | **Deliberately unmapped.** Navi 44 is the same ISA family but nobody has measured it; a runner label is a claim of support |
| gfx11 (RDNA3) | gfx1100/1101 | Not in this branch yet |
| Instinct | gfx942/gfx950 | Unchanged from upstream |

Multi-GPU: measured on a **2-card** host only. On this box the two GPUs report
`peers: false` and host↔GPU is capped at PCIe Gen3 (~12.9 GB/s measured, an
on-board switch upstream port downgrades the link). A 4-card rig will hit
different walls; nothing here is validated at that width.

## Quickstart

```bash
export HYPERLOOM_BYPASS_SCRIPTS_DIR=$PWD/examples/rdna
export MODEL=/path/to/model.gguf
export LLAMA_CPP_DIR=/path/to/llama.cpp        # expects build/bin/llama-bench
```

Hyperloom resolves `custom_r9700.sh` from that directory by the
`custom_{runner_type}.sh` convention once the board is detected as `r9700`.

Verify detection before running anything long:

```python
from hyperloom.inference_optimizer.gpu_types import amd_gpu_dispatch_identity
assert amd_gpu_dispatch_identity() == ("gfx1201", 64)
```

Run the runner standalone (it is a normal script — no Hyperloom required):

```bash
MODEL=/path/model.gguf LLAMA_BENCH=/path/llama-bench RESULT_DIR=/tmp/out \
  PP=512 TG=64 REPS=3 ./examples/rdna/custom_r9700.sh
```

### Why llama.cpp and not vLLM/SGLang

Below ~32 GB the serving stacks' VRAM and concurrency assumptions do not
transfer, and llama.cpp is what this class of card is actually served with. The
runner is a **scriptable** (server-less) workload: no OpenAI endpoint, no HTTP
client. It writes the InferenceX-shaped `inferencex_result.json` that
Hyperloom's collectors already consume, so nothing downstream needed changing.

### Correctness gate

`PPL_FILE` + `PPL_MAX` run `llama-perplexity` and emit a `quality_gate`. A run
that fails the gate is **unselectable regardless of throughput** — which is the
point of shipping one, since a wrong-but-fast kernel otherwise wins the
benchmark. With `PPL_FILE` set but no `PPL_MAX`, the perplexity is recorded and
explicitly marked `skipped` rather than being silently treated as a pass.

## RDNA4 profiling realities

These will silently break tooling that assumes Instinct. All measured on
gfx1201:

| Reality | Consequence |
|---|---|
| PC sampling is **host_trap only**, min interval 512 | `stochastic` is unsupported; a harness requesting it gets nothing |
| Several PMC derived counters (e.g. `FETCH_SIZE`) collect **zeros** | Reads as "no memory traffic" rather than as an unsupported counter |
| `GL2C` counters need a **fixed DPM profile** to leave zero | The counters are perf-level gated, not broken firmware |
| `profile_peak` vs `profile_standard` | Both un-gate the counters, but `profile_standard` **throttles to ~1593 MHz** vs ~2330 MHz — it suppresses any benchmark running alongside it. Prefer `profile_peak` |
| `--setperflevel high` **reduces** inference throughput ~10% vs `auto` | A trap for any harness that pins clocks for "stable" absolutes |
| No MFMA on this arch | Anything reaching for CDNA matrix cores must branch on gfx arch, not on "is AMD" |

## Quantization schemes on gfx1201

`fp8` / `ptpc_fp8` are available; RDNA4 has WMMA fp8 and a native E4M3 path.

`mxfp4` / `mxfp4_fp8` are **unavailable**, and for a stronger reason than the
upstream MI355X gate: the MX scaled-convert instructions are gfx950/gfx1250
gated, so an assembler capability probe on gfx1201 grades them REJECTED. There
is no hardware datapath to fall back to.

## Upstream posture

`main` on this fork tracks `upstream/main` byte-for-byte so PR branches cherry-
pick cleanly. AMD has said RDNA is triaged after 2026-08-30; this branch exists
so the hardware is usable now, not to route around that.
