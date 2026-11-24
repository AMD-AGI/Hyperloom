---
myst:
    html_meta:
        "description": "KERNEL lane behaviour reference for Hyperloom: forge-only behaviors including fusion recipe cap, GEMM per-tuner candidates, fp8 to bf16 retry, tuner cap, and subprocess output recording."
        "keywords": "Hyperloom, kernel lane, forge, KERNEL_OPT_BACKEND_ORDER, fusion lane, GEMM tuning, kernel rewrite controller, subprocess"
---
# KERNEL lane behaviour

No variable toggles the behaviours below; `KERNEL_OPT_BACKEND_ORDER` only decides
which of them apply.

Globally, on every backend including the default `geak`:

- **`kernel_opt` and `gemm_tuning` are no longer model-issued.** Orchestration
  can neither propose either action nor request `run_optimization` /
  `run_gemm_tuning` of the kernel agent; both are refused as
  `phase_incompatible`. The Coordinator dispatches each lane once at KERNEL
  entry from a lane budget, the way the fusion lane already worked. `integrate` stays proposable, so draining the KEEP queue is still the
  model's job.

On the `forge` KERNEL path only (`KERNEL_OPT_BACKEND_ORDER=forge`; the default
`geak` backend hands to SWEEP before reaching any of these):

- **Fusion authors one independent sibling patch per confirmed recipe** rather
  than a single combined patch.
- **The fusion lane caps its recipe count** at 3 (`--max-recipes`), funded from
  its share of the phase. A session with no finite remaining time derives no
  share and stays uncapped.
- **GEMM reports carry a per-tuner `candidates[]` list**, so one tuner's table
  can be kept while another is reverted, instead of `recommended_env` landing
  all-or-nothing.
- **The fp8 → bf16 dense GEMM retry runs inside the same gemm call.** The
  follow-up subprocess Hyperloom used to launch is gone; the bf16 dense pass is
  selected up front as a fallback tuner and executes only when no earlier tuner
  produced a candidate, so a winning fp8 run pays nothing for it.
- **The gemm lane caps how many routed tuners run** (`--max-tuners`), priced on
  the router's own per-tuner estimates.
- **Kernel subprocess output is recorded as activity** for diagnostics. Activity
  does not prove useful progress and does not extend benchmark hard deadlines.
- **A `status: partial` invocation spec is declined on the rewrite route**
  rather than admitted because the file exists. A partial spec leaves the
  producer on its placeholder driver, which burns the whole budget and then
  exits non-zero.
