# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brief handed to whoever writes a generated tuner.

Four things, because a tuner cannot be written without any of them: the output
contract, the demand it must cover, why the existing tiers did not, and a
skeleton that already runs on this hardware.

Three of the clauses below are not style preferences. They come from the first
real trial of this on MI355X, where both an LLM-written tuner and aiter's own
official tuner produced confident, wrong answers in the same two ways:

* **Correctness has to be re-checked, on fresh inputs, several times.** Four
  split-K winners -- two chosen by the generated tuner, two by aiter's -- were
  wrong on 1.25-3.98% of output elements, and *which* elements changed between
  identical calls on identical inputs. A single check passes such a kernel
  roughly at random; the generated tuner's own report claimed a worst-case
  relative error of 7.65e-3 for candidates that a repeated audit measured at 17
  to 50.
* **A Python-loop timer cannot rank these kernels.** One dispatch costs ~12us on
  this box against kernels of 5-13us, so every candidate collapses to about the
  same number and the fastest one is invisible. Capturing N calls into a graph
  and replaying it removes the host cost from the measurement; without that step
  the honest conclusion from the same data was "there is nothing to tune here".
* **Its own timings decide nothing.** :mod:`.referee` re-times the proposed
  candidates with forge's clock, and only those numbers reach a KEEP. This is
  what makes the rest survivable: a script that mistimes itself, or benchmarks
  an empty kernel, costs machine time and nothing else.

The mandate is data. Rendering it as text is a convenience for a human or an
agent; the fields are what downstream code checks against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Columns every generated tuner must produce, whatever it searches. The three
# timing columns are what makes a result auditable at all: without default_us
# the improvement cannot be checked, and without both times the referee cannot
# tell a real win from a mis-scaled one.
REQUIRED_OUTPUT_COLUMNS = ("default_us", "tuned_us", "improved")

# Repeats of the correctness check, on fresh inputs each time, worst result
# counted. Eight was enough to catch every intermittently-wrong kernel observed;
# one was not enough to catch any of them.
CORRECTNESS_TRIALS = 8

# Relative error above which a candidate is discarded, measured against the
# magnitude of the reference as a whole -- see MAX_RELATIVE_ERROR_DEFINITION.
# On MI355X the unmodified torch.matmul scores 0.015 by this measure, so the
# limit leaves roughly 3x headroom over correct-but-rounded while staying far
# below the 17-50 seen from broken kernels.
MAX_RELATIVE_ERROR = 5e-2

# How to compute it, stated because the obvious reading is unusable: dividing
# element by element and flooring the denominator makes any element where the
# reference lands near zero dominate, and a K=7168 random GEMM produces plenty
# of those. Measured that way the unmodified torch.matmul scores 1.375 -- a
# gate at any sane threshold would reject the default path itself.
MAX_RELATIVE_ERROR_DEFINITION = "max|got - ref| / mean|ref|, over the whole output tensor, with ref computed in fp32"


@dataclass
class TunerMandate:
    """Everything needed to write one generated tuner, and nothing else."""

    table: str
    key_schema: list[str]
    demand_shapes: list[dict[str, Any]]
    why_existing_tiers_failed: str
    gpu: str = ""
    framework: str = ""
    dtype_note: str = ""
    reference_skeleton: str = ""
    budget_seconds: int = 1500
    output_csv: str = "/tmp/generated_tuner/out.csv"
    candidates_json: str = "/tmp/generated_tuner/candidates.json"
    max_candidates_per_shape: int = 5

    @property
    def output_columns(self) -> list[str]:
        """Key columns first, then the search's own, then the three timings."""
        return [*self.key_schema, "backend", "config", *REQUIRED_OUTPUT_COLUMNS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "key_schema": list(self.key_schema),
            "output_columns": self.output_columns,
            "demand_shapes": list(self.demand_shapes),
            "why_existing_tiers_failed": self.why_existing_tiers_failed,
            "gpu": self.gpu,
            "framework": self.framework,
            "dtype_note": self.dtype_note,
            "budget_seconds": self.budget_seconds,
            "output_csv": self.output_csv,
            "candidates_json": self.candidates_json,
            "max_candidates_per_shape": self.max_candidates_per_shape,
            "correctness_trials": CORRECTNESS_TRIALS,
            "max_relative_error": MAX_RELATIVE_ERROR,
            "max_relative_error_definition": MAX_RELATIVE_ERROR_DEFINITION,
        }

    def render(self) -> str:
        """The mandate as a brief. Kept in one place so the constraints travel."""
        shapes = "\n".join("  " + ", ".join(f"{k}={v}" for k, v in s.items()) for s in self.demand_shapes)
        return _TEMPLATE.format(
            table=self.table,
            gpu=self.gpu or "(unspecified)",
            framework=self.framework or "(unspecified)",
            dtype_note=self.dtype_note or "(none)",
            key_schema=", ".join(self.key_schema),
            shapes=shapes or "  (none)",
            columns=",".join(self.output_columns),
            output_csv=self.output_csv,
            candidates_json=self.candidates_json,
            top_k=self.max_candidates_per_shape,
            why=self.why_existing_tiers_failed,
            trials=CORRECTNESS_TRIALS,
            max_rel=MAX_RELATIVE_ERROR,
            max_rel_def=MAX_RELATIVE_ERROR_DEFINITION,
            budget=self.budget_seconds,
            skeleton=self.reference_skeleton or "(none supplied)",
        )


_TEMPLATE = """\
# Write a tuner for {table}

## Target
- GPU: {gpu}
- Framework: {framework}
- Key schema: {key_schema}
- dtype: {dtype_note}

## Shapes it must cover
These are the keys the runtime looked up and did not find. They are the whole
job; a config that is fast on other shapes is worth nothing here.
{shapes}

## Why the existing tuners cannot do this
{why}

## Output contract (binding)
Write `{output_csv}` with exactly this header:

    {columns}

- `config` describes the choice your search varies. Use `;` between fields,
  never a comma.
- `default_us` is the unmodified path at that shape; `tuned_us` is your best
  candidate; `improved` is True when tuned_us < default_us.
- Emit one row per shape even when nothing beat the default.

Also write `{candidates_json}`: for each shape, up to {top_k} candidates ranked
best first, each carrying enough detail to be dispatched by code that did not
write your script.

## Correctness
Check every candidate against a reference implementation {trials} times, on
fresh inputs each time, and keep the worst result. Discard anything above
{max_rel}, where the error is `{max_rel_def}`. Report how many you discarded.

Use that definition and not an element-wise ratio. Dividing element by element
and flooring the denominator lets any output element that happens to land near
zero dominate the result, and a large-K random GEMM produces plenty of those:
measured that way the unmodified `torch.matmul` scores 1.375 against its own
fp32 reference, so such a gate rejects the default path itself.

One check is not enough, and this is not a hypothetical: four split-K winners
measured on this hardware -- two picked by a generated tuner, two by the vendor's
own official tuner -- were wrong on 1.25-3.98% of output elements, and which
elements were wrong changed between identical calls. A single check passes such
a kernel roughly at random.

## Timing
Measure with a captured graph replayed N times, not a Python loop. One dispatch
costs ~12us on this hardware while the kernels under test cost 5-13us, so a loop
timer flattens every candidate to about the same number and hides the fastest
one. Warm the clocks before the first measurement.

Your timings are informational. The harness re-times your candidates with its
own clock and only those numbers decide anything, so do not tune the benchmark
-- propose genuinely fast configurations and describe them precisely enough to
be re-dispatched.

## Budget
About {budget}s of wall time. Explore what is callable before committing to a
search: if you cannot find an axis beyond calling the default, say so. That is a
valid and useful finding, and far better than a script that only measures the
default.

## Reference skeleton
{skeleton}
"""


def build_mandate(
    gap: Any,
    demand_shapes: list[dict[str, Any]],
    *,
    gpu: str = "",
    framework: str = "",
    dtype_note: str = "",
    reference_skeleton: str = "",
    budget_seconds: int = 1500,
) -> TunerMandate:
    """Turn a coverage gap plus its demanded shapes into a mandate."""
    return TunerMandate(
        table=str(getattr(gap, "table", "") or ""),
        key_schema=list(getattr(gap, "key_schema", []) or []),
        demand_shapes=list(demand_shapes),
        why_existing_tiers_failed=str(getattr(gap, "reason", "") or ""),
        gpu=gpu,
        framework=framework,
        dtype_note=dtype_note,
        reference_skeleton=reference_skeleton,
        budget_seconds=budget_seconds,
    )


def write_mandate(mandate: TunerMandate, path: Any) -> Any:
    """Persist a mandate as JSON beside its rendered brief."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mandate.to_dict(), indent=2), encoding="utf-8")
    p.with_suffix(".md").write_text(mandate.render(), encoding="utf-8")
    return p
