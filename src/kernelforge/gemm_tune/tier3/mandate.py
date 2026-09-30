# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the contract given to a generated-tuner author.

The mandate supplies demand, prior-tier gaps, the dispatchable candidate
protocol, a runnable skeleton, repeated correctness checks, graph timing, and
crash-resumable output. The independent referee remains authoritative; the
structured fields, not rendered prose, are the downstream contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Columns every generated tuner must produce, whatever it searches.
REQUIRED_OUTPUT_COLUMNS = ("default_us", "tuned_us", "improved")

# Repeats of the correctness check, on fresh inputs each time, worst result counted.
CORRECTNESS_TRIALS = 8

# Relative error above which a candidate is discarded, measured against the magnitude of the reference as a whole --
# see MAX_RELATIVE_ERROR_DEFINITION.
MAX_RELATIVE_ERROR = 5e-2

# How to compute it, stated because the obvious reading is unusable: dividing element by element and flooring the
# denominator makes any element where the reference lands near zero dominate, and a K=7168 random GEMM produces plenty
# of those.
MAX_RELATIVE_ERROR_DEFINITION = "max|got - ref| / mean|ref|, over the whole output tensor, with ref computed in fp32"

# Dense default only; adapters with different screening provide their own rule.
DENSE_CORRECTNESS_NOTE = """\
Use that definition and not an element-wise ratio. Dividing element by element
and flooring the denominator lets any output element that happens to land near
zero dominate the result, and a large-K random GEMM produces plenty of those:
measured that way the unmodified `torch.matmul` scores 1.375 against its own
fp32 reference, so such a gate rejects the default path itself.

One check is not enough, and this is not a hypothetical: four split-K winners
measured on this hardware -- two picked by a generated tuner, two by the vendor's
own official tuner -- were wrong on 1.25-3.98% of output elements, and which
elements were wrong changed between identical calls. A single check passes such
a kernel roughly at random."""


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
    candidate_protocol: str = ""
    reference_skeleton: str = ""
    budget_seconds: int = 1500
    output_csv: str = "/tmp/generated_tuner/out.csv"
    candidates_json: str = "/tmp/generated_tuner/candidates.json"
    max_candidates_per_shape: int = 5
    # The screen the referee will apply. Defaults are the dense adapter's; a table whose adapter differs overrides all
    # four together, because a limit without its metric is not a rule.
    correctness_trials: int = CORRECTNESS_TRIALS
    max_relative_error: float = MAX_RELATIVE_ERROR
    max_relative_error_definition: str = MAX_RELATIVE_ERROR_DEFINITION
    correctness_note: str = DENSE_CORRECTNESS_NOTE

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
            "candidate_protocol": self.candidate_protocol,
            "budget_seconds": self.budget_seconds,
            "output_csv": self.output_csv,
            "candidates_json": self.candidates_json,
            "max_candidates_per_shape": self.max_candidates_per_shape,
            "correctness_trials": self.correctness_trials,
            "max_relative_error": self.max_relative_error,
            "max_relative_error_definition": self.max_relative_error_definition,
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
            protocol=self.candidate_protocol
            or (
                "Nothing on this box knows how to re-dispatch a candidate for this table, so\n"
                "describe each one however is clearest and expect the result to be reported\n"
                "rather than promoted."
            ),
            trials=self.correctness_trials,
            max_rel=self.max_relative_error,
            max_rel_def=self.max_relative_error_definition,
            correctness_note=self.correctness_note,
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

Also write `{candidates_json}`: a JSON object mapping each shape to up to {top_k}
candidates, ranked best first. This file is the only part of your work that can
be promoted, because the harness re-times what is in it and nothing else.

{protocol}

## Correctness
Check every candidate against a reference implementation {trials} times, on
fresh inputs each time, and keep the worst result. Discard anything above
{max_rel}, where the error is `{max_rel_def}`. Report how many you discarded.

This is the screen the harness itself applies to your candidates, quoted from the
code that applies it. Applying it yourself is not duplicated work: every
candidate you send up that fails it is a slot spent on something that cannot be
promoted.

{correctness_note}

## Timing
Measure with a captured graph replayed N times, not a Python loop. One dispatch
costs ~12us on this hardware while the kernels under test cost 5-13us, so a loop
timer flattens every candidate to about the same number and hides the fastest
one. Warm the clocks before the first measurement.

Your timings are informational. The harness re-times your candidates with its
own clock and only those numbers decide anything, so do not tune the benchmark
-- propose genuinely fast configurations and describe them precisely enough to
be re-dispatched.

## Surviving the search
A bad launch on this hardware does not raise. It ends the process:
`Memory access fault by GPU node-N ... Write access to a read-only page`, no
traceback, no `except` that can see it, and everything still in memory is gone.
Measured:
the first tuner written from this mandate died on shape 2 of 42 while sweeping
one backend's kernel ids, four minutes in, and kept a single row. The fault is
asynchronous too, so it can surface several launches after the one that caused
it; the candidate the process died on is not reliably the guilty one.

So keep the state on disk and do the GPU work in short-lived subprocesses.
Record each candidate before you launch it, so a worker that dies names what it
was running and the driver can skip that pair and carry on. Write each shape's
result -- the default first, before anything risky -- as soon as it exists, and
rebuild both output files from that state after every worker exits. A fault then
costs one candidate instead of the whole run.

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
    candidate_protocol: str | None = None,
    reference_skeleton: str = "",
    budget_seconds: int = 1500,
) -> TunerMandate:
    """Build a mandate from a gap and its demanded shapes.

    Candidate format and correctness defaults come from the same adapter the
    referee uses, preventing un-dispatchable or unpromotable proposals.
    """
    table = str(getattr(gap, "table", "") or "")
    if candidate_protocol is None:
        from .dispatch import describe_candidate_protocol

        candidate_protocol = describe_candidate_protocol(table)
    from .dispatch import describe_correctness_rule

    rule = describe_correctness_rule(table) or {}
    return TunerMandate(
        table=table,
        key_schema=list(getattr(gap, "key_schema", []) or []),
        demand_shapes=list(demand_shapes),
        why_existing_tiers_failed=str(getattr(gap, "reason", "") or ""),
        gpu=gpu,
        framework=framework,
        dtype_note=dtype_note,
        candidate_protocol=candidate_protocol,
        reference_skeleton=reference_skeleton,
        budget_seconds=budget_seconds,
        correctness_trials=int(rule.get("trials", CORRECTNESS_TRIALS)),
        max_relative_error=float(rule.get("limit", MAX_RELATIVE_ERROR)),
        max_relative_error_definition=str(rule.get("definition", MAX_RELATIVE_ERROR_DEFINITION)),
        correctness_note=str(rule.get("note", DENSE_CORRECTNESS_NOTE)),
    )


def write_mandate(mandate: TunerMandate, path: Any) -> Any:
    """Persist a mandate as JSON beside its rendered brief."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mandate.to_dict(), indent=2), encoding="utf-8")
    p.with_suffix(".md").write_text(mandate.render(), encoding="utf-8")
    return p
