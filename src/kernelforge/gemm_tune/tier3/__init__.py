# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generated tuners: the third source of a tuner, after aiter's and forge's own.

The order is deliberate. Tier 1 is an official aiter script; Tier 2 is a tuner
forge implements because aiter ships none for that capability. Tier 3 is a tuner
written for the occasion, and it only earns a turn when a demand entry is served
by neither of the first two -- which, on every combination measured so far, has
not happened. :func:`coverage_gaps` exists to keep that answer honest rather than
assumed: it names what fell through, so "is this needed" is something the fleet
answers instead of something the design asserts.

The rule that makes a generated tuner safe is that it never gets to decide
anything. It proposes configurations; :mod:`.referee` re-times the ones it
proposes with forge's own clock, and only those numbers reach a KEEP. A script
that mistimes its own benchmark, or writes one that measures an empty kernel,
therefore costs machine time and nothing else.

Two hazards from the first real trial of this, both of which produced confident
and wrong answers, are encoded in :mod:`.mandate` rather than left to the author:

* a single correctness check passes kernels that are wrong intermittently. Four
  split-K winners -- two picked by an LLM-written tuner, two by aiter's own --
  computed 1.25-3.98% of elements incorrectly, with *which* elements changing
  between identical calls;
* a Python-loop timer cannot rank these kernels at all. One dispatch costs ~12us
  on MI355X against kernels of 5-13us, so every candidate flattens to roughly
  the same number and the fastest becomes invisible.
"""

from .contract import ContractViolation, validate_output_csv
from .coverage import CoverageGap, coverage_gaps
from .gate import GateDecision, should_generate
from .ledger import TunerRecord, is_trusted, record_outcome, script_digest
from .mandate import TunerMandate, build_mandate
from .referee import Judgement, PairedTiming, judge_candidates, time_paired
from .runner import Tier3Outcome, attempt_generated_tuner
from .sandbox import SandboxResult, run_generated_tuner

__all__ = [
    "ContractViolation",
    "CoverageGap",
    "GateDecision",
    "Judgement",
    "PairedTiming",
    "SandboxResult",
    "Tier3Outcome",
    "TunerMandate",
    "TunerRecord",
    "attempt_generated_tuner",
    "build_mandate",
    "coverage_gaps",
    "is_trusted",
    "judge_candidates",
    "record_outcome",
    "run_generated_tuner",
    "script_digest",
    "should_generate",
    "time_paired",
    "validate_output_csv",
]
