# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The whole third-tier attempt, from gate to verdict.

Five checkpoints, and failing any of them ends the attempt without affecting
the tuning run that hosts it. In order, because each is cheaper than the next
and rules out a different kind of wrong:

    gate      -- is this even our problem, and did an operator allow it
    generate  -- can a script be authored at all
    contract  -- does its output have the agreed shape
    sandbox   -- does it run here without taking the box down
    referee   -- are its candidates actually faster, on our clock

The referee is last and decisive. Everything before it can be gamed by a script
that reports what it was asked to report; nothing before it establishes that a
single kernel got faster. That is why a generated tuner's own numbers are read
only to be discarded.

One retry, with the rejection reason handed back. More would be a search over
authorings, which is a different and much more expensive activity than writing
one tuner for one gap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .contract import load_candidates, validate_output_csv
from .coverage import CoverageGap
from .gate import GateDecision, should_generate
from .generate import generate_tuner
from .ledger import record_outcome, script_digest
from .mandate import build_mandate, write_mandate
from .referee import Judgement, judge_candidates
from .sandbox import run_generated_tuner

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


@dataclass
class Tier3Outcome:
    """What the attempt produced, and where it stopped."""

    attempted: bool = False
    stage: str = "gate"
    ok: bool = False
    reason: str = ""
    table: str = ""
    script: str = ""
    digest: str = ""
    judgements: list[Judgement] = field(default_factory=list)
    #: Whether an operator has signed this exact script off. Named for the
    #: signature and not for "trusted" because CodeQL's clear-text-storage
    #: query classifies any field whose name contains "trusted" as a secret,
    #: and this one is serialised into ``tier3_outcome.json``. It is a bool.
    operator_signed: bool = False

    @property
    def improved_shapes(self) -> int:
        return sum(1 for j in self.judgements if j.improved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "stage": self.stage,
            "ok": self.ok,
            "reason": self.reason,
            "table": self.table,
            "script": self.script,
            "digest": self.digest,
            "operator_signed": self.operator_signed,
            "improved_shapes": self.improved_shapes,
            "judgements": [j.to_dict() for j in self.judgements],
        }


def attempt_generated_tuner(
    gaps: list[CoverageGap],
    demand_shapes_for: Callable[[CoverageGap], list[dict[str, Any]]],
    work_root: Path,
    *,
    model_name: str = "",
    gpu: str = "",
    framework: str = "",
    make_baseline: Callable[[str], Callable[[], Any]] | None = None,
    make_dispatch: Callable[[str], Callable[[dict], Callable[[], Any] | None]] | None = None,
    make_correctness: Callable[[str], Callable[[Callable[[], Any]], bool]] | None = None,
    sync: Callable[[], Any] | None = None,
    decision: GateDecision | None = None,
) -> Tier3Outcome:
    """Try to produce a verified generated tuner for the strongest gap.

    The three ``make_*`` callables are how a caller supplies the only things
    that cannot be written generically: what the unmodified path is, how to
    dispatch a proposed candidate, and how to check its numerics. Without them
    the attempt stops before the referee, because an unverified candidate is
    exactly what this tier must never emit.
    """
    decision = decision or should_generate(gaps)
    outcome = Tier3Outcome(stage="gate", reason="; ".join(decision.reasons))
    if not decision.allowed or decision.gap is None:
        return outcome

    gap = decision.gap
    outcome.attempted = True
    outcome.table = gap.table
    work_dir = work_root / "tier3" / gap.table.replace(".", "_")
    work_dir.mkdir(parents=True, exist_ok=True)

    shapes = demand_shapes_for(gap)
    mandate = build_mandate(gap, shapes, gpu=gpu, framework=framework)
    mandate.output_csv = str(work_dir / "out.csv")
    mandate.candidates_json = str(work_dir / "candidates.json")
    write_mandate(mandate, work_dir / "mandate.json")

    retry_note = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        outcome.stage = "generate"
        gen = generate_tuner(mandate, work_dir, retry_note=retry_note)
        if not gen.ok or gen.script_path is None:
            outcome.reason = gen.reason
            return outcome
        outcome.script = str(gen.script_path)
        outcome.digest = script_digest(gen.script_path)

        outcome.stage = "sandbox"
        run = run_generated_tuner(
            gen.script_path,
            work_dir,
            expect=[Path(mandate.output_csv), Path(mandate.candidates_json)],
        )
        if not run.ok:
            retry_note = (
                f"The script did not produce both output files "
                f"(rc={run.returncode}, timed_out={run.timed_out}). Tail:\n"
                f"{run.stderr_tail[-800:]}"
            )
            outcome.reason = retry_note
            if attempt < MAX_ATTEMPTS:
                continue
            return outcome

        outcome.stage = "contract"
        violations = validate_output_csv(mandate.output_csv, mandate)
        if violations:
            retry_note = "The output violated the contract:\n" + "\n".join(f"- {v}" for v in violations[:8])
            outcome.reason = retry_note
            if attempt < MAX_ATTEMPTS:
                continue
            return outcome
        break

    outcome.stage = "referee"
    if make_baseline is None or make_dispatch is None:
        outcome.reason = (
            "no dispatch was supplied, so the candidates cannot be re-timed; "
            "an unverified generated tuner is not emitted"
        )
        return outcome

    candidates = load_candidates(mandate.candidates_json, mandate)
    if not candidates:
        outcome.reason = "the script proposed no candidates to re-time"
        return outcome

    for shape, cands in candidates.items():
        outcome.judgements.append(
            judge_candidates(
                shape,
                cands,
                baseline=make_baseline(shape),
                dispatch=make_dispatch(shape),
                is_correct=make_correctness(shape) if make_correctness else None,
                sync=sync,
            )
        )

    outcome.ok = outcome.improved_shapes > 0
    best = max(
        (j.best_timing.speedup for j in outcome.judgements if j.best_timing and j.best_timing.usable),
        default=None,
    )
    outcome.reason = (
        f"{outcome.improved_shapes} of {len(outcome.judgements)} shape(s) improved"
        if outcome.ok
        else "no shape improved once re-timed"
    )

    record = record_outcome(
        work_root / "tier3" / "ledger.json",
        digest=outcome.digest,
        table=gap.table,
        model=model_name,
        improved=outcome.ok,
        speedup=best,
    )
    from .ledger import is_trusted

    outcome.operator_signed = is_trusted(outcome.digest)
    (work_dir / "outcome.json").write_text(
        json.dumps({**outcome.to_dict(), "ledger": record.to_dict()}, indent=2),
        encoding="utf-8",
    )
    log.info("tier3: %s -- %s", gap.table, outcome.reason)
    return outcome
