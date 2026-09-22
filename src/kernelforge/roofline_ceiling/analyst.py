# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The analyst session: an agent estimates the ceiling, the framework measures the box.

Deriving the minimum legal work of an arbitrary operator is not something a
table can do, and neither is composing it into a latency. MoE routing, paged
attention, segmented reductions, fusion legality, how much of one stage overlaps
the next, whether a shape that fills eight CUs is limited by peak throughput at
all -- these are judgements, and a fixed composition rule imposed on them makes
the analyst distort its model to fit the rule.

So the whole estimate is the analyst's, and the audit trail is the derivation it
writes rather than a schema this module can parse. What is *not* the analyst's
is the hardware: every peak, bandwidth and launch cost in the request below was
measured on this box, so that two ceilings taken a month apart are at least
divided by the same numbers, and so ``peak_source`` on the report is true.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import (
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.roofline_ceiling.contract import (
    CeilingContractError,
    CeilingReport,
    build_report,
    response_schema,
)
from kernelforge.roofline_ceiling.evidence import OBSERVED_CAMPAIGN, EvidenceBundle
from kernelforge.roofline_ceiling.specs import CANONICAL_INSTRUCTION_PATHS, peak_source_meaning
from kernelforge.orchestrator.structured_output import (
    build_repair_prompt,
    extract_json_object,
)
from kernelforge.resources import resource_path

log = logging.getLogger("kernelforge.roofline_ceiling")

ROLE_FILENAME = "ceiling_analyst.md"

#: The analyst reads source, driver and profiling artifacts. Reading a trace
#: costs turns, so this is generous; it is bounded by the session timeout too.
DEFAULT_ANALYST_TURNS = 80

#: One repair round. The failure a repair fixes is a malformed or incomplete
#: response, and an analyst that cannot produce the schema twice is not going to
#: produce it on the third ask -- it is going to spend budget agreeing with the
#: error message.
MAX_REPAIR_ROUNDS = 1


class CeilingAnalysisError(RuntimeError):
    """Raised when no usable ceiling model could be obtained."""


def load_role(project_root: str | Path | None = None) -> str:
    """Read the analyst role document shipped with the package."""
    return (resource_path("roofline_ceiling", project_root) / ROLE_FILENAME).read_text(encoding="utf-8")


def build_request(
    *,
    kernel_files: Sequence[str],
    driver_script: str,
    performance_command: Sequence[str],
    case_ids: Sequence[str],
    case_params: Mapping[str, Any],
    evidence: EvidenceBundle,
) -> str:
    """Build the analyst's request payload.

    Every figure is stated here rather than left for the analyst to look up. A
    peak it reads off a knowledge-base card is the vendor datasheet, roughly
    twice what the chip sustains, and a ceiling divided by that while the report
    says ``reference_profile`` is a report that lies about its own inputs.
    """
    hardware = evidence.hardware
    payload: dict[str, Any] = {
        "task": (
            "Estimate the theoretical achievable latency of every scored case of this kernel, "
            "against the measured hardware figures below. Return one JSON object matching "
            "output_schema and nothing else."
        ),
        "kernel_files": list(kernel_files),
        "driver_script": driver_script,
        "performance_command": list(performance_command),
        "scored_case_ids": list(case_ids),
        "case_parameters": dict(case_params),
        "hardware": {
            "arch": hardware.arch,
            "peak_source": hardware.peak_source,
            "peak_source_meaning": peak_source_meaning(hardware.peak_source),
            "units": "peak_flops in FLOP/s (OP/s for integer paths); bandwidth in bytes/s; times in seconds",
            "peak_flops_by_instruction_path": dict(hardware.peak_flops),
            "bandwidth_bytes_per_s_by_memory_level": dict(hardware.bandwidth),
            "dispatch_floor_s": hardware.dispatch_floor_s,
            "canonical_instruction_paths": list(CANONICAL_INSTRUCTION_PATHS),
            "note": (
                "Use these and only these. A memory level or instruction path absent from the "
                "tables above was not measured on this box: say so rather than substituting a "
                "datasheet figure or a neighbouring rate."
            ),
            "provenance": hardware.provenance,
        },
        "evidence_dir": str(evidence.artifacts_dir),
        "evidence_files": evidence.artifact_paths(),
        "observed_ms": dict(evidence.observed_ms),
        "observed_ms_origin": evidence.observed_origin,
        "observed_ms_meaning": (
            (
                "latency measured by the campaign over repeated runs, with no profiler attached"
                if evidence.observed_origin == OBSERVED_CAMPAIGN
                else "latency seen while profiling, so inflated by the profiler's own overhead"
            )
            + ". A sanity reference only: no ceiling may be back-solved from it."
        ),
        "output_schema": response_schema(),
    }
    if evidence.notes:
        payload["evidence_notes"] = list(evidence.notes)
    return json.dumps(payload, indent=2, sort_keys=True)


def _spec(
    *,
    system_prompt: str,
    user_prompt: str,
    workdir: str,
    model: str,
    timeout_sec: int,
    evidence_dir: str,
    turns: int,
) -> AgentRunSpec:
    """One read-only analyst session."""
    return AgentRunSpec(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        role="ceiling analyst",
        cwd=workdir,
        model=model,
        writable=False,
        timeout_sec=max(1, int(timeout_sec)),
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=False,
            shell=False,
            max_turns=turns,
        ),
        # The analyst may only read. Evidence collection is the deterministic
        # side's job precisely so the numbers it reasons over are the numbers the
        # framework will divide by.
        protected_globs=["*"],
        additional_directories=[evidence_dir],
        # The kernel under analysis is routinely a dirty checkout mid-campaign,
        # and a read-only session has no business demanding a clean tree.
        allow_dirty_baseline=True,
    )


async def _ask(backend: Any, spec: AgentRunSpec) -> str:
    """Run one session and return its final text."""
    result = await asyncio.wait_for(
        backend.run(spec),
        timeout=watchdog_timeout_sec(spec.timeout_sec or 0),
    )
    text = str(getattr(result, "text", "") or "").strip()
    end_reason = str(getattr(result, "end_reason", "agent_stopped") or "agent_stopped")
    if not text:
        raise CeilingAnalysisError(f"analyst returned no text (end_reason={end_reason})")
    return text


async def run_ceiling_analysis(
    backend: Any,
    *,
    canonical_id: str,
    workdir: str,
    kernel_files: Sequence[str],
    driver_script: str,
    performance_command: Sequence[str],
    case_ids: Sequence[str],
    case_params: Mapping[str, Any],
    evidence: EvidenceBundle,
    model: str = "",
    timeout_sec: int = 3600,
    turns: int = DEFAULT_ANALYST_TURNS,
    project_root: str | Path | None = None,
) -> CeilingReport:
    """Obtain one validated ceiling report, repairing a malformed answer once."""
    system_prompt = load_role(project_root)
    user_prompt = build_request(
        kernel_files=kernel_files,
        driver_script=driver_script,
        performance_command=performance_command,
        case_ids=case_ids,
        case_params=case_params,
        evidence=evidence,
    )

    attempt_prompt = user_prompt
    last_error = ""
    last_text = ""
    for attempt in range(MAX_REPAIR_ROUNDS + 1):
        spec = _spec(
            system_prompt=system_prompt,
            user_prompt=attempt_prompt,
            workdir=workdir,
            model=model,
            timeout_sec=timeout_sec,
            evidence_dir=str(evidence.artifacts_dir),
            turns=turns,
        )
        last_text = await _ask(backend, spec)
        try:
            payload = extract_json_object(last_text, "ceiling analysis")
            return build_report(
                payload,
                canonical_id=canonical_id,
                hardware=evidence.hardware,
                expected_case_ids=case_ids,
                observed_ms=evidence.observed_ms,
            )
        except (CeilingContractError, ValueError) as exc:
            last_error = str(exc)
            log.warning("ceiling analysis attempt %d rejected: %s", attempt + 1, last_error)
            if attempt >= MAX_REPAIR_ROUNDS:
                break
            attempt_prompt = build_repair_prompt(
                label="ceiling analysis",
                original_response=last_text,
                validation_error=last_error,
                output_schema=response_schema(),
            )

    raise CeilingAnalysisError(f"analyst produced no valid ceiling model: {last_error}")


__all__ = [
    "DEFAULT_ANALYST_TURNS",
    "MAX_REPAIR_ROUNDS",
    "ROLE_FILENAME",
    "CeilingAnalysisError",
    "build_request",
    "load_role",
    "run_ceiling_analysis",
]
