# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-iteration objective outcome ledger for the forge-loop.

Persists the objective facts written by the loop and gate: the ``git`` diff
summary of each iteration's net change, the measured outcome
(validation/bench/keep-revert), and real error signatures. Free-form Implementer
session records live only in :mod:`kernelforge.loop.lessons`; this ledger does
not compress them into one-line conclusions.

Rendered into the next agent prompt as:
    ## Observed toolchain constraints      <- deduped, distilled from failures
    ## Recent iterations                   <- last K compact entries

Also flushed to ``<workspace>/forge_experiments/forge_experience.md`` for
inspection and possible later promotion into the knowledge base. Scope is
per-campaign; cross-campaign accumulation is intentionally out of scope.
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from kernelforge.durable_io import atomic_write_text
from kernelforge.experience_distillation import (
    ConstraintMemory,
    extract_signature,
    render_ledger,
)


# Known error-signature -> crisp, reusable constraint. Extend as new recurring
# failure modes are observed. Keep each constraint short and actionable.
_CONSTRAINT_RULES: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"#arith\.fastmath<(?:True|False)>|FastMathFlags"),
        "FlyDSL `fastmath=` accepted a FastMathFlags value (e.g. `fm_fast` / "
        "`arith.FastMathFlags.fast`); a Python bool serialized to an invalid "
        "`#arith.fastmath<True>` attribute and failed to compile.",
    ),
    (
        re.compile(r"max_flat_work", re.IGNORECASE),
        "A launch block size past the device limit failed; for example, "
        "BLOCK_THREADS=512 exceeded the AMDGPU default max flat workgroup size.",
    ),
    (
        re.compile(r"invalid cast", re.IGNORECASE),
        "A register-vector width that did not match the copy-atom width triggered "
        "an 'Invalid cast!' backend assertion. A 128-bit copy atom used "
        "VEC_WIDTH = 128 // elem_bits (4 for f32, 8 for bf16/f16).",
    ),
    (
        re.compile(r"same[- ]type cast|to\(Float32\).*f32|invalid same", re.IGNORECASE),
        "A same-type cast, such as `.to(Float32)` on data already in f32, failed; "
        "the successful form guarded the conversion on dtype.",
    ),
]

# Heuristic markers for the single most informative line in an error blob.
_ERR_MARKERS = (
    "error",
    "assert",
    "exception",
    "traceback",
    "failed",
    "not faster",
    "allclose",
    "mlirerror",
    "unable to parse",
)


# How much of the chosen line survives into the prompt.
_SIGNATURE_CHARS = 180


def _extract_signature(text: str) -> str:
    """Pull one normalized, informative line out of an error/outcome blob."""
    return extract_signature(text, markers=_ERR_MARKERS, limit=_SIGNATURE_CHARS)


@dataclass
class ExperienceEntry:
    """One iteration's record."""

    iteration: int
    outcome: str
    diff_summary: str = ""
    error_sig: str = ""


class ExperienceLedger:
    """Per-run experience store, injected into each next iteration's prompt."""

    def __init__(self, workspace_dir: str, keep_recent: int = 6, max_constraints: int = 15):
        workspace = Path(workspace_dir)
        self.root = workspace / "forge_experiments"
        self.path = self.root / "forge_experience.md"
        self.jsonl_path = self.root / "experience.jsonl"
        self.keep_recent = keep_recent
        self.memory = ConstraintMemory(_CONSTRAINT_RULES, max_constraints=max_constraints)
        self.entries: list[ExperienceEntry] = []
        self._load()

    @property
    def constraints(self) -> list[str]:
        """The distilled constraints carried into the next prompt."""
        return self.memory.constraints

    @staticmethod
    def _entry_from_payload(payload: object) -> ExperienceEntry | None:
        """Validate and convert one JSONL record."""
        if not isinstance(payload, dict):
            return None
        expected_fields = {
            "iteration",
            "outcome",
            "diff_summary",
            "error_sig",
        }
        if set(payload) != expected_fields:
            return None
        iteration = payload.get("iteration")
        outcome = payload.get("outcome")
        if type(iteration) is not int or not isinstance(outcome, str):
            return None
        if any(not isinstance(payload[field], str) for field in ("diff_summary", "error_sig")):
            return None
        return ExperienceEntry(
            iteration=iteration,
            outcome=outcome,
            diff_summary=payload["diff_summary"],
            error_sig=payload["error_sig"],
        )

    def _load(self) -> None:
        """Reload the exact current structured history."""
        if self.jsonl_path.is_file():
            with contextlib.suppress(OSError):
                contents = self.jsonl_path.read_bytes()
                for line in contents.splitlines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    entry = self._entry_from_payload(payload)
                    if entry is None:
                        continue
                    self.entries.append(entry)
                    self._learn_from_entry(entry)
            return

    # ── recording ────────────────────────────────────────────────────────────
    def _learn_from_entry(self, entry: ExperienceEntry) -> None:
        """Promote objective failure signatures into reusable constraints.

        Only machine-verifiable error signatures feed the factual toolchain
        observations. The agent's narrative lives in full in the per-iteration
        lesson documents and is never distilled here.
        """
        self.memory.distill(entry.error_sig, entry.outcome)

    def record_iteration(
        self,
        iteration: int,
        outcome: str,
        diff_summary: str = "",
        error_text: str = "",
    ) -> None:
        """Record one iteration's objective outcome and error evidence."""
        entry = ExperienceEntry(
            iteration=iteration,
            outcome=(outcome or "").strip(),
            diff_summary=(diff_summary or "").strip(),
            error_sig=_extract_signature(error_text),
        )
        self._learn_from_entry(entry)
        self.entries.append(entry)
        self.flush()

    # ── rendering ────────────────────────────────────────────────────────────
    @staticmethod
    def _entry_lines(entry: ExperienceEntry) -> list[str]:
        lines = [f"- iter {entry.iteration}: {entry.outcome}"]
        lines.extend(f"    {ln}" for ln in entry.diff_summary.splitlines()[:8])
        if entry.error_sig:
            lines.append(f"    error: {entry.error_sig}")
        return lines

    def _render(self, entries: list[ExperienceEntry]) -> str:
        return render_ledger(
            constraints_heading="## Observed toolchain constraints",
            constraints=self.constraints,
            entries_heading="## Recent iterations",
            entry_lines=[self._entry_lines(entry) for entry in entries],
        )

    def render_for_prompt(self, include_recent: bool = True) -> str:
        """Bounded text for the agent prompt.

        With ``include_recent=True`` (default) renders distilled constraints AND
        the last K iteration entries. With ``include_recent=False`` renders ONLY
        the constraints — used when the candidate-archive digest is present,
        since its trajectory and diffs would duplicate the ledger's recent
        iteration rows.
        """
        entries = self.entries[-self.keep_recent :] if include_recent else []
        return self._render(entries)

    def flush(self) -> None:
        """Persist structured JSONL and the full Markdown inspection view."""
        with contextlib.suppress(Exception):
            self.root.mkdir(parents=True, exist_ok=True)
            payload = "".join(json.dumps(asdict(entry), sort_keys=True) + "\n" for entry in self.entries)
            atomic_write_text(self.jsonl_path, payload)
        with contextlib.suppress(Exception):
            self.root.mkdir(parents=True, exist_ok=True)
            header = "# Forge experience ledger\n\n"
            self.path.write_text(header + self._render(self.entries) + "\n")
