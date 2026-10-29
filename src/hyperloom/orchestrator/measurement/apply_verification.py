# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check whether a tuned artifact was actually read by the server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.workflows.gemm_tune import parse_log_file

# Verdicts that should block a KEEP.
BLOCKING_VERDICTS = frozenset({"not_merged", "zero_hit"})


@dataclass(frozen=True)
class ApplyVerdict:
    """Whether the tuned table reached the server and was read."""

    verdict: str
    hits: int = 0
    misses: int = 0
    merged_tables: list[str] = field(default_factory=list)
    unmerged_artifacts: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def blocks_keep(self) -> bool:
        return self.verdict in BLOCKING_VERDICTS

    @property
    def conclusive(self) -> bool:
        return self.verdict in BLOCKING_VERDICTS | {"served"}

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "hits": self.hits,
            "misses": self.misses,
            "blocks_keep": self.blocks_keep,
            "conclusive": self.conclusive,
            "merged_tables": list(self.merged_tables),
            "unmerged_artifacts": list(self.unmerged_artifacts),
            "detail": self.detail,
        }


def verify_applied(
    server_log: Path | str,
    artifact_paths: list[str] | None = None,
    *,
    hit_logging: bool | None = None,
    runtime_table_names: list[str] | None = None,
) -> ApplyVerdict:
    """Decide whether the tuned artifacts were merged and read."""
    path = Path(server_log)
    if not path.is_file():
        return ApplyVerdict("unknown", detail=f"no serving log at {path}")

    report = parse_log_file(path, hit_logging=hit_logging)
    hits = report["apply_verdict"]["hit"]
    misses = report["apply_verdict"]["miss"]
    merged = report["merged_tables"]
    seen = {Path(p).name for p in (*report["consulted_tables"], *merged)}

    wanted = [str(a) for a in (artifact_paths or []) if str(a).strip()]
    if wanted and seen and not seen & set(runtime_table_names or ()):
        missing = [a for a in wanted if Path(a).name not in seen]
        if len(missing) == len(wanted):
            return ApplyVerdict(
                "not_merged",
                hits,
                misses,
                merged,
                missing,
                detail=(
                    f"none of the {len(wanted)} tuned table(s) appear in what the "
                    f"runtime consulted ({sorted(seen)}); the server loaded its "
                    "bundled defaults"
                ),
            )

    verdict = report["apply_verdict"]["verdict"]
    details = {
        "served": f"{hits} lookup(s) hit the tuned table",
        "zero_hit": f"{misses} lookup(s) with hit logging on, none matched the tuned table",
        "inconclusive_no_hit_logging": (
            "misses logged but hit logging was off or unknown; cannot "
            "distinguish 'never read' from 'not recorded' -- set "
            "AITER_LOG_TUNED_CONFIG=1"
        ),
        "no_lookups": "the server made no tuned-config lookups at all",
    }
    return ApplyVerdict(verdict, hits, misses, merged, detail=details[verdict])
