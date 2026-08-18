# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check whether a tuned artifact was actually read by the server.

Three things being true at once -- the artifact exists, the env var is set, and
e2e throughput went up -- still does not mean the tuning did anything. Two
independent ways for it to mean nothing have both been observed:

* **the keys are unreachable**: the table has rows, but none of them match the
  (M, N, K, ...) the runtime asks for;
* **the table never arrived**: the artifact was written, but the merge step did
  not pick it up and the server loaded its bundled default.

Neither is a tuner-selection problem, so no amount of choosing the right tuner
detects them. This is a separate, deterministic check on the serving log.

The one trap it has to avoid: aiter logs a *miss* unconditionally but a *hit*
only when ``AITER_LOG_TUNED_CONFIG=1``. Reading "no hit lines" as "zero hits"
would mark every arm that ran without the flag as a failed apply -- and a scan
of 60 production logs found the flag set in none of them. So "we cannot tell"
is a distinct verdict from "it was not used", and only the latter reverts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

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
        return self.verdict in {"served", "not_merged", "zero_hit"}

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


def _parse(server_log: Path) -> dict | None:
    """Parse the serving log with forge's evidence module, if it is installed."""
    try:
        from forge_gemm_tune.evidence import parse_log_file
    except ImportError:
        log.info("forge_gemm_tune not importable; apply verification unavailable")
        return None
    try:
        return parse_log_file(server_log)
    except Exception:  # noqa: BLE001 - a parse failure must not fail the run
        log.debug("apply verification parse failed for %s", server_log, exc_info=True)
        return None


def verify_applied(
    server_log: Path | str,
    artifact_paths: list[str] | None = None,
) -> ApplyVerdict:
    """Decide whether the tuned artifacts were merged and read.

    Args:
        server_log: The serving log written by the run under test.
        artifact_paths: Tuned CSVs that were supposed to be deployed. Checked by
            basename against the merge line, because the server copies tables
            into its own config directory before loading them.

    Returns:
        A verdict. ``blocks_keep`` is true only for the two cases that are
        positively wrong; everything else, including "cannot tell", leaves the
        decision to the caller.
    """
    path = Path(server_log)
    if not path.is_file():
        return ApplyVerdict("unknown", detail=f"no serving log at {path}")

    report = _parse(path)
    if report is None:
        return ApplyVerdict("unknown", detail="log parser unavailable")

    av = report.get("apply_verdict") or {}
    hits = int(av.get("hit") or 0)
    misses = int(av.get("miss") or 0)
    merged = [str(m) for m in (report.get("merged_tables") or [])]

    # 1. Did the artifact reach the server at all? Compare basenames: the merge
    #    step copies tables into the server's own config dir, so the deployed
    #    path is not the path we wrote.
    wanted = [str(a) for a in (artifact_paths or []) if str(a).strip()]
    if wanted and merged:
        merged_names = {Path(m).name for m in merged}
        missing = [a for a in wanted if Path(a).name not in merged_names]
        if missing:
            return ApplyVerdict(
                "not_merged", hits, misses, merged, missing,
                detail=(
                    f"{len(missing)} tuned table(s) absent from the server's merge list; "
                    "the server loaded its bundled defaults"
                ),
            )

    # 2. Was anything read? Hit lines are gated behind AITER_LOG_TUNED_CONFIG,
    #    so their absence is only informative when misses were logged too.
    if hits > 0:
        return ApplyVerdict(
            "served", hits, misses, merged,
            detail=f"{hits} lookup(s) hit the tuned table",
        )
    if misses > 0:
        # Misses logged, no hits: either genuinely nothing matched, or hit
        # logging was off. Only the parser knows which, and it says so.
        if str(av.get("verdict")) == "inconclusive_no_hit_logging":
            return ApplyVerdict(
                "inconclusive_no_hit_logging", hits, misses, merged,
                detail=(
                    "misses logged but hit logging was off; cannot distinguish "
                    "'never read' from 'not recorded' -- set AITER_LOG_TUNED_CONFIG=1"
                ),
            )
        return ApplyVerdict(
            "zero_hit", hits, misses, merged,
            detail=f"{misses} lookup(s), none matched the tuned table",
        )

    return ApplyVerdict(
        "no_lookups", hits, misses, merged,
        detail="the server made no tuned-config lookups at all",
    )
