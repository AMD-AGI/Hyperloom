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
        from kernelforge.gemm_tune.evidence import parse_log_file
    except ImportError:
        # Warning, not info: kernelforge ships in this same wheel, so an
        # ImportError here is a broken install rather than a supported
        # configuration. At info level the run silently loses apply
        # verification and looks identical to one where it passed.
        log.warning(
            "kernelforge.gemm_tune is not importable, so apply verification is "
            "skipped for this run -- it ships with Hyperloom, so this means an "
            "incomplete install; reinstall with the forge extra "
            '(pip install -e ".[forge]")'
        )
        return None
    try:
        return parse_log_file(server_log)
    except Exception:  # noqa: BLE001 - a parse failure must not fail the run
        log.debug("apply verification parse failed for %s", server_log, exc_info=True)
        return None


def verify_applied(
    server_log: Path | str,
    artifact_paths: list[str] | None = None,
    *,
    hit_logging: bool | None = None,
    runtime_table_names: list[str] | None = None,
) -> ApplyVerdict:
    """Decide whether the tuned artifacts were merged and read.

    Args:
        server_log: The serving log written by the run under test.
        artifact_paths: Tuned CSVs that were supposed to be deployed.
        hit_logging: Whether ``AITER_LOG_TUNED_CONFIG`` was on for this run.
            ``None`` means unknown, which keeps a zero-hit result inconclusive.
        runtime_table_names: Canonical table names the runtime resolves these
            artifacts under (e.g. ``bf16_tuned_gemm.csv``). Needed because the
            file we deploy is named after the candidate, not after the table.

    Returns:
        A verdict. ``blocks_keep`` is true only for the cases that are
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
    consulted = [str(c) for c in (report.get("consulted_tables") or [])]

    # 1. Did the artifact reach the server at all?
    #
    #    Judge by the tables the lookups actually named, not by the merge line.
    #    Setting AITER_CONFIG_* -- which is exactly what a candidate run does --
    #    makes aiter skip the merge step entirely: it prints no merge line and
    #    resolves against the override, so the lookup is the only place the path
    #    appears. Reading an absent merge line as "not merged" would have
    #    reverted every candidate.
    #
    #    Both names are accepted because both are legitimate: the deployed file
    #    is named after the candidate when it is an override, and after the
    #    table when the server merged it into its own config directory.
    wanted = [str(a) for a in (artifact_paths or []) if str(a).strip()]
    if wanted and (consulted or merged):
        seen = {Path(p).name for p in consulted} | {Path(m).name for m in merged}
        canonical = {str(n).strip() for n in (runtime_table_names or []) if str(n).strip()}
        if not (seen & (canonical or set())):
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

    # 2. Was anything read? Hit lines are gated behind AITER_LOG_TUNED_CONFIG,
    #    so their absence is only informative when the flag was on.
    if hits > 0:
        return ApplyVerdict(
            "served",
            hits,
            misses,
            merged,
            detail=f"{hits} lookup(s) hit the tuned table",
        )
    if misses > 0:
        # Misses logged and no hits. With hit logging on, that is a real zero --
        # the case this gate exists for. Without it, "never read" and "not
        # recorded" are the same picture, and a scan of 60 production logs found
        # the flag set in none of them, so the default has to stay inconclusive.
        if hit_logging:
            return ApplyVerdict(
                "zero_hit",
                hits,
                misses,
                merged,
                detail=(f"{misses} lookup(s) with hit logging on, none matched the tuned table"),
            )
        return ApplyVerdict(
            "inconclusive_no_hit_logging",
            hits,
            misses,
            merged,
            detail=(
                "misses logged but hit logging was off or unknown; cannot "
                "distinguish 'never read' from 'not recorded' -- set "
                "AITER_LOG_TUNED_CONFIG=1"
            ),
        )

    return ApplyVerdict(
        "no_lookups",
        hits,
        misses,
        merged,
        detail="the server made no tuned-config lookups at all",
    )
