# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Universal patch-safety contract for specialist worker output.

This is the canonical home for the anti-hallucination guards that apply to
*every* specialist patch, regardless of scope (single domain / cross-domain /
freeform). It absorbs the primitives that previously lived only on the
``dynamic_action`` path:

* unified-diff structural validation (a patch must carry at least one hunk),
* git-grounding (``git apply --check`` against a clean checkout so a fabricated
  patch that does not apply to real source is flagged),
* quantitative-claim guards (forbidden numeric fields + numeric-claim regex on
  the qualitative argument),
* the cross-domain Critic rule descriptors, surfaced when ``scope == 'domains'``.

Pure / dependency-light: imports only stdlib + git via subprocess so it can be
imported from the runner, the Critic backend, and tests without cycles.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Quantitative / priority fields rejected outright on any patch proposal:
# throughput / gain numbers are the Coordinator's measured truth, never a
# self-reported claim from the worker.
FORBIDDEN_PROPOSAL_FIELDS: frozenset[str] = frozenset({
    "expected_gain",
    "expected_gain_pct",
    "bench_evidence",
    "confidence",
    "score",
    "rank",
    "force_provenance",
})


# Numeric speedup claims smuggled into a qualitative argument / summary.
_NUMERIC_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|us|tok/s|qps|tps)\b", re.I),
    re.compile(r"\bspeedup\s*(?:of|=)?\s*\d", re.I),
)

# Unified diff sanity: must contain at least one @@ hunk header.
_UNIFIED_DIFF_HUNK_RE: re.Pattern[str] = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M,
)

# git-only metadata stripped before semantic diff comparison.
_DIFF_NORMALISE_DROP_LINES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+(?: \d+)?$", re.M),
    re.compile(r"^diff --git a/.* b/.*$", re.M),
    re.compile(r"^new file mode \d+$", re.M),
    re.compile(r"^deleted file mode \d+$", re.M),
    re.compile(r"^similarity index \d+%$", re.M),
)


# Patch path within a unified diff (``--- a/<p>`` / ``+++ b/<p>``).
_PATCH_PATH_RE: re.Pattern[str] = re.compile(
    r"^(?:---|\+\+\+) (?:a|b)/(?P<path>.+)$", re.M,
)


# Candidate ``-p`` strip levels for resolving a diff header path to a real file,
# mirroring ``action_executors.integrate_patch._P_LEVELS``. Specialists author
# patches with heterogeneous path prefixes (``a/vllm/...`` -> strip 1,
# ``b/_aiter_ops.py`` -> strip 0/2, full absolute
# ``b/usr/local/lib/python3.12/dist-packages/vllm/...`` -> strip 7), so target
# existence must be probed across levels rather than assuming ``-p1``.
_P_STRIP_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)

# Sentinel the post-/pre-image path takes for a created/deleted file.
_DEV_NULL = "/dev/null"


def _strip_path_prefix(path: str, level: int) -> str:
    """Strip ``level`` leading path components, mimicking ``git apply -p<level>``."""
    if level <= 0:
        return path
    parts = path.split("/")
    if len(parts) <= level:
        return parts[-1]
    return "/".join(parts[level:])


def patch_file_targets(patch_text: str) -> list[tuple[str, str]]:
    """Return ``(old_path, new_path)`` header pairs from a unified diff.

    Paths are the raw ``--- ``/``+++ `` tokens (may carry an ``a/``/``b/``
    prefix, a deep absolute prefix, or the ``/dev/null`` sentinel for a
    created/deleted file). Trailing ``\\t<timestamp>`` is stripped.
    """
    pairs: list[tuple[str, str]] = []
    lines = (patch_text or "").splitlines()
    i = 0
    while i < len(lines):
        if (
            lines[i].startswith("--- ")
            and i + 1 < len(lines)
            and lines[i + 1].startswith("+++ ")
        ):
            old = lines[i][4:].strip().split("\t")[0]
            new = lines[i + 1][4:].strip().split("\t")[0]
            pairs.append((old, new))
            i += 2
        else:
            i += 1
    return pairs


def patch_targets_missing(
    patch_text: str,
    root: Path,
    *,
    strip_levels: tuple[int, ...] = _P_STRIP_LEVELS,
) -> list[str]:
    """Return modify/delete target paths absent from ``root`` at every ``-p`` level.

    A patch hunk that *modifies* or *deletes* an existing file (pre-image path
    is not ``/dev/null``) can never apply if that file is absent from the
    framework source tree — a clear hallucination of the framework layout
    (e.g. patching a CUDA-only file on a ROCm build). Pure file *creations*
    (pre-image ``/dev/null``) are exempt. The returned paths are the raw
    pre-image tokens, suitable for an advisory back to the specialist.
    """
    missing: list[str] = []
    for old, _new in patch_file_targets(patch_text):
        if old == _DEV_NULL:
            continue
        found = False
        for lvl in strip_levels:
            try:
                if (root / _strip_path_prefix(old, lvl)).exists():
                    found = True
                    break
            except OSError:
                continue
        if not found:
            missing.append(old)
    return missing


# Scope literal that triggers the cross-domain Critic rules (mirrors
# specialist_profile.SCOPE_DOMAINS; duplicated here to keep this module
# dependency-light for the Critic backend import).
SCOPE_DOMAINS_LITERAL: str = "domains"


@dataclass(frozen=True)
class CrossDomainRule:
    """One review rule injected into the Critic prompt when a proposal is
    cross-domain (``scope == 'domains'``)."""

    rule_id: str
    description: str
    failure_verdict: str
    failure_reason_code: str


CROSS_DOMAIN_RULES: tuple[CrossDomainRule, ...] = (
    CrossDomainRule(
        rule_id="rationale_per_domain",
        description=(
            "Proposal SHOULD give an independent rationale for each "
            "domain in scope — why this change is necessary within that "
            "domain's boundary."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_rationale_incomplete",
    ),
    CrossDomainRule(
        rule_id="coupling_and_side_effects",
        description=(
            "Proposal SHOULD name the cross-domain coupling points "
            "(why these changes must happen together) AND at least "
            "one potential side effect of the combination."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_coupling_unspecified",
    ),
    CrossDomainRule(
        rule_id="motivation_gap_valid",
        description=(
            "Proposal SHOULD show that no single-domain specialist could "
            "surface this combination within its own-domain prompt. A "
            "simple specialist-A + specialist-B concatenation is a grid "
            "combo (explore grid), not a cross-domain change; advise when "
            "the motivation degenerates so the stack rebench + KEEP "
            "threshold can adjudicate."
        ),
        failure_verdict="advise",
        failure_reason_code="cross_domain_motivation_invalid",
    ),
)


def cross_domain_rule_descriptors() -> list[dict[str, str]]:
    """Return the cross-domain rules as the dict shape the Critic bundle uses."""
    return [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "failure_verdict": r.failure_verdict,
            "failure_reason_code": r.failure_reason_code,
        }
        for r in CROSS_DOMAIN_RULES
    ]


def numeric_claims(text: str) -> list[str]:
    """Return numeric-speedup claim substrings found in ``text``."""
    hits: list[str] = []
    for pattern in _NUMERIC_CLAIM_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits


def is_unified_diff(text: str) -> bool:
    """True iff ``text`` carries at least one unified-diff hunk header."""
    return bool(_UNIFIED_DIFF_HUNK_RE.search(text or ""))


def normalise_diff_for_compare(text: str) -> str:
    """Strip git-only metadata + trailing whitespace for semantic comparison."""
    body = text or ""
    for pat in _DIFF_NORMALISE_DROP_LINES:
        body = pat.sub("", body)
    return "\n".join(
        line.rstrip() for line in body.splitlines() if line.strip()
    )


def patch_escapes_tree(patch_text: str) -> str | None:
    """Return the first offending path that escapes the tree, else ``None``."""
    for hit in _PATCH_PATH_RE.finditer(patch_text or ""):
        cand = hit.group("path").strip()
        if cand.startswith("/") or ".." in Path(cand).parts:
            return cand
    return None


# Patch grounding verdicts.
GROUND_APPLIES = "applies"      # git apply --check succeeded against clean base
GROUND_STALE = "stale"          # valid diff but does not apply to clean base
GROUND_NOT_DIFF = "not_diff"    # not a unified diff (no hunk header)
GROUND_PATH_ESCAPE = "path_escape"  # patch path escapes the tree
GROUND_MISSING_TARGET = "missing_target"  # modify/delete target absent from base
GROUND_UNCHECKED = "unchecked"  # no base available / git unavailable


@dataclass(frozen=True)
class PatchGroundingResult:
    """Outcome of grounding one patch file against a clean checkout."""

    verdict: str
    detail: str = ""

    @property
    def is_garbage(self) -> bool:
        """True for verdicts that should drop the patch (clear hallucination).

        ``missing_target`` joins the structural failures: a patch modifying or
        deleting a file that does not exist in the framework source tree can
        never apply, so it is dropped before it wastes an ``integrate_patch``
        benchmark slot (unlike ``stale``, which is kept because integrate's
        ``-p`` auto-detect / 3-way merge may still salvage it).
        """
        return self.verdict in (
            GROUND_NOT_DIFF, GROUND_PATH_ESCAPE, GROUND_MISSING_TARGET,
        )


def ground_patch_text(
    patch_text: str,
    *,
    base_checkout: Path | None,
    git_timeout_sec: float = 30.0,
) -> PatchGroundingResult:
    """Validate + git-ground one patch.

    Structural checks (unified diff, no path escape) always run. The
    ``git apply --check`` grounding runs only when ``base_checkout`` is a real
    git checkout; otherwise the result is ``GROUND_UNCHECKED`` (advisory, never
    drops the patch) so a missing base never produces a false negative.
    """
    if not is_unified_diff(patch_text):
        return PatchGroundingResult(GROUND_NOT_DIFF, "no unified-diff hunk header")
    escape = patch_escapes_tree(patch_text)
    if escape is not None:
        return PatchGroundingResult(GROUND_PATH_ESCAPE, f"path={escape!r}")
    if base_checkout is None or not Path(base_checkout).is_dir():
        return PatchGroundingResult(GROUND_UNCHECKED, "no base checkout")
    # Hallucinated-layout guard: a modify/delete hunk whose target file is
    # absent from the framework tree can never apply (e.g. patching a
    # CUDA-only file on a ROCm build). Catch it precisely here rather than
    # letting it surface downstream as an opaque ``git_apply_failed``.
    missing = patch_targets_missing(patch_text, Path(base_checkout))
    if missing:
        return PatchGroundingResult(
            GROUND_MISSING_TARGET,
            "target file(s) not in framework tree: " + ", ".join(missing[:5]),
        )
    try:
        proc = subprocess.run(
            ["git", "-C", str(base_checkout), "apply", "--check", "-"],
            input=patch_text if patch_text.endswith("\n") else patch_text + "\n",
            capture_output=True,
            text=True,
            timeout=git_timeout_sec,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PatchGroundingResult(GROUND_UNCHECKED, f"git unavailable: {exc!r}")
    if proc.returncode == 0:
        return PatchGroundingResult(GROUND_APPLIES)
    return PatchGroundingResult(
        GROUND_STALE, (proc.stderr or "").strip()[:240],
    )


@dataclass
class PatchSafetyReport:
    """Aggregate patch-safety findings for one specialist_done payload."""

    kept_patches: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    grounding: dict[str, str] = field(default_factory=dict)
    numeric_warnings: list[str] = field(default_factory=list)
    forbidden_fields: list[str] = field(default_factory=list)

    def notes(self) -> list[str]:
        """Render audit notes for SpecialistRunResult / session_breakdown."""
        out: list[str] = []
        if self.dropped:
            out.append(
                "patch_safety_dropped:"
                + ",".join(f"{d['path']}({d['verdict']})" for d in self.dropped[:8])
            )
        missing = [
            d for d in self.dropped if d.get("verdict") == GROUND_MISSING_TARGET
        ]
        if missing:
            out.append(
                "patch_safety_missing_target:"
                + ",".join(d.get("detail", d["path"]) for d in missing[:4])
                + " — author patches against files that exist in the framework "
                "source tree (inspect it with Glob/Grep first)."
            )
        stale = [p for p, v in self.grounding.items() if v == GROUND_STALE]
        if stale:
            out.append("patch_safety_stale:" + ",".join(stale[:8]))
        if self.numeric_warnings:
            out.append("patch_safety_numeric:" + ",".join(self.numeric_warnings[:8]))
        if self.forbidden_fields:
            out.append(
                "patch_safety_forbidden_fields:" + ",".join(self.forbidden_fields[:8])
            )
        return out


def scan_quantitative_claims(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(forbidden_fields_present, numeric_warning_strings)``.

    Forbidden quantitative fields are a hard signal; numeric claims in the
    summary / qualitative argument are advisory warnings (the Coordinator's
    measured gain is the truth, not the claim).
    """
    forbidden = sorted(set((payload or {}).keys()) & FORBIDDEN_PROPOSAL_FIELDS)
    warnings: list[str] = []
    for key in ("summary", "expected_qualitative_argument", "cross_domain_rationale"):
        hits = numeric_claims(str((payload or {}).get(key) or ""))
        if hits:
            warnings.extend(hits)
    for proposal in (payload or {}).get("proposal_set") or []:
        if not isinstance(proposal, dict):
            continue
        forbidden.extend(
            sorted(set(proposal.keys()) & FORBIDDEN_PROPOSAL_FIELDS)
        )
        hits = numeric_claims(
            str(proposal.get("expected_qualitative_argument") or "")
        )
        if hits:
            warnings.extend(hits)
    # de-dupe, preserve order
    forbidden = list(dict.fromkeys(forbidden))
    warnings = list(dict.fromkeys(warnings))
    return forbidden, warnings


def vet_patches(
    patch_paths: list[str],
    *,
    base_checkout: Path | None,
) -> tuple[list[str], list[dict[str, str]], dict[str, str]]:
    """Ground each patch file; drop clear garbage (non-diff / path escape).

    Returns ``(kept_paths, dropped_records, grounding_by_path)``. Stale-but-valid
    patches are kept (integrate_patch + Critic adjudicate) with a grounding note.
    """
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    grounding: dict[str, str] = {}
    for path in patch_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            dropped.append({"path": path, "verdict": "unreadable", "detail": repr(exc)})
            continue
        res = ground_patch_text(text, base_checkout=base_checkout)
        grounding[path] = res.verdict
        if res.is_garbage:
            dropped.append({"path": path, "verdict": res.verdict, "detail": res.detail})
            continue
        kept.append(path)
    return kept, dropped, grounding


__all__ = [
    "CROSS_DOMAIN_RULES",
    "CrossDomainRule",
    "FORBIDDEN_PROPOSAL_FIELDS",
    "GROUND_APPLIES",
    "GROUND_MISSING_TARGET",
    "GROUND_NOT_DIFF",
    "GROUND_PATH_ESCAPE",
    "GROUND_STALE",
    "GROUND_UNCHECKED",
    "PatchGroundingResult",
    "PatchSafetyReport",
    "SCOPE_DOMAINS_LITERAL",
    "cross_domain_rule_descriptors",
    "ground_patch_text",
    "is_unified_diff",
    "normalise_diff_for_compare",
    "numeric_claims",
    "patch_escapes_tree",
    "patch_file_targets",
    "patch_targets_missing",
    "scan_quantitative_claims",
    "vet_patches",
]
