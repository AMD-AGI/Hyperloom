# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Universal patch-safety contract for specialist worker output.

This is the canonical home for the anti-hallucination guards that apply to
*every* specialist patch, regardless of scope (single domain / cross-domain /
freeform). It provides:

* unified-diff structural validation (a patch must carry at least one hunk),
* git-grounding (``git apply --check`` against a clean checkout so a fabricated
  patch that does not apply to real source is flagged),
* quantitative-claim guards (forbidden numeric fields, stripped rather than
  merely reported, + numeric-claim regex on the qualitative argument),
* the cross-domain Critic rule descriptors, surfaced when ``scope == 'domains'``.

Pure / dependency-light: imports only stdlib + git via subprocess so it can be
imported from the runner, the Critic backend, and tests without cycles.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


# Quantitative / priority fields rejected outright on any patch proposal:
# throughput / gain numbers are the Coordinator's measured truth, never a
# self-reported claim from the worker.
#
# The ban is scoped to specialist-authored output by where it is applied, not
# by what it lists: ``strip_forbidden_proposal_fields`` runs on the specialist
# exit payload alone. ``predicted_gain_pct`` therefore belongs here even though
# it is a *required* field of a ``propose_action`` intent -- there the number
# is the Coordinator's estimate, in a specialist's ``proposal_set`` it is the
# same self-reported claim as ``expected_gain_pct`` under a different name.
FORBIDDEN_PROPOSAL_FIELDS: frozenset[str] = frozenset(
    {
        "expected_gain",
        "expected_gain_pct",
        "predicted_gain_pct",
        "bench_evidence",
        "confidence",
        "score",
        "rank",
        "force_provenance",
    }
)

# The same guard at the payload's top level, where ``confidence`` means
# something else: the output schema asks for a round-level self-assessment and
# the specialist-round audit rows record it. That is not a per-proposal gain
# claim and cannot bias which variant gets benched, so banning it here only
# made the schema contradict itself -- the guard's own scope, per the Critic
# rules, is ``proposal_set[*]``.
FORBIDDEN_PAYLOAD_FIELDS: frozenset[str] = FORBIDDEN_PROPOSAL_FIELDS - {"confidence"}


# Numeric speedup claims smuggled into a qualitative argument / summary.
_NUMERIC_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    re.compile(r"\b\d+(?:\.\d+)?\s*x\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|us|tok/s|qps|tps)\b", re.I),
    re.compile(r"\bspeedup\s*(?:of|=)?\s*\d", re.I),
)

# Unified diff sanity: must contain at least one @@ hunk header.
_UNIFIED_DIFF_HUNK_RE: re.Pattern[str] = re.compile(
    r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@",
    re.M,
)

# Patch path within a unified diff (``--- a/<p>`` / ``+++ b/<p>``).
_PATCH_PATH_RE: re.Pattern[str] = re.compile(
    r"^(?:---|\+\+\+) (?:a|b)/(?P<path>.+)$",
    re.M,
)


# Candidate ``-p`` strip levels for resolving a diff header path to a real file.
# Specialists author patches with heterogeneous path prefixes, so target
# existence is probed across levels rather than assuming ``-p1``.
_P_STRIP_LEVELS: tuple[int, ...] = (1, 0, 2, 3, 4, 5, 6, 7, 8)

# Sentinel the post-/pre-image path takes for a created/deleted file.
_DEV_NULL = "/dev/null"


@dataclass(frozen=True)
class ParsedPatchTargets:
    """Safe repo-relative targets split by whether they must already exist."""

    existing: tuple[str, ...]
    created: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.existing, *self.created)))


def _safe_patch_path(raw: str) -> str:
    value = str(raw or "").strip().split("\t", 1)[0]
    if value in {"", _DEV_NULL}:
        return value
    if value.startswith(("a/", "b/")):
        value = value[2:]
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
        raise ValueError(f"unsafe patch target path: {raw!r}")
    return parsed.as_posix()


def parse_patch_targets(patch_text: str) -> ParsedPatchTargets:
    """Parse unified-diff targets, including create/delete/rename/mode-only.

    Standard ``---``/``+++`` pairs are authoritative. If a patch has no such
    pairs (for example a mode-only or metadata-only rename), ``diff --git``
    headers provide the fallback paths.
    """
    pairs = patch_file_targets(patch_text)
    if not pairs:
        for line in (patch_text or "").splitlines():
            if not line.startswith("diff --git "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                pairs.append((parts[2], parts[3]))

    existing: list[str] = []
    created: list[str] = []
    for raw_old, raw_new in pairs:
        old = _safe_patch_path(raw_old)
        new = _safe_patch_path(raw_new)
        if old and old != _DEV_NULL and old not in existing:
            existing.append(old)
        if new and new != _DEV_NULL and new != old and new not in created:
            created.append(new)
        elif old == _DEV_NULL and new and new not in created:
            created.append(new)
    if not existing and not created:
        raise ValueError("patch declares no safe target files")
    return ParsedPatchTargets(tuple(existing), tuple(created))


def _strip_path_prefix(path: str, level: int) -> str:
    """Strip ``level`` leading path components, mimicking ``git apply -p<level>``.

    Args:
        path: The diff header path to strip.
        level: Number of leading components to drop (``<= 0`` is a no-op).

    Returns:
        The path with ``level`` leading components removed (basename floor).
    """
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

    Args:
        patch_text: The unified-diff text to scan.

    Returns:
        The list of ``(old_path, new_path)`` header pairs.
    """
    pairs: list[tuple[str, str]] = []
    lines = (patch_text or "").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
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

    Args:
        patch_text: The unified-diff text to scan.
        root: Framework source-tree root the targets are probed against.
        strip_levels: ``-p`` strip levels to try when resolving each path.

    Returns:
        The raw pre-image token paths absent from ``root`` at every level.
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


# Scope literal that triggers the cross-domain Critic rules. Duplicated from
# specialists.profile.SCOPE_DOMAINS to keep this module dependency-light.
SCOPE_DOMAINS_LITERAL: str = "domains"

# The verdict a rule declares when its violation is advisory: the proposal still
# reaches the Coordinator, carrying the reason code as a note. Spelled once so a
# typo in one rule cannot quietly drop it from
# :func:`advisory_only_reason_codes` and re-arm the reject it asked to avoid.
ADVISE_VERDICT: str = "advise"


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
        failure_verdict=ADVISE_VERDICT,
        failure_reason_code="cross_domain_rationale_incomplete",
    ),
    CrossDomainRule(
        rule_id="coupling_and_side_effects",
        description=(
            "Proposal SHOULD name the cross-domain coupling points "
            "(why these changes must happen together) AND at least "
            "one potential side effect of the combination."
        ),
        failure_verdict=ADVISE_VERDICT,
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
        failure_verdict=ADVISE_VERDICT,
        failure_reason_code="cross_domain_motivation_invalid",
    ),
)


def cross_domain_rule_descriptors() -> list[dict[str, str]]:
    """Return the cross-domain rules as the dict shape the Critic bundle uses.

    Returns:
        One dict per cross-domain rule with ``rule_id`` / ``description`` /
        ``failure_verdict`` / ``failure_reason_code`` keys.
    """
    return [
        {
            "rule_id": r.rule_id,
            "description": r.description,
            "failure_verdict": r.failure_verdict,
            "failure_reason_code": r.failure_reason_code,
        }
        for r in CROSS_DOMAIN_RULES
    ]


# Audit code the Critic cites when a self-reported gain field reaches review.
QUANTITATIVE_CLAIM_REASON_CODE: str = "specialist_quantitative_claim_violation"


def quantitative_claim_rule_descriptor() -> dict[str, Any]:
    """Return the self-reported-gain rule in the shape the Critic bundle uses.

    Single-sources the field list from :data:`FORBIDDEN_PROPOSAL_FIELDS` so the
    Critic's copy cannot drift from the one the runner enforces, and carries
    ``advise`` as the verdict: the fields are stripped before review, so one
    arriving anyway is a format problem, and rejecting over format costs the
    round every proposal in the set.

    Returns:
        A ``rule_id`` / ``description`` / ``forbidden_proposal_fields`` /
        ``failure_verdict`` / ``failure_reason_code`` dict.
    """
    return {
        "rule_id": "no_self_reported_gain",
        "description": (
            "proposal_set[*] must not carry a self-reported gain, priority or "
            "confidence field: measured gain is the Coordinator's, never the "
            "worker's claim. These fields are stripped from specialist output "
            "before review, so treat any that still reach you -- including an "
            "equivalent smuggled under another name -- as advisory: ignore the "
            "field and judge the proposal on its merits."
        ),
        "forbidden_proposal_fields": sorted(FORBIDDEN_PROPOSAL_FIELDS),
        "failure_verdict": ADVISE_VERDICT,
        "failure_reason_code": QUANTITATIVE_CLAIM_REASON_CODE,
    }


def advisory_only_reason_codes() -> frozenset[str]:
    """Return the reason codes whose owning rule asked for ``advise``, not ``reject``.

    Derived from the descriptors the Critic is actually handed rather than
    restated, so a rule that changes its ``failure_verdict`` cannot leave a
    stale entry behind. Lets the verdict path hold a ``reject`` citing one of
    these to the verdict its own rule declared: every rule here is a format or
    strategy hint, and a reject costs the round every proposal in the set.

    Returns:
        The ``failure_reason_code`` of every rule declaring
        ``failure_verdict == "advise"``.
    """
    descriptors: list[dict[str, Any]] = [quantitative_claim_rule_descriptor()]
    descriptors.extend(cross_domain_rule_descriptors())
    codes = {
        str(d.get("failure_reason_code") or "").strip()
        for d in descriptors
        if str(d.get("failure_verdict") or "").strip() == ADVISE_VERDICT
    }
    codes.discard("")
    return frozenset(codes)


# The proposal kinds the advisory rules speak about. Both rule families are
# about a specialist-authored payload -- ``proposal_set[*]`` for the
# quantitative-claim rule, ``scope=domains`` for the cross-domain ones -- which
# reaches review as a ``specialist`` proposal or as the ``explore`` grid that
# ``proposal_set`` is materialised into. ``framework_agent`` is here because the
# quantitative-claim rule names it by exception ("never fire the rule on them",
# see prompts/critic.md): its payload always carries ``predicted_gain_pct``, so
# a verdict citing the rule there is a misapplication of the rule itself.
#
# Spelled out rather than derived: no ACTION_CATALOGUE field separates these
# from ``integrate_patch``, which shares their ``exploration`` verdict class,
# ``shallow`` family and ``workspace_write`` side effect while being the one
# action whose materialisation lands the patch under review.
ADVISORY_RULE_PROPOSAL_KINDS: frozenset[str] = frozenset(
    {
        "explore",
        "framework_agent",
        "specialist",
    }
)


def advisory_rules_govern(action_name: str) -> bool:
    """Return whether the advisory review rules speak about ``action_name``.

    Args:
        action_name: The proposed action's name.

    Returns:
        True when the action is one of :data:`ADVISORY_RULE_PROPOSAL_KINDS`.
    """
    return str(action_name or "").strip() in ADVISORY_RULE_PROPOSAL_KINDS


def numeric_claims(text: str) -> list[str]:
    """Return numeric-speedup claim substrings found in ``text``.

    Args:
        text: The free-text argument/summary to scan.

    Returns:
        The matched numeric-claim substrings (may be empty).
    """
    hits: list[str] = []
    for pattern in _NUMERIC_CLAIM_PATTERNS:
        for match in pattern.finditer(text or ""):
            hits.append(match.group(0))
    return hits


def is_unified_diff(text: str) -> bool:
    """True iff ``text`` carries at least one unified-diff hunk header.

    Args:
        text: The candidate diff text.

    Returns:
        True when at least one ``@@`` hunk header is present.
    """
    return bool(_UNIFIED_DIFF_HUNK_RE.search(text or ""))


def patch_escapes_tree(patch_text: str) -> str | None:
    """Return the first offending path that escapes the tree, else ``None``.

    Args:
        patch_text: The unified-diff text to scan.

    Returns:
        The first absolute or ``..``-containing path, or ``None`` when none
        escape the tree.
    """
    for hit in _PATCH_PATH_RE.finditer(patch_text or ""):
        cand = hit.group("path").strip()
        if cand.startswith("/") or ".." in Path(cand).parts:
            return cand
    return None


# Patch grounding verdicts.
GROUND_APPLIES = "applies"  # git apply --check succeeded against clean base
GROUND_STALE = "stale"  # valid diff but does not apply to clean base
GROUND_NOT_DIFF = "not_diff"  # not a unified diff (no hunk header)
GROUND_PATH_ESCAPE = "path_escape"  # patch path escapes the tree
GROUND_MISSING_TARGET = "missing_target"  # modify/delete target absent from base
GROUND_UNCHECKED = "unchecked"  # no base available / git unavailable


@dataclass(frozen=True)
class PatchGroundingResult:
    """Outcome of grounding one patch file against a clean checkout."""

    verdict: str
    detail: str = ""
    # The allowlisted root that the patch was successfully grounded against, or
    # empty when the verdict is not ``applies`` / ``stale``.
    root: str = ""

    @property
    def is_garbage(self) -> bool:
        """True for verdicts that should drop the patch (clear hallucination).

        ``missing_target`` joins the structural failures: a patch modifying or
        deleting a file that does not exist in the framework source tree can
        never apply, so it is dropped before it wastes an ``integrate_patch``
        benchmark slot (unlike ``stale``, which is kept because integrate's
        ``-p`` auto-detect / 3-way merge may still salvage it).

        Returns:
            True for structural-failure verdicts that should drop the patch.
        """
        return self.verdict in (
            GROUND_NOT_DIFF,
            GROUND_PATH_ESCAPE,
            GROUND_MISSING_TARGET,
        )


def ground_patch_text(
    patch_text: str,
    *,
    base_checkout: Path | None,
    candidate_roots: tuple[Path, ...] = (),
    git_timeout_sec: float = 30.0,
) -> PatchGroundingResult:
    """Validate + git-ground one patch.

    Structural checks (unified diff, no path escape) always run.  The
    ``git apply --check`` grounding runs against ``base_checkout`` first; when
    that root returns ``missing_target`` and ``candidate_roots`` is non-empty,
    each candidate root is tried in order before the patch is dropped.  This
    handles the common case where a specialist writes a sglang patch from an
    aiter worktree: the patch targets sglang files, so grounding must use the
    sglang checkout.

    Args:
        patch_text: The unified-diff text to validate and ground.
        base_checkout: Primary clean git checkout to ground against, or
            ``None`` to skip the ``git apply --check`` step.
        candidate_roots: Additional allowlisted checkouts to retry against
            when the primary root returns ``missing_target``.  Tried in order;
            the first that either finds all targets or lets ``git apply --check``
            pass is used.
        git_timeout_sec: Timeout for each ``git apply --check`` subprocess.

    Returns:
        The :class:`PatchGroundingResult` with the verdict and detail.
        ``result.root`` is the checkout path (as a str) when the verdict is
        ``applies`` or ``stale``; empty otherwise.
    """
    if not is_unified_diff(patch_text):
        return PatchGroundingResult(GROUND_NOT_DIFF, "no unified-diff hunk header")
    escape = patch_escapes_tree(patch_text)
    if escape is not None:
        return PatchGroundingResult(GROUND_PATH_ESCAPE, f"path={escape!r}")
    if base_checkout is None or not Path(base_checkout).is_dir():
        return PatchGroundingResult(GROUND_UNCHECKED, "no base checkout")

    def _try_root(root: Path) -> PatchGroundingResult | None:
        """Attempt grounding against one root.  Returns None to keep trying."""
        missing = patch_targets_missing(patch_text, root)
        if missing:
            return None  # targets not in this tree; try the next root
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), "apply", "--check", "-"],
                input=patch_text if patch_text.endswith("\n") else patch_text + "\n",
                capture_output=True,
                text=True,
                timeout=git_timeout_sec,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return PatchGroundingResult(GROUND_UNCHECKED, f"git unavailable: {exc!r}")
        if proc.returncode == 0:
            return PatchGroundingResult(GROUND_APPLIES, root=str(root))
        return PatchGroundingResult(GROUND_STALE, (proc.stderr or "").strip()[:240], root=str(root))

    # Try primary root first.
    result = _try_root(Path(base_checkout))
    if result is not None:
        return result

    # Primary root has missing targets.  Walk candidate roots.
    for alt in candidate_roots:
        if not alt.is_dir() or alt.resolve() == Path(base_checkout).resolve():
            continue
        result = _try_root(alt)
        if result is not None:
            return result

    # All roots have missing targets — hallucinated layout.
    missing = patch_targets_missing(patch_text, Path(base_checkout))
    return PatchGroundingResult(
        GROUND_MISSING_TARGET,
        "target file(s) not in any framework tree: " + ", ".join(missing[:5]),
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
        """Render audit notes for SpecialistRunResult / session_breakdown.

        Returns:
            Human-readable audit note strings for the recorded findings.
        """
        out: list[str] = []
        if self.dropped:
            out.append("patch_safety_dropped:" + ",".join(f"{d['path']}({d['verdict']})" for d in self.dropped[:8]))
        missing = [d for d in self.dropped if d.get("verdict") == GROUND_MISSING_TARGET]
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
            out.append("patch_safety_forbidden_fields:" + ",".join(self.forbidden_fields[:8]))
        return out


def scan_numeric_claims(payload: dict[str, Any]) -> list[str]:
    """Return the numeric speedup claims smuggled into ``payload``'s prose.

    A number in a summary or qualitative argument is advisory: the Coordinator's
    measured gain is the truth, not the claim, and the audit note this feeds is
    how a smuggled one stays visible.

    The forbidden *fields* are a different question, and
    :func:`strip_forbidden_proposal_fields` is the one place that answers it: it
    removes them and returns what it took. Answering it a second time here would
    be a copy of that same ``keys & FORBIDDEN_*`` intersection with nothing
    holding the two in step.

    Args:
        payload: The specialist_done payload to scan.

    Returns:
        The matched numeric-claim substrings, de-duped with order preserved.
    """
    warnings: list[str] = []
    for key in ("summary", "expected_qualitative_argument", "cross_domain_rationale"):
        warnings.extend(numeric_claims(str((payload or {}).get(key) or "")))
    for proposal in (payload or {}).get("proposal_set") or []:
        if not isinstance(proposal, dict):
            continue
        warnings.extend(numeric_claims(str(proposal.get("expected_qualitative_argument") or "")))
    return list(dict.fromkeys(warnings))


def strip_forbidden_proposal_fields(payload: dict[str, Any]) -> list[str]:
    """Remove the forbidden quantitative keys from ``payload`` in place.

    Uses :data:`FORBIDDEN_PAYLOAD_FIELDS` at the top level and
    :data:`FORBIDDEN_PROPOSAL_FIELDS` on each ``proposal_set`` entry.

    Detecting a self-reported gain number and then forwarding it is what turns a
    format slip into a lost round: the Critic is told to reject the whole
    ``proposal_set`` over it, so the specialist's ideas never reach a benchmark
    and there is rarely budget to resubmit. The claim is worthless either way —
    measured gain is the Coordinator's — so dropping it costs nothing and makes
    the violation unreachable rather than merely audited. The names returned are
    what the caller's audit note records.

    Args:
        payload: The ``specialist_done`` payload, mutated in place. Both the
            top level and each ``proposal_set`` entry are cleaned.

    Returns:
        The removed field names, de-duped with first-seen order preserved.
    """
    if not isinstance(payload, dict):
        return []
    removed: list[str] = []
    for key in sorted(set(payload.keys()) & FORBIDDEN_PAYLOAD_FIELDS):
        payload.pop(key, None)
        removed.append(key)
    for proposal in payload.get("proposal_set") or []:
        if not isinstance(proposal, dict):
            continue
        for key in sorted(set(proposal.keys()) & FORBIDDEN_PROPOSAL_FIELDS):
            proposal.pop(key, None)
            removed.append(key)
    return list(dict.fromkeys(removed))


def vet_patches(
    patch_paths: list[str],
    *,
    base_checkout: Path | None,
    candidate_roots: tuple[Path, ...] = (),
) -> tuple[list[str], list[dict[str, str]], dict[str, str], dict[str, str]]:
    """Ground each patch file; drop clear garbage (non-diff / path escape).

    Stale-but-valid patches are kept (integrate_patch + Critic adjudicate)
    with a grounding note.  When a patch's primary root has missing targets,
    ``candidate_roots`` are tried in order so cross-repo patches (e.g. a sglang
    fix written from an aiter worktree) are rescued instead of silently dropped.

    Args:
        patch_paths: File paths of the candidate patches to vet.
        base_checkout: Primary clean git checkout to ground against, or
            ``None``.
        candidate_roots: Additional allowlisted checkouts to retry against
            for patches that fail ``missing_target`` on ``base_checkout``.

    Returns:
        A ``(kept_paths, dropped_records, grounding_by_path, patch_roots)``
        tuple.  ``patch_roots`` maps each kept patch path to the checkout
        root string where it was successfully grounded.
    """
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    grounding: dict[str, str] = {}
    patch_roots: dict[str, str] = {}
    for path in patch_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            dropped.append({"path": path, "verdict": "unreadable", "detail": repr(exc)})
            continue
        res = ground_patch_text(text, base_checkout=base_checkout, candidate_roots=candidate_roots)
        grounding[path] = res.verdict
        if res.is_garbage:
            dropped.append({"path": path, "verdict": res.verdict, "detail": res.detail})
            continue
        kept.append(path)
        if res.root:
            patch_roots[path] = res.root
    return kept, dropped, grounding, patch_roots


__all__ = [
    "ADVISE_VERDICT",
    "ADVISORY_RULE_PROPOSAL_KINDS",
    "CROSS_DOMAIN_RULES",
    "CrossDomainRule",
    "FORBIDDEN_PAYLOAD_FIELDS",
    "FORBIDDEN_PROPOSAL_FIELDS",
    "GROUND_APPLIES",
    "GROUND_MISSING_TARGET",
    "GROUND_NOT_DIFF",
    "GROUND_PATH_ESCAPE",
    "GROUND_STALE",
    "GROUND_UNCHECKED",
    "PatchGroundingResult",
    "PatchSafetyReport",
    "QUANTITATIVE_CLAIM_REASON_CODE",
    "SCOPE_DOMAINS_LITERAL",
    "advisory_only_reason_codes",
    "advisory_rules_govern",
    "cross_domain_rule_descriptors",
    "ground_patch_text",
    "is_unified_diff",
    "numeric_claims",
    "patch_escapes_tree",
    "patch_file_targets",
    "patch_targets_missing",
    "quantitative_claim_rule_descriptor",
    "scan_numeric_claims",
    "strip_forbidden_proposal_fields",
    "vet_patches",
    "parse_patch_targets",
]
