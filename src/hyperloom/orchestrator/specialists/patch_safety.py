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
from collections.abc import Sequence
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

#: The one absolute header a legitimate diff carries: the missing side of an
#: add or delete. Never treated as an escape.
_DEV_NULL_PATHS: frozenset[str] = frozenset({"/dev/null", "dev/null"})


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


@dataclass(frozen=True)
class PatchRootResolution:
    """One unambiguous checkout selected from Patch pre-image targets.

    Attributes:
        root: The resolved checkout, or ``None`` when the set fails closed.
        reason: Why resolution failed; one of :data:`PATCH_ROOT_FAIL_REASONS`.
            Empty on success.
        matches: Every candidate that held the whole set. Populated only for
            ``ambiguous_root``, where naming the collision is the diagnosis.
    """

    root: Path | None
    reason: str = ""
    matches: tuple[Path, ...] = ()


#: Why :func:`resolve_patch_apply_root` refused to name a root. Callers map
#: these onto their own vocabulary (warm replay rewrites two of them into
#: allowlist-flavoured reasons) and persist them, so they are a wire contract.
PATCH_ROOT_FAIL_REASONS: tuple[str, ...] = (
    "explicit_root_invalid",  # declared root is unreadable or not a directory
    "explicit_root_target_mismatch",  # declared root lacks a pre-image target
    "patch_targets_invalid",  # a diff names no safe target path
    "patch_content_missing",  # no readable diff text to match a root against
    "no_candidate_roots",  # no tree offered at all -- says nothing about the patch
    "pure_create_requires_explicit_root",  # no pre-image can identify a root
    "no_matching_root",  # no candidate holds every pre-image
    "ambiguous_root",  # more than one candidate holds every pre-image
)

# Reasons that report an absent tree rather than a fact about the patch: with no
# pre-image, or nothing to match it against, there is no hallucinated target to
# catch, so vetting defers to the applying caller.
_GROUNDING_UNDECIDABLE_REASONS: frozenset[str] = frozenset(
    {
        "no_candidate_roots",
        "pure_create_requires_explicit_root",
    }
)


def _normalize_patch_path(raw: str) -> str:
    """Strip the header decoration ``git apply -p1`` drops, without judging it."""
    value = str(raw or "").strip().split("\t", 1)[0]
    if value in {"", _DEV_NULL}:
        return value
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def _safe_patch_path(raw: str) -> str:
    value = _normalize_patch_path(raw)
    if value in {"", _DEV_NULL}:
        return value
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
    pairs = patch_file_targets(patch_text)
    existing = [old for old, _new in pairs if old != _DEV_NULL]
    if not pairs:
        try:
            existing = list(parse_patch_targets(patch_text).existing)
        except ValueError:
            return ["<invalid>"]
    missing: list[str] = []
    for old in existing:
        found = False
        for lvl in strip_levels:
            try:
                stripped = _strip_path_prefix(old, lvl)
                # A bare filename matches any root holding that name, so deep
                # strips of a nested path would implicate unrelated repos.
                if "/" not in stripped and lvl > 1:
                    continue
                if (root / stripped).exists():
                    found = True
                    break
            except OSError:
                continue
        if not found:
            missing.append(old)
    return missing


def _collapse_nested_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep only the outermost of any nested match, leaving disjoint ones alone.

    An editable install puts a package parent inside its own checkout, so
    ``/sgl-workspace/sglang`` and ``/sgl-workspace/sglang/python`` both hold a
    ``python/sglang/...`` target at different strip levels. They are one tree,
    and the outer root is the one whose strip level matches ``git diff`` output.

    Args:
        roots: Match candidates from :func:`resolve_patch_apply_root`.

    Returns:
        Roots with any descendant of another entry removed.
    """
    resolved = {root: root.resolve() for root in roots}
    return tuple(
        root
        for root in roots
        if not any(other != resolved[root] and other in resolved[root].parents for other in resolved.values())
    )


def resolve_patch_apply_root(
    patch_texts: Sequence[str],
    *,
    explicit_root: Path | None,
    candidate_roots: Sequence[Path] = (),
    default_root: Path | None = None,
) -> PatchRootResolution:
    """Resolve one checkout under the shared Enablement/warm-replay rules.

    The whole set resolves together, so a set split across two checkouts is
    refused rather than half-applied. An explicit root is authoritative. Failing
    that, the pre-images must single out exactly one candidate: zero and several
    both fail closed, because guessing here mutates a checkout the patch was
    never written against.

    A create-only set carries no pre-image, so no candidate can be matched and
    only a root the caller already knows will do -- an explicit one, or the
    ``default_root`` a caller supplies when it has independent grounds for it,
    such as the checkout a specialist's worktree was cut from.

    When pre-images do exist but no candidate was offered, the answer is
    ``no_candidate_roots`` rather than a miss, because absence of a tree is not
    evidence about the patch. Callers decide what that means for them: a vetting
    gate declines to judge, an applying one still refuses.

    Args:
        patch_texts: The diffs to place. Blank entries are ignored.
        explicit_root: The checkout the caller declared, if any.
        candidate_roots: Checkouts to match the pre-images against when no
            explicit root is declared.
        default_root: The checkout to use for a create-only set. Never
            consulted while a pre-image can pick a candidate.

    Returns:
        A :class:`PatchRootResolution` naming the checkout, or carrying one of
        :data:`PATCH_ROOT_FAIL_REASONS`.
    """
    texts = tuple(str(text or "") for text in patch_texts if str(text or "").strip())
    resolved_explicit: Path | None = None
    if explicit_root is not None:
        try:
            resolved_explicit = Path(explicit_root).resolve()
        except (OSError, RuntimeError):
            return PatchRootResolution(None, "explicit_root_invalid")
        if not resolved_explicit.is_dir():
            return PatchRootResolution(None, "explicit_root_invalid")
        if not texts:
            return PatchRootResolution(resolved_explicit)

    parsed: list[ParsedPatchTargets] = []
    try:
        parsed = [parse_patch_targets(text) for text in texts]
    except ValueError:
        return PatchRootResolution(None, "patch_targets_invalid")
    if not parsed:
        return PatchRootResolution(None, "patch_content_missing")

    has_existing = any(targets.existing for targets in parsed)
    if resolved_explicit is not None:
        if any(patch_targets_missing(text, resolved_explicit) for text in texts):
            return PatchRootResolution(None, "explicit_root_target_mismatch")
        return PatchRootResolution(resolved_explicit)
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_roots:
        try:
            root = Path(candidate).resolve()
        except (OSError, RuntimeError):
            continue
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        roots.append(root)

    resolved_default: Path | None = None
    if default_root is not None:
        try:
            candidate_default = Path(default_root).resolve()
        except (OSError, RuntimeError):
            candidate_default = None
        if candidate_default is not None and candidate_default.is_dir():
            resolved_default = candidate_default

    # Settled before the candidates are considered at all: a create-only set has
    # no pre-image, so whether any candidate exists says nothing about it.
    if not has_existing:
        if resolved_default is None:
            return PatchRootResolution(None, "pure_create_requires_explicit_root")
        return PatchRootResolution(resolved_default)

    # Nothing to match the pre-images against is not evidence against the patch.
    # Say so separately so a vetting caller can decline to judge while an
    # applying caller still refuses to write into a tree it cannot name.
    if not roots:
        return PatchRootResolution(None, "no_candidate_roots")

    matches = tuple(root for root in roots if not any(patch_targets_missing(text, root) for text in texts))
    if not matches:
        return PatchRootResolution(None, "no_matching_root")
    if len(matches) > 1:
        matches = _collapse_nested_roots(matches)
    if len(matches) > 1:
        return PatchRootResolution(None, "ambiguous_root", matches)
    return PatchRootResolution(matches[0], matches=matches)


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
            "the motivation degenerates so the KEEP threshold can "
            "adjudicate."
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
# ``proposal_set`` is materialised into.
#
# Spelled out rather than derived: no ACTION_CATALOGUE field separates these
# from ``integrate_patch``, which shares their ``exploration`` verdict class,
# ``shallow`` family and ``workspace_write`` side effect while being the one
# action whose materialisation lands the patch under review.
ADVISORY_RULE_PROPOSAL_KINDS: frozenset[str] = frozenset(
    {
        "explore",
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

    Reads the same ``---``/``+++`` header pairs the apply path resolves its
    targets from, so the gate and the applier cannot disagree on which paths a
    patch touches.

    Args:
        patch_text: The unified-diff text to scan.

    Returns:
        The first absolute or ``..``-containing path, or ``None`` when none
        escape the tree.
    """
    for old, new in patch_file_targets(patch_text):
        for raw in (old, new):
            cand = _normalize_patch_path(raw)
            if not cand or cand in _DEV_NULL_PATHS:
                continue
            if cand.startswith("/") or ".." in PurePosixPath(cand).parts:
                return cand
    return None


# Patch grounding verdicts.
GROUND_APPLIES = "applies"  # git apply --check succeeded against clean base
GROUND_STALE = "stale"  # valid diff but does not apply to clean base
GROUND_NOT_DIFF = "not_diff"  # not a unified diff (no hunk header)
GROUND_PATH_ESCAPE = "path_escape"  # patch path escapes the tree
GROUND_MISSING_TARGET = "missing_target"  # modify/delete target absent from base
GROUND_AMBIGUOUS_ROOT = "ambiguous_root"  # patch targets match more than one disjoint tree
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
        deleting a file absent from *every* candidate source tree can never
        apply, so it is dropped before it wastes an ``integrate_patch``
        benchmark slot (unlike ``stale``, which is kept because integrate's
        ``-p`` auto-detect / 3-way merge may still salvage it).

        Returns:
            True for structural-failure verdicts that should drop the patch.
        """
        return self.verdict in (
            GROUND_NOT_DIFF,
            GROUND_PATH_ESCAPE,
            GROUND_MISSING_TARGET,
            GROUND_AMBIGUOUS_ROOT,
        )


def ground_patch_text(
    patch_text: str,
    *,
    base_checkout: Path | None,
    candidate_roots: tuple[Path, ...] = (),
    explicit_root: Path | None = None,
    git_timeout_sec: float = 30.0,
) -> PatchGroundingResult:
    """Validate + git-ground one patch.

    Structural checks (unified diff, no path escape) always run. The
    ``git apply --check`` grounding runs against ``base_checkout`` first, then
    against each entry of ``candidate_roots`` whose tree holds the patch's
    targets. A specialist handed an aiter worktree still writes sglang patches,
    so grounding only against the worktree base drops them as ``missing_target``
    when the sglang checkout would have accepted them.

    Args:
        patch_text: The unified-diff text to validate and ground.
        base_checkout: Primary clean git checkout to ground against, or
            ``None`` to skip the ``git apply --check`` step.
        candidate_roots: Further checkouts to try when ``base_checkout`` does
            not hold the patch's targets.
        explicit_root: Authoritative target checkout. Required for create-only
            patches because no pre-image can identify a candidate.
        git_timeout_sec: Timeout for each ``git apply --check`` subprocess.

    Returns:
        The :class:`PatchGroundingResult` with the verdict and detail.
    """
    if not is_unified_diff(patch_text):
        return PatchGroundingResult(GROUND_NOT_DIFF, "no unified-diff hunk header")
    escape = patch_escapes_tree(patch_text)
    if escape is not None:
        return PatchGroundingResult(GROUND_PATH_ESCAPE, f"path={escape!r}")
    candidates = tuple(
        root
        for root in ((base_checkout,) if base_checkout is not None else ()) + tuple(candidate_roots)
        if Path(root).is_dir()
    )
    if explicit_root is None and not candidates:
        return PatchGroundingResult(GROUND_UNCHECKED, "no base checkout")
    resolution = resolve_patch_apply_root(
        (patch_text,),
        explicit_root=explicit_root,
        candidate_roots=candidates,
        default_root=base_checkout,
    )
    if resolution.root is None:
        detail = resolution.reason
        if resolution.matches:
            detail += ": " + ", ".join(str(root) for root in resolution.matches)
        if resolution.reason == "ambiguous_root":
            return PatchGroundingResult(GROUND_AMBIGUOUS_ROOT, detail)
        return PatchGroundingResult(GROUND_MISSING_TARGET, detail)
    root = resolution.root
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
        return PatchGroundingResult(GROUND_APPLIES)
    return PatchGroundingResult(
        GROUND_STALE,
        (proc.stderr or "").strip()[:240],
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
                + " — the patch names a file that does not exist in any"
                " allowlisted framework source tree; verify the target path"
                " with Glob/Grep before authoring the diff."
            )
        ambiguous = [d for d in self.dropped if d.get("verdict") == GROUND_AMBIGUOUS_ROOT]
        if ambiguous:
            out.append(
                "patch_safety_ambiguous_root:"
                + ",".join(d.get("detail", d["path"]) for d in ambiguous[:4])
                + " — the patch targets match more than one disjoint source"
                " tree; declare an explicit framework_source_root so the"
                " correct tree is selected without guessing."
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
    explicit_root: Path | None = None,
) -> tuple[list[str], list[dict[str, str]], dict[str, str], bool]:
    """Ground each patch against the candidate checkouts, one root per patch.

    Structural rejects (unreadable / non-diff / path escape) are dropped first.
    Each survivor then resolves its own root, so a cross-repo set survives even
    though no single root holds every target. Only a patch absent from every
    candidate is dropped. Stale-but-valid patches are kept for integrate_patch
    and the Critic to adjudicate.

    Args:
        patch_paths: File paths of the candidate patches to vet.
        base_checkout: The checkout the specialist worktree was cut from. It
            is offered to root resolution first and is the root a create-only
            patch lands in.
        candidate_roots: Further checkouts offered to root resolution.
        explicit_root: Authoritative target checkout, when declared.

    Returns:
        A ``(kept_paths, dropped_records, grounding_by_path, spans_multiple_roots)``
        tuple. ``spans_multiple_roots`` is ``True`` when the kept patches
        resolved to more than one distinct checkout.
    """
    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    grounding: dict[str, str] = {}
    readable: list[tuple[str, str]] = []
    for path in patch_paths:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            dropped.append({"path": path, "verdict": "unreadable", "detail": repr(exc)})
            continue
        if not is_unified_diff(text):
            dropped.append(
                {
                    "path": path,
                    "verdict": GROUND_NOT_DIFF,
                    "detail": "no unified-diff hunk header",
                }
            )
            grounding[path] = GROUND_NOT_DIFF
            continue
        escape = patch_escapes_tree(text)
        if escape is not None:
            dropped.append(
                {
                    "path": path,
                    "verdict": GROUND_PATH_ESCAPE,
                    "detail": f"path={escape!r}",
                }
            )
            grounding[path] = GROUND_PATH_ESCAPE
            continue
        readable.append((path, text))

    if not readable:
        return kept, dropped, grounding, False

    candidates = tuple(
        root
        for root in ((base_checkout,) if base_checkout is not None else ()) + tuple(candidate_roots)
        if Path(root).is_dir()
    )

    resolved_roots: set[Path] = set()
    for path, text in readable:
        resolution = resolve_patch_apply_root(
            [text],
            explicit_root=explicit_root,
            candidate_roots=candidates,
            default_root=base_checkout,
        )
        if resolution.reason in _GROUNDING_UNDECIDABLE_REASONS:
            grounding[path] = GROUND_UNCHECKED
            kept.append(path)
            continue
        if resolution.root is None:
            detail = resolution.reason
            if resolution.matches:
                detail += ": " + ", ".join(str(r) for r in resolution.matches)
            verdict = GROUND_AMBIGUOUS_ROOT if resolution.reason == "ambiguous_root" else GROUND_MISSING_TARGET
            grounding[path] = verdict
            dropped.append({"path": path, "verdict": verdict, "detail": detail})
            continue
        res = ground_patch_text(text, base_checkout=None, explicit_root=resolution.root)
        grounding[path] = res.verdict
        if res.is_garbage:
            dropped.append({"path": path, "verdict": res.verdict, "detail": res.detail})
            continue
        resolved_roots.add(resolution.root)
        kept.append(path)
    return kept, dropped, grounding, len(resolved_roots) > 1


__all__ = [
    "ADVISE_VERDICT",
    "ADVISORY_RULE_PROPOSAL_KINDS",
    "CROSS_DOMAIN_RULES",
    "CrossDomainRule",
    "FORBIDDEN_PAYLOAD_FIELDS",
    "FORBIDDEN_PROPOSAL_FIELDS",
    "GROUND_AMBIGUOUS_ROOT",
    "GROUND_APPLIES",
    "GROUND_MISSING_TARGET",
    "GROUND_NOT_DIFF",
    "GROUND_PATH_ESCAPE",
    "GROUND_STALE",
    "GROUND_UNCHECKED",
    "PATCH_ROOT_FAIL_REASONS",
    "PatchGroundingResult",
    "PatchRootResolution",
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
    "resolve_patch_apply_root",
    "scan_numeric_claims",
    "strip_forbidden_proposal_fields",
    "vet_patches",
]
