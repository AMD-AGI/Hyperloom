# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Enablement failure-signature classifier.

Parses a server-launch log / Python traceback / build error into a structured
:class:`FailureSignature` — a failure ``kind`` plus the offending file/symbol
a downstream authoring sub-agent should target. Performs no network, LLM, or
filesystem access.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Pattern


# --- Failure kinds ---------------------------------------------------------
# String ids for each failure kind.

MISSING_MODEL_ARCH = "missing_model_arch"
UNSUPPORTED_DTYPE = "unsupported_dtype"
HIP_KERNEL_MISSING = "hip_kernel_missing"
IMPORT_ERROR = "import_error"
SHAPE_MISMATCH = "shape_mismatch"
MISSING_WEIGHT = "missing_weight"
NOT_IMPLEMENTED = "not_implemented"
CAPABILITY_DISABLED = "capability_disabled"
UNKNOWN = "unknown"

# Ordered most-specific to least-specific.
FAILURE_KINDS: tuple[str, ...] = (
    MISSING_MODEL_ARCH,
    HIP_KERNEL_MISSING,
    UNSUPPORTED_DTYPE,
    SHAPE_MISMATCH,
    MISSING_WEIGHT,
    NOT_IMPLEMENTED,
    CAPABILITY_DISABLED,
    IMPORT_ERROR,
    UNKNOWN,
)


# --- Result model ----------------------------------------------------------


@dataclass(frozen=True)
class FailureSignature:
    """Structured classification of a launch/import/build failure.

    Attributes:
        kind: One of :data:`FAILURE_KINDS`.
        offending_file: Best-guess source file the fix should target
            (from the last traceback frame or an inline path), or ``""``.
        offending_symbol: Best-guess symbol (arch name, function, module,
            dtype, undefined symbol), or ``""``.
        raw_excerpt: The matched log line(s), trimmed, for audit.
        confidence: Heuristic confidence in ``[0.0, 1.0]``; ``0.0`` for
            :data:`UNKNOWN`.
        bridge_layer: Where a bridging patch most likely lands
            (``"framework"``, ``"rocm_hip"``, ``"build"``, or ``""``).
        secondary_kinds: Other failure kinds whose rules also matched, most
            specific first (excludes the primary ``kind``). Empty when only one
            rule matched.
    """

    kind: str
    offending_file: str = ""
    offending_symbol: str = ""
    raw_excerpt: str = ""
    confidence: float = 0.0
    bridge_layer: str = ""
    secondary_kinds: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        """True when the signature is anything other than :data:`UNKNOWN`.

        Returns:
            bool: ``True`` unless ``kind`` is :data:`UNKNOWN`.
        """
        return self.kind != UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output.

        Returns:
            dict[str, Any]: A dataclass-derived dict of all fields.
        """
        return asdict(self)


# --- Rule table ------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    """One classification rule: patterns + how to extract the offending symbol."""

    kind: str
    bridge_layer: str
    patterns: tuple[Pattern[str], ...]
    confidence: float
    symbol_from: Callable[[re.Match[str]], str] | None = None


def _grp(match: re.Match[str]) -> str:
    """Return the first non-empty capture group of a match, else ``""``."""
    for g in match.groups():
        if g:
            return g.strip()
    return ""


_RULES: tuple[_Rule, ...] = (
    _Rule(
        kind=MISSING_MODEL_ARCH,
        bridge_layer="framework",
        patterns=(
            re.compile(r"[Mm]odel architecture[s]?\s+['\"]?([A-Za-z0-9_]+)['\"]?\s+(?:is|are)?\s*not\s+supported"),
            re.compile(r"[Uu]nsupported\s+model\s+architecture[:\s]+['\"]?([A-Za-z0-9_]+)"),
            re.compile(r"[Aa]rchitectures?\s+\[?['\"]([A-Za-z0-9_]+)['\"].*?not\s+(?:yet\s+)?supported"),
        ),
        confidence=0.95,
        symbol_from=_grp,
    ),
    _Rule(
        kind=HIP_KERNEL_MISSING,
        bridge_layer="rocm_hip",
        patterns=(
            re.compile(r"hipError[A-Za-z]*"),
            re.compile(r"no kernel image is available"),
            re.compile(r"hipErrorNoBinaryForGpu"),
            re.compile(r"undefined symbol:\s*([A-Za-z0-9_:]+)"),
            re.compile(r"HSA_STATUS_ERROR[A-Za-z_]*"),
        ),
        confidence=0.85,
        symbol_from=_grp,
    ),
    _Rule(
        kind=UNSUPPORTED_DTYPE,
        bridge_layer="framework",
        patterns=(
            re.compile(r"not implemented for\s+['\"]?([A-Za-z0-9_]+)['\"]?"),
            re.compile(r"\b(fp8|bfloat16|bf16|float8|e4m3|e5m2|int4|fp4)\b[^\n]*?(?:unsupported|not\s+supported)"),
            re.compile(r"(?:dtype|data type)\s+['\"]?([A-Za-z0-9_]+)['\"]?\s+(?:is\s+)?not\s+supported"),
        ),
        confidence=0.8,
        symbol_from=_grp,
    ),
    _Rule(
        kind=SHAPE_MISMATCH,
        bridge_layer="framework",
        patterns=(
            re.compile(r"shape\s+['\"]?\[?[\d,\s]+\]?['\"]?\s+(?:is\s+)?invalid for input of size"),
            re.compile(r"size mismatch"),
            re.compile(r"mat1 and mat2 shapes cannot be multiplied"),
            re.compile(r"[Ee]xpected .*? but got .*? \(size"),
            # torch .narrow()/.slice() bounds error surfaced by weight loaders
            # when a fused projection's checkpoint shard width does not match
            # the model class's expected dim (e.g. a new attention variant
            # loaded through an older model implementation).
            re.compile(r"start\s*\(\s*\d+\s*\)\s*\+\s*length\s*\(\s*\d+\s*\)\s*exceeds dimension size"),
        ),
        confidence=0.7,
    ),
    _Rule(
        # A model implementation that declares parameters the checkpoint does
        # not carry (or vice-versa): the strict weight-init check refuses to
        # boot. Common when a new architecture shares/omits per-layer tensors
        # (e.g. index-sharing) but the framework instantiates them on every
        # layer. Distinct from SHAPE_MISMATCH so it registers as a *different*
        # (deeper) failure once a prior shape fix is applied — this is what lets
        # the enablement loop detect forward progress instead of re-deriving the
        # same fix (see ``enablement_made_progress``).
        kind=MISSING_WEIGHT,
        bridge_layer="framework",
        patterns=(
            re.compile(r"(?:were|was)\s+not\s+initialized\s+from\s+(?:the\s+)?checkpoint"),
            re.compile(r"not\s+initialized\s+from\s+checkpoint"),
            re.compile(r"[Mm]issing\s+key\(?s?\)?\s+in\s+state_dict"),
            re.compile(r"[Uu]nexpected\s+key\(?s?\)?\s+in\s+state_dict"),
            re.compile(r"[Ee]rror\(s\)\s+in\s+loading\s+state_dict"),
            re.compile(r"KeyError:\s*['\"]([\w.]+\.(?:weight|bias))['\"]"),
        ),
        confidence=0.72,
        symbol_from=_grp,
    ),
    _Rule(
        kind=NOT_IMPLEMENTED,
        bridge_layer="framework",
        patterns=(
            re.compile(r"NotImplementedError:?\s*(.*)"),
            re.compile(r"raise\s+NotImplementedError"),
        ),
        confidence=0.75,
        symbol_from=_grp,
    ),
    _Rule(
        kind=CAPABILITY_DISABLED,
        bridge_layer="framework",
        patterns=(
            re.compile(r"([A-Za-z_][A-Za-z0-9_]*_supported)\s*\(\s*\)\s*(?:returned|is|==)?\s*False"),
            re.compile(r"falling back to (?:the\s+)?(?:naive|slow|reference) (?:path|implementation)"),
            re.compile(r"disabled on (?:ROCm|HIP|AMD)"),
        ),
        confidence=0.6,
        symbol_from=_grp,
    ),
    _Rule(
        kind=IMPORT_ERROR,
        bridge_layer="build",
        patterns=(
            re.compile(r"ModuleNotFoundError:\s*No module named\s+['\"]([A-Za-z0-9_.]+)['\"]"),
            re.compile(r"ImportError:\s*(?:cannot import name\s+['\"]?([A-Za-z0-9_]+)['\"]?)?"),
        ),
        confidence=0.7,
        symbol_from=_grp,
    ),
)


_TB_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\S+)')
_INLINE_PATH = re.compile(r"([/\w.\-]+\.(?:py|cpp|cc|cu|hip|h|hpp|cuh))(?::\d+)?")


def _extract_offending_file(text: str, *, near: int | None = None) -> str:
    """Return the most relevant source file from a traceback / build log.

    When ``near`` is given, prefer the traceback frame / inline path closest to
    (and at or before) that offset. Otherwise fall back to the last Python
    traceback frame, then the last inline source-path mention. Returns ``""``
    when nothing matches.

    Args:
        text: The raw log / traceback text.
        near: Optional character offset of the primary rule match; frames at or
            before it are preferred.

    Returns:
        str: A file path, or ``""`` when none is found.
    """
    if near is not None:
        for finder in (_TB_FRAME, _INLINE_PATH):
            before = [m for m in finder.finditer(text) if m.start() <= near]
            if before:
                nearest = max(before, key=lambda m: m.start())
                return nearest.group(1).strip()
    frames = _TB_FRAME.findall(text)
    if frames:
        return frames[-1][0].strip()
    paths = _INLINE_PATH.findall(text)
    if paths:
        return paths[-1].strip()
    return ""


def _excerpt_for(match: re.Match[str], text: str, span: int = 200) -> str:
    """Return a trimmed one-line-ish excerpt around a regex match.

    Args:
        match: The regex match whose neighbourhood to excerpt.
        text: The full source text.
        span: Max characters to keep from the match start.

    Returns:
        str: The trimmed excerpt (whitespace collapsed).
    """
    start = match.start()
    raw = text[start : start + span]
    return re.sub(r"\s+", " ", raw).strip()


@dataclass(frozen=True)
class _RuleHit:
    """One rule that matched, with the match used to extract its symbol/excerpt."""

    rule: _Rule
    match: re.Match[str]

    @property
    def rule_index(self) -> int:
        """Position of ``rule`` in :data:`_RULES` (lower == more specific)."""
        return _RULES.index(self.rule)


def _collect_hits(text: str) -> list[_RuleHit]:
    """Return the first matching pattern per rule, in :data:`_RULES` order.

    At most one hit per rule kind (the first pattern that fires), so a rule is
    not double-counted when several of its patterns match.

    Args:
        text: The raw log / traceback text.

    Returns:
        list[_RuleHit]: Hits ordered as the rules are declared (most specific
        first); empty when nothing matches.
    """
    hits: list[_RuleHit] = []
    for rule in _RULES:
        for pat in rule.patterns:
            m = pat.search(text)
            if m is not None:
                hits.append(_RuleHit(rule=rule, match=m))
                break
    return hits


def classify_failure(log_text: str) -> FailureSignature:
    """Classify a launch/import/build failure into a :class:`FailureSignature`.

    Collects every rule in :data:`_RULES` that matches, elects a primary by
    ``(rule_index, match_start)``, and surfaces the remaining matched kinds as
    :attr:`FailureSignature.secondary_kinds`. Confidence rises slightly with
    each corroborating rule. Returns an :data:`UNKNOWN` signature
    (``confidence=0.0``) when nothing matches.

    Args:
        log_text: Raw server-launch stderr/stdout, Python traceback, or
            build-error text.

    Returns:
        FailureSignature: The classification result; ``kind == UNKNOWN`` when
        no rule matches (including for empty/blank input).
    """
    text = log_text or ""
    if not text.strip():
        return FailureSignature(kind=UNKNOWN)

    hits = _collect_hits(text)
    if not hits:
        return FailureSignature(
            kind=UNKNOWN,
            offending_file=_extract_offending_file(text),
            raw_excerpt=re.sub(r"\s+", " ", text[-200:]).strip(),
            confidence=0.0,
        )

    primary = min(hits, key=lambda h: (h.rule_index, h.match.start()))
    secondary = tuple(
        h.rule.kind for h in sorted(hits, key=lambda h: h.rule_index) if h.rule.kind != primary.rule.kind
    )

    symbol = ""
    if primary.rule.symbol_from is not None:
        symbol = primary.rule.symbol_from(primary.match)

    confidence = min(1.0, primary.rule.confidence + 0.02 * len(secondary))

    return FailureSignature(
        kind=primary.rule.kind,
        offending_file=_extract_offending_file(text, near=primary.match.start()),
        offending_symbol=symbol,
        raw_excerpt=_excerpt_for(primary.match, text),
        confidence=confidence,
        bridge_layer=primary.rule.bridge_layer,
        secondary_kinds=secondary,
    )


# --- Request model ---------------------------------------------------------


@dataclass(frozen=True)
class EnablementRequest:
    """Top-level request describing a non-runnable ``(model, backend)`` combo.

    Attributes:
        framework: Serving framework (``sglang`` / ``vllm`` / ``atom`` ...).
        model: Model id / path that fails to launch.
        repo_url: Canonical framework repo URL (see :mod:`repo_map`).
        launch_log: Raw failure text fed to :func:`classify_failure`.
        work_dir: Scratch root for candidate worktrees.
        gpu_type: Target GPU (``mi300x`` ...); feeds keyword ranking.
        launch_probe: Command that must exit 0 for the combo to count as
            "runs" (the runnable gate). Empty disables the probe.
        max_search_candidates: Cap on bridging PRs to consider.
    """

    framework: str
    model: str
    repo_url: str
    launch_log: str = ""
    work_dir: Path = field(default=Path("/tmp/framework-agent-enablement"))
    gpu_type: str = ""
    launch_probe: str = ""
    max_search_candidates: int = 5

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnablementRequest":
        """Parse a JSON payload into an :class:`EnablementRequest`.

        Args:
            raw: Decoded JSON with at least ``framework``, ``model``,
                ``repo_url``.

        Returns:
            EnablementRequest: The parsed request.

        Raises:
            ValueError: If a required field is missing/empty.
        """
        framework = str(raw.get("framework") or "").strip().lower()
        if not framework:
            raise ValueError("framework is required")
        model = str(raw.get("model") or "").strip()
        if not model:
            raise ValueError("model is required")
        repo_url = str(raw.get("repo_url") or "").strip()
        if not repo_url:
            raise ValueError("repo_url is required")
        return cls(
            framework=framework,
            model=model,
            repo_url=repo_url,
            launch_log=str(raw.get("launch_log") or ""),
            work_dir=Path(str(raw.get("work_dir") or "/tmp/framework-agent-enablement")).expanduser(),
            gpu_type=str(raw.get("gpu_type") or "").strip().lower(),
            launch_probe=str(raw.get("launch_probe") or "").strip(),
            max_search_candidates=int(raw.get("max_search_candidates", 5)),
        )

    @property
    def signature(self) -> FailureSignature:
        """Classify :attr:`launch_log` on demand.

        Returns:
            FailureSignature: Result of :func:`classify_failure` over
            ``launch_log``.
        """
        return classify_failure(self.launch_log)


# --- Runnable gate ---------------------------------------------------------


def runnable_decision(
    *,
    probe_returncode: int | None,
    correctness_ok: bool | None,
    probe_timed_out: bool = False,
    before_signature: FailureSignature | None = None,
    after_signature: FailureSignature | None = None,
) -> tuple[bool, str]:
    """Decide whether an enablement patch made the combo *run*.

    Returns KEEP only when the launch probe now exits 0 (and did not time out)
    and, when a correctness check was run, it passed. When both
    ``before_signature`` and ``after_signature`` are supplied, the same
    actionable failure re-appearing after the patch returns REVERT.

    Args:
        probe_returncode: Launch-probe exit code; ``None`` if the probe did not
            run.
        correctness_ok: Minimal-correctness result; ``None`` if not evaluated.
        probe_timed_out: Whether the probe hit its wall-clock budget.
        before_signature: Failure signature before applying the patch.
        after_signature: Failure signature captured from the post-patch probe
            output (``UNKNOWN`` / non-actionable when it booted cleanly).

    Returns:
        tuple[bool, str]: ``(runs, reason)``.
    """
    if probe_timed_out:
        return False, "launch probe timed out"
    if probe_returncode is None:
        return False, "launch probe did not run"
    if probe_returncode != 0:
        return False, f"launch probe exited {probe_returncode} (still not runnable)"
    if (
        before_signature is not None
        and after_signature is not None
        and after_signature.is_actionable
        and after_signature.kind == before_signature.kind
    ):
        return False, f"same failure {after_signature.kind} persists after patch"
    if correctness_ok is False:
        return False, "launch succeeded but minimal correctness check failed"
    return True, "combo now launches" + ("" if correctness_ok is None else " and passes minimal correctness")


def _failure_identity(sig: FailureSignature | None) -> tuple[str, str, str]:
    """A coarse, taxonomy-independent identity for a failure signature.

    Deliberately does NOT rely on the enumerated ``kind`` alone: two *different*
    boot crashes can both classify as ``UNKNOWN`` (a brand-new failure the rule
    table has never seen), yet still represent real forward progress when the
    error text / offending site changed. The identity is ``(kind, offending_file,
    normalized_excerpt)`` where the excerpt is whitespace-collapsed, lower-cased,
    truncated, and has run-to-run numeric operands (sizes / addresses / layer
    indices) masked to ``#`` so "size 704 vs 576" and "size 512 vs 384" compare
    equal (same failure, different operands) but a genuinely different error does
    not.

    Args:
        sig: The failure signature (may be ``None``).

    Returns:
        tuple[str, str, str]: ``(kind, offending_file, normalized_excerpt)``.
    """
    if sig is None:
        return ("", "", "")
    excerpt = re.sub(r"\s+", " ", (sig.raw_excerpt or "")).strip().lower()
    excerpt = re.sub(r"\d+", "#", excerpt)[:160]
    return (sig.kind or "", (sig.offending_file or "").strip(), excerpt)


def _has_failure(sig: FailureSignature | None) -> bool:
    """True when a signature represents a real (post-)boot failure, not a clean boot.

    ``kind`` is always populated (at least ``UNKNOWN``), so it cannot be used to
    tell "no failure" apart from "unclassified failure". A real failure is either
    actionable OR carries error text / an offending file; a clean boot is a
    non-actionable signature with no content.
    """
    if sig is None:
        return False
    if sig.is_actionable:
        return True
    return bool((sig.raw_excerpt or "").strip() or (sig.offending_file or "").strip())


def enablement_made_progress(
    before_signature: FailureSignature | None,
    after_signature: FailureSignature | None,
) -> bool:
    """Whether a patch advanced the boot to a *new, deeper* failure.

    Enablement gaps are frequently **serial**: fixing gap #1 (e.g. a shape
    mismatch in the weight loader) only reveals gap #2 (e.g. a missing-weight
    error deeper in model construction). A patch that clears the original crash
    but stops at a *different* failure has made real forward progress and its
    diff is a necessary building block — it must be **kept / stacked**, not
    reverted and re-derived from the stale original log.

    **Taxonomy-independent (see Q1 hardening):** progress is judged by whether
    the failure *identity* changed (:func:`_failure_identity`), NOT by whether
    the enumerated ``kind`` changed. So a brand-new gap that the classifier has
    never seen (``kind == UNKNOWN``) still registers as progress as long as its
    error text / offending site differs from the prior failure. This removes the
    dependency on adding a new ``FAILURE_KINDS`` entry for every novel gap.

    A clean boot (``after`` carries no error text at all) is **not** "progress"
    here; that is the terminal *runnable* case handled by
    :func:`runnable_decision`.

    Args:
        before_signature: Failure signature before applying the patch.
        after_signature: Failure signature captured from the post-patch probe.

    Returns:
        bool: ``True`` when the patch moved the boot to a new failure identity.
    """
    # No post-patch failure at all -> clean boot, handled by runnable_decision.
    if not _has_failure(after_signature):
        return False
    if not _has_failure(before_signature):
        # No known prior failure to compare against: any post-patch failure is
        # treated as a (first) forward step.
        return True
    return _failure_identity(after_signature) != _failure_identity(before_signature)


__all__ = [
    "CAPABILITY_DISABLED",
    "FAILURE_KINDS",
    "HIP_KERNEL_MISSING",
    "IMPORT_ERROR",
    "MISSING_MODEL_ARCH",
    "MISSING_WEIGHT",
    "NOT_IMPLEMENTED",
    "SHAPE_MISMATCH",
    "UNKNOWN",
    "UNSUPPORTED_DTYPE",
    "EnablementRequest",
    "FailureSignature",
    "classify_failure",
    "enablement_made_progress",
    "runnable_decision",
]
