# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Enablement failure-signature classifier.

The enablement path answers a different question than the perf path: not
"is this candidate faster?" but "why does this ``(model, backend)`` combo
fail to *run at all*, and where would a bridging patch land?".

This module is the deterministic, GPU-free front door to that path. It parses
a server-launch log / Python traceback / build error into a structured
:class:`FailureSignature` — a failure ``kind`` plus the offending file/symbol
a downstream authoring sub-agent should target. It performs **no** network,
LLM, or filesystem access, so it is fully unit-testable.

See ``framework_ref1_design.md`` §2 for the taxonomy rationale and its
relationship to the existing ``static_recon_specialist`` (which handles the
orthogonal "runs but a fast path is silently disabled" case).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Pattern


# --- Failure kinds ---------------------------------------------------------
# Stable string ids (not an Enum) so they serialize verbatim into JSON
# summaries and downstream KB entries without an encoder shim.

MISSING_MODEL_ARCH = "missing_model_arch"
UNSUPPORTED_DTYPE = "unsupported_dtype"
HIP_KERNEL_MISSING = "hip_kernel_missing"
IMPORT_ERROR = "import_error"
SHAPE_MISMATCH = "shape_mismatch"
NOT_IMPLEMENTED = "not_implemented"
CAPABILITY_DISABLED = "capability_disabled"
UNKNOWN = "unknown"

# Ordered most-specific → least-specific. ``classify_failure`` returns the
# first rule that matches, so ambiguous text (e.g. an ImportError whose real
# cause is a missing HIP symbol) resolves to the more actionable bridging
# layer first.
FAILURE_KINDS: tuple[str, ...] = (
    MISSING_MODEL_ARCH,
    HIP_KERNEL_MISSING,
    UNSUPPORTED_DTYPE,
    SHAPE_MISMATCH,
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
    """

    kind: str
    offending_file: str = ""
    offending_symbol: str = ""
    raw_excerpt: str = ""
    confidence: float = 0.0
    bridge_layer: str = ""

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


# Each pattern's first capture group (when present) is the offending symbol.
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
        ),
        confidence=0.7,
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


# Last Python traceback frame: File "<path>", line N, in <func>
_TB_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\S+)')
# Inline path mention (e.g. compiler errors: /path/to/file.cpp:123:4)
_INLINE_PATH = re.compile(r"([/\w.\-]+\.(?:py|cpp|cc|cu|hip|h|hpp|cuh))(?::\d+)?")


def _extract_offending_file(text: str) -> str:
    """Return the most relevant source file from a traceback / build log.

    Prefers the *last* Python traceback frame (closest to the raise site);
    falls back to the last inline source-path mention. Returns ``""`` when
    nothing matches.

    Args:
        text: The raw log / traceback text.

    Returns:
        str: A file path, or ``""`` when none is found.
    """
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


def classify_failure(log_text: str) -> FailureSignature:
    """Classify a launch/import/build failure into a :class:`FailureSignature`.

    Scans the ordered :data:`_RULES` table and returns the first match,
    enriching it with the offending file (from the traceback) and symbol
    (from the matching pattern's first capture group, when any). Returns an
    :data:`UNKNOWN` signature (``confidence=0.0``) when nothing matches.

    This is pure text analysis — no network, LLM, or filesystem access.

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

    offending_file = _extract_offending_file(text)

    for rule in _RULES:
        for pat in rule.patterns:
            m = pat.search(text)
            if m is None:
                continue
            symbol = ""
            if rule.symbol_from is not None:
                symbol = rule.symbol_from(m)
            return FailureSignature(
                kind=rule.kind,
                offending_file=offending_file,
                offending_symbol=symbol,
                raw_excerpt=_excerpt_for(m, text),
                confidence=rule.confidence,
                bridge_layer=rule.bridge_layer,
            )

    return FailureSignature(
        kind=UNKNOWN,
        offending_file=offending_file,
        raw_excerpt=re.sub(r"\s+", " ", text[-200:]).strip(),
        confidence=0.0,
    )


# --- Request model ---------------------------------------------------------


@dataclass(frozen=True)
class EnablementRequest:
    """Top-level request describing a non-runnable ``(model, backend)`` combo.

    Mirrors the shape of :class:`framework_agent.models.ExploreRequest` but is
    gated on *runnability* rather than throughput — there is no baseline
    throughput because the combo does not run yet.

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

    The enablement analogue of
    :func:`framework_agent.decision.winner_decision`: the gate is **runnability**,
    not throughput. A patch is KEPT only when the launch probe now exits 0 (and
    did not time out) and, when a correctness check was run, it passed.

    Short-circuits on the first failing condition; the reason is always set for
    audit. As defence-in-depth, when both ``before_signature`` and
    ``after_signature`` are supplied, the *same* actionable failure re-appearing
    after the patch is treated as "not fixed" even if the probe superficially
    returned 0.

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


__all__ = [
    "CAPABILITY_DISABLED",
    "FAILURE_KINDS",
    "HIP_KERNEL_MISSING",
    "IMPORT_ERROR",
    "MISSING_MODEL_ARCH",
    "NOT_IMPLEMENTED",
    "SHAPE_MISMATCH",
    "UNKNOWN",
    "UNSUPPORTED_DTYPE",
    "EnablementRequest",
    "FailureSignature",
    "classify_failure",
    "runnable_decision",
]
