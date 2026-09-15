# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement failure-signature classifier."""

from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Pattern


# --- Failure kinds ---------------------------------------------------------

MISSING_MODEL_ARCH = "missing_model_arch"
UNSUPPORTED_DTYPE = "unsupported_dtype"
HIP_KERNEL_MISSING = "hip_kernel_missing"
IMPORT_ERROR = "import_error"
SHAPE_MISMATCH = "shape_mismatch"
MISSING_WEIGHT = "missing_weight"
NOT_IMPLEMENTED = "not_implemented"
CAPABILITY_DISABLED = "capability_disabled"
TOKENIZER_ERROR = "tokenizer_error"
SERVE_FLAG = "serve_flag"
# Resource constraints (OOM, TP/GPU count) are NOT code acquisition targets.
RESOURCE_CONSTRAINT = "resource_constraint"
# Accuracy-eval triggers (values match _accuracy_gate EVAL_KIND_*): a booting baseline whose accuracy is below the
# floor, an eval cut short because generation never terminated, and a crashed eval run.
ACCURACY_BELOW_FLOOR = "accuracy_below_floor"
EVAL_GENERATION_PATHOLOGY = "eval_generation_pathology"
EVAL_RUNTIME_FAILURE = "eval_runtime_failure"
UNKNOWN = "unknown"

# Ordered most-specific to least-specific.
FAILURE_KINDS: tuple[str, ...] = (
    MISSING_MODEL_ARCH,
    EVAL_GENERATION_PATHOLOGY,
    ACCURACY_BELOW_FLOOR,
    RESOURCE_CONSTRAINT,
    HIP_KERNEL_MISSING,
    UNSUPPORTED_DTYPE,
    SHAPE_MISMATCH,
    MISSING_WEIGHT,
    NOT_IMPLEMENTED,
    CAPABILITY_DISABLED,
    TOKENIZER_ERROR,
    SERVE_FLAG,
    IMPORT_ERROR,
    EVAL_RUNTIME_FAILURE,
    UNKNOWN,
)


# --- Result model ----------------------------------------------------------


@dataclass(frozen=True)
class FailureSignature:
    """Structured classification of a launch/import/build failure."""

    kind: str
    offending_file: str = ""
    offending_symbol: str = ""
    raw_excerpt: str = ""
    confidence: float = 0.0
    bridge_layer: str = ""
    secondary_kinds: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        """True when the signature is anything other than :data:`UNKNOWN`."""
        return self.kind != UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return asdict(self)


# --- CapabilityGap projection -----------------------------------------------


@dataclass(frozen=True)
class CapabilityGap:
    """Thin overlay on FailureSignature exposing code-acquisition semantics."""

    kind: str
    bridge_layer: str = ""
    requires_code_acquisition: bool = True

    @classmethod
    def from_signature(cls, sig: FailureSignature) -> "CapabilityGap":
        """Project a :class:`FailureSignature` onto a :class:`CapabilityGap`."""
        return cls(
            kind=sig.kind,
            bridge_layer=sig.bridge_layer,
            requires_code_acquisition=(sig.kind != RESOURCE_CONSTRAINT),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "kind": self.kind,
            "bridge_layer": self.bridge_layer,
            "requires_code_acquisition": self.requires_code_acquisition,
        }


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
            # Transformers/HF: a checkpoint whose ``model_type`` predates the installed transformers (or vLLM's
            # ModelConfig validation wrapping it).
            re.compile(r"model type\s+[`'\"]?([A-Za-z0-9_]+)[`'\"]?\s+but\s+Transformers\s+does\s+not\s+recognize"),
            re.compile(r"does\s+not\s+recognize\s+this\s+architecture"),
            re.compile(r"[Tt]he\s+checkpoint\s+.*?model\s+type\s+[`'\"]?([A-Za-z0-9_]+)[`'\"]?"),
        ),
        confidence=0.95,
        symbol_from=_grp,
    ),
    _Rule(
        # The eval was cut short because generation never terminated, so the ~0 score says nothing about answer
        # quality.
        kind=EVAL_GENERATION_PATHOLOGY,
        bridge_layer="",
        patterns=(re.compile(r"eval_generation_pathology"),),
        confidence=0.95,
    ),
    _Rule(
        # A booting baseline whose accuracy is below the floor.
        kind=ACCURACY_BELOW_FLOOR,
        bridge_layer="",
        patterns=(
            re.compile(r"accuracy\s+.*?(?:did not meet|below)\s+.*?floor"),
            re.compile(r"baseline[_ ]accuracy[_ ]below[_ ]floor"),
        ),
        confidence=0.9,
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
            # torch .narrow()/.slice() bounds error from weight loaders on shard-width mismatch.
            re.compile(r"start\s*\(\s*\d+\s*\)\s*\+\s*length\s*\(\s*\d+\s*\)\s*exceeds dimension size"),
        ),
        confidence=0.7,
    ),
    _Rule(
        # Model params mismatch the checkpoint's tensors; the strict weight-init check refuses to boot.
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
        # Resource constraints: OOM, insufficient GPU count, TP requirements. bridge_layer="" means no bridge repo is
        # searched and CapabilityGap marks requires_code_acquisition=False — do not try to patch for these.
        kind=RESOURCE_CONSTRAINT,
        bridge_layer="",
        patterns=(
            re.compile(r"[Oo]ut\s+of\s+memory"),
            re.compile(r"[Hh][Ii][Pp]\s+out\s+of\s+memory"),
            re.compile(r"[Cc][Uu][Dd][Aa]\s+out\s+of\s+memory"),
            re.compile(r"no\s+GPU\s+memory\s+(?:for|left\s+for)\s+the\s+KV\s+[Cc]ache", re.IGNORECASE),
            re.compile(r"[Cc]annot\s+allocate\s+(?:memory|cuda|hip)"),
            re.compile(r"[Oo]ut[Oo]f[Mm]emory[Ee]rror"),
            re.compile(r"requires?\s+(?:at\s+least\s+)?(\d+)\s+GPU[s]?"),
            re.compile(r"[Tt]ensor\s+[Pp]arallel\s+[Ss]ize\s+.*?(\d+).*?GPU"),
            re.compile(r"available\s+GPU\s+count\s+\((\d+)\)\s+(?:is\s+)?less\s+than"),
        ),
        confidence=0.88,
        symbol_from=_grp,
    ),
    _Rule(
        kind=TOKENIZER_ERROR,
        bridge_layer="framework",
        patterns=(
            re.compile(
                r"[Tt]okenizer(?:\s+mode)?\s+['\"]?([A-Za-z0-9_]+)['\"]?\s+(?:is\s+)?not\s+(?:supported|found|recognized)"
            ),
            re.compile(r"[Uu]nknown\s+tokenizer\s+(?:class|type|backend):\s*['\"]?([A-Za-z0-9_]+)"),
            re.compile(r"[Ff]ailed\s+to\s+(?:load|initialize)\s+tokenizer"),
            re.compile(r"[Cc]annot\s+(?:load|find|locate)\s+tokenizer"),
            re.compile(r"--tokenizer-mode\s+([A-Za-z0-9_]+)\s+(?:is\s+)?not\s+supported"),
        ),
        confidence=0.75,
        symbol_from=_grp,
    ),
    _Rule(
        kind=SERVE_FLAG,
        bridge_layer="framework",
        patterns=(
            re.compile(r"unrecognized arguments?:\s*(-+[A-Za-z0-9_-]+)"),
            re.compile(r"error:\s+argument\s+(-+[A-Za-z0-9_-]+)"),
            re.compile(r"invalid\s+choice.*?for.*?argument\s+(-+[A-Za-z0-9_-]+)"),
            re.compile(r"([A-Za-z][A-Za-z0-9_-]+):\s+error:\s+unrecognized"),
        ),
        confidence=0.70,
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
    _Rule(
        # LAST rule: a generic eval-run crash.
        kind=EVAL_RUNTIME_FAILURE,
        bridge_layer="",
        patterns=(
            re.compile(r"run_eval failed with exit code"),
            re.compile(r"ERROR: run_eval failed"),
            re.compile(r"accuracy eval (?:failed|crashed|did not run)"),
        ),
        confidence=0.6,
    ),
)


_TB_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\S+)')
_INLINE_PATH = re.compile(r"([/\w.\-]+\.(?:py|cpp|cc|cu|hip|h|hpp|cuh))(?::\d+)?")


def _extract_offending_file(text: str, *, near: int | None = None) -> str:
    """Return the most relevant source file from a traceback / build log."""
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
    """Return a trimmed one-line-ish excerpt around a regex match."""
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
    """Return the first matching pattern per rule, in :data:`_RULES` order."""
    hits: list[_RuleHit] = []
    for rule in _RULES:
        for pat in rule.patterns:
            m = pat.search(text)
            if m is not None:
                hits.append(_RuleHit(rule=rule, match=m))
                break
    return hits


def classify_failure(log_text: str) -> FailureSignature:
    """Classify a launch/import/build failure into a :class:`FailureSignature`."""
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
    secondary = tuple(h.rule.kind for h in sorted(hits, key=lambda h: h.rule_index) if h.rule.kind != primary.rule.kind)

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
    """Top-level request describing a non-runnable ``(model, backend)`` combo."""

    framework: str
    model: str
    repo_url: str
    launch_log: str = ""
    work_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "framework-agent-enablement")
    gpu_type: str = ""
    max_search_candidates: int = 5

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EnablementRequest":
        """Parse a JSON payload into an :class:`EnablementRequest`."""
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
            work_dir=Path(
                str(raw.get("work_dir") or (Path(tempfile.gettempdir()) / "framework-agent-enablement"))
            ).expanduser(),
            gpu_type=str(raw.get("gpu_type") or "").strip().lower(),
            max_search_candidates=int(raw.get("max_search_candidates", 5)),
        )

    @property
    def signature(self) -> FailureSignature:
        """Classify :attr:`launch_log` on demand."""
        return classify_failure(self.launch_log)


# --- Runnable gate ---------------------------------------------------------


def runnable_decision(
    *,
    booted: bool | None,
    correctness_ok: bool | None,
    boot_timed_out: bool = False,
) -> tuple[bool, str]:
    """Decide whether an enablement patch made the combo *run*.

    ``booted`` is the boot verdict off the attempt's ladder observation, never a
    throughput number: a server that comes up and serves slowly has been
    enabled. Whether the boot got *further* than the last one is a separate
    question, answered by ladder arithmetic over two boot observations.

    Args:
        booted: Whether the attempt reached a serving server; ``None`` when no
            observation was recorded and the question was never answered.
        correctness_ok: Minimal-correctness result; ``None`` if not evaluated.
        boot_timed_out: Whether the attempt was reaped on its wall-clock budget.

    Returns:
        tuple[bool, str]: ``(runs, reason)``.
    """
    if boot_timed_out:
        return False, "the bring-up was reaped on its budget"
    if booted is None:
        return False, "no boot observation was recorded for this attempt"
    if not booted:
        return False, "the server did not come up (still not runnable)"
    if correctness_ok is False:
        return False, "the server came up but the minimal correctness check failed"
    return True, "the server now comes up" + ("" if correctness_ok is None else " and passes minimal correctness")


# Evidence that a dtype/capability miss is backed by a *compiled* op rather than pure-Python guard logic: a native
# symbol, an .so/kernel/op reference, or a named compiled backend.
_NATIVE_EVIDENCE_RE = re.compile(
    r"undefined symbol|\.so\b|_C\b|aiter|sgl[_-]?kernel|hip[a-z]*kernel|"
    r"\bkernel\b|custom[_ ]?op|torch\.ops|extension module|"
    r"\bfp4\b|\bfp8\b|marlin|cutlass",
    re.IGNORECASE,
)


def is_targeted_build_candidate(
    signature: FailureSignature,
    launch_log: str = "",
) -> bool:
    """Whether a residual gap is a *compiled* component miss."""
    if signature is None:
        return False
    if signature.bridge_layer in ("build", "rocm_hip"):
        return True
    if signature.kind == HIP_KERNEL_MISSING:
        return True
    if signature.kind == UNSUPPORTED_DTYPE:
        evidence = f"{signature.offending_symbol or ''}\n{signature.raw_excerpt or ''}\n{launch_log or ''}"
        return bool(_NATIVE_EVIDENCE_RE.search(evidence))
    return False


__all__ = [
    "ACCURACY_BELOW_FLOOR",
    "CAPABILITY_DISABLED",
    "EVAL_GENERATION_PATHOLOGY",
    "EVAL_RUNTIME_FAILURE",
    "FAILURE_KINDS",
    "HIP_KERNEL_MISSING",
    "IMPORT_ERROR",
    "MISSING_MODEL_ARCH",
    "MISSING_WEIGHT",
    "NOT_IMPLEMENTED",
    "RESOURCE_CONSTRAINT",
    "SERVE_FLAG",
    "SHAPE_MISMATCH",
    "TOKENIZER_ERROR",
    "UNKNOWN",
    "UNSUPPORTED_DTYPE",
    "CapabilityGap",
    "EnablementRequest",
    "FailureSignature",
    "classify_failure",
    "is_targeted_build_candidate",
    "runnable_decision",
]
