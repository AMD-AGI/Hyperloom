"""Workspace artifact scan — converts files on disk into a typed snapshot.

``collect_artifacts`` is a pure function: it does NOT classify outcomes, it
only inventories what SKILL.md left behind in ``<workspace>/`` and in the
quantized model directory (resolved from ``run_manifest.yaml``). The classifier
in ``driver/assessment.py`` consumes the snapshot and decides which ``OutcomeId``
applies.

The split exists so the classifier can stay declarative (decision table) while
disk I/O lives here in one place. It also keeps tests cheap — feed a fixture
workspace directory, get a frozen snapshot, assert on fields.

Only the **MUST-have** and **MUST-validate** files from §5.4 are interpreted;
SHOULD-have and NICE-to-have presence is recorded but not graded — that's
classifier territory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


STRICT_VALIDATION_ENV = "HYPERLOOM_QUANT_STRICT_VALIDATION"

# §5.4 MUST-have file globs for the quantized model directory.
# (Multi-shard models also require ``model.safetensors.index.json``, but its
# absence on single-file models is not a failure — the weights-glob covers
# both cases.)
_WEIGHT_GLOBS = ("*.safetensors", "*.bin")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
)


@dataclass(frozen=True)
class ValidationSteps:
    """Per-step status parsed from ``validation_report.md``.

    ``None`` means the step heading wasn't found in the report (distinct from
    ``"skipped"`` which the validator emits explicitly). The classifier needs
    that distinction: an absent step usually means the validator was never
    run / crashed early, not that the step was skipped on purpose.
    """

    auxiliary: str | None = None   # Step 1
    md5: str | None = None         # Step 2
    config: str | None = None      # Step 3
    fuzzy: str | None = None       # Step 4


@dataclass(frozen=True)
class CollectedArtifacts:
    workspace: Path
    strict_validation: bool

    # Manifest + resolution
    manifest_present: bool
    manifest_parse_error: str | None
    quantized_model_dir: Path | None

    # SHOULD-have / Plan artifacts
    model_analysis_present: bool
    quant_plan_present: bool
    session_context_present: bool

    # MUST-have on quantized_model_dir
    quantized_dir_exists: bool
    has_config_json: bool
    has_weights: bool
    has_tokenizer: bool  # any tokenizer file present

    # Validator
    validation_report_present: bool
    validation_steps: ValidationSteps

    # Eval
    source_eval_present: bool
    quantized_eval_present: bool
    eval_report_present: bool
    eval_report_data: dict | None       # parsed eval_report.json (or None)
    eval_skipped_reason: str | None     # contents of eval_skipped.txt if present

    # SKILL.md control-plane files
    last_phase: str | None              # contents of last_phase.txt
    blocked_reason: str | None          # contents of blocked.md
    fix_hypothesis_attempts: tuple[int, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────────
# parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches a line like ``**Step 2 — MD5 spot-check**: ok``.
# The validator emits one of ``ok`` / ``FAIL`` / ``skipped`` exactly.
_STEP_LINE_RE = re.compile(
    r"^\*\*Step\s+(?P<num>[1-4])\b[^*]*\*\*\s*:\s*(?P<status>ok|FAIL|skipped)\b",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_validation_report(text: str) -> ValidationSteps:
    by_num: dict[str, str] = {}
    for m in _STEP_LINE_RE.finditer(text):
        # Normalize to lower-case "ok"/"fail"/"skipped" for downstream compares.
        status = m.group("status").lower()
        if status == "fail":
            status = "FAIL"  # keep FAIL upper-case to match the spec text
        by_num[m.group("num")] = status
    return ValidationSteps(
        auxiliary=by_num.get("1"),
        md5=by_num.get("2"),
        config=by_num.get("3"),
        fuzzy=by_num.get("4"),
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    raw = _read_text(path)
    if raw is None:
        return None, None
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc.msg} at line {exc.lineno}"


def _resolve_quantized_dir(workspace: Path) -> tuple[Path | None, bool, str | None]:
    """Read ``run_manifest.yaml`` and pull ``outputs.quantized_model_dir``.

    Returns ``(path, manifest_present, parse_error)``. PyYAML is imported
    lazily so the agent stays installable without it — falling back to the
    ``nice_to_have_skipped`` (#20) outcome in that case.
    """

    manifest = workspace / "run_manifest.yaml"
    if not manifest.is_file():
        return None, False, None

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return None, True, "pyyaml_missing"

    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, True, f"yaml_error: {exc}"

    if not isinstance(data, dict):
        return None, True, "manifest_not_mapping"

    outputs = data.get("outputs") or {}
    raw_path = outputs.get("quantized_model_dir") if isinstance(outputs, dict) else None
    if not raw_path:
        return None, True, "missing_outputs_quantized_model_dir"

    path = Path(str(raw_path))
    if not path.is_absolute():
        # Manifest paths are conventionally relative to workspace.
        path = (workspace / path).resolve()
    return path, True, None


def _has_any(directory: Path, names: Iterable[str]) -> bool:
    for name in names:
        if (directory / name).is_file():
            return True
    return False


def _has_glob(directory: Path, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        for _ in directory.glob(pat):
            return True
    return False


def _scan_hypothesis_attempts(workspace: Path) -> tuple[int, ...]:
    """Find ``fix_hypothesis_attempt_N.md`` files and return the sorted Ns.

    The classifier uses these to decide if SKILL.md actually diagnosed a fix
    before the retry was attempted (precondition for incrementing the
    retry counter — see §A.10).
    """

    pattern = re.compile(r"^fix_hypothesis_attempt_(\d+)\.md$")
    ns: list[int] = []
    try:
        for entry in workspace.iterdir():
            m = pattern.match(entry.name)
            if m and entry.is_file():
                ns.append(int(m.group(1)))
    except FileNotFoundError:
        return ()
    return tuple(sorted(ns))


def _strict_validation_enabled(env: dict[str, str] | None = None) -> bool:
    raw = (env if env is not None else os.environ).get(STRICT_VALIDATION_ENV, "1")
    return raw.strip().lower() not in ("0", "false", "no", "")


# ─────────────────────────────────────────────────────────────────────────────
# public entry
# ─────────────────────────────────────────────────────────────────────────────

def collect_artifacts(
    workspace: Path,
    *,
    env: dict[str, str] | None = None,
) -> CollectedArtifacts:
    workspace = Path(workspace)

    qdir, manifest_present, manifest_err = _resolve_quantized_dir(workspace)

    quantized_dir_exists = bool(qdir and qdir.is_dir())
    has_config = bool(quantized_dir_exists and (qdir / "config.json").is_file())  # type: ignore[union-attr]
    has_weights = bool(quantized_dir_exists and _has_glob(qdir, _WEIGHT_GLOBS))   # type: ignore[arg-type]
    has_tokenizer = bool(quantized_dir_exists and _has_any(qdir, _TOKENIZER_FILES))  # type: ignore[arg-type]

    validation_text = _read_text(workspace / "validation_report.md")
    validation_present = validation_text is not None
    validation_steps = (
        _parse_validation_report(validation_text) if validation_text else ValidationSteps()
    )

    eval_data, _ = _read_json(workspace / "eval_report.json")

    return CollectedArtifacts(
        workspace=workspace,
        strict_validation=_strict_validation_enabled(env),
        manifest_present=manifest_present,
        manifest_parse_error=manifest_err,
        quantized_model_dir=qdir,
        model_analysis_present=(workspace / "model_analysis.json").is_file(),
        quant_plan_present=(workspace / "quant_plan.json").is_file(),
        session_context_present=(workspace / "session_context.json").is_file(),
        quantized_dir_exists=quantized_dir_exists,
        has_config_json=has_config,
        has_weights=has_weights,
        has_tokenizer=has_tokenizer,
        validation_report_present=validation_present,
        validation_steps=validation_steps,
        source_eval_present=(workspace / "source_eval.md").is_file(),
        quantized_eval_present=(workspace / "quantized_eval.md").is_file(),
        eval_report_present=(workspace / "eval_report.json").is_file(),
        eval_report_data=eval_data,
        eval_skipped_reason=_read_text(workspace / "eval_skipped.txt"),
        last_phase=(_read_text(workspace / "last_phase.txt") or "").strip() or None,
        blocked_reason=_read_text(workspace / "blocked.md"),
        fix_hypothesis_attempts=_scan_hypothesis_attempts(workspace),
    )


__all__ = [
    "CollectedArtifacts",
    "ValidationSteps",
    "STRICT_VALIDATION_ENV",
    "collect_artifacts",
]
