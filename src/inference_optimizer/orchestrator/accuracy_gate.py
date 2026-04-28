"""Accuracy gate — DESIGN §7.5.

Risk-budgeted accuracy validation. Only run for actions whose
``accuracy_risk > 0`` (see DESIGN §7.5.1 risk table).

STATUS (v0.7):
    Pure-Python implementation. ``run_gsm8k`` shells out to
    ``scripts/eval_accuracy.sh`` via a subprocess seam (``_run_eval``)
    that tests patch with deterministic stubs. ``optional_kernel_micro_check``
    works with ``numpy`` arrays / nested lists / tuples without requiring
    PyTorch.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

# Per DESIGN §7.5.1 — actions that DO NOT require gating.
_GATE_FREE_ACTIONS: tuple[str, ...] = (
    "setup",
    "classify",
    "target-analysis",
    "baseline",
    "profile",
    "sweep",
    "report",
    "dream",
    "re-explore",
    "recover",
)


class Verdict(str, Enum):
    KEEP = "keep"
    REVERT = "revert"
    FAIL = "fail"


@dataclass
class GateResult:
    verdict: Verdict
    baseline_acc: float | None
    new_acc: float | None
    delta: float | None
    notes: str = ""


# ---------------------------------------------------------------------------
# §7.5.1 accuracy_risk table — must mirror actions/_meta/*.yaml.
# ---------------------------------------------------------------------------
ACCURACY_RISK_TABLE: dict[str, float] = {
    # prep / analysis
    "setup": 0.0,
    "classify": 0.0,
    "target-analysis": 0.0,
    "baseline": 0.0,
    "profile": 0.0,
    # shallow
    "backends": 0.10,
    "params": 0.0,            # default; some param sweeps (kv-cache fp8) override to 0.30
    "sweep": 0.0,
    "report": 0.0,
    # deep_kernel
    "kernel-opt": 0.10,
    "integrate": 0.15,
    "deep-kernel-analysis": 0.0,
    "operator-tuning": 0.10,
    "vendor-kernel-config": 0.10,
    # long
    "framework-rebuild": 0.15,
    "comm-optimization": 0.05,
    "compiler-tuning": 0.05,
    # creative / resilience
    "dream": 0.0,
    "re-explore": 0.0,
    "recover": 0.0,
}


def _normalize_action_name(name: str) -> str:
    """Map hyphen/underscore variants to a single canonical form (hyphen)."""
    return str(name).strip().lower().replace("_", "-")


def requires_gate(action_name: str) -> bool:
    """True if this action MUST go through the accuracy gate.

    Accepts both hyphenated (``target-analysis``) and underscored
    (``target_analysis``) names so callers don't have to remember which
    variant the YAML uses.
    """
    canonical = _normalize_action_name(action_name)
    free = {_normalize_action_name(n) for n in _GATE_FREE_ACTIONS}
    if canonical in free:
        return False
    table_canon = {_normalize_action_name(k): v for k, v in ACCURACY_RISK_TABLE.items()}
    return table_canon.get(canonical, 0.0) > 0.0


class AccuracyGateError(RuntimeError):
    """Raised when the gate cannot make a meaningful decision."""


_EVAL_SCRIPT_NAME = "eval_accuracy.sh"
_DEFAULT_TIMEOUT_S = 30 * 60  # 30 min hard cap


def _default_eval_script() -> Path:
    """Resolve the bundled ``eval_accuracy.sh`` lazily so import succeeds
    even when the skill root cannot be located (e.g. during unit tests
    that monkeypatch :func:`_run_eval`).

    Falls back to the bare relative path so callers can still see *some*
    diagnostic when the skill is missing — they get
    ``ProcessManagementError`` / ``AccuracyGateError`` from the next
    layer rather than an ImportError.
    """
    try:
        from ..paths import skill_script
        return skill_script(_EVAL_SCRIPT_NAME)
    except Exception:  # noqa: BLE001 — best-effort resolution
        return Path("scripts") / _EVAL_SCRIPT_NAME


def _run_eval(
    *,
    script_path: Path,
    server_port: int,
    model_path: str,
    results_dir: Path,
    eval_task: str = "gsm8k",
    num_fewshot: int = 5,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> int:
    """Subprocess seam — patched by tests to avoid invoking the real script."""
    env = dict(os.environ)
    env.update(
        EVAL_TASK=eval_task,
        NUM_FEWSHOT=str(num_fewshot),
        PORT=str(server_port),
        MODEL=str(model_path),
        RESULTS_DIR=str(results_dir),
    )
    return subprocess.call(
        [str(script_path)],
        env=env,
        timeout=timeout_s,
    )


def run_gsm8k(
    server_port: int,
    model_path: str,
    results_dir: Path,
    *,
    script_path: Path | None = None,
    eval_task: str = "gsm8k",
    num_fewshot: int = 5,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> float:
    """Invoke ``scripts/eval_accuracy.sh`` and read back the GSM8K score.

    Returns the unfewshot accuracy ∈ [0, 1] reported in
    ``<results_dir>/eval_summary_<eval_task>.json``.

    Raises :class:`AccuracyGateError` when the script exits non-zero or
    the summary is missing/malformed.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if script_path is None:
        script_path = _default_eval_script()
    rc = _run_eval(
        script_path=Path(script_path),
        server_port=int(server_port),
        model_path=str(model_path),
        results_dir=results_dir,
        eval_task=eval_task,
        num_fewshot=num_fewshot,
        timeout_s=timeout_s,
    )
    if rc != 0:
        raise AccuracyGateError(
            f"eval script exited rc={rc} (port={server_port}, task={eval_task})"
        )
    summary = results_dir / f"eval_summary_{eval_task}.json"
    return extract_score_from_summary(summary)


def extract_score_from_summary(summary_path: Path) -> float:
    """Parse ``eval_summary_<task>.json`` and return the score ∈ [0, 1].

    Accepts both the lm-evaluation-harness shape (``{"results": {"task":
    {"acc": 0.42, ...}}}``) and a flat ``{"score": 0.42}`` shape used by
    the dry-run fixture.
    """
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise AccuracyGateError(f"summary file not found: {summary_path}")
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccuracyGateError(
            f"failed to read accuracy summary {summary_path}: {exc}"
        ) from exc

    score = _extract_score_field(data)
    if score is None:
        raise AccuracyGateError(
            f"could not locate accuracy score in {summary_path}"
        )
    if not 0.0 <= score <= 1.0:
        raise AccuracyGateError(
            f"accuracy score out of range [0,1]: {score!r}"
        )
    return float(score)


def _extract_score_field(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("score"), (int, float)):
        return float(data["score"])
    # lm-evaluation-harness layout
    results = data.get("results")
    if isinstance(results, dict) and results:
        # take the first task's acc
        first = next(iter(results.values()))
        if isinstance(first, dict):
            for key in ("acc_norm,none", "acc,none", "acc", "exact_match"):
                v = first.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
    # nested {"task": {"score": x}} layout
    for v in data.values():
        if isinstance(v, dict):
            inner = _extract_score_field(v)
            if inner is not None:
                return inner
    return None


def compare_to_baseline(
    baseline_acc: float, new_acc: float, threshold: float = 0.01
) -> Verdict:
    """KEEP if ``new_acc >= baseline_acc - threshold`` else REVERT.

    Returns FAIL only when either input is NaN / negative / >1.
    """
    if any(
        not isinstance(x, (int, float)) or math.isnan(x) or x < 0.0 or x > 1.0
        for x in (baseline_acc, new_acc)
    ):
        return Verdict.FAIL
    if new_acc >= baseline_acc - threshold:
        return Verdict.KEEP
    return Verdict.REVERT


def _flatten(seq: Any) -> Iterable[float]:
    """Best-effort flattener for nested lists / tuples / numpy arrays."""
    try:
        import numpy as np  # type: ignore[import-untyped]
        if isinstance(seq, np.ndarray):
            yield from (float(x) for x in seq.flatten().tolist())
            return
    except Exception:
        pass
    if isinstance(seq, (list, tuple)):
        for item in seq:
            yield from _flatten(item)
    elif isinstance(seq, (int, float)):
        yield float(seq)
    else:
        # final fallback — try iterating
        try:
            for item in seq:
                yield from _flatten(item)
        except TypeError:
            yield float(seq)


def optional_kernel_micro_check(
    orig_out: Any,
    opt_out: Any,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """``allclose`` check between two tensor-like outputs.

    Accepts numpy arrays, nested lists/tuples, or python floats. Returns
    ``True`` when every element pair satisfies
    ``|a - b| <= atol + rtol * |b|``.
    """
    a_vals = list(_flatten(orig_out))
    b_vals = list(_flatten(opt_out))
    if len(a_vals) != len(b_vals):
        return False
    for a, b in zip(a_vals, b_vals):
        if math.isnan(a) or math.isnan(b):
            return False
        if abs(a - b) > atol + rtol * abs(b):
            return False
    return True


__all__ = [
    "Verdict",
    "GateResult",
    "ACCURACY_RISK_TABLE",
    "AccuracyGateError",
    "requires_gate",
    "run_gsm8k",
    "extract_score_from_summary",
    "compare_to_baseline",
    "optional_kernel_micro_check",
]
