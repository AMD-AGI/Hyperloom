# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK same-harness revalidation candidate identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_FALLBACK_FAILED,
    REBENCH_NO_MATERIAL,
    REBENCH_NO_PROMOTE,
    REBENCH_VALIDATED,
)

INCOMPARABLE_REVALIDATION = "incomparable"

# Verdict annotations are not candidate identity.
_REVALIDATION_ANNOTATION_KEYS: frozenset[str] = frozenset(
    {
        "kernel_event_id",
        "final_validation",
        "native_source_revalidation",
        "canonical_revalidation",
        "revalidation_status",
        "revalidation_error",
        "revalidation_error_class",
        "revalidation_blocked_overlay",
    }
)

# Failed measurements remain retryable.
_TERMINAL_REVALIDATION_STATUSES: frozenset[str] = frozenset({REBENCH_NO_MATERIAL, REBENCH_NO_PROMOTE})


def geak_harness_replays_workload(state: Any) -> bool:
    """Whether GEAK's own harness can replay this session's workload.

    The canonical AgentX workload it cannot: ``validate_geak_via_geak_harness``
    refuses before launch, which is what makes a refusal there structural rather
    than a run that might land next time. Persisted mode wins over the ambient
    switch, matching that refusal, so a session recorded as synthetic keeps
    replaying.
    """
    from hyperloom.common.perf_metric import agentx_enabled, is_agentx_mode

    mode = str(state.benchmark_mode or "").strip()
    return not (is_agentx_mode(mode) if mode else agentx_enabled())


def geak_verdict_is_terminal(persisted: Any) -> bool:
    """True when a replay of the same candidate could not change the verdict.

    A refusal counts only with its typed class (:data:`INCOMPARABLE_REVALIDATION`),
    so state written before the class was recorded stays retryable instead of
    being read out of a reason string.
    """
    prev = persisted if isinstance(persisted, dict) else {}
    status = str(prev.get("revalidation_status") or "")
    return status in _TERMINAL_REVALIDATION_STATUSES or (
        status == REBENCH_FALLBACK_FAILED
        and str(prev.get("revalidation_error_class") or "") == INCOMPARABLE_REVALIDATION
    )


def geak_candidate_is_adjudicated(persisted: Any, recovered: Any, *, harness_can_replay: bool) -> bool:
    """True when ``persisted``'s verdict already settles the ``recovered`` result.

    Guards KERNEL crash-recovery, whose job is to promote a ``result.json`` whose
    handback was lost. Where the harness can replay the workload the verdict
    alone decides, as it always has. Where it cannot, the verdict must also have
    been reached on the result now on disk: identity is the runner's own result
    content minus this module's verdict stamps, so any field a rerun moves reads
    as new evidence. A pre-dispatch overlay refusal also checks whether the
    blocked overlay has become loadable; other artifact contents are not read.

    Args:
        persisted: ``shared_state.geak_result`` — the last adjudicated result.
        recovered: The result parsed from ``result.json``.
        harness_can_replay: See :func:`geak_harness_replays_workload`.

    Returns:
        Whether recovery must leave the recovered result alone.
    """
    prev = persisted if isinstance(persisted, dict) else {}
    if harness_can_replay:
        return str(prev.get("revalidation_status") or "") in _TERMINAL_REVALIDATION_STATUSES
    if prev.get("native_source_revalidation"):
        return geak_candidate_matches(prev, recovered) and source_verdict_matches(prev)
    if not geak_verdict_is_terminal(prev):
        return False
    blocked_overlay = str(prev.get("revalidation_blocked_overlay") or "")
    if blocked_overlay:
        from ..kernel.geak_config import geak_overlay_is_loadable, _normalize_geak_overlay_dir

        if geak_overlay_is_loadable(_normalize_geak_overlay_dir(blocked_overlay)):
            return False
    return geak_candidate_matches(prev, recovered)


def geak_candidate_matches(persisted: Any, recovered: Any) -> bool:
    """Whether a raw GEAK result is the same product, ignoring coordinator annotations."""
    prev = persisted if isinstance(persisted, dict) else {}
    raw = recovered if isinstance(recovered, dict) else {}
    # The phase stamps the runner's exit code onto state with ``setdefault``, so
    # ``returncode`` is an annotation only where the file carries none: it is a
    # property of the process, not of the product. A value GEAK wrote itself
    # stays part of the product and is compared.
    stamped = _REVALIDATION_ANNOTATION_KEYS if "returncode" in raw else _REVALIDATION_ANNOTATION_KEYS | {"returncode"}
    return _geak_candidate_identity(prev, stamped) == _geak_candidate_identity(raw, stamped)


def _geak_candidate_identity(result: Any, stamped: frozenset[str]) -> dict[str, Any]:
    """The GEAK result content that identifies a candidate, ``stamped`` keys aside."""
    payload = result if isinstance(result, dict) else {}
    return {key: value for key, value in payload.items() if key not in stamped}


__all__ = [
    "INCOMPARABLE_REVALIDATION",
    "geak_candidate_is_adjudicated",
    "geak_candidate_matches",
    "geak_harness_replays_workload",
    "geak_verdict_is_terminal",
]


def source_digest(path: Path) -> str:
    """Hash the exact source bytes bound to a review."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay_source_files(overlay: str) -> dict[str, str]:
    """Use the native launch's source evidence for every deployable overlay file."""
    if not overlay:
        return {}
    return {
        str(path): source_digest(path)
        for path in sorted(Path(overlay).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def verify_source_artifacts(params: dict[str, Any]) -> None:
    """Reject changes to a reviewed patch or deployable overlay."""
    for filename, expected in params["native_geak_files"].items():
        path = Path(filename)
        if not path.is_file() or source_digest(path) != expected:
            raise ValueError(f"reviewed GEAK source patch changed or disappeared: {path}")
    overlay = str(params.get("overlay_pythonpath") or "")
    if overlay and overlay_source_files(overlay) != params.get("native_geak_overlay_files"):
        raise ValueError("reviewed GEAK overlay changed or disappeared")


def source_verdict_matches(result: dict[str, Any]) -> bool:
    """A canonical verdict settles only the source bytes that were reviewed."""
    settled = result["native_source_revalidation"]
    if settled.get("status") not in {REBENCH_VALIDATED, REBENCH_NO_PROMOTE}:
        return False
    params = settled["params"]
    try:
        verify_source_artifacts(params)
        return all(
            Path(name).is_file() and source_digest(Path(name)) == digest
            for name, digest in params["native_geak_original_files"].items()
        )
    except (OSError, ValueError):
        return False
