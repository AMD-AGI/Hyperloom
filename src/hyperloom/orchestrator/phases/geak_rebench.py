# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK same-harness revalidation candidate identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.geak_rebench_verdict import (
    REBENCH_FAILED,
    REBENCH_NO_MATERIAL,
    REBENCH_NO_PROMOTE,
    REBENCH_UNMEASURED_GIVEN_UP,
    REBENCH_VALIDATED,
)

#: Beside a candidate's settled status: how many of its measurements ended with no verdict.
REVALIDATION_UNMEASURED = "revalidation_unmeasured"
#: How many measurements of one candidate may end with no verdict before it is given up. Two: one repeat
#: separates a passing disturbance from a product that cannot be measured, and each further one costs a
#: server boot or a full round and holds back a fresh GEAK run.
MAX_UNMEASURED_REVALIDATIONS = 2

# Coordinator annotations are not candidate identity.
_REVALIDATION_ANNOTATION_KEYS: frozenset[str] = frozenset(
    {
        "absent_backends",
        "kernel_event_id",
        "final_validation",
        "native_source_revalidation",
        "canonical_revalidation",
        "revalidation_status",
        "revalidation_error",
        "revalidation_error_class",
        "revalidation_blocked_overlay",
        REVALIDATION_UNMEASURED,
    }
)

# Failed measurements remain retryable until the candidate is given up.
_TERMINAL_REVALIDATION_STATUSES: frozenset[str] = frozenset({REBENCH_NO_MATERIAL, REBENCH_NO_PROMOTE})


def geak_session_is_agentx(state: Any) -> bool:
    """Whether the session GEAK is handed runs the AgentX workload; the persisted mode wins over the ambient switch."""
    from hyperloom.common.perf_metric import agentx_enabled, is_agentx_mode

    mode = str(state.benchmark_mode or "").strip()
    return is_agentx_mode(mode) if mode else agentx_enabled()


def geak_verdict_is_terminal(persisted: Any) -> bool:
    """True when the session does not measure the same candidate again.

    A failed measurement counts only once it was given up (``REBENCH_UNMEASURED_GIVEN_UP``).
    """
    prev = persisted if isinstance(persisted, dict) else {}
    status = str(prev.get("revalidation_status") or "")
    error_class = str(prev.get("revalidation_error_class") or "")
    return status in _TERMINAL_REVALIDATION_STATUSES or (
        status == REBENCH_FAILED and error_class == REBENCH_UNMEASURED_GIVEN_UP
    )


def geak_candidate_is_adjudicated(persisted: Any, recovered: Any) -> bool:
    """True when ``persisted``'s verdict already settles the ``recovered`` result.

    Guards KERNEL crash-recovery, whose job is to promote a ``result.json`` whose
    handback was lost. The verdict must have been reached on the result now on
    disk: identity is the runner's own result content minus this module's verdict
    stamps, so any field a rerun moves reads as new evidence. A settlement that
    named an unloadable overlay also checks whether that overlay has become
    loadable; other artifact contents are not read.

    Args:
        persisted: ``shared_state.geak_result`` — the last adjudicated result.
        recovered: The result parsed from ``result.json``.

    Returns:
        Whether recovery must leave the recovered result alone.
    """
    prev = persisted if isinstance(persisted, dict) else {}
    if prev.get("native_source_revalidation"):
        return geak_candidate_matches(prev, recovered) and source_verdict_matches(prev)
    if not geak_verdict_is_terminal(prev):
        return False
    blocked_overlay = str(prev.get("revalidation_blocked_overlay") or "")
    if blocked_overlay:
        from hyperloom.common.overlay import overlay_is_loadable

        from ..kernel.geak_config import _normalize_geak_overlay_dir

        if overlay_is_loadable(_normalize_geak_overlay_dir(blocked_overlay)):
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


def keep_unmeasured_count(persisted: dict[str, Any], recovered: dict[str, Any]) -> None:
    """Carry onto a raw GEAK result the unmeasured count the session holds for the same product."""
    count = persisted.get(REVALIDATION_UNMEASURED)
    if count and geak_candidate_matches(persisted, recovered):
        recovered[REVALIDATION_UNMEASURED] = count


def _geak_candidate_identity(result: Any, stamped: frozenset[str]) -> dict[str, Any]:
    """The GEAK result content that identifies a candidate, ``stamped`` keys aside."""
    payload = result if isinstance(result, dict) else {}
    return {key: value for key, value in payload.items() if key not in stamped}


__all__ = [
    "MAX_UNMEASURED_REVALIDATIONS",
    "REVALIDATION_UNMEASURED",
    "geak_candidate_is_adjudicated",
    "geak_candidate_matches",
    "geak_session_is_agentx",
    "geak_verdict_is_terminal",
    "keep_unmeasured_count",
]


def source_digest(path: Path) -> str:
    """Hash the exact source bytes a revalidation is bound to."""
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
    """Reject changes to a staged patch or deployable overlay."""
    for filename, expected in params["native_geak_files"].items():
        path = Path(filename)
        if not path.is_file() or source_digest(path) != expected:
            raise ValueError(f"staged GEAK source patch changed or disappeared: {path}")
    overlay = str(params.get("overlay_pythonpath") or "")
    if overlay and overlay_source_files(overlay) != params.get("native_geak_overlay_files"):
        raise ValueError("staged GEAK overlay changed or disappeared")


def source_verdict_matches(result: dict[str, Any]) -> bool:
    """A canonical verdict, or a measurement given up unmeasured, settles only the source bytes it was for."""
    settled = result["native_source_revalidation"]
    if settled.get("status") not in {REBENCH_VALIDATED, REBENCH_NO_PROMOTE} and not geak_verdict_is_terminal(result):
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
