# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the best prior solutions from the KB Store.

Used by the forge-loop's warm-start: before optimization begins, look up what
past runs recorded under this kernel's five-tuple, ranked on measured evidence
ahead of unverified claims. The GPU is part of the address, so every candidate
returned was recorded on the machine's own architecture and there is nothing to
filter afterwards. Only an exact implementation signature is eligible for the
downstream auto-apply gate; every mismatch remains reference.

Identity comes from the SAME resolver the write side uses
(:mod:`kernelforge.knowledge.loop_identity`), so a read reliably resolves to
the address a prior run wrote to. Best-effort: missing config, a transport
error, or an empty store all yield no candidates and the loop cold-starts.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Protocol

from kernelforge.knowledge.experience_sink import (
    detect_framework as detect_framework,
)
from kernelforge.knowledge.implementation_identity import (
    canonical_editable_source_map,
    implementation_signature,
)
from kernelforge.knowledge.loop_identity import PATCH_ARTIFACT

#: Workspace-relative home for the candidates a warm start materialized.
_CANDIDATE_REL = "forge_experiments/kb_candidates"

log = logging.getLogger(__name__)

_MAX_READ_ERROR_LENGTH = 240
_BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[^\s,;}\]]+")
_NAMED_SECRET_RE = re.compile(
    r"(?i)\b(token|password|secret|credential|authorization|api[_-]?key)"
    r"(\s*[:=]\s*)[^\s,;}\]]+"
)
_URL_CREDENTIAL_RE = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class _CandidateBundle(Protocol):
    """Materialized candidate fields consumed by the experience reader."""

    files_dir: Path


def sanitize_read_error(exc: Exception, *, secrets: tuple[str, ...] = ()) -> str:
    """Return a bounded exception summary with credential-like values redacted."""
    message = f"{type(exc).__name__}: {exc}"
    message = _BEARER_SECRET_RE.sub("Bearer [REDACTED]", message)
    message = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        message,
    )
    message = _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", message)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:_MAX_READ_ERROR_LENGTH]


def _set_read_status(
    read_status: dict[str, str] | None,
    reason: str,
    error: str = "",
) -> None:
    if read_status is not None:
        read_status["read_reason"] = reason
        read_status["read_error"] = error


@contextlib.contextmanager
def _candidate_destination(workspace: str) -> Iterator[Path]:
    """Where the SDK may drop this read's candidates.

    Inside the caller's workspace when there is one, so a run that misapplied a
    candidate can still be inspected after it ends. Without a workspace the
    kernel may well live in site-packages, and materializing a patch into a
    framework install is not something a read should ever do, so an anonymous
    temporary directory takes its place.

    The directory is never created here: the SDK creates it per candidate, so a
    cold identity leaves nothing behind.
    """
    root = str(workspace or "").strip()
    if not root:
        with tempfile.TemporaryDirectory(prefix="forge-loop-kb-") as temporary:
            yield Path(temporary)
        return
    destination = Path(root) / _CANDIDATE_REL
    # One generation at a time. Bundles are a cache of what this read selected,
    # so an earlier read's leftovers must not sit beside them looking current.
    with contextlib.suppress(OSError):
        shutil.rmtree(destination)
    yield destination


def _bundle_patch(bundle: _CandidateBundle) -> str:
    """Read one materialized candidate's diff off disk, ``""`` when absent.

    Read as bytes: a text handle folds ``\\r\\n`` to ``\\n``, and a patch whose
    newlines no longer match its source is one ``git apply`` will refuse while
    still looking like a valid diff.
    """
    path = Path(bundle.files_dir) / PATCH_ARTIFACT
    if not path.is_file() or path.is_symlink():
        return ""
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_best_solution(
    *,
    config,
    kernel_path: str,
    kernel_source: str,
    kernel_backend: str,
    target_functions: list[str] | None = None,
    framework: str = "",
    source_files: list[str] | None = None,
    workspace: str = "",
    operator_name: str = "",
) -> dict[str, Any] | None:
    """Return the best prior solution for this operator on this GPU, or None.

    The returned dict carries everything the warm-start needs::

        {
          "kernel_slug": str, "session_id": str, "solution_slug": str,
          "speedup": float, "measured_speedup": float | None,
          "patch_content": str, "strategy": str, "recipe": str,
          "lessons": str, "metric": dict,
        }

    Never raises - returns None on any failure so the loop cold-starts.
    """
    try:
        return _read_best_solution_impl(
            config=config,
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            kernel_backend=kernel_backend,
            target_functions=target_functions,
            framework=framework,
            source_files=source_files,
            workspace=workspace,
            operator_name=operator_name,
        )
    except Exception as exc:  # noqa: BLE001 - warm-start read must never break a run
        log.warning("experience read failed (cold start): %r", exc)
        return None


def read_top_solutions(
    *,
    config,
    kernel_path: str,
    kernel_source: str,
    kernel_backend: str,
    target_functions: list[str] | None = None,
    framework: str = "",
    top_k: int = 3,
    source_files: list[str] | None = None,
    workspace: str = "",
    operator_name: str = "",
    read_status: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``top_k`` prior solutions for this operator.

    Candidates recorded for this GPU are ranked on measured evidence first and
    on the claimed speedup only when no consumer has measured them. Each dict
    has the same shape as ``read_best_solution`` plus reference metadata. Never
    raises: returns ``[]`` on any failure so the loop cold-starts. Adoption is
    gated downstream by implementation identity, apply, correctness, and
    performance checks.
    When supplied, ``read_status`` receives stable ``read_reason`` and
    ``read_error`` fields without changing the list return API.
    """
    _set_read_status(read_status, "read_error")
    try:
        return _read_top_solutions_impl(
            config=config,
            kernel_path=kernel_path,
            kernel_source=kernel_source,
            kernel_backend=kernel_backend,
            target_functions=target_functions,
            framework=framework,
            top_k=max(1, int(top_k)),
            source_files=source_files,
            workspace=workspace,
            operator_name=operator_name,
            read_status=read_status,
        )
    except Exception as exc:  # noqa: BLE001 - warm-start read must never break a run
        error = sanitize_read_error(
            exc,
            secrets=(
                str(getattr(config, "gbrain_token", "") or ""),
                os.environ.get("GBRAIN_TOKEN", ""),
            ),
        )
        log.warning("experience top-k read failed (cold start): %s", error)
        _set_read_status(
            read_status,
            "read_error",
            error,
        )
        return []


def _read_best_solution_impl(
    *,
    config,
    kernel_path,
    kernel_source,
    kernel_backend,
    target_functions=None,
    framework="",
    source_files=None,
    workspace="",
    operator_name="",
):
    """Single highest-speedup solution — thin wrapper over the top-k impl."""
    top = _read_top_solutions_impl(
        config=config,
        kernel_path=kernel_path,
        kernel_source=kernel_source,
        kernel_backend=kernel_backend,
        target_functions=target_functions,
        framework=framework,
        top_k=1,
        source_files=source_files,
        workspace=workspace,
        operator_name=operator_name,
    )
    return top[0] if top else None


def _build_solution_dict(
    *, canonical_id, prior, patch, consumer_signature, consumer_identity, consumer_source_map
) -> dict[str, Any]:
    """Assemble the warm-start payload for one recorded candidate."""
    attrs = dict(prior.value)
    candidate_signature = str(attrs.get("implementation_signature") or "")
    implementation_match = bool(candidate_signature and candidate_signature == consumer_signature)
    return {
        "kernel_slug": canonical_id,
        "session_id": prior.session_id,
        "solution_slug": f"{canonical_id}/{prior.session_id}",
        "speedup": float(prior.speedup or 0.0),
        "measured_speedup": prior.measured_speedup,
        "match_mode": "exact" if implementation_match else "reference",
        "implementation_signature": candidate_signature,
        "consumer_implementation_signature": consumer_signature,
        "implementation_identity": (
            attrs.get("implementation_identity") if isinstance(attrs.get("implementation_identity"), dict) else {}
        ),
        "consumer_implementation_identity": consumer_identity,
        "consumer_source_map": consumer_source_map,
        "implementation_match": implementation_match,
        "patch_content": patch,
        "strategy": str(attrs.get("strategy") or ""),
        "recipe": str(attrs.get("recipe") or ""),
        "lessons": str(attrs.get("lessons") or ""),
        "metric": attrs.get("metric") if isinstance(attrs.get("metric"), dict) else {},
    }


def _read_top_solutions_impl(
    *,
    config,
    kernel_path,
    kernel_source,
    kernel_backend,
    target_functions=None,
    framework="",
    top_k=3,
    source_files=None,
    workspace="",
    operator_name="",
    read_status=None,
):
    # Must be the same dimension the write side addresses by, or the read
    # resolves to an address no run ever wrote to and every start looks cold.
    gpu_type = str(getattr(config, "gpu_type", "") or "").strip()
    if not gpu_type:
        log.info("experience read skipped: GPU hardware model is required")
        _set_read_status(read_status, "missing_gpu_type")
        return []

    # Identity MUST match the write side exactly, or a read resolves to an
    # address no prior write reached. Both sides call one resolver so the two
    # cannot drift apart.
    from kernelforge.knowledge.loop_identity import resolve_loop_identity

    identity, _concrete_op, framework = resolve_loop_identity(
        kernel_path=kernel_path,
        kernel_source=kernel_source,
        kernel_backend=kernel_backend,
        gpu_type=gpu_type,
        target_functions=target_functions,
        source_files=source_files,
        framework=framework,
        operator_name=operator_name,
        producer=getattr(config, "producer", ""),
    )

    # Imported here rather than at module scope: the facade's identity module
    # imports the sink this module also imports, so a top-level import would
    # close a cycle.
    from kernelforge.rewrite_by_flydsl.agent_kb import KernelRecipeKB

    kb = KernelRecipeKB.open_identity(identity, config)
    if not kb.active:
        log.info("experience read skipped: %s", kb.reason or "not_configured")
        _set_read_status(read_status, kb.reason or "not_configured")
        return []

    consumer_workspace = workspace or str(Path(kernel_path).resolve().parent)
    consumer_signature, consumer_identity = implementation_signature(
        workspace=consumer_workspace,
        kernel_path=kernel_path,
        source_files=source_files,
        framework=framework,
    )
    consumer_source_map = canonical_editable_source_map(
        workspace=consumer_workspace,
        kernel_path=kernel_path,
        source_files=source_files,
        framework=framework,
    )
    log.info("experience read: identity=%s", kb.canonical_id)

    # Materialize the Top-N and read them back off disk: that is the SDK's
    # normal consumer path, it downloads each selected session once under its
    # integrity checks, and it leaves the candidates where a person debugging
    # the run can look at them. The GPU and the producer are part of the
    # address, so nothing returned here needs filtering out.
    #
    # The patch is read inside the block because a workspace-less caller gets a
    # temporary destination that disappears on the way out.
    with _candidate_destination(workspace) as destination:
        candidates = kb.read_top_n(destination, limit=top_k)
        if kb.reason:
            # The facade swallows transport failures into ``reason``; surface it
            # as a read error so a cold start is not mistaken for an empty store.
            log.warning("experience read failed (cold start): %s", kb.reason)
            _set_read_status(read_status, "read_error", kb.reason)
            return []
        if not candidates:
            log.info("experience read: no prior record for %s", kb.canonical_id)
            _set_read_status(read_status, "no_prior_record")
            return []

        out: list[dict[str, Any]] = [
            _build_solution_dict(
                canonical_id=kb.canonical_id,
                prior=bundle,
                patch=_bundle_patch(bundle),
                consumer_signature=consumer_signature,
                consumer_identity=consumer_identity,
                consumer_source_map=consumer_source_map,
            )
            for bundle in candidates
        ]
    if not out:
        log.info("experience read: no usable record for %s", kb.canonical_id)
        _set_read_status(read_status, "no_prior_record")
    else:
        log.info(
            "experience read: %d solution(s), best %s (speedup=%.3f)",
            len(out),
            out[0]["solution_slug"],
            out[0]["speedup"],
        )
        _set_read_status(read_status, "hit")
    return out
