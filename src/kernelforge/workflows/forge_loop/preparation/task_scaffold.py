# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Files task preparation lays out around the driver: reference bundle, harnesses, invocation spec, external staging."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from kernelforge.config import Config
from kernelforge.config.resources import resource_path
from kernelforge.workflows.forge_loop.preparation.external_artifacts import (
    ExternalArtifactError,
    ExternalArtifactTransaction,
)
from kernelforge.workflows.forge_loop.preparation.task_preparer_prompts import (
    DRIVER_CONTRACT_SPEC,
    REFERENCE_DRIVER_TEMPLATE,
)

log = logging.getLogger(__name__)


# The prep agent reads REAL reference example files (not just prompt text).
REFERENCE_SUBDIR = ".forge_task_reference"
INVOCATION_SPEC_FILENAME = "invocation_spec.json"
MAX_INVOCATION_SPEC_BYTES = 1024 * 1024
_INVOCATION_SPEC_NAME_RE = re.compile(r"^invocation_spec_[A-Za-z0-9._-]+\.json$")
_REFERENCE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "forge_experiments", ".git")


def _safe_rmtree(path: Path | None) -> None:
    """Remove a tree best-effort; the caller checks whether it actually went."""
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


def _dist_harness_text() -> str | None:
    """Return the shipped distributed harness's own source.

    Read from this package rather than from the reference examples the way
    ``graph_harness`` is: the examples tree is authoring scaffolding that
    preparation deletes, while a multi-rank driver imports this module on every
    later run of the loop, so it has to come from something the wheel carries.
    """
    try:
        return (Path(__file__).parent / "dist_harness.py").read_text(encoding="utf-8")
    except OSError:
        return None


def _find_reference_harness(ref_dir: Path | None) -> str | None:
    """Return the text of a capture-guarded graph harness from the reference tree."""
    if ref_dir is None or not ref_dir.is_dir():
        return None
    for cand in sorted(ref_dir.rglob("graph_harness.py")):
        try:
            text = cand.read_text()
        except OSError:
            continue
        if "def cuda_graph_bench" in text and "dirty" in text:
            return text
    return None


def _materialize_reference(workspace: Path) -> Path | None:
    """Make the shipped reference examples available for the agent to Read."""
    ref_dir = workspace / REFERENCE_SUBDIR
    _safe_rmtree(ref_dir)

    examples = resource_path("examples", missing_ok=True)
    try:
        if examples and Path(examples).is_dir():
            shutil.copytree(examples, ref_dir, ignore=_REFERENCE_IGNORE)
            return ref_dir
    except OSError:
        _safe_rmtree(ref_dir)

    # Fallback: no examples tree resolved — materialize the compact contract and a driver template so the agent still
    # has real files to Read.
    try:
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "CONTRACT.md").write_text(DRIVER_CONTRACT_SPEC)
        (ref_dir / "driver_template.py").write_text(REFERENCE_DRIVER_TEMPLATE.lstrip("\n"))
        return ref_dir
    except OSError:
        _safe_rmtree(ref_dir)
        return None


def _materialize_invocation_spec(
    source_file: str,
    durable_dir: Path | None,
) -> tuple[Path | None, str]:
    """Place the invocation spec where the prepared driver can keep reading it."""
    if not source_file or durable_dir is None:
        return None, ""
    try:
        source = Path(source_file).expanduser().resolve()
        if not source.is_file() or source.stat().st_size > MAX_INVOCATION_SPEC_BYTES:
            return None, ""
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, ""
        destination_name = source.name if _INVOCATION_SPEC_NAME_RE.fullmatch(source.name) else INVOCATION_SPEC_FILENAME
        destination = durable_dir / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            log.warning(
                "invocation specification destination %s is a symbolic link; "
                "leaving it alone rather than writing through it",
                destination,
            )
            return None, ""
        if destination.exists():
            with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
                existing = destination.read_text(encoding="utf-8")
                if json.loads(existing) == payload:
                    return destination, existing
            log.warning(
                "invocation specification destination %s already holds different "
                "content; leaving it alone rather than replacing it",
                destination,
            )
            return None, ""
        canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        destination.write_text(canonical, encoding="utf-8")
        return destination, canonical
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None, ""


def _place_invocation_spec(invocation_spec_file: str, driver_dir: Path) -> tuple[Path | None, str]:
    """Place the caller's invocation spec beside the driver, warning when a supplied one cannot be used."""
    spec_path, canonical_spec = _materialize_invocation_spec(invocation_spec_file, driver_dir)
    if invocation_spec_file and spec_path is None:
        # Not a default: an explicitly supplied spec that cannot be used costs the prompt's case table, the driver's
        # durable runtime input and the committed-alongside check all at once, and the caller asked for it.
        log.warning(
            "could not materialize the invocation specification %s beside the "
            "driver; preparation continues without its case table, without a "
            "durable runtime input for the driver, and without the check that the "
            "spec is committed with it",
            invocation_spec_file,
        )
    return spec_path, canonical_spec


def declared_case_ids(spec_path: Path | str | None) -> list[str]:
    """Case ids the task declares its driver must benchmark, sorted."""
    if not spec_path:
        return []
    try:
        payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read the invocation specification {spec_path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invocation specification {spec_path} is not a JSON object")
    tests = payload.get("tests")
    contract = tests.get("driver_contract") if isinstance(tests, dict) else None
    selectors = contract.get("case_selectors") if isinstance(contract, dict) else None
    if not isinstance(selectors, list):
        return []
    return sorted(
        {str(selector["CASE_ID"]) for selector in selectors if isinstance(selector, dict) and selector.get("CASE_ID")}
    )


def _driver_is_inside(workspace: Path, driver_input_path: Path, driver_path: Path) -> bool:
    """Whether the driver lives in the workspace both lexically and after resolution.

    Both have to hold: a path that only looks internal, or only resolves
    internal, is a symlink out of the tree and is staged like any other
    external driver.
    """
    for candidate in (driver_input_path, driver_path):
        try:
            candidate.relative_to(workspace)
        except ValueError:
            return False
    return True


@dataclass(frozen=True)
class _AuditDirs:
    """Where a preparation's experiments live, and its audit trail when that directory could be created."""

    experiments_dir: Path | None
    audit_dir: Path | None


def _preparation_audit_dir(config: Config) -> _AuditDirs:
    """Where this preparation writes its experiments and its audit trail, as far as each resolves.

    They are reported separately because they fail separately: a directory that
    cannot be created costs the audit trail, while the experiments path it sits
    under is still the one the external-artifact transaction has to exclude --
    that tree changes while the agent runs, which is what the exclusion is for.
    """
    experiments_dir: Path | None = None
    try:
        experiments_dir = Path(config.experiments_dir)
        audit_dir = experiments_dir / "task_preparation"
        audit_dir.mkdir(parents=True, exist_ok=True)
    except (AttributeError, OSError, TypeError, ValueError):
        return _AuditDirs(experiments_dir, None)
    return _AuditDirs(experiments_dir, audit_dir)


def _stage_external_driver(
    workspace: Path,
    *,
    driver_input_path: Path,
    driver_path: Path,
    audit_dirs: _AuditDirs,
    read_only_paths: list[Path],
) -> tuple[ExternalArtifactTransaction | None, str]:
    """Stage a driver that lives outside the workspace, or say why it cannot be staged."""
    external_exclusions = [workspace]
    experiments_dir, audit_dir = audit_dirs.experiments_dir, audit_dirs.audit_dir
    if experiments_dir is not None:
        try:
            experiments_rel = experiments_dir.resolve().relative_to(driver_path.parent)
            if experiments_rel.parts:
                # Runtime logs/results can change while the agent is running.
                external_exclusions.append(driver_path.parent / experiments_rel.parts[0])
            elif audit_dir is not None:
                external_exclusions.append(audit_dir)
        except (OSError, ValueError):
            # Keep the conservative workspace exclusion when the external experiments path cannot be relativized
            # safely.
            pass
    try:
        transaction = ExternalArtifactTransaction(
            driver_path=driver_input_path,
            excluded_paths=external_exclusions,
            passthrough_paths=[workspace],
            read_only_paths=read_only_paths,
        )
    except ExternalArtifactError as exc:
        return None, f"could not stage external driver artifacts: {exc}"
    return transaction, ""


def _place_graph_harness(
    harness_path: Path,
    *,
    ref_dir: Path | None,
    driver_external: bool,
    reference_prefix: str,
) -> tuple[str | None, str]:
    """Pre-place a correct, capture-guarded ``graph_harness.py`` beside the driver.

    The agent can then import a known-good ``cuda_graph_bench`` (accepting
    ``dirty``/``verify``) instead of writing its own -- a self-written harness
    can silently mismatch its driver calls and degrade graph timing to eager.
    A harness the caller already placed is left alone.
    """
    canonical_harness = None if harness_path.is_file() else _find_reference_harness(ref_dir)
    if canonical_harness is None:
        return None, reference_prefix
    harness_path.write_text(canonical_harness)
    harness_display = str(harness_path) if driver_external else "./graph_harness.py"
    return canonical_harness, (
        "## Graph timing harness (already available beside the driver)\n"
        f"`{harness_display}` is a correct, capture-guarded harness. Import it —\n"
        "`from graph_harness import cuda_graph_bench` (it accepts optional\n"
        "`dirty`/`verify`). Do NOT rewrite it. Only implement custom timing in the\n"
        "driver if this operator genuinely cannot be captured into a static-input\n"
        "graph.\n\n" + reference_prefix
    )


def _place_dist_harness(dist_harness_path: Path, nproc_per_node: int) -> str | None:
    """Place ``dist_harness.py`` before the agent is asked for anything.

    A multi-rank task measures inside it or it is not measured. A missing
    packaged source is not fatal here, so the failure is reported by preflight,
    against the driver, in the same shape as every other distributed refusal.
    """
    if nproc_per_node <= 1:
        return None
    canonical_dist_harness = _dist_harness_text()
    if canonical_dist_harness is None:
        log.error(
            "could not read the packaged dist_harness source; this %d-rank task has no harness to "
            "measure inside and its driver will be refused",
            nproc_per_node,
        )
        return None
    dist_harness_path.write_text(canonical_dist_harness, encoding="utf-8")
    return canonical_dist_harness


def _external_staging_note(driver_path: Path) -> str:
    """The prompt section that confines an external driver's edits to its staging directory."""
    return (
        "## Transactional external driver staging\n"
        f"The driver and every helper it imports MUST be written under the "
        f"isolated staging directory "
        f"`{driver_path.parent.resolve()}`. The kernel workspace is source evidence only; "
        "changes are published to the caller's artifact directory only after "
        "deterministic validation succeeds. Existing task metadata and invocation "
        "specifications are read-only.\n\n"
    )
